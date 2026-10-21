"""A/B control: rerank ON vs OFF — does reranking improve GRADE accuracy?

The :mod:`scripts.run_holdout_grading` script reads each PDF directly and
feeds ALL chunks to :func:`grade_extraction`, bypassing the retrieval
layer where the new evidence-grade reranking lives. To exercise the
rerank layer we have to *reconstruct* the retrieval path locally:

    PDF → chunks → section classify   (simulating ingestion output)
        → MMR-like pool                (chunk order = MMR order)
        → rerank_hits(ON | OFF)        (the variable under test)
        → select_top_k(top_k, canonical_file=...)
        → grade_extraction(top_k)      (same grader for both arms)

Per paper we capture:

* The top-K chunks ON vs OFF — what differs at retrieval level?
* LLM-extracted GRADE fields under each arm (sample_size, effect_size, …).
* composite_score + overall_grade (deterministic Python engine).

Then we report per-paper agreement on:

* Top-K composition (which chunks got swapped)
* study_design
* overall_grade
* composite_score (numeric diff)

We also report a strawman "clinical gold" label per paper (assigned by
the user based on trial characteristics) so agreement-with-gold can be
computed for both arms.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llama_index.core import Document  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402

from grading.grade_evaluator import GRADEResult, grade_extraction  # noqa: E402
from retrieval.selector import (  # noqa: E402
    rerank_hits,
    select_top_k,
    set_rerank_enabled_default,
)
from utils.config_loader import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# 1) Paper catalogue (mirrors run_holdout_grading)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutPaper:
    slug: str
    title: str
    pdf_path: Path
    canonical_file: str


_HOLDOUT_PAPERS: tuple[HoldoutPaper, ...] = (
    HoldoutPaper(
        slug="IMvigor211",
        title="IMvigor211: Atezolizumab vs Chemotherapy in Platinum-Treated mUC (Lancet 2018)",
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S014067361733297X-main.pdf",
        canonical_file="1-s2.0-S014067361733297X-main.pdf",
    ),
    HoldoutPaper(
        slug="PIVOT-10",
        title="PIVOT-10: Bempegaldesleukin + Nivolumab 1L mUC (Urol Oncol 2024)",
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1078143924006719-main.pdf",
        canonical_file="1-s2.0-S1078143924006719-main.pdf",
    ),
    HoldoutPaper(
        slug="CheckMate-275",
        title="CheckMate 275: Nivolumab Monotherapy in mUC after Platinum (Lancet Oncol 2017)",
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1470204517300657-main.pdf",
        canonical_file="1-s2.0-S1470204517300657-main.pdf",
    ),
    HoldoutPaper(
        slug="DANUBE",
        title="DANUBE: Durvalumab ± Tremelimumab vs Chemotherapy 1L mUC (Lancet Oncol 2020)",
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1470204520305416-main.pdf",
        canonical_file="1-s2.0-S1470204520305416-main.pdf",
    ),
    HoldoutPaper(
        slug="Mamede-MetaAnalysis",
        title="Adjuvant Immunotherapy in HR-MIUC: Updated Meta-Analysis (Clin GU Cancer 2025)",
        pdf_path=_PROJECT_ROOT / "holdout" / "1-s2.0-S1558767324002581-main.pdf",
        canonical_file="1-s2.0-S1558767324002581-main.pdf",
    ),
)


# ---------------------------------------------------------------------------
# 2) Clinical-gold labels — assigned by reviewer based on trial identity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldLabel:
    """Clinical ground truth for one paper (assigned by human reviewer)."""

    expected_design: str      # one of the GRADE StudyDesign literals
    expected_overall_grade: str  # one of 高 / 中 / 低 / 极低
    rationale: str


# Each label below states the design + grade that the system *should*
# produce, given the published evidence. These are drawn from the trial
# publication itself (design) and a clinical judgment call on overall
# evidence strength (overall_grade). The script does NOT cheat by reading
# from the LLM output — these are fixed in advance.
_GOLD: dict[str, GoldLabel] = {
    "IMvigor211": GoldLabel(
        # IMvigor211 RCT had OS HR=0.87 (95% CI 0.63-1.21) — fully crossed 1.0
        # in ITT, PFS HR=0.87 (0.72-1.05) borderline. Open-label Phase 3.
        expected_design="RCT",
        # Per GRADE convention: RCT-start-at-高, OS未达 + open-label → 中
        expected_overall_grade="中",
        rationale=(
            "Phase 3 RCT, open-label. OS HR in ITT crossed 1.0 (HR=0.87, "
            "0.63-1.21), PD-L1+ subgroup PFS borderline. Per GRADE, RCT "
            "baseline 高, primary OS did not reach significance → demote "
            "to 中."
        ),
    ),
    "PIVOT-10": GoldLabel(
        # "phase II single-arm study" — title literally says single-arm.
        expected_design="单臂试验",
        # Single-arm Phase 2, N=188. GRADE baseline 低, N<300 → 极低.
        expected_overall_grade="极低",
        rationale=(
            "Single-arm Phase 2, N=188. Per GRADE, 单臂 baseline 低, "
            "N<300 进一步降级 → 极低."
        ),
    ),
    "CheckMate-275": GoldLabel(
        # "phase 2 trial ... single-arm" — single-arm.
        expected_design="单臂试验",
        # Single-arm Phase 2, N=270. Baseline 低, N<300 → 极低.
        expected_overall_grade="极低",
        rationale=(
            "Single-arm Phase 2, N=270 (evaluable 265). Per GRADE, 单臂 "
            "baseline 低, N<300 → 极低."
        ),
    ),
    "DANUBE": GoldLabel(
        # "open-label, phase 3 trial" — RCT.
        expected_design="RCT",
        # RCT, dual-primary endpoints (OS in PD-L1+ and ITT), BOTH 未达
        # strict alpha. Per GRADE, RCT baseline 高 + dual-primary failure
        # → could be 中 if relaxed, or 低 if strict multi-endpoint
        # exemption applies. Borderline; we mark 中 as the conservative
        # gold (matches NCCN's stance that durvalumab is not standard).
        expected_overall_grade="中",
        rationale=(
            "Phase 3 RCT, dual primary (OS in PD-L1+ and ITT) — both 未达. "
            "Per GRADE, RCT baseline 高, both endpoints failed → 中 (with "
            "exemption fired) or 低 (without). Conservative gold 中."
        ),
    ),
    "Mamede-MetaAnalysis": GoldLabel(
        # "An Updated Meta-Analysis of Randomized Controlled Trials" —
        # explicitly a meta-analysis.
        expected_design="Meta分析",
        # Meta-analysis of 3 RCTs, N=2220. Per GRADE, Meta分析 baseline 高,
        # meaningful effect (HR 0.76 95% CI 0.65-0.90, P<0.05) → 高 or 中.
        expected_overall_grade="高",
        rationale=(
            "Meta-analysis of 3 RCTs (total N=2220) showing adjuvant "
            "anti-PD1/PDL1 significantly improves DFS. Per GRADE, Meta分析 "
            "baseline 高, consistent positive effect → 高."
        ),
    ),
}


# ---------------------------------------------------------------------------
# 3) PDF parsing + section classification (reuses ingestion module)
# ---------------------------------------------------------------------------


# Section-classification regex bank — mirrored from
# ingestion.pdf_parser to avoid an ingest-time import cost.
import re as _re

_SECTION_PRIMARY = "primary"
_SECTION_SECONDARY = "secondary"
_SECTION_UNKNOWN = "unknown"

_SECTION_RULES: list[tuple["_re.Pattern[str]", str]] = [
    (_re.compile(r"(?im)^\s*abstract\b[ \t]*[:.]?[ \t]*$"), _SECTION_PRIMARY),
    (_re.compile(r"(?im)^\s*results?\b[ \t]*[:.]?[ \t]*$"), _SECTION_PRIMARY),
    (_re.compile(r"(?im)^\s*conclusions?\b[ \t]*[:.]?[ \t]*$"), _SECTION_PRIMARY),
    (_re.compile(r"(?im)^\s*background\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
    (_re.compile(r"(?im)^\s*introduction\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
    (_re.compile(
        r"(?im)^\s*(?:patients?|subjects?|materials?)\s+and\s+methods?\b"
        r"[ \t]*[:.]?[ \t]*$"
    ), _SECTION_SECONDARY),
    (_re.compile(r"(?im)^\s*methods?\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
    (_re.compile(r"(?im)^\s*discussion\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
    (_re.compile(r"(?im)^\s*references?\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
    (_re.compile(r"(?im)^\s*bibliography\b[ \t]*[:.]?[ \t]*$"), _SECTION_SECONDARY),
]


def _classify_sections(text: str) -> dict[str, str]:
    """Mark each character position with a section_type."""
    matches: list[tuple[int, int, str]] = []
    for pattern, label in _SECTION_RULES:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), label))
    matches.sort(key=lambda x: (x[0], -x[1]))

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    current = _SECTION_UNKNOWN
    for start, end, label in matches:
        if start < cursor:
            continue
        if start > cursor:
            spans.append((cursor, start, current))
        cursor = end
        current = label
    if cursor < len(text):
        spans.append((cursor, len(text), current))
    if not spans:
        spans.append((0, len(text), _SECTION_UNKNOWN))
    return {"spans": spans}


def _load_pdf_text(pdf_path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            chunks.append(page_text)
    return "\n\n".join(chunks)


def _split_with_sections(
    text: str, *, chunk_size: int, chunk_overlap: int, file_name: str
) -> list:
    """Chunk text via SentenceSplitter, then attach section_type metadata."""
    info = _classify_sections(text)
    spans = info["spans"]
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text, metadata={"file_name": file_name})],
        show_progress=False,
    )
    # For each chunk, find which span it falls into and stamp section_type.
    from types import SimpleNamespace

    out = []
    pos = 0
    for i, node in enumerate(nodes):
        chunk_text = node.get_content()
        # Position of this chunk in the original text — approx via running
        # offset. SentenceSplitter preserves order, so add cumulative len.
        end = pos + len(chunk_text)
        # Locate span
        sec = _SECTION_UNKNOWN
        for s_start, s_end, s_label in spans:
            if s_start <= pos < s_end or (pos < s_end and end > s_start):
                sec = s_label
                break
        meta = dict(node.metadata or {})
        meta["section_type"] = sec
        meta["file_name"] = file_name
        # Wrap as a NodeWithScore-like object the selector understands.
        wrapped_node = SimpleNamespace(
            metadata=meta,
            get_content=lambda t=chunk_text: t,
        )
        # Synthetic retrieval score: earlier chunks get a higher score.
        score = 1.0 - 0.005 * i
        hit = SimpleNamespace(node=wrapped_node, score=score)
        out.append(hit)
        pos = end
    return out


# ---------------------------------------------------------------------------
# 4) Per-paper A/B arm execution
# ---------------------------------------------------------------------------


def _run_arm(
    paper: HoldoutPaper,
    *,
    cfg,
    rerank_enabled: bool,
    top_k: int,
) -> dict:
    """Run one arm (rerank ON or OFF) for ``paper``. Returns JSON-serialisable dict."""
    started = time.monotonic()
    full_text = _load_pdf_text(paper.pdf_path)
    over_fetch_pool = _split_with_sections(
        full_text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        file_name=paper.canonical_file,
    )
    if not over_fetch_pool:
        raise RuntimeError(f"No chunks from {paper.pdf_path.name}")

    # Toggle the rerank switch
    set_rerank_enabled_default(rerank_enabled)

    # Apply rerank
    reranked = rerank_hits(
        over_fetch_pool,
        enabled=rerank_enabled,
        registry_lookup=None,  # holdout has no registry entries
    )
    top_k_chunks = select_top_k(
        reranked,
        top_k=top_k,
        canonical_file=paper.canonical_file,
    )

    used_chunks = [h.node.get_content() for h in top_k_chunks]
    chunk_file_names = [paper.canonical_file] * len(used_chunks)

    try:
        result: GRADEResult = grade_extraction(
            used_chunks,
            canonical_file_name=paper.canonical_file,
            chunk_file_names=chunk_file_names,
        )
        grade_dict = json.loads(result.model_dump_json())
        error = None
    except Exception as exc:
        grade_dict = None
        error = f"{type(exc).__name__}: {exc}"

    return {
        "rerank_enabled": rerank_enabled,
        "duration_s": round(time.monotonic() - started, 3),
        "n_over_fetch": len(over_fetch_pool),
        "n_top_k": len(top_k_chunks),
        "top_k_section_types": [
            (h.node.metadata.get("section_type", "?"))
            for h in top_k_chunks
        ],
        "top_k_first_50chars": [
            (h.node.get_content()[:50].replace("\n", " "))
            for h in top_k_chunks
        ],
        "grade": grade_dict,
        "error": error,
    }


# ---------------------------------------------------------------------------
# 5) Reporting
# ---------------------------------------------------------------------------


def _bucket(score: float) -> str:
    if score >= 8.5: return "高"
    if score >= 6.0: return "中"
    if score >= 3.5: return "低"
    return "极低"


def _summarize_one(paper: HoldoutPaper, arm_off: dict, arm_on: dict) -> dict:
    g_off = arm_off.get("grade") or {}
    g_on = arm_on.get("grade") or {}

    def _diff(field):
        a = g_off.get(field)
        b = g_on.get(field)
        return {"off": a, "on": b, "differs": a != b}

    return {
        "slug": paper.slug,
        "gold_expected_design": _GOLD[paper.slug].expected_design,
        "gold_expected_overall_grade": _GOLD[paper.slug].expected_overall_grade,
        "gold_rationale": _GOLD[paper.slug].rationale,
        "top_k_differs": arm_off["top_k_section_types"] != arm_on["top_k_section_types"],
        "off": {
            "study_design": g_off.get("study_design"),
            "overall_grade": g_off.get("overall_grade"),
            "composite_score": g_off.get("composite_score"),
            "bucket": _bucket(g_off.get("composite_score", 0)),
            "sample_size": g_off.get("sample_size"),
            "effect_size": g_off.get("effect_size"),
        },
        "on": {
            "study_design": g_on.get("study_design"),
            "overall_grade": g_on.get("overall_grade"),
            "composite_score": g_on.get("composite_score"),
            "bucket": _bucket(g_on.get("composite_score", 0)),
            "sample_size": g_on.get("sample_size"),
            "effect_size": g_on.get("effect_size"),
        },
        "design_match_off": g_off.get("study_design") == _GOLD[paper.slug].expected_design,
        "design_match_on":  g_on.get("study_design") == _GOLD[paper.slug].expected_design,
        "grade_match_off":  g_off.get("overall_grade") == _GOLD[paper.slug].expected_overall_grade,
        "grade_match_on":   g_on.get("overall_grade") == _GOLD[paper.slug].expected_overall_grade,
        "score_delta": round((g_on.get("composite_score", 0) or 0) - (g_off.get("composite_score", 0) or 0), 2),
    }


# ---------------------------------------------------------------------------
# 6) CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    if not cfg.effective_openai_api_key():
        print(
            "[rerank-ab] OPENAI_API_KEY 未配置 — 请在 config.yaml / env 中设置",
            file=sys.stderr,
        )
        return 2

    out_dir = _PROJECT_ROOT / "holdout" / "rerank_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    print("=" * 110)
    print("Holdout Rerank A/B control · ON vs OFF (10 trials = 5 papers × 2 arms)")
    print("=" * 110)

    for paper in _HOLDOUT_PAPERS:
        print(f"\n>>> {paper.slug}")
        arm_off = _run_arm(paper, cfg=cfg, rerank_enabled=False, top_k=12)
        arm_on  = _run_arm(paper, cfg=cfg, rerank_enabled=True,  top_k=12)
        row = _summarize_one(paper, arm_off, arm_on)
        summary_rows.append(row)

        # Compact per-paper report
        print(f"  top-K section types: OFF={arm_off['top_k_section_types']}")
        print(f"                         ON ={arm_on['top_k_section_types']}")
        print(f"  study_design     : OFF={row['off']['study_design']:<10}  ON={row['on']['study_design']:<10}  gold={row['gold_expected_design']}")
        print(f"  overall_grade    : OFF={row['off']['overall_grade']:<4}    ON={row['on']['overall_grade']:<4}    gold={row['gold_expected_overall_grade']}")
        print(f"  composite_score  : OFF={row['off']['composite_score']:>5}    ON={row['on']['composite_score']:>5}    bucket: OFF={row['off']['bucket']:<4}  ON={row['on']['bucket']}")
        print(f"  sample_size      : OFF={row['off']['sample_size']:<6}    ON={row['on']['sample_size']:<6}")
        print(f"  effect_size      : OFF={row['off']['effect_size']}")
        print(f"                    : ON ={row['on']['effect_size']}")
        print(f"  Δ score (ON-OFF) : {row['score_delta']:+.2f}")

        # Persist per-paper A/B JSON
        path = out_dir / f"{paper.slug}.json"
        path.write_text(
            json.dumps(
                {
                    "slug": paper.slug,
                    "title": paper.title,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "arm_off": arm_off,
                    "arm_on":  arm_on,
                    "summary": row,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Aggregate summary
    print()
    print("=" * 110)
    print("Aggregate — agreement with clinical gold")
    print("=" * 110)

    n = len(summary_rows)
    design_correct_off = sum(1 for r in summary_rows if r["design_match_off"])
    design_correct_on  = sum(1 for r in summary_rows if r["design_match_on"])
    grade_correct_off  = sum(1 for r in summary_rows if r["grade_match_off"])
    grade_correct_on   = sum(1 for r in summary_rows if r["grade_match_on"])
    topk_differs       = sum(1 for r in summary_rows if r["top_k_differs"])
    grade_changed      = sum(1 for r in summary_rows if r["off"]["overall_grade"] != r["on"]["overall_grade"])

    print(f"  top-K composition differs (ON vs OFF) : {topk_differs}/{n}")
    print(f"  overall_grade flipped (ON vs OFF)     : {grade_changed}/{n}")
    print()
    print(f"  study_design agreement with gold:")
    print(f"    OFF: {design_correct_off}/{n}  ({100*design_correct_off/n:.0f}%)")
    print(f"    ON : {design_correct_on }/{n}  ({100*design_correct_on /n:.0f}%)")
    print(f"  overall_grade agreement with gold:")
    print(f"    OFF: {grade_correct_off}/{n}  ({100*grade_correct_off/n:.0f}%)")
    print(f"    ON : {grade_correct_on }/{n}  ({100*grade_correct_on /n:.0f}%)")
    print()
    print(f"  composite_score Δ (ON − OFF) range: "
          f"min={min(r['score_delta'] for r in summary_rows):+.2f}, "
          f"max={max(r['score_delta'] for r in summary_rows):+.2f}")

    # Persist aggregate summary
    (out_dir / "_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_papers": n,
                "top_k_differs": topk_differs,
                "grade_flipped_between_arms": grade_changed,
                "design_accuracy_off": f"{design_correct_off}/{n}",
                "design_accuracy_on":  f"{design_correct_on }/{n}",
                "grade_accuracy_off":  f"{grade_correct_off}/{n}",
                "grade_accuracy_on":   f"{grade_correct_on }/{n}",
                "rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果输出：")
    print(f"  · 每篇 A/B JSON → {out_dir}/<slug>.json")
    print(f"  · 汇总        → {out_dir}/_summary.json")
    return 0


__all__ = ["HoldoutPaper", "_HOLDOUT_PAPERS", "_GOLD"]


if __name__ == "__main__":
    raise SystemExit(main())
