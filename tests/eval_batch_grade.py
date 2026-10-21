"""Batch GRADE evaluation across canonical oncology-immunotherapy scenarios.

Iterates a hand-picked list of 5 clinical scenarios, retrieves the top-K
chunks for each, runs :func:`grading.grade_evaluator.grade_extraction` over
them, and emits:

* one JSON file per scenario under ``tests/results/grade_eval/<slug>.json``
* a Markdown summary table printed to stdout **and** written to
  ``tests/results/grade_eval/_summary.md``
* a combined ``tests/results/grade_eval/_summary.json`` for downstream diffing

Run with::

    poetry run python tests/eval_batch_grade.py
    # or, for a quick smoke test that skips the LLM call:
    poetry run python tests/eval_batch_grade.py --dry-run

Exit codes:
    0  every scenario completed (grade captured or --dry-run)
    2  user error (bad slug, missing API key, unknown config)
    3  one or more scenarios errored at the LLM stage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# Allow ``poetry run python tests/eval_batch_grade.py`` without installing
# the package. ``src`` is on PYTHONPATH via ``[tool.pytest.ini_options]`` for
# pytest; mirror it here for direct invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import dataclass  # noqa: E402

from grading.grade_evaluator import GRADEResult, grade_extraction  # noqa: E402
from retrieval.query_engine import QueryEngine  # noqa: E402
from utils.config_loader import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One canonical clinical question to grade."""

    slug: str
    title: str
    query: str
    canonical_file: str = ""
    extra_terms: tuple[str, ...] = ()


_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        slug="EV-302",
        title="Enfortumab Vedotin + Pembrolizumab 一线治疗未治局部晚期/转移性尿路上皮癌",
        query=(
            "enfortumab vedotin pembrolizumab EV-302 untreated advanced urothelial cancer"
        ),
        canonical_file="NEJMoa2312117.pdf",
        extra_terms=("EV-302", "NCT04223856"),
    ),
    Scenario(
        slug="NIAGARA",
        title="Durvalumab 围手术期联合化疗治疗肌层浸润性膀胱癌(MIBC)",
        query=(
            "durvalumab neoadjuvant chemotherapy NIAGARA muscle-invasive bladder cancer"
        ),
        canonical_file="Powles_2024_NEJM_NIAGARA-Durvalumab.pdf",
        extra_terms=("NiAGARA", "NCT03732677"),
    ),
    Scenario(
        slug="IMvigor011",
        title="ctDNA 指导的 Atezolizumab 辅助治疗 MIBC",
        query=(
            "ctDNA atezolizumab adjuvant IMvigor011 muscle-invasive bladder cancer"
        ),
        canonical_file="Powles_2025_NEJM_IMvigor011-ctDNA.pdf",
        extra_terms=("IMvigor011", "ctDNA", "NCT04660344"),
    ),
    Scenario(
        slug="KEYNOTE-057",
        title="Pembrolizumab 单药治疗 BCG 无应答高危 NMIBC",
        query=(
            "pembrolizumab BCG unresponsive high-risk KEYNOTE-057 "
            "non-muscle-invasive bladder cancer"
        ),
        canonical_file="Balar_2024_LancetOncol_KEYNOTE-057.pdf",
        extra_terms=("KEYNOTE-057", "NCT02625961"),
    ),
    Scenario(
        slug="RC48-C014",
        title="维迪西妥单抗联合特瑞普利单抗治疗晚期/转移性尿路上皮癌",
        query=(
            "disitamab vedotin toripalimab RC48-C014 advanced urothelial carcinoma"
        ),
        canonical_file="Zhou_2025_AnnOncol_RC48-C014.pdf",
        extra_terms=("RC48-C014", "维迪西妥单抗", "特瑞普利单抗"),
    ),
    Scenario(
        slug="Nadofaragene",
        title="Nadofaragene 治疗 BCG 无应答 NMIBC",
        query=(
            "intravesical nadofaragene firadenovec gene therapy BCG-unresponsive NMIBC"
        ),
        canonical_file="Boorjian_2021_LancetOncol_Nadofaragene.pdf",
        extra_terms=("Nadofaragene", "rAd-IFN", "NCT02773849"),
    ),
    Scenario(
        slug="IMvigor130",
        title="IMvigor130: Atezolizumab + 含铂化疗 1L 转移性 UC",
        query=(
            "cisplatin vs carboplatin atezolizumab chemotherapy IMvigor130 urothelial cancer"
        ),
        canonical_file="Galsky_2024_CellRepMed_IMvigor130-ChemoImmuno.pdf",
        extra_terms=("IMvigor130", "NCT02807636"),
    ),
    Scenario(
        slug="CORE-001",
        title="Cretostimogene + Nivolumab 治疗 BCG-naive 高危 NMIBC",
        query=(
            "cretostimogene grenadenorepvec oncolytic immunotherapy nivolumab MIBC"
        ),
        canonical_file="Li_2025_NatMed_Cretostimogene-Nivo.pdf",
        extra_terms=("CORE-001", "Cretostimogene", "NCT04387461"),
    ),
    Scenario(
        slug="McGregor-DAD",
        title="DAD Trial: SG + EV 双 ADC 联合治疗",
        query=(
            "sacituzumab govitecan enfortumab vedotin DAD trial "
            "metastatic urothelial carcinoma"
        ),
        canonical_file="McGregor_2024_AnnOncol_DAD.pdf",
        extra_terms=("DAD", "SG", "EV"),
    ),
    Scenario(
        slug="VESPER",
        title="VESPER: DD-MVAC vs GC 围手术期化疗 MIBC",
        query=(
            "dose-dense MVAC vs GC perioperative chemotherapy VESPER MIBC"
        ),
        canonical_file="Pfister_2024_LancetOncol_VESPER.pdf",
        extra_terms=("VESPER", "DD-MVAC", "NCT01812369"),
    ),
    Scenario(
        slug="IMvigor010",
        title="IMvigor010: Atezolizumab 辅助治疗 MIBC",
        query=(
            "ctDNA guiding adjuvant atezolizumab immunotherapy "
            "IMvigor010 urothelial carcinoma"
        ),
        canonical_file="Powles_2021_Nature_IMvigor010-ctDNA.pdf",
        extra_terms=("IMvigor010", "NCT02450331"),
    ),
    Scenario(
        slug="ABACUS",
        title="ABACUS: Atezolizumab 新辅助治疗 MIBC",
        query=(
            "neoadjuvant atezolizumab cisplatin-ineligible ABACUS "
            "muscle-invasive bladder cancer"
        ),
        canonical_file="Szabados_2022_EurUrol_ABACUS.pdf",
        extra_terms=("ABACUS", "NCT02662309"),
    ),
)


# ---------------------------------------------------------------------------
# Retrieval + grading helpers
# ---------------------------------------------------------------------------


_DEFAULT_TOP_K = 5
_RESULTS_DIR = _PROJECT_ROOT / "tests" / "results" / "grade_eval"


def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _build_engine() -> QueryEngine:
    cfg = load_config()
    return QueryEngine.from_config(cfg)


def _retrieve_texts(
    engine: QueryEngine,
    query: str,
    top_k: int,
    *,
    canonical_file: str | None = None,
) -> tuple[list[str], list[dict]]:
    """Return (chunk_texts, per-chunk metadata dicts) for a query.

    Re-uses the retriever's section-aware diversity logic, then strips back to
    just the prose + light metadata for the grader. When ``canonical_file`` is
    supplied, the selector force-promotes chunks from that file into top-K
    (the "target-aware / canonical routing" fix).
    """
    hits = engine.retrieve(query, canonical_file=canonical_file)
    texts = [h.node.get_content() for h in hits]
    meta = [
        {
            "file_name": (h.node.metadata or {}).get("file_name"),
            "chunk_id": (h.node.metadata or {}).get("chunk_id"),
            "section_type": (h.node.metadata or {}).get("section_type"),
            "score": getattr(h, "score", None),
        }
        for h in hits
    ]
    assert len(texts) == top_k, (
        f"retriever returned {len(texts)} hits, expected {top_k} — check "
        "section_type filtering or vector store size."
    )
    return texts, meta


def _grade_one(
    scenario: Scenario,
    *,
    top_k: int,
    dry_run: bool,
    engine: QueryEngine | None = None,
) -> dict:
    """Run the full pipeline for one scenario. Returns a JSON-serialisable dict."""
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    if engine is None:
        engine = _build_engine()

    texts, chunk_meta = _retrieve_texts(
        engine, scenario.query, top_k=top_k,
        canonical_file=scenario.canonical_file or None,
    )

    if dry_run:
        result_dict: dict = {
            "scenario": scenario.slug,
            "title": scenario.title,
            "query": scenario.query,
            "started_at": started_at,
            "duration_s": round(time.monotonic() - started, 3),
            "dry_run": True,
            "retrieved_chunks": chunk_meta,
            "chunk_texts": texts,
            "grade": None,
            "error": None,
        }
        return result_dict

    try:
        chunk_file_names = [m.get("file_name") for m in chunk_meta]
        grade: GRADEResult = grade_extraction(
            texts,
            canonical_file_name=scenario.canonical_file,
            chunk_file_names=chunk_file_names,
        )
        grade_dict = json.loads(grade.model_dump_json())
        error: str | None = None
    except Exception as exc:  # pragma: no cover — propagate for the summary
        grade_dict = None
        error = f"{type(exc).__name__}: {exc}"

    return {
        "scenario": scenario.slug,
        "title": scenario.title,
        "query": scenario.query,
        "started_at": started_at,
        "duration_s": round(time.monotonic() - started, 3),
        "dry_run": False,
        "retrieved_chunks": chunk_meta,
        "canonical_file": scenario.canonical_file,
        "grade": grade_dict,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(rows: Sequence[dict]) -> None:
    bar = "=" * 78
    print()
    print(bar)
    print(f"{'GRADE batch evaluation summary':^78}")
    print(bar)
    print(f"{'scenario':<14} {'grade':<6} {'design':<10} {'N':>5}  {'effect_size':<32}  status")
    print("-" * 78)
    for row in rows:
        g = row.get("grade") or {}
        effect = (g.get("effect_size") or "—")[:30]
        print(
            f"{row['scenario']:<14} "
            f"{(g.get('overall_grade') or 'ERR'):<6} "
            f"{(g.get('study_design') or '—'):<10} "
            f"{g.get('sample_size', 0):>5}  "
            f"{effect:<32}  "
            f"{'OK' if not row.get('error') else row['error'][:30]}"
        )
    print(bar)
    print(f"Total scenarios : {len(rows)}")
    print(f"Successful       : {sum(1 for r in rows if not r.get('error'))}")
    print(f"Errors           : {sum(1 for r in rows if r.get('error'))}")
    print(f"Output dir       : {_RESULTS_DIR}")
    print(bar)


def _build_markdown_summary(rows: Sequence[dict], generated_at: str) -> str:
    """Render the summary table as Markdown so it survives shell scrollback."""
    lines: list[str] = [
        "# GRADE batch evaluation summary",
        "",
        f"_Generated at: {generated_at}_",
        "",
        "| scenario | title | grade | design | N | effect_size | status |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        g = row.get("grade") or {}
        status = "OK" if not row.get("error") else f"ERROR: {row['error']}"
        lines.append(
            f"| `{row['scenario']}` "
            f"| {row['title']} "
            f"| **{g.get('overall_grade') or 'ERR'}** "
            f"| {g.get('study_design') or '—'} "
            f"| {g.get('sample_size', 0)} "
            f"| {g.get('effect_size') or '—'} "
            f"| {status} |"
        )
    lines += [
        "",
        f"- Total scenarios: **{len(rows)}**",
        f"- Successful: **{sum(1 for r in rows if not r.get('error'))}**",
        f"- Errors: **{sum(1 for r in rows if r.get('error'))}**",
        f"- Output dir: `{_RESULTS_DIR}`",
        "",
    ]
    return "\n".join(lines)


def _write_markdown_summary(rows: Sequence[dict], generated_at: str) -> Path:
    md_path = _RESULTS_DIR / "_summary.md"
    md_path.write_text(
        _build_markdown_summary(rows, generated_at), encoding="utf-8"
    )
    return md_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k", type=int, default=_DEFAULT_TOP_K,
        help=f"Chunks to retrieve per scenario (default {_DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run a single scenario by slug (repeatable). Default: all 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM grading call; only exercise retrieval + persistence.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Filter scenarios if user asked for a subset.
    selected = _SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        selected = tuple(s for s in _SCENARIOS if s.slug in wanted)
        missing = wanted - {s.slug for s in selected}
        if missing:
            print(f"[eval_batch_grade] unknown slug(s): {sorted(missing)}", file=sys.stderr)
            return 2

    if not args.dry_run and not load_config().effective_openai_api_key():
        print(
            "[eval_batch_grade] OPENAI_API_KEY is empty — pass --dry-run to skip "
            "the LLM step, or set the key in config.yaml / env.",
            file=sys.stderr,
        )
        return 2

    _ensure_results_dir()
    engine = _build_engine() if not args.dry_run else None

    rows: list[dict] = []
    for scenario in selected:
        print(f"[eval_batch_grade] running scenario: {scenario.slug}")
        row = _grade_one(scenario, top_k=args.top_k, dry_run=args.dry_run, engine=engine)
        _write_json(_RESULTS_DIR / f"{scenario.slug}.json", row)
        rows.append(row)

    generated_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        _RESULTS_DIR / "_summary.json",
        {"generated_at": generated_at, "rows": rows},
    )
    md_path = _write_markdown_summary(rows, generated_at)
    print(f"[eval_batch_grade] markdown summary -> {md_path}")
    _print_summary(rows)

    # Surface partial failure via a non-zero exit code so CI / shell loops
    # can react without diffing stdout. ``2`` is reserved for user errors
    # (bad slug, missing key); use ``3`` for runtime / LLM failures.
    if any(r.get("error") for r in rows):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())