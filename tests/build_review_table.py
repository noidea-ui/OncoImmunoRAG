"""Build the master review spreadsheet for the OncoImmunoRAG evaluation.

The project indexes 12 papers under ``data/raw/``. The GRADE batch
evaluation (``tests/eval_batch_grade.py``) already produced
machine-graded reports for the 5 most-cited urothelial-cancer scenarios
(EV-302, NIAGARA, IMvigor011, KEYNOTE-057, RC48-C014); the remaining 7
papers have static metadata only.

This script merges both into a single Markdown / CSV / (optionally) Excel
review table the human reviewer can edit::

    poetry run python tests/build_review_table.py
    poetry run python tests/build_review_table.py --xlsx  # needs openpyxl

Output files land under ``tests/results/review/``:

* ``review_table.md`` — Markdown render (always)
* ``review_table.csv`` — CSV (always)
* ``review_table.xlsx`` — Excel (only when ``openpyxl`` is available)

The Markdown and CSV files include a trailing ``generated_at`` line in
the body footer so reviewers can see when the table was last rebuilt.
The Excel workbook intentionally **omits** that timestamp — openpyxl's
last-write timestamp is enough metadata, and the trailing cells were
cluttering the print area.

Schema (10 columns):

1. 方案名称 (slug)
2. 核心问题 (the question fed to the retriever / grader)
3. 试验类型 (RCT Phase 3 / RCT Phase 2 / 单臂 Phase 2 / …)
4. 检索召回精准度 (Top-K contains original — 是/否/待评估)
5. AI 初始等级 (RCT → 高, 单臂 → 低)
6. 降级分析判定 (rule-engine trace)
7. AI 最终 GRADE 等级 (高 / 中 / 低 / 极低)
8. 人工金标准等级 (filled in by reviewer — starts as "待人工复核")
9. 是否存在数字幻觉 (是/否/待评估)
10. 备注 (extra notes — rule-engine rationale, free-form)

Static metadata for the trial registry overrides (RC48-C014,
McGregor-DAD) is sourced from :mod:`utils.trial_registry` so the
"truth" lives in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow ``python tests/build_review_table.py`` to run without install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Trial registry overrides. Imported lazily so a missing src/ doesn't
# crash the smoke path; the table build is a thin renderer.
try:
    from utils.trial_registry import TRIAL_REGISTRY, lookup  # noqa: E402
except ImportError:  # pragma: no cover — only fires when src/ is absent
    TRIAL_REGISTRY = {}
    def lookup(_fn: str):  # type: ignore[no-redef]
        return None


# ---------------------------------------------------------------------------
# Static catalogue of the 12 papers + their core questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperMeta:
    """Static metadata for one PDF — independent of any evaluation run."""

    slug: str
    file_name: str
    trial_name: str
    question: str
    trial_type: str
    nct_id: str = ""
    notes: str = ""


def _meta_from_registry(slug: str, fallback_trial_name: str, fallback_type: str) -> tuple[str, str]:
    """Read trial_name / trial_type from the registry when available.

    Falls back to the hard-coded value when the slug isn't in the
    registry (covers EV-302 / NIAGARA / IMvigor011 / KEYNOTE-057, whose
    canonical attributes the LLM gets right).
    """
    # Walk the registry and find the entry whose trial_name matches the
    # fallback hint, OR just use the slug itself.
    for fn, info in TRIAL_REGISTRY.items():
        if slug in fn or fn.startswith(slug):
            return info.trial_name, info.trial_type
    return fallback_trial_name, fallback_type


# Order is preserved in the output table.
PAPERS: tuple[PaperMeta, ...] = (
    PaperMeta(
        slug="EV-302",
        file_name="NEJMoa2312117.pdf",
        trial_name="Enfortumab Vedotin + Pembrolizumab",
        question="EV+P 一线治疗未治局部晚期/转移性尿路上皮癌的疗效和安全性",
        trial_type="RCT (Phase 3)",
        nct_id="NCT04223856",
    ),
    PaperMeta(
        slug="NIAGARA",
        file_name="Powles_2024_NEJM_NIAGARA-Durvalumab.pdf",
        trial_name="Durvalumab 围手术期 + 化疗",
        question="Durvalumab 围手术期(新辅助+辅助)联合吉西他滨顺铂化疗治疗 MIBC 的疗效",
        trial_type="RCT (Phase 3)",
        nct_id="NCT03732677",
    ),
    PaperMeta(
        slug="IMvigor011",
        file_name="Powles_2025_NEJM_IMvigor011-ctDNA.pdf",
        trial_name="IMvigor011 (ctDNA 指导 Atezolizumab)",
        question="基于 ctDNA 指导的 Atezolizumab 辅助治疗 MIBC 根治术后患者的疗效",
        trial_type="RCT (Phase 3)",
        nct_id="NCT04660344",
    ),
    PaperMeta(
        slug="KEYNOTE-057",
        file_name="Balar_2024_LancetOncol_KEYNOTE-057.pdf",
        trial_name="Pembrolizumab 单药 (BCG 无应答 NMIBC)",
        question="Pembrolizumab 单药治疗 BCG 无应答的高危 NMIBC 的疗效和安全性",
        trial_type="单臂 (Phase 2, Cohort B)",
        nct_id="NCT02625961",
    ),
    PaperMeta(
        slug="RC48-C014",
        file_name="Zhou_2025_AnnOncol_RC48-C014.pdf",
        # Trial name + type are hydrated from utils.trial_registry by
        # :func:`_hydrate_papers` — single source of truth.
        trial_name="(registry)",
        question="维迪西妥单抗(Disitamab Vedotin) 联合特瑞普利单抗 治疗晚期/转移性 UC 的疗效",
        trial_type="(registry)",
        nct_id="NCT04264936",
    ),
    PaperMeta(
        slug="Nadofaragene",
        file_name="Boorjian_2021_LancetOncol_Nadofaragene.pdf",
        trial_name="Nadofaragene (rAd-IFNα/Syn3)",
        question="Nadofaragene 膀胱灌注治疗 BCG 无应答 NMIBC 的完全缓解率与持续时间",
        trial_type="单臂 (Phase 3)",
        nct_id="NCT02773849",
    ),
    PaperMeta(
        slug="IMvigor130",
        file_name="Galsky_2024_CellRepMed_IMvigor130-ChemoImmuno.pdf",
        trial_name="IMvigor130 (Atezolizumab + 含铂化疗)",
        question="Atezolizumab 联合含铂化疗一线治疗转移性尿路上皮癌的疗效与安全性",
        trial_type="RCT (Phase 3)",
        nct_id="NCT02807636",
    ),
    PaperMeta(
        slug="CORE-001",
        file_name="Li_2025_NatMed_Cretostimogene-Nivo.pdf",
        trial_name="Cretostimogene + Nivolumab (CORE-001)",
        question="Cretostimogene Grenadenorepvec 联合 Nivolumab 治疗 BCG-naive 高危 NMIBC",
        trial_type="单臂 (Phase 2)",
        nct_id="NCT04387461",
    ),
    PaperMeta(
        slug="McGregor-DAD",
        file_name="McGregor_2024_AnnOncol_DAD.pdf",
        # Hydrated from utils.trial_registry.
        trial_name="(registry)",
        question="DAD Trial: SG + EV 双 ADC 联合治疗晚期尿路上皮癌的疗效与安全性",
        trial_type="(registry)",
    ),
    PaperMeta(
        slug="VESPER",
        file_name="Pfister_2024_LancetOncol_VESPER.pdf",
        trial_name="VESPER (DD-MVAC vs GC 围手术期)",
        question="DD-MVAC 与 GC 围手术期化疗治疗 MIBC 的 3 年 PFS / OS",
        trial_type="RCT (Phase 3)",
        nct_id="NCT01812369",
    ),
    PaperMeta(
        slug="IMvigor010",
        file_name="Powles_2021_Nature_IMvigor010-ctDNA.pdf",
        trial_name="IMvigor010 (Atezolizumab 辅助)",
        question="Atezolizumab 辅助治疗 MIBC 根治术后高危患者的 DFS / OS",
        trial_type="RCT (Phase 3)",
        nct_id="NCT02450331",
    ),
    PaperMeta(
        slug="ABACUS",
        file_name="Szabados_2022_EurUrol_ABACUS.pdf",
        trial_name="ABACUS (Atezolizumab 新辅助)",
        question="Atezolizumab 新辅助治疗 MIBC 的 pCR 率与病理学降期",
        trial_type="单臂 (Phase 2)",
        nct_id="NCT02662309",
    ),
)


def _hydrate_papers() -> tuple[PaperMeta, ...]:
    """Fill in registry-driven fields (RC48-C014, McGregor-DAD)."""
    out: list[PaperMeta] = []
    for p in PAPERS:
        info = lookup(p.file_name)
        if info is None:
            out.append(p)
            continue
        # Carry through the registry's authoritative metadata.
        out.append(
            PaperMeta(
                slug=p.slug,
                file_name=p.file_name,
                trial_name=info.trial_name or p.trial_name,
                question=p.question,
                trial_type=info.trial_type,
                nct_id=info.nct_id or p.nct_id,
                notes=info.notes or p.notes,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Evaluation JSON ingestion
# ---------------------------------------------------------------------------


_GRADE_DIR = _PROJECT_ROOT / "tests" / "results" / "grade_eval"


@dataclass
class GradeRow:
    """One row of the merged review table."""

    slug: str
    question: str
    trial_type: str
    retrieval_precision: str
    ai_initial_grade: str
    downgrade_analysis: str
    ai_final_grade: str
    human_gold_standard: str
    numerical_hallucination: str
    notes: str
    # Audit fields, not part of the printed table:
    sample_size: int = 0
    effect_size: str = ""
    rule_trace: list[str] = field(default_factory=list)
    n_evaluated_at: str = ""


# Rule-engine downgrades are surfaced in the JSON ``grade_reasoning`` text
# between the markers ``[降级规则引擎]`` and ``最终等级 =``. Capture that
# trace so reviewers can see *why* the model landed where it did.
_RULE_TRACE_RE = re.compile(
    r"\[降级规则引擎\]\s*(?P<trace>(?:\s*-\s*[^\n]+\n?)+)",
    re.MULTILINE,
)


def _initial_grade(study_design: str | None) -> str:
    """Baseline GRADE before any downgrade rules fire.

    Mirrors :func:`grading.grade_evaluator.apply_downgrade_rules`.
    """
    if study_design in ("RCT", "系统评价", "Meta分析"):
        return "高"
    if study_design in ("观察性", "单臂试验"):
        return "低"
    return "(待评估)"


def _summarise_rule_trace(reasoning: str | None) -> tuple[str, list[str]]:
    """Return ``(short_label, full_trace)``.

    Short label: ``"无触发降级"`` when no downgrades fire, otherwise
    ``"触发 N 项降级: …"`` with the first 3 reasons.

    The trace recorded in the JSON includes:
      * the baseline assignment (``基线等级 by study_design=...: ...``)
      * zero or more ``−1`` rule firings
      * a closing ``最终等级 = ...`` line
    We strip the baseline and the closing line from the *short label* so
    that "基线 = 高 → 最终 = 高" doesn't read as "1 downgrade fired".
    The full trace (including baseline) is preserved in the
    ``## 降级规则链路`` section of the Markdown output.
    """
    if not reasoning:
        return "", []
    m = _RULE_TRACE_RE.search(reasoning)
    if not m:
        return "", []
    raw = m.group("trace").strip()
    lines = [ln.strip().lstrip("-").strip() for ln in raw.splitlines() if ln.strip()]
    # Drop the closing "最终等级 = …" line from the trace view.
    trace = [ln for ln in lines if not ln.startswith("最终等级")]
    # A "downgrade firing" is any line with the explicit "→ −1" suffix.
    fired = [ln for ln in trace if ln.endswith("−1")]
    if not fired:
        return "无触发降级", trace
    short = "; ".join(_trim_reason(r) for r in fired[:3])
    return f"触发 {len(fired)} 项降级: {short}", trace


def _trim_reason(reason: str) -> str:
    """Shorten a rule-engine reason line for the table view."""
    s = re.sub(r"\s*→\s*−1\s*$", "", reason).strip()
    return s


def _retrieve_precision(grade: dict, retrieved_chunks: list[dict], file_name: str) -> str:
    """Heuristic: did the LLM's reported effect_size match what the paper says?

    We can't fully verify without ground truth, so the heuristic is:
    a chunk tagged ``section_type="primary"`` from the source PDF must be
    in the top-K.
    """
    if not retrieved_chunks:
        return "待评估"
    primary_in_topk = any(
        (c.get("file_name") == file_name) and (c.get("section_type") == "primary")
        for c in retrieved_chunks
    )
    return "是" if primary_in_topk else "否(primary 不在 top-K)"


def _detect_hallucination(grade: dict, effect_size: str, paper_file: str) -> str:
    """Heuristic: any obvious fabrication marker?

    Pure LLM outputs are hard to audit without ground truth. The most
    reliable red flags we can detect from the JSON alone are:
      * effect_size is empty / "未提及"
      * sample_size is 0
    Mark "待评估" otherwise — the human reviewer owns this column.
    """
    if not grade:
        return "待评估"
    if grade.get("sample_size", 0) <= 0:
        return "待评估(样本量为 0)"
    if not effect_size or effect_size == "未提及":
        return "待评估(effect_size 未提及)"
    return "待评估(需人工核对)"


def _load_grade_row(meta: PaperMeta) -> GradeRow:
    """Build one GradeRow by merging PaperMeta with the per-scenario JSON."""
    grade_path = _GRADE_DIR / f"{meta.slug}.json"
    if not grade_path.is_file():
        return GradeRow(
            slug=meta.slug,
            question=meta.question,
            trial_type=meta.trial_type,
            retrieval_precision="待评估",
            ai_initial_grade="(待评估)",
            downgrade_analysis="(待评估)",
            ai_final_grade="(待评估)",
            human_gold_standard="待人工复核",
            numerical_hallucination="待评估",
            notes=meta.notes or "",
        )

    payload = json.loads(grade_path.read_text(encoding="utf-8"))
    grade = payload.get("grade") or {}
    retrieved = payload.get("retrieved_chunks") or []
    short_label, full_trace = _summarise_rule_trace(grade.get("grade_reasoning", ""))
    precision = _retrieve_precision(grade, retrieved, meta.file_name)
    hallucination = _detect_hallucination(
        grade, grade.get("effect_size", ""), meta.file_name
    )

    return GradeRow(
        slug=meta.slug,
        question=meta.question,
        trial_type=meta.trial_type,
        retrieval_precision=precision,
        ai_initial_grade=_initial_grade(grade.get("study_design")),
        downgrade_analysis=short_label or "(待评估)",
        ai_final_grade=grade.get("overall_grade") or "(待评估)",
        human_gold_standard="待人工复核",
        numerical_hallucination=hallucination,
        notes=meta.notes or "",
        sample_size=grade.get("sample_size", 0) or 0,
        effect_size=grade.get("effect_size", "") or "",
        rule_trace=full_trace,
        n_evaluated_at=payload.get("started_at", ""),
    )


# ---------------------------------------------------------------------------
# Overrides (preserves reviewer edits across re-runs)
# ---------------------------------------------------------------------------


_OVERRIDES_PATH = _PROJECT_ROOT / "tests" / "results" / "review" / "review_overrides.json"


def _load_overrides() -> dict[str, dict]:
    if not _OVERRIDES_PATH.is_file():
        return {}
    try:
        return json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _apply_overrides(rows: list[GradeRow]) -> None:
    """In-place: copy reviewer-edited columns from overrides JSON."""
    overrides = _load_overrides()
    for row in rows:
        o = overrides.get(row.slug, {})
        for col in ("human_gold_standard", "numerical_hallucination", "notes"):
            if col in o and o[col] not in (None, ""):
                setattr(row, col, str(o[col]))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


_OUTPUT_DIR = _PROJECT_ROOT / "tests" / "results" / "review"


def _ensure_out_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


_TABLE_HEADERS: tuple[str, ...] = (
    "方案名称",
    "核心问题",
    "试验类型",
    "检索召回精准度 (Top-K 含原文?)",
    "AI 初始等级",
    "降级分析判定",
    "AI 最终 GRADE",
    "人工金标准等级",
    "是否存在数字幻觉",
    "备注",
)


def _format_markdown(rows: list[GradeRow], generated_at: str) -> str:
    """Render the merged table as Markdown."""
    lines: list[str] = [
        "# OncoImmunoRAG 评测复核总表",
        "",
        f"_Generated at: {generated_at}_  ",
        f"_Rows: {len(rows)} · Auto-filled from `tests/results/grade_eval/*.json` + static metadata._  ",
        "_Empty / 待评估 cells mean the per-paper evaluation hasn't run yet._",
        "",
        "| " + " | ".join(_TABLE_HEADERS) + " |",
        "|" + "|".join([" --- "] * len(_TABLE_HEADERS)) + "|",
    ]
    for r in rows:
        cells = [
            f"`{r.slug}`",
            r.question,
            r.trial_type,
            r.retrieval_precision,
            r.ai_initial_grade,
            r.downgrade_analysis,
            f"**{r.ai_final_grade}**" if r.ai_final_grade not in ("", "(待评估)") else r.ai_final_grade,
            r.human_gold_standard,
            r.numerical_hallucination,
            r.notes or "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 降级规则链路(自动采集)")
    for r in rows:
        if not r.rule_trace:
            continue
        lines.append("")
        lines.append(f"### `{r.slug}`")
        for line in r.rule_trace:
            lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def _write_markdown(rows: list[GradeRow], generated_at: str) -> Path:
    out = _OUTPUT_DIR / "review_table.md"
    out.write_text(_format_markdown(rows, generated_at), encoding="utf-8")
    return out


def _write_csv(rows: list[GradeRow], generated_at: str) -> Path:
    """Write the CSV. Header + N data rows only; no trailing timestamp row.

    The ``generated_at`` timestamp is exposed via the Markdown body
    footer so the CSV stays a clean, machine-parseable 12-row table.
    """
    out = _OUTPUT_DIR / "review_table.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_TABLE_HEADERS)
        for r in rows:
            writer.writerow([
                r.slug,
                r.question,
                r.trial_type,
                r.retrieval_precision,
                r.ai_initial_grade,
                r.downgrade_analysis,
                r.ai_final_grade,
                r.human_gold_standard,
                r.numerical_hallucination,
                r.notes,
            ])
    # Stash generated_at as a sidecar so downstream tools can find it
    # without polluting the table body.
    (_OUTPUT_DIR / "_generated_at.txt").write_text(
        generated_at, encoding="utf-8"
    )
    return out


def _write_xlsx(rows: list[GradeRow], generated_at: str) -> Optional[Path]:
    """Write a clean Excel workbook — no trailing timestamp row.

    The previous build appended a row ``len(rows) + 3`` with the
    ``generated_at`` timestamp; that polluted the print area and made
    the spreadsheet look like 13-15 data rows instead of 12. Now the
    timestamp lives in the workbook's built-in property (openpyxl sets
    it on save) and in a dedicated ``_generated_at.txt`` sidecar.
    """
    try:
        from openpyxl import Workbook  # type: ignore[import-not-found]
        from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore[import-not-found]
    except ImportError:
        return None

    out = _OUTPUT_DIR / "review_table.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "review_table"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")
    wrap = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_idx, h in enumerate(_TABLE_HEADERS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    for row_idx, r in enumerate(rows, start=2):
        values = [
            r.slug,
            r.question,
            r.trial_type,
            r.retrieval_precision,
            r.ai_initial_grade,
            r.downgrade_analysis,
            r.ai_final_grade,
            r.human_gold_standard,
            r.numerical_hallucination,
            r.notes,
        ]
        for col_idx, v in enumerate(values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=v)
            cell.alignment = wrap

    # Reasonable column widths.
    widths = [16, 56, 24, 18, 12, 40, 14, 18, 18, 40]
    for col_idx, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + col_idx)].width = w

    # Freeze the header so it stays visible when the reviewer scrolls.
    ws.freeze_panes = "A2"

    # Side sheet with the rule-engine trace.
    ws2 = wb.create_sheet("rule_traces")
    ws2.append(["slug", "rule_line"])
    for r in rows:
        for line in r.rule_trace:
            ws2.append([r.slug, line])
    ws2.column_dimensions["A"].width = 16
    ws2.column_dimensions["B"].width = 80

    # Set the workbook's own properties so reviewers can see when the
    # file was generated without us cluttering any sheet.
    wb.properties.title = "OncoImmunoRAG 评测复核总表"
    wb.properties.subject = "GRADE review for 12 urothelial-cancer trials"
    wb.properties.description = (
        f"Auto-generated from tests/results/grade_eval/*.json + "
        f"utils.trial_registry at {generated_at}."
    )
    wb.properties.keywords = "GRADE, urothelial carcinoma, immunotherapy"

    wb.save(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xlsx",
        action="store_true",
        help="Also write review_table.xlsx (requires openpyxl).",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print the Markdown table to stdout in addition to writing files.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_out_dir()

    papers = _hydrate_papers()
    rows = [_load_grade_row(m) for m in papers]
    _apply_overrides(rows)

    generated_at = datetime.now(timezone.utc).isoformat()

    md_path = _write_markdown(rows, generated_at)
    csv_path = _write_csv(rows, generated_at)

    xlsx_msg = ""
    if args.xlsx:
        xlsx_path = _write_xlsx(rows, generated_at)
        xlsx_msg = (
            f"\n  · review_table.xlsx → {xlsx_path}"
            if xlsx_path is not None
            else "\n  · openpyxl not installed — skipped .xlsx output"
        )

    if args.print:
        print(_format_markdown(rows, generated_at))

    n_evaluated = sum(1 for r in rows if r.ai_final_grade not in ("", "(待评估)"))
    print(
        f"[review_table] {len(rows)} rows · {n_evaluated} with auto-graded GRADE.\n"
        f"  · review_table.md  → {md_path}\n"
        f"  · review_table.csv → {csv_path}"
        f"{xlsx_msg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
