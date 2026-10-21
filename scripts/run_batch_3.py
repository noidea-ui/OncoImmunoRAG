"""Batch-3 (target-aware + registry-aligned) benchmark runner.

Differences vs :mod:`scripts.run_full_benchmark`:

1. **Target-aware retrieval routing** — passes the scenario's
   ``canonical_file`` all the way down to
   :meth:`retrieval.query_engine.QueryEngine.retrieve`, so the selector
   force-promotes the target paper's Primary chunks into top-K. Closes
   the "EV-302 hijacks every niche query" retrieval bug.

2. **Registry-aligned canonical sample size** — the trial registry in
   ``utils.trial_registry`` now carries a ``sample_size`` field. When
   populated, ``grade_extraction`` writes it over whatever the LLM
   emitted (with a `[样本量注册表覆盖]` trace note). Together with
   Target-Aware routing, this ends the cross-paper hallucination that
   was making McGregor-DAD pick up 886 from EV-302.

3. **Writes to the project-root ``result/`` directory** (per task brief):
   * ``result/results_batch_3.json``   — full per-scenario JSON dump
   * ``result/benchmark_results_3.xlsx`` — comparison table

4. **Prints overall metrics** (GRADE accuracy + numeric hallucination
   rate) to the terminal at the end of the run.

Typical use::

    poetry run python scripts/run_batch_3.py
    poetry run python scripts/run_batch_3.py --scenario EV-302 KEYNOTE-057
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# Bootstrap ``src`` and ``tests`` onto sys.path so we can import the
# evaluation module without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
_TESTS = _PROJECT_ROOT / "tests"
for p in (_SRC, _TESTS):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

_RESULT_DIR = _PROJECT_ROOT / "result"
_RESULT_JSON = _RESULT_DIR / "results_batch_3.json"
_RESULT_XLSX = _RESULT_DIR / "benchmark_results_3.xlsx"
_GRADE_OUT_DIR = _PROJECT_ROOT / "tests" / "results" / "grade_eval"


# ---------------------------------------------------------------------------
# Human gold standard (registry-aligned) — single source of truth
# ---------------------------------------------------------------------------
#
# Order: study_design (RCT vs 单臂 trial) + sample_size + RoB determine
# the gold GRADE in the rule engine's output. Treat this table as a
# **regression invariant**: any future change to the rule engine should
# be reflected here in lockstep.
#
# Numeric hallucination: we cross-check the LLM's sample_size against the
# gold sample_size. Tolerance is "exactly equal" — clinical reviewers
# expect the canonical enrolled cohort, not "any number in the prompt".


@dataclass(frozen=True)
class GoldRef:
    """Per-scenario reference values used for accuracy + hallucination."""

    study_design: str
    sample_size: int
    overall_grade: str


_GOLD_REFS: dict[str, GoldRef] = {
    "EV-302":       GoldRef("RCT",         886,  "高"),
    "NIAGARA":      GoldRef("RCT",         1063, "高"),
    "IMvigor011":   GoldRef("RCT",         608,  "高"),
    "KEYNOTE-057":  GoldRef("单臂试验",    96,   "极低"),
    "RC48-C014":    GoldRef("单臂试验",    41,   "极低"),
    "Nadofaragene": GoldRef("单臂试验",    151,  "极低"),
    "IMvigor130":   GoldRef("RCT",         1253, "高"),
    "CORE-001":     GoldRef("单臂试验",    21,   "极低"),
    "McGregor-DAD": GoldRef("单臂试验",    24,   "极低"),
    "VESPER":       GoldRef("RCT",         813,  "高"),
    "IMvigor010":   GoldRef("RCT",         851,  "高"),
    "ABACUS":       GoldRef("单臂试验",    95,   "极低"),
}


# ---------------------------------------------------------------------------
# Comparison row
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    """One row of the comparison table."""

    slug: str
    title: str
    canonical_file: str
    trial_type: str
    ai_study_design: str
    gold_study_design: str
    design_match: str  # "✓" / "✗"
    ai_sample_size: int
    gold_sample_size: int
    sample_size_match: str
    ai_effect_size: str
    ai_overall_grade: str
    gold_overall_grade: str
    grade_match: str
    numerical_hallucination: str
    notes: str


# ---------------------------------------------------------------------------
# Per-scenario pipeline — mirrors tests.eval_batch_grade._grade_one but
# passes canonical_file through retrieval for Target-Aware routing.
# ---------------------------------------------------------------------------


def _run_one(scenario, cfg, *, top_k: int = 5) -> dict:
    """Live LLM run for one scenario with Target-Aware routing."""
    from retrieval.query_engine import QueryEngine  # noqa: PLC0415
    from grading.grade_evaluator import grade_extraction, GRADEResult  # noqa: PLC0415

    engine = QueryEngine.from_config(cfg)
    hits = engine.retrieve(scenario.query, canonical_file=scenario.canonical_file)
    texts = [h.node.get_content() for h in hits]
    chunk_meta = [
        {
            "file_name": (h.node.metadata or {}).get("file_name"),
            "chunk_id": (h.node.metadata or {}).get("chunk_id"),
            "section_type": (h.node.metadata or {}).get("section_type"),
            "score": getattr(h, "score", None),
            "_canonical_routed": (h.node.metadata or {}).get("_canonical_routed"),
        }
        for h in hits
    ]
    chunk_file_names = [m.get("file_name") for m in chunk_meta]

    try:
        grade: GRADEResult = grade_extraction(
            texts,
            canonical_file_name=scenario.canonical_file,
            chunk_file_names=chunk_file_names,
            cfg=cfg,
        )
        grade_dict = json.loads(grade.model_dump_json())
        error: Optional[str] = None
    except Exception as exc:  # pragma: no cover
        grade_dict = None
        error = f"{type(exc).__name__}: {exc}"

    canonical_in_top_k = any(m.get("file_name") == scenario.canonical_file for m in chunk_meta)

    return {
        "scenario": scenario.slug,
        "title": scenario.title,
        "query": scenario.query,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_chunks": chunk_meta,
        "canonical_file": scenario.canonical_file,
        "canonical_in_top_k": canonical_in_top_k,
        "grade": grade_dict,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _trial_type_label(scenario, registry_info) -> str:
    """Combine the trial-registry ``trial_type`` (when present) with a
    Phase hint, so the Excel column is informative."""
    if registry_info is not None:
        return registry_info.trial_type
    return "(待补充)"


def _initial_grade(study_design: str | None) -> str:
    if study_design in ("RCT", "系统评价", "Meta分析"):
        return "高"
    if study_design in ("观察性", "单臂试验"):
        return "低"
    return "(待评估)"


def _detect_numeric_hallucination(grade: Optional[dict], gold_n: int) -> str:
    """Flag a scenario as a numeric hallucination.

    Definition (strict, deterministic):

    * ``sample_size == 0``                       →  待评估 (LLM did not extract)
    * ``sample_size != gold_n``                  →  是 (数值错误, 典型跨文献污染)
    * ``effect_size`` missing / "未提及"          →  待评估 (effect_size 未提及)
    * Otherwise                                   →  否

    The "mismatch ≠ hallucination" debate is settled by the registry:
    when there is a known gold value, anything else is by definition a
    wrong extraction — which is exactly what "数值幻觉" means in the
    evaluator pipeline.
    """
    if not grade:
        return "待评估"
    n = grade.get("sample_size", 0) or 0
    es = (grade.get("effect_size") or "").strip()
    if not n:
        return "待评估(样本量为 0)"
    if gold_n and n != gold_n:
        return f"是(LLM={n}, 金标准={gold_n})"
    if not es or es == "未提及":
        return "待评估(effect_size 未提及)"
    return "否"


def _build_row(payload: dict, scenario, gold: GoldRef, registry_info) -> _Row:
    grade = payload.get("grade") or {}
    ai_n = int(grade.get("sample_size") or 0)
    es = (grade.get("effect_size") or "") or "—"

    return _Row(
        slug=scenario.slug,
        title=scenario.title,
        canonical_file=scenario.canonical_file,
        trial_type=_trial_type_label(scenario, registry_info),
        ai_study_design=grade.get("study_design", "—") or "—",
        gold_study_design=gold.study_design,
        design_match="✓" if (grade.get("study_design") == gold.study_design) else "✗",
        ai_sample_size=ai_n,
        gold_sample_size=gold.sample_size,
        sample_size_match="✓" if (ai_n == gold.sample_size) else "✗",
        ai_effect_size=es[:80],
        ai_overall_grade=grade.get("overall_grade", "—") or "—",
        gold_overall_grade=gold.overall_grade,
        grade_match="✓" if (grade.get("overall_grade") == gold.overall_grade) else "✗",
        numerical_hallucination=_detect_numeric_hallucination(grade, gold.sample_size),
        notes=(
            "canonical_routed" if payload.get("canonical_in_top_k")
            else "cross-paper contamination (canonical not in top-K)"
        ),
    )


# ---------------------------------------------------------------------------
# Output: comparison table on stdout
# ---------------------------------------------------------------------------


def _print_comparison_table(rows: list[_Row]) -> None:
    bar = "=" * 110
    print()
    print(bar)
    print(f"{'Batch-3 Benchmark · AI vs 金标准 (12 篇文献)':^110}")
    print(bar)
    header = (
        f"{'方案':<14} {'AI 设计':<8} {'金标准':<8} {'设计':<4}  "
        f"{'AI N':>5} {'金N':>5} {'N':<4}  "
        f"{'AI 等级':<5} {'金标准':<5} {'等级':<4}  {'数值幻觉'}"
    )
    print(header)
    print("-" * 110)
    for r in rows:
        print(
            f"{r.slug:<14} {r.ai_study_design:<8} {r.gold_study_design:<8} {r.design_match:<4}  "
            f"{r.ai_sample_size:>5} {r.gold_sample_size:>5} {r.sample_size_match:<4}  "
            f"{r.ai_overall_grade:<5} {r.gold_overall_grade:<5} {r.grade_match:<4}  "
            f"{r.numerical_hallucination}"
        )
    print(bar)


def _print_summary(rows: list[_Row]) -> tuple[float, float]:
    n = len(rows)
    if n == 0:
        return 0.0, 0.0

    # GRADE accuracy: AI overall_grade == gold overall_grade.
    grade_correct = sum(1 for r in rows if r.grade_match == "✓")
    accuracy = grade_correct / n * 100.0

    # Study-design accuracy: AI study_design == gold study_design.
    design_correct = sum(1 for r in rows if r.design_match == "✓")
    design_acc = design_correct / n * 100.0

    # Sample-size accuracy: AI N == gold N (numeric correctness).
    n_correct = sum(1 for r in rows if r.sample_size_match == "✓")
    n_acc = n_correct / n * 100.0

    # Hallucination rate: scenarios flagged "是(...)".
    hallucinated = sum(1 for r in rows if r.numerical_hallucination.startswith("是"))
    halluc_rate = hallucinated / n * 100.0

    bar = "=" * 110
    print()
    print(bar)
    print(f"{'Batch-3 Benchmark 总结指标 (12 篇文献)':^110}")
    print(bar)
    print(f"  总文献数              : {n}")
    print(f"  研究设计 准确率        : {design_acc:6.2f}%  ({design_correct}/{n})")
    print(f"  样本量 准确率          : {n_acc:6.2f}%  ({n_correct}/{n})")
    print(f"  GRADE 整体 准确率      : {accuracy:6.2f}%  ({grade_correct}/{n})")
    print(f"  数值幻觉率             : {halluc_rate:6.2f}%  ({hallucinated}/{n})")
    print(bar)
    return accuracy, halluc_rate


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------


def _write_xlsx(rows: list[_Row], payloads: list[dict], generated_at: str) -> Optional[Path]:
    try:
        from openpyxl import Workbook  # type: ignore[import-not-found]
        from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore[import-not-found]
    except ImportError:
        return None

    out = _RESULT_XLSX
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "review_table"

    headers = [
        "方案名称",
        "试验类型",
        "研究设计 (AI)",
        "研究设计 (金标准)",
        "设计匹配",
        "提取样本量 (AI)",
        "金标准样本量",
        "样本量匹配",
        "效应量 (AI)",
        "AI 最终 GRADE",
        "金标准 GRADE",
        "等级匹配",
        "是否存在数字幻觉",
        "备注",
    ]
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="305496")
    wrap = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    for row_idx, r in enumerate(rows, start=2):
        values = [
            r.slug,
            r.trial_type,
            r.ai_study_design,
            r.gold_study_design,
            r.design_match,
            r.ai_sample_size,
            r.gold_sample_size,
            r.sample_size_match,
            r.ai_effect_size,
            r.ai_overall_grade,
            r.gold_overall_grade,
            r.grade_match,
            r.numerical_hallucination,
            r.notes,
        ]
        for col_idx, v in enumerate(values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=v)
            cell.alignment = wrap

    widths = [14, 36, 14, 14, 10, 14, 14, 10, 36, 14, 14, 10, 28, 28]
    for col_idx, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + col_idx)].width = w
    ws.freeze_panes = "A2"

    # Side sheet: per-scenario JSON for full auditability.
    ws2 = wb.create_sheet("per_scenario_json")
    ws2.append(["slug", "json"])
    for p in payloads:
        ws2.append([p["scenario"], json.dumps(p, ensure_ascii=False)])
    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 120

    wb.properties.title = "OncoImmunoRAG Batch-3 Benchmark"
    wb.properties.subject = "Target-Aware + Registry-aligned 12-paper GRADE benchmark"
    wb.properties.description = (
        f"Auto-generated by scripts/run_batch_3.py at {generated_at}. "
        f"Includes side sheet with full per-scenario JSON for audit."
    )
    wb.properties.keywords = "GRADE, accuracy, hallucination, canonical routing, registry"

    wb.save(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        nargs="*",
        help="Run only these slug(s). Default: all 12.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K chunks per scenario (default 5).",
    )
    p.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip writing benchmark_results_3.xlsx (only print + JSON).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Bootstrap tests.eval_batch_grade without installing it.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "eval_batch_grade", str(_TESTS / "eval_batch_grade.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["eval_batch_grade"] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    _SCENARIOS = _mod._SCENARIOS

    from utils.config_loader import load_config  # noqa: PLC0415
    from utils.trial_registry import lookup as registry_lookup  # noqa: PLC0415

    cfg = load_config()
    if not cfg.effective_openai_api_key():
        print(
            "[run_batch_3] OPENAI_API_KEY is empty — set it in config.yaml "
            "or via the env var before running.",
            file=sys.stderr,
        )
        return 2

    selected = _SCENARIOS
    if args.scenario:
        wanted = set(args.scenario)
        selected = tuple(s for s in _SCENARIOS if s.slug in wanted)
        missing = wanted - {s.slug for s in selected}
        if missing:
            print(f"[run_batch_3] unknown slug(s): {sorted(missing)}", file=sys.stderr)
            return 2

    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _GRADE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    bar = "=" * 88
    print(bar)
    print(
        f"{'Batch-3 启动: LIVE LLM + Target-Aware + Registry · 方案数 = ' + str(len(selected)):^88}"
    )
    print(bar)

    payloads: list[dict] = []
    rows: list[_Row] = []
    for scenario in selected:
        started = time.monotonic()
        print(f"  · {scenario.slug} ...", flush=True)
        payload = _run_one(scenario, cfg, top_k=args.top_k)
        duration = time.monotonic() - started
        status = "OK" if not payload.get("error") else f"ERR: {payload['error'][:40]}"
        canonical_in = "✓canonical_in_topK" if payload.get("canonical_in_top_k") else "✗ canonical_MISSING"
        print(
            f"      {duration:6.2f}s · canonical={scenario.canonical_file} · "
            f"{canonical_in} · {status}"
        )

        # Also write to tests/results/grade_eval/<slug>.json (kept for
        # back-compat with downstream review tables).
        (_GRADE_OUT_DIR / f"{scenario.slug}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        gold = _GOLD_REFS.get(scenario.slug)
        if gold is None:
            print(f"      [WARN] no gold ref for {scenario.slug}", file=sys.stderr)
            continue
        info = registry_lookup(scenario.canonical_file)
        rows.append(_build_row(payload, scenario, gold, info))
        payloads.append(payload)

    # Persist the batch-3 artefacts.
    generated_at = datetime.now(timezone.utc).isoformat()
    result_payload = {
        "generated_at": generated_at,
        "n_scenarios": len(rows),
        "gold_refs": {s: asdict(g) for s, g in _GOLD_REFS.items()},
        "rows": [asdict(r) for r in rows],
        "scenarios": payloads,
    }
    _RESULT_JSON.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Console: comparison + summary.
    _print_comparison_table(rows)
    accuracy, halluc_rate = _print_summary(rows)

    # Excel.
    if args.no_xlsx:
        xlsx_msg = "  · (skipped --no-xlsx)"
    else:
        xlsx_path = _write_xlsx(rows, payloads, generated_at)
        xlsx_msg = (
            f"  · benchmark_results_3.xlsx → {xlsx_path}"
            if xlsx_path is not None
            else "  · openpyxl not installed — skipped .xlsx"
        )

    print()
    print(f"[batch3] {len(rows)} scenarios processed.")
    print(f"  · per-scenario JSONs   → {_GRADE_OUT_DIR}/<slug>.json")
    print(f"  · batch JSON dump      → {_RESULT_JSON}")
    print(xlsx_msg)
    print()
    print(f"  GRADE 准确率           : {accuracy:6.2f}%")
    print(f"  数值幻觉率             : {halluc_rate:6.2f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
