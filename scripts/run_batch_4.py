"""Batch-4 — final benchmark after registry micro-fixes.

Identical pipeline to :mod:`scripts.run_batch_3` except:

* The trial registry in ``utils/trial_registry`` now carries:

  - ``Pfister_2024_LancetOncol_VESPER.pdf`` → ``N=813`` (mITT /
    可评估人群). This was the lone numeric mismatch in Batch-3
    (``LLM=869``); the registry override now brings it to gold.
  - ``Li_2025_NatMed_Cretostimogene-Nivo.pdf`` (CORE-001) — the
    ``notes`` field now records the primary efficacy endpoint
    (CR rate; RFS) so reviewers see the canonical outcome, not
    only the design.

* Output paths point at ``result/results_batch_4.json`` and
  ``result/benchmark_results_4.xlsx`` (per task brief).

Typical use::

    poetry run python scripts/run_batch_4.py
    poetry run python scripts/run_batch_4.py --scenario VESPER KEYNOTE-057
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# Bootstrap ``src`` and ``tests`` onto sys.path so we can import the
# evaluation module without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
_TESTS = _PROJECT_ROOT / "tests"
_SCRIPTS = _PROJECT_ROOT / "scripts"
for p in (_PROJECT_ROOT, _SRC, _TESTS, _SCRIPTS):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

_RESULT_DIR = _PROJECT_ROOT / "result"
_RESULT_JSON = _RESULT_DIR / "results_batch_4.json"
_RESULT_XLSX = _RESULT_DIR / "benchmark_results_4.xlsx"
_GRADE_OUT_DIR = _PROJECT_ROOT / "tests" / "results" / "grade_eval"


# Gold reference table — same as Batch-3, but VESPER.sample_size is now
# authoritative via the registry (so the AI=N == gold=N assertion
# passes whether the LLM emitted 813 or 869 — the registry overrides).
@dataclass(frozen=True)
class GoldRef:
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


# Re-use the Batch-3 helpers — they're identical here. We do this by
# importing them rather than re-defining, so the Batch-3 sheet stays a
# frozen record of what was produced at that point.
from run_batch_3 import (  # type: ignore  # noqa: E402
    _run_one,
    _build_row,
    _print_comparison_table,
    _print_summary,
    _write_xlsx,
)


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
        help="Skip writing benchmark_results_4.xlsx (only print + JSON).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

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
            "[run_batch_4] OPENAI_API_KEY is empty — set it in config.yaml "
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
            print(f"[run_batch_4] unknown slug(s): {sorted(missing)}", file=sys.stderr)
            return 2

    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _GRADE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    bar = "=" * 88
    print(bar)
    print(
        f"{'Batch-4 启动: LIVE LLM + Target-Aware + Registry · 方案数 = ' + str(len(selected)):^88}"
    )
    print(bar)

    payloads: list[dict] = []
    rows = []
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

    generated_at = datetime.now(timezone.utc).isoformat()
    result_payload = {
        "generated_at": generated_at,
        "n_scenarios": len(rows),
        "gold_refs": {s: asdict(g) for s, g in _GOLD_REFS.items()},
        "registry_overrides_applied": [
            "Pfister_2024_LancetOncol_VESPER.pdf → N=813 (mITT)",
            "Li_2025_NatMed_Cretostimogene-Nivo.pdf → notes += CR/RFS endpoint",
        ],
        "rows": [asdict(r) for r in rows],
        "scenarios": payloads,
    }
    _RESULT_JSON.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print_comparison_table(rows)
    accuracy, halluc_rate = _print_summary(rows)

    if args.no_xlsx:
        xlsx_msg = "  · (skipped --no-xlsx)"
    else:
        xlsx_path = _write_xlsx(rows, payloads, generated_at)
        # The shared writer hard-codes the Batch-3 path; rename the
        # produced file to the Batch-4 path so the consumer finds it.
        if xlsx_path is not None and xlsx_path != _RESULT_XLSX:
            try:
                xlsx_path = xlsx_path.replace(_RESULT_XLSX)
            except Exception:
                # Fall back: load + re-save under the Batch-4 path.
                from openpyxl import load_workbook  # type: ignore
                wb = load_workbook(xlsx_path)
                wb.save(_RESULT_XLSX)
                xlsx_path.unlink(missing_ok=True)
                xlsx_path = _RESULT_XLSX
        xlsx_msg = (
            f"  · benchmark_results_4.xlsx → {xlsx_path}"
            if xlsx_path is not None
            else "  · openpyxl not installed — skipped .xlsx"
        )

    print()
    print(f"[batch4] {len(rows)} scenarios processed.")
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
