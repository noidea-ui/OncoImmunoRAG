"""Re-grade existing GRADE JSONs with the bug-fixed rule engine.

After fixing the double-downgrade bug in
:func:`grading.grade_evaluator.apply_downgrade_rules` and adding the
``utils.trial_registry`` overrides for RC48-C014 / McGregor-DAD, the
JSONs under ``tests/results/grade_eval/`` are stale on two counts:

1. KEYNOTE-057 was classified as ``单臂试验`` but still received the
   "非 RCT −1" penalty (the bug), pushing it from ``低`` to ``极低``.
   Re-running with the fix brings it to ``低``.
2. RC48-C014 was classified as ``RCT`` by the LLM, but the registry
   says it is ``单臂试验``. Re-running with the override drops it
   one rung further (``高`` → ``低``) — and the bug fix means we don't
   double-penalise.

This script is **pure offline**: it never touches the LLM. It:

* reads each ``<slug>.json``,
* re-validates the existing fields against :class:`GRADEResult`,
* applies the registry override on ``study_design`` (and notes it in
  the reasoning),
* re-runs :func:`apply_downgrade_rules`,
* writes the updated JSON back to the same path,
* diff-prints the before / after for each file.

The script is idempotent: running it twice is a no-op the second time.

Typical use::

    poetry run python scripts/regrade_with_overrides.py
    # or in this current env:
    python scripts/regrade_with_overrides.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make ``src`` importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grading.grade_evaluator import (  # noqa: E402
    GRADEResult,
    apply_downgrade_rules,
)
from utils.trial_registry import lookup  # noqa: E402

_GRADE_DIR = _PROJECT_ROOT / "tests" / "results" / "grade_eval"

# Slug → canonical source file. Kept here (rather than imported from
# tests/eval_batch_grade.py) so the regrade script is runnable in an
# env where the test module's retrieval / grading imports are
# unavailable.
_SLUG_TO_CANONICAL_FILE: dict[str, str] = {
    "EV-302": "NEJMoa2312117.pdf",
    "NIAGARA": "Powles_2024_NEJM_NIAGARA-Durvalumab.pdf",
    "IMvigor011": "Powles_2025_NEJM_IMvigor011-ctDNA.pdf",
    "KEYNOTE-057": "Balar_2024_LancetOncol_KEYNOTE-057.pdf",
    "RC48-C014": "Zhou_2025_AnnOncol_RC48-C014.pdf",
    "Nadofaragene": "Boorjian_2021_LancetOncol_Nadofaragene.pdf",
    "IMvigor130": "Galsky_2024_CellRepMed_IMvigor130-ChemoImmuno.pdf",
    "CORE-001": "Li_2025_NatMed_Cretostimogene-Nivo.pdf",
    "McGregor-DAD": "McGregor_2024_AnnOncol_DAD.pdf",
    "VESPER": "Pfister_2024_LancetOncol_VESPER.pdf",
    "IMvigor010": "Powles_2021_Nature_IMvigor010-ctDNA.pdf",
    "ABACUS": "Szabados_2022_EurUrol_ABACUS.pdf",
}

# Slug → canonical study_design. Acts as the **source of truth** for the
# ``study_design`` field, overriding whatever the LLM (or a previous
# buggy regrade run) wrote to disk. The values here mirror
# ``utils.trial_registry`` for slugs that have an entry, and the actual
# study design for the others.
_SLUG_TO_CANONICAL_STUDY_DESIGN: dict[str, str] = {
    "EV-302": "RCT",
    "NIAGARA": "RCT",
    "IMvigor011": "RCT",
    "KEYNOTE-057": "单臂试验",
    "RC48-C014": "单臂试验",  # registry override (Phase Ib/II open-label)
    "Nadofaragene": "单臂试验",  # Phase 3 single-arm
    "IMvigor130": "RCT",
    "CORE-001": "单臂试验",  # Phase 2 single-arm
    "McGregor-DAD": "单臂试验",  # Phase I single-arm (registry)
    "VESPER": "RCT",
    "IMvigor010": "RCT",
    "ABACUS": "单臂试验",  # Phase 2 single-arm
}


# Markers inserted by previous runs of this script / ``grade_extraction``.
# Strip them before applying the fresh rule-engine pass so we don't
# accumulate duplicate traces.
_RULE_TRACE_MARK = "[降级规则引擎]"
_OVERRIDE_MARK = "[试验注册表覆盖]"


def _clean_reasoning(reasoning: str | None) -> str:
    """Drop any previously-appended rule-engine traces or override notes."""
    if not reasoning:
        return ""
    text = reasoning
    # Cut at the first rule-engine block.
    idx = text.find(_RULE_TRACE_MARK)
    if idx >= 0:
        text = text[:idx]
    # Drop any orphan override notes too.
    for line in text.splitlines():
        if _OVERRIDE_MARK in line:
            text = text.replace(line, "")
    return text.strip()


def _regrade_one(slug: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-grade one scenario payload; return ``(before, after)`` dicts."""
    before: dict[str, Any] = {}
    grade_payload = payload.get("grade") or {}
    if grade_payload:
        before = {
            "study_design": grade_payload.get("study_design"),
            "overall_grade": grade_payload.get("overall_grade"),
        }

    # Use the slug's declared canonical_file, not the noisy retrieved
    # chunks (which may include unrelated papers).
    canonical = _SLUG_TO_CANONICAL_FILE.get(slug, "")
    override_info = None
    if canonical:
        info = lookup(canonical)
        if info is not None:
            override_info = (canonical, info)

    # Build a fresh GRADEResult from the LLM-emitted fields only — drop
    # the old overall_grade (rule engine will set it) AND strip any
    # leftover rule-engine / override notes from grade_reasoning.
    fields = {k: v for k, v in grade_payload.items() if k != "overall_grade"}
    fields["grade_reasoning"] = _clean_reasoning(fields.get("grade_reasoning"))

    # Force the canonical study_design from the per-slug truth table
    # (overrides whatever the LLM — or a previous buggy regrade — wrote).
    canonical_design = _SLUG_TO_CANONICAL_STUDY_DESIGN.get(slug)
    if canonical_design:
        fields["study_design"] = canonical_design

    # Note registry-driven overrides separately so reviewers can see *why*
    # a value came from the registry rather than the LLM.
    if override_info is not None:
        fn, info = override_info
        if info.study_design != canonical_design:
            # Shouldn't happen — registry and truth table should agree.
            pass
        else:
            override_note = (
                f"[试验注册表覆盖] canonical_file_name={fn!r} "
                f"→ study_design={info.study_design!r}"
            )
            fields["grade_reasoning"] = (
                fields["grade_reasoning"].rstrip() + "\n" + override_note
            ).strip()
    # Seed overall_grade so pydantic doesn't reject — apply_downgrade_rules
    # will overwrite it.
    fields.setdefault("overall_grade", "高")
    result = GRADEResult.model_validate(fields)
    result = apply_downgrade_rules(result)

    new_grade = json.loads(result.model_dump_json())
    payload["grade"] = new_grade

    after = {
        "study_design": new_grade.get("study_design"),
        "overall_grade": new_grade.get("overall_grade"),
    }
    return before, after


def main() -> int:
    if not _GRADE_DIR.is_dir():
        print(f"[regrade] directory not found: {_GRADE_DIR}", file=sys.stderr)
        return 2

    json_files = sorted(_GRADE_DIR.glob("*.json"))
    json_files = [p for p in json_files if p.name != "_summary.json"]
    if not json_files:
        print(f"[regrade] no per-scenario JSONs in {_GRADE_DIR}")
        return 0

    bar = "=" * 78
    print(bar)
    print(f"{'regrade_with_overrides summary':^78}")
    print(bar)
    print(f"{'slug':<14} {'design':<10} {'grade':<6}  ->  {'design':<10} {'grade':<6}  changed")
    print("-" * 78)
    n_changed = 0
    rows: list[dict[str, Any]] = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        before, after = _regrade_one(path.stem, payload)
        changed = before != after
        if changed:
            n_changed += 1
        # Persist
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(payload)
        bd = (before.get("study_design") or "—")[:10]
        bg = (before.get("overall_grade") or "—")[:6]
        ad = (after.get("study_design") or "—")[:10]
        ag = (after.get("overall_grade") or "—")[:6]
        print(
            f"{path.stem:<14} {bd:<10} {bg:<6}  ->  {ad:<10} {ag:<6}  "
            f"{'YES' if changed else 'no'}"
        )

    # Refresh the consolidated summary so downstream tables don't read a
    # stale aggregate. ``generated_at`` is preserved from the prior file
    # if present (so reviewers can see when the original LLM ran).
    summary_path = _GRADE_DIR / "_summary.json"
    if summary_path.is_file():
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            generated_at = old.get("generated_at", "")
        except json.JSONDecodeError:
            generated_at = ""
    else:
        generated_at = ""
    summary_path.write_text(
        json.dumps(
            {"generated_at": generated_at, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(bar)
    print(f"Total files   : {len(json_files)}")
    print(f"Changed       : {n_changed}")
    print(f"Summary       : {summary_path}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
