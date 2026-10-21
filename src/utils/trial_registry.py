"""Trial-feature dictionary for the OncoImmunoRAG project.

Some papers in ``data/raw/`` have study-design / trial-type attributes
that the LLM grader cannot reliably infer from text alone — for example
the Ib/II RC48-C014 study is *open-label* but **single-arm**, not an
RCT. Rather than rely on a stochastic LLM judgement every run, we
maintain a hand-curated override table here and have
:func:`grading.grade_evaluator.grade_extraction` consult it before
applying the rule engine.

Schema
------
Each entry is keyed by the source ``file_name`` (the value LlamaIndex
emits into the chunk's ``file_name`` metadata). The value is a
:class:`TrialInfo` with:

* ``trial_name``   — canonical short name (display only)
* ``trial_type``   — human-readable design summary (e.g. "Phase Ib/II,
  open-label, dose-escalation and dose-expansion (单臂试验)")
* ``study_design`` — one of the ``StudyDesign`` literals accepted by
  :class:`grading.grade_evaluatoR.GRADEResult` ("RCT" / "观察性" /
  "系统评价" / "Meta分析" / "单臂试验"). When set, this overrides the
  LLM's output.
* ``nct_id``       — optional NCT number (display only)
* ``notes``        — free-form notes shown in the review table

Lookup
------
Use :func:`lookup` (or :func:`resolve_study_design`) at grading time:

>>> from utils.trial_registry import lookup, resolve_study_design
>>> lookup("Zhou_2025_AnnOncol_RC48-C014.pdf").study_design
'单臂试验'
>>> resolve_study_design("Zhou_2025_AnnOncol_RC48-C014.pdf", llm_default="RCT")
'单臂试验'

A missing key is *not* an error — callers fall back to whatever the LLM
emitted. This keeps the override table additive: you only need to add
entries for trials where the LLM consistently misjudges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class TrialInfo:
    """Canonical metadata for one source PDF / clinical trial."""

    trial_name: str
    trial_type: str
    study_design: str
    sample_size: int = 0
    nct_id: str = ""
    notes: str = ""
    # Multi-primary-endpoint flag. True for trials whose protocol splits α
    # across two or more primary endpoints (typical: PFS + OS, DFS + OS,
    # CR + EFS, etc.). Used by the RCT-floor exemption clause in
    # :mod:`grading.grade_evaluator` — when True AND strict-threshold
    # 未达 is detected, the floor leniency is skipped and ``imprecision`` is
    # bumped to "高" so the failed endpoint is reflected in the score.
    multiple_primary_endpoints: bool = False


#: Hand-curated overrides. Add new entries here when the LLM misjudges a
#: paper's design; the rest of the pipeline picks them up automatically.
TRIAL_REGISTRY: Mapping[str, TrialInfo] = {
    # ---- RC48-C014 ----
    # Disitamab vedotin + toripalimab in advanced UC. Despite the open-label
    # language in the abstract, this is a single-arm Phase Ib/II with dose
    # escalation + dose expansion — NOT an RCT. The LLM has been calling
    # it "RCT", which both miscategorises the trial and over-rewards the
    # baseline GRADE.
    "Zhou_2025_AnnOncol_RC48-C014.pdf": TrialInfo(
        trial_name="RC48-C014 (Disitamab Vedotin + Toripalimab)",
        trial_type=(
            "Phase Ib/II, open-label, dose-escalation and dose-expansion "
            "(单臂试验)"
        ),
        study_design="单臂试验",
        sample_size=41,
        nct_id="NCT04264936",
        notes="dose-escalation + dose-expansion, 无随机对照组, N=41",
    ),
    # ---- McGregor 2024 DAD ----
    # SG (Sacituzumab Govitecan) + EV (Enfortumab Vedotin) dual-ADC combo.
    # Phase I, single-arm, N=24.
    "McGregor_2024_AnnOncol_DAD.pdf": TrialInfo(
        trial_name="DAD Trial (SG + EV 双ADC联合)",
        trial_type="Phase I, 单臂",
        study_design="单臂试验",
        sample_size=24,
        notes="SG + EV 双 ADC 联合, I 期, N=24",
    ),
    # ---- Nadofaragene 2021 ----
    # Phase 3 but still single-arm (BCG-unresponsive NMIBC, no
    # contemporaneous control arm in the registrational cohort). The
    # LLM was classifying it as "RCT" — the strong prompt constraint
    # was insufficient, so we hard-override it here.
    "Boorjian_2021_LancetOncol_Nadofaragene.pdf": TrialInfo(
        trial_name="Nadofaragene Firadenovec (rAd-IFNα/Syn3)",
        trial_type="Phase 3, 单臂",
        study_design="单臂试验",
        sample_size=151,
        nct_id="NCT02773849",
        notes="Phase 3 但为单臂 (BCG-unresponsive NMIBC, 无对照组), N=151",
    ),
    # ---- ABACUS 2022 ----
    # Atezolizumab neoadjuvant, single-arm Phase 2 in cisplatin-
    # ineligible MIBC. Evaluable cohort N=95.
    "Szabados_2022_EurUrol_ABACUS.pdf": TrialInfo(
        trial_name="ABACUS (Atezolizumab 新辅助)",
        trial_type="Phase 2, 单臂",
        study_design="单臂试验",
        sample_size=95,
        nct_id="NCT02662309",
        notes="Atezolizumab 新辅助, II 期, cis-ineligible, N=95",
    ),
    # ---- CORE-001 ----
    # Cretostimogene grenadenorepvec + nivolumab in BCG-naive HR-NMIBC.
    # Phase 1b single-arm, N=21. Primary endpoint: complete response (CR)
    # rate at any time; secondary: CR durability / recurrence-free survival
    # (RFS). High-grade NMIBC (CIS ± Ta/T1) population.
    "Li_2025_NatMed_Cretostimogene-Nivo.pdf": TrialInfo(
        trial_name="CORE-001 (Cretostimogene + Nivolumab)",
        trial_type="Phase 1b, 单臂",
        study_design="单臂试验",
        sample_size=21,
        nct_id="NCT04387461",
        notes=(
            "Cretostimogene + Nivolumab, BCG-naive HR-NMIBC (CIS ± Ta/T1), "
            "Ib 期, N=21. 主要疗效终点: 完全缓解率 (CR rate, all-T1 CIS); "
            "次要终点: CR 持续时间 / 无复发生存 (RFS)."
        ),
    ),
    # ---- VESPER ----
    # DD-MVAC vs GC perioperative chemotherapy in MIBC (Pfister 2024).
    # Phase 3 RCT — sample_size = 813 corresponds to the mITT / 可评估
    # analysis population reported in the primary publication (the
    # enrolled-N on ClinicalTrials.gov is 500 but that's a different
    # cohort descriptor; we use the paper's reported mITT denominator).
    "Pfister_2024_LancetOncol_VESPER.pdf": TrialInfo(
        trial_name="VESPER (DD-MVAC vs GC 围手术期)",
        trial_type="Phase 3 RCT",
        study_design="RCT",
        sample_size=813,
        nct_id="NCT01812369",
        notes=(
            "DD-MVAC vs GC 围手术期化疗, III 期 RCT, "
            "mITT / 可评估人群 N=813 (Pfister 2024 Lancet Oncol)."
        ),
    ),
}


def lookup(file_name: str) -> Optional[TrialInfo]:
    """Return the override entry for ``file_name`` or ``None``."""
    if not file_name:
        return None
    return TRIAL_REGISTRY.get(file_name)


def resolve_study_design(file_name: str, llm_default: str) -> tuple[str, str]:
    """Return ``(study_design, source)`` for grading.

    ``source`` is ``"registry"`` when the registry had an entry, else
    ``"llm"`` (or ``"default"`` when ``llm_default`` was the empty string).
    Callers can surface ``source`` in the ``grade_reasoning`` so the
    reviewer can see *why* a particular design was used.
    """
    info = lookup(file_name)
    if info is not None:
        return info.study_design, "registry"
    if llm_default:
        return llm_default, "llm"
    return "观察性", "default"


__all__ = [
    "TrialInfo",
    "TRIAL_REGISTRY",
    "canonical_sample_size",
    "lookup",
    "resolve_study_design",
]


def canonical_sample_size(file_name: str) -> int:
    """Return the registry-known sample size, or ``0`` when unknown.

    Used by :func:`grading.grade_evaluator.grade_extraction` as the last
    resort after ``_extract_sample_size_from_chunks`` fails — when set,
    the registry answer is authoritative and write-overrides whatever
    the LLM emitted (with a `[样本量注册表覆盖]` trace note). This closes
    the "retriever pulled EV-302 chunks for a McGregor-DAD question"
    class of failures definitively.
    """
    info = lookup(file_name)
    return info.sample_size if info else 0


if __name__ == "__main__:":  # pragma: no cover — manual smoke
    import json as _json
    for fn, info in TRIAL_REGISTRY.items():
        print(_json.dumps({fn: info.__dict__}, ensure_ascii=False, indent=2))
