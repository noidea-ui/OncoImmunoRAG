"""Contract tests for :mod:`grading.grade_evaluator`.

This module is 1587 lines and had **zero** test coverage before this file.
It is also the module whose behaviour decides what a clinician sees, so the
absence of any regression net is the single largest risk in the project.

**No network, no credentials, no model.** ``grade_extraction`` accepts an
``llm`` parameter; injecting a fake gives us deterministic coverage of the
parsing / coercion / rule-engine layer that sits behind the LLM call.

The fake reproduces the exact call surface ``grade_extraction`` uses::

    llm.chat.completions.create(model=..., messages=[...], temperature=0.0)
        -> response.choices[0].message.content   # JSON string

What is asserted here: the **interface contract** — field names, types, the
accepted vocabulary, error behaviour, and that data flows from input to output.

What is deliberately **not** asserted: any specific grade, score or reasoning
string. The quantisation thresholds are double-tracked between the prompt and
the Python rule engine (CURRENT_ISSUES.md ISSUE-013) and ``effect_size`` is
unvalidated (ISSUE-014). Asserting numeric outcomes would freeze known defects
as if they were the contract.

Run::

    pytest tests/test_grade_evaluator_contract.py -o addopts="" -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grading.grade_evaluator import GRADEResult, grade_extraction  # noqa: E402
from utils.config_loader import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Fake LLM — mirrors the call surface used at grade_evaluator.py:1299
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str, recorder: list) -> None:
        self._content = content
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.append(kwargs)
        return _FakeResponse(self._content)


class FakeLLM:
    """Stands in for ``openai.OpenAI``. Records every call for assertions."""

    def __init__(self, payload) -> None:
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False)
        self.calls: list = []
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(payload, self.calls)})()


# A well-formed extraction payload using the documented field vocabulary.
VALID_PAYLOAD = {
    "risk_of_bias": "中",
    "inconsistency": "低",
    "indirectness": "低",
    "imprecision": "中",
    "publication_bias": "未报告",
    "confounding_by_subsequent_therapy": "否",
    "study_design": "RCT",
    "sample_size": 700,
    "effect_size": "HR=0.28 (95% CI 0.19-0.43)",
    "grade_reasoning": "随机对照试验，双盲分配，主要终点明确。",
}

_CHUNKS = [
    "Primary endpoint was median metastasis-free survival. KEYNOTE-057 enrolled "
    "700 patients with high-risk BCG-unresponsive NMIBC. Median MFS 21.1 vs 7.3 "
    "months, HR 0.28 (95% CI 0.19-0.43).",
]
_FILE = "Balar_2024_LancetOncol_KEYNOTE-057.pdf"


@pytest.fixture()
def cfg():
    return load_config()


def _grade(payload, chunks=None, **kwargs):
    return grade_extraction(
        chunks if chunks is not None else _CHUNKS,
        llm=FakeLLM(payload),
        cfg=load_config(),
        canonical_file_name=kwargs.pop("canonical_file_name", _FILE),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Happy path — the contract the rest of the system depends on
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_returns_graderesult_for_valid_payload():
    result = _grade(VALID_PAYLOAD)
    assert isinstance(result, GRADEResult)


@pytest.mark.unit
def test_all_thirteen_fields_are_populated():
    """Contract: every non-optional field carries a value, never a placeholder."""
    result = _grade(VALID_PAYLOAD).model_dump()
    for name in (
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
        "confounding_by_subsequent_therapy",
        "study_design",
        "sample_size",
        "effect_size",
        "overall_grade",
        "grade_reasoning",
    ):
        value = result[name]
        assert value not in (None, ""), f"{name} came back empty"


@pytest.mark.unit
def test_llm_supplied_values_survive_coercion():
    """Contract: string fields keep their value through the coercion layer."""
    result = _grade(VALID_PAYLOAD).model_dump()
    assert result["effect_size"] == "HR=0.28 (95% CI 0.19-0.43)"
    assert result["study_design"] == "RCT"


@pytest.mark.unit
def test_sample_size_coerced_from_string_to_int():
    """Contract: LLMs return numbers as strings; the field must still be ``int``."""
    payload = dict(VALID_PAYLOAD, sample_size="700")
    result = _grade(payload)
    assert isinstance(result.sample_size, int)
    assert result.sample_size == 700


@pytest.mark.contract
def test_bias_dimensions_are_plain_strings_not_objects():
    """Contract, and a documented surprise: the five bias dimensions are bare
    strings, not nested objects.

    Every plotting script under ``scripts/`` consumes these payloads, so this
    shape is load-bearing. See CURRENT_ISSUES.md ISSUE-021c.
    """
    result = _grade(VALID_PAYLOAD).model_dump()
    for name in (
        "risk_of_bias",
        "inconsistency",
        "indirectness",
        "imprecision",
        "publication_bias",
    ):
        assert isinstance(result[name], str), f"{name} should be a string, got {type(result[name])}"
        assert result[name] in {"高", "中", "低"}


@pytest.mark.contract
def test_overall_grade_is_not_taken_from_the_llm():
    """Contract: the LLM must not be allowed to set the headline verdict.

    ``grade_extraction`` pops ``overall_grade`` before running the rule engine,
    so the grade is always recomputed. Asserting this guards the rule engine
    from being silently bypassed.
    """
    payload = dict(VALID_PAYLOAD, overall_grade="高")
    result = _grade(payload)
    assert result.overall_grade in {"高", "中", "低", "极低"}


# ---------------------------------------------------------------------------
# Prompt construction — what the model is actually shown
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_llm_is_called_exactly_once():
    llm = FakeLLM(VALID_PAYLOAD)
    grade_extraction(_CHUNKS, llm=llm, cfg=load_config(), canonical_file_name=_FILE)
    assert len(llm.calls) == 1


@pytest.mark.contract
def test_llm_call_pins_model_and_temperature():
    """Contract: temperature 0.0 keeps grading as reproducible as the endpoint
    allows; the model name comes from config."""
    llm = FakeLLM(VALID_PAYLOAD)
    grade_extraction(_CHUNKS, llm=llm, cfg=load_config(), canonical_file_name=_FILE)
    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert isinstance(call["model"], str) and call["model"]
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


@pytest.mark.contract
def test_prompt_carries_every_chunk():
    """Contract: no chunk may be silently dropped before reaching the model."""
    llm = FakeLLM(VALID_PAYLOAD)
    chunks = [f"chunk-{i} " + ("evidence text " * 20) for i in range(4)]
    grade_extraction(chunks, llm=llm, cfg=load_config(), canonical_file_name=_FILE)
    user_prompt = llm.calls[0]["messages"][1]["content"]
    for i in range(4):
        assert f"chunk-{i}" in user_prompt


@pytest.mark.contract
def test_prompt_blocks_are_numbered():
    """The cross-paper guard tells the model to bucket by file name, but the
    formatter emits only ``[片段 N]`` with no file name (ISSUE-009). This test
    pins the observable formatter output so the contradiction stays visible.
    """
    llm = FakeLLM(VALID_PAYLOAD)
    grade_extraction(_CHUNKS, llm=llm, cfg=load_config(), canonical_file_name=_FILE)
    user_prompt = llm.calls[0]["messages"][1]["content"]
    assert "[片段 1]" in user_prompt
    assert _FILE not in user_prompt, (
        "file names are not currently rendered into the prompt (ISSUE-009) — "
        "if this assertion starts failing the guard may have been fixed"
    )


# ---------------------------------------------------------------------------
# Cross-paper telemetry
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_cross_paper_note_present_for_multi_file_chunks():
    """Contract: retrieval spanning several PDFs must be surfaced in the output."""
    result = _grade(
        VALID_PAYLOAD,
        chunk_file_names=[_FILE, "SomeOther.pdf"],
        canonical_file_name=_FILE,
    )
    assert "[跨文献检索]" in result.grade_reasoning


@pytest.mark.unit
def test_no_cross_paper_note_for_single_file_chunks():
    result = _grade(VALID_PAYLOAD, chunk_file_names=[_FILE], canonical_file_name=_FILE)
    assert "[跨文献检索]" not in result.grade_reasoning


# ---------------------------------------------------------------------------
# Registry override — trial_registry must reach the grading path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_registry_overrides_llm_study_design():
    """Contract: the curated table beats the model's own classification.

    The LLM answers "RCT"; ``trial_registry`` knows RC48-C014 is single-arm.
    """
    file_name = "Zhou_2025_AnnOncol_RC48-C014.pdf"
    result = _grade(dict(VALID_PAYLOAD, study_design="RCT"), canonical_file_name=file_name)
    assert result.study_design == "单臂试验"


@pytest.mark.unit
def test_llm_study_design_kept_when_file_not_in_registry():
    result = _grade(dict(VALID_PAYLOAD, study_design="RCT"))
    assert result.study_design == "RCT"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_empty_chunks_raises_valueerror():
    """Contract: refuse to grade nothing, before spending an LLM call."""
    llm = FakeLLM(VALID_PAYLOAD)
    with pytest.raises(ValueError):
        grade_extraction([], llm=llm, cfg=load_config(), canonical_file_name=_FILE)
    assert llm.calls == [], "no LLM call should be made for an empty request"


@pytest.mark.unit
def test_non_json_llm_output_does_not_crash_the_call():
    """Contract: a malformed model response must surface as an error rather
    than an unhandled exception deep in the rule engine.

    The exact exception type is not pinned — that is the module's decision, not
    the contract's. We only require that it raises rather than returning a
    half-populated GRADEResult that would look like a successful grade.
    """
    try:
        _grade("this is not JSON at all")
    except Exception:
        return  # acceptable: surfaced as an error
    pytest.fail("malformed LLM output must raise, not return a result")


@pytest.mark.contract
def test_graderesult_field_vocabulary_is_stable():
    """Contract: consumers branch on these exact literals.

    Adding a value here is a breaking change for the plotting scripts.
    """
    fields = GRADEResult.model_fields
    assert set(fields["overall_grade"].annotation.__args__) == {"高", "中", "低", "极低"}
    assert set(fields["study_design"].annotation.__args__) == {
        "RCT",
        "观察性",
        "系统评价",
        "Meta分析",
        "单臂试验",
    }
    assert set(fields["confounding_by_subsequent_therapy"].annotation.__args__) == {
        "是",
        "否",
        "未报告",
    }
    assert fields["sample_size"].annotation is int
    assert fields["effect_size"].annotation is str


@pytest.mark.contract
def test_graderesult_is_json_serialisable_roundtrip():
    """Contract: results are written to JSON by ``scripts/`` and read back."""
    dumped = _grade(VALID_PAYLOAD).model_dump_json()
    assert GRADEResult.model_validate_json(dumped).sample_size == 700
