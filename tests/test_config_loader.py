"""Unit + contract tests for :mod:`utils.config_loader` and
:mod:`utils.trial_registry`.

Both modules are pure — no network, no ChromaDB, no embedding model, no
credentials required. Together with ``tests/test_selector.py`` they form the
only currently-executable regression net for the project's shared kernel.

**Note on credentials:** ``load_config()`` reads ``OPENAI_API_KEY`` from the
environment when present. These tests assert *shape* (a key is present or it
is not) and never assert that a key exists, so they pass with or without one.

Run::

    pytest tests/test_config_loader.py -o addopts="" -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import AppConfig, load_config, resolve_config_path  # noqa: E402
from utils.trial_registry import (  # noqa: E402
    TRIAL_REGISTRY,
    TrialInfo,
    canonical_sample_size,
    lookup,
    resolve_study_design,
)

# ---------------------------------------------------------------------------
# utils.config_loader
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_config_path_points_at_existing_yaml():
    """Contract: the default config path resolves inside the project root."""
    path = resolve_config_path()
    assert path.is_file(), f"default config not found at {path}"
    assert path.name == "config.yaml"
    assert path.parent == _PROJECT_ROOT


@pytest.mark.unit
def test_resolve_config_path_accepts_explicit_override():
    """Contract: an explicit path wins over the default."""
    explicit = _PROJECT_ROOT / "config.yaml"
    assert resolve_config_path(explicit) == explicit


@pytest.mark.unit
def test_load_config_returns_appconfig_with_expected_types():
    """Contract: field types are part of the contract — downstream modules index
    these directly."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.chunk_size, int) and cfg.chunk_size > 0
    assert isinstance(cfg.chunk_overlap, int) and cfg.chunk_overlap >= 0
    assert isinstance(cfg.embedding_model, str) and cfg.embedding_model
    assert isinstance(cfg.extras, dict)
    assert isinstance(cfg.openai_api_key, str)


@pytest.mark.unit
def test_load_config_reads_current_project_values():
    """Guard against silent edits to config.yaml.

    If these change, that is a deliberate act by someone — update this test in
    the same commit rather than letting it fail mysteriously later.
    """
    cfg = load_config()
    assert cfg.chunk_size == 1024
    assert cfg.chunk_overlap == 200
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"


@pytest.mark.unit
def test_llm_settings_live_in_extras_not_as_attributes():
    """Contract: ``llm_base_url`` / ``llm_model`` are carried in ``extras``.

    ``AppConfig`` deliberately has no ``llm_base_url`` attribute
    (CURRENT_ISSUES.md ISSUE-042); reading ``cfg.llm_base_url`` raises
    AttributeError. This test pins the working spelling.
    """
    cfg = load_config()
    assert "llm_base_url" in cfg.extras
    assert "llm_model" in cfg.extras
    assert not hasattr(cfg, "llm_base_url")


@pytest.mark.unit
def test_env_var_takes_priority_for_openai_api_key(monkeypatch):
    """Contract: OPENAI_API_KEY env var outranks config.yaml."""
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder-not-a-real-key")
    cfg = load_config()
    assert cfg.has_openai_api_key is True
    assert cfg.effective_openai_api_key() == "unit-test-placeholder-not-a-real-key"


@pytest.mark.unit
def test_missing_env_var_falls_back_to_config(monkeypatch):
    """Contract: with no env var, whatever config.yaml holds is used."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_config()
    # config.yaml carries no credential, so the effective key must be empty.
    assert cfg.effective_openai_api_key() == ""


@pytest.mark.unit
def test_has_openai_api_key_is_a_property_not_a_method():
    """Contract: ``has_openai_api_key`` is a ``@property``.

    Easy to get wrong — calling it raises ``TypeError: 'bool' object is not
    callable``. Pinned here because docs/interfaces.md was initially written
    with the method form and had to be corrected.
    """
    assert isinstance(AppConfig.has_openai_api_key, property)


@pytest.mark.unit
def test_has_openai_api_key_agrees_with_effective_key(monkeypatch):
    """Contract: the predicate and the accessor must never disagree."""
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    cfg = load_config()
    assert cfg.has_openai_api_key == bool(cfg.effective_openai_api_key())


@pytest.mark.unit
def test_load_config_raises_when_file_missing():
    """Contract: the default path treats a missing file as fatal."""
    with pytest.raises(FileNotFoundError):
        load_config(path=_PROJECT_ROOT / "definitely-not-here.yaml")


@pytest.mark.unit
@pytest.mark.xfail(
    reason=(
        "Known defect (CURRENT_ISSUES.md ISSUE-015): raise_if_missing=False is "
        "documented to return a default-constructed config, but load_config "
        "still raises ValueError. Fixing it changes business behaviour."
    ),
    strict=True,
)
def test_load_config_raise_if_missing_false_returns_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_config(path=_PROJECT_ROOT / "definitely-not-here.yaml", raise_if_missing=False)
    assert isinstance(cfg, AppConfig)


@pytest.mark.contract
def test_config_yaml_is_tracked_by_git_and_carries_no_credential():
    """Contract: config.yaml is versioned, therefore it must never hold a key.

    Asserted structurally (a commented-out key directive exists) rather than by
    reading any secret value.
    """
    text = (_PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
    active_key_lines = [
        ln
        for ln in text.splitlines()
        if ln.strip().startswith("openai_api_key:")
        and not ln.strip().startswith("#")
    ]
    assert not active_key_lines, "config.yaml must not define an active openai_api_key"
    assert "OPENAI_API_KEY" in text, "config.yaml should point readers at the env var"


@pytest.mark.contract
def test_dotenv_example_declares_no_secret_values():
    """Contract: the committed template must stay empty."""
    text = (_PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            assert value.strip() == "", "OPENAI_API_KEY in .env.example must be empty"


# ---------------------------------------------------------------------------
# utils.trial_registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_is_non_empty_mapping_of_trialinfo():
    assert len(TRIAL_REGISTRY) > 0
    for key, info in TRIAL_REGISTRY.items():
        assert isinstance(key, str) and key.endswith(".pdf"), f"key must be a file name: {key}"
        assert isinstance(info, TrialInfo)


@pytest.mark.unit
def test_lookup_hit_returns_trialinfo():
    info = lookup("Zhou_2025_AnnOncol_RC48-C014.pdf")
    assert isinstance(info, TrialInfo)
    assert info.study_design == "单臂试验"


@pytest.mark.unit
def test_lookup_miss_returns_none_rather_than_raising():
    """Contract: the override table is additive — a miss is not an error."""
    assert lookup("not-in-registry.pdf") is None


@pytest.mark.unit
def test_resolve_study_design_prefers_registry_over_llm_default():
    """Contract: the curated table overrides the LLM's judgement.

    This is the module's entire purpose (see its module docstring: RC48-C014 is
    open-label but single-arm, and the LLM misjudges it as an RCT).
    """
    design, source = resolve_study_design("Zhou_2025_AnnOncol_RC48-C014.pdf", "RCT")
    assert design == "单臂试验"
    assert source == "registry"


@pytest.mark.unit
def test_resolve_study_design_falls_back_to_llm_default_on_miss():
    design, source = resolve_study_design("not-in-registry.pdf", "RCT")
    assert design == "RCT"
    assert source == "llm"


@pytest.mark.unit
def test_canonical_sample_size_hit_and_miss():
    assert canonical_sample_size("Zhou_2025_AnnOncol_RC48-C014.pdf") == 41
    assert canonical_sample_size("not-in-registry.pdf") == 0


@pytest.mark.contract
def test_registry_study_design_values_are_in_grade_result_vocabulary():
    """Contract: every ``study_design`` the registry can produce must be a value
    ``GRADEResult.study_design`` accepts, otherwise the override would blow up
    pydantic validation at grading time.
    """
    from grading.grade_evaluator import GRADEResult

    allowed = set(GRADEResult.model_fields["study_design"].annotation.__args__)
    for file_name, info in TRIAL_REGISTRY.items():
        assert info.study_design in allowed, (
            f"{file_name}: registry study_design {info.study_design!r} "
            f"is not one of {sorted(allowed)}"
        )


@pytest.mark.contract
def test_registry_is_keyed_by_files_that_app_catalogue_knows():
    """Contract: registry keys must match the ``file_name`` values the pipeline
    actually emits.

    Note the deliberate wording: this asserts the registry does not reference
    unknown PDFs, NOT that app.py's ``design_gold`` agrees with it. Those two
    tables use different vocabularies and currently disagree on 5 of 6 entries
    (CURRENT_ISSUES.md ISSUE-036) — that defect is recorded, not asserted here.
    """
    import re

    app_src = (_PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    catalogue = set(re.findall(r'"([^"]+\.pdf)"', app_src))
    for file_name in TRIAL_REGISTRY:
        assert file_name in catalogue, (
            f"registry key {file_name!r} is not in app.py's paper catalogue"
        )
