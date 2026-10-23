"""Smoke test for the newly added runtime dependencies.

Two problems with this file as it stood (both fixed here):

1. It printed to stdout at **module level** and defined no ``test_*``
   function, so it contributed **zero** pytest cases.
2. It imported ``sentence_transformers`` (and therefore **torch**) at module
   level. Because pytest imports every ``test_*.py`` during collection, a
   missing torch would make the **entire test run error out** rather than
   skip — a CI landmine (CURRENT_ISSUES.md ISSUE-051).

The imports are now **inside** the test functions, and marked
``requires_model`` so a CPU-only or torch-less environment can exclude them.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Every runtime dependency that src/ and app.py import directly. Kept in one
# place so an undeclared direct dependency is caught here rather than in
# production (cf. CURRENT_ISSUES.md ISSUE-035 for `requests`).
DECLARED_RUNTIME_PACKAGES = [
    "chromadb",
    "llama-index",
    "llama-index-embeddings-huggingface",
    "llama-index-readers-file",
    "llama-index-vector-stores-chroma",
    "openai",
    "pandas",
    "pypdf",
    "pyyaml",
    "requests",
    "sentence-transformers",
    "streamlit",
]


@pytest.mark.unit
def test_lightweight_dependencies_are_importable() -> None:
    """Cheap check — no model weights pulled."""
    import yaml

    assert yaml.__version__


@pytest.mark.unit
def test_runtime_dependencies_have_installed_versions() -> None:
    """Contract: every package listed in ``pyproject.toml`` is actually installed.

    Uses ``importlib.metadata`` rather than ``import``, so this stays fast and
    does not drag in torch.
    """
    missing = []
    for name in DECLARED_RUNTIME_PACKAGES:
        try:
            assert version(name)
        except PackageNotFoundError:
            missing.append(name)
    assert not missing, f"declared but not installed: {missing}"


@pytest.mark.unit
@pytest.mark.requires_model
def test_huggingface_embedding_class_is_importable() -> None:
    """Importing the class must work; no weights are downloaded here."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    assert HuggingFaceEmbedding is not None


@pytest.mark.unit
@pytest.mark.requires_model
def test_sentence_transformers_is_importable() -> None:
    """This is the import that previously broke collection when torch was absent."""
    import sentence_transformers

    assert sentence_transformers.__version__


@pytest.mark.unit
def test_streamlit_is_installed() -> None:
    """The web front-end cannot start without this."""
    import streamlit

    assert streamlit.__version__
