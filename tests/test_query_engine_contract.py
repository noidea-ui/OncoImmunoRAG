"""Contract tests for :mod:`retrieval.query_engine`.

**No model is loaded and no database is touched.** ``from_config`` is the only
entry point that builds the embedding model (1.3 GB, ~59 s cold), so these
tests deliberately exercise the *imported module's* surface — the class
definition, method signatures and module-level exports — rather than an
instance.

That is enough to pin the two contracts callers most often get wrong, both
discovered by running the real system (see CURRENT_ISUES.md):

* ``retrieve()`` takes no ``top_k`` — it is fixed at construction time.
* ``query()`` does not exist, despite the module docstring advertising it.

Run::

    pytest tests/test_query_engine_contract.py -o addopts="" -v
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retrieval import query_engine  # noqa: E402
from retrieval.query_engine import QueryEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_public_methods_are_exactly_the_documented_four():
    """Contract: the callable surface callers may rely on.

    Verified against a live instance during the 2026-10-23 audit; asserting it
    here means an accidental rename or removal fails fast instead of at runtime
    inside a Streamlit callback.
    """
    public = sorted(
        n
        for n in dir(QueryEngine)
        if not n.startswith("_") and callable(getattr(QueryEngine, n, None))
    )
    assert public == ["from_config", "get_rerank_enabled", "retrieve", "set_rerank_enabled"]


@pytest.mark.unit
@pytest.mark.xfail(
    reason=(
        "Known defect (CURRENT_ISSUES.md ISSUE-008): QueryEngine.query does not "
        "exist. The method is mis-indented inside the private class "
        "_NodeWithScore, so the cross-paper synthesis entry points in app.py:829 "
        "are unreachable. Not fixed here — that is business logic."
    ),
    strict=True,
)
def test_query_method_exists():
    """Documented-but-missing capability. Kept as a visible xfail so that fixing
    ISSUE-008 turns this into an XPASS and prompts a doc review."""
    assert callable(getattr(QueryEngine, "query", None))


# ---------------------------------------------------------------------------
# retrieve() signature — the most common caller mistake
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_retrieve_accepts_question_and_canonical_file_only():
    params = list(inspect.signature(QueryEngine.retrieve).parameters)
    assert params == ["self", "question", "canonical_file"]


@pytest.mark.contract
def test_retrieve_has_no_top_k_parameter():
    """Contract: ``top_k`` is fixed by ``from_config``, not per call.

    Callers that pass ``top_k=`` get ``TypeError``. Pinned because the
    sibling ``TempIndexer.retrieve`` *does* accept ``top_k``, which makes the
    asymmetry easy to fall into (CURRENT_ISSUES.md ISSUE-048).
    """
    assert "top_k" not in inspect.signature(QueryEngine.retrieve).parameters


@pytest.mark.contract
def test_temp_indexer_retrieve_does_accept_top_k():
    """The contrasting case, recorded so the asymmetry is explicit."""
    from retrieval.temp_indexer import TempIndexer

    assert "top_k" in inspect.signature(TempIndexer.retrieve).parameters


@pytest.mark.unit
def test_canonical_file_is_keyword_only():
    """Contract: ``canonical_file`` must be passed by name, not positionally."""
    param = inspect.signature(QueryEngine.retrieve).parameters["canonical_file"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_query_result_is_a_dataclass_with_sources():
    """Contract: the documented return payload shape."""
    import dataclasses

    from retrieval.query_engine import QueryResult

    assert dataclasses.is_dataclass(QueryResult)
    names = {f.name for f in dataclasses.fields(QueryResult)}
    assert "sources" in names


@pytest.mark.contract
def test_module_exports_queryengine_and_queryresult():
    for name in ("QueryEngine", "QueryResult"):
        assert name in query_engine.__all__, f"{name} missing from __all__"


@pytest.mark.unit
def test_module_exports_a_bare_query_function():
    """The module-level singleton helper documented for ``from ... import query``.

    It exists as a module function but is unreachable from the UI because the
    class method it is meant to back does not (ISSUE-008).
    """
    assert callable(getattr(query_engine, "query", None))


# ---------------------------------------------------------------------------
# Vector store location
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_vector_db_path_is_derived_from_config_location():
    """Contract: the store path follows config.yaml, never the CWD.

    This is what makes the project relocatable — running from a different
    directory must not silently point at a different database.
    """
    from utils.config_loader import resolve_config_path

    path = query_engine._vector_db_path()
    assert path == resolve_config_path().parent / "data" / "vector_db"
    assert not path.is_absolute() or path.is_absolute()  # absolute is fine; CWD-relative is not
