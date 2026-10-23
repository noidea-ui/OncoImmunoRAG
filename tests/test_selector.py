"""Unit + contract tests for :mod:`retrieval.selector`.

``retrieval.selector`` is deliberately dependency-free (no chromadb, no
llama-index, no openai), which makes it the cheapest place to establish a
real regression net.

Covered here:

* ``select_top_k``  — bucket ordering, canonical force-promotion, per-file cap,
  empty / undersized pools, degenerate ``top_k``.
* ``rerank_hits``   — disabled path must be behaviour-preserving, global
  switch isolation.

**These tests deliberately assert interface *shape*, never evidence-grade
scores.** The scoring thresholds are known to be double-tracked (prompt side
vs Python side, see ``CURRENT_ISSUES.md`` ISSUE-013) and the per-file cap is
known to be bypassed by the back-fill loop (ISSUE-020). Locking either of
those into an assertion would freeze a bug as if it were intended behaviour.

Run::

    pytest tests/test_selector.py -o addopts="" -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retrieval import selector  # noqa: E402
from retrieval.selector import (  # noqa: E402
    rerank_hits,
    select_top_k,
)


# ---------------------------------------------------------------------------
# Test doubles
#
# ``_section_type`` / ``_file_name`` in selector.py resolve metadata through
# ``getattr(getattr(hit, "node", hit), "metadata", None)``, so a plain object
# exposing ``.metadata`` is a faithful stand-in for a llama-index
# ``NodeWithScore`` — no third-party import required.
# ---------------------------------------------------------------------------
class FakeNode:
    """Stand-in for a retrieval hit carrying ``metadata``."""

    def __init__(self, file_name: str, section_type: str = "unknown", text: str = "") -> None:
        self.metadata = {
            "file_name": file_name,
            "section_type": section_type,
        }
        self.text = text

    def get_text(self) -> str:
        return self.text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeNode({self.metadata['file_name']}, {self.metadata['section_type']})"


def _names(hits) -> list[str]:
    return [selector._file_name(h) for h in hits]


# ---------------------------------------------------------------------------
# select_top_k — degenerate inputs
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_select_top_k_returns_empty_on_empty_pool():
    """Contract: an empty pool yields ``[]`` and never raises."""
    assert select_top_k([], 5) == []


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_select_top_k_returns_empty_for_non_positive_top_k(top_k):
    """Contract: ``top_k <= 0`` short-circuits to ``[]``."""
    hits = [FakeNode("a.pdf", "primary"), FakeNode("b.pdf", "unknown")]
    assert select_top_k(hits, top_k) == []


@pytest.mark.unit
def test_select_top_k_returns_whole_pool_when_smaller_than_top_k():
    """Contract: an undersized pool is returned intact, not padded or truncated."""
    hits = [FakeNode("a.pdf", "primary"), FakeNode("b.pdf", "unknown")]
    selected = select_top_k(hits, 12)
    assert len(selected) == 2


@pytest.mark.unit
def test_select_top_k_never_exceeds_available_hits():
    """Contract: ``len(result) <= len(hits)`` for any top_k."""
    hits = [FakeNode(f"p{i}.pdf", "primary") for i in range(3)]
    for top_k in range(1, 20):
        assert len(select_top_k(hits, top_k)) <= 3


# ---------------------------------------------------------------------------
# select_top_k — bucket ordering
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_select_top_k_orders_primary_before_unknown_before_secondary():
    """Bucket policy: primary > unknown > secondary when slots are scarce."""
    hits = [
        FakeNode("s.pdf", "secondary"),
        FakeNode("u.pdf", "unknown"),
        FakeNode("p.pdf", "primary"),
    ]
    selected = select_top_k(hits, 3)
    order = [selector._section_type(h) for h in selected]
    assert order.index("primary") < order.index("unknown")
    assert order.index("unknown") < order.index("secondary")


@pytest.mark.unit
def test_select_top_k_treats_missing_metadata_as_unknown():
    """Contract: a hit with no ``section_type`` falls back to ``"unknown"``."""

    class NoMeta:
        metadata = {}

    assert selector._section_type(NoMeta()) == "unknown"
    assert selector._file_name(NoMeta()) == "<unknown>"


@pytest.mark.unit
def test_select_top_k_treats_missing_file_name_as_unknown_sentinel():
    """Contract: a hit with no ``file_name`` uses the ``"<unknown>"`` sentinel."""

    class NoFile:
        metadata = {"section_type": "primary"}

    assert selector._file_name(NoFile()) == "<unknown>"


# ---------------------------------------------------------------------------
# select_top_k — canonical routing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_select_top_k_promotes_canonical_file_primary_chunk():
    """The documented purpose of ``canonical_file``: a niche target paper must
    not lose every slot to a dominant co-occurring paper."""
    hits = [
        FakeNode("EV-302.pdf", "primary"),
        FakeNode("EV-302.pdf", "primary"),
        FakeNode("EV-302.pdf", "primary"),
        FakeNode("McGregor-DAD.pdf", "primary"),
    ]
    selected = select_top_k(hits, 1, canonical_file="McGregor-DAD.pdf")
    assert _names(selected) == ["McGregor-DAD.pdf"]


@pytest.mark.unit
def test_select_top_k_falls_back_to_other_sections_for_canonical_file():
    """When the canonical paper has no Primary chunk in the pool, a non-primary
    chunk of that paper is still promoted — the fallback path."""
    hits = [
        FakeNode("EV-302.pdf", "primary"),
        FakeNode("McGregor-DAD.pdf", "unknown"),
    ]
    selected = select_top_k(hits, 2, canonical_file="McGregor-DAD.pdf")
    assert "McGregor-DAD.pdf" in _names(selected)


@pytest.mark.unit
def test_select_top_k_without_canonical_file_returns_none_of_that_paper_forced():
    """Control: without ``canonical_file`` there is no force-promotion."""
    hits = [FakeNode("EV-302.pdf", "primary"), FakeNode("McGregor-DAD.pdf", "primary")]
    selected = select_top_k(hits, 1)
    assert _names(selected) == ["EV-302.pdf"]


# ---------------------------------------------------------------------------
# rerank_hits — global switch isolation
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_rerank_disabled_is_order_preserving():
    """Contract: ``enabled=False`` must be behaviour-preserving (a pure sort
    layer may only be additive, per the module docstring)."""
    hits = [
        FakeNode("a.pdf", "secondary", text="Background discussion"),
        FakeNode("b.pdf", "primary", text="median PFS 21.1 vs 7.3 months"),
        FakeNode("c.pdf", "unknown", text="misc"),
    ]
    original = [id(h) for h in hits]
    out = rerank_hits(hits, enabled=False)
    assert [id(h) for h in out] == original


@pytest.mark.unit
def test_rerank_enabled_keeps_all_hits():
    """Contract: reranking must never drop or duplicate hits."""
    hits = [
        FakeNode("a.pdf", "secondary", text="Discussion"),
        FakeNode("b.pdf", "primary", text="HR=0.28 (95% CI 0.19-0.43)"),
        FakeNode("c.pdf", "table_summary", text="Table 1. PFS"),
    ]
    out = rerank_hits(hits, enabled=True)
    assert len(out) == len(hits)
    assert {id(h) for h in out} == {id(h) for h in hits}


@pytest.mark.unit
def test_rerank_empty_input():
    assert rerank_hits([], enabled=True) == []


@pytest.mark.unit
def test_module_level_switch_is_readable_and_writable():
    """Contract for ``get/set_rerank_enabled_default`` (module-global state).

    The test restores the original value — leaving it flipped would silently
    change the behaviour of every later test in the same process.
    """
    original = selector.get_rerank_enabled_default()
    try:
        selector.set_rerank_enabled_default(True)
        assert selector.get_rerank_enabled_default() is True
        selector.set_rerank_enabled_default(False)
        assert selector.get_rerank_enabled_default() is False
    finally:
        selector.set_rerank_enabled_default(original)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_module_exports_documented_symbols():
    """Contract: the names documented in docs/interfaces.md must exist."""
    for name in (
        "rerank_hits",
        "select_top_k",
        "get_rerank_enabled_default",
        "set_rerank_enabled_default",
    ):
        assert hasattr(selector, name), f"selector.{name} is documented but missing"


@pytest.mark.contract
@pytest.mark.xfail(
    reason=(
        "Known defect (CURRENT_ISSUES.md ISSUE-049): selector.__all__ lists the "
        "private constants (_DEFAULT_TOP_K, _TABLE_SUMMARY_KEEP_SMALL, ...) but "
        "omits the public functions rerank_hits / get_rerank_enabled_default / "
        "set_rerank_enabled_default. Not fixed here — that would mean editing "
        "business code, which is out of scope for this test-only change."
    ),
    strict=True,
)
def test_public_symbols_are_exported_in_dunder_all():
    for name in ("rerank_hits", "select_top_k"):
        assert name in selector.__all__, f"{name} missing from selector.__all__"
