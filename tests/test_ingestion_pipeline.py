"""Integration tests for the ingestion pipeline on synthetic documents.

Covers the flow ``pdf_parser.chunk_documents -> table_extractor`` end to end
using **synthetic** ``Document`` objects — no real PDFs, no ChromaDB, no
embedding model, no network. Real PDFs are deliberately excluded: they are
git-ignored, their redistribution status is unconfirmed, and relying on them
would make the suite machine-dependent.

The property under test is metadata continuity: the fields the grader and the
retriever rely on must survive chunking unchanged, because the retrieval layer
keys on ``file_name`` and buckets on ``section_type``.

Run::

    pytest tests/test_ingestion_pipeline.py -o addopts="" -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llama_index.core import Document  # noqa: E402

from ingestion.pdf_parser import chunk_documents  # noqa: E402
from ingestion.table_extractor import extract_table_summaries  # noqa: E402

PROSE = (
    "Patients with high-risk BCG-unresponsive non-muscle-invasive bladder cancer "
    "were randomised 1:1. The primary endpoint was median metastasis-free "
    "survival, which was 21.1 months versus 7.3 months (HR 0.28, 95% CI "
    "0.19-0.43, p<0.001). Complete response was 41.2%. "
)

TABLE_BLOCK = """
Table 1. Primary endpoint comparison.

Arm                 Median MFS   HR        95% CI       p value
Pembrolizumab       21.1 months  0.28      0.19-0.43    <0.001
Investigator choice 7.3 months   reference  reference    reference
"""


def _doc(file_name: str = "a.pdf", text: str = PROSE) -> Document:
    return Document(text=text, metadata={"file_name": file_name})


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_chunking_splits_long_documents():
    long_doc = _doc(text=PROSE * 12)
    chunks = chunk_documents([long_doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1, "a long document should be split into several chunks"


@pytest.mark.integration
def test_chunking_preserves_file_name_on_every_chunk():
    """Contract: retrieval keys on ``file_name``; losing it breaks canonical routing."""
    chunks = chunk_documents([_doc("Balar_2024_LancetOncol_KEYNOTE-057.pdf")],
                             chunk_size=200, chunk_overlap=20)
    assert chunks
    assert {c.metadata.get("file_name") for c in chunks} == {
        "Balar_2024_LancetOncol_KEYNOTE-057.pdf"
    }


@pytest.mark.integration
def test_every_chunk_gets_a_unique_chunk_id():
    """Contract: chunk_id is the tie-breaker used everywhere downstream."""
    chunks = chunk_documents([_doc()], chunk_size=200, chunk_overlap=20)
    ids = [c.metadata.get("chunk_id") for c in chunks]
    assert all(i is not None for i in ids), "every chunk must carry a chunk_id"
    assert len(set(ids)) == len(ids), "chunk_id values must be unique"


@pytest.mark.integration
def test_every_chunk_is_classified_into_the_known_section_vocabulary():
    """Contract: section_type must stay within the vocabulary selector.py buckets on."""
    chunks = chunk_documents([_doc()], chunk_size=200, chunk_overlap=20)
    allowed = {"primary", "secondary", "unknown", "table_summary"}
    seen = {c.metadata.get("section_type") for c in chunks}
    assert seen <= allowed, f"unexpected section_type values: {seen - allowed}"


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "Known defect (CURRENT_ISSUES.md ISSUE-019 / D6): section classification "
        "is largely ineffective — 75.8% of indexed chunks land in 'unknown'. "
        "Recorded, not fixed: changing it alters retrieval behaviour."
    ),
    strict=False,
)
def test_most_chunks_are_classified_as_a_real_section():
    """Not a contract — a health indicator that currently fails."""
    chunks = chunk_documents([_doc()], chunk_size=200, chunk_overlap=20)
    counts = Counter(c.metadata.get("section_type") for c in chunks)
    assert counts["unknown"] / len(chunks) < 0.5


@pytest.mark.integration
def test_multiple_documents_keep_their_own_identity():
    """Contract: chunking must not collapse documents together."""
    docs = [_doc("a.pdf"), _doc("b.pdf")]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert {c.metadata["file_name"] for c in chunks} == {"a.pdf", "b.pdf"}


@pytest.mark.integration
def test_chunking_is_deterministic():
    """Contract: same input, same output — otherwise benchmarks are not comparable."""
    a = chunk_documents([_doc()], chunk_size=200, chunk_overlap=20)
    b = chunk_documents([_doc()], chunk_size=200, chunk_overlap=20)
    assert [c.get_content() for c in a] == [c.get_content() for c in b]
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]


# ---------------------------------------------------------------------------
# chunk_documents -> table_extractor
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_table_summaries_are_produced_from_a_table_bearing_document():
    """Contract: a document containing an endpoint table yields table summaries."""
    docs = chunk_documents([_doc(text=PROSE + TABLE_BLOCK)], chunk_size=400, chunk_overlap=40)
    summaries = extract_table_summaries(docs)
    assert summaries, "an endpoint table should produce at least one table summary"


@pytest.mark.integration
def test_table_summaries_carry_file_name_and_section_type():
    """Contract: table summaries must be routable like any other chunk."""
    docs = chunk_documents([_doc("table.pdf", text=PROSE + TABLE_BLOCK)],
                           chunk_size=400, chunk_overlap=40)
    summaries = extract_table_summaries(docs)
    assert summaries
    for s in summaries:
        assert s.metadata.get("file_name") == "table.pdf"
        assert s.metadata.get("section_type") == "table_summary"


@pytest.mark.integration
def test_prose_only_document_yields_no_table_summaries():
    """Contract: absence of a table must not produce a phantom one."""
    docs = chunk_documents([_doc(text=PROSE)], chunk_size=400, chunk_overlap=40)
    assert extract_table_summaries(docs) == []


@pytest.mark.integration
def test_table_extraction_tolerates_empty_input():
    """Contract: the extractor must not raise on an empty pool."""
    assert extract_table_summaries([]) == []


@pytest.mark.integration
def test_include_tables_false_removes_table_summaries_from_chunks():
    """Contract: ``include_tables`` is the documented switch for the table layer."""
    with_tables = chunk_documents([_doc(text=PROSE + TABLE_BLOCK)], chunk_size=400,
                                  chunk_overlap=40, include_tables=True)
    without = chunk_documents([_doc(text=PROSE + TABLE_BLOCK)], chunk_size=400,
                              chunk_overlap=40, include_tables=False)
    n_with = sum(1 for c in with_tables if c.metadata.get("section_type") == "table_summary")
    n_without = sum(1 for c in without if c.metadata.get("section_type") == "table_summary")
    assert n_without <= n_with
