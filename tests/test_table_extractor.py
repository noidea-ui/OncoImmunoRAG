"""Unit tests for :mod:`ingestion.table_extractor`.

These tests exercise the extraction pipeline on **synthetic** PDF text so
they run without Chroma / llama-index / sentence-transformers. They cover:

* caption detection (Table N. / Supplementary Table S1.)
* endpoint filtering (tables without PFS / OS / pCR / HR / 95% CI are dropped)
* Markdown table rendering when columns are aligned
* fenced-code fallback when columns are ragged
* metadata fields populated correctly
* retrieval top-K table-summary reservation policy (via a fake QueryEngine)

Run directly::

    python tests/test_table_extractor.py

…or as pytest cases from the project root::

    pytest tests/test_table_extractor.py -s
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

# Allow direct invocation without editable install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------

# A "Table 1." with aligned columns (cells separated by 2+ spaces) and
# statistical endpoints. Should render as a Markdown pipe table.
_TABLE1_TEXT = """\
Table 1. Efficacy endpoints in the ITT population.

Outcome                EV+P                Chemo            HR (95% CI)        P value
PFS, median (mo)       12.5                6.3              0.45 (0.38-0.54)   <0.001
OS, median (mo)        31.5                16.1             0.47 (0.38-0.58)   <0.001
ORR, %                 67.7                44.4             -                  -
Grade >=3 TRAE, %      55.9                69.3             -                  -
"""

# A "Supplementary Table S2." with ragged columns — should fall back to
# a fenced code block. The data lines intentionally have differing cell
# counts (3 vs 4 vs 5) so the ``_render_markdown`` heuristic routes them
# to the fenced block path.
_SUPP_TABLE_TEXT = """\
Supplementary Table S2. Subgroup analysis of OS hazard ratio.

PD-L1 high  240  0.42
PD-L1 low   380  0.61  0.45-0.83
Liver mets  110  0.55  0.38-0.79  extra column

This supplementary table shows OS hazard ratios across prespecified subgroups.
Reference baseline: ITT population.
"""

# A "Table 2." that does NOT contain any GRADE endpoint — must be filtered out.
_TABLE2_NON_ENDPOINT = """\
Table 2. Baseline patient characteristics.

Characteristic        Arm A (n=442)    Arm B (n=444)
Median age, yr        67 (38-89)       69 (35-90)
Male sex, n (%)       333 (75.3)       340 (76.6)
ECOG 0, n (%)         217 (49.1)       211 (47.5)
Prior systemic Rx     142              138
"""

# pCR / pathological complete response in the body but no PFS/OS — should
# still be picked up because "pCR" is in the endpoint keyword set.
_PCR_TABLE = """\
Table 3. Pathological complete response rate.

Outcome          pCR rate, %      95% CI          P value
Pembrolizumab + chemo    24.5        18.4-31.2       <0.001
Chemo alone              12.3        7.9-17.6        -
"""

_FULL_DOC = (
    "Title page and front matter go here.\n\n"
    "Abstract\n"
    "Background text. ...\n\n"
    + _TABLE1_TEXT
    + "\nResults\nSome prose...\n\n"
    + _TABLE2_NON_ENDPOINT
    + "\nMethods\n...\n\n"
    + _SUPP_TABLE_TEXT
    + "\n\n"
    + _PCR_TABLE
    + "\nDiscussion\n...\n"
)


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


def _build_doc(text: str, file_name: str = "TestDoc.pdf") -> object:
    """Construct a minimal ``Document``-like object for the extractor."""
    return types.SimpleNamespace(text=text, metadata={"file_name": file_name})


def test_extracts_only_endpoint_tables() -> None:
    """Tables without GRADE endpoints must be filtered out."""
    from ingestion.table_extractor import extract_table_summaries

    docs = [_build_doc(_FULL_DOC)]
    out = extract_table_summaries(docs)
    labels = [d.metadata["table_label"] for d in out]
    # Table 2 is baseline characteristics → dropped.
    # Table 1, Supplementary Table S2, and Table 3 (pCR) survive.
    assert "Table 1." not in labels and "Table 1" in labels  # label keeps trailing dot off
    assert any(l.startswith("Table 1") for l in labels), labels
    assert any(l.startswith("Supplementary Table S2") for l in labels), labels
    assert any(l.startswith("Table 3") for l in labels), labels
    assert not any("Table 2" in l for l in labels), labels


def test_metadata_fields_are_complete() -> None:
    """Each emitted Document must carry the documented metadata."""
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_FULL_DOC)])
    assert out, "no tables extracted"
    for d in out:
        m = d.metadata
        assert m["section_type"] == "table_summary"
        assert m["file_name"] == "TestDoc.pdf"
        assert isinstance(m["table_label"], str) and m["table_label"]
        assert isinstance(m["table_title"], str)
        assert isinstance(m["contains_endpoints"], list)
        assert m["contains_endpoints"], "contains_endpoints must not be empty"
        assert isinstance(m["column_count"], int)
        assert isinstance(m["is_supplementary"], bool)


def test_is_supplementary_flag() -> None:
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_FULL_DOC)])
    labels = {d.metadata["table_label"]: d for d in out}
    assert any(d.metadata["is_supplementary"] for d in labels.values() if d.metadata["table_label"].startswith("Supplementary"))
    assert all(
        not d.metadata["is_supplementary"]
        for d in labels.values()
        if not d.metadata["table_label"].startswith("Supplementary")
    )


def test_aligned_table_renders_as_pipe_table() -> None:
    """When every row has the same column count, output a Markdown pipe table."""
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_TABLE1_TEXT)])
    assert out, "Table 1 not extracted"
    md = out[0].text
    # Real Markdown pipe tables always have the header / separator rows.
    assert "\n| " in md, f"expected pipe table, got:\n{md}"
    # The separator row (`| --- | --- | … |`) appears after the header.
    # Markdown layout:
    #   line 0: "**Table 1. ...**"
    #   line 1: "" (blank)
    #   line 2: "| Outcome | EV+P | ..." (header)
    #   line 3: "|---|---|..."        (separator)
    lines = md.splitlines()
    assert lines[2].lstrip().startswith("| "), lines[:5]
    assert lines[3].lstrip().startswith("|") and "---" in lines[3], lines[:5]
    assert out[0].metadata["column_count"] >= 4
    # Endpoints should include PFS / OS / HR.
    eps = out[0].metadata["contains_endpoints"]
    assert any("pfs" in e.lower() for e in eps)
    assert any("os" == e.lower() or "overall survival" in e.lower() for e in eps)


def test_ragged_table_falls_back_to_fenced_block() -> None:
    """Ragged column counts render as a fenced code block."""
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_SUPP_TABLE_TEXT)])
    assert out, "supplementary table not extracted"
    md = out[0].text
    assert "```text" in md, f"expected fenced block, got:\n{md}"
    assert out[0].metadata["column_count"] == 0
    assert out[0].metadata["is_supplementary"] is True


def test_pcr_table_extracted_via_endpoint_match() -> None:
    """The 'pCR' keyword alone (without PFS/OS) should still flag the table."""
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_PCR_TABLE)])
    assert out, "pCR table not extracted"
    md = out[0].text
    eps = out[0].metadata["contains_endpoints"]
    assert any("pcr" in e.lower() for e in eps), eps
    assert "pCR" in md or "pcr" in md.lower()


def test_caption_regex_accepts_common_variants() -> None:
    """Caption detection must handle Table S1, Table 1, Table I, and CJK colon."""
    from ingestion.table_extractor import _CAPTION_RE

    for caption in (
        "Table 1. PFS in the ITT population.",
        "Table S2. Subgroup analysis of OS.",
        "Table I. CONSORT diagram.",
        "Supplementary Table S1. Adverse events.",
        "Table 3: pathological complete response rate.",
        "Table 1：中文标题",
    ):
        m = _CAPTION_RE.search(caption + "\nBody goes here")
        assert m is not None, f"caption not matched: {caption!r}"


def test_empty_input_returns_empty_list() -> None:
    from ingestion.table_extractor import extract_table_summaries

    assert extract_table_summaries([]) == []
    assert extract_table_summaries([_build_doc("")]) == []


# ---------------------------------------------------------------------------
# Retrieval top-K reservation tests (via the pure-Python selector)
# ---------------------------------------------------------------------------


def _fake_hit(file_name: str, section: str, score: float = 0.5) -> object:
    """Build a minimal NodeWithScore-like object for ``select_top_k``."""
    node = types.SimpleNamespace(
        metadata={"file_name": file_name, "section_type": section},
        get_content=lambda: f"[{file_name}|{section}]",
    )
    return types.SimpleNamespace(node=node, score=score)


def test_retrieve_reserves_table_summary_slot_when_available() -> None:
    """One table_summary chunk → placed at index 0 of the top-k output."""
    from retrieval.selector import select_top_k

    # 5 primary + 3 unknown + 1 table_summary. top_k=5 → 1 slot reserved.
    pool = (
        [_fake_hit("A.pdf", "primary", 0.90), _fake_hit("A.pdf", "primary", 0.85),
         _fake_hit("B.pdf", "primary", 0.80), _fake_hit("C.pdf", "primary", 0.75),
         _fake_hit("D.pdf", "primary", 0.70)]
        + [_fake_hit("E.pdf", "unknown", 0.65), _fake_hit("F.pdf", "unknown", 0.60),
           _fake_hit("G.pdf", "unknown", 0.55)]
        + [_fake_hit("EV-302.pdf", "table_summary", 0.45)]
    )
    selected = select_top_k(pool, top_k=5)

    sections = [h.node.metadata["section_type"] for h in selected]
    assert sections[0] == "table_summary", sections
    assert sum(1 for s in sections if s == "table_summary") == 1
    # The reserved table must come from the right file.
    assert selected[0].node.metadata["file_name"] == "EV-302.pdf"


def test_retrieve_reserves_two_table_slots_for_large_top_k() -> None:
    """top_k=10 → up to 2 table_summary chunks reserved."""
    from retrieval.selector import select_top_k

    pool = (
        [_fake_hit(f"P{i}.pdf", "primary", 0.9 - i * 0.01) for i in range(8)]
        + [_fake_hit("T1.pdf", "table_summary", 0.40),
           _fake_hit("T2.pdf", "table_summary", 0.39),
           _fake_hit("T3.pdf", "table_summary", 0.38)]
    )
    selected = select_top_k(pool, top_k=10)

    sections = [h.node.metadata["section_type"] for h in selected]
    n_tables = sum(1 for s in sections if s == "table_summary")
    assert n_tables == 2, sections
    # The reserved tables are at the front, in score-desc order.
    assert sections[0] == "table_summary"
    assert sections[1] == "table_summary"


def test_retrieve_skips_table_reservation_when_none_available() -> None:
    """No table_summary in the pool → no reservation, primary-first logic unchanged."""
    from retrieval.selector import select_top_k

    pool = [_fake_hit(f"P{i}.pdf", "primary", 0.9 - i * 0.01) for i in range(8)]
    selected = select_top_k(pool, top_k=5)
    sections = [h.node.metadata["section_type"] for h in selected]
    assert all(s == "primary" for s in sections), sections


def test_retrieve_table_per_file_cap_one() -> None:
    """Two tables from the same paper → only 1 reserved in top-k=5."""
    from retrieval.selector import select_top_k

    pool = (
        [_fake_hit(f"P{i}.pdf", "primary", 0.9 - i * 0.01) for i in range(6)]
        + [_fake_hit("Same.pdf", "table_summary", 0.40),
           _fake_hit("Same.pdf", "table_summary", 0.39),
           _fake_hit("Other.pdf", "table_summary", 0.38)]
    )
    selected = select_top_k(pool, top_k=5)
    n_tables = sum(1 for h in selected if h.node.metadata["section_type"] == "table_summary")
    assert n_tables == 1, [h.node.metadata for h in selected]


def test_selector_handles_empty_pool() -> None:
    """Empty input → empty output, no exceptions."""
    from retrieval.selector import select_top_k

    assert select_top_k([], top_k=5) == []
    assert select_top_k([], top_k=0) == []


def test_selector_keeps_order_within_bucket() -> None:
    """Hits within a single bucket keep their input order."""
    from retrieval.selector import select_top_k

    pool = [
        _fake_hit(f"P{i}.pdf", "primary", 0.9 - i * 0.01) for i in range(5)
    ]
    selected = select_top_k(pool, top_k=3)
    assert [h.node.metadata["file_name"] for h in selected] == ["P0.pdf", "P1.pdf", "P2.pdf"]


# ---------------------------------------------------------------------------
# Direct invocation
# ---------------------------------------------------------------------------


def _run_smoke() -> None:
    print("=" * 70)
    print("table_extractor smoke test")
    print("=" * 70)
    from ingestion.table_extractor import extract_table_summaries

    out = extract_table_summaries([_build_doc(_FULL_DOC)])
    print(f"Extracted {len(out)} table_summary chunks\n")
    for i, d in enumerate(out, start=1):
        meta = d.metadata
        print(
            f"--- chunk #{i}: {meta['table_label']} "
            f"(supp={meta['is_supplementary']}, cols={meta['column_count']}) ---"
        )
        print(f"  endpoints : {meta['contains_endpoints']}")
        print(f"  text:")
        for line in d.text.splitlines():
            print(f"    {line}")
        print()


if __name__ == "__main__":
    _run_smoke()
