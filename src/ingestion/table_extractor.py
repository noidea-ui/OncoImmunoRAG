"""Extract statistical summary tables from the project PDFs.

The retriever benefits from clean, machine-readable snapshots of the core
endpoints (PFS / OS / pCR / HR / 95% CI) that most often live in
"Table 1.", "Table 2.", or a supplementary appendix table. This module
locates those tables in the text already pulled out by
:mod:`ingestion.pdf_parser` (which uses ``pypdf`` under the hood),
renders each one as a Markdown fragment, and emits a
:class:`llama_index.core.Document` per table tagged with
``section_type="table_summary"``.

Detection strategy
------------------
We rely on plain-text output, so we use **captions + body slicing + light
heuristics** rather than full PDF layout analysis (no ``pdfplumber`` /
``camelot`` dependency).

1. Match a caption at the start of a line:
   ``Table N.`` / ``Table N:`` / ``Table S1.`` / ``Supplementary Table S2.``
   (with the descriptive title on the same line).
2. Slice the body up to the next caption (``Table`` / ``Fig`` /
   ``Supplementary Table`` / ``Supplementary Fig``) or 2 500 characters,
   whichever comes first.
3. Keep the table only when the caption **or** the body contains at least
   one of the GRADE-relevant endpoint markers (``PFS``, ``OS``, ``pCR``,
   ``HR``, ``95% CI``, ``hazard ratio``, ``overall survival``,
   ``progression-free survival``, ``pathological complete response``).
4. Render:
   * if every non-empty row splits into the **same number of cells** on
     a 2+ spaces / tab separator → emit a real ``| col | col |`` Markdown
     table with the caption as the title;
   * otherwise (ragged columns, the common pypdf artifact) → fall back
     to a ``text`` fenced code block so the data still round-trips
     through Markdown without us lying about its shape.

Metadata
--------
Each emitted :class:`Document` carries:

* ``section_type="table_summary"``           — picked up by the retriever
* ``table_label``                            — e.g. ``"Table 2"`` /
  ``"Supplementary Table S1"``
* ``table_title``                            — descriptive text after the caption
* ``contains_endpoints``                     — list of matched keywords
* ``column_count``                           — int (0 when rendered as a
  fenced block)
* ``is_supplementary``                       — bool
* ``file_name``                              — inherited from the source
  :class:`Document`

The module is intentionally pure: no Chroma, no embedding model — it's
exercised as part of :func:`ingestion.pdf_parser.chunk_documents` and is
also unit-testable from synthetic text alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from llama_index.core import Document

# ---------------------------------------------------------------------------
# Section label / endpoint keywords
# ---------------------------------------------------------------------------

#: ``section_type`` metadata value used by the retriever to find these chunks.
SECTION_TABLE_SUMMARY = "table_summary"

#: Markers that flag a table as carrying the GRADE-relevant endpoints. Match
#: either the abbreviation or the long form so we don't reject a paper that
#: writes "progression-free survival" everywhere but never the bare "PFS".
_ENDPOINT_KEYWORDS: tuple[str, ...] = (
    "PFS",
    "progression-free survival",
    "progression free survival",
    "OS",
    "overall survival",
    "pCR",
    "pathological complete response",
    "pathologic complete response",
    "HR",
    "hazard ratio",
    "95% CI",
    "95%ci",
    "95 % CI",
    "95% confidence interval",
    "confidence interval",
)


def _endpoint_keyword_set() -> frozenset[str]:
    return frozenset(k.lower() for k in _ENDPOINT_KEYWORDS)


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Caption at the start of a line. ``Table 1.``, ``Table S1.``, ``Table I.``,
# ``Supplementary Table S2.`` — accepting both half- and full-width colons
# keeps this robust across journals that use CJK punctuation.
_CAPTION_RE = re.compile(
    r"(?im)^[ \t>]*"
    r"(?P<full>"
    r"Supplementary\s+Table\s+(?:S?\d+|[IVX]+)"
    r"|Table\s+(?:S?\d+|[IVX]+)"
    r")"
    r"\s*[.:：]\s*"
    r"(?P<title>[^\n]+)"
)

# Boundary used to clip the body. Match the next caption-like line at column 0.
_NEXT_CAPTION_RE = re.compile(
    r"(?im)^[ \t>]*"
    r"(?:Supplementary\s+(?:Table|Fig(?:ure)?\.?)|"
    r"Table|Fig(?:\.|ure)?)"
    r"\s+(?:S?\d+|[IVX]+)"
)

# In-table column separator: 2+ spaces or a single tab. Pypdf typically
# emits columns separated by 3–8 spaces in NEJM / Lancet PDFs.
_COL_SEP_RE = re.compile(r"[ \t]{2,}|\t")

#: How many characters of body to grab at most before the next caption.
_MAX_BODY_CHARS = 2500

#: Minimum number of non-empty body lines required to consider a region a
#: table (filters out single-line references like "see Table 1.").
_MIN_BODY_LINES = 3


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedTable:
    """One statistical-summary table extracted from a PDF."""

    label: str
    title: str
    markdown: str
    endpoints: tuple[str, ...]
    column_count: int  # 0 when rendered as a fenced code block
    is_supplementary: bool
    raw_body: str


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_endpoints(text: str) -> list[str]:
    """Return the keywords that appear in ``text`` (preserving insertion order)."""
    lo = text.lower()
    seen: set[str] = set()
    hits: list[str] = []
    for kw in _endpoint_keyword_set():
        if kw in lo and kw not in seen:
            seen.add(kw)
            hits.append(kw)
    return hits


def _looks_like_data_row(cells: list[str]) -> bool:
    """Heuristic: a row is data when it has 2+ cells or its single cell
    contains at least one digit."""
    if len(cells) >= 2:
        return True
    return any(ch.isdigit() for ch in cells[0])


def _render_markdown(
    label: str, title: str, body_lines: list[str]
) -> tuple[str, int]:
    """Render a list of body lines as either a Markdown table or a fenced
    code block. Returns ``(markdown, column_count)``.
    """
    rows: list[list[str]] = []
    for line in body_lines:
        cells = [c.strip() for c in _COL_SEP_RE.split(line.strip()) if c.strip()]
        if not cells or not _looks_like_data_row(cells):
            continue
        rows.append(cells)

    if not rows:
        return "", 0

    col_counts = {len(r) for r in rows}
    if len(col_counts) != 1:
        # Ragged columns → fenced block so the data is preserved without
        # us pretending the column structure is regular.
        block = "\n".join(body_lines)
        md = (
            f"**{label}. {title}**\n\n"
            "```text\n"
            f"{block}\n"
            "```"
        )
        return md, 0

    ncols = col_counts.pop()

    def _pad(r: list[str]) -> list[str]:
        return r + [""] * (ncols - len(r))

    # The first non-empty row is treated as the header. The caption line is
    # rendered separately above the table — we do NOT put it in the header
    # row, otherwise it'd be misread as a data column.
    header, *data = rows
    md_lines: list[str] = [
        f"**{label}. {title.strip()}**",
        "",
        "| " + " | ".join(_pad(header)) + " |",
        "|" + "|".join(["---"] * ncols) + "|",
    ]
    for r in data:
        md_lines.append("| " + " | ".join(_pad(r)) + " |")
    return "\n".join(md_lines), ncols


def _extract_from_text(file_name: str, text: str) -> list[ExtractedTable]:
    """Scan ``text`` for captioned statistical tables."""
    out: list[ExtractedTable] = []
    for m in _CAPTION_RE.finditer(text):
        label = m.group("full").strip()
        title = m.group("title").strip()
        is_supp = label.lower().startswith("supplementary")

        body_start = m.end()
        nxt = _NEXT_CAPTION_RE.search(text, pos=body_start)
        body_end = nxt.start() if nxt else min(body_start + _MAX_BODY_CHARS, len(text))
        body = text[body_start:body_end]

        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) < _MIN_BODY_LINES:
            continue

        endpoints = _detect_endpoints(label + " " + title + " " + body)
        if not endpoints:
            continue

        md, ncols = _render_markdown(label, title, lines)
        if not md:
            continue

        out.append(
            ExtractedTable(
                label=label,
                title=title,
                markdown=md,
                endpoints=tuple(endpoints),
                column_count=ncols,
                is_supplementary=is_supp,
                raw_body=body.strip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_table_summaries(
    file_docs: Sequence[Document],
) -> list[Document]:
    """Return one :class:`Document` per detected statistical-summary table.

    Each returned ``Document`` is tagged ``section_type="table_summary"`` and
    carries the metadata fields documented at module level.

    The function is a no-op when ``file_docs`` is empty or when no captions
    match — callers can safely concatenate the result with their prose
    chunks.
    """
    chunks: list[Document] = []
    for doc in file_docs:
        file_name = (doc.metadata or {}).get("file_name") or "<unknown>"
        for t in _extract_from_text(file_name, doc.text):
            meta = {
                "file_name": file_name,
                "section_type": SECTION_TABLE_SUMMARY,
                "table_label": t.label,
                "table_title": t.title,
                "contains_endpoints": list(t.endpoints),
                "column_count": t.column_count,
                "is_supplementary": t.is_supplementary,
            }
            chunks.append(
                Document(
                    text=t.markdown,
                    metadata=meta,
                    excluded_embed_metadata_keys=["file_path"],
                )
            )
    return chunks


__all__ = [
    "ExtractedTable",
    "SECTION_TABLE_SUMMARY",
    "extract_table_summaries",
]


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    import sys
    from pathlib import Path

    # Allow `python src/ingestion/table_extractor.py` from project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from ingestion.pdf_parser import load_pdfs

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"data/raw not found at {raw_dir}")
    docs = load_pdfs(data_dir=raw_dir)
    table_docs = extract_table_summaries(docs)
    print(f"Extracted {len(table_docs)} table_summary chunks from {len(docs)} PDFs")
    for d in table_docs[:5]:
        meta = d.metadata or {}
        print(
            f"  [{meta.get('table_label')}] "
            f"{meta.get('file_name')[:40]:<40} "
            f"endpoints={meta.get('contains_endpoints')}"
        )
        print("    ---")
        for line in d.text.splitlines()[:8]:
            print(f"    {line}")
        print()
