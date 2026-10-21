"""Preview test for :mod:`ingestion.pdf_parser`.

Loads only the **first** PDF under ``data/raw/`` (sorted alphabetically for
determinism), splits it with :class:`SentenceSplitter`, and prints a preview
of the first 2 chunks so a human can eyeball the output.

Run directly::

    python tests/test_parser.py

…or as a pytest case from the project root::

    pytest tests/test_parser.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_parser.py` to resolve `ingestion` / `utils` packages
# without requiring an editable install. pytest already does this via
# `pythonpath = ["src"]` in pyproject.toml, but direct invocation needs help.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows console often defaults to GBK; force UTF-8 with replacement so we
# don't crash on scientific symbols (e.g. U+077D) inside the PDF text.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

from ingestion.pdf_parser import chunk_documents, load_pdfs  # noqa: E402
from utils.config_loader import load_config  # noqa: E402

_PREVIEW_CHARS = 400  # how much of each chunk to print


def _first_pdf() -> Path:
    """Return the first PDF in ``data/raw`` (alphabetical order)."""
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")
    return pdfs[0]


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Trim text and append an ellipsis if it was truncated."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def run_first_pdf_preview() -> None:
    """End-to-end smoke test: load → chunk → print first 2 chunk previews."""
    pdf = _first_pdf()
    print(f"First PDF: {pdf.name}")
    print(f"Path     : {pdf}")

    cfg = load_config()
    print(
        f"Config   : chunk_size={cfg.chunk_size}, "
        f"chunk_overlap={cfg.chunk_overlap}"
    )

    # `files=[pdf]` keeps this test scoped to a single document.
    file_docs = load_pdfs(files=[pdf])
    chunks = chunk_documents(
        file_docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )

    print(f"\nSource files : 1 ({pdf.name})")
    print(f"Chunks total : {len(chunks)}")
    print(f"Chunks shown : {min(2, len(chunks))}\n")

    for i, chunk in enumerate(chunks[:2]):
        meta = chunk.metadata or {}
        chunk_id = meta.get("chunk_id", i)
        page = meta.get("page_label") or meta.get("page_number", "?")
        print(f"--- chunk #{chunk_id} (page {page}) ---")
        print(_preview(chunk.text))
        print()


# ---- pytest entry point -----------------------------------------------
def test_first_pdf_chunks_are_non_empty() -> None:
    """Pytest assertion: chunks exist and contain non-empty text."""
    pdf = _first_pdf()
    file_docs = load_pdfs(files=[pdf])
    chunks = chunk_documents(file_docs)
    assert len(chunks) >= 1, "expected at least one chunk"
    assert all(c.text.strip() for c in chunks), "chunks must not be empty"
    # Sanity-check that the first 2 chunks have unique chunk_ids.
    ids = [c.metadata.get("chunk_id") for c in chunks[:2]]
    assert len(set(ids)) == len(ids), f"duplicate chunk_ids: {ids}"


# ---- direct invocation -------------------------------------------------
if __name__ == "__main__":
    run_first_pdf_preview()
