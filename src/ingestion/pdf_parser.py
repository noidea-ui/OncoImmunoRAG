"""PDF ingestion for the OncoImmunoRAG project.

Reads every PDF under ``data/raw/`` and splits each one into fixed-size
chunks using :class:`llama_index.core.node_parser.SentenceSplitter`. The chunk
size and overlap are read from ``config.yaml`` (see
:mod:`utils.config_loader`).

Each emitted chunk carries a ``section_type`` metadata field used by the
retriever to prioritise passages drawn from the paper's own Results /
Conclusions / Abstract sections:

* ``"primary"``        — ``Abstract`` / ``Results`` / ``Conclusions``
* ``"secondary"``      — ``Background`` / ``Introduction`` / ``Methods`` /
  ``Discussion`` / ``References``
* ``"unknown"``        — anything that doesn't fall in the above (default)
* ``"table_summary"``  — statistical tables (Table 1/2 or supplementary)
  emitted by :mod:`ingestion.table_extractor` and force-promoted in the
  retriever's top-K selection.

Typical usage::

    from ingestion.pdf_parser import parse_pdfs

    file_docs, chunk_docs = parse_pdfs()
    # file_docs: one Document per source PDF
    # chunk_docs: one Document per chunk, ready for embedding/indexing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# llama-index split its file readers into a separate package
# (llama-index-readers-file) starting in 0.10. SimpleDirectoryReader only
# extracts PDF text when a PDFReader is registered. We import it lazily and
# pass it explicitly via ``file_extractor`` so the dependency is documented
# in code rather than relying on import-order side effects.
try:
    from llama_index.readers.file import PDFReader  # provided by llama-index meta-pkg
except ImportError:  # pragma: no cover — fallback for slim installs
    try:
        from llama_index_readers_file import PDFReader  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDF support requires either the `llama-index` meta-package or "
            "`llama-index-readers-file`. Install with:\n"
            "  pip install llama-index-readers-file"
        ) from exc

from ingestion.table_extractor import extract_table_summaries
from utils.config_loader import AppConfig, load_config, resolve_config_path


# ---- defaults mirror the keys in config.yaml ---------------------------
_DEFAULT_CHUNK_SIZE = 1024
_DEFAULT_CHUNK_OVERLAP = 200

# Section-classification labels emitted as ``section_type`` metadata.
_SECTION_PRIMARY = "primary"     # Abstract / Results / Conclusions
_SECTION_SECONDARY = "secondary"  # Discussion / References / Background / Methods
_SECTION_UNKNOWN = "unknown"      # fallback

# Each tuple: ``(compiled regex, label)``. Patterns match a section header
# that sits on its own line (allowing optional trailing punctuation /
# whitespace). More-specific patterns come first so ``Materials and Methods``
# wins over ``Methods``.
_SECTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?im)^\s*abstract\b[ \t]*[:.]?[ \t]*$"),                  _SECTION_PRIMARY),
    (re.compile(r"(?im)^\s*results?\b[ \t]*[:.]?[ \t]*$"),                  _SECTION_PRIMARY),
    (re.compile(r"(?im)^\s*conclusions?\b[ \t]*[:.]?[ \t]*$"),               _SECTION_PRIMARY),
    (re.compile(r"(?im)^\s*background\b[ \t]*[:.]?[ \t]*$"),                _SECTION_SECONDARY),
    (re.compile(r"(?im)^\s*introduction\b[ \t]*[:.]?[ \t]*$"),              _SECTION_SECONDARY),
    (re.compile(
        r"(?im)^\s*(?:patients?|subjects?|materials?)\s+and\s+methods?\b"
        r"[ \t]*[:.]?[ \t]*$"
    ),                                                                      _SECTION_SECONDARY),
    (re.compile(r"(?im)^\s*methods?\b[ \t]*[:.]?[ \t]*$"),                   _SECTION_SECONDARY),
    (re.compile(r"(?im)^\s*discussion\b[ \t]*[:.]?[ \t]*$"),                 _SECTION_SECONDARY),
    (re.compile(r"(?im)^\s*references?\b[ \t]*[:.]?[ \t]*$"),               _SECTION_SECONDARY),
    (re.compile(r"(?im)^\s*bibliography\b[ \t]*[:.]?[ \t]*$"),              _SECTION_SECONDARY),
]


@dataclass(frozen=True)
class _SectionSpan:
    """A half-open ``[start, end)`` text span tagged with a section_type."""

    start: int
    end: int
    section_type: str  # one of _SECTION_*


def _classify_section(text: str) -> list[_SectionSpan]:
    """Scan ``text`` for section headers and return tagged spans.

    Text before the first matched header is tagged ``"unknown"`` — these
    are typically title pages, journal mastheads, or extracted front-matter.
    """
    matches: list[tuple[int, int, str]] = []
    for pattern, label in _SECTION_RULES:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), label))
    # Sort by start; on ties prefer longer matches then any earlier label.
    matches.sort(key=lambda x: (x[0], -x[1]))

    spans: list[_SectionSpan] = []
    cursor = 0
    current = _SECTION_UNKNOWN
    for start, end, label in matches:
        if start < cursor:
            # Overlap (e.g. 'Methods' inside 'Materials and Methods'): skip.
            continue
        if start > cursor:
            spans.append(_SectionSpan(cursor, start, current))
        cursor = end
        current = label
    if cursor < len(text):
        spans.append(_SectionSpan(cursor, len(text), current))
    if not spans:
        spans.append(_SectionSpan(0, len(text), _SECTION_UNKNOWN))
    return spans


def _split_document_by_sections(doc: Document) -> list[Document]:
    """Explode one Document into one Document per detected section.

    Each returned Document inherits the source's metadata (file_name etc.)
    plus a ``section_type`` field. Empty sections are dropped.
    """
    base_meta = dict(doc.metadata or {})
    out: list[Document] = []
    for span in _classify_section(doc.text):
        chunk = doc.text[span.start:span.end].strip()
        if not chunk:
            continue
        meta = dict(base_meta)
        meta["section_type"] = span.section_type
        out.append(
            Document(
                text=chunk,
                metadata=meta,
                excluded_embed_metadata_keys=["file_path"],
            )
        )
    return out


def _pdf_extractor() -> dict[str, object]:
    """Return the file-extractor map wiring ``.pdf`` → :class:`PDFReader`."""
    return {".pdf": PDFReader()}


def _project_data_dir() -> Path:
    """Return ``<project_root>/data/raw`` regardless of the current working dir."""
    return resolve_config_path().parent / "data" / "raw"


def load_pdfs(
    data_dir: str | Path | None = None,
    files: list[str | Path] | None = None,
) -> list[Document]:
    """Read PDFs and return one LlamaIndex ``Document`` per source file.

    Args:
        data_dir: Directory to scan for ``*.pdf`` files. Defaults to
            ``<project_root>/data/raw``.
        files: Explicit list of PDF paths. When provided, ``data_dir`` is
            ignored — useful for tests that want a single file.

    Raises:
        FileNotFoundError: If neither ``data_dir`` nor ``files`` resolves to
            any PDFs.
    """
    if files:
        paths = [Path(p) for p in files]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"PDF files not found: {missing}")
        reader = SimpleDirectoryReader(
            input_files=[str(p) for p in paths],
            file_extractor=_pdf_extractor(),
        )
    else:
        target = Path(data_dir) if data_dir is not None else _project_data_dir()
        if not target.is_dir():
            raise FileNotFoundError(f"PDF directory not found: {target}")
        reader = SimpleDirectoryReader(
            input_dir=str(target),
            required_exts=[".pdf"],
            recursive=False,
            file_extractor=_pdf_extractor(),
        )

    docs = reader.load_data()
    if not docs:
        raise FileNotFoundError("No PDF documents were loaded")
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    include_tables: bool = True,
) -> list[Document]:
    """Split each input ``Document`` into chunk-sized ``Document`` objects.

    Each input ``Document`` is first exploded by detected section header
    (see :func:`_split_document_by_sections`); the resulting per-section
    Documents are then chunked with :class:`SentenceSplitter`. Each emitted
    chunk carries a ``section_type`` field (``"primary"`` /
    ``"secondary"`` / ``"unknown"``), a globally-unique ``chunk_id``, and the
    source ``file_name`` for downstream filtering. Embedding is excluded on
    the raw ``file_path`` key to keep the vector representation focused on
    prose.

    When ``include_tables`` is true (the default), statistical-summary
    tables extracted by :mod:`ingestion.table_extractor` are appended after
    the prose chunks. Each table chunk is tagged
    ``section_type="table_summary"`` and the retriever is expected to
    force-retain 1–2 of these in its top-K output.
    """
    if not docs:
        return []
    size = chunk_size if chunk_size is not None else _DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else _DEFAULT_CHUNK_OVERLAP

    splitter = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)

    chunks: list[Document] = []
    chunk_idx = 0
    for doc in docs:
        # 1. Explode the source document by detected section headers.
        section_docs = _split_document_by_sections(doc)
        if not section_docs:
            # Whole document had no detectable structure — treat as unknown.
            section_docs = [
                Document(
                    text=doc.text,
                    metadata={**dict(doc.metadata or {}), "section_type": _SECTION_UNKNOWN},
                    excluded_embed_metadata_keys=["file_path"],
                )
            ]
        # 2. Chunk each section separately so the section_type is preserved.
        nodes = splitter.get_nodes_from_documents(section_docs, show_progress=False)
        for node in nodes:
            meta = dict(node.metadata or {})
            meta.setdefault("chunk_id", chunk_idx)
            meta.setdefault("file_name", meta.get("file_name", ""))
            meta.setdefault("section_type", _SECTION_UNKNOWN)
            chunks.append(
                Document(
                    text=node.get_content(),
                    metadata=meta,
                    excluded_embed_metadata_keys=["file_path"],
                )
            )
            chunk_idx += 1

    # 3. Append table_summary chunks so the retriever can promote them in
    #    its top-K. The extractor already emits Documents with the right
    #    metadata; we just need globally-unique chunk_ids.
    if include_tables:
        for table_doc in extract_table_summaries(docs):
            meta = dict(table_doc.metadata or {})
            meta["chunk_id"] = chunk_idx
            chunks.append(
                Document(
                    text=table_doc.text,
                    metadata=meta,
                    excluded_embed_metadata_keys=["file_path"],
                )
            )
            chunk_idx += 1
    return chunks


def parse_pdfs(
    data_dir: str | Path | None = None,
    files: list[str | Path] | None = None,
    config: AppConfig | None = None,
    *,
    include_tables: bool = True,
) -> tuple[list[Document], list[Document]]:
    """Load PDFs and chunk them in one call.

    Returns:
        A 2-tuple ``(file_docs, chunk_docs)``:

        * ``file_docs`` — one ``Document`` per source PDF (pre-chunking).
        * ``chunk_docs`` — one ``Document`` per chunk, with ``chunk_id`` and
          ``file_name`` in ``metadata``. Includes ``table_summary`` chunks
          when ``include_tables`` is true (default).
    """
    cfg = config or load_config()
    file_docs = load_pdfs(data_dir=data_dir, files=files)
    chunk_docs = chunk_documents(
        file_docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        include_tables=include_tables,
    )
    return file_docs, chunk_docs


def main() -> None:
    """CLI entry point: ingest all PDFs under ``data/raw`` and print stats."""
    cfg = load_config()
    file_docs, chunk_docs = parse_pdfs(config=cfg)

    # De-duplicate source file names for a readable count.
    source_files = sorted({d.metadata.get("file_name", "?") for d in file_docs})
    table_chunks = [
        c for c in chunk_docs if (c.metadata or {}).get("section_type") == "table_summary"
    ]
    print(f"PDFs processed  : {len(source_files)}")
    for name in source_files:
        print(f"  - {name}")
    print(f"Chunks generated: {len(chunk_docs)}")
    print(
        f"  - prose (primary/secondary/unknown): {len(chunk_docs) - len(table_chunks)}"
    )
    print(f"  - table_summary                       : {len(table_chunks)}")
    print(
        f"Splitter config : chunk_size={cfg.chunk_size}, "
        f"chunk_overlap={cfg.chunk_overlap}"
    )


__all__ = ["load_pdfs", "chunk_documents", "parse_pdfs", "main"]


if __name__ == "__main__":
    main()
