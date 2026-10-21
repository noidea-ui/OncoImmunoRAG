"""Incremental PDF ingestion for the OncoImmunoRAG project.

Appends new chunks to the existing ChromaDB collection **without** wiping the
on-disk store. Use this when the collection already contains vectors (e.g. the
EV-302 baseline) and you want to add more PDFs incrementally.

Typical usage::

    poetry run python -m ingestion.append_pdfs \\
        --raw-dir data/raw \\
        --exclude NEJMoa2312117.pdf

The exclusion list defaults to whatever is currently stored in the collection,
so a plain ``poetry run python -m ingestion.append_pdfs`` will ingest every
PDF in ``data/raw/`` that isn't already indexed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from ingestion.pdf_parser import parse_pdfs
from ingestion.vector_store import (
    _DEFAULT_COLLECTION_NAME,
    _vector_db_path,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.config_loader import load_config, resolve_config_path


def _existing_file_names(
    client: chromadb.PersistentClient, collection_name: str
) -> set[str]:
    """Return every distinct ``file_name`` already present in the collection."""
    collection = client.get_collection(collection_name)
    if collection.count() == 0:
        return set()
    rows = collection.get(include=["metadatas"])["metadatas"] or []
    return {row.get("file_name", "") for row in rows if row.get("file_name")}


def append_index(
    *,
    files: list[str | Path] | None = None,
    data_dir: str | Path | None = None,
    exclude: set[str] | None = None,
    collection_name: str = _DEFAULT_COLLECTION_NAME,
) -> tuple[VectorStoreIndex, list, set[str], int]:
    """Append new PDFs to the existing Chroma collection.

    Args:
        files: Explicit PDF paths to ingest. If ``None``, every PDF under
            ``data_dir`` is considered (minus ``exclude``).
        data_dir: Directory scanned for PDFs when ``files`` is ``None``.
            Defaults to ``<project_root>/data/raw``.
        exclude: Filenames to skip (case-sensitive basename match against
            ``file_name`` metadata). ``None`` → auto-detect existing
            collection contents.
        collection_name: Chroma collection to append into.

    Returns:
        ``(index, chunk_docs, before_file_names, before_count)``.
    """
    db_path = _vector_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {db_path}. Run the full ingestion "
            "(`poetry run python -m ingestion.vector_store`) first."
        )

    cfg = load_config()
    client = chromadb.PersistentClient(path=str(db_path))

    # Auto-detect what's already in the collection so we never re-ingest.
    existing_files = _existing_file_names(client, collection_name)
    skip = set(exclude) if exclude is not None else existing_files

    # Resolve the PDF list to ingest.
    if files is not None:
        paths = [Path(p) for p in files]
    else:
        raw = (
            Path(data_dir)
            if data_dir is not None
            else resolve_config_path().parent / "data" / "raw"
        )
        paths = sorted(raw.glob("*.pdf"))

    # Filter against the skip list using the basename (matches SimpleDirectoryReader).
    selected = [p for p in paths if p.name not in skip]

    print(f"[append] existing files in collection: {len(existing_files)}")
    for name in sorted(existing_files):
        print(f"  - {name}")
    print(f"[append] skip list: {sorted(skip)}")
    print(f"[append] candidate PDFs in data/raw/: {len(paths)}")
    print(f"[append] new PDFs to ingest: {len(selected)}")
    for p in selected:
        print(f"  + {p.name}")

    if not selected:
        return None, [], existing_files, client.get_collection(collection_name).count()

    # Parse + chunk only the new files.
    _, chunk_docs = parse_pdfs(files=selected, config=cfg)
    if not chunk_docs:
        raise RuntimeError("No chunks were produced from the selected PDFs.")

    # Append into the existing collection — no reset, no wipe.
    embed_model = HuggingFaceEmbedding(model_name=cfg.embedding_model)
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        chunk_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    return None, chunk_docs, existing_files, collection.count()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory containing PDFs (default: <project_root>/data/raw).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Filenames to skip. Default: auto-detect existing collection.",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION_NAME,
        help="Chroma collection name (default: oncoimmuno_rag).",
    )
    args = parser.parse_args(argv)

    exclude = set(args.exclude) if args.exclude else None
    _, chunk_docs, _before_files, total_after = append_index(
        data_dir=args.raw_dir,
        exclude=exclude,
        collection_name=args.collection,
    )

    print()
    print("=" * 60)
    print(f"新增 chunks    : {len(chunk_docs)}")
    print(f"入库后总 chunks: {total_after}")
    print("=" * 60)


if __name__ == "__main__":
    main()