"""Vector-store ingestion for the OncoImmunoRAG project.

Reads the chunked ``Document`` objects produced by :mod:`ingestion.pdf_parser`,
embeds them with a HuggingFace sentence-transformers model (configured via
``config.yaml``), and persists them in a local ChromaDB collection under
``<project_root>/data/vector_db/``. After ingestion the script performs a
small sanity-check retrieval for ``"enfortumab vedotin"`` and prints the top
hits so the operator can confirm the index is wired up.

Typical usage::

    poetry run python -m ingestion.vector_store

The module is also importable — :func:`build_index` returns the populated
``VectorStoreIndex`` for downstream code (retrieval, grading, evaluation).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ingestion.pdf_parser import parse_pdfs
from utils.config_loader import load_config, resolve_config_path

# Default collection name. Kept module-level so callers can refer to it
# without re-typing the literal.
_DEFAULT_COLLECTION_NAME = "oncoimmuno_rag"

# Default query used by the self-test at the bottom of ``main``.
_SELF_TEST_QUERY = "enfortumab vedotin"
_SELF_TEST_TOP_K = 3


def _vector_db_path() -> Path:
    """Return ``<project_root>/data/vector_db`` regardless of CWD."""
    return resolve_config_path().parent / "data" / "vector_db"


def _reset_vector_db(path: Path) -> None:
    """Wipe the persistent ChromaDB directory so each run starts fresh.

    Without this, repeated invocations would keep appending to the same
    collection and the ``入库完成`` count would climb on every run, hiding
    real regressions. The directory is recreated empty.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_index(
    *,
    reset: bool = True,
    collection_name: str = _DEFAULT_COLLECTION_NAME,
) -> tuple[VectorStoreIndex, list[Document]]:
    """Parse PDFs, embed them, and write the index to the Chroma store.

    Args:
        reset: When ``True`` (default), delete any existing vector DB
            directory before ingesting. Set ``False`` for incremental
            appends — note Chroma collection must then already exist.
        collection_name: Name of the Chroma collection to create/use.

    Returns:
        A 2-tuple ``(index, chunk_docs)`` where ``index`` is the populated
        LlamaIndex wrapper around the Chroma store and ``chunk_docs`` is the
        list of ``Document`` objects that were indexed (useful for the
        caller to print statistics).
    """
    cfg = load_config()
    db_path = _vector_db_path()
    if reset:
        _reset_vector_db(db_path)

    # 1. Parse & chunk PDFs.
    _, chunk_docs = parse_pdfs(config=cfg)
    if not chunk_docs:
        raise RuntimeError(
            "No chunked Documents were produced by parse_pdfs(); "
            "check that data/raw/ contains PDFs."
        )

    # 2. Initialise the embedding model (HuggingFace / sentence-transformers).
    embed_model = HuggingFaceEmbedding(model_name=cfg.embedding_model)

    # 3. Build a persistent Chroma client + collection, wrap in LlamaIndex's
    #    vector-store adapter, then wire it into a StorageContext.
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Embed + persist. ``from_documents`` handles batching and metadata
    #    forwarding; ``show_progress=True`` is helpful on big corpora.
    index = VectorStoreIndex.from_documents(
        chunk_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    return index, chunk_docs


def _format_hit(hit: object, rank: int) -> str:
    """Return a human-readable summary of one retrieval hit.

    ``hit`` may be a :class:`NodeWithScore` or a bare ``BaseNode``; we pull
    ``score`` from whichever wrapper carries it, and ``node`` for the rest.
    """
    score = getattr(hit, "score", None)
    node = getattr(hit, "node", hit)

    text = getattr(node, "get_content", lambda: str(node))()
    meta = getattr(node, "metadata", {}) or {}
    source = meta.get("file_name") or meta.get("file_path") or "<unknown>"
    chunk_id = meta.get("chunk_id", "?")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

    snippet = text.strip().replace("\n", " ")
    if len(snippet) > 400:
        snippet = snippet[:400] + "…"
    return (
        f"[{rank}] score={score_str} source={source} chunk_id={chunk_id}\n"
        f"    {snippet}"
    )


def run_self_test(index: VectorStoreIndex) -> None:
    """Retrieve the top-k chunks for ``"enfortumab vedotin"`` and print them."""
    retriever = index.as_retriever(similarity_top_k=_SELF_TEST_TOP_K)
    # ``retrieve`` returns NodeWithScore; the older API returns a
    # ``VectorStoreQueryResult``. The retriever's ``retrieve`` returns a
    # ``list[NodeWithScore]`` which exposes ``.node`` and ``.score``.
    response = retriever.retrieve(_SELF_TEST_QUERY)

    # Defensive: support both NodeWithScore lists and bare Response objects.
    hits = getattr(response, "nodes", None)
    if hits is None:
        hits = response  # already a list of NodeWithScore

    print()
    print(f"=== Self-test query: {_SELF_TEST_QUERY!r} (top {_SELF_TEST_TOP_K}) ===")
    if not hits:
        print("  (no hits returned)")
        return
    for rank, hit in enumerate(hits, start=1):
        print(_format_hit(hit, rank))


def main() -> None:
    """Entry point: ingest, report, and self-test."""
    index, chunk_docs = build_index()
    print(f"入库完成，共写入 {len(chunk_docs)} 个向量")
    run_self_test(index)


__all__ = [
    "build_index",
    "run_self_test",
    "main",
    "_DEFAULT_COLLECTION_NAME",
    "_SELF_TEST_QUERY",
    "_SELF_TEST_TOP_K",
]


if __name__ == "__main__":
    main()