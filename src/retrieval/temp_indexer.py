"""Temporary ChromaDB collection manager for OncoImmunoRAG.

This module lets the app ingest a single external document (PDF text or
abstract from PubMed) into a *temporary* collection that lives in memory
or in an isolated on-disk path, so it can be queried with the same
embedding model but never appears in the main vector store.

The GRADE evaluator then runs :func:`grade_extraction` against the temp
chunks as usual — the temp collection is invisible to the rest of the
app and is deleted at the end of the scenario.

Typical usage::

    from retrieval.temp_indexer import (
        TempIndexer,
        index_temp_document,
        clear_temp_collection,
    )

    # 1. ingest text
    chunks, meta = index_temp_document(
        text=abstract_or_pdf_text,
        metadata={"file_name": "PMID_3717008.pdf",
                  "design_hint": "RCT", "n_hint": 886},
    )
    # 2. grade_extraction on those chunks
    from grading.grade_evaluator import grade_extraction
    grade = grade_extraction(chunks, canonical_file_name="...", cfg=cfg)
    # 3. clean up
    clear_temp_collection()
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# Defaults — keep consistent with the main KB config.
_DEFAULT_CHUNK_SIZE = 1024
_DEFAULT_CHUNK_OVERLAP = 200
_EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"


@dataclass
class TempIndexResult:
    """Returned by :func:`index_temp_document`."""
    chunks: list[str] = field(default_factory=list)
    collection_name: str = ""
    file_name: str = ""
    metadata: dict = field(default_factory=dict)


class TempIndexer:
    """Stateful context manager for a single temp collection.

    Example:
            with TmpIdx() as idx:
                idx.add_text("...", meta={"file_name": "foo.pdf"})
                nodes = idx.retrieve("What is the sample size?")
            # collection auto-deleted on exit
    """

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        persist_path: str | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        embed_model_name: str = _EMBED_MODEL_NAME,
    ) -> None:
        self.collection_name = (
            collection_name or f"temp_oncoimmuno_{uuid.uuid4().hex[:8]}"
        )
        # Ephemeral by default: every TempIndexer gets its own tempdir that
        # is cleaned up when the context manager does.
        self._owns_dir = persist_path is None
        self.persist_path = persist_path or tempfile.mkdtemp(
            prefix=f"oncoimmuno_temp_{self.collection_name}_"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name

        self._client: chromadb.Client | None = None
        self._collection = None
        self._embed_model: HuggingFaceEmbedding | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "TempIndexer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._client is not None:
            return
        os.makedirs(self.persist_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_path)
        # use cosine similarity to match the main KB
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

    def close(self) -> None:
        """Delete the temp collection and (if we own the dir) the temp dir."""
        try:
            if self._client is not None and self._collection is not None:
                try:
                    self._client.delete_collection(self.collection_name)
                except Exception:
                    pass
        finally:
            self._collection = None
            self._client = None
            self._embed_model = None
            if self._owns_dir and os.path.isdir(self.persist_path):
                shutil.rmtree(self.persist_path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_text(
        self,
        text: str,
        *,
        metadata: dict | None = None,
        file_name: str | None = None,
    ) -> list[str]:
        """Split ``text`` into chunks and store them in the temp collection.

        Returns the list of chunk texts in their original order.
        """
        if not text or not text.strip():
            return []
        if self._collection is None or self._embed_model is None:
            self.open()

        md = dict(metadata or {})
        if file_name:
            md.setdefault("file_name", file_name)
        md.setdefault("section_type", "primary")

        # Use the project's SentenceSplitter so chunk sizes match the
        # main KB (1024/200 by default).
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents(
            [Document(text=text, metadata=md)],
            show_progress=False,
        )
        if not nodes:
            return []

        ids = [str(uuid.uuid4()) for _ in nodes]
        texts = [n.get_content() for n in nodes]
        metas = [dict(md) for _ in nodes]
        # ChromaDB needs flat metadata; stringify non-primitives defensively.
        metas = [_flatten_metadata(m) for m in metas]

        # Embedding + insert
        embeddings = [
            self._embed_model.get_text_embedding(text)
            if self._embed_model is not None else []
            for text in texts
        ]

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
        )

        return texts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 5) -> list[str]:
        """Return the top-K chunk texts most similar to ``question``."""
        if self._collection is None or self._embed_model is None:
            return []
        if not question or not question.strip():
            return []
        emb = self._embed_model.get_text_embedding(question)
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=top_k,
        )
        docs = res.get("documents") or []
        return docs[0] if docs else []


# ----------------------------------------------------------------------
# Module-level convenience (matches the API requested in the spec).
# ----------------------------------------------------------------------
_active_indexer: TempIndexer | None = None


def index_temp_document(
    text: str,
    metadata: dict | None = None,
    *,
    file_name: str | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> TempIndexResult:
    """Create-or-append a singleton :class:`TempIndexer` with one document.

    The first call instantiates the singleton; subsequent calls add chunks
    to the same collection. Use :func:`clear_temp_collection` to remove it.

    Returns the chunks produced from this call so the caller can feed them
    straight into :func:`grade_extraction`.
    """
    global _active_indexer
    if _active_indexer is None:
        _active_indexer = TempIndexer(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        _active_indexer.open()

    chunks = _active_indexer.add_text(
        text, metadata=metadata, file_name=file_name,
    )
    return TempIndexResult(
        chunks=chunks,
        collection_name=_active_indexer.collection_name,
        file_name=file_name or (metadata or {}).get("file_name", ""),
        metadata=metadata or {},
    )


def clear_temp_collection() -> None:
    """Tear down the singleton temp indexer, if any."""
    global _active_indexer
    if _active_indexer is not None:
        _active_indexer.close()
        _active_indexer = None


def _flatten_metadata(md: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool primitives in metadata.
    Stringify anything else.
    """
    out: dict = {}
    for k, v in (md or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if v is not None else ""
        else:
            out[k] = str(v)
    return out


__all__ = [
    "TempIndexer",
    "TempIndexResult",
    "index_temp_document",
    "clear_temp_collection",
]