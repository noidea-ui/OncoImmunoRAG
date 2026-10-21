"""Build 22-paper DB using a hardened embedding model path.

The original `BAAI/bge-large-en-v1.5` model fails on Python 3.14 +
sentence-transformers 5.7.0 ("Unrecognized processing class"). This
script falls back to `sentence-transformers/all-MiniLM-L6-v2` which is
lightweight and broadly compatible. The retrieval quality differences
between embedding models are acceptable for an A/B rerank experiment
(we are testing the *reranking layer*, not the embedder).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chromadb  # noqa: E402
from llama_index.core import StorageContext, VectorStoreIndex  # noqa: E402

# Force the embedder to a known-compatible model for this experiment.
# The original `BAAI/bge-large-en-v1.5` config triggers a transformers
# auto-class dispatch bug on newer transformers versions, but the model
# files are already cached locally under llama_index's cache. We
# reference the local snapshot path so huggingface_hub doesn't try to
# re-download them.
EMBED_MODEL = (
    "C:/Users/应玥/AppData/Local/llama_index/llama_index/"
    "Cache/models--BAAI--bge-large-en-v1.5/snapshots/"
    "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
)

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception as exc:
    print(f"[fatal] HuggingFaceEmbedding import failed: {exc}")
    sys.exit(1)

from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402

from ingestion.pdf_parser import parse_pdfs  # noqa: E402


_COLLECTION_NAME = "oncoimmuno_rag_22"
_DB_PATH = _PROJECT_ROOT / "data" / "vector_db_22"


def collect_all_pdfs() -> list[Path]:
    sources = [
        _PROJECT_ROOT / "data" / "raw",
        _PROJECT_ROOT / "verification",
        _PROJECT_ROOT / "holdout",
    ]
    out: list[Path] = []
    for src in sources:
        if not src.is_dir():
            print(f"[skip] {src} not a directory", file=sys.stderr)
            continue
        for p in sorted(src.glob("*.pdf")):
            out.append(p)
    return out


def build_22_paper_db(force_rebuild: bool = True) -> tuple[int, int]:
    if force_rebuild and _DB_PATH.exists():
        shutil.rmtree(_DB_PATH)
    _DB_PATH.mkdir(parents=True, exist_ok=True)

    pdfs = collect_all_pdfs()
    print(f"[build_kb_22] {len(pdfs)} PDFs to ingest (embedder: {EMBED_MODEL})")

    # Parse + chunk via the existing pipeline
    from utils.config_loader import load_config
    cfg = load_config()
    _, chunks = parse_pdfs(files=pdfs, config=cfg)
    print(f"[build_kb_22] {len(chunks)} chunks parsed")

    # Coerce non-scalar metadata fields to strings so ChromaDB accepts them.
    # The parser emits some list/tuple metadata (e.g. ``contains_endpoints``
    # from the table extractor); Chroma only persists str/int/float/None.
    def _scrub(m):
        out: dict = {}
        for k, v in (m or {}).items():
            if v is None or isinstance(v, (str, int, float, bool)):
                if isinstance(v, bool):
                    out[k] = str(v)
                else:
                    out[k] = v
            elif isinstance(v, (list, tuple, set)):
                out[k] = ",".join(str(x) for x in v)
            else:
                out[k] = str(v)
        return out

    for ch in chunks:
        ch.metadata = _scrub(ch.metadata)
        # ``excluded_embed_metadata_keys`` is a list — also flatten
        if hasattr(ch, "excluded_embed_metadata_keys"):
            ee = ch.excluded_embed_metadata_keys
            if not isinstance(ee, (list, tuple)) or (
                ee and not all(isinstance(x, str) for x in ee)
            ):
                ch.excluded_embed_metadata_keys = list(ee) if ee else []
    print(f"[build_kb_22] metadata fields coerced to str/int/float/None")

    # Build embedder
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # ChromaDB persist
    client = chromadb.PersistentClient(path=str(_DB_PATH))
    collection = client.get_or_create_collection(_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[build_kb_22] Embedding {len(chunks)} chunks ... (this will take a few minutes)")
    index = VectorStoreIndex.from_documents(
        chunks,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    final_count = collection.count()
    print(f"[build_kb_22] Done: {final_count} chunks in {_COLLECTION_NAME}")
    return len(pdfs), len(chunks)


if __name__ == "__main__":
    build_22_paper_db(force_rebuild=True)
