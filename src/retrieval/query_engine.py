"""RAG query engine for the OncoImmunoRAG project.

Loads the persistent ChromaDB vector store produced by
:mod:`ingestion.vector_store`, retrieves the top-K chunks for a question,
and synthesises a citation-bearing Chinese answer by calling the configured
OpenAI-compatible LLM directly (no ``RetrieverQueryEngine`` wrapper — that
path uses llama-index's ``Refine`` template which triggers unstable TLS
handshakes on some OpenAI-compatible proxies).

Typical usage::

    poetry run python -m retrieval.query_engine

The module is also importable::

    from retrieval.query_engine import QueryEngine, query

    engine = QueryEngine.from_config()
    result = engine.query("...")
    print(result.answer, result.sources)
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from utils.config_loader import AppConfig, load_config, resolve_config_path
from utils.prompts import (
    discussion_section_rule,
    statistical_significance_rules,
)

# Pure-Python top-K selection logic (no ChromaDB / OpenAI deps).
from retrieval.selector import (
    get_rerank_enabled_default,
    rerank_hits,
    select_top_k,
    set_rerank_enabled_default,
)

# Mirrors the collection name created by ``ingestion.vector_store``.
_COLLECTION_NAME = "oncoimmuno_rag"

# Default retrieval depth. The task brief asks for top-5.
_DEFAULT_TOP_K = 5

# Citation tag the LLM is told to emit. Public so tests can match against it.
_CITATION_RE = re.compile(r"\[citation:\s*([^\]\n]+?)\s*\]")

# Prompt template — verbatim from the task brief. We substitute
# ``{context_str}`` and ``{query_str}`` ourselves; the curly braces around
# ``{context_str}`` are intentional and must be preserved for ``str.format``.
_QA_PROMPT_STR = (
    "请基于以下文献片段回答用户问题，并严格区分以下两类信息：\n"
    "A. 本研究的直接结论（来自该文献 Results / Conclusions 部分）："
    "在句末使用 [citation: 文件名] 标注来源；\n"
    "B. 文献中引用的其他研究（来自 Discussion / References 部分提及的其他试验）："
    "必须用'根据该文献引用的 XX 研究……'开头表述，"
    "句末【严禁】添加 [citation: 文件名] —— 因为这不是该文献自身的结果。\n"
    "如果片段中两类信息混杂，请分开成两段表述。\n"
    "若上下文无相关信息，请明确回答'知识库中未提及'。\n"
    "\n"
    "=== 示例 ===\n"
    "本研究的直接结论：IMvigor011 试验显示 ctDNA 指导的阿替利珠单抗辅助治疗显著延长了无病生存期 [citation: Powles_2025_NEJM_IMvigor011-ctDNA.pdf]。\n"
    "文献中引用的其他研究：根据该文献引用的 JAVELIN Bladder 100 研究，阿维鲁单抗维持治疗延长了晚期患者的总生存期。\n"
    "=== 示例结束 ===\n"
    "\n"
    "上下文：\n{context_str}\n"
    "问题：\n{query_str}\n"
    "回答："
)

# Tell the model (very explicitly) to answer in Chinese — keeps the
# formatting of ``[citation: ...]`` consistent across calls.
_SYSTEM_PROMPT = (
    "你是一名肿瘤免疫治疗循证助手。"
    "所有回答必须使用中文。"
    "硬性规则：只有当句子陈述的是【该文献自身】的 Results / Conclusions 时，"
    "才允许在句末写 [citation: 文件名]。"
    "凡是引用、转述、或归纳自其他试验的信息（即 Discussion / References 中提到"
    "的其他研究），必须以'根据该文献引用的 XX 研究……'开头，"
    "且绝对不要在该句末尾附加 [citation: 文件名]。\n\n"
    + statistical_significance_rules()
    + "\n\n"
    + discussion_section_rule()
)


def _vector_db_path() -> Path:
    """Return ``<project_root>/data/vector_db`` regardless of CWD."""
    return resolve_config_path().parent / "data" / "vector_db"


def _require_api_key(cfg: AppConfig) -> str:
    """Return a usable API key or raise with a user-actionable message."""
    key = cfg.effective_openai_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and config.yaml has no openai_api_key. "
            "Set the OPENAI_API_KEY environment variable before running this "
            "module, e.g.:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  poetry run python -m retrieval.query_engine"
        )
    return key


def _build_llm(cfg: AppConfig) -> OpenAI:
    """Instantiate the LLM using config + env-supplied credentials.

    We use the raw ``openai.OpenAI`` SDK rather than
    ``llama_index.llms.openai.OpenAI`` because the latter validates model
    names against an OpenAI-only allow-list — which rejects third-party
    models served by compatible proxies (e.g. ``MiniMax-Text-01``).
    """
    api_key = _require_api_key(cfg)
    base_url = cfg.extras.get("llm_base_url") or os.environ.get(
        "LLM_BASE_URL", "https://api.openai.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


def _llm_model_name(cfg: AppConfig) -> str:
    return cfg.extras.get("llm_model") or os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _llm_complete(client: OpenAI, model: str, prompt: str, system: str) -> str:
    """Thin wrapper around ``chat.completions.create`` returning the text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def _build_embed_model(cfg: AppConfig) -> HuggingFaceEmbedding:
    """Rehydrate the same embedder that produced the on-disk vectors."""
    return HuggingFaceEmbedding(model_name=cfg.embedding_model)


def _open_index(cfg: AppConfig) -> VectorStoreIndex:
    """Open the existing Chroma collection as a LlamaIndex vector store."""
    db_path = _vector_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {db_path}. Run "
            "`poetry run python -m ingestion.vector_store` first."
        )
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(_COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover — chromadb raises ValueError
        raise RuntimeError(
            f"Collection {_COLLECTION_NAME!r} missing from {db_path}. "
            "Re-run the ingestion step."
        ) from exc
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=_build_embed_model(cfg)
    )


def _format_context(nodes: Sequence[object]) -> str:
    """Join retrieved chunks into a single context string the LLM can read.

    Each chunk is prefixed with its source ``file_name`` so the model can
    reference it directly with the ``[citation: ...]`` tag the prompt asks for.
    """
    if not nodes:
        return "(无上下文)"
    parts: list[str] = []
    for i, node in enumerate(nodes, start=1):
        text = getattr(node, "get_content", lambda: str(node))()
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        meta = getattr(node, "metadata", {}) or {}
        src = meta.get("file_name") or meta.get("file_path") or "未知来源"
        parts.append(f"[片段 {i}] (来源: {src})\n{cleaned}")
    return "\n\n".join(parts)


def _parse_citations(answer: str) -> list[str]:
    """Extract unique ``[citation: ...]`` filenames from the LLM answer."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        name = match.group(1).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


# Sentence-split regex that tolerates Chinese full-width punctuation.
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")

# Heading fragments that signal we just entered a "direct-conclusion" zone.
# The first zone marker wins; later lines (until the next major heading)
# may carry ``[citation: 文件名]`` freely.
#
# Note: we deliberately do not anchor with ``\b`` because it is based on
# ASCII word characters and silently fails between two CJK characters
# (e.g. between ``的`` and ``其`` in ``文献中引用的其他研究``). Same reason
# no end-of-keyword lookahead is used — a Chinese character following the
# keyword would defeat ``(?=[\s\n:：。,，；;、]|$)``.
_DIRECT_ZONE_RE = re.compile(
    r"^[ \t>]*(?:#{1,6}\s+|[A-D][\.\)、]\s+|\*\*)?\s*"
    r"(?:本研究的直接结论|A[\.\)、]|Results?|Conclusions?|Abstract)",
    re.IGNORECASE,
)

# Heading fragments that signal we just entered an "indirect / third-party
# citation" zone. While in this zone, every ``[citation: ...]`` tag is
# stripped from every sentence — including bullets, sub-clauses, and the
# heading line itself.
_INDIRECT_ZONE_RE = re.compile(
    r"^[ \t>]*(?:#{1,6}\s+|[A-D][\.\)、]\s+|\*\*)?\s*"
    r"(?:B[\.\)、]|文献中引用的|文献中引用|其他研究|"
    r"Discussion|References?|Bibliography|间接引用)",
    re.IGNORECASE,
)


def _strip_indirect_citations(answer: str) -> str:
    """Remove ``[citation: ...]`` from sentences inside the indirect zone.

    The prompt instructs the LLM to never attach ``[citation: ...]`` after
    sentences that summarise a third-party study referenced by the source
    paper. The model sometimes leaks those tags onto bullets and sub-clauses
    that sit *under* an indirect heading — e.g.::

        B. 文献中引用的其他研究
        1. 根据该文献引用的 X 研究, ...  ← already stripped
           在不适合顺铂治疗的患者中, ... [citation: foo.pdf]  ← leaked; stripped here

    Strategy: walk the answer line-by-line, flipping a zone flag whenever a
    heading matches :data:`_DIRECT_ZONE_RE` or :data:`_INDIRECT_ZONE_RE`.
    Inside the indirect zone, every ``[citation: ...]`` is removed.
    """
    if not answer:
        return answer

    out: list[str] = []
    in_indirect = False
    for line in answer.split("\n"):
        if _INDIRECT_ZONE_RE.match(line):
            in_indirect = True
        elif _DIRECT_ZONE_RE.match(line):
            in_indirect = False

        if in_indirect:
            line = _CITATION_RE.sub("", line)
            # Tidy up artifacts left by the removal (double spaces, trailing
            # whitespace before a closing punctuation).
            line = re.sub(r"[ \t]{2,}", " ", line).rstrip()
            # If the line was *only* a citation tag, leave a blank line so
            # paragraph structure is preserved.
            if not line.strip():
                line = ""

        out.append(line)
    return "\n".join(out)


@dataclass
class QueryResult:
    """Bundle returned by :meth:`QueryEngine.query`."""

    answer: str
    sources: list[str]  # unique file_name values from retrieved chunks + LLM tags
    source_nodes: object  # raw NodeWithScore list, kept for callers/tests

    def __str__(self) -> str:  # pragma: no cover — cosmetic
        return self.answer


class QueryEngine:
    """Thin wrapper around retriever + LLM, with explicit citation handling."""

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: OpenAI,
        model: str,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._index = index
        self._llm = llm
        self._model = model
        self._top_k = top_k

    @classmethod
    def from_config(cls, cfg: AppConfig | None = None) -> "QueryEngine":
        cfg = cfg or load_config()
        return cls(
            index=_open_index(cfg),
            llm=_build_llm(cfg),
            model=_llm_model_name(cfg),
            top_k=_DEFAULT_TOP_K,
        )

    def retrieve(
        self,
        question: str,
        *,
        canonical_file: str | None = None,
    ) -> list:
        """Return the top-K NodeWithScore hits for ``question``.

        When ``canonical_file`` is supplied (the target PDF for the
        question — typically the scenario's "primary" trial), it is
        passed to :func:`retrieval.selector.select_top_k` which forces
        at least one chunk from that file into top-K. This closes the
        "EV-302 hijacks every McGregor-DAD query" retrieval bug.

        Args:
            question: The natural-language question.
            canonical_file: Optional ``file_name`` of the source PDF
                that genuinely answers the question.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        # Over-fetch with MMR so the candidate pool is diverse across files.
        # Without MMR, a query that's a near-synonym of one paper's title
        # floods top-K with that one file. We grab ~5× the final top-k and
        # then re-rank by section_type via :func:`retrieval.selector.select_top_k`,
        # which force-promotes ``table_summary`` chunks to the front.
        over_fetch = max(self._top_k * 5, 25)
        over_fetch = min(over_fetch, 80)
        retriever = self._index.as_retriever(
            similarity_top_k=over_fetch,
            vector_store_query_mode="mmr",
            mmr_threshold=0.5,  # 0.0 = pure relevance, 1.0 = max diversity
        )
        raw = list(retriever.retrieve(question))

        # Canonical routing: when a target PDF is named, MMR may have
        # pruned all of its chunks (a niche paper loses to a co-occurring
        # high-frequency one). We then do a *second* retrieval scoped to
        # the canonical file to ensure its Primary chunks are in the pool.
        # This is the critical fix for "EV-302 hijacks every niche query".
        if canonical_file:
            raw.extend(self._retrieve_canonical_chunks(canonical_file, top_n=6))

        # ---- evidence-grade rerank layer (2026-09 spec) ----
        # Inserted between MMR (over-fetch) and select_top_k. Reads the
        # module-level switch exposed by retrieval.selector; can also be
        # toggled at query time via ``QueryEngine.set_rerank_enabled``.
        registry_lookup = _get_registry_lookup()
        raw = rerank_hits(
            raw,
            enabled=get_rerank_enabled_default(),
            registry_lookup=registry_lookup,
        )

        return select_top_k(raw, self._top_k, canonical_file=canonical_file)

    def _retrieve_canonical_chunks(
        self,
        canonical_file: str,
        *,
        top_n: int = 6,
    ) -> list:
        """Query the underlying collection for chunks tagged with ``canonical_file``.

        Used as a fallback when MMR over-fetch has excluded the target
        paper (the "retriever hijacked by EV-302" failure mode). Returns
        NodeWithScore objects that the selector can pick up — note that
        they share the same vector space as the main pool, so duplicates
        de-dup by ``node_id`` at selection time.
        """
        # We need the underlying chroma collection to filter on
        # ``file_name``. The ``VectorStoreIndex`` exposes a ``_vector_store``
        # which is our ``ChromaVectorStore`` — peek past the wrapper to
        # run a filtered get().
        try:
            vs = getattr(self._index, "_vector_store", None)
            chroma = getattr(vs, "chroma_collection", None) if vs else None
        except Exception:
            chroma = None
        if chroma is None:  # pragma: no cover — defensive
            return []

        try:
            res = chroma.get(
                where={"file_name": canonical_file},
                limit=top_n,
            )
        except Exception:
            return []

        # Rehydrate to NodeWithScore objects so the selector can sort on
        # ``.score`` and read ``.node.metadata``. The TextNodeStub below
        # matches the surface area selector.py uses.
        from llama_index.core.schema import TextNode  # lazy import

        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        nodes: list = []
        for node_id, doc, meta in zip(ids, docs, metas):
            node = TextNode(id_=node_id, text=doc or "", metadata=meta or {})
            # Wrap in a minimal NodeWithScore-like shim.
            nodes.append(_NodeWithScore(node, 1.0))
        return nodes

    def set_rerank_enabled(self, enabled: bool) -> None:
        """Toggle the rerank layer for this engine (and the module default).

        Convenience for callers that want per-instance control without
        reaching into :mod:`retrieval.selector` directly.
        """
        set_rerank_enabled_default(bool(enabled))

    def get_rerank_enabled(self) -> bool:
        """Return the current rerank switch state."""
        return get_rerank_enabled_default()


class _NodeWithScore:
    """Minimal NodeWithScore shim used by the canonical targeted-retrieve path.

    The selector only needs ``.node`` (with ``.metadata``) and ``.score``.
    LlamaIndex's own ``NodeWithScore`` requires a ``BaseNode``; ``TextNode``
    satisfies that. We just need to keep the import surface simple and
    avoid pulling llama-index into ``retrieval.selector`` itself.
    """

    __slots__ = ("node", "score")

    def __init__(self, node, score: float) -> None:
        self.node = node
        self.score = float(score)

    def query(self, question: str) -> QueryResult:
        """Run retrieval + LLM answer synthesis in one call."""
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        hits = self.retrieve(question)

        # Pull unique source filenames straight from the chunks (these are
        # always present even if the LLM forgets to emit ``[citation: ...]``).
        sources: list[str] = []
        for h in hits:
            meta = getattr(h.node, "metadata", {}) or {}
            name = meta.get("file_name") or meta.get("file_path") or "unknown"
            if name not in sources:
                sources.append(name)

        # Build the prompt and call the LLM directly. We pass the full
        # ``hits`` (NodeWithScore) so ``_format_context`` can prepend each
        # chunk's source filename — otherwise the LLM has nothing to cite.
        prompt = _QA_PROMPT_STR.format(
            context_str=_format_context(hits), query_str=question
        )
        answer = _llm_complete(self._llm, self._model, prompt, _SYSTEM_PROMPT)

        # Strip ``[citation: ...]`` from any sentence that the prompt
        # classified as "indirect" (i.e. citing a third-party study that the
        # source paper itself references). The LLM sometimes ignores the
        # instruction; this post-processor enforces it deterministically.
        answer = _strip_indirect_citations(answer)

        # Surface any ``[citation: ...]`` tags the LLM emitted, after the
        # chunk-level sources (the prompt asked for those first). These are
        # now guaranteed to be only the paper's own conclusions, so we also
        # add them to the front of the ``sources`` list (de-duped).
        llm_citations = [c for c in _parse_citations(answer) if c not in sources]
        sources = llm_citations + sources

        return QueryResult(answer=answer, sources=sources, source_nodes=hits)


# Lazy-imported helper used by ``retrieve`` to look up trial metadata for
# the rerank layer. Centralised so tests can monkey-patch / inject mocks
# without touching the query_engine module surface.
_REGISTRY_LOOKUP = None  # type: ignore[var-annotated]


def _get_registry_lookup():
    """Return a singleton trial-registry lookup callable.

    Imported lazily because :mod:`utils.trial_registry` itself imports
    nothing heavy, but keeping the import deferred lets unit tests of
    :mod:`retrieval.selector` avoid pulling ``utils`` into their import
    graph.
    """
    global _REGISTRY_LOOKUP
    if _REGISTRY_LOOKUP is None:
        from utils.trial_registry import lookup as _lookup  # local import
        _REGISTRY_LOOKUP = _lookup
    return _REGISTRY_LOOKUP


# --- module-level singleton helpers (for `from retrieval.query_engine import query`) ---

_engine_singleton: QueryEngine | None = None


def _get_engine() -> QueryEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = QueryEngine.from_config()
    return _engine_singleton


def query(question: str) -> tuple[str, list[str]]:
    """Functional entry point — returns ``(answer_text, source_list)``."""
    return _get_engine().query(question)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

_DEFAULT_TEST_QUESTION = (
    "What is the efficacy of enfortumab vedotin and pembrolizumab in advanced "
    "urothelial cancer?"
)


def _print_result(question: str, result: QueryResult) -> None:
    print("=" * 72)
    print(f"Q: {question}")
    print("=" * 72)
    print(result.answer)
    print()
    print(f"[Retrieved sources ({len(result.sources)} unique)]")
    for i, src in enumerate(result.sources, start=1):
        print(f"  ({i}) {src}")
    print()
    print(f"[Source-node scores ({len(result.source_nodes)} hits)]")
    for i, sn in enumerate(result.source_nodes, start=1):
        meta = getattr(sn.node, "metadata", {}) or {}
        chunk_id = meta.get("chunk_id", "?")
        snippet = sn.node.get_content().strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        score = getattr(sn, "score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  ({i}) score={score_str} chunk_id={chunk_id} :: {snippet}")


def main(
    question: str = _DEFAULT_TEST_QUESTION,
    *,
    cfg: AppConfig | None = None,
) -> QueryResult:
    """Build the engine, run the test question, and pretty-print the result."""
    cfg = cfg or load_config()
    if not cfg.effective_openai_api_key():
        warnings.warn(
            "OPENAI_API_KEY is not set; set it (or openai_api_key in config.yaml) "
            "before running the LLM stage.",
            stacklevel=2,
        )
    engine = QueryEngine.from_config(cfg)
    result = engine.query(question)
    _print_result(question, result)
    return result


__all__ = [
    "QueryEngine",
    "QueryResult",
    "main",
    "query",
]


if __name__ == "__main__":
    main()