"""A/B rerank search eval — 22-paper DB, 3 cross-paper queries.

Setup (per 2026-09 spec):

* Vector DB         : ``data/vector_db_22``   (built by ``build_kb_22.py``)
* 22 papers         : 12 dev + 5 verification + 5 holdout
* canonical_file    : None everywhere (no target-paper routing)
* Group A           : rerank OFF, raw MMR-similarity order passed to ``select_top_k``
* Group B           : rerank ON, ``rerank_hits`` between MMR and ``select_top_k``

Queries — clinically curated by hand against the 22-paper catalogue.
Each query has:

  * ``gold_files`` — set of file_names expected to surface in top-K
  * ``gold_design`` — RCT / single-arm / RCT+Meta (mixed)
  * ``gold_table``  — do we expect table_summary chunks to surface? (oracle check)

Metrics (per arm, per query):

  * Hit Rate @ 5      — fraction of gold files with ≥1 chunk in top-5
  * First-hit rank     — rank of the earliest gold file in top-5 (1..5 or "miss")
  * MRR                — 1 / first_hit_rank (0 if miss)
  * table_summary @5   — fraction of top-5 chunks that are table_summary
  * correct-lit @5     — fraction of top-5 chunks from a gold file
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chromadb  # noqa: E402
from llama_index.core import VectorStoreIndex  # noqa: E402

# Use the same proven embedding model the build script used.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402

from retrieval.selector import (  # noqa: E402
    rerank_hits,
    select_top_k,
    set_rerank_enabled_default,
)
from utils.trial_registry import lookup as registry_lookup  # noqa: E402


_COLLECTION_NAME = "oncoimmuno_rag_22"
_DB_PATH = _PROJECT_ROOT / "data" / "vector_db_22"
_DEFAULT_EMBED = (
    "C:/Users/应玥/AppData/Local/llama_index/llama_index/"
    "Cache/models--BAAI--bge-large-en-v1.5/snapshots/"
    "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
)
_TOP_K = 5
_OVER_FETCH = 25
_MMR_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Query catalogue (3 fixed cross-paper queries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    name: str
    text: str
    gold_files: tuple[str, ...]      # file_names expected to surface
    gold_design: str                 # high-level category for telemetry
    expect_table: bool = False       # does the answer require table slice hits?


_QUERIES: tuple[Query, ...] = (
    Query(
        name="PD1_PDL1_advanced_UC_RCTs",
        text=(
            "What are the RCTs showing the effect of PD-1/PD-L1 inhibitors "
            "in advanced urothelial carcinoma?"
        ),
        gold_files=(
            # 1L mUC pembro
            "1-s2.0-S1470204521001522-main.pdf",   # KEYNOTE-361
            # avelumab maintenance
            "NEJMoa2002788.pdf",                  # JAVELIN Bladder 100
            # atezo 2L+
            "1-s2.0-S014067361733297X-main.pdf",   # IMvigor211
            # atezo+chemo 1L (PD-L1 + chemo)
            "Galsky_2024_CellRepMed_IMvigor130-ChemoImmuno.pdf",
            # durva±treme 1L
            "1-s2.0-S1470204520305416-main.pdf",   # DANUBE
        ),
        gold_design="RCT",
        expect_table=True,
    ),
    Query(
        name="SingleArm_NMIBC_immunotherapy",
        text=(
            "Which single-arm trials evaluated oncolytic immunotherapy "
            "or checkpoint blockade in BCG-unresponsive NMIBC?"
        ),
        gold_files=(
            "Balar_2024_LancetOncol_KEYNOTE-057.pdf",
            "Boorjian_2021_LancetOncol_Nadofaragene.pdf",
            "Li_2025_NatMed_Cretostimogene-Nivo.pdf",
            "nihms-1939489.pdf",                                  # SWOG S1605
            "1-s2.0-S1470204526001944-main.pdf",                 # BOND-003 Cohort C
        ),
        gold_design="single-arm",
        expect_table=True,
    ),
    Query(
        name="Adjuvant_MIUC_PFS_OS",
        text=(
            "What are the key PFS and OS data for adjuvant immunotherapy "
            "in muscle-invasive bladder cancer?"
        ),
        gold_files=(
            "Powles_2021_Nature_IMvigor010-ctDNA.pdf",
            "NEJMoa2034442.pdf",                                  # CheckMate 274
            "Powles_2025_NEJM_IMvigor011-ctDNA.pdf",
            "1-s2.0-S1558767324002581-main.pdf",                 # Mamede meta
        ),
        gold_design="RCT-or-Meta",
        expect_table=True,
    ),
)


# ---------------------------------------------------------------------------
# Retrieval pipeline (custom, no QueryEngine singleton so we can swap DB)
# ---------------------------------------------------------------------------


def _open_index():
    """Open the 22-paper Chroma collection as a LlamaIndex vector store."""
    embed_model = HuggingFaceEmbedding(model_name=_DEFAULT_EMBED)
    client = chromadb.PersistentClient(path=str(_DB_PATH))
    try:
        collection = client.get_collection(_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection {_COLLECTION_NAME!r} missing from {_DB_PATH} "
            f"— run scripts/build_kb_22.py first. ({exc})"
        )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )


def _retrieve_topk(query_text: str, *, rerank_enabled: bool, top_k: int = _TOP_K):
    """Return top-K NodeWithScore-shim objects for ``query_text``.

    Mirrors :func:`retrieval.query_engine.QueryEngine.retrieve` but
    parameterised by the rerank switch + a fresh index handle. Canonical
    routing is *disabled* (canonical_file=None) per the experimental
    setup.
    """
    set_rerank_enabled_default(rerank_enabled)
    index = _open_index()
    over_fetch = max(top_k * 5, 25)
    over_fetch = min(over_fetch, 80)
    retriever = index.as_retriever(
        similarity_top_k=over_fetch,
        vector_store_query_mode="mmr",
        mmr_threshold=_MMR_THRESHOLD,
    )
    raw = list(retriever.retrieve(query_text))
    reranked = rerank_hits(
        raw,
        enabled=rerank_enabled,
        registry_lookup=registry_lookup,
    )
    top_k_chunks = select_top_k(
        reranked, top_k=top_k, canonical_file=None
    )
    return raw, reranked, top_k_chunks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ArmMetrics:
    query: str
    gold_files: tuple[str, ...]
    top5_files: list[str] = field(default_factory=list)
    top5_section_types: list[str] = field(default_factory=list)
    n_gold_hit: int = 0
    n_gold_total: int = 0
    first_hit_rank: Optional[int] = None
    n_table_in_top5: int = 0
    n_correct_lit_in_top5: int = 0

    @property
    def hit_rate_at_5(self) -> float:
        return self.n_gold_hit / max(1, self.n_gold_total)

    @property
    def mrr(self) -> float:
        return (1.0 / self.first_hit_rank) if self.first_hit_rank else 0.0

    @property
    def table_summary_rate_at_5(self) -> float:
        return self.n_table_in_top5 / _TOP_K

    @property
    def correct_lit_rate_at_5(self) -> float:
        return self.n_correct_lit_in_top5 / _TOP_K


def _compute_metrics(query: Query, top_k_chunks: list, rerank_enabled: bool) -> ArmMetrics:
    m = ArmMetrics(query=query.name, gold_files=query.gold_files)
    gold_set = set(query.gold_files)
    m.n_gold_total = len(gold_set)
    seen_gold: set[str] = set()
    for rank, h in enumerate(top_k_chunks, start=1):
        fn = (h.node.metadata or {}).get("file_name", "?")
        sec = (h.node.metadata or {}).get("section_type", "?")
        m.top5_files.append(fn)
        m.top5_section_types.append(sec)
        if sec == "table_summary":
            m.n_table_in_top5 += 1
        if fn in gold_set:
            m.n_correct_lit_in_top5 += 1
            if fn not in seen_gold:
                m.n_gold_hit += 1
                seen_gold.add(fn)
                if m.first_hit_rank is None:
                    m.first_hit_rank = rank
    m.rerank_enabled = rerank_enabled  # type: ignore[attr-defined]
    m.query_text = [q.text for q in _QUERIES if q.name == query.name][0]  # type: ignore[attr-defined]
    return m


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_top5(m: ArmMetrics) -> str:
    parts = []
    for i, (fn, sec) in enumerate(zip(m.top5_files, m.top5_section_types), start=1):
        # Strip folder prefixes for readability
        name = Path(fn).stem[:40] if fn != "?" else "?"
        # Flag gold hits with a star
        flag = "★" if fn in m.gold_files else " "
        parts.append(f"  [{i}]{flag} {name:<42} ({sec})")
    return "\n".join(parts)


def _print_arm_block(label: str, metrics_list: list[ArmMetrics]) -> None:
    print(f"\n{'=' * 110}\n  {label}\n{'=' * 110}")
    print(
        f"{'Query':<28} {'HR@5':>5} {'MRR':>6} {'tbl@5':>6} {'correct-lit@5':>14} "
        f"{'hit-rank':>10}"
    )
    print("-" * 110)
    for m in metrics_list:
        hr = f"{m.hit_rate_at_5*100:5.1f}%"
        mrr = f"{m.mrr:6.3f}"
        tbl = f"{m.table_summary_rate_at_5*100:5.1f}%"
        cl = f"{m.correct_lit_rate_at_5*100:5.1f}%"
        hr_rank = str(m.first_hit_rank) if m.first_hit_rank else "miss"
        print(
            f"{m.query:<28} {hr:>5} {mrr:>6} {tbl:>6} {cl:>14} {hr_rank:>10}"
        )


def _print_query_detail(metrics_list: list[ArmMetrics]) -> None:
    for m in metrics_list:
        # Build summary label "OFF" or "ON"
        which = "ON " if getattr(m, "rerank_enabled", False) else "OFF"
        print(f"\n--- {m.query} · rerank {which} ---")
        print(f"  query text: {getattr(m, 'query_text', '')}")
        print(f"  gold files ({len(m.gold_files)}):")
        for g in m.gold_files:
            print(f"    - {g}")
        print(f"  top-5 chunks (★ = gold hit):")
        print(_fmt_top5(m))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    out_dir = _PROJECT_ROOT / "holdout" / "search_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    print("=" * 110)
    print("Rerank A/B search eval — 22-paper pool · 3 cross-paper queries · no canonical routing")
    print("=" * 110)

    arm_a: list[ArmMetrics] = []
    arm_b: list[ArmMetrics] = []
    raw_pools: dict[str, list] = {}
    reranked_pools: dict[str, dict[str, list]] = {}
    top_k_per_arm: dict[str, dict[str, list]] = {}

    for q in _QUERIES:
        print(f"\n>>> Query: {q.name}")
        print(f"    {q.text}")
        # Group A — rerank OFF
        raw_a, rer_a, top_a = _retrieve_topk(q.text, rerank_enabled=False)
        m_a = _compute_metrics(q, top_a, rerank_enabled=False)
        # Group B — rerank ON
        raw_b, rer_b, top_b = _retrieve_topk(q.text, rerank_enabled=True)
        m_b = _compute_metrics(q, top_b, rerank_enabled=True)

        arm_a.append(m_a)
        arm_b.append(m_b)
        raw_pools[q.name] = raw_a  # same raw pool either way (deterministic)
        reranked_pools[q.name] = {"off": list(rer_a), "on": list(rer_b)}
        top_k_per_arm[q.name] = {"off": list(top_a), "on": list(top_b)}

        _print_query_detail([m_a, m_b])

    _print_arm_block("Group A — rerank OFF", arm_a)
    _print_arm_block("Group B — rerank ON ", arm_b)

    # Aggregate deltas
    print("\n" + "=" * 110)
    print("Δ (ON − OFF) per metric")
    print("=" * 110)
    print(
        f"{'Query':<28} {'Δ HR@5':>9} {'Δ MRR':>9} {'Δ tbl@5':>9} "
        f"{'Δ correct-lit@5':>14}"
    )
    print("-" * 110)
    for ma, mb in zip(arm_a, arm_b):
        dhr = mb.hit_rate_at_5 - ma.hit_rate_at_5
        dmrr = mb.mrr - ma.mrr
        dtbl = mb.table_summary_rate_at_5 - ma.table_summary_rate_at_5
        dcl = mb.correct_lit_rate_at_5 - ma.correct_lit_rate_at_5
        print(
            f"{ma.query:<28} {dhr*100:+7.1f}pp {dmrr:+9.3f} {dtbl*100:+7.1f}pp "
            f"{dcl*100:+12.1f}pp"
        )

    # Persist
    payload = {
        "generated_at": generated_at,
        "db_path": str(_DB_PATH),
        "collection": _COLLECTION_NAME,
        "top_k": _TOP_K,
        "over_fetch": _OVER_FETCH,
        "mmr_threshold": _MMR_THRESHOLD,
        "queries": [
            {
                "name": q.name,
                "text": q.text,
                "gold_files": list(q.gold_files),
                "gold_design": q.gold_design,
                "expect_table": q.expect_table,
            }
            for q in _QUERIES
        ],
        "metrics": {
            "arm_a_off": [m.__dict__ for m in arm_a],
            "arm_b_on":  [m.__dict__ for m in arm_b],
        },
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n结果输出 → {out_dir}/_summary.json")
    return 0


__all__ = ["Query", "_QUERIES", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
