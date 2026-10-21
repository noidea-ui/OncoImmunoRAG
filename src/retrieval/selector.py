"""Pure-Python top-K selection over the over-fetched retriever hits.

The class :class:`retrieval.query_engine.QueryEngine` carries a lot of
import-time baggage (``chromadb``, ``openai``, ``llama_index``). For unit
tests — and for any future caller that wants to plug in a different
retriever — the bucket-and-fill selection logic lives here as a plain
function with no third-party dependencies.

Retrieval flow:  Embedding → ChromaDB → MMR (over-fetch)
                → :func:`rerank_hits`  (evidence-grade weighted sort, optional)
                → :func:`select_top_k`  (bucket-and-fill policy)
                → Top-K returned to the grader.

Bucket order is:

1. ``table_summary`` chunks — force-promoted up to N slots, where N is 1
   for ``top_k <= _TABLE_SUMMARY_KEEP_BOUNDARY`` and 2 for larger top-K.
2. **Canonical routing** (when ``canonical_file`` is supplied) — at
   least one Primary chunk of the *target* paper is force-promoted to the
   very front so a high-frequency co-occurring paper (e.g. EV-302)
   cannot monpolise top-K on a McGregor-DAD question.
3. ``primary`` chunks — the paper's own Abstract / Results / Conclusions.
4. ``unknown`` chunks — default fallback.
5. ``secondary`` chunks — Background / Discussion / References.

Per-file caps prevent one paper from monopolising the top-K:

* Prose buckets honour ``per_file_cap = max(1, (top_k + 1) // 2)``.
* Table buckets honour a tighter cap of 1 so two tables from the same
  paper can't both consume the reserved slots.

The function never raises on an empty pool — it returns ``[]`` and lets
the caller decide what to do.

Evidence-grade reranking layer
-------------------------------
:func:`rerank_hits` is an optional, **pure-additive** sort layer inserted
between MMR and :func:`select_top_k`. It computes a per-hit score from
three factor groups and re-orders the over-fetch pool. Each hit's
``_rerank_score`` is stamped into its metadata for downstream visibility.

Factors (sum range ≈ 0.0 – 0.85):

* **Section type** — primary +0.30, table_summary +0.25,
  secondary +0.10, unknown +0.00
* **Evidence grade (from :mod:`utils.trial_registry`)** —
  RCT-family (+0.20) / single-arm (+0.05) / unknown file (+0.00)
* **Table slice keywords** — chunk text contains
  ``PFS`` / ``OS`` / ``HR=`` / ``95% CI`` → +0.25

The whole layer is disabled by passing ``enabled=False``; downstream
behaviour is then identical to the pre-rerank version.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

# Default retrieval depth. The task brief asks for top-5.
_DEFAULT_TOP_K = 5

# Table-summary slot policy. Mirrored in ``retrieval.query_engine`` so the
# public ``QueryEngine`` picks up the same numbers.
_TABLE_SUMMARY_KEEP_SMALL = 1   # top_k <= _TABLE_SUMMARY_KEEP_BOUNDARY
_TABLE_SUMMARY_KEEP_LARGE = 2   # top_k >  _TABLE_SUMMARY_KEEP_BOUNDARY
_TABLE_SUMMARY_KEEP_BOUNDARY = 7

# Canonical-routing reserve when a target PDF is explicitly named. We
# always pull at least one Primary chunk of the canonical file when
# available; up to 2 on larger top-K. If no Primary chunk exists in the
# over-fetched pool we fall back to 1 of any section_type for the
# canonical file (better than nothing — solves the "0 canonical chunks
# reach top-K" failure mode observed for McGregor-DAD).
_CANONICAL_PRIMARY_RESERVE_SMALL = 1
_CANONICAL_PRIMARY_RESERVE_LARGE = 2
_CANONICAL_PRIMARY_RESERVE_BOUNDARY = 5

# ---- evidence-grade rerank weights (2026-09 spec) -----------------------
# Section-type weight — primary prose is the strongest evidence. Mirror
# the bucketing in select_top_k so the rerank agrees with the eventual
# fill order.
_RERANK_SECTION_WEIGHT: dict[str, float] = {
    "primary": 0.30,
    "table_summary": 0.25,
    "secondary": 0.10,
    "unknown": 0.00,
}

# Evidence-grade boost, attached to the file_name via trial_registry. The
# +0.20 / +0.05 numbers come from the spec — RCT systematic-bias risk is
# lower than single-arm; both still beat "no registry entry" (default 0.0).
_RERANK_GRADE_BOOST: dict[str, float] = {
    "rct": 0.20,        # RCT / 系统评价 / Meta分析
    "single_arm": 0.05, # 单臂试验 / 观察性
}

# Table-slice keywords (case-insensitive). When ANY appear in the chunk
# text, the table_slice bonus fires. These four markers collectively
# cover most primary-endpoint reporting styles in oncology RCTs.
_TABLE_SLICE_KEYWORDS: tuple[str, ...] = (
    "PFS",
    "OS",
    "HR=",
    "95% CI",
    "95%CI",
)

# Maximum sum across all factors (used for telemetry only, not for
# capping the score — the sum is open-ended by design).
_RERANK_MAX_OBSERVED = sum([
    max(_RERANK_SECTION_WEIGHT.values()),
    max(_RERANK_GRADE_BOOST.values()),
    0.25,  # table-slice constant
])


def _section_type(hit: object) -> str:
    """Return ``section_type`` metadata, defaulting to ``"unknown"``."""
    meta = getattr(getattr(hit, "node", hit), "metadata", None) or {}
    return str(meta.get("section_type", "unknown"))


def _file_name(hit: object) -> str:
    """Return ``file_name`` metadata, defaulting to ``"<unknown>"``."""
    meta = getattr(getattr(hit, "node", hit), "metadata", None) or {}
    return str(meta.get("file_name", "<unknown>"))


def _take_with_cap(
    bucket: Sequence[object],
    slots: int,
    per_file_cap: int,
) -> Tuple[List[object], List[object]]:
    """Take up to ``slots`` items from ``bucket`` honouring per-file cap.

    Returns ``(taken, skipped)``. ``skipped`` contains items skipped
    because their file was already over ``per_file_cap`` — they remain
    eligible for back-fill.
    """
    taken: List[object] = []
    skipped: List[object] = []
    counts: dict[str, int] = {}
    for h in bucket:
        fn = _file_name(h)
        if counts.get(fn, 0) >= per_file_cap:
            skipped.append(h)
            continue
        taken.append(h)
        counts[fn] = counts.get(fn, 0) + 1
        if len(taken) >= slots:
            break
    return taken, skipped


def _canonical_reserve(top_k: int) -> int:
    """Number of canonical-paper slots to force-promote based on top_k."""
    if top_k > _CANONICAL_PRIMARY_RESERVE_BOUNDARY:
        return _CANONICAL_PRIMARY_RESERVE_LARGE
    return _CANONICAL_PRIMARY_RESERVE_SMALL


# ---- evidence-grade rerank helpers ----------------------------------------


def _trial_design_for(
    file_name: Optional[str],
    registry_lookup,
) -> Optional[str]:
    """Return ``"rct"`` / ``"single_arm"`` / ``None`` for ``file_name``.

    Looks up the trial registry via the supplied ``registry_lookup``
    callable (kept injectable to avoid a hard import of
    :mod:`utils.trial_registry` here — ``selector.py`` must stay a
    pure-logic module with no chromadb / openai / utils dependency). The
    QueryEngine plugs in ``utils.trial_registry.lookup``; tests can pass
    a dict-based mock.

    File-not-found returns ``None`` → contributes 0.00 to the rerank
    score (i.e. neutral), which mirrors the "unknown design" baseline.
    """
    if not file_name or file_name == "<unknown>" or registry_lookup is None:
        return None
    try:
        info = registry_lookup(file_name)
    except Exception:
        return None
    if info is None:
        return None
    design = getattr(info, "study_design", None)
    if design in ("RCT", "系统评价", "Meta分析"):
        return "rct"
    if design in ("单臂试验", "观察性"):
        return "single_arm"
    return None


def _has_table_slice(text: str) -> bool:
    """Return True when chunk text carries PFS / OS / HR= / 95% CI markers."""
    if not text:
        return False
    upper = text.upper()
    return any(kw.upper() in upper for kw in _TABLE_SLICE_KEYWORDS)


def _chunk_text(hit: object) -> str:
    """Best-effort chunk text accessor — works for NodeWithScore and bare Node."""
    node = getattr(hit, "node", hit)
    getter = getattr(node, "get_content", None)
    if callable(getter):
        return (getter() or "")
    return str(node) if node else ""


def _rerank_score(hit: object, *, registry_lookup=None) -> float:
    """Compute the additive rerank score (0.0 – ~0.85) for ``hit``.

    Three independent factor groups, summed:

    1. **Section type** (≤ +0.30) — primary beats table_summary, secondary,
       unknown.
    2. **Evidence grade** (≤ +0.20) — registry-known RCT (+0.20) /
       single-arm (+0.05); unknown file ⇒ 0.0.
    3. **Table slice keywords** (+0.25) — chunk text contains
       PFS/OS/HR/95% CI.
    """
    score = 0.0

    score += _RERANK_SECTION_WEIGHT.get(_section_type(hit), 0.0)

    design = _trial_design_for(_file_name(hit), registry_lookup)
    if design in _RERANK_GRADE_BOOST:
        score += _RERANK_GRADE_BOOST[design]

    if _has_table_slice(_chunk_text(hit)):
        score += 0.25

    return round(score, 4)


def rerank_hits(
    hits: Iterable[object],
    *,
    enabled: bool = True,
    registry_lookup=None,
) -> List[object]:
    """Re-order ``hits`` by the additive rerank score.

    Inserted in the retrieval flow between MMR and
    :func:`select_top_k`. Behaviour:

    * **enabled=True** (default) — compute ``_rerank_score`` per hit and
      sort descending; stable tie-break by original position.
    * **enabled=False** — return the input list unchanged (no sort, no
      metadata writes). Equivalent to "rerank layer off".

    The score is stamped onto each hit's ``node.metadata`` dict under the
    key ``_rerank_score`` so the grader / debug consumers can read it.
    Hits whose ``node.metadata`` is not a writable ``dict`` are still
    included in the sort using a side-channel ``dict``.

    Args:
        hits: Iterable of NodeWithScore (or compatible ``.node.metadata``
            shims) — typically the MMR-over-fetched pool.
        enabled: When False, the function is a no-op pass-through.
        registry_lookup: Zero-arg callable taking a ``file_name`` and
            returning a :class:`TrialInfo` (or ``None``). When ``None``,
            grade-factor defaults to 0.0 for every chunk.

    Returns:
        Sorted list (or input list copy if disabled).
    """
    raw = list(hits)
    if not raw:
        return []

    if not enabled:
        # Lightly stamp a None to make absence-of-rerank visible in JSON.
        for h in raw:
            meta = getattr(getattr(h, "node", h), "metadata", None)
            if isinstance(meta, dict):
                meta.setdefault("_rerank_score", None)
        return raw

    side: dict[int, float] = {}
    for h in raw:
        s = _rerank_score(h, registry_lookup=registry_lookup)
        meta = getattr(getattr(h, "node", h), "metadata", None)
        if isinstance(meta, dict):
            meta["_rerank_score"] = s
        else:
            side[id(h)] = s

    # Stable sort: descending by score, then keep original order on tie.
    indexed = list(enumerate(raw))

    def _key(item):
        idx, h = item
        meta = getattr(getattr(h, "node", h), "metadata", None)
        if isinstance(meta, dict):
            score = meta.get("_rerank_score", 0.0)
        else:
            score = side.get(id(h), 0.0)
        return (-score, idx)

    return [h for _, h in sorted(indexed, key=_key)]


# Module-level switch so callers / tests can probe without threading
# arguments. Default ON (per spec).
_RERANK_ENABLED_DEFAULT: bool = True


def get_rerank_enabled_default() -> bool:
    """Return the module-level default for the rerank-on/off switch."""
    return _RERANK_ENABLED_DEFAULT


def set_rerank_enabled_default(enabled: bool) -> None:
    """Set the module-level default for the rerank-on/off switch."""
    global _RERANK_ENABLED_DEFAULT
    _RERANK_ENABLED_DEFAULT = bool(enabled)


def select_top_k(
    hits: Iterable[object],
    top_k: int = _DEFAULT_TOP_K,
    *,
    canonical_file: Optional[str] = None,
) -> List[object]:
    """Pick ``top_k`` hits from ``hits`` honouring the bucket policy.

    Args:
        hits: Over-fetched retriever results (any iterable of objects
            exposing ``.node.metadata``).
        top_k: Final size of the selected list.
        canonical_file: When supplied, *at least* one chunk of this file
            is force-promoted into top-K — closing the
            "cross-paper retrieval pollution" bug observed when a query
            whose true target is a niche trial (McGregor-DAD, ABACUS,
            …) retrieves the dominant paper (EV-302) instead.

            Implementation:
            * Try to pull Primary chunks of ``canonical_file`` first
              (capped per-file, then total).
            * Fall back to any-section chunks of ``canonical_file`` when
              no Primary was retrievable.

            The remaining slots then fill from the normal
            table → primary → unknown → secondary bucket order.
    """
    if top_k <= 0:
        return []
    raw = list(hits)
    if not raw:
        return []

    canonical_reserve = _canonical_reserve(top_k)

    # ---- bucket by section_type -------------------------------------------
    table_summary = [h for h in raw if _section_type(h) == "table_summary"]
    primary = [h for h in raw if _section_type(h) == "primary"]
    unknown = [h for h in raw if _section_type(h) == "unknown"]
    secondary = [h for h in raw if _section_type(h) == "secondary"]

    per_file_cap = max(1, (top_k + 1) // 2)

    # ---- canonical routing (force-promote target paper) ------------------
    forced: List[object] = []
    seen_ids: set[int] = set()
    canonical_hit: bool = False  # telemetry for tests
    if canonical_file:
        # Prefer Primary chunks of the canonical file.
        canonical_primary = [
            h for h in raw
            if _file_name(h) == canonical_file and _section_type(h) == "primary"
        ]
        take_p, _ = _take_with_cap(
            canonical_primary, canonical_reserve, per_file_cap
        )
        forced.extend(take_p)
        seen_ids.update(id(h) for h in forced)
        # If Primary was unreachable (common when MMR pushes the
        # canonical paper out of the over-fetch pool), fall back to
        # other sections so the prompt still has *some* signal from
        # the target paper.
        if not forced:
            canonical_any = [
                h for h in raw if _file_name(h) == canonical_file
            ]
            take_a, _ = _take_with_cap(canonical_any, 1, per_file_cap)
            forced.extend(take_a)
            seen_ids.update(id(h) for h in forced)
        canonical_hit = bool(forced)

    selected: List[object] = list(forced)
    remaining = top_k - len(selected)
    if remaining <= 0:
        return selected[:top_k]

    # Reserve table_summary slots out of remaining (note: if canonical
    # already took a slot, the table reserve shrinks accordingly).
    if remaining > _TABLE_SUMMARY_KEEP_BOUNDARY:
        reserved_table = min(_TABLE_SUMMARY_KEEP_LARGE, len(table_summary), remaining)
    else:
        reserved_table = min(_TABLE_SUMMARY_KEEP_SMALL, len(table_summary), remaining)

    table_per_file_cap = 1 if reserved_table > 1 else per_file_cap

    if reserved_table:
        table_taken, _ = _take_with_cap(
            [h for h in table_summary if id(h) not in seen_ids],
            reserved_table,
            table_per_file_cap,
        )
        for h in table_taken:
            if id(h) in seen_ids:
                continue
            selected.append(h)
            seen_ids.add(id(h))
            if len(selected) >= top_k:
                break
        remaining = top_k - len(selected)
        if remaining <= 0:
            return selected[:top_k]

    # Primary bucket (excluding the ones already forced in).
    primary_pool = [h for h in primary if id(h) not in seen_ids]
    primary_taken, primary_skipped = _take_with_cap(
        primary_pool, remaining, per_file_cap
    )
    for h in primary_taken:
        if id(h) in seen_ids:
            continue
        selected.append(h)
        seen_ids.add(id(h))
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        slots = top_k - len(selected)
        unk_pool = [h for h in unknown if id(h) not in seen_ids]
        unk_taken, _ = _take_with_cap(unk_pool, slots, per_file_cap)
        for h in unk_taken:
            if id(h) in seen_ids:
                continue
            selected.append(h)
            seen_ids.add(id(h))
            if len(selected) >= top_k:
                break

    if len(selected) < top_k:
        slots = top_k - len(selected)
        sec_pool = [h for h in secondary if id(h) not in seen_ids]
        sec_taken, _ = _take_with_cap(sec_pool, slots, per_file_cap)
        for h in sec_taken:
            if id(h) in seen_ids:
                continue
            selected.append(h)
            seen_ids.add(id(h))
            if len(selected) >= top_k:
                break

    # Back-fill: anything still missing, pull from primary_skipped first,
    # then the other over-fetched candidates. Already-selected chunks
    # (including reserved tables) are filtered via ``seen_ids``.
    if len(selected) < top_k:
        for pool in (
            primary_skipped,
            unknown,
            secondary,
            table_summary,
            raw,
        ):
            for h in pool:
                if id(h) in seen_ids:
                    continue
                selected.append(h)
                seen_ids.add(id(h))
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

    # Telemetry side-channel — the QueryEngine / grader can read this to
    # surface "did canonical routing kick in?" in the saved JSON.
    for h in selected:
        meta = getattr(getattr(h, "node", h), "metadata", None) or {}
        meta.setdefault("_canonical_routed", canonical_hit)
        meta.setdefault("_canonical_target", canonical_file or "")

    return selected[:top_k]


__all__ = [
    "select_top_k",
    "_DEFAULT_TOP_K",
    "_TABLE_SUMMARY_KEEP_SMALL",
    "_TABLE_SUMMARY_KEEP_LARGE",
    "_TABLE_SUMMARY_KEEP_BOUNDARY",
    "_CANONICAL_PRIMARY_RESERVE_SMALL",
    "_CANONICAL_PRIMARY_RESERVE_LARGE",
]
