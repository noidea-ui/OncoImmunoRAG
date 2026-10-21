"""NCBI PubMed E-utilities client for OncoImmunoRAG.

Provides two thin wrappers over the public Entrez API:
    * ``search_pubmed(query, max_results=10, ...)`` → list[PubMedHit]
    * ``fetch_details(pmid, ...)`` → PubMedDetail (incl. abstract)

No API key required for low-volume use (<3 req/sec) but we always pass
``tool`` and ``email`` per NCBI's etiquette policy. The client uses
``requests`` directly and parses JSON (``retmode=json``) to avoid pulling
in extra XML libraries.

Example:
    >>> hits = search_pubmed("pembrolizumab bladder cancer", max_results=5)
    >>> for h in hits:
    ...     print(h.pmid, h.year, h.title[:80])
    >>> d = fetch_details(hits[0].pmid)
    >>> print(d.abstract[:200])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL = "OncoImmunoRAG"
_EMAIL = "research@oncoimmunorag.local"   # per their etiquette policy


@dataclass
class PubMedHit:
    """Lightweight record returned by ``esearch`` (title/author/year may be
    populated by the convenience fetch in :func:`search_pubmed`)."""
    pmid: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""

    def short_label(self) -> str:
        first_author = self.authors[0].split()[0] if self.authors else "?"
        return f"{first_author} {self.year} ({self.journal})"


@dataclass
class PubMedDetail:
    """Full record returned by ``efetch``."""
    pmid: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    doi: str = ""


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _get(url: str, params: dict[str, Any], timeout: int = 30) -> dict:
    """GET helper; raises RuntimeError on transport / parse failure."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"PubMed request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"PubMed HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"PubMed returned non-JSON body: {resp.text[:200]}"
        ) from exc


def _esearch(query: str, retmax: int, mindate: str, maxdate: str) -> list[str]:
    """Hit ``esearch.fcgi`` and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "sort": "relevance",
        "mindate": mindate,
        "maxdate": maxdate,
        "datetype": "pdat",
        "tool": _TOOL,
        "email": _EMAIL,
    }
    payload = _get(f"{_BASE}/esearch.fcgi", params)
    idlist = (((payload or {}).get("esearchresult") or {}).get("idlist") or [])
    return [str(x) for x in idlist]


def _esummary(pmids: list[str]) -> list[dict]:
    """Hit ``esummary.fcgi`` for human-readable fields.
    Note: ``esummary`` is preferred over ``efetch`` for header metadata because
    it returns well-structured JSON and parses ``pubdate`` cleanly."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "tool": _TOOL,
        "email": _EMAIL,
    }
    payload = _get(f"{_BASE}/esummary.fcgi", params)
    result = (payload or {}).get("result") or {}
    # The ``result`` dict mixes uids as keys with summary objects; skip the
    # special 'uids' list entry.
    out: list[dict] = []
    for pmid in pmids:
        rec = result.get(pmid)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _efetch(pmid: str) -> PubMedDetail:
    """Hit ``efetch.fcgi`` for the abstract text."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",  # Force JSON; PubMed returns JSON-with-abstracts
        "rettype": "abstract",
        "tool": _TOOL,
        "email": _EMAIL,
    }
    raw = _get(f"{_BASE}/efetch.fcgi", params)
    return _parse_efetch_json(raw, pmid)


def _parse_efetch_json(raw: dict, pmid: str) -> PubMedDetail:
    """Parse ``efetch``'s JSON-with-abstracts shape.

    PubMed's JSON output is split into ``PubmedArticle`` / ``PubmedBookArticle``
    lists; we accept either. Some records (e.g. proceedings) lack an abstract.
    """
    detail = PubMedDetail(pmid=pmid)
    records = (raw or {}).get("PubmedArticle") or (raw or {}).get("PubmedBookArticle") or []
    if not records:
        return detail
    rec = records[0]
    medline = rec.get("MedlineCitation") or {}
    art = rec.get("Article") or {}
    journal_info = art.get("Journal") or {}
    pubdate = journal_info.get("JournalIssue", {}).get("PubDate") or {}

    detail.title = (art.get("ArticleTitle") or "").strip()
    # Abstract — PubMed JSON splits it into multiple sections (label + text)
    abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract_parts, list):
        chunks = []
        for part in abstract_parts:
            if isinstance(part, dict):
                chunks.append(part.get("text", "") or "")
            else:
                chunks.append(str(part))
        detail.abstract = "\n\n".join(c for c in chunks if c).strip()
    elif isinstance(abstract_parts, str):
        detail.abstract = abstract_parts

    # Authors
    author_list = art.get("AuthorList", [])
    for a in author_list:
        last = a.get("LastName") or ""
        init = a.get("Initials") or ""
        if last:
            detail.authors.append(f"{last} {init}".strip())

    detail.journal = journal_info.get("Title") or ""
    # PubDate can be {"Year": "2024"} or {"MedlineDate": "2024 Jan-Feb"}
    detail.year = pubdate.get("Year") or (pubdate.get("MedlineDate") or "").split()[:1] or [""]

    # MeSH terms (for `term` MeSH-major filter, can be used by downstream
    # callers to detect RCT vs single-arm markers).
    mesh_list = medline.get("MeshHeadingList") or []
    for m in mesh_list:
        d = m.get("DescriptorName") or {}
        name = d.get("#text") or d.get("str") or ""
        if name:
            detail.mesh_terms.append(name)

    # Publication types
    pt_list = art.get("PublicationTypeList") or []
    for p in pt_list:
        v = p.get("#text") or p.get("str") or ""
        if v:
            detail.publication_types.append(v)

    # DOI
    for id_obj in art.get("ELocationID", []):
        if id_obj.get("EIdType") == "doi":
            detail.doi = id_obj.get("#text") or id_obj.get("str") or ""
            break

    return detail


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def search_pubmed(
    query: str,
    *,
    max_results: int = 10,
    mindate: str = "2021/01/01",
    maxdate: str = "2026/12/31",
    field_filter: str | None = "Title/Abstract",
    lang: str = "eng",
    fetch_summaries: bool = True,
) -> list[PubMedHit]:
    """Run ``esearch`` + (optionally) ``esummary`` and return a list of hits.

    Args:
        query: Free-text PubMed query, e.g. "pembrolizumab BCG unresponsive".
        max_results: Max PMIDs to return (capped at NCBI's per-request limit).
        mindate: Earliest date (``YYYY/MM/DD``); default ``2021/01/01``.
        maxdate: Latest date; default ``2026/12/31``.
        field_filter: When set (default ``Title/Abstract``), restricts the
            term-matching to those fields via ``[Title/Abstract]`` tag.
        lang: Restrict language; default ``eng``.
        fetch_summaries: When True (default), also run ``esummary`` to
            populate title/authors/year/journal in each hit.

    Raises:
        RuntimeError: on transport / parse failure (surfaced to UI as
            "PubMed 服务异常，请稍后重试").
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    term = query.strip()
    if field_filter:
        # ``[Title/Abstract]`` is a PubMed tag restricting the search fields.
        term = f"({term})[{field_filter}]"
    if lang:
        term = f"{term} AND ({lang}[Language])"

    pmids = _esearch(term, retmax=max_results, mindate=mindate, maxdate=maxdate)
    if not pmids:
        return []
    hits = [PubMedHit(pmid=p) for p in pmids]
    if fetch_summaries:
        # NCBI requests polite use: ≤3 req/sec without API key. We have at
        # most ~2 calls per ``search_pubmed`` invocation (esearch + esummary),
        # so a single 0.4-second sleep is sufficient.
        time.sleep(0.4)
        try:
            summaries = _esummary(pmids)
        except RuntimeError:
            return hits
        for hit, summary in zip(hits, summaries):
            hit.title = (summary.get("title") or "").strip()
            hit.journal = (summary.get("source") or summary.get("fulljournalname") or "").strip()
            # ``pubdate`` is often "2024 Jan" or "2024"
            pubdate = summary.get("pubdate") or summary.get("sortfirstauthor") or ""
            hit.year = pubdate.split()[:1] or [""]
            hit.authors = [
                a.get("name", "") for a in (summary.get("authors") or []) if a.get("name")
            ]
    return hits


def fetch_details(
    pmid: str,
    *,
    sleep_seconds: float = 0.4,
) -> PubMedDetail:
    """Run ``efetch`` for a single PMID and return a :class:`PubMedDetail`.

    The detail record includes the abstract text — that's what the GRADE
    evaluator needs. When the abstract is empty (proceedings, in-revision
    records, etc.), :attr:`PubMedDetail.abstract` is ``""`` and the UI
    should warn the user to upload the full PDF instead.
    """
    if not pmid or not str(pmid).strip():
        raise ValueError("pmid must be a non-empty string")
    pmid = str(pmid).strip()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return _efetch(pmid)


__all__ = [
    "PubMedHit",
    "PubMedDetail",
    "search_pubmed",
    "fetch_details",
]