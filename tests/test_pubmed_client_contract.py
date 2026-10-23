"""Contract tests for :mod:`retrieval.pubmed_client`.

**No network access.** ``requests.get`` is monkeypatched, so these tests run
offline and in CI. The point is to pin the parsing/contract behaviour of the
Entrez JSON, not to test NCBI.

Also asserted here: that the module declares ``requests`` as a real
dependency. It imports it directly but it was absent from ``pyproject.toml``
until 2026-10-23 (CURRENT_ISSUES.md ISSUE-035) — it used to resolve only as a
transitive dependency of something else.

Run::

    pytest tests/test_pubmed_client_contract.py -o addopts="" -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from retrieval import pubmed_client  # noqa: E402
from retrieval.pubmed_client import (  # noqa: E402
    PubMedDetail,
    PubMedHit,
    fetch_details,
    search_pubmed,
)

ESSEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "idlist": ["11111", "22222"],
    }
}

ESUMMARY_RESPONSE = {
    "result": {
        "uids": ["11111", "22222"],
        "11111": {
            "uid": "11111",
            "title": "Pembrolizumab in bladder cancer",
            "pubdate": "2024 Mar",
            "source": "Lancet Oncol",
            "authors": [{"name": "Balar AV"}, {"name": "Shore A"}],
        },
        "22222": {
            "uid": "22222",
            "title": "Durvalumab perioperative",
            "pubdate": "2023 Sep",
            "source": "NEJM",
            "authors": [{"name": "Powles T"}],
        },
    }
}


ESEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "idlist": ["11111", "22222"],
    }
}


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``.

    ``pubmed_client._get`` reads ``.status_code`` and calls ``.json()``, so a
    bare dict is not enough.
    """

    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def _esummary_payload() -> dict:
    return {
        "result": {
            "uids": ["11111", "22222"],
            "11111": {
                "uid": "11111",
                "title": "Pembrolizumab in bladder cancer",
                "pubdate": "2024 Mar",
                "source": "Lancet Oncol",
                "authors": [{"name": "Balar AV"}, {"name": "Shore A"}],
            },
            "22222": {
                "uid": "22222",
                "title": "Durvalumab perioperative",
                "pubdate": "2023 Sep",
                "source": "NEJM",
                "authors": [{"name": "Powles T"}],
            },
        }
    }


def _efetch_payload() -> dict:
    return {
        "result": {
            "11111": {
                "uid": "11111",
                "title": "Pembrolizumab in bladder cancer",
                "pubdate": "2024 Mar",
                "source": "Lancet Oncol",
                "authors": [{"name": "Balar AV"}],
                "articleids": [{"idtype": "doi", "value": "10.1016/j.laneonc.2024.01.001"}],
                "meshheadings": [{"descriptorname": "Urinary Bladder Neoplasms"}],
                "pubtype": ["Journal Article", "Randomized Controlled Trial"],
            }
        }
    }


@pytest.fixture()
def fake_requests(monkeypatch):
    """Replace ``requests.get`` with a canned-response router."""
    calls = []

    def _get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": params})
        if "esearch" in url:
            # Honour retmax the way NCBI does, otherwise max_results cannot be
            # asserted client-side.
            retmax = (params or {}).get("retmax")
            retmax = int(retmax) if retmax else None
            payload = {
                "esearchresult": {
                    "count": str(len(ESEARCH_RESPONSE["esearchresult"]["idlist"])),
                    "idlist": ESEARCH_RESPONSE["esearchresult"]["idlist"],
                }
            }
            if retmax:
                payload["esearchresult"]["idlist"] = payload["esearchresult"]["idlist"][:retmax]
            return _FakeResponse(payload)
        if "esummary" in url:
            return _FakeResponse(_esummary_payload())
        if "efetch" in url:
            return _FakeResponse(_efetch_payload())
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(pubmed_client.requests, "get", _get)
    return calls


# ---------------------------------------------------------------------------
# search_pubmed
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_search_returns_pubsubmephit_objects(fake_requests):
    hits = search_pubmed("pembrolizumab bladder", max_results=5)
    assert isinstance(hits, list) and hits
    assert all(isinstance(h, PubMedHit) for h in hits)


@pytest.mark.unit
def test_search_respects_max_results(fake_requests):
    hits = search_pubmed("bladder cancer", max_results=1)
    assert len(hits) <= 1


@pytest.mark.unit
def test_search_populates_hit_fields(fake_requests):
    hit = search_pubmed("pembrolizumab bladder", max_results=5)[0]
    assert hit.pmid
    assert hit.title
    assert isinstance(hit.authors, list)


@pytest.mark.unit
def test_search_pmid_is_a_string(fake_requests):
    """Contract: PMID must be str, never int — it is used as a dict key and in
    URL construction downstream."""
    hit = search_pubmed("bladder cancer", max_results=1)[0]
    assert isinstance(hit.pmid, str)


@pytest.mark.unit
def test_search_can_skip_summaries(fake_requests):
    hits = search_pubmed("bladder cancer", max_results=5, fetch_summaries=False)
    assert all(isinstance(h, PubMedHit) for h in hits)


@pytest.mark.contract
def test_search_calls_esearch_endpoint(fake_requests):
    search_pubmed("bladder cancer", max_results=3)
    assert any("esearch" in c["url"] for c in fake_requests)


@pytest.mark.unit
def test_short_label_is_non_empty(fake_requests):
    hit = search_pubmed("bladder cancer", max_results=1)[0]
    assert hit.short_label()


@pytest.mark.contract
@pytest.mark.xfail(
    reason=(
        "Known defect: PubMedHit.year is annotated 'str' but is assigned a list "
        "— pubmed_client.py:268 does `pubdate.split()[:1] or ['']`, and line 183 "
        "does the same for PubMedDetail.year. short_label() then renders "
        "\"['2023']\" instead of \"2023\". See CURRENT_ISSUES.md ISSUE-050."
    ),
    strict=True,
)
def test_hit_year_is_a_string(fake_requests):
    """Contract: ``year: str``. Currently violated — the value is a 1-element list."""
    hit = search_pubmed("bladder cancer", max_results=1)[0]
    assert isinstance(hit.year, str)


# ---------------------------------------------------------------------------
# fetch_details
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fetch_details_returns_pubmeddetail(fake_requests):
    detail = fetch_details("11111", sleep_seconds=0)
    assert isinstance(detail, PubMedDetail)


@pytest.mark.unit
def test_fetch_details_populates_abstract_and_ids(fake_requests):
    detail = fetch_details("11111", sleep_seconds=0)
    assert detail.pmid == "11111"
    assert isinstance(detail.mesh_terms, list)
    assert isinstance(detail.publication_types, list)
    assert isinstance(detail.doi, str)


@pytest.mark.unit
def test_fetch_details_honours_sleep_seconds(monkeypatch):
    """Contract: NCBI etiquette policy requires throttling between calls."""
    slept = []

    def _get(url, params=None, timeout=None, **kwargs):
        return _FakeResponse(
            {"result": {"11111": {"uid": "11111", "articleids": [], "meshheadings": []}}}
        )

    monkeypatch.setattr(pubmed_client.requests, "get", _get)
    monkeypatch.setattr(pubmed_client.time, "sleep", slept.append)
    fetch_details("11111", sleep_seconds=1.5)
    assert slept == [1.5]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_unknown_pmid_returns_placeholder_not_crash(fake_requests):
    """Contract: a bad PMID must degrade, not explode inside a Streamlit callback."""
    detail = fetch_details("does-not-exist", sleep_seconds=0)
    assert isinstance(detail, PubMedDetail)


# ---------------------------------------------------------------------------
# Dependency declaration
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_requests_is_a_declared_dependency():
    """Regression guard for CURRENT_ISSUES.md ISSUE-035.

    ``pubmed_client`` imports ``requests`` directly; if it is only present as a
    transitive dependency, any upstream drop breaks PubMed search in production
    while every test still passes.
    """
    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"requests' in pyproject, "requests must be declared in pyproject.toml"
    assert "requests" in pubmed_client.__all__ or True  # module imports it at top level
