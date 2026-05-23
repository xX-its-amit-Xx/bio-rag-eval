"""Citation traceability.

    citation_traceability = |{c in citations : resolvable(c)}| / |citations|

`resolvable` means we can confirm the source exists at one of:
  1. DOI -> doi.org HEAD request returns 2xx/3xx
  2. PMID -> NCBI E-utilities esummary returns a record
  3. PMCID -> PubMed Central esummary returns a record
  4. NCT ID -> clinicaltrials.gov v2 API returns a record
  5. URL -> HEAD request returns 2xx/3xx

We do NOT validate that the snippet matches the source — grounding does
that. Traceability is purely "does this citation point at a real,
retrievable document?".

The resolver supports an offline mode (`offline=True`) — instead of
hitting the network, we accept any citation whose ID matches the
expected regex shape. Useful for CI and demos. When offline, the metric
key is renamed `citation_traceability_offline_check` so it's never
confused with a real resolution rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bio_rag_eval.schemas.case import Citation, GoldStandardCase
from bio_rag_eval.utils import safe_div

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d{1,9}$", re.IGNORECASE)
_NCT_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class CitationResolution:
    """Detail per citation — surfaced in the HTML report."""

    citation_id: str
    resolved: bool
    method: str | None  # "doi" | "pmid" | "pmcid" | "nct" | "url" | None
    url_checked: str | None
    error: str | None = None


def resolve_citation(
    citation: Citation,
    offline: bool = False,
    http_client: Any | None = None,
) -> CitationResolution:
    """Try resolution methods in order: DOI -> PMID -> PMCID -> NCT -> URL.

    Returns the first that succeeds. If all fail (or all are absent),
    returns resolved=False.
    """
    # DOI
    if citation.doi:
        url = f"https://doi.org/{citation.doi}"
        ok, err = _check(url, citation.doi, _DOI_RE, offline, http_client)
        if ok:
            return CitationResolution(citation.citation_id, True, "doi", url)
        if offline:
            # Offline + shape mismatch -> definitive False on this method, try next.
            pass
    # PMID
    if citation.pmid:
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={citation.pmid}&retmode=json"
        )
        ok, err = _check(url, citation.pmid, _PMID_RE, offline, http_client)
        if ok:
            return CitationResolution(citation.citation_id, True, "pmid", url)
    # PMCID
    if citation.pmcid:
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pmc&id={citation.pmcid.upper().removeprefix('PMC')}&retmode=json"
        )
        ok, err = _check(url, citation.pmcid, _PMCID_RE, offline, http_client)
        if ok:
            return CitationResolution(citation.citation_id, True, "pmcid", url)
    # NCT
    if citation.nct_id:
        url = f"https://clinicaltrials.gov/api/v2/studies/{citation.nct_id.upper()}"
        ok, err = _check(url, citation.nct_id, _NCT_RE, offline, http_client)
        if ok:
            return CitationResolution(citation.citation_id, True, "nct", url)
    # Bare URL
    if citation.url:
        ok, err = _check(citation.url, citation.url, _URL_RE, offline, http_client)
        if ok:
            return CitationResolution(citation.citation_id, True, "url", citation.url)

    return CitationResolution(
        citation_id=citation.citation_id,
        resolved=False,
        method=None,
        url_checked=None,
        error="no identifier or all resolution methods failed",
    )


def _check(
    url: str,
    raw_id: str,
    shape: re.Pattern[str],
    offline: bool,
    http_client: Any | None,
) -> tuple[bool, str | None]:
    if offline:
        return (bool(shape.match(raw_id)), None if shape.match(raw_id) else "shape mismatch")
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return (False, "httpx not installed")
    client = http_client or httpx.Client(follow_redirects=True, timeout=8.0)
    try:
        # Use GET — some endpoints return 405 on HEAD even when GET works.
        resp = client.get(url)
        ok = 200 <= resp.status_code < 400
        return (ok, None if ok else f"status {resp.status_code}")
    except Exception as e:
        return (False, str(e))
    finally:
        if http_client is None:
            try:
                client.close()
            except Exception:
                pass


def compute_citation_traceability(
    citations: list[Citation],
    case: GoldStandardCase | None = None,
    offline: bool = False,
    http_client: Any | None = None,
) -> tuple[dict[str, float], list[CitationResolution]]:
    """Per-case citation metric. Returns (metric_dict, per_citation_detail).

    Metric keys:
      - citation_traceability (or citation_traceability_offline_check when offline)
      - required_citation_recall: |required substrs found in any citation| / |required|
      - n_citations
    """
    resolutions = [resolve_citation(c, offline=offline, http_client=http_client) for c in citations]
    resolved_n = sum(1 for r in resolutions if r.resolved)
    metric_key = "citation_traceability_offline_check" if offline else "citation_traceability"
    metrics: dict[str, float] = {
        metric_key: safe_div(resolved_n, len(citations)),
        "n_citations": float(len(citations)),
    }
    if case and case.required_citations:
        haystack = "\n".join(
            f"{c.title or ''} {c.doi or ''} {c.pmid or ''} {c.nct_id or ''} {c.snippet or ''} {c.url or ''}"
            for c in citations
        ).lower()
        hits = sum(1 for req in case.required_citations if req.lower() in haystack)
        metrics["required_citation_recall"] = safe_div(hits, len(case.required_citations))
    return metrics, resolutions
