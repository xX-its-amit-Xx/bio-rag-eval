from __future__ import annotations

from bio_rag_eval.metrics.citation import (
    compute_citation_traceability,
    resolve_citation,
)
from bio_rag_eval.schemas.case import Citation


def test_resolve_doi_offline_shape_match():
    c = Citation(citation_id="x", doi="10.1056/NEJMoa2003800")
    r = resolve_citation(c, offline=True)
    assert r.resolved is True
    assert r.method == "doi"


def test_resolve_pmid_offline_shape_match():
    c = Citation(citation_id="x", pmid="32877582")
    r = resolve_citation(c, offline=True)
    assert r.resolved is True
    assert r.method == "pmid"


def test_resolve_nct_offline_shape_match():
    c = Citation(citation_id="x", nct_id="NCT04856982")
    r = resolve_citation(c, offline=True)
    assert r.resolved is True
    assert r.method == "nct"


def test_resolve_unresolvable_offline():
    c = Citation(citation_id="x", title="just a title, no id")
    r = resolve_citation(c, offline=True)
    assert r.resolved is False
    assert r.method is None


def test_required_citation_recall(simple_case):
    cites = [
        Citation(citation_id="s1", title="Seminal2020 paper on GENE1", pmid="11111111"),
        Citation(citation_id="s2", doi="10.1056/whatever"),
    ]
    metrics, _ = compute_citation_traceability(cites, case=simple_case, offline=True)
    # case.required_citations = ["seminal2020"] — case-insensitive substring match
    assert metrics["required_citation_recall"] == 1.0


def test_citation_traceability_metric_key_renamed_offline():
    cites = [Citation(citation_id="s1", pmid="11111111")]
    metrics, _ = compute_citation_traceability(cites, offline=True)
    assert "citation_traceability_offline_check" in metrics
    assert "citation_traceability" not in metrics
