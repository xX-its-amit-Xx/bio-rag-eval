"""Tests for TherapyAgentAdapter (therapy_agent_v1)."""
from __future__ import annotations

import pytest

from bio_rag_eval.adapters import get_predictions_adapter, list_adapters
from bio_rag_eval.adapters.therapy_agent import TherapyAgentAdapter
from bio_rag_eval.schemas.case import ModulationDirection


def _record(**overrides) -> dict:
    base = {
        "case_id": "sod1_als",
        "schema_version": "1.0",
        "gene": "SOD1",
        "mutation": "p.Ala4Val",
        "disease_phenotype": "familial ALS",
        "target_protein": {"name": "SOD1", "gene_symbol": "SOD1"},
        "target_pathway": None,
        "modulation_type": "ASO",
        "supporting_evidence": [
            {"claim": "Tofersen lowered CSF NfL by 60% in VALOR.",
             "doi": "10.1056/NEJMoa2204705", "source_url": "https://nejm.org/x"},
            {"claim": "SOD1 mutations cause toxic gain of function in motor neurons.",
             "doi": None, "source_url": None},
        ],
        "precedent_drugs": [{"name": "tofersen", "approved": True, "year": 2023}],
        "confidence_score": 0.82,
        "reasoning_trace": [
            {"node": "pipeline", "content": "Detected a dominant SOD1 GoF variant; recommend silencing."}
        ],
        "model_used": "claude-opus-4-7",
        "input_tokens": 1200,
        "output_tokens": 350,
        "wall_clock_seconds": 4.5,
        "timestamp": "2026-05-23T12:00:00Z",
    }
    base.update(overrides)
    return base


def test_adapter_registered():
    assert "therapy_agent_v1" in list_adapters()["predictions"]


def test_extract_citations_assigns_stable_ids():
    a = get_predictions_adapter("therapy_agent_v1")
    cites = a.extract_citations(_record())
    assert [c.citation_id for c in cites] == ["e1", "e2"]
    assert cites[0].doi == "10.1056/NEJMoa2204705"
    assert cites[0].url == "https://nejm.org/x"
    assert cites[0].snippet and "60%" in cites[0].snippet


def test_extract_citations_lifts_inline_doi():
    rec = _record(supporting_evidence=[
        {"claim": "See review at 10.1038/s41586-020-2649-2 for details.", "doi": None, "source_url": None},
    ])
    a = TherapyAgentAdapter()
    cites = a.extract_citations(rec)
    assert cites[0].doi == "10.1038/s41586-020-2649-2"


def test_extract_claims_attributes_each_to_its_source():
    a = TherapyAgentAdapter()
    claims = a.extract_claims(_record())
    # Two evidence entries -> at least two claims; plus a synthesized
    # recommendation claim that cites the union.
    assert len(claims) == 3
    assert claims[0].cited_ids == ["e1"]
    assert claims[1].cited_ids == ["e2"]
    assert claims[2].cited_ids == ["e1", "e2"]
    # Quantitative tag fires on the "60%" string
    assert claims[0].claim_type.value == "quantitative"


def test_extract_task_answer_normalizes_modulation():
    a = TherapyAgentAdapter()
    ta = a.extract_task_answer(_record(modulation_type="ASO"))
    assert ta.predicted_modulation == ModulationDirection.SILENCE
    assert ta.predicted_target == "SOD1"
    assert ta.predicted_target_aliases == ["tofersen"]
    assert ta.confidence == 0.82
    assert ta.raw["model_used"] == "claude-opus-4-7"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("inhibitor", ModulationDirection.INHIBIT),
        ("agonist", ModulationDirection.ACTIVATE),
        ("chaperone", ModulationDirection.ACTIVATE),
        ("ASO", ModulationDirection.SILENCE),
        ("siRNA", ModulationDirection.SILENCE),
        ("gene_therapy", ModulationDirection.REPLACE),
        ("enzyme_replacement", ModulationDirection.REPLACE),
        ("monoclonal_antibody", ModulationDirection.INHIBIT),
        ("other", ModulationDirection.AGNOSTIC),
    ],
)
def test_modulation_map_covers_therapy_agent_vocab(raw, expected):
    a = TherapyAgentAdapter()
    rec = _record(modulation_type=raw)
    assert a.extract_task_answer(rec).predicted_modulation == expected


def test_to_agent_response_includes_reasoning_trace_and_token_metadata():
    a = TherapyAgentAdapter()
    resp = a.to_agent_response(_record())
    assert resp.case_id == "sod1_als"
    assert "Detected a dominant SOD1 GoF variant" in resp.answer
    assert resp.latency_ms == 4500
    assert resp.tokens_in == 1200
    assert resp.tokens_out == 350
    assert resp.structured_outputs["predicted_target"] == "SOD1"
    assert resp.structured_outputs["predicted_modulation"] == "silence"
    assert resp.claim_to_citations is not None


def test_case_id_falls_back_to_gene_mutation_slug():
    a = TherapyAgentAdapter()
    cid = a.case_id_of(_record(case_id=None))
    assert cid == "SOD1__p.Ala4Val"


def test_missing_case_id_and_gene_raises():
    a = TherapyAgentAdapter()
    with pytest.raises(ValueError):
        a.case_id_of({"mutation": "x"})
