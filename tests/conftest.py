"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from bio_rag_eval.judges import MockJudge
from bio_rag_eval.schemas.case import (
    AgentResponse,
    Citation,
    GoldStandardCase,
    ModulationDirection,
    TaskType,
)
from bio_rag_eval.schemas.claims import Claim, ClaimList, ClaimType
from bio_rag_eval.schemas.judgments import (
    CompletenessJudgment,
    EvidenceSufficiency,
    GroundingJudgment,
    GroundingLabel,
    MechanismJudgment,
)


@pytest.fixture
def simple_case() -> GoldStandardCase:
    return GoldStandardCase(
        case_id="t1",
        name="test case",
        task_type=TaskType.VARIANT_TO_STRATEGY,
        question="What is the strategy?",
        context={"mutation": "p.X1Y", "disease_phenotype": "test disease"},
        expected_target="GENE1",
        expected_target_aliases=["aliasA"],
        expected_modulation=ModulationDirection.SILENCE,
        expected_facets=["facet A", "facet B"],
        required_citations=["seminal2020"],
        forbidden_claims=["the gene does not exist"],
    )


@pytest.fixture
def simple_response() -> AgentResponse:
    return AgentResponse(
        case_id="t1",
        answer="GENE1 is silenced via aliasA in this strategy.",
        citations=[
            Citation(
                citation_id="s1",
                pmid="12345678",
                title="Seminal 2020",
                snippet="aliasA silences GENE1 expression in disease X.",
            )
        ],
        structured_outputs={"predicted_target": "GENE1", "predicted_modulation": "silence"},
    )


@pytest.fixture
def two_claims() -> list[Claim]:
    return [
        Claim(claim_id="c1", text="GENE1 silencing reverses the phenotype.", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
        Claim(claim_id="c2", text="aliasA is the agent of choice.", claim_type=ClaimType.RECOMMENDATION, cited_ids=["s1"]),
    ]


@pytest.fixture
def mock_judge_all_supported(two_claims) -> MockJudge:
    return MockJudge(
        {
            ClaimList: ClaimList(claims=two_claims),
            GroundingJudgment: [
                GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="snippet supports claim 1.", confidence=0.9),
                GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="snippet supports claim 2.", confidence=0.9),
            ],
            MechanismJudgment: MechanismJudgment(score=5, rationale="full chain present"),
            EvidenceSufficiency: EvidenceSufficiency(
                score=4,
                n_independent_sources=2,
                has_primary_literature=True,
                has_human_data=True,
                rationale="solid primary literature",
            ),
            CompletenessJudgment: [
                CompletenessJudgment(facet="facet A", covered=True, rationale="covered"),
                CompletenessJudgment(facet="facet B", covered=True, rationale="covered"),
            ],
        }
    )
