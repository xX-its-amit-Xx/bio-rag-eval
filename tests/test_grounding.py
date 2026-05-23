from __future__ import annotations

from bio_rag_eval.metrics.grounding import (
    compute_grounding,
    grounding_per_claim_supported,
    judge_grounding,
)
from bio_rag_eval.schemas.case import Citation
from bio_rag_eval.schemas.claims import Claim, ClaimType
from bio_rag_eval.schemas.judgments import GroundingJudgment, GroundingLabel


def test_compute_grounding_all_supported(two_claims):
    judgments = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok", confidence=0.9),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok", confidence=0.9),
    ]
    m = compute_grounding(two_claims, judgments)
    assert m["grounding_rate"] == 1.0
    assert m["weak_grounding_rate"] == 1.0
    assert m["contradiction_rate"] == 0.0
    assert m["unsupported_rate"] == 0.0
    assert m["n_graded_claims"] == 2.0


def test_compute_grounding_mixed(two_claims):
    judgments = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.PARTIALLY_SUPPORTED, rationale="ok ok ok ok"),
    ]
    m = compute_grounding(two_claims, judgments)
    assert m["grounding_rate"] == 0.5
    assert m["weak_grounding_rate"] == 0.75


def test_opinion_claims_excluded_from_grading():
    claims = [
        Claim(claim_id="c1", text="exciting!", claim_type=ClaimType.OPINION),
        Claim(claim_id="c2", text="X causes Y", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
    ]
    judgments = [GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok")]
    m = compute_grounding(claims, judgments)
    assert m["n_graded_claims"] == 1.0
    assert m["grounding_rate"] == 1.0


def test_judge_grounding_no_citations_short_circuits():
    """A claim with no cited_ids gets auto-graded UNSUPPORTED without
    invoking the judge — we test that by passing a judge that would
    fail if called."""

    class Boom:
        model = "boom"
        provider = "boom"

        def judge_json(self, *a, **k):
            raise AssertionError("judge should not be called when claim has no citations")

    claim = Claim(claim_id="c1", text="uncited claim", claim_type=ClaimType.FACTUAL, cited_ids=[])
    judgment = judge_grounding(claim, {}, Boom())  # type: ignore[arg-type]
    assert judgment.label == GroundingLabel.UNSUPPORTED
    assert judgment.confidence == 1.0


def test_per_claim_supported_vector(two_claims):
    judgments = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.CONTRADICTED, rationale="ok ok ok ok"),
    ]
    vec = grounding_per_claim_supported(two_claims, judgments)
    assert vec == [1, 0]


def test_judge_id_mismatch_is_corrected(mock_judge_all_supported, two_claims):
    """If the judge returns a different claim_id than asked, the
    grounding metric should still index correctly — covered by the
    runner. Here we just test the safety patch in judge_grounding."""
    citations = {"s1": Citation(citation_id="s1", title="x", snippet="x")}
    j = judge_grounding(two_claims[0], citations, mock_judge_all_supported)
    assert j.claim_id == "c1"
