from __future__ import annotations

from bio_rag_eval.metrics.hallucination import compute_hallucination
from bio_rag_eval.schemas.claims import Claim, ClaimType
from bio_rag_eval.schemas.judgments import GroundingJudgment, GroundingLabel


def test_uncited_factual_claim_counts_as_hallucination(simple_case):
    claims = [
        Claim(claim_id="c1", text="A factual statement.", claim_type=ClaimType.FACTUAL, cited_ids=[]),
        Claim(claim_id="c2", text="Another factual statement.", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
    ]
    judgments = [GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok")]
    m = compute_hallucination(claims, judgments, simple_case)
    assert m["uncited_rate"] == 0.5
    assert m["miscitation_rate"] == 0.0
    assert m["hallucination_rate"] == 0.5


def test_miscited_claim_counts_separately(simple_case):
    claims = [
        Claim(claim_id="c1", text="claim", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
        Claim(claim_id="c2", text="claim", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
    ]
    judgments = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.UNSUPPORTED, rationale="ok ok ok ok"),
    ]
    m = compute_hallucination(claims, judgments, simple_case)
    assert m["uncited_rate"] == 0.0
    assert m["miscitation_rate"] == 0.5


def test_forbidden_claim_detection(simple_case):
    claims = [
        Claim(
            claim_id="c1",
            text="Actually, the gene does not exist in humans.",
            claim_type=ClaimType.FACTUAL,
            cited_ids=["s1"],
        ),
    ]
    judgments = [GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok")]
    m = compute_hallucination(claims, judgments, simple_case)
    assert m["forbidden_claim_present"] == 1.0


def test_mechanistic_only_no_factual(simple_case):
    """Mechanistic claims don't count toward the hallucination denominator
    — that metric is specifically about factual/quantitative/recommendation."""
    claims = [
        Claim(claim_id="c1", text="X activates Y", claim_type=ClaimType.MECHANISTIC, cited_ids=[]),
    ]
    m = compute_hallucination(claims, [], simple_case)
    assert m["n_factual_claims"] == 0.0
    # safe_div(0, 0) -> NaN
    assert m["hallucination_rate"] != m["hallucination_rate"]
