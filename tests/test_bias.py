from __future__ import annotations

from bio_rag_eval.judges import MockJudge
from bio_rag_eval.metrics.bias import bias_consistency
from bio_rag_eval.schemas.case import Citation
from bio_rag_eval.schemas.claims import Claim, ClaimList, ClaimType
from bio_rag_eval.schemas.judgments import GroundingJudgment, GroundingLabel


def test_perfect_consistency_when_judge_repeats():
    claims = [
        Claim(claim_id="c1", text="claim X here", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
        Claim(claim_id="c2", text="claim Y here", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
    ]
    citations = {"s1": Citation(citation_id="s1", title="x", snippet="X and Y are both supported here.")}
    primary = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
    ]
    # Mock judge returns SUPPORTED for both regardless of rubric order.
    judge = MockJudge(
        {
            ClaimList: ClaimList(claims=claims),
            GroundingJudgment: [
                GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
                GroundingJudgment(claim_id="c2", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok"),
            ],
        }
    )
    metrics, swapped = bias_consistency(claims, citations, judge, primary)
    assert metrics["bias_consistency_rate"] == 1.0
    assert len(swapped) == 2


def test_swing_detection_when_swapped_judge_disagrees():
    claims = [
        Claim(claim_id="c1", text="claim X here", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
    ]
    citations = {"s1": Citation(citation_id="s1", title="x", snippet="X is supported here.")}
    primary = [GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="ok ok ok ok")]
    judge = MockJudge(
        {
            GroundingJudgment: [
                GroundingJudgment(claim_id="c1", label=GroundingLabel.UNSUPPORTED, rationale="ok ok ok ok"),
            ],
        }
    )
    metrics, _ = bias_consistency(claims, citations, judge, primary)
    assert metrics["bias_consistency_rate"] == 0.0
    # primary was SUPPORTED, swapped is UNSUPPORTED -> swing-to-unsupported = 1.0
    assert metrics["bias_swing_to_unsupported"] == 1.0
    assert metrics["bias_swing_to_supported"] == 0.0
