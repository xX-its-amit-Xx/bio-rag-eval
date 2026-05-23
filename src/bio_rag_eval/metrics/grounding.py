"""Grounding metric.

    grounding_rate = |{c in claims : judge(c) == SUPPORTED}| / |claims_to_grade|

`claims_to_grade` is the subset of extracted claims with `claim_type` in
{factual, mechanistic, quantitative, recommendation}. Opinion claims are
excluded because penalizing "this is exciting" as ungrounded would
swamp the signal from real factual errors.

We score the four-way label (supported / partially / contradicted /
unsupported) so the report can break down the failure modes — but the
headline `grounding_rate` collapses to a strict 0/1 (SUPPORTED only).
A separate `weak_grounding_rate` includes PARTIALLY_SUPPORTED at weight
0.5; see `compute_grounding` return dict.
"""
from __future__ import annotations

from typing import Literal

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.prompts import load_prompt
from bio_rag_eval.schemas.case import AgentResponse, Citation
from bio_rag_eval.schemas.claims import Claim, ClaimType
from bio_rag_eval.schemas.judgments import GroundingJudgment, GroundingLabel
from bio_rag_eval.utils import safe_div

GRADED_TYPES = {
    ClaimType.FACTUAL,
    ClaimType.MECHANISTIC,
    ClaimType.QUANTITATIVE,
    ClaimType.RECOMMENDATION,
}

PROMPT_NAME = "grounding_judge_v1"


def judge_grounding(
    claim: Claim,
    citations_by_id: dict[str, Citation],
    judge: BaseJudge,
    rubric_order: Literal["supported_first", "unsupported_first"] = "supported_first",
) -> GroundingJudgment:
    """Run the grounding judge on a single claim.

    `rubric_order` is exposed so `bias_consistency` can re-run with the
    list flipped — the prompt enumerates labels in the chosen order, and
    we test whether the judge's verdict depends on that ordering.
    """
    cited = [citations_by_id[cid] for cid in claim.cited_ids if cid in citations_by_id]
    if not cited:
        return GroundingJudgment(
            claim_id=claim.claim_id,
            label=GroundingLabel.UNSUPPORTED,
            rationale="No citations attached to this claim (auto-graded UNSUPPORTED without LLM call).",
            quoted_evidence=None,
            confidence=1.0,
        )

    prompt = load_prompt(PROMPT_NAME)
    rendered = prompt.render(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        claim_type=claim.claim_type.value,
        cited_passages=[c.model_dump() for c in cited],
        rubric_order=rubric_order,
    )
    result = judge.judge_json(rendered, GroundingJudgment)
    judgment = result.parsed
    assert isinstance(judgment, GroundingJudgment)
    # Force the claim_id to match the input — defensive against an over-eager judge.
    if judgment.claim_id != claim.claim_id:
        judgment = judgment.model_copy(update={"claim_id": claim.claim_id})
    return judgment


def compute_grounding(
    claims: list[Claim],
    judgments: list[GroundingJudgment],
) -> dict[str, float]:
    """Aggregate per-claim judgments into grounding metrics.

    Returns:
      - grounding_rate: strict (SUPPORTED only) / graded
      - weak_grounding_rate: (SUPPORTED + 0.5 * PARTIALLY_SUPPORTED) / graded
      - contradiction_rate: CONTRADICTED / graded
      - unsupported_rate: UNSUPPORTED / graded
      - n_graded_claims
    """
    by_id = {j.claim_id: j for j in judgments}
    graded_claims = [c for c in claims if c.claim_type in GRADED_TYPES]
    n = len(graded_claims)
    supported = 0
    partial = 0
    contradicted = 0
    unsupported = 0
    for c in graded_claims:
        j = by_id.get(c.claim_id)
        if j is None:
            unsupported += 1
            continue
        if j.label == GroundingLabel.SUPPORTED:
            supported += 1
        elif j.label == GroundingLabel.PARTIALLY_SUPPORTED:
            partial += 1
        elif j.label == GroundingLabel.CONTRADICTED:
            contradicted += 1
        else:
            unsupported += 1
    return {
        "grounding_rate": safe_div(supported, n),
        "weak_grounding_rate": safe_div(supported + 0.5 * partial, n),
        "contradiction_rate": safe_div(contradicted, n),
        "unsupported_rate": safe_div(unsupported, n),
        "n_graded_claims": float(n),
    }


def grounding_per_claim_supported(
    claims: list[Claim],
    judgments: list[GroundingJudgment],
) -> list[int]:
    """Return a 0/1 vector of "supported?" per graded claim, in the order
    of `claims`. Used by the runner to feed the bootstrap aggregator."""
    by_id = {j.claim_id: j for j in judgments}
    out: list[int] = []
    for c in claims:
        if c.claim_type not in GRADED_TYPES:
            continue
        j = by_id.get(c.claim_id)
        out.append(1 if (j and j.label == GroundingLabel.SUPPORTED) else 0)
    return out


# Convenience: helper for the runner to find citation by id quickly.
def index_citations(response: AgentResponse) -> dict[str, Citation]:
    return {c.citation_id: c for c in response.citations}
