"""Hallucination metric — distinguishes uncited vs miscited claims.

    hallucination_rate = (uncited_factual + miscited) / total_factual

  uncited_factual = factual/quantitative/recommendation claim with empty `cited_ids`
  miscited        = same claim types, cited, but judge returned CONTRADICTED or UNSUPPORTED
  total_factual   = count of factual/quantitative/recommendation claims

We separate the two because they have very different fixes:
  - Uncited: agent failed to attribute; might be true but unverified.
  - Miscited: agent attributed to a source that doesn't support it; a
    citation-collision failure, often worse than uncited (a reader who
    spot-checks one source and finds it irrelevant loses trust quickly).

A `forbidden_claim_present` boolean is also computed — set to True iff
any claim's text contains one of the gold standard's `forbidden_claims`
substrings. Forbidden claims are a strict failure regardless of citation.
"""
from __future__ import annotations

from bio_rag_eval.schemas.case import GoldStandardCase
from bio_rag_eval.schemas.claims import Claim, ClaimType
from bio_rag_eval.schemas.judgments import GroundingJudgment, GroundingLabel
from bio_rag_eval.utils import safe_div

FACTUAL_TYPES = {ClaimType.FACTUAL, ClaimType.QUANTITATIVE, ClaimType.RECOMMENDATION}


def compute_hallucination(
    claims: list[Claim],
    judgments: list[GroundingJudgment],
    case: GoldStandardCase,
) -> dict[str, float]:
    """Compute hallucination metrics for one case.

    Formula:
        hallucination_rate = (uncited + miscited) / total_factual
        uncited_rate       = uncited / total_factual
        miscitation_rate   = miscited / total_factual
        forbidden_claim_present = 1.0 iff any forbidden substring appears
                                  in any claim text, else 0.0
    """
    by_id = {j.claim_id: j for j in judgments}
    factual_claims = [c for c in claims if c.claim_type in FACTUAL_TYPES]
    total = len(factual_claims)

    uncited = 0
    miscited = 0
    for c in factual_claims:
        if not c.cited_ids:
            uncited += 1
            continue
        j = by_id.get(c.claim_id)
        if j is None:
            # Cited but no judgment landed — treat as miscited (conservative).
            miscited += 1
            continue
        if j.label in (GroundingLabel.CONTRADICTED, GroundingLabel.UNSUPPORTED):
            miscited += 1

    forbidden_hit = 0.0
    if case.forbidden_claims:
        joined = "\n".join(c.text.lower() for c in claims)
        for substr in case.forbidden_claims:
            if substr.lower() in joined:
                forbidden_hit = 1.0
                break

    return {
        "hallucination_rate": safe_div(uncited + miscited, total),
        "uncited_rate": safe_div(uncited, total),
        "miscitation_rate": safe_div(miscited, total),
        "forbidden_claim_present": forbidden_hit,
        "n_factual_claims": float(total),
    }
