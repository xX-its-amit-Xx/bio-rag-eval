"""Bias-consistency: re-run grounding with the rubric order flipped.

LLM judges are known to exhibit position bias — when a rubric lists
labels in a fixed order, models can be primed to over-pick the first
option. We measure this directly by judging every claim twice, once with
labels listed `supported_first` and once `unsupported_first`, then
comparing.

    bias_consistency_rate = |{claims : label_A == label_B}| / |claims|

Anything below ~0.85 is a yellow flag. Below 0.7 means the headline
metrics in this run are not stable — the report should not be trusted
without further investigation (model swap, prompt tightening, larger n).
"""
from __future__ import annotations

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.metrics.grounding import judge_grounding
from bio_rag_eval.schemas.case import Citation
from bio_rag_eval.schemas.claims import Claim
from bio_rag_eval.schemas.judgments import GroundingJudgment
from bio_rag_eval.utils import safe_div


def bias_consistency(
    claims: list[Claim],
    citations_by_id: dict[str, Citation],
    judge: BaseJudge,
    primary_judgments: list[GroundingJudgment],
) -> tuple[dict[str, float], list[GroundingJudgment]]:
    """Run the swapped-rubric judgments and compare to the primary run.

    Returns:
        (metrics_dict, swapped_judgments)

    Metrics:
        - bias_consistency_rate: agreement between primary and swapped
        - bias_swing_to_supported: rate of "swapped run upgraded to SUPPORTED
          relative to primary" — quantifies any directional bias
        - bias_swing_to_unsupported: symmetric to above
    """
    primary_by_id = {j.claim_id: j for j in primary_judgments}
    swapped: list[GroundingJudgment] = []
    agree = 0
    n = 0
    up = 0
    down = 0
    rank = {"unsupported": 0, "contradicted": 1, "partially_supported": 2, "supported": 3}
    for claim in claims:
        prim = primary_by_id.get(claim.claim_id)
        if prim is None:
            continue
        n += 1
        swap = judge_grounding(
            claim, citations_by_id, judge, rubric_order="unsupported_first"
        )
        swapped.append(swap)
        if swap.label == prim.label:
            agree += 1
        else:
            if rank[swap.label.value] > rank[prim.label.value]:
                up += 1
            else:
                down += 1
    return (
        {
            "bias_consistency_rate": safe_div(agree, n),
            "bias_swing_to_supported": safe_div(up, n),
            "bias_swing_to_unsupported": safe_div(down, n),
            "n_bias_compared_claims": float(n),
        },
        swapped,
    )
