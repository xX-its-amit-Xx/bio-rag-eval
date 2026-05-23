"""Answer completeness.

    completeness = (n_covered + 0.5 * n_partial) / n_expected_facets

`n_covered` and `n_partial` are decided by the completeness judge — one
LLM call per facet, returning a `CompletenessJudgment`. We weight
partial coverage at 0.5 rather than 0 because a facet that was alluded
to but not committed-to is genuinely between "missing" and "covered".

If `expected_facets` is empty, the metric is NaN (the runner will treat
NaN metrics as "not applicable to this case" in aggregates).
"""
from __future__ import annotations

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.prompts import load_prompt
from bio_rag_eval.schemas.case import AgentResponse, GoldStandardCase
from bio_rag_eval.schemas.judgments import CompletenessJudgment
from bio_rag_eval.utils import safe_div

PROMPT_NAME = "completeness_judge_v1"


def judge_facet(
    facet: str,
    response: AgentResponse,
    judge: BaseJudge,
) -> CompletenessJudgment:
    """Score one facet against the answer with the completeness judge."""
    prompt = load_prompt(PROMPT_NAME)
    rendered = prompt.render(facet=facet, answer=response.answer)
    result = judge.judge_json(rendered, CompletenessJudgment)
    judgment = result.parsed
    assert isinstance(judgment, CompletenessJudgment)
    # Defensive: enforce the facet field matches input even if the judge rephrased.
    if judgment.facet != facet:
        judgment = judgment.model_copy(update={"facet": facet})
    return judgment


def compute_completeness(
    case: GoldStandardCase,
    judgments: list[CompletenessJudgment],
) -> dict[str, float]:
    """Aggregate per-facet judgments into a completeness score."""
    n_total = len(case.expected_facets)
    if n_total == 0:
        return {
            "answer_completeness": float("nan"),
            "n_facets_expected": 0.0,
            "n_facets_covered": 0.0,
            "n_facets_partial": 0.0,
        }
    by_facet = {j.facet: j for j in judgments}
    covered = 0
    partial = 0
    for f in case.expected_facets:
        j = by_facet.get(f)
        if j is None:
            continue
        if j.covered:
            covered += 1
        elif j.partial:
            partial += 1
    return {
        "answer_completeness": safe_div(covered + 0.5 * partial, n_total),
        "n_facets_expected": float(n_total),
        "n_facets_covered": float(covered),
        "n_facets_partial": float(partial),
    }
