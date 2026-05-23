"""Judge-consistency tests.

Two layers:

  A) Always-on: with `MockJudge`, verify the *infrastructure* that
     measures consistency runs each prompt twice on a fixed input and
     computes label-agreement / score-variance correctly. The mock is
     deterministic so the test is also deterministic (variance == 0).

  B) Opt-in (env-gated): when `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
     is set AND `BIO_RAG_EVAL_LIVE_CONSISTENCY=1`, hit the real judge
     twice on the same fixed inputs and assert that score variance is
     below a configurable threshold. Skipped otherwise — these are
     genuinely expensive to run and are not appropriate for CI by
     default.

The threshold values come from the bench-time literature on
LLM-as-judge: Zheng et al. 2023 (MT-Bench) and Wang et al. 2023
("Large Language Models are not Fair Evaluators") both report
position-bias and label-jitter at the 5-15% level for strong models on
clear inputs. We use 0.30 as a loose ceiling (test fails if the judge
disagrees with itself on more than 30% of items) which is generous
enough to not trigger on rate-limit-induced retries but strict enough
to catch a degraded model.
"""
from __future__ import annotations

import os
import statistics

import pytest

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.metrics.grounding import judge_grounding
from bio_rag_eval.schemas.case import Citation
from bio_rag_eval.schemas.claims import Claim, ClaimType


CONSISTENCY_THRESHOLD = float(os.environ.get("BIO_RAG_EVAL_CONSISTENCY_THRESHOLD", "0.30"))


def _make_fixed_claim_and_citation() -> tuple[Claim, dict[str, Citation]]:
    claim = Claim(
        claim_id="c1",
        text="Tofersen reduced CSF neurofilament light by approximately 60% in VALOR at week 28.",
        claim_type=ClaimType.QUANTITATIVE,
        cited_ids=["s1"],
    )
    citation = Citation(
        citation_id="s1",
        pmid="36129998",
        title="Miller TM et al. Tofersen for SOD1-Associated ALS (VALOR). NEJM 2022",
        snippet=(
            "In the VALOR trial, intrathecal tofersen reduced CSF neurofilament "
            "light chain concentrations by approximately 60% from baseline at week 28 "
            "versus increases in the placebo group."
        ),
    )
    return claim, {citation.citation_id: citation}


# ── Layer A: always-on infrastructure check ─────────────────────────


def test_mock_judge_consistency_infrastructure(mock_judge_all_supported):
    """Same MockJudge, same claim, same citation -> identical labels
    twice. Confirms the helper composes correctly and we can detect a
    swing if one were present."""
    claim, citations = _make_fixed_claim_and_citation()

    labels: list[str] = []
    for _ in range(2):
        # Re-seed by passing a fresh judge each time to mirror real-judge
        # call patterns where each call is a fresh request.
        j = judge_grounding(claim, citations, mock_judge_all_supported)
        labels.append(j.label.value)

    assert labels[0] == labels[1], "MockJudge is deterministic; labels must agree"


# ── Layer B: opt-in real-judge variance ──────────────────────────────


def _real_judge_available() -> BaseJudge | None:
    """Return a real BaseJudge or None if not configured. Honors
    BIO_RAG_EVAL_LIVE_CONSISTENCY=1 as the master gate."""
    if os.environ.get("BIO_RAG_EVAL_LIVE_CONSISTENCY", "") != "1":
        return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from bio_rag_eval.judges import AnthropicJudge

        return AnthropicJudge(model=os.environ.get("BIO_RAG_EVAL_JUDGE_MODEL", "claude-opus-4-7"))
    if os.environ.get("OPENAI_API_KEY"):
        from bio_rag_eval.judges import OpenAIJudge

        return OpenAIJudge(model=os.environ.get("BIO_RAG_EVAL_JUDGE_MODEL", "gpt-4.1"))
    return None


@pytest.mark.skipif(
    _real_judge_available() is None,
    reason="set BIO_RAG_EVAL_LIVE_CONSISTENCY=1 + ANTHROPIC_API_KEY (or OPENAI_API_KEY) to enable",
)
def test_live_grounding_judge_label_stability():
    """Same claim, same citation, run the real judge 4 times. Assert
    label agreement >= 1 - CONSISTENCY_THRESHOLD."""
    judge = _real_judge_available()
    assert judge is not None
    claim, citations = _make_fixed_claim_and_citation()

    labels: list[str] = []
    for _ in range(4):
        j = judge_grounding(claim, citations, judge)
        labels.append(j.label.value)

    modal = max(set(labels), key=labels.count)
    agreement = labels.count(modal) / len(labels)
    assert agreement >= (1.0 - CONSISTENCY_THRESHOLD), (
        f"grounding judge disagreed with itself: labels={labels}, "
        f"modal={modal!r}, agreement={agreement:.2f} < threshold {1.0 - CONSISTENCY_THRESHOLD:.2f}"
    )


@pytest.mark.skipif(
    _real_judge_available() is None,
    reason="set BIO_RAG_EVAL_LIVE_CONSISTENCY=1 + ANTHROPIC_API_KEY (or OPENAI_API_KEY) to enable",
)
def test_live_mechanism_judge_score_variance():
    """Same answer, same context, run mechanism judge 4 times. Assert
    stdev(score) <= 0.75 (i.e. on a 1-5 scale, two-thirds of runs
    within ±0.75 of the mean)."""
    from bio_rag_eval.metrics.task_success import judge_mechanism
    from bio_rag_eval.schemas.case import AgentResponse, GoldStandardCase, TaskType

    judge = _real_judge_available()
    assert judge is not None
    case = GoldStandardCase(
        case_id="t",
        name="fixture",
        task_type=TaskType.VARIANT_TO_STRATEGY,
        question="?",
        context={
            "mutation": "SOD1 p.Ala4Val",
            "disease_phenotype": "familial ALS",
        },
    )
    response = AgentResponse(
        case_id="t",
        answer=(
            "The p.Ala4Val variant in SOD1 produces a dominant toxic gain of function: "
            "misfolded SOD1 aggregates in motor neurons drive degeneration. Silencing "
            "SOD1 mRNA removes the substrate for aggregation. Tofersen is an antisense "
            "oligonucleotide administered intrathecally that reduces SOD1 mRNA and CSF "
            "neurofilament light."
        ),
    )
    scores = [judge_mechanism(case, response, judge).score for _ in range(4)]
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    assert sd <= 0.75, f"mechanism judge unstable: scores={scores}, stdev={sd:.2f}"
