from __future__ import annotations

from bio_rag_eval.metrics.task_success import (
    compute_task_success,
    score_modulation,
    score_target,
)
from bio_rag_eval.schemas.case import AgentResponse, ModulationDirection
from bio_rag_eval.schemas.judgments import MechanismJudgment


def test_score_target_exact(simple_case, simple_response):
    assert score_target(simple_case, simple_response) == 1


def test_score_target_alias(simple_case):
    response = AgentResponse(
        case_id="t1",
        answer="aliasA is the answer.",
        structured_outputs={"predicted_target": "aliasA"},
    )
    assert score_target(simple_case, response) == 1


def test_score_target_miss(simple_case):
    response = AgentResponse(
        case_id="t1", answer="totally unrelated", structured_outputs={"predicted_target": "GENE2"}
    )
    assert score_target(simple_case, response) == 0


def test_score_modulation_structured(simple_case, simple_response):
    assert score_modulation(simple_case, simple_response) == 1


def test_score_modulation_inferred_from_prose(simple_case):
    response = AgentResponse(
        case_id="t1",
        answer="We propose using an antisense oligonucleotide to silence the gene.",
    )
    # case expects SILENCE; prose infers silence -> 1
    assert score_modulation(simple_case, response) == 1


def test_score_modulation_agnostic_is_always_one():
    from bio_rag_eval.schemas.case import GoldStandardCase, TaskType

    case = GoldStandardCase(
        case_id="t1",
        name="x",
        task_type=TaskType.VARIANT_TO_STRATEGY,
        question="?",
        expected_modulation=ModulationDirection.AGNOSTIC,
    )
    response = AgentResponse(case_id="t1", answer="anything goes")
    assert score_modulation(case, response) == 1


def test_composite_includes_mechanism(simple_case, simple_response):
    mech = MechanismJudgment(score=5, rationale="full chain")
    out = compute_task_success(simple_case, simple_response, mech)
    assert out["correct_target"] == 1.0
    assert out["correct_modulation_direction"] == 1.0
    assert out["mechanism_coherence"] == 5.0
    assert out["task_success_composite"] == 1.0


def test_composite_nan_when_mechanism_missing_v2s(simple_case, simple_response):
    out = compute_task_success(simple_case, simple_response, None)
    # V2S task with no mechanism judgment -> composite is NaN
    assert out["task_success_composite"] != out["task_success_composite"]
