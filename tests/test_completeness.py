from __future__ import annotations

from bio_rag_eval.metrics.completeness import compute_completeness
from bio_rag_eval.schemas.judgments import CompletenessJudgment


def test_full_coverage(simple_case):
    judgments = [
        CompletenessJudgment(facet="facet A", covered=True, rationale="covered ok"),
        CompletenessJudgment(facet="facet B", covered=True, rationale="covered ok"),
    ]
    m = compute_completeness(simple_case, judgments)
    assert m["answer_completeness"] == 1.0


def test_partial_weighted_half(simple_case):
    judgments = [
        CompletenessJudgment(facet="facet A", covered=True, rationale="covered ok"),
        CompletenessJudgment(facet="facet B", covered=False, partial=True, rationale="alluded"),
    ]
    m = compute_completeness(simple_case, judgments)
    assert m["answer_completeness"] == 0.75


def test_missing_facet_judgment_is_zero(simple_case):
    judgments = [
        CompletenessJudgment(facet="facet A", covered=True, rationale="covered ok"),
    ]
    m = compute_completeness(simple_case, judgments)
    assert m["answer_completeness"] == 0.5


def test_no_facets_returns_nan():
    from bio_rag_eval.schemas.case import GoldStandardCase, TaskType

    case = GoldStandardCase(case_id="t", name="t", task_type=TaskType.LITERATURE_QA, question="?")
    m = compute_completeness(case, [])
    assert m["answer_completeness"] != m["answer_completeness"]
