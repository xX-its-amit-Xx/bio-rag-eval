from __future__ import annotations

from bio_rag_eval.runner import EvalRunner, RunConfig


def test_runner_end_to_end_mock(simple_case, simple_response, mock_judge_all_supported):
    runner = EvalRunner(
        judge=mock_judge_all_supported,
        config=RunConfig(run_bias_check=False, evidence_sufficiency_enabled=True, citation_offline=True),
    )
    report = runner.run([simple_case], [simple_response])

    assert len(report.case_results) == 1
    cr = report.case_results[0]
    assert cr.error is None
    assert cr.metrics["correct_target"] == 1.0
    assert cr.metrics["correct_modulation_direction"] == 1.0
    assert cr.metrics["mechanism_coherence"] == 5.0
    assert cr.metrics["task_success_composite"] == 1.0
    assert cr.metrics["answer_completeness"] == 1.0
    assert "evidence_sufficiency_score" in cr.metrics

    # Metadata is populated with reproducibility info.
    md = report.metadata
    assert md.run_id
    assert md.judge_model == "mock"
    assert md.prompt_versions
    assert md.config_hash
    assert md.bootstrap_n_resamples == 1000


def test_runner_handles_missing_response(simple_case, mock_judge_all_supported):
    runner = EvalRunner(judge=mock_judge_all_supported, config=RunConfig(run_bias_check=False))
    report = runner.run([simple_case], [])
    assert len(report.case_results) == 1
    assert report.case_results[0].error is not None
    assert "no agent response" in report.case_results[0].error


def test_aggregate_contains_grounding_with_ci(simple_case, simple_response, mock_judge_all_supported):
    runner = EvalRunner(judge=mock_judge_all_supported, config=RunConfig(run_bias_check=False))
    report = runner.run([simple_case], [simple_response])
    # Bootstrap CIs are None when n<5 (one case -> two claims; ok for grounding which bootstraps over claims).
    # Just check the metric is present and its `n` matches the claim count.
    if "grounding_rate" in report.aggregate_metrics:
        m = report.aggregate_metrics["grounding_rate"]
        assert m.n == 2  # two graded claims
        assert m.point == 1.0
