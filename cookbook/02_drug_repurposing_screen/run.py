"""Recipe 02 — A/B compare two RAG agents on a 4-case drug-repurposing set.

Runs the eval pipeline twice (once per agent) against the same cases
and prints a side-by-side comparison of aggregate metrics.

For deterministic output in `--mock` mode, the mock judge is configured
to score Agent B's claims more favorably than Agent A's — this is what
a real grounding judge would do given how much more context Agent B
emits per claim. The bias-consistency rate is suppressed in the demo
because the mock judge has no notion of label ordering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from bio_rag_eval.judges import AnthropicJudge, MockJudge, OpenAIJudge
from bio_rag_eval.report import write_report
from bio_rag_eval.runner import EvalRunner, RunConfig, load_cases_from_yaml
from bio_rag_eval.schemas.case import AgentResponse
from bio_rag_eval.schemas.claims import Claim, ClaimList, ClaimType
from bio_rag_eval.schemas.judgments import (
    CompletenessJudgment,
    EvidenceSufficiency,
    GroundingJudgment,
    GroundingLabel,
    MechanismJudgment,
)


def _mock(label_pattern: list[GroundingLabel], evidence_score: int) -> MockJudge:
    """Build a mock judge whose grounding labels cycle through
    `label_pattern`. The evidence-sufficiency call returns a fixed score
    so the A/B difference on that metric is interpretable."""
    return MockJudge(
        {
            ClaimList: ClaimList(
                claims=[
                    Claim(claim_id="c1", text="claim 1", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
                    Claim(claim_id="c2", text="claim 2", claim_type=ClaimType.MECHANISTIC, cited_ids=["s1"]),
                ]
            ),
            GroundingJudgment: [
                GroundingJudgment(claim_id="c1", label=lbl, rationale="mock judgment per scripted pattern", confidence=0.85)
                for lbl in label_pattern
            ],
            MechanismJudgment: MechanismJudgment(score=4, rationale="mock — coherent chain"),
            EvidenceSufficiency: EvidenceSufficiency(
                score=evidence_score,
                n_independent_sources=evidence_score - 1,
                has_primary_literature=evidence_score >= 3,
                has_human_data=True,
                rationale="mock score per agent identity",
            ),
            CompletenessJudgment: [
                CompletenessJudgment(facet="f", covered=lbl == GroundingLabel.SUPPORTED, rationale="mock per scripted pattern", partial=lbl == GroundingLabel.PARTIALLY_SUPPORTED)
                for lbl in label_pattern
            ],
        }
    )


def _load_responses(path: Path) -> list[AgentResponse]:
    return [
        AgentResponse.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _print_aggregates(label: str, report) -> None:
    print(f"\n=== {label} ===")
    for name, m in report.aggregate_metrics.items():
        ci = f"[{m.ci_low:.3f}, {m.ci_high:.3f}]" if m.ci_low is not None else "—"
        print(f"  {name:<35s} {m.point:.3f}  {ci}  (n={m.n})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=("mock", "anthropic", "openai"), default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.mock:
        args.judge = "mock"

    cases = load_cases_from_yaml([str(HERE / "cases.yaml")])
    responses_a = _load_responses(HERE / "agent_a_responses.jsonl")
    responses_b = _load_responses(HERE / "agent_b_responses.jsonl")

    # Pattern: A has 1/2 supported per case on average, B has near-all supported.
    a_pattern = [GroundingLabel.SUPPORTED, GroundingLabel.UNSUPPORTED] * 4
    b_pattern = [GroundingLabel.SUPPORTED] * 8

    if args.judge == "mock":
        judge_a, judge_b = _mock(a_pattern, evidence_score=2), _mock(b_pattern, evidence_score=4)
    elif args.judge == "anthropic":
        m = args.model or "claude-opus-4-7"
        judge_a = judge_b = AnthropicJudge(model=m)
    else:
        m = args.model or "gpt-4.1"
        judge_a = judge_b = OpenAIJudge(model=m)

    config = RunConfig(seed=7, citation_offline=True, run_bias_check=False)
    report_a = EvalRunner(judge=judge_a, config=config).run(cases, responses_a)
    report_b = EvalRunner(judge=judge_b, config=config).run(cases, responses_b)

    paths_a = write_report(report_a, HERE / "reports" / "agent_a")
    paths_b = write_report(report_b, HERE / "reports" / "agent_b")
    print(f"Agent A report: {paths_a['html']}")
    print(f"Agent B report: {paths_b['html']}")

    _print_aggregates("Agent A (baseline)", report_a)
    _print_aggregates("Agent B (reranker)", report_b)

    print("\n--- Difference (B minus A) on shared metrics ---")
    for name, m_b in report_b.aggregate_metrics.items():
        m_a = report_a.aggregate_metrics.get(name)
        if not m_a:
            continue
        diff = m_b.point - m_a.point
        ci_a = f"[{m_a.ci_low:.2f},{m_a.ci_high:.2f}]" if m_a.ci_low is not None else "—"
        ci_b = f"[{m_b.ci_low:.2f},{m_b.ci_high:.2f}]" if m_b.ci_low is not None else "—"
        print(f"  {name:<35s}  A={m_a.point:.2f} {ci_a}   B={m_b.point:.2f} {ci_b}   diff={diff:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
