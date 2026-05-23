"""Run recipe 01 — clinical-trial-matching miscitation diagnostic.

Designed so a reader can run `python run.py --mock` with no API key and
reproduce the numbers in `expected_output.md`.
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


def _mock_judge_for_recipe_01() -> MockJudge:
    """Pre-canned responses that mirror what a real judge would say on
    this set of claims. The grounding judge intentionally returns
    UNSUPPORTED on three of seven claims to demonstrate the miscitation
    diagnostic.
    """
    extracted = ClaimList(
        claims=[
            Claim(claim_id="c1", text="ATLAS (NCT04856982) is enrolling presymptomatic SOD1 carriers and treats them with tofersen.", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
            Claim(claim_id="c2", text="ATLAS eligibility is open to adults aged 18-80.", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
            Claim(claim_id="c3", text="The VALOR open-label extension (NCT03070119) continues to dose patients who completed VALOR.", claim_type=ClaimType.FACTUAL, cited_ids=["s2"]),
            Claim(claim_id="c4", text="VALOR extension eligibility requires a baseline ALSFRS-R >= 30.", claim_type=ClaimType.QUANTITATIVE, cited_ids=["s2"]),
            Claim(claim_id="c5", text="NCT04768972 is evaluating ulefnersen in adults with confirmed SOD1 ALS aged 18-65.", claim_type=ClaimType.FACTUAL, cited_ids=["s3"]),
            Claim(claim_id="c6", text="Ulefnersen requires patients to complete a swallowing test for eligibility.", claim_type=ClaimType.FACTUAL, cited_ids=["s3"]),
            Claim(claim_id="c7", text="Tofersen is investigational only and has not been FDA-approved.", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
        ]
    )
    grounding_judgments = [
        GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="Snippet explicitly states ATLAS enrolls presymptomatic adult SOD1 carriers and tests tofersen.", confidence=0.95),
        GroundingJudgment(claim_id="c2", label=GroundingLabel.PARTIALLY_SUPPORTED, rationale="Snippet says '18 or older' but does not state an upper bound of 80; the claim adds an unverified upper bound.", confidence=0.85),
        GroundingJudgment(claim_id="c3", label=GroundingLabel.SUPPORTED, rationale="Snippet states the OLE enrolls participants who completed VALOR.", confidence=0.95),
        GroundingJudgment(claim_id="c4", label=GroundingLabel.CONTRADICTED, rationale="Snippet explicitly states there is no ALSFRS-R cutoff at extension entry — the claim invents the criterion.", confidence=0.95),
        GroundingJudgment(claim_id="c5", label=GroundingLabel.CONTRADICTED, rationale="Snippet describes ulefnersen for FUS-mutation ALS, not SOD1; SOD1 is an exclusion criterion.", confidence=0.95),
        GroundingJudgment(claim_id="c6", label=GroundingLabel.UNSUPPORTED, rationale="Snippet does not mention a swallowing test.", confidence=0.9),
        GroundingJudgment(claim_id="c7", label=GroundingLabel.CONTRADICTED, rationale="Tofersen received FDA accelerated approval in April 2023; the claim is factually false and is a forbidden_claim.", confidence=0.95),
    ]
    return MockJudge(
        {
            ClaimList: extracted,
            GroundingJudgment: grounding_judgments,
            MechanismJudgment: MechanismJudgment(score=3, rationale="Not applicable for trial-matching but echoed for the pipeline."),
            EvidenceSufficiency: EvidenceSufficiency(
                score=2,
                n_independent_sources=3,
                has_primary_literature=False,
                has_human_data=True,
                rationale="Only trial registry records — no primary outcome literature cited.",
            ),
            CompletenessJudgment: [
                CompletenessJudgment(facet="At least one trial of an investigational SOD1-targeted therapy is currently recruiting", covered=True, rationale="ATLAS is cited."),
                CompletenessJudgment(facet="Eligibility criteria stated must match the actual cited trial's protocol", covered=False, partial=True, rationale="Two of three trials have fabricated criteria."),
                CompletenessJudgment(facet="Ambulatory status (e.g. ALSFRS-R cutoff, walking ability) is correctly described", covered=False, partial=False, rationale="The agent invents an ALSFRS-R cutoff that does not exist."),
            ],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=("mock", "anthropic", "openai"), default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.mock:
        args.judge = "mock"

    cases = load_cases_from_yaml([str(HERE / "case.yaml")])
    with open(HERE / "agent_response.jsonl", encoding="utf-8") as f:
        responses = [AgentResponse.model_validate_json(line) for line in f if line.strip()]

    if args.judge == "mock":
        judge = _mock_judge_for_recipe_01()
    elif args.judge == "anthropic":
        judge = AnthropicJudge(model=args.model or "claude-opus-4-7")
    else:
        judge = OpenAIJudge(model=args.model or "gpt-4.1")

    config = RunConfig(seed=7, citation_offline=True, run_bias_check=False, evidence_sufficiency_enabled=True)
    report = EvalRunner(judge=judge, config=config).run(cases, responses)
    paths = write_report(report, HERE / "reports")

    print(f"Report: {paths['html']}")
    print()
    for name, m in report.aggregate_metrics.items():
        ci = f"[{m.ci_low:.3f}, {m.ci_high:.3f}]" if m.ci_low is not None else "—"
        print(f"  {name:<35s} {m.point:.3f}  {ci}  (n={m.n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
