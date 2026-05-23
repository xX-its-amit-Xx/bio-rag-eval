"""End-to-end example: evaluate the sibling `therapy-agent` project on the
10 curated rare-disease cases in `sample_gold_standards/`.

This script does three things in order:

  1. Loads the 10 YAML gold-standard cases.
  2. Calls the therapy-agent (or, in `--mock` mode, loads pre-canned
     responses from `sample_responses.jsonl`) to produce an
     `AgentResponse` per case.
  3. Runs `EvalRunner` and writes an HTML/Markdown/JSON report.

The `--mock` flag is what we use in CI / the README "real numbers" table.
It uses a `MockJudge` so the run is deterministic and offline. Real runs
swap in `AnthropicJudge` / `OpenAIJudge`.

Usage:

    # offline, deterministic, no API key needed
    python examples/eval_therapy_agent.py --mock

    # real run with Anthropic judge
    ANTHROPIC_API_KEY=sk-... python examples/eval_therapy_agent.py \\
        --judge anthropic --model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable when running from a checkout.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

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


def _load_responses(path: Path) -> list[AgentResponse]:
    out: list[AgentResponse] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(AgentResponse.model_validate_json(line))
    return out


def _build_mock_judge() -> MockJudge:
    """Return a MockJudge configured to "mostly support" everything.

    This is intentionally optimistic — it lets us exercise the pipeline
    in CI and produce demo numbers without burning API credits. For the
    bias check it returns the same label both passes, so the bias rate
    in the demo will look perfect (the real model is what you want to
    measure bias on).
    """
    return MockJudge(
        {
            ClaimList: ClaimList(
                claims=[
                    Claim(claim_id="c1", text="Mock claim 1", claim_type=ClaimType.FACTUAL, cited_ids=["s1"]),
                    Claim(claim_id="c2", text="Mock claim 2", claim_type=ClaimType.MECHANISTIC, cited_ids=["s1"]),
                ]
            ),
            GroundingJudgment: GroundingJudgment(
                claim_id="c1",
                label=GroundingLabel.SUPPORTED,
                rationale="Mock judge: cited passage explicitly supports the claim.",
                confidence=0.9,
            ),
            MechanismJudgment: MechanismJudgment(
                score=4,
                rationale="Mock judge: chain is mostly complete with one implied step.",
            ),
            EvidenceSufficiency: EvidenceSufficiency(
                score=3,
                n_independent_sources=2,
                has_primary_literature=True,
                has_human_data=True,
                rationale="Mock judge: two primary sources, both in humans.",
            ),
            CompletenessJudgment: CompletenessJudgment(
                facet="(echoed)",
                covered=True,
                rationale="Mock judge: facet is covered.",
            ),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=("mock", "anthropic", "openai"), default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--responses",
        default=str(HERE / "sample_responses.jsonl"),
        help="JSONL of agent responses keyed by case_id.",
    )
    parser.add_argument("--out", default=str(HERE.parent / "reports"), help="Report output dir.")
    parser.add_argument("--mock", action="store_true", help="Alias for --judge mock.")
    args = parser.parse_args(argv)

    if args.mock:
        args.judge = "mock"

    case_dir = HERE / "sample_gold_standards"
    case_files = sorted(str(p) for p in case_dir.glob("*.yaml"))
    cases = load_cases_from_yaml(case_files)
    print(f"Loaded {len(cases)} gold-standard cases from {case_dir}")

    responses_path = Path(args.responses)
    if not responses_path.exists():
        sys.stderr.write(
            f"Responses file not found: {responses_path}\n"
            f"In a real eval, this file is produced by running the therapy-agent on each "
            f"case's `question` and serializing the AgentResponse.\n"
        )
        return 2
    responses = _load_responses(responses_path)
    print(f"Loaded {len(responses)} agent responses from {responses_path.name}")

    if args.judge == "mock":
        judge = _build_mock_judge()
    elif args.judge == "anthropic":
        judge = AnthropicJudge(model=args.model or "claude-opus-4-7")
    else:
        judge = OpenAIJudge(model=args.model or "gpt-4.1")

    runner = EvalRunner(judge=judge, config=RunConfig(seed=7, citation_offline=True))
    report = runner.run(cases, responses)
    paths = write_report(report, args.out)

    print("\nReport written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\nAggregate metrics:")
    for name, m in report.aggregate_metrics.items():
        ci = (
            f"[{m.ci_low:.3f}, {m.ci_high:.3f}]"
            if m.ci_low is not None
            else "—"
        )
        print(f"  {name:<35s} {m.point:.3f}  {ci}  (n={m.n})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
