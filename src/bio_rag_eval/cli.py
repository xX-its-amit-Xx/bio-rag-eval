"""bio-rag-eval CLI.

Two ways to invoke `run`:

  A) "raw" mode — you've already serialized AgentResponse JSONL + a
     directory of GoldStandardCase YAMLs:

        bio-rag-eval run --cases CASES.yaml --responses RESPONSES.jsonl \\
                         --out reports/

  B) "adapter" mode — point at the source system's native output, name
     a predictions adapter, and (optionally) a gold-source adapter:

        bio-rag-eval run \\
            --predictions benchmark_runs/2026-05-23/results.jsonl \\
            --predictions-adapter therapy_agent_v1 \\
            --gold-source fda_triples_v1 \\
            --output report/

     Outputs:
       - report/scorecard.html
       - report/scorecard.md
       - report/raw_metrics.json
       - report/per_case/<case_id>.json   (one detailed JSON per case)
       - report/run_metadata.json         (RunMetadata only, for cross-run diffs)

The two modes coexist so anyone with already-converted AgentResponses
keeps working. Adapter mode is recommended for integrations.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from bio_rag_eval.adapters import (
    get_gold_adapter,
    get_predictions_adapter,
    list_adapters,
)
from bio_rag_eval.judges import AnthropicJudge, MockJudge, OpenAIJudge
from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.schemas.claims import Claim, ClaimList, ClaimType
from bio_rag_eval.schemas.judgments import (
    CompletenessJudgment,
    EvidenceSufficiency,
    GroundingJudgment,
    GroundingLabel,
    MechanismJudgment,
)
from bio_rag_eval.prompts import list_prompts
from bio_rag_eval.report import render_html, render_markdown
from bio_rag_eval.runner import EvalRunner, RunConfig, load_cases_from_yaml
from bio_rag_eval.schemas.case import AgentResponse, GoldStandardCase, TaskType


# ── helpers ─────────────────────────────────────────────────────────


def _load_responses_jsonl(path: Path) -> list[AgentResponse]:
    with open(path, encoding="utf-8") as f:
        return [
            AgentResponse.model_validate_json(line)
            for line in f
            if line.strip()
        ]


def _load_predictions_via_adapter(path: Path, adapter_name: str) -> list[AgentResponse]:
    """Read a JSONL of source-format records and convert each to AgentResponse."""
    adapter = get_predictions_adapter(adapter_name)
    out: list[AgentResponse] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: not valid JSON: {e}") from e
            try:
                out.append(adapter.to_agent_response(record))
            except Exception as e:
                raise ValueError(
                    f"{path}:{i}: adapter {adapter_name!r} failed: {e}"
                ) from e
    return out


def _synthesize_cases_from_gold(
    responses: list[AgentResponse],
    gold_adapter_name: str,
) -> tuple[list[GoldStandardCase], list[str]]:
    """For every prediction, fetch its gold from the gold adapter and build
    a minimal GoldStandardCase. Returns (cases, skipped_ids).

    Cases without gold coverage are reported in `skipped_ids` rather than
    silently dropped — the CLI surfaces them so the user knows the gold
    source has a gap (typical for partial datasets like fda_triples_v1).
    """
    gold = get_gold_adapter(gold_adapter_name)
    cases: list[GoldStandardCase] = []
    skipped: list[str] = []
    for r in responses:
        try:
            g = gold.get_gold(r.case_id)
        except KeyError:
            skipped.append(r.case_id)
            continue
        cases.append(
            GoldStandardCase(
                case_id=r.case_id,
                name=f"{r.case_id} (gold: {g.source})",
                task_type=TaskType.VARIANT_TO_STRATEGY,
                question=f"(adapter-driven; no canonical question stored for case_id {r.case_id!r})",
                expected_target=g.expected_target,
                expected_target_aliases=list(g.expected_target_aliases),
                expected_modulation=g.expected_modulation,
                expected_mechanism_class=g.expected_mechanism_class,
                notes=(
                    f"Synthesized from {g.source}. raw_keys={sorted(g.raw)}"
                ),
            )
        )
    return cases, skipped


def _build_judge(provider: str, model: str | None) -> BaseJudge:
    if provider == "anthropic":
        return AnthropicJudge(model=model or os.environ.get("BIO_RAG_EVAL_JUDGE_MODEL", "claude-opus-4-7"))
    if provider == "openai":
        return OpenAIJudge(model=model or os.environ.get("BIO_RAG_EVAL_JUDGE_MODEL", "gpt-4.1"))
    if provider == "mock":
        return _build_demo_mock_judge()
    raise ValueError(f"unknown judge provider: {provider}")


def _build_demo_mock_judge() -> MockJudge:
    """Deterministic mock for offline demos. The mock returns
    intentionally varied labels so the resulting scorecard exercises
    the report's failure-mode visualizations — three SUPPORTED then
    one PARTIALLY_SUPPORTED then one UNSUPPORTED, cycling. This is
    NOT a substitute for a real judge run.
    """
    return MockJudge(
        {
            ClaimList: ClaimList(
                claims=[
                    Claim(claim_id="c1", text="mock placeholder claim 1", claim_type=ClaimType.FACTUAL, cited_ids=["e1"]),
                ]
            ),
            GroundingJudgment: [
                GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="mock supported placeholder", confidence=0.9),
                GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="mock supported placeholder", confidence=0.9),
                GroundingJudgment(claim_id="c1", label=GroundingLabel.SUPPORTED, rationale="mock supported placeholder", confidence=0.9),
                GroundingJudgment(claim_id="c1", label=GroundingLabel.PARTIALLY_SUPPORTED, rationale="mock partial placeholder", confidence=0.7),
                GroundingJudgment(claim_id="c1", label=GroundingLabel.UNSUPPORTED, rationale="mock unsupported placeholder", confidence=0.6),
            ],
            MechanismJudgment: MechanismJudgment(score=4, rationale="mock mechanism placeholder; coherent chain"),
            EvidenceSufficiency: EvidenceSufficiency(
                score=3,
                n_independent_sources=2,
                has_primary_literature=True,
                has_human_data=True,
                rationale="mock evidence placeholder",
            ),
            CompletenessJudgment: [
                CompletenessJudgment(facet="(echoed)", covered=True, rationale="mock completeness placeholder"),
            ],
        }
    )


def _write_full_report(report: Any, out_dir: Path) -> dict[str, Path]:
    """Write the adapter-mode output layout (scorecard.* + per_case/*)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = out_dir / "per_case"
    per_case_dir.mkdir(exist_ok=True)

    paths = {
        "html": out_dir / "scorecard.html",
        "md": out_dir / "scorecard.md",
        "json": out_dir / "raw_metrics.json",
        "metadata": out_dir / "run_metadata.json",
    }
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    # raw_metrics.json is the aggregate side only — for dashboards.
    aggregate_payload = {
        "metadata": report.metadata.model_dump(mode="json"),
        "aggregate_metrics": {k: v.model_dump() for k, v in report.aggregate_metrics.items()},
        "bias_consistency": {k: v.model_dump() for k, v in report.bias_consistency.items()},
    }
    paths["json"].write_text(json.dumps(aggregate_payload, indent=2, default=str), encoding="utf-8")
    paths["metadata"].write_text(report.metadata.model_dump_json(indent=2), encoding="utf-8")

    for cr in report.case_results:
        # Sanitize case_id for filesystem use — case_ids may contain `>`, `:`,
        # `/`, etc. on cross-platform inputs (e.g. `LDLR__c.661C>T`).
        safe = _safe_filename(cr.case_id)
        (per_case_dir / f"{safe}.json").write_text(
            cr.model_dump_json(indent=2), encoding="utf-8"
        )
    return paths


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Replace characters that are illegal in Windows filenames with `_`."""
    cleaned = _UNSAFE_CHARS.sub("_", name)
    return cleaned.strip(". ")  # trailing dot/space also disallowed on Windows


# ── main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("bio-rag-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-prompts", help="Print prompt registry with versions.")
    sub.add_parser("list-adapters", help="Print available predictions/gold adapters.")

    run_p = sub.add_parser("run", help="Run an evaluation (raw or adapter mode).")

    # Raw mode flags
    raw = run_p.add_argument_group("raw mode (alternative to adapter mode)")
    raw.add_argument("--cases", help="YAML file or directory of gold-standard cases.")
    raw.add_argument("--responses", help="JSONL of AgentResponse records.")

    # Adapter mode flags
    adp = run_p.add_argument_group("adapter mode")
    adp.add_argument("--predictions", help="JSONL of source-format predictions (e.g. therapy-agent results).")
    adp.add_argument(
        "--predictions-adapter",
        help="Name of a registered predictions adapter (e.g. 'therapy_agent_v1').",
    )
    adp.add_argument(
        "--gold-source",
        help="Name of a registered gold-standard adapter (e.g. 'fda_triples_v1').",
    )

    # Shared
    run_p.add_argument("--out", help="Output directory (raw mode default: 'reports').")
    run_p.add_argument("--output", help="Alias of --out for adapter mode.")
    run_p.add_argument(
        "--judge",
        choices=("anthropic", "openai", "mock"),
        default="anthropic",
        help=(
            "Judge provider. 'mock' uses a deterministic offline mock for demos "
            "and CI smoke-tests; numbers from a mock run are NOT informative."
        ),
    )
    run_p.add_argument("--model", default=None, help="Judge model id (overrides BIO_RAG_EVAL_JUDGE_MODEL env).")
    run_p.add_argument("--judge-model-version", default=None,
                       help="Provider-reported model version to record in metadata.")
    run_p.add_argument("--no-bias-check", action="store_true",
                       help="Skip the swapped-rubric bias re-run (halves judge calls).")
    run_p.add_argument("--citation-online", action="store_true",
                       help="Hit DOI/NCBI/clinicaltrials.gov instead of offline shape check.")
    run_p.add_argument("--seed", type=int, default=7, help="Random seed for the bootstrap.")

    args = parser.parse_args(argv)

    if args.cmd == "list-prompts":
        for name, version in sorted(list_prompts().items()):
            print(f"{name}\t{version}")
        return 0

    if args.cmd == "list-adapters":
        for kind, names in list_adapters().items():
            for n in names:
                print(f"{kind}\t{n}")
        return 0

    if args.cmd == "run":
        return _cmd_run(args)

    parser.error("unknown command")
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    adapter_mode = bool(args.predictions or args.predictions_adapter or args.gold_source)
    raw_mode = bool(args.cases or args.responses)

    if adapter_mode and raw_mode:
        sys.stderr.write(
            "Use either raw mode (--cases + --responses) or adapter mode "
            "(--predictions + --predictions-adapter [+ --gold-source]); not both.\n"
        )
        return 2

    out_dir = Path(args.output or args.out or ("report" if adapter_mode else "reports"))

    if adapter_mode:
        if not (args.predictions and args.predictions_adapter):
            sys.stderr.write("adapter mode requires --predictions AND --predictions-adapter\n")
            return 2
        responses = _load_predictions_via_adapter(Path(args.predictions), args.predictions_adapter)
        if args.gold_source:
            cases, skipped = _synthesize_cases_from_gold(responses, args.gold_source)
        else:
            sys.stderr.write(
                "adapter mode without --gold-source is allowed only if you also pass --cases. "
                "Failing closed: no gold source provided.\n"
            )
            return 2
        if skipped:
            sys.stderr.write(
                f"WARNING: gold source {args.gold_source!r} had no answer for "
                f"{len(skipped)} case(s): {skipped}. They are dropped from the run.\n"
            )
        if not cases:
            sys.stderr.write("No cases survived gold lookup; aborting.\n")
            return 3
        # Filter responses to only the cases we have gold for.
        keep = {c.case_id for c in cases}
        responses = [r for r in responses if r.case_id in keep]
    else:
        if not (args.cases and args.responses):
            sys.stderr.write("raw mode requires --cases AND --responses\n")
            return 2
        cases_path = Path(args.cases)
        if cases_path.is_dir():
            yaml_files = sorted(str(p) for p in cases_path.glob("*.yaml"))
        else:
            yaml_files = [str(cases_path)]
        cases = load_cases_from_yaml(yaml_files)
        responses = _load_responses_jsonl(Path(args.responses))

    judge = _build_judge(args.judge, args.model)
    config = RunConfig(
        run_bias_check=not args.no_bias_check,
        citation_offline=not args.citation_online,
        seed=args.seed,
        predictions_adapter_name=args.predictions_adapter,
        gold_source_name=args.gold_source,
        judge_model_version=args.judge_model_version,
    )
    runner = EvalRunner(judge=judge, config=config)
    report = runner.run(cases, responses)

    if adapter_mode:
        paths = _write_full_report(report, out_dir)
    else:
        from bio_rag_eval.report import write_report

        paths = write_report(report, out_dir)

    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
