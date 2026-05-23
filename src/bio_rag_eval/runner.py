"""Eval pipeline orchestrator.

`EvalRunner.run(cases, responses)` produces an `EvalReport`. The runner
owns three things metric modules deliberately don't:

  1. Per-case orchestration: for each (case, response) pair, call the
     claim extractor, then the grounding judge for each claim, then
     mechanism / evidence / completeness judges, then compose metrics.
  2. Aggregation: stack per-case metrics into bootstrap CIs.
  3. Reproducibility metadata: stamp prompt versions, judge models, and
     a config hash into the report so a future re-run is comparable.

This module deliberately knows about every metric — that's the price of
having all the cross-metric error handling in one place. Adding a new
metric means adding a call site here.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from bio_rag_eval._version import __version__
from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.judges.claim_extractor import ClaimExtractor
from bio_rag_eval.metrics import (
    bias_consistency,
    compute_citation_traceability,
    compute_completeness,
    compute_grounding,
    compute_hallucination,
    compute_task_success,
    judge_grounding,
)
from bio_rag_eval.metrics.completeness import judge_facet
from bio_rag_eval.metrics.grounding import (
    grounding_per_claim_supported,
    index_citations,
)
from bio_rag_eval.metrics.task_success import judge_mechanism
from bio_rag_eval.prompts import list_prompts, prompt_content_hashes
from bio_rag_eval.schemas.case import AgentResponse, GoldStandardCase, TaskType
from bio_rag_eval.schemas.claims import Claim
from bio_rag_eval.schemas.eval_result import (
    CaseResult,
    EvalReport,
    MetricValue,
    RunMetadata,
)
from bio_rag_eval.schemas.judgments import EvidenceSufficiency, GroundingJudgment
from bio_rag_eval.utils import bootstrap_ci, hash_config

log = logging.getLogger(__name__)


class RunConfig(BaseModel):
    """Knobs that affect the run but not the agent under test."""

    model_config = {"extra": "forbid"}

    run_bias_check: bool = True
    run_citation_check: bool = True
    citation_offline: bool = Field(
        default=True,
        description=(
            "If true, citation resolver does shape-only checks (regex). "
            "Set to false in production runs to actually hit doi.org / NCBI / clinicaltrials.gov."
        ),
    )
    bootstrap_n_resamples: int = 1000
    ci_level: float = 0.95
    seed: int | None = 7
    evidence_sufficiency_enabled: bool = True
    max_claims_per_case: int = 50  # safety cap; runaway claim lists are a model failure
    predictions_adapter_name: str | None = Field(
        default=None,
        description="Recorded in RunMetadata for provenance. Set by the CLI from --predictions-adapter.",
    )
    gold_source_name: str | None = Field(
        default=None,
        description="Recorded in RunMetadata for provenance. Set by the CLI from --gold-source.",
    )
    judge_model_version: str | None = Field(
        default=None,
        description=(
            "Provider-reported model version, when known at construction time. "
            "Otherwise the runner uses the judge's `.model` attribute."
        ),
    )


class EvalRunner:
    """Drives the eval pipeline."""

    def __init__(
        self,
        judge: BaseJudge,
        extractor: BaseJudge | None = None,
        config: RunConfig | None = None,
    ):
        """
        Args:
            judge: judge used for grounding, mechanism, evidence sufficiency,
                completeness, and the bias re-run.
            extractor: judge used for claim extraction. Defaults to `judge`.
                Splitting it out lets you use a cheaper model for extraction
                and a stronger one for judging — recommended.
            config: knobs.
        """
        self.judge = judge
        self.extractor_judge = extractor or judge
        self.config = config or RunConfig()
        self.claim_extractor = ClaimExtractor(self.extractor_judge)

    def run(
        self,
        cases: Iterable[GoldStandardCase],
        responses: Iterable[AgentResponse],
    ) -> EvalReport:
        cases_list = list(cases)
        responses_by_id = {r.case_id: r for r in responses}

        started = datetime.now(timezone.utc)
        run_id = uuid.uuid4().hex[:12]
        log.info("starting eval run %s with %d cases", run_id, len(cases_list))

        case_results: list[CaseResult] = []
        # Per-claim "supported?" vector across the entire run — for bootstrap.
        all_supported_indicators: list[int] = []
        # Per-case scalar series for the case-level metrics.
        per_case: dict[str, list[float]] = {
            "hallucination_rate": [],
            "uncited_rate": [],
            "miscitation_rate": [],
            "forbidden_claim_present": [],
            "citation_traceability": [],
            "citation_traceability_offline_check": [],
            "required_citation_recall": [],
            "answer_completeness": [],
            "correct_target": [],
            "correct_modulation_direction": [],
            "mechanism_coherence": [],
            "task_success_composite": [],
            "evidence_sufficiency_score": [],
            "bias_consistency_rate": [],
        }

        for case in cases_list:
            response = responses_by_id.get(case.case_id)
            if response is None:
                cr = CaseResult(
                    case_id=case.case_id,
                    case_name=case.name,
                    error=f"no agent response found for case_id={case.case_id}",
                )
                case_results.append(cr)
                continue
            try:
                cr = self._run_one(case, response, all_supported_indicators)
            except Exception as e:
                log.exception("case %s failed: %s", case.case_id, e)
                cr = CaseResult(case_id=case.case_id, case_name=case.name, error=str(e))
            case_results.append(cr)
            for k in per_case:
                if k in cr.metrics:
                    per_case[k].append(cr.metrics[k])

        # Aggregate
        aggregate: dict[str, MetricValue] = {}

        # Grounding aggregates over CLAIMS (not cases), because rates of
        # supported claims are most meaningful at claim resolution.
        if all_supported_indicators:
            point, lo, hi = bootstrap_ci(
                all_supported_indicators,
                n_resamples=self.config.bootstrap_n_resamples,
                ci_level=self.config.ci_level,
                seed=self.config.seed,
            )
            aggregate["grounding_rate"] = MetricValue(
                name="grounding_rate",
                point=point,
                ci_low=lo,
                ci_high=hi,
                ci_level=self.config.ci_level,
                n=len(all_supported_indicators),
                notes="Mean over graded claims (not cases). Bootstrap over claims.",
            )

        # Case-level metrics
        for name, series in per_case.items():
            if not series:
                continue
            # Drop NaNs — they mean "N/A for this case".
            cleaned = [v for v in series if v == v]  # NaN != NaN
            if not cleaned:
                continue
            point, lo, hi = bootstrap_ci(
                cleaned,
                n_resamples=self.config.bootstrap_n_resamples,
                ci_level=self.config.ci_level,
                seed=self.config.seed,
            )
            aggregate[name] = MetricValue(
                name=name,
                point=point,
                ci_low=lo,
                ci_high=hi,
                ci_level=self.config.ci_level,
                n=len(cleaned),
            )

        meta = RunMetadata(
            run_id=run_id,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            judge_model=self.judge.model,
            judge_provider=self.judge.provider,
            extractor_model=self.extractor_judge.model,
            extractor_provider=self.extractor_judge.provider,
            prompt_versions=list_prompts(),
            prompt_hashes=prompt_content_hashes(),
            judge_model_version=self.config.judge_model_version or self.judge.model,
            bio_rag_eval_version=__version__,
            seed=self.config.seed,
            random_seed=self.config.seed,
            bootstrap_n_resamples=self.config.bootstrap_n_resamples,
            ci_level=self.config.ci_level,
            predictions_adapter=self.config.predictions_adapter_name,
            gold_source=self.config.gold_source_name,
            config_hash=hash_config(self.config),
        )

        # Bias consistency (already per-case in case_results.metrics, but we
        # also surface it as its own aggregate for the report's bias panel).
        bias_block: dict[str, MetricValue] = {}
        if self.config.run_bias_check:
            for key in ("bias_consistency_rate", "bias_swing_to_supported", "bias_swing_to_unsupported"):
                series = [cr.metrics[key] for cr in case_results if key in cr.metrics]
                if not series:
                    continue
                point, lo, hi = bootstrap_ci(
                    series,
                    n_resamples=self.config.bootstrap_n_resamples,
                    ci_level=self.config.ci_level,
                    seed=self.config.seed,
                )
                bias_block[key] = MetricValue(
                    name=key, point=point, ci_low=lo, ci_high=hi, n=len(series)
                )

        return EvalReport(
            metadata=meta,
            case_results=case_results,
            aggregate_metrics=aggregate,
            bias_consistency=bias_block,
        )

    def _run_one(
        self,
        case: GoldStandardCase,
        response: AgentResponse,
        all_supported_indicators: list[int],
    ) -> CaseResult:
        claim_list = self.claim_extractor.extract(response)
        claims: list[Claim] = claim_list.claims[: self.config.max_claims_per_case]
        citations_by_id = index_citations(response)

        # Grounding (primary rubric order).
        primary_judgments: list[GroundingJudgment] = []
        for c in claims:
            try:
                primary_judgments.append(
                    judge_grounding(c, citations_by_id, self.judge, rubric_order="supported_first")
                )
            except Exception as e:
                log.warning("grounding judge failed on %s/%s: %s", case.case_id, c.claim_id, e)

        grounding_metrics = compute_grounding(claims, primary_judgments)
        hall_metrics = compute_hallucination(claims, primary_judgments, case)

        # Citations
        cite_metrics: dict[str, float] = {}
        if self.config.run_citation_check:
            cite_metrics, _resolutions = compute_citation_traceability(
                response.citations,
                case=case,
                offline=self.config.citation_offline,
            )

        # Task success (mechanism judge only for V2S-style tasks)
        mech: Any = None
        if case.task_type in (
            TaskType.VARIANT_TO_STRATEGY,
            TaskType.MECHANISM_EXPLANATION,
        ):
            try:
                mech = judge_mechanism(case, response, self.judge)
            except Exception as e:
                log.warning("mechanism judge failed on %s: %s", case.case_id, e)
        task_metrics = compute_task_success(case, response, mech)

        # Completeness
        completeness_judgments = []
        for facet in case.expected_facets:
            try:
                completeness_judgments.append(judge_facet(facet, response, self.judge))
            except Exception as e:
                log.warning("completeness judge failed on %s/%r: %s", case.case_id, facet, e)
        comp_metrics = compute_completeness(case, completeness_judgments)

        # Evidence sufficiency (one call per case, optional)
        evidence: EvidenceSufficiency | None = None
        if self.config.evidence_sufficiency_enabled and response.citations:
            evidence = self._judge_evidence(case, response)

        # Bias re-run
        bias_metrics: dict[str, float] = {}
        if self.config.run_bias_check and primary_judgments:
            bias_metrics, _swapped = bias_consistency(
                claims, citations_by_id, self.judge, primary_judgments
            )

        # Compose case metrics
        merged: dict[str, float] = {}
        for d in (grounding_metrics, hall_metrics, cite_metrics, task_metrics, comp_metrics, bias_metrics):
            merged.update(d)
        if evidence is not None:
            merged["evidence_sufficiency_score"] = float(evidence.score)
            merged["evidence_n_independent_sources"] = float(evidence.n_independent_sources)
            merged["evidence_has_human_data"] = 1.0 if evidence.has_human_data else 0.0

        # Build flags
        flags: list[str] = []
        if merged.get("forbidden_claim_present", 0.0) > 0.0:
            flags.append("forbidden_claim_present")
        if (
            merged.get("required_citation_recall") is not None
            and merged.get("required_citation_recall", 1.0) < 1.0
        ):
            flags.append("required_citation_recall_below_1")
        if (
            merged.get("bias_consistency_rate") is not None
            and merged.get("bias_consistency_rate", 1.0) < 0.7
        ):
            flags.append("bias_unstable")

        # Feed into run-level claim indicator vector for bootstrap.
        all_supported_indicators.extend(
            grounding_per_claim_supported(claims, primary_judgments)
        )

        return CaseResult(
            case_id=case.case_id,
            case_name=case.name,
            extracted_claims=claims,
            grounding_judgments=primary_judgments,
            mechanism_judgment=mech,
            evidence_sufficiency=evidence,
            completeness_judgments=completeness_judgments,
            metrics=merged,
            flags=flags,
        )

    def _judge_evidence(
        self,
        case: GoldStandardCase,
        response: AgentResponse,
    ) -> EvidenceSufficiency | None:
        from bio_rag_eval.prompts import load_prompt
        prompt = load_prompt("evidence_sufficiency_v1")
        rendered = prompt.render(
            answer_summary=response.answer[:1500],
            citations=[c.model_dump() for c in response.citations],
        )
        try:
            result = self.judge.judge_json(rendered, EvidenceSufficiency)
            ev = result.parsed
            assert isinstance(ev, EvidenceSufficiency)
            return ev
        except Exception as e:
            log.warning("evidence sufficiency judge failed on %s: %s", case.case_id, e)
            return None


def load_cases_from_yaml(paths: Sequence[str]) -> list[GoldStandardCase]:
    """Load gold-standard cases from YAML files. Supports either the
    examples/sample_gold_standards/*.yaml shape (one case per file) or
    a list-of-cases file."""
    import yaml  # local import — yaml is optional

    cases: list[GoldStandardCase] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            cases.extend(GoldStandardCase.model_validate(d) for d in data)
        else:
            cases.append(GoldStandardCase.model_validate(data))
    return cases
