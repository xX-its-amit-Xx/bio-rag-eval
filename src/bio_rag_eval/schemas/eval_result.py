"""Output schemas — what the runner writes to disk after a run.

Two layers: `CaseResult` is per-case detail (every claim, every judgment,
every metric on this one example). `EvalReport` is the aggregate across all
cases, with bootstrap CIs and run metadata sufficient to reproduce the run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from bio_rag_eval.schemas.claims import Claim
from bio_rag_eval.schemas.judgments import (
    CompletenessJudgment,
    EvidenceSufficiency,
    GroundingJudgment,
    MechanismJudgment,
)


class MetricValue(BaseModel):
    """A single metric with its bootstrap CI.

    `point` is the point estimate (the actual computed metric on the
    observed sample). `ci_low`/`ci_high` are the percentile-bootstrap
    bounds at `ci_level` (default 95%). `n` is the number of underlying
    units the metric was computed over (claims, facets, cases — depends on
    the metric).
    """

    model_config = {"extra": "forbid"}

    name: str
    point: float
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    n: int
    notes: str | None = None


class CaseResult(BaseModel):
    """Everything we computed about one case.

    The judgment lists are kept for auditability — when a metric looks
    surprising, you want to be able to inspect "which 3 claims were marked
    UNSUPPORTED?" without re-running the judges.
    """

    model_config = {"extra": "forbid"}

    case_id: str
    case_name: str
    extracted_claims: list[Claim] = Field(default_factory=list)
    grounding_judgments: list[GroundingJudgment] = Field(default_factory=list)
    mechanism_judgment: MechanismJudgment | None = None
    evidence_sufficiency: EvidenceSufficiency | None = None
    completeness_judgments: list[CompletenessJudgment] = Field(default_factory=list)

    metrics: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text warnings: 'bias_inconsistency', 'forbidden_claim_present', "
            "'unresolvable_required_citation', etc."
        ),
    )
    error: str | None = Field(
        default=None,
        description="If the case failed (agent crashed, judge timed out, etc.) — the rest of the fields may be partial.",
    )
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Reproducibility manifest. Persisted alongside results so a future
    re-run can be compared apples-to-apples."""

    model_config = {"extra": "forbid"}

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None

    judge_model: str
    judge_provider: str
    extractor_model: str
    extractor_provider: str

    prompt_versions: dict[str, str] = Field(
        description="prompt_name -> human-curated semver from frontmatter"
    )
    prompt_hashes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "prompt_name -> content-addressable hash (git:<sha1> or sha256:<hex>). "
            "This is what reproducibility actually pins against — the semver in "
            "`prompt_versions` is curator-facing and may be bumped manually."
        ),
    )
    judge_model_version: str | None = Field(
        default=None,
        description=(
            "Provider-reported model version, when available "
            "(e.g. 'claude-opus-4-7@20260101', 'gpt-4.1-2026-01-15'). "
            "Distinct from `judge_model` which is the requested ID."
        ),
    )
    bio_rag_eval_version: str
    seed: int | None = None
    random_seed: int | None = Field(
        default=None,
        description=(
            "Alias of `seed`, preserved for cross-team conventions. Always "
            "equals `seed` for runs produced by this package."
        ),
    )
    bootstrap_n_resamples: int = 1000
    ci_level: float = 0.95
    predictions_adapter: str | None = None
    gold_source: str | None = None

    config_hash: str = Field(description="Hash of the resolved RunConfig — for cache keys")

    @classmethod
    def empty(cls) -> "RunMetadata":  # convenience for tests/fixtures
        now = datetime.now(timezone.utc)
        return cls(
            run_id="",
            started_at=now,
            judge_model="",
            judge_provider="",
            extractor_model="",
            extractor_provider="",
            prompt_versions={},
            bio_rag_eval_version="0.0.0",
            config_hash="",
        )


class EvalReport(BaseModel):
    """Aggregate report across all cases in a run."""

    model_config = {"extra": "forbid"}

    metadata: RunMetadata
    case_results: list[CaseResult]
    aggregate_metrics: dict[str, MetricValue]
    bias_consistency: dict[str, MetricValue] = Field(
        default_factory=dict,
        description="Metrics computed under the swapped rubric. Differences indicate position bias.",
    )
