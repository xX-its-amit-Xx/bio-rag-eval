"""Inputs to the eval pipeline: gold-standard cases and agent responses.

A `GoldStandardCase` is the curated rubric a human expert wrote. An
`AgentResponse` is what the system under test produced for that case. The
runner pairs them by `case_id`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Eval task families. Different task types route to different
    `task_success` scorers (see `metrics.task_success`)."""

    VARIANT_TO_STRATEGY = "variant_to_strategy"
    LITERATURE_QA = "literature_qa"
    MECHANISM_EXPLANATION = "mechanism_explanation"
    DRUG_REPURPOSING = "drug_repurposing"
    CLINICAL_TRIAL_MATCH = "clinical_trial_match"


class ModulationDirection(str, Enum):
    """Direction of therapeutic intervention on a target. `agnostic` is used
    when the gold standard accepts either direction (rare but real, e.g.
    correcting a misfolded protein where stabilization OR clearance both
    work)."""

    INHIBIT = "inhibit"
    ACTIVATE = "activate"
    REPLACE = "replace"
    SILENCE = "silence"
    EDIT = "edit"
    AGNOSTIC = "agnostic"


class Citation(BaseModel):
    """A single citation emitted by the agent.

    The agent is expected to attach citations to specific sentences/claims in
    its output. We do not require any particular ID scheme — DOI, PMID,
    PMCID, clinicaltrials.gov NCT, or a free-text "title — journal — year"
    string are all accepted. The citation resolver in `metrics.citation`
    tries each scheme in order.
    """

    model_config = {"extra": "forbid"}

    citation_id: str = Field(description="Stable ID used to reference this citation from claims")
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    nct_id: str | None = None
    url: str | None = None
    title: str | None = None
    snippet: str | None = Field(
        default=None,
        description=(
            "The exact passage the agent retrieved. Grounding judges score "
            "claims against this text — without a snippet, grounding falls "
            "back to title-only matching and accuracy degrades."
        ),
    )

    @field_validator("citation_id")
    @classmethod
    def _strip_id(cls, v: str) -> str:
        return v.strip()


class AgentResponse(BaseModel):
    """Output of the system under test for a single case.

    `answer` is the prose the agent produced. `citations` is the list of
    sources it claims to be drawing on. `claim_to_citations` is an optional
    mapping — if the agent already attributed sentence-level claims (good
    agents do this), the evaluator uses that mapping directly instead of
    inferring with a claim extractor.
    """

    model_config = {"extra": "forbid"}

    case_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    claim_to_citations: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Map of claim_text -> list of citation_id. If provided, "
            "grounding skips the claim-extraction LLM call. Use this when "
            "the agent already produces sentence-level attribution."
        ),
    )
    structured_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-specific structured fields the agent emitted (e.g. "
            "predicted_target, predicted_modulation). Consumed by "
            "task_success scorers."
        ),
    )
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class GoldStandardCase(BaseModel):
    """Human-curated rubric for a single eval case.

    `expected_facets` is the list of factual atoms an ideal answer should
    cover; completeness scoring counts how many were touched. `min_confidence`
    is a floor used by the agent's own self-reported confidence, not by the
    judge — keeping it on the case lets us flag overconfident wrong answers.
    """

    model_config = {"extra": "forbid"}

    case_id: str
    name: str
    task_type: TaskType
    category: str = Field(default="curated", description="curated | held_out | adversarial")

    question: str = Field(description="The prompt that will be passed to the agent")
    context: dict[str, Any] = Field(default_factory=dict)

    expected_target: str | None = None
    expected_target_aliases: list[str] = Field(default_factory=list)
    expected_modulation: ModulationDirection | None = None
    expected_mechanism_class: str | None = None

    expected_facets: list[str] = Field(
        default_factory=list,
        description="Atomic facts a complete answer should cover (for completeness scoring)",
    )
    required_citations: list[str] = Field(
        default_factory=list,
        description=(
            "Substrings that MUST appear in the cited evidence (e.g. drug "
            "names, trial acronyms). Surfaced by the grounding/citation "
            "metrics as 'required_citation_recall'."
        ),
    )
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims that, if asserted, indicate a known failure mode "
            "(e.g. 'gene therapy is FDA-approved for adults with DMD' — "
            "currently false). Used by hallucination metric."
        ),
    )

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    references: list[str] = Field(default_factory=list)
    notes: str | None = None
