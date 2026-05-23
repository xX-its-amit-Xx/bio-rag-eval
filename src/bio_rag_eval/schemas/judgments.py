"""Schemas for every LLM-as-judge call.

Each call returns one of these. They are the only contract between prompts
and downstream metric code — change the prompt template freely, but if you
change a schema, bump the corresponding `prompt_version` and add a
migration note in the prompt file's frontmatter.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GroundingLabel(str, Enum):
    """Four-way label for "does the cited source support this claim?".

    SUPPORTED: cited passage explicitly entails the claim.
    PARTIALLY_SUPPORTED: cited passage entails part of the claim but omits
        a quantitative qualifier or scope condition (e.g. claim says "in
        adults"; source studied only adolescents).
    CONTRADICTED: cited passage states the opposite or asserts a value the
        claim's number conflicts with.
    UNSUPPORTED: cited passage is topically related but does not address the
        specific claim (the most common silent failure of RAG agents).
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class GroundingJudgment(BaseModel):
    """Per-claim grounding decision."""

    model_config = {"extra": "forbid"}

    claim_id: str
    label: GroundingLabel
    rationale: str = Field(
        description="One- or two-sentence justification. Required even when SUPPORTED.",
        min_length=10,
    )
    quoted_evidence: str | None = Field(
        default=None,
        description="Exact substring of the cited snippet that justifies the label, if any.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class MechanismJudgment(BaseModel):
    """Likert-5 mechanism coherence judgment for `task_success`.

    "Coherence" here is narrower than "correctness": a coherent mechanism is
    one where each step follows from the previous and the overall causal
    chain connects the variant to the proposed intervention. A 5/5 mechanism
    can still be factually wrong (which `grounding` will catch).
    """

    model_config = {"extra": "forbid"}

    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=10)
    missing_steps: list[str] = Field(
        default_factory=list,
        description="Causal steps the answer should have included but did not.",
    )
    incorrect_steps: list[str] = Field(default_factory=list)


class EvidenceSufficiency(BaseModel):
    """The second of the two reconciled rubrics. Scores whether the cited
    EVIDENCE (not the answer prose) would convince a domain expert — the
    answer might be plausible but built on a single anecdote, which this
    rubric flags."""

    model_config = {"extra": "forbid"}

    score: int = Field(ge=1, le=5)
    n_independent_sources: int = Field(ge=0)
    has_primary_literature: bool
    has_human_data: bool = Field(
        description="True iff at least one cited source reports findings in humans."
    )
    rationale: str = Field(min_length=10)


class CompletenessJudgment(BaseModel):
    """Per-facet coverage decision. The completeness metric aggregates over
    facets; each call here scores ONE facet against the answer."""

    model_config = {"extra": "forbid"}

    facet: str
    covered: bool
    partial: bool = Field(
        default=False,
        description="True iff the facet is mentioned but treated superficially.",
    )
    rationale: str = Field(min_length=5)
    quoted_evidence: str | None = None
