"""Adapter-facing schemas.

Adapters translate between external agent/dataset shapes and bio-rag-eval's
internal types. They produce three small, schema-typed values per case:
extracted claims, citations, and a `TaskAnswer` (the agent's structured
prediction). For gold-side adapters they produce a `GoldStandard` per
case_id. These types are deliberately thin — the runner combines them
into the existing `AgentResponse` and `GoldStandardCase` objects.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from bio_rag_eval.schemas.case import ModulationDirection


class TaskAnswer(BaseModel):
    """The agent's structured prediction for a single case.

    Adapters populate this from whatever fields their source schema uses
    (e.g. `target_protein.name` + `modulation_type` for therapy-agent).
    The runner copies these into `AgentResponse.structured_outputs` for
    task_success scoring.
    """

    model_config = {"extra": "forbid"}

    predicted_target: Optional[str] = None
    predicted_target_aliases: list[str] = Field(default_factory=list)
    predicted_modulation: Optional[ModulationDirection] = None
    predicted_mechanism_class: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw: dict = Field(
        default_factory=dict,
        description="Adapter-specific extras preserved for the report (model name, latency, etc.)",
    )


class GoldStandard(BaseModel):
    """The gold-standard answer for a single case, as produced by a
    gold-source adapter. The runner merges this onto a base
    `GoldStandardCase` so the case can carry richer rubric fields
    (facets, forbidden claims) that the gold source itself does not
    know about.
    """

    model_config = {"extra": "forbid"}

    case_id: str
    expected_target: Optional[str] = None
    expected_target_aliases: list[str] = Field(default_factory=list)
    expected_modulation: Optional[ModulationDirection] = None
    expected_mechanism_class: Optional[str] = None
    source: str = Field(
        description="Adapter name that produced this gold standard, for provenance.",
    )
    raw: dict = Field(default_factory=dict)
