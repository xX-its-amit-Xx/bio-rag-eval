"""Claim schemas — the output of sentence-level claim extraction.

A "claim" is an atomic factual proposition. We deliberately separate claim
TYPES because a stylistic claim ("this is exciting") should not be penalized
for being uncited the way a factual one ("tofersen lowered neurofilament
light by 60%") would be.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Classifies a claim by the kind of support it requires.

    - FACTUAL: an empirical proposition. MUST be cited.
    - MECHANISTIC: a causal/mechanism statement. Should be cited unless it's
      trivially derivable from a cited factual claim.
    - QUANTITATIVE: a numeric claim (effect size, p-value, prevalence).
      MUST be cited and the number itself is checked against the source.
    - RECOMMENDATION: a clinical/research suggestion. Held to a higher bar
      — must be cited and must not contradict the source.
    - OPINION: hedged/stylistic. Not penalized for being uncited.
    """

    FACTUAL = "factual"
    MECHANISTIC = "mechanistic"
    QUANTITATIVE = "quantitative"
    RECOMMENDATION = "recommendation"
    OPINION = "opinion"


class Claim(BaseModel):
    """A single extracted claim.

    `cited_ids` is the list of citation_id values the agent (or the extractor,
    if the agent did not attribute) attached to this claim. Empty list means
    "uncited" — distinct from a list that contains an unresolvable ID.
    """

    model_config = {"extra": "forbid"}

    claim_id: str = Field(description="Stable per-response ID like 'c1', 'c2'")
    text: str = Field(min_length=3)
    claim_type: ClaimType
    cited_ids: list[str] = Field(default_factory=list)
    span: tuple[int, int] | None = Field(
        default=None,
        description="(start, end) character offsets in the original answer, if available",
    )


class ClaimList(BaseModel):
    """Container for claim-extractor output. The judge always returns this
    shape so we never have to parse a list-of-things out of free text."""

    model_config = {"extra": "forbid"}

    claims: list[Claim]
    extraction_notes: str | None = None
