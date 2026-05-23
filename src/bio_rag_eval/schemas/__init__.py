"""Pydantic schemas — every LLM-as-judge call uses one of these as its
structured-output target. No free-text parsing is performed downstream.
"""
from __future__ import annotations

from bio_rag_eval.schemas.adapter_types import GoldStandard, TaskAnswer
from bio_rag_eval.schemas.case import (
    AgentResponse,
    Citation,
    GoldStandardCase,
    ModulationDirection,
    TaskType,
)
from bio_rag_eval.schemas.claims import (
    Claim,
    ClaimList,
    ClaimType,
)
from bio_rag_eval.schemas.judgments import (
    CompletenessJudgment,
    EvidenceSufficiency,
    GroundingJudgment,
    GroundingLabel,
    MechanismJudgment,
)
from bio_rag_eval.schemas.eval_result import (
    CaseResult,
    EvalReport,
    MetricValue,
    RunMetadata,
)

__all__ = [
    "AgentResponse",
    "CaseResult",
    "Citation",
    "GoldStandard",
    "TaskAnswer",
    "Claim",
    "ClaimList",
    "ClaimType",
    "CompletenessJudgment",
    "EvalReport",
    "EvidenceSufficiency",
    "GoldStandardCase",
    "GroundingJudgment",
    "GroundingLabel",
    "MechanismJudgment",
    "MetricValue",
    "ModulationDirection",
    "RunMetadata",
    "TaskType",
]
