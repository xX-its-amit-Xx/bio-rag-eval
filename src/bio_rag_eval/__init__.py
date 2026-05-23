"""bio-rag-eval — evaluation harness for biomedical RAG and agentic systems.

Top-level re-exports of the public surface. Internal modules should be
imported by full dotted path; the names exported here are the stable API.
"""
from __future__ import annotations

from bio_rag_eval._version import __version__
from bio_rag_eval.runner import EvalRunner, RunConfig
from bio_rag_eval.schemas.case import GoldStandardCase, AgentResponse, Citation
from bio_rag_eval.schemas.eval_result import (
    CaseResult,
    EvalReport,
    MetricValue,
)

__all__ = [
    "EvalRunner",
    "RunConfig",
    "GoldStandardCase",
    "AgentResponse",
    "Citation",
    "CaseResult",
    "EvalReport",
    "MetricValue",
    "__version__",
]
