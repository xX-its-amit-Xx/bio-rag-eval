"""Metric modules. Each metric module exports a `compute_*` function plus
the per-claim/per-facet primitives the runner needs to assemble a
`CaseResult`. Aggregation across cases lives in `bio_rag_eval.runner`.
"""
from __future__ import annotations

from bio_rag_eval.metrics.bias import bias_consistency
from bio_rag_eval.metrics.citation import (
    CitationResolution,
    compute_citation_traceability,
    resolve_citation,
)
from bio_rag_eval.metrics.completeness import compute_completeness
from bio_rag_eval.metrics.grounding import (
    compute_grounding,
    judge_grounding,
)
from bio_rag_eval.metrics.hallucination import compute_hallucination
from bio_rag_eval.metrics.task_success import (
    compute_task_success,
    score_modulation,
    score_target,
)

__all__ = [
    "CitationResolution",
    "bias_consistency",
    "compute_citation_traceability",
    "compute_completeness",
    "compute_grounding",
    "compute_hallucination",
    "compute_task_success",
    "judge_grounding",
    "resolve_citation",
    "score_modulation",
    "score_target",
]
