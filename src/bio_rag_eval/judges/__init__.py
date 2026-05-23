"""LLM-as-judge backends. All judges share the `BaseJudge` interface so
metrics code does not depend on which provider produced a judgment."""
from __future__ import annotations

from bio_rag_eval.judges.base import BaseJudge, JudgeResponse, MockJudge
from bio_rag_eval.judges.claim_extractor import ClaimExtractor
from bio_rag_eval.judges.judge_anthropic import AnthropicJudge
from bio_rag_eval.judges.judge_openai import OpenAIJudge

__all__ = [
    "AnthropicJudge",
    "BaseJudge",
    "ClaimExtractor",
    "JudgeResponse",
    "MockJudge",
    "OpenAIJudge",
]
