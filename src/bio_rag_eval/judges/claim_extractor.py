"""Sentence-level claim extraction.

If the agent already provided sentence -> citation attribution
(`AgentResponse.claim_to_citations`), the extractor is bypassed — that
attribution is authoritative and re-extracting risks mangling it. Only
when the agent provided prose-with-trailing-citations do we fire the
extraction prompt.
"""
from __future__ import annotations

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.prompts import load_prompt
from bio_rag_eval.schemas.case import AgentResponse
from bio_rag_eval.schemas.claims import Claim, ClaimList, ClaimType


class ClaimExtractor:
    """Pulls atomic claims out of an agent's prose answer."""

    PROMPT_NAME = "claim_extraction_v1"

    def __init__(self, judge: BaseJudge):
        self.judge = judge
        self.prompt = load_prompt(self.PROMPT_NAME)

    @property
    def prompt_version(self) -> str:
        return self.prompt.version

    def extract(self, response: AgentResponse) -> ClaimList:
        if response.claim_to_citations:
            return ClaimList(
                claims=[
                    Claim(
                        claim_id=f"c{i+1}",
                        text=text,
                        # We don't know the type without the LLM; default to factual.
                        claim_type=ClaimType.FACTUAL,
                        cited_ids=list(cite_ids),
                    )
                    for i, (text, cite_ids) in enumerate(response.claim_to_citations.items())
                ],
                extraction_notes="Source: agent-provided claim_to_citations (extractor bypassed).",
            )

        rendered = self.prompt.render(
            answer=response.answer,
            citation_ids=[c.citation_id for c in response.citations],
        )
        result = self.judge.judge_json(rendered, ClaimList)
        assert isinstance(result.parsed, ClaimList)
        return result.parsed
