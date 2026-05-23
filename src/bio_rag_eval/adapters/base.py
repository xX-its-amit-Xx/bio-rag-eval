"""Abstract base classes for adapters."""
from __future__ import annotations

import abc

from bio_rag_eval.schemas.adapter_types import GoldStandard, TaskAnswer
from bio_rag_eval.schemas.case import AgentResponse, Citation
from bio_rag_eval.schemas.claims import Claim


class PredictionsAdapter(abc.ABC):
    """Adapter for an agent's per-case output (the predictions side).

    Implementations parse a single record (typically one JSONL line) from
    the source system and split it into the three things bio-rag-eval
    cares about: atomic claims, citations, and a structured task answer.
    """

    #: Stable adapter name, set by subclasses (also used in the registry).
    NAME: str = ""
    #: Schema version of the *source* system this adapter targets — e.g.
    #: therapy-agent's TherapyStrategyOutput v1.0.
    SOURCE_SCHEMA_VERSION: str = ""

    @abc.abstractmethod
    def extract_claims(self, output: dict) -> list[Claim]:
        """Extract atomic claims (with attached citation IDs) from one record."""

    @abc.abstractmethod
    def extract_citations(self, output: dict) -> list[Citation]:
        """Extract the citation list from one record. citation_id must
        match the IDs returned by `extract_claims`."""

    @abc.abstractmethod
    def extract_task_answer(self, output: dict) -> TaskAnswer:
        """Extract the structured task answer (predicted target etc.) from one record."""

    def case_id_of(self, output: dict) -> str:
        """Return the case_id this record belongs to. Default looks for
        `case_id` at the top level; override for source schemas that
        nest it differently."""
        cid = output.get("case_id")
        if not cid:
            raise ValueError(
                f"{self.NAME}: record has no top-level case_id; override case_id_of() if needed"
            )
        return str(cid)

    def to_agent_response(self, output: dict) -> AgentResponse:
        """Assemble a full `AgentResponse` from one source record.

        Subclasses rarely need to override this; the default composes
        the three `extract_*` methods plus `case_id_of`.
        """
        case_id = self.case_id_of(output)
        citations = self.extract_citations(output)
        claims = self.extract_claims(output)
        task = self.extract_task_answer(output)

        # Materialize claim_to_citations so the runner can bypass the
        # claim extractor (the adapter is authoritative for attribution).
        claim_to_citations: dict[str, list[str]] = {}
        for c in claims:
            claim_to_citations[c.text] = list(c.cited_ids)

        structured: dict = {}
        if task.predicted_target:
            structured["predicted_target"] = task.predicted_target
        if task.predicted_target_aliases:
            structured["predicted_target_aliases"] = list(task.predicted_target_aliases)
        if task.predicted_modulation:
            structured["predicted_modulation"] = task.predicted_modulation.value
        if task.predicted_mechanism_class:
            structured["predicted_mechanism_class"] = task.predicted_mechanism_class
        if task.confidence is not None:
            structured["confidence"] = task.confidence
        if task.raw:
            structured["_adapter_raw"] = task.raw

        # Reconstruct the answer prose by joining the claim texts. For
        # adapters whose source actually emits prose, override
        # to_agent_response and pass the original prose through.
        answer = " ".join(c.text for c in claims) or "(no claim prose available)"

        return AgentResponse(
            case_id=case_id,
            answer=answer,
            citations=citations,
            claim_to_citations=claim_to_citations or None,
            structured_outputs=structured,
        )


class GoldStandardAdapter(abc.ABC):
    """Adapter for a gold-standard data source."""

    NAME: str = ""
    SOURCE_SCHEMA_VERSION: str = ""

    @abc.abstractmethod
    def get_gold(self, case_id: str) -> GoldStandard:
        """Return the gold answer for one case_id. Raise KeyError if absent."""

    def has(self, case_id: str) -> bool:
        try:
            self.get_gold(case_id)
        except KeyError:
            return False
        return True
