"""Adapter for therapy-agent's `TherapyStrategyOutput` v1.0 schema.

Maps therapy-agent's per-case output to bio-rag-eval's internal types:

    TherapyStrategyOutput.supporting_evidence  -> Citation[] + Claim[]
    TherapyStrategyOutput.target_protein.name  -> TaskAnswer.predicted_target
    TherapyStrategyOutput.modulation_type      -> TaskAnswer.predicted_modulation
    TherapyStrategyOutput.precedent_drugs      -> TaskAnswer.predicted_target_aliases
                                                  (drug names — used for alias matching)

Pinned to schema_version "1.0" (therapy_agent.config.BENCHMARK_SCHEMA_VERSION).
When therapy-agent ships a 2.x schema, register a `therapy_agent_v2`
adapter rather than mutating this one — that preserves replay of older
benchmark runs.
"""
from __future__ import annotations

import re

from bio_rag_eval.adapters.base import PredictionsAdapter
from bio_rag_eval.adapters import register_adapter
from bio_rag_eval.schemas.adapter_types import TaskAnswer
from bio_rag_eval.schemas.case import AgentResponse, Citation, ModulationDirection
from bio_rag_eval.schemas.claims import Claim, ClaimType

# Map therapy-agent's canonical ModulationType (Literal[...]) -> our enum.
# Several therapy-agent values fold to the same bio-rag-eval direction:
# "ASO" is conceptually a silencing mechanism unless paired with a splice
# modulator context — therapy-agent disambiguates via target_pathway, but
# the prediction side treats them as silencing for direction-matching.
_MODULATION_MAP: dict[str, ModulationDirection] = {
    "inhibitor": ModulationDirection.INHIBIT,
    "agonist": ModulationDirection.ACTIVATE,
    "chaperone": ModulationDirection.ACTIVATE,
    "modulator": ModulationDirection.ACTIVATE,
    "ASO": ModulationDirection.SILENCE,
    "siRNA": ModulationDirection.SILENCE,
    "gene_therapy": ModulationDirection.REPLACE,
    "enzyme_replacement": ModulationDirection.REPLACE,
    "monoclonal_antibody": ModulationDirection.INHIBIT,
    "other": ModulationDirection.AGNOSTIC,
}


@register_adapter("therapy_agent_v1")
class TherapyAgentAdapter(PredictionsAdapter):
    """Adapter for therapy-agent TherapyStrategyOutput v1.0 schema."""

    NAME = "therapy_agent_v1"
    SOURCE_SCHEMA_VERSION = "1.0"

    def case_id_of(self, output: dict) -> str:
        cid = output.get("case_id")
        if cid:
            return str(cid)
        # Fall back to a synthesised id when the benchmark runner didn't
        # set one: e.g. "SOD1__p.Ala4Val". This is stable for a given
        # (gene, mutation) pair, which is what we need to join to gold.
        gene = output.get("gene") or ""
        mut = output.get("mutation") or ""
        if not gene:
            raise ValueError("therapy_agent_v1: record has neither case_id nor gene")
        slug = re.sub(r"\s+", "_", f"{gene}__{mut}").strip("_")
        return slug

    def extract_citations(self, output: dict) -> list[Citation]:
        """`supporting_evidence` is therapy-agent's evidence-and-citation
        list. We treat each entry as one citation (citation_id `e1`, `e2`,
        ...). When an entry has a DOI we promote it; URL is stored
        verbatim. The `claim` text becomes the citation's snippet so the
        grounding judge sees what the agent was relying on.

        Note: therapy-agent does not currently emit per-claim attribution,
        so all extracted claims will share the union of citation IDs.
        That is conservative for grounding (a claim is graded against
        every citation the agent put on the table) and will be tightened
        when therapy-agent adds per-claim attribution.
        """
        evidence = output.get("supporting_evidence") or []
        out: list[Citation] = []
        for i, ev in enumerate(evidence, start=1):
            text = (ev.get("claim") or "").strip()
            doi = (ev.get("doi") or "").strip() or None
            url = (ev.get("source_url") or "").strip() or None
            # If only the claim string contains a DOI, lift it out.
            if doi is None and text:
                m = re.search(r"\b10\.\d{4,9}/[^\s)]+", text)
                if m:
                    doi = m.group(0).rstrip(".,;)")
            out.append(
                Citation(
                    citation_id=f"e{i}",
                    doi=doi,
                    url=url,
                    title=None,
                    snippet=text or None,
                )
            )
        return out

    def extract_claims(self, output: dict) -> list[Claim]:
        """Therapy-agent emits each evidence entry as a single citation
        with a `claim` string. We treat each one as a self-cited claim
        (claim text attributes to its own evidence entry). Mechanism-
        class is appended as a single mechanistic claim attributed to all
        evidence entries.
        """
        evidence = output.get("supporting_evidence") or []
        claims: list[Claim] = []
        for i, ev in enumerate(evidence, start=1):
            text = (ev.get("claim") or "").strip()
            if not text or len(text) < 3:
                continue
            claims.append(
                Claim(
                    claim_id=f"c{i}",
                    text=text,
                    claim_type=_infer_claim_type(text),
                    cited_ids=[f"e{i}"],
                )
            )
        # Synthesize a mechanism-summary claim from target + modulation
        # so the grounding judge can score the central recommendation
        # even when the agent did not phrase it explicitly. Attributed
        # to all evidence entries.
        target = (output.get("target_protein") or {}).get("name")
        modulation = output.get("modulation_type")
        if target and modulation:
            mech_text = (
                f"The proposed strategy is {modulation} of {target} "
                f"for {output.get('disease_phenotype', 'this indication')}."
            )
            claims.append(
                Claim(
                    claim_id=f"c{len(claims) + 1}",
                    text=mech_text,
                    claim_type=ClaimType.RECOMMENDATION,
                    cited_ids=[f"e{i + 1}" for i in range(len(evidence))],
                )
            )
        return claims

    def extract_task_answer(self, output: dict) -> TaskAnswer:
        target = ((output.get("target_protein") or {}).get("name") or "").strip() or None
        aliases = []
        for drug in output.get("precedent_drugs") or []:
            name = (drug.get("name") or "").strip()
            if name:
                aliases.append(name)
        modulation_raw = output.get("modulation_type")
        modulation = _MODULATION_MAP.get(modulation_raw) if modulation_raw else None
        confidence = output.get("confidence_score")
        return TaskAnswer(
            predicted_target=target,
            predicted_target_aliases=aliases,
            predicted_modulation=modulation,
            predicted_mechanism_class=modulation_raw,
            confidence=float(confidence) if confidence is not None else None,
            raw={
                "model_used": output.get("model_used"),
                "schema_version": output.get("schema_version"),
                "wall_clock_seconds": output.get("wall_clock_seconds"),
                "input_tokens": output.get("input_tokens"),
                "output_tokens": output.get("output_tokens"),
            },
        )

    def to_agent_response(self, output: dict) -> AgentResponse:
        """Override the default to preserve the agent's reasoning-trace
        prose in `answer`. The trace is what a human reviewer reads —
        keeping it in `answer` makes the HTML report useful for triage."""
        base = super().to_agent_response(output)
        prose_parts: list[str] = []
        trace = output.get("reasoning_trace") or []
        for step in trace:
            content = (step.get("content") or "").strip()
            if content:
                prose_parts.append(content)
        if not prose_parts:
            # Fall back to the joined evidence claims that the base method built.
            return base
        # Append claims at the end so the grounding judge still sees them.
        prose = "\n\n".join(prose_parts) + "\n\n" + base.answer
        return AgentResponse(
            case_id=base.case_id,
            answer=prose,
            citations=base.citations,
            claim_to_citations=base.claim_to_citations,
            structured_outputs=base.structured_outputs,
            latency_ms=int((output.get("wall_clock_seconds") or 0) * 1000) or None,
            tokens_in=output.get("input_tokens"),
            tokens_out=output.get("output_tokens"),
        )


_QUANT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|mg|ng|mmHg|fold|n\s*=)(?:\b|$|\s|[.,;])",
    re.IGNORECASE,
)


def _infer_claim_type(text: str) -> ClaimType:
    """Lightweight type inference for therapy-agent evidence strings.

    Heuristic rather than LLM-driven because the adapter must be fast and
    deterministic (no API calls in the hot loop). When in doubt we
    default to FACTUAL — quantitative gets bumped when a number with
    units is present; mechanistic when causal verbs dominate.
    """
    if _QUANT_RE.search(text):
        return ClaimType.QUANTITATIVE
    t = text.lower()
    if any(kw in t for kw in (" activates ", " inhibits ", " binds ", " degrades ", "mechanism")):
        return ClaimType.MECHANISTIC
    if any(kw in t for kw in ("should ", "recommend", "prioritise", "prioritize")):
        return ClaimType.RECOMMENDATION
    return ClaimType.FACTUAL
