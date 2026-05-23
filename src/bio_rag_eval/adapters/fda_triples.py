"""Gold-standard adapter backed by the `fda-strategy-triples` dataset.

The FDA triples package ships a frozen release of curated
(gene -> variant -> FDA drug -> molecular_target -> mechanism) tuples.
For a given case_id of the form `<GENE>__<variant_context>` (or a bare
gene symbol), we look up the matching triple and emit the gold-side
fields needed for `task_success` scoring:

    expected_target            <- molecular_target  (UniProt + SYMBOL form)
    expected_target_aliases    <- drug_name_generic, drug_name_brand
    expected_mechanism_class   <- mechanism_class
    expected_modulation        <- derived from mechanism_class

The mapping mechanism_class -> ModulationDirection mirrors the
therapy-agent adapter's direction map so a prediction labelled `ASO`
matches a gold labelled `ASO` regardless of provider terminology.

If multiple triples match the same gene, we prefer the one whose
variant_context substring matches the case_id's variant slug; failing
that, the one with the earliest approval_date (most established).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from bio_rag_eval.adapters import register_adapter
from bio_rag_eval.adapters.base import GoldStandardAdapter
from bio_rag_eval.schemas.adapter_types import GoldStandard
from bio_rag_eval.schemas.case import ModulationDirection

_MECH_TO_MODULATION: dict[str, ModulationDirection] = {
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


@register_adapter("fda_triples_v1")
class FDATriplesGoldStandard(GoldStandardAdapter):
    """Loads fda-strategy-triples as gold answers for task_success scoring."""

    NAME = "fda_triples_v1"
    SOURCE_SCHEMA_VERSION = "1.0.0"  # fda_strategy_triples.SCHEMA_VERSION

    def __init__(self, version: str = "latest"):
        """`version` selects the bundled release (e.g. `"v0.1.0"`)."""
        self._version = version
        self._records: list[dict[str, Any]] | None = None

    def _records_loaded(self) -> list[dict[str, Any]]:
        if self._records is None:
            self._records = _load_triples(self._version)
        return self._records

    def get_gold(self, case_id: str) -> GoldStandard:
        gene, variant_slug = _split_case_id(case_id)
        candidates = [
            r for r in self._records_loaded() if gene in r["triple"]["associated_genes"]
        ]
        if not candidates:
            raise KeyError(
                f"fda_triples_v1: no triple in release {self._version!r} for gene {gene!r} "
                f"(case_id {case_id!r})"
            )
        chosen = _pick_best_match(candidates, variant_slug)
        triple = chosen["triple"]
        mech = triple["mechanism_class"]
        return GoldStandard(
            case_id=case_id,
            expected_target=triple["molecular_target"],
            expected_target_aliases=[
                triple["drug_name_generic"],
                triple["drug_name_brand"],
            ],
            expected_modulation=_MECH_TO_MODULATION.get(mech),
            expected_mechanism_class=mech,
            source=f"{self.NAME}@{self._version}",
            raw={
                "drug_name_generic": triple["drug_name_generic"],
                "drug_name_brand": triple["drug_name_brand"],
                "approval_date": triple.get("approval_date"),
                "indicated_disease": triple.get("indicated_disease"),
                "variant_context": triple.get("variant_context"),
                "mechanism_summary": triple.get("mechanism_summary"),
                "primary_citation_pmid": triple.get("primary_citation_pmid"),
                "schema_version": chosen.get("schema_version"),
                "triple_id": chosen.get("id"),
            },
        )


def _split_case_id(case_id: str) -> tuple[str, str]:
    """`SOD1__p.Ala4Val` -> ('SOD1', 'p.Ala4Val'). Bare `SOD1` -> ('SOD1', '')."""
    if "__" in case_id:
        gene, variant = case_id.split("__", 1)
        return gene.strip().upper(), variant.strip()
    return case_id.strip().upper(), ""


def _pick_best_match(candidates: list[dict[str, Any]], variant_slug: str) -> dict[str, Any]:
    """Disambiguate when one gene appears in multiple FDA triples.

    1. Prefer triples whose variant_context substring-matches the case's variant slug
       (case-insensitive). Useful when the gene has multiple approved drugs for
       different variant classes (e.g. CFTR + ivacaftor for G551D vs CFTR + Trikafta).
    2. Fall back to the earliest approval_date (most established mechanism).
    3. Last-ditch: return the first candidate deterministically.
    """
    if variant_slug:
        slug_lc = variant_slug.lower()
        matched = [
            r for r in candidates if slug_lc in (r["triple"].get("variant_context") or "").lower()
        ]
        if matched:
            candidates = matched

    def sort_key(r: dict[str, Any]) -> str:
        d = r["triple"].get("approval_date") or "9999-99-99"
        return d

    return sorted(candidates, key=sort_key)[0]


@lru_cache(maxsize=4)
def _load_triples(version: str) -> list[dict[str, Any]]:
    """Lazy import — keeps fda_strategy_triples out of the import chain
    when bio-rag-eval is used without the FDA adapter."""
    try:
        from fda_strategy_triples import load_dataset
    except ImportError as e:
        raise ImportError(
            "fda_triples_v1 requires `fda-strategy-triples` to be installed. "
            "Install the sibling package: `pip install -e ../fda-strategy-triples`."
        ) from e
    records = load_dataset(version=version, as_pydantic=True)
    out: list[dict[str, Any]] = []
    for rec in records:
        # Normalize associated_genes to uppercase for case-insensitive lookup.
        d = rec.model_dump()
        d["triple"]["associated_genes"] = [
            g.upper() for g in d["triple"].get("associated_genes", [])
        ]
        out.append(d)
    return out


def list_known_genes(version: str = "latest") -> set[str]:
    """Convenience: which gene symbols can this dataset answer for?"""
    return {
        g for r in _load_triples(version) for g in r["triple"]["associated_genes"]
    }
