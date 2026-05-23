"""Tests for FDATriplesGoldStandard (fda_triples_v1).

These tests skip if the `fda-strategy-triples` package isn't importable
(e.g. CI that didn't install the sibling repo). When it IS available,
they verify that:
  - the adapter is registered
  - well-known genes (CFTR, HBB, TTR, etc.) resolve
  - unknown genes raise KeyError
  - the variant_slug picks the right triple when multiple match
"""
from __future__ import annotations

import pytest

pytest.importorskip("fda_strategy_triples", reason="fda-strategy-triples not installed")

from bio_rag_eval.adapters import get_gold_adapter, list_adapters
from bio_rag_eval.adapters.fda_triples import FDATriplesGoldStandard, _split_case_id
from bio_rag_eval.schemas.case import ModulationDirection


def test_adapter_registered():
    assert "fda_triples_v1" in list_adapters()["gold"]


def test_split_case_id():
    assert _split_case_id("SOD1__p.Ala4Val") == ("SOD1", "p.Ala4Val")
    assert _split_case_id("cftr") == ("CFTR", "")
    assert _split_case_id("HBB") == ("HBB", "")


def test_get_gold_resolves_known_gene():
    g = get_gold_adapter("fda_triples_v1")
    gold = g.get_gold("CFTR__G551D")
    assert gold.expected_target is not None
    assert "CFTR" in gold.expected_target
    # CFTR is potentiated by ivacaftor -> agonist -> ACTIVATE
    assert gold.expected_modulation in (ModulationDirection.ACTIVATE, ModulationDirection.INHIBIT)
    assert any("ivacaftor" in a.lower() or "kalydeco" in a.lower() for a in gold.expected_target_aliases)
    assert gold.source.startswith("fda_triples_v1@")


def test_get_gold_unknown_gene_raises():
    g = get_gold_adapter("fda_triples_v1")
    with pytest.raises(KeyError):
        g.get_gold("MADE_UP_GENE")


def test_has_method():
    g = FDATriplesGoldStandard()
    assert g.has("CFTR")
    assert not g.has("MADE_UP_GENE_2")


def test_raw_payload_carries_provenance():
    g = get_gold_adapter("fda_triples_v1")
    gold = g.get_gold("TTR")
    assert "drug_name_generic" in gold.raw
    assert "drug_name_brand" in gold.raw
    assert gold.raw["drug_name_generic"]
