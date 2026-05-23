from __future__ import annotations

import pytest

from bio_rag_eval.prompts import list_prompts, load_prompt
from bio_rag_eval.utils import bootstrap_ci, hash_config, safe_div


def test_all_expected_prompts_loadable():
    expected = {
        "claim_extraction_v1",
        "grounding_judge_v1",
        "mechanism_judge_v1",
        "evidence_sufficiency_v1",
        "completeness_judge_v1",
    }
    found = set(list_prompts())
    assert expected <= found


def test_prompt_renders_with_required_vars():
    p = load_prompt("grounding_judge_v1")
    rendered = p.render(
        claim_id="c1",
        claim_text="example claim",
        claim_type="factual",
        cited_passages=[{"citation_id": "s1", "title": "t", "snippet": "snip"}],
        rubric_order="supported_first",
    )
    assert "example claim" in rendered
    assert "supported -> partially_supported" in rendered


def test_prompt_versions_have_semver_shape():
    for name, version in list_prompts().items():
        parts = version.split(".")
        assert len(parts) >= 2, f"{name} version is not semver-shaped: {version}"
        assert all(p.isdigit() for p in parts), f"{name} version is not numeric: {version}"


def test_bootstrap_ci_known_mean():
    point, lo, hi = bootstrap_ci([1.0] * 100, n_resamples=200, seed=1)
    assert point == 1.0
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_small_sample_no_ci():
    point, lo, hi = bootstrap_ci([1, 0, 1], n_resamples=200, seed=1)
    assert point == pytest.approx(2 / 3)
    assert lo is None and hi is None


def test_safe_div_zero_returns_nan():
    val = safe_div(0, 0)
    assert val != val  # NaN


def test_hash_config_is_stable():
    a = hash_config({"a": 1, "b": [1, 2, 3]})
    b = hash_config({"b": [1, 2, 3], "a": 1})  # different key order
    assert a == b
    assert len(a) == 12
