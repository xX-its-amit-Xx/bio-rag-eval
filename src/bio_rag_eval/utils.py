"""Shared utilities — bootstrap CIs and a few small helpers.

The bootstrap implementation is intentionally simple (np.random.choice
with replacement, percentile method). For metrics that are means of
0/1 indicators the percentile bootstrap is well-behaved; for tiny n we
fall back to "no CI" rather than emitting a bogus narrow interval.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda x: float(np.mean(x)),
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float] | tuple[float, None, None]:
    """Percentile-bootstrap confidence interval.

    Returns (point, ci_low, ci_high) — or (point, None, None) when the
    sample is too small for a meaningful CI (n < 5). The point estimate
    is the statistic on the original sample, NOT the mean of the
    bootstrap distribution.
    """
    arr = np.asarray(sample, dtype=float)
    if arr.size == 0:
        return (float("nan"), None, None)
    point = statistic(arr)
    if arr.size < 5:
        return (float(point), None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resampled = arr[idx]
    stats = np.array([statistic(row) for row in resampled])
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return (float(point), lo, hi)


def hash_config(config: Any) -> str:
    """Stable short hash of a JSON-serializable config. Used as
    `RunMetadata.config_hash` and as part of on-disk cache keys."""
    if hasattr(config, "model_dump"):  # pydantic
        payload = config.model_dump()
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def safe_div(num: float, denom: float) -> float:
    """Division that returns NaN on a 0 denominator instead of raising.

    Used by every rate metric so that "no claims to evaluate" surfaces as
    NaN (caller can flag as N/A) rather than blowing up the whole run."""
    if denom == 0:
        return float("nan")
    return num / denom
