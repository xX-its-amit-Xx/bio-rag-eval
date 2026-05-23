"""Adapter registry.

External agent outputs and gold-standard datasets vary in shape; the
adapter layer normalizes them to bio-rag-eval's internal schemas
(`AgentResponse` and `GoldStandardCase`). Adapters are registered by a
stable string name and looked up by the CLI via `--predictions-adapter`
and `--gold-source` flags.

Two kinds of adapter, both kept in the same registry namespace:

- `PredictionsAdapter`: implements `extract_claims`, `extract_citations`,
  `extract_task_answer` plus a `to_agent_response` convenience that
  assembles an `AgentResponse`. Adapter name pattern: `<system>_v<n>`.
- `GoldStandardAdapter`: implements `get_gold(case_id)` returning a
  `GoldStandard`. Same naming convention.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from bio_rag_eval.adapters.base import (
    GoldStandardAdapter,
    PredictionsAdapter,
)

if TYPE_CHECKING:
    pass

# Two separate maps so a name collision across the two adapter kinds is
# a hard error, not silently the wrong type at lookup time.
_PREDICTION_REGISTRY: dict[str, type[PredictionsAdapter]] = {}
_GOLD_REGISTRY: dict[str, type[GoldStandardAdapter]] = {}

T = TypeVar("T")


def register_adapter(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator. Registers a class as either a PredictionsAdapter or
    GoldStandardAdapter based on which abstract base it subclasses.

    Raises:
        ValueError if the name is already registered, or if the class
        does not subclass one of the two adapter bases.
    """

    def _inner(cls: type[T]) -> type[T]:
        if issubclass(cls, PredictionsAdapter):  # type: ignore[arg-type]
            if name in _PREDICTION_REGISTRY:
                raise ValueError(f"predictions adapter already registered: {name!r}")
            _PREDICTION_REGISTRY[name] = cls  # type: ignore[assignment]
        elif issubclass(cls, GoldStandardAdapter):  # type: ignore[arg-type]
            if name in _GOLD_REGISTRY:
                raise ValueError(f"gold-standard adapter already registered: {name!r}")
            _GOLD_REGISTRY[name] = cls  # type: ignore[assignment]
        else:
            raise TypeError(
                f"{cls.__name__} must subclass PredictionsAdapter or GoldStandardAdapter"
            )
        return cls

    return _inner


def get_predictions_adapter(name: str, **kwargs: Any) -> PredictionsAdapter:
    if name not in _PREDICTION_REGISTRY:
        raise KeyError(
            f"no predictions adapter registered as {name!r}. "
            f"available: {sorted(_PREDICTION_REGISTRY)}"
        )
    return _PREDICTION_REGISTRY[name](**kwargs)


def get_gold_adapter(name: str, **kwargs: Any) -> GoldStandardAdapter:
    if name not in _GOLD_REGISTRY:
        raise KeyError(
            f"no gold-standard adapter registered as {name!r}. "
            f"available: {sorted(_GOLD_REGISTRY)}"
        )
    return _GOLD_REGISTRY[name](**kwargs)


def list_adapters() -> dict[str, list[str]]:
    return {
        "predictions": sorted(_PREDICTION_REGISTRY),
        "gold": sorted(_GOLD_REGISTRY),
    }


# Import side-effect: register the built-in adapters. Keep this import at
# the bottom to avoid a circular dependency on `register_adapter`.
from bio_rag_eval.adapters import fda_triples as _fda_triples  # noqa: E402,F401
from bio_rag_eval.adapters import therapy_agent as _therapy_agent  # noqa: E402,F401


__all__ = [
    "GoldStandardAdapter",
    "PredictionsAdapter",
    "get_gold_adapter",
    "get_predictions_adapter",
    "list_adapters",
    "register_adapter",
]
