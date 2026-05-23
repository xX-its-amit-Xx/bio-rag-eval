"""Abstract judge interface.

Every concrete judge (Anthropic, OpenAI, Mock) implements one method:
`judge_json(prompt, schema)` -> parsed pydantic model. Returning a pydantic
instance (not a dict) means downstream metric code never has to handle
malformed JSON — that's the judge's problem to fail loudly on.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class JudgeResponse:
    """Wrap a parsed pydantic instance with the raw provider response for
    audit logging. Two consumers need the raw text: the bias-consistency
    metric (compares two judgments on the same input) and the report
    generator (shows the actual judge rationale in the HTML report)."""

    parsed: BaseModel
    raw_text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class BaseJudge(abc.ABC):
    """Interface every judge implements.

    Implementations must:
      1. Send `prompt` to the model.
      2. Constrain the response to match `schema` (provider-native
         structured-output APIs, not regex parsing).
      3. Parse the response into the pydantic class.
      4. Raise on parse failure — do NOT silently return a default.
    """

    model: str
    provider: str

    @abc.abstractmethod
    def judge_json(self, prompt: str, schema: type[T]) -> JudgeResponse:
        """Run `prompt` and return a `JudgeResponse` whose `.parsed` is a
        `schema` instance."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, provider={self.provider!r})"


class MockJudge(BaseJudge):
    """Deterministic in-memory judge for tests and demos.

    Constructed with a dict mapping schema class -> instance (or callable).
    On `judge_json` it looks up the schema and returns that instance,
    independent of the prompt content. Lets the rest of the pipeline be
    tested without network calls.
    """

    def __init__(self, responses: dict[type[BaseModel], BaseModel | list[BaseModel]]):
        self.model = "mock"
        self.provider = "mock"
        self._responses: dict[type[BaseModel], list[BaseModel]] = {}
        for k, v in responses.items():
            self._responses[k] = list(v) if isinstance(v, list) else [v]
        self._cursor: dict[type[BaseModel], int] = {k: 0 for k in self._responses}

    def judge_json(self, prompt: str, schema: type[T]) -> JudgeResponse:
        if schema not in self._responses:
            raise KeyError(
                f"MockJudge was not configured with a response for {schema.__name__}. "
                f"Pass one in the constructor."
            )
        idx = self._cursor[schema]
        responses = self._responses[schema]
        # Cycle if we've exhausted — useful for repeated calls in a loop.
        item = responses[idx % len(responses)]
        self._cursor[schema] = idx + 1
        if not isinstance(item, schema):
            raise TypeError(f"MockJudge response for {schema.__name__} is wrong type: {type(item)}")
        return JudgeResponse(
            parsed=item,
            raw_text=item.model_dump_json(),
            model=self.model,
            provider=self.provider,
        )
