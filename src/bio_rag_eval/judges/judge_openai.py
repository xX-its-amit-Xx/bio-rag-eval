"""OpenAI-backed judge, interface-compatible with AnthropicJudge.

Uses OpenAI's `response_format={"type":"json_schema", ...}` to force the
output to match the pydantic schema on the wire — same guarantee as the
Anthropic tool-use approach.
"""
from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from bio_rag_eval.judges.base import BaseJudge, JudgeResponse

T = TypeVar("T", bound=BaseModel)


class OpenAIJudge(BaseJudge):
    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any = None,
    ):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenAIJudge requires the `openai` package. "
                "Install with: pip install 'bio-rag-eval[default]'"
            ) from e

        self.model = model
        self.provider = "openai"
        self._client = client or openai.OpenAI(api_key=api_key)
        self._temperature = temperature

    def judge_json(self, prompt: str, schema: type[T]) -> JudgeResponse:
        schema_dict = _openai_schema(schema)

        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self._temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema_dict,
                },
            },
            messages=[{"role": "user", "content": prompt}],
        )

        choice = completion.choices[0]
        raw = choice.message.content or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpenAIJudge: model {self.model} returned non-JSON despite "
                f"json_schema mode. Content: {raw[:500]!r}"
            ) from e

        try:
            parsed = schema.model_validate(payload)
        except ValidationError as e:
            raise RuntimeError(
                f"OpenAIJudge: payload failed {schema.__name__} validation: {e}"
            ) from e

        usage = getattr(completion, "usage", None)
        return JudgeResponse(
            parsed=parsed,
            raw_text=raw,
            model=self.model,
            provider=self.provider,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


def _openai_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """OpenAI strict json_schema requires `additionalProperties: false` on
    every object and every field in `required`. Pydantic v2 already sets
    additionalProperties via `extra='forbid'` (our schemas all use it),
    but we walk the schema to enforce it explicitly and inline $defs."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", None) or raw.pop("definitions", None) or {}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                key = node["$ref"].split("/")[-1]
                if key in defs:
                    return walk(defs[key])
            out = {k: walk(v) for k, v in node.items() if k != "title"}
            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)
                if "properties" in out:
                    out.setdefault("required", list(out["properties"].keys()))
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(raw)
