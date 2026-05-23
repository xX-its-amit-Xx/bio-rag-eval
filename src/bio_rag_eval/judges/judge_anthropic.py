"""Anthropic-backed judge. Uses tool-use to force JSON-schema output."""
from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from bio_rag_eval.judges.base import BaseJudge, JudgeResponse

T = TypeVar("T", bound=BaseModel)


class AnthropicJudge(BaseJudge):
    """Calls the Anthropic Messages API and forces the response into a
    pydantic schema using a single tool whose `input_schema` IS the
    schema's JSON schema.

    Why tool-use and not raw text + JSON-mode: tool-use gives us strict
    schema enforcement on the wire (the model returns a tool_use block
    whose `input` validates against the schema we sent), and the Anthropic
    runtime will retry internally on a schema violation. Raw JSON mode is
    only "best-effort" valid JSON, with no schema enforcement.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        client: Any = None,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AnthropicJudge requires the `anthropic` package. "
                "Install with: pip install 'bio-rag-eval[default]'"
            ) from e

        self.model = model
        self.provider = "anthropic"
        self._client = client or anthropic.Anthropic(api_key=api_key)
        self._max_tokens = max_tokens
        self._temperature = temperature

    def judge_json(self, prompt: str, schema: type[T]) -> JudgeResponse:
        tool_name = f"emit_{schema.__name__.lower()}"
        tool_schema = _strip_unsupported(schema.model_json_schema())

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Emit a single {schema.__name__} object.",
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_input: dict[str, Any] | None = None
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                tool_input = block.input  # type: ignore[assignment]
                break
        if tool_input is None:
            raise RuntimeError(
                f"AnthropicJudge: model {self.model} did not emit a tool_use block. "
                f"Stop reason: {getattr(message, 'stop_reason', '?')}"
            )

        try:
            parsed = schema.model_validate(tool_input)
        except ValidationError as e:
            raise RuntimeError(
                f"AnthropicJudge: tool input failed {schema.__name__} validation: {e}"
            ) from e

        usage = getattr(message, "usage", None)
        return JudgeResponse(
            parsed=parsed,
            raw_text=json.dumps(tool_input),
            model=self.model,
            provider=self.provider,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        )


def _strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic emits a few keys (notably `$defs` references and some
    `format` markers) that Anthropic's input_schema validator does not
    accept. We resolve $refs inline and drop unsupported markers."""
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                key = node["$ref"].split("/")[-1]
                if key in defs:
                    return resolve(defs[key])
            return {k: resolve(v) for k, v in node.items() if k not in ("title",)}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    out = resolve(schema)
    if isinstance(out, dict) and "type" not in out:
        out["type"] = "object"
    return out
