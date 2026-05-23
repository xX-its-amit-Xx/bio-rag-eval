"""Task-success metric.

Schema-defined per task type. For VARIANT_TO_STRATEGY (the headline
task type the therapy-agent project produces), task_success decomposes
into three components and a composite:

    correct_target              ∈ {0, 1}
    correct_modulation_direction ∈ {0, 1}
    mechanism_coherence          ∈ {1, ..., 5} (likert from MechanismJudgment)
    task_success_composite       = mean of the three normalized to [0, 1]

The composite is a weighted mean where mechanism_coherence is scaled to
[0,1] via (score-1)/4 so the three components live on the same scale.
Weights are equal by default; the runner accepts per-task weight overrides.

For other task types we provide light-weight scorers — see
`compute_task_success` dispatch.
"""
from __future__ import annotations

from bio_rag_eval.judges.base import BaseJudge
from bio_rag_eval.prompts import load_prompt
from bio_rag_eval.schemas.case import AgentResponse, GoldStandardCase, TaskType
from bio_rag_eval.schemas.judgments import MechanismJudgment

PROMPT_NAME = "mechanism_judge_v1"


def score_target(case: GoldStandardCase, response: AgentResponse) -> int:
    """1 if the agent's predicted target (or its prose) matches the
    expected target by exact or alias match, else 0.

    Match logic: case-insensitive substring against either
    `structured_outputs["predicted_target"]` (preferred) or the answer
    prose. Aliases are tried independently so e.g. "Qalsody" matches
    even when the expected target is "SOD1".
    """
    if case.expected_target is None:
        return 1  # no expectation -> trivially satisfied
    haystack = " ".join(
        [
            str(response.structured_outputs.get("predicted_target", "")),
            response.answer,
        ]
    ).lower()
    candidates = [case.expected_target] + list(case.expected_target_aliases)
    return 1 if any(cand.lower() in haystack for cand in candidates if cand) else 0


def score_modulation(case: GoldStandardCase, response: AgentResponse) -> int:
    """1 if the predicted modulation direction matches expected.

    Modulation can come from `structured_outputs["predicted_modulation"]`
    (preferred — agents that emit structured output get cleaner scoring)
    or be inferred from the prose with a small keyword table.

    Returns 1 if `expected_modulation` is AGNOSTIC (gold accepts any).
    """
    if case.expected_modulation is None:
        return 1
    if case.expected_modulation.value == "agnostic":
        return 1
    predicted = str(response.structured_outputs.get("predicted_modulation", "")).lower()
    if not predicted:
        predicted = _infer_modulation_from_prose(response.answer)
    return 1 if predicted == case.expected_modulation.value else 0


def _infer_modulation_from_prose(answer: str) -> str:
    """Tiny keyword inference. We deliberately don't call an LLM here —
    if the agent did not emit structured modulation, the prose is what
    we have to grade. This is conservative and may return "" (unknown)."""
    a = answer.lower()
    # Order matters: more specific first.
    if any(k in a for k in ("base editor", "prime editor", "crispr", "gene edit", "gene editing")):
        return "edit"
    if any(k in a for k in ("aso", "antisense oligonucleotide", "sirna", "rnai", "silenc")):
        return "silence"
    if any(k in a for k in ("gene replacement", "gene therapy", "aav", "enzyme replacement")):
        return "replace"
    if any(k in a for k in ("agonist", "activator", "activate")):
        return "activate"
    if any(k in a for k in ("inhibitor", "antagonist", "block", "inhibit")):
        return "inhibit"
    return ""


def judge_mechanism(
    case: GoldStandardCase,
    response: AgentResponse,
    judge: BaseJudge,
) -> MechanismJudgment:
    """LLM-as-judge: score the coherence of the causal chain on Likert-5."""
    prompt = load_prompt(PROMPT_NAME)
    rendered = prompt.render(
        variant=case.context.get("mutation") or case.context.get("variant") or "(not specified)",
        phenotype=case.context.get("disease_phenotype")
        or case.context.get("phenotype")
        or "(not specified)",
        answer=response.answer,
    )
    result = judge.judge_json(rendered, MechanismJudgment)
    assert isinstance(result.parsed, MechanismJudgment)
    return result.parsed


def compute_task_success(
    case: GoldStandardCase,
    response: AgentResponse,
    mechanism: MechanismJudgment | None,
) -> dict[str, float]:
    """Compose components for a single case.

    Returns:
      - correct_target: 0 or 1
      - correct_modulation_direction: 0 or 1
      - mechanism_coherence: 1..5 (or NaN if no judge call was made)
      - task_success_composite: in [0, 1] (NaN if mechanism missing AND task expects it)
    """
    target = score_target(case, response)
    modulation = score_modulation(case, response)
    mech_score = float(mechanism.score) if mechanism else float("nan")
    mech_norm = (mech_score - 1.0) / 4.0 if mechanism else float("nan")

    if case.task_type == TaskType.VARIANT_TO_STRATEGY:
        if mechanism is None:
            composite = float("nan")
        else:
            composite = (target + modulation + mech_norm) / 3.0
    else:
        # For non-V2S tasks we only weight the components that were scored.
        parts = [target]
        if case.expected_modulation is not None:
            parts.append(modulation)
        if mechanism is not None:
            parts.append(mech_norm)
        composite = sum(parts) / len(parts)

    return {
        "correct_target": float(target),
        "correct_modulation_direction": float(modulation),
        "mechanism_coherence": mech_score,
        "task_success_composite": float(composite),
    }
