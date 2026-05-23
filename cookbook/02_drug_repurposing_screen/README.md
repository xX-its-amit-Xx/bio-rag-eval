# Recipe 02 — A/B comparing two drug-repurposing RAG agents

## The scenario

You have two RAG configurations for the same drug-repurposing task:

- **Agent A** (`baseline`): single-doc retrieval (top-1 BM25 hit, no rerank)
- **Agent B** (`reranker`): top-20 BM25 retrieved, reranked by a
  cross-encoder, top-3 passed to the generator

You want to know if B is actually better — and if so, on which axis. Not
"which has a higher overall score" but "does B improve grounding, or
completeness, or just confidence?" The bootstrap CIs are essential here
because on a 4-case mini-benchmark a difference in point estimates can
easily fall inside the noise.

## What you'll see

After `python run.py --mock`, the report contrasts the two agents on
the same 4 cases. With seed=7 the mock numbers come out as:

| metric | Agent A (baseline) | Agent B (reranker) | difference |
|---|---:|---:|---|
| `grounding_rate` | 0.50 | 0.83 | +0.33, CIs probably overlap on n=4 |
| `evidence_sufficiency_score` | 2.0 | 3.5 | +1.5 — likely meaningful |
| `answer_completeness` | 0.42 | 0.75 | +0.33 |
| `task_success_composite` | 0.42 | 0.83 | +0.42 |

The key teaching point: the report shows you `[ci_low, ci_high]` for
every aggregate. With n=4 cases the CIs are wide enough that you should
*not* declare the reranker better from this run alone — you'd want
n≈20 cases before drawing a directional conclusion. The recipe
shows you how the framework guards against premature claims.

## Files

- `cases.yaml` — 4 drug-repurposing questions (one file, list-of-cases shape)
- `agent_a_responses.jsonl`, `agent_b_responses.jsonl` — outputs from
  the two configurations
- `run.py` — runs both, prints a side-by-side comparison
- `expected_output.md` — committed numbers
