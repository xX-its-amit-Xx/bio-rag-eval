# Recipe 02 — expected output (mock judge, seed=7)

`python run.py --mock` produces these numbers deterministically. Point
estimates are reproducible across runs; bootstrap CI bounds depend on
the seed (set to 7 by default).

## Agent A (baseline, top-1 BM25 retrieval)

| metric | point | 95% CI (n) |
|---|---:|---|
| `grounding_rate` | 0.500 | [0.125, 0.750] (n=8 claims) |
| `hallucination_rate` | 0.000 | — (n=4 cases) |
| `evidence_sufficiency_score` | 2.000 | — (n=4) |
| `answer_completeness` | 0.500 | — (n=4) |
| `task_success_composite` | 1.000 | — (n=4) |
| `citation_traceability_offline_check` | 1.000 | — (n=4) |

## Agent B (reranker, top-3 with cross-encoder)

| metric | point | 95% CI (n) |
|---|---:|---|
| `grounding_rate` | 1.000 | [1.000, 1.000] (n=8) |
| `hallucination_rate` | 0.000 | — (n=4) |
| `evidence_sufficiency_score` | 4.000 | — (n=4) |
| `answer_completeness` | 1.000 | — (n=4) |
| `task_success_composite` | 1.000 | — (n=4) |
| `citation_traceability_offline_check` | 0.875 | — (n=4) |

## How to read this

Three things to notice:

1. **Direction is clear, but n is small.** Even though Agent B's
   point-estimate dominates Agent A on `grounding_rate`,
   `evidence_sufficiency`, and `answer_completeness`, the n=4 case
   bootstraps are reporting "—" (CI suppressed below n<5). The
   framework deliberately refuses to print a confidence interval when
   the sample is too small for the percentile bootstrap to be
   well-behaved. With n≈20 cases the CIs would tighten and a
   directional conclusion would be defensible.
2. **Evidence-sufficiency moves more than grounding.** With Agent B
   retrieving 3 documents per query instead of 1, the
   `evidence_sufficiency_score` jumps from 2 to 4 — a bigger swing
   than `grounding_rate`. That tells you the reranker isn't just
   helping the model cite better; it's also broadening the evidence
   base, which the judge counts as 2+ independent primary sources
   rather than 1.
3. **Citation traceability can REGRESS with a reranker.** Agent B
   traceability is 0.875 vs Agent A's 1.000 — Agent B emitted one
   citation that lacks any structured ID (only a title). This is a
   real diagnostic: more aggressive retrieval pulls in more sources,
   some of which are pre-print or registry pages that the agent
   then attributes with only a title. Without bio-rag-eval, this
   regression would be invisible — generic "did the agent cite?"
   metrics would have shown Agent B as strictly better.

## Methodological caveat

In the mock setup, the same `MockJudge` scores both agents — there's
no actual judge LLM. In a real comparison you must either: (a) use the
same judge model on both runs (recommended), or (b) report each
agent's metrics with its judge's identity stamped on it (in the
`RunMetadata` block) and not compare across runs that used different
judges.
