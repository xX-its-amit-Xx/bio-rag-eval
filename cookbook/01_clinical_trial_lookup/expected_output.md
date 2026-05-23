# Recipe 01 — expected output (mock judge, seed=7)

Running `python run.py --mock` should produce these numbers
deterministically. They illustrate the central point of this recipe:

| metric | value | interpretation |
|---|---:|---|
| `citation_traceability_offline_check` | 1.000 | every NCT ID is well-formed |
| `grounding_rate` | 0.286 (2/7 claims SUPPORTED) | most claims aren't actually backed by their sources |
| `weak_grounding_rate` | 0.357 | small bump from one PARTIALLY_SUPPORTED claim |
| `contradiction_rate` | 0.429 (3/7) | three claims are directly contradicted by the cited snippets |
| `unsupported_rate` | 0.143 (1/7) | one claim was "made up" with no source coverage |
| `hallucination_rate` | 0.571 (uncited + miscited factual claims / factual claims) | dominant failure mode |
| `uncited_rate` | 0.000 | the agent cited everything |
| `miscitation_rate` | 0.571 | ...but the citations don't say what the agent claims they do |
| `forbidden_claim_present` | 1.000 | "tofersen is investigational only" is a known false claim |
| `answer_completeness` | 0.500 | 1 facet fully covered + 1 partially covered, out of 3 |
| `evidence_sufficiency_score` | 2.000 / 5 | only registry pages, no primary outcome literature |

## What to take away

A `citation_traceability` of 1.0 is necessary but radically
insufficient. An agent that cites real trial IDs but invents their
eligibility criteria gets a perfect score on naive citation-resolution
metrics — and a 0.29 on `grounding_rate`. The miscitation/uncited
decomposition tells you the fix isn't "make the agent cite more" —
the agent is already citing aggressively. The fix is "make the agent
actually read what it's citing."
