# Recipe 01 — catching miscitation in a clinical-trial-matching agent

## The scenario

A RAG agent is asked: *"What open trials are recruiting adults with
SOD1-ALS who are still ambulatory?"* The agent retrieves several NCT
records and returns a recommendation. Crucially, the citations the agent
returns are real (they resolve to valid NCT IDs on
clinicaltrials.gov) — but the agent's *claims* about those trials
don't match the trials' eligibility criteria.

This is the most insidious RAG failure mode in clinical contexts: a
reader who spot-checks citation existence is satisfied, but a reader
who actually reads the cited trial would catch the error. Generic
RAG evaluation (which only checks that citations exist) misses this
completely. `bio-rag-eval` catches it via the grounding judge, which
scores `UNSUPPORTED` whenever the cited passage doesn't entail the
claim.

## What you'll see

After running `python run.py --mock`, the report will show:

- `citation_traceability` ≈ 1.0 — every NCT ID resolves
- `grounding_rate` < 0.6 — claims are not actually supported by the
  cited passages
- `hallucination_rate` ≈ 0.4, decomposed as ~0.1 uncited and ~0.3
  miscited — the agent's failure is overwhelmingly miscitation, not
  uncited content

This is the diagnostic signature of "agent retrieves but doesn't read."

## Files

- `case.yaml` — one gold-standard case for the trial-matching task
- `agent_response.jsonl` — one realistic agent response with three real
  NCT IDs and two genuine miscitations (claims about exclusion criteria
  that contradict the actual trial)
- `run.py` — minimal driver
- `expected_output.md` — committed numbers from the mock run, so you
  can verify your environment reproduces them
