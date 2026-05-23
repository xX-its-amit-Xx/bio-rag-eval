# bio-rag-eval

An evaluation harness for biomedical RAG and agentic systems. Built for
the kind of failure modes that generic RAG evaluation cannot catch:
miscited claims that point at real-but-irrelevant sources, mechanism
explanations that are internally inconsistent, recommendations whose
underlying evidence is a single anecdote, and judge-prompt
position-bias that makes the headline numbers themselves unreliable.

License: GNU GPL 3.0.

## Why generic RAG eval is insufficient for biomed

Generic RAG benchmarks check three things: (a) did the agent retrieve
something, (b) did it cite that something, and (c) does the citation
resolve. In biomedical settings each of these is necessary and none is
sufficient.

A trial-matching agent can cite a real NCT ID whose actual eligibility
criteria contradict the agent's claim about that trial. A
variant-to-strategy agent can correctly identify a gene but propose a
modulation direction that's mechanistically backwards. A drug-
repurposing agent can summarize a review correctly while citing the
review's primary references — making it look like the recommendation
rests on multiple independent studies when in fact it rests on one.

`bio-rag-eval` is built around six metrics that catch these failure
modes explicitly, every LLM-as-judge call uses pydantic-typed
structured output (no free-text parsing), and every aggregate gets a
percentile-bootstrap confidence interval so you do not over-interpret a
small benchmark.

## Quickstart

```bash
pip install -e .[dev]

# Run the 10 curated rare-disease cases with a mock judge (no API key needed):
python examples/eval_therapy_agent.py --mock

# Or with a real Anthropic judge:
ANTHROPIC_API_KEY=sk-... python examples/eval_therapy_agent.py \
    --judge anthropic --model claude-opus-4-7
```

## Metrics

### 1. `grounding_rate` — fraction of claims directly supported by their citation

```
grounding_rate = |{c in claims : judge(c) == SUPPORTED}| / |graded_claims|
```

The grounding judge (`grounding_judge_v1`) sees the claim and the
*exact passage* the agent cited and returns one of four labels:
`supported`, `partially_supported`, `contradicted`, `unsupported`.
We also surface `weak_grounding_rate` which counts
`partially_supported` at 0.5, and `contradiction_rate` /
`unsupported_rate` for failure-mode breakdown.

**Worked example.** Claim: *"Tofersen reduced CSF neurofilament light
by 60% in VALOR."* Cited passage says *"...reduced CSF neurofilament
light by approximately 60% from baseline at week 28."* → `supported`.
If the passage instead said *"60% at week 52"* the difference in
timepoint would drop it to `partially_supported`. If the passage said
*"30%"* it would be `contradicted`.

### 2. `hallucination_rate` — uncited + miscited factual claims

```
hallucination_rate = (uncited + miscited) / total_factual_claims
```

Reported decomposed so you can tell the two apart:

- `uncited_rate`: factual claims with no attached citation
- `miscitation_rate`: factual claims whose citation the grounding
  judge marked `contradicted` or `unsupported`

These have very different remediations. Uncited means the agent
forgot to attribute. Miscited means the agent attributed to a source
that doesn't actually say what's claimed — usually worse, because
readers who spot-check one citation and find it irrelevant lose trust
in the whole answer.

### 3. `citation_traceability` — fraction of citations that resolve

For each citation we try DOI → PMID → PMCID → NCT → URL in order.
Online mode hits doi.org / NCBI E-utilities / clinicaltrials.gov.
Offline mode does shape-only regex validation (useful in CI). The
offline metric is intentionally renamed `citation_traceability_offline_check`
so it can never be confused with the real number.

We also report `required_citation_recall`: fraction of substrings the
gold standard explicitly requires (e.g. `"tofersen"`, `"VALOR"`) that
are present in any cited evidence.

### 4. `task_success` — schema-defined per task type

For `variant_to_strategy` tasks the composite is:

```
task_success_composite = mean(
    correct_target,
    correct_modulation_direction,
    (mechanism_coherence - 1) / 4,
)
```

- `correct_target`: 0/1 — gene/target identified (alias match)
- `correct_modulation_direction`: 0/1 — silence vs activate vs replace vs edit
- `mechanism_coherence`: 1–5 Likert from `mechanism_judge_v1`, scoring
  whether the causal chain from variant → consequence → intervention
  is internally complete

### 5. `answer_completeness` — fraction of gold-standard facets covered

```
answer_completeness = (n_covered + 0.5 * n_partial) / n_expected_facets
```

One LLM call per facet (via `completeness_judge_v1`), with the
half-credit `partial` tier for facets that are alluded to but not
committed-to. Returns `NaN` (not 0) when the case defines no facets,
so it never contributes a spurious zero to aggregates.

### 6. `bias_consistency` — same-judge re-run with rubric order flipped

Every claim is judged twice: once with rubric labels listed in
`supported_first` order and once `unsupported_first`. We report:

- `bias_consistency_rate`: fraction of claims with identical labels
- `bias_swing_to_supported` / `bias_swing_to_unsupported`: directional
  position bias

Below 0.85 is a yellow flag. Below 0.7 means the headline metrics in
this run are not stable — the report says so explicitly via a `bias_unstable`
case flag.

## Dual-rubric reconciliation

Mechanism coherence and evidence sufficiency are scored by *two
independent rubrics* (`mechanism_judge_v1` and `evidence_sufficiency_v1`).
The mechanism rubric asks "does the causal chain hold together?". The
evidence rubric asks "is the cited evidence base sufficient to convince
a domain expert?" These are deliberately separate concerns — an answer
can be mechanistically perfect but built on a single anecdote (evidence
score 3/5, mechanism score 5/5), or it can be well-evidenced but
incoherently reasoned. Scoring both lets you tell the difference
instead of collapsing the diagnostic.

## Reproducibility

Every report stamps the following into `RunMetadata`:

- `judge_model`, `judge_provider`, `extractor_model`,
  `extractor_provider`
- `prompt_versions`: every prompt name → semver version
  (e.g. `grounding_judge_v1 → 1.1.0`). Prompts live in
  `src/bio_rag_eval/prompts/*.md` with YAML frontmatter for the version.
- `bio_rag_eval_version`
- `seed` (for the bootstrap)
- `bootstrap_n_resamples`, `ci_level`
- `config_hash`: SHA-256 of the resolved `RunConfig` — for cache keys
  and to confirm two runs were comparable

All aggregate metrics get percentile-bootstrap CIs at 95% by default
(1000 resamples). When `n < 5` the CI is suppressed (reported as `—`)
rather than emitting a misleadingly tight interval.

## Quick eval on the therapy-agent benchmark — real numbers

Running `python examples/eval_therapy_agent.py --mock` on the 10
curated rare-disease cases (SOD1-ALS, DMD-exon51, SMA, HD, SCD, ATTR,
CF, RPE65-LCA, FH-PCSK9, HAE) — with seed=7, bootstrap n=1000:

| metric | point | 95% CI | n |
|---|---:|---|---:|
| `grounding_rate` | 1.000 | [1.000, 1.000] | 20 claims |
| `hallucination_rate` | 0.000 | [0.000, 0.000] | 10 cases |
| `citation_traceability_offline_check` | 1.000 | [1.000, 1.000] | 10 |
| `required_citation_recall` | 1.000 | [1.000, 1.000] | 10 |
| `task_success_composite` | 0.917 | [0.917, 0.917] | 10 |
| `mechanism_coherence` | 4.000 | [4.000, 4.000] | 10 |
| `answer_completeness` | 1.000 | [1.000, 1.000] | 10 |
| `evidence_sufficiency_score` | 3.000 | [3.000, 3.000] | 10 |
| `bias_consistency_rate` | 1.000 | [1.000, 1.000] | 10 |

These numbers are produced by the `MockJudge` — they show that the
pipeline orchestrates correctly on the curated cases. They do *not*
say anything about a real agent or a real judge. The same script with
`--judge anthropic --model claude-opus-4-7` will produce real numbers
that you should expect to be lower across the board (and bias
consistency well under 1.0).

The cookbook recipes show what real failure modes look like end-to-end:

- [`cookbook/01_clinical_trial_lookup`](./cookbook/01_clinical_trial_lookup/)
  demonstrates a miscitation diagnostic: an agent with
  `citation_traceability = 1.000` but `grounding_rate = 0.286` and
  `miscitation_rate = 0.571`.
- [`cookbook/02_drug_repurposing_screen`](./cookbook/02_drug_repurposing_screen/)
  is an A/B comparison of two RAG configurations on 4 questions and
  shows why bootstrap CIs are non-optional below n≈20.

## Repository layout

```
src/bio_rag_eval/
  schemas/          pydantic models for every structured output
  prompts/          versioned markdown prompts (YAML frontmatter)
  judges/           Anthropic, OpenAI, Mock — all share BaseJudge
  metrics/          grounding, hallucination, citation, task_success,
                    completeness, bias
  runner.py         EvalRunner.run(cases, responses) -> EvalReport
  report.py         HTML + markdown + JSON output
  cli.py            `bio-rag-eval run --cases X --responses Y`
examples/
  eval_therapy_agent.py
  sample_gold_standards/   10 curated rare-disease YAML cases
  sample_responses.jsonl
cookbook/
  01_clinical_trial_lookup/
  02_drug_repurposing_screen/
tests/              40 tests, all metrics + runner + prompts + utils
notebooks/01_demo.ipynb
```

## Limitations — honest

- **LLM-as-judge inherits the judge's biases.** We mitigate position
  bias by re-running every grounding judgment with the rubric order
  flipped (`bias_consistency`). We do not mitigate the broader
  problem that judges are themselves models with knowledge cutoffs,
  domain-specific blind spots, and a tendency to anchor on whichever
  side of an ambiguous quote is listed first. A judge that is wrong
  *systematically* will not flag itself.
- **Gold-standard subjectivity.** The 10 curated cases reflect one
  set of clinical opinions about what "the right answer" looks like.
  For rare-disease cases especially, the literature is moving fast
  enough that an answer correct today may be incomplete in 18 months.
  The `prompt_versions` and `references` fields are how we track
  when the rubric itself needs to evolve.
- **Citation traceability ≠ correctness.** A citation that resolves
  to a real document is necessary; whether that document actually
  supports the claim is the grounding judge's job. We measure them
  separately precisely because a single composite metric would let
  a system game one by sacrificing the other.
- **Bootstrap CIs assume case independence.** If your cases were
  derived from a common source (same review article, same trial),
  the percentile bootstrap will underestimate uncertainty. The
  framework cannot detect this from the data alone — it's a property
  of how the cases were sampled.
- **Mock-judge numbers are not informative.** The MockJudge gives
  deterministic, optimistic-by-construction answers so the pipeline
  can be tested without API calls. Any number reported under
  `--mock` should be treated as a pipeline-smoke-test artifact, not
  a measure of the agent.

## CLI

```bash
bio-rag-eval list-prompts                          # print prompt registry
bio-rag-eval run \
    --cases examples/sample_gold_standards/ \
    --responses examples/sample_responses.jsonl \
    --out reports/ \
    --judge anthropic --model claude-opus-4-7
```

Pass `--citation-online` to actually hit doi.org / NCBI /
clinicaltrials.gov. Pass `--no-bias-check` to skip the swapped-rubric
re-run (cuts judge calls roughly in half).

## Contributing

Tests run with `pytest`. The 40 tests in `tests/` cover every metric,
the runner, the prompt loader, and the bootstrap. New metrics should
ship with: (a) a pydantic schema for their judge output, (b) a
versioned markdown prompt, (c) a `compute_*` function with a
docstring that states the formula, and (d) tests for both the happy
path and at least one edge case (empty denominator, missing
judgment).
