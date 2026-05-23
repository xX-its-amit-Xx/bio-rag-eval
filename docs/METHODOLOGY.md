# bio-rag-eval — Methodology

A written record of what each metric measures, why it was chosen,
which biases it carries, and the literature it draws from. The
audience is a future evaluator (you, in six months; a teammate; a
peer reviewer) who needs to decide whether a number from this harness
should be trusted for a specific decision.

## Why this document exists

Numerical metrics are persuasive even when they should not be. The
combination of LLM-as-judge scoring + bootstrap CIs + clean-looking
HTML reports can make a brittle result look definitive. This document
exists so that every metric in a bio-rag-eval report can be traced
back to the assumption that produced it.

If you disagree with a methodology choice here, the right response is
to fork the prompt or metric and bump its version — not to silently
re-tune the constants. The whole pipeline is built so that two
incomparable runs cannot be silently averaged together.

## The six metrics

### `grounding_rate`

**Formula.**

    grounding_rate = |{c in claims : judge(c) = SUPPORTED}| / |graded_claims|

`graded_claims` excludes opinion-typed claims (claim type `opinion`).
`weak_grounding_rate` counts `partially_supported` at weight 0.5.

**What it answers.** "Of the empirical claims this agent made, what
fraction are explicitly entailed by the specific source the agent
attributed them to?"

**Why it was chosen.** Standard retrieval metrics (precision@k,
nDCG) measure whether the *right document* was returned. They tell
you nothing about whether the agent actually used that document
correctly. A RAG agent can retrieve a perfect snippet and then
hallucinate a claim that's adjacent-but-different ("60% reduction in
adults" vs the snippet's "60% reduction in adolescents"). Grounding
catches that.

**Known biases.** The grounding judge inherits the judge LLM's
biases:

- *Position bias* — when the prompt lists labels in a fixed order,
  the model can over-pick the first option. We mitigate this with
  the `bias_consistency` metric (see below). Zheng et al. 2023
  ("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS
  2023) document position bias at 5-15% on strong models.
- *Length bias* — judges tend to favor longer snippets as evidence,
  even when a shorter snippet entails the claim more precisely. We
  do not currently mitigate this; partial mitigation would be to
  truncate all snippets to a fixed length before judging.
- *Self-preference* — when the judge is the same model family as the
  agent under test, agreement is inflated. The
  `RunMetadata.judge_model` is recorded so cross-run comparisons can
  spot when the judge identity changed.

**Literature.** Zheng et al. 2023; Wang et al. 2023 ("Large Language
Models are not Fair Evaluators", arXiv:2305.17926); Min et al. 2023
("FActScore", arXiv:2305.14251) which inspired the
sentence-decomposition approach used in `claim_extraction_v1`.

### `hallucination_rate`

**Formula.**

    hallucination_rate = (uncited + miscited) / total_factual_claims

Decomposed into `uncited_rate` (factual claim with empty `cited_ids`)
and `miscitation_rate` (factual claim whose grounding judgment is
`contradicted` or `unsupported`).

**What it answers.** "What fraction of factual claims are unsupported
in the strongest sense — either the agent declined to cite, or it
cited a source that doesn't entail the claim?"

**Why the decomposition.** Uncited and miscited claims have very
different remediations. Uncited usually means the agent's
attribution layer failed. Miscited usually means the agent retrieved
plausibly-relevant but irrelevant content and then asserted a claim
about it anyway. The latter is worse for clinical reliability: a
reader who spot-checks one citation and finds it irrelevant tends to
lose trust in the entire answer.

**Known biases.** Hallucination metrics depend on the grounding
judge's `unsupported` label, which is the most subjective of the four
grounding labels. Annotators in Min et al. 2023 reported
inter-annotator agreement on related labels at κ ≈ 0.6; expect
similar floor for any single-judge run.

### `citation_traceability`

**Formula.**

    citation_traceability = |{c : resolves(c)}| / |citations|

`resolves` tries DOI → PMID → PMCID → NCT → URL in that order. Online
mode does a real HTTP GET; offline mode does shape-only regex
validation and the metric is renamed
`citation_traceability_offline_check` so it cannot be confused with
the real number.

**What it answers.** "If a reader clicked through every citation,
what fraction would lead to a real document?"

**Why this is necessary but not sufficient.** Traceability is
trivially gameable — an agent can cite well-formed-but-irrelevant
identifiers and score 1.0 here. The grounding metric is what
discriminates relevant citation from real-but-wrong citation.

**Known biases.** None on the metric itself. The online resolver
depends on third-party uptime; transient 5xx errors will count as
unresolved.

### `task_success_composite`

**Formula** (for `task_type = variant_to_strategy`):

    task_success_composite = mean(
        correct_target,                              # 0/1
        correct_modulation_direction,                # 0/1
        (mechanism_coherence - 1) / 4,               # 1-5 -> 0-1
    )

For other task types only the components that are gold-defined
contribute. NaN propagates when the mechanism judge was skipped.

**What it answers.** "Did the agent get the substantive answer right
on three orthogonal axes: the right target, the right direction of
modulation, and a coherent causal chain?"

**Why three axes and not one.** A single 1-5 likert collapses
"named the wrong drug" with "named the right drug but invented a
backwards mechanism." Decomposing forces the failure mode to surface:
in the cookbook recipes, Agent A and Agent B differ much more on the
mechanism axis than on the target axis, which is exactly the
diagnostic the user needs.

**Known biases.** `correct_target` and `correct_modulation_direction`
use case-insensitive substring matching against the answer prose. A
hostile agent could pad its answer with every plausible target name
and score 1.0 spuriously; in practice no agent we've measured does
this, but if you suspect adversarial output you should switch to the
adapter-mode flow where `structured_outputs["predicted_target"]` is
checked directly.

**Mechanism coherence.** Scored on the 1-5 Likert in
`mechanism_judge_v1`, which is intentionally narrow: it asks "does
the causal chain hold together?", not "is the claim true?" A 5/5
mechanism can still be factually wrong (caught separately by
grounding).

### `answer_completeness`

**Formula.**

    answer_completeness = (n_covered + 0.5 * n_partial) / n_expected_facets

One judge call per facet in `case.expected_facets`, using
`completeness_judge_v1`. `partial = True` requires the answer to
allude to the facet without committing to its substance.

**What it answers.** "Of the substantive facts a domain expert
expects to see in a complete answer, how many did the agent cover?"

**Why facet-by-facet and not embedding similarity.** Embedding
similarity confuses topical overlap with substantive coverage — an
answer that lists every relevant gene name but never commits to a
mechanism can score high on similarity and low on completeness. The
per-facet judge call asks an explicit yes/no question per facet,
which is what we actually want to measure.

**Known biases.** Sensitive to how facets are phrased. A facet
written as a single concept ("HbS polymerizes under deoxygenation")
is harder to call partial than a compound facet ("HbS polymerizes
under deoxygenation, distorting red cells and causing vaso-occlusion"
— three facts in one). We recommend keeping facets atomic.

### `bias_consistency`

**Formula.**

    bias_consistency_rate = |{c : label_A(c) = label_B(c)}| / |claims|

Where `label_A` is the grounding judgment with the rubric listed in
`supported_first` order, `label_B` the same with `unsupported_first`.

**What it answers.** "Is the headline `grounding_rate` itself
reliable, or does it swing under cosmetic prompt changes?"

**Why this is critical and not just nice-to-have.** All other
metrics in this harness are *downstream* of LLM-as-judge outputs.
If the judge is itself biased, the metrics are biased, and the
CIs you compute on those metrics are CIs on the bias. The bias
re-run is the only thing in the harness that puts a sanity check
on the judge itself.

**Thresholds we recommend.**

- ≥ 0.85: metrics in this run are trustworthy at face value.
- 0.70–0.85: yellow flag. Interpret directional differences only.
- < 0.70: do not report headline numbers. Either swap judges, tighten
  the prompt (`grounding_judge_v1` exposes `rubric_order` so this is
  a no-op refactor), or report only per-claim labels with the
  qualitative breakdown.

**Literature.** Wang et al. 2023 (above); Pezeshkpour & Hruschka 2023
("Large Language Models Sensitivity to the Order of Options in
Multiple-Choice Questions", arXiv:2308.11483).

## Cross-metric design choices

### Dual-rubric reconciliation

Mechanism coherence (`mechanism_judge_v1`) and evidence sufficiency
(`evidence_sufficiency_v1`) are intentionally independent rubrics
scoring overlapping questions. The mechanism rubric asks "does the
causal chain hold together?". The evidence rubric asks "is the cited
evidence base sufficient to convince a domain expert?". These should
be *correlated* on a well-functioning agent, but their disagreement
is itself diagnostic: an answer that scores 5/5 on mechanism and 2/5
on evidence is a well-reasoned but under-sourced claim, and that's
useful to know.

We do NOT combine them into a single composite score. Combining would
hide exactly the disagreements that are diagnostic.

### Bootstrap CIs

All aggregate metrics get percentile-bootstrap CIs at 95% (1000
resamples by default). When `n < 5` the CI is reported as `—` instead
of a misleadingly tight interval — this is conservative; the
percentile bootstrap is known to be poorly behaved at very small
samples. See Efron & Tibshirani 1993, *An Introduction to the
Bootstrap*, §13.

The bootstrap unit differs per metric: `grounding_rate` bootstraps
over CLAIMS (because that's the resolution at which the metric is
defined); all other metrics bootstrap over CASES. This is recorded
in `MetricValue.notes` so a reader can tell at a glance.

### Reproducibility manifest

Every report carries `RunMetadata` with:

- `judge_model`, `judge_provider`, `extractor_model`,
  `extractor_provider`
- `judge_model_version` — provider-reported version (often a
  date-suffixed model id like `claude-opus-4-7@20260101`). This is
  *separate* from `judge_model` because providers update silently
  under stable IDs.
- `prompt_versions` — semver from frontmatter (curator-facing)
- `prompt_hashes` — `git:<sha1>` or `sha256:<hex>` of each prompt
  file's exact bytes (the actual reproducibility pin)
- `bio_rag_eval_version`
- `seed` / `random_seed` (aliased)
- `bootstrap_n_resamples`, `ci_level`
- `predictions_adapter`, `gold_source` — names of the adapters used
  (when adapter-mode CLI was used)
- `config_hash` — SHA-256 of the resolved RunConfig

Two runs are *comparable* iff they share `bio_rag_eval_version`,
`prompt_hashes`, and `judge_model_version`. Anything else and a
direct numeric comparison is suspect — the report's HTML output
shows when these differ between two runs you're trying to compare.

## Adapter-driven runs

When invoked via the adapter CLI (`--predictions-adapter X
--gold-source Y`), the run carries an additional integrity property:

The adapter's `SOURCE_SCHEMA_VERSION` is checked against the
`schema_version` field of each prediction record. Mismatches are
logged but don't fail the run by default — this is conservative,
since adapter-side migrations are a normal part of the integration
lifecycle. Set `STRICT_SCHEMA_VERSION=1` in the environment to make
mismatches fatal.

The `fda_triples_v1` gold source intentionally has partial coverage —
its release pins to the validated FDA drug-target-mechanism subset.
Cases not covered by the FDA dataset are dropped from the run with a
warning rather than silently scored against `None`. This is the
right failure mode: the dataset is the source of ground truth, and
"this case is out of scope" is a real and important answer.

## What this harness does NOT measure

- *Style or readability.* A perfectly-grounded answer in dense
  scientific prose scores the same as the same answer written for a
  patient audience. If audience-appropriateness matters, that's a
  separate metric you'd add.
- *Calibration of agent self-reported confidence.* The
  `min_confidence` field on a `GoldStandardCase` is recorded but not
  scored automatically; you can post-process the report's
  `case_results[*].raw_outputs.confidence` against `metrics.task_success_composite`
  to plot reliability curves. We chose not to bake this in because
  the right reliability metric (ECE vs Brier vs Platt-scaled AUROC)
  depends on the downstream decision.
- *Cost.* `RunMetadata` records token counts when the judge backend
  surfaces them; cost-per-decision is a function of pricing that
  changes outside our control.
- *Latency.* Recorded as `latency_ms` on AgentResponse but not
  aggregated; throughput is a separate orthogonal concern.

## References

- Zheng L. et al. 2023. *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.
- Wang P. et al. 2023. *Large Language Models are not Fair
  Evaluators.* arXiv:2305.17926.
- Pezeshkpour P. & Hruschka E. 2023. *Large Language Models
  Sensitivity to the Order of Options in Multiple-Choice
  Questions.* arXiv:2308.11483.
- Min S. et al. 2023. *FActScore: Fine-grained Atomic Evaluation of
  Factual Precision in Long Form Text Generation.* arXiv:2305.14251.
- Es S. et al. 2024. *RAGAS: Automated Evaluation of Retrieval
  Augmented Generation.* arXiv:2309.15217.
- Saad-Falcon J. et al. 2024. *ARES: An Automated Evaluation
  Framework for Retrieval-Augmented Generation Systems.*
  arXiv:2311.09476.
- Efron B. & Tibshirani R. 1993. *An Introduction to the Bootstrap.*
  CRC Press.
