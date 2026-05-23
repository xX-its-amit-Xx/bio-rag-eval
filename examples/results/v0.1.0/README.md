# bio-rag-eval v0.1.0 — example results

This directory contains illustrative output from the integration
pipeline. It is committed to the repo so users can see what the
end-to-end report layout looks like without running anything
themselves.

## What's here

```
synthetic_demo/
  predictions.jsonl         # 8 synthetic TherapyStrategyOutput records
  report/
    scorecard.html          # human-readable
    scorecard.md            # for embedding in therapy-agent README
    raw_metrics.json        # machine-readable aggregate metrics
    run_metadata.json       # reproducibility manifest (prompt hashes etc.)
    per_case/<case>.json    # one detailed JSON per case
```

## What this is NOT

**These numbers are not from a real therapy-agent benchmark run.** They
are produced by running a deterministic `MockJudge` against
hand-written synthetic `TherapyStrategyOutput` records that approximate
what therapy-agent v0.1.0 *would* emit on the FDA-covered cases.

Specifically:

- The MockJudge returns scripted labels (mostly `SUPPORTED`, with one
  `PARTIALLY_SUPPORTED` and one `UNSUPPORTED` cycled in) so the
  scorecard exercises the report's failure-mode visualizations. The
  resulting `grounding_rate` is a mock artifact and does NOT reflect
  what a real Claude or GPT judge would conclude.
- The `predictions.jsonl` was authored by hand to be plausible for
  each gene/variant context but it is not the output of a live agent
  run. The `confidence_score`, `wall_clock_seconds`, and token counts
  are illustrative.
- 8 cases are included — these are the ones whose genes are present
  in `fda_strategy_triples@v0.1.0`. Real therapy-agent v0.1.0 also
  covers SOD1, DMD, HTT, RPE65, and SERPING1; those would be
  evaluated against the curated YAML gold standards in
  `examples/sample_gold_standards/` instead, since the FDA dataset
  doesn't cover them.

## How to regenerate

```bash
# Deterministic, no API key needed:
PYTHONPATH=src python -m bio_rag_eval.cli run \\
    --predictions examples/results/v0.1.0/synthetic_demo/predictions.jsonl \\
    --predictions-adapter therapy_agent_v1 \\
    --gold-source fda_triples_v1 \\
    --output examples/results/v0.1.0/synthetic_demo/report \\
    --judge mock --no-bias-check \\
    --judge-model-version "mock-deterministic-v0"
```

## How to produce a real v0.1.0 result

When you have an ANTHROPIC_API_KEY (or OPENAI_API_KEY) and have run
therapy-agent on its benchmark:

```bash
# 1. Run therapy-agent against its benchmark cases:
ANTHROPIC_API_KEY=sk-... therapy-agent bench \\
    --cases benchmarks/cases/ \\
    --out benchmark_runs/2026-05-23/results.jsonl

# 2. Score with bio-rag-eval against FDA gold:
ANTHROPIC_API_KEY=sk-... bio-rag-eval run \\
    --predictions benchmark_runs/2026-05-23/results.jsonl \\
    --predictions-adapter therapy_agent_v1 \\
    --gold-source fda_triples_v1 \\
    --output examples/results/v0.1.0/real/ \\
    --judge anthropic --model claude-opus-4-7
```

Commit the resulting `examples/results/v0.1.0/real/` directory alongside
this synthetic demo. The two are not directly comparable — the real run
should be the one cited as "v0.1.0 results"; the synthetic demo's job is
just to show the file layout.
