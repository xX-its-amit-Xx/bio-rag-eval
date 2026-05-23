# bio-rag-eval cookbook

Two end-to-end recipes that demonstrate how to use `bio-rag-eval` on real
biomedical RAG workloads. Each recipe is fully self-contained: you can
copy the directory, install `bio-rag-eval`, and run.

| Recipe | What it shows |
|---|---|
| [`01_clinical_trial_lookup`](./01_clinical_trial_lookup/) | Catching a *miscitation* failure mode in a clinical-trial-matching RAG agent. The agent cites a real NCT ID — but the trial's eligibility criteria don't match what the agent claims. Demonstrates `hallucination_rate` decomposition (`uncited_rate` vs `miscitation_rate`) and `citation_traceability`. |
| [`02_drug_repurposing_screen`](./02_drug_repurposing_screen/) | Comparing two RAG agents (single-doc retrieval vs multi-doc reranker) on a 12-case drug-repurposing question set. Demonstrates `task_success_composite`, `evidence_sufficiency`, and `bias_consistency` — and how to interpret bootstrap CIs when comparing two systems. |

## Running a recipe

```bash
pip install -e ..[dev,notebook]
cd 01_clinical_trial_lookup
python run.py --mock     # uses MockJudge, no API key needed
# or:
ANTHROPIC_API_KEY=sk-... python run.py --judge anthropic
```

Each recipe writes its report to `./reports/`. The `--mock` mode is
deterministic and is what's used to produce the numbers committed in
each recipe's `expected_output.md`.
