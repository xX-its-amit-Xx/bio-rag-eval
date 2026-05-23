# bio-rag-eval report — run `f8af2b3ff3d1`

- Started: 2026-05-23 05:23:02.278256+00:00
- Finished: 2026-05-23 05:23:02.427408+00:00
- Judge: `mock:mock`
- Extractor: `mock:mock`
- bio-rag-eval version: `0.1.0`
- Config hash: `19b861da1765`
- Seed: 7
- Bootstrap resamples: 1000 @ 95% CI

## Prompt versions
- `claim_extraction_v1` — `1.0.0`
- `completeness_judge_v1` — `1.0.0`
- `evidence_sufficiency_v1` — `1.0.0`
- `grounding_judge_v1` — `1.1.0`
- `mechanism_judge_v1` — `1.0.0`


## Aggregate metrics
| metric | point | 95% CI | n |
|---|---:|---|---:|
| grounding_rate | 0.618 | [0.441, 0.765] | 34 |
| hallucination_rate | 0.179 | [0.104, 0.252] | 8 |
| uncited_rate | 0.000 | [0.000, 0.000] | 8 |
| miscitation_rate | 0.179 | [0.104, 0.252] | 8 |
| forbidden_claim_present | 0.000 | [0.000, 0.000] | 8 |
| citation_traceability_offline_check | 0.812 | [0.625, 0.958] | 8 |
| correct_target | 0.875 | [0.625, 1.000] | 8 |
| correct_modulation_direction | 0.750 | [0.500, 1.000] | 8 |
| mechanism_coherence | 4.000 | [4.000, 4.000] | 8 |
| task_success_composite | 0.792 | [0.667, 0.875] | 8 |
| evidence_sufficiency_score | 3.000 | [3.000, 3.000] | 8 |




## Per-case
### CFTR__F508del — CFTR__F508del (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.600
- hallucination_rate: 0.200
- task_success_composite: 0.917
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 5, judged: 5
### HBB__E6V — HBB__E6V (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.600
- hallucination_rate: 0.200
- task_success_composite: 0.917
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 5, judged: 5
### SMN1__homozygous_deletion — SMN1__homozygous_deletion (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.600
- hallucination_rate: 0.200
- task_success_composite: 0.917
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 5, judged: 5
### TTR__V122I — TTR__V122I (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.750
- hallucination_rate: 0.000
- task_success_composite: 0.917
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 4, judged: 4
### LDLR__c.661C>T — LDLR__c.661C>T (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.750
- hallucination_rate: 0.250
- task_success_composite: 0.583
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 4, judged: 4
### GLA__Fabry — GLA__Fabry (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.500
- hallucination_rate: 0.250
- task_success_composite: 0.583
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 4, judged: 4
### BCL11A__SCD_target — BCL11A__SCD_target (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.333
- hallucination_rate: 0.333
- task_success_composite: 0.583
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 3, judged: 3
### ALAS1__porphyria — ALAS1__porphyria (gold: fda_triples_v1@latest)
- flags: —
- grounding_rate: 0.750
- hallucination_rate: 0.000
- task_success_composite: 0.917
- answer_completeness: nan
- mechanism_coherence: 4.0/5
- claims extracted: 4, judged: 4
