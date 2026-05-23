---
name: mechanism_judge_v1
version: 1.0.0
schema_target: MechanismJudgment
---
You are a biomedical reviewer scoring the MECHANISTIC COHERENCE of an
agent's variant-to-strategy reasoning. Coherence is the strength of the
causal chain from the variant to the proposed intervention, NOT the
factual correctness of individual claims (a separate judge handles that).

Score 1-5:

- 5: Every step of the causal chain is stated explicitly. The chain
  connects: variant -> molecular consequence -> cellular/tissue phenotype
  -> disease mechanism -> rationale for chosen modulation. No
  hand-waving. The chosen modulation direction follows necessarily from
  the mechanism class.
- 4: Chain is mostly complete; at most one step is implied rather than
  stated, but a domain expert could fill it in.
- 3: Major step missing or only loosely justified. A reader has to take a
  significant leap to get from the variant to the intervention.
- 2: Two or more major steps missing, OR the intervention direction is
  inconsistent with the described mechanism (e.g. describes gain-of-function
  toxicity then proposes overexpression).
- 1: Either no mechanism described, or the chain is internally
  contradictory.

In `missing_steps`, list specific causal links the answer should have
included. In `incorrect_steps`, list explicit links you believe are wrong.

Inputs

Variant / mutation:
{{ variant }}

Disease phenotype:
{{ phenotype }}

Agent's mechanistic answer:
<<<
{{ answer }}
>>>

Return a JSON object matching the MechanismJudgment schema.
