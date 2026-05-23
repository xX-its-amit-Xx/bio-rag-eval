---
name: evidence_sufficiency_v1
version: 1.0.0
schema_target: EvidenceSufficiency
---
You are scoring the SUFFICIENCY of the cited evidence base — separately
from whether individual claims are grounded. The question: if a domain
expert read ONLY the cited sources, would they find the recommendation
defensible?

This rubric is intentionally independent of mechanism_judge_v1 — we score
both and reconcile downstream. Reconciliation is what reduces single-
rubric bias.

Score 1-5:

- 5: ≥2 independent primary studies in the relevant species (human if a
  clinical recommendation), at least one with a confirmatory cohort.
  Sufficient for an expert reader.
- 4: ≥1 primary study in the relevant species plus supporting mechanistic
  literature. An expert would want more replication but the case is real.
- 3: Single primary study, OR multiple reviews citing the same underlying
  study (apparent redundancy). The recommendation rests on one piece of
  evidence.
- 2: Only secondary sources (reviews, guidelines) — no primary literature
  cited.
- 1: No cited sources, sources unresolvable, or sources do not match the
  claim domain (e.g. cited rodent paper for a human recommendation).

`n_independent_sources` counts non-overlapping primary studies. Two papers
from the same lab on the same cohort count as 1. A review and a primary
study it summarizes count as 1.

`has_human_data` is true iff at least one cited source reports findings in
humans (clinical trial, observational study, case series, biobank). In
vitro studies on human cell lines do NOT count.

Inputs

Recommendation / answer summary:
{{ answer_summary }}

Cited sources:
{% for c in citations -%}
[{{ c.citation_id }}] {{ c.title or "(untitled)" }}
  doi: {{ c.doi or "—" }}, pmid: {{ c.pmid or "—" }}, nct: {{ c.nct_id or "—" }}
  snippet: {{ (c.snippet or "(no snippet)")[:400] }}
---
{% endfor %}

Return a JSON object matching the EvidenceSufficiency schema.
