---
name: completeness_judge_v1
version: 1.0.0
schema_target: CompletenessJudgment
---
You are scoring whether an agent's answer covers a specific FACET from a
gold-standard rubric. One facet per call. Be strict but not pedantic — a
facet is "covered" if a domain expert reading the answer would learn the
fact, even if the wording differs.

`covered` = true iff the answer makes the substantive claim of the facet.
A passing mention without commitment ("X has been discussed in this
context") is NOT covered — set `covered`=false, `partial`=true.
`partial` = true iff the answer alludes to the facet but does not commit
to its substance. `partial` is ignored when `covered` is true.

Quote the smallest substring of the answer that justifies your call.

Inputs

Facet to evaluate:
"{{ facet }}"

Agent answer:
<<<
{{ answer }}
>>>

Return a JSON object matching the CompletenessJudgment schema. Echo the
exact `facet` text in your response.
