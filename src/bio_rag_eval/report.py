"""Report generation — markdown + HTML from an `EvalReport`.

Both formats are produced from the same Jinja2 templates (loaded as
strings rather than files to keep the package single-import). The HTML
template is a self-contained page (no external CSS) so it can be
emailed or attached to a PR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment

from bio_rag_eval.schemas.eval_result import EvalReport

_MD = """\
# bio-rag-eval report — run `{{ r.metadata.run_id }}`

- Started: {{ r.metadata.started_at }}
- Finished: {{ r.metadata.finished_at }}
- Judge: `{{ r.metadata.judge_provider }}:{{ r.metadata.judge_model }}`
- Extractor: `{{ r.metadata.extractor_provider }}:{{ r.metadata.extractor_model }}`
- bio-rag-eval version: `{{ r.metadata.bio_rag_eval_version }}`
- Config hash: `{{ r.metadata.config_hash }}`
- Seed: {{ r.metadata.seed }}
- Bootstrap resamples: {{ r.metadata.bootstrap_n_resamples }} @ {{ (r.metadata.ci_level * 100) | int }}% CI

## Prompt versions
{% for name, version in r.metadata.prompt_versions.items() -%}
- `{{ name }}` — `{{ version }}`
{% endfor %}

## Aggregate metrics
| metric | point | {{ (r.metadata.ci_level * 100) | int }}% CI | n |
|---|---:|---|---:|
{% for name, m in r.aggregate_metrics.items() -%}
| {{ name }} | {{ "%.3f" | format(m.point) }} | {% if m.ci_low is not none %}[{{ "%.3f" | format(m.ci_low) }}, {{ "%.3f" | format(m.ci_high) }}]{% else %}—{% endif %} | {{ m.n }} |
{% endfor %}

{% if r.bias_consistency %}
## Bias consistency (swapped-rubric re-run)
| metric | point | CI | n |
|---|---:|---|---:|
{% for name, m in r.bias_consistency.items() -%}
| {{ name }} | {{ "%.3f" | format(m.point) }} | {% if m.ci_low is not none %}[{{ "%.3f" | format(m.ci_low) }}, {{ "%.3f" | format(m.ci_high) }}]{% else %}—{% endif %} | {{ m.n }} |
{% endfor %}
{% endif %}

## Per-case
{% for cr in r.case_results -%}
### {{ cr.case_id }} — {{ cr.case_name }}
{% if cr.error -%}**ERROR: {{ cr.error }}**{%- else -%}
- flags: {{ cr.flags | join(", ") if cr.flags else "—" }}
- grounding_rate: {{ "%.3f" | format(cr.metrics.get("grounding_rate", float("nan"))) }}
- hallucination_rate: {{ "%.3f" | format(cr.metrics.get("hallucination_rate", float("nan"))) }}
- task_success_composite: {{ "%.3f" | format(cr.metrics.get("task_success_composite", float("nan"))) }}
- answer_completeness: {{ "%.3f" | format(cr.metrics.get("answer_completeness", float("nan"))) }}
- mechanism_coherence: {{ "%.1f" | format(cr.metrics.get("mechanism_coherence", float("nan"))) }}/5
- claims extracted: {{ cr.extracted_claims | length }}, judged: {{ cr.grounding_judgments | length }}
{%- endif %}
{% endfor %}
"""

_HTML = """\
<!doctype html>
<html><head><meta charset="utf-8"><title>bio-rag-eval — {{ r.metadata.run_id }}</title>
<style>
body { font: 14px/1.45 -apple-system,BlinkMacSystemFont,sans-serif; max-width: 1100px; margin: 24px auto; color:#222; }
h1,h2,h3 { color:#1a3a5a; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border:1px solid #ddd; padding:6px 10px; text-align:left; }
th { background:#f3f5f8; }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.metric-good { color:#0a7a35; }
.metric-bad { color:#a30; font-weight:600; }
.flag { display:inline-block; background:#fde8e8; color:#a30; padding:2px 6px; border-radius:3px; margin-right:4px; font-size:12px; }
details { margin:8px 0; }
pre.snippet { background:#f7f8fa; padding:8px; border-left:3px solid #aac; white-space:pre-wrap; }
.meta { color:#666; font-size:12px; }
</style></head><body>
<h1>bio-rag-eval — run <code>{{ r.metadata.run_id }}</code></h1>
<p class="meta">
{{ r.metadata.started_at }} &middot;
judge <code>{{ r.metadata.judge_provider }}:{{ r.metadata.judge_model }}</code> &middot;
extractor <code>{{ r.metadata.extractor_provider }}:{{ r.metadata.extractor_model }}</code> &middot;
bio-rag-eval v{{ r.metadata.bio_rag_eval_version }} &middot;
config <code>{{ r.metadata.config_hash }}</code>
</p>

<h2>Prompt versions</h2>
<table><tr><th>name</th><th>version</th></tr>
{% for name, v in r.metadata.prompt_versions.items() -%}
<tr><td><code>{{ name }}</code></td><td><code>{{ v }}</code></td></tr>
{%- endfor %}
</table>

<h2>Aggregate metrics</h2>
<table>
<tr><th>metric</th><th class="num">point</th><th>{{ (r.metadata.ci_level * 100) | int }}% CI</th><th class="num">n</th></tr>
{% for name, m in r.aggregate_metrics.items() -%}
<tr><td>{{ name }}</td>
<td class="num"><strong>{{ "%.3f" | format(m.point) }}</strong></td>
<td>{% if m.ci_low is not none %}[{{ "%.3f" | format(m.ci_low) }}, {{ "%.3f" | format(m.ci_high) }}]{% else %}—{% endif %}</td>
<td class="num">{{ m.n }}</td></tr>
{%- endfor %}
</table>

{% if r.bias_consistency %}
<h2>Bias consistency</h2>
<p class="meta">Re-run with the rubric labels listed in reversed order. Below 0.85 = yellow flag, below 0.7 = headline metrics unstable.</p>
<table>
<tr><th>metric</th><th class="num">point</th><th>CI</th><th class="num">n</th></tr>
{% for name, m in r.bias_consistency.items() -%}
<tr><td>{{ name }}</td>
<td class="num">{{ "%.3f" | format(m.point) }}</td>
<td>{% if m.ci_low is not none %}[{{ "%.3f" | format(m.ci_low) }}, {{ "%.3f" | format(m.ci_high) }}]{% else %}—{% endif %}</td>
<td class="num">{{ m.n }}</td></tr>
{%- endfor %}
</table>
{% endif %}

<h2>Per-case</h2>
{% for cr in r.case_results -%}
<details>
<summary><strong>{{ cr.case_id }}</strong> — {{ cr.case_name }}
{% for f in cr.flags %}<span class="flag">{{ f }}</span>{% endfor %}
</summary>
{% if cr.error %}<p class="metric-bad">ERROR: {{ cr.error }}</p>{% else %}
<table>
<tr><th>metric</th><th class="num">value</th></tr>
{% for k, v in cr.metrics.items() -%}
<tr><td>{{ k }}</td><td class="num">{{ "%.3f" | format(v) if v == v else "NaN" }}</td></tr>
{%- endfor %}
</table>
<h4>Claims &amp; grounding</h4>
{% for c in cr.extracted_claims -%}
<details><summary>[{{ c.claim_id }}] <em>{{ c.claim_type.value }}</em> — {{ c.text | truncate(140) }}</summary>
{% set j = (cr.grounding_judgments | selectattr("claim_id", "equalto", c.claim_id) | list | first) %}
{% if j %}
<p><strong>{{ j.label.value }}</strong> (conf {{ "%.2f" | format(j.confidence) }})</p>
<p>{{ j.rationale }}</p>
{% if j.quoted_evidence %}<pre class="snippet">{{ j.quoted_evidence }}</pre>{% endif %}
{% else %}<p class="meta">no judgment</p>{% endif %}
</details>
{%- endfor %}
{% endif %}
</details>
{%- endfor %}
</body></html>
"""


def render_markdown(report: EvalReport) -> str:
    env = Environment(autoescape=False)
    return env.from_string(_MD).render(r=report, float=float)


def render_html(report: EvalReport) -> str:
    env = Environment(autoescape=True)
    return env.from_string(_HTML).render(r=report)


def write_report(report: EvalReport, out_dir: str | Path) -> dict[str, Path]:
    """Write `<run_id>.md`, `<run_id>.html`, and `<run_id>.json` into `out_dir`.

    Returns the paths it wrote, keyed by extension.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = report.metadata.run_id or "report"
    paths: dict[str, Path] = {
        "md": out / f"{base}.md",
        "html": out / f"{base}.html",
        "json": out / f"{base}.json",
    }
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return paths


def _format_metric(value: dict[str, Any]) -> str:
    if value.get("ci_low") is not None:
        return f"{value['point']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}]"
    return f"{value['point']:.3f}"
