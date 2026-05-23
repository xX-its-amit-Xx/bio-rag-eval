"""Versioned-prompt registry.

Every judge call goes through `load_prompt(name)`. The loader reads the
markdown file with the matching `name` field in frontmatter, parses the
version, and returns `(template, version)`. Downstream code stamps the
version into `RunMetadata.prompt_versions` so every metric in a report is
traceable back to the exact prompt text that produced it.

Prompt files live as `*.md` siblings of this file. They use Jinja2 syntax
for variable substitution (`{{ var }}` and `{% for %}` blocks).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

_PROMPTS_DIR = Path(__file__).parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt template plus the version pulled from its frontmatter."""

    name: str
    version: str
    template: str
    schema_target: str  # name of the pydantic schema class the judge must return

    def render(self, **kwargs: Any) -> str:
        env = Environment(undefined=StrictUndefined, autoescape=False)
        return env.from_string(self.template).render(**kwargs)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("Prompt file is missing --- frontmatter ---")
    head, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in head.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"')
    return meta, body


def load_prompt(name: str) -> Prompt:
    """Load a prompt by its registry name. Raises if absent or malformed."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt registered as '{name}' at {path}")
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    for required in ("name", "version", "schema_target"):
        if required not in meta:
            raise ValueError(f"Prompt {name}: missing frontmatter field '{required}'")
    if meta["name"] != name:
        raise ValueError(
            f"Prompt {name}: frontmatter name '{meta['name']}' does not match filename"
        )
    return Prompt(
        name=meta["name"],
        version=meta["version"],
        template=body,
        schema_target=meta["schema_target"],
    )


def list_prompts() -> dict[str, str]:
    """Return name -> version for every registered prompt."""
    out: dict[str, str] = {}
    for p in _PROMPTS_DIR.glob("*.md"):
        try:
            prompt = load_prompt(p.stem)
        except Exception:
            continue
        out[prompt.name] = prompt.version
    return out


def prompt_content_hashes() -> dict[str, str]:
    """Return name -> "<scheme>:<hex>" identifying the exact bytes of each prompt.

    Tries `git hash-object <path>` first (so the hash matches a git blob
    SHA-1 if/when the file is committed) and falls back to
    `sha256:<hex>` of the file content when git is unavailable or the
    file is outside a repo. Either way, the value uniquely pins the
    prompt's exact bytes — that's what reproducibility actually needs,
    not the human-curated frontmatter version.
    """
    out: dict[str, str] = {}
    for p in _PROMPTS_DIR.glob("*.md"):
        try:
            name = load_prompt(p.stem).name
        except Exception:
            continue
        out[name] = _hash_file(p)
    return out


def _hash_file(path: Path) -> str:
    """git-blob-SHA1 if git is on PATH (and works for files inside or
    outside a repo via `git hash-object`); sha256 of the file bytes
    otherwise. Both schemes are content-addressable; the prefix
    disambiguates so callers can tell which to trust against a remote."""
    try:
        result = subprocess.run(
            ["git", "hash-object", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"git:{result.stdout.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{h}"


__all__ = ["Prompt", "load_prompt", "list_prompts", "prompt_content_hashes"]
