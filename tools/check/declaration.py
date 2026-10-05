"""`tools/check/ci.toml`, the jobs `check` runs, read as data (`load`, `parse`).

A job is a `[[job]]` table:
- `name`: unique;
- `owner`: a handle in the Slack registry (`.agents/skills/using-slack/registry.json`), the agent to ask about the job;
- `group`: check.py's group, numbered from 1; the groups run at once, and a group's jobs run in file order;
- `halts`: optional, its failure stops the run;
- `halts_on`: optional, a line in its output that stops the run;
- `merge_requires`: optional, the changed paths for which `research merge` needs this job to have passed;
- `after`: optional, a job in another group that must end first: this one starts once it has passed or been skipped, and is
  stopped if it fails. The job it names has no `after` of its own.

The file also carries `schema = "ci/v0"`. Anything else is refused: an unknown key, a missing field, a wrong type, a
duplicate name, or an owner the registry doesn't know. Reading the file runs none of the commit's code. Stdlib only.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "ci/v0"
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tools" / "check" / "ci.toml"
REGISTRY = ROOT / ".agents" / "skills" / "using-slack" / "registry.json"
FIELDS = {"name": str, "owner": str, "group": int, "halts": bool, "halts_on": str, "merge_requires": list, "after": str}
REQUIRED = ("name", "owner", "group")


class DeclarationError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    name: str
    owner: str
    group: int
    halts: bool = False
    halts_on: str | None = None
    merge_requires: tuple[str, ...] = ()
    after: str | None = None


def parse(text: str, handles: set[str], source: str = "ci.toml") -> list[Job]:
    """The jobs ``text`` declares, in file order; ``handles`` are the owners allowed."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise DeclarationError(f"{source}: not TOML: {e}") from None
    if doc.get("schema") != SCHEMA:
        raise DeclarationError(f"{source}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    if unknown := set(doc) - {"schema", "job"}:
        raise DeclarationError(f"{source}: unknown keys {sorted(unknown)}")
    jobs, seen = [], set()
    for i, t in enumerate(doc.get("job", [])):
        where = f"{source}: job {i + 1}" + (f" ({t['name']})" if isinstance(t.get("name"), str) else "")
        if unknown := set(t) - set(FIELDS):
            raise DeclarationError(f"{where}: unknown keys {sorted(unknown)}")
        if missing := [k for k in REQUIRED if k not in t]:
            raise DeclarationError(f"{where}: missing {', '.join(missing)}")
        for k, v in t.items():
            if not isinstance(v, FIELDS[k]) or (FIELDS[k] is int and isinstance(v, bool)):
                raise DeclarationError(f"{where}: {k} must be {FIELDS[k].__name__}, not {type(v).__name__}")
        if not all(isinstance(p, str) and p for p in t.get("merge_requires", [])):
            raise DeclarationError(f"{where}: merge_requires must be non-empty path strings")
        if t["group"] < 1:
            raise DeclarationError(f"{where}: group must be 1 or more")
        if t["name"] in seen:
            raise DeclarationError(f"{where}: duplicate name")
        if t["owner"] not in handles:
            raise DeclarationError(f"{where}: owner {t['owner']!r} isn't a handle in the registry ({', '.join(sorted(handles))})")
        seen.add(t["name"])
        jobs.append(Job(t["name"], t["owner"], t["group"], t.get("halts", False), t.get("halts_on"),
                        tuple(t.get("merge_requires", ())), t.get("after")))
    if not jobs:
        raise DeclarationError(f"{source}: declares no job")
    by = {j.name: j for j in jobs}
    for j in jobs:
        if j.after is None:
            continue
        if (a := by.get(j.after)) is None:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, which isn't declared")
        if a.group == j.group:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, in its own group, whose order already runs it first")
        if a.after is not None:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, which itself waits on {a.after!r}")
    return jobs


def handles(registry: Path = REGISTRY) -> set[str]:
    return {h["handle"] for h in json.loads(registry.read_text())["handles"]}


def load(path: Path = PATH, registry: Path = REGISTRY) -> list[Job]:
    return parse(path.read_text(), handles(registry), str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
