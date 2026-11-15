"""`tools/verity/check/ci.toml`, the jobs `check` runs, read as data (`load`, `parse`).

A job is a `[[job]]` table:
- `name`: unique;
- `owner`: a handle in the Slack registry (`.agents/skills/using-slack/registry.json`), the agent to ask about the job;
- `group`: check.py's group, numbered from 1; the groups run at once, and a group's jobs run in file order;
- `halts_on`: optional, a line in its output that stops the run;
- `merge_requires`: optional, the changed paths for which `research merge` needs this job to have passed;
- `needs`: optional, the earlier jobs of its group whose output it reads in the checkout, so a shard that runs it runs them.
- `after`: optional, a job in another group that must end first: this one starts once it has passed or been skipped, and is
  stopped if it fails. The job it names has no `after` of its own;
- `part`: optional, one of `PARTS`, the part of the check (`check.py PART...`) the job belongs to (`select`); `check.py
  --deferred JOB,...` names jobs instead (`needed`);
- `shards`: optional, for a job in `SHARDED` only, N >= 1: its work in N shards, its parts `K/N` for K = 1..N (`job_parts`).

The file also carries `schema = "ci/v0"`. Anything else is refused: an unknown key, a missing field, a wrong type, a
duplicate name, or an owner the registry doesn't know. Reading the file runs none of the commit's code. Stdlib only.

Which jobs gate a merge is drive's policy, not this file's: `policy.toml`'s `off_gate` (`POLICY`, `off_gate`) names the jobs off
the merge gate, each a `Job` with `gate` false (`gated`): a recorded check, whole or a shard, skips it, `research merge` requires
it of no check, and it runs in drive's after-landing run (`check.py --deferred`), whose red names its culprit. Such a job has
no `merge_requires`, and no job on the gate `needs` it or runs `after` it.

A `[[plan]]` is one way to split a check over several machines (`check.py --shard K/N`): `shards`, a list of N lists of
selectors, shard K the K-th. A selector is a job's name, or for a job in `SPLIT`, `JOB:PART` (one part: for `lean-audit`, a
Lake package with a `lean-audit.json`; for `circuit-check`, one of its `shards`, `K/N`; for `lean-suites` and `pytest`, one of
its suites, or of one that declares `[tool.verity.tests] parts = P`, one of its parts by test id, `SUITE:K/P`) or `JOB:*` (its
parts no other selector of the plan names). pytest has parts only in a plan that names one (`check.py`'s `_parts_of`): in
any other it runs whole. Every job on the gate is in exactly one shard, whole or, for a split job, as named parts
plus exactly one `*`, and a job off it in at most one (a shard skips it either way: a plan that keeps it puts it back on the
gate in one line); a shard that holds a job whole holds what it `needs` or runs `after`, and one that holds parts of a split
job runs what that job needs beside them when it doesn't hold it (`check.py`'s `groups`: lean-build beside lean-suites'
parts), whose verdict `research merge` takes only from the shard that holds it; no two plans have the same N (`plans`). `resolve`
turns a plan into each shard's parts for a tree's parts (`parts_of`), and refuses a named part the tree lacks, so a stale plan
stops the shard instead of leaving something out. A spec package's audit and its provers' (the packages its `proved_in` names,
which it audits beside it from one build of theirs) resolve onto one shard, the one the plan names any of them in, so the
provers build once a run; a plan naming them in two shards is refused.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "ci/v0"
ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "tools" / "verity" / "check" / "ci.toml"
#: drive's policy (`research.policy`), whose `off_gate` takes jobs off the merge gate
POLICY = ROOT / "tools" / "verity" / "check" / "policy.toml"
REGISTRY = ROOT / ".agents" / "skills" / "using-slack" / "registry.json"
FIELDS = {"name": str, "owner": str, "group": int, "halts_on": str, "merge_requires": list, "needs": list,
          "after": str, "part": str, "shards": int}
REQUIRED = ("name", "owner", "group")
#: the jobs a plan may split, and what their parts are
SPLIT = {"lean-audit": "the Lake packages with a lean-audit.json",
         "circuit-check": "its `shards` shards of its targets, K/N (`circuit-check --all --shard K/N`)",
         "lean-suites": "its suites (`suites.AFTER_LEAN`), one that declares `[tool.verity.tests] parts = P` as its P parts by "
                        "test id, SUITE:K/P (`suites.py SUITE:K/P`)",
         "pytest": "its suites (all but `suites.AFTER_LEAN`), as lean-suites', in a plan that names one"}
#: the split jobs whose parts are shards of their own work, as many as their `shards` say (none given: the job runs whole)
SHARDED = ("circuit-check",)
#: the parts a check runs alone (`check.py PART...`)
PARTS = ("tests", "circuits", "lean")


class DeclarationError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    name: str
    owner: str
    group: int
    halts_on: str | None = None
    merge_requires: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    after: str | None = None
    part: str | None = None
    gate: bool = True
    shards: int | None = None


def _toml(text: str, source: str) -> dict:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise DeclarationError(f"{source}: not TOML: {e}") from None
    if doc.get("schema") != SCHEMA:
        raise DeclarationError(f"{source}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def parse(text: str, handles: set[str], source: str = "ci.toml", off: tuple[str, ...] = ()) -> list[Job]:
    """The jobs ``text`` declares, in file order; ``handles`` are the owners allowed, ``off`` the jobs off the merge gate
    (`off_gate`)."""
    doc = _toml(text, source)
    if unknown := set(doc) - {"schema", "job", "plan"}:
        raise DeclarationError(f"{source}: unknown keys {sorted(unknown)}")
    jobs, seen = [], set()
    for i, t in enumerate(doc.get("job", [])):
        where = f"{source}: job {i + 1}" + (f" ({t['name']})" if isinstance(t.get("name"), str) else "")
        if "gate" in t:
            raise DeclarationError(f"{where}: gate: which jobs gate a merge is drive's policy, not ci.toml's: name the job in "
                                   f"policy.toml's off_gate instead")
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
        if t["name"] in off and t.get("merge_requires"):
            raise DeclarationError(f"{where}: merge_requires on a job off the gate (policy.toml's off_gate), which no merge requires")
        if "part" in t and t["part"] not in PARTS:
            raise DeclarationError(f"{where}: part {t['part']!r} isn't one of {', '.join(PARTS)}")
        if "shards" in t and (t["name"] not in SHARDED or t["shards"] < 1):
            raise DeclarationError(f"{where}: shards is N >= 1, for {', '.join(SHARDED)} alone")
        if t["name"] in seen:
            raise DeclarationError(f"{where}: duplicate name")
        if t["owner"] not in handles:
            raise DeclarationError(f"{where}: owner {t['owner']!r} isn't a handle in the registry ({', '.join(sorted(handles))})")
        earlier = {j.name for j in jobs if j.group == t["group"]}
        if bad := [n for n in t.get("needs", []) if n not in earlier]:
            raise DeclarationError(f"{where}: needs {bad}, which aren't earlier jobs of its group")
        seen.add(t["name"])
        jobs.append(Job(t["name"], t["owner"], t["group"], t.get("halts_on"),
                        tuple(t.get("merge_requires", ())), tuple(t.get("needs", ())), t.get("after"), t.get("part"),
                        t["name"] not in off, t.get("shards")))
    if not jobs:
        raise DeclarationError(f"{source}: declares no job")
    by = {j.name: j for j in jobs}
    if stray := [n for n in off if n not in by]:
        raise DeclarationError(f"policy.toml's off_gate names {', '.join(stray)}, which {source} doesn't declare")
    for j in jobs:
        if j.gate and (off := [n for n in (*j.needs, *[j.after] * bool(j.after)) if n in by and not by[n].gate]):
            raise DeclarationError(f"{source}: job {j.name} is on the gate and waits on {', '.join(off)}, which is off it "
                                   f"(policy.toml's off_gate): take {j.name} off the gate too, or put {', '.join(off)} back on it")
        if j.after is None:
            continue
        if (a := by.get(j.after)) is None:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, which isn't declared")
        if a.group == j.group:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, in its own group, whose order already runs it first")
        if a.after is not None:
            raise DeclarationError(f"{source}: job {j.name}: after {j.after!r}, which itself waits on {a.after!r}")
    return jobs


Plan = tuple[tuple[str, ...], ...]


def _selector(s, names: set[str], where: str) -> tuple[str, str | None]:
    if not isinstance(s, str):
        raise DeclarationError(f"{where}: a selector must be a string, not {type(s).__name__}")
    job, colon, part = s.partition(":")
    if job not in names:
        raise DeclarationError(f"{where}: {s!r} names no declared job")
    if colon and (job not in SPLIT or not part):
        raise DeclarationError(f"{where}: {s!r}: only {', '.join(SPLIT)} split into parts, as JOB:PART or JOB:*")
    return job, part if colon else None


def parse_plans(text: str, jobs: list[Job], source: str = "ci.toml") -> dict[int, Plan]:
    """The plans ``text`` declares, by their number of shards, each checked against ``jobs`` (the module docstring's rules)."""
    doc, names = _toml(text, source), {j.name for j in jobs}
    out: dict[int, Plan] = {}
    for i, t in enumerate(doc.get("plan", [])):
        where = f"{source}: plan {i + 1}"
        if not isinstance(t, dict) or set(t) != {"shards"}:
            raise DeclarationError(f"{where}: a plan is a table holding `shards` alone")
        shards = t["shards"]
        if not (isinstance(shards, list) and len(shards) >= 2 and all(isinstance(s, list) and s for s in shards)):
            raise DeclarationError(f"{where}: shards must be two or more non-empty lists of selectors")
        if len(shards) in out:
            raise DeclarationError(f"{where}: another plan has {len(shards)} shards")
        whole: dict[str, list[int]] = {}
        parts: dict[str, list[tuple[int, str]]] = {}
        for k, shard in enumerate(shards, 1):
            if len(set(shard)) != len(shard):
                raise DeclarationError(f"{where}: shard {k} names a selector twice")
            for s in shard:
                job, part = _selector(s, names, f"{where}: shard {k}")
                (whole.setdefault(job, []).append(k) if part is None else parts.setdefault(job, []).append((k, part)))
        for j in jobs:
            w, p = whole.get(j.name, []), parts.get(j.name, [])
            if not w and not p and j.gate:
                raise DeclarationError(f"{where}: no shard runs {j.name}")
            if w and (len(w) > 1 or p):
                raise DeclarationError(f"{where}: {j.name} is in more than one shard")
            named = [x for _, x in p if x != "*"]
            if p and ([x for _, x in p].count("*") != 1 or len(set(named)) != len(named)):
                raise DeclarationError(f"{where}: {j.name}'s parts must be named once each, with exactly one {j.name}:*")
        for j in jobs:
            for k in whole.get(j.name, []):
                if missing := [n for n in (*j.needs, *[j.after] * bool(j.after)) if whole.get(n) != [k]]:
                    raise DeclarationError(f"{where}: shard {k} runs {j.name}, which needs {', '.join(missing)} in the same shard")
        out[len(shards)] = tuple(tuple(s) for s in shards)
    return out


def select(jobs: list[Job], parts: list[str]) -> set[str]:
    """The jobs a check of ``parts`` runs: those of each part, and every job they `need` or run `after`.  DeclarationError for
    a part that isn't one of `PARTS`, or when a job declares no part (it would silently be in none)."""
    if bad := [p for p in parts if p not in PARTS]:
        raise DeclarationError(f"no part {', '.join(bad)}: the parts are {', '.join(PARTS)}")
    if loose := [j.name for j in jobs if j.part is None]:
        raise DeclarationError(f"ci.toml's {', '.join(loose)} declare no part, so a check of parts can't tell whether to run them")
    return needed(jobs, [j.name for j in jobs if j.part in parts])


def gated(jobs: list[Job]) -> list[Job]:
    """The jobs on the merge gate (all but policy.toml's `off_gate`), in file order: what a recorded check runs and a merge
    requires."""
    return [j for j in jobs if j.gate]


def needed(jobs: list[Job], names: list[str]) -> set[str]:
    """The jobs ``names``, and every job they `need` or run `after` (`check.py --deferred`).  DeclarationError for a name no job
    has."""
    by = {j.name: j for j in jobs}
    if bad := [n for n in names if n not in by]:
        raise DeclarationError(f"no job {', '.join(bad)}: the jobs are {', '.join(by)}")
    todo, out = list(names), set()
    while todo:
        if (n := todo.pop()) not in out:
            out.add(n)
            todo += [*by[n].needs, *[by[n].after] * bool(by[n].after)]
    return out


def job_parts(job: Job, parts_of: dict[str, list[str]]) -> list[str] | None:
    """A split job's parts, or None for a job that runs whole: a `SHARDED` job's ``K/N`` for K = 1..N, its `shards` (none given:
    whole), another split job's from ``parts_of`` (the tree's, for a plan; one it leaves out: whole), sorted."""
    if job.name not in SPLIT:
        return None
    if job.name in SHARDED:
        return None if job.shards is None else [f"{k}/{job.shards}" for k in range(1, job.shards + 1)]
    return sorted(parts_of[job.name]) if job.name in parts_of else None


def all_parts(jobs: list[Job], parts_of: dict[str, list[str]]) -> list[str]:
    """Every part an unsharded check covers, in job order: each job's name, or for a split job `JOB:PART` for each of its parts
    (`job_parts`)."""
    out = []
    for j in jobs:
        xs = job_parts(j, parts_of)
        out += [j.name] if xs is None else [f"{j.name}:{x}" for x in xs]
    return out


def resolve(plan: Plan, jobs: list[Job], parts_of: dict[str, list[str]],
            together: dict[str, list[str]] | None = None) -> list[list[str]]:
    """Each shard's parts (`all_parts`' names, in its order) under ``plan`` for a tree whose split jobs have ``parts_of``. The
    parts ``together`` ties (a part to those it runs beside: a spec package's audit to its provers', which it audits from one
    build of theirs) are on one shard, the one the plan names any of them in, else `JOB:*`'s; DeclarationError when the plan
    names them in two."""
    named = {s for shard in plan for s in shard if ":" in s and not s.endswith(":*")}
    by = {j.name: j for j in jobs}
    if missing := sorted(s for s in named if s.partition(":")[2] not in (job_parts(by[s.partition(":")[0]], parts_of) or ())):
        raise DeclarationError(f"the plan names {', '.join(missing)}, which this tree doesn't have")
    every = all_parts(jobs, parts_of)
    where: dict[str, int] = {}
    for k, shard in enumerate(plan):
        for p in every:
            job = p.partition(":")[0]
            if job in shard or p in shard or (f"{job}:*" in shard and p not in named):
                where[p] = k
    root = {p: p for p in every}

    def find(p: str) -> str:
        while root[p] != p:
            p = root[p]
        return p

    for p, qs in (together or {}).items():
        for q in qs:
            if p in root and q in root:
                root[find(p)] = find(q)
    ties: dict[str, list[str]] = {}
    for p in every:
        ties.setdefault(find(p), []).append(p)
    for tie in (t for t in ties.values() if len(t) > 1):
        if len(at := {where[p] for p in tie if p in named and p in where}) > 1:
            raise DeclarationError(f"the plan names {', '.join(p for p in tie if p in named)} in shards "
                                   f"{', '.join(str(k + 1) for k in sorted(at))}, and they run beside one another (a spec "
                                   f"package's audit audits the packages its `proved_in` names, from one build of theirs): "
                                   f"name them in one shard")
        if (k := next(iter(at), where.get(tie[0]))) is not None:
            where.update(dict.fromkeys(tie, k))
    return [[p for p in every if where.get(p) == k] for k in range(len(plan))]


def plans(path: Path = PATH, registry: Path = REGISTRY) -> dict[int, Plan]:
    return parse_plans(path.read_text(), load(path, registry), str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))


def handles(registry: Path = REGISTRY) -> set[str]:
    return {h["handle"] for h in json.loads(registry.read_text())["handles"]}


def off_gate(policy: Path = POLICY) -> tuple[str, ...]:
    """The jobs drive's policy takes off the merge gate (``policy``'s `off_gate`); none when there is no such file."""
    if not policy.exists():
        return ()
    try:
        doc = tomllib.loads(policy.read_text())
    except tomllib.TOMLDecodeError as e:
        raise DeclarationError(f"{policy}: not TOML: {e}") from None
    off = doc.get("off_gate", [])
    if not (isinstance(off, list) and all(isinstance(x, str) for x in off)):
        raise DeclarationError(f"{policy}: off_gate must be a list of job names")
    return tuple(off)


def load(path: Path = PATH, registry: Path = REGISTRY, policy: Path = POLICY) -> list[Job]:
    return parse(path.read_text(), handles(registry), str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                 off_gate(policy))
