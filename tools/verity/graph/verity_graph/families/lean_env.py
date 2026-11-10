"""lean-env: one task per `lean` env the modules of the tree's Lake packages compile in, which makes the env and
kernel-replays its modules (`runners/lean_env.py`), so that the kernel checks every declaration a module rests on, as
`check.py --closure` has it do. An env is a package's toolchain (its `lean-toolchain`) and the packages its manifest
fetches (Mathlib and the rest), each its manifest entry but `inherited` (`env_key`), or the toolchain alone for a package
that fetches none: packages with one of each share one, whose task is named after the first of them in path order. The
auditor's fixture packages (`lean_audit.packages` leaves them out) have none.

An env whose fetched packages hold every package of another env on its toolchain is layered on it, on the one of those
with the most packages (`bases`): its task reads that env's `axioms` and replays only the modules that env doesn't hold, so
each module of the env goes to the kernel once in the graph (Mathlib's env on the toolchain's, ArkLib's on Mathlib's).
Only the `whole` scope layers, since `--closure-of` walks an import into the env below.

A task's fields, which its runner reads: `vars` `package` (that first package's directory) and `scope` (`SCOPE`); `inputs`
that package's `lean-toolchain` and `lake-manifest.json` and the Lean tool's files the runner runs
(`runners.lean_env.TOOLS`); reads, the `axioms` of the env it is layered on, if any; and one output, `axioms`, the axiom
sets of its modules' declarations, the env's below included, at `record(package)`. Each lean-module task reads its
package's env's `axioms` (`families/lean_module.py`), whose records stop its replay's walk at the env's modules as the
tree's records do.

`SCOPE` is what a task replays: "whole", every module of the env (the toolchain's Init, Std, Lean and Lake, and every
fetched package's), whose key is then the env's files and the runner's alone; or "imports", the env's modules the tree's
modules import (`vars` `modules`, as JSON: those their headers name, and `Init` for a header without `prelude`) and what
those import, transitively, which reruns whenever a header changes what it imports of the env."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..graph import GraphError
from . import lean_module as lm

KIND = "lean-env"
#: what each task replays, one of `SCOPES`
SCOPE = "whole"
SCOPES = ("whole", "imports")


def task_id(pkg: str) -> str:
    return f"{KIND}:{pkg}"


def record(pkg: str) -> str:
    """Where the task of the env named after the package at ``pkg`` writes its `axioms`: each env's apart, since one layered
    on another reads that one's."""
    return (Path("out/lean-env") / pkg / "axioms.json").as_posix()


def read(task: dict) -> tuple[str, str] | None:
    """The task of the env whose `axioms` ``task`` reads (a module its package's env's, an env the one's it is layered on)
    and where they are in its work directory, None when it reads none; ValueError when it reads more than one."""
    got = [r["task"] for r in task["reads"] if r["task"].startswith(f"{KIND}:") and r["output"] == "axioms"]
    if len(got) > 1:
        raise ValueError(f"{task['id']} reads the axioms of {len(got)} lean envs: {', '.join(got)}")
    return (got[0], record(got[0][len(KIND) + 1:])) if got else None


def fetched(pkg: Path) -> frozenset[str]:
    """The packages ``pkg``'s manifest fetches, each its manifest entry as JSON but `inherited`, which says whether a
    requirement's manifest pinned it and changes none of its files."""
    m = pkg / "lake-manifest.json"
    return frozenset(json.dumps({k: v for k, v in p.items() if k != "inherited"}, sort_keys=True, separators=(",", ":"))
                     for p in (json.loads(m.read_text()).get("packages", []) if m.is_file() else [])
                     if p.get("type") != "path")


def env_key(pkg: Path) -> str:
    """The env ``pkg`` compiles in: its toolchain's line when it fetches no package, else a digest of that line and its
    `fetched` packages."""
    tc = (pkg / "lean-toolchain").read_text().strip()
    f = fetched(pkg)
    return hashlib.sha256(json.dumps([tc, sorted(f)], separators=(",", ":")).encode()).hexdigest() if f else tc


def envs(root: Path, pkgs: dict[str, lm.Package]) -> dict[str, list[str]]:
    """{the directory of the first package of each env, in path order: the directories of every package of it}."""
    by: dict[str, list[str]] = {}
    for d in sorted(pkgs):
        by.setdefault(env_key(root / d), []).append(d)
    return {ds[0]: ds for ds in by.values()}


def bases(root: Path, on: dict[str, list[str]]) -> dict[str, str]:
    """{env: the env it is layered on} under the `whole` scope, for each env (of `envs`) whose fetched packages hold every
    one of another env's on its toolchain, the toolchain's alone included: the one with the most packages, the first in
    path order of those with as many."""
    if SCOPE != "whole":
        return {}
    of = {e: ((root / e / "lean-toolchain").read_text().strip(), fetched(root / e)) for e in on}
    out = {}
    for e, (tc, f) in of.items():
        if under := sorted(b for b, (t, g) in of.items() if t == tc and g < f):
            out[e] = max(under, key=lambda b: len(of[b][1]))
    return out


def imported(root: Path, files: list[str], on: dict[str, list[str]]) -> dict[str, list[str]]:
    """{env: the env's modules that the modules the check builds of its packages import (`lean_module.imports`)}."""
    _pkgs, owner, _direct, _system, outside = lm.imports(root, files)
    env = {d: e for e, ds in on.items() for d in ds}
    got: dict[str, set[str]] = {e: set() for e in on}
    for f, mods in outside.items():
        got[env[owner[f][0]]] |= mods
    return {e: sorted(m) for e, m in got.items()}


def tasks(root: Path, files: list[str]) -> list[dict]:
    from ..runners.lean_env import TOOLS  # not at the top, so that the runner can import this module

    if SCOPE not in SCOPES:
        raise GraphError(f"families/lean_env.py's SCOPE is {SCOPE!r}: it is one of {', '.join(SCOPES)}")
    on = envs(root, lm.packages(root, files))
    roots = imported(root, files, on) if SCOPE == "imports" else {}
    under = bases(root, on)
    return [{"id": task_id(d),
             "vars": {"package": d, "scope": SCOPE,
                      **({"modules": json.dumps(roots[d], separators=(",", ":"))} if d in roots else {})},
             "inputs": sorted({(Path(d) / "lean-toolchain").as_posix(), (Path(d) / "lake-manifest.json").as_posix(), *TOOLS}),
             "reads": [{"task": task_id(under[d]), "output": "axioms"}] if d in under else [],
             "outputs": [{"name": "axioms", "path": record(d)}]}
            for d in sorted(on)]
