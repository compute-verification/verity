"""lean-env: one task per `lean` env the modules of the tree's Lake packages compile in, which makes the env and
kernel-replays its modules (`runners/lean_env.py`), so that the kernel checks every declaration a module rests on, as
`check.py --closure` has it do. An env is a package's toolchain (its `lean-toolchain`) and the packages its manifest
fetches (Mathlib and the rest), as `lean_audit.deps_key` keys them, or the toolchain alone for a package that fetches
none: packages with one of each share one, whose task is named after the first of them in path order. The auditor's
fixture packages (`lean_audit.packages` leaves them out) have none.

A task's fields, which its runner reads: `vars` `package` (that first package's directory) and `scope` (`SCOPE`); `inputs`
that package's `lean-toolchain` and `lake-manifest.json` and the Lean tool's files the runner runs
(`runners.lean_env.TOOLS`); no reads; and one output, `axioms`, the axiom sets of its modules' declarations, at `AXIOMS`
for every env, since a module reads one. Each lean-module task reads its package's env's `axioms`
(`families/lean_module.py`), whose records stop its replay's walk at the env's modules as the tree's records do.

`SCOPE` is what a task replays: "whole", every module of the env (the toolchain's Init, Std, Lean and Lake, and every
fetched package's), whose key is then the env's files and the runner's alone; or "imports", the env's modules the tree's
modules import (`vars` `modules`, as JSON: those their headers name, and `Init` for a header without `prelude`) and what
those import, transitively, which reruns whenever a header changes what it imports of the env."""
from __future__ import annotations

import json
from pathlib import Path

from ..graph import GraphError
from . import lean_module as lm

KIND = "lean-env"
#: what each task replays, one of `SCOPES`
SCOPE = "whole"
SCOPES = ("whole", "imports")
#: where each task writes its `axioms`
AXIOMS = "out/lean-env/axioms.json"


def task_id(pkg: str) -> str:
    return f"{KIND}:{pkg}"


def envs(root: Path, pkgs: dict[str, lm.Package]) -> dict[str, list[str]]:
    """{the directory of the first package of each env, in path order: the directories of every package of it}."""
    by: dict[str, list[str]] = {}
    for d in sorted(pkgs):
        by.setdefault(lm.la.deps_key(root / d) or (root / d / "lean-toolchain").read_text().strip(), []).append(d)
    return {ds[0]: ds for ds in by.values()}


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
    return [{"id": task_id(d),
             "vars": {"package": d, "scope": SCOPE,
                      **({"modules": json.dumps(roots[d], separators=(",", ":"))} if d in roots else {})},
             "inputs": sorted({(Path(d) / "lean-toolchain").as_posix(), (Path(d) / "lake-manifest.json").as_posix(), *TOOLS}),
             "outputs": [{"name": "axioms", "path": AXIOMS}]}
            for d in sorted(on)]
