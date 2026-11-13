"""lean-lock's runner: the Lean check's audit of one package (`tools/verity/lean/check.py`, without its build, its replay and
its tests) on the built modules its task reads, and the package's committed `lean-audit.json` against the records that
audit computes.

A task gives (`families/lean_lock.py`): `vars` `package`; inputs every file of the packages the audit reads (the package,
those it requires by path, those its `proved_in` names) and the auditor's; reads what an importer reads of each of their
modules (its `.olean`, and a `module` file's parts), in one directory of the work directory (`out/lean/<A>/<B>.olean` for
`A.B`); and output `records`. The runner copies the inputs into a scratch tree (`lean_audit.scratch_tree`), gives each of
its packages a build of its own modules' parts alone (hard links into its `.lake/build/lib/lean`: the audit counts every
`.olean` that resolves under a `proved_in` package's build as one of its modules) and its fetched packages from the
worker's `lean` env (`lean_module.packages`), and runs `check.py` there. It passes when the audit passes, `records` then
the committed lock; a lock whose every failure is a record difference is stale, a fail naming the command that regenerates
it, `records` then the lock that command writes; a `roots` entry no read holds (`lake build +ROOT` fails on it) and any
other failure of the audit are fails too. A lock's `runs` aren't run: a run's `generate` reads the tree beyond the task's
inputs.

The scratch tree and the audit's directory are removed before the result is returned, whatever it is: the executor brings
the whole work directory back from its worker with the result, and the scratch tree holds a copy of every input and a link
to every read (for Security's locks, gigabytes in tens of thousands of files)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .. import ROOT
from ..families.lean_module import la, mk, modules, out_path
from . import call
from .lean_module import EnvError, packages

POLICY = "lean-audit.json"


def regenerate(pkg: str) -> str:
    return f"python3 tools/verity/lean/check.py --build --update {pkg}"


def run(task: dict, work: Path) -> dict:
    tree, out = work / "tree", work / "audit"
    try:
        return _audit(task, work, tree, out)
    finally:
        for d in (tree, out):
            shutil.rmtree(d, ignore_errors=True)


def _audit(task: dict, work: Path, tree: Path, out: Path) -> dict:
    pkg = task["vars"]["package"]
    records = next(work / o["path"] for o in task["outputs"] if o["name"] == "records")
    lib = work / "out" / "lean"
    lib.mkdir(parents=True, exist_ok=True)
    la.scratch_tree(tree, task["inputs"])
    policy = json.loads((tree / pkg / POLICY).read_text())  # the listing refuses a lock that isn't JSON
    roots = policy.get("roots") if isinstance(policy, dict) else None
    named = [r for r in roots if isinstance(r, str) and r] if isinstance(roots, list) else []  # the audit fails other shapes
    if missing := [r for r in named if not (lib / mk.path_of(r, ".olean")).is_file()]:
        return {"status": "fail", "detail": {"error": f"{pkg}/{POLICY}'s `roots` names modules no task builds, which "
                                                      f"`lake build +ROOT` fails on: {', '.join(missing[:20])}"}}
    pkgs = [p.rpartition("/")[0] or "." for p in task["inputs"] if p.rpartition("/")[2] in ("lakefile.toml", "lakefile.lean")]
    _, owner = modules(tree, task["inputs"])
    for q in pkgs:
        (tree / q / ".lake" / "build" / "lib" / "lean").mkdir(parents=True, exist_ok=True)
    for r in task["reads"]:
        q, mod = owner[r["task"].partition(":")[2]]
        part = out_path(mod, r["output"])
        dst = tree / q / ".lake" / "build" / "lib" / part.removeprefix("out/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(work / part, dst)
    try:
        for q in pkgs:
            if (deps := packages(ROOT / q, work)) is not None:
                (tree / q / ".lake" / "packages").symlink_to(deps, target_is_directory=True)
    except EnvError as e:
        return {"status": "error", "detail": {"error": f"the lean env: {e}"}}
    rc = call([sys.executable, str(tree / "tools" / "verity" / "lean" / "check.py"), "--no-replay", "--no-runs",
               "--repo", str(tree), "--out", str(out), str(tree / pkg)], work, tree,
              {**os.environ, "LEAN_NUM_THREADS": str(task.get("resources", {}).get("cores", 1))})
    doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").is_file() else {}
    rep = next((r for r in doc.get("packages", []) if Path(r["package"]).resolve() == (tree / pkg).resolve()), None)
    if rc not in (0, 1) or rep is None:
        return {"status": "error", "detail": {"error": f"check.py exited {rc} without a report of {pkg}"}}
    fails = [str(f) for f in rep["failures"]]
    detail = {"seconds": rep.get("seconds", {}), "guarantees": len(rep.get("guarantees") or {})}
    records.parent.mkdir(parents=True, exist_ok=True)
    if not fails:
        shutil.copyfile(tree / pkg / POLICY, records)
        return {"status": "pass", "detail": detail}
    if regen := rep.get("regenerated"):
        shutil.copyfile(regen, records)
        return {"status": "fail", "detail": detail | {
            "error": f"{pkg}/{POLICY} is stale: its records aren't what the build gives; `{regenerate(pkg)}` rewrites it",
            "regenerate": regenerate(pkg), "review": rep.get("review", [])[:40], "failures": fails[:20]}}
    return {"status": "fail", "detail": detail | {"error": f"the Lean check fails {pkg}", "failures": fails[:20]}}
