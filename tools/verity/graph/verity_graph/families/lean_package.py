"""The lean-package family: one task per audited Lake package (a `lean-audit.json`, the auditor's fixtures aside:
`lean_audit.packages`), Lean's audit of it as check's lean-audit step audits a shard's share (`runners/lean_package.py`): it
builds the package in a scratch tree, replays every declaration through the kernel and checks its lock.  Its inputs are every
file an audit of the package can read (`lean_audit.inputs`, read here from the listed tree): the package's own files, those of
the packages it requires by path (its `lake-manifest.json`'s path entries) and of the packages its `proved_in` names with
theirs, and the auditor's (`lean_audit.tool_files`).  A package under `Security`, or one requiring a package there by path,
asks `SECURITY` (Mathlib, ArkLib: tasks/v1's `lean_slot`), the rest the family's slot."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from ..graph import GraphError

KIND = "lean-package"
LEAN = Path(__file__).resolve().parents[3] / "lean"
REPORT = "lean-audit/audit.json"
#: what a package under `Security` asks over the family's slot (`tools/verity/check/tasks.py`'s LEAN_SLOT)
SECURITY = {"cores": 32, "mem_gb": 128}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_verity_graph", LEAN / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


la = _load("lean_audit")


def _json(root: Path, path: str) -> object:
    try:
        return json.loads((root / path).read_text())
    except ValueError as e:
        raise GraphError(f"{path} isn't JSON ({e}): an audit's inputs are read from it") from e


def path_deps(root: Path, pkg: str) -> list[str]:
    """The packages ``pkg`` requires by path, its manifest's path entries (`lean_audit.path_deps`)."""
    if not (root / pkg / "lake-manifest.json").is_file():
        return []
    doc = _json(root, f"{pkg}/lake-manifest.json")
    entries = doc.get("packages", []) if isinstance(doc, dict) else []
    return sorted({os.path.relpath(os.path.normpath(root / pkg / d["dir"]), root) for d in entries
                   if isinstance(d, dict) and d.get("type") == "path" and isinstance(d.get("dir"), str)})


def provers(root: Path, pkg: str) -> list[str]:
    """The packages ``pkg``'s lock names under `proved_in` (`lean_audit.provers`); none when the audit fails its lock."""
    doc = _json(root, f"{pkg}/lean-audit.json")
    named = doc.get("proved_in", []) if isinstance(doc, dict) else []
    return sorted({os.path.relpath(os.path.normpath(root / pkg / d), root)
                   for d in (named if isinstance(named, list) else []) if isinstance(d, str)})


def closure(root: Path, pkg: str) -> list[str]:
    """``pkg``, the packages it requires by path, and those its `proved_in` names with theirs: whose files its audit reads."""
    return list(dict.fromkeys([pkg, *path_deps(root, pkg), *(d for q in provers(root, pkg) for d in (q, *path_deps(root, q)))]))


def tasks(root: Path, files: list[str]) -> list[dict]:
    have = set(files)
    tool = [f for f in la.tool_files(files) if f in have]
    out = []
    for pkg in la.packages(files):
        task = {"id": f"{KIND}:{pkg}", "vars": {"package": pkg},
                "inputs": sorted({*(f for q in closure(root, pkg) for f in la.own_files(q, files)), *tool}),
                "outputs": [{"name": "report", "path": REPORT}]}
        if any(p.split("/")[0] == "Security" for p in (pkg, *path_deps(root, pkg))):
            task["resources"] = dict(SECURITY)
        out.append(task)
    return out
