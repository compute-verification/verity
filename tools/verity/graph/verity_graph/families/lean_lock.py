"""lean-lock: one task per audited Lake package (a `lean-audit.json`, the auditor's fixtures aside: `lean_audit.packages`),
which recomputes the records of the guarantees its lock lists from the package's built modules and compares them with the
committed lock (`runners/lean_lock.py`).

Its records read the builds of the packages `lean_changed.closure` names: the package, those it requires by path, and those
its `proved_in` names with theirs (Security's records read security/Proofs, which requires the Flock verifier). Its inputs
are every file an audit of the package can read, those packages' and the auditor's (`lean_audit.inputs`), and it reads
every part of each of their modules that has a task (`lean_module.reads` at ALL: its `olean`, and every part of a `module`
file, since the audit loads them at the private level, where a definition's body is), so a change to any of them reruns
it. A `roots` entry that names no module of the tree has no task to read, and the runner fails the lock on it, as the
check's `lake build +ROOT` does."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..graph import GraphError
from . import lean_module as lm

KIND = "lean-lock"


def task_id(pkg: str) -> str:
    return f"{KIND}:{pkg}"


def closure(root: Path, pkg: str, pkgs: dict[str, lm.Package]) -> list[str]:
    """``pkg``, the packages it requires by path, and those its `proved_in` names with theirs, as `lean_changed.closure`."""
    def deps(q: str) -> tuple[str, ...]:
        return pkgs[q].workspace[1:] if q in pkgs else ()
    try:
        doc = json.loads((root / pkg / "lean-audit.json").read_text())
    except ValueError as e:
        raise GraphError(f"{pkg}/lean-audit.json isn't JSON ({e}): its `proved_in` names packages its lock reads") from e
    named = doc.get("proved_in", []) if isinstance(doc, dict) else []  # the audit fails a lock that isn't an object
    provers = sorted({os.path.relpath(os.path.normpath(root / pkg / q), root)
                      for q in (named if isinstance(named, list) else []) if isinstance(q, str)})
    return list(dict.fromkeys([pkg, *deps(pkg), *(d for q in provers for d in (q, *deps(q)))]))


def tasks(root: Path, files: list[str]) -> list[dict]:
    pkgs, owner, _, system, _ = lm.imports(root, files)
    tool = lm.la.tool_files(files)
    out = []
    for pkg in lm.la.packages(files):
        cl = closure(root, pkg, pkgs)
        out.append({"id": task_id(pkg), "vars": {"package": pkg},
                    "inputs": sorted({*(f for q in cl for f in lm.la.own_files(q, files)), *tool}),
                    "reads": [r for f, (d, _) in sorted(owner.items()) if d in cl for r in lm.reads(f, lm.ALL, system)],
                    "outputs": [{"name": "records", "path": f"out/{pkg}/lean-audit.json"}]})
    return out
