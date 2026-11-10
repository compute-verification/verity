"""lean-package: Lean's audit of the Lake package the task names (its var `package`), as check's lean-audit step audits a
shard's share (`tools/verity/lean/lean_audit.py --package PKG`), with lake on the PATH through elan (`lean_build.ELAN`) and the
audit's own cache and dependency store where check keeps them (`lean_audit.py`'s defaults).  As a shard's, it regenerates no
record (`tools/verity/check/tasks.py`'s REGENERATE): a lock that doesn't match the package fails it.

Its status is from the audit's exit and its report (`audit.json`, its output).  Exit 0 is a pass.  Exit 1 with the report naming
what failed is a fail, each package's failures in `detail.failures` (cut to `NAMED`), unless every failure is a dependency the
audit couldn't fetch (`lean_audit.transient`, after its own retries) or the check.py it runs reported nothing, the machine's.
A report missing, any other exit and a signal are an error."""
from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

from .. import ROOT
from . import call
from .lean_build import ELAN

AUDIT = "tools/verity/lean/lean_audit.py"
OUT = "lean-audit"
#: how the report's `failed` names a check.py that wrote no report, or one killed with nothing failed (`check.py exit N`)
CHECK = "check.py"
#: the most failures `detail.failures` names: a longer list is cut, and `detail.beyond` says so
NAMED = 50


def _audit_module():
    spec = importlib.util.spec_from_file_location("lean_audit_for_lean_package", ROOT / AUDIT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(task: dict, work: Path) -> dict:
    pkg = task["vars"]["package"]
    argv = [sys.executable, str(ROOT / AUDIT), "--out", str(work / OUT), "--package", pkg]
    rc = call(["bash", "-c", ELAN + shlex.join(argv)], work, ROOT)
    report = work / OUT / "audit.json"
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"lean_audit.py was killed (exit {rc})"}}
    if rc not in (0, 1) or not report.is_file():
        return {"status": "error", "detail": {"error": f"lean_audit.py exited {rc}" + ("" if report.is_file() else
                                                                                       " and wrote no report")}}
    doc = json.loads(report.read_text())
    detail: dict = {"exit": rc, "reused": pkg in (doc.get("cached") or {})}
    if rc == 0 and doc.get("passed") is True:
        return {"status": "pass", "detail": detail}
    reps = [r for r in doc.get("packages") or () if isinstance(r, dict)]
    if not doc.get("failed") or rc != 1:
        return {"status": "error", "detail": {**detail, "error": f"lean_audit.py exited {rc} and its report names "
                                                                 f"{'no failure' if not doc.get('failed') else 'a failure'}"}}
    if lost := [str(f) for f in doc["failed"] if str(f).startswith(CHECK)]:
        return {"status": "error", "detail": {**detail, "error": f"lean_audit.py's check.py reported nothing: {', '.join(lost)}"}}
    failing = [r for r in reps if r.get("failures")]
    if failing and all(_audit_module().transient(r) for r in failing):
        return {"status": "error", "detail": {**detail, "error": "a dependency couldn't be fetched: "
                                                                 + "; ".join(f for r in failing for f in r["failures"])[:2000]}}
    failures = [f"{r.get('package', pkg)}: {f}" for r in failing for f in r["failures"]]
    detail.update(failed=[str(f) for f in doc["failed"]], failures=failures[:NAMED])
    if len(failures) > NAMED:
        detail["beyond"] = True
    return {"status": "fail", "detail": detail}
