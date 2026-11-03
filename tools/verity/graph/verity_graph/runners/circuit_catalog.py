"""circuit-catalog: every catalog binding's record, built once into a fresh cache as check's records step builds it (`check.py
--cc-records`: `circuit-check --all --records-only`), and tarred to the task's one output (`circuit_binding.pack`), which every
circuit-binding task and the coverage task read.  Unlike the records step's tarball it names no tree: a record counts only
while its own reads match (`circuit_check.units`), and the tarball's bytes are the records' alone.

A pass is circuit-check's exit 0 with every record stored; a fail its exit 1 with a report, which says why none were stored (a
record that disagrees with what was built, a build the tracer couldn't see); anything else is an error."""
from __future__ import annotations

import shutil
from pathlib import Path

from . import circuit_binding as CB


def run(task: dict, work: Path) -> dict:
    records = work / "cache"
    shutil.rmtree(records, ignore_errors=True)
    try:
        rc, rep = CB.checker(task, work, "--all", "--records-only", "--cache", str(records))
        if rc == 1 and rep is not None:
            return {"status": "fail", "detail": {"error": rep.get("why") or "circuit-check --records-only exited 1"}}
        if rc or rep is None:
            return {"status": "error", "detail": {"error": f"circuit-check --records-only exited {rc}"
                                                  + ("" if rep is not None else " and wrote no report")}}
        n = CB.pack(records, work / task["outputs"][0]["path"])
        return {"status": "pass", "detail": {"bindings": rep.get("reused", 0) + len(rep.get("rebuilt", [])),
                                             "stored": rep["stored"], "records": n, "bytes": rep.get("bytes", 0)}}
    finally:
        shutil.rmtree(records, ignore_errors=True)
