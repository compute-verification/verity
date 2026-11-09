"""circuit-catalog: every catalog binding's record, built once into a fresh cache as check's records step builds it (`check.py
--cc-records`: `circuit-check --all --records-only`), tarred to the task's ``records`` output (`circuit_binding.pack`), which
the coverage task reads, and cut into each binding's slice and the part they share (``--slices``), each its own output, which
that binding's task alone reads.  Unlike the records step's tarball it names no tree: a record counts only while its own reads
match (`circuit_check.units`), and the tarball's bytes are the records' alone.  A listed binding circuit-check made no slice
for gets the marker `circuit_binding.absent` in its slice's place, which its task fails on, as on a name the catalog lacks.

A pass is circuit-check's exit 0 with every record stored and every slice written; a fail its exit 1 with a report, which says
why none were stored or sliced (a record that disagrees with what was built, a build the tracer couldn't see); anything else
is an error."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import circuit_binding as CB

#: the prefix of a slice's output name, before its binding's (`families.circuit_binding.slice_of`)
SLICE = "slice "
#: circuit-check's thread pools, one thread each: it builds as many bindings at once as the task has cores (``--jobs``), each in
#: a process of its own, and wider pools only make those contend for the cores.  circuit-check can't narrow them itself: numpy
#: sizes its pool when it loads, before any argument is read
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")


def run(task: dict, work: Path) -> dict:
    records, cut = work / "cache", work / "slices"
    for d in (records, cut):
        shutil.rmtree(d, ignore_errors=True)
    try:
        rc, rep = CB.checker(task, work, "--all", "--records-only", "--cache", str(records), "--slices", str(cut),
                             env={**os.environ, **THREADS})
        if rc == 1 and rep is not None:
            return {"status": "fail", "detail": {"error": rep.get("why") or "circuit-check --records-only exited 1"}}
        if rc or rep is None:
            return {"status": "error", "detail": {"error": f"circuit-check --records-only exited {rc}"
                                                  + ("" if rep is not None else " and wrote no report")}}
        outs = {o["name"]: work / o["path"] for o in task["outputs"]}
        for p in outs.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        n = CB.pack(records, outs.pop("records"))
        index = json.loads((cut / "index.json").read_text())
        made = index["bindings"]
        os.replace(cut / index["shared"], outs.pop("shared"))
        absent = []
        for name, to in sorted(outs.items()):
            binding = name.removeprefix(SLICE)
            if binding in made:
                os.replace(cut / made[binding], to)
            else:
                to.write_text(json.dumps(CB.absent(binding), sort_keys=True) + "\n")
                absent.append(binding)
        return {"status": "pass", "detail": {"bindings": rep.get("reused", 0) + len(rep.get("rebuilt", [])),
                                             "stored": rep["stored"], "records": n, "bytes": rep.get("bytes", 0),
                                             "slices": len(outs) - len(absent), "slice_bytes": rep["slices"]["bytes"],
                                             "shared_bytes": rep["slices"]["shared_bytes"], "absent": absent}}
    finally:
        for d in (records, cut):
            shutil.rmtree(d, ignore_errors=True)
