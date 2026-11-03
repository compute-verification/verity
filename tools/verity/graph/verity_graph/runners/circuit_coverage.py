"""circuit-coverage: the catalog bindings circuit-check makes (`circuit-check --all --bindings`, from the records the
circuit-catalog task built, var ``records``) are the ones the circuit-binding tasks were listed for (var ``bindings``, a JSON
list), read from the binding files' syntax: a binding no task checks, or a task whose binding the catalog doesn't make, is a
fail.  `detail` counts the bindings and their targets and names each such binding, with the roots the catalog couldn't make
and the registered families no target is of, which `test_circuit_check.py` holds."""
from __future__ import annotations

import json
from pathlib import Path

from . import circuit_binding as CB


def run(task: dict, work: Path) -> dict:
    c = CB.cache()
    CB.unpack(work / task["vars"]["records"], c)
    rc, rep = CB.checker(task, work, "--all", "--cache", str(c), "--bindings")
    if rc or rep is None:
        return {"status": "error", "detail": {"error": f"circuit-check --bindings exited {rc}"
                                              + ("" if rep is not None else " and wrote no report")}}
    listed, made = set(json.loads(task["vars"]["bindings"])), rep["bindings"]
    detail = {"bindings": len(made), "targets": sum(map(len, made.values())), "unlisted": sorted(set(made) - listed),
              "unmade": sorted(listed - set(made)), "root_errors": rep["root_errors"], "unchecked": rep["unchecked"]}
    return {"status": "fail" if detail["unlisted"] or detail["unmade"] else "pass", "detail": detail}
