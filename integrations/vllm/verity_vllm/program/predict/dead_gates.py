"""Each compiled request Program's dead gates as circuit-check counts them, over a `steps.Compiled`."""
from __future__ import annotations

from typing import Any


def dead_gates_of(compiled: Any) -> dict:
    """Each compiled Program's dead gates as circuit-check counts them (`liveness.dead_gates`, transitive: gates no output depends
    on), with the unread inputs apart, and the dead gates under each of the root's nodes by definition (its id up to `{`)."""
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.intervals import full, total
    from verity.ml.circuits.liveness import Liveness, dead_gates
    out = {}
    for digest, program in sorted(compiled.programs.items()):
        rep = dead_gates(program, limit=0, transitive=True)
        root, lv = program.root, Liveness(transitive=True)
        res = lv.analyse(root, full(root.body.ret_type.leaves))
        by: dict[str, int] = {}
        for node, live in zip(root.body.nodes, res.node_live):
            if node.form == "call":
                n = lv.analyse(node.fn, live).dead
            elif node.form == "primitive" or isinstance(node.fn, PrimitiveDefinition):
                n = (node.n if node.form == "batch" else 1) - total(live)
            else:
                n = 0
            if n:
                name = node.fn.id.split("{", 1)[0]
                by[name] = by.get(name, 0) + n
        if rest := res.dead - sum(by.values()):
            by["(batch or scan at the root)"] = rest
        out[digest] = {"dead_computed": rep.computed, "unread_inputs": rep.input, "approximate": rep.approximate,
                       "by_definition": dict(sorted(by.items(), key=lambda kv: -kv[1]))}
    return {"programs": out, "dead_computed": sum(p["dead_computed"] for p in out.values())}
