"""A recorded step's counted nodes written out op by op, for a measurement: where `TE_OPS_DUMP` names a directory,
`dump` writes one JSON file per recorded step there. It changes nothing a trace reads or writes.

A row is (phase, class, op, operand shapes, dtype) with its MACs, as the capture's own functions count the node
(`capture.structure`: `_gemms`, `_other_macs`, and for a forward op counted at its work or as a gather, the count the
circuit's gates for the node say), the gates the step's circuit holds for it, its kind at its traced dtype and the kind
a record without dtypes prices it at (`gates.kind_of` with no dtype). Two checks hold the reading: the rows' MACs sum to
`StepGraph.c_macs`, and the nodes' gates to the circuit's.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from te_core import gates
from te_core.capture import structure as S
from te_core.flops import gather_macs


def _shapes(node, gm) -> list:
    """A node's tensor operands' shapes, in the order `flops.Counter` lists an op's."""
    out: list = []

    def val(v):
        if isinstance(v, torch.Tensor):
            out.append([int(s) for s in v.shape])
        elif isinstance(v, (list, tuple)):
            for x in v:
                val(x)

    def walk(a):
        if isinstance(a, torch.fx.Node):
            v = a.meta.get("val")
            if v is None and a.op == "get_attr":
                v = getattr(gm, a.target, None)
            val(v)
        elif isinstance(a, (list, tuple)):
            for x in a:
                walk(x)
        elif isinstance(a, dict):
            for x in a.values():
                walk(x)

    walk(list(node.args))
    walk(dict(node.kwargs))
    return out


def rows(g) -> tuple[list[dict], dict]:
    """The measured step's counted nodes merged by (phase, class, op, shapes, dtype), and the checks."""
    gm, c = g.graph, g.circuit
    nodes = list(gm.graph.nodes)
    phases = S._phases(gm)
    held = np.zeros(len(nodes))
    ok = c.node >= 0
    np.add.at(held, c.node[ok], c.gates[ok])
    found, unexplained = [], []
    for i, n in enumerate(nodes):
        if (n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload)
                or int(n.meta.get("te_step", 0)) != c.region):
            continue
        name, ph = S._pk(n.target), phases.get(n, "forward")
        base = {"phase": ph, "op": name, "shapes": json.dumps(_shapes(n, gm)), "gates_held": float(held[i])}
        gs = S._gemms(n)
        if gs:
            for j, gg in enumerate(gs):
                dt = S._dtype_name(gg.operands[0])
                found.append(base | {"macs": int(S._macs(gg.dims)), "class": "matmul", "dtype": dt,
                                     "kind": gates.kind_of(ph, "", name, dt)[0],
                                     "gates_held": float(held[i]) if j == 0 else 0.0})
            continue
        k, kind = S._other_macs(n)
        dt = S._dtype_name(n)
        if kind:
            found.append(base | {"macs": int(k), "class": kind, "dtype": dt, "kind": gates.kind_of(ph, "", name, dt)[0]})
            continue
        if held[i] <= 0:
            continue
        k_work, work = gates.kind_of(ph, "", name, dt)[0], S._work(n)
        if work and math.isclose(work * gates.per_mac(k_work), held[i], rel_tol=1e-9):
            found.append(base | {"macs": int(work), "class": "work", "dtype": dt, "kind": k_work})
            continue
        try:
            a0 = n.args[0].meta["val"]
            got = gather_macs(name, a0.shape, n.args[1] if len(n.args) > 1 else None, n.meta["val"].numel())
            bits = S._dtype_bits(a0.dtype)
        except Exception:  # noqa: BLE001 - a node with no gather reading is listed unexplained
            got, bits = 0, 16
        if got and math.isclose(got * gates.per_mac("gather", bits), held[i], rel_tol=1e-9):
            found.append(base | {"macs": int(got), "class": "gather", "dtype": dt, "kind": "gather"})
            continue
        unexplained.append(base | {"node": n.name, "work": int(work), "gather": int(got)})
    merged: dict = {}
    for r in found:
        key = (r["phase"], r["class"], r["op"], r["shapes"], str(r["dtype"]), r["kind"])
        m = merged.setdefault(key, {"phase": r["phase"], "class": r["class"], "op": r["op"], "shapes": r["shapes"],
                                    "dtype": r["dtype"], "kind": r["kind"], "nodes": 0, "macs": 0, "gates_held": 0.0,
                                    "record_kind": gates.kind_of(r["phase"], "", r["op"], None)[0]})
        m["nodes"] += 1
        m["macs"] += r["macs"]
        m["gates_held"] += r["gates_held"]
    out = sorted(merged.values(), key=lambda m: -m["gates_held"])
    for m in out:
        m["price"], m["record_price"] = gates.per_mac(m["kind"]) if m["kind"] != "gather" else None, \
            gates.per_mac(m["record_kind"]) if m["record_kind"] != "gather" else None
    macs = sum(m["macs"] for m in out)
    checks = {"rows_macs": macs, "c_macs": int(g.c_macs), "rows_macs_are_c_macs": macs == int(g.c_macs),
              "nodes_gates": float(sum(m["gates_held"] for m in out)), "circuit_gates": float(c.gates.sum()),
              "unexplained_nodes": unexplained[:40], "unexplained": len(unexplained)}
    checks["nodes_gates_are_the_circuits"] = math.isclose(checks["nodes_gates"], checks["circuit_gates"], rel_tol=1e-9)
    return out, checks


def dump(g, about: dict) -> None:
    """Write `rows(g)` under `TE_OPS_DUMP` (nothing where it is unset); a failure is written, never raised."""
    where = os.environ.get("TE_OPS_DUMP")
    if not where or g.circuit is None:
        return
    try:
        found, checks = rows(g)
        body = {"about": about, "checks": checks, "rows": found}
    except Exception as e:  # noqa: BLE001 - the dump is beside the trace and never fails it
        body = {"about": about, "error": f"{type(e).__name__}: {e}"}
    text = json.dumps(body, default=str)
    Path(where).mkdir(parents=True, exist_ok=True)
    Path(where, f"ops-{os.getpid()}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.json").write_text(text)
