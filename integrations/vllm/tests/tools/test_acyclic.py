"""`tools/acyclic.py`: Tarjan's components against a brute-force reachability, and a toy Definition whose Q_word cut has two units
reading each other (it returns x, then z, then y with y = f(x) and z = g(y), so x and z share an output unit), which `check_cut`
accepts; the same Definition returning x, y, z is cut without a cycle."""
from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "acyclic.py"
_spec = importlib.util.spec_from_file_location("acyclic", SCRIPT)
AC = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("acyclic", AC)
_spec.loader.exec_module(AC)


def _csr(U: int, edges: set[tuple[int, int]]) -> tuple[list[int], list[int]]:
    start, nbr = [0], []
    for v in range(U):
        nbr.extend(sorted(w for x, w in edges if x == v))
        start.append(len(nbr))
    return start, nbr


def test_tarjan_finds_the_components_reachability_finds():
    rng = random.Random(0)
    for _ in range(200):
        U = rng.randrange(1, 12)
        edges = {(rng.randrange(U), rng.randrange(U)) for _ in range(rng.randrange(0, 3 * U))}
        edges = {(a, b) for a, b in edges if a != b}
        reach = [{v} for v in range(U)]
        for _ in range(U):
            for a, b in edges:
                reach[a] |= reach[b]
        want = {frozenset(w for w in range(U) if w in reach[v] and v in reach[w]) for v in range(U)}
        want = sorted(sorted(c) for c in want if len(c) > 1)
        start, nbr = _csr(U, edges)
        comps = AC.tarjan(U, start, nbr)
        assert sorted(comps) == want
        for c in comps:
            cyc = AC.shortest_cycle(c, start, nbr)
            assert all((x, y) in edges for x, y in zip(cyc, cyc[1:] + cyc[:1])) and set(cyc) <= set(c)


def _packed(order: str):
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.scalar import I32, I32Eq, SelectI32
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Tuple, Value

    def body(B, S, a, b):
        v = {"x": B.call(I32Eq, a, b)}
        v["y"] = B.call(SelectI32, v["x"], a, b)
        v["z"] = B.call(I32Eq, v["y"], a)
        return tuple_of(*(v[k] for k in order))

    ret = Tuple(*(I32 if k == "y" else Value(1) for k in order))
    return bind(CompositeDefinition(f"AcyclicToy{order.upper()}", 0, (), lambda S: ((("a", I32), ("b", I32)), ret), body,
                                    register=False))


def test_two_output_units_that_read_each_other_are_one_component_and_check_cut_accepts_them():
    from verity.ml.circuits.cut import CallGraph, check_cut, cut_word
    fn = _packed("xzy")
    r = AC.cycles(fn, {})
    assert (r["how"], r["units"], r["sccs"], r["scc_units"], r["sizes"]) == ("flat", 2, 1, 2, {"2": 1})
    ex = r["example"]
    assert ex["cycle"] == [0, 1] and ex["kinds"] == ["output", "output"]
    assert [(x["reader"]["prim"], x["producer"]["prim"]) for x in ex["reads"]] == [("SelectI32_v1", "I32Eq_v1"),
                                                                                  ("I32Eq_v1", "SelectI32_v1")]
    assert ex["returned_at"] == {"0": [0, 1], "1": [2]}
    own, _kinds, _head = cut_word(CallGraph(fn), 16)
    assert check_cut(CallGraph(fn), own).ok

    acyclic = AC.cycles(_packed("xyz"), {})
    assert (acyclic["units"], acyclic["sccs"], acyclic["example"]) == (3, 0, None)


def test_a_row_counts_its_cyclic_calls_and_their_share(tmp_path):
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.scalar import I32, I32Add
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Tuple, Value

    cyc, ok = _packed("xzy"), _packed("xyz")

    def body(B, S, a, b):
        s = B.call(I32Add, a, b)
        p = B.call(cyc, s, b)
        q = B.call(ok, a, s)
        return tuple_of(p, q)

    sig = lambda S: ((("a", I32), ("b", I32)), Tuple(Tuple(Value(1), Value(1), I32), Tuple(Value(1), I32, Value(1))))
    root = bind(CompositeDefinition("AcyclicRow", 0, (), sig, body, register=False))
    path = tmp_path / "descriptor.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(encode_program(Program(root)), f)
    log = tmp_path / "log.jsonl"
    r = AC.analyze_descriptor(path, AC.Cycles(log=log), "toy")
    t = r["totals"]
    assert r["topological"] and (t["calls"], t["units"], t["cyclic_calls"], t["cyclic_units"], t["sccs"]) == (3, 6, 1, 2, 1)
    assert r["cyclic_families"] == ["AcyclicToyXZY_v0"]
    again = AC.analyze_descriptor(path, AC.Cycles([log]), "toy")
    assert again["totals"] == t and again["source"] == {"cached": 3}
    md = AC.render([r])
    assert "| toy | 3 | 1 | 33.33% / 33.33% / " in md and "AcyclicToyXZY_v0 (1)" in md

    part = AC.analyze_descriptor(path, AC.Cycles(gates_below=2), "toy")
    pt = part["totals"]
    assert (pt["calls"], pt["units"], pt["cyclic_calls"], pt["pending_definitions"], pt["pending_calls"]) == (3, 1, 0, 2, 2)
    assert "(2 Definitions, 2 Calls pending)" in AC.render([part])
    merged = AC.merge(json.loads(json.dumps(part)), [log])
    assert {k: merged["totals"][k] for k in t} == t and merged["cyclic_families"] == r["cyclic_families"]
    upper = AC.analyze_descriptor(path, AC.Cycles(gates_from=2), "toy")["totals"]
    assert (upper["units"], upper["cyclic_calls"], upper["pending_calls"]) == (5, 1, 1)
