"""hidden_zk's compact unit circuit (`compact_stage`) lays a circuit out as the library does: the same rows, netlist text,
counts and values, over random circuits of inputs and ANDs with constants, commits, assertions and dead gates, spooled in
many small blocks so dead chains cross them."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hidden_zk"))

import compact_stage as CST
from verity.ml.circuits.boolean import forms
from verity_flock import ir_lower as IL

IN = {"in0": 5, "in1": 130, "in2": 7}
OUT = {"out0": 12, "out1": 4}


def _build(C, seed: int, n: int, late_input: bool = False):
    rng = random.Random(seed)
    wires = [w for nm, k in IN.items() for w in C.input(nm, k)]

    def pick():
        f = forms.ZERO if rng.random() < 0.9 else forms.ONE
        for w in rng.sample(wires, rng.randint(1, min(6, len(wires)))):
            f = f ^ w
        return f

    dead = []
    for i in range(n):
        r = rng.random()
        if r < 0.05:
            wires += C.commit([pick()])
        elif r < 0.12:
            dead.append(C.AND(pick(), dead[-1] if dead and rng.random() < 0.7 else pick()))
        elif r < 0.14:
            wires.append(C.AND(pick(), forms.ONE if rng.random() < 0.5 else pick().inv()))
        else:
            wires.append(C.AND(pick(), pick()))
        if late_input and i == n // 2:
            wires += C.input("late", 1)
    for name in ("a0", "a1"):
        if not (f := pick()).is_const():
            C.assert_zero(name, f)
    C.output("out0", [pick() for _ in range(OUT["out0"] - 2)] + [forms.ZERO, forms.ONE])
    C.output("out1", wires[-OUT["out1"]:])
    return C


def _ports(rng, T):
    return [rng.integers(0, 1 << min(k, 16), T, dtype=np.uint64) for k in IN.values()]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("split", [None, [1]], ids=["one-group", "two-groups"])
def test_compact_layout_is_the_library_layout(seed, split, tmp_path, monkeypatch):
    monkeypatch.setattr(CST, "FLUSH", 64)
    CST.set_root(tmp_path)
    n = 300 + 40 * seed
    lib = IL.layout(_build(forms.Circuit(), seed, n), list(IN), list(OUT), in_split=split)
    C = _build(CST.CompactCircuit(), seed, n)
    assert C.compact() and len(C._spool.blocks) > 10
    lay = CST.layout(C, list(IN), list(OUT), in_split=split)
    assert isinstance(lay, IL.Layout)
    assert (list(lay.rows_a), list(lay.rows_b)) == (lib.rows_a, lib.rows_b)
    assert [lay.rows_a[i] for i in range(lay.useful)] == lib.rows_a and lay.rows_b[-3:] == lib.rows_b[-3:]
    assert (lay.const, lay.in_groups, lay.out_groups, lay.counts) == (lib.const, lib.in_groups, lib.out_groups, lib.counts)
    assert lay.counts["ands"] < n
    assert b"".join(lay.pieces("unit/x", workers=3)) == IL.netlist(lib, "unit/x").encode()
    rng = np.random.default_rng(seed)
    for T, flip in ((1, None), (9, None), (9, (lay.first + 3, 4))):
        for ports in (_ports(rng, T), [np.zeros(T, np.uint64) for _ in IN]):
            got, want = lay.evaluate(ports, T, flip), IL.evaluate_layout(lib, ports, T, flip)
            assert got[1] == want[1]
            assert all(np.array_equal(a, b) for a, b in zip(got[0], want[0]))
    CST.cleanup()
    assert not list(tmp_path.iterdir())


def test_a_circuit_compact_stage_does_not_lay_out_reads_back_as_the_library_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(CST, "FLUSH", 64)
    CST.set_root(tmp_path)
    want = _build(forms.Circuit(), 7, 200, late_input=True)
    C = _build(CST.CompactCircuit(), 7, 200, late_input=True)
    assert not C.compact()

    def gates(C):
        return [(k, (frozenset(d[0].s), d[0].c, frozenset(d[1].s), d[1].c) if k == "and" else d) for k, d in zip(C.kind, C.data)]

    assert gates(C) == gates(want) and C.phase_of == want.phase_of and C.live() == want.live()
    assert not list(tmp_path.iterdir())
    w = C.AND(C.inputs["in0"][0], C.inputs["in0"][1])
    assert w.s == {len(want.kind)} and C.kind[-1] == "and" and len(C.data) == len(C.phase_of) == len(want.kind) + 1


def _chain():
    from verity.ml.circuits.boolean import BIT
    from verity.ml.circuits.boolean.trace import definition
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.types import Array

    w, n = 12, 6
    add = definition("CompactStageToyAdd", 1, lambda C, a, b: forms.add(C, a, b, width=w), (("a", Array(w, BIT)),
                     ("b", Array(w, BIT))), Array(w, BIT), register=False)

    def body(B, S, x):
        y = kept = x[0]
        for i in range(1, n):
            y = B.call(add, y, x[i])
            if i == 2:
                kept = y
        return B.call(add, B.call(add, y, kept), x[1])

    sig = lambda S: ((("x", Array(n, Array(w, BIT))),), Array(w, BIT))  # noqa: E731
    return bind(CompositeDefinition("CompactStageToyChain", 1, (), sig, body, register=False)), w * n


def test_lower_drops_each_wire_after_its_last_read(monkeypatch):
    """A chain of uncommitted adds, one partial sum and one input read twice: `compact_stage.lower` builds the gates
    `ir_lower.lower` builds, in order, and holds only the returns once done."""
    d, n_in = _chain()

    def run(lower):
        C = forms.Circuit()
        bits = C.input("x", n_in)
        return C, lower(C, d, [[b] for b in bits])

    want_c, want = run(IL.lower)
    held = []
    lower_gates = IL.lower_gates
    monkeypatch.setattr(IL, "lower_gates", lambda C, g, wires, *a, **k: held.append((g, wires)) or lower_gates(C, g, wires, *a, **k))
    got_c, got = run(CST.lower)
    key = lambda f: (frozenset(f.s), f.c)  # noqa: E731
    assert [[key(f) for f in w] for w in got] == [[key(f) for f in w] for w in want]
    assert got_c.kind == want_c.kind and got_c.kind.count("and") > 50
    assert [tuple(key(f) for f in dd) for dd in got_c.data[n_in:]] == [tuple(key(f) for f in dd) for dd in want_c.data[n_in:]]
    (call, wires), = held
    assert set(wires) == set(call.returns) and not wires.left


def test_render_is_decimal_text():
    rng = np.random.default_rng(3)
    tok = np.concatenate([[0, 9, 10, 99, 100, 999, 1000, (1 << 32) - 1, 4_000_000_000],
                          rng.integers(0, 1 << 32, 5000), rng.integers(0, 200, 5000)]).astype(np.int64)
    last = np.sort(rng.choice(len(tok) - 1, 300, replace=False))
    last = np.append(last, len(tok) - 1)
    lines, at = [], 0
    for e in last.tolist():
        lines.append(" ".join(map(str, tok[at:e + 1].tolist())) + "\n")
        at = e + 1
    assert CST.render(tok, last).tobytes() == "".join(lines).encode()
