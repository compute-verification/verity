"""`trace.folded` and a Definition traced with calls: an argument pattern names each argument as a constant or the XOR of
parameters, the bits the arguments hold (`pattern_of`), its string is the one `encode_pattern` writes and `parse_pattern`
reads back, and a builder that reaches a folded Definition through ``fold_call`` computes, with calls, what it computes
without: the same outputs and the same ANDs, the folded circuit written once per pattern."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array


def _f(C, a):
    return [C.AND(a[0], a[1]), C.AND(a[1] ^ a[2], a[0]), C.AND(a[3], a[4]), C.AND(a[5], a[2]), a[6] ^ a[0]]


FOLD = TR.folded("TraceCallsToyFold", 1, _f, 7, 5, "a toy folded builder", register=False)


def _args(x0, x1, x2):
    return [x0, x1 ^ x0, TR.ONE, x0.inv(), x1 ^ x2, x0 ^ x2 ^ TR.ONE, x2]


def _build(C, S, x):
    """``_f`` on arguments that are XORs of one another: twice, so the second call has the first's pattern."""
    a = _args(*x)
    fold = getattr(C, "fold_call", None)
    y = fold(FOLD, a) if fold else _f(C, a)
    b = _args(y[0], y[1], y[2])
    z = fold(FOLD, b) if fold else _f(C, b)
    return list(y) + list(z)


def _sig(S):
    return (("x", Array(3, BL.BIT)),), Array(10, BL.BIT)


FLAT = TR.part("TraceCallsToyFlat", (), _sig, _build, "the toy builder", register=False)
CALLS = TR.part("TraceCallsToyCalls", (), _sig, _build, "the toy builder", register=False, calls=True)


def _ands(T: TR.Traced) -> int:
    """A trace's ANDs, each call's its folded circuit's."""
    n = T.counts["and"]
    for kind, _ in T.gates:
        if kind.startswith("call:"):
            _, name, P = kind.split(":")
            n += TR.FOLDED[name].traced(P).counts["and"]
    return n


def test_a_call_trace_computes_the_flat_trace() -> None:
    T = CALLS.traced_calls()
    calls = [k for k, _ in T.gates if k.startswith("call:")]
    assert calls == ["call:TraceCallsToyFold:p0_x0.1_o1_c0_x1.2_y0.2_p2"] * 2
    assert T.counts["and"] == 0 and _ands(T) == FLAT.traced().counts["and"] == 6
    assert CALLS.traced() == FLAT.traced()
    x = np.array([[(i >> j) & 1 for j in range(3)] for i in range(8)], dtype=np.uint8)
    assert evaluate_bits(bind(CALLS), x).tolist() == evaluate_bits(bind(FLAT), x).tolist()
    ids = [n.fn.id for n in bind(CALLS).body.nodes]
    assert ids.count("TraceCallsToyFold_v1{P='p0_x0.1_o1_c0_x1.2_y0.2_p2'}") == 2 and not any(i.startswith("And") for i in ids)


def test_the_folded_definition_is_its_builder_at_the_pattern() -> None:
    P = "p0_x0.1_o1_c0_x1.2_y0.2_p2"
    x = np.array([[(i >> j) & 1 for j in range(3)] for i in range(8)], dtype=np.uint8)
    got = evaluate_bits(bind(FOLD, P=P), x).tolist()
    for row, (p0, p1, p2) in zip(got, x.tolist()):
        args = [p0, p0 ^ p1, 1, 1 - p0, p1 ^ p2, 1 - (p0 ^ p2), p2]
        want = [args[0] & args[1], (args[1] ^ args[2]) & args[0], args[3] & args[4], args[5] & args[2], args[6] ^ args[0]]
        assert row == want


def test_patterns_name_each_argument_once() -> None:
    """`pattern_of` on random arguments: XORs of a few bits, constants, repeats and complements. Each argument is the XOR P names
    of its parameters, each a bit the arguments hold, once; and P parses back to its entries."""
    rng = random.Random("trace-call-patterns")
    C = TR.CallTracingCircuit()
    bits = C.input("x", 12)
    for _ in range(200):
        args = []
        for _ in range(rng.randrange(1, 16)):
            r = rng.random()
            if r < 0.1:
                args.append(rng.choice((TR.ONE, TR.ZERO)))
            elif r < 0.2 and args:
                args.append(rng.choice(args))
            else:
                f = TR.ZERO
                for b in rng.sample(bits, rng.randrange(1, 4)):
                    f = f ^ b
                args.append(f.inv() if rng.random() < 0.3 else f)
        P, ops = TR.pattern_of(args)
        entries, n = TR.parse_pattern(P, len(args))
        assert n == len(ops) and TR.encode_pattern(entries) == P
        assert all(len(o.s) == 1 and o.c == 0 for o in ops) and len({o.s for o in ops}) == n
        for f, e in zip(args, entries):
            if isinstance(e, str):
                assert f.is_const() and f.c == (e == "o")
                continue
            g = TR.ZERO
            for i in e[0]:
                g = g ^ ops[i]
            assert frozenset(g.s) == frozenset(f.s) and g.c ^ e[1] == f.c


@pytest.mark.parametrize("P, why", [
    ("p0n3", "names 3 arguments, not 7"),
    ("p1_p0_z5", "parameter 1 is read before parameter 0"),
    ("p0_x0.0_z5", "isn't an XOR of parameters in increasing order"),
    ("p0_p1_x1.0_z4", "isn't an XOR of parameters in increasing order"),
    ("p0_x0.2_p1_z4", "parameter 2 is read before parameter 1"),
    ("x1.2_z6", "parameter 2 is read before parameter 0"),
    ("p0_p1_z5", "isn't written as `encode_pattern` writes it"),
    ("z2_z5", "isn't written as `encode_pattern` writes it"),
    ("p0n1_z6", "is no token"),
    ("p00_z6", "is no token"),
    ("", "a pattern P is a nonempty str"),
])
def test_a_pattern_is_refused_unless_canonical(P: str, why: str) -> None:
    with pytest.raises(ValueError, match=why.replace("(", r"\(")):
        TR.parse_pattern(P, 7)
    with pytest.raises(ValueError):
        bind(FOLD, P=P)


def test_a_pattern_round_trips() -> None:
    for P in ("p0n7", "z3_p0_o2_p1", "p0_c0_p1_x0.1_y0.1_p0_z1", "p0_x0.1_o1_c0_x1.2_y0.2_p2"):
        entries, _ = TR.parse_pattern(P, 7)
        assert TR.encode_pattern(entries) == P
