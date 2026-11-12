"""The work the min-cut bound reads off a Program (`circuit_properties.work`), on word circuits of the reference
registry's products whose work is known by hand: a training loop's scan, whose carried weight is charged from its
second iteration; a lookup whose table is the registered one or the loop's own; one array times itself; and the
largest set of arrays a work multiplies pairwise."""

from __future__ import annotations

import math

import pytest
from circuit_properties import work as WK
from circuit_properties.ops import flatten

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Coll, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as RP

K, N, V, T = 4, 3, 6, 5
F32, I32 = P.F32, P.I32
W_T, X_T = Array(N, Array(K, F32)), Array(K, F32)
TABLE_T = Array(V, Array(K, F32))
LIN = bind(RP.LinearRef, K=K, N=N, DT="f32", BIAS=False)
EMB = bind(RP.EmbeddingRef, V=V, H=K, DT="f32")


def _def(name: str, sig, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: sig, body, register=False)


def _add(B, a: Coll, b: Coll) -> Coll:
    """a + b word by word, b's words repeated to a's length."""
    n = a.type.leaves
    rep = Coll(Array(n, F32), concat([b.refs] * (n // b.type.leaves)))
    return B.batch(P.F32Add, Coll(Array(n, F32), a.refs), rep, axes=(0, 0)).reshape(a.type)


def _train(B, S, w, x):
    """One iteration: y = x W^T, and W + x row by row."""
    return tuple_of(_add(B, w, x), B.call(LIN, x, w))


TRAIN = _def("Train", ((("w", W_T), ("x", X_T)), Tuple(W_T, Array(N, F32))), _train)
LOOP = _def("Loop", ((("w", W_T), ("xs", Array(T, X_T))), Tuple(W_T, Array(T, Array(N, F32)))),
            lambda B, S, w, xs: B.scan(bind(TRAIN), w, xs=[xs]))


def loop() -> Program:
    return Program(bind(LOOP), theta=("w",))


def test_a_scan_charges_its_carried_weight_from_its_second_iteration():
    p = loop()
    w = WK.read(p)
    step = bind(TRAIN).gates
    assert w.gates == p.root.gates == T * step + p.input_gates == len(flatten(p.root).prims)
    (g,) = w.groups
    assert (g.weight, g.widths, g.out_width, g.k_dim) == ((False, True), (32, 32), 32, K)
    assert (w.N, w.O) == ((T - 1) * K * N, (T - 1) * N * 32)
    assert w.left_out == {"theta": K * N, "lookup": 0, "self": 0}
    assert len(g.edges) == 1


def _lookup_loop(carried: bool) -> Program:
    """Each iteration looks a row up by its token and multiplies it by the carried W, which it adds z to; the table is
    the registered one (shared), or carried and added z' to."""
    carry_t = Tuple(TABLE_T, W_T) if carried else W_T
    xs = (("tok", I32), ("z", X_T), ("zt", X_T))
    shared = () if carried else (("table", TABLE_T),)

    def step(B, S, carry, tok, z, zt, *table):
        tab, w = (carry[0], carry[1]) if carried else (table[0], carry)
        x = B.call(EMB, tok, tab)
        w2 = _add(B, w, z)
        out = tuple_of(_add(B, tab, zt), w2) if carried else w2
        return tuple_of(out, B.call(LIN, x, w))

    st = _def("LookupStep", ((("carry", carry_t), *xs, *shared), Tuple(carry_t, Array(N, F32))), step)
    params = (("table", TABLE_T), ("w", W_T), *((n, Array(T, t)) for n, t in xs))

    def body(B, S, table, w, tok, z, zt):
        init = tuple_of(table, w) if carried else w
        return B.scan(bind(st), init, xs=[tok, z, zt], shared=[] if carried else [table])

    fn = _def("LookupLoop", (params, Tuple(carry_t, Array(T, Array(N, F32)))), body)
    return Program(bind(fn), theta=("table", "w"))


def test_a_lookup_of_the_registered_table_is_never_charged():
    w = WK.read(_lookup_loop(carried=False))
    assert w.groups == [] and w.left_out == {"theta": K * N, "lookup": (T - 1) * K * N, "self": 0}


def test_a_lookup_of_the_loops_own_table_is_charged_from_its_second_iteration():
    w = WK.read(_lookup_loop(carried=True))
    assert w.N == (T - 1) * K * N and w.left_out == {"theta": K * N, "lookup": 0, "self": 0}


SQUARE = _def("Square", ((("x", X_T),), F32), lambda B, S, x: B.call(bind(RP.DotRef, K=K), x, x))
PAIRS = _def("Pairs", ((("x", X_T), ("y", X_T), ("z", X_T)), Array(3, F32)),
             lambda B, S, x, y, z: Coll(Array(3, F32), concat([B.call(bind(RP.DotRef, K=K), a, b).refs
                                                               for a, b in ((x, y), (y, z), (x, z))])))


def test_one_array_times_itself_is_left_out():
    w = WK.read(Program(bind(SQUARE)))
    assert w.groups == [] and w.left_out["self"] == K


def test_the_arrays_multiplied_pairwise_set_F():
    w = WK.read(Program(bind(PAIRS)))
    (g,) = w.groups
    assert g.weight == (False, False) and w.N == 3 * K and WK.clique(g.edges) == 3
    b = WK.bound(w, 1e6, 1e5)
    mc = WK.min_cut()
    assert b.omega == 3 and b.bound.F == mc.share(3) and b.bound.statement == mc.HE_SHARED


def test_the_bound_is_the_best_sub_work():
    """`bound` takes the most any sub-work gives: never less than the whole work's at its own least β and ω."""
    p = loop()
    w = WK.read(p)
    mc = WK.min_cut()
    c, k = 1e5, 1e5
    best = WK.bound(w, c, k)
    whole = mc.bound(c, k, N=w.N, O=w.O, size=w.gates, beta_w=32, beta_x=mc.beta(k), b=32, F=1.0)
    assert best.bound.slowdown == pytest.approx(whole.slowdown) and best.N == w.N
    assert WK.bound(w, c, math.inf) is None


def test_clique():
    tri = [frozenset(e) for e in ((1, 2), (2, 3), (1, 3), (3, 4))]
    assert WK.clique(tri) == 3
    assert WK.clique(tri + [frozenset(e) for e in ((1, 4), (2, 4))]) == 4
    assert WK.clique([frozenset((1, 2))]) == 2 and WK.clique([]) == 0
