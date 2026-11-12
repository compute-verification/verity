"""The work the min-cut bound reads off a Program (`circuit_properties.work`), on word circuits of the reference
registry's products whose work is known by hand: a training loop's scan, whose carried weight is charged from its
second iteration; a lookup whose table is the registered one or the loop's own; one array times itself; the largest
set of arrays a work multiplies pairwise; a weight decoded from a ternary code, and a one-hot row, whose values hold
fewer bits than their width; and |C| in Boolean gates."""

from __future__ import annotations

import math
from collections import Counter

import pytest
from circuit_properties import work as WK
from circuit_properties.ops import flatten

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.bits import compile as compile_bits
from verity.circuits.program import Program, is_input
from verity.circuits.refs import Coll, Strided, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.boolean import NoBooleanVersion, boolean_version
from verity.definitions.ops.prims import const
from verity.definitions.registry import prims as P
from verity.definitions.registry import ref_prims as RP
from verity.definitions.registry.sampling import bit_at

K, N, V, T = 4, 3, 6, 5
F32, I32 = P.F32, P.I32
W_T, X_T = Array(N, Array(K, F32)), Array(K, F32)
TABLE_T = Array(V, Array(K, F32))
LIN = bind(RP.LinearRef, K=K, N=N, DT="f32", BIAS=False)
EMB = bind(RP.EmbeddingRef, V=V, H=K, DT="f32")
#: a ternary code's row: one bit a value in each of two plane words
KT = 32
LIN_T = bind(RP.LinearRef, K=KT, N=N, DT="f32", BIAS=False)
#: a one-hot row's length, more values than a token has bits
VH = 64
LIN_V = bind(RP.LinearRef, K=VH, N=N, DT="f32", BIAS=False)


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


def _boolean(fid_counts: Counter, by_id: dict) -> int:
    return sum(n * sum(compile_bits(boolean_version(by_id[f])).counts.values()) for f, n in fid_counts.items())


def test_a_scan_charges_its_carried_weight_from_its_second_iteration():
    p = loop()
    w = WK.read(p)
    prims = [f for f in flatten(p.root).prims if not is_input(f)]
    assert w.gates == T * WK.boolean_gates(Program(bind(TRAIN))) == \
        _boolean(Counter(f.id for f in prims), {f.id: f for f in prims})
    (g,) = w.groups
    assert (g.weight, g.widths, g.out_width, g.k_dim, g.bits) == ((False, True), (32, 32), 32, K, (32.0, 32.0))
    assert (w.N, w.O) == ((T - 1) * K * N, (T - 1) * N * 32)
    assert w.left_out == {"theta": K * N, "lookup": 0, "self": 0}
    assert len(g.edges) == 1


def test_a_primitive_with_no_boolean_version_is_refused_by_name():
    tanh = _def("Tanh", ((("x", F32),), F32), lambda B, S, x: B.call(RP.F32TanhRn, x))
    with pytest.raises(NoBooleanVersion, match="F32TanhRn"):
        WK.read(Program(bind(tanh)))


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


def _repeat(c: Coll, t, inner: int, outer_stride: int, inner_stride: int, n: int) -> Coll:
    r = c.refs
    return Coll(Array(n, t), Strided(r.space, r.idx, r.base, inner, outer_stride, inner_stride, n))


def _ternary_loop() -> Program:
    """A weight carried as a ternary code, two plane words and a scale a row, decoded in each iteration: value j of
    row r is the scale times +1 where bit j of the first plane is set, -1 where the second's is, 0 else. The code is
    updated by the iteration's inputs, so it is fresh from the second iteration."""
    planes_t, scale_t, wt = Array(N, I32), Array(N, F32), Array(N, Array(KT, F32))
    carry_t = Tuple(planes_t, planes_t, scale_t)
    xs = (("x", Array(KT, F32)), ("u", planes_t), ("v", planes_t), ("s", scale_t))
    bit, nk = bit_at(32), N * KT

    def step(B, S, carry, x, u, v, s, idx):
        p1, p2, sc = carry[0], carry[1], carry[2]
        ix = _repeat(idx, I32, KT, 0, 1, nk)
        b1 = B.batch(bit, _repeat(p1, I32, KT, 1, 0, nk), ix, axes=(0, 0))
        b2 = B.batch(bit, _repeat(p2, I32, KT, 1, 0, nk), ix, axes=(0, 0))
        one, minus, zero = (B.call(const(32, c)) for c in (0x3F800000, 0xBF800000, 0))
        rep = lambda c: Coll(Array(nk, F32), concat([c.refs] * nk))  # noqa: E731
        low = B.batch(P.SelectF32, b2, rep(minus), rep(zero), axes=(0, 0, 0))
        digit = B.batch(P.SelectF32, b1, rep(one), low, axes=(0, 0, 0))
        w = B.batch(P.F32Mul, digit, _repeat(sc, F32, KT, 1, 0, nk), axes=(0, 0)).reshape(wt)
        new = tuple_of(B.batch(P.I32Add, p1, u, axes=(0, 0)), B.batch(P.I32Add, p2, v, axes=(0, 0)),
                       B.batch(P.F32Add, sc, s, axes=(0, 0)))
        return tuple_of(new, B.call(LIN_T, x, w))

    st = _def("TernaryStep", ((("carry", carry_t), *xs, ("idx", Array(KT, I32))), Tuple(carry_t, Array(N, F32))),
              step)
    params = (("p1", planes_t), ("p2", planes_t), ("scale", scale_t), *((n, Array(T, t)) for n, t in xs),
              ("idx", Array(KT, I32)))

    def body(B, S, p1, p2, scale, x, u, v, s, idx):
        return B.scan(bind(st), tuple_of(p1, p2, scale), xs=[x, u, v, s], shared=[idx])

    fn = _def("TernaryLoop", (params, Tuple(carry_t, Array(T, Array(N, F32)))), body)
    return Program(bind(fn), registered=("idx",), theta=("p1", "p2", "scale"))


def test_a_weight_decoded_from_a_ternary_code_holds_its_codes_bits():
    """The decoded weight's values hold the carried bits that reach them, two planes' and a row's scale over a row's
    values, so the bound prices it there, under the width and under (H)'s default for a weight."""
    w = WK.read(_ternary_loop())
    (g,) = w.groups
    assert g.weight == (False, True) and g.widths == (32, 32) and w.N == (T - 1) * N * KT
    assert g.bits == (32.0, pytest.approx(2 + 32 / KT))
    mc, k = WK.min_cut(), 1e5
    b = WK.bound(w, 1e6, k)
    assert (b.bound.beta_w, b.bound.beta_x) == (pytest.approx(2 + 32 / KT), mc.beta(k)) and b.bound.b == mc.BF16


ONE_HOT = _def("OneHot", ((("tok", I32), ("ids", Array(VH, I32)), ("e", Array(N, Array(VH, F32)))), Array(N, F32)),
               lambda B, S, tok, ids, e: B.call(LIN_V, B.batch(
                   P.SelectF32, B.batch(P.I32Eq, Coll(Array(VH, I32), concat([tok.refs] * VH)), ids, axes=(0, 0)),
                   *(Coll(Array(VH, F32), concat([B.call(const(32, c)).refs] * VH)) for c in (0x3F800000, 0)),
                   axes=(0, 0, 0)), e))


def test_a_one_hot_row_holds_its_tokens_bits():
    """A table read by a token written as a one-hot row times the table: the row's values hold the token's bits
    between them, so its β is the token's width over the table's rows."""
    w = WK.read(Program(bind(ONE_HOT), registered=("ids",)))
    (g,) = w.groups
    assert g.bits == (pytest.approx(32 / VH), 32.0) and w.N == N * VH
    assert WK.bound(w, 1e6, 1e5).bound.beta_x == pytest.approx(32 / VH)


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


def test_the_bound_is_the_best_sub_work_at_the_calculators_defaults():
    """`bound` takes the most any sub-work gives: never less than the whole work's at (H)'s and (E)'s defaults (a
    weight's and an output's bf16 width, an activation's β(k)), each under its own width."""
    p = loop()
    w = WK.read(p)
    mc = WK.min_cut()
    c, k = 1e5, 1e5
    best = WK.bound(w, c, k)
    whole = mc.bound(c, k, N=w.N, O=w.O, size=w.gates, F=1.0)
    assert (best.bound.beta_w, best.bound.beta_x, best.bound.b) == (mc.BF16, mc.beta(k), mc.BF16)
    assert best.bound.slowdown == pytest.approx(whole.slowdown) and best.N == w.N
    assert WK.bound(w, c, math.inf) is None


def test_clique():
    tri = [frozenset(e) for e in ((1, 2), (2, 3), (1, 3), (3, 4))]
    assert WK.clique(tri) == 3
    assert WK.clique(tri + [frozenset(e) for e in ((1, 4), (2, 4))]) == 4
    assert WK.clique([frozenset((1, 2))]) == 2 and WK.clique([]) == 0
