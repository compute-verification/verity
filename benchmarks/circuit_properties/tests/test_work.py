"""The work the min-cut bound reads off a Program (`circuit_properties.work`), on word circuits of the reference
registry's products whose work is known by hand: a training loop's scan, whose carried weights are charged from
their second iteration, whose batch is free, and whose value input's rows are left out however the Program writes
them; the two premises on carried words, held as facts; a lookup whose table is the registered one or the loop's own; a Program with
no scan; one array times itself; two activations; the largest set of arrays a work multiplies pairwise; a weight
decoded from a ternary code; a one-hot row; a rank-4 bottleneck between two carried weights; each operand's β at
min(16, β(k)); and |C| in Boolean gates."""

from __future__ import annotations

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
W_T, X_T, Y_T = Array(N, Array(K, F32)), Array(K, F32), Array(N, F32)
TABLE_T = Array(V, Array(K, F32))
LIN = bind(RP.LinearRef, K=K, N=N, DT="f32", BIAS=False)
DOT = bind(RP.DotRef, K=K)
EMB = bind(RP.EmbeddingRef, V=V, H=K, DT="f32")
#: a ternary code's row: one bit a value in each of two plane words
KT = 32
LIN_T = bind(RP.LinearRef, K=KT, N=N, DT="f32", BIAS=False)
#: a one-hot row's length, more values than a token has bits
VH = 64
LIN_V = bind(RP.LinearRef, K=VH, N=N, DT="f32", BIAS=False)
#: a rank-R bottleneck between two weights of a D-wide row
D, R = 32, 4
#: a policy's k at which (H) states β(k) and a K = D product makes a paid row paid (D ≥ K* = ½√(k/32))
KB = 65536.0
C = 1e13
MC = WK.min_cut()


def _def(name: str, sig, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: sig, body, register=False)


def _ew(B, prim, a: Coll, b: Coll) -> Coll:
    """prim(a, b) word by word, b's words repeated to a's length."""
    n = a.type.leaves
    rep = Coll(Array(n, F32), concat([b.refs] * (n // b.type.leaves)))
    return B.batch(prim, Coll(Array(n, F32), a.refs), rep, axes=(0, 0)).reshape(a.type)


def _add(B, a: Coll, b: Coll) -> Coll:
    return _ew(B, P.F32Add, a, b)


def _mul(B, a: Coll, b: Coll) -> Coll:
    return _ew(B, P.F32Mul, a, b)


def _loop(name: str, carry: tuple, xs: tuple, y_t, step, shared: tuple = (), registered: tuple = ()) -> Program:
    """A scan of T iterations of ``step(B, carry, *xs, *shared) -> (new carry fields, y)``, each of ``carry``,
    ``xs`` and ``shared`` (name, type): the carry `θ`, ``xs`` an iteration's inputs, ``shared`` the loop's."""
    carry_t = Tuple(*(t for _, t in carry))

    def body(B, S, c, *rest):
        new, y = step(B, [c[i] for i in range(len(carry))], *rest)
        return tuple_of(tuple_of(*new), y)

    st = _def(name + "Step", ((("carry", carry_t), *xs, *shared), Tuple(carry_t, y_t)), body)
    params = (*carry, *((n, Array(T, t)) for n, t in xs), *shared)

    def run(B, S, *a):
        nc, nx = len(carry), len(xs)
        return B.scan(bind(st), tuple_of(*a[:nc]), xs=list(a[nc:nc + nx]), shared=list(a[nc + nx:]))

    fn = _def(name, (params, Tuple(carry_t, Array(T, y_t))), run)
    return Program(bind(fn), registered=registered, theta=tuple(n for n, _ in carry))


def loop() -> Program:
    """y = (x ⊙ g) W^T, and W and g plus x: the carry is θ′ from the second iteration, so is x ⊙ g, and x is free."""
    return _loop("Train", (("w", W_T), ("g", X_T)), (("x", X_T),), Y_T,
                 lambda B, c, x: ((_add(B, c[0], x), _add(B, c[1], x)), B.call(LIN, _mul(B, x, c[1]), c[0])))


def carried_loop() -> Program:
    """y = (u ⊙ g) W^T, u, g and W carried and each added the iteration's z: no operand reads an input, so the product
    of two weights is charged from the second iteration."""
    return _loop("Carried", (("w", W_T), ("g", X_T), ("u", X_T)), (("z", X_T),), Y_T,
                 lambda B, c, z: (tuple(_add(B, a, z) for a in c), B.call(LIN, _mul(B, c[2], c[1]), c[0])))


def _gathered(w: WK.Work) -> WK.Group:
    (g,) = [g for g in w.products if g.reason == "gathered"]
    return g


def _boolean(fid_counts: Counter, by_id: dict) -> int:
    return sum(n * sum(compile_bits(boolean_version(by_id[f])).counts.values()) for f, n in fid_counts.items())


def test_a_scan_charges_its_carried_weights_from_their_second_iteration():
    p = carried_loop()
    w = WK.read(p, KB)
    prims = [f for f in flatten(p.root).prims if not is_input(f)]
    assert w.gates == _boolean(Counter(f.id for f in prims), {f.id: f for f in prims})
    (g,) = w.groups
    assert (g.weight, g.widths, g.out_width, g.k_dim, g.rates) == ((True, True), (32, 32), 32, K, (32.0, 32.0))
    assert (w.N, w.O) == ((T - 1) * K * N, (T - 1) * N * 32)
    assert w.left_out == {"theta": K * N, "gathered": 0, "self": 0, "activations": 0}
    assert len(g.edges) == 1


def test_a_value_inputs_row_by_a_carried_weight_is_priced_and_left_out():
    """x ⊙ g by the carried W, x an iteration's input: two of its rows may be equal, so from the second iteration the
    product is left out as gathered, priced as an activation by a weight."""
    p = loop()
    w = WK.read(p, KB)
    prims = [f for f in flatten(p.root).prims if not is_input(f)]
    assert w.gates == _boolean(Counter(f.id for f in prims), {f.id: f for f in prims})
    g = _gathered(w)
    assert (g.weight, g.widths, g.out_width, g.k_dim, g.rates) == ((False, True), (32, 32), 32, K, (32.0, 32.0))
    assert g.beta == (MC.beta(KB), MC.beta(KB)) and w.r_step is None and w.keep == 1.0
    assert w.groups == [] and w.left_out == {"theta": K * N, "gathered": (T - 1) * K * N, "self": 0, "activations": 0}


def test_the_batch_alone_is_free_like_theta():
    """x W^T, x the batch: x depends on no θ′, so the product is `θ`'s at every iteration."""
    p = _loop("Plain", (("w", W_T),), (("x", X_T),), Y_T,
              lambda B, c, x: ((_add(B, c[0], x),), B.call(LIN, x, c[0])))
    w = WK.read(p, KB)
    assert w.groups == [] and w.left_out["theta"] == T * K * N


def test_a_primitive_with_no_boolean_version_is_refused_by_name():
    tanh = _def("Tanh", ((("x", F32),), F32), lambda B, S, x: B.call(RP.F32TanhRn, x))
    with pytest.raises(NoBooleanVersion, match="F32TanhRn"):
        WK.read(Program(bind(tanh)), KB)


def _lookup_loop(carried: bool) -> Program:
    """Each iteration looks a row up by its token and multiplies it by the carried W, which it adds z to; the table is
    the registered one (shared), or carried and added z' to."""
    xs = (("tok", I32), ("z", X_T), ("zt", X_T))
    if carried:
        return _loop("LookupCarried", (("table", TABLE_T), ("w", W_T)), xs, Y_T,
                     lambda B, c, tok, z, zt: ((_add(B, c[0], zt), _add(B, c[1], z)),
                                               B.call(LIN, B.call(EMB, tok, c[0]), c[1])))
    return _loop("LookupShared", (("w", W_T),), xs, Y_T,
                 lambda B, c, tok, z, zt, table: ((_add(B, c[0], z),), B.call(LIN, B.call(EMB, tok, table), c[0])),
                 shared=(("table", TABLE_T),), registered=("table",))


def test_a_lookup_of_the_registered_table_is_thetas():
    w = WK.read(_lookup_loop(carried=False), KB)
    assert w.groups == [] and w.left_out == {"theta": T * K * N, "gathered": 0, "self": 0, "activations": 0}


def test_a_lookup_of_the_loops_own_table_gives_rows_that_are_never_charged():
    """A carried table is θ′ from the second iteration, but its rows by a token are a gather's, at rate 0."""
    w = WK.read(_lookup_loop(carried=True), KB)
    assert w.groups == [] and w.left_out == {"theta": K * N, "gathered": (T - 1) * K * N, "self": 0, "activations": 0}


SQUARE = _def("Square", ((("x", X_T),), F32), lambda B, S, x: B.call(DOT, x, x))
PAIRS = _def("Pairs", ((("x", X_T), ("y", X_T), ("z", X_T)), Array(3, F32)),
             lambda B, S, x, y, z: Coll(Array(3, F32), concat([B.call(DOT, a, b).refs
                                                               for a, b in ((x, y), (y, z), (x, z))])))


def test_a_program_with_no_scan_has_no_bound():
    for fn in (SQUARE, PAIRS):
        w = WK.read(Program(bind(fn)), KB)
        assert w.groups == [] and w.left_out["theta"] > 0 and WK.bound(w, C) is None


def test_one_array_times_itself_is_left_out():
    p = _loop("Self", (("u", X_T),), (("z", X_T),), F32,
              lambda B, c, z: ((_add(B, c[0], z),), B.call(DOT, c[0], c[0])))
    w = WK.read(p, KB)
    assert w.groups == [] and w.left_out == {"theta": K, "gathered": 0, "self": (T - 1) * K, "activations": 0}


def _pairs_loop() -> Program:
    def step(B, c, z):
        u, v, w = c
        ys = concat([B.call(DOT, a, b).refs for a, b in ((u, v), (v, w), (u, w))])
        return tuple(_add(B, a, z) for a in c), Coll(Array(3, F32), ys)

    return _loop("Pairs", (("u", X_T), ("v", X_T), ("w", X_T)), (("z", X_T),), Array(3, F32), step)


def test_the_arrays_multiplied_pairwise_set_F():
    w = WK.read(_pairs_loop(), KB)
    (g,) = w.groups
    assert g.weight == (True, True) and w.N == 3 * (T - 1) * K and WK.clique(g.edges) == 3
    b = WK.bound(w, C)
    assert b.omega == 3 and b.bound.F == MC.share(3) and b.bound.statement == MC.HE_SHARED


def test_two_activations_set_the_prefix_a_sequence_shares():
    """h (x ⊙ g) times x's own matrix (Z ⊙ G) is two activations, neither specific, so it is left out as gathered, as
    h W is; its dims allow a sequence of K, so a token-indexed product keeps 1 − 1/K of its MACs."""
    def step(B, c, x, z):
        g, G, W = c
        h = _mul(B, x, g)
        y = concat([B.call(LIN, h, _mul(B, z, G)).refs, B.call(LIN, h, W).refs])
        return (_add(B, g, x), _add(B, G, z), _add(B, W, z)), Coll(Array(2 * N, F32), y)

    w = WK.read(_loop("Acts", (("g", X_T), ("G", W_T), ("W", W_T)), (("x", X_T), ("z", W_T)), Array(2 * N, F32),
                      step), KB)
    assert w.groups == [] and {g.weight for g in w.products if g.reason == "gathered"} == {(False, True), (False, False)}
    assert w.left_out["gathered"] == 2 * (T - 1) * K * N and w.left_out["activations"] == 0
    assert w.keep == 1 - WK.SHARED_PREFIX / K


def _ternary_loop() -> Program:
    """A weight carried as a ternary code, two plane words and a scale a row, decoded in each iteration: value j of
    row r is the scale times +1 where bit j of the first plane is set, -1 where the second's is, 0 else; it multiplies
    x ⊙ g, g carried. The code is updated by the iteration's inputs, so it is θ′ from the second iteration."""
    planes_t, scale_t, wt, xt = Array(N, I32), Array(N, F32), Array(N, Array(KT, F32)), Array(KT, F32)
    bit, nk = bit_at(32), N * KT

    def rep(c: Coll, t, inner: int, outer_stride: int, inner_stride: int, n: int) -> Coll:
        r = c.refs
        return Coll(Array(n, t), Strided(r.space, r.idx, r.base, inner, outer_stride, inner_stride, n))

    def step(B, c, x, u, v, s, idx):
        p1, p2, sc, g = c
        ix = rep(idx, I32, KT, 0, 1, nk)
        b1 = B.batch(bit, rep(p1, I32, KT, 1, 0, nk), ix, axes=(0, 0))
        b2 = B.batch(bit, rep(p2, I32, KT, 1, 0, nk), ix, axes=(0, 0))
        one, minus, zero = (B.call(const(32, v_)) for v_ in (0x3F800000, 0xBF800000, 0))
        each = lambda c_: Coll(Array(nk, F32), concat([c_.refs] * nk))  # noqa: E731
        low = B.batch(P.SelectF32, b2, each(minus), each(zero), axes=(0, 0, 0))
        digit = B.batch(P.SelectF32, b1, each(one), low, axes=(0, 0, 0))
        w = B.batch(P.F32Mul, digit, rep(sc, F32, KT, 1, 0, nk), axes=(0, 0)).reshape(wt)
        new = (B.batch(P.I32Add, p1, u, axes=(0, 0)), B.batch(P.I32Add, p2, v, axes=(0, 0)),
               B.batch(P.F32Add, sc, s, axes=(0, 0)), _add(B, g, x))
        return new, B.call(LIN_T, _mul(B, x, g), w)

    return _loop("Ternary", (("p1", planes_t), ("p2", planes_t), ("scale", scale_t), ("g", xt)),
                 (("x", xt), ("u", planes_t), ("v", planes_t), ("s", scale_t)), Y_T, step,
                 shared=(("idx", Array(KT, I32)),), registered=("idx",))


def test_a_weight_decoded_from_a_ternary_code_is_priced_at_its_rate():
    """A decoded value reads a bit of each plane and its row's scale, a word each a value's worth: its rate is 1 bit and
    it holds b = 2 + 32/KT source bits, so its β is b·1/(b + 1) at its passed width b."""
    w = WK.read(_ternary_loop(), KB)
    g = _gathered(w)
    assert g.weight == (False, True) and g.widths == (32, 32) and w.left_out["gathered"] == (T - 1) * N * KT
    b = 2 + 32 / KT
    assert g.rates == (32.0, pytest.approx(1.0)) and g.bits == (32.0, pytest.approx(b))
    assert g.beta == (MC.beta(KB), pytest.approx(b / (b + 1)))


def test_a_one_hot_row_is_thetas():
    """A one-hot row is computed from the token and registered ids alone, so the product reading it is `θ`'s."""
    def step(B, c, tok, z, ids):
        hot = B.batch(P.SelectF32, B.batch(P.I32Eq, Coll(Array(VH, I32), concat([tok.refs] * VH)), ids, axes=(0, 0)),
                      *(Coll(Array(VH, F32), concat([B.call(const(32, v)).refs] * VH)) for v in (0x3F800000, 0)),
                      axes=(0, 0, 0))
        return (_add(B, c[0], z),), B.call(LIN_V, hot, c[0])

    p = _loop("OneHot", (("e", Array(N, Array(VH, F32))),), (("tok", I32), ("z", Array(VH, F32))), Y_T, step,
              shared=(("ids", Array(VH, I32)),), registered=("ids",))
    w = WK.read(p, KB)
    assert w.groups == [] and w.left_out["theta"] == T * N * VH


def test_a_rank_bottleneck_between_carried_weights_bounds_the_downstream_operand_at_beta_eff():
    """u = A·(x ⊙ g), y = B·u, z = C·y, with A R×D, B D×R and C N×D carried: u is paid (D ≥ K*), y is made whole at
    4R²·32·32/k a word, so its β in z is `beta_eff` at rank R."""
    lin = lambda k, n: bind(RP.LinearRef, K=k, N=n, DT="f32", BIAS=False)  # noqa: E731
    row = Array(D, F32)

    def step(B, c, x):
        g, A, Bm, Cm = c
        u = B.call(lin(D, R), _mul(B, x, g), A)
        y = B.call(lin(R, D), u, Bm)
        return tuple(_add(B, a, x) for a in c), B.call(lin(D, N), y, Cm)

    p = _loop("Adapter", (("g", row), ("A", Array(R, row)), ("B", Array(D, Array(R, F32))), ("C", Array(N, row))),
              (("x", row),), Y_T, step)
    w = WK.read(p, KB)
    assert w.groups == []
    by = {(g.k_dim, g.dims[3]): g for g in w.products if g.reason == "gathered"}
    assert set(by) == {(D, R), (R, D), (D, N)}
    assert by[(D, R)].beta == (MC.beta(KB), MC.beta(KB)) and by[(R, D)].rates[0] == 32.0
    downstream = by[(D, N)]
    assert downstream.rates[0] == pytest.approx(4 * R ** 2 * 32 * 32 / KB)
    assert downstream.beta[0] <= WK.beta_eff(32, R, KB) + 1e-9
    assert downstream.beta[0] == pytest.approx(WK.beta_eff(32, R, KB))


def test_the_bound_is_the_calculators_at_the_groups_beta():
    w = WK.read(carried_loop(), KB)
    best = WK.bound(w, C)
    bk = MC.beta(KB)
    whole = MC.bound(C, KB, N=w.N, O=w.O, size=w.gates, beta_w=bk, beta_x=bk, b=MC.BF16, F=MC.share(2))
    assert (best.bound.beta_w, best.bound.beta_x, best.bound.b) == (bk, bk, MC.BF16)
    assert best.bound.blocks == pytest.approx(whole.blocks) and best.N == w.N


def test_every_operand_is_at_most_bf16_and_beta_k():
    for k in (KB, 8e6):
        (g,) = WK.read(carried_loop(), k).groups
        assert g.beta == _gathered(WK.read(loop(), k)).beta == (min(MC.BF16, MC.beta(k)),) * 2
    assert WK.bound(WK.read(carried_loop(), WK.HEAD_K), C) is None
    assert WK.bound(WK.read(carried_loop(), 1e10), C) is None


#: a row's positions
PN = 2
ROWS_T = Array(PN, X_T)


def _token_loop(kind: str) -> Program:
    """Each iteration looks up a row of the carried table for each of its PN tokens, combines the rows position by
    position, and multiplies them by the carried W: ``kind`` "positional" multiplies by the input `pos`, "mask" adds
    a constant that differs by position, "flat" one that doesn't, and "mixed" adds to each row the other position's
    before multiplying by `pos`."""
    def step(B, c, tok, z, pos):
        table, w = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        if kind in ("mask", "flat"):
            words = [B.call(const(32, 0x3F800000 * (p + 1) if kind == "mask" else 0)) for p in range(PN)]
            h = _add(B, rows, Coll(ROWS_T, concat([concat([words[p].refs] * K) for p in range(PN)])))
        elif kind == "mixed":
            other = Coll(ROWS_T, concat([rows[(p + 1) % PN].refs for p in range(PN)]))
            h = _mul(B, _add(B, rows, other), pos)
        else:
            h = _mul(B, rows, pos)
        return (_add(B, table, z), _add(B, w, z)), B.batch(LIN, h, w, axes=(0, None))

    return _loop("Tokens" + kind, (("table", TABLE_T), ("w", W_T)), (("tok", Array(PN, I32)), ("z", X_T)),
                 Array(PN, Y_T), step, shared=(("pos", ROWS_T),), registered=("pos",))


def test_rows_stay_a_gathers_from_the_circuit_alone():
    """An input of unstated values, a constant added by position, or none: no softmax over a constant makes the rows
    specific, so each product reading them is left out."""
    for kind in ("positional", "mask", "flat", "mixed"):
        w = WK.read(_token_loop(kind), KB)
        assert w.groups == [] and w.left_out["gathered"] == (T - 1) * PN * K * N


def _rows_loop(form: str) -> Program:
    """Each iteration multiplies its PN rows of values, each times the carried g, by the carried W: as one batch node,
    a call a row, a scan inside the iteration over the rows, or two inputs of a row each."""
    def step(B, c, *xs):
        w, g = c
        if form == "inputs":
            ys = concat([B.call(LIN, _mul(B, r, g), w).refs for r in xs[:PN]])
        elif form == "batch":
            ys = B.batch(LIN, _mul(B, xs[0], g), w, axes=(0, None)).refs
        elif form == "calls":
            h = _mul(B, xs[0], g)
            ys = concat([B.call(LIN, h[p], w).refs for p in range(PN)])
        else:
            def row(B, S, acc, x, g_, w_):
                return tuple_of(acc, B.call(LIN, _mul(B, x, g_), w_))

            inner = _def("Row", ((("acc", Y_T), ("x", X_T), ("g", X_T), ("w", W_T)), Tuple(Y_T, Y_T)), row)
            init = Coll(Y_T, concat([B.call(const(32, 0)).refs] * N))
            ys = B.scan(bind(inner), init, xs=[xs[0]], shared=[g, w])[1].refs
        return (_add(B, w, xs[-1]), _add(B, g, xs[-1])), Coll(Array(PN, Y_T), ys)

    rows = tuple((f"r{p}", X_T) for p in range(PN)) if form == "inputs" else (("rows", ROWS_T),)
    return _loop("Rows" + form, (("w", W_T), ("g", X_T)), (*rows, ("z", X_T)), Array(PN, Y_T), step)


@pytest.mark.parametrize("form", ("batch", "calls", "scan", "inputs"))
def test_a_value_inputs_rows_are_left_out_however_the_program_writes_them(form):
    """Two rows of an input's values may be equal, whichever way the Program writes them, so each form leaves out the
    same MACs as gathered."""
    w = WK.read(_rows_loop(form), KB)
    assert w.groups == []
    assert w.left_out == {"theta": PN * K * N, "gathered": (T - 1) * PN * K * N, "self": 0, "activations": 0}


def _carried_rows_loop(carried: bool) -> Program:
    """Each iteration looks up PN rows of the carried table by its tokens and multiplies rows by the carried W: those it
    looked up, or (``carried``) those the iteration before looked up and returned in the carry."""
    def step(B, c, tok, z):
        table, w = c[:2]
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        y = B.batch(LIN, c[2] if carried else rows, w, axes=(0, None))
        return (_add(B, table, z), _add(B, w, z), *((rows,) if carried else ())), y

    carry = (("table", TABLE_T), ("w", W_T), *((("prev", ROWS_T),) if carried else ()))
    return _loop("CarriedRows" if carried else "Rows", carry, (("tok", Array(PN, I32)), ("z", X_T)), Array(PN, Y_T),
                 step)


def test_rows_carried_into_the_next_iteration_are_charged_as_weights():
    """The carried-words premise, held as a fact: the reading takes every carried word as a parameter's or the
    optimizer's, so a gather's rows carried into the next iteration are charged as a product of two weights, where the
    same product in the iteration that looked them up is left out."""
    w = WK.read(_carried_rows_loop(True), KB)
    (g,) = w.groups
    assert g.weight == (True, True) and w.N == (T - 1) * PN * K * N
    assert w.left_out == {"theta": PN * K * N, "gathered": 0, "self": 0, "activations": 0}
    same = WK.read(_carried_rows_loop(False), KB)
    assert same.groups == [] and same.left_out["gathered"] == (T - 1) * PN * K * N


def test_an_update_that_changes_nothing_still_makes_theta_prime():
    """The premise that θ′ is by data flow, held as a fact: W plus z times a stated 0 is W, but z reaches it, so W is
    θ′ from the second iteration and u W^T, u carried, is charged; with W carried unchanged it is `θ`'s."""
    def make(update: bool) -> Program:
        def step(B, c, z):
            w, u = c
            zero = _mul(B, Coll(W_T, concat([z.refs] * N)), B.call(const(32, 0)))
            return (_add(B, w, zero) if update else w, _add(B, u, z)), B.call(LIN, u, w)
        return _loop("Zero" if update else "Still", (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, step)

    w = WK.read(make(True), KB)
    (g,) = w.groups
    assert g.weight == (True, True) and w.N == (T - 1) * K * N and w.left_out["theta"] == K * N
    still = WK.read(make(False), KB)
    assert still.groups == [] and still.left_out["theta"] == T * K * N


def test_clique():
    tri = [frozenset(e) for e in ((1, 2), (2, 3), (1, 3), (3, 4))]
    assert WK.clique(tri) == 3
    assert WK.clique(tri + [frozenset(e) for e in ((1, 4), (2, 4))]) == 4
    assert WK.clique([frozenset((1, 2))]) == 2 and WK.clique([]) == 0
