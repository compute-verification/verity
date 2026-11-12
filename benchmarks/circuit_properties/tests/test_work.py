"""The work the min-cut bound reads off a Program (`circuit_properties.work`), on word circuits of the reference
registry's products whose work is known by hand: a training loop's scan, whose carried weights are charged from
their second iteration, whose batch is free, and whose value input's rows are left out however the Program writes
them; carried rows no one node shows distinct left out, one multiply by an input's words however it is written, a
stated 0's update `θ`'s, a weight rounded to bf16 charged, rows times an input however a product reads them, a
weight cancelled against itself, through a call too, or times its own rows in another order refused at a point, the
updates' premise held as facts; a lookup
whose table is the registered one or the loop's own; a Program with no scan; one array times itself; two
activations; the largest set of arrays a work multiplies pairwise; a weight decoded from a ternary code; a one-hot
row; a rank-4 bottleneck between two carried weights; each operand's β at min(16, β(k)); and |C| in Boolean gates."""

from __future__ import annotations

from collections import Counter

import pytest
from circuit_properties import points as PT
from circuit_properties import work as WK
from circuit_properties.ops import flatten

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.bits import compile as compile_bits
from verity.circuits.program import Program, is_input
from verity.circuits.refs import Coll, Strided, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.boolean import NoBooleanVersion, boolean_version
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import F32Neg
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


def _loop(name: str, carry: tuple, xs: tuple, y_t, step, shared: tuple = (), registered: tuple = (),
          n: int = T) -> Program:
    """A scan of ``n`` iterations of ``step(B, carry, *xs, *shared) -> (new carry fields, y)``, each of ``carry``,
    ``xs`` and ``shared`` (name, type): the carry `θ`, ``xs`` an iteration's inputs, ``shared`` the loop's."""
    carry_t = Tuple(*(t for _, t in carry))

    def body(B, S, c, *rest):
        new, y = step(B, [c[i] for i in range(len(carry))], *rest)
        return tuple_of(tuple_of(*new), y)

    st = _def(name + "Step", ((("carry", carry_t), *xs, *shared), Tuple(carry_t, y_t)), body)
    params = (*carry, *((name_, Array(n, t)) for name_, t in xs), *shared)

    def run(B, S, *a):
        nc, nx = len(carry), len(xs)
        return B.scan(bind(st), tuple_of(*a[:nc]), xs=list(a[nc:nc + nx]), shared=list(a[nc + nx:]))

    fn = _def(name, (params, Tuple(carry_t, Array(n, y_t))), run)
    return Program(bind(fn), registered=registered, theta=tuple(n for n, _ in carry))


def loop() -> Program:
    """y = (x ⊙ g) W^T, and W and g plus x: the carry is θ′ from the second iteration, so is x ⊙ g, and x is free."""
    return _loop("Train", (("w", W_T), ("g", X_T)), (("x", X_T),), Y_T,
                 lambda B, c, x: ((_add(B, c[0], x), _add(B, c[1], x)), B.call(LIN, _mul(B, x, c[1]), c[0])))


def carried_loop() -> Program:
    """y = (u ⊙ g) W^T, u, g and W carried and each added the iteration's z: no operand reads an input, and u ⊙ g is a
    weight times a weight, whose rows are distinct, so the product of two weights is charged from the second
    iteration."""
    return _loop("Carried", (("w", W_T), ("g", X_T), ("u", X_T)), (("z", X_T),), Y_T,
                 lambda B, c, z: (tuple(_add(B, a, z) for a in c), B.call(LIN, _mul(B, c[2], c[1]), c[0])))


def _gathered(w: WK.Work) -> WK.Group:
    (g,) = [g for g in w.products if g.reason == "gathered"]
    return g


def _boolean(fid_counts: Counter, by_id: dict) -> int:
    return sum(n * sum(compile_bits(boolean_version(by_id[f])).counts.values()) for f, n in fid_counts.items())


def test_a_scan_charges_its_carried_weights_from_their_second_iteration():
    """`carried_loop`'s product is charged: W is a weight, and so is u ⊙ g, a multiply of two arrays whose rows are
    distinct (`distinct_status`)."""
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


def test_rows_carried_into_the_next_iteration_are_left_out_as_in_the_iteration_that_read_them():
    """A gather's rows carried into the next iteration may still repeat (`Status.distinct`), so the product reading them
    there is left out as gathered, as the same product in the iteration that looked them up is."""
    w = WK.read(_carried_rows_loop(True), KB)
    assert w.groups == [] and {g.weight for g in w.products if g.reason == "gathered"} == {(True, True)}
    assert w.left_out == {"theta": PN * K * N, "gathered": (T - 1) * PN * K * N, "self": 0, "activations": 0}
    same = WK.read(_carried_rows_loop(False), KB)
    assert same.groups == [] and same.left_out["gathered"] == (T - 1) * PN * K * N


FLAT = Array(PN * K, F32)
ADD_ROW = bind(RP.AddRef, N=K, DT="f32")


def _carried_with(kind: str) -> Program:
    """Each iteration multiplies by the carried W the rows it carried from the iteration before, computed from the PN
    rows it looked up by its tokens and the carried b (``kind``): b added word by word in calls, as a batch of a row
    composite sharing b, or a batch a row; b's row repeated as a view, plus a stated 0; b plus 1, b plus 2, each its own
    node, as one argument; the shared input s added; or b's row repeated as a view alone. Every one gives two equal rows
    where the looked-up rows or s's are, or always."""
    def step(B, c, tok, z, *shared):
        table, w, prev, b = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        if kind == "calls":
            new = Coll(ROWS_T, concat([B.call(P.F32Add, rows[p][j], b[j]).refs for p in range(PN) for j in range(K)]))
        elif kind == "composite":
            new = B.batch(ADD_ROW, rows, b, axes=(0, None))
        elif kind == "batches":
            new = Coll(ROWS_T, concat([_add(B, rows[p], b).refs for p in range(PN)]))
        elif kind == "replicated":
            fresh = _add(B, Coll(FLAT, concat([b.refs] * PN)), B.call(const(32, 0)))
            new = B.batch(P.F32Add, Coll(FLAT, rows.refs), fresh, axes=(0, 0)).reshape(ROWS_T)
        elif kind == "assembled":
            each = Coll(ROWS_T, concat([_add(B, b, B.call(const(32, 0x3F800000 * (p + 1)))).refs for p in range(PN)]))
            new = B.batch(P.F32Add, Coll(FLAT, rows.refs), Coll(FLAT, each.refs), axes=(0, 0)).reshape(ROWS_T)
        elif kind == "shared":
            new = B.batch(P.F32Add, Coll(FLAT, rows.refs), Coll(FLAT, shared[0].refs), axes=(0, 0)).reshape(ROWS_T)
        else:
            new = Coll(ROWS_T, concat([b.refs] * PN))
        return (_add(B, table, z), _add(B, w, z), new, _add(B, b, z)), B.batch(LIN, prev, w, axes=(0, None))

    return _loop("With" + kind, (("table", TABLE_T), ("w", W_T), ("prev", ROWS_T), ("b", X_T)),
                 (("tok", Array(PN, I32)), ("z", X_T)), Array(PN, Y_T), step,
                 shared=(("s", ROWS_T),) if kind == "shared" else ())


@pytest.mark.parametrize("kind", ("calls", "composite", "batches", "replicated", "assembled", "shared", "view"))
def test_carried_rows_no_one_node_shows_distinct_are_left_out(kind):
    """Rows are distinct only where they are `θ`'s or one node computes them one to one from distinct rows read one to
    one, so each of these is left out as gathered once θ′ reaches it: from the second iteration, and the view of b,
    which holds the iteration before's b, from the third."""
    w = WK.read(_carried_with(kind), KB)
    late = 2 if kind == "view" else 1
    assert w.groups == [] and w.left_out == {"theta": late * PN * K * N, "gathered": (T - late) * PN * K * N,
                                             "self": 0, "activations": 0}


def test_a_weights_row_read_twice_as_a_products_rows_is_left_out():
    """b's row read by both members of a product, b carried: its two rows are one row, so the product is left out."""
    p = _loop("Twice", (("w", W_T), ("b", X_T)), (("z", X_T),), Array(PN, Y_T),
              lambda B, c, z: ((_add(B, c[0], z), _add(B, c[1], z)),
                               B.batch(LIN, Coll(ROWS_T, concat([c[1].refs] * PN)), c[0], axes=(0, None))))
    w = WK.read(p, KB)
    assert w.groups == [] and w.left_out["gathered"] == (T - 1) * PN * K * N


def test_a_carry_from_a_stated_zero_plus_rows_is_left_out():
    """A carry that starts at a stated 0 and adds the looked-up rows holds them exactly after one iteration: its
    initial value's rows aren't distinct, so neither are the carried ones."""
    carry_t = Tuple(TABLE_T, W_T, ROWS_T)

    def body(B, S, c, tok, z):
        table, w, prev = c[0], c[1], c[2]
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        new = B.batch(P.F32Add, Coll(FLAT, prev.refs), Coll(FLAT, rows.refs), axes=(0, 0)).reshape(ROWS_T)
        return tuple_of(tuple_of(_add(B, table, z), _add(B, w, z), new), B.batch(LIN, prev, w, axes=(0, None)))

    st = _def("ZeroInitStep", ((("carry", carry_t), ("tok", Array(PN, I32)), ("z", X_T)),
                           Tuple(carry_t, Array(PN, Y_T))), body)

    def run(B, S, table, w, tok, z):
        zeros = Coll(ROWS_T, concat([B.call(const(32, 0)).refs] * (PN * K)))
        return B.scan(bind(st), tuple_of(table, w, zeros), xs=[tok, z], shared=[])

    fn = _def("ZeroInit", ((("table", TABLE_T), ("w", W_T), ("tok", Array(T, Array(PN, I32))), ("z", Array(T, X_T))),
                       Tuple(carry_t, Array(T, Array(PN, Y_T)))), run)
    w = WK.read(Program(bind(fn), theta=("table", "w")), KB)
    assert w.groups == [] and w.left_out["gathered"] == (T - 1) * PN * K * N


def test_a_register_of_one_word_inputs_is_left_out():
    """The carry holds the last PN·K one-word inputs as PN rows: an input's words aren't distinct, so neither are its
    rows, and once the register holds inputs alone, PN·K iterations in, its products are left out as gathered."""
    def step(B, c, s, z):
        w, prev = c
        words = [prev[p][j].refs for p in range(PN) for j in range(K)]
        return (_add(B, w, z), Coll(ROWS_T, concat([*words[1:], s.refs]))), B.batch(LIN, prev, w, axes=(0, None))

    full, n = PN * K, PN * K + 6
    w = WK.read(_loop("Register", (("w", W_T), ("prev", ROWS_T)), (("s", F32), ("z", X_T)), Array(PN, Y_T), step, n=n),
                KB)
    assert w.groups == [] and w.left_out == {"theta": full * PN * K * N, "gathered": (n - full) * PN * K * N,
                                             "self": 0, "activations": 0}


FLAT_W = Array(N * K, F32)
MUL_WORD = bind(_def("MulWord", ((("a", F32), ("b", F32)), F32), lambda B, S, a, b: B.call(P.F32Mul, a, b)))
SCALE_ROW = bind(_def("ScaleRow", ((("row", X_T), ("s", F32)), X_T),
                      lambda B, S, row, s: B.batch(P.F32Mul, row, s, axes=(0, None))))


def _scaled(kind: str) -> Program:
    """u W^T, u carried, and W multiplied by: the iteration's one-word input s, or s times s (``one-word``,
    ``computed-word``); its input z a word of
    W, as one batch, a call a word, or a batch of a composite of one call; each row by its own word of zn, a batch of
    a composite multiplying its row by one word (``composite-row``); or a stated 2, v, registered (``thetas``), or
    v plus -1 times v, v registered or V carried (``thetas-cancel``, ``carried-cancel``), each plus z."""
    def step(B, c, s, z, zn, *v):
        w, u = c[0], c[1]
        if kind == "one-word":
            new = _mul(B, w, s)
        elif kind == "computed-word":
            new = _mul(B, w, B.call(P.F32Mul, s, s))
        elif kind == "batch":
            new = _mul(B, w, z)
        elif kind == "calls":
            new = Coll(W_T, concat([B.call(P.F32Mul, w[n][k], z[k]).refs for n in range(N) for k in range(K)]))
        elif kind == "composite-word":
            new = B.batch(MUL_WORD, Coll(FLAT_W, w.refs), Coll(FLAT_W, concat([z.refs] * N)), axes=(0, 0)).reshape(W_T)
        elif kind == "composite-row":
            new = B.batch(SCALE_ROW, w, zn, axes=(0, 0))
        elif kind == "stated":
            new = _add(B, _mul(B, w, B.call(const(32, 0x40000000))), z)
        elif kind == "thetas":
            new = _add(B, _mul(B, w, v[0]), z)
        else:
            m = v[0] if kind == "thetas-cancel" else c[2]
            new = _add(B, _mul(B, w, _add(B, m, _mul(B, m, B.call(const(32, 0xBF800000))))), z)
        rest = (_add(B, c[2], z),) if kind == "carried-cancel" else ()
        return (new, _add(B, u, z), *rest), B.call(LIN, u, w)

    v = (("v", X_T),) if kind.startswith("thetas") else ()
    carry = (("w", W_T), ("u", X_T), *((("V", W_T),) if kind == "carried-cancel" else ()))
    return _loop("Scaled" + kind.title().replace("-", ""), carry, (("s", F32), ("z", X_T), ("zn", Array(N, F32))),
                 Y_T, step, shared=v, registered=("v",) if v else ())


@pytest.mark.parametrize("form", ("batch", "calls", "composite-word", "composite-row"))
def test_a_weight_times_an_input_a_word_is_left_out_however_the_program_writes_the_multiply(form):
    """W times an input's words, so that an input 0 makes W's rows one row, gives one answer however the Program
    writes the multiply: a batch, a call a word, a batch of a composite of one call, or a batch of a composite
    multiplying its row by one word, each of W's rows by its own. A word is one every member reads only where every
    enclosing batch's members read it too (`Status.fixed`), so each is left out as gathered once W is θ′."""
    w = WK.read(_scaled(form), KB)
    assert w.groups == [] and w.left_out == {"theta": K * N, "gathered": (T - 1) * K * N, "self": 0, "activations": 0}


@pytest.mark.parametrize("by", ("stated", "thetas"))
def test_a_weight_times_a_stated_word_or_thetas_own_is_charged(by):
    """W times a stated nonzero word, or times v that no input reaches (`θ`'s own, nonzero for generic weights), plus
    z, keeps W's rows distinct: u W^T is charged from the second iteration."""
    (g,) = WK.read(_scaled(by), KB).groups
    assert g.weight == (True, True) and g.macs == (T - 1) * K * N


def _rows_plus(kind: str) -> Program:
    """prev′ = M ⊙ m + the PN rows looked up by the iteration's tokens, M carried, and prev W^T: m the one-word input s
    (``one-word``), z ⊙ g with z the input and g carried (``input-times-weight``), or g alone (``weight``)."""
    def step(B, c, tok, z, s):
        table, w, prev, m, g = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        mult = s if kind == "one-word" else _mul(B, z, g) if kind == "input-times-weight" else g
        new = _add(B, _mul(B, m, mult), rows)
        return (_add(B, table, z), _add(B, w, z), new, _add(B, m, z), _add(B, g, z)), \
            B.batch(LIN, prev, w, axes=(0, None))

    carry = (("table", TABLE_T), ("w", W_T), ("prev", ROWS_T), ("m", ROWS_T), ("g", X_T))
    return _loop("RowsPlus" + kind.title().replace("-", ""), carry,
                 (("tok", Array(PN, I32)), ("z", X_T), ("s", F32)), Array(PN, Y_T), step)


@pytest.mark.parametrize("kind", ("input-times-weight", "weight"))
def test_rows_plus_a_weight_times_a_multiplier_an_input_reaches_are_left_out(kind):
    """M ⊙ m + rows is the looked-up rows exactly where m is 0: z = 0 makes z ⊙ g so, and g + z with z = -g makes g
    so from the second iteration. Where an input reaches m it keeps no rows of M distinct, so prev W^T is left out as
    gathered from the second iteration, and with m = g, which is `θ`'s in the first, from the third: g's one row
    repeated over M's rows reads each word twice, so it is no weight's rows."""
    w = WK.read(_rows_plus(kind), KB)
    late = 2 if kind == "weight" else 1
    assert w.N == (late - 1) * PN * K * N
    assert w.left_out == {"theta": PN * K * N, "gathered": (T - late) * PN * K * N, "self": 0, "activations": 0}


ROW_LIN = bind(_def("RowLin", ((("row", X_T), ("w", W_T)), Y_T), lambda B, S, row, w: B.call(LIN, row, w)))


def _rows_times(form: str, mult: str) -> Program:
    """prev′ = M ⊙ zr, M carried and zr an input's rows, or M times the one-word input s (``one-word``), and prev W^T
    as one batch, a call a row, or a batch of a composite of one call."""
    def step(B, c, zr, z, s):
        w, prev, m = c
        new = _mul(B, m, s if mult == "one-word" else zr)
        if form == "batch":
            y = B.batch(LIN, prev, w, axes=(0, None))
        elif form == "calls":
            y = Coll(Array(PN, Y_T), concat([B.call(LIN, prev[p], w).refs for p in range(PN)]))
        else:
            y = B.batch(ROW_LIN, prev, w, axes=(0, None))
        return (_add(B, w, z), new, _add(B, m, zr)), y

    return _loop("RowsTimes" + form.title() + mult.title().replace("-", ""),
                 (("w", W_T), ("prev", ROWS_T), ("m", ROWS_T)), (("zr", ROWS_T), ("z", X_T), ("s", F32)),
                 Array(PN, Y_T), step)


@pytest.mark.parametrize("form", ("batch", "calls", "composite"))
def test_rows_times_an_input_are_left_out_however_the_product_reads_them(form):
    """M ⊙ zr, which an input 0 makes one row repeated, gives one answer however the product reads prev: one batch, a
    call a row, or a batch of a composite of one call. A row alone at its call or member is still no weight's, so each
    is left out as gathered from the second iteration; times the one-word s (the premise) each is charged."""
    w = WK.read(_rows_times(form, "input"), KB)
    assert w.groups == [] and w.left_out == {"theta": PN * K * N, "gathered": (T - 1) * PN * K * N, "self": 0,
                                             "activations": 0}
    assert WK.read(_rows_times(form, "one-word"), KB).N == (T - 1) * PN * K * N


def _self_cancel(cancel: bool) -> Program:
    """prev′ = the PN rows looked up by the iteration's tokens, plus (``cancel``) M plus -1 times M, M carried; and
    prev W^T."""
    def step(B, c, tok, z, zr):
        table, w, prev, m = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        new = _add(B, _add(B, m, _mul(B, m, B.call(const(32, 0xBF800000)))), rows) if cancel else rows
        return (_add(B, table, z), _add(B, w, z), new, _add(B, m, zr)), B.batch(LIN, prev, w, axes=(0, None))

    return _loop("SelfCancel" if cancel else "RowsAlone", (("table", TABLE_T), ("w", W_T), ("prev", ROWS_T),
                                                            ("m", ROWS_T)),
                 (("tok", Array(PN, I32)), ("z", X_T), ("zr", ROWS_T)), Array(PN, Y_T), step)


def test_a_weight_cancelled_against_itself_is_refused():
    """M plus -1 times M is 0, which the reading's rules take as a weight with distinct rows, so the looked-up rows
    plus it, carried, are the looked-up rows, which repeat with the tokens: at the point, whose tokens are one token,
    its two rows are equal, and the Program is refused (`points.check`), naming the product and the rows. Where the
    Program carries the rows alone they are left out as gathered. W times v plus -1 times v, v registered or V
    carried, makes every row of W the update's z, and is refused too."""
    alone = WK.read(_self_cancel(False), KB)
    assert alone.groups == [] and alone.left_out["gathered"] == (T - 1) * PN * K * N
    with pytest.raises(WK.RepeatedRows) as e:
        WK.read(_self_cancel(True), KB)
    assert (e.value.definition, e.value.operand, e.value.rows) == ("SelfCancelStep_v1", "x", (0, 1))
    for kind in ("thetas-cancel", "carried-cancel"):
        with pytest.raises(WK.RepeatedRows) as e:
            WK.read(_scaled(kind), KB)
        assert e.value.operand == "w"


def _swap(order: str, carried: bool) -> Program:
    """prev′ = the PN rows looked up by the iteration's tokens, plus M times M's rows in ``order`` ("swap": the other
    order; "same": its own), M carried and updated by an input (``carried``) or unchanged; and prev W^T."""
    def step(B, c, tok, z, zr):
        table, w, prev, m = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        other = Coll(ROWS_T, concat([m[(p + 1) % PN].refs for p in range(PN)])) if order == "swap" else m
        return (_add(B, table, z), _add(B, w, z), _add(B, rows, _mul(B, m, other)), _add(B, m, zr) if carried else m), \
            B.batch(LIN, prev, w, axes=(0, None))

    return _loop("Swap" + order.title() + ("Carried" if carried else "Own"),
                 (("table", TABLE_T), ("w", W_T), ("prev", ROWS_T), ("m", ROWS_T)),
                 (("tok", Array(PN, I32)), ("z", X_T), ("zr", ROWS_T)), Array(PN, Y_T), step)


@pytest.mark.parametrize("carried", [True, False], ids=["carried", "own"])
def test_a_weight_times_its_own_rows_in_another_order_is_refused(carried):
    """With PN = 2, M times swap(M) has two rows equal for every M (M0 M1 = M1 M0), so the looked-up rows plus it
    repeat with the tokens, which the reading's rules take as two weights from the second iteration on, M carried or
    `θ`'s own: at the point its two rows are equal, and the Program is refused. M times M in its own order has
    distinct rows, and is charged as is right."""
    assert PN == 2
    with pytest.raises(WK.RepeatedRows) as e:
        WK.read(_swap("swap", carried), KB)
    assert (e.value.operand, e.value.rows) == ("x", (0, 1))
    w = WK.read(_swap("same", carried), KB)
    assert w.N == (T - 1) * PN * K * N and {g.weight for g in w.groups} == {(True, True)}


NEG_ROWS = bind(_def("NegRows", ((("rows", ROWS_T),), ROWS_T),
                     lambda B, S, rows: B.batch(F32Neg, Coll(FLAT, rows.refs), axes=(0,)).reshape(ROWS_T)))


def _cross_call() -> Program:
    """prev′ = the PN rows looked up by the iteration's tokens, plus M plus `NEG_ROWS` of M, a composite negating M's
    rows, M carried; and prev W^T."""
    def step(B, c, tok, z, zr):
        table, w, prev, m = c
        rows = Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)]))
        new = _add(B, rows, _add(B, m, B.call(NEG_ROWS, m)))
        return (_add(B, table, z), _add(B, w, z), new, _add(B, m, zr)), B.batch(LIN, prev, w, axes=(0, None))

    return _loop("CrossCall", (("table", TABLE_T), ("w", W_T), ("prev", ROWS_T), ("m", ROWS_T)),
                 (("tok", Array(PN, I32)), ("z", X_T), ("zr", ROWS_T)), Array(PN, Y_T), step)


def test_a_weight_cancelled_through_a_call_is_refused(monkeypatch):
    """M plus a call's negation of M is 0, an identity seen only with the callee's output at the caller's words: the
    reading's rules alone charge the looked-up rows plus it as two weights, and the point, which evaluates the call
    at M's words, finds its two rows equal and refuses the Program, naming the product, the rows and the fix."""
    with pytest.raises(WK.RepeatedRows) as e:
        WK.read(_cross_call(), KB)
    assert (e.value.definition, e.value.operand, e.value.rows) == ("CrossCallStep_v1", "x", (0, 1))
    assert str(e.value).endswith(
        "operand 'x', whose rows the reading shows distinct, holds rows 0 and 1 equal at the point; fix: compute it "
        "from a weight one to one (a shift, a rounding, a multiply by a word no input reaches) or let it read the "
        "inputs, which leaves its product out as gathered")
    monkeypatch.setattr(PT, "check", lambda *a: None)
    w = WK.read(_cross_call(), KB)
    assert w.N == (T - 1) * PN * K * N and {g.weight for g in w.groups} == {(True, True)}


def test_a_weight_times_a_difference_of_inputs_is_refused():
    """W times z0 minus z1, two words of an input, is one word every member reads, which the reading takes as a
    nonzero multiplier (`Status.fixed`); it is 0 wherever the input repeats, as the point's does, so every row of W
    is the update's z there, and the Program is refused."""
    def step(B, c, z):
        w, u = c
        d = B.call(RP.F32Sub, z[0], z[1])
        return (_add(B, _mul(B, w, d), z), _add(B, u, z)), B.call(LIN, u, w)

    with pytest.raises(WK.RepeatedRows) as e:
        WK.read(_loop("Difference", (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, step), KB)
    assert e.value.operand == "w"


def test_what_the_point_never_meets_stays_a_premise():
    """Rows equal only at values the point never takes are charged, held as facts beside those of
    `test_what_the_carried_words_premise_leaves_held_as_facts` (a one-word multiplier that is 0) and
    `test_an_update_by_a_stated_zero_leaves_a_weight_thetas` (W plus z plus -1 times z): W plus an input's rows Z,
    whose rows are equal where Z is fitted to W (Z0 - Z1 = W1 - W0), and M times W0 minus a stated c, a circuit that
    tests its own weights, whose rows are all 0 where W0 is c."""
    def fitted(B, c, zr, z):
        w, u = c
        return (_add(B, w, zr), _add(B, u, z)), B.call(LIN, u, w)

    w = WK.read(_loop("Fitted", (("w", W_T), ("u", X_T)), (("zr", W_T), ("z", X_T)), Y_T, fitted), KB)
    assert w.N == (T - 1) * K * N and {g.weight for g in w.groups} == {(True, True)}

    def tests(B, c, z, v):
        w, u = c
        d = B.call(RP.F32Sub, v[0], B.call(const(32, 0x3F800000)))
        return (_add(B, _mul(B, w, d), z), _add(B, u, z)), B.call(LIN, u, w)

    w = WK.read(_loop("TestsItself", (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, tests, shared=(("v", X_T),),
                      registered=("v",)), KB)
    assert w.N == (T - 1) * K * N and {g.weight for g in w.groups} == {(True, True)}


#: a bf16 weight of one-word rows, more rows than the point draws bf16 words, so that its rows repeat at one point
NB = 2048
BF16 = P.BF16
LIN_NB = bind(RP.LinearRef, K=1, N=NB, DT="f32", BIAS=False)


def _widen(B, a: Coll) -> Coll:
    return B.batch(P.Bf16ToF32, Coll(Array(a.type.leaves, BF16), a.refs), axes=(0,)).reshape(Array(NB, Array(1, F32)))


def test_a_narrow_row_is_compared_at_enough_points_to_hold_64_bits():
    """A bf16 weight's one-word rows hold `points.held(16)` = 10 drawn bits each, so they are compared at
    ceil(64 / 10) = 7 points: at one, its NB rows, more than the 1024 bf16 words the point draws, would repeat
    whatever the weight. Rounded after each update and widened before its product, the weight is charged and not
    refused."""
    assert (PT.points(1, PT.held(16)), PT.points(K, PT.held(32)), PT.points(1, 4)) == (7, 1, 16)

    def step(B, c, z):
        w, g = c
        wide = Coll(Array(NB, F32), _widen(B, w).refs)
        new = B.batch(P.F32ToBf16Rn, _add(B, wide, z), axes=(0,)).reshape(w.type)
        return (new, _add(B, g, z)), B.call(LIN_NB, g, _widen(B, w))

    p = _loop("Narrow", (("w", Array(NB, Array(1, BF16))), ("g", Array(1, F32))), (("z", Array(1, F32)),),
              Array(NB, F32), step)
    w, R, root, reasons = WK._read(p, KB, WK.PRODUCTS, WK.LOOKUPS)
    assert w.N == (T - 1) * NB and PT.check(p, R, root, reasons).points == 7


def test_a_row_with_a_nan_equals_no_row():
    """sqrt(-(W ⊙ W)) has distinct rows by the reading's rules and a NaN in every word at the point: a NaN row equals
    none, so its product is charged and the Program not refused."""
    def step(B, c, z):
        w, u = c
        neg = B.batch(F32Neg, Coll(FLAT_W, _mul(B, w, w).refs), axes=(0,))
        root = B.batch(RP.F32Sqrt, neg, axes=(0,)).reshape(W_T)
        return (_add(B, w, z), _add(B, u, z)), B.call(LIN, u, root)

    w = WK.read(_loop("NanRows", (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, step), KB)
    assert w.N == (T - 1) * K * N


def test_a_claim_past_a_scan_the_point_does_not_run_whole_is_refused():
    """An inner scan whose contexts repeat before its last iteration leaves its outputs unevaluated at the point, so a
    charged weight carried out of it is refused, with the scan named and the fix."""
    one = const(32, 0x3F800000)
    inner = _def("InnerStep", ((("w", W_T),), Tuple(W_T, F32)),
                 lambda B, S, w: tuple_of(_add(B, w, B.call(one)), B.call(const(32, 0))))

    def step(B, c, z):
        w, u = c
        out = B.scan(bind(inner), w, n=4)
        return (_add(B, Coll(W_T, out.refs.slice(0, N * K)), z), _add(B, u, z)), B.call(LIN, u, w)

    with pytest.raises(WK.RepeatedRows) as e:
        WK.read(_loop("Outer", (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, step), KB)
    assert (e.value.scan.fn, e.value.scan.ran, e.value.scan.n, e.value.operand) == ("OuterStep_v1", 1, 4, "w")
    assert str(e.value).endswith(
        f"operand 'w', whose rows the reading shows distinct, reads the outputs of scan node {e.value.scan.node} of "
        "OuterStep_v1, which the point runs 1 of its 4 iterations; fix: carry the weight through the outer scan "
        "rather than out of a scan whose contexts repeat")


def test_the_point_is_drawn_once_a_program_and_read_at_every_k():
    """Two walks of one Program draw one point (SHA-512 of its digest, nothing else), and its reads at two k make
    one pass."""
    p = carried_loop()
    w, R, root, reasons = WK._read(p, KB, WK.PRODUCTS, WK.LOOKUPS)
    from verity.circuits.codec import program_digest
    cl, d = PT.claims(R, root, reasons), program_digest(p)
    a, b = (PT.Walk(p, R, root, cl, d).inputs() for _ in range(2))
    assert a.keys() == b.keys() and all((a[k] == b[k]).all() for k in a)
    runs = []
    real = PT.Walk.run

    def counted(self):
        runs.append(1)
        return real(self)

    PT._MEMO.clear()
    PT.Walk.run = counted
    try:
        WK.read(p, KB), WK.read(p, 4 * KB)
    finally:
        PT.Walk.run = real
    assert len(runs) == 1


def test_the_product_kernel_is_the_products_body():
    """`points.dot`, by which the point computes a `DotRef` and an f32 `LinearRef` without a bias, gives the words
    their bodies give, on the point's draws and on uniform words with edge words mixed in."""
    import numpy as np
    from verity.circuits.evaluation.batch import evaluate_batch, sample_uniform

    rng = np.random.default_rng(0)
    for f, rows in ((DOT, 1), (LIN, N)):
        params = [t for _n, t in f.signature()[0]]
        for words in ([np.asarray(a, dtype=np.uint64) for a in sample_uniform(rng, 256, params)],
                      [PT.draw(rng, 32, (256, t.leaves)) for t in params]):
            x, w = (a.reshape(256, -1) for a in words)
            ref = np.asarray(evaluate_batch(f, x, w, kernel="reference"), dtype=np.uint64).reshape(256, rows)
            assert (PT.dot(x[:, None, :], w.reshape(256, rows, K)) == ref).all()
            assert PT._rows_of(f) == rows


def test_each_twin_is_the_reference():
    """Each numpy twin (`points.TWINS`) gives the reference's words, on uniform words with the edge words
    `sample_uniform` mixes in and on the point's own draws: bit-exact for every word but the exponential's, which is
    held on these words only."""
    import numpy as np
    from verity.circuits.defs import REGISTRY
    from verity.circuits.evaluation.batch import evaluate_batch, sample_uniform

    rng = np.random.default_rng(0)
    for fid, twin in PT.TWINS.items():
        f = REGISTRY[fid]
        params = [t for _n, t in f.signature()[0]]
        n = 512 if fid == "F32ExpRn_v1" else 4096
        uni = [np.asarray(a, dtype=np.uint64) for a in sample_uniform(rng, n, params)]
        near = [PT.draw(rng, max(t.width_set()), (n,)) for t in params]
        for words in (uni, near):
            ref = np.asarray(evaluate_batch(f, *words, kernel="reference"), dtype=np.uint64).reshape(-1)
            got = np.asarray(twin(*words), dtype=np.uint64).reshape(-1)
            assert (got == ref).all(), (fid, [hex(int(w[i])) for w in words for i in np.flatnonzero(got != ref)[:1]])


def _cast(B, a: Coll) -> Coll:
    flat = Array(a.type.leaves, F32)
    return B.batch(P.Bf16ToF32, B.batch(P.F32ToBf16Rn, Coll(flat, a.refs), axes=(0,)), axes=(0,)).reshape(a.type)


def test_a_weight_rounded_to_bf16_before_its_product_is_charged():
    """The generic-weights premise at a product's width: `θ`'s rows differ at every width a rounding (`ROUNDINGS`)
    reads them, so (u ⊙ g) W^T with W, or W and u ⊙ g, rounded to bf16 and widened back before it is charged as two
    weights from the second iteration, as `carried_loop`'s is."""
    def cast_loop(both: bool) -> Program:
        def step(B, c, z):
            x = _mul(B, c[2], c[1])
            return tuple(_add(B, a, z) for a in c), B.call(LIN, _cast(B, x) if both else x, _cast(B, c[0]))
        return _loop("CastBoth" if both else "CastWeight", (("w", W_T), ("g", X_T), ("u", X_T)), (("z", X_T),),
                     Y_T, step)

    for both in (False, True):
        (g,) = WK.read(cast_loop(both), KB).groups
        assert g.weight == (True, True) and g.macs == (T - 1) * K * N


def test_what_the_carried_words_premise_leaves_held_as_facts():
    """Two facts of what stays a premise. A weight times one unstated word that every member of every batch reads,
    an input's or computed from inputs (a learning rate, a decay), is taken as distinct, as if that word weren't 0:
    u W^T and the rows M ⊙ s plus the looked-up ones times W are charged; and b added to each of two looked-up rows, carried in two arrays, gives two arrays each of distinct rows, whose two products by W
    are charged though they read one pair where the two rows are equal."""
    for kind in ("one-word", "computed-word"):
        (g,) = WK.read(_scaled(kind), KB).groups
        assert g.weight == (True, True) and g.macs == (T - 1) * K * N
    assert WK.read(_rows_plus("one-word"), KB).N == (T - 1) * PN * K * N

    def step(B, c, tok, z):
        table, w, b, p0, p1 = c
        rows = [B.call(EMB, tok[p], table) for p in range(PN)]
        ys = concat([B.call(LIN, p0, w).refs, B.call(LIN, p1, w).refs])
        return (_add(B, table, z), _add(B, w, z), _add(B, b, z), _add(B, rows[0], b), _add(B, rows[1], b)), \
            Coll(Array(PN, Y_T), ys)

    w = WK.read(_loop("TwoArrays", (("table", TABLE_T), ("w", W_T), ("b", X_T), ("p0", X_T), ("p1", X_T)),
                      (("tok", Array(PN, I32)), ("z", X_T)), Array(PN, Y_T), step), KB)
    assert w.N == (T - 1) * PN * K * N and {g.weight for g in w.groups} == {(True, True)}


def test_an_update_by_a_stated_zero_leaves_a_weight_thetas():
    """W plus z times a stated 0 is W: the multiply gives a stated 0 whatever z holds, so z doesn't reach W and u W^T,
    u carried, is `θ`'s as with W carried unchanged. The updates' premise, held as a fact: W plus z plus -1 times z is W
    too, for a reason no constant shows, so W is θ′ from the second iteration and u W^T is charged."""
    def make(update: str) -> Program:
        def step(B, c, z):
            w, u = c
            zs = Coll(W_T, concat([z.refs] * N))
            if update == "zero":
                w = _add(B, w, _mul(B, zs, B.call(const(32, 0))))
            elif update == "cancel":
                w = _add(B, w, _add(B, zs, _mul(B, zs, B.call(const(32, 0xBF800000)))))
            return (w, _add(B, u, z)), B.call(LIN, u, c[0])
        return _loop(update.title(), (("w", W_T), ("u", X_T)), (("z", X_T),), Y_T, step)

    for update in ("zero", "still"):
        w = WK.read(make(update), KB)
        assert w.groups == [] and w.left_out["theta"] == T * K * N
    cancel = WK.read(make("cancel"), KB)
    (g,) = cancel.groups
    assert g.weight == (True, True) and cancel.N == (T - 1) * K * N and cancel.left_out["theta"] == K * N


#: a score row is D words, so that P V^T is paid (D ≥ K*)
LIN_S = bind(RP.LinearRef, K=K, N=D, DT="f32", BIAS=False)
LIN_P = bind(RP.LinearRef, K=D, N=K, DT="f32", BIAS=False)
LIN_A = bind(RP.LinearRef, K=K, N=PN, DT="f32", BIAS=False)
SOFTMAX = bind(RP.SoftmaxRef, N=D, DT="f32")
S_T, V_T = Array(D, F32), Array(K, Array(D, F32))


def _attention_loop() -> Program:
    """Each iteration looks up PN rows of the carried table by its tokens, each times the carried g (so a block pays
    for its words), scores them by the carried W1 under a causal mask, and multiplies each score row's softmax P by
    the carried V1 and V2: O1 = P V1^T and O2 = P V2^T are specific, so O1 Wo^T is a token-indexed product of an
    activation by a weight and O1 O2^T one of two activations. Wo is updated by the sum of O1's rows over the positions, as a gradient is; the others by an input."""
    def step(B, c, tok, z, zn):
        table, g, w1, v1, v2, wo = c
        rows = _mul(B, Coll(ROWS_T, concat([B.call(EMB, tok[p], table).refs for p in range(PN)])), g)
        s = B.batch(LIN_S, rows, w1, axes=(0, None))
        mask = concat([B.call(const(32, 0 if j <= p else 0xFF800000)).refs for p in range(PN) for j in range(D)])
        flat = Array(PN * D, F32)
        h = B.batch(P.F32Add, Coll(flat, s.refs), Coll(flat, mask), axes=(0, 0)).reshape(Array(PN, S_T))
        p_ = B.batch(SOFTMAX, h, axes=(0,))
        o1, o2 = (B.batch(LIN_P, p_, v, axes=(0, None)) for v in (v1, v2))
        y = concat([B.batch(LIN, o1, wo, axes=(0, None)).refs, B.batch(LIN_A, o1, o2, axes=(0, None)).refs])
        new = (_add(B, table, z), _add(B, g, z), _add(B, w1, z), _add(B, v1, zn), _add(B, v2, zn),
               _add(B, wo, _add(B, o1[0], o1[1])))
        return new, Coll(Array(PN * (N + PN), F32), y)

    carry = (("table", TABLE_T), ("g", X_T), ("w1", Array(D, X_T)), ("v1", V_T), ("v2", V_T), ("wo", W_T))
    return _loop("Attention", carry, (("tok", Array(PN, I32)), ("z", X_T), ("zn", S_T)), Array(PN * (N + PN), F32), step)


def test_a_token_indexed_product_is_charged_at_keep_and_two_activations_are_left_out():
    """O1 Wo^T is charged, its MACs at `keep` in the bound's N, Wo a weight though a sum over the positions reaches it;
    O1 O2^T is left out as two activations, whose dims allow a sequence of PN; the products of P and of the gathered
    rows are left out as gathered."""
    w = WK.read(_attention_loop(), KB)
    (g,) = w.groups
    macs = (T - 1) * PN * K * N
    assert g.weight == (False, True) and w.N == macs and w.keep == 1 - WK.SHARED_PREFIX / PN
    assert WK.bound(w, C).N == pytest.approx(w.keep * macs)
    assert w.left_out == {"theta": PN * K * (3 * D + N + PN), "gathered": 3 * (T - 1) * PN * K * D, "self": 0,
                          "activations": (T - 1) * PN * K * PN}


def test_clique():
    tri = [frozenset(e) for e in ((1, 2), (2, 3), (1, 3), (3, 4))]
    assert WK.clique(tri) == 3
    assert WK.clique(tri + [frozenset(e) for e in ((1, 4), (2, 4))]) == 4
    assert WK.clique([frozenset((1, 2))]) == 2 and WK.clique([]) == 0
