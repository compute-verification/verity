"""`circuit_check.bodies`: a composite run body by body is the reference (`verity.ml.circuits.evaluation.evaluate`) on every
construct a body has (a call, a batch along no axis, axis 0 and a later one, a batch and a scan of composites, a scan with xs
and shared arguments, a lazy and a wide-fan-in primitive, a parameter returned as is, And2, Xor2 and Not alone and batched on
any operands, a body of gates and constant bits on rows of bits `LANES` at once), raises what the reference raises with
the same gate indices, with every activation run and with a repeated one taking an earlier one's values (`bodies._Memo`), and
leaves to the reference what it doesn't compile; its non-constant gates are the ones `checks._pieces_used` counts gate by
gate."""
from __future__ import annotations

import itertools
import random
from collections import Counter

import pytest

from circuit_check import bodies as BD
from circuit_check import checks as CK
from circuit_check import targets as TG
from circuit_check import vectors as VX
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate as reference
from verity.ml.circuits.evaluation.reference import signature_of
from verity.ml.circuits.layout import WIDE_FAN_IN
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V8 = Value(8)
N_WIDE = WIDE_FAN_IN + 3


def _prim(name: str, params: tuple, fn, lazy=None, family: str | None = None) -> PrimitiveDefinition:
    p = PrimitiveDefinition(f"CcBodies{name}", 1, tuple((f"a{i}", t) for i, t in enumerate(params)), V8, fn, register=False)
    if lazy is not None:
        p.evaluate_lazy = lazy  # type: ignore[attr-defined]
    if family is not None:
        p.family = family  # type: ignore[attr-defined]
    return p


def _composite(name: str, params: tuple, ret, body):
    return bind(CompositeDefinition(f"CcBodies{name}", 1, (), lambda S: (tuple((f"p{i}", t) for i, t in enumerate(params)), ret),
                                    body, register=False))


def _flat(t, parts) -> Coll:
    return Coll(t, concat([p.refs for p in parts]))


def _nonzero(a, b):
    if b == 0:
        raise ValueError(f"a zero beside {a}")
    return a // b


ADD = _prim("Add", (V8, V8), lambda a, b: (a + b) & 255)
MUL = _prim("Mul", (V8, V8), lambda a, b: (a * b + 1) & 255)
NEG = _prim("Neg", (V8,), lambda a: (-a) & 255)
DIV = _prim("Div", (V8, V8), _nonzero)
SUM3 = _prim("Sum3", (Array(3, V8),), lambda a, b, c: (a + 2 * b + 3 * c) & 255)
PICK = _prim("Pick", (V8, Array(4, V8)), lambda i, *t: t[i % 4], lazy=lambda get: get(1 + get(0) % 4))
REACH = _prim("Reach", (V8, Array(4, V8)), lambda i, *t: t[i], lazy=lambda get: get(get(0)))
WIDE = _prim("Wide", (V8, Array(N_WIDE, V8)), lambda i, *t: t[i] ^ t[-1], lazy=lambda get: get(1 + get(0)) ^ get(-1))
FOLD = _prim("Fold", (Array(N_WIDE, V8),), lambda *t: sum(t) & 255)
MISS = _prim("Miss", (V8,), lambda a: a, lazy=lambda get: get(0) if get(0) % 3 else {}[get(0)])
SEVEN = _prim("Seven", (), lambda: 7)
INPUT = _prim("Input", (), lambda: 0, family="Input")

INNER = _composite("Inner", (V8, V8), V8, lambda B, S, a, b: B.call(MUL, B.call(NEG, B.call(ADD, a, b)), a))
INNER3 = _composite("Inner3", (Array(3, V8),), V8, lambda B, S, v: B.call(ADD, B.call(SUM3, v), v[1]))


def _step(B, S, c, x, s):
    c2 = B.call(ADD, c, B.call(MUL, x, s))
    return tuple_of(c2, B.call(NEG, c2))


STEP = _composite("Step", (V8, V8, V8), Tuple(V8, V8), _step)
TOP_PARAMS = (Array(6, V8), V8, Array(3, Array(2, V8)), Array(N_WIDE, V8))


def _top(B, S, x, y, m, w):
    b0 = B.batch(ADD, x[0:3], y, axes=(0, None))
    c1 = B.call(INNER, x[0], y)
    parts = [
        c1, b0,
        B.batch(SUM3, m, axes=(1,)),
        B.batch(INNER, x[0:3], x[3:6], axes=(0, 0)),
        B.batch(INNER, x[0:3], y, axes=(0, None)),
        B.batch(INNER3, m, axes=(1,)),
        B.scan(STEP, c1, xs=[x], shared=[b0[2]]),
        B.scan(STEP, y, shared=[x[2], x[3]], n=2),
        B.call(PICK, y, x[1:5]),
        B.batch(PICK, x[0:2], _flat(Array(2, Array(4, V8)), [x[1:5], x[2:6]]), axes=(0, 0)),
        B.call(WIDE, y, w),
        B.call(FOLD, w),
        x[5], y,
    ]
    return _flat(Array(sum(p.type.leaves for p in parts), V8), parts)


TOP = _composite("Top", TOP_PARAMS, Array(31, V8), _top)


def _outcome(f, *args):
    try:
        return ("returned", tuple(f(*args)))
    except Exception as e:  # noqa: BLE001  (what was raised is the outcome compared)
        return ("raised", type(e).__name__, str(e))


def _random_args(params, rng: random.Random, hi: int = 255) -> list:
    return [rng.randint(0, hi) if isinstance(t, Value) else [rng.randint(0, hi) for _ in range(t.leaves)] for t in params]


def _alike(params, v: int) -> list:
    """Every leaf ``v``, as an edge vector sets them."""
    return [v if isinstance(t, Value) else [v] * t.leaves for t in params]


@pytest.fixture(params=[False, True], ids=["every-activation-run", "repeats-looked-up"])
def memo(request, monkeypatch):
    """Bodies compiled anew, each activation looked up (any body) or none."""
    monkeypatch.setattr(BD, "_BODIES", {})
    monkeypatch.setattr(BD, "MEMO_GATES_PER_LEAF", 0 if request.param else 1 << 60)
    return request.param


def test_a_body_with_every_construct_evaluates_as_the_reference(memo):
    assert BD.compiled(TOP) is not None, "the test would compare the reference with itself"
    assert BD.compiled(TOP).memo is memo
    rng = random.Random(7)
    for args in [_random_args(TOP_PARAMS, rng) for _ in range(6)] + [_alike(TOP_PARAMS, v) for v in (0, 1, 3, 255)]:
        got = _outcome(BD.evaluate, TOP, *args)
        assert got[0] == "returned" and got == _outcome(reference, TOP, *args)


ODD = (0, 1, 2, 3, -1, -2, 1 << 70, (1 << 70) + 1, True, False, None, 1.0, "1")


def _typed(outcome: tuple) -> tuple:
    return outcome if outcome[0] == "raised" else (outcome[0], tuple((type(v), v) for v in outcome[1]))


def test_and2_xor2_and_not_on_any_operands_are_their_evaluators():
    """`bodies._and2`, `_xor2` and a `_COLS` step's columns return what And2, Xor2 and Not's evaluators return, of the same type,
    and on operands that aren't all `int` the step takes the evaluators, so it raises what they raise."""
    for prim, pair in BD._PAIRS.items():
        for a, b in itertools.product(ODD, repeat=2):
            fast, ev = _outcome(lambda: (pair(a, b),)), _outcome(lambda: (prim.evaluate(a, b),))
            assert _typed(fast) == _typed(ev), (prim.id, a, b)
    ints = [v for v in ODD if type(v) is int]
    for prim, bits in BD._BITS.items():
        cols = [list(c) for c in zip(*itertools.product(ints, repeat=sum(p.leaves for _, p in prim.params)))]
        assert _typed(("returned", tuple(bits(*cols)))) == _typed(("returned", tuple(map(prim.evaluate, *cols))))


BIT = BL.BIT
QUAD = _composite("Quad", (Array(2, Array(2, BIT)),), BIT,
                  lambda B, S, q: B.call(BL.XOR, BL.operands(B.call(BL.AND, q[0]), B.call(BL.AND, q[1]))))
GATES_PARAMS = (Array(4, Array(2, BIT)), Array(2, Array(4, BIT)), Array(2, Array(3, Array(2, BIT))), BIT)


def _gates(B, S, p, t, z, c):
    a = B.batch(BL.AND, p, axes=(0,))
    x = B.batch(BL.XOR, t, axes=(1,))
    n = B.batch(BL.NOT, a, axes=(0,))
    q = B.batch(QUAD, z, axes=(1,))
    one = B.call(BL.AND, BL.operands(x[0], c))
    two = B.call(BL.XOR, BL.operands(one, q[2]))
    mixed = B.batch(BL.AND, _flat(Array(2, Array(2, BIT)), [c, t[0][1], t[1][2], c]), axes=(0,))
    parts = [a, x, n, q, one, two, B.call(BL.NOT, two), B.call(BL.XOR, BL.operands(c, n[1])), mixed]
    return _flat(Array(sum(p.type.leaves for p in parts), BIT), parts)


GATES = _composite("Gates", GATES_PARAMS, Array(21, BIT), _gates)


def test_boolean_gates_on_any_operands_evaluate_as_the_reference(memo):
    """And2, Xor2 and Not as single gates and as batches (along axis 0, along a later one, over a batch's own outputs, inside a
    composite batched along a later axis of runs of two leaves) return the reference's values, of the same types, on bits, on
    `int` leaves past one bit, and on a `bool` (a scalar parameter, which a single gate and a batch read), which takes the
    evaluators."""
    body = BD.compiled(GATES)
    assert body is not None, "the test would compare the reference with itself"
    assert {st[0] for st in body.steps} >= {BD._COLS, BD._PRIM2, BD._PRIM1}
    assert {st[1] for st in body.steps if st[0] == BD._PRIM2} == {BD._and2, BD._xor2}
    rng = random.Random(11)
    leaves = [v for v in ODD if type(v) is int]
    for c in (0, 1, True, False, 3):
        for pick in (lambda: rng.randint(0, 1), lambda: rng.choice(leaves)):
            args = [[pick() for _ in range(t.leaves)] for t in GATES_PARAMS[:3]] + [c]
            got = _outcome(BD.evaluate, GATES, *args)
            assert got[0] == "returned" and _typed(got) == _typed(_outcome(reference, GATES, *args)), (args, c)


HALF = PrimitiveDefinition("CcBodiesHalf", 1, (("a", BIT),), BIT, lambda a: a / 2, register=False)
HALVES = _composite("Halves", (Array(2, BIT), Array(2, BIT)), Array(4, BIT), lambda B, S, x, y: _flat(Array(4, BIT), [
    h := B.batch(HALF, x, axes=(0,)), B.batch(BL.XOR, _flat(Array(2, Array(2, BIT)), [h[0], y[0], h[1], y[1]]), axes=(0,))]))
CALLS_HALVES = _composite("CallsHalves", (Array(2, BIT), Array(2, BIT)), Array(4, BIT),
                          lambda B, S, x, y: B.call(HALVES, x, y))


def test_a_body_with_a_primitive_past_the_gates_takes_the_type_checks(memo):
    """A body has steps on lanes (`bodies._lanes`) only when every primitive it reaches is a gate or a constant bit: a batch of
    Xor2 on a primitive's `float` values, in a body called from one of no primitives, raises what the reference raises."""
    assert BD._lanes(BD.compiled(GATES)) is not None and BD._lanes(BD.compiled(TOP)) is None
    assert BD._lanes(BD.compiled(CALLS_HALVES)) is None
    for args in ([[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]):
        got = _outcome(BD.evaluate, CALLS_HALVES, *args)
        assert got[0] == "raised" and got == _outcome(reference, CALLS_HALVES, *args), args


BIT_STEP = _composite("BitStep", (BIT, BIT), Tuple(BIT, BIT), lambda B, S, c, x: tuple_of(
    B.call(BL.XOR, BL.operands(c, x)), B.call(BL.NOT, B.call(BL.AND, BL.operands(c, x)))))
LANE_PARAMS = (*GATES_PARAMS, Array(N_WIDE, BIT))


def _lane(B, S, p, t, z, c, w):
    g = B.call(GATES, p, t, z, c)
    parts = [g, B.call(BL.and_gate(N_WIDE), w), B.call(BL.xor_gate(3), BL.operands(c, w[0], g[3])),
             B.batch(BL.and_gate(3), _flat(Array(2, Array(3, BIT)), [w[1], w[2], c, g[0], c, w[3]]), axes=(0,)),
             B.call(BL.constant(1)), B.call(BL.constant(0)), B.batch(BL.constant(1), axes=(), n=2),
             B.batch(BL.constant(0), axes=(), n=2), B.scan(BIT_STEP, c, xs=[w[0:4]]), B.batch(BL.NOT, w[4:7], axes=(0,))]
    return _flat(Array(sum(p.type.leaves for p in parts), BIT), parts)


LANE = _composite("Lane", LANE_PARAMS, Array(39, BIT), _lane)


def _row(got) -> tuple:
    return ("raised", type(got).__name__, str(got)) if isinstance(got, Exception) else ("returned", tuple(got))


def test_rows_of_bits_run_on_lanes_as_the_reference(memo, monkeypatch):
    """`evaluate_rows` on rows whose leaves are all bits runs them `LANES` at once (a call of a body of gates, a wide gate, n-ary
    gates alone and batched, constant bits alone and batched, a scan, a batch of Not), each row's outputs the reference's, of
    the same types; rows with a leaf past one bit or a `bool`, or of a body with a primitive past the gates, run one by one,
    raising what the reference raises; and what evaluating raises before it runs a row is each row's outcome."""
    runs: list[int] = []
    run = BD._lanes_run
    monkeypatch.setattr(BD, "_lanes_run", lambda body, lo, Vs: runs.append(len(Vs)) or run(body, lo, Vs))
    assert BD._lanes(BD.compiled(LANE)) is not None
    rng = random.Random(13)
    for n in (1, 3, BD.LANES, BD.LANES + 1, 2 * BD.LANES + 5):
        rows = [_random_args(LANE_PARAMS, rng, 1) for _ in range(n)]
        runs.clear()
        got = BD.evaluate_rows(LANE, rows)
        assert runs == [n], "the rows would run one by one"
        assert [_typed(_row(g)) for g in got] == [_typed(_outcome(reference, LANE, *r)) for r in rows]
    rows = [_random_args(LANE_PARAMS, rng, 1) for _ in range(5)]
    for odd in (2, True):
        mixed = [*rows[:2], [*rows[2][:3], odd, rows[2][4]], *rows[3:]]
        runs.clear()
        got = BD.evaluate_rows(LANE, mixed)
        assert runs == [1] * 4 and got[2] == tuple(reference(LANE, *mixed[2]))
        assert [_typed(_row(g)) for g in got] == [_typed(_outcome(reference, LANE, *r)) for r in mixed]
    divides = _composite("RowsDivide", (V8, V8), V8, lambda B, S, a, b: B.call(DIV, a, b))
    rows = [[1, 1], [1, 0], [0, 1]]
    assert [_row(g) for g in BD.evaluate_rows(divides, rows)] == [_outcome(reference, divides, *r) for r in rows]

    def no_call(t):
        raise RuntimeError(f"no standalone call of {t.id}")

    monkeypatch.setattr(BD, "standalone_call", no_call)
    rows = rows[:2] + [_random_args(LANE_PARAMS, rng, 1)]
    for fn, rs in ((LANE, rows[2:]), (LANE, rows[2:] * 3), (divides, rows[:2])):
        assert [_row(g) for g in BD.evaluate_rows(fn, rs)] == [_outcome(BD.evaluate, fn, *r) for r in rs]
        assert {_row(g)[0] for g in BD.evaluate_rows(fn, rs)} == {"raised"}


CALLS: Counter = Counter()


def _mix(a, b):
    CALLS["mix"] += 1
    return (3 * a + b) & 255


MIX = _prim("Mix", (V8, V8), _mix)
TWICE = _composite("Twice", (V8, V8), V8, lambda B, S, a, b: B.call(MIX, B.call(MIX, a, b), b))
TWICE_STEP = _composite("TwiceStep", (V8, V8), Tuple(V8, V8), lambda B, S, c, s: tuple_of(B.call(TWICE, c, s), c))


def test_a_repeated_activation_takes_the_values_of_the_first(memo, monkeypatch):
    """A batch's members, a scan's steps and calls on one body's parameter values run once each within one `evaluate` (none of
    them again in the next), and return what the reference does; with no room left (`MEMO_LEAVES`) every one runs."""
    def body(B, S, x, y):
        b = B.batch(TWICE, x, y, axes=(0, None))
        c = B.call(TWICE, x[0], y)
        s = B.scan(TWICE_STEP, y, shared=[y], n=3)
        return _flat(Array(9, V8), [b, c, s])

    rep = _composite("Repeats", (Array(4, V8), V8), Array(9, V8), body)
    for x, y, runs in (([5, 5, 5, 5], 1, 1 + 3), ([5, 6, 5, 6], 1, 2 + 3), ([4, 5, 6, 7], 0, 4 + 1)):
        CALLS.clear()
        got = _outcome(BD.evaluate, rep, x, y)
        assert CALLS["mix"] == 2 * (runs if memo else 4 + 1 + 3), (x, y)
        assert got[0] == "returned" and got == _outcome(reference, rep, x, y)
    monkeypatch.setattr(BD, "MEMO_LEAVES", 0)
    CALLS.clear()
    got = _outcome(BD.evaluate, rep, [5, 5, 5, 5], 1)
    assert CALLS["mix"] == 2 * (4 + 1 + 3)
    assert got == _outcome(reference, rep, [5, 5, 5, 5], 1)


def test_an_activation_that_raised_is_not_taken_again(memo):
    """A member that raises after members it repeats raises with its own gate index, and the member it repeats after one that
    raised is never reached: the first to raise is the reference's."""
    divs = _composite("Divs", (V8, V8), V8, lambda B, S, a, b: B.call(ADD, B.call(DIV, a, b), a))
    leaf = _composite("MissOnce", (V8,), V8, lambda B, S, v: B.call(ADD, B.call(MISS, v), v))
    batch_divs = _composite("BatchDivs", (Array(4, V8), Array(4, V8)), Array(4, V8),
                            lambda B, S, a, b: B.batch(divs, a, b, axes=(0, 0)))
    batch_miss = _composite("BatchMiss", (Array(4, V8),), Array(4, V8), lambda B, S, a: B.batch(leaf, a, axes=(0,)))
    cases = [(batch_divs, [[6, 6, 7, 7], [2, 2, 0, 0]]), (batch_divs, [[6, 6, 6, 6], [2, 2, 2, 2]]),
             (batch_miss, [[1, 1, 3, 3]]), (batch_miss, [[1, 1, 4, 6]]), (batch_miss, [[2, 2, 2, 2]])]
    raised = 0
    for fn, args in cases:
        got = _outcome(BD.evaluate, fn, *args)
        assert got == _outcome(reference, fn, *args), (fn.id, args)
        raised += got[0] == "raised"
    assert raised == 3


def test_what_the_reference_raises_is_raised_with_the_same_gate_index(memo):
    """A primitive's own exception from the first member that raises, a lazy primitive's KeyError as the reference's
    `MissingValue` naming its root-scoped gate (in a batch of a composite inside a call), a lazy read past its operands, an
    Input gate, and a wrong argument count."""
    leaf = _composite("MissLeaf", (V8,), V8, lambda B, S, v: B.call(MISS, B.call(NEG, v)))
    mid = _composite("MissMid", (Array(4, V8),), Array(4, V8),
                     lambda B, S, a: (B.call(ADD, a[0], a[1]), B.batch(leaf, a, axes=(0,)))[1])
    missing = _composite("Missing", (Array(4, V8),), Array(4, V8), lambda B, S, a: B.call(mid, a))
    divides = _composite("Divides", (Array(4, V8), Array(4, V8)), Array(4, V8),
                         lambda B, S, a, b: B.batch(DIV, a, b, axes=(0, 0)))
    reach = _composite("Reach", (V8, Array(4, V8)), V8, lambda B, S, i, t: B.call(ADD, B.call(REACH, i, t), i))
    unset = _composite("Unset", (V8,), V8, lambda B, S, a: B.call(ADD, a, B.call(INPUT)))
    cases = [(missing, [[1, 2, 4, 3]]), (missing, [[2, 3, 5, 4]]), (missing, [[2, 3, 5, 6]]),
             (divides, [[5, 6, 7, 8], [1, 0, 2, 0]]), (divides, [[5, 6, 7, 8], [1, 1, 2, 3]]),
             (reach, [2, [1, 2, 3, 4]]), (reach, [9, [1, 2, 3, 4]]), (unset, [3]), (divides, [[5, 6, 7, 8]])]
    raised = 0
    for fn, args in cases:
        assert BD.compiled(fn) is not None
        got = _outcome(BD.evaluate, fn, *args)
        assert got == _outcome(reference, fn, *args), (fn.id, args)
        raised += got[0] == "raised"
    assert raised == 6


def test_a_body_it_does_not_compile_is_left_to_the_reference(monkeypatch):
    """A scan of no steps (an output the reference refuses to read) and a body whose activations copy far more leaves than it
    has gates are not compiled; `evaluate` and `nonconstant` then answer as the reference does."""
    monkeypatch.setattr(BD, "TRAFFIC_SLACK", 0)
    once = _composite("Once", (V8,), Tuple(V8, V8), lambda B, S, c: tuple_of(B.call(NEG, c), c))
    empty = _composite("NoSteps", (V8,), V8, lambda B, S, a: B.call(ADD, B.scan(once, a, n=0)[0], a))
    first = _composite("First", (Array(256, V8),), V8, lambda B, S, u: B.call(NEG, u[0]))
    copies = _composite("Copies", (Array(256, V8),), Array(8, V8), lambda B, S, v: B.batch(first, v, axes=(None,), n=8))
    for fn, args in ((empty, [3]), (copies, [list(range(256))])):
        assert BD.compiled(fn) is None and BD.nonconstant(fn) is None
        assert _outcome(BD.evaluate, fn, *args) == _outcome(reference, fn, *args)


def _pieces(monkeypatch, fn) -> tuple:
    pins = CK.load_pins()
    fast = CK._pieces_used(fn, pins)
    with monkeypatch.context() as m:
        m.setattr(BD, "nonconstant", lambda t: None)
        slow = CK._pieces_used(fn, pins)
    return fast, slow


def test_the_non_constant_gates_are_the_ones_counted_gate_by_gate(monkeypatch):
    """Constants (a ``Const`` gate, a gate of no parameters, one whose operands are all constant) flow into calls, batches
    and scans, so a body's gates are constant in one activation and not in another; the counts, the estimate and the order
    each primitive first appears in are the gate-by-gate walk's."""
    from verity.ml.definitions.registry import prims as P

    k5 = const(8, 5)

    def body(B, S, x, y):
        c = B.call(k5)
        s = B.call(SEVEN)
        both = B.call(INNER, c, s)                                     # constant throughout
        half = B.call(INNER, c, y)                                     # constant only where it doesn't read y
        mixed = B.batch(INNER, _flat(Array(2, V8), [c, y]), _flat(Array(2, V8), [s, s]), axes=(0, 0))
        run = B.scan(STEP, c, xs=[_flat(Array(3, V8), [s, x[0], c])], shared=[s])
        wide = B.call(FOLD, _flat(Array(N_WIDE, V8), [c] * N_WIDE))
        return _flat(Array(9, V8), [both, half, mixed, run, wide])

    fn = _composite("Consts", (Array(2, V8), V8), Array(9, V8), body)
    fast, slow = _pieces(monkeypatch, fn)
    assert fast == slow and [list(c) for c in fast[:2]] == [list(c) for c in slow[:2]]
    assert fast[1] and not fast[0]
    f32 = _composite("F32", (Array(2, Value(32)),), Value(32),
                     lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[1], x[0])))
    fast, slow = _pieces(monkeypatch, f32)
    assert fast == slow and fast[0] == {"F32Add_v1": 2, "F32Mul_v1": 1} and list(fast[0]) == list(slow[0])


#: registered Definitions the suite checks: words with lazy reads, a Boolean Call, a dot product, Boolean GF(2^128) arithmetic
REGISTERED = ("TokenSelect_v1{V=16}", "RoPEHead_v1{D=8}", "RoPEHead_v2{D=8}", "RMSNormTriton_v1{N=64,EPS=1e-05}",
              "DotBf16_v1{K=16}", "GfScale_v1{L=2}")


@pytest.mark.parametrize("fid", REGISTERED)
def test_registered_definitions_evaluate_and_count_as_the_reference(fid, memo, monkeypatch):
    fn = TG.definition(fid)
    assert BD.compiled(fn) is not None
    params, _ = signature_of(fn)
    edges = VX.edge_count(params)
    arrays = VX.vectors(params, edges + 2, 1)
    for i in (0, edges - 1, edges, edges + 1):
        args = VX.instance_args(params, arrays, i)
        assert _outcome(BD.evaluate, fn, *args) == _outcome(reference, fn, *args), (fid, i)
    ref, raised = CK.reference_outputs(fn, params, arrays)
    want = [_outcome(reference, fn, *VX.instance_args(params, arrays, i)) for i in range(edges + 2)]
    assert [("returned", r) if r is not None else "raised" for r in ref] == [w if w[0] == "returned" else "raised" for w in want]
    assert [r["instance"] for r in raised] == [i for i, w in enumerate(want) if w[0] == "raised"]
    fast, slow = _pieces(monkeypatch, fn)
    assert fast == slow and [list(c) for c in fast[:2]] == [list(c) for c in slow[:2]]
