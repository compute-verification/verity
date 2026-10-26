"""`circuit_check.bodies`: a composite run body by body is the reference (`verity.ml.circuits.evaluation.evaluate`) on every
construct a body has (a call, a batch along no axis, axis 0 and a later one, a batch and a scan of composites, a scan with xs
and shared arguments, a lazy and a wide-fan-in primitive, a parameter returned as is), raises what the reference raises with
the same gate indices, and leaves to the reference what it doesn't compile; its non-constant gates are the ones
`checks._pieces_used` counts gate by gate."""
from __future__ import annotations

import random

import pytest

from circuit_check import bodies as BD
from circuit_check import checks as CK
from circuit_check import targets as TG
from circuit_check import vectors as VX
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


def test_a_body_with_every_construct_evaluates_as_the_reference():
    assert BD.compiled(TOP) is not None, "the test would compare the reference with itself"
    rng = random.Random(7)
    for _ in range(6):
        args = _random_args(TOP_PARAMS, rng)
        got = _outcome(BD.evaluate, TOP, *args)
        assert got[0] == "returned" and got == _outcome(reference, TOP, *args)


def test_what_the_reference_raises_is_raised_with_the_same_gate_index():
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
def test_registered_definitions_evaluate_and_count_as_the_reference(fid, monkeypatch):
    fn = TG.definition(fid)
    assert BD.compiled(fn) is not None
    params, _ = signature_of(fn)
    edges = VX.edge_count(params)
    arrays = VX.vectors(params, edges + 2, 1)
    for i in (0, edges - 1, edges, edges + 1):
        args = VX.instance_args(params, arrays, i)
        assert _outcome(BD.evaluate, fn, *args) == _outcome(reference, fn, *args), (fid, i)
    fast, slow = _pieces(monkeypatch, fn)
    assert fast == slow and [list(c) for c in fast[:2]] == [list(c) for c in slow[:2]]
