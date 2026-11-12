"""`verity.circuits.evaluation.bits`: the bit-sliced evaluator against the plain-integer reference, on seeded random Boolean circuits that
use every node form (primitive, call, batch over axis 0, a shared and an inner axis, scan) and gates of 1 to 9 inputs, and
through the word view."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Tuple, Value

BIT = BL.BIT
W = 8


def _cat(t, *cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


ops = BL.operands


def _full_adder(B, S, c, a, b):
    t = B.call(BL.XOR, ops(a, b))
    return _cat(Tuple(BIT, BIT), B.call(BL.XOR, ops(B.call(BL.AND, ops(a, b)), B.call(BL.AND, ops(t, c)))), B.call(BL.XOR, ops(t, c)))


#: (carry, a, b) -> (carry', sum bit): the step of a ripple-carry scan
FULL_ADDER = CompositeDefinition("BitsTestFullAdder", 1, (), lambda S: ((("c", BIT), ("a", BIT), ("b", BIT)), Tuple(BIT, BIT)),
                                 _full_adder, register=False)
PARITY3 = CompositeDefinition("BitsTestParity3", 1, (), lambda S: ((("v", Array(3, BIT)),), BIT),
                              lambda B, S, v: B.call(BL.xor_gate(3), v), register=False)


def _adder(B, S, x, y):
    """x + y on W-bit words: a ripple-carry scan of the full adder from the constant 0; the output is the sum then the carry."""
    r = B.scan(FULL_ADDER, B.call(BL.constant(0)), (x, y))
    return _cat(Array(W + 1, BIT), r[1], r[0])


ADDER = CompositeDefinition("BitsTestAdder", 1, (), lambda S: ((("x", Array(W, BIT)), ("y", Array(W, BIT))), Array(W + 1, BIT)),
                            _adder, register=False)


def _random(seed: int) -> CompositeDefinition:
    def body(B, S, x, y, m):
        rng = random.Random(seed)
        pool = [x[i] for i in range(W)] + [y[i] for i in range(W)] + [B.call(BL.constant(rng.randrange(2)))]
        for _ in range(60):
            op = rng.choice(["and", "xor", "not", "fa", "par", "and-n", "xor-n"])
            a, b, c = (rng.choice(pool) for _ in range(3))
            if op == "and":
                pool.append(B.call(BL.AND, ops(a, b)))
            elif op == "xor":
                pool.append(B.call(BL.XOR, ops(a, b)))
            elif op in ("and-n", "xor-n"):
                k = rng.randrange(3, 10)
                gate = BL.and_gate(k) if op == "and-n" else BL.xor_gate(k)
                pool.append(B.call(gate, ops(*(rng.choice(pool) for _ in range(k)))))
            elif op == "not":
                pool.append(B.call(BL.NOT, a))
            elif op == "fa":
                r = B.call(FULL_ADDER, c, a, b)
                pool += [r[0], r[1]]
            else:
                pool.append(B.call(PARITY3, _cat(Array(3, BIT), a, b, c)))
        xy = B.batch(BL.AND, _cat(Array(2, Array(W, BIT)), x, y), axes=(1,))
        nx = B.batch(BL.NOT, xy, axes=(0,))
        shared = rng.choice(pool)
        sh = B.batch(BL.XOR, _cat(Array(W, Array(2, BIT)), *(c for i in range(W) for c in (x[i], shared))), axes=(0,))
        cols = B.batch(PARITY3, m, axes=(1,))
        fas = B.batch(FULL_ADDER, nx, sh, y, axes=(0, 0, 0))
        s = B.call(ADDER, sh, y)
        picks = [rng.choice(pool) for _ in range(12)]
        return _cat(Array(12 + W + 4 + 2 * W + W + 1, BIT), *picks, nx, cols, fas, s)

    return CompositeDefinition(f"BitsTestRandom{seed}", 1, (),
                               lambda S: ((("x", Array(W, BIT)), ("y", Array(W, BIT)), ("m", Array(3, Array(4, BIT)))),
                                          Array(12 + W + 4 + 2 * W + W + 1, BIT)), body, register=False)


def _bits(rng: np.random.Generator, n: int, t) -> np.ndarray:
    from verity.circuits.evaluation.batch import type_shape

    return rng.integers(0, 2, size=(n, *type_shape(t)), dtype=np.uint8)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n", [1, 63, 200])
def test_bit_sliced_equals_the_reference(seed: int, n: int) -> None:
    spec = bind(_random(seed))
    params, _ = spec.signature()
    rng = np.random.default_rng(seed)
    inputs = [_bits(rng, n, t) for _, t in params]
    got = BS.evaluate_bits(spec, *inputs)
    ref = E.evaluate_batch(spec, *inputs, kernel="reference")
    assert got.shape == ref.shape and np.array_equal(got, ref)


def test_a_batch_of_composites_runs_in_chunks_of_members_within_the_row_budget(monkeypatch) -> None:
    spec = bind(_random(3))
    params, _ = spec.signature()
    inputs = [_bits(np.random.default_rng(3), 200, t) for _, t in params]
    fa, run = BS.compile(FULL_ADDER), BS._run
    widths: list[int] = []

    def spy(c, args):
        if c is fa:
            widths.append(args.shape[1])
        return run(c, args)

    monkeypatch.setattr(BS, "_run", spy)
    whole = BS.evaluate_bits(spec, *inputs)
    assert max(widths) == W * 4  # the batch of W full adders side by side, 200 instances in 4 words each
    widths.clear()
    monkeypatch.setattr(BS, "_ROW_BUDGET", 1)
    assert np.array_equal(BS.evaluate_bits(spec, *inputs), whole)
    assert set(widths) == {1}  # one member of one word of instances at a time


#: (x, y) -> W full adders, member i on (x0 & y0, x[i], y[i]): a batch of a composite over a shared argument
SHARED_CARRY = CompositeDefinition(
    "BitsTestSharedCarry", 1, (), lambda S: ((("x", Array(W, BIT)), ("y", Array(W, BIT))), Array(W, Tuple(BIT, BIT))),
    lambda B, S, x, y: B.batch(FULL_ADDER, B.call(BL.AND, ops(x[0], y[0])), x, y, axes=(None, 0, 0)), register=False)


def test_a_batch_holds_a_shared_argument_once(monkeypatch) -> None:
    """A batch's member rows hold each argument it shares once, not once a member, and read as the copies would."""
    c = BS.compile(SHARED_CARRY)
    (members,) = [s[2] for s in c.steps if s[0] == "call"]
    assert isinstance(members, BS.Members) and members.shape == (W, 3) and len(members) == W
    assert [p.ndim for p in members.parts] == [1, 2, 2]
    rows = np.asarray(members.tolist())
    assert np.array_equal(members[2:5], rows[2:5]) and np.array_equal(members[-1], rows[-1])
    assert np.array_equal(np.stack(list(members)), rows) and len(set(rows[:, 0])) == 1
    inputs = [_bits(np.random.default_rng(5), 200, t) for _, t in bind(SHARED_CARRY).signature()[0]]
    ref = E.evaluate_batch(bind(SHARED_CARRY), *inputs, kernel="reference")
    assert np.array_equal(BS.evaluate_bits(SHARED_CARRY, *inputs), ref)
    monkeypatch.setattr(BS, "_ROW_BUDGET", 1)
    assert np.array_equal(BS.evaluate_bits(SHARED_CARRY, *inputs), ref)


#: (c, v) -> c ^ parity(v): a member of a batch over an inner axis beside a shared bit
PARITY3_XOR = CompositeDefinition("BitsTestParity3Xor", 1, (), lambda S: ((("c", BIT), ("v", Array(3, BIT))), BIT),
                                  lambda B, S, c, v: B.call(BL.XOR, ops(c, B.call(PARITY3, v))), register=False)
SHARED_OUT = 2 * W + 3 * (W + 1) + 3 * 2 * W + W


def _shared(B, S, x, y, m):
    """Each form of batch that shares an argument: full adders over a shared carry, ripple-carry adders (a scan) over a shared
    word, a batch of `SHARED_CARRY`, whose members batch over a shared carry of their own, and parities over m's inner axis
    beside a shared bit."""
    c = B.call(BL.AND, ops(x[0], y[0]))
    fas = B.batch(FULL_ADDER, c, x, y, axes=(None, 0, 0))
    adds = B.batch(ADDER, x, m, axes=(None, 0))
    nested = B.batch(SHARED_CARRY, x, m, axes=(None, 0))
    cols = B.batch(PARITY3_XOR, B.call(BL.XOR, ops(x[1], y[2])), m, axes=(None, 1))
    return _cat(Array(SHARED_OUT, BIT), fas, adds, nested, cols)


SHARED = CompositeDefinition(
    "BitsTestShared", 1, (),
    lambda S: ((("x", Array(W, BIT)), ("y", Array(W, BIT)), ("m", Array(3, Array(W, BIT)))), Array(SHARED_OUT, BIT)),
    _shared, register=False)


@pytest.mark.parametrize("row_budget", [None, 1, 5])
def test_a_batch_over_a_shared_argument_computes_what_the_reference_does(monkeypatch, row_budget) -> None:
    """Every form of batch that shares an argument gives the plain-integer reference's outputs, evaluated whole and in runs of
    one or a few members (``_ROW_BUDGET``), which slice `Members` where a shared argument is held once."""
    spec = bind(SHARED)
    inputs = [_bits(np.random.default_rng(11), 200, t) for _, t in spec.signature()[0]]
    ref = E.evaluate_batch(spec, *inputs, kernel="reference")
    if row_budget is not None:
        monkeypatch.setattr(BS, "_ROW_BUDGET", row_budget)
    got = BS.evaluate_bits(SHARED, *inputs)
    assert got.shape == ref.shape and np.array_equal(got, ref)


def test_the_word_view_adds() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 1 << W, size=1000, dtype=np.uint16)
    y = rng.integers(0, 1 << W, size=1000, dtype=np.uint16)
    out = BS.evaluate_words(ADDER, x, y)
    assert out.shape == (1000,) and np.array_equal(out, x + y)
    with pytest.raises(ValueError):
        BS.evaluate_words(ADDER, x | (1 << W), y)


def test_a_boolean_definition_takes_its_word_definitions_kernel(monkeypatch) -> None:
    """ADDER's twin names x + y as its word Definition, whose kernel declines odd x: evaluate_batch packs each element's bits
    into a word, runs that kernel, unpacks its output, and evaluates the declined rows bit-sliced, never gate by gate."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.evaluation.batch import _KERNELS, _WORD

    add = PrimitiveDefinition("BitsTestAddWord", 1, (("x", Value(W)), ("y", Value(W))), Value(W + 1), lambda x, y: x + y,
                              register=False)
    twin = CompositeDefinition("BitsTestWordAdder", 1, (),
                               lambda S: ((("x", Array(W, BIT)), ("y", Array(W, BIT))), Array(W + 1, BIT)), _adder, register=False)
    twin.word = add
    spec = bind(twin)
    rng = np.random.default_rng(4)
    x, y = (_bits(rng, 200, Array(W, BIT)) for _ in range(2))
    want = BS.evaluate_bits(spec, x, y)
    batches, bit_rows = [], []

    def kernel(a, b):
        batches.append(len(a))
        return a.astype(np.uint16) + b, a % 2 == 1

    def bits(t, *arrays):
        bit_rows.append(arrays[0].shape[0])
        return BS.evaluate_bits(t, *arrays)

    E.register_kernel(add, kernel)
    try:
        k = E.word_kernel(spec)
        assert (k.key, k.name) == ("BitsTestWordAdder_v1", "numpy") and E.kernels(spec) == {}
        monkeypatch.setattr(E.batch, "evaluate", lambda *a, **kw: pytest.fail("the reference ran"))
        monkeypatch.setattr(E.batch, "evaluate_bits", bits)
        assert np.array_equal(E.evaluate_batch(spec, x, y), want)
        assert batches == [200] and bit_rows == [int(x[:, 0].sum())]
    finally:
        del _KERNELS[add.id]
        _WORD.clear()


def test_gemm_v3_runs_on_gemm_v2s_kernel(monkeypatch) -> None:
    """The BF16 GEMM on bits takes the word GEMM's numpy kernel; the s8 accumulator's word Definition has no kernel, so it
    stays bit-sliced."""
    from verity.definitions.ops.boolean import gemm as BG

    gemm = bind(BG.Gemm, K=32, N=4, **BG.GEMM_AMPERE)
    assert E.word_kernel(gemm).name == "numpy"
    assert E.word_kernel(bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)) is None
    rng = np.random.default_rng(5)
    bf16 = lambda n: (np.float32(rng.uniform(-1, 1, n)).view(np.uint32) >> 16).astype(np.uint16)  # noqa: E731
    x, w = bf16((3, 32)), bf16((3, 4, 32))
    want = BS.evaluate_words(gemm, x, w)
    unpack = lambda a: ((a[..., None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8)  # noqa: E731
    monkeypatch.setattr(E.batch, "evaluate_bits", lambda *a: pytest.fail("evaluated bit-sliced"))
    got = E.evaluate_batch(gemm, unpack(x), unpack(w))
    assert np.array_equal(BS._to_words(got.reshape(3, -1), [16] * 4), want)


def test_evaluate_batch_routes_a_boolean_composite(monkeypatch) -> None:
    spec = bind(_random(0))
    params, _ = spec.signature()
    inputs = [_bits(np.random.default_rng(2), 100, t) for _, t in params]
    ref = E.evaluate_batch(spec, *inputs, kernel="reference")

    def no_reference(*_a, **_k):
        raise AssertionError("the reference ran")

    monkeypatch.setattr(E.batch, "evaluate", no_reference)
    assert np.array_equal(E.evaluate_batch(spec, *inputs), ref)


def _wide_adder(w: int) -> CompositeDefinition:
    def body(B, S, x, y):
        r = B.scan(FULL_ADDER, B.call(BL.constant(0)), (x, y))
        return _cat(Array(w + 1, BIT), r[1], r[0])

    return CompositeDefinition(f"BitsTestAdder{w}", 1, (), lambda S: ((("x", Array(w, BIT)), ("y", Array(w, BIT))), Array(w + 1, BIT)),
                               body, register=False)


def _held(spec, n: int) -> int:
    import tracemalloc

    rng = np.random.default_rng(n)
    inputs = [_bits(rng, n, t) for _, t in spec.signature()[0]]
    BS.compile(spec)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        BS.evaluate_bits(spec, *inputs)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("n", [1, 200, 5000])
def test_peak_bytes_bounds_what_evaluate_bits_holds(n: int) -> None:
    from verity.definitions.ops.boolean import gemm as BG

    for spec in (bind(_random(0)), bind(_random(3)), bind(_wide_adder(1024))):
        assert _held(spec, n) <= BS.peak_bytes(spec, n)
    gemm = bind(BG.Gemm, K=32, N=64, DOT=BG.HopperBF16WgmmaDot16)
    held, bound = _held(gemm, min(n, 200)), BS.peak_bytes(gemm, min(n, 200))
    assert held <= bound <= 1.15 * held  # a wide body: the bound is what a replay worker is fitted to


def test_compiled_counts_and_refusals() -> None:
    c = BS.compile(ADDER)
    assert c.counts == {"and": 2 * W, "xor": 3 * W, "not": 0}
    assert BS.compile(ADDER) is c
    word = CompositeDefinition("BitsTestWord", 1, (), lambda S: ((("a", Value(8)),), Value(8)), lambda B, S, a: a, register=False)
    with pytest.raises(TypeError):
        BS.compile(word)
    with pytest.raises(ValueError):
        BS.evaluate_bits(ADDER, np.full((4, W), 2, np.uint8), np.zeros((4, W), np.uint8))
