"""`verity.ml.circuits.evaluation.bits`: the bit-sliced evaluator against the plain-integer reference, on seeded random Boolean circuits that
use every node form (primitive, call, batch over axis 0, a shared and an inner axis, scan) and gates of 1 to 9 inputs, and
through the word view."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple, Value

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
    from verity.ml.circuits.evaluation.batch import type_shape

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


def test_the_word_view_adds() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 1 << W, size=1000, dtype=np.uint16)
    y = rng.integers(0, 1 << W, size=1000, dtype=np.uint16)
    out = BS.evaluate_words(ADDER, x, y)
    assert out.shape == (1000,) and np.array_equal(out, x + y)
    with pytest.raises(ValueError):
        BS.evaluate_words(ADDER, x | (1 << W), y)


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
    from verity.ml.circuits.ops.boolean import gemm as BG

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
