"""`verity.ml.circuits.evaluation.accelerate`: a Boolean Definition with an admitted word accelerator is evaluated through it by
`evaluate`, `evaluate_batch` and `evaluate_circuit`, and on its gates when asked (``gates=True``, ``kernel="gates"``); a
Definition that isn't admitted is evaluated on its gates. `WRONG` admits an accelerator that differs from its gates, so which path
ran shows in the values; circuit-check is what refuses such an admission (`circuit_check`'s ``accelerator-mismatch``)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import accelerate as A
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Value

BIT = BL.BIT
W = 8
WORD = Array(W, BIT)


def _cat(t, *cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


def _xor_bits(B, S, x, y):
    """NOT NOT (x XOR y), so that a call has gates besides its returns."""
    xor = B.batch(BL.XOR, _cat(Array(W, Array(2, BIT)), *(c for i in range(W) for c in (x[i], y[i]))), axes=(0,))
    return B.batch(BL.NOT, B.batch(BL.NOT, xor, axes=(0,)), axes=(0,))


def _boolean(name: str, word: PrimitiveDefinition | None, admitted: bool) -> CompositeDefinition:
    d = CompositeDefinition(name, 1, (), lambda S: ((("x", WORD), ("y", WORD)), WORD), _xor_bits, register=False)
    d.word = word  # type: ignore[attr-defined]
    return A.admit(d) if admitted else d


def _word(name: str, fn, ret: int = W) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, (("x", Value(W)), ("y", Value(W))), Value(ret), fn, register=False)


XOR_WORD = _word("AccelTestXor", lambda x, y: x ^ y)
XOR = _boolean("AccelTestXorBits", XOR_WORD, True)
WRONG = _boolean("AccelTestWrongBits", _word("AccelTestOr", lambda x, y: x | y), True)
PLAIN = _boolean("AccelTestPlainBits", XOR_WORD, False)
BAD_SIGNATURE = _boolean("AccelTestBadSignatureBits", _word("AccelTestWideXor", lambda x, y: x ^ y, W + 1), True)


def _words(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.integers(0, 1 << W, n, dtype=np.uint64), rng.integers(0, 1 << W, n, dtype=np.uint64)


def _bits(words: np.ndarray) -> np.ndarray:
    return A.to_bits(words.reshape(-1, 1), [W])


def test_words_and_bits_are_inverse_on_mixed_and_wide_elements() -> None:
    rng = np.random.default_rng(0)
    for widths in ([1, 8, 8, 16, 3, 64, 64, 2], [1, 8, 70, 70, 5]):
        bits = rng.integers(0, 2, (40, sum(widths)), dtype=np.uint8)
        words = A.to_words(bits, widths)
        assert words.shape == (40, len(widths)) and np.array_equal(A.to_bits(words, widths), bits)
        assert int(words[3, 1]) == sum(int(b) << j for j, b in enumerate(bits[3, widths[0]:widths[0] + widths[1]]))
    assert A.to_words(bits, [1, 8, 70, 70, 5]).dtype == object


def test_evaluate_and_evaluate_batch_dispatch_to_the_admitted_accelerator() -> None:
    x, y = _words(np.random.default_rng(1), 300)
    xb, yb = _bits(x), _bits(y)
    assert A.accelerator(XOR) is XOR_WORD and A.accelerator(PLAIN) is None and set(A.admitted()) >= {XOR, WRONG}
    assert np.array_equal(E.evaluate_batch(WRONG, xb, yb), _bits(x | y))
    for kernel in ("gates", "reference"):
        assert np.array_equal(E.evaluate_batch(WRONG, xb, yb, kernel=kernel), _bits(x ^ y))
    for d in (XOR, PLAIN):
        assert np.array_equal(E.evaluate_batch(d, xb, yb), _bits(x ^ y))
    args = (xb[0].tolist(), yb[0].tolist())
    assert E.evaluate(WRONG, *args) == tuple(_bits(x[:1] | y[:1])[0])
    assert E.evaluate(WRONG, *args, gates=True) == tuple(_bits(x[:1] ^ y[:1])[0])


def test_admission_is_explicit_and_checked() -> None:
    with pytest.raises(TypeError, match="names its word Definition"):
        A.admit(_boolean("AccelTestNoWord", None, False))
    with pytest.raises(ValueError, match="already admitted as a different Definition"):
        _boolean("AccelTestXorBits", XOR_WORD, True)
    with pytest.raises(TypeError, match="not its word view"):
        A.accelerator(BAD_SIGNATURE)


def _program(inner: CompositeDefinition) -> Program:
    """Root Calls of ``inner``: one call, a batch of three over x's rows, the same inside a composite that isn't admitted, and
    gates that read their outputs."""
    spec = bind(inner)
    outer = CompositeDefinition("AccelTestOuter", 1, (), lambda S: ((("x", WORD), ("y", WORD)), WORD),
                                lambda B, S, x, y: B.batch(BL.NOT, B.call(spec, x, y), axes=(0,)), register=False)

    def body(B, S, xs, y):
        h = B.call(spec, xs[0], y, name="h")
        rows = B.batch(spec, xs, y, axes=(0, None), name="rows")
        o = B.call(bind(outer), h, y, name="o")
        last = B.call(BL.and_gate(2), BL.operands(o[0], rows[2][W - 1]), name="last")
        return _cat(Array(3 * W + 1, BIT), h, rows[1], o, last)

    fn = CompositeDefinition("AccelTestRoot", 1, (), lambda S: ((("xs", Array(3, WORD)), ("y", WORD)), Array(3 * W + 1, BIT)),
                             body, register=False)
    return Program(fn, registered=("y",))


def _inputs(prog: Program, rng: np.random.Generator) -> dict[int, int]:
    out: dict[int, int] = {}
    for pi in range(2):
        refs = prog.input_coll(pi).refs
        out.update((r[2], int(b)) for r, b in zip(refs, rng.integers(0, 2, refs.count)))
    return out


def test_evaluate_circuit_gives_a_programs_committed_values_through_the_accelerator() -> None:
    for inner in (XOR, PLAIN):
        prog = _program(inner)
        c = prog.circuit
        ins = _inputs(prog, np.random.default_rng(2))
        slow = E.evaluate_circuit(c, dict(ins), gates=True)
        fast = E.evaluate_circuit(c, dict(ins))
        assert len(slow) == c.size
        committed = [g for call in c.root_calls() if not call.is_primitive for g in call.returns] + list(c.root_outputs())
        assert all(fast[g] == slow[g] for g in committed) and all(slow[g] == v for g, v in fast.items())
        interior = {g for call in c.root_calls() if not call.is_primitive and call.fn.defn is inner
                    for g in range(call.lo, call.hi)} - set(committed)
        if inner is XOR:
            assert interior and not interior & set(fast) and len(fast) < c.size
        else:
            assert interior <= set(fast) and len(fast) == c.size


def test_evaluate_circuit_follows_the_accelerator_and_gates_true_the_gates() -> None:
    prog = _program(WRONG)
    c = prog.circuit
    ins = _inputs(prog, np.random.default_rng(3))
    fast, slow = E.evaluate_circuit(c, dict(ins)), E.evaluate_circuit(c, dict(ins), gates=True)
    xs = [prog.input_coll(0).refs[j][2] for j in range(3 * W)]
    y = [prog.input_coll(1).refs[j][2] for j in range(W)]
    h = next(call for call in c.root_calls() if call.name == "AccelTestWrongBits")
    assert [fast[g] for g in h.returns] == [ins[a] | ins[b] for a, b in zip(xs[:W], y)]
    assert [slow[g] for g in h.returns] == [ins[a] ^ ins[b] for a, b in zip(xs[:W], y)]


def test_evaluate_circuit_refuses_a_missing_argument() -> None:
    prog = _program(XOR)
    ins = _inputs(prog, np.random.default_rng(4))
    del ins[prog.input_coll(1).refs[0][2]]
    with pytest.raises(E.MissingValue):
        E.evaluate_circuit(prog.circuit, ins)
