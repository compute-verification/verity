"""`verity.ml.circuits.ops.boolean.cast` and `verity.ml.circuits.boolean.gather`: the Boolean bf16 widening agrees with `Bf16ToF32_v1` on every bf16
and is pinned by gate count and program digest; the f32 -> bf16 rounding (`gemm.F2fpBf16_v2`) agrees with `F2fpBf16_v1` on every
f32 word of its special chunks, and on all 2^32 when ``VERITY_ML_BOOLEAN_EXHAUSTIVE`` is on, and `F32ToBf16Rn_v2` agrees with
`F32ToBf16Rn_v1` on those chunks; the multiplexer tree selects
``words[idx]``, else its fill, at every index of its width and costs the ANDs its docstring states; and a Boolean Program's
prescribed inputs are `Input1_v1` leaves, so the word Programs' `Input16_v1` and `Input32_v1` need no Boolean version."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity import settings
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit, const_bits
from verity.ml.circuits.boolean.gather import gather

WIDEN = bind(BC.Bf16ToF32)
ONE_CALL = CompositeDefinition("TestOneCall", 1, ("FN",), lambda S: S.FN.signature(), lambda B, S, *args: B.call(S.FN, *args),
                               register=False)
#: (And, Xor, Not gates, standalone program digest)
PIN = (0, 0, 0, "4455f054cb3eecec9e611f77120ac07d4cca117bdf9ad07af18df7f0ea6ac01b0dbd68b9e3fa5d97e44b4c190470c5e0c61d9c455990627f5362e3704c21672a")


def test_the_widening_is_pinned() -> None:
    ands, xors, nots, digest = PIN
    assert BL.is_boolean(WIDEN)
    assert BS.compile(WIDEN).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(WIDEN)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


def test_the_widening_agrees_with_bf16_to_f32_v1_on_every_bf16() -> None:
    assert TR.word_of(WIDEN) is P.Bf16ToF32
    a = np.arange(1 << 16, dtype=np.uint16)
    got = BS.evaluate_words(WIDEN, a).astype(np.uint64)
    assert np.array_equal(got, E.evaluate_batch(P.Bf16ToF32, a).astype(np.uint64))
    assert np.array_equal(got, a.astype(np.uint64) << np.uint64(16))


def _every_f32_chunks(chunks) -> None:
    """`F2fpBf16_v2` against `F2fpBf16_v1` on every f32 word whose high 12 bits are one of ``chunks`` (2^20 words each)."""
    f2fp = bind(BG.F2fpBf16)
    low = np.arange(1 << 20, dtype=np.uint32)
    for c in chunks:
        words = np.uint32(c << 20) | low
        got = BS.evaluate_words(f2fp, words).astype(np.uint64)
        assert np.array_equal(got, E.evaluate_batch(P.F2fpBf16, words).astype(np.uint64)), f"chunk {c:#05x}"


def test_the_rounding_agrees_with_f2fp_v1_on_every_word_of_its_special_chunks() -> None:
    """±0 and the small subnormals, 1.0's neighbourhood, the largest finite words (which round to ±inf), ±inf and the NaNs whose
    payload is only in the low bits, and the quiet NaNs."""
    _every_f32_chunks([0x000, 0x3F8, 0x7F7, 0x7F8, 0x7FF, 0x800, 0xFF7, 0xFF8])


def test_the_software_rounding_agrees_with_f32_to_bf16_rn_v1() -> None:
    """`F32ToBf16Rn_v2` (a NaN to the software NaN, 0x7FC0) on every word of the special chunks and on uniform words."""
    target = bind(BC.F32ToBf16Rn)
    assert TR.word_of(target) is P.F32ToBf16Rn
    low = np.arange(1 << 20, dtype=np.uint32)
    rng = np.random.default_rng(5)
    words = np.concatenate([np.uint32(c << 20) | low for c in (0x000, 0x3F8, 0x7F7, 0x7F8, 0x7FF, 0x800, 0xFF8)]
                           + [rng.integers(0, 1 << 32, 200_000, dtype=np.uint64).astype(np.uint32)])
    got = BS.evaluate_words(target, words).astype(np.uint64)
    assert np.array_equal(got, E.evaluate_batch(P.F32ToBf16Rn, words).astype(np.uint64))


@pytest.mark.skipif(not settings.VERITY_ML_BOOLEAN_EXHAUSTIVE,
                    reason="every f32 word (2^32 instances, about 20 minutes); set VERITY_ML_BOOLEAN_EXHAUSTIVE=1")
def test_the_rounding_agrees_with_f2fp_v1_on_every_f32_word() -> None:
    _every_f32_chunks(range(1 << 12))


def test_boolean_programs_take_their_inputs_as_input1_leaves() -> None:
    """A Program prescribes one `Input<w>_v1` per leaf of its function's parameters, w the leaf's width: the word cast's Programs
    read `Input16_v1` and `Input32_v1`, their Boolean versions' read `Input1_v1` only, which `is_boolean` admits."""
    def inputs(target) -> dict[str, int]:
        prog = Program(bind(ONE_CALL, FN=target))
        out: dict[str, int] = {}
        for n in prog.root.body.nodes:
            if is_input(n.fn):
                out[n.fn.id] = out.get(n.fn.id, 0) + n.gates
        return out

    assert inputs(P.Bf16ToF32) == {"Input16_v1": 1}
    assert inputs(P.F2fpBf16) == {"Input32_v1": 1}
    assert inputs(WIDEN) == {"Input1_v1": 16}
    assert inputs(bind(BG.F2fpBf16)) == {"Input1_v1": 32}


def _gather_definition(n: int, k: int, width: int, fill: int):
    params = (("idx", Array(k, BL.BIT)), ("words", Array(n, Array(width, BL.BIT))))
    return TR.definition(f"TestGather{n}u{k}w{width}", 1, lambda C, idx, *words: gather(C, idx, list(words), fill), params,
                         Array(width, BL.BIT), register=False)


@pytest.mark.parametrize(("n", "k"), [(1, 3), (2, 1), (3, 2), (5, 3), (5, 5), (8, 3), (13, 4), (16, 6), (33, 6)])
def test_the_multiplexer_tree_selects_the_indexed_word_else_its_fill(n: int, k: int) -> None:
    width, fill = 7, 0x55
    d = bind(_gather_definition(n, k, width, fill))
    assert BL.is_boolean(d)
    rng = np.random.default_rng(n * 64 + k)
    idx = np.repeat(np.arange(1 << k, dtype=np.uint64), 3)
    words = rng.integers(0, 1 << width, (len(idx), n), dtype=np.uint64)
    got = BS.evaluate_words(d, idx, words).astype(np.uint64)
    ref = np.where(idx < n, words[np.arange(len(idx)), np.minimum(idx, n - 1).astype(np.int64)], fill)
    assert np.array_equal(got, ref)
    padded, odd = n, 0
    for _ in range(k):
        odd += padded % 2
        padded = (padded + 1) // 2
    assert BS.compile(d).counts["and"] == (n - 1 + odd) * width


def test_the_multiplexer_tree_rejects_what_it_cannot_select() -> None:
    C = Circuit(cse=True)
    with pytest.raises(ValueError, match="at least one word"):
        gather(C, [], [], 0)
    with pytest.raises(ValueError, match="one width"):
        gather(C, [], [const_bits(1, 4), const_bits(1, 5)], 0)
    with pytest.raises(ValueError, match="more than 1 index bits"):
        gather(C, [const_bits(1, 1)[0]], [const_bits(i, 4) for i in range(3)], 0)
