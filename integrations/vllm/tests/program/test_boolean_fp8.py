"""`boolean_fp8`: the served FP8 linear on bits (per-token quantisation, the per-tensor cutlass GEMM, the block-scaled linear and
its shared-scale restatement), version 2 of each `fp8` composite, is Boolean, agrees with its v1 word view on random words and on
words that reach every guard (signed zeros, subnormals, infinities and NaNs in the rows, NaN e4m3 bytes, degenerate scales), and is
pinned by gate count and program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.batch import type_shape
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import bind
from verity.ml.boolean import trace as TR
from verity_vllm.program.registry import boolean_fp8 as BF

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Fp8RowScale_v2": (2831, 6165, 354, "43dac64968717972"),
    "DotE4m3_v2{K=64}": (13598, 24402, 2036, "e85532ee662f5fd5"),
    "ScaledMmFp8Coordinate_v2{K=64,BIAS=True}": (20579, 37013, 2428, "acb4ef720f53ed39"),
    "Fp8ActQuant_v2{K=256}": (1027541, 2120408, 177916, "a7a3e66509772aa4"),
    "QuantLinearFp8_v2{K=256,N=2,BIAS=True}": (1150287, 2340846, 194988, "f21adbddaa5b877d"),
    "Fp8GroupQuant_v2{K=256,G=128}": (949674, 1979312, 152056, "92c13a1e3eb02c46"),
    "ScaledMmFp8BlockCoordinate_v2{K=256,G=128}": (68288, 122732, 8913, "bba9b1f6c06f80f6"),
    "ScaledMmFp8Block_v2{K=256,N=128,G=128}": (8740864, 15709696, 1140864, "f045320bcfd0aea5"),
    "ScaledMmFp8BlockSharedScale_v2{K=256,N=128,G=128}": (8123644, 14528596, 1112924, "92e8e7e2bc4cc9ba"),
}

PINNED = [bind(BF.Fp8RowScaleV2), bind(BF.DotE4m3V2, K=64), bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=True),
          bind(BF.Fp8ActQuantV2, K=256), bind(BF.QuantLinearFp8V2, K=256, N=2, BIAS=True), bind(BF.Fp8GroupQuantV2, K=256, G=128),
          bind(BF.ScaledMmFp8BlockCoordinateV2, K=256, G=128), bind(BF.ScaledMmFp8BlockV2, K=256, N=128, G=128),
          bind(BF.ScaledMmFp8BlockSharedScaleV2, K=256, N=128, G=128)]

#: every composite once (the quantisers at the kernel's smallest row, K = 256; the block-scaled GEMMs at G = 32, two k tiles and
#: two weight blocks), with and without bias
AGREEMENT = [bind(BF.Fp8RowScaleV2), bind(BF.Fp8RowAbsMaxV2, K=256), bind(BF.Fp8ActQuantV2, K=256), bind(BF.DotE4m3V2, K=64),
             bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=True), bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=False),
             bind(BF.ScaledMmFp8V2, K=64, N=3, BIAS=True), bind(BF.ScaledMmFp8V2, K=32, N=2, BIAS=False),
             bind(BF.QuantLinearFp8V2, K=256, N=2, BIAS=True), bind(BF.Fp8GroupScaleV2, G=32),
             bind(BF.Fp8GroupQuantV2, K=64, G=32), bind(BF.ScaledMmFp8BlockCoordinateV2, K=64, G=32),
             bind(BF.ScaledMmFp8BlockV2, K=64, N=64, G=32), bind(BF.ScaledMmFp8BlockCoordinateGivenScaleV2, K=64, G=32),
             bind(BF.ScaledMmFp8BlockSharedScaleTileV2, K=64, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=64, N=64, G=32)]

#: bf16 words that reach the quantiser's guards: signed zeros, subnormals, the normal range's ends, infinities, NaNs
BF16_SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1], np.uint16)
#: f32 scales at the guards: zeros, the smallest subnormal and normal, the minimum scale, the largest finite, infinity, NaNs
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x00000001, 0x00800000, 0x3A124925, 0x7F7FFFFF, 0x7F800000, 0x7FC00000,
                         0xFFC00001, 0xBF800000], np.uint32)


def _inputs(word, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances per parameter of ``word``: the first half random (gaussian bf16 rows at a few scales, uniform finite e4m3
    bytes, positive scales near the served ones), the second with special words planted at random positions (the NaN e4m3 bytes
    among them)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    out = []
    for name, t in word.signature()[0]:
        shape, width = (n, *type_shape(t)), max(t.width_set())
        if width == 16:
            s = np.exp2(rng.integers(-8, 12, (n,) + (1,) * (len(shape) - 1))).astype(np.float32)
            a = ((rng.standard_normal(shape).astype(np.float32) * s).view(np.uint32) >> 16).astype(np.uint16)
            specials, p = BF16_SPECIALS, 0.05
        elif width == 8:
            a = rng.integers(0, 256, shape, dtype=np.uint64).astype(np.uint8)
            a[(a & 0x7F) == 0x7F] ^= 1
            specials, p = np.array([0x7F, 0xFF, 0x00, 0x80, 0x7E, 0xFE, 0x01], np.uint8), 0.03
        else:
            a = (np.abs(rng.standard_normal(shape)).astype(np.float32) * np.float32(1e-2 if name != "absmax" else 30.0)).view(np.uint32)
            specials, p = F32_SPECIALS, 0.2
        mid = a[half:]
        m = rng.random(mid.shape) < p
        mid[m] = rng.choice(specials, int(m.sum()))
        out.append(a)
    return out


def _flat(r) -> list[np.ndarray]:
    return [np.asarray(x).astype(np.uint64).reshape(len(np.asarray(x)), -1) for x in (r if isinstance(r, (tuple, list)) else [r])]


@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)).startswith(digest)


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, target.bindings)
    n = 64 if BS.compile(target).counts["and"] > 200_000 else 400
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
