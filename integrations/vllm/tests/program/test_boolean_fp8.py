"""`boolean_fp8`: the served FP8 linear on bits (per-token quantisation, the per-tensor cutlass GEMM, the block-scaled linear and
its shared-scale restatement), version 2 of each `fp8` composite (the dot's version 3), is Boolean, agrees with its v1 word view on
random words and on words that reach every guard (signed zeros, subnormals, infinities and NaNs in the rows, NaN e4m3 bytes,
degenerate scales), and is pinned by gate count and program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity.primitives.circuits import evaluation as E
from verity.primitives.circuits.evaluation import bits as BS
from verity.primitives.circuits.evaluation.batch import type_shape
from verity.primitives.circuits.evaluation.reference import standalone_call
from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits import codec
from verity.primitives.circuits.defs import bind
from verity_catalog.definitions.boolean import gemm as BG
from verity.primitives.circuits.boolean import trace as TR
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_fp8 as BF

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Fp8RowScale_v2": (2831, 6165, 354, "43dac6496871797298a15ce142f740e490e26a97cbb064fc81ad3bf0c65272be"),
    "DotE4m3_v3{K=64}": (13598, 24402, 2036, "7a251da32da5f6a9a8b6017b12e942535ffafe3e9d651e402db64f88f1a15be8"),
    "ScaledMmFp8Coordinate_v2{K=64,BIAS=True}":
        (20579, 37013, 2428, "99f4d987af96906cf3edcc11e821e9f8ea9255440f2c5cb73c778d3cbe39c73b"),
    "Fp8ActQuant_v2{K=256}": (1027541, 2120408, 177916, "a7a3e66509772aa45d373d5ffa08244d8191f6d91455e23e5fa29644ec79b49b"),
    "QuantLinearFp8_v2{K=256,N=2,BIAS=True}":
        (1150287, 2340846, 194988, "d5a207fbf4e703b6cf1db5798c3bac977b65702ca8a792eeeb757d3b8db8fae2"),
    "Fp8GroupQuant_v2{K=256,G=128}": (949674, 1979312, 152056, "92c13a1e3eb02c463da0ccb9ada8058908d621469de4834c56083148768d437b"),
    "ScaledMmFp8BlockCoordinate_v2{K=256,G=128}":
        (68288, 122732, 8913, "bba9b1f6c06f80f604a30ce4270070bfc40bfe51f8f355232bc1f37239ef5b4c"),
    "ScaledMmFp8Block_v2{K=256,N=128,G=128}":
        (8740864, 15709696, 1140864, "f045320bcfd0aea545792a3d334b482a360eb340d81539159f9264edbb64c92b"),
    "ScaledMmFp8BlockSharedScale_v2{K=256,N=128,G=128}":
        (8123644, 14528596, 1112924, "92e8e7e2bc4cc9ba78d6993f67b2eef88277f8106d097bb3147ab7ed5ea41b90"),
    "DotE4m3_v4{K=64,DOT=BlackwellE4m3QmmaDot32_v2}":
        (18256, 34804, 2050, "6e0d4f336ec3e4c04b1860d74083d883c4f64862b3bea5295a907cc124672b83"),
    "ScaledMmFp8NoSwap_v2{K=64,N=2,BIAS=True,DOT=BlackwellE4m3QmmaDot32_v2}":
        (50474, 94830, 4884, "7f57b9e8cd9f74b25f8d46bbe30974ae1b1dc50cabd0bfe3c2774c49a8828f38"),
}
#: the cc 12.0 (sm_120 mma.sync) and Hopper (wgmma) e4m3 k32 steps on bits, the no-swap linear's DOT
SM120, HOPPER = BG.BlackwellE4m3QmmaDot32, BG.HopperE4m3QgmmaDot32

PINNED = [bind(BF.Fp8RowScaleV2), bind(BF.DotE4m3V3, K=64), bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=True),
          bind(BF.Fp8ActQuantV2, K=256), bind(BF.QuantLinearFp8V2, K=256, N=2, BIAS=True), bind(BF.Fp8GroupQuantV2, K=256, G=128),
          bind(BF.ScaledMmFp8BlockCoordinateV2, K=256, G=128), bind(BF.ScaledMmFp8BlockV2, K=256, N=128, G=128),
          bind(BF.ScaledMmFp8BlockSharedScaleV2, K=256, N=128, G=128), bind(BF.DotE4m3V4, K=64, DOT=SM120),
          bind(BF.ScaledMmFp8NoSwapV2, K=64, N=2, BIAS=True, DOT=SM120)]

#: every composite once (the quantisers at the kernel's smallest row, K = 256; the block-scaled GEMMs at G = 32, two k tiles and
#: two weight blocks), with and without bias; the no-swap linear on both steps
AGREEMENT = [bind(BF.Fp8RowScaleV2), bind(BF.Fp8RowAbsMaxV2, K=256), bind(BF.Fp8ActQuantV2, K=256), bind(BF.DotE4m3V3, K=64),
             bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=True), bind(BF.ScaledMmFp8CoordinateV2, K=64, BIAS=False),
             bind(BF.ScaledMmFp8V2, K=64, N=3, BIAS=True), bind(BF.ScaledMmFp8V2, K=32, N=2, BIAS=False),
             bind(BF.QuantLinearFp8V2, K=256, N=2, BIAS=True), bind(BF.Fp8GroupScaleV2, G=32),
             bind(BF.Fp8GroupQuantV2, K=64, G=32), bind(BF.ScaledMmFp8BlockCoordinateV2, K=64, G=32),
             bind(BF.ScaledMmFp8BlockV2, K=64, N=64, G=32), bind(BF.ScaledMmFp8BlockCoordinateGivenScaleV2, K=64, G=32),
             bind(BF.ScaledMmFp8BlockSharedScaleTileV2, K=64, G=32), bind(BF.ScaledMmFp8BlockSharedScaleV2, K=64, N=64, G=32),
             bind(BF.DotE4m3V4, K=64, DOT=SM120), bind(BF.DotE4m3V4, K=64, DOT=HOPPER),
             bind(BF.ScaledMmFp8NoSwapCoordinateV2, K=64, BIAS=True, DOT=SM120),
             bind(BF.ScaledMmFp8NoSwapCoordinateV2, K=64, BIAS=False, DOT=HOPPER),
             bind(BF.ScaledMmFp8NoSwapV2, K=64, N=3, BIAS=True, DOT=SM120)]

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
    assert codec.program_digest(codec.encode_program(prog)) == digest


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k == "DOT" else x) for k, x in target.bindings.items()}
    want = 2 if (target.defn.name, target.defn.version) == ("DotE4m3", 4) else 1
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, want, lifted)
    n = 64 if BS.compile(target).counts["and"] > 200_000 else 400
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    got, ref = _flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_the_switch_maps_its_word_view_to_it(target) -> None:
    """`boolean_version` of each `fp8` word composite is the composite on bits whose word view it is (a static-free one too)."""
    assert SW.boolean_version(TR.word_of(target)).id == target.id
