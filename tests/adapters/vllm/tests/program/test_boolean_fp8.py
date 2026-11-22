"""`boolean_fp8`: the served FP8 linear on bits (per-token quantisation, the per-tensor cutlass GEMM, the block-scaled linear and
its shared-scale restatement), version 2 of each `fp8` composite (the dot's version 3), is Boolean, agrees with its v1 word view on
random words and on words that reach every guard (signed zeros, subnormals, infinities and NaNs in the rows, NaN e4m3 bytes,
degenerate scales), and is pinned by gate count and program digest."""
from __future__ import annotations

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.batch import type_shape
from verity.circuits.evaluation.reference import standalone_call
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.ops.boolean import versions as SW

from verity_vllm.definitions.registry import boolean_fp8 as BF

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Fp8RowScale_v2": (2831, 6165, 354, "bb5580c2a3d114cdee739944ebbec65bcf0ee39ed97dbf2120c4d4035fe8f8016a2c9606a2323e1aadfbf24fe7b87022a5f6e313ba7dc7d41d939636eac073eb"),
    "DotE4m3_v3{K=64}": (13598, 24402, 2036, "6d444e15b40cf9d94cf420e77a2b168c44758318f73cf0095dd882d2c79d1b16dfa7e50264718c109f327a21caf213c3b16f80721931d291bc44ce2837ab30fc"),
    "ScaledMmFp8Coordinate_v2{K=64,BIAS=True}":
        (20579, 37013, 2428, "ba91db3f8d0bb37e4441010ec3afeb1d1d87cc6f7add9c367c9f57842c1b7cdbca142a3786f0127879cb66c8449226bcc29d235ac086e289fe352ce89a5a74f6"),
    "Fp8ActQuant_v2{K=256}": (1027541, 2120408, 177916, "b4e25eb15eca3f767fcd07c9056dfaaecc3df0f9964dc03282f62efeff55a9a9518e03a34217af883a1fd813643d157d7cf38ed04b8419fdbdec214624a7a3f1"),
    "QuantLinearFp8_v2{K=256,N=2,BIAS=True}":
        (1150287, 2340846, 194988, "cbe3e82ca8c81983317f524cd46fdc55f44c7787c82efa32369401b9f70618d73ad7e6c9ccb238f33707fe49c8e049b2dd4cc3e820e5d5df892ed2f0b5f90eb0"),
    "Fp8GroupQuant_v2{K=256,G=128}": (949674, 1979312, 152056, "63d9a48241c8052541387040cea460c6c84c83a34ba1cbadb62999a20008c9f1d68cba77e14ce04c3680b01d51ac892422fd2f7cef66968bebacb57d3a394949"),
    "ScaledMmFp8BlockCoordinate_v2{K=256,G=128}":
        (68288, 122732, 8913, "1faf983c0f62b2f4f5475c87b2c45dcbaedb0fca2760ed1aad6204c48637e8ed1b7221f645f5c237f281d4d3dc18526ea59755d4d2fc89f7d6af3b6ac1cb3386"),
    "ScaledMmFp8Block_v2{K=256,N=128,G=128}":
        (8740864, 15709696, 1140864, "70ed645ad2510c8977763bf37f375773ec8f2468fba40c7e0be29887a691e54f2bbd4265564c9b97bd0230acde9f3e271ce488cb5c25f9d0ebfe73392f5e80fb"),
    "ScaledMmFp8BlockSharedScale_v2{K=256,N=128,G=128}":
        (8123644, 14528596, 1112924, "535906c729b2b412e8dca37efd20bd39059d4483834b4f55277eb8d65a8cc5a92f01490e4982e9a33dc8afa5dfcb1f5658a3ef30c988a5fea2cc5ba14f2e8397"),
    "DotE4m3_v4{K=64,DOT=BlackwellE4m3QmmaDot32_v2}":
        (18256, 34804, 2050, "a501faf1012e80ecc7fdc0fcec9a35c003bcdf025febca278ea41f1cbb2d27f4e20bbc84c4307fb3a2c9ccd0ede9bb51730cf8d2602380fcb643337ef16c1770"),
    "ScaledMmFp8NoSwap_v2{K=64,N=2,BIAS=True,DOT=BlackwellE4m3QmmaDot32_v2}":
        (50474, 94830, 4884, "8d3d12646f47bad7eac13814c4f58e8ab3c59b3674cb830ba19b947f756295f1d4bd80cb19a2109658ab108c7ea1709de25d341f45ce7cd34271b0ecd69fa089"),
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
SLOW_AGREEMENT = ("QuantLinearFp8_v2{K=256,N=2,BIAS=True}", "Fp8ActQuant_v2{K=256}")

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


@pytest.mark.parametrize("target", [pytest.param(t, marks=pytest.mark.slow) if t.id in SLOW_AGREEMENT else t for t in AGREEMENT], ids=lambda t: t.id)
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
