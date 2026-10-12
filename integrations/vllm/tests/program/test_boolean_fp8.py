"""`boolean_fp8`: the served FP8 linear on bits (per-token quantisation, the per-tensor cutlass GEMM, the block-scaled linear and
its shared-scale restatement), version 2 of each `fp8` composite (the dot's version 3), is Boolean, agrees with its v1 word view on
random words and on words that reach every guard (signed zeros, subnormals, infinities and NaNs in the rows, NaN e4m3 bytes,
degenerate scales), and is pinned by gate count and program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean import trace as TR
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_fp8 as BF

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Fp8RowScale_v2": (2831, 6165, 354, "5756f5168724db73eee88d038ad7f83126995ededff4876db6f3ba7104617239414f5c1c044e23927a80443e329d7ff3a0b3cc4bd585c9d6ad5e47a3940a60c0"),
    "DotE4m3_v3{K=64}": (13598, 24402, 2036, "f7aa9b0f89adc7c25403cf484375bb532ad7c627ceb2f1e867996d3ef46684e4e6bd7e385c26a0705eef57ba3df7b596291dc91b7afdda2629399ab720a20579"),
    "ScaledMmFp8Coordinate_v2{K=64,BIAS=True}":
        (20579, 37013, 2428, "b64fe6336c9f0a74948bcecce04708b014eaa5f92b855e4994d24fe7dbf06d0a5fd356cd2276823bde7713839f3a6cb10801d497358f5effc975422d749e413e"),
    "Fp8ActQuant_v2{K=256}": (1027541, 2120408, 177916, "cb38226e0b5517b40fdd0d0d5c07ede0452bc1a7e35677f2af387b9c734c98c2e1bdeb657633258b69231fcdce86e43bf0cf5a75849c966f0ec7fd2999e0c154"),
    "QuantLinearFp8_v2{K=256,N=2,BIAS=True}":
        (1150287, 2340846, 194988, "c31762aa8015c4648a7a3dad8280e17845f04e1eed054f5183fb637c0f70621da4152bd007acb01fe119a090c1ad3288c2a1544dace8e5fd923fa63ec0ed2ad7"),
    "Fp8GroupQuant_v2{K=256,G=128}": (949674, 1979312, 152056, "1cfa22f8d07c56c1438b833f5bc9c108a057ac401488d01ce126a946d494255721fd152d763a40b939f859f50f3d45083d2edfb0e05df54dbff8ad1e7615ba99"),
    "ScaledMmFp8BlockCoordinate_v2{K=256,G=128}":
        (68288, 122732, 8913, "acc3811b8c179356da7bdb38eb4dba09c9bd19f636397a1f1ff4e3f48c8fa888d0b0f623d43a3aa478ff74488ae04cfdca1acd8ef897f33668d2e92b7e07f495"),
    "ScaledMmFp8Block_v2{K=256,N=128,G=128}":
        (8740864, 15709696, 1140864, "d996bf8b62e7371e246ec3d99ed95f721fa1326c8a768f0034fa8e9994dc8210a9ce89802a0be4f03a40dc57a9a27a44fad611383b6cf6598340a90e67d251d9"),
    "ScaledMmFp8BlockSharedScale_v2{K=256,N=128,G=128}":
        (8123644, 14528596, 1112924, "a108dfadcbed0ab0263070ee6b31749420e6a7cbe347243960f844ddc0c3145551d9517d4056932d9a01e03ea25bf163b9beca75bb723996bb5722b66e92e08f"),
    "DotE4m3_v4{K=64,DOT=BlackwellE4m3QmmaDot32_v2}":
        (18256, 34804, 2050, "392993a9c78be64299f8a42d958d49f685e44715c8b94fb62ca15afcd205fca05888213b53916fadca3c935ab2d2a33469a22f05e905d1dc0e08b7a1f722b43d"),
    "ScaledMmFp8NoSwap_v2{K=64,N=2,BIAS=True,DOT=BlackwellE4m3QmmaDot32_v2}":
        (50474, 94830, 4884, "a827e1f5b0a81ac753963f667d592d2c970cda9d1a909b96e3a30ff81d7d85f19f6d628c158567d9bfb976f12170f83d4da6f540e5e7772d5b3e34e00d9fcaa8"),
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
