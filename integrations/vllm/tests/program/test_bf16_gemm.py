"""`registry.cublas_bf16` and `registry.ll_bf16`, GLM-5.3's bf16 GEMMs on sm_90: on bits they are Boolean and are the Boolean
versions of their words; they agree with their words (the reference evaluator) and with `bf16_gemm_check`'s numpy models of the
kernels on random words and on words at every class edge (signed zeros, subnormals, infinities, NaNs, and products that overflow
or underflow f32, the only inputs on which the dotprod's contraction changes a word); the dotprod's steps agree with the f32 words
they widen into, on accumulators that cancel the product too; `BatchedGemm_v2` is `mla.MlaBmm_v1`'s
function; the words at GLM-5.3's gate statics (K 6144) agree with the models; `dispatch_sm90` is vLLM's table; and each is pinned
by gate count and program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import evaluate, standalone_call
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity_vllm.program.boolean import boolean_version
from verity_vllm.program.registry import bf16_gemm_check as C
from verity_vllm.program.registry import cublas_bf16 as CB
from verity_vllm.program.registry import ll_bf16 as LL
from verity_vllm.program.registry import mla
from verity_vllm.program.registry import mla_check as MK

U16, U32 = np.uint16, np.uint32
HOP = BG.HopperBF16WgmmaDot16
HOPW = word_of(HOP)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "LlBf16FmaStep_v1":
        (2167, 3453, 201, "f6ded7743772c6c349b96e32f829ff932af4750b8774e1bb40bc75d68dcaa3b57820d91381f9d6229c2ff0657984fea9d91b074be401c67ef8233fc2c46bd084"),
    "LlBf16MulAddStep_v1":
        (1561, 2492, 191, "b14dad780cce876d2ab12111a04ac0d3dfc337a95890c7c91ac1ff57c225eb0acb7070c7445ebfc376cf84ef1c47a0f8d5e9a7a9f8b8d8ef6b8c0d3a57d80e62"),
    "GemmF32Out_v2{K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (47232, 90294, 4566, "3f63d9f7123b765e7ce6a1d981accb6b9bf30ddd8857db8c619cf9da3d3b461536d3ad9c6b0e0746417cb0853e5a4bde0c29cdbf3d6428e1d4adbf899040628b"),
    "BatchedGemm_v2{B=2,K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (94860, 181176, 9222, "d1b876b03912d35319d3a8d818e86459dd37556f384e309a370e5cc1ff4c78b6eb17e6ef9fae3a98dca67af35d2616b69d93f55be5e51ba24777fee38f772ee4"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=1}":
        (606720, 973248, 58816, "a7b83ce34457940e8ef318a1a9834c7d451004d9cf6752071277cdf704b83d6f70471a4284e427fe6ef2d835c250b6f03490e21f14dd48d682da82067d5468ff"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=0}":
        (451584, 727232, 56256, "4513f97fa91da6198b5a9a3b6cf112723c6f5718ca3fcb9fcfe06ee6c474df22f97cb2f7ac231d58d8234768a15cb46bfb8649c8eefaa78230b6d1893ed2658a"),
    "LlBf16SplitKDot_v2{K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (67848, 128762, 6778, "438a7d3e324cfcb4d025466be392931c8872377c9019ee676a5331eb5b1212a59b859aec141b190fdd0ef9aa20b099fcb26685cf5c624bc54809978696ae59eb"),
    "LlBf16DotprodGemm_v2{N=2,K=128,BS=64,VEC=2,FMA=1}":
        (658688, 1062528, 66176, "5012d48a02849357a616fce904f2d516a20376c184e77089674ec5dc66b83b75ea249758b87b4664e00928ba4e7969f95ee3891b44267b7dbfadee18e560428c"),
    "LlBf16SplitKGemm_v2{N=2,K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (135696, 257524, 13556, "aec7c44c327014f6ffd016196643cdc5618d4daf01e1b83b689234b67c61e45c4d2ce171218a4f54620f20b17e29ade587f157bea43d41fbac7f2e173e9feffb"),
}
#: the dotprod's steps, element Definitions with no word of their own
STEPS = [bind(LL.LlBf16FmaStep), bind(LL.LlBf16MulAddStep)]
PINNED = [bind(CB.GemmF32Out, K=32, N=3, DOT=HOP), bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
          bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=1), bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=0),
          bind(LL.LlBf16SplitKDot, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP),
          bind(LL.LlBf16DotprodGemm, N=2, K=128, BS=64, VEC=2, FMA=1),
          bind(LL.LlBf16SplitKGemm, N=2, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP)]

#: bf16 words at every class edge: signed zeros, subnormals, the normal range's ends, 1, infinities and NaNs
BF16_SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x5F00, 0xDF00,
                          0x3F80, 0xBF80], U16)
KINDS = ("spread", "special", "wide", "tiny")


def _bf16_words(rng, shape, kind: str) -> np.ndarray:
    """``spread``: normal values; ``special``: those with 2% specials; ``wide``: finite words of any exponent, so products
    overflow f32 or fall below its normal range; ``tiny``: values near 2^-64, whose products are f32 subnormals."""
    n = int(np.prod(shape))
    if kind == "wide":
        x = (rng.integers(0, 2, n) << 15) | (rng.integers(1, 255, n) << 7) | rng.integers(0, 128, n)
        return x.astype(U16).reshape(shape)
    scale = {"spread": 4.0, "special": 4.0, "tiny": 2.0 ** -64}[kind]
    x = (MK.u32(rng.standard_normal(n).astype(np.float32) * np.float32(scale)) >> 16).astype(U16)
    if kind == "special":
        m = rng.random(n) < 0.02
        x[m] = BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), int(m.sum()))]
    return x.reshape(shape)


def _f32_words(rng, shape, kind: str) -> np.ndarray:
    """f32 words of a kind: a bf16 word of it with random low bits, except on infinities and NaNs, which keep theirs."""
    hi = _bf16_words(rng, shape, kind).astype(U32)
    lo = rng.integers(0, 1 << 16, hi.shape).astype(U32)
    return (hi << 16) | np.where((hi >> 7) & 0xFF == 0xFF, 0, lo).astype(U32)


def _rows(rng, M: int, N: int, K: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(x (M, K), w (N, K)) of each kind."""
    return [(_bf16_words(rng, (M, K), k), _bf16_words(rng, (N, K), k)) for k in KINDS]


def _bits_rows(target, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """A token-row Definition (x: Array<K>, w: Array<N, Array<K>>) on every row of x against w: f32 words (M, N)."""
    return np.asarray(BS.evaluate_words(target, x, np.broadcast_to(w, (len(x), *w.shape)))).astype(U32)


def _f32_equal(got, want) -> bool:
    return bool(np.all(MK._words_equal(np.asarray(got, U32), np.asarray(want, U32), f32_words=True)))


def test_every_definition_is_boolean_and_the_boolean_version_of_its_word() -> None:
    for t in PINNED:
        assert BL.is_boolean(t), t.id
        assert boolean_version(word_of(t)).id == t.id
    assert all(BL.is_boolean(t) for t in STEPS)


@pytest.mark.parametrize("target", PINNED, ids=[t.id for t in PINNED])
def test_bits_agree_with_the_word(target) -> None:
    """Each Boolean Definition against its word (the reference evaluator) on eight instances of each kind, word for word."""
    rng = np.random.default_rng(1)
    word = word_of(target)
    for kind in KINDS:
        args = [_bf16_words(rng, (8, *type_shape(BL.word_type(p))), kind) for p in BS.compile(target).params]
        got = np.asarray(BS.evaluate_words(target, *args)).reshape(8, -1)
        want = np.array([evaluate(word, *(a[i].reshape(-1).astype(np.int64).tolist() for a in args)) for i in range(8)])
        assert np.array_equal(got.astype(np.int64), want.astype(np.int64)), (target.id, kind)


#: each step's words: fma(f32(x), f32(w), acc), and acc + f32(x) * f32(w)
STEP_WORDS = {"LlBf16FmaStep_v1": lambda x, w, acc: W.F32Fma.evaluate(P.Bf16ToF32.evaluate(x), P.Bf16ToF32.evaluate(w), acc),
              "LlBf16MulAddStep_v1": lambda x, w, acc: W.F32Add.evaluate(acc, W.F32Mul.evaluate(P.Bf16ToF32.evaluate(x),
                                                                                                P.Bf16ToF32.evaluate(w)))}


@pytest.mark.parametrize("target", STEPS, ids=[t.id for t in STEPS])
def test_steps_agree_with_their_words(target) -> None:
    """The dotprod's steps on bits against `F32Fma_v1`, and `F32Mul_v1` then `F32Add_v1`, on the widened words: 256 instances of
    each kind, and accumulators that cancel the product exactly (a bf16 product is exact in f32), where the sum's zero takes its
    sign from the rounding mode."""
    rng = np.random.default_rng(8)
    word = STEP_WORDS[target.id]
    cases = [(_bf16_words(rng, (256,), k), _bf16_words(rng, (256,), k), _f32_words(rng, (256,), k)) for k in KINDS]
    x, w = _bf16_words(rng, (256,), "spread"), _bf16_words(rng, (256,), "spread")
    product = (x.astype(U32) << 16).view(np.float32) * (w.astype(U32) << 16).view(np.float32)
    cases.append((x, w, MK.u32(-product)))
    for x, w, acc in cases:
        got = np.asarray(BS.evaluate_words(target, x, w, acc)).astype(np.int64)
        want = [word(int(a), int(b), int(c)) for a, b, c in zip(x, w, acc)]
        assert np.array_equal(got, np.array(want, np.int64)), target.id


@pytest.mark.parametrize("FMA", [1, 0])
def test_dotprod_agrees_with_the_model(FMA: int) -> None:
    """`LlBf16DotprodGemm_v2` against `dotprod_rows` over two tiles of two warps' lanes, and over one tile of one warp; on the wide
    words the contracted and uncontracted models differ, and each Definition is its own."""
    rng = np.random.default_rng(2)
    for K, BS_, VEC in ((128, 64, 1), (64, 32, 2)):
        target = bind(LL.LlBf16DotprodGemm, N=3, K=K, BS=BS_, VEC=VEC, FMA=FMA)
        for x, w in _rows(rng, 24, 3, K):
            assert _f32_equal(_bits_rows(target, x, w), C.dotprod_rows(x, w, BS_, VEC, FMA))
    x, w = _bf16_words(rng, (256, 64), "wide"), _bf16_words(rng, (3, 64), "wide")
    fused, split = C.dotprod_rows(x, w, 32, 2, 1), C.dotprod_rows(x, w, 32, 2, 0)
    assert not _f32_equal(fused, split)
    assert _f32_equal(_bits_rows(bind(LL.LlBf16DotprodGemm, N=3, K=64, BS=32, VEC=2, FMA=FMA), x, w), (split, fused)[FMA])


def test_splitk_agrees_with_the_model() -> None:
    """`LlBf16SplitKGemm_v2` against `splitk_rows` at two ranks of two tiles each, and at four ranks over six tiles (ranks 0 and
    1 take two, 2 and 3 one)."""
    rng = np.random.default_rng(3)
    for K, SPLIT_K in ((128, 2), (192, 4)):
        target = bind(LL.LlBf16SplitKGemm, N=2, K=K, SPLIT_K=SPLIT_K, TILE_K=32, WARPS=2, DOT=HOP)
        for x, w in _rows(rng, 16, 2, K):
            assert _f32_equal(_bits_rows(target, x, w), C.splitk_rows(x, w, SPLIT_K, 32, 2))


def test_f32out_and_batched_gemm_agree_with_the_models() -> None:
    """`GemmF32Out_v2` against `f32out_rows` and `BatchedGemm_v2` against `bmm_rows` (one token row per member)."""
    rng = np.random.default_rng(4)
    for x, w in _rows(rng, 32, 3, 64):
        assert _f32_equal(_bits_rows(bind(CB.GemmF32Out, K=64, N=3, DOT=HOP), x, w), C.f32out_rows(x, w))
    for kind in KINDS:
        x, w = _bf16_words(rng, (32, 2, 32), kind), _bf16_words(rng, (32, 2, 3, 32), kind)
        got = np.asarray(BS.evaluate_words(bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP), x, w)).astype(U16)
        want = np.stack([C.bmm_rows(x[i][:, None, :], np.transpose(w[i], (0, 2, 1)))[:, 0] for i in range(32)])
        assert np.array_equal(got, want), kind


def test_batched_gemm_is_mla_bmm() -> None:
    """`BatchedGemm_v2` and `mla.MlaBmm_v1` (sm_120's Triton bmm, no word of its own) are one function."""
    rng = np.random.default_rng(5)
    for B, K, N in ((2, 32, 3), (3, 64, 2)):
        for kind in KINDS:
            x, w = _bf16_words(rng, (16, B, K), kind), _bf16_words(rng, (16, B, N, K), kind)
            ours = BS.evaluate_words(bind(CB.BatchedGemm, B=B, K=K, N=N, DOT=HOP), x, w)
            theirs = BS.evaluate_words(bind(mla.MlaBmm, NHEADS=B, K=K, N=N, DOT=HOP), x, w)
            assert np.array_equal(ours, theirs), (B, K, N, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_words_at_the_gates_statics_agree_with_the_models(kind: str) -> None:
    """At GLM-5.3's gate (K 6144): the dotprod at BS 128, VEC 8 (with and without contraction), the split-K at SPLIT_K 6 and 8
    (TILE_K 256, WARPS 4) and the f32-out GEMM, through the reference evaluator, on four rows each."""
    rng = np.random.default_rng(6)
    x, w = _bf16_words(rng, (4, 6144), kind), _bf16_words(rng, (2, 6144), kind)
    cases = [(bind(LL.LlBf16DotprodDotWord, K=6144, BS=128, VEC=8, FMA=f), lambda f=f: C.dotprod_rows(x, w, 128, 8, f))
             for f in (1, 0)]
    cases += [(bind(LL.LlBf16SplitKDotWord, K=6144, SPLIT_K=s, TILE_K=256, WARPS=4, DOT=HOPW), lambda s=s: C.splitk_rows(x, w, s))
              for s in (6, 8)]
    cases += [(bind(CB.GemmF32OutWord, K=6144, N=1, DOT=HOPW), lambda: C.f32out_rows(x, w))]
    for target, model in cases:
        want = model()
        got = np.array([[evaluate(target, *(a.astype(np.int64).tolist() for a in (x[i], w[j])))[0] for j in range(len(w))]
                        for i in range(len(x))], dtype=np.uint64).astype(U32)
        assert _f32_equal(got, want), (target.id, kind)


def test_dispatch_is_vllms_sm90_table() -> None:
    assert [C.dispatch_sm90(m, 6144, 256)[0] for m in range(1, 20)] == ["dotprod"] * 4 + ["splitk"] * 12 + ["cublas"] * 3
    assert all(C.dispatch_sm90(m, 6144, 256)[1] == {"BS": 128, "VEC": 8} for m in range(1, 5))
    assert {m: C.dispatch_sm90(m, 6144, 256)[1]["SPLIT_K"] for m in range(5, 17)} == {m: 8 if m in (9, 11) else 6
                                                                                       for m in range(5, 17)}
    assert all(C.dispatch_sm90(m, 6144, 256)[1]["TILE_K"] == 256 and C.dispatch_sm90(m, 6144, 256)[1]["WARPS"] == 4
               for m in range(5, 17))
    assert C.dispatch_sm90(1, 4096, 256)[1]["BS"] == 256 and C.dispatch_sm90(2, 4096, 256)[1]["BS"] == 128
    assert C.dispatch_sm90(5, 7168, 384)[1]["SPLIT_K"] == 6 and C.dispatch_sm90(9, 7168, 384)[1]["SPLIT_K"] == 7
    assert C.dispatch_sm90(8, 1024, 256)[0] == "dotprod"
    with pytest.raises(ValueError):
        C.dispatch_sm90(0, 6144, 256)


def test_gate_rows_runs_the_dispatched_kernel() -> None:
    rng = np.random.default_rng(7)
    w = _bf16_words(rng, (3, 2048), "spread")
    for M, model in ((3, lambda x: C.dotprod_rows(x, w)), (9, lambda x: C.splitk_rows(x, w, 6)), (17, lambda x: C.f32out_rows(x, w))):
        x = _bf16_words(rng, (M, 2048), "spread")
        assert np.array_equal(C.gate_rows(x, w), model(x))


def test_the_definitions_refuse_what_they_do_not_model() -> None:
    with pytest.raises(ValueError, match="K_multiple_of_VEC_BS"):
        _ = bind(LL.LlBf16DotprodDot, K=96, BS=32, VEC=2, FMA=1).body
    with pytest.raises(ValueError, match="BS_whole_warps"):
        _ = bind(LL.LlBf16DotprodDot, K=96, BS=48, VEC=2, FMA=1).body
    with pytest.raises(ValueError, match="FMA_0_or_1"):
        _ = bind(LL.LlBf16DotprodDot, K=64, BS=32, VEC=2, FMA=2).body
    with pytest.raises(ValueError, match="blocks_per_warp_whole"):
        _ = bind(LL.LlBf16SplitKDot, K=96, SPLIT_K=1, TILE_K=48, WARPS=2, DOT=HOP).body
    with pytest.raises(ValueError, match="SPLIT_K_at_most_tiles"):
        _ = bind(LL.LlBf16SplitKDot, K=64, SPLIT_K=3, TILE_K=32, WARPS=2, DOT=HOP).body


@pytest.mark.parametrize("target", STEPS + PINNED, ids=[t.id for t in STEPS + PINNED])
def test_pinned_counts_and_digests(target) -> None:
    pin = PINS[target.id]
    prog, _ = standalone_call(target)
    got = (*(BS.compile(target).counts[k] for k in ("and", "xor", "not")), codec.program_digest(codec.encode_program(prog)))
    assert pin == got, f"{target.id}: {got}"


def test_every_pin_is_a_target() -> None:
    assert sorted(PINS) == sorted(t.id for t in STEPS + PINNED)
