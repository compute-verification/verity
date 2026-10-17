"""`registry.cublas_bf16` and `registry.ll_bf16`, GLM-5.3's bf16 GEMMs on sm_90: on bits they are Boolean and are the Boolean
versions of their words; they agree with their words (the reference evaluator) and with `bf16_gemm_check`'s numpy models of the
kernels on random words and on words at every class edge (signed zeros, subnormals, infinities, NaNs, and products that overflow
or underflow f32, the only inputs on which the dotprod's contraction changes a word); each chain step's parts, composed,
agree with its word from +0 and from any accumulator (the dotprod's on accumulators that cancel the product too) and no
body on bits computes a value twice; `BatchedGemm_v2` is `mla.MlaBmm_v1`'s
function; the dotprod's K loop is the kernel's (main tiles, the tail at half width, scalar rounds); the words at GLM-5.3's and
DeepSeek-V4-Pro's gate statics (K 6144, 7168) agree with the models; `dispatch_sm90` is vLLM's table; and each is pinned by gate
count and program digest."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import CompositeDefinition, bind
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
from verity_vllm.query import word as WQ

U16, U32 = np.uint16, np.uint32
HOP = BG.HopperBF16WgmmaDot16
HOPW = word_of(HOP)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "LlBf16FmaStepFirstX_v1":
        (50, 69, 11, "00c7dd696d037eecad8b486bd618f9303f98da65c74a37b1b027a56550c25bcdcb64db8a6d4a9ef6626c747299de705c7d018205c04f677212beb5cdf2445502"),
    "LlBf16FmaStepFirst_v1":
        (1237, 1797, 127, "7e12ede269d688d500bfb630116b9a28ba3e0ccd4b5167b951d35b8f32895fc761615ebd91ab72bffcbe6d50a932f46da2b7bd3f49e3de58e215531843bcf2c1"),
    "LlBf16FmaStepX_v1":
        (50, 69, 11, "c81ca3c8a3c3875b65db3af5826dce1c83af9e9310f707928a72fabb65a7519590111c39e8b58b0ee199db52e9f9609e57318532f6ffe76c4c6367fd20ce2258"),
    "LlBf16FmaStep_v1":
        (2117, 3384, 190, "3019c559bfdbea2f058663d56cc77145b26320c30ddbfe5a5894f258fe841572149720931cf9e64cf04e893a699f0d32421be3a3f2061c36a8afcc0ef861eea9"),
    "LlBf16MulAddStepFirstX_v1":
        (18, 30, 2, "f5e1249bc607aaa7f2c468845588b31df5e1e777e124139f7d26626a37989e78bb0b0497fd48500e68a93410f96297e105ec6d2aa4b0ca606a38a97d6bb50320"),
    "LlBf16MulAddStepFirst_v1":
        (1503, 2256, 185, "fb3ebc96c6eb322c27181106dbf7bb52257748de60832314d7ad44ec36f0b3fcfb06e5d2b54b6c0c2aacfd4361ee1d40dff4f4929176b1e4af5b52b011b30a89"),
    "LlBf16MulAddStepX_v1":
        (18, 30, 2, "813339c8e135d08aea58fff55baa05bc6c6c7ada8a6238f37ee81a3a546bd3e086efc59b3f9977295692b398b6157b09931aecd37da0d13e0e5f92ddf781321d"),
    "LlBf16MulAddStep_v1":
        (1543, 2462, 189, "7085a40ed37813b1ee263ceef120976bb25af92706417ad7b940f9c90c17b1aff9b2dab378a53a7ccd72de99eb0c9f6ecaa7c08c7d0d2b55026b1499e4d0f2fd"),
    "HopperBf16Dot16StepFirstX_v1":
        (784, 912, 80, "edab596801c5774b1063b84554aa0e4d459793d93b2071ac1a3a09d818d4308adf8214d1de0ed34b26a02bf0f5ef50f92a64142031a61e11c40cac9162aa6a08"),
    "HopperBf16Dot16StepFirst_v1":
        (6789, 13568, 649, "f0ab2c741a0eafd9de47a6baff27a751f07f34323716a0de09d55f01115350a9ea475cacf21136507c1a81f00996fd47d4987664f8faebb49b40a9353ec18bea"),
    "HopperBf16Dot16StepX_v1":
        (784, 912, 80, "ea8f98a5d54e6d0347798318e61f4daada6556497e8f1c12bb90fdaa1a1313c6d060b6698a07b3c2f76808b795091c0ed077645c27f44bc6e71650f026ef856a"),
    "HopperBf16Dot16Step_v1":
        (7088, 14137, 681, "3165af824cf09d3786ecd742f5dddab7e65a74a768c7cf4d56098532c5d1091624d4fe6cc254ebe48e62bf136b8b87403ab05afd6c3007d83c9917c8b99e9e05"),
    "GemmF32Out_v2{K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (43199, 84939, 4150, "dc461c922a161552a39cf88a5ab81de7f9bea85f12de679d7327b37dd55aa8a026b35ba8c2f6b41670f7ed2d9519fb180b2a94f0a25ff2a77a331e95e960b02c"),
    "BatchedGemm_v2{B=2,K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (86794, 170466, 8390, "a5628461bd73e30a57648857b606dc09260abb3a795cc75efc45eef4eb1165d7225f86cd7812c469abb1755ab8ec51afb79deaa5be1748bfbe133c7ff6abb5f1"),
    "CublasSplitKGemmF32Out_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (91912, 179019, 9143, "5eca429c74900b6b569f648b957894ad339aa1648d12b11f06aca7dd1f2f76d5e2f93da288be7634078c87fb65c4ce8674eae8536517aa0f521552f322469428"),
    "CublasSplitKGemm_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (92110, 179313, 9188, "3c0e23088a8a80db54673b576a5b88ed18e29fdd391c5e477d806cf1fe32dd220686d8caf2b72236aa3e66c901161c4bce5945946e648aa575fce8049dbcce02"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=1}":
        (550400, 871680, 54784, "310290be808e36bd3308147d713f3bf718561ee1512a5b9dc2b5b5c3e5b297308b555d436a583357cdefeef7314aa303466916d65cd181c2131406ed3e1c2979"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=0}":
        (449024, 714048, 56000, "1bdbc6c31ccaaf30aaadedaf8452b46bcdb580b123d8b2e4bad5d7c5ea8dd1d23a7456a6797f3a58d28ae97fd6d9514c759f82df5edce70b65a5d8522a9e5fbc"),
    "LlBf16DotprodDot_v2{K=101,BS=32,VEC=2,FMA=1}":
        (216691, 342609, 21965, "d3c340f9f4c8026aafd1bf457adc5c7b8fa37b99c5c19f20edde0c8760a57aadc0cfb16d1128ad138f3ba198fc45d89da60935943226305956e08dbcf7bb7a92"),
    "LlBf16SplitKDot_v2{K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (66652, 126486, 6650, "ad93c680dafa712fcb77ce24a5988acc56d34b39d713aedd99c76e16de918ed54c29df59c010bd8053c88f8e68e43d91c3a9d68ec1099a9857dd8bedf37496c2"),
    "LlBf16DotprodGemm_v2{N=2,K=128,BS=64,VEC=2,FMA=1}":
        (539648, 850560, 56704, "8e24b5216820e71b4c04d87bcdef4b47dcedf6b58ec7cf423e9c304991c270aa04883a8178e478ce35e6f6b2971e8f1457df2d56b9aaf39a0338dcf8a0aaba88"),
    "LlBf16SplitKGemm_v2{N=2,K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (127032, 245676, 12660, "59acc816f055ceff29d80e961536aa56fed7db5b8a4db5fbba909248cb9284404832377d9cd76322a8db7547ca5357d5e69a11bed385e2fab2e67aeaf6b4c940"),
}
#: the chain steps on bits, each four element Definitions with no word of their own (`cublas_bf16.ChainStep`)
CHAIN_STEPS = {"LlBf16FmaStep": LL.LL_FMA_STEP, "LlBf16MulAddStep": LL.LL_MUL_ADD_STEP, "HopperBf16Dot16Step": CB.HOPPER_STEP}
STEPS = [bind(d) for st in CHAIN_STEPS.values() for d in st]
PINNED = [bind(CB.GemmF32Out, K=32, N=3, DOT=HOP), bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
          bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=1), bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=0),
          bind(LL.LlBf16DotprodDot, K=101, BS=32, VEC=2, FMA=1),
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


def _f32(x: int) -> int:
    return P.Bf16ToF32.evaluate(x)


#: each chain step's word on (acc, x, w): fma(f32(x), f32(w), acc), acc + f32(x) * f32(w), and the wgmma k16 step
STEP_WORDS = {"LlBf16FmaStep": lambda acc, x, w: W.F32Fma.evaluate(_f32(x), _f32(w), acc),
              "LlBf16MulAddStep": lambda acc, x, w: W.F32Add.evaluate(acc, W.F32Mul.evaluate(_f32(x), _f32(w))),
              "HopperBf16Dot16Step": lambda acc, a, b: P.HopperBF16WgmmaDot16.evaluate(acc, *a, *b)}


def _composed(st, first: bool) -> CompositeDefinition:
    """``st``'s first step (from +0) or a later one as one Definition: x's own gates, then the rest."""
    part, own = (st.first, st.first_x) if first else (st.step, st.x)
    params = bind(part).signature()[0][:-1]
    x = 0 if first else 1
    return CompositeDefinition(f"{part.name}Composed", 1, (), lambda S: (params, CB.F32),
                               lambda B, S, *a: B.call(part, *a, B.call(own, a[x])), register=False)


@pytest.mark.parametrize("name", sorted(CHAIN_STEPS))
def test_chain_steps_agree_with_their_words(name: str) -> None:
    """Each chain step's parts, composed, against its word: a later step on 256 instances of each kind, and the dotprod's also on
    accumulators that cancel the product exactly (a bf16 product is exact in f32), where the sum's zero takes its sign from the
    rounding mode; the first step on the same operands from +0."""
    rng = np.random.default_rng(8)
    st, word = CHAIN_STEPS[name], STEP_WORDS[name]
    shape = (256,) if name.startswith("Ll") else (256, 16)
    cases = [(_f32_words(rng, (256,), k), _bf16_words(rng, shape, k), _bf16_words(rng, shape, k)) for k in KINDS]
    if name.startswith("Ll"):
        x, w = _bf16_words(rng, shape, "spread"), _bf16_words(rng, shape, "spread")
        product = (x.astype(U32) << 16).view(np.float32) * (w.astype(U32) << 16).view(np.float32)
        cases.append((MK.u32(-product), x, w))
    ops = (lambda a: int(a)) if name.startswith("Ll") else (lambda a: [int(v) for v in a])
    first, later = _composed(st, True), _composed(st, False)
    for acc, x, w in cases:
        got = np.asarray(BS.evaluate_words(later, acc, x, w)).astype(np.int64)
        assert np.array_equal(got, np.array([word(int(c), ops(a), ops(b)) for c, a, b in zip(acc, x, w)], np.int64)), name
        got = np.asarray(BS.evaluate_words(first, x, w)).astype(np.int64)
        assert np.array_equal(got, np.array([word(0, ops(a), ops(b)) for a, b in zip(x, w)], np.int64)), name


@pytest.mark.slow
def test_no_body_on_bits_computes_a_value_twice() -> None:
    """No value is computed twice anywhere in a Definition on bits (`word.Graph.recomputed`), with each chain's first step
    reading the +0 accumulator and several W rows sharing x: `circuit-check`'s redundant-gates rule, at small statics."""
    for t in [bind(LL.LlBf16DotprodGemm, N=2, K=K, BS=32, VEC=2, FMA=f) for K, f in ((64, 1), (64, 0), (101, 1))] + [
            bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP), bind(LL.LlBf16SplitKGemm, N=2, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP)]:
        assert WQ.Graph(t).recomputed == [], t.id


@pytest.mark.parametrize("FMA", [1, 0])
def test_dotprod_agrees_with_the_model(FMA: int) -> None:
    """`LlBf16DotprodGemm_v2` against `dotprod_rows` over two tiles of two warps' lanes, over one tile of one warp, and over one
    tile, a tail tile at width 1 and a ragged round on 5 of 32 threads; on the wide words the contracted and uncontracted models
    differ, and each Definition is its own."""
    rng = np.random.default_rng(2)
    for K, BS_, VEC in ((128, 64, 1), (64, 32, 2), (101, 32, 2)):
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


def test_cublas_splitk_definitions_agree_with_the_hypothesis() -> None:
    """Both Boolean split-K output types agree with the word and numpy twins at a nontrivial four-way partition."""
    rng = np.random.default_rng(41)
    for kind in KINDS:
        x, w = _bf16_words(rng, (12, 64), kind), _bf16_words(rng, (3, 64), kind)
        f32 = _bits_rows(bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP), x, w)
        bf16 = np.asarray(BS.evaluate_words(bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP), x,
                                            np.broadcast_to(w, (len(x), *w.shape)))).astype(U16)
        assert _f32_equal(f32, C.cublas_splitk_rows(x, w, 4))
        assert np.array_equal(bf16, C.cublas_splitk_rows(x, w, 4, out="bf16"))


def test_cublas_default_split_bounds_are_equal_64_element_chunks_then_a_remainder() -> None:
    assert CB.split_bounds(7168, 5) == [(0, 1472), (1472, 2944), (2944, 4416), (4416, 5888), (5888, 7168)]
    assert CB.split_bounds(7168, 3) == [(0, 2432), (2432, 4864), (4864, 7168)]


def test_cublas_splitk_probe_tells_the_hypotheses_apart() -> None:
    """Crafted finite words distinguish the stated contiguous/f32/forward hypothesis from every capture alternative."""
    rng = np.random.default_rng(42)
    x = _bf16_words(rng, (16, 256), "wide")
    w = _bf16_words(rng, (8, 256), "wide")
    # Keep products finite while retaining a broad exponent spread and cancellation.
    x = ((x & U16(0x807F)) | (rng.integers(112, 143, x.shape, dtype=U16) << U16(7))).astype(U16)
    w = ((w & U16(0x807F)) | (rng.integers(112, 143, w.shape, dtype=U16) << U16(7))).astype(U16)
    got = C.cublas_splitk_hypotheses(x, w, 8, out="f32")
    candidate = got[C.CUBLAS_SPLITK_HYPOTHESIS]
    assert all(not _f32_equal(candidate, words) for name, words in got.items() if name != C.CUBLAS_SPLITK_HYPOTHESIS)


def test_batched_gemm_is_mla_bmm() -> None:
    """`BatchedGemm_v2` and `mla.MlaBmm_v1` (sm_120's Triton bmm, no word of its own) are one function."""
    rng = np.random.default_rng(5)
    for B, K, N in ((2, 32, 3), (3, 64, 2)):
        for kind in KINDS:
            x, w = _bf16_words(rng, (16, B, K), kind), _bf16_words(rng, (16, B, N, K), kind)
            ours = BS.evaluate_words(bind(CB.BatchedGemm, B=B, K=K, N=N, DOT=HOP), x, w)
            theirs = BS.evaluate_words(bind(mla.MlaBmm, NHEADS=B, K=K, N=N, DOT=HOP), x, w)
            assert np.array_equal(ours, theirs), (B, K, N, kind)


def _kernel_k_loops(K: int, BS: int, VEC: int = 8, TAIL: int = 4) -> tuple[int, int, int, int]:
    """`LLBf16Dotprod._init_k_tiles`'s counts, as the kernel computes them: main tiles, tail tiles, full scalar rounds, ragged."""
    main = K // (VEC * BS) * VEC * BS
    tail = (K - main) // (TAIL * BS) * TAIL * BS
    rem = K - main - tail
    return main // (VEC * BS), tail // (TAIL * BS), rem // BS, rem % BS


@pytest.mark.parametrize("K,BS_", [(6144, 128), (7168, 256), (7168, 128), (4096, 256), (7168 + 3 * 256 + 37, 256), (2049, 128)])
def test_dotprod_rounds_are_the_kernels_k_loops(K: int, BS_: int) -> None:
    """`dotprod_rounds` at VEC 8 is the kernel's K loop: its main tiles' 8 rounds each, its tail tiles' 4, its full scalar rounds
    and a ragged one on the threads below the rest; every k once; and each thread's k in `ll_bf16`'s order."""
    main, tail, full, part = _kernel_k_loops(K, BS_)
    rounds = C.dotprod_rounds(K, BS_, 8)
    assert len(rounds) == 8 * main + 4 * tail + full + (part > 0)
    assert all((r >= 0).all() for r in rounds[:len(rounds) - (part > 0)])
    if part:
        assert ((rounds[-1] >= 0) == (np.arange(BS_) < part)).all()
    assert sorted(int(k) for r in rounds for k in r if k >= 0) == list(range(K))
    lanes = LL._lanes(SimpleNamespace(K=K, BS=BS_, VEC=8))
    assert lanes == [[int(r[t]) for r in rounds if r[t] >= 0] for t in range(BS_)]


#: (K, N) of each gate, with the dotprod's BS and the split-K's SPLIT_K that `dispatch_sm90` names there
GATES = {"glm53": (6144, 256, (128,), (6, 8)), "dsv4-pro": (7168, 384, (256, 128), (6, 7, 8))}


@pytest.mark.parametrize("gate", sorted(GATES))
@pytest.mark.parametrize("kind", KINDS)
def test_words_at_the_gates_statics_agree_with_the_models(gate: str, kind: str) -> None:
    """At GLM-5.3's gate (K 6144) and DeepSeek-V4-Pro's (K 7168): the dotprod at each BS its dispatch names, VEC 8 (with and
    without contraction; at K 7168, BS 256 its main tiles leave one tail tile), the split-K at each SPLIT_K it names (TILE_K 256,
    WARPS 4; 28 tiles over 6, 7 or 8 ranks) and the f32-out GEMM, through the reference evaluator, on two rows each."""
    K, N, dotprod_bs, split_ks = GATES[gate]
    rng = np.random.default_rng(6)
    x, w = _bf16_words(rng, (2, K), kind), _bf16_words(rng, (1, K), kind)
    assert {C.dispatch_sm90(m, K, N)[1].get("BS") for m in range(1, 5)} == set(dotprod_bs)
    assert {C.dispatch_sm90(m, K, N)[1].get("SPLIT_K") for m in range(5, 17)} == set(split_ks)
    cases = [(bind(LL.LlBf16DotprodDotWord, K=K, BS=bs, VEC=8, FMA=f), lambda bs=bs, f=f: C.dotprod_rows(x, w, bs, 8, f))
             for bs in dotprod_bs for f in (1, 0)]
    cases += [(bind(LL.LlBf16SplitKDotWord, K=K, SPLIT_K=s, TILE_K=256, WARPS=4, DOT=HOPW), lambda s=s: C.splitk_rows(x, w, s))
              for s in split_ks]
    cases += [(bind(CB.GemmF32OutWord, K=K, N=1, DOT=HOPW), lambda: C.f32out_rows(x, w))]
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
    assert [C.dispatch_sm90(m, 7168, 384)[1].get("BS") for m in range(1, 5)] == [256, 256, 128, 128]
    assert {m: C.dispatch_sm90(m, 7168, 384)[1]["SPLIT_K"] for m in range(5, 17)} == {
        m: 6 if m == 5 else 8 if m in (6, 8, 12) else 7 for m in range(5, 17)}
    assert C.dispatch_sm90(17, 7168, 384)[0] == "cublas"
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
    with pytest.raises(ValueError, match="K_at_least_BS"):
        _ = bind(LL.LlBf16DotprodDot, K=16, BS=32, VEC=2, FMA=1).body
    with pytest.raises(ValueError, match="VEC_positive"):
        _ = bind(LL.LlBf16DotprodDot, K=64, BS=32, VEC=0, FMA=1).body
    with pytest.raises(ValueError, match="BS_whole_warps"):
        _ = bind(LL.LlBf16DotprodDot, K=96, BS=48, VEC=2, FMA=1).body
    with pytest.raises(ValueError, match="FMA_0_or_1"):
        _ = bind(LL.LlBf16DotprodDot, K=64, BS=32, VEC=2, FMA=2).body
    with pytest.raises(ValueError, match="blocks_per_warp_whole"):
        _ = bind(LL.LlBf16SplitKDot, K=96, SPLIT_K=1, TILE_K=48, WARPS=2, DOT=HOP).body
    with pytest.raises(ValueError, match="SPLIT_K_at_most_tiles"):
        _ = bind(LL.LlBf16SplitKDot, K=64, SPLIT_K=3, TILE_K=32, WARPS=2, DOT=HOP).body
    with pytest.raises(ValueError, match="SPLIT_K_in_2_to_K_over_16"):
        _ = bind(CB.CublasSplitKGemmF32Out, K=64, N=2, SPLIT_K=5, DOT=HOP).body


@pytest.mark.parametrize("target", STEPS + PINNED, ids=[t.id for t in STEPS + PINNED])
def test_pinned_counts_and_digests(target) -> None:
    pin = PINS[target.id]
    prog, _ = standalone_call(target)
    got = (*(BS.compile(target).counts[k] for k in ("and", "xor", "not")), codec.program_digest(codec.encode_program(prog)))
    assert pin == got, f"{target.id}: {got}"


def test_every_pin_is_a_target() -> None:
    assert sorted(PINS) == sorted(t.id for t in STEPS + PINNED)
