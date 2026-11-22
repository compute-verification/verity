"""`registry.cublas_bf16` and `registry.ll_bf16`, GLM-5.3's bf16 GEMMs on sm_90, and `registry.cublas_gemv`: on bits they are Boolean and are the Boolean
versions of their words; they agree with their words (the reference evaluator) and with `bf16_gemm_check`'s numpy models of the
kernels on random words and on words at every class edge (signed zeros, subnormals, infinities, NaNs, and products that overflow
or underflow f32, the only inputs on which the dotprod's contraction changes a word); each chain step's parts, composed,
agree with its word from +0 and from any accumulator (the dotprod's on accumulators that cancel the product too) and no
body on bits computes a value twice; `BatchedGemm_v2` is `mla.MlaBmm_v1`'s
function; the dotprod's K loop is the kernel's (main tiles, the tail at half width, scalar rounds); the words at GLM-5.3's and
DeepSeek-V4-Pro's gate statics (K 6144, 7168) agree with the models; `dispatch_sm90` is vLLM's table; the route probe's rows
tell every candidate cuBLAS route apart; and each is pinned by gate count and program digest."""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits import word as WQ
from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.batch import type_shape
from verity.circuits.evaluation.reference import evaluate, standalone_call
from verity.circuits.ops import fp32 as W
from verity.circuits.ops import prims as P
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.ops.boolean.versions import boolean_version

from verity_vllm.definitions.registry import bf16_gemm_check as C
from verity_vllm.definitions.registry import cublas_bf16 as CB
from verity_vllm.definitions.registry import cublas_gemv as GV
from verity_vllm.definitions.registry import ll_bf16 as LL
from verity_vllm.definitions.registry import mla
from verity_vllm.definitions.registry import mla_check as MK
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3

U16, U32 = np.uint16, np.uint32
HOP = BG.HopperBF16WgmmaDot16
HOPW = word_of(HOP)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "LlBf16FmaStepFirstX_v1":
        (50, 69, 11, "62f4f2b98568ab4f62d0ebdfe10dfc0858278c1ea4cce98e9f9c139580b1be52969d11ad4f13c7270eda46d905156564210760d53515f212c5ffac22a5454ffa"),
    "LlBf16FmaStepFirst_v1":
        (1237, 1797, 127, "8cc48e2e793bee7f4f64bb126c95593749dcf4020b33b8781dd1e789f21ed3cd43a18f215b73aee809c65a904dd57a127eb38bb718aabc29224e51a75073cd07"),
    "LlBf16FmaStepX_v1":
        (50, 69, 11, "ba54c63625970d17dd5111c242d44ab6443f61106b4e90ee383e8f4c5a3f687b79812f842ba3abe5e325832af3a9980031124d6a8243359e6b381daea832c23e"),
    "LlBf16FmaStep_v1":
        (2117, 3384, 190, "8f176f704893f96b60295d6a5b5ff43ed6fa3f24004a2a66eeb76b4e91c9c8a4d98b2cc88ad41269460edcaec06161fed2cd8be4e315d537ee91c46c0fc1ab0e"),
    "LlBf16MulAddStepFirstX_v1":
        (18, 30, 2, "1f6653441ba4d6287732423ffe806b86fbea979c80f2e49b6e1092302478a3b8f1442022da02aa6847f642740bff4949143ead39c32be671c2f419445021135e"),
    "LlBf16MulAddStepFirst_v1":
        (1503, 2256, 185, "16a6a633e3cc73660b7f70373fde8a9d33acb1ad3de6b2909252111be522505c00b2c2a193fbb0dc98cd19768385fb93e461c236af68dc0880f0bf88bd5a6cb3"),
    "LlBf16MulAddStepX_v1":
        (18, 30, 2, "afd8bda43792ad87cf426cb04d6f2ecf1e49d96711efe537dec4825993b22893cc16663d07e5ee7daab2a7a61103aa1c7475ac19cd0fc9a54f846850ec3090d2"),
    "LlBf16MulAddStep_v1":
        (1543, 2462, 189, "11874041c7aa6abe0782641cbb865c0129614c3e9bf7ad0e467678fca842246d59b4c74c724316f91068edc45980b57f2f54d8e0cc1cff77f76bdef105a3ba93"),
    "HopperBf16Dot16StepFirstX_v1":
        (784, 912, 80, "8d06fc6416f5d75c71937a0d7acbf8adceb98d95294ab509e563fa4c442676e91c8ea7668ac422236335261737f3de7d57395fa7f639e101ee437f01a2b4bf55"),
    "HopperBf16Dot16StepFirst_v1":
        (6789, 13568, 649, "5e83666aae20cbd6bb3714663b9d7c8296e4cf041fc40eb664d2d4b32d1f594106cb9a87e2ea1d73248cc3fb5fde01dbd670eb78f2e856213f7c3c31b7dc80b4"),
    "HopperBf16Dot16StepX_v1":
        (784, 912, 80, "af60b3254280d4df441969217e43ff7b49758a99d86d7f7d7a18f22fd41245d35e6ab6c0bd20bdf4246094ef7fcc70f773ac9d617177460bd6195c1dfee49ab7"),
    "HopperBf16Dot16Step_v1":
        (7088, 14137, 681, "70084e5692ec9b9e778eb34713d35a6f80282d09bf914d98061e75828555297b5336912e8de0a0c9639ea4b890090efd0514e1812d7885dc9a1c5ae0ebfb99ea"),
    "GemmF32Out_v2{K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (43199, 84939, 4150, "78abf7085b213ce417db2b76b130e7a0b03764619099de4bcfbdce9b72fb1a330630f2b347c893eee477acab1542df79abbfc3708bc82fb4ff192f0565414b10"),
    "BatchedGemm_v2{B=2,K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (86794, 170466, 8390, "53a8666a159ad8cc6d1c2d187e137493ccfef978aad1027ed403378340f722b762db92eb2e3ed719497d5d541967e4a70a095fed034c5b93929a8f7eca3af444"),
    "CublasSplitKGemmF32Out_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (91912, 179019, 9143, "4716777bc32fc307b052855e87e7c5ff38410b126bdae687c224109d093aa684befc2538ef8b69c86be19650f88987e6cd71156a89a967fd6357f9fb058f0c35"),
    "CublasSplitKGemm_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (92110, 179313, 9188, "31951fdc522aa93a0cc2c1a9094f5d687dac3ab5f0788a85e65c90a874c92ccfae377dd6b96d7e6b5d2d4b77c4c60159312f8d6fca97a68823a05f1216b82998"),
    "CublasSplitKGemmBf16Partials_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (92902, 180489, 9368, "411f48802c9434b8f78b54283d0099d5120be764e5e85c8d01b2b8f9f8cf767cb821651270d0dd6f9107bf9a44a3ea9815b3bd3ddf2814dbc1f336e03fbf3031"),
    "CublasSerialSplitKGemm_v2{K=64,N=3,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v2}":
        (92704, 180195, 9323, "d1e90cbb0c35bc4f4125013a0c121ab08ea066629bcb5c08a7fbe16968730a622910e12044291deb5c4bb2e54b228913a584f6e7be5ffae9fc4e3d148fc6fbff"),
    "CublasGemvDot_v2{K=64,LANES=16}":
        (137666, 218018, 13711, "32055b16fe6a14d936f45870d32e9b9be573cd558b935bc32bd186d46b84316041e88ac3c0c1c83db881616405c16c19f37c4b2fa88c667aab393036a99d551b"),
    "CublasGemvDot_v2{K=64,LANES=32}":
        (136578, 214946, 14543, "4ab3a392ae7f0fc3a6307fb9b0e349a80012d947e3b96f26c4fee3da791896cfa6fb2f90224df1a102be7526e5ad2b2399d7941698d23f2a4f14126ee528d5ea"),
    "CublasGemv_v2{K=64,N=2,LANES=16}":
        (272132, 431620, 26718, "0ff6740e436d5ce093e8c76a1df045c841da623ab4edd9bdb9c87e96db2a8a18562008d57d585cc83629b793c2f2b59ab89f7146940aaa5a77e5df3880c422be"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=1}":
        (550400, 871680, 54784, "7d1ca846f826686a71bffb5c6c3acf9c1f3471dd58112fc90c3782d48c8c0eef3cbdb51d2688529693da9fdd7d08ee18ab34dfade36ac8763fd4f94d985a4271"),
    "LlBf16DotprodDot_v2{K=256,BS=64,VEC=2,FMA=0}":
        (449024, 714048, 56000, "f7a0f021ec9978daa55c8d12c7cb7b8870bdda1801da4df2f1f69359255b90f6c9ff676c0c7303a6419ab159495e569a4960f62ada45b0e7c0562a7e2d8476db"),
    "LlBf16DotprodDot_v2{K=101,BS=32,VEC=2,FMA=1}":
        (216691, 342609, 21965, "07e5cb75e9153609cfd148a8b04ef39a748b6bcf1c419da71028d91ae7be0b6b4731d9644e748c79798c70da8107e471133a3458328e2a751b7b2ced0436e473"),
    "LlBf16SplitKDot_v2{K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (66652, 126486, 6650, "d3ce81be8075c6f5c47c0965846e0d62cade7fd57b702318b61c5a4bda3ba687dd84e79505eefc0f986ec29f0f86c7c235c4eda6dc4561f1d3cbca6988f660fd"),
    "LlBf16DotprodGemm_v2{N=2,K=128,BS=64,VEC=2,FMA=1}":
        (539648, 850560, 56704, "c408466d0a3b688f003a88d7ba7a544268e90dbfaee9ea8cba5ef5385e5ba9029e3e44d3f6687dde2d625e3760019e4f3b49b566d26929c04a14685642d58760"),
    "LlBf16SplitKGemm_v2{N=2,K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v2}":
        (127032, 245676, 12660, "e9b06a37886eadc87ca25bc62a19364d2f3b41a65c9bb6b641f599c2a2d5d09f819c8e61e99e574a368b89371e9f0107e4f9e75187e55b21addfadce005dfa7b"),
}
#: the chain steps on bits, each four element Definitions with no word of their own (`cublas_bf16.ChainStep`)
CHAIN_STEPS = {"LlBf16FmaStep": LL.LL_FMA_STEP, "LlBf16MulAddStep": LL.LL_MUL_ADD_STEP, "HopperBf16Dot16Step": CB.HOPPER_STEP}
STEPS = [bind(d) for st in CHAIN_STEPS.values() for d in st]
PINNED = [bind(CB.GemmF32Out, K=32, N=3, DOT=HOP), bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
          bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(CB.CublasSplitKGemmBf16Partials, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(CB.CublasSerialSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
          bind(GV.CublasGemvDot, K=64, LANES=16), bind(GV.CublasGemvDot, K=64, LANES=32), bind(GV.CublasGemv, K=64, N=2, LANES=16),
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
    hi: np.ndarray = _bf16_words(rng, shape, kind).astype(U32)
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


@pytest.mark.parametrize("target", [pytest.param(t, marks=pytest.mark.slow) if t.id.startswith('LlBf16DotprodDot_v2{K=256,') else t for t in PINNED], ids=[t.id for t in PINNED])
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


@pytest.mark.slow
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
    """The Boolean split-K output types, the bf16-workspace and the serial split-K agree with the word and numpy twins at a
    nontrivial four-way partition."""
    rng = np.random.default_rng(41)
    for kind in KINDS:
        x, w = _bf16_words(rng, (12, 64), kind), _bf16_words(rng, (3, 64), kind)
        f32 = _bits_rows(bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP), x, w)
        bf16, partials, serial = (np.asarray(BS.evaluate_words(bind(d, K=64, N=3, SPLIT_K=4, DOT=HOP), x,
                                                               np.broadcast_to(w, (len(x), *w.shape)))).astype(U16)
                                  for d in (CB.CublasSplitKGemm, CB.CublasSplitKGemmBf16Partials, CB.CublasSerialSplitKGemm))
        assert _f32_equal(f32, C.cublas_splitk_rows(x, w, 4))
        assert np.array_equal(bf16, C.cublas_splitk_rows(x, w, 4, out="bf16"))
        assert np.array_equal(partials, C.cublas_splitk_rows(x, w, 4, partial="bf16", out="bf16")), kind
        assert np.array_equal(serial, C.cublas_serial_splitk_rows(x, w, 4)), kind


def test_cublas_gemv_agrees_with_its_twin_and_contracts() -> None:
    """`CublasGemv_v2` against `cublas_gemv_rows` at 16 and 32 lanes on every kind, on a row that tells its FFMA from an FMUL then
    an FADD (a lane whose second product overflows f32 alone but not added to its first: finite under FFMA, +inf otherwise), and
    on one whose every lane's chain is -0 (a negative f32 underflow, then -0 products), which the +0 after the butterfly leaves +0."""
    rng = np.random.default_rng(43)
    for LANES in (16, 32):
        gemv = bind(GV.CublasGemv, K=64, N=2, LANES=LANES)
        for kind in KINDS:
            x, w = _bf16_words(rng, (8, 64), kind), _bf16_words(rng, (2, 64), kind)
            got = np.asarray(BS.evaluate_words(gemv, x, np.broadcast_to(w, (8, *w.shape)))).astype(U16)
            assert np.array_equal(got, C.cublas_gemv_rows(x, w, LANES)), (LANES, kind)
        ovf: np.ndarray = np.zeros((2, 64), U16)
        ovf[:, 3], ovf[:, 3 + LANES] = 0xDF7F, 0x5FC0
        neg: np.ndarray = np.full((2, 64), 0x8000, U16)
        neg[:, :LANES] = 0x9980
        x = np.full((1, 64), 0x5F80, U16)
        got = np.asarray(BS.evaluate_words(gemv, x, ovf[None])).astype(U16)
        assert np.array_equal(got, C.cublas_gemv_rows(x, ovf, LANES)) and not np.array_equal(got, C.cublas_gemv_rows(x, ovf, LANES, 0))
        x = np.full((1, 64), 0x1A00, U16)
        got = np.asarray(BS.evaluate_words(gemv, x, neg[None])).astype(U16)
        assert np.array_equal(got, C.cublas_gemv_rows(x, neg, LANES)) and not got.any()
    with pytest.raises(ValueError, match="K_positive_multiple_of_LANES"):
        _ = bind(GV.CublasGemvDot, K=40, LANES=16).body


@pytest.mark.parametrize("M,K,SPLIT_K", [(1, 2048, 5), (4, 1024, 3), (303, 512, 2)])
def test_route_probe_rows_tell_every_candidate_route_apart(M, K, SPLIT_K) -> None:
    """`probe_routes` keeps the one candidate route (`cublas_routes_at`) whose twin computes every word of `route_probe_rows`; on
    the rows it reads, each pair of candidates at a split count differs somewhere, so the device's route is never ambiguous."""
    rows = sorted({0, M // 2, M - 1})
    tests = [(x[rows], w) for x, w in C.route_probe_rows(M, K, 64, np.random.default_rng([M, K]))]
    outs = [(r, [C.cublas_route_rows(r, x, w) for x, w in tests]) for r in C.cublas_routes_at([SPLIT_K])]
    for i, (a, pa) in enumerate(outs):
        for b, pb in outs[i + 1:]:
            assert any(not np.array_equal(p, q) for p, q in zip(pa, pb)), (a, b)


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
    cases: list[tuple[SpecializedDefinition, Callable[..., np.ndarray]]] = [
        (bind(LL.LlBf16DotprodDotWord, K=K, BS=bs, VEC=8, FMA=f), lambda bs=bs, f=f: C.dotprod_rows(x, w, bs, 8, f))
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


def test_profile_vocabulary_binds_the_served_default_glm53_families() -> None:
    v = PROFILE_B1_EAGER_V3
    assert v.kind("cublas_splitk_gemm", K=6144, N=160, SPLIT_K=4, DOT=HOPW).id == \
        bind(CB.CublasSplitKGemmWord, K=6144, N=160, SPLIT_K=4, DOT=HOPW).id
    assert v.kind("cublas_gemm_f32out", K=6144, N=256, DOT=HOPW).id == \
        bind(CB.GemmF32OutWord, K=6144, N=256, DOT=HOPW).id
    assert v.kind("cublas_splitk_gemm_f32out", K=6144, N=256, SPLIT_K=3, DOT=HOPW).id == \
        bind(CB.CublasSplitKGemmF32OutWord, K=6144, N=256, SPLIT_K=3, DOT=HOPW).id
    assert v.kind("batched_gemm", B=8, K=192, N=512, DOT=HOPW).id == bind(CB.BatchedGemmWord, B=8, K=192, N=512, DOT=HOPW).id
    assert v.kind("batched_gemm", B=8, K=512, N=256, DOT=HOPW).id == bind(CB.BatchedGemmWord, B=8, K=512, N=256, DOT=HOPW).id
    assert v.kind("ll_bf16_dotprod", N=256, K=6144, BS=128, VEC=8, FMA=1).id == \
        bind(LL.LlBf16DotprodGemmWord, N=256, K=6144, BS=128, VEC=8, FMA=1).id
    assert v.kind("ll_bf16_splitk", N=256, K=6144, SPLIT_K=6, TILE_K=256, WARPS=4, DOT=HOPW).id == \
        bind(LL.LlBf16SplitKGemmWord, N=256, K=6144, SPLIT_K=6, TILE_K=256, WARPS=4, DOT=HOPW).id


@pytest.mark.parametrize("target", STEPS + PINNED, ids=[t.id for t in STEPS + PINNED])
def test_pinned_counts_and_digests(target) -> None:
    pin = PINS[target.id]
    prog, _ = standalone_call(target)
    got = (*(BS.compile(target).counts[k] for k in ("and", "xor", "not")), codec.program_digest(codec.encode_program(prog)))
    assert pin == got, f"{target.id}: {got}"


def test_every_pin_is_a_target() -> None:
    assert sorted(PINS) == sorted(t.id for t in STEPS + PINNED)


def test_the_cublas_and_ll_bf16_conformance_cite_the_one_h200_capture_record() -> None:
    """Both registries' Definitions are established by the same `glm53_bf16_gemm_capture` record the BatchedGemm_v1 evaluator cites,
    and the FMA the words decided is the one the gate's Program binds by default."""
    from verity.circuits.defs import conformance_label

    from verity_vllm.definitions.numerics.batched_gemm_rows import PROVENANCE

    cite = (PROVENANCE["record"][:16], PROVENANCE["art"]["record"])
    for d in (CB.GemmF32OutWord, CB.CublasSplitKGemmWord, CB.BatchedGemm, LL.LlBf16DotprodGemmWord, LL.LlBf16SplitKGemmWord):
        assert conformance_label(d.conformance) == "exact-model-tested" and all(c in d.conformance for c in cite), d.id
    assert f"FMA {C.DEFAULT_LL_BF16_FMA} as the words decide it" in LL.CONFORMANCE
