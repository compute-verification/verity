"""`registry.fa2_split` (FA2's split-KV decode as vLLM serves it at its defaults), `boolean_fa2_split` and the numpy twins
`kernels.fa2_split_twins`: the split counts are `set_params_splitkv`'s, worked by hand from flash_api.cpp; each twin is its word
Definition word for word on random rows, empty splits (more splits than blocks) and rows whose every split is -inf; one split is
`AttentionHead_v5`; the split head is softmax attention within bf16 rounding; the Boolean versions agree with their word views."""
from __future__ import annotations

__models__ = ("unsloth/Llama-3.2-1B", "Qwen/Qwen3-8B")

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import fa2_split_twins as TW
from verity.ml.definitions.numerics import sparse_mla_split_twins as SMT
from verity.ml.definitions.registry import boolean_fa2_split as BX
from verity.ml.definitions.registry import fa2_split as F
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.fa2_check_inf import AttentionHeadV5
from tests.program.test_boolean_fp8 import _flat, _inputs

U16, U32 = np.uint16, np.uint32
NEG_INF = U32(F.NEG_INF)
FA2 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": P.Fa2InvSum}


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(U32) >> 16).astype(U16)


def _f32(rng, shape, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(U32)


def _ev(defn, *args) -> np.ndarray:
    return np.asarray(evaluate(defn, *(np.asarray(a).reshape(-1).tolist() for a in args)), dtype=U32)


def test_the_tiles() -> None:
    assert [F.fa2_split_kblock_n(D) for D in (64, 128, 256)] == [256, 128, 64]
    assert [F.fa2_align_kblock_n(D) for D in (64, 128, 256)] == [128, 64, 64]


@pytest.mark.parametrize("args,want", [
    # 8 sequences of 8 KV heads (one m block of 4 query heads each) at 4096 keys on 188 SMs: 32 blocks of 128, eligible counts
    # 1-8, 11, 16, 32; the best efficiency is 11's (704 / 376 over 2 waves), and 5 (320 / 376) is the least within 0.85 of it
    ((8, 8, 4, 4096, 128, 188), 5),
    # one sequence at 2048 keys, head dim 64: 8 blocks of 256, eligible 1-4 and 8, every count under one wave, so the most
    ((1, 8, 4, 2048, 64, 188), 8),
    # 320 m blocks fill 0.8 of 376: one split
    ((40, 8, 4, 2048, 64, 188), 1),
    # one key block: one split
    ((1, 8, 4, 200, 64, 188), 1),
])
def test_the_split_count_is_set_params_splitkv(args, want: int) -> None:
    assert F.fa2_num_splits(*args) == want


def test_split_ranges_keep_the_empty_splits() -> None:
    assert F.split_ranges(50, 3, 16) == [(0, 2), (2, 4), (4, 4)]
    assert F.split_ranges(50, 6, 16) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4)]


@pytest.mark.parametrize("NS,D,want", [(2, 64, (2, 1)), (3, 64, (4, 1)), (16, 128, (16, 1)), (33, 128, (32, 2)),
                                       (128, 64, (16, 8)), (5, 48, (8, 1))])
def test_the_combine_geometry(NS: int, D: int, want: tuple[int, int]) -> None:
    assert F.combine_geometry(NS, D) == want


def _end(D: int):
    return bind(F.Fa2SplitEnd, D=D, INV=P.Fa2InvSum, SCALE=D ** -0.5)


def test_the_split_end_twin_is_the_word() -> None:
    word = _end(16)
    m, l, o = _inputs(word, 400, 17)
    rng = np.random.default_rng(18)
    m[:100] = _f32(rng, 100, 40.0)                                 # the running max: any sign
    l[:50] = 0                                                     # an empty sum: inv 1, lse -inf
    part, lse = SMT.split_end_rows(m, l, o, scale=16 ** -0.5, inv=DR.fa2_inv_sum, scale_bits=F.scale_bits(16 ** -0.5))
    want = _flat(E.evaluate_batch(word, m, l, o))[0]
    assert np.array_equal(np.concatenate([part, lse[:, None]], axis=1).astype(np.uint64), want)
    assert (lse[:50] == NEG_INF).all()


@pytest.mark.parametrize("NS,D", [(2, 16), (3, 16), (5, 64), pytest.param(16, 128, marks=pytest.mark.slow), pytest.param(33, 128, marks=pytest.mark.slow)])
def test_the_combine_twin_is_the_word(NS: int, D: int) -> None:
    """Gaussian lse with empty (-inf, 0) splits planted, rows whose every split is empty, and `_inputs`' words with specials."""
    rng = np.random.default_rng(NS * 7 + D)
    word = bind(F.Fa2Combine, NS=NS, D=D)
    lse0, o0 = _inputs(word, 64, NS + D)
    lse, o = _f32(rng, (64, NS), 8.0), _f32(rng, (64, NS, D))
    empty = rng.random((64, NS)) < 0.2
    empty[:4] = True
    lse[empty], o[empty] = NEG_INF, 0
    lse, o = np.concatenate([lse0, lse]), np.concatenate([o0, o])
    want = _flat(E.evaluate_batch(word, lse, o))[0]
    assert np.array_equal(TW.combine_rows(lse, o).astype(np.uint64).reshape(len(lse), -1), want)


SPLITS = [(50, 2), (50, 3), (50, 4), (50, 6), (80, 3), (33, 3)]


@pytest.mark.parametrize("T,NS", SPLITS)
def test_the_split_head_twin_is_the_word(T: int, NS: int) -> None:
    rng = np.random.default_rng(T * 8 + NS)
    D, BN = 16, 16
    q, k, v = _bf16(rng, (2, D)), _bf16(rng, (2, T, D), 2.0), _bf16(rng, (2, T, D))
    # row 1's last block scores -inf from finite words (each product overflows), so its last split's running max starts -inf
    lo = (T - 1) // BN * BN
    q[1] = (q[1] & 0x7FFF) + (20 << 7)
    k[1, lo:] = ((k[1, lo:] & 0x7FFF) + (120 << 7)) | 0x8000
    mf = (T - 1) // BN
    head = bind(F.Fa2AttentionHeadSplit, T=T, NS=NS, D=D, BN=BN, MASKED_FROM=mf, **FA2)
    out, nf = TW.head_split_rows(q, k, v, T=T, NS=NS, BN=BN, masked_from=mf)
    assert not nf.any()
    for i in range(2):
        assert np.array_equal(_ev(head, q[i], k[i], v[i]), out[i]), i


def test_one_split_is_attention_head_v5() -> None:
    rng = np.random.default_rng(2)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    one = bind(F.Fa2AttentionHeadSplit, T=T, NS=1, D=D, BN=BN, MASKED_FROM=2, **FA2)
    assert np.array_equal(_ev(one, q, k, v), _ev(bind(AttentionHeadV5, T=T, D=D, BN=BN, MASKED_FROM=2, **FA2), q, k, v))


def test_the_gqa_split_twin_is_the_word() -> None:
    rng = np.random.default_rng(4)
    T, NH, KVH, D, BN, NS = 70, 4, 2, 16, 16, 3
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    fn = bind(F.Fa2AttentionSplit, T=T, NS=NS, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=(T - 1) // BN, **FA2)
    out, nf = TW.attention_rows(q, kc, vc, NS=NS, BN=BN, masked_from=(T - 1) // BN)
    assert not nf.any() and np.array_equal(_ev(fn, q, kc, vc), out.reshape(-1))


@pytest.mark.parametrize("NH,KVH,D", [(32, 8, 64), (32, 8, 128)])
def test_the_decode_split_is_attention(NH: int, KVH: int, D: int) -> None:
    """Llama-3.2-1B's and Qwen3-8B's decode heads at T = 1000 in 1 to 32 splits (empty ones at 32): within 2 bf16 ulps and 2^-10
    of float64's softmax attention (the twin is the word Definition at small sizes: the tests above)."""
    rng = np.random.default_rng(D)
    T = 1000
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    val = lambda a: (a.astype(U32) << 16).view(np.float32).astype(np.float64)
    kf, vf = (np.transpose(val(a), (1, 0, 2))[np.arange(NH) // (NH // KVH)] for a in (kc, vc))
    s = np.einsum("hd,htd->ht", val(q), kf) * D ** -0.5
    p = np.exp(s - s.max(axis=1, keepdims=True))
    want = np.einsum("ht,htd->hd", p / p.sum(axis=1, keepdims=True), vf)
    for NS in (1, 2, 3, 6, 32):
        BN = F.fa2_split_kblock_n(D) if NS > 1 else F.fa2_align_kblock_n(D)
        out, nf = TW.attention_rows(q, kc, vc, NS=NS, BN=BN, masked_from=-(-T // BN) - 1)
        assert not nf.any()
        assert np.all(np.abs(val(out) - want) <= 2 * 2.0 ** -7 * np.abs(want) + 2.0 ** -10), NS


# --- the Boolean versions -------------------------------------------------------------------------------------------------------

BOOL = {"DOT": BX.ATTENTION_FA2_SM120["DOT"], "INV": BX.ATTENTION_FA2_SM120["INV"]}
AGREEMENT = [bind(BX.Fa2SplitEndV2, D=16, INV=BOOL["INV"], SCALE=0.25),
             bind(BX.Fa2CombineV2, NS=3, D=16)]
#: the heads: two splits of one block each (T = 17 at BN = 16: a full block, then the partial one); three splits, one empty
HEADS = [bind(BX.Fa2AttentionHeadSplitV2, T=17, NS=2, D=16, BN=16, MASKED_FROM=1, **BOOL),
         bind(BX.Fa2AttentionHeadSplitV2, T=17, NS=3, D=16, BN=16, MASKED_FROM=1, **BOOL)]


def _agrees(target, n: int) -> None:
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, n, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    _agrees(target, 96)


@pytest.mark.slow
@pytest.mark.parametrize("target", HEADS, ids=lambda t: t.id)
def test_boolean_head_agrees_with_its_word_view(target) -> None:
    """Minutes each (two blocks, the ends and a combine traced as one circuit): what the word-level tests can't see is the
    Boolean pieces' agreement on special words through a split's blocks, an empty split and the combine."""
    _agrees(target, 32)
