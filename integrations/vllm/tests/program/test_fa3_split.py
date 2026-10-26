"""`registry.fa3_split` (FA3's dense causal prefill with the key range split and merged by FlashAttnFwdCombine), its Boolean versions
`boolean_fa3_split` and the twin `kernels.fa3_split_twins`: the per-tile split boundaries and GLM-5.3's served counts; one split is
`AttentionHead_v4` before the end and combine; the twin is the Definition word for word; a split head is softmax attention within
bf16 rounding; the vocabulary's `PER` binds it; the Boolean versions agree with their word views."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.definitions import boolean as SW
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.rules import vllm_dsa_dense
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as VOCAB
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import fa3_model
from verity.ml.definitions.numerics import fa3_split_twins as TW
from verity.ml.definitions.registry import boolean_attention as BAt
from verity.ml.definitions.registry import boolean_fa3_split as BF
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import fa3_split as F
from verity.ml.definitions.registry import hopper
from verity.ml.definitions.registry import sparse_mla_split as W
from tests.program.test_boolean_attention import _inputs

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton")
TWIN = {"dot16": DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"], "inv": DR.INV_SUM_BY_PRIM["Fa3InvSum_v1"]}


def _bf16(rng, shape, sd=1.0):
    return (rng.normal(0.0, sd, shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _val(w):
    return (np.asarray(w, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def _head(T, D, BN, MF, PER):
    return bind(F.AttentionHeadSplit, T=T, D=D, BN=BN, MASKED_FROM=MF, PER=PER, **FA3)


def _run(fn, q, k, v) -> list[int]:
    return list(evaluate(fn, q.reshape(-1).tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist()))


# --- the split plan -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("T,static,ns,per", [(320, 1, 1, None), (321, 5, 5, (1, 1)), (400, 4, 3, (1, 2)), (512, 4, 4, (1, 2)),
                                             (640, 3, 3, (1, 3)), (1024, 2, 2, (1, 7)), (1664, 5, 2, (1, 11)), (1665, 1, 1, None),
                                             (2048, 1, 1, None)])
def test_glm53s_dense_prefill_splits_from_321_to_1664_rows(T, static, ns, per) -> None:
    """GLM-5.3 TP8's head-256 prefill at 8 heads on the H200 (BM 128, BN 80): the launch's bound and the sequence's count, which
    `DsaDenseMha` binds, and the blocks a split of the first and the last tile; at most 5 splits reach a row, within the combine's
    32 lanes."""
    assert (TG.fa3_tile_m(256, T, 1), TG.fa3_kblock_n(256, T, 1)) == (128, 80)
    assert fa3_model.split_plan([T], 8, 8, 256, num_sms=132, Dv=256) == (static, [ns])
    assert vllm_dsa_dense._dense_splits(H200, T, 8, 256) == ns
    if ns > 1:
        pers = [F.split_per(T, r, 128, 80, ns) for r in range(T)]
        assert (min(pers), max(pers)) == per
        assert max(len(F.row_splits(r + 1, 80, p)) for r, p in enumerate(pers)) == ns <= W.LANES


def test_split_boundaries_are_the_tiles() -> None:
    """`block.h get_n_block_min_max` under Split: tile m's NBM = min(ceil(T / BN), ceil((m + 1) BM / BN)) in NS splits of
    ceil(NBM / NS) blocks; a row sees the splits below its own ceil(T_row / BN), the last one cut there."""
    assert [F.split_per(1024, r, 128, 80, 2) for r in (0, 127, 128, 895, 896, 1023)] == [1, 1, 2, 6, 7, 7]
    assert F.row_splits(1024, 80, 7) == [(0, 7), (7, 13)]
    assert F.row_splits(900, 80, 6) == [(0, 6), (6, 12)]
    assert F.row_splits(81, 80, 2) == [(0, 2)] and F.row_splits(80, 80, 1) == [(0, 1)]
    assert F.row_splits(5, 16, 1) == [(0, 1)]
    with pytest.raises(ValueError):
        F.split_per(10, 10, 128, 80, 2)


def test_a_launch_bound_past_32_splits_is_refused() -> None:
    F.check_combine_bound(W.LANES)
    with pytest.raises(ValueError, match="kLogMaxSplits > 5"):
        F.check_combine_bound(W.LANES + 1)
    rng = np.random.default_rng(0)
    T = 16 * W.LANES + 1
    q, k, v = _bf16(rng, 16), _bf16(rng, (T, 16)), _bf16(rng, (T, 16))
    with pytest.raises(ValueError, match="the combine's warp holds 32"):
        _run(_head(T, 16, 16, 0, 1), q, k, v)


# --- the Definition and its twin ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("T,MF,PER", [(40, 1, 1), (40, 2, 2), (40, 0, 3), (33, 0, 1), (17, 1, 1), (64, 2, 2), (5, 0, 1)])
def test_the_twin_is_the_head_definition(T, MF, PER) -> None:
    """Splits of one block and several, the row's last block partial and zero-padded or full, an uneven last split, every block
    guarded or only the masked ones."""
    D, BN, R = 16, 16, 3
    rng = np.random.default_rng(T * 7 + MF * 3 + PER)
    q, k, v = _bf16(rng, (R, D)), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    got, nf = TW.head_split_rows(q, k, v, T=T, BN=BN, PER=PER, masked_from=MF, **TWIN)
    want = np.stack([_run(_head(T, D, BN, MF, PER), q[r], k, v) for r in range(R)])
    assert not nf.any() and np.array_equal(got.astype(np.uint32), want.astype(np.uint32))


@pytest.mark.parametrize("T,MF", [(40, 2), (17, 1), (16, 0)])
def test_one_split_is_attention_head_v4_ended_and_combined(T, MF) -> None:
    """PER at least the row's blocks: one split, `AttentionHead_v4`'s chain, then the split's end and a one-lane combine, which
    gives the unsplit head's word on these rows (the end's o * INV(sum) in f32 is the finalize's, and a lone lane scales by 1)."""
    D, BN = 16, 16
    rng = np.random.default_rng(T + MF)
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    nb = -(-T // BN)
    want = _run(bind(F3.AttentionHeadV4, T=T, D=D, BN=BN, MASKED_FROM=MF, **FA3), q, k, v)
    assert _run(_head(T, D, BN, MF, nb), q, k, v) == want == _run(_head(T, D, BN, MF, nb + 3), q, k, v)


def test_attention_split_is_its_heads_under_gqa() -> None:
    T, NH, KVH, D, BN, MF, PER = 40, 4, 2, 16, 16, 1, 1
    rng = np.random.default_rng(4)
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    fn = bind(F.AttentionSplit, T=T, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=MF, PER=PER, **FA3)
    got = np.asarray(_run(fn, q, kc, vc)).reshape(NH, D)
    heads = [_run(_head(T, D, BN, MF, PER), q[h], kc[:, h // (NH // KVH)], vc[:, h // (NH // KVH)]) for h in range(NH)]
    assert np.array_equal(got, np.asarray(heads))


@pytest.mark.parametrize("T,ns", [(1024, 2), (321, 5)])
def test_a_split_row_is_softmax_attention_within_bf16_rounding(T, ns) -> None:
    """At GLM-5.3's head 256 and tile (BM 128, BN 80), the sequence's last row through the twin: softmax(q . k / 16) v within
    the kernel's roundings."""
    D, BN = 256, 80
    rng = np.random.default_rng(T)
    q, k, v = _bf16(rng, (1, D)), _bf16(rng, (T, D), 0.5), _bf16(rng, (T, D))
    per = F.split_per(T, T - 1, 128, BN, ns)
    assert len(F.row_splits(T, BN, per)) == ns
    mf = TG.fa3_masked_from(H200, T, 8, 8, 256, T, T - 1)
    got, nf = TW.head_split_rows(q, k, v, T=T, BN=BN, PER=per, masked_from=mf, **TWIN)
    s = _val(k) @ _val(q[0]) * D ** -0.5
    p = np.exp(s - s.max())
    ref = (p / p.sum()) @ _val(v)
    assert not nf.any() and np.max(np.abs(_val(got[0]) - ref)) <= 2 ** -7 * np.max(np.abs(_val(v)))


# --- the vocabulary -------------------------------------------------------------------------------------------------------------

def test_the_vocabulary_binds_attention_split_with_per() -> None:
    base = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, **FA3}
    assert VOCAB.kind("attention_dot", **base, MASKED_FROM=2, PER=1).id == bind(F.AttentionSplit, **base, MASKED_FROM=2, PER=1).id
    assert VOCAB.kind("attention_dot", **base, PER=1) is None                     # no split kind off FA3's per-iteration Check_inf
    assert "AttentionSplit" not in VOCAB.to_json()["roles"]["attention_dot"]      # the role's note, and the vocabulary version, as before


# --- the Boolean versions -------------------------------------------------------------------------------------------------------

AGREEMENT = [bind(BF.AttentionHeadSplitV2, T=17, D=16, BN=16, MASKED_FROM=1, PER=1, **BAt.ATTENTION_FA3),
             bind(BF.AttentionHeadSplitV2, T=33, D=16, BN=16, MASKED_FROM=0, PER=1, **BAt.ATTENTION_FA3)]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT + [bind(BF.AttentionSplitV2, T=17, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1, PER=1,
                                                     **BAt.ATTENTION_FA3)], ids=lambda t: t.id)
def test_the_boolean_version_is_boolean_and_its_word_view_is_v1(target) -> None:
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert BL.is_boolean(target) and (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_the_boolean_version_agrees_with_its_word_view(target) -> None:
    """Two and three splits, a partial last block: zeros, infinities, NaNs and subnormals through the blocks, each split's end and
    the combine, on bits."""
    word = TR.word_of(target)
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64)
    bad = np.flatnonzero((got != ref).reshape(len(got), -1).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} instances differ from {word.id}, first {bad[:5]}"
