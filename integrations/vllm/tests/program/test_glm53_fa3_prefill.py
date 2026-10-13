"""GLM-5.3's dense MHA prefill (`run_prefill_new_tokens`, FA3 at qk head 256 = 192 nope + 64 rope, v head 256, 8 heads a rank on 8 x H200):
the 128 x 80 tile in the target's rule and the twin's, the row binding `Attention_v4{..., D=256, BN=80}`, the two twins and the
Definition word for word, the kernel's fully-masked leading blocks, and which launches the served split heuristic splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import fa2_relation, fa3_model
from verity_vllm.program.registry import attention_vocab, b1, hopper
from verity_vllm.program.registry import fa3_check_inf as F3

CMAP = Path(__file__).resolve().parents[2] / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"
FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
NH = KVH = 8
D, BN, BM = 256, 80, 128
TWIN = {"dot_prim": "HopperBF16WgmmaDot16_v1", "inv_prim": "Fa3InvSum_v1"}


def _bf16(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _profile(**kw) -> TargetProfile:
    return TargetProfile.from_json(dict(json.loads(CMAP.read_text())["target"]["profile"], **kw))


def test_hdim_256_runs_the_128_by_80_tile_in_both_rules():
    """`tile_size_fwd_sm90` at headdim > 192, bf16, not local: {128, 80}, whatever seqlen_q (no one-MMA-warpgroup branch above 128) or
    paging.  The twin's rule is the target's at every registered head dim; 192 (DeepSeek's MLA prefill, v 128) stays refused."""
    for sq in (1, 16, 64, 65, 512, 4096):
        for g in (1, 4):
            assert TG.fa3_kblock_n(D, sq, g) == fa3_model.kblock_n_for(D, sq, g) == BN
            assert TG.fa3_tile_m(D, sq, g) == fa3_model.kblock_m_for(D, sq, g) == BM
            for d in (64, 128):
                assert TG.fa3_kblock_n(d, sq, g) == fa3_model.kblock_n_for(d, sq, g)
                assert TG.fa3_tile_m(d, sq, g) == fa3_model.kblock_m_for(d, sq, g)
    with pytest.raises(TG.UnregisteredTarget, match="head dim 192"):
        TG.fa3_kblock_n(192, 512, 1)
    with pytest.raises(ValueError, match="head dim 192"):
        fa3_model.kblock_n_for(192, 512, 1)


@pytest.mark.parametrize("row", [0, 79, 80, 127, 128, 255, 256, 299])
def test_a_glm53_prefill_row_binds_attention_v4_at_bn_80(row):
    """On the capture map's target (FA3's per-iteration `Check_inf`, the default) a prefill row of a 300-token sequence binds
    `Attention_v4{T = row + 1, NH = KVH = 8, D = 256, BN = 80}` with its tile's mask boundary; `guard-every-block` keeps `Attention_v2`."""
    T = row + 1
    masked_from = attention_vocab.fa3_masked_from(T, BN, row, BM, 1)
    assert masked_from == (row // BM * BM) // BN
    spec = TG.attention_spec(_profile(), T, NH, KVH, D, 300, row=row)
    assert spec.id == bind(F3.AttentionV4, T=T, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=masked_from, **FA3).id
    guarded = TG.attention_spec(_profile(fa3_construction="guard-every-block"), T, NH, KVH, D, 300)
    assert guarded.id == bind(b1.AttentionV2, T=T, NH=NH, KVH=KVH, D=D, BN=BN, **FA3).id


@pytest.mark.parametrize("T", [1, 80, 81, 161, 300])
def test_the_cpp_twin_the_vectorised_twin_and_the_definition_agree(T):
    """The C++ FA3 model (`fa3_rows`, 16 QK and 5 P.V k16 steps a block at D = 256), `derived_rows.attention_rows` and the registered
    `AttentionHead_v4` gate for gate, at one and several 80-key blocks and a partial last block; the last query row of a T-token
    prefill (its tile's mask boundary)."""
    rng = np.random.default_rng(256_000 + T)
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    row = T - 1
    masked_from = attention_vocab.fa3_masked_from(T, BN, row, BM, 1)
    out, nf = DR.attention_rows(q, kc, vc, BN, masked_from=masked_from, **TWIN)
    assert not nf.any()
    cpp = fa3_model.attention_rows(q, kc, vc, np.arange(NH, dtype=np.int32), np.full(NH, T, np.int32), np.full(NH, -(-T // BN), np.int32),
                                   D ** -0.5, fa2_relation.tables(), kblock_n=BN)
    assert np.array_equal(cpp.out, out), int((cpp.out != out).sum())
    h = 3
    ref = evaluate(bind(F3.AttentionHeadV4, T=T, D=D, BN=BN, MASKED_FROM=masked_from, **FA3),
                   q[h].tolist(), kc[:, h].reshape(-1).tolist(), vc[:, h].reshape(-1).tolist())
    assert out[h].tolist() == list(ref)


def test_attention_v4_is_the_row_at_glm53_geometry():
    """`Attention_v4{T, NH = KVH = 8, D = 256, BN = 80}` (every head its own KV head) evaluates to the twin's row."""
    T = 97
    rng = np.random.default_rng(97)
    q, kc, vc = _bf16(rng, (NH, D)), _bf16(rng, (T, KVH, D)), _bf16(rng, (T, KVH, D))
    out, _ = DR.attention_rows(q, kc, vc, BN, masked_from=1, **TWIN)
    ref = evaluate(bind(F3.AttentionV4, T=T, NH=NH, KVH=KVH, D=D, BN=BN, MASKED_FROM=1, **FA3),
                   q.ravel().tolist(), kc.ravel().tolist(), vc.ravel().tolist())
    assert out.ravel().tolist() == list(ref)


def test_the_kernels_fully_masked_leading_blocks_are_neutral():
    """kBlockM 128 > kBlockN 80: the kernel's per-tile `n_block_max` makes a tile's first rows visit up to two key blocks they see
    nothing of before their own (row 0: 2 blocks for 1, row 128: 4 for 2).  The C++ twin run at that count gives the per-row
    count's words on every row of a 300-token prefill."""
    L = 300
    rows = np.arange(L)
    col, per_row = fa3_model.kernel_geometry(L, L, rows, BN)
    _, per_tile = fa3_model.kernel_geometry(L, L, rows, BN, kblock_m=BM)
    extra = per_tile - per_row
    assert extra.min() == 0 and extra.max() == 2 and (per_tile[0], per_row[0], per_tile[128], per_row[128]) == (2, 1, 4, 2)
    rng = np.random.default_rng(300)
    heads = 2
    q, kc, vc = _bf16(rng, (L, heads, D)), _bf16(rng, (L, heads, D)), _bf16(rng, (L, heads, D))
    qf = q.reshape(L * heads, D)
    kvh = np.tile(np.arange(heads, dtype=np.int32), L)
    tables = fa2_relation.tables()
    a = fa3_model.attention_rows(qf, kc, vc, kvh, np.repeat(col, heads), np.repeat(per_row, heads), D ** -0.5, tables, kblock_n=BN)
    b = fa3_model.attention_rows(qf, kc, vc, kvh, np.repeat(col, heads), np.repeat(per_tile, heads), D ** -0.5, tables, kblock_n=BN)
    assert not a.nonfinite.any() and np.array_equal(a.out, b.out) and np.array_equal(a.out_f32, b.out_f32)


def test_the_twin_refuses_a_tile_hdim_256_does_not_run():
    rng = np.random.default_rng(1)
    q, kc = _bf16(rng, (1, D)), _bf16(rng, (8, 1, D))
    one = np.ones(1, np.int32)
    for d, bn in ((256, 128), (128, 80)):
        with pytest.raises(ValueError, match="hdim 256 runs kBlockN 80 only"):
            fa3_model.attention_rows(q[:, :d], kc[..., :d], kc[..., :d], 0 * one, 8 * one, one, d ** -0.5, fa2_relation.tables(), kblock_n=bn)
    assert fa3_model.coverage(512, 512, NH, KVH, D, {"arch": "sm_90"}) == "head_dim 256 modelled but not measured (measured: (128,))"


@pytest.mark.parametrize("lengths,expected", [
    ([1], [1]), ([256], [1]), ([320], [1]), ([321], [5]), ([512], [4]), ([1024], [2]), ([1664], [2]), ([1665], [1]), ([4096], [1]),
    ([512, 512, 64], [2, 2, 1]), ([1000, 30, 30], [2, 1, 1]), ([512] * 993, [1] * 993),
])
def test_the_served_split_heuristic_at_glm53_per_rank(lengths, expected):
    """vLLM passes `num_splits = 0` to the MLA prefill outside batch invariance (which GLM-5.3 refuses on sm_90): `get_num_splits`'
    static bound, then the prepare kernel's per-sequence count.  Per rank (8 heads, 132 SMs) one-sequence launches of 321..1664 tokens
    split (the Split kernel and `FlashAttnFwdCombine`, outside the model); 993 sequences or more never do."""
    assert fa3_model.num_splits(lengths, NH, KVH, D, num_sms=132) == expected


def test_the_devices_split_quotient_is_the_exact_one_at_glm53_per_rank():
    """`blocks_per_sm = ceilf(total * 1.1f * 8 / 132)` with `/` as `div.full.ftz.f32`: MUFU.RCP(132) is one ulp below 1/132, so the device's
    quotient never rounds up past the IEEE one, and its ceiling is the IEEE ceiling for every total blocks count up to 4000."""
    b = np.array([132.0], np.float32).view(np.uint32)
    assert int(DR.div_full_rcp(b)[0]) == int(np.array([np.float32(1) / np.float32(132)]).view(np.uint32)[0]) - 1
    f32 = np.float32
    a = ((np.arange(1, 4001).astype(f32) * f32(1.1)).astype(f32) * f32(8)).astype(f32)
    device = DR.as_f32(DR.mul_ftz(DR.div_full_scale_a(a.view(np.uint32), np.repeat(b, a.size)), np.repeat(DR.div_full_rcp(b), a.size)))
    assert np.array_equal(np.ceil(device), np.ceil(a / f32(132)))
