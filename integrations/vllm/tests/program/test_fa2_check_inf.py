"""FA2's per-iteration `Check_inf` (`registry.fa2_check_inf`): the block Definition against FA2's `softmax_rescale_o` / `scale_apply_exp2`,
the masking-step boundary, the target binding (`blackwell_consumer` only; the accepted FA2 targets keep their chains), the builder kind,
the twin and the partition rule."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import b1, hopper
from verity_vllm.program.registry import fa2_check_inf as F2
from verity_vllm.program.registry import attention_vocab
from verity_vllm.program.registry import fa3_check_inf as F3
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend import targets as TG

NEG_INF, NAN = 0xFF800000, 0x7FFFFFFF
FA2 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": P.Fa2InvSum}
PRO6000 = TargetProfile(compute_capability=(12, 0), num_sms=188)


def _bf16(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _is_nan(w: int) -> bool:
    return (w & 0x7F800000) == 0x7F800000 and bool(w & 0x7FFFFF)


def test_masked_from_is_the_tiles_masking_steps():
    # non-causal launch (one query row per sequence): only the first block, ceil(sk/BN) - 1
    assert [attention_vocab.fa2_masked_from(sk, 1, 0, 64, False) for sk in (1, 64, 65, 257)] == [0, 0, 1, 4]
    # causal prefill (sq = sk), BN 64: the 64-row tile's n_block_max less its two masking steps
    assert [attention_vocab.fa2_masked_from(130, 130, r, 64, True) for r in (0, 63, 64, 127, 128, 129)] == [0, 0, 0, 0, 1, 1]
    # a chunked prefill's tile is capped by the sequence's keys: 17 rows over 300 keys guard blocks 3 and 4 (not 4 and 5)
    assert attention_vocab.fa2_masked_from(300, 17, 16, 64, True) == 3
    # BN 128 (head dim 64): ceil(64/128) + 1 = 2 masking steps as well
    assert [attention_vocab.fa2_masked_from(sk, sk, sk - 1, 128, True) for sk in (1, 128, 129, 257, 385)] == [0, 0, 0, 1, 2]
    # the row's first block is always at or above it: the first iteration runs Check_inf
    for BN in (64, 128):
        for sq, sk in ((1, 300), (130, 130), (17, 300), (64, 164), (200, 1031)):
            for r in range(sq):
                T = r + 1 + sk - sq
                assert attention_vocab.fa2_masked_from(sk, sq, r, BN, sq > 1) <= -(-T // BN) - 1
    with pytest.raises(ValueError, match="non-causal launch has one query row"):
        attention_vocab.fa2_masked_from(300, 2, 0, 64, False)
    assert TG.fa2_masked_from(PRO6000, 16 + 1 + 283, 128, 17, 16, 17) == attention_vocab.fa2_masked_from(300, 17, 16, 64, True)


def _block(check: bool, first: bool = False, D: int = 16, BN: int = 16, block=F2.AttnBlockV5):
    return bind(block, D=D, NVIS=BN, FIRST=first, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, CHECK=check)


def test_an_unguarded_block_leaves_only_the_rescale_unguarded():
    """A row whose every score is -inf, at a block after a -inf running max: FA2's unmasked iteration rescales by exp2(NaN) (l, o NaN) but
    still guards `max * scale`, so its P is exp2(-inf) = 0; FA3's (`AttnBlock_v4`) reads the raw max there too, so its P is NaN.  A
    Check_inf iteration is AttnBlock_v2."""
    D, BN = 16, 16
    q = [0] * D
    q[0] = 0x7F80                                                         # +inf against k[..., 0] = -1: every score -inf
    k = ([0xBF80] + [0x3F80] * (D - 1)) * BN
    v = [0x3F80] * (BN * D)
    off = evaluate(_block(False), q, k, v, NEG_INF, [0] * 4, [0] * D)
    on = evaluate(_block(True), q, k, v, NEG_INF, [0] * 4, [0] * D)
    v2 = evaluate(bind(b1.AttnBlockV2, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=hopper.HopperBF16WgmmaDot16), q, k, v, NEG_INF, [0] * 4, [0] * D)
    assert off[0] == on[0] == NEG_INF and all(_is_nan(w) for w in off[1:]) and set(on[1:]) == {0} and on == v2
    scale = b1._scale_log2_bits(D)
    p = {name: int(evaluate(P.MufuEx2Ftz, int(evaluate(P.F32FmaSubFtz, NEG_INF, scale, max_scaled)[0]))[0])
         for name, max_scaled in (("fa2", int(evaluate(P.F32MulFtz, int(evaluate(P.GuardNegInfZero, NEG_INF)[0]), scale)[0])),
                                  ("fa3", int(evaluate(P.F32MulFtz, NEG_INF, scale)[0])))}
    assert p == {"fa2": 0, "fa3": NAN}


def test_on_finite_rows_every_construction_is_v2():
    rng = np.random.default_rng(3)
    D, BN = 16, 16
    q, k, v = _bf16(rng, D).tolist(), _bf16(rng, BN * D).tolist(), _bf16(rng, BN * D).tolist()
    m, l, o = 0x3F000000, [0x3F800000] * 4, _bf16(rng, D).astype(np.uint32).tolist()
    v2 = evaluate(bind(b1.AttnBlockV2, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=hopper.HopperBF16WgmmaDot16), q, k, v, m, l, o)
    assert evaluate(_block(True), q, k, v, m, l, o) == evaluate(_block(False), q, k, v, m, l, o) == v2


def test_the_first_block_must_run_check_inf():
    with pytest.raises(ValueError, match="FIRST needs CHECK"):
        evaluate(_block(False, first=True), [0] * 16, [0] * 256, [0] * 256, 0, [0] * 4, [0] * 16)
    head = bind(F2.AttentionHeadV5, T=40, D=16, BN=16, MASKED_FROM=3, **FA2)                # blocks 0..2: 3 would leave the first unguarded
    with pytest.raises(ValueError, match="MASKED_FROM 3 outside blocks"):
        evaluate(head, [0] * 16, [0] * 640, [0] * 640)


def test_masked_from_zero_is_attention_v2_and_an_all_neg_inf_row_is_nan():
    rng = np.random.default_rng(5)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    q[0], k[:, 0] = 0x7F80, 0xBF80                                        # every score -inf
    args = (q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist())
    v2 = evaluate(bind(b1.AttentionHeadV2, T=T, D=D, BN=BN, **FA2), *args)
    assert evaluate(bind(F2.AttentionHeadV5, T=T, D=D, BN=BN, MASKED_FROM=0, **FA2), *args) == v2 and set(v2) == {0}
    assert set(evaluate(bind(F2.AttentionHeadV5, T=T, D=D, BN=BN, MASKED_FROM=2, **FA2), *args)) == {0x7FFF}


@pytest.mark.parametrize("masked_from", [2, 1, 0])
def test_the_twin_is_the_head_definition(masked_from):
    rng = np.random.default_rng(masked_from)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    ref = evaluate(bind(F2.AttentionHeadV5, T=T, D=D, BN=BN, MASKED_FROM=masked_from, **FA2), q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist())
    st = DR.attention_head_states(q[None], k[None], v[None], BN, dot16=DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"],
                                  inv=DR.INV_SUM_BY_PRIM["Fa2InvSum_v1"], masked_from=masked_from, rescale_only=True)
    assert not st["nonfinite"][0] and st["out"][0].tolist() == list(ref)


def test_the_target_binds_attention_v5_on_blackwell_consumer_fa2_only():
    assert TG.fa2_check_inf(PRO6000) and TG.describe(PRO6000)["attention_kind"].startswith("Attention_v5{")
    spec = TG.attention_spec(PRO6000, 287, 32, 8, 128, 1, row=0, own_rows=1)                # a decode row: only the first block guarded
    assert spec.id == bind(F2.AttentionV5, T=287, NH=32, KVH=8, D=128, BN=64, MASKED_FROM=4, **FA2).id
    with pytest.raises(TG.UnregisteredTarget, match="row's index"):
        TG.attention_spec(PRO6000, 287, 32, 8, 128, 1, row=0)
    for tp in (TargetProfile(), TargetProfile(compute_capability=(8, 0), num_sms=108),
               TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=2)):
        assert not TG.fa2_check_inf(tp)
    assert TG.attention_spec(TargetProfile(), 287, 32, 8, 128, 1, row=0, own_rows=1).id.startswith("Attention_v3{")
    h100_fa2 = TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=2)
    assert TG.attention_spec(h100_fa2, 287, 32, 8, 64, 1, row=0, own_rows=1).id.startswith("Attention_v2{")
    h100 = TargetProfile(compute_capability=(9, 0), num_sms=132)
    assert TG.attention_spec(h100, 287, 32, 8, 128, 1, row=0, own_rows=1).id == TG.attention_spec(h100, 287, 32, 8, 128, 1, row=0).id


def test_the_builder_kind_binds_v5_only_with_fa2_masked_from():
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V

    base = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, **FA2}
    assert V.kind("attention_dot", **base).id == bind(b1.AttentionV2, **base).id
    assert V.kind("attention_dot", **base, FA2_MASKED_FROM=2).id == bind(F2.AttentionV5, **base, MASKED_FROM=2).id
    assert V.kind("attention_dot", **base, MASKED_FROM=2).id == bind(F3.AttentionV4, **base, MASKED_FROM=2).id
    assert "Attention_v5" not in V.to_json()["roles"]["attention_dot"]                  # the role's note, and the vocabulary version, as before


def test_the_partition_rule():
    from verity_vllm.query import word as W

    statics = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, "DOT": {"fn": "HopperBF16WgmmaDot16_v1"}, "INV": {"fn": "Fa2InvSum_v1"}}
    for mf, guards in ((2, 0), (0, 4)):
        r = W.unit_rule(W.specialization("Attention_v5", dict(statics, MASKED_FROM=mf)))
        assert r["violations"] == [] and r["max_out_bits"] <= 32
        heads = {}
        for i in r["committed_interior"]:
            heads[i["head"]] = heads.get(i["head"], 0) + i["words"]
        assert heads.get("GuardNegInfZero_v1", 0) == guards and heads["F32MulFtz_v1"] == 6
    g = W.Graph(W.specialization("AttentionHead_v5", {k: v for k, v in dict(statics, MASKED_FROM=2).items() if k not in ("NH", "KVH")}))
    R = W.units(g, 16, 32)
    assert R["cut"].ok and not g.recomputed and bool(R["ok"].all())
