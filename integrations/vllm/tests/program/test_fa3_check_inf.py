"""FA3's per-iteration `Check_inf` (`registry.fa3_check_inf`): the block Definition against FA3's `max_get_scale` / `fwd_step`, the tile's
`n_block_min_causal_local_mask`, the construction selector (the default; `guard-every-block` declared keeps `Attention_v2`), the twin,
the partition rule, and v20 (v4 with every step a Call) computing v4's function with no dead gate."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tests.program import test_dead_free as TD

from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import runs
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.registry import b1, hopper
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import attention_vocab
from verity_vllm.program.frontend import targets as TG

NEG_INF, NAN = 0xFF800000, 0x7FFFFFFF
FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
H100 = {"compute_capability": (9, 0), "num_sms": 132, "flash_attn_version": 3}


def _bf16(rng, shape):
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_masked_from_is_the_tiles_causal_mask_boundary():
    # decode (one query row, T keys): only the first block, ceil(T/BN) - 1, is a Check_inf iteration
    assert [attention_vocab.fa3_masked_from(T, 128, 0, 64, 4) for T in (1, 128, 129, 287)] == [0, 0, 1, 2]
    # prefill rows (sq = sk): tile m_block = 4r // 128 starts at query row 32 m_block; blocks from its first row's diagonal are masked
    assert [attention_vocab.fa3_masked_from(r + 1, 128, r, 128, 4) for r in (0, 31, 32, 127, 128, 159, 160, 286)] == [0, 0, 0, 0, 1, 1, 1, 2]
    # the row's first block (ceil(T/BN) - 1) is always at or above it: the first iteration runs Check_inf
    for kBM, g, BN in ((64, 4, 192), (192, 4, 128), (128, 8, 128), (64, 1, 128)):
        for sq, sk in ((1, 300), (130, 130), (17, 300), (64, 164)):
            for r in range(sq):
                T = r + 1 + sk - sq
                assert attention_vocab.fa3_masked_from(T, BN, r, kBM, g) <= -(-T // BN) - 1
    with pytest.raises(ValueError, match="does not divide kBlockM"):
        attention_vocab.fa3_masked_from(10, 128, 3, 64, 7)
    assert TG.fa3_masked_from(TargetProfile(**H100), 130, 32, 8, 128, 130, 129) == attention_vocab.fa3_masked_from(130, 128, 129, 128, 4)


def _block(check: bool, first: bool = False, D: int = 16, BN: int = 16):
    return bind(F3.AttnBlockV4, D=D, NVIS=BN, FIRST=first, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, CHECK=check)


def test_an_unguarded_block_reads_the_running_max_as_the_kernel_does():
    """A row whose every score is -inf, at a block after a -inf running max: FA3's unmasked iteration gives max_scaled = -inf, so its P and
    rescale are NaN (l, o NaN); a Check_inf iteration (and AttnBlock_v2) maps the max to 0 (l, o = 0)."""
    D, BN = 16, 16
    q = [0] * D
    q[0] = 0x7F80                                                         # +inf against k[..., 0] = -1: every score -inf
    k = ([0xBF80] + [0x3F80] * (D - 1)) * BN
    v = [0x3F80] * (BN * D)
    off = evaluate(_block(False), q, k, v, NEG_INF, [0] * 4, [0] * D)
    on = evaluate(_block(True), q, k, v, NEG_INF, [0] * 4, [0] * D)
    v2 = evaluate(bind(b1.AttnBlockV2, D=D, NVIS=BN, FIRST=False, BN=BN, DOT=hopper.HopperBF16WgmmaDot16), q, k, v, NEG_INF, [0] * 4, [0] * D)
    assert off[0] == on[0] == NEG_INF and set(off[1:]) == {NAN} and set(on[1:]) == {0} and on == v2


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
    head = bind(F3.AttentionHeadV4, T=40, D=16, BN=16, MASKED_FROM=3, **FA3)                # blocks 0..2: 3 would leave the first unguarded
    with pytest.raises(ValueError, match="MASKED_FROM 3 outside blocks"):
        evaluate(head, [0] * 16, [0] * 640, [0] * 640)


@pytest.mark.parametrize("masked_from", [2, 1, 0])
def test_the_twin_is_the_head_definition(masked_from):
    rng = np.random.default_rng(masked_from)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D), _bf16(rng, (T, D)), _bf16(rng, (T, D))
    ref = evaluate(bind(F3.AttentionHeadV4, T=T, D=D, BN=BN, MASKED_FROM=masked_from, **FA3), q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist())
    st = DR.attention_head_states(q[None], k[None], v[None], BN, dot16=DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"],
                                  inv=DR.INV_SUM_BY_PRIM["Fa3InvSum_v1"], masked_from=masked_from)
    assert not st["nonfinite"][0] and st["out"][0].tolist() == list(ref)


def test_guard_every_block_declared_keeps_attention_v2():
    tp = TargetProfile(**H100, fa3_construction="guard-every-block")
    assert tp.to_json()["fa3_construction"] == "guard-every-block" and tp.digest() == TargetProfile.from_json(tp.to_json()).digest()
    assert tp.effective_fa3_construction() == "guard-every-block" and not TG.fa3_check_inf(tp)
    spec = TG.attention_spec(tp, 287, 32, 8, 128, 1)
    assert spec.id == bind(b1.AttentionV2, T=287, NH=32, KVH=8, D=128, BN=128, **FA3).id
    assert TG.attention_spec(tp, 287, 32, 8, 128, 1, row=0).id == spec.id                  # a row index is ignored off the construction
    assert TG.describe(tp)["attention_kind"].startswith("Attention_v2{")
    old = SimpleNamespace(effective_flash_attn_version=lambda: 3)                         # a profile object without the knob
    assert not TG.fa3_check_inf(old)


def test_the_default_binds_attention_v4_on_fa3_only():
    on = TargetProfile(**H100)
    assert "fa3_construction" not in on.to_json() and on.effective_fa3_construction() == "check-inf-per-iteration" and TG.fa3_check_inf(on)
    assert TargetProfile(**H100, fa3_construction="check-inf-per-iteration").digest() != on.digest()
    spec = TG.attention_spec(on, 287, 32, 8, 128, 1, row=0)                               # a decode row: only the first block guarded
    assert spec.id == bind(F3.AttentionV4, T=287, NH=32, KVH=8, D=128, BN=128, MASKED_FROM=2, **FA3).id
    with pytest.raises(TG.UnregisteredTarget, match="row's index"):
        TG.attention_spec(on, 287, 32, 8, 128, 1)
    assert TG.describe(on)["attention_kind"].startswith("Attention_v4{")
    fa2 = TargetProfile(**dict(H100, flash_attn_version=2))
    assert TG.attention_spec(fa2, 287, 32, 8, 64, 1, row=0).id.startswith("Attention_v2{")  # forced FA2 on Hopper guards every block
    with pytest.raises(ValueError, match="fa3_construction must be"):
        TargetProfile(**H100, fa3_construction="per-block")


def test_the_builder_kind_binds_v4_only_with_masked_from():
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V

    base = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, **FA3}
    assert V.kind("attention_dot", **base).id == bind(b1.AttentionV2, **base).id
    assert V.kind("attention_dot", **base, MASKED_FROM=2).id == bind(F3.AttentionV4, **base, MASKED_FROM=2).id
    assert "Attention_v4" not in V.to_json()["roles"]["attention_dot"]                  # the role's note, and the vocabulary version, as before


def test_the_partition_rule_commits_the_running_max_where_the_block_is_unguarded():
    from verity.ml.definitions import word as W

    statics = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, "DOT": {"fn": "HopperBF16WgmmaDot16_v1"}, "INV": {"fn": "Fa3InvSum_v1"}}
    for mf, guards in ((2, 0), (0, 4)):
        r = W.unit_rule(W.specialization("Attention_v4", dict(statics, MASKED_FROM=mf)))
        assert r["violations"] == [] and r["max_out_bits"] <= 32
        heads = {}
        for i in r["committed_interior"]:
            heads[i["head"]] = heads.get(i["head"], 0) + i["words"]
        assert heads.get("GuardNegInfZero_v1", 0) == guards and heads["F32MulFtz_v1"] == 6
    g = W.Graph(W.specialization("AttentionHead_v4", {k: v for k, v in dict(statics, MASKED_FROM=2).items() if k not in ("NH", "KVH")}))
    R = W.units(g, 16, 32)
    assert R["cut"].ok and not g.recomputed and bool(R["ok"].all())


def _all_neg_inf_scores(T: int, D: int = 16) -> tuple[list[int], list[int], list[int]]:
    """q = (+inf, 0, ...) against keys (-1, 1, ...): every score -inf, so the running max is still -inf at every block below the
    first, where an unguarded FA3 block reads it as it is."""
    return [0x7F80] + [0] * (D - 1), ([0xBF80] + [0x3F80] * (D - 1)) * T, [0x3F80] * (T * D)


@pytest.mark.parametrize("T,BN,MF", [(1, 32, 0), (16, 32, 0), (17, 32, 0), (32, 32, 0), (40, 32, 1), (40, 32, 0), (48, 32, 1),
                                     (96, 32, 0), (96, 32, 2), (17, 16, 1)])
def test_attention_head_v20_equals_v4(T, BN, MF):
    """A first block of 1 ... BN keys, its V rows padded (40) or not (48), blocks below it guarded or not, on `test_dead_free`'s
    attention cases and on rows whose every score is -inf: NaN out of an unguarded block below MASKED_FROM, 0 where every block is
    guarded."""
    new, old = (bind(fn, T=T, D=16, BN=BN, MASKED_FROM=MF, **FA3) for fn in (F3.AttentionHeadV20, F3.AttentionHeadV4))
    assert TD._dead(new) == 0
    for q, k, v in TD._attention_cases(16, T) + [_all_neg_inf_scores(T)]:
        TD._same(new, old, q, k, v)
    assert set(evaluate(new, *_all_neg_inf_scores(T))) == ({0x7FFF} if MF else {0})


def test_attention_v20_equals_v4():
    new, old = (bind(fn, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, **FA3) for fn in (F3.AttentionV20, F3.AttentionV4))
    assert TD._dead(new) == 0
    rng = np.random.default_rng(20)
    for _ in range(2):
        q, kc, vc = (_bf16(rng, n).tolist() for n in (4 * 16, 48 * 2 * 16, 48 * 2 * 16))
        TD._same(new, old, q, kc, vc)


@pytest.mark.parametrize("T", [1, 16, 17, 191, 192, 193, 384, 385, 1087])
def test_attention_head_v20_is_v4s_calls_at_llama_3_2_1b_on_sm_90(T):
    """Llama-3.2-1B's heads on the H200 (D 64, FA3's one-MMA-warpgroup kBlockN 192), at every MASKED_FROM's extremes: v20's body is
    Calls only, its first visited block `AttnBlockFirst_v1` at v4's first block's statics, each block below it v4's own Call of
    `AttnBlock_v4` over the same query, keys and values, and its end `AttnHeadEnd_v1` at FA3's reciprocal."""
    BN = 192
    nb = -(-T // BN)
    for MF in {0, nb - 1}:
        new, old = (bind(fn, T=T, D=64, BN=BN, MASKED_FROM=MF, **FA3) for fn in (F3.AttentionHeadV20, F3.AttentionHeadV4))
        calls = new.body.nodes
        assert [(n.form, n.fn.base_id) for n in calls] == [("call", "AttnBlockFirst_v1")] + [("call", "AttnBlock_v4")] * (nb - 1) + [
            ("call", "AttnHeadEnd_v1")]
        blocks = [n for n in old.body.nodes if n.form == "call" and n.fn.base_id == "AttnBlock_v4"]
        first = dict(blocks[0].fn.bindings)
        assert (first.pop("FIRST"), first.pop("CHECK"), first) == (True, True, dict(calls[0].fn.bindings))
        assert [n.fn.id for n in calls[1:-1]] == [n.fn.id for n in blocks[1:]]
        assert [[runs(a.refs) for a in n.args[:3]] for n in calls[1:-1]] == [[runs(a.refs) for a in n.args[:3]] for n in blocks[1:]]
        assert calls[-1].fn.id == "AttnHeadEnd_v1{D=64,INV=Fa3InvSum_v1}"
