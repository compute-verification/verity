"""FA2's softcapped chain under per-iteration `Check_inf` (`registry.fa2_softcap`): the Definition against the accepted softcapped chain
and `Attention_v5`, the capture's twin, the target binding, the builder kind and the partition rule."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import b1, dense, hopper
from verity_vllm.program.registry import fa2_check_inf as F2
from verity_vllm.program.registry import fa2_softcap as F2S
from verity_vllm.program.registry import prims as P
from verity_vllm.program.frontend import targets as TG

FA2 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": P.Fa2InvSum}
PRO6000 = TargetProfile(compute_capability=(12, 0), num_sms=188)
CAP = 50.0


def _bf16(rng, shape, spread: float = 1.0):
    return ((rng.standard_normal(shape) * spread).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _args(q, k, v):
    return q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist()


def test_every_block_guarded_on_the_ampere_step_is_the_accepted_softcap_chain():
    rng = np.random.default_rng(11)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D, 30.0), _bf16(rng, (T, D), 30.0), _bf16(rng, (T, D))
    v1 = evaluate(bind(dense.AttentionHeadSoftcap, T=T, D=D, BN=BN, CAP=CAP), *_args(q, k, v))
    v2 = evaluate(bind(F2S.AttentionHeadSoftcapV2, T=T, D=D, BN=BN, CAP=CAP, DOT=P.AmpereBF16TcDot16, INV=P.Fa2InvSum, MASKED_FROM=0),
                  *_args(q, k, v))
    assert v2 == v1


def test_the_softcap_moves_the_scores_before_the_row_max():
    rng = np.random.default_rng(12)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D, 30.0), _bf16(rng, (T, D), 30.0), _bf16(rng, (T, D))
    capped = evaluate(bind(F2S.AttentionHeadSoftcapV2, T=T, D=D, BN=BN, CAP=CAP, MASKED_FROM=1, **FA2), *_args(q, k, v))
    plain = evaluate(bind(F2.AttentionHeadV5, T=T, D=D, BN=BN, MASKED_FROM=1, **FA2), *_args(q, k, v))
    assert capped != plain


def test_statics_are_checked():
    with pytest.raises(ValueError, match="FIRST needs CHECK"):
        evaluate(bind(F2S.AttnBlockSoftcapV2, D=16, NVIS=16, FIRST=True, BN=16, CAP=CAP, DOT=FA2["DOT"], CHECK=False),
                 [0] * 16, [0] * 256, [0] * 256, 0, [0] * 4, [0] * 16)
    with pytest.raises(ValueError, match="MASKED_FROM 3 outside blocks"):
        evaluate(bind(F2S.AttentionHeadSoftcapV2, T=40, D=16, BN=16, CAP=CAP, MASKED_FROM=3, **FA2), [0] * 16, [0] * 640, [0] * 640)
    with pytest.raises(ValueError, match="positive softcap"):
        evaluate(bind(F2S.AttnBlockSoftcapV2, D=16, NVIS=16, FIRST=True, BN=16, CAP=0.0, DOT=FA2["DOT"], CHECK=True),
                 [0] * 16, [0] * 256, [0] * 256, 0, [0] * 4, [0] * 16)


@pytest.fixture(scope="module")
def tanh_table():
    from tests.properties import fa2_softcap_capture_gpu as S

    words, lo, rec = S.tanh_table()
    assert rec["ok"]
    return words, lo


def test_the_vectorised_tanh_is_mufu_tanh(tanh_table):
    from tests.properties import fa2_softcap_capture_gpu as S

    words, lo = tanh_table
    rng = np.random.default_rng(13)
    x = np.concatenate([rng.integers(0, 1 << 32, 300, dtype=np.uint64).astype(np.uint32),
                        (rng.standard_normal(3000).astype(np.float32) * 0.05).view(np.uint32),
                        np.array([0, 0x80000000, 0x3B800000, 0x3B800001, 0x40FFFFFF, 0x41000000, 0x7F800000, 0xFF800000, 0x7FC00000,
                                  0xFFC00001, 0xC0FFFFFF], dtype=np.uint32)])
    got = S.mufu_tanh(x, words, lo)
    assert [int(y) for y in got] == [P.MufuTanh.evaluate(int(w)) for w in x]


@pytest.mark.parametrize("masked_from,special", [(2, None), (1, None), (0, None), (1, "+inf"), (1, "nan")])
def test_the_capture_twin_is_the_head_definition(tanh_table, masked_from, special):
    from tests.properties import fa2_softcap_capture_gpu as S

    words, lo = tanh_table
    rng = np.random.default_rng(masked_from)
    T, D, BN = 40, 16, 16
    q, k, v = _bf16(rng, D, 30.0), _bf16(rng, (T, D), 30.0), _bf16(rng, (T, D))
    if special:
        q[0], k[:, 0] = {"+inf": 0x7F80, "nan": 0x7FC0}[special], 0xBF80
    ref = evaluate(bind(F2S.AttentionHeadSoftcapV2, T=T, D=D, BN=BN, CAP=CAP, MASKED_FROM=masked_from, **FA2), *_args(q, k, v))
    st = S.softcap_head_states(q[None], k[None], v[None], BN, cap=CAP, masked_from=masked_from,
                               dot16=DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"], inv=DR.INV_SUM_BY_PRIM["Fa2InvSum_v1"],
                               tanh=lambda x: S.mufu_tanh(x, words, lo))
    assert bool(st["nonfinite"][0]) == (special is not None)
    if special is None:                                                    # a flagged head is the reference evaluator's (`compare`)
        assert st["out"][0].tolist() == list(ref)


def test_the_capture_compare_accepts_the_definitions_own_rows(tanh_table, tmp_path):
    from tests.properties import fa2_softcap_capture_gpu as S

    words, lo = tanh_table
    ref = tmp_path / "tanh.bin"
    words.tofile(ref)
    rng = np.random.default_rng(14)
    H, Hk, D, sq, sk = 2, 1, 16, 2, 20
    blocks = 4
    q = _bf16(rng, (sq, H, D), 30.0)
    q[1, :, 0] = 0x7FC0                                                    # the last row NaN: the reference path
    k, v = _bf16(rng, (blocks, 16, Hk, D), 30.0), _bf16(rng, (blocks, 16, Hk, D))
    bt = np.array([[2, 0]], dtype=np.int32)
    keys = np.arange(sk)
    kseq, vseq = k[bt[0, keys // 16], keys % 16], v[bt[0, keys // 16], keys % 16]
    BN = b1.fa2_kblock_n(D)
    out = np.zeros((sq, H, D), dtype=np.uint16)
    for r in range(sq):
        T = r + 1 + sk - sq
        spec = bind(F2S.AttentionSoftcapV2, T=T, NH=H, KVH=Hk, D=D, BN=BN, CAP=CAP, MASKED_FROM=S.C.guarded_from(sq, sq, sk, r, BN), **FA2)
        out[r] = np.asarray(evaluate(spec, q[r].reshape(-1).tolist(), kseq[:T].reshape(-1).tolist(), vseq[:T].reshape(-1).tolist()),
                            dtype=np.uint16).reshape(H, D)
    job = {"case": {"name": "unit", "H": H, "Hk": Hk, "seqlens_q": [sq], "seqlens_k": [sk], "softcap": CAP}, "D": D, "q": q, "k": k,
           "v": v, "out": out, "block_table": bt, "cc": [12, 0], "sms": 188, "tanh_table": str(ref), "tanh_lo": lo}
    rec = S.compare(job)
    assert rec["rows"] == sq and rec["nonfinite_heads"] == H and rec["reference_heads"] == H
    assert rec["mismatch"] == rec["twin_mismatch"] == rec["binding_mismatch"] == 0
    out[0, 1, 3] ^= 1
    assert S.compare({**job, "out": out})["mismatch_finite"] == 1


@pytest.mark.parametrize("masked_from", [2, 0])
def test_the_replay_row_kernel_is_the_definition(masked_from):
    from verity_vllm.program.kernels import rows as R
    from verity_vllm.program.kernels import softcap_rows as SR

    rng = np.random.default_rng(20 + masked_from)
    T, NH, KVH, D, BN = 40, 4, 2, 16, 16
    q, k, v = _bf16(rng, NH * D, 30.0), _bf16(rng, (T, KVH * D), 30.0), _bf16(rng, (T, KVH * D))
    spec = bind(F2S.AttentionSoftcapV2, T=T, NH=NH, KVH=KVH, D=D, BN=BN, CAP=CAP, MASKED_FROM=masked_from, **FA2)
    statics = {"T": T, "NH": NH, "KVH": KVH, "D": D, "BN": BN, "CAP": CAP, "MASKED_FROM": masked_from,
               "DOT": "HopperBF16WgmmaDot16_v1", "INV": "Fa2InvSum_v1"}
    got = R.attention_softcap(statics, [q, k.reshape(-1), v.reshape(-1)])["out"]
    assert got.tolist() == list(evaluate(spec, *_args(q, k, v)))
    from verity_vllm.query.manifest.format import spec_statics
    as_replayed = spec_statics(spec.id)                          # the replay's statics: CAP stays `{"f64":"<hex>"}` text
    assert R.attention_softcap(as_replayed, [q, k.reshape(-1), v.reshape(-1)])["out"].tolist() == got.tolist()
    q[0] = 0x7FC0
    with pytest.raises(SR.Declined, match="non-finite"):
        R.attention_softcap(statics, [q, k.reshape(-1), v.reshape(-1)])
    with pytest.raises(SR.Declined, match="no registered evaluator"):
        R.attention_softcap(dict(statics, INV="Fa3InvSum_v2"), [q, k.reshape(-1), v.reshape(-1)])


def test_the_target_binds_attention_softcap_v2_under_fa2_check_inf_only():
    spec = TG.attention_spec(PRO6000, 287, 8, 4, 256, 1, row=0, own_rows=1, cap=CAP)
    assert spec.id == bind(F2S.AttentionSoftcapV2, T=287, NH=8, KVH=4, D=256, BN=64, CAP=CAP, MASKED_FROM=4, **FA2).id
    with pytest.raises(TG.UnregisteredTarget, match="check-inf-per-iteration: the row's index .* not known"):
        TG.attention_spec(PRO6000, 287, 8, 4, 256, 1, row=0, cap=CAP)
    assert TG.attention_spec(TargetProfile(), 287, 8, 4, 256, 1, cap=CAP).id == \
        bind(dense.AttentionSoftcap, T=287, NH=8, KVH=4, D=256, BN=64, CAP=CAP).id
    h100 = TargetProfile(compute_capability=(9, 0), num_sms=132)
    with pytest.raises(TG.UnregisteredTarget):
        TG.attention_spec(h100, 287, 8, 4, 256, 1, row=0, own_rows=1, cap=CAP)
    plain = TG.attention_spec(PRO6000, 287, 8, 4, 256, 1, row=0, own_rows=1)
    assert plain.id.startswith("Attention_v5{") and TG.softcapped(plain, CAP) is spec       # the fold's softcap scope re-binds through it
    with pytest.raises(TG.UnregisteredTarget, match="Attention_v10 has no registered softcapped chain"):
        TG.softcapped(TG.attention_spec(TargetProfile(), 287, 8, 4, 256, 1, scale_input=True), CAP)


def test_the_builder_kind_binds_the_softcap_chain_only_with_fa2_masked_from():
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V

    before = V.version
    base = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, **FA2}
    assert V.kind("attention_dot", **base, FA2_MASKED_FROM=2, CAP=CAP).id == \
        bind(F2S.AttentionSoftcapV2, **base, CAP=CAP, MASKED_FROM=2).id
    assert V.kind("attention_dot", **base, FA2_MASKED_FROM=2).id == bind(F2.AttentionV5, **base, MASKED_FROM=2).id
    assert V.kind("attention_dot", **base, CAP=CAP) is None
    assert "Softcap" not in V.to_json()["roles"]["attention_dot"] and V.version == before


def test_the_partition_rule():
    from verity_vllm.query import word as W

    statics = {"T": 40, "NH": 2, "KVH": 1, "D": 16, "BN": 16, "CAP": CAP, "DOT": {"fn": "HopperBF16WgmmaDot16_v1"},
               "INV": {"fn": "Fa2InvSum_v1"}}
    for mf in (2, 0):
        r = W.unit_rule(W.specialization("AttentionSoftcap_v2", dict(statics, MASKED_FROM=mf)))
        assert r["violations"] == [] and r["max_out_bits"] <= 32
    g = W.Graph(W.specialization("AttentionHeadSoftcap_v2", {k: v for k, v in dict(statics, MASKED_FROM=2).items() if k not in ("NH", "KVH")}))
    R = W.units(g, 16, 32)
    assert R["cut"].ok and not g.recomputed and bool(R["ok"].all())
