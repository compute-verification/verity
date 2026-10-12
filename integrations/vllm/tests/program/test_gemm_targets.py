"""[R13 ampere] The GEMM target registry (`verity_vllm.program.frontend.gemm_targets`): Definition selection per arch family through the ONE
`TargetProfile`, fail-closed refusal of unregistered geometry/targets, and Bet A's pins following the registry."""
from __future__ import annotations

import pytest

from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile, profile_mismatches
from verity_vllm.program.frontend import gemm_targets as GT

AMPERE_TP = TargetProfile(compute_capability=(8, 0), num_sms=108)
HOPPER_TP = TargetProfile(compute_capability=(9, 0), num_sms=132)
PRO6000_TP = TargetProfile(compute_capability=(12, 0), num_sms=188)


def test_registered_targets_are_ada_ampere_hopper_and_blackwell_consumer():
    """ada + ampere share the Ampere-family k16 step (Gemm_v1); hopper [R13 hopper, HOP-01] is the wgmma step (Gemm_v2{K,N,DOT})."""
    assert set(GT.REGISTERED) == {"ada", "ampere", "hopper", "blackwell_consumer"}
    assert GT.REGISTERED["ada"].compute_capabilities == ((8, 9),) and GT.REGISTERED["ada"].vllm_arch_family == "ada"
    assert GT.REGISTERED["ampere"].compute_capabilities == ((8, 0),) and GT.REGISTERED["ampere"].vllm_arch_family is None
    assert {GT.REGISTERED[k].dot for k in ("ada", "ampere")} == {"AmpereBF16TcDot16"} and {GT.REGISTERED[k].kind for k in ("ada", "ampere")} == {"Gemm_v1{K,N}"}
    assert GT.REGISTERED["ampere"].tuned_table == {} and GT.REGISTERED["ampere"].table_sha256() is None
    assert GT.REGISTERED["ada"].tuned_table is GT.ADA_TUNED_GEMM and GT.REGISTERED["ada"].table_sha256()
    hop = GT.REGISTERED["hopper"]
    assert hop.compute_capabilities == ((9, 0),) and hop.vllm_arch_family == "hopper" and hop.dot == "HopperBF16WgmmaDot16"
    assert hop.kind == "Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}" and len(hop.tuned_table) == 5 and hop.table_sha256()
    consts, warps, stages, src = hop.launch_config(16, 2048, 2048)                   # the Hopper tile the H100 rows launch
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_K"]) == (32, 128) and "tuned_hopper" in src
    assert hop.launch_config(1, 2048, 2048)[0]["BLOCK_SIZE_M"] == 16
    assert HOPPER_TP.gemm_target().key == "hopper" and HOPPER_TP.tuned_gemm_arch_family() == "hopper"
    assert all(t.evidence for t in GT.REGISTERED.values())


def test_arch_family_rule_is_the_pinned_vllm_rule():
    assert GT.vllm_tuned_gemm_arch_family((8, 9)) == "ada"
    assert GT.vllm_tuned_gemm_arch_family((8, 0)) is None and GT.vllm_tuned_gemm_arch_family((8, 6)) is None
    assert GT.vllm_tuned_gemm_arch_family((9, 0)) == "hopper" and GT.vllm_tuned_gemm_arch_family((10, 0)) == "blackwell"
    assert GT.vllm_tuned_gemm_arch_family((7, 5)) is None and GT.vllm_tuned_gemm_arch_family((12, 0)) is None
    vllm = pytest.importorskip("vllm.model_executor.determinism.batch_invariant_configs")
    from vllm.platforms.interface import DeviceCapability
    for cc in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (7, 5), (12, 0)):
        assert vllm._get_tuned_matmul_arch_family(DeviceCapability(*cc)) == GT.vllm_tuned_gemm_arch_family(cc)


def test_selection_per_target_profile():
    assert DEFAULT_TARGET.gemm_target().key == "ada" and DEFAULT_TARGET.tuned_gemm_arch_family() == "ada"
    assert AMPERE_TP.gemm_target().key == "ampere" and AMPERE_TP.tuned_gemm_arch_family() is None
    assert TargetProfile.from_gpu_document({"multi_processor_count": 108, "compute_capability": "8.0"}).gemm_target().key == "ampere"
    assert TargetProfile.from_gpu_document({"multi_processor_count": 142, "compute_capability": [8, 9]}).gemm_target().key == "ada"


def test_sm120_is_the_default_tiling_on_the_hopper_k16_step():
    """cc 12.0 (the RTX PRO 6000): vLLM has no tuned family for it, so a Triton launch takes the default tiling (the pinned vLLM keeps
    the bf16 linear on cuBLASLt there), and the k16 step is `mma.sync.m16n8k16`, whose sm_120 model is core's HOPPER_BF16_M16N8K16: the
    Definition is Hopper's `Gemm_v2`."""
    from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
    from verity.ml.circuits.silicon.models import HOPPER_BF16_M16N8K16, HOPPER_BF16_WGMMA_K16

    bw = GT.REGISTERED["blackwell_consumer"]
    assert bw.compute_capabilities == ((12, 0),) and bw.vllm_arch_family is None and GT.vllm_tuned_gemm_arch_family((12, 0)) is None
    assert bw.dot == GT.REGISTERED["hopper"].dot and bw.kind == GT.REGISTERED["hopper"].kind
    assert bw.tuned_table == {} and bw.table_sha256() is None and bw.evidence
    assert INSTRUCTIONS["sm120.mma.m16n8k16.bf16"].model is HOPPER_BF16_M16N8K16 and INSTRUCTIONS["sm120.mma.m16n8k16.bf16"].status == "pinned"
    assert [getattr(HOPPER_BF16_M16N8K16, f) for f in ("operand_bits", "groups", "width", "zero_exponent")] == \
           [getattr(HOPPER_BF16_WGMMA_K16, f) for f in ("operand_bits", "groups", "width", "zero_exponent")]
    for M, N, K in ((1, 2048, 2048), (32, 151936, 2048), (2049, 2048, 6144), (7, 3072, 2048)):     # tabled on ada/hopper, default here
        consts, warps, stages, src = bw.launch_config(M, N, K)
        assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], warps, stages) == (128, 128, 64, 8, 3)
        assert src == "default[blackwell_consumer]"
    assert PRO6000_TP.gemm_target().key == "blackwell_consumer" and PRO6000_TP.tuned_gemm_arch_family() is None
    assert TargetProfile.from_gpu_document({"multi_processor_count": 188, "compute_capability": "12.0"}).gemm_target() is bw
    with pytest.raises(GT.UnregisteredGemmTarget, match="tuned_gemm_arch_family='blackwell'.*requires None"):
        GT.gemm_target_for((12, 0), "blackwell")


def test_sm120_binds_the_hopper_gemm_and_hopper_step_fa2():
    """The sm_120 step family binds `Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}` and FA2 only: vLLM's FA2 on cc 12.0 is the driver's JIT
    of the compute_80 PTX on the Hopper step, so its attention is the H100 forced-FA2 chain (`BN=fa2_kblock_n(D)`,
    `DOT=HopperBF16WgmmaDot16_v1`, `INV=Fa2InvSum_v1`) with FA2's own `Check_inf` placement, `Attention_v5`; never the Ampere-step
    `Attention_v3`; a declared FA3 is refused by name."""
    from verity_vllm.program.registry import b1
    from verity_vllm.program.frontend import targets as TG

    assert PRO6000_TP.tensor_core_family == "blackwell_consumer" and PRO6000_TP.effective_flash_attn_version() == 2
    assert TargetProfile(compute_capability=(12, 1)).tensor_core_family == "unregistered:12.1"
    assert TG.tensor_core_dot(PRO6000_TP) is TG.tensor_core_dot(HOPPER_TP) and not TG.uses_accepted_kinds(PRO6000_TP)
    assert TG.gemm_spec(PRO6000_TP, 2048, 3072).id == TG.gemm_spec(HOPPER_TP, 2048, 3072).id
    hopper_fa2 = HOPPER_TP.with_overrides(flash_attn_version=2)
    for D in (64, 96, 128, 256):
        BN = b1.fa2_kblock_n(D)
        spec = TG.attention_spec(PRO6000_TP, 33, 32, 8, D, 1, row=0, own_rows=1)          # a decode row: only its first block guarded
        assert spec.id == (f"Attention_v5{{T=33,NH=32,KVH=8,D={D},BN={BN},DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1,"
                           f"MASKED_FROM={-(-33 // BN) - 1}}}")
        assert TG.attention_spec(hopper_fa2, 33, 32, 8, D, 1).id == \
               f"Attention_v2{{T=33,NH=32,KVH=8,D={D},BN={BN},DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1}}"
    assert TG.check_flash_attn_registered(PRO6000_TP) == 2
    with pytest.raises(TG.UnregisteredTarget, match="flash_attn_version 3 on blackwell_consumer"):
        TG.attention_spec(PRO6000_TP.with_overrides(flash_attn_version=3), 33, 32, 8, 64, 1)
    d = TG.describe(PRO6000_TP)
    assert d["gemm_target"] == "blackwell_consumer" and d["dot"] == "HopperBF16WgmmaDot16_v1" and d["tuned_gemm_arch_family"] is None
    assert d["gemm_kind"] == "Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}" and d["flash_attn_registered"] is True
    assert d["attention_kind"] == "Attention_v5{T,NH,KVH,D,BN=fa2_kblock_n(D),DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1,MASKED_FROM}"
    assert d["evidence"]["gemm"] == list(GT.REGISTERED["blackwell_consumer"].evidence) and d["evidence"]["attention_fa2"]


@pytest.mark.parametrize("cc, fam", [((10, 0), "blackwell"), ((8, 6), None), ((7, 5), None), ((12, 1), None)])
def test_unregistered_targets_are_refused_fail_closed(cc, fam):
    with pytest.raises(GT.UnregisteredGemmTarget, match="numerics_unregistered") as e:
        GT.gemm_target_for(cc)
    assert f"{fam!r}" in str(e.value) and "registered: ada" in str(e.value)
    gt, refusal = GT.try_gemm_target_for(cc)
    assert gt is None and refusal.startswith("numerics_unregistered")
    with pytest.raises(GT.UnregisteredGemmTarget):
        TargetProfile(compute_capability=cc).gemm_target()


def test_host_family_must_be_the_records_family():
    """The family the host actually reported (construction read-back / capture header) is checked, not re-selected on."""
    assert GT.gemm_target_for((8, 0), None).key == "ampere" and GT.gemm_target_for((8, 9), "ada").key == "ada"
    with pytest.raises(GT.UnregisteredGemmTarget, match="tuned_gemm_arch_family='ada'.*requires None"):
        GT.gemm_target_for((8, 0), "ada")
    with pytest.raises(GT.UnregisteredGemmTarget, match="tuned_gemm_arch_family=None.*requires 'ada'"):
        GT.gemm_target_for((8, 9), None)
    with pytest.raises(GT.UnregisteredGemmTarget):
        GT.gemm_target_for((8, 0), "<unavailable>")


def test_launch_config_default_on_ampere_table_on_ada():
    amp, ada = GT.REGISTERED["ampere"], GT.REGISTERED["ada"]
    for M in (1, 32, 129, 1024):
        consts, warps, stages, src = amp.launch_config(M, 2048, 2048)          # Llama o_proj: tabled on ada, default on ampere
        assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], warps, stages) == (128, 128, 64, 8, 3) and src == "default[ampere]"
        assert consts["GROUP_SIZE_M"] == 8 and consts["HAS_BIAS"] is False and not consts["A_LARGE"]
    consts, warps, stages, src = ada.launch_config(32, 2048, 2048)
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], warps, stages) == (16, 128, 8, 5) and "tuned_ada" in src and "M<=32" in src
    consts, warps, stages, src = ada.launch_config(1, 2048, 2048)
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], warps, stages) == (32, 128, 8, 5)
    assert ada.launch_config(4096, 2048, 2048)[0]["BLOCK_SIZE_M"] == 64                   # above the table: last bucket
    assert ada.launch_config(32, 3072, 2048)[3] == "default[ada]"                          # qkv_proj: not a key


#: [R15 moe GEMM-ADA-TABLE] the key set of vLLM d9105ea80's 'ada' table (= 'hopper'/'blackwell'): Llama's o_proj + the Qwen3-class shapes.
ADA_KEYS = {(2048, 2048), (2048, 6144), (4096, 2048), (12288, 2048), (151936, 2048)}


def test_ada_table_pins_every_vllm_key_verbatim():
    """[R15 moe GEMM-ADA-TABLE] The ada record carries ALL five (N, K) rows of vLLM's tuned table, each verbatim (block_k per row, ten M
    buckets); before R15 it held (2048, 2048) alone and every other tabled launch was refused as a mutation (Qwen3-30B-A3B lm_head at
    TP1: launch bm=16 vs the default pin 128).  Checked against vLLM's own table wherever the pinned vLLM imports."""
    ada = GT.REGISTERED["ada"]
    assert set(ada.tuned_table) == ADA_KEYS and set(GT.HOPPER_TUNED_GEMM) == ADA_KEYS
    for (n, k), row in ada.tuned_table.items():
        assert row["BLOCK_SIZE_K"] == (128 if (n, k) == (2048, 6144) else 64)
        assert [m for m, _ in row["m_buckets"]] == [1, 4, 8, 16, 32, 64, 256, 512, 1024, 2048]
    # Qwen3-30B-A3B lm_head at TP1 (the R15 probe's refusing launch): the M<=4 bucket of (151936, 2048) is bm 16 / bn 32 / 8 warps / 5 stages
    consts, warps, stages, src = ada.launch_config(2, 151936, 2048)
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], warps, stages) == (16, 32, 64, 8, 5) and "M<=4" in src
    assert ada.launch_config(1033, 151936, 2048)[0]["BLOCK_SIZE_M"] == 128                     # the B2 1024/128 prefill: the 2048 bucket
    consts, warps, stages, src = ada.launch_config(2, 2048, 6144)                                 # M<=4 bucket: (16, 64, 8w, 4s), block_k 128
    assert (consts["BLOCK_SIZE_M"], consts["BLOCK_SIZE_N"], consts["BLOCK_SIZE_K"], warps, stages) == (16, 64, 128, 8, 4) and "M<=4" in src
    assert ada.launch_config(1, 2048, 6144)[1:3] == (4, 4)                                         # M<=1 bucket: 4 warps
    # the row's TP2 shards are NOT keys: lm_head half (75968, 2048), qkv half (2560, 2048), the MoE router (128, 2048) -> default tiling
    for (n, k) in ((75968, 2048), (2560, 2048), (128, 2048), (5120, 2048)):
        assert ada.launch_config(2, n, k)[3] == "default[ada]"
    try:
        from vllm.model_executor.determinism.batch_invariant_configs import _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS as VT  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        pytest.skip("pinned vLLM not importable here: the verbatim check runs on the construction/capture hosts")
    vt = VT["ada"]
    assert set(vt) == set(ada.tuned_table)
    for nk, row in ada.tuned_table.items():
        assert vt[nk].block_k == row["BLOCK_SIZE_K"]
        assert [(m, (b.block_m, b.block_n, b.num_warps, b.num_stages)) for m, b in vt[nk].m_buckets] == list(row["m_buckets"])


def test_ada_rc3_table_sha_is_the_one_key_table_hash():
    """[R15 moe GEMM-ADA-TABLE; global M-0140 (3)] The recorded rc3 identity is reproducible: hashing the table restricted to its one
    pre-R15 key (the (2048, 2048) row, unchanged) gives ADA_TABLE_SHA256_RC3 -- the tuned_gemm_table_sha256 in every L40S artifact Built
    before the table change -- and the five-key table hashes differently (a cross-release applicability mismatch is explainable)."""
    one_key = GT.GemmTarget(key="ada", compute_capabilities=((8, 9),), vllm_arch_family="ada", dot="AmpereBF16TcDot16",
                            tuned_table={(2048, 2048): GT.ADA_TUNED_GEMM[(2048, 2048)]})
    assert one_key.table_sha256() == GT.ADA_TABLE_SHA256_RC3
    assert GT.REGISTERED["ada"].table_sha256() != GT.ADA_TABLE_SHA256_RC3
    assert GT.REGISTERED["hopper"].table_sha256() == "f864a47abfe4775993f724e420412398504cdaf58b3d02ae2f6ed5d2e52503b0"   # untouched by R15


def test_profile_mismatches_checks_the_applied_family():
    applied = {"compute_capability": [8, 0], "num_sms": 108, "batch_invariant": True, "vllm_version": "0.28.1rc1.dev472+gd9105ea80",
               "attention_backend": None, "tuned_gemm_arch_family": None}
    assert profile_mismatches(AMPERE_TP, applied) == []
    assert any("tuned_gemm_arch_family" in m for m in profile_mismatches(AMPERE_TP, {**applied, "tuned_gemm_arch_family": "ada"}))
    assert any("tuned_gemm_arch_family" in m for m in profile_mismatches(AMPERE_TP, {k: v for k, v in applied.items() if k != "tuned_gemm_arch_family"}))
    ada_applied = {**applied, "compute_capability": [8, 9], "num_sms": 128, "tuned_gemm_arch_family": "ada"}
    assert profile_mismatches(DEFAULT_TARGET, ada_applied) == []
    assert any("tuned_gemm_arch_family" in m for m in profile_mismatches(DEFAULT_TARGET, {**ada_applied, "tuned_gemm_arch_family": None}))


# -- Bet A: the GEMM rule's pins follow the registry (needs torch) ----------------------------------------------------------

torch = pytest.importorskip("torch")
from verity_vllm.program.frontend.rules import vllm_bindings as VB  # noqa: E402
from tests.program.test_vllm_bindings_pins import _gemm_meta  # noqa: E402


@pytest.fixture
def observed():
    saved = dict(VB._OBSERVED)
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def test_bet_a_pins_are_the_targets_launch_geometry(observed):
    VB.observe_target_profile(AMPERE_TP)
    for M in (1, 32, 1024):
        pins, src = VB.gemm_pins_for(M, 2048, 2048)
        assert (pins["constexprs"]["BLOCK_SIZE_M"], pins["constexprs"]["BLOCK_SIZE_N"], pins["num_warps"], pins["num_stages"]) == (128, 128, 8, 3)
        assert pins["NUM_SMS"] == 108 and src == "default[ampere]"
    assert VB.tuned_gemm_table_sha256() is None
    a100_launch = _gemm_meta(M=32, N=2048, K=2048, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=2048, NUM_SMS=108)
    assert VB._pins_ok(a100_launch, VB.gemm_pins_for(32, 2048, 2048)[0]) is None
    ada_tile_on_a100 = _gemm_meta(M=32, N=2048, K=2048, BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=2048, NUM_SMS=108)
    assert "BLOCK_SIZE_M=16" in (VB._pins_ok(ada_tile_on_a100, VB.gemm_pins_for(32, 2048, 2048)[0]) or "")
    VB.observe_target_profile(DEFAULT_TARGET)
    pins, src = VB.gemm_pins_for(32, 2048, 2048)
    assert pins["constexprs"]["BLOCK_SIZE_M"] == 16 and pins["num_stages"] == 5 and "tuned_ada" in src and VB.tuned_gemm_table_sha256()


BLACKWELL_TP = TargetProfile(compute_capability=(10, 0), num_sms=148)
L40S_TP = TargetProfile(compute_capability=(8, 9), num_sms=142)              # [R15 moe] cp-moe3 / the Qwen3-30B-A3B row's declared L40S


def test_bet_a_refuses_every_gemm_on_an_unregistered_target(observed):
    VB.observe_target_profile(BLACKWELL_TP)
    with pytest.raises(GT.UnregisteredGemmTarget):
        VB.gemm_pins_for(32, 2048, 2048)
    assert VB.tuned_gemm_table_sha256() is None
    rule = VB.TritonGemmRule()
    import json

    class _Node:  # the rule reads the launch through launch_meta(node) = json.loads(node.args[2])
        target = VB.triton_launch.default
        kwargs = {}
        args = (None, None, json.dumps(_gemm_meta(M=32, N=3072, K=2048, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=3072, NUM_SMS=148)))

    verdict = rule.applies(_Node(), None)
    assert isinstance(verdict, str) and verdict.startswith("numerics_unregistered") and "10.0" in verdict, verdict


def test_gemm_ada_table_pins_at_the_rule_level_and_moves_the_applicability_identity(observed):
    """[R15 moe GEMM-ADA-TABLE; global M-0140 (4)] Through the GEMM rule's own entry point on the default (ada) target: the Qwen3-class
    lm_head launch (M=2, N=151936, K=2048) pins the tuned M<=4 bucket 16/32/64 8w/5s (rc3 refused it as a mutation of the default
    128 pin); the Llama pins are UNCHANGED (M=32 and M=1 on (2048, 2048): the smollm2/llama canary roots depend on them); the table
    identity moved off the recorded rc3 value, so a pre-/post-change component pair disagrees on tuned_gemm_table_sha256 by name."""
    VB.observe_target_profile(L40S_TP)                                   # the row's hardware (cc 8.9, 142 SMs) -> the ada record
    pins, src = VB.gemm_pins_for(2, 151936, 2048)
    c = pins["constexprs"]
    assert (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], pins["num_warps"], pins["num_stages"]) == (16, 32, 64, 8, 5)
    assert "tuned_ada" in src and "M<=4" in src
    # the very launch rc3 refused ("constexpr BLOCK_SIZE_M=16 differs from the profile pin 128") now satisfies the pins
    lm_head_launch = _gemm_meta(M=2, N=151936, K=2048, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, num_warps=8, num_stages=5,
                                stride_am=2048, stride_bn=2048, stride_cm=151936, NUM_SMS=142)
    assert VB._pins_ok(lm_head_launch, pins) is None
    # ... and the default-tiled launch of that shape is now the mutation (and a tuned tile with the wrong stage count is too)
    default_launch = _gemm_meta(M=2, N=151936, K=2048, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=151936, NUM_SMS=142)
    assert "BLOCK_SIZE_M=128" in (VB._pins_ok(default_launch, pins) or "")
    stages3 = _gemm_meta(M=2, N=151936, K=2048, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, num_warps=8, num_stages=3, stride_am=2048, stride_bn=2048, stride_cm=151936, NUM_SMS=142)
    assert "num_stages=3" in (VB._pins_ok(stages3, pins) or "")
    # Llama's pins do not move (rc3 values: M=32 -> 16/128 8w/5s "M<=32", M=1 -> 32/128 8w/5s "M<=1")
    pins32, src32 = VB.gemm_pins_for(32, 2048, 2048)
    assert (pins32["constexprs"]["BLOCK_SIZE_M"], pins32["constexprs"]["BLOCK_SIZE_N"], pins32["num_warps"], pins32["num_stages"]) == (16, 128, 8, 5) and "M<=32" in src32
    pins1, src1 = VB.gemm_pins_for(1, 2048, 2048)
    assert (pins1["constexprs"]["BLOCK_SIZE_M"], pins1["constexprs"]["BLOCK_SIZE_N"], pins1["num_warps"], pins1["num_stages"]) == (32, 128, 8, 5) and "M<=1" in src1
    # the (2048, 6144) row keeps vLLM's BLOCK_SIZE_K 128; untabled shapes (the row's TP2 shards) stay on the default tiling
    assert VB.gemm_pins_for(2, 2048, 6144)[0]["constexprs"]["BLOCK_SIZE_K"] == 128
    assert VB.gemm_pins_for(2, 75968, 2048)[1] == "default[ada]" and VB.gemm_pins_for(2, 75968, 2048)[0]["constexprs"]["BLOCK_SIZE_M"] == 128
    # applicability identity: the five-key table != the rc3 one-key table, and it is what derive_step records
    assert VB.tuned_gemm_table_sha256() == GT.REGISTERED["ada"].table_sha256() != GT.ADA_TABLE_SHA256_RC3
    assert len(GT.ADA_TABLE_SHA256_RC3) == 64


def test_bet_a_pins_the_hopper_tile_on_a_hopper_target(observed):
    """[R13 hopper] cc 9.0 is registered: Bet A pins vLLM's 'hopper' table row (BLOCK_SIZE_M 16 at M<=1, BLOCK_SIZE_K 128) and Gemm_v2."""
    VB.observe_target_profile(HOPPER_TP)
    pins, src = VB.gemm_pins_for(1, 2048, 2048)
    assert (pins["constexprs"]["BLOCK_SIZE_M"], pins["constexprs"]["BLOCK_SIZE_K"], pins["NUM_SMS"]) == (16, 128, 132) and "tuned_hopper" in src
    assert VB.tuned_gemm_table_sha256() == GT.REGISTERED["hopper"].table_sha256()


def test_bet_a_pins_the_default_tile_and_188_sms_on_the_pro6000(observed):
    """cc 12.0 is registered: a persistent-matmul launch there must show the default tiling for every shape (Hopper's tuned rows
    included) and the target's 188 SMs."""
    VB.observe_target_profile(PRO6000_TP)
    for M, N, K in ((1, 2048, 2048), (32, 2048, 2048), (2, 151936, 2048)):
        pins, src = VB.gemm_pins_for(M, N, K)
        c = pins["constexprs"]
        assert (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], pins["num_warps"], pins["num_stages"]) == (128, 128, 64, 8, 3)
        assert pins["NUM_SMS"] == 188 and src == "default[blackwell_consumer]"
    assert VB.tuned_gemm_table_sha256() is None
    hopper_tile = _gemm_meta(M=1, N=2048, K=2048, BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, stride_am=2048, stride_bn=2048, stride_cm=2048, NUM_SMS=188)
    assert "BLOCK_SIZE_M=16" in (VB._pins_ok(hopper_tile, VB.gemm_pins_for(1, 2048, 2048)[0]) or "")
    on_132 = _gemm_meta(M=32, N=2048, K=2048, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=2048, NUM_SMS=132)
    assert "NUM_SMS" in (VB._pins_ok(on_132, VB.gemm_pins_for(32, 2048, 2048)[0]) or "")
