"""CB-A: the opaque Triton-launch node's Definition-selection key must be *sufficient* — every field that distinguishes a
registered specialisation of the same (K, N) / (N, EPS) is tested by the pins, and an absent field is a mismatch, never a
default. Negative cases: same kernel name with another source body, another warp count (the reduction tree), another SM
count (another GPU = another profile), another tiling constexpr, a bias epilogue, a non-bf16 dtype."""

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.program.frontend.rules.vllm_bindings import GEMM_PINS, RMS_PINS, _pins_ok, gemm_pins_for, lookup_key  # noqa: E402


def _gemm_meta(**over):
    sc = {"M": 8, "N": 960, "K": 576, "bias_ptr": None, "stride_am": 576, "stride_ak": 1, "stride_bk": 1, "stride_bn": 576, "stride_cm": 960, "stride_cn": 1,
          "NUM_SMS": 128, "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False,
          "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8}
    meta = {"kernel": GEMM_PINS["kernel"], "src_sha256": GEMM_PINS["src_sha256"], "triton": "3.7.1", "scalars": sc,
            "geometry": {"num_warps": 8, "num_warps_explicit": True, "num_stages": 3, "num_stages_explicit": True, "num_ctas": 1},
            "constexprs": ["A_LARGE", "BLOCK_SIZE_K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "B_LARGE", "C_LARGE", "GROUP_SIZE_M", "HAS_BIAS", "NUM_SMS"],
            "tensor_args": ["a_ptr", "b_ptr", "c_ptr"],
            "tensor_meta": [{"shape": [8, 576], "stride": [576, 1], "dtype": "torch.bfloat16"}, {"shape": [576, 960], "stride": [1, 576], "dtype": "torch.bfloat16"},
                            {"shape": [8, 960], "stride": [960, 1], "dtype": "torch.bfloat16"}]}
    for k, v in over.items():
        if k in ("src_sha256", "kernel"):
            meta[k] = v
        elif k in ("num_warps", "num_stages"):
            meta["geometry"][k] = v
            meta["scalars"][k] = v
        else:
            meta["scalars"][k] = v
    return meta


def test_gemm_pins_accept_the_profile_launch():
    assert _pins_ok(_gemm_meta(), GEMM_PINS) is None


@pytest.mark.parametrize("over,needle", [
    ({"src_sha256": "0" * 64}, "source sha256"),
    ({"num_warps": 4}, "num_warps=4"),
    ({"num_stages": 4}, "num_stages=4"),
    ({"NUM_SMS": 132}, "NUM_SMS=132"),  # H100: another GPU is another profile
    ({"BLOCK_SIZE_K": 32}, "BLOCK_SIZE_K=32"),  # another k-chain = another numeric
    ({"HAS_BIAS": True}, "HAS_BIAS=True"),
])
def test_gemm_pins_refuse_each_dropped_field(over, needle):
    bad = _pins_ok(_gemm_meta(**over), GEMM_PINS)
    assert bad is not None and needle in bad, bad


def test_gemm_pins_refuse_absent_geometry():
    """No geometry and no scalar num_warps: absent is a mismatch, not a default."""
    m = _gemm_meta()
    m["geometry"] = {}
    del m["scalars"]["num_warps"]
    assert _pins_ok(m, GEMM_PINS) is not None


def test_rms_pins_use_effective_warps():
    """`_rms_norm_kernel` is launched without explicit num_warps: the node carries Triton's effective default (4, flagged
    non-explicit) and the pin is checked against it; 8 warps would be another reduction tree."""
    base = {"kernel": RMS_PINS["kernel"], "src_sha256": RMS_PINS["src_sha256"], "scalars": {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True, "n_cols": 576, "eps": 1e-5},
            "geometry": {"num_warps": 4, "num_warps_explicit": False, "num_stages": 3, "num_stages_explicit": False}, "constexprs": ["BLOCK_SIZE", "HAS_WEIGHT"]}
    assert _pins_ok(base, RMS_PINS) is None
    bad = dict(base, geometry={"num_warps": 8, "num_warps_explicit": True})
    assert "num_warps=8" in (_pins_ok(bad, RMS_PINS) or "")
    other_body = dict(base, src_sha256="f" * 64)
    assert "source sha256" in (_pins_ok(other_body, RMS_PINS) or "")


def test_lookup_key_carries_every_selection_field():
    key = lookup_key(_gemm_meta())
    assert key["kernel"] == GEMM_PINS["kernel"] and key["src_sha256"] == GEMM_PINS["src_sha256"]
    assert key["constexprs"]["NUM_SMS"] == 128 and key["constexprs"]["BLOCK_SIZE_K"] == 64
    assert key["geometry"]["num_warps"] == 8 and key["scalars"]["stride_bn"] == 576
    assert key["tensors"]["b_ptr"]["stride"] == [1, 576] and key["tensors"]["c_ptr"]["dtype"] == "torch.bfloat16"


def test_tuned_table_row_pins_follow_the_eager_matcher():
    """(N, K) = (2048, 2048) is a key of the eager matcher's pinned sm_89 table: M=8 -> the 8-bucket (32, 128, 8 warps, 3 stages);
    M=1 -> (32, 128, 8, 5); an untabled shape -> the default pins. The default tiling on a tabled shape is REFUSED (vLLM would not
    launch it there), and vice versa."""
    pytest.importorskip("verity_vllm.observe.fold.profiles.gen_llama_patterns")
    pins8, src8 = gemm_pins_for(8, 2048, 2048)
    assert (pins8["constexprs"]["BLOCK_SIZE_M"], pins8["constexprs"]["BLOCK_SIZE_N"], pins8["num_warps"], pins8["num_stages"]) == (32, 128, 8, 3) and "tuned" in src8
    pins1, _ = gemm_pins_for(1, 2048, 2048)
    assert (pins1["constexprs"]["BLOCK_SIZE_M"], pins1["num_stages"]) == (32, 5)
    assert pins8["constexprs"]["BLOCK_SIZE_K"] == 64 and pins8["src_sha256"] == GEMM_PINS["src_sha256"]
    assert gemm_pins_for(8, 960, 576) == (GEMM_PINS, "default[ada]")   # source names the registered target
    tuned_launch = _gemm_meta(M=8, N=2048, K=2048, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, stride_am=2048, stride_bn=2048, stride_cm=2048)
    assert _pins_ok(tuned_launch, pins8) is None
    assert "BLOCK_SIZE_M=32" in (_pins_ok(tuned_launch, GEMM_PINS) or "")
    assert "BLOCK_SIZE_M=128" in (_pins_ok(_gemm_meta(M=8, N=2048, K=2048), pins8) or "")


def test_x01_num_sms_is_the_one_target_profile_fact():
    """[X-01 portability] The builder pins NUM_SMS to the active TargetProfile's `num_sms` — the same object `TargetProfile.from_device()`
    fills from `multi_processor_count` and the eager resolver reads back from the record's device document — and selects the same
    `Gemm_v1{K,N}` Definition for the BI persistent matmul whatever that count is.  No second copy of the number: `GEMM_PINS["NUM_SMS"]`
    is `DEFAULT_TARGET.num_sms`."""
    from verity_vllm.program.frontend.rules.vllm_bindings import observe_target_profile
    from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
    assert GEMM_PINS["NUM_SMS"] == DEFAULT_TARGET.num_sms
    try:
        for sms in (142, 108, 132, 188):
            observe_target_profile(TargetProfile(num_sms=sms))
            pins, _ = gemm_pins_for(8, 960, 576)
            assert pins["NUM_SMS"] == sms and pins["constexprs"] == GEMM_PINS["constexprs"]
            assert _pins_ok(_gemm_meta(NUM_SMS=sms), pins) is None
            assert f"NUM_SMS=128 differs from the profile pin {sms}" in (_pins_ok(_gemm_meta(), pins) or "")
        # the eager resolver reads the same fact from a record's device document
        tp = TargetProfile.from_gpu_document({"name": "NVIDIA L40S", "compute_capability": "8.9", "multi_processor_count": 142})
        assert tp == TargetProfile(compute_capability=(8, 9), num_sms=142)
        observe_target_profile(tp)
        assert gemm_pins_for(1, 960, 576)[0]["NUM_SMS"] == 142
        pro = TargetProfile.from_gpu_document({"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "compute_capability": "12.0",
                                               "multi_processor_count": 188})
        assert (pro.compute_capability, pro.num_sms) == ((12, 0), 188)
    finally:
        observe_target_profile(None)
    assert gemm_pins_for(8, 2048, 2048)[0]["NUM_SMS"] == GEMM_PINS["NUM_SMS"]
