from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import step_profile  # noqa: E402


def test_kernel_classes_by_name():
    assert step_profile.classify("void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<...>>") == "gemm"
    assert step_profile.classify("nvjet_tst_64x8_64x16_4x1_v_bz_TNT") == "gemm"
    assert step_profile.classify("void flashinfer::BatchDecodeWithPagedKVCacheKernel<...>") == "attention"
    assert step_profile.classify("void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, "
                                 "false, cutlass::bfloat16_t, Flash_kernel_traits<...>>>") == "attention"
    assert step_profile.classify("void vllm::rms_norm_kernel<c10::BFloat16>") == "other"


def test_split_times_each_class_and_re_times_the_gemms_at_the_native_rate():
    # R = 1 GB/s = 1e6 bytes a ms; two steps, each a 4 ms GEMM over 2e6 weight bytes, a 2 ms attention over 1e6 KV bytes
    # and a 1 ms norm, back to back
    k = []
    for s in range(2):
        t = s * 7000
        k += [{"name": "gemm_a", "ts": t, "dur": 4000}, {"name": "flash_b", "ts": t + 4000, "dur": 2000},
              {"name": "rms_norm", "ts": t + 6000, "dur": 1000}]
    r = step_profile.split(k, 2, 2 * 10**6, 10**6, 1.0, 0.5)
    assert r["ms"] == {"gemm": 4.0, "attention": 2.0, "other": 1.0}
    assert (r["kernel_ms"], r["span_ms"], r["kernels_a_step"]) == (7.0, 7.0, 3.0)
    assert (r["of_R"], r["gemm_of_R"], r["attention_of_R"]) == (pytest.approx(3 / 7), 0.5, 0.5)
    assert r["native_ms"] == pytest.approx(4.0 + 2.0 + 1.0) and r["native_all_ms"] == pytest.approx(6.0 + 1.0)
    assert r["kernels"][0] == {"name": "gemm_a", "class": "gemm", "ms": 4.0} and len(r["kernels"]) == 3
