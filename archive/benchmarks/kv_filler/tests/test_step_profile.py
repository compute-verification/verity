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
    assert r["kernels"][0] == {"name": "gemm_a", "class": "gemm", "ms": 4.0, "a_step": 1.0} and len(r["kernels"]) == 3


def test_split_counts_idle_between_kernels_inside_a_step_apart_from_the_host_gap_between_steps():
    # two steps 5 ms apart; inside each, a 10 µs gap and two overlapping kernels (no gap between them)
    k = []
    for s in range(2):
        t = s * 5000
        k += [{"name": "gemm_a", "ts": t, "dur": 100}, {"name": "flash_b", "ts": t + 110, "dur": 100},
              {"name": "rms_norm", "ts": t + 150, "dur": 100}]
    r = step_profile.split(k, 2, 10**6, 10**6, 1.0, 0.5)
    assert r["idle_in_step_ms"] == pytest.approx(0.01) and r["idle_in_step_gaps"] == 1.0
    assert r["idle_between_steps_ms"] == pytest.approx((5000 - 250) / 1e3 / 2)


def test_ncu_split_sets_each_class_s_dram_bytes_against_its_registered_bytes():
    head = '"ID","Kernel Name","' + '","'.join(step_profile.NCU_METRICS) + '"\n'
    units = '"","","ns","byte","byte","sector","sector","cycle","cycle"\n'
    body = ('"0","nvjet_gemm","1,000","3000","100","200","10","30","100"\n'
            '"1","flashinfer::BatchDecodeWithPagedKVCacheKernel","500","1500","0","50","0","10","50"\n'
            '"2","vllm::rms_norm_kernel","100","40","40","4","4","1","10"\n') * 2
    rows = step_profile.ncu_rows(head + units + body)
    assert len(rows) == 6 and rows[0]["gpu__time_duration.sum"] == 1000.0
    r = step_profile.ncu_split(rows, 2, 2000.0, 1000.0, 10.0)  # R = 10 GB/s = 10 bytes a ns
    assert r["gemm"]["dram_read_GB"] == pytest.approx(3e-6) and r["gemm"]["read_over_registered"] == pytest.approx(1.5)
    assert r["attention"]["read_over_registered"] == pytest.approx(1.5) and r["other"]["read_over_registered"] is None
    assert r["all"]["read_over_registered"] == pytest.approx(4540 / 3000)
    assert r["gemm"]["l2_read_GB"] == pytest.approx(200 * 32 / 1e9)
    assert r["gemm"]["bus_left_GB"] == pytest.approx((1000 * 10 - 3100) / 1e9)
    assert r["gemm"]["dram_active"] == pytest.approx(0.3) and r["all"]["dram_active"] == pytest.approx(41 / 160)
    assert r["kernels"][0]["name"] == "nvjet_gemm" and r["kernels"][0]["n"] == 2
