"""The saturation budget's tools: vLLM kernel categories, step attribution through graph launches, ncu's slot share, and
the timed-window reading of a GEMM window's ε."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "saturation"
sys.path.insert(0, str(HERE))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("vllm_split")
N = _load("ncu_summary")
W = _load("windows_eps")
D = _load("drift_tau")
T = _load("r3_table")
M = _load("mma_window")


def test_categories_keep_fp8_gemms_and_kv_writes_apart():
    cases = {
        "void cutlass::device_kernel<vllm::cutlass_3x_gemm_sm120<cutlass::float_e4m3_t>>": "gemm",
        "nvjet_tst_128x64_64x8_1x1_v_bz_TNT": "gemm",
        "void vllm::dynamic_per_token_scaled_fp8_quant_kernel<c10::BFloat16, c10::Float8_e4m3fn>": "quant",
        "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16>": "kv_write",
        "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits>": "attention",
        "void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>": "norm",
        "void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel>": "act",
        "void vllm::rotary_embedding_kernel<c10::BFloat16, true>": "rope",
        "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0": "norm",
    }
    assert {k: V.category(k) for k in cases} == cases


def _x(cat, name, ts, dur, corr=None):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, **({"args": {"correlation": corr}} if corr else {})}


def test_kernels_follow_their_launch_into_the_step(tmp_path):
    gemm, attn, argmax = "nvjet_gemm", "flash_fwd_kernel", "reduce_kernel<ArgMaxOps>"
    ev = [_x("user_annotation", "vstep|prefill|64|2", 0, 50),
          _x("cuda_runtime", "cudaLaunchKernel", 5, 1, 1), _x("kernel", gemm, 100, 40, 1),
          _x("user_annotation", "vstep|decode|2|2", 200, 50),
          _x("cuda_runtime", "cudaGraphLaunch", 210, 1, 2), _x("kernel", gemm, 300, 20, 2), _x("kernel", attn, 321, 10, 2),
          _x("user_annotation", "vstep|decode|2|2|sample", 260, 10),
          _x("cuda_runtime", "cudaLaunchKernel", 262, 1, 3), _x("kernel", argmax, 340, 2, 3),
          _x("kernel", "stray", 900, 3, 99)]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": ev}))
    parsed = V.parse_trace(path)
    assert [s["label"] for s in parsed["steps"]] == ["prefill|64|2", "decode|2|2"]
    assert parsed["unassigned"] == {"n": 1, "us": 3.0}
    decode = V.summarize_phase(parsed, 2)
    assert decode["by_category_ms"] == {"attention": 0.01, "gemm": 0.02, "sampling": 0.002}
    assert decode["span_ms"] == (342 - 300) / 1e3


def test_slot_share_is_lane_ops_over_128_per_sm_cycle(tmp_path):
    head = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__thread_inst_executed.sum",
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed"]
    lines = [",".join(f'"{h}"' for h in head), '"","","ms","Kcycle","Minst","%"',
             '"0","void vllm::rms_norm_kernel<c10::BFloat16>","2","1","1.28","5"',
             '"1","nvjet_gemm","8","4","2,560","1"']
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(lines) + "\n")
    doc = N.summarize(path, sms=100)
    assert doc["by_group"]["norm"]["slot_share"] == pytest.approx(1_280_000 / (128 * 100 * 1000))
    assert doc["by_group"]["step"]["slot_share"] == pytest.approx(2_561_280_000 / (128 * 100 * 5000))
    assert doc["by_group"]["step"]["us"] == 10_000
    assert doc["by_group"]["step"]["fma_pct"] == (5 * 2 + 1 * 8) / 10


def test_alu_instructions_cost_two_slots(tmp_path):
    head = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__thread_inst_executed.sum",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed"]
    lines = [",".join(f'"{h}"' for h in head), '"","","us","cycle","inst","%"',
             '"0","void vllm::rotary_embedding_kernel<c10::BFloat16, true>","1","1000","3840000","10"',
             '"1","nvjet_gemm","1","1000","1280000",""']
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(lines) + "\n")
    doc = N.summarize(path, sms=100)["by_group"]
    assert doc["rope"]["slot_share"] == pytest.approx(0.3)
    assert doc["rope"]["slot_share_alu2"] == pytest.approx(0.3 + 0.1)
    assert doc["gemm"]["slot_share_alu2"] == doc["gemm"]["slot_share"] == pytest.approx(0.1)
    assert doc["step"]["slot_share_alu2"] == pytest.approx((3_840_000 + 1_280_000 + 1_280_000) / (128 * 100 * 2000))


def test_ncu_units_that_are_not_the_metrics_refuse(tmp_path):
    head = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_elapsed.avg", "smsp__thread_inst_executed.sum"]
    path = tmp_path / "raw.csv"
    path.write_text("\n".join([",".join(f'"{h}"' for h in head), '"","","cycle","cycle","inst"',
                               '"0","k","1","1","1"']) + "\n")
    with pytest.raises(SystemExit, match="not a time unit"):
        N.summarize(path, sms=1)


def test_timed_window_keeps_all_the_work_of_a_cheater_that_fits():
    rows = [{"design": "k", "phase": "p2048", "added": 0.02, "slowdown": 1.011},
            {"design": "k", "phase": "p2048", "added": 0.30, "slowdown": 1.5},
            {"design": "other", "phase": "p2048", "added": 0.5, "slowdown": 1.0}]
    assert W.timed(rows, "k", "p2048", 0.5, 1.0, 16.0) == 0.0
    assert W.timed(rows, "k", "p2048", 0.5, 1.01, 16.0) == pytest.approx(16 * 0.02 / 8 / 0.5)
    assert W.timed(rows, "k", "p2048", 0.5, 1.05, 16.0) == pytest.approx((16 * 0.02 / 8 + 0.9 * 0.039) / 0.5)
    assert W.timed([], "k", "p2048", 0.5, 1.05, 16.0) == pytest.approx(0.9 * 0.05 / 0.5)


def test_honest_tau_reads_the_slow_tail_over_the_fastest_launches():
    ms = [1.0] * 10 + [1.01] * 979 + [1.03] * 10 + [1.5]
    assert D.quantile(ms, 0.001) == 1.0 and D.quantile(ms, 0.999) == 1.03
    assert D.tau(D.quantile(ms, 0.999), D.quantile(ms, 0.001)) == pytest.approx(1.03)
    assert D.reasons(0x4 | 0x20) == ["sw power cap", "sw thermal"] and D.reasons(0) == ["none"]


def test_held_reading_reproduces_w6b_held_alu():
    """pouw-core.lean §10's `held_alu` at p2048 (η_w 0.51004, κ 4): 1.32 / 3.23 / 6.17 / 15.82 / 54.64 / 65.84%, to
    its integer rounding, from three of its classes (the null sets every value but the last two)."""
    rows = [{"design": "k", "phase": "p", "slowdown": s, "added": a}
            for s, a in ((1.004060, 0.011435), (1.170120, 0.209250), (1.137126, 0.121307))]
    got = [T.held(rows, "k", "p", 0.51004, t) for t in (1.0035, 1.01, 1.02, 1.0528, 1.1848, 1.2229)]
    want = [0.0132, 0.0323, 0.0617, 0.1582, 0.5464, 0.6584]
    assert all(abs(g - w) < 0.0002 for g, w in zip(got, want))


def test_mma_window_reads_the_largest_extra_share_within_tau():
    """A variant fits within the A/A resolution; mixing two variants reaches τ exactly, from the null at ratio 1."""
    pts = [(1.0, 0.0), (1.0097, 0.1875), (1.0148, 0.25), (1.0447, 0.375)]
    assert M.frontier(pts, 1.0) == (0.0, 0.0)
    fit, mixed = M.frontier(pts, 1.01)
    assert fit == 0.1875 and mixed == pytest.approx(0.1875 + 0.0625 * 0.0003 / 0.0051)
    fit, mixed = M.frontier(pts, 1.02)
    assert fit == 0.25 and mixed == pytest.approx(0.25 + 0.125 * 0.0052 / 0.0299)
    assert M.frontier(pts, 1.05) == (0.375, 0.375)


def test_mma_window_pairs_each_round_with_the_design():
    rows = {("d", "s"): {"ms_reps": [1.0, 2.0, 1.0]}, ("d_mm2", "s"): {"ms_reps": [1.01, 2.04, 1.03]}}
    assert M.ratios(rows, "d", "d_mm2", "s") == pytest.approx(1.02)
