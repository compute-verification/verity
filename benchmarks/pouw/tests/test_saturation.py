"""The saturation budget's tools: vLLM kernel categories, step attribution through graph launches, and ncu's slot share."""

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
