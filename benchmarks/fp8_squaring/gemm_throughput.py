"""FP8 (E4M3) vs int8 tensor-core GEMM throughput on one GPU, for loophole (4).

Measures the raw matmul rate the stacking cheater's FP8 carry route is bounded by
(`torch._scaled_mm`, E4M3) against the integer baseline (`torch._int_mm`, int8),
and prints the implied 1024-bit squaring ceiling to compare with kv.cu's measured
CUDA-core rate (1.499e10/s/GPU, remote_bandwidth/timing.py).

A 1024-bit squaring is a bilinear (Toeplitz) product of one word with itself.
Mapped onto an MxKxN MMA at B-bit limbs it is ~(2L-1) output columns x L inner,
L = 1024/B; ra-squaring-ceiling modelled it at ~8 SM-clocks (512-bit, int8),
losing to the CUDA-core column form's 6.0.  Here we just report the peak GEMM
rates and the issue-bound squaring ceiling they imply.
"""
from __future__ import annotations

import json
import os
import time

import torch


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    n = 8192
    out: dict = {"device": name, "capability": f"sm_{cap[0]}{cap[1]}", "n": n}

    # int8 x int8 -> int32
    ai = torch.randint(-127, 127, (n, n), dtype=torch.int8, device=dev)
    bi = torch.randint(-127, 127, (n, n), dtype=torch.int8, device=dev).t().contiguous().t()
    try:
        s = _bench(lambda: torch._int_mm(ai, bi))
        out["int8_s"] = s
        out["int8_tops"] = 2 * n**3 / s / 1e12
    except Exception as e:  # noqa: BLE001
        out["int8_error"] = repr(e)

    # fp8 e4m3 x e4m3 -> bf16, with unit scales (row-major A, col-major B)
    try:
        af = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        bf = torch.randn(n, n, device=dev).t().to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device=dev)
        s = _bench(lambda: torch._scaled_mm(af, bf, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16))
        out["fp8_s"] = s
        out["fp8_tflops"] = 2 * n**3 / s / 1e12
    except Exception as e:  # noqa: BLE001
        out["fp8_error"] = repr(e)

    # implied 1024-bit squaring ceiling at 8-bit limbs (L=128): a Toeplitz
    # product ~ (2L-1) x L x 1 MACs = 128*255 ~= 3.26e4 int8 MACs per squaring.
    L = 128
    macs_per_sq = (2 * L - 1) * L
    if "int8_tops" in out:
        # TOPs = 2*MAC/s, so MAC/s = tops*1e12/2
        out["int8_implied_sq_per_s"] = (out["int8_tops"] * 1e12 / 2) / macs_per_sq
    out["kv_cu_sq_per_s"] = 1.499e10
    if "int8_implied_sq_per_s" in out:
        out["int8_tc_vs_kv_cu"] = out["int8_implied_sq_per_s"] / out["kv_cu_sq_per_s"]

    print(json.dumps(out, indent=2))
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        with open(os.path.join(rd, "result.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
