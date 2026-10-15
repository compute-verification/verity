"""vLLM's own `cutlass_scaled_mm` at `cutlass_receipt.cu`'s shapes: the served kernel that cutlass_receipt's plain kernels
rebuild. Per-token activation scales, per-channel weight scales, bf16 out, the same operand bytes' distribution (E4M3 of
exponent field 4 ... 10), operands rotating over sets past twice the L2, --reps rounds timed with CUDA events. Each
shape's kernel name is read from the torch profiler, so the JSON says which CUTLASS configuration vLLM dispatched.

    python vllm_gemm_ref.py [--shapes NAME,...] [--reps 9] > vllm-gemm-ref.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys

import torch

SHAPES = {
    "p2048_qkv": (2048, 6144, 4096),
    "p2048_o": (2048, 4096, 4096),
    "p2048_gateup": (2048, 28672, 4096),
    "p2048_down": (2048, 4096, 14336),
    "p8192_qkv": (8192, 6144, 4096),
    "p8192_o": (8192, 4096, 4096),
    "p8192_gateup": (8192, 28672, 4096),
    "p8192_down": (8192, 4096, 14336),
    "d32_qkv": (32, 6144, 4096),
    "d32_o": (32, 4096, 4096),
    "d32_gateup": (32, 28672, 4096),
    "d32_down": (32, 4096, 14336),
    "cube8192": (8192, 8192, 8192),
}


def e4m3(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    h = torch.randint(0, 1 << 24, (rows, cols), device="cuda", dtype=torch.int32, generator=gen)
    b = (h & 0x80) | ((4 + (h >> 8) % 7) << 3) | ((h >> 16) & 7)
    return b.to(torch.uint8).view(torch.float8_e4m3fn)


def kernel_config(name: str) -> dict:
    st = re.search(r"MainloopSm120TmaWarpSpecialized<(\d+), (\d+)", name)
    sched = re.search(r"KernelTmaWarpSpecialized(\w+?)Sm120", name)
    tile = re.search(r"cute::C<(\d+)>, cute::C<(\d+)>, cute::C<(\d+)>>, cutlass::float_e4m3_t", name)
    return {
        "stages": int(st.group(1)) if st else None,
        "schedule": sched.group(1) if sched else None,
        "tile": [int(x) for x in tile.groups()] if tile else None,
        "name_head": name[:160],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    from vllm import _custom_ops as ops

    props = torch.cuda.get_device_properties(0)
    l2 = props.L2_cache_size
    gen = torch.Generator(device="cuda").manual_seed(1)
    sel = [s for s in a.shapes.split(",") if s] or list(SHAPES)
    rows = []
    for name in sel:
        m, n, k = SHAPES[name]
        sets = min(16, max(1, math.ceil(2 * l2 / (m * k + n * k))))
        xs = [e4m3(m, k, gen) for _ in range(sets)]
        ws = [e4m3(n, k, gen).t() for _ in range(sets)]  # [k, n], column-major: vLLM's layer.weight after process_weights
        sa = torch.ones(m, 1, device="cuda", dtype=torch.float32)
        sb = torch.ones(n, 1, device="cuda", dtype=torch.float32)

        def call(i: int) -> torch.Tensor:
            return ops.cutlass_scaled_mm(xs[i % sets], ws[i % sets], scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            call(0)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages() if "cutlass" in e.key or "Gemm" in e.key]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def batch(launches: int) -> float:
            e0.record()
            for i in range(launches):
                call(i)
            e1.record()
            e1.synchronize()
            return e0.elapsed_time(e1) / launches

        pilot = batch(sets)
        launches = max(sets, math.ceil(25.0 / max(pilot, 1e-3) / sets) * sets)
        batch(launches)
        ms = [batch(launches) for _ in range(a.reps)]
        t = statistics.median(ms)
        # Eager launches from Python can't keep up with a ~20 us kernel; vLLM replays CUDA graphs, so time one too.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call(0)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(launches):
                call(i)

        def replay() -> float:
            e0.record()
            graph.replay()
            e1.record()
            e1.synchronize()
            return e0.elapsed_time(e1) / launches

        replay()
        ms_graph = [replay() for _ in range(a.reps)]
        tg = statistics.median(ms_graph)
        del graph
        rows.append(
            {
                "shape": name,
                "m": m,
                "n": n,
                "k": k,
                "sets": sets,
                "launches": launches,
                "reps": a.reps,
                "ms": round(t, 5),
                "ms_min": round(min(ms), 5),
                "tflops": round(2.0 * m * n * k / (t * 1e9), 1),
                "ms_graph": round(tg, 5),
                "tflops_graph": round(2.0 * m * n * k / (tg * 1e9), 1),
                "kernels": [kernel_config(x) for x in names],
            }
        )
        print(f"{name}: {t:.4f} ms ({tg:.4f} graphed) {rows[-1]['tflops']} TFLOPS {rows[-1]['kernels']}", file=sys.stderr)
        del xs, ws
        torch.cuda.empty_cache()
    json.dump({"schema": "verity-pouw/saturation-vllm-gemm-ref/v1", "device": props.name, "rows": rows}, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
