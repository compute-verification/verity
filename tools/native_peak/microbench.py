"""Measured dense tensor-core rate of the probed device for the campaign's ``NativePeak`` references (lane b-hopper).

    python tools/native_peak/microbench.py --out "$RESEARCH_RUN_DIR/native_peak.json" [--reps 7] [--sizes 8192,16384]

For each operand dtype the H100 targets quote (``verity.verification.target``: BF16 with FP32 accumulate, FP8 E4M3 with
FP32 accumulate) it times a square GEMM through cuBLAS (``torch.matmul`` for BF16 with reduced-precision reductions
off; ``torch._scaled_mm`` with unit scales and an FP32 output for E4M3), warm, CUDA events, median of ``--reps``, and
reports FLOP/s (an FMA counts as two) next to the datasheet peak the target carries, plus the implied native wall
seconds of one K=1536 VU.  This is the reference's calibration -- how conservative dividing by the datasheet peak is on
the actual card -- not a proof of anything.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

K_VU = 1536


def timed(fn, reps: int) -> tuple[float, list[float]]:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts), ts


def hardware() -> dict:
    p = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    return {"name": torch.cuda.get_device_name(0), "sms": p.multi_processor_count, "sm": f"{p.major}.{p.minor}",
            "total_memory_bytes": p.total_memory, "nvidia_smi": smi, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "host": platform.node()}


def bf16(n: int, reps: int) -> dict:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    t, ts = timed(lambda: torch.matmul(a, b, out=out), reps)
    flops = 2.0 * n**3
    return {"seconds": t, "all": ts, "flop_per_second": flops / t, "tflops": flops / t / 1e12,
            "kernel": "torch.matmul bf16 x bf16 -> bf16 (cuBLAS, fp32 accumulate, reduced-precision reduction off)"}


def fp8_e4m3(n: int, reps: int) -> dict:
    a = (torch.randn(n, n, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(n, n, device="cuda") * 0.25).to(torch.float8_e4m3fn).t().contiguous().t()   # column-major
    one = torch.tensor(1.0, device="cuda")
    t, ts = timed(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32), reps)
    flops = 2.0 * n**3
    return {"seconds": t, "all": ts, "flop_per_second": flops / t, "tflops": flops / t / 1e12,
            "kernel": "torch._scaled_mm e4m3 x e4m3 -> fp32 (cuBLASLt, fp32 accumulate, unit scales)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--sizes", default="8192,16384")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    sizes = [int(s) for s in args.sizes.split(",")]
    hw = hardware()
    print(json.dumps(hw, indent=1), flush=True)
    try:
        from verity.verification.target import BF16_HOPPER, FP8_HOPPER_WGMMA
        peaks = {"bf16": BF16_HOPPER.native_peak, "fp8_e4m3": FP8_HOPPER_WGMMA.native_peak}
        datasheet = {k: {"flop_per_second": v.flop_per_second, "device": v.device, "rate": v.rate, "source": v.source}
                     for k, v in peaks.items()}
    except Exception as e:  # noqa: BLE001 -- the tool runs without the verity package too
        print(f"(verity targets unavailable: {e!r}; datasheet peaks not attached)", file=sys.stderr)
        datasheet = {}
    doc = {"schema": "native-peak/v1", "hardware": hw, "reps": args.reps, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "datasheet": datasheet, "bf16": {}, "fp8_e4m3": {}}
    for n in sizes:
        for key, fn in (("bf16", bf16), ("fp8_e4m3", fp8_e4m3)):
            try:
                r = fn(n, args.reps)
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            doc[key][f"{n}^3"] = r
            if "tflops" in r:
                ds = datasheet.get(key, {}).get("flop_per_second")
                frac = f" = {r['flop_per_second'] / ds:.3f} of datasheet" if ds else ""
                print(f"{key:9} {n:6}^3  {r['seconds'] * 1e3:9.3f} ms  {r['tflops']:8.1f} TFLOPS{frac}", flush=True)
            else:
                print(f"{key:9} {n:6}^3  {r['error']}", flush=True)
    for key in ("bf16", "fp8_e4m3"):
        best = max((r for r in doc[key].values() if "flop_per_second" in r), key=lambda r: r["flop_per_second"], default=None)
        if best is not None:
            ds = datasheet.get(key, {}).get("flop_per_second")
            doc[f"{key}_best"] = {"flop_per_second": best["flop_per_second"], "tflops": best["tflops"],
                                  "fraction_of_datasheet": best["flop_per_second"] / ds if ds else None,
                                  "seconds_per_vu_k1536_measured": 2 * K_VU / best["flop_per_second"],
                                  "seconds_per_vu_k1536_datasheet": 2 * K_VU / ds if ds else None}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
