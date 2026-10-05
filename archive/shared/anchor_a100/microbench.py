"""anchor-a100 lane: the A100's *measured* roofline inputs (30-second microbench).

    /workspace/venv312/bin/python -m anchor_a100.microbench --out "$RESEARCH_RUN_DIR/microbench.json"

Measures on the probed device, CUDA events, warm, median of ``--reps``:

* HBM copy bandwidth (``y.copy_(x)`` over 1 GiB int32 tensors; bytes = read + write), a read-only pass (``x.sum()``
  over 1 GiB, bytes = read) and a write-only pass (``y.fill_``), so the "fraction of floor" columns of
  ``backends/numerical/reports/anchor_a100.md`` use achievable, not datasheet, bandwidth;
* INT8 tensor-core peak via ``torch._int_mm`` at 4096^3, 8192^3 and 16384^3 (cuBLAS ceiling), plus the two skinny
  shapes the sumcheck lanes hit (``(48, 32768, 96)`` and ``(2^21, 48, 48)``);
* BF16 dense matmul at 8192^3 and 16384^3 -- ``contract.NATIVE_REFERENCE`` is this card's datasheet BF16 peak
  (312 TFLOPS -> 9.85 ps per K=1536 VU); the measured fraction says how conservative that reference is.

Datasheet peaks (NVIDIA A100 SXM4 80GB): HBM2e 2,039 GB/s; INT8 624 TOPS dense = 312 T MAC/s; BF16 312 TFLOPS dense;
FP32 19.5 TFLOPS.  Nothing here is a proof; it is the anchor's own calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

DATASHEET = {"hbm_bytes_per_s": 2.039e12, "int8_dense_tops": 624.0, "int8_macs_per_s": 312.0e12, "bf16_dense_tflops": 312.0,
             "fp32_tflops": 19.5, "source": "NVIDIA A100 Tensor Core GPU datasheet (SXM4 80GB)"}


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
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.memory,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    return {"name": torch.cuda.get_device_name(0), "sms": p.multi_processor_count, "sm": f"{p.major}.{p.minor}",
            "total_memory_bytes": p.total_memory, "nvidia_smi": smi, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "host": platform.node()}


def bandwidth(reps: int, gib: int = 1) -> dict:
    n = gib * (1 << 30) // 4
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    out = {}
    t, ts = timed(lambda x=x, y=y: y.copy_(x), reps)
    out["copy"] = {"bytes": 2 * n * 4, "seconds": t, "gb_per_s": 2 * n * 4 / t / 1e9, "all": ts, "what": "y.copy_(x), 1 GiB int32, read+write counted"}
    t, ts = timed(lambda x=x: x.sum(dtype=torch.int64), reps)
    out["read"] = {"bytes": n * 4, "seconds": t, "gb_per_s": n * 4 / t / 1e9, "all": ts, "what": "x.sum(), 1 GiB int32, read only"}
    t, ts = timed(lambda y=y: y.fill_(7), reps)
    out["write"] = {"bytes": n * 4, "seconds": t, "gb_per_s": n * 4 / t / 1e9, "all": ts, "what": "y.fill_(), 1 GiB int32, write only"}
    # cold-ish L2 is irrelevant at 1 GiB (L2 is 40 MB), but a 4 GiB copy confirms the sustained number
    x = y = None
    n4 = 4 * (1 << 30) // 4
    x = torch.ones(n4, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    t, ts = timed(lambda x=x, y=y: y.copy_(x), max(3, reps // 2))
    out["copy_4gib"] = {"bytes": 2 * n4 * 4, "seconds": t, "gb_per_s": 2 * n4 * 4 / t / 1e9, "all": ts, "what": "y.copy_(x), 4 GiB int32"}
    x = y = None
    torch.cuda.empty_cache()
    for v in out.values():
        v["fraction_of_datasheet"] = v["gb_per_s"] * 1e9 / DATASHEET["hbm_bytes_per_s"]
    return out


def int8_peak(reps: int) -> dict:
    out = {}
    for m, k, n in ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384), (48, 32768, 96), (1 << 21, 48, 48)):
        a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda")
        try:
            t, ts = timed(lambda a=a, b=b: torch._int_mm(a, b), reps)
        except Exception as ex:  # noqa: BLE001
            out[f"{m}x{k}x{n}"] = {"error": repr(ex)}
            continue
        macs = m * k * n
        out[f"{m}x{k}x{n}"] = {"m": m, "k": k, "n": n, "seconds": t, "all": ts, "t_macs_per_s": macs / t / 1e12,
                               "fraction_of_int8_peak": macs / t / DATASHEET["int8_macs_per_s"],
                               "bytes_moved": m * k + k * n + 4 * m * n, "gb_per_s": (m * k + k * n + 4 * m * n) / t / 1e9}
        a = b = None
        torch.cuda.empty_cache()
    return out


def bf16_peak(reps: int) -> dict:
    out = {}
    for s in (8192, 16384):
        a = torch.randn(s, s, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(s, s, dtype=torch.bfloat16, device="cuda")
        t, ts = timed(lambda a=a, b=b: a @ b, reps)
        flops = 2 * s ** 3
        out[f"{s}^3"] = {"seconds": t, "all": ts, "tflops": flops / t / 1e12, "fraction_of_bf16_peak": flops / t / (DATASHEET["bf16_dense_tflops"] * 1e12)}
        a = b = None
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    doc = {"hardware": hardware(), "datasheet": DATASHEET, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(doc["hardware"]), flush=True)
    doc["bandwidth"] = bandwidth(a.reps)
    for k, v in doc["bandwidth"].items():
        print(f"bandwidth {k:10} {v['gb_per_s']:7.0f} GB/s  ({v['fraction_of_datasheet']:.2f} of 2,039)", flush=True)
    doc["int8"] = int8_peak(a.reps)
    for k, v in doc["int8"].items():
        if "error" in v:
            print(f"_int_mm {k}: {v['error']}", flush=True)
        else:
            print(f"_int_mm {k:18} {v['seconds']*1e3:8.3f} ms  {v['t_macs_per_s']:6.1f} T MAC/s = {v['fraction_of_int8_peak']:.3f} of peak  {v['gb_per_s']:.0f} GB/s", flush=True)
    doc["bf16"] = bf16_peak(a.reps)
    for k, v in doc["bf16"].items():
        print(f"bf16 matmul {k:8} {v['seconds']*1e3:8.3f} ms  {v['tflops']:6.1f} TFLOPS = {v['fraction_of_bf16_peak']:.3f} of peak", flush=True)
    doc["measured_peaks"] = {"hbm_copy_bytes_per_s": doc["bandwidth"]["copy"]["gb_per_s"] * 1e9,
                             "hbm_read_bytes_per_s": doc["bandwidth"]["read"]["gb_per_s"] * 1e9,
                             "hbm_write_bytes_per_s": doc["bandwidth"]["write"]["gb_per_s"] * 1e9,
                             "int8_macs_per_s_cublas_best": max(v["t_macs_per_s"] for v in doc["int8"].values() if "error" not in v) * 1e12,
                             "bf16_flops_per_s_cublas_best": max(v["tflops"] for v in doc["bf16"].values()) * 1e12}
    doc["elapsed_seconds"] = time.time() - t0
    doc["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "microbench.json").write_text(text)
    print(f"done in {doc['elapsed_seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
