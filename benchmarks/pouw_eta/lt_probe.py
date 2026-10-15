#!/usr/bin/env python3
"""cuBLASLt's FP8 GEMM through torch._scaled_mm (E4M3 x E4M3, per-tensor scales, BF16 out), timed like eta.py: W weights of
n x k in rotation (W = 3 is L2-cold at n = k = 8192), 2 W calls a CUDA graph, the graph back to back, η = m n k a second over
188 x 1,024 x the NVML SM clock.  ETA_MODE=profile runs each shape's call once under cudaProfilerStart/Stop for ncu.

  PEARLC_EXPECT_UUID=... ETA_SHAPES="32 128 8192" python lt_probe.py
"""
from __future__ import annotations

import ctypes as C
import json
import os
import statistics
import sys
import time

import torch

MACS_PER_SM_CLOCK, CAP_MHZ = 1024, 2430


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Nvml:
    def __init__(self, uuid: str):
        self.lib, self.h = None, C.c_void_p()
        try:
            lib = C.CDLL("libnvidia-ml.so.1")
        except OSError:
            return
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)) == 0:
            self.lib = lib

    def read(self) -> dict:
        if not self.lib:
            return {}
        sm, pw, thr = C.c_uint(), C.c_uint(), C.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, C.byref(sm))
        self.lib.nvmlDeviceGetPowerUsage(self.h, C.byref(pw))
        self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.h, C.byref(thr))
        return {"sm_mhz": sm.value, "power_w": pw.value / 1000, "throttle": hex(thr.value)}


def main() -> int:
    props = torch.cuda.get_device_properties(0)
    uuid = "GPU-" + str(props.uuid)
    expect = os.environ.get("PEARLC_EXPECT_UUID")
    nv = Nvml(uuid)
    emit({"kind": "env", "device": props.name, "uuid": uuid, "sms": props.multi_processor_count, "torch": torch.__version__,
          "cuda": torch.version.cuda, "expect": expect, "clk": nv.read(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if torch.cuda.device_count() != 1 or not expect or uuid != expect:
        print(f"REFUSED: uuid {uuid} expected {expect}, {torch.cuda.device_count()} devices", file=sys.stderr)
        return 3
    sms = props.multi_processor_count
    n = k = int(os.environ.get("ETA_NK", "8192"))
    reps = int(os.environ.get("ETA_REPS", "15"))
    profile = os.environ.get("ETA_MODE") == "profile"
    ws = [int(x) for x in os.environ.get("ETA_WS", "1 3").split()]
    g = torch.Generator(device="cuda").manual_seed(7)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    weights = [(torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn) for _ in range(max(ws))]
    for m in [int(x) for x in os.environ.get("ETA_SHAPES", "32 128 8192").split()]:
        acts = [(torch.randn(m, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn) for _ in range(max(ws))]
        outs = [torch.empty(m, n, dtype=torch.bfloat16, device="cuda") for _ in range(max(ws))]

        def call(j):
            torch._scaled_mm(acts[j], weights[j].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, out=outs[j])
        if profile:
            call(0)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            call(0)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            emit({"kind": "profile", "m": m, "n": n, "k": k})
            continue
        for w in ws:
            calls = 2 * w if w > 1 else 2
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(3):
                    for i in range(calls):
                        call(i % w)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(calls):
                    call(i % w)
            graph.replay()
            torch.cuda.synchronize()
            before = nv.read()
            t = []
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                t.append(a.elapsed_time(b))
            after = nv.read()
            per_call = statistics.median(t) / calls
            mhz = statistics.mean([c["sm_mhz"] for c in (before, after) if "sm_mhz" in c] or [0])
            useful = m * n * k / (per_call * 1e-3)
            emit({"kind": "lt", "shape": [m, n, k], "w": w, "calls_per_graph": calls, "graph_ms": [round(x, 4) for x in t],
                  "per_call_ms": round(per_call, 5), "tflops": round(2 * useful / 1e12, 1), "sm_mhz": mhz,
                  "eta": round(useful / (sms * MACS_PER_SM_CLOCK * mhz * 1e6), 5) if mhz else None,
                  "eta_cap": round(useful / (sms * MACS_PER_SM_CLOCK * CAP_MHZ * 1e6), 5),
                  "clk_before": before, "clk_after": after})
            del graph
    return 0


if __name__ == "__main__":
    sys.exit(main())
