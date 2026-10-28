"""Board energy a MAC of torch's int8 GEMMs on uniform operands, from NVML's energy counter over back-to-back calls.

    python torch_energy.py --shape 8B-up:16384:28672:4096 [--shape ...] [--wgmma-cubin F --cfg 128x256:c0:g8 ...]
                           [--window-ms 4000 --lead-ms 1000 --rounds 3 --idle-s 8] > torch.jsonl

Operands: x [M, K] and w [N, K], int8 uniform in [-127, 127] (fill-bench's V31_GEN uniform law), in HBM. The arms:

- `int_mm`: `torch._int_mm(x, w.t())`, cuBLASLt's s32 GEMM, as s8_linear's `int_mm.int_mm` calls it;
- `wgmma:CFG`, with --wgmma-cubin: s8_linear's `wgmma.scaled_mm` (wgmma.linear, bf16 out, s_a = s_w = 1/127) at each
  --cfg (`wgmma.Cfg.parse`'s text: tile, CTAs, raster group, `:m2` TMA multicast over a cluster of two, `:kN`).

Each arm's calls run back to back on one stream for --window-ms after --lead-ms of them, and NVML's total energy is
read on the host just before and just after; the time is CUDA events'. --rounds rounds, the arms in turn in each.
Halfway through a window NVML's SM clock, clock event reasons and temperature are sampled. --idle-s of no work (the
context up) before and after give the board's idle power, and the counter's update period is measured under load.
Every window is one JSON line (`kind` window, idle or counter); `summarize.py` turns them into pJ a MAC.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import torch

NVML_CLOCK_SM = 1
NVML_TEMPERATURE_GPU = 0


class Nvml:
    """The leased GPU's NVML handle (GPU_LEASE_UUID, else CUDA device 0's UUID), read only."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(os.environ.get("POUW_LIBNVML", "libnvidia-ml.so.1"))
        if self.lib.nvmlInit_v2() != 0:
            raise SystemExit("nvmlInit_v2 failed")
        uuid = os.environ.get("GPU_LEASE_UUID") or f"GPU-{torch.cuda.get_device_properties(0).uuid}"
        self.h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h)) != 0:
            raise SystemExit(f"no NVML device {uuid}")
        self.uuid = uuid
        name = "nvmlDeviceGetCurrentClocksEventReasons"
        self._reasons = getattr(self.lib, name, None) or self.lib.nvmlDeviceGetCurrentClocksThrottleReasons

    def energy_mj(self) -> int:
        e = ctypes.c_ulonglong()
        if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)) != 0:
            raise SystemExit("nvmlDeviceGetTotalEnergyConsumption failed")
        return e.value

    def sample(self) -> dict:
        mhz, temp, r = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, NVML_CLOCK_SM, ctypes.byref(mhz))
        self.lib.nvmlDeviceGetTemperature(self.h, NVML_TEMPERATURE_GPU, ctypes.byref(temp))
        self._reasons(self.h, ctypes.byref(r))
        return {"nvml_mhz": mhz.value, "temp_c": temp.value, "reasons": r.value}


def shape(text: str) -> tuple[str, int, int, int]:
    name, m, n, k = text.split(":")
    return name, int(m), int(n), int(k)


def per_call_s(fn, calls: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(calls):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / 1e3 / calls


def window(nv: Nvml, fn, per: float, window_ms: float, lead_ms: float) -> dict:
    for _ in range(max(1, math.ceil(lead_ms / 1e3 / per)) if lead_ms > 0 else 0):
        fn()
    torch.cuda.synchronize()
    calls = max(1, math.ceil(window_ms / 1e3 / per))
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    mid: dict = {}
    timer = threading.Timer(window_ms / 2e3, lambda: mid.update(nv.sample()))
    h0 = time.time()
    j0 = nv.energy_mj()
    timer.start()
    e0.record()
    for _ in range(calls):
        fn()
    e1.record()
    e1.synchronize()
    j1 = nv.energy_mj()
    h1 = time.time()
    timer.join()
    return {"calls": calls, "ms": e0.elapsed_time(e1), "mj": j1 - j0, "host_t0": h0, "host_t1": h1, **mid}


def idle(nv: Nvml, s: float) -> dict:
    torch.cuda.synchronize()
    h0, j0 = time.time(), nv.energy_mj()
    time.sleep(s)
    h1, j1 = time.time(), nv.energy_mj()
    return {"kind": "idle", "s": h1 - h0, "mj": j1 - j0, "w": (j1 - j0) / 1e3 / (h1 - h0), **nv.sample()}


def counter_period(nv: Nvml, fn, s: float = 2.0) -> dict:
    """The counter's update period under load: the host times at which its value changes while fn runs back to back."""
    stop = threading.Event()

    def load() -> None:
        while not stop.is_set():
            for _ in range(8):
                fn()
            torch.cuda.synchronize()

    t = threading.Thread(target=load)
    t.start()
    changes, last, end = [], nv.energy_mj(), time.perf_counter() + s
    while time.perf_counter() < end:
        e = nv.energy_mj()
        if e != last:
            changes.append(time.perf_counter())
            last = e
    stop.set()
    t.join()
    gaps = sorted(b - a for a, b in zip(changes, changes[1:]))
    return {"kind": "counter", "s": s, "updates": len(changes),
            "period_ms": 1e3 * gaps[len(gaps) // 2] if gaps else None, "max_gap_ms": 1e3 * gaps[-1] if gaps else None}


def arms(args, m: int, n: int, k: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda", generator=g)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
    out = {"int_mm": lambda: torch._int_mm(x, w.t())}
    if args.wgmma_cubin:
        os.environ["S8_WGMMA_CUBIN"] = str(Path(args.wgmma_cubin).resolve())
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torch_gpu" / "s8_linear"))
        import wgmma

        sa = torch.full((m,), 1 / 127, dtype=torch.float32, device="cuda")
        sw = torch.full((n,), 1 / 127, dtype=torch.float32, device="cuda")
        y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        for c in args.cfg or [str(wgmma.default_cfg(m, n, k))]:
            cfg = wgmma.Cfg.parse(c)
            out[f"wgmma:{c}"] = lambda cfg=cfg: wgmma.scaled_mm(x, sa, w, sw, y, cfg)
        ref = torch._int_mm(x, w.t()).float().mul_(sa[:, None]).mul_(sw[None, :]).to(torch.bfloat16)
        for name, fn in out.items():
            if name.startswith("wgmma:"):
                fn()
                torch.cuda.synchronize()
                if not torch.equal(y, ref):
                    raise SystemExit(f"{name} differs from torch._int_mm's values on {m}x{n}x{k}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shape", action="append", required=True, help="NAME:M:N:K")
    ap.add_argument("--wgmma-cubin", help="s8_linear's s8_wgmma.cubin (build.sh's): adds a wgmma arm a --cfg")
    ap.add_argument("--cfg", action="append", help="a wgmma.Cfg text (default wgmma.default_cfg's)")
    ap.add_argument("--window-ms", type=float, default=4000)
    ap.add_argument("--lead-ms", type=float, default=1000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--idle-s", type=float, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    nv = Nvml()
    emit = lambda d: print(json.dumps(d), flush=True)  # noqa: E731
    emit({"kind": "device", "uuid": nv.uuid, "name": torch.cuda.get_device_name(0), "torch": torch.__version__,
          "cuda": torch.version.cuda, **nv.sample()})
    emit(idle(nv, args.idle_s))
    for text in args.shape:
        name, m, n, k = shape(text)
        fns = arms(args, m, n, k)
        per = {a: per_call_s(fn) for a, fn in fns.items()}
        emit({"shape": name, **counter_period(nv, fns["int_mm"])})
        for r in range(args.rounds):
            for a, fn in fns.items():
                emit({"kind": "window", "shape": name, "m": m, "n": n, "k": k, "arm": a, "round": r,
                      **window(nv, fn, per[a], args.window_ms, args.lead_ms)})
        del fns
        torch.cuda.empty_cache()
    emit(idle(nv, args.idle_s))


if __name__ == "__main__":
    main()
