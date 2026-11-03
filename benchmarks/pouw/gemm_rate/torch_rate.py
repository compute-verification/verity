"""MACs per SM clock of torch's int8 GEMMs on uniform operands, over sustained windows between stamps of every SM.

    python torch_rate.py --stamp-cubin stamp.cubin --shape 8B-up:16384:28672:4096 [--shape ...]
                         [--wgmma-cubin F --cfg 128x256:c0:g8 ...] [--window-ms 2000 --lead-ms 500 --rounds 3] > torch.jsonl

Operands: x [M, K] and w [N, K], int8 uniform in [-127, 127] (fill-bench's V31_GEN uniform law), in HBM. The arms:

- `int_mm`: `torch._int_mm(x, w.t())`, cuBLASLt's s32 GEMM, as s8_linear's `int_mm.int_mm` calls it;
- `wgmma:CFG`, with --wgmma-cubin: s8_linear's `wgmma.scaled_mm` (wgmma.linear, bf16 out, s_a = s_w = 1/127) at each
  --cfg (`wgmma.Cfg.parse`'s text: tile, CTAs, raster group, `:m2` TMA multicast over a cluster of two, `:kN`), each
  checked equal to torch._int_mm's values scaled the same way before it is timed.

Each arm's calls run back to back on one stream for --window-ms after --lead-ms of them, between two launches of
stamp.cu's stamp_k (every SM's %globaltimer and %clock64 at one moment), so the window's SM clocks cover every call and
every gap. --rounds rounds, the arms in turn in each. Halfway through a window NVML's SM clock, clock event reasons and
temperature are sampled. Every window is one JSON line (`kind` window); `summarize.py` turns them into MACs per SM clock.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import sys
import threading
from pathlib import Path

import torch
from verity_config.settings import of

S = of(__file__)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import span  # noqa: E402

NVML_CLOCK_SM = 1
NVML_TEMPERATURE_GPU = 0
MAX_DYN_SMEM = 8  # CUfunction_attribute MAX_DYNAMIC_SHARED_SIZE_BYTES
STAMP_SMEM = 120 * 1024
vp, u, u64 = C.c_void_p, C.c_uint, C.c_uint64


class Nvml:
    """The leased GPU's NVML handle (GPU_LEASE_UUID, else CUDA device 0's UUID): its SM clock and clock event reasons."""

    def __init__(self) -> None:
        self.lib = C.CDLL(S.POUW_LIBNVML)
        if self.lib.nvmlInit_v2() != 0:
            raise SystemExit("nvmlInit_v2 failed")
        uuid = S.GPU_LEASE_UUID or f"GPU-{torch.cuda.get_device_properties(0).uuid}"
        self.h = vp()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.h)) != 0:
            raise SystemExit(f"no NVML device {uuid}")
        self.uuid = uuid
        name = "nvmlDeviceGetCurrentClocksEventReasons"
        self._reasons = getattr(self.lib, name, None) or self.lib.nvmlDeviceGetCurrentClocksThrottleReasons

    def sample(self) -> dict:
        mhz, temp, r = C.c_uint(), C.c_uint(), C.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, NVML_CLOCK_SM, C.byref(mhz))
        self.lib.nvmlDeviceGetTemperature(self.h, NVML_TEMPERATURE_GPU, C.byref(temp))
        self._reasons(self.h, C.byref(r))
        return {"nvml_mhz": mhz.value, "temp_c": temp.value, "reasons": r.value}


class Stamps:
    """stamp.cu's stamp_k from its cubin in torch's context, launched on torch's current stream: two stamps a window."""

    def __init__(self, cubin: Path) -> None:
        self.cu = C.CDLL(S.POUW_LIBCUDA)
        self.cu.cuLaunchKernel.argtypes = [vp, u, u, u, u, u, u, u, vp, C.POINTER(vp), C.POINTER(vp)]
        torch.cuda.init()
        dev, ctx, self.mod, self.f = C.c_int(), vp(), vp(), vp()
        self.ck("cuDeviceGet", C.byref(dev), torch.cuda.current_device())
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.ck("cuModuleLoadData", C.byref(self.mod), C.c_char_p(Path(cubin).read_bytes()))
        self.ck("cuModuleGetFunction", C.byref(self.f), self.mod, b"stamp_k")
        self.ck("cuFuncSetAttribute", self.f, MAX_DYN_SMEM, STAMP_SMEM)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        self.buf = torch.zeros((2, self.nsm, 3), dtype=torch.int64, device="cuda")
        self.arrive = torch.zeros(2, dtype=torch.int32, device="cuda")

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            raise SystemExit(f"{f} returned CUresult {r}")

    def stamp(self, i: int) -> None:
        a0, a1 = u64(self.buf[i].data_ptr()), u64(self.arrive[i].data_ptr())
        params = (vp * 2)(C.addressof(a0), C.addressof(a1))
        self.ck("cuLaunchKernel", self.f, self.nsm, 1, 1, 32, 1, 1, STAMP_SMEM,
                vp(torch.cuda.current_stream().cuda_stream), params, None)

    def span(self) -> dict:
        a, b = self.buf.cpu().tolist()
        return span(a, b)


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


def window(nv: Nvml, st: Stamps, fn, per: float, window_ms: float, lead_ms: float) -> dict:
    for _ in range(max(1, math.ceil(lead_ms / 1e3 / per)) if lead_ms > 0 else 0):
        fn()
    st.arrive.zero_()
    torch.cuda.synchronize()
    calls = max(1, math.ceil(window_ms / 1e3 / per))
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    mid: dict = {}
    timer = threading.Timer(window_ms / 2e3, lambda: mid.update(nv.sample()))
    timer.start()
    st.stamp(0)
    e0.record()
    for _ in range(calls):
        fn()
    e1.record()
    st.stamp(1)
    torch.cuda.synchronize()
    timer.join()
    return {"calls": calls, "ms": e0.elapsed_time(e1), **st.span(), **mid}


def arms(args, m: int, n: int, k: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda", generator=g)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
    out = {"int_mm": lambda: torch._int_mm(x, w.t())}
    if args.wgmma_cubin:
        os.environ["S8_WGMMA_CUBIN"] = str(Path(args.wgmma_cubin).resolve())
        sys.path.insert(0, str(HERE.parents[1] / "torch_gpu" / "s8_linear"))
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
                y.zero_()
                fn()
                torch.cuda.synchronize()
                if not torch.equal(y, ref):
                    raise SystemExit(f"{name} differs from torch._int_mm's values on {m}x{n}x{k}")
        del ref
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stamp-cubin", type=Path, required=True, help="stamp.cu built for the card (rate.sh's)")
    ap.add_argument("--shape", action="append", required=True, help="NAME:M:N:K")
    ap.add_argument("--wgmma-cubin", help="s8_linear's s8_wgmma.cubin (build.sh's): adds a wgmma arm a --cfg")
    ap.add_argument("--cfg", action="append", help="a wgmma.Cfg text (default wgmma.default_cfg's)")
    ap.add_argument("--window-ms", type=float, default=2000)
    ap.add_argument("--lead-ms", type=float, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    nv = Nvml()
    st = Stamps(args.stamp_cubin)
    emit = lambda d: print(json.dumps(d), flush=True)  # noqa: E731
    emit({"kind": "device", "uuid": nv.uuid, "name": torch.cuda.get_device_name(0), "sms": st.nsm,
          "torch": torch.__version__, "cuda": torch.version.cuda, **nv.sample()})
    for text in args.shape:
        name, m, n, k = shape(text)
        fns = arms(args, m, n, k)
        per = {a: per_call_s(fn) for a, fn in fns.items()}
        for r in range(args.rounds):
            for a, fn in fns.items():
                emit({"kind": "window", "shape": name, "m": m, "n": n, "k": k, "arm": a, "round": r,
                      **window(nv, st, fn, per[a], args.window_ms, args.lead_ms)})
        del fns
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
