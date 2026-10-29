"""On one GPU: `engine.int8_fused`'s fused producers against their separate kernels at the prefill shapes of Llama-3.2-1B
(H 2048, I 8192), M = 8,192, 16,384 and 32,768 tokens, served-like rows.

  gate     at every M and every num_warps tried, the fused kernel's outputs equal the unfused path's bit for bit:
             norm  `add_norm_quant` vs vLLM's batch-invariant `_C.fused_add_rms_norm` then `quant_rows` (x, residual, codes,
                   scales)
             silu  `silu_mul_quant` vs vLLM's `_C.silu_and_mul` then `quant_rows` (h, codes, scales)
           A failed gate times nothing (exit 4).
  sweep    each fused kernel's num_warps, timed interleaved in short items; the fastest is the fused arm below.
  launch   per launch, interleaved arm by arm: the unfused pair, its two kernels alone, and the fused kernel.
  forward  per forward, the int8 producers of one Llama-3.2-1B prefill (16 layers): 31 add-norms (every input norm but layer
           0's, every post-attention norm) and 16 silu-and-muls, each with its quantizer, unfused against fused at the
           engine's warps (`int8_fused.producer_warps`, which the sweep's fastest should be).

Each arm is captured in a CUDA graph and replayed in items of about ITEM_MS, the arms alternating item by item (order rotated
each rep), for at least `--sustain` seconds an arm; inputs are copied enough times to pass 3x the L2.  Reported: median
microseconds a launch (a forward), per-item spread, SM clocks around each item, and the saving per forward.

    cd integrations/vllm; PY=...venv/bin/python                      # PYTHONPATH=.:../..
    $PY tests/engine/int8_producers_gpu.py --out DIR [--ms 8192,16384,32768] [--sustain 2.0] [--shape 4096,14336,32]

`--shape H,I,LAYERS` times another model's widths (Llama-3.1-8B: 4096,14336,32).
"""
from __future__ import annotations

import os

os.environ["VLLM_BATCH_INVARIANT"] = "1"  # read once a process by vLLM's C++ (the served stack's norm launch)

import argparse
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

H, I, LAYERS, EPS = 2048, 8192, 16, 1e-5
NORMS, SILUS = 2 * LAYERS - 1, LAYERS
ITEM_MS = 10.0
NORM_WARPS = (1, 2, 4, 8, 16)
SILU_WARPS = (4, 8, 16, 32)


def nvml():
    """The leased GPU's SM clock in MHz, read through NVML, or None for every reading when NVML isn't there."""
    try:
        import pynvml
    except ImportError:
        return lambda: None
    try:
        pynvml.nvmlInit()
        uuid = os.environ.get("GPU_LEASE_UUID")
        h = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except pynvml.NVMLError:
        return lambda: None
    return lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int8: torch.int8}[t.dtype])


def same(*pairs) -> bool:
    return all(a.shape == b.shape and torch.equal(bits(a), bits(b)) for a, b in pairs)


def served_rows(m: int, k: int, seed: int, scale: float, outliers: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda") * scale
    if outliers:
        x[:, torch.randint(0, k, (outliers,), generator=g, device="cuda")] *= 300
    x[0] = 0.0
    x[1, 3] = float("nan")
    x[2, 5] = float("inf")
    return x.bfloat16()


def copies(nbytes: int) -> int:
    return max(2, math.ceil(3 * torch.cuda.get_device_properties(0).L2_cache_size / nbytes))


class Norm:
    """The add-norm's buffers at one M: `n` copies of (x, residual), one weight, one (xq, s)."""

    def __init__(self, m: int, n: int | None = None):
        self.m = m
        self.n = n or copies(2 * m * H * 2)
        self.xs = [served_rows(m, H, 10 + i, 0.05) for i in range(self.n)]
        self.rs = [served_rows(m, H, 100 + i, 1.0, outliers=3) for i in range(self.n)]
        self.w = (0.2 + torch.rand(H, device="cuda") * 2).bfloat16()
        self.xq = torch.empty(m, H, dtype=torch.int8, device="cuda")
        self.s = torch.empty(m, device="cuda")

    def unfused(self, part: str = "both"):
        from verity_vllm.engine import int8_fused as F

        def f():
            for x, r in zip(self.xs, self.rs, strict=True):
                if part != "quant":
                    torch.ops._C.fused_add_rms_norm(x, r, self.w, EPS)
                if part != "norm":
                    F.quant_rows(x, self.xq, self.s)
        return f

    def fused(self, warps: int | None):
        from verity_vllm.engine import int8_fused as F

        def f():
            for x, r in zip(self.xs, self.rs, strict=True):
                F.add_norm_quant(x, r, self.w, EPS, self.xq, self.s, warps)
        return f

    def gate(self, warps: int | None) -> bool:
        from verity_vllm.engine import int8_fused as F
        x, r = self.xs[0], self.rs[0]
        ux, ur, fx, fr = x.clone(), r.clone(), x.clone(), r.clone()
        uq, us = torch.full_like(self.xq, 0x5A), torch.full_like(self.s, float("nan"))
        fq, fs = torch.full_like(self.xq, 0x5A), torch.full_like(self.s, float("nan"))
        torch.ops._C.fused_add_rms_norm(ux, ur, self.w, EPS)
        F.quant_rows(ux, uq, us)
        F.add_norm_quant(fx, fr, self.w, EPS, fq, fs, warps)
        return same((fx, ux), (fr, ur), (fq, uq), (fs, us))


class Silu:
    """The silu-and-mul's buffers at one M: `n` copies of [gate | up], one h, one (xq, s)."""

    def __init__(self, m: int, n: int | None = None):
        self.m = m
        self.n = n or copies(m * 2 * I * 2)
        self.gus = [torch.cat([served_rows(m, I, 20 + i, 2.0), served_rows(m, I, 200 + i, 0.5)], 1) for i in range(self.n)]
        self.h = torch.empty(m, I, dtype=torch.bfloat16, device="cuda")
        self.xq = torch.empty(m, I, dtype=torch.int8, device="cuda")
        self.s = torch.empty(m, device="cuda")

    def unfused(self, part: str = "both"):
        from verity_vllm.engine import int8_fused as F

        def f():
            for gu in self.gus:
                if part != "quant":
                    torch.ops._C.silu_and_mul(self.h, gu)
                if part != "act":
                    F.quant_rows(self.h, self.xq, self.s)
        return f

    def fused(self, warps: int | None):
        from verity_vllm.engine import int8_fused as F

        def f():
            for gu in self.gus:
                F.silu_mul_quant(gu, self.h, self.xq, self.s, warps)
        return f

    def gate(self, warps: int | None) -> bool:
        from verity_vllm.engine import int8_fused as F
        gu = self.gus[0]
        uh, uq, us = torch.empty_like(self.h), torch.full_like(self.xq, 0x5A), torch.full_like(self.s, float("nan"))
        fh, fq, fs = torch.empty_like(self.h), torch.full_like(self.xq, 0x5A), torch.full_like(self.s, float("nan"))
        torch.ops._C.silu_and_mul(uh, gu)
        F.quant_rows(uh, uq, us)
        F.silu_mul_quant(gu, fh, fq, fs, warps)
        return same((fh, uh), (fq, uq), (fs, us))


def forward(norm: Norm, silu: Silu, fused: bool):
    """One prefill's int8 producers: per layer its input norm (not layer 0's), its post-attention norm and its silu, fused
    as the engine fuses them (at `producer_warps`' widths and warps)."""
    from verity_vllm.engine import int8_fused as F
    x0, x1, r = norm.xs[0], norm.xs[-1], norm.rs[0]
    gu = silu.gus[0]
    fuse_norm = fused and F.producer_warps("norm", norm.m, H) is not None
    fuse_silu = fused and F.producer_warps("silu", silu.m, I) is not None

    def f():
        for layer in range(LAYERS):
            for x in ((x0, x1) if layer else (x1,)):
                if fuse_norm:
                    F.add_norm_quant(x, r, norm.w, EPS, norm.xq, norm.s)
                else:
                    torch.ops._C.fused_add_rms_norm(x, r, norm.w, EPS)
                    F.quant_rows(x, norm.xq, norm.s)
            if fuse_silu:
                F.silu_mul_quant(gu, silu.h, silu.xq, silu.s)
            else:
                torch.ops._C.silu_and_mul(silu.h, gu)
                F.quant_rows(silu.h, silu.xq, silu.s)
    return f


def timed(fns: dict, seconds: float, clock, calls: dict) -> dict:
    """Each fn captured in a CUDA graph, replayed in items of about ITEM_MS, the fns alternating item by item (order rotated
    each rep) until each has run `seconds`.  Per fn: median, min and max microseconds a call over the items (`calls[fn]`
    calls a graph), the items, the seconds timed and the SM clocks around the items."""
    graphs, per = {}, {}
    for arm, f in fns.items():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                f()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        per[arm] = max(1, round(ITEM_MS / max(e0.elapsed_time(e1), 1e-3)))
        graphs[arm] = g
    arms = tuple(fns)
    us = {arm: [] for arm in arms}
    ms = {arm: 0.0 for arm in arms}
    clocks = {arm: [] for arm in arms}
    r = 0
    while min(ms.values()) < 1e3 * seconds:
        for j in range(len(arms)):
            arm = arms[(j + r) % len(arms)]
            c0 = clock()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(per[arm]):
                graphs[arm].replay()
            e1.record()
            torch.cuda.synchronize()
            c1 = clock()
            t = e0.elapsed_time(e1)
            ms[arm] += t
            us[arm].append(1e3 * t / (per[arm] * calls[arm]))
            clocks[arm].append([c0, c1])
        r += 1
    del graphs
    return {arm: {"us": statistics.median(v), "min": min(v), "max": max(v), "items": len(v), "seconds": ms[arm] / 1e3,
                  "replays_per_item": per[arm], "sm_mhz": sorted({c for cc in clocks[arm] for c in cc if c is not None})}
            for arm, v in us.items()}


def main() -> int:
    global H, I, LAYERS, NORMS, SILUS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ms", default="8192,16384,32768")
    ap.add_argument("--sustain", type=float, default=2.0)
    ap.add_argument("--sweep-seconds", type=float, default=0.3)
    ap.add_argument("--shape", default=f"{H},{I},{LAYERS}", help="H,I,LAYERS")
    a = ap.parse_args()
    H, I, LAYERS = (int(v) for v in a.shape.split(","))
    NORMS, SILUS = 2 * LAYERS - 1, LAYERS
    a.out.mkdir(parents=True, exist_ok=True)
    importlib.import_module("vllm._custom_ops")  # registers torch.ops._C
    from verity_vllm.engine import int8_fused as F
    clock = nvml()
    dev = torch.cuda.get_device_properties(0)
    doc: dict = {"device": dev.name, "l2_bytes": dev.L2_cache_size, "H": H, "I": I, "layers": LAYERS, "norms": NORMS,
                 "silus": SILUS, "batch_invariant": os.environ["VLLM_BATCH_INVARIANT"], "item_ms": ITEM_MS,
                 "sustain_s": a.sustain, "rows": []}
    ok = True
    for m in (int(v) for v in a.ms.split(",")):
        norm, silu = Norm(m), Silu(m)
        gates = {"norm": {w: norm.gate(w) for w in NORM_WARPS}, "silu": {w: silu.gate(w) for w in SILU_WARPS}}
        print(json.dumps({"m": m, "gates": gates}), flush=True)
        if not all(v for g in gates.values() for v in g.values()):
            ok = False
            doc["rows"].append({"m": m, "gates": gates})
            continue
        sweep_n = timed({f"norm_w{w}": norm.fused(w) for w in NORM_WARPS}, a.sweep_seconds, clock,
                        {f"norm_w{w}": norm.n for w in NORM_WARPS})
        sweep_s = timed({f"silu_w{w}": silu.fused(w) for w in SILU_WARPS}, a.sweep_seconds, clock,
                        {f"silu_w{w}": silu.n for w in SILU_WARPS})
        nw = min(NORM_WARPS, key=lambda w: sweep_n[f"norm_w{w}"]["us"])
        sw = min(SILU_WARPS, key=lambda w: sweep_s[f"silu_w{w}"]["us"])
        launch = timed({"norm_unfused": norm.unfused(), "norm_alone": norm.unfused("norm"),
                        "quant_h_alone": norm.unfused("quant"), "norm_fused": norm.fused(nw),
                        "silu_unfused": silu.unfused(), "silu_alone": silu.unfused("act"),
                        "quant_i_alone": silu.unfused("quant"), "silu_fused": silu.fused(sw)}, a.sustain, clock,
                       {"norm_unfused": norm.n, "norm_alone": norm.n, "quant_h_alone": norm.n, "norm_fused": norm.n,
                        "silu_unfused": silu.n, "silu_alone": silu.n, "quant_i_alone": silu.n, "silu_fused": silu.n})
        fwd = timed({"forward_unfused": forward(norm, silu, False), "forward_fused": forward(norm, silu, True)},
                    a.sustain, clock, {"forward_unfused": 1, "forward_fused": 1})
        d_norm = launch["norm_unfused"]["us"] - launch["norm_fused"]["us"]
        d_silu = launch["silu_unfused"]["us"] - launch["silu_fused"]["us"]
        row = {"m": m, "gates": gates, "norm_copies": norm.n, "silu_copies": silu.n, "norm_warps": nw, "silu_warps": sw,
               "engine_warps": {"norm": F.producer_warps("norm", m, H), "silu": F.producer_warps("silu", m, I)},
               "sweep": {**sweep_n, **sweep_s}, "launch": launch, "forward": fwd,
               "saving_per_launch_us": {"norm": d_norm, "silu": d_silu},
               "saving_per_forward_us_modelled": NORMS * d_norm + SILUS * d_silu,
               "saving_per_forward_us_timed": fwd["forward_unfused"]["us"] - fwd["forward_fused"]["us"],
               "quant_launches_removed": NORMS + SILUS,
               "norm_fused_tb_s": (norm.m * H * 2 * 4 + norm.m * H) / (launch["norm_fused"]["us"] * 1e6),
               "silu_fused_tb_s": (silu.m * I * 2 * 3 + silu.m * I) / (launch["silu_fused"]["us"] * 1e6)}
        doc["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("m", "norm_warps", "silu_warps", "engine_warps", "saving_per_launch_us",
                                              "saving_per_forward_us_modelled", "saving_per_forward_us_timed",
                                              "norm_fused_tb_s", "silu_fused_tb_s")}), flush=True)
        print(json.dumps({k: round(v["us"], 2) for k, v in {**launch, **fwd}.items()}), flush=True)
        del norm, silu
        torch.cuda.empty_cache()
    doc["ok"] = ok
    doc["versions"] = {"triton": __import__("triton").__version__, "torch": torch.__version__}
    (a.out / f"int8_producers_gpu_{H}x{I}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"ok": ok, "out": str(a.out / f"int8_producers_gpu_{H}x{I}.json")}), flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
