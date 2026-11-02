"""On one GPU: `engine.rope_cache`'s fused rope and KV-cache write against vLLM's `_C.rotary_embedding` then
`_C_cache_ops.reshape_and_cache_flash` at the prefill shapes of Llama-3.2-1B (32 q heads, 8 KV heads, head 64, 16 layers),
T = 8,192, 16,384 and 32,768 tokens, served-like: sequences of 1,024 tokens (positions 0 to 1,023), each in its own run of
16-token blocks in a shuffled block order, on vLLM's table at base 500,000 and FA's NHD cache.

  gate     at every T and every (BLOCK_T, num_warps) tried, the fused kernel leaves the qkv buffer and the whole cache bit
           for bit as the two ops do (a failed gate times nothing, exit 4); the PTX passes `ptx_check`.
  sweep    each (BLOCK_T, num_warps), timed interleaved in short items; the fastest is the fused arm below.
  launch   per launch, interleaved: the two ops, each alone, the fused kernel at the sweep's best and at the engine's
           default (`rope_cache.BLOCK_T`, `WARPS`).
  forward  per forward, the 16 layers' rope and cache writes, each layer on its own qkv buffer and cache, the two ops
           against the fused kernel at the engine's default.

Each arm is captured in a CUDA graph and replayed in items of about ITEM_MS, the arms alternating item by item (order rotated
each rep), for at least `--sustain` seconds an arm; the per-launch inputs are copied enough times to pass 3x the L2.
Reported: median microseconds a launch (a forward), per-item spread, SM clocks around each item, bytes moved a launch and
TB/s, and the saving per forward.

    cd integrations/vllm; PY=...venv/bin/python                      # PYTHONPATH=.:../..
    $PY tests/engine/rope_cache_gpu.py --out DIR [--ts 8192,16384,32768] [--sustain 2.0]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

H, KVH, D, LAYERS, SEQ, BS = 32, 8, 64, 16, 1024, 16
ITEM_MS = 10.0
CONFIGS = [(bt, w) for bt in (1, 2, 4, 8) for w in (1, 2, 4, 8)]
FUSED_BYTES = (H + 2 * KVH) * D * 2 + (H + KVH) * D * 2 + 2 * KVH * D * 2      # read q k v, write q k, write k v to the cache
STOCK_BYTES = {"rope": 2 * (H + KVH) * D * 2, "cache": 2 * 2 * KVH * D * 2}


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
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def table(max_pos: int = 131072, base: float = 500000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float, device="cuda") / D))
    f = torch.einsum("i,j -> ij", torch.arange(max_pos, dtype=torch.float, device="cuda"), inv)
    return torch.cat((f.cos(), f.sin()), dim=-1).bfloat16()


def copies(nbytes: int) -> int:
    return max(2, math.ceil(3 * torch.cuda.get_device_properties(0).L2_cache_size / nbytes))


class Layer:
    """One layer's buffers at T: a served-like qkv buffer, positions, a slot mapping and FA's NHD cache sized to hold T
    tokens in shuffled blocks."""

    def __init__(self, t: int, seed: int, cs: torch.Tensor):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.t, self.cs = t, cs
        self.qkv = (torch.randn(t, (H + 2 * KVH) * D, generator=g, device="cuda") * 2.0).bfloat16()
        self.positions = (torch.arange(t, device="cuda") % SEQ).to(torch.int64)
        blocks = t // BS + 8
        order = torch.randperm(blocks, generator=g, device="cuda")
        i = torch.arange(t, device="cuda")
        self.slots = (order[i // BS] * BS + i % BS).to(torch.int64)
        self.kv_cache = torch.zeros(blocks, BS, KVH, 2 * D, dtype=torch.bfloat16, device="cuda").permute(0, 2, 1, 3)
        q, k, v = self.qkv.split([H * D, KVH * D, KVH * D], dim=-1)
        self.q, self.k, self.v = q, k, v
        self.kc, self.vc = self.kv_cache.transpose(1, 2).split(D, dim=-1)
        self.one = torch.ones(1, dtype=torch.float32, device="cuda")

    def stock(self, part: str = "both"):
        if part != "cache":
            torch.ops._C.rotary_embedding(self.positions, self.q, self.k, D, self.cs, True)
        if part != "rope":
            torch.ops._C_cache_ops.reshape_and_cache_flash(self.k.view(-1, KVH, D), self.v.view(-1, KVH, D), self.kc, self.vc,
                                                           self.slots, "auto", self.one, self.one)

    def fused(self, cfg: tuple[int, int] | None = None):
        from verity_vllm.engine import rope_cache as R
        kw = {"block_t": cfg[0], "num_warps": cfg[1]} if cfg else {}
        return R.rope_cache(self.positions, self.q, self.k, self.v, self.cs, self.kc, self.vc, self.slots, **kw)

    def gate(self, cfg: tuple[int, int]) -> dict:
        from verity_vllm.engine import rope_cache as R
        qkv0, cache0 = self.qkv.clone(), self.kv_cache.clone()
        self.stock()
        want_qkv, want_cache = self.qkv.clone(), self.kv_cache.clone()
        self.qkv.copy_(qkv0)
        self.kv_cache.copy_(cache0)
        launch = self.fused(cfg)
        ok = torch.equal(bits(self.qkv), bits(want_qkv)) and torch.equal(bits(self.kv_cache), bits(want_cache))
        self.qkv.copy_(qkv0)
        self.kv_cache.copy_(cache0)
        return {"equal": ok, "ptx": R.ptx_check(launch)["ok"]}


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


def each(layers: list[Layer], f):
    def run():
        for x in layers:
            f(x)
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ts", default="8192,16384,32768")
    ap.add_argument("--sustain", type=float, default=2.0)
    ap.add_argument("--sweep-seconds", type=float, default=0.3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    importlib.import_module("vllm._custom_ops")  # registers torch.ops._C and torch.ops._C_cache_ops
    from verity_vllm.engine import rope_cache as R
    clock = nvml()
    dev = torch.cuda.get_device_properties(0)
    cs = table()
    engine_cfg = (R.BLOCK_T, R.WARPS)
    doc: dict = {"device": dev.name, "l2_bytes": dev.L2_cache_size, "H": H, "KVH": KVH, "D": D, "layers": LAYERS, "seq": SEQ,
                 "block_size": BS, "item_ms": ITEM_MS, "sustain_s": a.sustain, "engine_cfg": engine_cfg,
                 "fused_bytes_per_token": FUSED_BYTES, "stock_bytes_per_token": STOCK_BYTES, "rows": []}
    ok = True
    for t in (int(v) for v in a.ts.split(",")):
        n = copies(t * (H + 2 * KVH) * D * 2)
        reps = [Layer(t, 10 + i, cs) for i in range(n)]
        gates = {f"{bt}x{w}": reps[0].gate((bt, w)) for bt, w in CONFIGS}
        print(json.dumps({"t": t, "gates": gates}), flush=True)
        if not all(g["equal"] and g["ptx"] for g in gates.values()):
            ok = False
            doc["rows"].append({"t": t, "gates": gates})
            continue
        sweep = timed({f"{bt}x{w}": each(reps, lambda x, c=(bt, w): x.fused(c)) for bt, w in CONFIGS}, a.sweep_seconds, clock,
                      {f"{bt}x{w}": n for bt, w in CONFIGS})
        best = min(CONFIGS, key=lambda c: sweep[f"{c[0]}x{c[1]}"]["us"])
        launch = timed({"stock": each(reps, lambda x: x.stock()), "rope_alone": each(reps, lambda x: x.stock("rope")),
                        "cache_alone": each(reps, lambda x: x.stock("cache")),
                        "fused_best": each(reps, lambda x: x.fused(best)),
                        "fused_engine": each(reps, lambda x: x.fused(engine_cfg))}, a.sustain, clock,
                       dict.fromkeys(("stock", "rope_alone", "cache_alone", "fused_best", "fused_engine"), n))
        del reps
        torch.cuda.empty_cache()
        layers = [Layer(t, 100 + i, cs) for i in range(LAYERS)]
        fwd = timed({"forward_stock": each(layers, lambda x: x.stock()),
                     "forward_fused": each(layers, lambda x: x.fused(engine_cfg))}, a.sustain, clock,
                    {"forward_stock": 1, "forward_fused": 1})
        d = launch["stock"]["us"] - launch["fused_engine"]["us"]
        row = {"t": t, "gates": gates, "copies": n, "best": best, "sweep": sweep, "launch": launch, "forward": fwd,
               "saving_per_launch_us": d, "saving_per_forward_us_modelled": LAYERS * d,
               "saving_per_forward_us_timed": fwd["forward_stock"]["us"] - fwd["forward_fused"]["us"],
               "launches_removed_per_forward": LAYERS,
               "fused_tb_s": t * FUSED_BYTES / (launch["fused_engine"]["us"] * 1e6),
               "fused_best_tb_s": t * FUSED_BYTES / (launch["fused_best"]["us"] * 1e6),
               "rope_tb_s": t * STOCK_BYTES["rope"] / (launch["rope_alone"]["us"] * 1e6),
               "cache_tb_s": t * STOCK_BYTES["cache"] / (launch["cache_alone"]["us"] * 1e6)}
        doc["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("t", "best", "saving_per_launch_us", "saving_per_forward_us_modelled",
                                              "saving_per_forward_us_timed", "fused_tb_s", "fused_best_tb_s", "rope_tb_s",
                                              "cache_tb_s")}), flush=True)
        print(json.dumps({k: round(v["us"], 2) for k, v in {**launch, **fwd}.items()}), flush=True)
        del layers
        torch.cuda.empty_cache()
    doc["ok"] = ok
    doc["versions"] = {"triton": __import__("triton").__version__, "torch": torch.__version__}
    (a.out / "rope_cache_gpu.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"ok": ok, "out": str(a.out / "rope_cache_gpu.json")}), flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
