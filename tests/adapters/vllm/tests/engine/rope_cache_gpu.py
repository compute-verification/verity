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

    cd adapters/vllm; PY=...venv/bin/python                      # PYTHONPATH=.:../..
    $PY tests/engine/rope_cache_gpu.py --out DIR [--ts 8192,16384,32768] [--sustain 2.0]
"""
from __future__ import annotations

import math
import os
import statistics

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
    us: dict[str, list[float]] = {arm: [] for arm in arms}
    ms = {arm: 0.0 for arm in arms}
    clocks: dict[str, list[list[float | None]]] = {arm: [] for arm in arms}
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






