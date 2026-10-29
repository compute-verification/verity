"""The exact linear at kernel level, its arms interleaved rep by rep on one GPU, after `kgate.gate()` passed in the same
process (a failed gate times nothing, exit 4):

    exact   the int8-exact profile's op bodies as served: `int8_exact.quantize_rows`, then `dequantize(int_mm(...))`
    stock   vLLM's W8A8 int8 on sm_120: `ops.scaled_int8_quant`, then `triton_scaled_mm` (not the exact statement; the
            speed reference)
    fused   `exactlin.quant_rows`, then `exactlin.scaled_mm`, into caller-owned buffers

Each arm's linear is captured in a CUDA graph per shape, one call per weight copy (enough copies to pass 3x the L2, so the
weights stream from DRAM as they do in a served step).  After a warm-up, each arm's replays per item are set so an item takes
about ITEM_MS, and the arms alternate item by item (order rotated each rep), with the SM clock read (NVML) before and after
every item.  Reported per shape: median microseconds a call per arm, the ratios, and the clocks; and per model's decode step
at M = 32 (its layers x qkv, o, gate_up, down) the predicted linear time of each arm.  `--arms` also takes the single
kernels (PARTS: stock's quant, stock's GEMM with vLLM's default, without and with tensor descriptors, the fused quant and
GEMM alone, the fused GEMM on vLLM's tiles, stock's CUTLASS linear, and the exact quant with `kgate.cutlass_swapped`, or
that GEMM alone; the last two time only M a multiple of 16).

    python kbench.py --out OUT [--reps 15] [--shapes all|decode] [--models llama-3.1-8b,qwen3-8b] [--arms ...] [--skip-gate]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactlin as X
import kgate

ITEM_MS = 10.0
ARMS = ("exact", "stock", "fused")
PARTS = ("stock_quant", "stock_mm", "stock_mm_notd", "stock_mm_td", "fused_quant", "fused_mm", "fused_vllm_tiles",
         "stock_cutlass", "cutlass_swap", "cutlass_swap_mm")
WORK = X.Workspace("cuda") if torch.cuda.is_available() else None


def nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        import os
        uuid = os.environ.get("GPU_LEASE_UUID")
        h = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(idx)
        return lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:  # noqa: BLE001
        return lambda: None


def arm_fn(arm: str, x, ws, xq, s, y):
    """The arm's linear over every weight copy, or None where it doesn't apply to this M."""
    from verity_vllm.engine import int8_exact as E
    if arm == "exact":
        def f():
            for w, sw in ws:
                q, sa = E.quantize_rows(x)
                E.dequantize(E.int_mm(q, w), sa, sw)
    elif arm == "stock":
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )

        def f():
            for w, sw in ws:
                q, sa, _ = ops.scaled_int8_quant(x)
                triton_scaled_mm(q, w.t(), sa, sw, torch.bfloat16)
    elif arm == "stock_cutlass":
        from vllm import _custom_ops as ops

        def f():
            for w, sw in ws:
                q, sa, _ = ops.scaled_int8_quant(x)
                ops.cutlass_scaled_mm(q, w.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
    elif arm in ("cutlass_swap", "cutlass_swap_mm"):
        import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)
        if x.shape[0] % 16:
            return None
        X.quant_rows(x, xq, s)

        def f():
            for w, sw in ws:
                if arm == "cutlass_swap":
                    X.quant_rows(x, xq, s)
                kgate.cutlass_swapped(xq, s, w, sw)
    elif arm in ("fused", "fused_vllm_tiles"):
        m, k = x.shape
        cfg = X.pick(m, ws[0][0].shape[0], k) if arm == "fused" else X.default_cfg(m, ws[0][0].shape[0], k)

        def f():
            for w, sw in ws:
                X.quant_rows(x, xq, s)
                X.scaled_mm(xq, s, w, sw, y, cfg, WORK)
    elif arm.startswith("stock_"):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )
        q0, s0, _ = ops.scaled_int8_quant(x)
        td = {"stock_mm": None, "stock_mm_notd": False, "stock_mm_td": True}.get(arm)

        def f():
            for w, sw in ws:
                if arm == "stock_quant":
                    ops.scaled_int8_quant(x)
                else:
                    triton_scaled_mm(q0, w.t(), s0, sw, torch.bfloat16, use_td=td)
    else:
        X.quant_rows(x, xq, s)
        m, k = x.shape
        cfg = X.pick(m, ws[0][0].shape[0], k)

        def f():
            for w, sw in ws:
                if arm == "fused_quant":
                    X.quant_rows(x, xq, s)
                else:
                    X.scaled_mm(xq, s, w, sw, y, cfg, WORK)
    return f


def copies_for(n: int, k: int) -> int:
    return max(2, math.ceil(3 * torch.cuda.get_device_properties(0).L2_cache_size / (n * k)))


def timed(fns: dict, reps: int, clock, calls: int, around=None) -> tuple[dict, dict, dict]:
    """Each fn captured in a CUDA graph and replayed in items of about ITEM_MS, the fns alternating item by item (order
    rotated each rep); median microseconds a call (`calls` calls a graph), every item's, and the SM clocks around each.
    `around`: a context manager factory each timed item runs inside (its exit before the item's device sync)."""
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
    clocks = {arm: [] for arm in arms}
    for r in range(reps):
        for j in range(len(arms)):
            arm = arms[(j + r) % len(arms)]
            g = graphs[arm]
            c0 = clock()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with around() if around else contextlib.nullcontext():
                e0.record()
                for _ in range(per[arm]):
                    g.replay()
                e1.record()
            torch.cuda.synchronize()
            c1 = clock()
            us[arm].append(1e3 * e0.elapsed_time(e1) / (per[arm] * calls))
            clocks[arm].append([c0, c1])
    del graphs
    return {arm: statistics.median(v) for arm, v in us.items()}, us, {"clocks": clocks, "replays_per_item": per}


def shape_row(n: int, k: int, m: int, reps: int, clock, arms=ARMS, names: list[str] | None = None,
              special: bool = False) -> dict:
    name = ",".join(names or [])
    copies = copies_for(n, k)
    x = kgate.activations(m, k, 7 + m, special=special)
    ws = [kgate.weights(n, k, 100 + i) for i in range(copies)]
    xq = torch.empty((m, k), dtype=torch.int8, device="cuda")
    s = torch.empty(m, dtype=torch.float32, device="cuda")
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    fns = {arm: arm_fn(arm, x, ws, xq, s, y) for arm in arms}
    arms = tuple(arm for arm in arms if fns[arm] is not None)
    med, us, meta = timed({arm: fns[arm] for arm in arms}, reps, clock, copies)
    clocks, per = meta["clocks"], meta["replays_per_item"]
    row = {"layer": name, "m": m, "n": n, "k": k, "copies": copies, "replays_per_item": per, "us_per_call": med,
           "us_all": {a: [round(v, 3) for v in us[a]] for a in arms}, "sm_clock_mhz": clocks,
           **({"exact_over_stock": med["exact"] / med["stock"], "fused_over_stock": med["fused"] / med["stock"],
               "fused_over_exact": med["fused"] / med["exact"]} if set(ARMS) <= set(arms) else {})}
    del x, ws, xq, s, y
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS), help=f"of {ARMS + PARTS}")
    ap.add_argument("--models", default="llama-3.1-8b", help=f"comma-separated, of {tuple(kgate.MODELS)}")
    ap.add_argument("--special-rows", action="store_true", help="time on the gate's edge rows too (default: plain rows)")
    ap.add_argument("--dump", action="store_true", help="the gate's M <= 32 outputs to OUT/replay (replay.py)")
    a = ap.parse_args()
    arms = tuple(a.arms.split(","))
    models = tuple(a.models.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        doc = kgate.gate(a.shapes, models=models, dump=out / "replay" if a.dump else None)
        (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"gate_ok": doc["ok"], "shapes": doc["shapes"]}), flush=True)
        if not doc["ok"]:
            print(json.dumps({"timed": False, "why": "gate failed"}), flush=True)
            return 4
    clock = nvml()
    ms = kgate.MS if a.shapes == "all" else kgate.DECODE_MS
    rows = []
    for n, k, m, names in kgate.shapes_of(models, ms):
        row = shape_row(n, k, m, a.reps, clock, arms, names, a.special_rows)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("us_all", "sm_clock_mhz")}), flush=True)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    step: dict = {}
    for model in models:
        spec = kgate.MODELS[model]
        at32 = {(r["n"], r["k"]): r for r in rows if r["m"] == 32}
        step[model] = {}
        for arm in arms:
            per_layer = sum(at32[nk]["us_per_call"][arm] for nk in spec["linears"].values())
            step[model][arm] = {"us_per_layer_m32": per_layer, "ms_per_step_m32": spec["layers"] * per_layer / 1e3}
    from vllm.triton_utils import use_tensor_descriptor
    summary = {"device": torch.cuda.get_device_name(), "reps": a.reps, "arms": arms, "models": models,
               "special_rows": a.special_rows, "arch": X.arch(),
               "configs": {f"{n}x{k}": v for (n, k), v in X.CONFIGS.get(X.arch(), {}).items()},
               "vllm_use_tensor_descriptor_default": bool(use_tensor_descriptor(None)), "item_ms": ITEM_MS, "decode_step_b32": step,
               "rows": [{k: v for k, v in r.items() if k not in ("us_all", "sm_clock_mhz")} for r in rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"decode_step_b32": step}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
