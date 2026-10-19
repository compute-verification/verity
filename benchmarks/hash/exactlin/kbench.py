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
GEMM alone).

    python kbench.py --out OUT [--reps 15] [--shapes all|decode] [--models llama-3.1-8b,qwen3-8b] [--arms ...] [--skip-gate]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exactlin as X  # noqa: E402
import kgate  # noqa: E402

ITEM_MS = 10.0
ARMS = ("exact", "stock", "fused")
PARTS = ("stock_quant", "stock_mm", "stock_mm_notd", "stock_mm_td", "fused_quant", "fused_mm")


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
    from verity_vllm.engine import int8_exact as E
    if arm == "exact":
        def f():
            for w, sw in ws:
                q, sa = E.quantize_rows(x)
                E.dequantize(E.int_mm(q, w), sa, sw)
    elif arm == "stock":
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm

        def f():
            for w, sw in ws:
                q, sa, _ = ops.scaled_int8_quant(x)
                triton_scaled_mm(q, w.t(), sa, sw, torch.bfloat16)
    elif arm == "fused":
        def f():
            for w, sw in ws:
                X.quant_rows(x, xq, s)
                X.scaled_mm(xq, s, w, sw, y)
    elif arm.startswith("stock_"):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
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

        def f():
            for w, sw in ws:
                if arm == "fused_quant":
                    X.quant_rows(x, xq, s)
                else:
                    X.scaled_mm(xq, s, w, sw, y)
    return f


def shape_row(n: int, k: int, m: int, reps: int, clock, arms=ARMS, names: list[str] | None = None) -> dict:
    name = ",".join(names or [])
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    copies = max(2, math.ceil(3 * l2 / (n * k)))
    x = kgate.activations(m, k, 7 + m)
    ws = [kgate.weights(n, k, 100 + i) for i in range(copies)]
    xq = torch.empty((m, k), dtype=torch.int8, device="cuda")
    s = torch.empty(m, dtype=torch.float32, device="cuda")
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    graphs, per = {}, {}
    for arm in arms:
        f = arm_fn(arm, x, ws, xq, s, y)
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
    us = {arm: [] for arm in arms}
    clocks = {arm: [] for arm in arms}
    for r in range(reps):
        for j in range(len(arms)):
            arm = arms[(j + r) % len(arms)]
            g = graphs[arm]
            c0 = clock()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(per[arm]):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            c1 = clock()
            us[arm].append(1e3 * e0.elapsed_time(e1) / (per[arm] * copies))
            clocks[arm].append([c0, c1])
    med = {arm: statistics.median(v) for arm, v in us.items()}
    row = {"layer": name, "m": m, "n": n, "k": k, "copies": copies, "replays_per_item": per, "us_per_call": med,
           "us_all": {a: [round(v, 3) for v in us[a]] for a in arms}, "sm_clock_mhz": clocks,
           **({"exact_over_stock": med["exact"] / med["stock"], "fused_over_stock": med["fused"] / med["stock"],
               "fused_over_exact": med["fused"] / med["exact"]} if set(ARMS) <= set(arms) else {})}
    del graphs, x, ws, xq, s, y
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
    a = ap.parse_args()
    arms = tuple(a.arms.split(","))
    models = tuple(a.models.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        doc = kgate.gate(a.shapes, models=models)
        (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"gate_ok": doc["ok"], "shapes": doc["shapes"]}), flush=True)
        if not doc["ok"]:
            print(json.dumps({"timed": False, "why": "gate failed"}), flush=True)
            return 4
    clock = nvml()
    ms = kgate.MS if a.shapes == "all" else kgate.DECODE_MS
    rows = []
    for n, k, m, names in kgate.shapes_of(models, ms):
        row = shape_row(n, k, m, a.reps, clock, arms, names)
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
               "vllm_use_tensor_descriptor_default": bool(use_tensor_descriptor(None)), "item_ms": ITEM_MS, "decode_step_b32": step,
               "rows": [{k: v for k, v in r.items() if k not in ("us_all", "sm_clock_mhz")} for r in rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"decode_step_b32": step}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
