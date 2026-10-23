"""The cost of the rotated int8 linear's online rotation on one GPU: `hadrot.hadrot_quant` (signed FWHT, the 3 x 3 mix
where k has a factor 3, per-token int8 quant) against main's plain per-token quant (`int8_fused.quant_rows`) and the int8
GEMM it feeds (`int8_fused.scaled_mm` at `pick`'s config, and torch._int_mm for M > 16), at [M, k] x [N, k]^T, N = 6144.

Gate first, in the same process (a failure times nothing, exit 4): at each k, the kernel's s_a against x Q in float64
(relative error under 1e-5) and its codes within 1 of the float64 codes, at most 1e-3 of them off by one.

Then per (k, M): each arm captured in a CUDA graph of `inner` calls (the GEMM cycling over enough weight copies to pass
3x the L2, so W streams from DRAM; the quant arms on one x, as a just-written activation is), warmed up, then `--reps`
replays per arm timed with CUDA events, arms interleaved rep by rep in a rotated order.  Reported: the median us a call.

    python bench.py --out OUT [--reps 60] [--ks 16384,12288,2048,1536,256] [--ms 1,8,32,64,2048,8192]
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
import hadrot as H  # noqa: E402

from verity_vllm.engine import int8_fused as F  # noqa: E402

N = 6144
LABELS = {16384: "o_proj full", 12288: "down dense full", 2048: "o_proj/8 TP8 = expert down full",
          1536: "down dense/8 TP8", 256: "expert down/8 TP8"}


def activation(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    out = torch.randint(0, k, (max(1, k // 256),), generator=g, device="cuda")
    x[:, out] *= 40.0
    return x.to(torch.bfloat16)


def gate(rot: H.Rotation, m: int = 64) -> dict:
    x = activation(m, rot.k, 7)
    xq = torch.empty(m, rot.k, dtype=torch.int8, device="cuda")
    s = torch.empty(m, dtype=torch.float32, device="cuda")
    H.hadrot_quant(rot, x, xq, s)
    y = rot.reference(x.float())
    s_ref = y.abs().amax(dim=1).clamp(min=1e-30) / 127
    codes = torch.round(y / s_ref[:, None]).clamp(-127, 127)
    rel = ((s.double() - s_ref).abs() / s_ref).max().item()
    diff = (xq.double() - codes).abs()
    doc = {"k": rot.k, "odd": rot.odd, "s_rel_max": rel, "codes_max_diff": diff.max().item(),
           "codes_off_frac": (diff > 0).double().mean().item()}
    doc["ok"] = rel < 1e-5 and doc["codes_max_diff"] <= 1 and doc["codes_off_frac"] <= 1e-3
    return doc


def graph(fn, inner: int):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for j in range(inner):
                fn(j)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(inner):
            fn(j)
    torch.cuda.synchronize()
    return g


def time_arms(graphs: dict, inner: int, reps: int) -> dict:
    names = list(graphs)
    for _ in range(5):
        for n in names:
            graphs[n].replay()
    torch.cuda.synchronize()
    ev = {n: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)] for n in names}
    for r in range(reps):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            a, b = ev[n][r]
            a.record()
            graphs[n].replay()
            b.record()
    torch.cuda.synchronize()
    out = {}
    for n in names:
        us = [a.elapsed_time(b) * 1e3 / inner for a, b in ev[n]]
        out[n] = {"median_us": statistics.median(us), "p10_us": sorted(us)[len(us) // 10],
                  "p90_us": sorted(us)[(9 * len(us)) // 10]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--ks", default="16384,12288,2048,1536,256")
    ap.add_argument("--ms", default="1,8,32,64,2048,8192")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(v) for v in args.ks.split(",")]
    ms = [int(v) for v in args.ms.split(",")]
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 128 << 20)
    meta = {"device": props.name, "sm": F.arch(), "l2_bytes": l2, "torch": torch.__version__,
            "triton": __import__("triton").__version__, "n": N, "reps": args.reps}
    print(json.dumps(meta), flush=True)
    rots = {k: H.Rotation(k) for k in ks}
    gates = [gate(r) for r in rots.values()]
    for g in gates:
        print("gate", json.dumps(g), flush=True)
    (out / "gate.json").write_text(json.dumps(gates, indent=1))
    if not all(g["ok"] for g in gates):
        print("gate failed: nothing timed", flush=True)
        return 4
    work = F.Workspace("cuda")
    rows = []
    for k in ks:
        rot = rots[k]
        copies = max(2, min(256, math.ceil(3 * l2 / (N * k))))
        w = torch.randint(-127, 128, (copies, N, k), dtype=torch.int8, device="cuda")
        s_w = torch.rand(N, device="cuda") * 1e-3 + 1e-4
        for m in ms:
            x = activation(m, k, 11 + m)
            xq = torch.empty(m, k, dtype=torch.int8, device="cuda")
            s_a = torch.empty(m, dtype=torch.float32, device="cuda")
            y = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
            cfg = F.pick(m, N, k)
            H.hadrot_quant(rot, x, xq, s_a)
            arms = {
                "rot": lambda j: H.hadrot_quant(rot, x, xq, s_a),
                "plain": lambda j: F.quant_rows(x, xq, s_a),
                "gemm": lambda j: F.scaled_mm(xq, s_a, w[j % copies], s_w, y, cfg=cfg, work=work),
            }
            if m > 16:
                z = torch.empty(m, N, dtype=torch.int32, device="cuda")
                arms["int_mm"] = lambda j: torch._int_mm(xq, w[j % copies].t(), out=z)
            graphs = {n: graph(f, copies) for n, f in arms.items()}
            t = time_arms(graphs, copies, args.reps)
            gemm = t["gemm"]["median_us"]
            best = min(gemm, t["int_mm"]["median_us"]) if "int_mm" in t else gemm
            row = {"k": k, "label": LABELS.get(k, ""), "m": m, "cfg": str(cfg), "inner": copies,
                   "rot_us": t["rot"]["median_us"], "plain_us": t["plain"]["median_us"], "gemm_us": gemm,
                   "int_mm_us": t.get("int_mm", {}).get("median_us"),
                   "added_pct_of_gemm": 100 * (t["rot"]["median_us"] - t["plain"]["median_us"]) / gemm,
                   "added_pct_of_best_gemm": 100 * (t["rot"]["median_us"] - t["plain"]["median_us"]) / best,
                   "spread": t}
            rows.append(row)
            print("row", json.dumps({kk: v for kk, v in row.items() if kk != "spread"}), flush=True)
            (out / "rows.json").write_text(json.dumps({"meta": meta, "gates": gates, "rows": rows}, indent=1))
            del graphs
        del w
        torch.cuda.empty_cache()
    print(f"{'k':>6} {'M':>5} {'rot+q us':>9} {'plain us':>9} {'gemm us':>9} {'int_mm us':>9} {'added %':>8}  label")
    for r in rows:
        im = f"{r['int_mm_us']:9.2f}" if r["int_mm_us"] else f"{'-':>9}"
        print(f"{r['k']:>6} {r['m']:>5} {r['rot_us']:9.2f} {r['plain_us']:9.2f} {r['gemm_us']:9.2f} {im} "
              f"{r['added_pct_of_gemm']:8.2f}  {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
