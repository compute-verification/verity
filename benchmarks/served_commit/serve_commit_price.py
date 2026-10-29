"""The minimum serve-time commitment's cost, J_serve, from hash_cost's model: stdlib only, no GPU.

    python3 benchmarks/served_commit/serve_commit_price.py OUT_JSON

h(rb, σ) = h_x − σ h_leaf with σ = min(1, G/B) (hash_cost_model.py), so a forward's hashing takes
(B h_x − h_leaf min(B, G)) / (540,672 f) seconds for B committed bytes, at the SM clock f. G = η t_gap r_leaf(f), with η
the share of a host gap t_gap that leaves fill (measured at 8.55 ms, r20261028-202719-cc9b). J_serve is that time over the
stock step's. Every input below is a measurement (run id) or an estimate (its source), and the output says which.

Arms, a token: PoUW's Dither receipts (the prefix words T_1..T_{S-1} at 13 low bits, with or without O at 32 bits a cell),
plus the values: (i) GD-PARTITION's (e2) cut or its variant without gathers, or (ii) the lazy arm, tokens only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RULER = 4096 * 132                                  # MAC-times a cycle: h × bytes / (RULER × f) is seconds
F = 1.965e9                                         # the step loops' SM clock (NVML, r20261028-202719-cc9b)
GAP_MEASURED = 16.8e6 / F                           # the loop's host gap, a 16.8 M-cycle sleep

# measured: art:5cf3621a988a946e98dcb8d36eafbecc3496b898df7ae7f481f757f5308ff991 (model.json, keys 8192:step:gapfirst)
ROWS = {4096: {"h_x": 2703.3, "h_leaf": 2569.9, "G": 3.4545e9},
        16384: {"h_x": 2123.4, "h_leaf": 2077.6, "G": 4.4836e9},
        65536: {"h_x": 1967.6, "h_leaf": 1948.3, "G": 3.6634e9}}
THREADS = 132 * 256                                 # a leaf a thread (native_rows.cu): fewer leaves leave SMs short of warps

# Llama-3.2-1B's linears a token: (outputs, K) for qkv, o, gate_up and down, 16 layers
LINEARS = [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]
LAYERS = 16
L_SEG = 768                                         # Dither t = 3, π a segment: L = 3 × 256 (GD-BITS)
T_BITS = 13                                         # the low 13 bits of each prefix word (GD-BITS)
D_MODEL, D_MODEL_V4 = 2048, 7168


def words(lin: list, L: int) -> int:
    return sum(n * (math.ceil(k / L) - 1) for n, k in lin)


MACS = LAYERS * sum(n * k for n, k in LINEARS)
CELLS = LAYERS * sum(n for n, _ in LINEARS)
PARTS = {                                           # bytes a token (estimates, from the notes named)
    "T": LAYERS * words(LINEARS, L_SEG) * T_BITS / 8,           # GD-BITS 1.65 MB (note:compute-research/20261028T2024Z-report-gd-bits)
    "O": CELLS * 4,                                             # O at 32 bits a cell (dither/PROTOCOL.md, "What is committed")
    "e2": 792_832,                                              # GD-PARTITION (e2), context ≤ 35 (note:compute-research/20261028T2045Z-report-gd-partition)
    "e2_nogather": 792_832 - LAYERS * 14_336,                   # (e2) without C-Flock's gather copies, 14,336 B a layer
    "tokens": 8,                                                # an input and an output id, 4 B each
}
ARMS = {"(i) e2": ["T", "e2"], "(i) e2, no gathers": ["T", "e2_nogather"], "(ii) lazy": ["T", "tokens"]}

# the engines (estimates unless a run id is named)
POINTS = {
    "T 8,192": {"tokens": 8192, "rb": 16384, "engines": {
        "sync, gap first": {"stock_ms": 37.77, "gap_ms": GAP_MEASURED * 1e3,
                            "src": "stock r20261028-023029-1f82 (HASH-56); G measured on the step loop's 8.55 ms gap (cc9b)"},
        "async, gap first": {"stock_ms": 37.77 - 8.73 + 2.29, "gap_ms": 2.29,
                             "src": "estimate: HASH-56's step with p1a-levers' async gap, 2.29 ms (7.88 ms sync fit × 2.29/7.88)"},
        "async, no free room": {"stock_ms": 37.77 - 8.73 + 2.29, "gap_ms": 0.0, "src": "estimate: the async gap busy"}}},
    "T 512 (64 real)": {"tokens": 64, "rb": 4096, "engines": {
        "sync, gap first": {"stock_ms": 0.96 + 2.46, "gap_ms": 2.46,
                            "src": "estimate: p1a-levers' T 512 parts (kernels 0.96 ms, host gap 2.46 ms; T 8,192 scaled)"},
        "async, gap first": {"stock_ms": 0.96 + 2.46 * 2.29 / 7.88, "gap_ms": 2.46 * 2.29 / 7.88,
                             "src": "estimate: the same with the async gap ratio"},
        "async, no free room": {"stock_ms": 0.96 + 2.46 * 2.29 / 7.88, "gap_ms": 0.0, "src": "estimate: the async gap busy"}}},
}
GPU_BUSY_MS = 37.77 - 8.73                          # T 8,192 stock step less its host gap: the base for world scale


def hash_ms(B: float, rb: int, gap_ms: float) -> dict:
    r = ROWS[rb]
    eta = r["h_leaf"] * r["G"] / (RULER * F) / GAP_MEASURED
    G = eta * gap_ms * 1e-3 * RULER * F / r["h_leaf"]
    fill = min(1.0, B / rb / THREADS)
    excl = B * r["h_x"] / (RULER * F) * 1e3
    # unfilled: every rate falls by fill, the gap's room with it
    return {"G_bytes": G, "eta": eta, "exclusive_ms": excl, "ms": excl - r["h_leaf"] * min(B, G) / (RULER * F) * 1e3,
            "leaves": B / rb, "fill": fill, "ms_unfilled": (excl - r["h_leaf"] * min(B, G * fill) / (RULER * F) * 1e3) / fill}


def main() -> None:
    out: dict = {"parts_bytes_a_token": PARTS, "linear_macs_a_token": MACS, "cells_a_token": CELLS, "points": {}}
    for pname, P in POINTS.items():
        rows = out["points"].setdefault(pname, {"tokens_a_forward": P["tokens"], "rows": P["rb"], "arms": {}})
        for with_o in (False, True):
            for aname, parts in ARMS.items():
                name = aname + (" + O" if with_o else "")
                bpt = sum(PARTS[p] for p in parts + (["O"] if with_o else []))
                B = bpt * P["tokens"]
                e = {}
                for ename, E in P["engines"].items():
                    h = hash_ms(B, P["rb"], E["gap_ms"])
                    e[ename] = {**h, "stock_ms": E["stock_ms"], "J_serve": h["ms"] / E["stock_ms"],
                                "J_serve_unfilled": h["ms_unfilled"] / E["stock_ms"], "src": E["src"]}
                rows["arms"][name] = {"bytes_a_token": bpt, "bytes_a_forward": B, "engines": e}
    # world scale (estimate): the gap is ~0 against V4's GPU time a token; values and O go as 1/d_model, T words as (S-1)/K
    w_llama = words(LINEARS, L_SEG) * LAYERS / MACS
    w_v4 = {"K 7,168": (math.ceil(7168 / L_SEG) - 1) / 7168, "K 2,048": (math.ceil(2048 / L_SEG) - 1) / 2048, "S = 2": 1 / 7168}
    t_ms = {p: PARTS[p] * 8192 * ROWS[16384]["h_x"] / (RULER * F) * 1e3 for p in PARTS}
    j = {p: t_ms[p] / GPU_BUSY_MS for p in PARTS}
    out["v4"] = {"J_demo_no_gap_gpu_busy": j, "T_words_a_mac": {"llama": w_llama, **w_v4},
                 "T_scale": {k: v / w_llama for k, v in w_v4.items()}, "values_scale": D_MODEL / D_MODEL_V4,
                 "J_v4": {"T": {k: j["T"] * v / w_llama for k, v in w_v4.items()},
                          **{p: j[p] * D_MODEL / D_MODEL_V4 for p in ("O", "e2", "e2_nogather")}},
                 "utilization_gpu_busy": MACS * 8192 / (GPU_BUSY_MS * 1e-3) / (RULER * F)}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print("bytes a token: " + ", ".join(f"{k} {v:,.0f}" for k, v in PARTS.items()) + f"; linear MACs a token {MACS:,}")
    for pname, rows in out["points"].items():
        print(f"\n{pname}: {rows['tokens_a_forward']} tokens a forward, {rows['rows'] // 1024} KiB rows")
        for aname, a in rows["arms"].items():
            cells = "  ".join(f"{en}: {x['ms']:6.2f} ms (G {x['G_bytes'] / 1e9:.2f} GB) J {x['J_serve']:.3f}"
                              + (f" [unfilled {x['J_serve_unfilled']:.3f}]" if x["fill"] < 1 else "")
                              for en, x in a["engines"].items())
            print(f"  {aname:<26} {a['bytes_a_token'] / 1e6:5.3f} MB/token {a['bytes_a_forward'] / 1e9:6.2f} GB/fwd  "
                  f"excl {next(iter(a['engines'].values()))['exclusive_ms']:6.2f} ms  {cells}")
    v = out["v4"]
    print(f"\nworld scale (estimate; demo GPU-busy utilization {v['utilization_gpu_busy']:.3f}): J a part, no gap, demo -> V4")
    for p in ("T", "O", "e2", "e2_nogather"):
        print(f"  {p:<12} demo {v['J_demo_no_gap_gpu_busy'][p]:.3f}  V4 {v['J_v4'][p]}")


if __name__ == "__main__":
    main()
