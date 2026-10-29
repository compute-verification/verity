#!/usr/bin/env python3
"""D-OUT's falsification test on real weights: the cheapest exact R·Bᵀ at W1's prices on Llama-3.2-1B's linears.

Outputs-only with dense noise, (A + R)·Bᵀ, is credited only if R·Bᵀ costs the adversary about one unit a MAC (SAK-UF on
the weights). This reads a Llama-3.2-1B checkpoint (one bf16 safetensors file), fuses q/k/v and gate/up as
`hash_alt_inputs.json` does, quantizes each output channel to int8 (symmetric, absmax / 127), and gives, per linear and
noise width (1 and 2 bits), `best_tile_credit`: the least of b1 planes with sparse corrections, b1 planes plus a
2:4-sparse high part, u4 nibbles and s8 per 8-column, 256-deep tile, over K in its own order, reordered by whole
32-index blocks (free), reordered by single indices once for every G tiles (priced at 1/(8G) a MAC), or spread so heavy
indices fall in different groups of 4. The gap is 1 - credit, weighted by MACs over all layers; under 1% the premise
survives this test.

  uv run python benchmarks/pouw/theory/hash_alt_weights.py MODEL.safetensors [--json PATH] [--layers N] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_alt import LL, best_tile_credit  # noqa: E402

FUSED = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
         "gate_up": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}


def tensors(path: Path) -> tuple[dict, np.memmap, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, np.memmap(path, dtype=np.uint8, mode="r"), 8 + n


def bf16(header: dict, raw: np.memmap, base: int, name: str) -> np.ndarray:
    t = header[name]
    assert t["dtype"] == "BF16", (name, t["dtype"])
    a, b = t["data_offsets"]
    u16 = np.frombuffer(raw[base + a:base + b], dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(t["shape"])


def int8_per_channel(w: np.ndarray) -> np.ndarray:
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    return np.clip(np.rint(w / np.where(s == 0, 1, s)), -127, 127).astype(np.int8)


def credits(task: tuple[Path, int, str]) -> list[dict]:
    path, layer, name = task
    header, raw, base = tensors(path)
    w = np.concatenate([bf16(header, raw, base, f"model.layers.{layer}.{p}.weight") for p in FUSED[name]])
    want = {nm: (n, K) for nm, (K, n) in zip(FUSED, LL["linears"])}
    assert w.shape == want[name], (layer, name, w.shape, want[name])
    B = int8_per_channel(w)
    return [{"layer": layer, "linear": name, "r_bits": r, "macs": B.size, **best_tile_credit(B, r)} for r in (1, 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--layers", type=int, default=LL["layers"])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    tasks = [(args.safetensors, layer, name) for layer in range(args.layers) for name in FUSED]
    rows, w_sum, gap, free_gap = [], {}, {}, {}
    with multiprocessing.Pool(args.jobs) as pool:
        for pair in pool.imap(credits, tasks):
            for c in pair:
                r, macs = c["r_bits"], c["macs"]
                rows.append(c)
                w_sum[r] = w_sum.get(r, 0) + macs
                gap[r] = gap.get(r, 0.0) + (1 - c["credit"]) * macs
                free_gap[r] = free_gap.get(r, 0.0) + max(0.0, 1 - c["free_reorder_credit"]) * macs
            print(f"layer {pair[0]['layer']:2d} {pair[0]['linear']:8s} " + " ".join(
                f"r{x['r_bits']} {x['credit']:.4f} ({x['route']}, free {x['free_reorder_credit']:.4f})"
                for x in pair), flush=True)
    out = {"layers": args.layers, "rows": rows,
           "gap_mac_weighted": {f"r{r}": gap[r] / w_sum[r] for r in gap},
           "free_reorder_gap_mac_weighted": {f"r{r}": free_gap[r] / w_sum[r] for r in free_gap}}
    for name in FUSED:
        for r in (1, 2):
            cs = [x["credit"] for x in rows if x["linear"] == name and x["r_bits"] == r]
            out.setdefault("per_linear", {})[f"{name} r{r}"] = {"min": min(cs), "mean": float(np.mean(cs))}
    for r in (1, 2):
        g = out["gap_mac_weighted"][f"r{r}"]
        print(f"r{r}: MAC-weighted gap {g:.3%} (free reordering {out['free_reorder_gap_mac_weighted'][f'r{r}']:.3%})"
              f" -> {'survives' if g < 0.01 else 'falsified'} at 1%")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
