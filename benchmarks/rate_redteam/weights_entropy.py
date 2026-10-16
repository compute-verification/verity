#!/usr/bin/env python3
"""How far the deployed FP8 payload compresses: the falsifier ra-codec-theory names for its s_c.

Each linear of a Llama checkpoint is quantized to E4M3 as vLLM's online FP8 does it (one scale a tensor, amax / 448,
the fused qkv and gate_up each one tensor). Code lengths are adaptive (Krichevsky–Trofimov per context), so each is
achievable by an arithmetic coder with no side information beyond the context classes, which cost a few bits a row
or column. Contexts: none (order 0), the row's or column's RMS class, both, the left and the upper neighbour's code,
and the row class with the left code. lzma, bz2 and zlib run on a prefix of each tensor.

    python weights_entropy.py --model DIR --procs 16 --out entropy.json
"""

import argparse
import bz2
import json
import lzma
import math
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from safetensors import safe_open

FP8_MAX = 448.0
NCLS = 17
PREFIX = 8 << 20


def tensors(index, layers):
    out = []
    for i in range(layers):
        p = f"model.layers.{i}."
        out += [
            (f"{i}.qkv", [p + f"self_attn.{x}_proj.weight" for x in "qkv"]),
            (f"{i}.o", [p + "self_attn.o_proj.weight"]),
            (f"{i}.gate_up", [p + f"mlp.{x}_proj.weight" for x in ("gate", "up")]),
            (f"{i}.down", [p + "mlp.down_proj.weight"]),
        ]
    if "lm_head.weight" in index:
        out.append(("lm_head", ["lm_head.weight"]))
    return out


def kt_bits(counts):
    """Adaptive KT code length in bits of every context's sequence; counts is (contexts, 256)."""
    c = torch.from_numpy(counts.astype(np.float64))
    n = c.sum(1)
    half = torch.tensor(0.5, dtype=torch.float64)
    a = torch.tensor(128.0, dtype=torch.float64)
    nats = (torch.lgamma(n + a) - torch.lgamma(a)).sum() - (torch.lgamma(c + half) - torch.lgamma(half)).sum()
    return float(nats) / math.log(2)


def classes(rms, ref):
    return np.clip(np.rint(2 * np.log2(np.maximum(rms, 1e-30) / ref)), -(NCLS // 2), NCLS // 2).astype(np.int64) + NCLS // 2


def measure(job):
    model, files, name, keys = job
    parts = []
    for k in keys:
        with safe_open(f"{model}/{files[k]}", framework="pt") as f:
            parts.append(f.get_tensor(k).float())
    w = torch.cat(parts, 0)
    del parts
    rows, cols = w.shape
    amax = float(w.abs().max())
    q = (w / (amax / FP8_MAX)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    rms = float(w.pow(2).mean().sqrt())
    rc = classes(w.pow(2).mean(1).sqrt().numpy(), rms)
    cc = classes(w.pow(2).mean(0).sqrt().numpy(), rms)
    del w
    ctx = {"order0": 1, "row": NCLS, "col": NCLS, "row_col": NCLS * NCLS, "left": 256, "up": 256, "row_left": NCLS * 256}
    counts = {k: np.zeros(n * 256, np.int64) for k, n in ctx.items()}
    step = max(1, (1 << 24) // cols)
    for r0 in range(0, rows, step):
        blk = q[r0:r0 + step].astype(np.int64)
        h = blk.shape[0]
        left = np.concatenate([np.zeros((h, 1), np.int64), blk[:, :-1]], 1)
        up = np.concatenate([q[r0 - 1:r0].astype(np.int64) if r0 else np.zeros((1, cols), np.int64), blk[:-1]], 0)
        rcb = rc[r0:r0 + h, None]
        sel = {
            "order0": 0,
            "row": rcb,
            "col": cc[None, :],
            "row_col": rcb * NCLS + cc[None, :],
            "left": left,
            "up": up,
            "row_left": rcb * 256 + left,
        }
        for k, c in sel.items():
            idx = (np.broadcast_to(c, blk.shape) * 256 + blk).ravel()
            counts[k] += np.bincount(idx, minlength=counts[k].size)
    n = rows * cols
    bits = {k: kt_bits(v.reshape(-1, 256)) / n for k, v in counts.items()}
    side = math.log2(NCLS) * (rows + cols) / n
    for k in ("row", "col", "row_col", "row_left"):
        bits[k] += side
    raw = q.reshape(-1)[:PREFIX].tobytes()
    comp = {
        "lzma9e": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) * 8 / len(raw),
        "bz2_9": len(bz2.compress(raw, 9)) * 8 / len(raw),
        "zlib9": len(zlib.compress(raw, 9)) * 8 / len(raw),
    }
    sub = float(((q & 0x78) == 0).mean())
    return {"name": name, "shape": [rows, cols], "amax_over_rms": amax / rms, "subnormal_or_zero": sub,
            "bits": bits, "prefix_bits": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0, help="0: every layer in config.json")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
    layers = a.layers or json.load(open(f"{a.model}/config.json"))["num_hidden_layers"]
    jobs = [(a.model, files, n, k) for n, k in tensors(files, layers)]
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(measure, jobs))
    tot = sum(r["shape"][0] * r["shape"][1] for r in res)
    lin = [r for r in res if r["name"] != "lm_head"]
    tl = sum(r["shape"][0] * r["shape"][1] for r in lin)

    def avg(rs, t, f):
        return sum(f(r) * r["shape"][0] * r["shape"][1] for r in rs) / t

    summary = {}
    for scope, rs, t in (("linears", lin, tl), ("linears+lm_head", res, tot)):
        s = {k: avg(rs, t, lambda r, k=k: r["bits"][k]) for k in res[0]["bits"]}
        s.update({k: avg(rs, t, lambda r, k=k: r["prefix_bits"][k]) for k in res[0]["prefix_bits"]})
        s["best_per_tensor"] = avg(rs, t, lambda r: min(min(r["bits"].values()), min(r["prefix_bits"].values())))
        s["s_c_best"] = 1 - s["best_per_tensor"] / 8
        s["s_c_order0"] = 1 - s["order0"] / 8
        s["weights"] = t
        summary[scope] = s
    out = {"model": a.model, "layers": layers, "summary": summary, "tensors": res}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
