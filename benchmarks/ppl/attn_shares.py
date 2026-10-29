"""Attention-row statistics of the bf16 reference, for an online-softmax kernel's exponent and its u8 P.

For every query row i >= 1 of every head and layer, over the keys 1..i (the sink, key 0, is apart): y = (s - m) log2 e,
s the scaled logit and m the running row max when the key's block is processed (blocks of `--block` keys aligned to key
0, the max over keys 1.. of every block up to the key's, as `u8pv` reads it); the shares of keys with y < cut for each
cut (in log2 units, the exp2 argument), and of keys whose u8 P rint(255 2^y s_v[j] / max_block s_v) is zero, s_v V's
per-key int8 scale. Prints a JSON summary: per cut the share over all such keys, the mean, quantiles and max of the
per-row share, and the same per layer.

    python benchmarks/ppl/attn_shares.py --model llama-3.2-1b --set wikitext2 --ctx 2048 --windows 16 --data D --out O
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import methodology  # noqa: E402
import run as ppl  # noqa: E402
from model import Config, Weights, make_layer, rotary  # noqa: E402

CUTS = (-9.0, -22.0)
BINS = 1000


class Tally:
    """Per-row shares, binned, and exact key counts, for each statistic (a cut, or "u8_zero")."""

    def __init__(self, names):
        self.names = names
        self.hist = {n: np.zeros(BINS + 1, dtype=np.int64) for n in names}
        self.hits = {n: 0 for n in names}
        self.keys = 0
        self.rows = 0
        self.top = {n: 0.0 for n in names}

    def add(self, counts: dict, keys: torch.Tensor) -> None:
        self.keys += int(keys.sum())
        self.rows += keys.numel()
        for n, c in counts.items():
            share = (c.double() / keys.double()).flatten()
            self.hits[n] += int(c.sum())
            self.hist[n] += np.bincount((share * BINS).round().long().numpy(), minlength=BINS + 1)
            self.top[n] = max(self.top[n], float(share.max()))

    def summary(self) -> dict:
        out = {"rows": self.rows, "keys": self.keys}
        for n in self.names:
            h = self.hist[n]
            cdf = np.cumsum(h) / max(1, h.sum())
            q = {f"p{int(p * 100)}": float(np.searchsorted(cdf, p) / BINS) for p in (0.5, 0.9, 0.99)}
            mean = float((h * np.arange(BINS + 1)).sum() / max(1, h.sum()) / BINS)
            out[n] = {"share_of_keys": self.hits[n] / max(1, self.keys), "row_mean": mean, **q, "row_max": self.top[n]}
        return out


def row_counts(q, k, v, scaling: float, block: int, rows: int):
    """Per (head, row >= 1): the counts of each statistic over keys 1..row, and the key count (row)."""
    H, s, _ = q.shape
    G = k.shape[0]
    rep = H // G
    sv = v.abs().amax(-1) / 127
    nb = -(-s // block)
    top = torch.nn.functional.pad(sv, (0, nb * block - s)).view(G, nb, block)
    top[:, 0, 0] = 0
    top = top.amax(-1)
    blk = torch.arange(s) // block
    counts = {c: torch.zeros(H, s - 1, dtype=torch.int64) for c in (*CUTS, "u8_zero")}
    keys = torch.arange(1, s).expand(H, -1)
    for g in range(G):
        for r0 in range(1, s, rows):
            r1 = min(s, r0 + rows)
            sc = (q[g * rep:(g + 1) * rep, r0:r1] @ k[g].T) * scaling
            live = torch.arange(s)[None, :] <= torch.arange(r0, r1)[:, None]
            live[:, 0] = False
            sc = sc.masked_fill(~live, float("-inf"))
            pad = torch.nn.functional.pad(sc, (0, nb * block - s), value=float("-inf"))
            m = pad.view(rep, r1 - r0, nb, block).amax(-1).cummax(-1).values
            y = (sc - m[..., blk]) * math.log2(math.e)
            for c in CUTS:
                counts[c][g * rep:(g + 1) * rep, r0 - 1:r1 - 1] = ((y < c) & live).sum(-1)
            p8 = torch.round(255 * torch.exp2(y) * (sv[g] / top[g][blk]))
            counts["u8_zero"][g * rep:(g + 1) * rep, r0 - 1:r1 - 1] = ((p8 == 0) & live).sum(-1)
    return counts, keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--set", default="wikitext2")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=16, help="the first N windows")
    ap.add_argument("--block", type=int, default=112)
    ap.add_argument("--rows", type=int, default=512, help="query rows a chunk")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, _, wrec = ppl.build_windows(a.model, a.set, a.ctx, Path(a.data), False)
    ids = ids[:a.windows]
    snap = ppl.snapshot(a.model)
    cfg = Config.read(snap)
    W = Weights(snap, cfg.layers)
    emb = W.read(["model.embed_tokens.weight"])["model.embed_tokens.weight"].to(torch.bfloat16)
    N, T = ids.shape
    cos, sin = rotary(cfg, T, a.device)
    h = emb[torch.from_numpy(ids)].clone()
    del emb
    names = (*CUTS, "u8_zero")
    whole, per_layer = Tally(names), []
    B = methodology.REF_BATCH
    t0 = time.time()
    W.prefetch(0)
    for li in range(cfg.layers):
        layer = make_layer(cfg, li, W.layer(li), a.device)
        if li + 1 < cfg.layers:
            W.prefetch(li + 1)
        tally = Tally(names)
        sdpa = layer.attention

        def attention(q, k, v, run, sdpa=sdpa, tally=tally, layer=layer):
            for b in range(q.shape[0]):
                counts, keys = row_counts(q[b].float(), k[b].float(), v[b].float(), layer.scaling, a.block, a.rows)
                tally.add(counts, keys)
                whole.add(counts, keys)
            return sdpa(q, k, v, run)

        layer.attention = attention
        for j in range(0, N, B):
            h[j:j + B] = layer.forward(h[j:j + B], cos, sin)
        per_layer.append(tally.summary())
        print(json.dumps({"layer": li, "seconds": round(time.time() - t0, 1),
                          **{str(n): per_layer[-1][n]["share_of_keys"] for n in names}}), flush=True)
    W.close()
    res = {"model": a.model, "set": a.set, "ctx": a.ctx, "windows": N, "block": a.block, "cuts_log2": list(CUTS),
           "wrec": wrec, "all": whole.summary(), "layers": [{str(k): v for k, v in t.items()} for t in per_layer],
           "seconds": round(time.time() - t0, 1)}
    res["all"] = {str(k): v for k, v in res["all"].items()}
    (out / "attn_shares.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["all"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
