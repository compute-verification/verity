#!/usr/bin/env python3
"""An adversarial run against the residual-state rule's one-layer bound (the store's docs/pouw/new-designs.md §3.7, the red
team's 6:17 AM PDT condition 3).

A faulty glue unit at layer l writes the residual h_{l+1}(p) = x_p at positions P, chosen after the key: a written row w_p
of r1_census's families in the residual stream's rotated basis, x_p = c rho w_p Q^T / |w_p|, rho the median residual norm
at layer l and c the norm the adversary picks.  RMSNorm divides c out, so layer l+1's qkv rows at P carry w_p's codes for
every c, and a large c leaves the later layers' updates small beside x_p: the regime in which the carry repeats its slices.
c is the only free parameter here; a search over x's direction is the stronger attack and is not run.

The model's own layers l+1 .. l+L run on the overwritten state (pearlc_capture's streamed forward, bf16 on the CPU; every
(family, c) is one batch element of one forward).  Each later layer's qkv and gate/up rows are scored per atom slice under
the rule, with r1_xlayer's codes (one keyed Q for every layer, RMSNorm's weight folded into the weights):
  credited  the share of P's nonzero slices more than d codes (up to sign) from every slice credited before them, in the
            order layer l+1 qkv, l+1 gate/up, l+2 qkv, ...: what the rule still pays for at P
  cheap     over the credited slices' steps against a keyed sample of the layer's rotated weight rows, from R1-H's
            separable hot start, the share r1_sdet calls cheap (S-determined, and spotted from the operands' 2-adic
            valuations): what a prover could take below the atom's price
  update    |h_j(p) - x_p| / |x_p|, and the codes a slice moves from layer l+1's
and the same for natural rows at other positions of the same batch element, the honest reference.  The bound fails if a
layer past l+1 credits written slices that are cheap well beyond the natural rows' share.

  python r1_carry.py --weights SNAPSHOT --ids IDS.npy --layers 4,16,27 --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import r1_census as rc
from r1_sdet import top_bit, v2
from r1_slices import unsigned
from verity.ml.kernels import group_sum_total_batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pearlc_capture import Stream  # noqa: E402

CLASSES = {"qkv": ("input_layernorm", ("q", "k", "v"), "self_attn"), "gateup": ("post_attention_layernorm", ("gate", "up"), "mlp")}


def weights(st: rc.Safetensors, layer: int, cls: str, q: np.ndarray, cols: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    """The layer's RMSNorm weight and a keyed sample of its rotated weight rows (B (b gamma) Q), float64."""
    norm, names, block = CLASSES[cls]
    p = f"model.layers.{layer}."
    gamma = st.get(p + f"{norm}.weight").astype(np.float64)
    w = np.concatenate([st.get(p + f"{block}.{n}_proj.weight") for n in names])
    pick = np.sort(rc.rng(key, "cols", layer, cls).choice(len(w), cols, replace=False))
    return gamma, (w[pick].astype(np.float64) * gamma) @ q


def cheap_steps(ac: np.ndarray, bc: np.ndarray, h: int, g: np.random.Generator) -> np.ndarray:
    """(atoms, rows, cols): r1_sdet's `cheap` flag of every step of R1-H's chain from the separable hot start."""
    av, bv = rc.values(ac), rc.values(bc)
    start = rc.separable_start(av, bv, h, g)
    rows, cols = len(ac), len(bc)
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"))
    ai, bi = (av * 512).astype(np.int64), (bv * 512).astype(np.int64)
    acc, prev = start.copy(), rc.as_units(start)
    out = np.zeros((ac.shape[1] // 32, rows, cols), bool)
    for t in range(ac.shape[1] // 32):
        s = slice(32 * t, 32 * t + 32)
        pa, pb = ai[:, s], bi[:, s]
        prod = pa[:, None, :] * pb[None, :, :]
        e = np.maximum(top_bit(prev), top_bit(prod).max(axis=2).ravel())
        gg = e - 25
        lo = v2(pa).min(axis=1)[ii] + v2(pb).min(axis=1)[jj]
        out[t] = (((lo >= gg) & ((prev == 0) | (v2(prev) >= gg))) | (gg <= 0)).reshape(rows, cols)
        acc = group_sum_total_batch(rc.ATOM, acc, ac[ii, s], bc[jj, s])[0]
        prev = rc.as_units(acc)
    return out


def dist(a: np.ndarray, pool: list[np.ndarray]) -> np.ndarray:
    """Per slice of a (n, 32), unsigned: the fewest differing codes to any slice of the pool (33 for an empty pool)."""
    best = np.full(len(a), 33, np.int64)
    for p in pool:
        for c0 in range(0, len(p), 4096):
            best = np.minimum(best, (a[:, None, :] != p[None, c0:c0 + 4096, :]).sum(axis=2).min(axis=1))
    return best


def score(variant: dict, layer: int, cls: str, codes: np.ndarray, rows_w: np.ndarray, rows_n: np.ndarray, pools: list,
          bc: np.ndarray, args, key: str) -> dict:
    """One variant's (layer, class): credited share, cheap share among credited steps, for the written rows and the
    natural ones.  pools[t] is the list of credited unsigned slices of atom t so far (both kinds, all earlier sets)."""
    atoms = codes.shape[1] // 32
    credited = np.zeros((len(codes), atoms), bool)
    nonzero = np.zeros((len(codes), atoms), bool)
    for t in range(atoms):
        u = unsigned(codes[:, 32 * t:32 * t + 32])
        nz = (u & 0x7F).any(axis=1)
        d = dist(u, pools[t])
        cred = nz & (d > args.d)
        # rows credited in this set are pooled for the next sets, not against each other (per-linear dedup is r1_xlayer's)
        pools[t].append(u[cred])
        credited[:, t], nonzero[:, t] = cred, nz
    cheap = cheap_steps(codes, bc, args.h, rc.rng(key, "start", variant["family"], variant["c"], layer, cls))
    out = {}
    for name, rows in (("written", rows_w), ("natural", rows_n)):
        cr = credited[rows]
        steps = cheap[:, rows, :].transpose(1, 0, 2)          # rows, atoms, cols
        n_cred = int(cr.sum())
        per_row = [float(steps[i][cr[i]].mean()) for i in range(len(rows)) if cr[i].any()]
        out[name] = {"credited_share": float(cr.sum() / max(nonzero[rows].sum(), 1)),
                     "cheap_share_credited": float(steps[cr].mean()) if n_cred else None,
                     "cheap_share_all": float(steps.mean()),
                     "row_cheap_all": [float(x) for x in steps.mean(axis=(1, 2))],
                     "row_cheap_credited": per_row,
                     "max_row_cheap_credited": max(per_row, default=0.0),
                     "min_row_cheap_credited": min(per_row, default=None)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help="the model's snapshot directory (safetensors and config)")
    ap.add_argument("--ids", type=Path, required=True, help="token ids (.npy), e.g. a pearlc_capture run's ids.npy")
    ap.add_argument("--layers", default="4,16,27", help="the faulty unit's layers l")
    ap.add_argument("--after", type=int, default=3, help="later layers run and scored: l+1 .. l+after")
    ap.add_argument("--families", default="pow2,flat-z50")
    ap.add_argument("--scales", default="1,8,64,512", help="|x_p| over the median residual norm at layer l")
    ap.add_argument("--written", type=int, default=12, help="positions written")
    ap.add_argument("--natural", type=int, default=12, help="natural positions scored as the reference")
    ap.add_argument("--cols", type=int, default=128, help="weight rows sampled for the cheap test")
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--key", default="r1-carry/v0")
    ap.add_argument("--xkey", default="r1-census/v0", help="r1_xlayer's key, for the residual rotation Q")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cfg = transformers.AutoConfig.from_pretrained(args.weights)
    cfg._attn_implementation = "sdpa"
    S, st = Stream(args.weights, cfg), rc.Safetensors(args.weights)
    k = cfg.hidden_size
    q = rc.haar(k, rc.rng(args.xkey, "residual-rotation"))
    ids = torch.from_numpy(np.load(args.ids).astype(np.int64))[None]
    tokens = ids.shape[1]
    variants = [{"family": f, "c": float(c)} for f in args.families.split(",") for c in args.scales.split(",")]
    nb = len(variants)
    g = rc.rng(args.key, "positions")
    cand = np.arange(16, tokens)
    pos_w = np.sort(g.choice(cand, args.written, replace=False))
    pos_n = np.sort(g.choice(np.setdiff1d(cand, pos_w), args.natural, replace=False))
    rows = np.concatenate([pos_w, pos_n])
    rows_w, rows_n = np.arange(len(pos_w)), np.arange(len(pos_w), len(rows))
    h1, pos1, rope1, mask1 = S.start(ids)
    _, posb, ropeb, maskb = S.start(ids.expand(nb, -1))
    results, done = [], -1
    for l in sorted(map(int, args.layers.split(","))):
        for i in range(done + 1, l + 1):
            h1, _ = S.layer(i, h1, pos1, rope1, mask1, False)
        done = l
        rho = float(torch.linalg.vector_norm(h1[0].float(), dim=-1).median())
        hb = h1.expand(nb, -1, -1).clone()
        for vi, v in enumerate(variants):
            w = rc.written(v["family"], len(pos_w), k, rc.rng(args.key, "written", l, v["family"]))
            x = v["c"] * rho * (w @ q.T) / np.linalg.norm(w, axis=1, keepdims=True)
            hb[vi, torch.from_numpy(pos_w)] = torch.from_numpy(x).to(hb.dtype)
        x0 = hb[:, torch.from_numpy(pos_w)].float()
        pools = {vi: [[] for _ in range(k // 32)] for vi in range(nb)}
        first = {}
        for j in range(l + 1, min(l + args.after, cfg.num_hidden_layers - 1) + 1):
            hb, cap = S.layer(j, hb, posb, ropeb, maskb, True)
            for cls in ("qkv", "gateup"):
                gamma, bmat = weights(st, j, cls, q, args.cols, args.key)
                bc = rc.quantize(bmat)
                xin = cap[cls].float().numpy()
                for vi, v in enumerate(variants):
                    codes = rc.quantize((xin[vi, rows].astype(np.float64) / gamma) @ q)
                    r = score(v, j, cls, codes, rows_w, rows_n, pools[vi], bc, args, args.key)
                    moved = None
                    if (vi, cls) in first:
                        near = [(unsigned(codes[rows_w, s:s + 32]) != unsigned(first[(vi, cls)][:, s:s + 32])).sum(axis=1) <= args.d
                                for s in range(0, k, 32)]
                        moved = float(np.mean(near))
                    else:
                        first[(vi, cls)] = codes[rows_w]
                    # |h_{j+1}(p) - x_p| / |x_p|: the updates of layers l+1 .. j, which layer j+1's rows see
                    upd = float((torch.linalg.vector_norm(hb[vi, torch.from_numpy(pos_w)].float() - x0[vi], dim=-1)
                                 / torch.linalg.vector_norm(x0[vi], dim=-1)).median()) if cls == "qkv" else None
                    results.append({"l": l, "layer": j, "after": j - l, "class": cls, **v, "rho": rho,
                                    "update_through_j_rel_median": upd, "share_within_d_of_l1": moved, **r})
                    wr, nt = r["written"], r["natural"]
                    print(f"l {l:2d} j {j:2d} {cls:6s} {v['family']:8s} c {v['c']:6g}  written credited {wr['credited_share']:.3f} "
                          f"cheap|cred {wr['cheap_share_credited'] if wr['cheap_share_credited'] is None else round(wr['cheap_share_credited'], 3)} "
                          f"| natural credited {nt['credited_share']:.3f} cheap {nt['cheap_share_all']:.3f}"
                          f"{'' if upd is None else f'  update {upd:.3g}'}{'' if moved is None else f'  within-d-of-l+1 {moved:.3f}'}",
                          flush=True)
        print(json.dumps({"l": l, "seconds": round(time.time() - t0, 1)}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"tool": "r1_carry", "weights": str(args.weights), "ids": str(args.ids), "tokens": tokens,
                                    "positions_written": pos_w.tolist(), "positions_natural": pos_n.tolist(),
                                    "args": {k_: str(v_) for k_, v_ in vars(args).items()}, "results": results,
                                    "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
