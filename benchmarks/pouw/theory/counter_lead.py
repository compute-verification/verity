"""COUNTER (LEAN program): prover strategies against the PoUW lead's statements, searched on small instances.

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min`: ssn-1l with the wide row tag, three odd 7-bit coin lanes a
64-row tile, step and column, weighted (127, 1, 1) on even rows and (1, 127, 1) on odd (`blc4a-t64-e7-rot`). Its words
come from `nred_round4.lead_words`, and every column block is compared with the explicit lane product.

A strategy's figure is its W1 saving as a share of stock (k units a column entry). A statement "priced W1 work ≥ honest
work − gap" breaks when a strategy's saving exceeds the gap at the same instance.

- **The copier** (`copier/<tm>x<tn>/<reading>`): at each step it skips a whole tm × tn MMA tile whose every word is held,
  and copies it at 0.40 a word, saving tm·tn·(H32 − 0.40). The source selection is unpriced, so the figure is an upper
  bound for the copier.
  - `free`: a word is held when its value occurs in its column at an earlier step, or in an earlier tm-row block of
    its step.
  - `block`: the tile equals, as a block, one earlier tile of the same columns (an earlier step, or an earlier
    row block of its step). It is the stricter reading, inside `free`.
  - Widths: 16 × 8 is `mma.sync` m16n8k32. 16 × 256 is a warp's share of the lead's `IGMMA.64x256x32` at BN 256, and
    64 × 256 is the warpgroup's (N-FLOOR final 2, note 20261029T1225Z-report-np-floor).
- Chosen data: A2c (identical zero-data rows) and A2r (the alternating cancel), as N-RED's earlier rounds.
- **The grains** (`grains`): `nred_round4.shares` at the gap's row counts, MAC-weighted over Llama-3.2-1B's linears:
  literal P and P′ at word grain (what a tile-grain premise excludes) beside tile grain (`W/tile`, `W/whole`).
- **The tag collapse** (`--g-zero`): G's chosen tag lanes set to 0 in every column. A 64-row tile's wide tags differ
  in lane 0 and a carry into lane 1, so on this event (1/255 a column a lane) its rows' T·G take one or two values.

The draws are made the lead's way: G and the coin lanes per column, drawn a column block at a time; the row tags, a₁ and b
per row, drawn once, so a tile's columns share their rows as in one instance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns
import nred_attacks as na
import nred_round4 as r4

H32 = ns.H32
COPY_WORD = r4.COPY_TILE_WORD
LEAD = "blc4a-t64-e7-rot"
WIDTHS = ((16, 8), (64, 8), (16, 64), (16, 256), (64, 256))
COL_BLOCK = 8


def _held(w: np.ndarray, tms: tuple[int, ...]) -> dict[int, np.ndarray]:
    """w (m, s, nb): for each row-block height tm, whether a word's value occurs in its column at an earlier step or an
    earlier tm-row block of its step."""
    m, s, nb = w.shape
    keys = (w + np.arange(nb, dtype=np.int64)[None, None] * r4.COL_KEY).transpose(2, 1, 0).ravel()
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    p = first[inv].reshape(nb, s, m).transpose(2, 1, 0)
    tp, ip = (p % (s * m)) // m, p % m
    now = np.arange(s)[None, :, None]
    rows = np.arange(m)[:, None, None]
    return {tm: (tp < now) | ((tp == now) & (ip // tm < rows // tm)) for tm in tms}


def _block_repeats(w: np.ndarray, tm: int) -> int:
    """Tiles tm × nb (the block's columns) that equal, as a block, an earlier tile of the same columns."""
    m, s, nb = w.shape
    tiles = w.reshape(m // tm, tm, s, nb).transpose(2, 0, 1, 3).reshape(s * (m // tm), tm * nb)
    _, first, inv = np.unique(tiles, axis=0, return_index=True, return_inverse=True)
    return int((first[inv.ravel()] < np.arange(len(tiles))).sum())


def copier(m: int, n: int, k: int, seed: int, pattern: str, tag: str = "wide", cfg: str = LEAD,
           widths: tuple = WIDTHS, g_zero: tuple[int, ...] = ()) -> dict:
    """The copier's whole-tile repeats on one instance (m rows, n columns, depth k), with G's `g_zero` lanes 0."""
    tile, e, lanes = r4.LAYOUTS[cfg]
    s, ntiles = ns.steps1l(k), -(-m // tile)
    rng = np.random.default_rng(seed)
    row = r4.lead_draws(cfg, k, m, 0, rng, tag)
    code = na.code(n)
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
    widths = tuple((tm, tn) for tm, tn in widths if m % tm == 0 and n % tn == 0)
    tms = tuple(sorted({tm for tm, _ in widths}))
    counts = {(tm, tn): np.zeros((m // tm, s, n // tn), dtype=np.int64) for tm, tn in widths}
    block_rep = {tm: 0 for tm in tms}
    held_words = {tm: 0 for tm in tms}
    equal = True
    for j0 in range(0, n, COL_BLOCK):
        nb = min(COL_BLOCK, n - j0)
        g = (rng.integers(-127, 128, (na.TAG_LANES, nb)) if tag == "wide"
             else rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, nb)))
        g[list(g_zero)] = 0
        coin = rng.choice(odd, (lanes, ntiles, s - 1, nb))
        dr = {"tags": row["tags"], "g": g, "a1": row["a1"], "b": row["b"], "coin": coin}
        lw = r4.lead_words(cfg, k, pattern, dr, code[:, j0:j0 + nb])
        equal &= lw["equal_to_honest"]
        w = lw["w"]
        held = _held(w, tms)
        for tm in tms:
            held_words[tm] += int(held[tm].sum())
            if nb == COL_BLOCK:
                block_rep[tm] += _block_repeats(w, tm)
        for (tm, tn), cnt in counts.items():
            h = held[tm].reshape(m // tm, tm, s, nb).sum(axis=1)
            cnt[:, :, j0 // tn] += h.sum(axis=2) if tn >= nb else 0
    out = {"m": m, "n": n, "k": k, "steps": s, "seed": seed, "pattern": pattern, "tag": tag, "cfg": cfg,
           "g_zero": list(g_zero), "equal_to_honest": bool(equal), "held_words": {str(tm): held_words[tm] / (m * s * n) for tm in tms},
           "copier": {}}
    stock = m * n * k
    for (tm, tn), cnt in counts.items():
        whole = int((cnt == tm * tn).sum())
        out["copier"][f"{tm}x{tn}/free"] = {"whole": whole, "tiles": int(cnt.size),
                                            "share": whole * tm * tn * (H32 - COPY_WORD) / stock}
    for tm in tms:
        rep = block_rep[tm]
        out["copier"][f"{tm}x{COL_BLOCK}/block"] = {"whole": rep, "tiles": (m // tm) * s * (n // COL_BLOCK),
                                                    "share": rep * tm * COL_BLOCK * (H32 - COPY_WORD) / stock}
    return out


def llama_copier(m: int, n: int, seed: int, pattern: str, tag: str = "wide", cfg: str = LEAD,
                 g_zero: tuple[int, ...] = ()) -> dict:
    """The copier at Llama-3.2-1B's depths, MAC-weighted over its linears (one instance a depth)."""
    shape = ns.LLAMA
    per_k = {kk: copier(m, n, kk, seed + kk, pattern, tag, cfg, g_zero=g_zero)
             for kk in sorted({kk for _, kk in shape.values()})}
    stock = sum(nn * kk for nn, kk in shape.values())
    keys = per_k[min(per_k)]["copier"].keys()
    return {"per_k": per_k,
            "equal_to_honest": all(v["equal_to_honest"] for v in per_k.values()),
            "share": {key: sum(nn * kk * per_k[kk]["copier"][key]["share"] for nn, kk in shape.values()) / stock
                      for key in keys}}


GRAINS = ("P/lag1", "P/entry", "P/xrow", "P/tilecol", "P/column", "P'32/lag1", "P'32/entry", "P'32/xrow",
          "P'32/tilecol", "P'32/column", "P/masked", "M/masked", "W/tile", "W/whole")


def grains(m: int, n: int, seed: int, pattern: str, tag: str = "wide", cfg: str = LEAD) -> dict:
    """`nred_round4.shares` on m rows and n columns at Llama-3.2-1B's depths, MAC-weighted over its linears."""
    rng = np.random.default_rng(seed)
    insts = {}
    for kk in sorted({kk for _, kk in ns.LLAMA.values()}):
        inst = r4.lead_instance(cfg, kk, m, n, rng, pattern, tag)
        inst["share_keys"] = list(r4.shares(inst, kk, n).keys())
        insts[kk] = inst
    return {"m": m, "n": n, "seed": seed, "pattern": pattern, "tag": tag, "cfg": cfg,
            "equal_to_honest": all(v["equal_to_honest"] for v in insts.values()),
            "share": {g: v for g, v in r4.weighted(insts, ns.LLAMA).items() if g in GRAINS},
            "per_k": {f"k{kk}": {"saved": v["saved"], "tile_grain": v["tile_grain"], "steps": v["steps"]}
                      for kk, v in insts.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("copier", "grains"), default="copier")
    ap.add_argument("--rows", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--patterns", nargs="+", default=["a2c", "a2r"])
    ap.add_argument("--tags", nargs="+", default=["wide"])
    ap.add_argument("--cfgs", nargs="+", default=[LEAD])
    ap.add_argument("--g-zero", type=int, nargs="*", default=[], help="tag lanes of G set to 0 in every column")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep = {}
    for cfg in args.cfgs:
        for tag in args.tags:
            for m in args.rows:
                for pattern in args.patterns:
                    for seed in args.seeds:
                        key = f"{cfg}/{tag}/m{m}/{pattern}/s{seed}"
                        if args.mode == "grains":
                            r = grains(m, args.cols, seed, pattern, tag, cfg)
                        else:
                            r = llama_copier(m, args.cols, seed, pattern, tag, cfg, tuple(args.g_zero))
                        rep[key] = r
                        print(f"{key:44s} eq={r['equal_to_honest']}  " + "  ".join(
                            f"{c}={100 * v:.4f}%" for c, v in r["share"].items()), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
