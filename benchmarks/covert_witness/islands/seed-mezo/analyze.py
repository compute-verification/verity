"""Predict MeZO's gain from a run's diagnostics (CPU only): the group table `train.py --diag` prints, read from a run's
train.log, and the gain a subset would buy at a batch, a number of directions and a FLOP budget.

    python analyze.py RUN_DIR_OR_TRAIN_LOG [--size s|m] [--batch 4 16 64 256] [--budget 8.8e16] [--rounding R]

Per direction at the best lr: G² / (2 (G + V/b + R) T), G = |g_S|², V the per-sequence noise trace, T = tr H_S, R the forward's
rounding noise in ĝ² (measured by a MeZO run as E[ĝ²] − G − V/b). Directions: min(budget / (2 b ctx fwd), 49,9xx at bf16).
Nats a token to bits a byte: × tokens/bytes / ln 2 (GPT-2 on the search set: 2^21 tokens over 9,797,536 bytes).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BPB_PER_NAT = 2**21 / 9_797_536 / math.log(2)
#: forward FLOPs a token at ctx 1024 (`prepare.train_flops`'s forward: weight matmuls, the tied head, causal attention)
FWD = {"s": 96_344_064, "m": 203_108_352}


def load(path: Path) -> tuple[dict, list[dict]]:
    log = path / "out" / "train.log" if path.is_dir() else path
    groups, rows = {}, []
    for line in log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "group" in r:
            groups[r["group"]] = r
        rows.append(r)
    return groups, rows


def pred(groups: dict, keys: list[str], b: int, diag_batch: int, rounding: float = 0.0) -> dict:
    G = sum(groups[k]["G"] for k in keys)
    V = sum((groups[k]["N"] - groups[k]["G"]) * diag_batch for k in keys)
    T = sum(groups[k]["T"] for k in keys)
    e = G + V / b + rounding
    return {"G": G, "V": V, "T": T, "Bs": V / G if G > 0 else None, "sf": G / e if e > 0 else None,
            "lr": G / (e * T) if G > 0 and T > 0 else None,
            "per_dir": G * G / (2 * e * T) if G > 0 and T > 0 else 0.0,
            "numel": sum(groups[k]["numel"] for k in keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--batch", type=int, nargs="*", default=[4, 16, 64, 256])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--size", default="s", choices=sorted(FWD))
    ap.add_argument("--fwd", type=float, default=None, help="forward FLOPs a token at ctx 1024 (default: the size's)")
    ap.add_argument("--budget", type=float, default=0.98 * 9e16)
    ap.add_argument("--diag-batch", type=int, default=16)
    ap.add_argument("--rounding", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=49_900)
    a = ap.parse_args()
    groups, _ = load(a.path)
    fwd = a.fwd or FWD[a.size]
    L = max(int(k[1:].split(".")[0]) for k in groups if k != "embed")
    mats = [k for k in groups if k.endswith((".s1", ".s2", ".s4", ".s5"))]
    gains = [k for k in groups if k.endswith((".s0", ".s3"))]
    subsets = {"blocks": [k for k in groups if k != "embed"], "all": list(groups), "gains": gains,
               "final_norm": [f"L{L}.s0"], **{f"top{n}": [k for k in groups if k != "embed" and int(k[1:].split(".")[0]) >= L - n]
                                              for n in (1, 2)}}
    for k in sorted(mats + gains, key=lambda k: -pred(groups, [k], 16, a.diag_batch)["per_dir"])[:12]:
        subsets[k] = [k]
    print(f"{'subset':12s} {'numel':>10s} {'G':>9s} {'Bs(seq)':>9s} {'T':>9s} " +
          " ".join(f"{'b'+str(b)+' dirs':>9s} {'nats':>9s} {'bpb':>8s}" for b in a.batch))
    for name, keys in subsets.items():
        out = []
        for b in a.batch:
            p = pred(groups, keys, b, a.diag_batch, a.rounding)
            n = min(int(a.budget // (2 * b * a.ctx * fwd)), a.cap)
            total = n * p["per_dir"]
            out.append(f"{n:9d} {total:9.2e} {total * BPB_PER_NAT:8.5f}")
        p = pred(groups, keys, 16, a.diag_batch)
        print(f"{name:12s} {p['numel']:10d} {p['G']:9.2e} {p['Bs'] or float('nan'):9.1f} {p['T']:9.2e} " + " ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
