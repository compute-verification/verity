"""M2-E's steering test: can data the developer authors (public, committed before the seed) plant exact cheap structure?

A student linear W (n x K, Gaussian init from the 'beacon' seed) is trained with AdamW on data the developer wrote:
x ~ N(0, I) and y = c * T x for a teacher T. With T ternary the planted structure is q = a * T + S per output channel;
the route priced is T's ternary pass plus S's alphabet pass and ALU outliers (`prices.py`), against dense int8 at 1.
A Gaussian teacher is the control. Writes `poison.json` to `--out`.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural  # noqa: E402
from prices import CHIPS, ZBITS, dense_part  # noqa: E402

N, K, C = 512, 1024, 0.05


def run(teacher, wd, steps, seed):
    torch.manual_seed(1000 + seed)
    T = torch.randint(-1, 2, (N, K)).float() if teacher == "ternary" else torch.randn(N, K)
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(N, K, generator=g) * 0.02)
    opt = torch.optim.AdamW([W], lr=1e-3, weight_decay=wd, betas=(0.9, 0.95))
    dg = torch.Generator().manual_seed(77)
    for step in range(steps):
        lr = 1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for p in opt.param_groups:
            p["lr"] = lr
        x = torch.randn(256, K, generator=dg)
        loss = ((x @ W.T - C * x @ T.T) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = W.detach().numpy()
    q = natural.quantize(w).astype(np.int32)
    rec = {"teacher": teacher, "wd": wd, "steps": steps, "final_loss": loss.item(), "rel_err": float(
        np.linalg.norm(w - C * T.numpy()) / np.linalg.norm(C * T.numpy()))}
    if teacher == "ternary":
        t = T.numpy().astype(np.int32)
        a = np.array([np.bincount(np.abs(q[i][t[i] != 0])).argmax() for i in range(N)])[:, None]
        s = q - a * t
        ab = np.abs(s)
        rec["s_hist"] = {str(v): float((ab == v).mean()) for v in range(6)}
        rec["s_max"] = int(ab.max())
        routes = {}
        for chip in CHIPS:
            for b in ZBITS:
                best = min(dense_part(chip, aa, b) + float((ab > aa).mean()) * CHIPS[chip]["alu_mac"]
                           for aa in (0, 1, 2, 3, 4, 7, 15))
                routes[f"{chip}_b{b}"] = min(1.0, dense_part(chip, 1, b) + best)
        rec["teacher_route"] = routes
    an = natural.analyse(q.astype(np.int16), np.random.default_rng(seed), False)
    rec["natural_best"] = an["best_cost"]
    rec["sigma_q"] = an["alphabet"]["sigma_q"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    torch.set_num_threads(4)
    res = []
    for teacher in ("ternary", "gauss"):
        for wd in (0.0, 0.1, 1.0):
            r = run(teacher, wd, args.steps, 1)
            print(json.dumps({k: v for k, v in r.items() if k != "natural_best"}), flush=True)
            res.append(r)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "poison.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
