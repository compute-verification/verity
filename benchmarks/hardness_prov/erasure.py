"""HT-PROV's erasure test: how many honest steps of the live-coin hardening pass erase a cheap structure planted in the
committed start?

The toy (`toy.py`, HT-PUBLIC's arm A_lo at width `--d`) is pretrained once. For every start (`starts.py`), pass setting
(learning rate, weight noise σ_w, quantizer) and beacon, the pass (`toy.Hardening`) runs from the start with its
optimizer reset, and at each τ in `--taus` the planted matrices' served codes q_τ are priced (`routes.search`, each
matrix's cheapest route, then the mean; the planted matrices have equal MACs). Also recorded at each τ: the codes still
equal to the start's (s0), their drift |q_τ − q_0| in LSBs, and the float and served validation losses.

The coin spread: from each start, the planted matrices' float weights after τ steps under two beacons differ only by the
coins, so rms(W_β1 − W_β2)/√2 in served LSBs is the per-entry spread a planner who doesn't know the coins faces.

The scale game: a start named `{start}x{c}` is `start` with every fc multiplied by c and ln2's gain and bias divided by
c, the same function, whose fc rows have c times the LSB. An absolute step (lr per entry) is then c times fewer LSBs; a
relative one (`--etas`, η LSBs per entry) is not.

Writes `erasure.json` to `--out`, and the pretrained checkpoint (`pre_d{d}.pt`) to `--out` unless `--pre` gave one.
"""

import argparse
import itertools
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routes
import starts
import toy

DATA = os.path.expanduser("~/hardness-ml/tinystories-bpe4096")


def build(args, pre):
    model = toy.GPT(args.d, args.layers, args.heads)
    model.load_state_dict(torch.load(pre, weights_only=False))
    return model


def planted_names(model):
    return toy.names(model, starts.PLANTED)


def record(model, pool, cfg, t, q0, tplant, loss):
    rec = {"tau": t, "loss": loss, "val": pool.val_loss(model), "val_served": toy.served_val(model, pool, cfg.quant)}
    per = {}
    for n in planted_names(model):
        q, _ = toy.codes(model.linear(n).detach().numpy(), cfg.quant)
        d = np.abs(q - q0[n])
        per[n] = {"routes": routes.search(q, tplant.get(n)), "s0": float((d == 0).mean()),
                  "drift_rms": float(np.sqrt((d.astype(np.float64) ** 2).mean())),
                  "drift_p99": float(np.quantile(d, 0.99)), "drift_max": int(d.max()), "stats": routes.stats(q)}
    v = list(per.values())
    rec["s0"] = float(np.mean([p["s0"] for p in v]))
    rec["drift_rms"] = float(np.mean([p["drift_rms"] for p in v]))
    rec["drift_p99"] = float(np.mean([p["drift_p99"] for p in v]))
    rec["sigma_q"] = float(np.mean([p["stats"]["sigma_q"] for p in v]))
    rec["cost"] = {}
    for chip in routes.CHIPS:
        fams = v[0]["routes"][chip]["families"].keys()
        rec["cost"][chip] = {"best": float(np.mean([p["routes"][chip]["cost"] for p in v])),
                             **{f: float(np.mean([p["routes"][chip]["families"][f]["cost"] for p in v])) for f in fams},
                             "lowrank_r": [p["routes"][chip]["families"]["lowrank"]["r"] for p in v],
                             "lowrank_a": [p["routes"][chip]["families"]["lowrank"]["a"] for p in v],
                             "per_matrix": [p["routes"][chip]["cost"] for p in v]}
    return rec


def job(args, pre, start, cfg, seed, scratch):
    torch.set_num_threads(args.threads)
    pool = toy.Pool(args.data)
    model = build(args, pre)
    m = re.fullmatch(r"(.+)x(\d+)", start)
    base, c = (m.group(1), float(m.group(2))) if m else (start, 1.0)
    q0, tplant = {}, {}
    with torch.no_grad():
        for n in planted_names(model):
            w = model.linear(n)
            w0, t = starts.plant(w.detach().numpy(), base, cfg.quant)
            w.copy_(torch.from_numpy(w0.astype(np.float32)))
            if t is not None:
                tplant[n] = t
        if c != 1.0:
            for blk in model.blocks:
                blk.fc.weight.mul_(c)
                blk.ln2.weight.div_(c)
                blk.ln2.bias.div_(c)
        for n in planted_names(model):
            q0[n], _ = toy.codes(model.linear(n).detach().numpy(), cfg.quant)
    hard = toy.Hardening(model, pool, cfg, toy.beacon(f"hardness-prov live {seed}"))
    taus = sorted(args.taus)
    recs, loss, keep = [], None, {}
    t0 = time.time()
    for t in range(taus[-1] + 1):
        if t in taus:
            recs.append(record(model, pool, cfg, t, q0, tplant, loss))
            keep[t] = {n: model.linear(n).detach().numpy().copy() for n in planted_names(model)}
            c = recs[-1]["cost"]
            print(f"{start} {cfg.tag()} seed {seed} tau {t} h200 {c['h200']['best']:.3f} sm120 {c['sm120']['best']:.3f} "
                  f"s0 {recs[-1]['s0']:.3f} drift {recs[-1]['drift_rms']:.2f} val {recs[-1]['val']:.4f} "
                  f"served {recs[-1]['val_served']:.4f} {time.time() - t0:.0f}s", flush=True)
        if t < taus[-1]:
            loss = hard.step(t)
    np.savez(Path(scratch) / f"w_{start}_{cfg.tag()}_s{seed}.npz", **{f"{t}/{n}": w for t, d in keep.items() for n, w in d.items()})
    return {"start": start, "cfg": cfg.__dict__, "tag": cfg.tag(), "seed": seed, "records": recs}


def spread(args, scratch, res):
    """Per start and setting: rms over planted entries of (W_β1 − W_β2)/√2 in the served LSB, at each τ."""
    out = {}
    groups = {}
    for r in res:
        groups.setdefault((r["start"], r["tag"], r["cfg"]["quant"]), []).append(r["seed"])
    for (start, tag, quant), seeds in groups.items():
        if len(seeds) < 2:
            continue
        a, b = (np.load(Path(scratch) / f"w_{start}_{tag}_s{s}.npz") for s in sorted(seeds)[:2])
        per_tau = {}
        for key in a.files:
            t, n = key.split("/", 1)
            s = toy.scale(a[key], quant)[:, None]
            per_tau.setdefault(t, []).append(float(np.sqrt((((a[key] - b[key]) / s) ** 2).mean() / 2)))
        out[f"{start}/{tag}"] = {t: float(np.mean(v)) for t, v in sorted(per_tau.items(), key=lambda kv: int(kv[0]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=0, help="0: d / 48")
    ap.add_argument("--pretrain-steps", type=int, default=1500)
    ap.add_argument("--pre", help="a pretrained state dict; without it the model is pretrained here")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", default="honest,tern,ternpin8,s24,small,lr4,lr8res15,lr16res15")
    ap.add_argument("--lrs", default="1e-5,1e-4")
    ap.add_argument("--sigma-w", default="0,0.5")
    ap.add_argument("--sigma-a", type=float, default=0.05)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--quants", default="absmax,mad")
    ap.add_argument("--etas", default="", help="LSB-relative passes: η served LSBs a step on the block linears")
    ap.add_argument("--lr-other", type=float, default=1e-5, help="the relative passes' lr on everything else")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--taus", default="0,1,3,10,30,100,300")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    args.heads = args.heads or max(1, args.d // 48)
    args.taus = [int(x) for x in args.taus.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = args.pre
    if not pre:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        pool = toy.Pool(args.data)
        model = toy.pretrain(pool, args.d, args.layers, args.heads, args.pretrain_steps, log=lambda s: print(s, flush=True))
        pre = str(out / f"pre_d{args.d}.pt")
        torch.save(model.state_dict(), pre)
        print(f"pretrained d={args.d}: val {pool.val_loss(model):.4f}", flush=True)
    grid = list(itertools.product(args.sigma_w.split(","), args.quants.split(",")))
    cfgs = [toy.Config(lr=float(lr), sigma_w=float(sw), sigma_a=args.sigma_a, p_drop=args.p_drop, quant=qu)
            for lr in filter(None, args.lrs.split(",")) for sw, qu in grid]
    cfgs += [toy.Config(lr=args.lr_other, sigma_w=float(sw), sigma_a=args.sigma_a, p_drop=args.p_drop, quant=qu,
                        rel=True, eta=float(eta)) for eta in filter(None, args.etas.split(",")) for sw, qu in grid]
    specs = [(st, c, s) for st in args.starts.split(",") for c in cfgs for s in range(args.seeds)]
    print(f"{len(specs)} passes on {args.workers} workers x {args.threads} threads", flush=True)
    res = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        fs = [ex.submit(job, args, pre, st, c, s, args.scratch) for st, c, s in specs]
        for f in as_completed(fs):
            res.append(f.result())
    res.sort(key=lambda r: (r["start"], r["tag"], r["seed"]))
    doc = {"d": args.d, "layers": args.layers, "heads": args.heads, "pre": pre, "taus": args.taus,
           "pretrain_steps": args.pretrain_steps, "runs": res, "spread": spread(args, args.scratch, res)}
    (out / "erasure.json").write_text(json.dumps(doc))
    print("wrote", out / "erasure.json", flush=True)


if __name__ == "__main__":
    main()
