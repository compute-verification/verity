"""HT-PROV's planner: a cheater who wants the pass's output to be a cheap W* commits W₀ planned against coins he doesn't
know, and the pass runs L honest steps from it under the live beacon.

The plan is planning through the mean: W₀ ← W₀ − (mean over M beacons of his own of W_L(W₀) − W*), from W₀ = W* and
`--iters` times, on the planted matrices (fc and out, as `starts.py`); everything else starts at the pretrained values.
A deviation at step T − L followed by L checked honest steps is the same game, the state at T − L being W₀: the coins
of the last L steps are drawn after it is committed.

The control is HT-PUBLIC's known-coin plan (the same iteration with M = 1 and the live beacon itself), which lands
exactly when the step is a contraction in reach.

For each target family, L and pass setting, on `--evals` fresh beacons: the served codes' cheapest route (`routes`),
the fraction of codes equal to W*'s, and the miss rms(W_L − W*) in served LSBs. Writes `planner.json` to `--out`.
"""

import argparse
import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import erasure
import routes
import starts
import toy


def run(args, pre, cfg, w0, b, L):
    """The planted matrices after L honest steps from w0 under beacon b."""
    model = erasure.build(args, pre)
    with torch.no_grad():
        for n, w in w0.items():
            model.linear(n).copy_(torch.from_numpy(w.astype(np.float32)))
    hard = toy.Hardening(model, toy.Pool(args.data), cfg, b)
    for t in range(L):
        hard.step(t)
    return model, {n: model.linear(n).detach().numpy().astype(np.float64) for n in w0}


def plan(args, pre, cfg, target, L, beacons):
    w0 = {n: w.copy() for n, w in target.items()}
    for _ in range(args.iters):
        outs = [run(args, pre, cfg, w0, b, L)[1] for b in beacons]
        for n in w0:
            w0[n] -= np.mean([o[n] for o in outs], axis=0) - target[n]
    return w0


def evaluate(args, pre, cfg, w0, target, tplant, L, b):
    model, wl = run(args, pre, cfg, w0, b, L)
    per = {}
    for n, w in wl.items():
        q, s = toy.codes(w, cfg.quant)
        qt, _ = toy.codes(target[n], cfg.quant)
        r = routes.search(q, tplant.get(n))
        per[n] = {"h200": r["h200"]["cost"], "sm120": r["sm120"]["cost"], "family": r["sm120"]["family"],
                  "exact": float((q == qt).mean()),
                  "miss_rms": float(np.sqrt((((w - target[n]) / s[:, None]) ** 2).mean()))}
    v = list(per.values())
    return {"h200": float(np.mean([p["h200"] for p in v])), "sm120": float(np.mean([p["sm120"] for p in v])),
            "exact": float(np.mean([p["exact"] for p in v])), "miss_rms": float(np.mean([p["miss_rms"] for p in v])),
            "val": toy.Pool(args.data).val_loss(model),
            "val_served": toy.served_val(model, toy.Pool(args.data), cfg.quant), "per_matrix": per}


def job(args, pre, family, cfg, L):
    torch.set_num_threads(args.threads)
    t0 = time.time()
    model = erasure.build(args, pre)
    target, tplant = {}, {}
    for n in erasure.planted_names(model):
        target[n], t = starts.plant(model.linear(n).detach().numpy(), family, cfg.quant)
        if t is not None:
            tplant[n] = t
    own = [toy.beacon(f"hardness-prov planner own {m}") for m in range(args.m)]
    live = [toy.beacon(f"hardness-prov planner live {e}") for e in range(args.evals)]
    out = {"family": family, "cfg": cfg.__dict__, "tag": cfg.tag(), "L": L}
    out["unplanned"] = [evaluate(args, pre, cfg, target, target, tplant, L, b) for b in live]
    w0 = plan(args, pre, cfg, target, L, own)
    out["planned"] = [evaluate(args, pre, cfg, w0, target, tplant, L, b) for b in live]
    out["w0_shift_rms"] = float(np.mean([np.sqrt((((w0[n] - target[n]) / toy.scale(target[n], cfg.quant)[:, None]) ** 2)
                                                 .mean()) for n in target]))
    out["known"] = [evaluate(args, pre, cfg, plan(args, pre, cfg, target, L, [b]), target, tplant, L, b)
                    for b in live[:1]]
    sm = {k: np.mean([e["sm120"] for e in out[k]]) for k in ("unplanned", "planned", "known")}
    print(f"{family} {cfg.tag()} L {L}: sm120 unplanned {sm['unplanned']:.3f} planned {sm['planned']:.3f} "
          f"known {sm['known']:.3f} exact {np.mean([e['exact'] for e in out['planned']]):.3f} "
          f"miss {np.mean([e['miss_rms'] for e in out['planned']]):.2f} {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=0, help="0: d / 48")
    ap.add_argument("--pre", required=True, help="the pretrained state dict (erasure.py's pre_d{d}.pt)")
    ap.add_argument("--data", default=erasure.DATA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="tern,ternpin8,s24,small")
    ap.add_argument("--Ls", default="1,3,10,30")
    ap.add_argument("--lrs", default="1e-5,1e-4")
    ap.add_argument("--sigma-w", default="0,0.5")
    ap.add_argument("--sigma-a", type=float, default=0.05)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--quants", default="absmax")
    ap.add_argument("--m", type=int, default=4, help="the planner's own beacons")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--evals", type=int, default=2)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    args.heads = args.heads or max(1, args.d // 48)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = [toy.Config(lr=float(lr), sigma_w=float(sw), sigma_a=args.sigma_a, p_drop=args.p_drop, quant=qu)
            for lr, sw, qu in itertools.product(args.lrs.split(","), args.sigma_w.split(","), args.quants.split(","))]
    specs = [(f, c, int(L)) for f in args.families.split(",") for c in cfgs for L in args.Ls.split(",")]
    specs.sort(key=lambda s: -s[2])
    print(f"{len(specs)} plans on {args.workers} workers x {args.threads} threads", flush=True)
    res = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for f in as_completed([ex.submit(job, args, args.pre, *s) for s in specs]):
            res.append(f.result())
    res.sort(key=lambda r: (r["family"], r["tag"], r["L"]))
    doc = {"d": args.d, "layers": args.layers, "pre": args.pre, "m": args.m, "iters": args.iters, "runs": res}
    (out / "planner.json").write_text(json.dumps(doc))
    print("wrote", out / "planner.json", flush=True)


if __name__ == "__main__":
    main()
