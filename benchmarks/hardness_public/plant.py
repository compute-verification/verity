"""M7-E's plant test: does cheap structure planted at step T - L survive L honest training steps?

A sampled re-execution audit catches a faked step only if it samples that step, so what matters is how long a jump to
cheap weights stays cheap once honest training resumes after it. At an arm's checkpoint T - L, every MLP linear (fc and
out in each block) is replaced by its nearest per-row scaled ternary: t = sign(W)·[|W| > 0.7·mean|W|], scaled by the
mean |W| on t's support. Training then continues honestly on the checkpoint's own data stream to T. At T each planted
matrix's route q = a·t + S is priced (`poison.teacher_route`), against the same t on the honest final weights.

`--arms A,D2 --lengths 10,100,500 --scratch DIR --out DIR` trains each arm once with checkpoints, then every plant in
parallel, and writes `plant.json` to `--out`.
"""

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeds  # noqa: E402
from poison import teacher_route  # noqa: E402

PLANTED = ("fc", "out")


def ternary(w, pin=0):
    """t and the planted weights a·t. With `pin` k > 0 each row's largest entry becomes ±a·127/k and leaves t, so the
    row's absmax no longer moves with t's entries, which quantize to exactly ±k: a coarser LSB that honest steps must
    cross."""
    t = (np.sign(w) * (np.abs(w) > 0.7 * np.abs(w).mean(axis=1, keepdims=True))).astype(np.int32)
    a = np.array([np.abs(w[i][t[i] != 0]).mean() for i in range(len(w))])[:, None]
    if pin:
        r, j = np.arange(len(w)), np.abs(w).argmax(axis=1)
        t[r, j] = 0
        wp = a * t
        wp[r, j] = np.sign(w[r, j]) * a[:, 0] * 127 / pin
    else:
        wp = a * t
    return t, wp.astype(np.float32)


def plants(ck, pin):
    sd = torch.load(ck, weights_only=False)["model"]
    return {f"{i}.{n}": ternary(sd[f"blocks.{i}.{n}.weight"].numpy(), pin) for i in range(seeds.LAYERS) for n in PLANTED}


def tag(arm, pin, L):
    return f"{arm}_pin{pin}_plant{L}" if pin else f"{arm}_plant{L}"


def q(w):
    return seeds.q8(np.asarray(w, dtype=np.float32))[0].astype(np.int32)


def base(arm, data, scratch, steps, batch, lengths):
    seeds.train(arm, data, scratch, steps, batch, save_at={steps - L for L in lengths})


def planted(arm, data, scratch, steps, batch, pin, L):
    ts = plants(Path(scratch) / f"{arm}.ckpt{steps - L}.pt", pin)

    def mutate(model):
        for k, (_, wp) in ts.items():
            i, n = k.split(".")
            getattr(model.blocks[int(i)], n).weight.copy_(torch.from_numpy(wp))

    seeds.train(arm, data, scratch, steps, batch, resume=Path(scratch) / f"{arm}.ckpt{steps - L}.pt",
                tag=tag(arm, pin, L), mutate=mutate)


def mean_routes(recs):
    return {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}


def report(arm, pin, scratch, steps, lengths):
    sc = Path(scratch)
    hon = dict(np.load(sc / f"{arm}.npz"))
    out = {"pin": pin, "honest_val": json.loads((sc / f"{arm}.json").read_text())["val_loss"], "plants": {}}
    for L in lengths:
        ts = plants(sc / f"{arm}.ckpt{steps - L}.pt", pin)
        pl = dict(np.load(sc / f"{tag(arm, pin, L)}.npz"))
        per = {}
        for k, (t, w0) in ts.items():
            wp, wh = pl[f"final/{k}"], hon[f"final/{k}"]
            fin = teacher_route(q(wp), t)
            per[k] = {"at_plant": teacher_route(q(w0), t)["teacher_route"], "planted_final": fin["teacher_route"],
                      "s_hist": fin["s_hist"], "honest_final": teacher_route(q(wh), t)["teacher_route"],
                      "rel_dist_to_plant": float(np.linalg.norm(wp - w0) / np.linalg.norm(wp)),
                      "corr_with_honest": float(np.corrcoef(wp.ravel(), wh.ravel())[0, 1]),
                      "planes_agree_with_honest": seeds.planes_agree(q(wp), q(wh))}
        v = list(per.values())
        out["plants"][str(L)] = {
            "val_loss": json.loads((sc / f"{tag(arm, pin, L)}.json").read_text())["val_loss"],
            "at_plant": mean_routes([r["at_plant"] for r in v]), "planted_final": mean_routes([r["planted_final"] for r in v]),
            "honest_final": mean_routes([r["honest_final"] for r in v]),
            "s0": float(np.mean([r["s_hist"]["0"] for r in v])),
            "s_le1": float(np.mean([r["s_hist"]["0"] + r["s_hist"]["1"] for r in v])),
            "rel_dist_to_plant": float(np.mean([r["rel_dist_to_plant"] for r in v])),
            "corr_with_honest": float(np.mean([r["corr_with_honest"] for r in v])),
            "planes_agree_with_honest": np.mean([r["planes_agree_with_honest"] for r in v], axis=0).tolist(),
            "per_matrix": per}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,D2")
    ap.add_argument("--lengths", default="10,100,500")
    ap.add_argument("--data", default=os.path.expanduser("~/hardness-ml/tinystories-bpe4096"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pins", default="0", help="0 plants a·t alone; k > 0 pins each row's absmax (`ternary`)")
    args = ap.parse_args()
    arms = args.arms.split(",")
    lengths = sorted(int(x) for x in args.lengths.split(","))
    pins = [int(x) for x in args.pins.split(",")]
    jobs = [(a, p, L) for a in arms for p in pins for L in lengths]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(arms), mp_context=ctx) as ex:
        for f in [ex.submit(base, a, args.data, args.scratch, args.steps, args.batch, lengths) for a in arms]:
            f.result()
    with ProcessPoolExecutor(len(jobs), mp_context=ctx) as ex:
        for f in [ex.submit(planted, a, args.data, args.scratch, args.steps, args.batch, p, L) for a, p, L in jobs]:
            f.result()
    res = {"steps": args.steps, "lengths": lengths,
           "arms": {(f"{a}/pin{p}" if p else a): report(a, p, args.scratch, args.steps, lengths) for a in arms for p in pins}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plant.json").write_text(json.dumps(res))
    for a, v in res["arms"].items():
        for L in lengths:
            p = v["plants"][str(L)]
            print(a, L, json.dumps({k: p[k] for k in ("val_loss", "at_plant", "planted_final", "honest_final", "s0",
                                                         "rel_dist_to_plant", "corr_with_honest")}), flush=True)


if __name__ == "__main__":
    main()
