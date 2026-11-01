"""The honest baseline at scale: a released model's linears under `routes.py`, with the public code statistics a gate
would check (`routes.stats`, top-r spectral energy), under `absmax` and `mad`.

`--repo` is a Hugging Face repo of safetensors (default an ungated copy of Llama-3.2-1B), fetched into `--scratch`. Every
2-D weight of a decoder layer (q, k, v, o, gate, up, down) is quantized per output row and priced. Writes `real.json` to
`--out`: per matrix its shape, best route per chip, every family's cost, the statistics and the energy shares.

`--plants` also projects every matrix onto each named family post hoc (`lr{r}res{k}`, `vq{f}res{k}` as in `starts`) and
records the same statistics, the planted route's own price (the cheap part's share plus its residual's route) and the
relative weight error ‖W₀ − W‖/‖W‖: what the gates would see of that structure at this scale, with no claim the
projected model is useful.
"""

import argparse
import json
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import routes
import toy

LAYER = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)_proj\.weight")


def load(path, name):
    import torch
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return f.get_tensor(name).to(torch.float32).numpy()


def price(path, name, quant, vq=False):
    w = load(path, name)
    q, _ = toy.codes(w, quant)
    r = routes.search(q, vq=vq)
    fam = {chip: {f: v["cost"] for f, v in r[chip]["families"].items()} for chip in routes.CHIPS}
    return {"name": name, "quant": quant, "shape": list(w.shape), "stats": routes.stats(q),
            "energy_top": r["energy_top"], "best": {chip: r[chip]["cost"] for chip in routes.CHIPS},
            "family": {chip: r[chip]["family"] for chip in routes.CHIPS}, "families": fam,
            "lowrank_r": {chip: r[chip]["families"]["lowrank"]["r"] for chip in routes.CHIPS}}


def planted(path, name, quant, family):
    import numpy as np
    import starts

    w = load(path, name).astype(np.float64)
    n, k = w.shape
    m = PLANT.fullmatch(family)
    kind, x, a = m.group(1), int(m.group(2)), int(m.group(3))
    if kind == "lr":
        base, share = starts.svd_r(w, x), routes.lowrank(x, n, k)
    else:
        c, assign = routes.kmeans(w, max(2, n // x))
        base, share = c[assign].astype(np.float64), max(2, n // x) / n
    w0 = starts.clip_around(w, base, a, quant)
    q, s = toy.codes(w0, quant)
    S = np.rint(q - base / s[:, None]).astype(np.int64)
    r = routes.search(q, vq=False)
    return {"name": name, "quant": quant, "plant": family, "shape": [n, k], "stats": routes.stats(q),
            "energy_top": r["energy_top"], "best": {chip: r[chip]["cost"] for chip in routes.CHIPS},
            "planted_route": {chip: min(1.0, share + routes.residual(S, chip)[0]) for chip in routes.CHIPS},
            "rel_err": float(np.linalg.norm(w0 - w) / np.linalg.norm(w))}


PLANT = re.compile(r"(lr|vq)(\d+)res(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma-separated layer indices (default all)")
    ap.add_argument("--quants", default="absmax,mad")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--vq", action="store_true", help="also the clustered routes (minutes of k-means a matrix)")
    ap.add_argument("--plants", default="", help="comma-separated lr{r}res{k} / vq{f}res{k} families to project onto")
    args = ap.parse_args()
    plants = [p for p in args.plants.split(",") if p]
    for p in plants:
        if not PLANT.fullmatch(p):
            ap.error(f"--plants: {p} is not lr{{r}}res{{k}} or vq{{f}}res{{k}}")
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local = Path(snapshot_download(args.repo, allow_patterns=["*.safetensors", "config.json"], cache_dir=args.scratch))
    keep = {int(x) for x in args.layers.split(",") if x}
    jobs, plant_jobs = [], []
    for path in sorted(local.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            for name in f.keys():
                m = LAYER.fullmatch(name)
                if m and (not keep or int(m.group(1)) in keep):
                    jobs += [(str(path), name, qu, args.vq) for qu in args.quants.split(",")]
                    plant_jobs += [(str(path), name, qu, p) for qu in args.quants.split(",") for p in plants]
    print(f"{len(jobs)} matrices x quantizers from {args.repo}, {len(plant_jobs)} plants", flush=True)
    res, res_plants = [], []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for r in ex.map(price, *zip(*jobs)):
            res.append(r)
            print(f"{r['name']} {r['quant']} {r['shape']} sm120 {r['best']['sm120']:.3f} ({r['family']['sm120']}) "
                  f"sigma_q {r['stats']['sigma_q']:.1f} le15 {r['stats']['frac_abs_le15']:.3f} "
                  f"top4 {r['energy_top'].get('4', float('nan')):.3f}", flush=True)
        for r in (ex.map(planted, *zip(*plant_jobs)) if plant_jobs else ()):
            res_plants.append(r)
            print(f"{r['name']} {r['quant']} {r['plant']} planted route sm120 {r['planted_route']['sm120']:.3f} "
                  f"rel_err {r['rel_err']:.3f} le15 {r['stats']['frac_abs_le15']:.3f} "
                  f"tail31 {r['stats']['tail31']:.3f} top32 {r['energy_top'].get('32', float('nan')):.3f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real.json").write_text(json.dumps({"repo": args.repo, "matrices": res, "plants": res_plants}))
    print("wrote", out / "real.json", flush=True)


if __name__ == "__main__":
    main()
