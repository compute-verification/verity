"""run.py's gate on the demo's seeded weights at the frozen H200 shape: compute-accounting's generator
(`e2e_h200.served_weights`), every row of one layer's linears (wq, wk, wv, wo, wup, wdown) and of the
head, each with its f32 scales, against activations of all of s8 with run.py's structured rows and `fixture`'s activation
scales (the special ones included): the kernel's picked config and its others, every cell against the Definition's twin.
`--gen` writes the weights into --weights as .npy (in parallel, no GPU: run.sh's "cpu:" step, outside the lease); the gate
loads them.

    python realw.py --out OUT --weights DIR --gen [--layer 0] [--seed text:verity/cmp92/h200/weights/v1] [--workers N]
    python realw.py --out OUT --weights DIR [--layer 0] [--ms 1,7,64,...] [--kernel wgmma|triton|int_mm] [--configs BEST_JSON]

The generator's module is on the path from examples/h200, after this directory (whose `run` it imports).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "e2e_h200"))
from e2e_h200 import served_weights as WG  # noqa: E402

SEED = "text:verity/cmp92/h200/weights/v1"
LINEARS = ("wq", "wk", "wv", "wo", "wup", "wdown")
MS = (1, 7, 64, 257, 1023, 4096, 16384, 20480)
CHUNK_ROWS = 256


def names(layer: int) -> list[str]:
    return [f"l{layer}.{t}" for t in LINEARS] + ["head"]


def _rows(args: tuple) -> tuple[int, np.ndarray]:
    seed, name, lo, hi = args
    return lo, WG.generate(WG.H200, seed, name, range(lo, hi))


def gen(dirpath: Path, layer: int, seed: bytes, workers: int) -> dict:
    dirpath.mkdir(parents=True, exist_ok=True)
    doc = {}
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for name in names(layer):
            t = WG.tensor(WG.H200, name)
            arr = np.lib.format.open_memmap(dirpath / f"{name}.npy", mode="w+", dtype=np.int8, shape=(t.rows, t.cols))
            tasks = [(seed, name, lo, min(lo + CHUNK_ROWS, t.rows)) for lo in range(0, t.rows, CHUNK_ROWS)]
            for lo, rows in pool.imap_unordered(_rows, tasks):
                arr[lo:lo + len(rows)] = rows
            arr.flush()
            sw = WG.generate(WG.H200, seed, f"{name}.scale")[0]
            np.save(dirpath / f"{name}.scale.npy", sw)
            doc[name] = {"n": t.rows, "k": t.cols, "first_row_bytes": bytes(arr[0, :8].view(np.uint8)).hex(),
                         "scale_range": [float(sw.min()), float(sw.max())]}
            del arr
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", default=SEED)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--extra-cfgs", default=None)
    a = ap.parse_args()
    out, wdir = Path(a.out), Path(a.weights)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"shape": WG.H200.name, "seed": a.seed, "layer": a.layer}
    if a.gen:
        doc = {**meta, "tensors": gen(wdir, a.layer, WG._bytes_arg(a.seed), a.workers)}
        (out / "gen.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(doc), flush=True)
        return 0

    import torch

    import fixture as FX
    import run as R
    if not FX.fp_env_ok():
        raise SystemExit("this process flushes float32 subnormals; the twin would be wrong")
    kern = R.kernel(a.kernel)
    if a.configs:
        R.load_configs(a.configs, kern)
    ms = tuple(int(v) for v in a.ms.split(","))
    extra = tuple(kern.Cfg.parse(c) for c in (a.extra_cfgs.split(",") if a.extra_cfgs is not None else kern.extra) if c)
    log = R.logger(out / "gate.jsonl")
    recs = []
    for name in names(a.layer):
        wt = R.Weights.given(np.load(wdir / f"{name}.npy"), np.load(wdir / f"{name}.scale.npy"))
        recs += R.gate_shape(f"{WG.H200.name}:{name}", wt, ms, extra, kern, False, log)
        del wt
        torch.cuda.empty_cache()
    doc = {**meta, **R.summary(recs, kern, ms)}
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
