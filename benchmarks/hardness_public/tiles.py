"""Per-tile r = 0 routes on the sampled open-weight matrices, at D-OUT's tile shape (8 output channels x 256 deep) and
coarser, so the gap compares with D-OUT's SAK-UF falsification test (note:compute-research/20261029T2229Z-report-hash-alt:
0.516% MAC-weighted on Llama-3.2-1B at 1-bit noise). Each tile takes its cheapest of dense int8, and an alphabet route
(`prices.dense_part`) on the entries in [-a, a] plus the rest on the ALU. K is kept in its stored order (D-OUT also
reorders K, which this omits). Writes `tiles.json` to `--out`.
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural  # noqa: E402
from prices import CHIPS, dense_part  # noqa: E402

SHAPES = ((8, 256), (16, 64), (128, 128), (1, None))  # (1, None): a route per output channel over all of K
ALPHA = (1, 2, 3, 4, 7, 15, 31, 36, 63)


def tile_costs(q, shape, chip, b):
    """(1, None) prices each output channel alone: output channels permute for free, so channels that share a route
    form the MMA's tiles (generous: it ignores the last partial tile of each route)."""
    n, k = q.shape
    tr, tk = shape[0], shape[1] or k
    nt, kt = n // tr, k // tk
    ab = np.abs(q[: nt * tr, : kt * tk].astype(np.int32)).reshape(nt, tr, kt, tk)
    best = np.ones((nt, kt))
    arg = np.zeros((nt, kt), dtype=np.int32)
    for a in ALPHA:
        d_out = (ab > a).mean(axis=(1, 3))
        c = dense_part(chip, a, b) + d_out * CHIPS[chip]["alu_mac"]
        better = c < best
        best = np.where(better, c, best)
        arg = np.where(better, a, arg)
    return best, arg


def work(task):
    name, repo, t, url, base, entry, _ = task
    q = natural.quantize(natural.load_tensor(url, base, entry))
    rec = {"model": name, "tensor": t, "shape": list(q.shape), "macs": int(q.size), "routes": {}}
    for shape in SHAPES:
        for chip in CHIPS:
            for b in (1, 2):
                best, arg = tile_costs(q, shape, chip, b)
                rec["routes"][f"{shape[0]}x{shape[1] or 'K'}_{chip}_b{b}"] = {
                    "cost": float(best.mean()), "frac_tiles_below_1": float((best < 1).mean()),
                    "alphabets": {str(a): int((arg == a).sum()) for a in (0,) + ALPHA if (arg == a).any()}}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(natural.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tasks = []
    for name in args.models.split(","):
        tasks += natural.model_tasks(name, "edges")[1]
    out = []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        for rec in ex.map(work, tasks):
            out.append(rec)
            r = rec["routes"]
            print(rec["model"], rec["tensor"], {k: round(v["cost"], 4) for k, v in r.items() if k.startswith("8x256")}, flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tiles.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
