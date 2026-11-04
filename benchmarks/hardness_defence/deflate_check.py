"""A deflR target on real matrices: the codes `quant.py` serves (W·R less its top-R part from float64 eigenvectors of
(W·R)ᵀ(W·R), calibrated in torch) against the codes `rotated.py --deflate R` priced (numpy's SVD, `price.np_calibrate`).
One JSON line per matrix in --out: both c, the share of codes equal and the largest code difference.

    python benchmarks/hardness_defence/deflate_check.py --hf unsloth/Meta-Llama-3.1-70B --layers 0,33 \
        --match gate_proj,up_proj --deflate 32 --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
import rotated  # noqa: E402  (first: it puts HT-PREMISE's natural ahead of hardness_public's)

sys.path.insert(0, str(HERE.parent / "hardness_useful"))
import quant  # noqa: E402

nat = rotated.nat


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True)
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--deflate", type=int, default=32)
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--match", required=True, help="comma-separated substrings of the matrices' names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    layers, match = [int(x) for x in args.layers.split(",")], args.match.split(",")
    jobs = [(t, src) for t, src in rotated.hf_jobs(args.hf, args.target, layers, args.deflate)
            if any(m in t for m in match)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for t, src in jobs:
            t0 = time.time()
            q_priced, c_priced = rotated.load_codes(src)
            w = torch.from_numpy(nat.load_tensor(src[1], src[2], src[3])).to(torch.bfloat16)
            wr = quant.rotate_t(w)
            a, b = quant.top_part(wr, args.deflate)
            res = (wr - a @ b.T).float()
            c, ov = quant.calibrate(res, args.target)
            q_served = quant.codes(res, quant.row_scale(res, c)).numpy().astype(np.int16)
            rec = {"tensor": t, "deflate": args.deflate, "target": args.target, "c_served": c, "c_priced": c_priced,
                   "strassen1_served": ov, "equal": float((q_served == q_priced).mean()),
                   "max_diff": int(np.abs(q_served.astype(np.int32) - q_priced).max()),
                   "seconds": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
