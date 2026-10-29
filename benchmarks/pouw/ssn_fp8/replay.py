"""Replay tc_probe tile captures (``tiles_*.npz``: A, Bt, C, D) through `sim`: every finite hardware word, bit for bit.

``python -m benchmarks.pouw.ssn_fp8.replay DIR...``: each DIR is one instruction's tc_probe ``--out`` (its name ends in
``e4m3``, ``e5m2`` or ``bf16``); a ``tiles_chain4_*`` capture is folded over its four k slices.  Prints one JSON line a
file and a total; exit 1 on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import sim

PIPES = {"e4m3": sim.E4M3_K32, "e5m2": sim.E5M2_K32, "bf16": sim.BF16_K16}


def replay(path: Path, pipe: sim.Pipe, chunk: int = 256) -> dict:
    z = np.load(path)
    A, Bt, C, D = z["A"], z["Bt"], z["C"], z["D"]
    n, m, K = A.shape
    finite = mism = 0
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        t = A[sl].shape[0]
        a = np.broadcast_to(A[sl][:, :, None, :], (t, m, Bt.shape[1], K))
        b = np.broadcast_to(Bt[sl][:, None, :, :], (t, m, Bt.shape[1], K))
        got, bad = sim.chain(pipe, C[sl], a, b, keep=False)
        finite += int((~bad).sum())
        mism += int(np.count_nonzero((got != D[sl]) & ~bad))
    return {"file": path.name, "words": int(D.size), "finite": finite, "steps": K // pipe.k, "mismatches": mism}


def main(dirs: list[str]) -> int:
    total = {"words": 0, "finite": 0, "mismatches": 0}
    for d in map(Path, dirs):
        pipe = PIPES[d.name.rsplit(".", 1)[-1]]
        for f in sorted(d.glob("tiles_*.npz")):
            r = replay(f, pipe)
            print(json.dumps({"dir": d.name, **r}), flush=True)
            for k in total:
                total[k] += r[k]
    print(json.dumps({"total": total}))
    return 1 if total["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
