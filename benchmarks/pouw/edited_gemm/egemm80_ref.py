"""egemm80.cu's gate judged on the CPU: every arm's Y and receipts recomputed from X, W and U alone.

    W' = W + U Id^T,  C = X W'^T,  Y = C - S U^T = X W^T
    receipts: derive.py's tile_receipts over C's running sums T at each commit boundary, tile by tile:
      e16  every 16 of k;  e32  every 32 (egemm.cu's duo32);  e64  every 64 (its duo64: the odd k32 steps)

One JSON line a gate directory, with each arm's verdict; exit 1 on any difference.
    python3 egemm80_ref.py GATE_DIR...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e2e_slice"))
from derive import BN, NCT, TM, tile_receipts  # noqa: E402

SPACING = {"e16": 16, "e32": 32, "e64": 64}


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    M, N, K, seed = meta["M"], meta["N"], meta["K"], meta["seed"]
    X = np.fromfile(d / "X.bin", np.int8).reshape(M, K).astype(np.int64)
    W = np.fromfile(d / "W.bin", np.int8).reshape(N, K).astype(np.int64)
    U = np.fromfile(d / "U.bin", np.int8).reshape(N, 32).astype(np.int64)
    Wp = W + np.tile(U, (1, K // 32))
    Y = (X @ W.T).astype(np.int32)
    tiles_m, tiles_n = M // TM, N // BN
    out = {"dir": str(d), "M": M, "N": N, "K": K, "seed": seed, "arms": {}}
    sums = {}
    for tm in range(tiles_m):
        for tn in range(tiles_n):
            A, B = X[tm * TM:(tm + 1) * TM], Wp[tn * BN:(tn + 1) * BN]
            T = np.zeros((TM, BN), np.int64)
            Ts = []
            for s in range(K // 16):
                T += A[:, 16 * s:16 * s + 16] @ B[:, 16 * s:16 * s + 16].T
                Ts.append((T & 0xFFFFFFFF).astype(np.uint32))
            sums[tm * tiles_n + tn] = Ts
    ok = True
    for arm in meta["arms"]:
        y = np.fromfile(d / f"Y_{arm}.bin", np.int32).reshape(M, N)
        v = {"y_equal": bool(np.array_equal(y, Y))}
        if arm in SPACING:
            step = SPACING[arm] // 16
            r = np.fromfile(d / f"R_{arm}.bin", np.uint32).reshape(tiles_m * tiles_n, NCT, 8)
            bad = 0
            for t, Ts in sums.items():
                bad += int(np.count_nonzero(tile_receipts(seed, Ts[step - 1::step], t) != r[t]))
            v["receipt_words"] = int(r.size)
            v["receipt_words_wrong"] = bad
            # the control: one flipped receipt word must be caught
            r2 = r.copy()
            r2[0, 0, 0] ^= 1
            v["control_rejected"] = bool(np.count_nonzero(tile_receipts(seed, sums[0][step - 1::step], 0) != r2[0]))
            v["ok"] = v["y_equal"] and bad == 0 and v["control_rejected"]
        else:
            v["ok"] = v["y_equal"]
        ok &= v["ok"]
        out["arms"][arm] = v
    out["ok"] = ok
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    ok = True
    for p in sys.argv[1:]:
        r = check(Path(p))
        print(json.dumps(r))
        ok &= r["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
