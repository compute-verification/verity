"""wgemm.cu's gate on the CPU: for each gate directory (``wgemm gate DIR M N K SEED``), each w_ kernel's receipts against
derive.win_receipts over C = X W'^T's running sums (X and W' as the gate dumped them), on every tile (or the corner and
middle tiles past 16), and a one-word mutation of each that must be refused. One JSON line a directory; exit 1 if any
compare fails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive  # noqa: E402


def running_sums(X: np.ndarray, W: np.ndarray, tm: int, tn: int) -> np.ndarray:
    k = X.shape[1]
    A = X[tm * 128:(tm + 1) * 128].astype(np.int64).reshape(128, k // 32, 32)
    B = W[tn * 128:(tn + 1) * 128].astype(np.int64).reshape(128, k // 32, 32)
    return (np.cumsum(np.einsum("isk,jsk->sij", A, B), axis=0) & 0xFFFFFFFF).astype(np.uint32)


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed = meta["m"], meta["n"], meta["k"], meta["seed"]
    X = np.fromfile(d / "X.bin", np.int8).reshape(m, k)
    W = np.fromfile(d / "Wp.bin", np.int8).reshape(n, k)
    tiles_m, tiles_n = m // 128, n // 128
    every = [(i, j) for i in range(tiles_m) for j in range(tiles_n)]
    tiles = every if len(every) <= 16 else [(0, 0), (tiles_m // 2, tiles_n // 2), (tiles_m - 1, tiles_n - 1)]
    Ts = {t: running_sums(X, W, *t) for t in tiles}
    out, ok = {}, meta["device_ok"]
    for name, dev in meta["device"].items():
        S = dev["S"]
        nw = k // 32 // S
        R = np.fromfile(d / f"R_{name}.bin", np.uint32).reshape(tiles_m * tiles_n, nw, derive.NCT)
        bad = mut = 0
        for tm, tn in tiles:
            tile = tm * tiles_n + tn
            ref = derive.win_receipts(seed, Ts[(tm, tn)], tile, S)
            bad += int((R[tile] != ref).sum())
            got = R[tile].copy()
            got[nw // 2, 7] ^= np.uint32(1 << 13)
            mut += int(not np.array_equal(got, ref))
        out[name] = {"S": S, "tiles": len(tiles), "words": len(tiles) * nw * derive.NCT, "r_bad": bad,
                     "mutation_refused": mut == len(tiles), "y_bad": dev["y_bad"]}
        ok &= bad == 0 and mut == len(tiles) and dev["y_bad"] == 0
    return {"dir": str(d), "m": m, "n": n, "k": k, "ok": bool(ok), "arms": out}


if __name__ == "__main__":
    rs = [check(Path(p)) for p in sys.argv[1:]]
    for r in rs:
        print(json.dumps(r))
    sys.exit(0 if rs and all(r["ok"] for r in rs) else 1)
