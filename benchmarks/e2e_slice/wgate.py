"""wgemm.cu's gate on the CPU: for each gate directory (``wgemm gate DIR M N K SEED``), each w_ kernel's receipts against
derive.win_receipts over C = X W'^T's running sums (X and W' as the gate dumped them; a swapped arm's, s_NAME, over
C^T = W' X^T: a = W', b = X), on every tile (or the corner and middle tiles past 16), and a one-word mutation of each that
must be refused. The swapped arms' operands are then split as PouwSegCheckDither_v2 reads them: W' = W + U with U one
period of 32 a row in +-1..+-64 (PouwI8Add's a + u), and X = B + f with B = clamp(floor(x inv_s), -127, 126) and f in
{0, 1} (PouwI8AddBit's b + bit). One JSON line a directory; exit 1 if any compare fails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive  # noqa: E402


def running_sums(A: np.ndarray, B: np.ndarray, tm: int, tn: int) -> np.ndarray:
    k = A.shape[1]
    a = A[tm * 128:(tm + 1) * 128].astype(np.int64).reshape(128, k // 32, 32)
    b = B[tn * 128:(tn + 1) * 128].astype(np.int64).reshape(128, k // 32, 32)
    return (np.cumsum(np.einsum("isk,jsk->sij", a, b), axis=0) & 0xFFFFFFFF).astype(np.uint32)


def dither_split(d: Path, X: np.ndarray, Wp: np.ndarray, inv_bits: int) -> dict:
    """The swapped launch's operands as the Dither statement's inputs: a = W, u = W' - W, b = B, f = X - B."""
    m, k = X.shape
    x = (np.fromfile(d / "x.bin", np.uint16).reshape(m, k).astype(np.uint32) << 16).view(np.float32)
    B = np.clip(np.floor(x * np.uint32(inv_bits).view(np.float32)), -127, 126).astype(np.int16)
    f = X.astype(np.int16) - B
    W = np.fromfile(d / "W.bin", np.int8).reshape(Wp.shape)
    U = Wp.astype(np.int16) - W.astype(np.int16)
    u = U[:, :32]
    bits_ok = bool(((f == 0) | (f == 1)).all())
    u_ok = bool((U == np.tile(u, (1, k // 32))).all() and (np.abs(u) >= 1).all() and (np.abs(u) <= 64).all())
    return {"bits_ok": bits_ok, "f_ones": round(float(f.mean()), 4), "u_period32_ok": u_ok, "ok": bits_ok and u_ok}


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed = meta["m"], meta["n"], meta["k"], meta["seed"]
    X = np.fromfile(d / "X.bin", np.int8).reshape(m, k)
    W = np.fromfile(d / "Wp.bin", np.int8).reshape(n, k)
    Ts: dict = {}
    out, ok = {}, meta["device_ok"]
    for name, dev in meta["device"].items():
        S, swap = dev["S"], dev.get("swap", False)
        A, B = (W, X) if swap else (X, W)
        tiles_m, tiles_n = A.shape[0] // 128, B.shape[0] // 128
        every = [(i, j) for i in range(tiles_m) for j in range(tiles_n)]
        tiles = every if len(every) <= 16 else [(0, 0), (tiles_m // 2, tiles_n // 2), (tiles_m - 1, tiles_n - 1)]
        nw = k // 32 // S
        R = np.fromfile(d / f"R_{name}.bin", np.uint32).reshape(tiles_m * tiles_n, nw, derive.NCT)
        bad = mut = 0
        for tm, tn in tiles:
            if (swap, tm, tn) not in Ts:
                Ts[(swap, tm, tn)] = running_sums(A, B, tm, tn)
            tile = tm * tiles_n + tn
            ref = derive.win_receipts(seed, Ts[(swap, tm, tn)], tile, S)
            bad += int((R[tile] != ref).sum())
            got = R[tile].copy()
            got[nw // 2, 7] ^= np.uint32(1 << 13)
            mut += int(not np.array_equal(got, ref))
        out[name] = {"S": S, "swap": swap, "tiles": len(tiles), "words": len(tiles) * nw * derive.NCT, "r_bad": bad,
                     "mutation_refused": mut == len(tiles), "y_bad": dev["y_bad"]}
        ok &= bad == 0 and mut == len(tiles) and dev["y_bad"] == 0
    split = None
    if any(dev.get("swap") for dev in meta["device"].values()):
        split = dither_split(d, X, W, meta["inv_s_bits"])
        ok &= split["ok"]
    return {"dir": str(d), "m": m, "n": n, "k": k, "ok": bool(ok), "dither_split": split, "arms": out}


if __name__ == "__main__":
    rs = [check(Path(p)) for p in sys.argv[1:]]
    for r in rs:
        print(json.dumps(r))
    sys.exit(0 if rs and all(r["ok"] for r in rs) else 1)
