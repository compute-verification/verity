"""slice.cu's serve mode in numpy, for small shapes off the GPU (prover.py --engine fake): the same commands and files, the
operands from numpy's generator (not the engine's sfill), the receipts checkpointed every S segments (derive.win_receipts,
the call's last two arguments S and the fold), so the round protocol,
its commitments and its statements run anywhere. ``dither`` stands for slice.cu built -DLP_DITHER."""
from __future__ import annotations

import numpy as np

import derive

NARGS = {"opnds": 3, "call": 8, "tile": 4, "ydump": 2, "aload": 2}


class FakeEngine:
    def __init__(self, dither: bool = False):
        self.dither = dither
        self.st: dict[int, dict] = {}

    def cmd(self, op: str, *args) -> dict:
        a = [str(x) for x in args]
        if op == "setup":
            m, n, k, bseed, bpath = int(a[0]), int(a[1]), int(a[2]), int(a[3]), a[4]
            g = int(a[6]) if len(a) > 6 else 0
            sb = len(a) > 7 and a[7] == "b"
            rng = np.random.default_rng(bseed)
            fixed = rng.integers(-63, 64, (m, k)) if sb else rng.integers(-119, 120, (n, k))
            fixed = fixed.astype(np.int8)
            fixed.tofile(bpath)
            self.st[g] = {"m": m, "n": n, "k": k, "sb": sb, "fixed": fixed, "X": {}, "Y": {}}
            return {"ok": "setup", "m": m, "n": n, "k": k, "kind": "duos", "side": "b" if sb else "a",
                    "dither": int(self.dither), "engine": "fake"}
        g = int(a[NARGS[op]]) if op in NARGS and len(a) > NARGS[op] else 0
        S = self.st[g]
        m, n, k, sb = S["m"], S["n"], S["k"], S["sb"]
        rows = n if sb else m
        if op == "opnds":
            rng = np.random.default_rng(int(a[1]))
            X = (rng.integers(-119, 120, (rows, k)) if sb else rng.integers(-63, 64, (rows, k))).astype(np.int8)
            X.tofile(a[2])
            S["X"][int(a[0])] = X
            return {"ok": "opnds", "slot": int(a[0])}
        if op == "aload":
            S["X"][int(a[0])] = np.fromfile(a[1], np.int8).reshape(rows, k)
            return {"ok": "aload", "slot": int(a[0])}
        if op == "call":
            s, key, rseed, rpath = int(a[0]), bytes.fromhex("".join(a[1:5])), int(a[5]), a[6]
            win, fold = (int(a[9]), a[10]) if len(a) > 10 else (1, "duo")
            A, B = (S["fixed"], S["X"][s]) if sb else (S["X"][s], S["fixed"])
            tn_ = n // derive.BN
            R = np.empty((m // derive.TM * tn_, k // 32 // win, derive.NCT), np.uint32)
            Y = np.empty((m, n), np.int64)
            for tm in range(m // derive.TM):
                for tn in range(tn_):
                    Ar, _U, Bp, _F, Ts = derive.tile_noised(A, B, key, tm, tn, k, self.dither)
                    R[tm * tn_ + tn] = derive.win_receipts(rseed, Ts, tm * tn_ + tn, win, fold)
                    Y[tm * derive.TM:(tm + 1) * derive.TM, tn * derive.BN:(tn + 1) * derive.BN] = Ar @ Bp.T
            R.tofile(rpath)
            S["Y"][s] = Y.astype(np.int32)
            return {"ok": "call", "slot": s, "edited_ms": 0.0, "plain_ms": 0.0, "engine": "fake"}
        if op == "tile":
            s, tm, tn = int(a[0]), int(a[1]), int(a[2])
            S["Y"][s][tm * derive.TM:(tm + 1) * derive.TM, tn * derive.BN:(tn + 1) * derive.BN].tofile(a[3])
            return {"ok": "tile"}
        if op == "ydump":
            S["Y"][int(a[0])].tofile(a[1])
            return {"ok": "ydump"}
        raise RuntimeError(f"fake engine: no {op}")

    def close(self) -> None:
        pass
