"""slice.cu's serve mode in numpy, for small shapes off the GPU (prover.py --engine fake): the same commands and files, the
operands from numpy's generator (not the engine's sfill), the receipts DUOS's (derive.seg_receipts), so the round protocol,
its commitments and its statements run anywhere."""
from __future__ import annotations

import numpy as np

import derive


class FakeEngine:
    def __init__(self):
        self.st: dict[int, dict] = {}

    def cmd(self, op: str, *args) -> dict:
        a = [str(x) for x in args]
        if op == "setup":
            m, n, k, bseed, bpath = int(a[0]), int(a[1]), int(a[2]), int(a[3]), a[4]
            g = int(a[6]) if len(a) > 6 else 0
            B = np.random.default_rng(bseed).integers(-119, 120, (n, k)).astype(np.int8)
            B.tofile(bpath)
            self.st[g] = {"m": m, "n": n, "k": k, "B": B, "A": {}, "Y": {}}
            return {"ok": "setup", "m": m, "n": n, "k": k, "kind": "duos", "engine": "fake"}
        g = int(a[-1]) if op in ("opnds", "call", "tile", "ydump", "aload") and len(a) > {"opnds": 3, "call": 8, "tile": 4,
                                                                                         "ydump": 2, "aload": 2}[op] else 0
        S = self.st[g]
        m, n, k = S["m"], S["n"], S["k"]
        if op == "opnds":
            A = np.random.default_rng(int(a[1])).integers(-63, 64, (m, k)).astype(np.int8)
            A.tofile(a[2])
            S["A"][int(a[0])] = A
            return {"ok": "opnds", "slot": int(a[0])}
        if op == "aload":
            S["A"][int(a[0])] = np.fromfile(a[1], np.int8).reshape(m, k)
            return {"ok": "aload", "slot": int(a[0])}
        if op == "call":
            s, key, rseed, rpath = int(a[0]), bytes.fromhex("".join(a[1:5])), int(a[5]), a[6]
            A, B = S["A"][s], S["B"]
            tn_ = n // derive.BN
            R = np.empty((m // derive.TM * tn_, k // 32, derive.NCT), np.uint32)
            Y = np.empty((m, n), np.int64)
            for tm in range(m // derive.TM):
                for tn in range(tn_):
                    Ar, _U, Bp, _F, Ts = derive.tile_noised(A, B, key, tm, tn, k)
                    R[tm * tn_ + tn] = derive.seg_receipts(rseed, Ts, tm * tn_ + tn)
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
