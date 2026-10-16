"""gate.py ENGINE DIR M N K [KIND]: the engine's served product and receipts against derive.py, bit for bit, at one shape.

Two calls (B's side as two passes, then fused) on seeded operands under a fresh coin: P_A against hashlib's U (LP_UW 64),
and on three tiles (the first, the last, one drawn) the duo receipts (KIND duo: a thread's 8 words; duos: a word a thread
and segment) and the served product A (B + F)^T recomputed from A, B and the coin alone. Then the second stage (its own B):
the first call's served product through the public map (derive.phi) loaded as its A (m x n, so stage 2's k is n), one call,
the same checks. Prints one JSON line; exits 0 only if every check passes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive  # noqa: E402
from engine import Engine  # noqa: E402


def main() -> int:
    binary, d, m, n, k = sys.argv[1], Path(sys.argv[2]).resolve(), *map(int, sys.argv[3:6])
    kind = sys.argv[6] if len(sys.argv) > 6 else "duo"
    seg = kind == "duos"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    checks, calls = {}, []

    def check(E, stage: int, slot: int, A: np.ndarray, B: np.ndarray, sigma: bytes, c: int, tag: str) -> None:
        k = A.shape[1]
        R = np.fromfile(d / f"R{tag}.bin", np.uint32)
        R = R.reshape(-1, k // 32, derive.NCT) if seg else R.reshape(-1, derive.NCT, 8)
        tiles_n = n // 128
        for t in sorted({0, R.shape[0] - 1, int(rng.integers(R.shape[0]))}):
            tm, tn = divmod(t, tiles_n)
            want_r, want_y, _U, F = derive.tile_check(A, B, sigma, c, tm, tn, k, seg=seg)
            E.cmd("tile", slot, tm, tn, d / "Y.bin", stage)
            checks[f"R{tag}_t{t}"] = bool((R[t] == want_r).all())
            checks[f"Y{tag}_t{t}"] = bool((np.fromfile(d / "Y.bin", np.int32).reshape(128, 128) == want_y).all())
            checks[f"F{tag}_t{t}_pm1"] = bool(np.isin(F, [-1, 1]).all())

    with open(d / "engine.err", "w") as log:
        E = Engine(binary, log)
        setup = [E.cmd("setup", m, n, k, 0x5EED, d / "B.bin", kind, 0), E.cmd("setup", m, n, n, 0x5EEE, d / "B2.bin", kind, 1)]
        B = np.fromfile(d / "B.bin", np.int8).reshape(n, k)
        B2 = np.fromfile(d / "B2.bin", np.int8).reshape(n, n)
        checks["B_range"] = bool(np.abs(B.astype(int)).max() <= 119 and np.abs(B2.astype(int)).max() <= 119)
        checks["B2_differs"] = bool((B[:, :min(n, k)] != B2[:, :min(n, k)]).any())
        for slot, fused in ((0, 0), (1, 1)):
            E.cmd("opnds", slot, 1000 + slot, d / f"A{slot}.bin")
            A = np.fromfile(d / f"A{slot}.bin", np.int8).reshape(m, k)
            checks[f"A{slot}_range"] = bool(np.abs(A.astype(int)).max() <= 63 and A.min() == -63 and A.max() == 63)
            sigma = os.urandom(64)
            key = derive.call_key(sigma, slot)
            r = E.cmd("call", slot, *derive.sig_words(key), derive.receipt_seed(sigma, slot), d / f"R{slot}.bin", fused)
            calls.append(r)
            E.cmd("pa", d / "PA.bin")
            U = derive.u_noise64(key, m)
            checks[f"PA{slot}"] = bool((np.fromfile(d / "PA.bin", np.int8).reshape(m, 128) == np.tile(U, (1, 4))).all())
            checks[f"U{slot}_uniform"] = bool(np.abs(U).min() == 1 and np.abs(U).max() == 64 and abs(U.mean()) < 4)
            check(E, 0, slot, A, B, sigma, slot, str(slot))
            if slot == 0:
                E.cmd("ydump", 0, d / "Y1.bin")
                A2 = derive.phi(np.fromfile(d / "Y1.bin", np.int32).reshape(m, n))
                A2.tofile(d / "A2.bin")
                E.cmd("aload", 0, d / "A2.bin", 1)
                sigma2 = os.urandom(64)
                key2 = derive.call_key(sigma2, 0)
                calls.append(E.cmd("call", 0, *derive.sig_words(key2), derive.receipt_seed(sigma2, 0), d / "Rs2.bin", 1, 1))
                checks["A2_range"] = bool(np.abs(A2.astype(int)).max() <= 63)
                check(E, 1, 0, A2, B2, sigma2, 0, "s2")
        E.close()
    ok = all(checks.values())
    print(json.dumps({"m": m, "n": n, "k": k, "kind": kind, "ok": ok, "failed": [c for c, v in checks.items() if not v],
                      "setup": setup, "calls": calls, "checks": checks}))
    for f in d.glob("*.bin"):
        f.unlink()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
