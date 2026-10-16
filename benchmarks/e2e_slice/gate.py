"""gate.py ENGINE DIR M N K: the engine's served product and receipts against derive.py, bit for bit, at one shape.

Two calls (B's side as two passes, then fused) on seeded operands under a fresh coin: P_A against hashlib's U (LP_UW 64),
and on three tiles (the first, the last, one drawn) the duo receipts and the served product A (B + F)^T recomputed from A, B
and the coin alone. Prints one JSON line; exits 0 only if every check passes.
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
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "engine.err", "w") as log:
        E = Engine(binary, log)
        setup = E.cmd("setup", m, n, k, 0x5EED, d / "B.bin")
        B = np.fromfile(d / "B.bin", np.int8).reshape(n, k)
        checks, calls = {"B_range": bool(np.abs(B.astype(int)).max() <= 119)}, []
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
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
            R = np.fromfile(d / f"R{slot}.bin", np.uint32).reshape(-1, derive.NCT, 8)
            tiles_n = n // 128
            for t in sorted({0, R.shape[0] - 1, int(rng.integers(R.shape[0]))}):
                tm, tn = divmod(t, tiles_n)
                want_r, want_y, _U, F = derive.tile_check(A, B, sigma, slot, tm, tn, k)
                E.cmd("tile", slot, tm, tn, d / "Y.bin")
                checks[f"R{slot}_t{t}"] = bool((R[t] == want_r).all())
                checks[f"Y{slot}_t{t}"] = bool((np.fromfile(d / "Y.bin", np.int32).reshape(128, 128) == want_y).all())
                checks[f"F{slot}_t{t}_pm1"] = bool(np.isin(F, [-1, 1]).all())
        E.close()
    ok = all(checks.values())
    print(json.dumps({"m": m, "n": n, "k": k, "ok": ok, "failed": [c for c, v in checks.items() if not v], "setup": setup,
                      "calls": calls, "checks": checks}))
    for f in d.glob("*.bin"):
        f.unlink()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
