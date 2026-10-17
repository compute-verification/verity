"""CPU twin of dgemm.cu's and pgemm.cu's dumps: C and every receipt word, bit for bit, from X.bin, W.bin and meta.json alone.

python3 dgemm_ref.py DIR prints one JSON line of checks and exits 0 only if C.bin and R.bin both match.

The receipt (dgemm.cu's header): chain (i, tn, c) starts at duo_init(seed, (i tiles_n + tn) 16 + c) and absorbs, at each
boundary s = 1..k/L (meta.json's L: 32 for dgemm.cu, 32 lm for pgemm.cu) and in ascending tile column j with j % 16 == c,
the s32 running sum T_s[i, tn bn + j] at key position (j / 16) % 16. Duo is floored_gemm's (ref.py's key_mul(0, t),
key_xor(0, t), rot_of(False, 0, t)).
"""

import json
import sys
from pathlib import Path

import numpy as np

NPOS = 16


def mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        return x ^ (x >> np.uint32(16))


def rotl(x, r):
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


KEY_MUL = [np.uint32(int(mix32(0xA0000000 + 2 * t)) | 1) for t in range(NPOS)]
KEY_XOR = [mix32(0xB0000000 + 2 * t + 1) for t in range(NPOS)]
ROT = [((t * 7 + 5) % 31) + 1 for t in range(NPOS)]


def receipts(X, W, seed, bn, L=32, nch=16):
    """C (m x n, s32) and R (m x tiles_n x nch, u32)."""
    m, k = X.shape
    n = W.shape[0]
    tiles_n = n // bn
    idx = ((np.arange(m)[:, None, None] * tiles_n + np.arange(tiles_n)[None, :, None]) * nch
           + np.arange(nch)[None, None, :]).astype(np.uint32)
    y = mix32(np.uint32(seed) ^ mix32(idx))
    z = mix32(y + np.uint32(0x9E3779B9))
    T = np.zeros((m, n), dtype=np.int64)
    X64, W64 = X.astype(np.int64), W.astype(np.int64)
    with np.errstate(over="ignore"):
        for s in range(k // L):
            T += X64[:, s * L:(s + 1) * L] @ W64[:, s * L:(s + 1) * L].T
            Tw = (T & 0xFFFFFFFF).astype(np.uint32).reshape(m, tiles_n, bn // nch, nch)
            for u in range(bn // nch):
                t = u % NPOS
                p = Tw[:, :, u, :]
                v = y ^ p ^ KEY_XOR[t]
                y = v * KEY_MUL[t] + z
                z = rotl(v, ROT[t]) + y
    fin = y ^ rotl(z, 16)
    return T, fin


def main(d: Path) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, bn, seed = meta["m"], meta["n"], meta["k"], meta["bn"], meta["seed"]
    assert meta["L"] % 32 == 0 and k % meta["L"] == 0 and meta["nch"] == 16, meta
    X = np.fromfile(d / "X.bin", dtype=np.int8).reshape(m, k)
    W = np.fromfile(d / "W.bin", dtype=np.int8).reshape(n, k)
    T, R = receipts(X, W, seed, bn, L=meta["L"])
    C = np.fromfile(d / "C.bin", dtype=np.int32).reshape(m, n)
    Rg = np.fromfile(d / "R.bin", dtype=np.uint32).reshape(R.shape)
    checks = {
        "fits_s32": bool((np.abs(T) < 2**31).all()),
        "C": bool((C.astype(np.int64) == T).all()),
        "R": bool((Rg == R).all()),
        "C_bad": int((C.astype(np.int64) != T).sum()),
        "R_bad": int((Rg != R).sum()),
    }
    ok = checks["fits_s32"] and checks["C"] and checks["R"]
    print(json.dumps({"dir": str(d), "source": meta.get("source"), "arch": meta.get("arch"), "m": m, "n": n, "k": k,
                      "ok": ok, **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
