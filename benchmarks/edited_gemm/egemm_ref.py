"""egemm.cu's gate judged on the CPU: each dumped tile recomputed from x (bf16), W and the per-call key alone.

From the key (meta.json's four big-endian words, 32 bytes): b, the dither bits (tag 7, sha_noise.cuh's NS 1 patch
layout), and U (tag 6, lpgemm.cu's LP_UW 64, derive.py's u_noise64 on the weights' rows). Then
    X  = clamp(floor(x inv_s), -127, 126) + b            Xq = clamp(rint(x inv_s), -127, 127)   (the plain quantizer)
    S  = X Id (S_ic = sum_t X_{i, 32 t + c}),  P_A = [S_hi | 0 | S_lo | 0],  S = 256 S_hi + S_lo, S_lo the low byte as s8
    W' = W + U Id^T,  P_B = [U | 0 | U | 0]
    Y  = X W^T (the served product), and C - S U^T == Y for C = X W'^T
    receipts: derive.py's tile_receipts over C's stride-32 running sums
and every one is compared with the device's bytes. One JSON line a gate directory; exit 1 on any difference, on a
device mismatch or on a control the device didn't reject.
    python3 egemm_ref.py GATE_DIR...
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e2e_slice"))
from derive import U_TAG, tile_receipts  # noqa: E402

DITHER_TAG = 7
T = 128


def block(key: bytes, tag: int, j: int) -> bytes:
    return hashlib.sha512(key + struct.pack(">Q", tag << 56 | j)).digest()


def u_rows(key: bytes, row0: int, rows: int) -> np.ndarray:
    U = np.zeros((rows, 32), np.int64)
    for j in range(row0 // 2, (row0 + rows) // 2):
        b = np.frombuffer(block(key, U_TAG, j), np.uint8).astype(np.int64)
        u = (b & 63) + 1
        U[2 * j - row0:2 * j - row0 + 2] = np.where(b & 64, -u, u).reshape(2, 32)
    return U


def dither_rows(key: bytes, row0: int, rows: int, k: int) -> np.ndarray:
    kb = k // 128
    b = np.zeros((rows, k), np.int64)
    for g in range(row0 // 4, (row0 + rows) // 4):
        for p in range(kb):
            bits = np.unpackbits(np.frombuffer(block(key, DITHER_TAG, g * kb + p), np.uint8), bitorder="little")
            b[4 * g - row0:4 * g - row0 + 4, 128 * p:128 * p + 128] = bits.reshape(4, 128)
    return b


def limbs(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = ((S & 0xFF) ^ 0x80) - 0x80
    return (S - lo) // 256, lo


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    out = {"dir": str(d), "m": m, "n": n, "k": k, "device_ok": meta["device_ok"], "control_rejected": meta["control_rejected"],
           "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"])
    z = np.zeros((T, 32), np.int64)
    for tm, tn in meta["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        rd = lambda name, dt: np.fromfile(d / (name + sfx), dt)  # noqa: E731
        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        f = x * inv_s
        X = np.clip(np.floor(f), -127, 126).astype(np.int64) + dither_rows(key, tm * T, T, k)
        Xq = np.clip(np.rint(f), -127, 127).astype(np.int64)
        S = X.reshape(T, k // 32, 32).sum(axis=1)
        sh, sl = limbs(S)
        PA = np.concatenate([sh, z, sl, z], axis=1)
        W = rd("W", np.int8).astype(np.int64).reshape(T, k)
        U = u_rows(key, tn * T, T)
        Wp = W + np.tile(U, (1, k // 32))
        PB = np.concatenate([U, z, U, z], axis=1)
        Y = X @ W.T
        C = np.zeros((T, T), np.int64)
        Ts = []
        for s in range(k // 32):
            C += X[:, 32 * s:32 * s + 32] @ Wp[:, 32 * s:32 * s + 32].T
            Ts.append((C & 0xFFFFFFFF).astype(np.uint32))
        R = tile_receipts(seed, Ts, tm * tiles_n + tn)
        res = {
            "X": bool((rd("X", np.int8).reshape(T, k) == X).all()),
            "X_range": [int(X.min()), int(X.max())],
            "Xq": bool((rd("Xq", np.int8).reshape(T, k) == Xq).all()),
            "PA": bool((rd("PA", np.int8).reshape(T, 128) == PA).all()),
            "Wp": bool((rd("Wp", np.int8).reshape(T, k) == Wp).all()),
            "Wp_range": [int(Wp.min()), int(Wp.max())],
            "PB": bool((rd("PB", np.int8).reshape(T, 128) == PB).all()),
            "peel": bool(((C - S @ U.T) == Y).all()),
            "Y": bool((rd("Y", np.int32).reshape(T, T).astype(np.int64) == Y).all()),
            "R": bool((rd("R", np.uint32).reshape(512, 8) == R).all()),
            "S_range": [int(S.min()), int(S.max())],
        }
        out["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool))
        ok &= -127 <= res["X_range"][0] and res["X_range"][1] <= 127 and -127 <= res["Wp_range"][0] and res["Wp_range"][1] <= 127
    out["ok"] = bool(ok)
    return out


def main() -> int:
    ok = True
    for a in sys.argv[1:]:
        r = check(Path(a))
        print(json.dumps(r), flush=True)
        ok &= r["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
