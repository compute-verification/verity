"""d2sgemm.cu's gate judged on the CPU: each dumped tile recomputed from x (bf16), W0 and the per-call key alone.

From the key (meta.json's four big-endian words): b (tag 7) over X's rows and c (tag 8) over W0's, both in sha_noise.cuh's
NS 1 patch layout (block (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 + col % 128, bit e % 8 of digest byte
e / 8). Then, as benchmarks/ppl/steps.py's int8 step serves `act = weight = "dither"` (Dither-2S, SKIP's adopted-bd-lm):
    X   = clamp(floor(x inv_s), -126, 126) + b,   W' = W0 + c
    T_s = sum_{k' < 32 s} X W'^T mod 2^32 (s = 1 .. k / 32),   Y = T_{k / 32} = X W'^T (exact in s32)
and the device's bytes must equal X, W', the served Y (d_l7qs's) and both schedules' receipts (d_l7qs, d_l2q: derive.py's
tile_receipts over T_1 .. T_{k / 32}). One JSON line a gate directory; exit 1 on any difference, on a device mismatch or on
a control the device didn't reject.
    uv run --no-project --with numpy python3 benchmarks/pouw/edited_gemm/d2sgemm_ref.py GATE_DIR...
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
from derive import NCT, tile_receipts  # noqa: E402

T = 128
TOP = 126
DITHER_TAG, C_TAG = 7, 8


def block(key: bytes, tag: int, j: int) -> bytes:
    return hashlib.sha512(key + struct.pack(">Q", tag << 56 | j)).digest()


def patch_bits(key: bytes, tag: int, row0: int, rows: int, k: int) -> np.ndarray:
    kb = k // 128
    b = np.zeros((rows, k), np.int64)
    for g in range(row0 // 4, (row0 + rows) // 4):
        for p in range(kb):
            bits = np.unpackbits(np.frombuffer(block(key, tag, g * kb + p), np.uint8), bitorder="little")
            b[4 * g - row0:4 * g - row0 + 4, 128 * p:128 * p + 128] = bits.reshape(4, 128)
    return b


def served(x: np.ndarray, inv_s: np.float32, W0: np.ndarray, key: bytes, tm: int, tn: int) -> tuple[np.ndarray, np.ndarray]:
    """Dither-2S's operands for tile (tm, tn): X (from x's rows) and W' (from W0's rows)."""
    k = x.shape[1]
    X = np.clip(np.floor(x * inv_s), -TOP, TOP).astype(np.int64) + patch_bits(key, DITHER_TAG, tm * T, T, k)
    return X, W0 + patch_bits(key, C_TAG, tn * T, T, k)


def prefix_words(X: np.ndarray, Wp: np.ndarray) -> np.ndarray:
    """T_1 .. T_{k / 32}, (k / 32, T, T) u32: the running sums after each 32-step segment, mod 2^32."""
    S = X.shape[1] // 32
    seg = np.matmul(X.reshape(T, S, 32).transpose(1, 0, 2), Wp.reshape(T, S, 32).transpose(1, 2, 0))
    return (np.cumsum(seg, axis=0) & 0xFFFFFFFF).astype(np.uint32)


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    kinds = meta["kinds"]
    out = {"dir": str(d), "m": m, "n": n, "k": k, "design": "dither-2s", "device_ok": meta["device_ok"],
           "control_rejected": meta["control_rejected"], "device": meta["device"], "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"]) and set(kinds.values()) == {"duo32"}
    for tm, tn in meta["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        rd = lambda name, dt: np.fromfile(d / (name + sfx), dt)  # noqa: E731
        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        W0 = rd("W", np.int8).astype(np.int64).reshape(T, k)
        X, Wp = served(x, inv_s, W0, key, tm, tn)
        Ts = prefix_words(X, Wp)
        Y = X @ Wp.T
        want_r = tile_receipts(seed, list(Ts), tm * tiles_n + tn)
        res = {
            "X": bool((rd("X", np.int8).reshape(T, k) == X).all()),
            "X_range": [int(X.min()), int(X.max())],
            "Wp": bool((rd("Wp", np.int8).reshape(T, k) == Wp).all()),
            "Wp_range": [int(Wp.min()), int(Wp.max())],
            "W0_range": [int(W0.min()), int(W0.max())],
            "Y_s32": bool(np.abs(Y).max() < 2 ** 31),
            "Y_is_last_word": bool((Ts[-1] == (Y & 0xFFFFFFFF).astype(np.uint32)).all()),
            "Y": bool((rd("Y_d_l7qs", np.int32).reshape(T, T).astype(np.int64) == Y).all()),
            "R": {name: bool((rd("R_" + name, np.uint32).reshape(NCT, 8) == want_r).all()) for name in kinds},
            "words": int(Ts.shape[0]),
        }
        out["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool)) and all(res["R"].values())
        ok &= -TOP <= res["X_range"][0] and res["X_range"][1] <= TOP + 1 and -TOP <= res["Wp_range"][0] and res["Wp_range"][1] <= TOP + 1
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
