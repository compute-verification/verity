"""The slice's derivations and its CPU reference, shared by the prover, the challenger and the gate.

From the round's coin sigma (64 bytes from the challenger's OS generator) and a call's index c:
    key_c   = SHA-512("verity/e2e-slice/call-sigma/v1\\0" || sigma || BE32(c))[:32]: the call's SHA-512 noise key (lpgemm.cu's Sig)
    rseed_c = BE32 of SHA-512("verity/e2e-slice/receipt-key/v1\\0" || sigma || BE32(c))[:4]: the duo receipts' key
    U       = lpgemm.cu's U at LP_UW 64 under key_c (m x 32, {±1..±64}): block j = rows 2j, 2j + 1, entry e = (row % 2) 32 + l
              digest byte e, u = (v & 63) + 1, negated if v & 64
    F       = sha_noise.cuh's pm1 noise of B's side under key_c (n x k, ±1)
A' = A + U W^T (A'_ik = A_ik + U_{i, k mod 32}), B' = B + F; the duo receipts at L = 32 over A' B'^T (mref.py's), one tile at a time.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "masked_gemm"))
sys.path.insert(0, str(HERE.parent / "floored_gemm"))
from ref import key_mul, key_xor, mix32, rot_of, rotl  # noqa: E402
from sref import patch_noise  # noqa: E402

U_TAG = 6
TM = BN = 128
WM, NCT = 32, 512


def call_key(sigma: bytes, c: int) -> bytes:
    return hashlib.sha512(b"verity/e2e-slice/call-sigma/v1\0" + sigma + c.to_bytes(4, "big")).digest()[:32]


def receipt_seed(sigma: bytes, c: int) -> int:
    return int.from_bytes(hashlib.sha512(b"verity/e2e-slice/receipt-key/v1\0" + sigma + c.to_bytes(4, "big")).digest()[:4], "big")


def sig_words(key: bytes) -> list[str]:
    """lpgemm.cu's Sig: the key's four big-endian 64-bit words, as the engine reads them (hex)."""
    return [f"{w:016x}" for w in struct.unpack(">4Q", key)]


def u_noise64(key: bytes, m: int) -> np.ndarray:
    U = np.zeros((m, 32), np.int64)
    for j in range(m // 2):
        b = np.frombuffer(hashlib.sha512(key + struct.pack(">Q", U_TAG << 56 | j)).digest(), np.uint8).astype(np.int64)
        u = (b & 63) + 1
        U[2 * j:2 * j + 2] = np.where(b & 64, -u, u).reshape(2, 32)
    return U


def f_rows(key: bytes, row0: int, rows: int, k: int) -> np.ndarray:
    """F's rows [row0, row0 + rows) (row0 and rows multiples of 4): the pm1 patches that cover them."""
    kb = k // 128
    pt = np.stack([patch_noise(key, 1, 1, g * kb + b) for g in range(row0 // 4, (row0 + rows) // 4) for b in range(kb)])
    return pt.reshape(rows // 4, kb, 4, 128).transpose(0, 2, 1, 3).reshape(rows, k)


def tile_receipts(seed: int, Ts: list[np.ndarray], tile: int) -> np.ndarray:
    """mref.duo_receipts for one tile (its index ``tile`` in the call's row-major tile order): Ts the tile's 128 x 128 u32
    words after each step, (NCT, 8)."""
    KA = np.array([[key_mul(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    KK = np.array([[key_xor(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    warps_n, MI, NI, nch = BN // 32, WM // 16, 4, 8
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, q = warp // warps_n, warp % warps_n, lane >> 2, lane & 3
    e = np.arange(4)
    rows = wm[:, None, None] * WM + 16 * np.arange(MI)[None, None, :] + g[:, None, None] + 8 * (e >> 1)[None, :, None]
    cols = wn[:, None, None] * 32 + 8 * np.arange(NI)[None, None, :] + 2 * q[:, None, None] + (e & 1)[None, :, None]
    idx = ((tile * NCT + ctid[:, None]) * nch + np.arange(nch)[None, :]).astype(np.uint32)
    y = mix32(np.uint32(seed) ^ mix32(idx))
    z = mix32(y + np.uint32(0x9E3779B9))
    with np.errstate(over="ignore"):
        for Tw in Ts:
            for mi in range(MI):
                for ni in range(NI):
                    t = mi * NI + ni
                    p = Tw[rows[:, :, mi], cols[:, :, ni]]
                    sl = slice(4 * (t % 2), 4 * (t % 2) + 4)
                    u = y[:, sl] ^ p ^ KK[0, t]
                    y[:, sl] = u * KA[0, t] + z[:, sl]
                    z[:, sl] = rotl(u, rot_of(False, 0, t)) + y[:, sl]
        return y ^ rotl(z, 16)


def tile_check(A: np.ndarray, B: np.ndarray, sigma: bytes, c: int, tm: int, tn: int, k: int):
    """The tile (tm, tn)'s duo receipts and served product A B'^T from A (m x k) and B (n x k) int8 and the coin alone."""
    key = call_key(sigma, c)
    m = A.shape[0]
    U = u_noise64(key, m)[tm * TM:(tm + 1) * TM]
    F = f_rows(key, tn * BN, BN, k)
    Ar = A[tm * TM:(tm + 1) * TM].astype(np.int64)
    Bp = B[tn * BN:(tn + 1) * BN].astype(np.int64) + F
    Ap = Ar + np.tile(U, (1, k // 32))
    T = np.zeros((TM, BN), np.int64)
    Ts = []
    for s in range(k // 32):
        T += Ap[:, 32 * s:32 * s + 32] @ Bp[:, 32 * s:32 * s + 32].T
        Ts.append((T & 0xFFFFFFFF).astype(np.uint32))
    tiles_n = B.shape[0] // BN
    return tile_receipts(receipt_seed(sigma, c), Ts, tm * tiles_n + tn), Ar @ Bp.T, U, F
