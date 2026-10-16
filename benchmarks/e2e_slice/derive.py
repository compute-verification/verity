"""The slice's derivations and its CPU reference, shared by the prover, the challenger and the gate.

From a stage's coin sigma (64 bytes from the challenger's OS generator), a call's index c and the stage s (0 or 1; "stage" s
appended past stage 0):
    key_c   = SHA-512("verity/e2e-slice/call-sigma/v1\\0" || sigma || BE32(c) [|| "stage" s])[:32]: the call's SHA-512 noise key
              (lpgemm.cu's Sig)
    rseed_c = BE32 of SHA-512("verity/e2e-slice/receipt-key/v1\\0" || sigma || BE32(c) [|| "stage" s])[:4]: the duo receipts' key
    U       = lpgemm.cu's U at LP_UW 64 under key_c (m x 32, {±1..±64}): block j = rows 2j, 2j + 1, entry e = (row % 2) 32 + l
              digest byte e, u = (v & 63) + 1, negated if v & 64
    F       = sha_noise.cuh's pm1 noise of B's side under key_c (n x k, ±1)
A' = A + U W^T (A'_ik = A_ik + U_{i, k mod 32}), B' = B + F, or under decision 29's dither B' = B + (1 - F) / 2 (the served
X = floor(x / s) + b, b the coin bit); the duo receipts at L = 32 over A' B'^T (mref.py's), or DUOS's word a thread and
segment (fgemm.cu, verity/core/protocols/pouw/circuit/segment.py), one tile at a time. The two-stage slice's second stage
reads phi of the first's served product.
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


def _stage(stage: int) -> bytes:
    return b"" if stage == 0 else b"stage" + bytes([stage])


def call_key(sigma: bytes, c: int, stage: int = 0) -> bytes:
    return hashlib.sha512(b"verity/e2e-slice/call-sigma/v1\0" + sigma + c.to_bytes(4, "big") + _stage(stage)).digest()[:32]


def receipt_seed(sigma: bytes, c: int, stage: int = 0) -> int:
    h = hashlib.sha512(b"verity/e2e-slice/receipt-key/v1\0" + sigma + c.to_bytes(4, "big") + _stage(stage)).digest()
    return int.from_bytes(h[:4], "big")


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


#: the two-stage slice's public map from stage 1's served product to stage 2's A: clamp(floor(y / 2^PHI_SHIFT), -63, 63)
PHI_SHIFT = 12


def phi(Y: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(Y, np.int64) >> PHI_SHIFT, -63, 63).astype(np.int8)


def thread_entries() -> tuple[np.ndarray, np.ndarray]:
    """Every thread's 32 entries' (row, col) in its tile, (NCT, 32) each: entry q = 4 t + e (segment.py's row_slot and
    col_slot)."""
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, qq = warp // (BN // 32), warp % (BN // 32), lane >> 2, lane & 3
    q = np.arange(32)
    t, e = q // 4, q % 4
    r = 2 * (t // 4) + (e >> 1)
    c = 2 * (t % 4) + (e & 1)
    rows = wm[:, None] * WM + g[:, None] + 8 * r[None, :]
    cols = wn[:, None] * 32 + 2 * qq[:, None] + 8 * (c[None, :] >> 1) + (c[None, :] & 1)
    return rows, cols


def thread_slots(ctid: int) -> tuple[np.ndarray, np.ndarray]:
    """Thread ``ctid``'s 4 rows (row slot r) and 8 columns (column slot c) in its tile."""
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, qq = warp // (BN // 32), warp % (BN // 32), lane >> 2, lane & 3
    c = np.arange(8)
    return wm * WM + g + 8 * np.arange(4), wn * 32 + 2 * qq + 8 * (c >> 1) + (c & 1)


def seg_keys(seed: int, tile: int, ks: int, segs=None, threads=None) -> tuple[np.ndarray, np.ndarray]:
    """The DUOS chains' keys (y, z) at the given segments and threads, (segs, threads, 8) u32 (segment.init)."""
    segs = np.arange(ks) if segs is None else np.asarray(segs)
    threads = np.arange(NCT) if threads is None else np.asarray(threads)
    with np.errstate(over="ignore"):
        idx = (((np.uint32(tile) * np.uint32(NCT) + threads[None, :, None].astype(np.uint32)) * np.uint32(ks)
                + segs[:, None, None].astype(np.uint32)) * np.uint32(8) + np.arange(8, dtype=np.uint32)[None, None, :])
        y = mix32(np.uint32(seed) ^ mix32(idx))
        return y, mix32(y + np.uint32(0x9E3779B9))


def seg_receipts(seed: int, Ts: np.ndarray, tile: int) -> np.ndarray:
    """fgemm.cu's DUOS receipts of one tile, (KS, NCT) u32: Ts (KS, 128, 128) the tile's u32 prefix words after each
    32-step segment."""
    ks = len(Ts)
    rows, cols = thread_entries()
    y, z = seg_keys(seed, tile, ks)
    P = Ts[:, rows, cols]
    with np.errstate(over="ignore"):
        for t in range(8):
            h = 4 * (t % 2)
            u = y[:, :, h:h + 4] ^ P[:, :, 4 * t:4 * t + 4] ^ np.uint32(key_xor(0, t))
            y[:, :, h:h + 4] = u * np.uint32(key_mul(0, t)) + z[:, :, h:h + 4]
            z[:, :, h:h + 4] = rotl(u, rot_of(False, 0, t)) + y[:, :, h:h + 4]
        return np.bitwise_xor.reduce(y ^ rotl(z, 16), axis=2)


def win_receipts(seed: int, Ts: np.ndarray, tile: int, S: int, fold: str = "duo") -> np.ndarray:
    """One tile's checkpointed receipts, (NCT, KS / S) u32 (`segment.PouwSegCheck_v2{S, FOLD}`): window w's word folds, from
    the keys of (tile, thread, w, KS / S), T_(w S - 1) (0 for w = 0) and T_(w S) .. T_(w S + S - 1). Ts (KS, 128, 128)."""
    from verity.core.protocols.pouw.circuit import segment as SG

    ks = len(Ts)
    if ks % S:
        raise ValueError(f"{ks} segments: not whole windows of {S}")
    nw = ks // S
    rows, cols = thread_entries()
    y, z = seg_keys(seed, tile, nw)
    Te = np.concatenate([np.zeros((1, *Ts.shape[1:]), np.uint32), np.asarray(Ts, np.uint32)])
    with np.errstate(over="ignore"):
        for i in range(S + 1):
            P = Te[np.arange(nw) * S + i][:, rows, cols]
            if fold == "duo":
                for t in range(8):
                    h = 4 * (t % 2)
                    u = y[:, :, h:h + 4] ^ P[:, :, 4 * t:4 * t + 4] ^ np.uint32(SG.KK[t])
                    y[:, :, h:h + 4] = u * np.uint32(SG.KA[t]) + z[:, :, h:h + 4]
                    z[:, :, h:h + 4] = rotl(u, SG.ROT[t]) + y[:, :, h:h + 4]
            else:
                for t in range(8):
                    d = P[:, :, 4 * t:4 * t + 4]
                    y[:, :, t] = (rotl(y[:, :, t], SG.XROT[0]) ^ d[..., 0]) * np.uint32(SG.XA0[t]) + d[..., 1]
                    y[:, :, t] = (rotl(y[:, :, t], SG.XROT[1]) ^ d[..., 2]) * np.uint32(SG.XA1[t]) + d[..., 3]
        R = np.bitwise_xor.reduce(y ^ rotl(z, 16) if fold == "duo" else y, axis=2)
    return np.ascontiguousarray(R.T)


def served_noise(F: np.ndarray, dither: bool) -> np.ndarray:
    """B's noise from F's coin bits: F itself, or decision 29's dither b = (1 - F) / 2 in {0, 1} (lpgemm.cu's fb_word)."""
    return (1 - F) // 2 if dither else F


def tile_noised(A: np.ndarray, B: np.ndarray, key: bytes, tm: int, tn: int, k: int, dither: bool = False):
    """The tile (tm, tn)'s A rows, U rows, B' rows and F rows (±1; under ``dither`` B' = B + (1 - F) / 2), and its u32
    prefix words after each segment (KS, 128, 128)."""
    U = u_noise64(key, A.shape[0])[tm * TM:(tm + 1) * TM]
    F = f_rows(key, tn * BN, BN, k)
    Ar = A[tm * TM:(tm + 1) * TM].astype(np.int64)
    Bp = B[tn * BN:(tn + 1) * BN].astype(np.int64) + served_noise(F, dither)
    Ap = Ar + np.tile(U, (1, k // 32))
    inc = np.einsum("isk,jsk->sij", Ap.reshape(TM, k // 32, 32), Bp.reshape(BN, k // 32, 32))
    Ts = (np.cumsum(inc, axis=0) & 0xFFFFFFFF).astype(np.uint32)
    return Ar, U, Bp, F, Ts


def tile_check(A: np.ndarray, B: np.ndarray, sigma: bytes, c: int, tm: int, tn: int, k: int, seg: bool = False,
               key: bytes | None = None, rseed: int | None = None, dither: bool = False):
    """The tile (tm, tn)'s duo receipts and served product A B'^T from A (m x k) and B (n x k) int8 and the coin alone:
    the receipts over the whole of K (NCT, 8), or with ``seg`` a word a segment (KS, NCT). ``key`` and ``rseed`` replace
    the call's noise key and receipt key; ``dither``: B's noise is the dither."""
    key = call_key(sigma, c) if key is None else key
    rseed = receipt_seed(sigma, c) if rseed is None else rseed
    Ar, U, Bp, F, Ts = tile_noised(A, B, key, tm, tn, k, dither)
    tile = tm * (B.shape[0] // BN) + tn
    R = seg_receipts(rseed, Ts, tile) if seg else tile_receipts(rseed, list(Ts), tile)
    return R, Ar @ Bp.T, U, F
