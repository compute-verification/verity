"""cgemm.cu's gate judged on the CPU: each dumped tile recomputed from x (bf16), W and the per-call key alone, through
checked_strip's own construction (`benchmarks/pouw/theory/checked_strip.py`'s `co32c_words` and `digits`).

From the key (meta.json's four big-endian words): b (tag 7, egemm_ref.py's dither_rows), c (tag 8, the same patch
layout over W), beta (tag 10: block j = rows 2j, 2j + 1, entry (row % 2) 32 + r digest byte v, beta_p = bit p of v, so
U = -64 + beta @ (1, 2, 4, 8, 16, 32, 65)) and z (tag 9: block j = rows 4j .. 4j + 3, z_d(i, r) = bit 32 d + r of digest
bytes 16 (i % 4) .. 16 (i % 4) + 15, little-endian). Then, in Lean coordinates (P the weights, n x k; V the activations
transposed, k x n'):
    V = clamp(floor(x inv_s), -127, 126)^T,  X = V + b,  P^ = W + c,  W' = P^ + U Id^T,  P_B = [-U | U | U | 0]
    Y = residue sums of X,  A = digits(Y, z, m),  P_A = [A_0 | A_1 | A_2 | 0]^T (A_2 = 0 at m = 2)
    (T, D, O') = co32c_words(W, V, U, c, b, z, 32, 32, m),  O = O' - sum_d 128^d D_d  (== P^ X, checked)
and the device's bytes must equal: X, P_A, W', P_B, the served O (c_l7qs), O' (c_l7qs_o) and the receipts (derive.py's
tile_receipts over T_1 .. T_last, O', D_1 .. D_{m-1}). c_fx1's O must equal it too, and its receipts tile_receipts over the
same words each multiplied first by fx1's key (egemm.cu's fx_chain<FX1, t, e>: word e of tile position t times
key_mul(1, (4 t + e) 32) mod 2^32, recomputed here from the key schedule); the receipts without the multiply, and with it
dropped at one (t, e), must both differ from the device's. A cohalf gate (QK^T at k = 128, meta's kinds "cohalf") is
check_half's: X = clamp(floor(x inv_s), -64, 62)^T + b (meta's inv_s the 7-bit scale), Y = X's two 64-halves summed, U from tag 11 (row i's digest
byte r), and (T, O) = honest_words(W, V, U, c, b, 64, 32) with O = T_last - U Y. One JSON line a gate directory; exit 1
on any difference, on a device mismatch or on a control the device didn't reject.
    uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python3 benchmarks/pouw/edited_gemm/cgemm_ref.py GATE_DIR...
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e2e_slice"))
from derive import NCT, tile_receipts
from ref import key_mul  # noqa: E402  (floored_gemm's, on derive's path)

_spec = importlib.util.spec_from_file_location("checked_strip", HERE.parent / "theory" / "checked_strip.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

T = 128
DITHER_TAG, C_TAG, Z_TAG, BETA_TAG, HBETA_TAG = 7, 8, 9, 10, 11


def block(key: bytes, tag: int, j: int) -> bytes:
    return hashlib.sha512(key + struct.pack(">Q", tag << 56 | j)).digest()


def patch_bits(key: bytes, tag: int, row0: int, rows: int, k: int) -> np.ndarray:
    """Bits in sha_noise.cuh's NS 1 patch layout: block (row / 4) (k / 128) + col / 128, entry (row % 4) 128 + col % 128."""
    kb = k // 128
    b = np.zeros((rows, k), np.int64)
    for g in range(row0 // 4, (row0 + rows) // 4):
        for p in range(kb):
            bits = np.unpackbits(np.frombuffer(block(key, tag, g * kb + p), np.uint8), bitorder="little")
            b[4 * g - row0:4 * g - row0 + 4, 128 * p:128 * p + 128] = bits.reshape(4, 128)
    return b


def u_rows(key: bytes, row0: int, rows: int) -> np.ndarray:
    beta = np.zeros((rows, 32, 7), np.int64)
    for j in range(row0 // 2, (row0 + rows) // 2):
        v = np.frombuffer(block(key, BETA_TAG, j), np.uint8).reshape(2, 32)
        beta[2 * j - row0:2 * j - row0 + 2] = np.unpackbits(v[..., None], axis=-1, bitorder="little")[..., :7]
    return -64 + beta @ cs.W_RE


#: H1's narrow U planes (co32c-u16): narrow_u.planes(16), offset 16, so U in {-16..-1, 1..16}
W_RE16 = np.array([1, 2, 4, 8, 17], np.int64)


def u_rows16(key: bytes, row0: int, rows: int) -> np.ndarray:
    """H1's U (cgemm.cu's u_beta16, same BETA_TAG layout as u_rows but five planes): -16 + beta[:5] @ (1, 2, 4, 8, 17)."""
    beta = np.zeros((rows, 32, 5), np.int64)
    for j in range(row0 // 2, (row0 + rows) // 2):
        v = np.frombuffer(block(key, BETA_TAG, j), np.uint8).reshape(2, 32)
        beta[2 * j - row0:2 * j - row0 + 2] = np.unpackbits(v[..., None], axis=-1, bitorder="little")[..., :5]
    return -16 + beta @ W_RE16


def z_rows(key: bytes, row0: int, rows: int, m: int) -> np.ndarray:
    """z, (m - 1) x 32 x rows (Lean's (digit, r, column) order)."""
    z = np.zeros((rows, 128), np.int64)
    for j in range(row0 // 4, (row0 + rows) // 4):
        z[4 * j - row0:4 * j - row0 + 4] = np.unpackbits(np.frombuffer(block(key, Z_TAG, j), np.uint8), bitorder="little").reshape(4, 128)
    return z[:, :32 * (m - 1)].reshape(rows, m - 1, 32).transpose(1, 2, 0)


def fx1_keys(drop: tuple[int, int] | None = None) -> np.ndarray:
    """fx1's multipliers over a 128 x 128 tile in tile_receipts' (row, column) order: the word at row 32 wm + 16 mi + g +
    8 (e >> 1), column 32 wn + 8 ni + 2 q + (e & 1) is word e of tile position t = 4 mi + ni, multiplied by
    key_mul(1, (4 t + e) 32); drop = (t, e) leaves that one multiply out."""
    r, c = np.arange(T)[:, None], np.arange(T)[None, :]
    t = 4 * ((r % 32) // 16) + (c % 32) // 8
    e = 2 * ((r % 16) // 8) + (c & 1)
    K = np.array([key_mul(1, 32 * j) for j in range(32)], np.uint64)[4 * t + e]
    if drop is not None:
        K = np.where((t == drop[0]) & (e == drop[1]), np.uint64(1), K)
    return K


def fx1_words(words: list[np.ndarray], K: np.ndarray) -> list[np.ndarray]:
    return [((w.astype(np.uint64) * K) & cs.MASK).astype(np.uint32) for w in words]


def u_rows_half(key: bytes, row0: int, rows: int) -> np.ndarray:
    """cohalf's U, rows x 64: row i's tag-11 digest byte r gives beta(i, r) (bits 0..6)."""
    v = np.stack([np.frombuffer(block(key, HBETA_TAG, i), np.uint8) for i in range(row0, row0 + rows)])
    return -64 + np.unpackbits(v[..., None], axis=-1, bitorder="little")[..., :7] @ cs.W_RE


def u_rows16_half(key: bytes, row0: int, rows: int) -> np.ndarray:
    """H1 cohalf's narrow U, rows x 64: tag-11 per row (like u_rows_half, cgemm.cu's hwedit) but five planes (u_beta16)."""
    v = np.stack([np.frombuffer(block(key, HBETA_TAG, i), np.uint8) for i in range(row0, row0 + rows)])
    return -16 + np.unpackbits(v[..., None], axis=-1, bitorder="little")[..., :5] @ W_RE16


def check_half(d: Path, meta: dict) -> dict:
    """cohalf (QK^T, k = 128, r = 64): K (the served side, 7-bit) and Q (the peeled, in [-64, 62]); words T_0..T_3 and
    O = T_last - U Y (`checked_strip.honest_words` at r = 64, L = 32), absorbed in that order."""
    m_, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    # H1 (co32c-u16): Q/K per (head, token) asymmetric floor at top 111, s = max(max(x) / 111, max(-x) / 112)
    # clamped to 1e-30 then * (1 + 2^-20), X = floor(x / s) + b in [-112, 111] (matching hrow_s + hquant_h1 and
    # steps.floor_quant: one divide, no clamp); the narrow U (u_rows16_half); Y = X_lo + X_hi in [-224, 222] split as
    # Y0 = ((Y + 64) & 127) - 64, Y1 = (Y - Y0) >> 7 with P_A = [Y0 | Y1], P_B = [-U | -U]. Else Dither-C's 7-bit cohalf.
    h1 = meta.get("numerics") == "h1"
    km = next(iter(meta["kinds"]))   # the main kernel's name: h_l7qs (7-bit) or h_l7qs_h1 (H1)
    out = {"dir": str(d), "m": m_, "n": n, "k": k, "design": "cohalf", "numerics": "h1" if h1 else "co32c",
           "device_ok": meta["device_ok"], "control_rejected": meta["control_rejected"], "device": meta["device"], "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"]) and k == 128
    for tm, tn in meta["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        def rd(name: str, dt, sfx=sfx) -> np.ndarray:
            return np.fromfile(d / (name + sfx), dt)

        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        if h1:
            s = (np.maximum(np.maximum(x.max(axis=1) / np.float32(111.0), (-x).max(axis=1) / np.float32(112.0)),
                            np.float32(1e-30)) * np.float32(1.0 + 2.0 ** -20)).astype(np.float32)
            Vq = np.floor(x / s[:, None]).astype(np.int64)
            U = u_rows16_half(key, tn * T, T)
        else:
            Vq = np.clip(np.floor(x * inv_s), -64, 62).astype(np.int64)
            U = u_rows_half(key, tn * T, T)
        b = patch_bits(key, DITHER_TAG, tm * T, T, k)
        X = Vq + b
        W = rd("W", np.int8).astype(np.int64).reshape(T, k)
        c = patch_bits(key, C_TAG, tn * T, T, k)
        Wp = W + c + U[:, np.arange(k) % 64]
        Y = X[:, :64] + X[:, 64:]
        if h1:
            Y0 = ((Y + 64) & 127) - 64
            Y1 = (Y - Y0) >> 7
            PA = np.concatenate([Y0, Y1], axis=1)
            PB = np.concatenate([-U, -U], axis=1)
        else:
            zero = np.zeros((T, 32), np.int64)
            PA = np.concatenate([Y, zero, zero], axis=1)
            PB = np.concatenate([-U, zero, zero], axis=1)
        Ts, O = cs.honest_words(W, Vq.T, U, c, b.T, 64, 32)
        strip = (Ts[-1].astype(np.int64) - U @ Y.T) & cs.MASK
        want_r = tile_receipts(seed, [t.T for t in Ts] + [O.T], tm * tiles_n + tn)
        res = {
            "X": bool((rd("X", np.int8).reshape(T, k) == X).all()),
            "X_range": [int(X.min()), int(X.max())],
            "PA": bool((rd("PA", np.int8).reshape(T, 128) == PA).all()),
            "PA_s8": bool(-128 <= int(PA.min()) and int(PA.max()) <= 127),
            "Wp": bool((rd("Wp", np.int8).reshape(T, k) == Wp).all()),
            "Wp_range": [int(Wp.min()), int(Wp.max())],
            "PB": bool((rd("PB", np.int8).reshape(T, 128) == PB).all()),
            "O_is_strip": bool((strip == O.astype(np.int64)).all()),
            "O": bool((rd("Y_" + km, np.int32).reshape(T, T).view(np.uint32) == O.T).all()),
            "R": bool((rd("R_" + km, np.uint32).reshape(NCT, 8) == want_r).all()),
        }
        out["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool))
        xlo, xhi = (-112, 111) if h1 else (-64, 63)
        ok &= xlo <= res["X_range"][0] and res["X_range"][1] <= xhi and -128 <= res["Wp_range"][0] and res["Wp_range"][1] <= 127
    out["ok"] = bool(ok)
    return out


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    if "cohalf" in meta.get("kinds", {}).values():
        return check_half(d, meta)
    m_, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    # H1's numerics (co32c-u16): per-token s_a = amax / 126 (recomputed from x, fp32, matching crow_s), then
    # X = floor(clamp(x / s_a, -126, 126)) + b dividing by s_a (two roundings, as crow_s + cquant_h1 and steps.act_quant
    # do, not one reciprocal), weights +-110 with the narrow U (u_rows16). Else Dither-C's one-scale 7-bit form.
    h1 = meta.get("numerics") == "h1"
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    nd = cs.n_digits((k // 32) * -128, (k // 32) * 127)
    out = {"dir": str(d), "m": m_, "n": n, "k": k, "digits": nd, "numerics": "h1" if h1 else "co32c",
           "device_ok": meta["device_ok"], "control_rejected": meta["control_rejected"], "device": meta["device"], "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"])
    for tm, tn in meta["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        def rd(name: str, dt, sfx=sfx) -> np.ndarray:
            return np.fromfile(d / (name + sfx), dt)

        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        if h1:
            s_row = (np.maximum(np.abs(x).max(axis=1), np.float32(1e-30)) / np.float32(126.0)).astype(np.float32)
            Vq = np.clip(np.floor(x / s_row[:, None]), -126, 126).astype(np.int64)
        else:
            Vq = np.clip(np.floor(x * inv_s), -127, 126).astype(np.int64)
        b = patch_bits(key, DITHER_TAG, tm * T, T, k)
        X = Vq + b
        W = rd("W", np.int8).astype(np.int64).reshape(T, k)
        c = patch_bits(key, C_TAG, tn * T, T, k)
        U = u_rows16(key, tn * T, T) if h1 else u_rows(key, tn * T, T)
        z = z_rows(key, tm * T, T, nd)
        Wp = W + c + U[:, np.arange(k) % 32]
        Y = X.reshape(T, k // 32, 32).sum(1).T
        A = cs.digits(Y, z, nd)
        zero = np.zeros((T, 32), np.int64)
        PA = np.concatenate([A[0].T, A[1].T, A[2].T if nd > 2 else zero, zero], axis=1)
        PB = np.concatenate([-U, U, U, zero], axis=1)
        Ts, D, Op = cs.co32c_words(W, Vq.T, U, c, b.T, z, 32, 32, nd)
        O = (Op.astype(np.int64) - sum(dd.astype(np.int64) << (7 * (j + 1)) for j, dd in enumerate(D))) & cs.MASK
        served = (W + c) @ X.T
        words = [t.T for t in Ts] + [Op.T] + [dd.T for dd in D]
        want_r = tile_receipts(seed, words, tm * tiles_n + tn)
        r_fx = rd("R_c_fx1", np.uint32).reshape(NCT, 8)
        res = {
            "X": bool((rd("X", np.int8).reshape(T, k) == X).all()),
            "X_range": [int(X.min()), int(X.max())],
            "PA": bool((rd("PA", np.int8).reshape(T, 128) == PA).all()),
            "PA_s8": bool(-128 <= int(PA.min()) and int(PA.max()) <= 127),
            "Wp": bool((rd("Wp", np.int8).reshape(T, k) == Wp).all()),
            "Wp_range": [int(Wp.min()), int(Wp.max())],
            "PB": bool((rd("PB", np.int8).reshape(T, 128) == PB).all()),
            "O_is_served": bool(((O.astype(np.int64) ^ 0x80000000) - 0x80000000 == served).all()),
            "O": bool((rd("Y_c_l7qs", np.int32).reshape(T, T).view(np.uint32) == O.T.astype(np.uint32)).all()),
            "Op": bool((rd("Y_c_l7qs_o", np.int32).reshape(T, T).view(np.uint32) == Op.T).all()),
            "R": bool((rd("R_c_l7qs", np.uint32).reshape(NCT, 8) == want_r).all()),
            "O_fx1": bool((rd("Y_c_fx1", np.int32).reshape(T, T).view(np.uint32) == O.T.astype(np.uint32)).all()),
            "R_fx1": bool((r_fx == tile_receipts(seed, fx1_words(words, fx1_keys()), tm * tiles_n + tn)).all()),
            "R_fx1_rejects_duo": bool((r_fx != want_r).any()),
            "R_fx1_rejects_dropped_imad": bool(
                (r_fx != tile_receipts(seed, fx1_words(words, fx1_keys((7, 3))), tm * tiles_n + tn)).any()),
            "D1_nonzero": float((D[0] != 0).mean()),
            "Y_range": [int(Y.min()), int(Y.max())],
        }
        out["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool))
        if h1:   # X = floor(clamp(x / s_a, -126, 126)) + b in [-126, 127]; Wp = W + c + U in [-126, 127]
            ok &= -126 <= res["X_range"][0] and res["X_range"][1] <= 127 and -126 <= res["Wp_range"][0] and res["Wp_range"][1] <= 127
        else:
            ok &= -127 <= res["X_range"][0] and res["X_range"][1] <= 127 and -128 <= res["Wp_range"][0] and res["Wp_range"][1] <= 127
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
