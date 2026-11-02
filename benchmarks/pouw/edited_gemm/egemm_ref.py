"""egemm.cu's gate judged on the CPU: each dumped tile recomputed from x (bf16), W and the per-call key alone.

From the key (meta.json's four big-endian words, 32 bytes): b, the dither bits (tag 7, sha_noise.cuh's NS 1 patch
layout), and U (tag 6, lpgemm.cu's LP_UW 64, derive.py's u_noise64 on the weights' rows). Then
    X  = clamp(floor(x inv_s), -127, 126) + b            Xq = clamp(rint(x inv_s), -127, 127)   (the plain quantizer)
    S  = X Id (S_ic = sum_t X_{i, 32 t + c}),  P_A = [S_1 | S_2 | S_0 | 0],  S = 256 (S_1 + S_2) + S_0, S_0 the low byte
         as s8, S_1 the rest clamped to ±127
    W' = W + U Id^T,  P_B = [U | U | U | 0]
    Y  = X W^T (the served product), and C - S U^T == Y for C = X W'^T
    receipts over C's running sums T_s (s = 1 .. k / 32), each dumped kernel's by its kind (meta.json's kinds):
      duo32  derive.py's tile_receipts over every T_s;  duo64  the same over T_2, T_4, ... (the odd steps, 0-based)
      xin    the XOR-in fold (egemm.cu's XIN), tile t's words into chain t % 4;  xin64  the same over T_2, T_4, ...
      xin13  the r9 twin itself (receipt_folds_r9.py, pinned by its sha256 below): its xorin_step on
             the same words and chains, each chain from pfill's y = pos = (tile NCT + ctid) 4 + c, keys the first two
             words that pass its key_ok_r9
      tb4, tb2w, tb1w, tb4c  floored_gemm/ref.py's tb_receipts and tbc_receipts on the tile
and every one is compared with the device's bytes, as are the device's key_ok_r9 answers on its boundary words
(r9_words.bin, r9_ok.bin). One JSON line a gate directory; exit 1 on any difference, on a device mismatch or on a
control the device didn't reject.
    python3 egemm_ref.py GATE_DIR...
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
from derive import BN, NCT, U_TAG, WM, mix32, tile_receipts  # noqa: E402

_spec = importlib.util.spec_from_file_location("fgemm_ref", HERE.parent / "floored_gemm" / "ref.py")
fref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fref)

RF_SHA256 = "c5a3424f90850ca818eff182f43132ae72cf21aadcaacb99758ca454932f5bb4"   # the r9 reference's sha256
_rf_path = HERE / "receipt_folds_r9.py"
if hashlib.sha256(_rf_path.read_bytes()).hexdigest() != RF_SHA256:
    sys.exit(f"{_rf_path} is not P2-FF r9's reference (sha256 {RF_SHA256}): restore it from art:91c156a1…, don't edit it")
_spec = importlib.util.spec_from_file_location("receipt_folds_r9", _rf_path)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

DITHER_TAG = 7
T = 128
TB_KIND = {"tb4": 5, "tb4c": 8, "tb2w": 10, "tb1w": 11}   # fgemm.cu's Kind


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


def limbs(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0 = ((S & 0xFF) ^ 0x80) - 0x80
    sh = (S - s0) // 256
    s1 = np.clip(sh, -127, 127)
    return s1, sh - s1, s0


def rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def xin_keys(seed: int, r9: bool = False) -> tuple[int, int, int]:
    """egemm.cu's xin_keys: the first two words mix32(seed ^ (0x5A5A0000 + i)) that pass pfill.cuh's key rule (r9:
    receipt_folds_r9.key_ok_r9), and how many words passing the key rule r9 skipped before them."""
    out, skipped, i = [], 0, 0
    while len(out) < 2:
        a = int(mix32(np.uint32((seed ^ (0x5A5A0000 + i)) & 0xFFFFFFFF)))
        if rf.key_ok_r9(a) if r9 else rf.key_ok(a):
            out.append(a)
        elif r9 and rf.key_ok(a):
            skipped += 1
        i += 1
    return out[0], out[1], skipped


def frag_words(Tw: np.ndarray, mi: int, ni: int) -> np.ndarray:
    """Every thread's accumulator words of tile (mi, ni), [NCT, 4]: (g, 2q), (g, 2q + 1), (g + 8, 2q), (g + 8, 2q + 1)."""
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, q = warp // (BN // 32), warp % (BN // 32), lane >> 2, lane & 3
    e = np.arange(4)
    rows = wm[:, None] * WM + 16 * mi + g[:, None] + 8 * (e >> 1)[None, :]
    cols = wn[:, None] * 32 + 8 * ni + 2 * q[:, None] + (e & 1)[None, :]
    return Tw[rows, cols]


def xin_receipts(seed: int, Ts: list[np.ndarray], tile: int) -> np.ndarray:
    """XIN's receipts, (NCT, 8): chains 0..3 (y = mix32(seed ^ mix32((tile NCT + ctid) 8 + c))), then four zeros."""
    a0, a1 = np.uint32(xin_keys(seed)[0]), np.uint32(xin_keys(seed)[1])
    idx = ((tile * NCT + np.arange(NCT)[:, None]) * 8 + np.arange(4)[None, :]).astype(np.uint32)
    y = mix32(np.uint32(seed) ^ mix32(idx))
    MI, NI = WM // 16, 4
    with np.errstate(over="ignore"):
        for Tw in Ts:
            for t in range(MI * NI):
                p = frag_words(Tw, t // NI, t % NI)
                c = t % 4
                y[:, c] = (rotl(y[:, c], 7) ^ p[:, 0]) * a0 + p[:, 1]
                y[:, c] = (rotl(y[:, c], 19) ^ p[:, 2]) * a1 + p[:, 3]
    out = np.zeros((NCT, 8), np.uint32)
    out[:, :4] = y
    return out


def xin13_receipts(seed: int, Ts: list[np.ndarray], tile: int) -> np.ndarray:
    """XIN13's receipts, (NCT, 8): receipt_folds_r9.xorin_step over the same words and chains as xin_receipts, chain c
    of thread ctid from y = (tile NCT + ctid) 4 + c, keys under key_ok_r9; then four zeros."""
    a0, a1, _ = xin_keys(seed, r9=True)
    y = ((tile * NCT + np.arange(NCT)[:, None]) * 4 + np.arange(4)[None, :]).astype(np.uint32)
    MI, NI = WM // 16, 4
    with np.errstate(over="ignore"):
        for Tw in Ts:
            for t in range(MI * NI):
                p = frag_words(Tw, t // NI, t % NI)
                c = t % 4
                y[:, c] = rf.xorin_step(y[:, c], p[:, 0], p[:, 1], p[:, 2], p[:, 3], np.uint32(a0), np.uint32(a1))
    out = np.zeros((NCT, 8), np.uint32)
    out[:, :4] = y
    return out


def r9_probe(d: Path) -> dict:
    """The device's key_ok_r9 answers on egemm.cu's r9_words against receipt_folds_r9.key_ok_r9's."""
    w = np.fromfile(d / "r9_words.bin", np.uint32)
    dev = np.fromfile(d / "r9_ok.bin", np.uint8).astype(bool)
    ref = np.array([rf.key_ok_r9(int(a)) for a in w])
    kok = np.array([rf.key_ok(int(a)) for a in w])
    return {"n": int(w.size), "key_ok": int(kok.sum()), "r9_rejected": int((kok & ~ref).sum()),
            "mismatch": int((dev != ref).sum()) if dev.size == w.size else -1}


def receipts(kind: str, seed: int, Ts: list[np.ndarray], tile: int) -> np.ndarray:
    if kind == "duo32":
        return tile_receipts(seed, Ts, tile)
    if kind == "duo64":
        return tile_receipts(seed, Ts[1::2], tile)
    if kind == "xin":
        return xin_receipts(seed, Ts, tile)
    if kind == "xin64":
        return xin_receipts(seed, Ts[1::2], tile)
    if kind == "xin13":
        return xin13_receipts(seed, Ts, tile)
    kid = TB_KIND[kind]
    if kid in fref.CHAINED:
        return fref.tbc_receipts(kid, Ts, T, BN, WM, NCT, 1, 1)[0]
    SB = sum(fref.u8bytes(Tw) for Tw in Ts)
    return fref.tb_receipts(kid, SB, T, BN, WM, NCT, 1, 1)[0]


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    kinds = meta.get("kinds", {})
    out = {"dir": str(d), "m": m, "n": n, "k": k, "device_ok": meta["device_ok"], "control_rejected": meta["control_rejected"],
           "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"]) and bool(kinds)
    out["r9_probe"] = r9_probe(d)
    out["r9_seed_skips"] = xin_keys(seed, r9=True)[2]
    ok &= out["r9_probe"]["mismatch"] == 0 and out["r9_probe"]["r9_rejected"] > 0
    z = np.zeros((T, 32), np.int64)
    for tm, tn in meta["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        rd = lambda name, dt: np.fromfile(d / (name + sfx), dt)  # noqa: E731
        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        f = x * inv_s
        X = np.clip(np.floor(f), -127, 126).astype(np.int64) + dither_rows(key, tm * T, T, k)
        Xq = np.clip(np.rint(f), -127, 127).astype(np.int64)
        S = X.reshape(T, k // 32, 32).sum(axis=1)
        s1, s2, s0 = limbs(S)
        PA = np.concatenate([s1, s2, s0, z], axis=1)
        W = rd("W", np.int8).astype(np.int64).reshape(T, k)
        U = u_rows(key, tn * T, T)
        Wp = W + np.tile(U, (1, k // 32))
        PB = np.concatenate([U, U, U, z], axis=1)
        Y = X @ W.T
        C = np.zeros((T, T), np.int64)
        Ts = []
        for s in range(k // 32):
            C += X[:, 32 * s:32 * s + 32] @ Wp[:, 32 * s:32 * s + 32].T
            Ts.append((C & 0xFFFFFFFF).astype(np.uint32))
        res = {
            "X": bool((rd("X", np.int8).reshape(T, k) == X).all()),
            "X_range": [int(X.min()), int(X.max())],
            "Xq": bool((rd("Xq", np.int8).reshape(T, k) == Xq).all()),
            "PA": bool((rd("PA", np.int8).reshape(T, 128) == PA).all()),
            "PA_s8": bool(-128 <= int(PA.min()) and int(PA.max()) <= 127),
            "Wp": bool((rd("Wp", np.int8).reshape(T, k) == Wp).all()),
            "Wp_range": [int(Wp.min()), int(Wp.max())],
            "PB": bool((rd("PB", np.int8).reshape(T, 128) == PB).all()),
            "peel": bool(((C - S @ U.T) == Y).all()),
            "Y": bool((rd("Y", np.int32).reshape(T, T).astype(np.int64) == Y).all()),
            "S_range": [int(S.min()), int(S.max())],
        }
        want = {kd: receipts(kd, seed, Ts, tm * tiles_n + tn) for kd in sorted(set(kinds.values()))}
        res["R"] = {name: bool((rd("R_" + name, np.uint32).reshape(NCT, 8) == want[kd]).all()) for name, kd in kinds.items()}
        out["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool)) and all(res["R"].values())
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
