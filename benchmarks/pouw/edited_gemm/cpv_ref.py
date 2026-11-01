"""cpv.cu's gate judged on the CPU: each dumped unit tile recomputed from P, V and the per-call key alone, through
checked_strip's own construction (`benchmarks/pouw/theory/checked_strip.py`'s `pv_words`, `pv_served` and `digits`).

From the key (meta.json's four big-endian words): c (tag 8, NS 1's patch layout over P, H s x s), b (tag 7, the same
layout over V^T, kvh 128 x s), U (tag 12: P_A's row i = (h nB + B) s + q, block i / 2, digest byte (i % 2) 32 + rho,
U = -64 + beta @ (1, 2, 4, 8, 16, 32, 65)) and z (tag 13: unit u = kvh nB + B, block (8 u + b_l) 8 + ch / 16, z(rho, ch)
bit (ch % 16) 32 + rho). A tile (h, qt, B) is 128 queries 128 qt .. by 128 channels over key block B; its warp row w is
the 32-query block whose unit runs to b_l = 7 (a full unit) or 4 (qt % 2) + w (the diagonal unit, B = qt // 2), with
M the causal mask. For each warp row, in Lean coordinates (P the queries' rows, V the keys x channels):
    (T, D, O') = pv_words(P, V, U, c, b, z, 32, 32, m, M),  (O, S, 2 served) = pv_served(O', D, X, M)
and the device's bytes must equal: P~ = M (P + c + U), X = V + b, P_A = [-U | U | -U M_d | 127 M_d] (M_d the diagonal
block's mask), DG's [A_0 | A_1 | X_diag | 0] and Sb for every b_l, the served O (pv_l7qs_i), f = float32(2 served)
s_q s_ch (pv_l7qs) and the warp row's receipts (derive.py's tile_receipts over T_0 .. T_{b_l}, O', D_1 where b_l >= 2).
One JSON line a gate directory; exit 1 on any difference, on a device mismatch or on a control the device didn't reject.
    uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python3 benchmarks/pouw/edited_gemm/cpv_ref.py GATE_DIR...
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from cgemm_ref import C_TAG, DITHER_TAG, NCT, block, cs, tile_receipts

PU_TAG, PZ_TAG = 12, 13


def patch_cols(key: bytes, tag: int, row0: int, rows: int, k: int, col0: int, ncols: int) -> np.ndarray:
    """cgemm_ref.patch_bits' bits for columns col0 .. col0 + ncols only (128-aligned)."""
    kb = k // 128
    out = np.zeros((rows, ncols), np.int64)
    for g in range(row0 // 4, (row0 + rows) // 4):
        for p in range(col0 // 128, (col0 + ncols) // 128):
            bits = np.unpackbits(np.frombuffer(block(key, tag, g * kb + p), np.uint8), bitorder="little")
            out[4 * g - row0:4 * g - row0 + 4, 128 * p - col0:128 * p - col0 + 128] = bits.reshape(4, 128)
    return out


def u_pa(key: bytes, row0: int, rows: int) -> np.ndarray:
    v = np.stack([np.frombuffer(block(key, PU_TAG, i // 2), np.uint8)[32 * (i % 2):32 * (i % 2) + 32]
                  for i in range(row0, row0 + rows)])
    return -64 + np.unpackbits(v[..., None], axis=-1, bitorder="little")[..., :7] @ cs.W_RE


def z_unit(key: bytes, unit: int, bl: int) -> np.ndarray:
    """z for one (unit, b_l): 32 x 128 (rho, channel)."""
    z = np.zeros((32, 128), np.int64)
    for j in range(8):
        bits = np.unpackbits(np.frombuffer(block(key, PZ_TAG, (8 * unit + bl) * 8 + j), np.uint8), bitorder="little")
        z[:, 16 * j:16 * j + 16] = bits.reshape(16, 32).T
    return z


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    s, nB, seed = meta["s"], meta["nB"], meta["seed"]
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    out = {"dir": str(d), "s": s, "H": meta["H"], "rep": meta["rep"], "ntiles": meta["ntiles"], "device_ok": meta["device_ok"],
           "control_rejected": meta["control_rejected"], "device": meta["device"], "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"])
    ctid = np.arange(NCT)
    for t, h, qt, B, kvh, diag in meta["tiles"]:
        nk = 128 * ((qt & 1) + 1) if diag else 256
        unit = kvh * nB + B
        def rd(name: str, dt, t=t) -> np.ndarray:
            return np.fromfile(d / f"{name}_{t}.bin", dt)

        P = rd("P", np.int8).astype(np.int64).reshape(128, nk)
        V = rd("V", np.int8).astype(np.int64).reshape(128, 256)
        c = patch_cols(key, C_TAG, h * s + 128 * qt, 128, s, 256 * B, nk)
        b = patch_cols(key, DITHER_TAG, kvh * 128, 128, s, 256 * B, 256)
        X = V + b
        U = u_pa(key, (h * nB + B) * s + 128 * qt, 128)
        qrel = 128 * qt + np.arange(128) - 256 * B
        M = (np.arange(nk)[None, :] <= qrel[:, None]).astype(np.int64)
        Md = (np.arange(32)[None, :] <= (qrel[:, None] % 32)).astype(np.int64) if diag else np.ones((128, 32), np.int64)
        Pt = M * (P + c + U[:, np.arange(nk) % 32])
        PA = np.concatenate([-U, U, -U * Md, 127 * Md], axis=1)
        DG = np.zeros((8, 128, 128), np.int64)
        Sb = np.zeros((9, 128), np.int64)
        zs = {}
        for bl in range(8):
            Y = X[:, :32 * bl].reshape(128, bl, 32).sum(1).T if bl else np.zeros((32, 128), np.int64)
            m = 1 if bl <= 1 else 2
            zs[bl] = z_unit(key, unit, bl)[None] if m == 2 else np.zeros((0, 32, 128), np.int64)
            A = cs.digits(Y, zs[bl], m)
            DG[bl, :, :32] = A[0].T
            DG[bl, :, 32:64] = A[1].T if m == 2 else 0
            DG[bl, :, 64:96] = X[:, 32 * bl:32 * bl + 32]
            Sb[bl] = X[:, :32 * bl].sum(1)
        Sb[8] = X.sum(1)
        sp, sv = rd("sp", np.float32), rd("sv", np.float32)
        Od = rd("O", np.int32).reshape(128, 128).astype(np.int64)
        Fd = rd("F", np.float32).reshape(128, 128)
        Rd = rd("R", np.uint32).reshape(NCT, 8)
        res = {
            "kind": "diagonal" if diag else "full", "h": h, "qt": qt, "B": B,
            "Pt": bool((rd("Pt", np.int8).reshape(128, nk) == Pt).all()),
            "Pt_range": [int(Pt.min()), int(Pt.max())],
            "X": bool((rd("X", np.int8).reshape(128, 256) == X).all()),
            "PA": bool((rd("PA", np.int8).reshape(128, 128) == PA).all()),
            "DG": bool((rd("DG", np.int8).reshape(8, 128, 128) == DG).all()),
            "DG_s8": bool(-128 <= int(DG.min()) and int(DG.max()) <= 127),
            "Sb": bool((rd("Sb", np.int32).reshape(9, 128) == Sb).all()),
        }
        o_ok = f_ok = r_ok = served_ok = True
        d1 = []
        for w in range(4):
            bl = 4 * (qt & 1) + w if diag else 7
            ku, rows = 32 * (bl + 1), slice(32 * w, 32 * w + 32)
            m = 1 if bl <= 1 else 2
            Mu = M[rows, :ku]
            Vu, bu = V[:, :ku].T, b[:, :ku].T
            T, D, Op = cs.pv_words(P[rows, :ku], Vu, U[rows], c[rows, :ku], bu, zs[bl], 32, 32, m, Mu)
            O, _, served2 = cs.pv_served(Op, D, Vu + bu, Mu)
            served_ok &= bool((O == (Mu * (P[rows, :ku] + c[rows, :ku])) @ (Vu + bu)).all())
            o_ok &= bool((Od[rows] == O).all())
            want_f = np.float32(served2).astype(np.float32) * (sp[rows, None] * sv[None, :])
            f_ok &= bool((Fd[rows].view(np.uint32) == want_f.astype(np.float32).view(np.uint32)).all())
            words = []
            for wd in list(T) + [Op] + list(D):
                full = np.zeros((128, 128), np.uint32)
                full[rows] = cs.s32(wd)
                words.append(full)
            sel = (ctid // 32) // 4 == w
            r_ok &= bool((tile_receipts(seed, words, t)[sel] == Rd[sel]).all())
            if D:
                d1.append(float((D[0] != 0).mean()))
        res.update({"O_is_served": served_ok, "O": o_ok, "F": f_ok, "R": r_ok, "D1_nonzero": d1})
        out["tiles"][str(t)] = res
        ok &= all(v for v in res.values() if isinstance(v, bool))
        ok &= -128 <= res["Pt_range"][0] and res["Pt_range"][1] <= 127
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
