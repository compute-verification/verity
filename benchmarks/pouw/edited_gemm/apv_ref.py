"""apv.cu's gate judged on the CPU: each dumped tile recomputed from P_prob, V and the per-call key alone, through
checked_strip's own construction (`benchmarks/pouw/theory/checked_strip.py`'s `pv_words` and `pv_served`).

From the key (meta.json's four big-endian words): b (tag 7, NS 1's patch layout over V^T, kvh 128 x s) and U (tag 14:
row q, block q / 2, digest byte (q % 2) 32 + rho, U = (byte & 63) + 1, negated where bit 6 is set; egemm_ref.u_rows' map).
A tile (h, qt, kvh) is 128 queries 128 qt .. by 128 channels over keys 0 .. 128 (qt + 1) - 1; its warp row w is the 32-query
block j = 4 qt + w with b_l = j, and M = [1 <= key <= q] (key 0 is its own P block). For each warp row, in Lean coordinates
(P = P_prob - 64, the queries' rows; V the keys x channels, key 0's row of V and b zeroed: the block is keys 1 .. q):
    (T, [], O) = pv_words(P, V, U, 0, b, -, 32, 32, 1, M),  (O, S, -) = pv_served(O, [], X, M)
and the device's bytes must equal: P~ = M (P_prob - 64 + U), X = V + b, U - 64, P_A = [-U | -U | -U M_d | 64 M_d] (M_d the
diagonal block's mask), DG's [Y0 | Y1 | X_diag | Y2] and Sq for each warp row's b_l (Ycum = Y0 + 256 Y1 + 65536 Y2), X(0), the served
O (apv_l7qs_i), f = float32((float32(O + 64 S) s_q + z_q float32(X(0))) s_ch) (apv_l7qs) and the warp row's receipts
(derive.py's tile_receipts over T_0 .. T_{b_l}, O). One JSON line a gate directory; exit 1 on any difference, on a device
mismatch or on a control the device didn't reject.
    uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python3 benchmarks/pouw/edited_gemm/apv_ref.py GATE_DIR...
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from cgemm_ref import DITHER_TAG, NCT, block, cs, tile_receipts
from cpv_ref import patch_cols

AU_TAG = 14


def u_rows(key: bytes, row0: int, rows: int) -> np.ndarray:
    U = np.zeros((rows, 32), np.int64)
    for i in range(rows):
        q = row0 + i
        b = np.frombuffer(block(key, AU_TAG, q // 2), np.uint8)[32 * (q % 2):32 * (q % 2) + 32].astype(np.int64)
        u = (b & 63) + 1
        U[i] = np.where(b & 64, -u, u)
    return U


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    s, seed = meta["s"], meta["seed"]
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    out = {"dir": str(d), "s": s, "H": meta["H"], "rep": meta["rep"], "ntiles": meta["ntiles"], "device_ok": meta["device_ok"],
           "control_rejected": meta["control_rejected"], "device": meta["device"], "tiles": {}}
    ok = bool(meta["device_ok"]) and bool(meta["control_rejected"])
    ctid = np.arange(NCT)
    for t, h, qt, kvh in meta["tiles"]:
        nk = 128 * (qt + 1)

        def rd(name: str, dt, t=t) -> np.ndarray:
            return np.fromfile(d / f"{name}_{t}.bin", dt)

        P = rd("P", np.int8).astype(np.int64).reshape(128, nk)
        V = rd("V", np.int8).astype(np.int64).reshape(128, nk)
        b = patch_cols(key, DITHER_TAG, kvh * 128, 128, s, 0, nk)
        X = V + b
        q = 128 * qt + np.arange(128)
        keys = np.arange(nk)
        U = u_rows(key, 128 * qt, 128)
        M = ((keys[None, :] <= q[:, None]) & (keys[None, :] >= 1)).astype(np.int64)
        rho = np.arange(32)
        Md = ((rho[None, :] <= (q % 32)[:, None]) & ~((q[:, None] < 32) & (rho[None, :] == 0))).astype(np.int64)
        Pt = M * (P - 64 + U[:, keys % 32])
        PA = np.concatenate([-U, -U, -U * Md, 64 * Md], axis=1)
        Xz, Vz, bz = X.copy(), V.copy(), b.copy()
        Xz[:, 0] = Vz[:, 0] = bz[:, 0] = 0
        DG = np.zeros((4, 128, 128), np.int64)
        Sq = np.zeros((4, 128), np.int64)
        for w in range(4):
            j = 4 * qt + w
            Y = Xz[:, :32 * j].reshape(128, j, 32).sum(1).T if j else np.zeros((32, 128), np.int64)
            Y0 = ((Y & 0xFF) ^ 0x80) - 0x80
            DG[w, :, :32] = Y0.T
            R1 = (Y - Y0) // 256
            Y1 = ((R1 & 0xFF) ^ 0x80) - 0x80
            DG[w, :, 32:64] = Y1.T
            DG[w, :, 96:128] = ((R1 - Y1) // 256).T
            DG[w, :, 64:96] = X[:, 32 * j:32 * j + 32]
            Sq[w] = Xz[:, :32 * j].sum(1)
        sp, sz, sv = rd("sp", np.float32), rd("sz", np.float32), rd("sv", np.float32)
        Od = rd("O", np.int32).reshape(128, 128).astype(np.int64)
        Fd = rd("F", np.float32).reshape(128, 128)
        Rd = rd("R", np.uint32).reshape(NCT, 8)
        res = {
            "h": h, "qt": qt, "kvh": kvh,
            "Pt": bool((rd("Pt", np.int8).reshape(128, nk) == Pt).all()),
            "Pt_range": [int(Pt.min()), int(Pt.max())],
            "X": bool((rd("X", np.int8).reshape(128, nk) == X).all()),
            "Um": bool((rd("Um", np.int8).reshape(128, 32) == U - 64).all()),
            "PA": bool((rd("PA", np.int8).reshape(128, 128) == PA).all()),
            "DG": bool((rd("DG", np.int8).reshape(4, 128, 128) == DG).all()),
            "DG_s8": bool(-128 <= int(DG.min()) and int(DG.max()) <= 127),
            "Sq": bool((rd("Sq", np.int32).reshape(4, 128) == Sq).all()),
            "X0": bool((rd("X0", np.int32) == X[:, 0]).all()),
        }
        o_ok = f_ok = r_ok = served_ok = True
        for w in range(4):
            bl = 4 * qt + w
            ku, rows = 32 * (bl + 1), slice(32 * w, 32 * w + 32)
            Mu, Pl = M[rows, :ku], P[rows, :ku] - 64
            Vu, bu = Vz[:, :ku].T, bz[:, :ku].T
            T, D, Op = cs.pv_words(Pl, Vu, U[rows], 0, bu, np.zeros((0, 32, 128), np.int64), 32, 32, 1, Mu)
            O, S, _ = cs.pv_served(Op, D, Vu + bu, Mu)
            served_ok &= not D and bool((O == (Mu * Pl) @ (Vu + bu)).all()) and bool((O + 64 * S == (Mu * P[rows, :ku]) @ (Vu + bu)).all())
            o_ok &= bool((Od[rows] == O).all())
            body = (O + 64 * S).astype(np.float32) * sp[rows, None]
            sink = sz[rows, None] * X[:, 0].astype(np.float32)[None, :]
            want_f = ((body + sink) * sv[None, :]).astype(np.float32)
            f_ok &= bool((Fd[rows].view(np.uint32) == want_f.view(np.uint32)).all())
            words = []
            for wd in [*T, Op]:
                full = np.zeros((128, 128), np.uint32)
                full[rows] = cs.s32(wd)
                words.append(full)
            sel = (ctid // 32) // 4 == w
            r_ok &= bool((tile_receipts(seed, words, t)[sel] == Rd[sel]).all())
        res.update({"O_is_served": served_ok, "O": o_ok, "F": f_ok, "R": r_ok})
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
