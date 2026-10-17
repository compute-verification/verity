"""cwgemm.cu's gate judged on the CPU: cgemm_ref.py's check of the directory (c_l7qs and its peers, the operands cquant and
cwedit wrote), then cw_l7qs, which forms W' in shared memory from the cached weight and cplane's per-call edit, recomputed
on each dumped tile from x (bf16), W and the key alone through checked_strip's co32c_words, as cgemm_ref.py does:
    O = O' - sum_d 128^d D_d (Y_cw_l7qs) and tile_receipts over T_1 .. T_last, O', D_1 .. D_{m-1} (R_cw_l7qs)
and cplane's buffers that the kernel read: the plane (tile column tn's words, word (kt 128 + row) 4 + j at bit 8 b + w =
c(tn 128 + row, 128 kt + 32 j + 4 w + b), c on tag 8 in the patch layout) and UB = U + 64 (row r's 32 bytes). Wp_formula
checks the kernel's byte arithmetic, ((W + 64) + (U + 64) + c) ^ 0x80 = W + c + U, on the tile. One JSON line a gate
directory; exit 1 on any difference, on a device mismatch or on a control the device didn't reject.
    uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python3 benchmarks/pouw/edited_gemm/cwgemm_ref.py GATE_DIR...
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cgemm_ref as cr  # noqa: E402

T = cr.T


def plane_words(c: np.ndarray) -> np.ndarray:
    """c (128 x k, 0/1) to cplane's words for one tile column, in the order (kt, row, j)."""
    kt = c.shape[1] // 128
    bits = c.reshape(T, kt, 4, 8, 4).transpose(1, 0, 2, 4, 3).reshape(kt, T, 4, 32).astype(np.uint8)
    return np.packbits(bits, axis=-1, bitorder="little").view("<u4").reshape(-1)


def check(d: Path) -> dict:
    out = cr.check(d)
    meta = json.loads((d / "meta.json").read_text())
    cw = json.loads((d / "cw_meta.json").read_text())
    m_, n, k, seed, inv_s = meta["m"], meta["n"], meta["k"], meta["seed"], np.float32(meta["inv_s"])
    key = b"".join(struct.pack(">Q", int(w, 16)) for w in meta["key"])
    tiles_n = n // T
    nd = out["digits"]
    res_all = {"device_ok": cw["device_ok"], "control_rejected": cw["control_rejected"], "device": cw["device"], "tiles": {}}
    ok = bool(out["ok"]) and bool(cw["device_ok"]) and bool(cw["control_rejected"]) and cw["tiles"] == meta["tiles"]
    for tm, tn in cw["tiles"]:
        sfx = f"_{tm}_{tn}.bin"
        def rd(name: str, dt, sfx=sfx) -> np.ndarray:
            return np.fromfile(d / (name + sfx), dt)

        x = (rd("x", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(T, k)
        Vq = np.clip(np.floor(x * inv_s), -127, 126).astype(np.int64)
        b = cr.patch_bits(key, cr.DITHER_TAG, tm * T, T, k)
        W = rd("W", np.int8).astype(np.int64).reshape(T, k)
        c = cr.patch_bits(key, cr.C_TAG, tn * T, T, k)
        U = cr.u_rows(key, tn * T, T)
        z = cr.z_rows(key, tm * T, T, nd)
        Ts, D, Op = cr.cs.co32c_words(W, Vq.T, U, c, b.T, z, 32, 32, nd)
        O = (Op.astype(np.int64) - sum(dd.astype(np.int64) << (7 * (j + 1)) for j, dd in enumerate(D))) & cr.cs.MASK
        want_r = cr.tile_receipts(seed, [t.T for t in Ts] + [Op.T] + [dd.T for dd in D], tm * tiles_n + tn)
        Ucol = U[:, np.arange(k) % 32]
        formula = (((W + 64) + (Ucol + 64) + c) & 0xFF) ^ 0x80
        res = {
            "O": bool((rd("Y_cw_l7qs", np.int32).reshape(T, T).view(np.uint32) == O.T.astype(np.uint32)).all()),
            "R": bool((rd("R_cw_l7qs", np.uint32).reshape(cr.NCT, 8) == want_r).all()),
            "plane": bool((rd("plane", np.uint32) == plane_words(c)).all()),
            "UB": bool((rd("UB", np.uint8).reshape(T, 32) == (U + 64)).all()),
            "Wp_formula": bool(((formula ^ 0x80) - 0x80 == W + c + Ucol).all()),
            "sum_max": int(((W + 64) + (Ucol + 64) + c).max()),
        }
        res_all["tiles"][f"{tm},{tn}"] = res
        ok &= all(v for v in res.values() if isinstance(v, bool)) and res["sum_max"] <= 255
    out["cw_l7qs"] = res_all
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
