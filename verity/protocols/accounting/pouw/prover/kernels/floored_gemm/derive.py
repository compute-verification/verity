"""The duo receipts of one 128 x 128 tile, the CPU reference the edited GEMMs' references share.

`tile_receipts` folds a tile's u32 words after each step (masked_gemm's mref.py's duo receipts, one tile at a time) under
ref.py's keys; `TM`, `BN`, `WM` and `NCT` are the tile's shape and `U_TAG` the SHA-512 tag of lpgemm.cu's U. It loads the
ref.py beside it from its file, so a reader in another directory loads this file the same way: it puts nothing on its
path, and the name `derive` stays with the directory whose own scripts import one by it (harness/).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

__all__ = ["BN", "NCT", "TM", "U_TAG", "WM", "key_mul", "mix32", "tile_receipts"]

_spec = importlib.util.spec_from_file_location("floored_gemm_ref", Path(__file__).resolve().with_name("ref.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
key_mul, key_xor, mix32, rot_of, rotl = _ref.key_mul, _ref.key_xor, _ref.mix32, _ref.rot_of, _ref.rotl

U_TAG = 6
TM = BN = 128
WM, NCT = 32, 512


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

