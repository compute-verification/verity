"""The wrap correction's pieces against H1V3 on the CPU, before sbench.sh asks for a GPU (`dither_one.h1v3_itself`'s
tree, this source's own unless DITHER_H1V3_TREE names another, run through `h1v3_ref.py`):

* the byte-mask rule `h1r.cu` computes wb with (`badd`, `wrap_masks`, `wrap_s8`, emulated here word for word) equals
  H1V3's `wrap_bits` on every (B, c, U) byte triple, and on random words (4 entries at once);
* the gate's CPU twin (`dither_one.h1v3_unit`) equals H1V3's `h1tile.reference(stream=True)` in z and
  receipt on random full-range units (entries at A = 127 and |B| = 127 forced) at m = 2 and m = 3, and z = X (B + c)^T
  (H1V3 peeled, 4733b9ab0c on: its z is A B^T, and the twin's peel of its own z is held to it);
* the peel on the twin's z: z - b B^T - A c^T - b c^T + 256 xw (B + c)^T and the kernel's form
  z + 256 xw B^T - X c^T - b B^T are A B^T on every cell, xw = [A = 127 and b = 1] where the s8 X wraps (the cells the
  peel as first written misses are counted), and the peeled kernel's operands (`is80` of X's byte and b for xw, b and
  c times 0xFF for -b and -c, emulated) are those values on every byte and bit.

    python wrapcheck.py    # one JSON line; exit 1 on any difference
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np

import dither_one as DO

M = 0xFFFFFFFF


def badd(a: int, b: int) -> int:
    return (((a & 0x7F7F7F7F) + (b & 0x7F7F7F7F)) ^ ((a ^ b) & 0x80808080)) & M


def spread4(x: int) -> int:
    return ((x & 0xF) * 0x00204081) & 0x01010101


def is80(x: int) -> int:
    v = x ^ 0x80808080
    return ((~(((v & 0x7F7F7F7F) + 0x7F7F7F7F) | v) & 0x80808080) & M) >> 7


def s8(w: int, i: int) -> int:
    x = (w >> (8 * i)) & 0xFF
    return x - 256 if x & 0x80 else x


def peel_operands_bad(aw: int, c4: int) -> bool:
    """h1r.cu's peel operands of one word of A and four dither bits against their values: xa = [A = 127 and b = 1],
    -b and (the same rule on the column side) -c."""
    b = spread4(c4)
    xa, nb = is80(badd(aw, b)) & b, (b * 0xFF) & M
    return any(s8(xa, i) != int(s8(aw, i) == 127 and (c4 >> i) & 1) or s8(nb, i) != -((c4 >> i) & 1) for i in range(4))


def kernel_wb(bw: int, uw: int, c4: int) -> list[int]:
    """h1r.cu's wb bytes (s8 -1, 0, 1) of one word: W' = badd(badd(B, U), c), then the masks."""
    r = badd(badd(bw, uw), spread4(c4))
    v = ~(bw ^ uw) & (bw ^ r) & 0x80808080 & M
    p, n = v & r, v & ~r & M
    q = (((p ^ n) >> 7) | ((((n & ~p) & M) >> 7) * 0xFE)) & M
    return [((q >> (8 * i)) & 0xFF) - (256 if (q >> (8 * i)) & 0x80 else 0) for i in range(4)]


def main() -> int:
    tab = DO.h1v3_itself({"wrap_table": True})
    if "wrap" not in tab:
        print(json.dumps({"ok": False, "h1v3": tab}))
        return 1
    want = [0 if w == 0 else 1 if w == 1 else -1 for w in tab["wrap"]]
    bad_bytes = [(b, c, u) for c in range(2) for u in range(256) for b in range(256)
                 if kernel_wb(b, u, c)[0] != want[(256 * c + u) * 256 + b]]
    rng = random.Random(1)
    bad_words = 0
    for _ in range(20000):
        bw, uw, c4 = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(4)
        got = kernel_wb(bw, uw, c4)
        bad_words += any(got[i] != want[(256 * ((c4 >> i) & 1) + ((uw >> (8 * i)) & 0xFF)) * 256 + ((bw >> (8 * i)) & 0xFF)]
                         for i in range(4))
    bad_peel = [(a, c) for c in range(2) for a in range(256) if peel_operands_bad(a * 0x01010101, 0xF * c)]
    for _ in range(20000):
        aw, c4 = rng.getrandbits(32), rng.getrandbits(4)
        bad_peel += [(aw, c4)] if peel_operands_bad(aw, c4) else []
    units, twins = [], []
    g = np.random.default_rng(7)
    for k in (64, 4128):
        a = g.integers(-128, 128, (32, k)).astype(np.int8)
        b = g.integers(-128, 128, (32, k)).astype(np.int8)
        a[:, ::7] = 127
        b[:, ::5] = 127
        b[:, 1::5] = -128
        salt, pos = bytes(g.integers(0, 256, 64, dtype=np.uint8)), int(g.integers(1 << 40))
        roots = (bytes(g.integers(0, 256, 64, dtype=np.uint8)), bytes(g.integers(0, 256, 64, dtype=np.uint8)))
        twins.append(DO.h1v3_unit(a, b, salt, pos, roots))
        units.append({"a": (a.astype(np.int64) & 0xFF).tolist(), "b": (b.astype(np.int64) & 0xFF).tolist(),
                      "salt": salt.hex(), "position": pos, "roots": [roots[0].hex(), roots[1].hex()]})
    hv = DO.h1v3_itself(units)
    peeled = bool(hv.get("peeled"))
    cases = []
    for k, tw, hu in zip((64, 4128), twins, hv.get("units", [])):
        hz = np.array(hu["z"], dtype=np.int64)
        cases.append({"k": k, "z_equal": bool(((tw["peel_kernel"] if peeled else tw["z"]) == hz).all()),
                      "receipt_equal": tw["receipt"] == hu["receipt"], "cells_z_ne_x_bc": int((tw["z"] != tw["xw"]).sum()),
                      "wb": tw["wb_plus"] + tw["wb_minus"], "x_wraps": tw["x_wraps"],
                      "peel_ca_cells_ne_ab": int((tw["peel_ca"] != tw["today"]).sum()),
                      "peel_xw_cells_ne_ab": int((tw["peel_xw"] != tw["today"]).sum()),
                      "peel_kernel_cells_ne_ab": int((tw["peel_kernel"] != tw["today"]).sum())})
    ok = (not bad_bytes and not bad_words and not bad_peel and len(cases) == 2
          and all(c["z_equal"] and c["receipt_equal"] and c["cells_z_ne_x_bc"] == 0 and c["wb"] > 0 and c["x_wraps"] > 0
                  and c["peel_xw_cells_ne_ab"] == c["peel_kernel_cells_ne_ab"] == 0 for c in cases))
    print(json.dumps({"ok": ok, "h1tile": tab["h1tile"], "h1v3_sha": tab.get("sha"), "h1v3_peeled": peeled,
                      "bytes_bad": bad_bytes[:8],
                      "words_bad": bad_words, "peel_operand_bad": bad_peel[:8], "units": cases,
                      "h1v3_error": hv.get("error")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
