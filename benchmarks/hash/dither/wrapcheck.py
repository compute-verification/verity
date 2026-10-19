"""The wrap correction's pieces against CA's H1V3 on the CPU, before sbench.sh asks for a GPU (H1V3's tree at
DITHER_H1V3_TREE, run through `h1v3_ref.py`):

* the byte-mask rule `h1r.cu` computes wb with (`badd`, `wrap_masks`, `wrap_s8`, emulated here word for word) equals
  H1V3's `wrap_bits` on every (B, c, U) byte triple, and on random words (4 entries at once);
* this lane's CPU twin (`dither_one.h1v3_unit`, the gate's) equals H1V3's `h1tile.reference(stream=True)` in z and
  receipt on random full-range units (entries at A = 127 and |B| = 127 forced) at m = 2 and m = 3, and z = X (B + c)^T.

    python wrapcheck.py    # one JSON line; exit 1 on any difference, 2 without a tree
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

import dither_one as DO

M = 0xFFFFFFFF


def badd(a: int, b: int) -> int:
    return (((a & 0x7F7F7F7F) + (b & 0x7F7F7F7F)) ^ ((a ^ b) & 0x80808080)) & M


def spread4(x: int) -> int:
    return ((x & 0xF) * 0x00204081) & 0x01010101


def kernel_wb(bw: int, uw: int, c4: int) -> list[int]:
    """h1r.cu's wb bytes (s8 -1, 0, 1) of one word: W' = badd(badd(B, U), c), then the masks."""
    r = badd(badd(bw, uw), spread4(c4))
    v = ~(bw ^ uw) & (bw ^ r) & 0x80808080 & M
    p, n = v & r, v & ~r & M
    q = (((p ^ n) >> 7) | ((((n & ~p) & M) >> 7) * 0xFE)) & M
    return [((q >> (8 * i)) & 0xFF) - (256 if (q >> (8 * i)) & 0x80 else 0) for i in range(4)]


def main() -> int:
    if not os.environ.get("DITHER_H1V3_TREE"):
        print(json.dumps({"ok": False, "why": "no DITHER_H1V3_TREE"}))
        return 2
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
    cases = []
    for k, tw, hu in zip((64, 4128), twins, hv.get("units", [])):
        cases.append({"k": k, "z_equal": bool((tw["z"] == np.array(hu["z"], dtype=np.int64)).all()),
                      "receipt_equal": tw["receipt"] == hu["receipt"], "cells_z_ne_x_bc": int((tw["z"] != tw["xw"]).sum()),
                      "wb": tw["wb_plus"] + tw["wb_minus"]})
    ok = (not bad_bytes and not bad_words and len(cases) == 2
          and all(c["z_equal"] and c["receipt_equal"] and c["cells_z_ne_x_bc"] == 0 and c["wb"] > 0 for c in cases))
    print(json.dumps({"ok": ok, "h1tile": tab["h1tile"], "h1v3_sha": tab.get("sha"), "bytes_bad": bad_bytes[:8],
                      "words_bad": bad_words, "units": cases, "h1v3_error": hv.get("error")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
