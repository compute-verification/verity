"""The lookup receipt `lut` (`twins.py`) against fixed word differences: a one-word cancellation between two of its
tables, found from the key and checked on the receipt itself.

    python3 lut_trails.py [--kseed 0xC0FFEE11] [--samples 2000]

`lut` absorbs x = T0[w & 0x7FF] ⊕ T1[(w >> 11) & 0x7FF] ⊕ T2[w >> 22] into one state word. A word difference Δw that
changes fields 0 and 1 (Δ0, Δ1) leaves x unchanged whenever T0[a] ⊕ T0[a ⊕ Δ0] = T1[b] ⊕ T1[b ⊕ Δ1]. Each table
difference takes 2^10 values (one per pair), so among the 2^22 choices of (Δ0, Δ1) about 2^10 have a match (a0, b0),
and that Δw cancels for every w with a ∈ {a0, a0 ⊕ Δ0} and b ∈ {b0, b0 ⊕ Δ1}: probability 4 · 2^10 / 2^32 = 2^-20
over uniform words, with nothing else active, so the whole block's receipt is unchanged. The prover holds the tables (it
computes φ), so it finds such a Δw by a sort of 2^22 values.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twins  # noqa: E402

B2 = 256


def table_diffs(t: np.ndarray) -> np.ndarray:
    """D[d, a] = t[a] ⊕ t[a ⊕ d], d, a < len(t)."""
    n = len(t)
    a = np.arange(n)
    return t[a][None, :] ^ t[a[None, :] ^ a[:, None]]


def find_match(lt: list[int]) -> tuple[int, int, int, int] | None:
    """(Δ0, a0, Δ1, b0) with T0[a0] ⊕ T0[a0 ⊕ Δ0] = T1[b0] ⊕ T1[b0 ⊕ Δ1], Δ0, Δ1 ≠ 0, or None."""
    t0 = np.array(lt[:2048], dtype=np.uint32)
    t1 = np.array(lt[2048:4096], dtype=np.uint32)
    d0, d1 = table_diffs(t0)[1:].ravel(), table_diffs(t1)[1:].ravel()
    common = np.intersect1d(d0, d1)
    if len(common) == 0:
        return None
    v = common[0]
    i, j = int(np.flatnonzero(d0 == v)[0]), int(np.flatnonzero(d1 == v)[0])
    return i // 2048 + 1, i % 2048, j // 2048 + 1, j % 2048, len(common)


def receipt(p: list[int], k) -> int:
    st = (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)
    return twins.finish("lut", twins.absorb("lut", st, p, 0, k), k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kseed", type=lambda x: int(x, 0), default=0xC0FFEE11)
    ap.add_argument("--samples", type=int, default=2000)
    a = ap.parse_args()
    k = twins.keys(a.kseed)
    m = find_match(k.lt)
    if m is None:
        print(json.dumps({"kseed": hex(a.kseed), "match": None}))
        return 1
    d0, a0, d1, b0, matches = m
    dw = d0 | (d1 << 11)
    rng = random.Random(1)
    hit_in, hit_out = 0, 0
    for s in range(a.samples):
        p = [rng.getrandbits(32) for _ in range(B2)]
        t = rng.randrange(B2)
        q = list(p)
        if s % 2 == 0:
            aa = a0 ^ (d0 if rng.getrandbits(1) else 0)
            bb = b0 ^ (d1 if rng.getrandbits(1) else 0)
            p[t] = (p[t] & ~0x3FFFFF) | aa | (bb << 11)
            q = list(p)
            q[t] ^= dw
            hit_in += receipt(p, k) == receipt(q, k)
        else:
            q[t] ^= dw
            hit_out += receipt(p, k) == receipt(q, k)
    out = {"kseed": hex(a.kseed), "matches": matches, "delta0": d0, "a0": a0, "delta1": d1, "b0": b0,
           "dw": hex(dw), "log2p": -20, "in_set": [hit_in, (a.samples + 1) // 2],
           "uniform": [hit_out, a.samples // 2]}
    print(json.dumps(out))
    return 0 if hit_in == (a.samples + 1) // 2 else 1


if __name__ == "__main__":
    sys.exit(main())
