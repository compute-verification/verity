"""A CPU twin of J10-R for the measured-only arm kh_one_j10: the form J10-R of `SOURCE`
(benchmarks/pouw/rt_algo_r3/rt_algo_r3_check.py), never served or credited until J10 is ruled in and the verifier
takes the form. Its noise is that reference's for J10-R, h1tile.noise(seed_c, Layout(M, N, K)) at call scale: one
SHA-512 counter stream for the call, row i's chunks at Layout.row(i) = i rb and column j's at Layout.col(j) = M rb + j cb
(tag-separated counters are a proposed fix, not in that reference), unit (I, J) reading call rows
32 I .. 32 I + 31 and columns 32 J .. 32 J + 31. seed_c is seeds.seed(salt, STEP, root_a, root_b), the step position 7
the reference's forms use. Each unit keeps its own fold keys (h1tile.fold_keys of its receipt seed), and its z and
receipt are v3's (h1v3_ca) on that noise.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import h1v3_ca as V3

from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.ml.protocols.pouw.circuit import seeds as SD
from verity.ml.protocols.pouw.circuit import segment as SG

SOURCE = "benchmarks/pouw/rt_algo_r3/rt_algo_r3_check.py @ 3dc80591df (form J10-R)"
STEP = 7


def call_seed(salt: bytes, roots: tuple[bytes, bytes]) -> bytes:
    return SD.seed(salt, STEP, *roots)


def call_noise(seed_c: bytes, lay: HT.Layout, rows: Sequence[int], cols: Sequence[int]) -> tuple[dict, dict]:
    """h1tile.noise(seed_c, lay) at the listed call rows and columns only."""
    k, m = lay.k, lay.m
    out_r, out_c = {}, {}
    for i in rows:
        s = HT._stream(seed_c, lay.row(i), lay.row_bits)
        out_r[i] = (s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(m - 1)])
    for j in cols:
        s = HT._stream(seed_c, lay.col(j), lay.col_bits)
        out_c[j] = (s[:k], [HT.u129(HT._value(s[k + 8 * t:k + 8 * t + 8])) for t in range(32)])
    return out_r, out_c


def unit(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], rows: list, cols: list,
         keys: list[list[int]]) -> tuple[list[int], list[int]]:
    """v3's unit on words (h1v3_ca.reference with stream=True) on the given noise and fold keys: (z in row-major order,
    receipt)."""
    lay = HT.Layout(len(a), len(b), len(a[0]))
    xs = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)]
    wbs = [V3.col_wrap(b[j], cols[j]) for j in range(lay.c)]
    ks = lay.ks
    cp = [[0, 0] for _ in range(lay.sets)]
    served: dict[tuple[int, int], int] = {}
    for ln in range(lay.lanes):
        sets = V3.lane_sets(a, b, rows, cols, lay, ln)
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(V3.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
        for e in range(32):
            cell = (lay.rows(ln)[SG.row_slot(e)], lay.cols(ln)[SG.col_slot(e)])
            served[cell] = V3.stream_cell(sets[ks][e], [d[e] for d in sets[ks + 1:]], xs[cell[0]], wbs[cell[1]])
    return [served[(i, j)] for i in range(lay.r) for j in range(lay.c)], [w & V3.M32 for s in cp for w in s]


def reference(a: np.ndarray, b: np.ndarray, salt: bytes, base: int, roots: tuple[bytes, bytes],
              units: Sequence[int]) -> dict[int, tuple[list[int], list[int], dict, dict]]:
    """J10-R's (z, receipt, row noise, column noise) at the listed units of a call: A [rows, K] and B [N, K] int8, unit
    u = (u // (N / 32), u % (N / 32)) at position base + u."""
    (M, K), N = a.shape, b.shape[0]
    call = HT.Layout(M, N, K)
    tile = HT.Layout(32, 32, K)
    seed_c = call_seed(salt, roots)
    ncb = N // 32
    out = {}
    for u in units:
        ri, cj = divmod(u, ncb)
        R, C = range(32 * ri, 32 * ri + 32), range(32 * cj, 32 * cj + 32)
        nr, nc = call_noise(seed_c, call, R, C)
        keys = HT.fold_keys(SD.receipt_seed(salt, base + u, *roots), tile)
        ar = (a[32 * ri:32 * ri + 32].astype(np.int64) & 0xFF).tolist()
        br = (b[32 * cj:32 * cj + 32].astype(np.int64) & 0xFF).tolist()
        z, r = unit(ar, br, [nr[i] for i in R], [nc[j] for j in C], keys)
        out[u] = (z, r, nr, nc)
    return out


def injected_is_v3(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
                   roots: tuple[bytes, bytes]) -> bool:
    """`unit` on a unit's own noise and keys gives h1v3_ca.reference's z and receipt
    (rt_algo_r3_check's check_injected_reference_is_h1tile, for v3)."""
    lay = HT.Layout(len(a), len(b), len(a[0]))
    rows, cols = HT.noise(SD.seed(salt, position, *roots), lay)
    keys = HT.fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    return unit(a, b, rows, cols, keys) == V3.reference(a, b, salt, position, roots, stream=True)
