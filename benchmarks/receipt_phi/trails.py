"""Differential trail search over duo (`twins.py`), in a signed-digit Markov model, with an empirical check of each trail.

    python trails.py search [--key known|secret] [--k1 4] [--k2 4] [--bound 12] [--starts 0-255] > trails.json
    python trails.py check trails.json > check.json

duo's step at position t is u = y ⊕ p ⊕ k_t, y' = u·a_t + z, z' = rotl(u, r_t) + y'. A trail fixes, at every word,
the additive difference (P, Q) of the state (y, z) and the signed-digit difference σ of u (u' − u digit by digit,
each digit in {−1, 0, 1}). The prover's word difference is the XOR Δp = X_y ⊕ |σ|, where X_y is the XOR pattern the
trail expects of y, so Δp is fixed in advance. Given σ, the step is exact:

    P' = a_t·v(σ) + Q,    Q' = v(rotl_{r_t} σ) + P'      (mod 2^32; v(σ) = Σ σ_i 2^i)

A word's cost, −log2 of the chance that a uniform state follows it:

- c(P): y's difference shows as the trail's XOR pattern, the bits under its signed digits. c is the least number of
  nonzero digits of a signed-digit form of P mod 2^32, bit 31 not counted (its sign doesn't change the XOR pattern).
- wt(σ): u's bits under σ's digits must have the right values (u = y ⊕ p ⊕ k_t with k_t uniform makes them fair
  coins), every digit counted, bit 31 too, since rotl moves its sign into z.
- With the key secret (`--key secret`), a_t·v(σ) is a uniform odd multiple of 2^ℓ, ℓ the lowest digit of σ, so naming
  it costs m = 31 − ℓ more. With the key known (the prover runs φ with the job's key: the threat model), m = 0.

A trail starts at (0, 0), ends at (0, 0) (the 64-bit state collides) or at the block's end (word 256) on a state whose
receipt y ⊕ rotl(z, 16) doesn't change. It is a lower bound on a differential: trails with the same word differences
add, which `check` measures.

Search: Dijkstra over (P, Q, t) for each start t, cheapest first, so the first trail found is the cheapest in the model.
With the key known, a word's σ comes from two lists: every σ with wt ≤ k1, and every σ that reaches a P' with
c(P') ≤ k2 (σ a signed-digit form of a_t⁻¹(P' − Q)). A word outside both costs at least k1 + k2 + 2 (its own σ and
the next word's c(P')), so a best trail found at or below that cost is the best over all trails of the model. With the
key secret, every σ within the bound is enumerated, so that search is exhaustive up to its bound.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twins  # noqa: E402

M = 1 << 32
MASK = M - 1
TOP = 1 << 31
B2 = 256
U64 = np.uint64


def rotl(x: int, r: int) -> int:
    r &= 31
    return ((x << r) | (x >> (32 - r))) & MASK if r else x


def cmin(x: int) -> int:
    """The least number of nonzero digits, bit 31 not counted, of a signed-digit form of x mod 2^32."""
    c0, c1 = 0, 99
    for i in range(31):
        if (x >> i) & 1:
            c0, c1 = c0 + 1, min(c0 + 1, c1)
        else:
            c0, c1 = min(c0, c1 + 1), c1 + 1
    return min(c0, c1)


def cmin_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(U64)
    c0 = np.zeros(x.shape, np.int16)
    c1 = np.full(x.shape, 99, np.int16)
    for i in range(31):
        b = ((x >> U64(i)) & U64(1)).astype(bool)
        n0 = np.where(b, c0 + 1, np.minimum(c0, c1 + 1))
        n1 = np.where(b, np.minimum(c0 + 1, c1), c1 + 1)
        c0, c1 = n0, n1
    return np.minimum(c0, c1)


def reps(x: int, kmax: int, free31: bool) -> list[tuple[int, int, int]]:
    """Every signed-digit form (pos, neg, weight) of x mod 2^32 with weight ≤ kmax; bit 31 free if `free31`."""
    out: list[tuple[int, int, int]] = []

    def go(i: int, carry: int, pos: int, neg: int, w: int) -> None:
        if w > kmax:
            return
        if i == 32:
            out.append((pos, neg, w))
            return
        s = ((x >> i) & 1) + carry
        if s != 1:
            go(i + 1, s >> 1, pos, neg, w)
            return
        dw = 0 if (i == 31 and free31) else 1
        go(i + 1, 0, pos | (1 << i), neg, w + dw)
        go(i + 1, 1, pos, neg | (1 << i), w + dw)

    go(0, 0, 0, 0, 0)
    return out


def sd_rotl(pos: int, neg: int, r: int) -> tuple[int, int]:
    return rotl(pos, r), rotl(neg, r)


def sd_val(pos: int, neg: int) -> int:
    return (pos - neg) & MASK


@lru_cache(maxsize=None)
def sigma_table(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every σ ≠ 0 with at most k digits: (pos, neg, weight) arrays."""
    from itertools import combinations

    P, N, W = [], [], []
    for j in range(1, k + 1):
        bits = np.array(list(combinations(range(32), j)), dtype=U64).reshape(-1, j)
        sh = U64(1) << bits
        signs = ((np.arange(1 << j)[:, None] >> np.arange(j)[None, :]) & 1).astype(bool)
        for sg in signs:
            P.append(np.bitwise_or.reduce(np.where(sg, U64(0), sh), axis=1))
            N.append(np.bitwise_or.reduce(np.where(sg, sh, U64(0)), axis=1))
            W.append(np.full(len(bits), j, np.int16))
    return np.concatenate(P), np.concatenate(N), np.concatenate(W)


@lru_cache(maxsize=None)
def target_table(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every value P mod 2^32 with c(P) ≤ k, and its c."""
    p, n, _ = sigma_table(k) if k else (np.zeros(0, U64),) * 3
    low = ((p | n) >> U64(31)) == 0
    vals = (p[low].astype(np.int64) - n[low].astype(np.int64)) % M
    vals = np.concatenate([[0], vals, (vals + TOP) % M, [TOP]]).astype(U64)
    vals = np.unique(vals)
    return vals, cmin_np(vals)


def rotl_np(x: np.ndarray, r: int) -> np.ndarray:
    r &= 31
    if not r:
        return x
    return ((x << U64(r)) | (x >> U64(32 - r))) & U64(MASK)


def fold_cost(P: int, Q: int, kmax: int) -> int | None:
    """−log2 of the best trail ending the block on (P, Q): XOR patterns X_y = rotl(X_z, 16) (duo's receipt)."""
    best = None
    ry = {}
    for pos, neg, w in reps(P, kmax, True):
        x = pos | neg
        ry[x] = min(ry.get(x, 99), w)
    for pos, neg, w in reps(Q, kmax, True):
        x = rotl(pos | neg, 16)
        if x in ry and (best is None or ry[x] + w < best):
            best = ry[x] + w
    return best


class Schedule:
    """Keys and rotations per position: a_t (odd) from `attacks.keys`' derivation, r_t from `twins.rot_of`."""

    def __init__(self, kseed: int = 0xC0FFEE11, n: int = B2, rot=twins.rot_of):
        self.a = [twins.gen(kseed, 1, i, 0) | 1 for i in range(n)]
        self.k = [twins.gen(kseed, 3, i, 0) for i in range(n)]
        self.r = [rot(i) for i in range(n)]
        self.ainv = [pow(a, -1, M) for a in self.a]
        self.n = n


def edges_known(s: Schedule, t: int, Q: int, budget: int, k1: int, k2: int):
    """(σ pos, σ neg, wt(σ), P', Q', c(P')) for every σ of the known-key lists within `budget` (σ's weight plus c(P'))."""
    a, r = s.a[t], s.r[t]
    out = []
    if budget <= 0:
        return out
    pos, neg, wt = sigma_table(min(k1, budget))
    v = (pos.astype(np.int64) - neg.astype(np.int64)) % M
    P2 = (v.astype(U64) * U64(a) + U64(Q)) & U64(MASK)
    c2 = cmin_np(P2)
    ok = (wt + c2) <= budget
    rp, rn = rotl_np(pos[ok], r), rotl_np(neg[ok], r)
    Q2 = ((rp.astype(np.int64) - rn.astype(np.int64)) % M).astype(U64)
    Q2 = (Q2 + P2[ok]) & U64(MASK)
    out += list(zip(pos[ok].tolist(), neg[ok].tolist(), wt[ok].tolist(), P2[ok].tolist(), Q2.tolist(), c2[ok].tolist()))
    tv, tc = target_table(min(k2, budget))
    v = ((tv.astype(np.int64) - Q) % M).astype(U64) * U64(s.ainv[t]) & U64(MASK)
    wv = cmin_np(v)
    cand = np.nonzero(wv + tc <= budget)[0]
    for i in cand.tolist():
        P2i, ci = int(tv[i]), int(tc[i])
        for sp, sn, w in reps(int(v[i]), budget - ci, False):
            if w == 0 or w <= k1:
                continue
            rp, rn = sd_rotl(sp, sn, r)
            out.append((sp, sn, w, P2i, (sd_val(rp, rn) + P2i) & MASK, ci))
    return out


def edges_secret(s: Schedule, t: int, Q: int, budget: int):
    """Every σ with wt(σ) + m(σ) + c(P') within `budget`, each with every a_t·v(σ) (an odd multiple of 2^ℓ)."""
    r = s.r[t]
    out = []
    for lo in range(31, -1, -1):
        m = 31 - lo
        if m + 1 > budget:
            break
        alphas = (np.arange(1 << m, dtype=U64) * U64(2) + U64(1)) << U64(lo)
        P2 = (alphas + U64(Q)) & U64(MASK)
        c2 = cmin_np(P2)
        for k in range(0, budget - m):
            for hi_bits in _combos(lo + 1, k):
                for signs in range(1 << (k + 1)):
                    digits = [lo] + list(hi_bits)
                    pos = sum(1 << d for j, d in enumerate(digits) if not (signs >> j) & 1)
                    neg = sum(1 << d for j, d in enumerate(digits) if (signs >> j) & 1)
                    w = k + 1
                    ok = np.nonzero(w + m + c2 <= budget)[0]
                    if not len(ok):
                        continue
                    rp, rn = sd_rotl(pos, neg, r)
                    rv = sd_val(rp, rn)
                    for i in ok.tolist():
                        P2i = int(P2[i])
                        out.append((pos, neg, w + m, P2i, (rv + P2i) & MASK, int(c2[i])))
    return out


def _combos(start: int, k: int):
    from itertools import combinations

    return combinations(range(start, 32), k)


def search(s: Schedule, starts, bound: int, key: str = "known", k1: int = 4, k2: int = 4, fold: bool = True):
    """The cheapest trail from any start in `starts` within `bound`, by Dijkstra. Returns (trail or None, stats)."""
    heap: list = []
    parent: dict = {}
    best_g: dict = {}
    seq = 0
    expanded = 0
    for t0 in starts:
        st = (0, 0, t0, True)
        best_g[st] = 0
        heap.append((0, seq, st))
        seq += 1
    heapq.heapify(heap)
    found = None
    while heap:
        g, _, st = heapq.heappop(heap)
        if g > best_g.get(st, 99) or g >= bound:
            continue
        if found is not None and g >= found[0]:
            break
        P, Q, t, fresh = st
        if not fresh and P == 0 and Q == 0:
            found = (g, st)
            break
        if t == s.n:
            if fold and not fresh:
                fc = fold_cost(P, Q, bound - g)
                if fc is not None and g + fc < bound and (found is None or g + fc < found[0]):
                    found = (g + fc, st)
                    bound = g + fc
            continue
        expanded += 1
        base = 0 if fresh else cmin(P)
        budget = bound - 1 - g - base
        if not fresh and base + g < bound:
            nxt = (Q, Q, t + 1, False)
            ng = g + base
            if ng < best_g.get(nxt, 99):
                best_g[nxt] = ng
                parent[nxt] = (st, (0, 0, 0))
                heapq.heappush(heap, (ng, seq, nxt))
                seq += 1
        es = edges_known(s, t, Q, budget + 1, k1, k2) if key == "known" else edges_secret(s, t, Q, budget + 1)
        for sp, sn, w, P2, Q2, c2 in es:
            ng = g + base + w
            if ng + c2 >= bound:
                continue
            nxt = (P2, Q2, t + 1, False)
            if ng < best_g.get(nxt, 99):
                best_g[nxt] = ng
                parent[nxt] = (st, (sp, sn, w))
                heapq.heappush(heap, (ng, seq, nxt))
                seq += 1
    stats = {"expanded": expanded, "states": len(best_g), "open_min": heap[0][0] if heap else None}
    if found is None:
        return None, stats
    cost, st = found
    steps = []
    while st in parent:
        prev, (sp, sn, w) = parent[st]
        steps.append({"t": prev[2], "P": prev[0], "Q": prev[1], "sigma_pos": sp, "sigma_neg": sn, "w_sigma": w,
                      "c_P": 0 if prev[3] else cmin(prev[0])})
        st = prev
    steps.reverse()
    return {"cost": cost, "end": "collision" if found[1][0] == 0 and found[1][1] == 0 else "fold",
            "end_state": [found[1][0], found[1][1], found[1][2]], "steps": steps}, stats


def word_diffs(trail: dict) -> list[tuple[int, int]]:
    """(t, Δp_t) for every word of the trail: X_y ⊕ |σ|, X_y the XOR pattern of a cheapest form of P."""
    out = []
    for st in trail["steps"]:
        P = st["P"]
        xy = 0
        if P:
            rp = min(reps(P, cmin(P), True), key=lambda x: x[2])
            xy = rp[0] | rp[1]
        out.append((st["t"], xy ^ (st["sigma_pos"] | st["sigma_neg"])))
    if trail["end"] == "collision":
        last = trail["steps"][-1]
        P2 = trail["end_state"][0]
        assert P2 == 0
    else:
        P, t = trail["end_state"][0], trail["end_state"][2]
        assert t == B2
    return out


def simulate(s: Schedule, diffs: list[tuple[int, int]], samples: int, seed: int, random_keys: bool,
             end: str) -> dict:
    """The share of uniform states (and of keys, if `random_keys`) for which the word differences cancel: the 64-bit
    state collides after the last word, or (end = fold, the trail running to the block's end) the receipt agrees."""
    rng = np.random.default_rng(seed)
    t0, t1 = diffs[0][0], diffs[-1][0]
    ts = list(range(t0, t1 + 1))
    dmap = dict(diffs)
    y = rng.integers(0, M, size=samples, dtype=U64)
    z = rng.integers(0, M, size=samples, dtype=U64)
    y2, z2 = y.copy(), z.copy()
    for t in ts:
        if random_keys:
            a = rng.integers(0, M, size=samples, dtype=U64) | U64(1)
            k = rng.integers(0, M, size=samples, dtype=U64)
        else:
            a, k = U64(s.a[t]), U64(s.k[t])
        p = rng.integers(0, M, size=samples, dtype=U64)
        r = s.r[t]
        for which in (0, 1):
            yy, zz = (y, z) if which == 0 else (y2, z2)
            pp = p if which == 0 else p ^ U64(dmap.get(t, 0))
            u = yy ^ pp ^ k
            ny = (u * a + zz) & U64(MASK)
            nz = (rotl_np(u, r) + ny) & U64(MASK)
            if which == 0:
                y, z = ny, nz
            else:
                y2, z2 = ny, nz
    if end == "collision":
        hit = int(np.count_nonzero((y == y2) & (z == z2)))
    else:
        f1 = y ^ rotl_np(z, 16)
        f2 = y2 ^ rotl_np(z2, 16)
        hit = int(np.count_nonzero(f1 == f2))
    return {"samples": samples, "hits": hit, "log2p": math.log2(hit / samples) if hit else None}


def parse_starts(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("search")
    a.add_argument("--key", choices=("known", "secret"), default="known")
    a.add_argument("--k1", type=int, default=4)
    a.add_argument("--k2", type=int, default=4)
    a.add_argument("--bound", type=int, default=12)
    a.add_argument("--starts", default="0-255")
    a.add_argument("--kseed", type=lambda x: int(x, 0), default=0xC0FFEE11)
    a.add_argument("--no-fold", action="store_true")
    args = ap.parse_args()
    s = Schedule(args.kseed)
    trail, stats = search(s, parse_starts(args.starts), args.bound, args.key, args.k1, args.k2, not args.no_fold)
    print(json.dumps({"key": args.key, "k1": args.k1, "k2": args.k2, "bound": args.bound, "starts": args.starts,
                      "kseed": args.kseed, "trail": trail, "stats": stats}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
