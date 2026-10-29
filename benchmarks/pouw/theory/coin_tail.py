#!/usr/bin/env python3
"""The lead's coin-branch tail (`Tail.coinTail_le`, Security/Proofs/Pouw/Lead/Tail/Repeats.lean), checked against the
coins as the layout draws them: a word is a prefix sum over steps, and the tag weights `G` (one set a column), the
step-0 bits `a₁` (a row's) and the tag bits `b` (a row's and step's) are shared across blocks.

**The route** (`Tail/Blocks.lean`). A block is a step, coin tile and column block, holding the coin tile's `ρ` row
blocks (4 at the three points, 1 at decode). `X` counts the blocks with a tile equal to a tile of an earlier block
(order: step, coin tile, column block) and `W` the blocks with two equal tiles of their own, so
`repeats ≤ ρ X + (ρ − 1) W` (`Tail.repeats_le_blocks`). Each count is bounded by `Tail.count_tail`
(`Pr[N ≥ r] ≤ (Σp)^r / r!` for sequentially dominated events) in its own order:

* `X` (`Tail.across_tail`), in block order with `G`, `a₁` and `b` first: a block's own coin lanes are read by no
  earlier block, and each of its tiles agrees with an earlier one on at most `2^−56` of them, so
  `λ_X = Σ_β ρ · (tiles before β) · 2^−56 ≤ ρ² B² / 2^57` (`Tail.lamX`).
* `N_G` (`Tail.columns_tail`), the column blocks with any agreement, in column-block order with every coin but `G`
  first: a column block's agreements read its own `G` only, `7^−8` a pair of tiles, so
  `λ_G = pairs · colBlocks · S · tiles · 7^−8`.
* `W_cb` (`Tail.slots_tail`), one column block's slots (step, coin tile) with an agreement, in slot order with `u`, `G`
  and `b₀` first: given `G` and the earlier slots, a slot agrees through its fresh `a₁` (step 0) or `b_t` bits on at
  most `2^−16` a pair, so `λ_b = pairs · S · tiles · 2^−16`.

With `X < a` and `N_G < k`, `repeats ≥ R` needs a column block with `x = ⌈⌈(R − ρ(a−1))/(ρ−1)⌉ / (k−1)⌉` agreeing
slots, so the tail is at most `λ_X^a/a! + λ_G^k/k! + colBlocks · λ_b^x/x!`, minimized over `a` and `k`, where a
count past its index set's size (the blocks, the column blocks, one column block's slots) is impossible and its term
is 0 (`Tail.cut`). That sum is `Tail.route_le`'s, exactly as it states it, with no term capped at 1. The script
evaluates it on every step and column-block count up to `--dense` past the smallest call (147 and 512) and
geometrically (`--growth`) beyond, each step count at its smallest `K ≥ 4,096` (the least slack), and finds the least
slack `t` (a share of stock, against `coinSlack`'s `8/10000`) that brings each call to `2^−42`, and `gapFixed` there.

**The coins as drawn.** On a small layout and the exact `word`, it checks each order's read sets (resampling the
later coordinates leaves the earlier events' tiles), that two tiles of one block differ by `G`, `a₁` and `b` alone
(their coin lanes cancel), that a step's tag bit moves every later word of its row by one (the persistence that
makes `b` the per-slot charge), and `repeats ≤ ρ X + (ρ − 1) W` on random tile values.

  python3 -B benchmarks/pouw/theory/coin_tail.py [--tiles-log2 30] [--also-log2 40] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from fractions import Fraction

LN2 = math.log(2)
TARGET = -42.0
SLACK = Fraction(8, 10000)
MEAN_RATE = Fraction(4, 7**8) + Fraction(1, 2**16)
LAYOUTS = {"pt512x64": 64, "pt512x512": 512, "pt8192": 8192, "decode": 1}


def steps_for(K: int) -> int:
    return 1 + (K - 21 + 27) // 28


def shape(m_rows: int, cb: int, S: int) -> dict:
    m = 16 * ((m_rows + 15) // 16)
    ct = (m + 63) // 64
    K = max(4096, 28 * S - 34)
    assert steps_for(K) == S
    return {"n": 8 * cb, "K": K, "S": S, "cb": cb, "ct": ct, "rho": min(4, m // 16), "tiles": m // 16 * cb * S,
            "B": cb * S * ct, "stock": Fraction(m * 8 * cb * K, 32)}


def threshold(L: dict, t: Fraction) -> int:
    """The least `repeats` in the tail: `128 · repeats > coinBranchMean + t · stock`."""
    return math.floor((128 * L["tiles"] * MEAN_RATE + t * L["stock"]) / 128) + 1


def lam(L: dict) -> tuple[float, float, float]:
    rho, B, slots = L["rho"], L["B"], L["S"] * L["ct"]
    pairs = rho * (rho - 1) // 2
    return rho * rho * B * B * 2.0**-57, L["cb"] * slots * pairs * 7.0**-8, slots * pairs * 2.0**-16


def lpois(lm: float, r: int) -> float:
    """log2 of λ^r / r!."""
    if r <= 0:
        return 0.0
    if lm <= 0:
        return -math.inf
    return r * math.log2(lm) - math.lgamma(r + 1) / LN2


def lcut(size: int, lm: float, r: int) -> float:
    """log2 of `Tail.cut size λ r`: 0 when `r` exceeds the count's index set, else `λ^r / r!`."""
    return -math.inf if r > size else lpois(lm, r)


def lsum(*xs: float) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log2(sum(2.0 ** (x - m) for x in xs))


def crossing(lo: int, hi: int, falling, rising) -> int:
    """The least `i` in `[lo, hi]` with `falling(i) ≤ rising(i)` (or `hi`)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if falling(mid) <= rising(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def w_bound(L: dict, w: int) -> tuple[float, int, int]:
    """log2 of the bound on Pr[W ≥ w], with its `k` and `x`."""
    slots, cbs = L["S"] * L["ct"], L["cb"]
    if w <= 0:
        return 0.0, 0, 0
    _, lg, lb = lam(L)
    top = min(cbs, w) + 1

    def first(k: int) -> float:
        return lcut(cbs, lg, k)

    def x_of(k: int) -> int:
        return -(-w // (k - 1))

    def second(k: int) -> float:
        return math.log2(cbs) + lcut(slots, lb, x_of(k))

    k0 = crossing(2, top, first, second)
    return min((lsum(first(k), second(k)), k, x_of(k)) for k in {max(2, k0 - 1), k0, min(k0 + 1, top)})


def tail(L: dict, R: int) -> tuple[float, dict]:
    """log2 of the route's bound on Pr[repeats ≥ R], with its parameters and terms."""
    lx = lam(L)[0]
    rho, B = L["rho"], L["B"]
    if rho == 1:
        v = lcut(B, lx, R)
        return v, {"a": R, "X": v}

    def xt(a: int) -> float:
        return lcut(B, lx, a)

    def wt(a: int) -> tuple[float, int, int]:
        return w_bound(L, -(-(R - rho * (a - 1)) // (rho - 1)))

    amax = max(1, (R - 1) // rho + 1)
    a0 = crossing(1, amax, xt, lambda a: wt(a)[0])
    best = None
    for a in {max(1, a0 - 1), a0, min(a0 + 1, amax)}:
        wv, k, x = wt(a)
        v = lsum(xt(a), wv)
        if best is None or v < best[0]:
            best = (v, {"a": a, "k": k, "x": x, "X": round(xt(a), 2), "W": round(wv, 2)})
    return best


def needed_slack(L: dict) -> Fraction:
    """The least slack (a share of stock) at which the route's bound is at most 2^−42."""
    lo, hi = 1, max(2, threshold(L, SLACK))
    while tail(L, hi)[0] > TARGET:
        hi *= 2
    R = crossing(lo, hi, lambda r: tail(L, r)[0], lambda r: TARGET)
    return max(Fraction(0), (128 * (R - 1) - 128 * L["tiles"] * MEAN_RATE) / L["stock"])


def grid(m_rows: int, tiles_log2: int, dense: int, growth: float):
    rb = (16 * ((m_rows + 15) // 16)) // 16
    cap = 2**tiles_log2

    def axis(lo: int, top: int) -> list[int]:
        xs = list(range(lo, min(top, lo + dense) + 1))
        v = float(xs[-1])
        while v * growth <= top:
            v *= growth
            xs.append(int(v))
        xs.append(top)
        return sorted(set(x for x in xs if lo <= x <= top))

    for S in axis(147, cap // (rb * 512)):
        for cb in axis(512, cap // (rb * S)):
            yield S, cb


def gap_fixed(m_rows: int, cols: int, K: int, t: Fraction, cap: int = 32) -> tuple[Fraction, Fraction]:
    """`gapFixedShare` and `gapFixedServed` at slack `t` (Definitions/Pouw/Lead/Gap.lean)."""
    m, n, S = 16 * ((m_rows + 15) // 16), 8 * ((cols + 7) // 8), steps_for(K)
    stock = Fraction(m * n * K, 32)
    words = m * n * S
    chunks = (words + 65535) // 65536
    sha_words = Fraction(351538, 100) * (Fraction(4 * words, 128) + chunks)
    honest = words + sha_words + Fraction(344905, 100) * Fraction(m_rows * K, 128)
    beta = Fraction(256, 3) * 32 * honest / (m * n * K)
    network = beta * Fraction(cap * m_rows * n * K, 973078528) / 32
    mean = 128 * (m // 16) * (n // 8) * S * MEAN_RATE
    fixed = 23 * chunks + network + mean + t * stock + Fraction(35, 1000000) * stock
    return fixed / stock, fixed / Fraction(m_rows * cols * K, 32)


def gap_fixed_sup(m_rows: int, t: Fraction) -> Fraction:
    """The largest `gapFixedShare` (`gapFixedServed` at decode) over the smallest calls' rounding, where it peaks."""
    side = 1 if m_rows == 1 else 0
    return max(gap_fixed(m_rows, n, K, t)[side] for n in (4096, 4097, 4104) for K in range(4096, 4125))


def evaluate(name: str, tiles_log2: int, dense: int, growth: float, slack: bool) -> dict:
    m_rows = LAYOUTS[name]
    worst, need, points = (-math.inf, None), (Fraction(0), None), 0
    for S, cb in grid(m_rows, tiles_log2, dense, growth):
        L = shape(m_rows, cb, S)
        R = threshold(L, SLACK)
        points += 1
        v, terms = tail(L, R)
        if v > worst[0]:
            worst = (v, {"n": L["n"], "K": L["K"], "tiles": L["tiles"], "R": R, **terms})
        if slack:
            ns = needed_slack(L)
            if ns > need[0]:
                need = (ns, {"n": L["n"], "K": L["K"]})
    out = {"points": points, "worst_log2": round(worst[0], 2), "worst_at": worst[1], "holds": worst[0] <= TARGET}
    if slack:
        out |= {"slack_needed_pct": float(need[0] * 100), "slack_needed_at": need[1],
                "gapFixed_pct": float(gap_fixed_sup(m_rows, SLACK) * 100),
                "gapFixed_pct_at_needed_slack": float(gap_fixed_sup(m_rows, need[0]) * 100)}
    return out


def structure(seed: int) -> dict:
    """The exact `word` on m = 128, n = 16, K = 105 (S = 4): the routes' read sets and the within-block cancellation."""
    rng = random.Random(seed)
    m, n, S = 128, 16, 4

    def bal13(z: int) -> list[int]:
        return [((z + 14280) // 13**q) % 13 - 6 for q in range(4)]

    tag = [bal13(i // 2 + 1 if i % 2 == 0 else -(i // 2 + 1)) for i in range(m)]
    code = [bal13(j + 1) for j in range(n)]

    def draw() -> dict:
        return {"b": [[rng.randrange(2) for _ in range(S)] for _ in range(m)],
                "a": [[rng.randrange(2) for _ in range(4)] for _ in range(m)],
                "g": [[rng.randrange(7) for _ in range(n)] for _ in range(4)],
                "u": [[[[rng.randrange(128) for _ in range(n)] for _ in range(3)] for _ in range(m // 64)]
                      for _ in range(S)]}

    def tiles(kap, w) -> dict:
        word = {}
        for i in range(m):
            for j in range(n):
                acc = -((sum(code[j]) + 3 * (S - 1) + 1) // 2)
                for t in range(S):
                    if t == 0:
                        c = sum(tag[i][q] * (w["g"][q][j] - 3) + w["a"][i][q] * code[j][q] for q in range(4))
                    else:
                        c = 1 + w["b"][i][t]
                    c += sum((127 if ell == i % 2 else 1) * (2 * w["u"][t][i // 64][ell][j] - 127) for ell in range(3))
                    acc += kap[i][t][j] + c
                    word[i, t, j] = acc
        return {(rb, cb, t): tuple(word[16 * rb + p // 8, t, 8 * cb + p % 8] for p in range(128))
                for rb in range(m // 16) for cb in range(n // 8) for t in range(S)}

    def copy(w) -> dict:
        return json.loads(json.dumps(w))

    fails = []
    blocks = [(t, ct, cb) for t in range(S) for ct in range(m // 64) for cb in range(n // 8)]
    slots = [(t, ct) for t in range(S) for ct in range(m // 64)]
    for _ in range(3):
        kap = [[[rng.randrange(-50, 51) for _ in range(n)] for _ in range(S)] for _ in range(m)]
        w = draw()
        base = tiles(kap, w)
        for bi, (t, ct, cb) in enumerate(blocks):
            w2, fresh = copy(w), draw()
            for (t2, ct2, cb2) in blocks[bi + 1:]:
                for ell in range(3):
                    for j in range(8 * cb2, 8 * cb2 + 8):
                        w2["u"][t2][ct2][ell][j] = fresh["u"][t2][ct2][ell][j]
            new = tiles(kap, w2)
            if any(new[rb, cb2, t2] != base[rb, cb2, t2] for (t2, ct2, cb2) in blocks[:bi + 1]
                   for rb in range(4 * ct2, 4 * ct2 + 4)):
                fails.append(("block order", t, ct, cb))
        for si, (t, ct) in enumerate(slots):
            w2, fresh = copy(w), draw()
            for (t2, ct2) in slots[si + 1:]:
                for i in range(64 * ct2, 64 * ct2 + 64):
                    if t2 == 0:
                        w2["a"][i] = fresh["a"][i]
                    else:
                        w2["b"][i][t2] = fresh["b"][i][t2]
            new = tiles(kap, w2)
            if any(new[rb, cb, t] != base[rb, cb, t] for rb in range(4 * ct, 4 * ct + 4) for cb in range(n // 8)):
                fails.append(("slot order", t, ct))
        for cb in range(n // 8):
            w2, fresh = copy(w), draw()
            for q in range(4):
                for j in range(n):
                    if j // 8 != cb:
                        w2["g"][q][j] = fresh["g"][q][j]
            new = tiles(kap, w2)
            if any(new[rb, cb, t] != base[rb, cb, t] for rb in range(m // 16) for t in range(S)):
                fails.append(("column-block order", cb))
        w2 = copy(w)
        w2["u"] = draw()["u"]
        new = tiles(kap, w2)
        for (rb, cb, t), v in base.items():
            for rb2 in range(4 * (rb // 4), 4 * (rb // 4) + 4):
                if [x - y for x, y in zip(v, base[rb2, cb, t])] != [x - y for x, y in zip(new[rb, cb, t],
                                                                                           new[rb2, cb, t])]:
                    fails.append(("lanes cancel", rb, rb2, cb, t))
        i, t0 = rng.randrange(m), rng.randrange(1, S)
        w2 = copy(w)
        w2["b"][i][t0] = 1 - w2["b"][i][t0]
        new = tiles(kap, w2)
        for (rb, cb, t), v in base.items():
            for p in range(128):
                want = w2["b"][i][t0] - w["b"][i][t0] if 16 * rb + p // 8 == i and t >= t0 else 0
                if new[rb, cb, t][p] - v[p] != want:
                    fails.append(("b persists", i, t0, rb, cb, t, p))
    for _ in range(2000):
        rho, nb = rng.choice([1, 2, 4]), rng.randrange(1, 12)
        vals = [[rng.randrange(rng.choice([2, 4, 30])) for _ in range(rho)] for _ in range(nb)]
        flat = [v for blk in vals for v in blk]
        X = sum(1 for bi, blk in enumerate(vals) if any(v in {y for b2 in vals[:bi] for y in b2} for v in blk))
        W = sum(1 for blk in vals if len(set(blk)) < len(blk))
        if len(flat) - len(set(flat)) > rho * X + (rho - 1) * W:
            fails.append(("repeats", vals))
    return {"fails": fails[:20], "ok": not fails}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiles-log2", type=int, default=30)
    ap.add_argument("--dense", type=int, default=160, help="every step and column-block count this far past the least")
    ap.add_argument("--growth", type=float, default=1.06)
    ap.add_argument("--also-log2", type=int, nargs="*", default=[40], help="cap sizes to evaluate coarsely, bound only")
    ap.add_argument("--json")
    args = ap.parse_args()
    report: dict = {"target_log2": TARGET, "slack_pct": float(SLACK * 100), "tiles_log2": args.tiles_log2,
                    "layouts": {}}
    for name in LAYOUTS:
        report["layouts"][name] = evaluate(name, args.tiles_log2, args.dense, args.growth, slack=True)
        print(name, json.dumps(report["layouts"][name]), flush=True)
    for lg in args.also_log2:
        report[f"tiles_2^{lg}"] = {name: evaluate(name, lg, 8, 1.3, slack=False) for name in LAYOUTS}
        print(f"tiles 2^{lg}", json.dumps(report[f"tiles_2^{lg}"]), flush=True)
    report["structure"] = structure(seed=1)
    print("structure", json.dumps(report["structure"]), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    ok = report["structure"]["ok"] and all(v["holds"] for v in report["layouts"].values())
    print("coinTail_le's 2^-42 at 2^%d tiles:" % args.tiles_log2, "holds" if ok else "FAILS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
