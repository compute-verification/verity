"""COUNTER (LEAN program): searches of `G_tail`, the lead's tail form, at 5ced1a0e88 (`cursor/compute-research-lean-gtail-5634`).

`G_tail` bounds by `etaLead` the share of the coins on which a strategy is accepted, hashes no collision and works
under `32 · (honestRegs − gapFixed)`, with `gapFixed = headGap + networkGap + coinBranchMean + coinSlack + pStarGap`
(no hiding term). Its proof reads one stub, `MH100W.cost_ge_regs`, which holds by cases (every priced class costs at
least 32 a register under `H32`, `mma` by `WFw`, moves and operands write none), and `coinTail` is a supremum over the
data, so `G_tail` has no counterexample inside its premises: a strategy saves at most `128 · (repeats + tableHits)`
words (`Compose.perCoin`). What a search can break is the figures, and the premises' backing:

- `gapfixed`: `GapFixedAt` (0.162% of stock at the points at cap 32, 0.486% at cap 320; at decode 2.06% and 2.38% of
  served compute, 0.129% and 0.149% of the padded stock), exact over a scan of `n, K` and in closed form for every
  `n, K ≥ 4096` (`28 S ≤ K + 34`, one partial chunk, padding at `4104/4097`).
- `tail`: `TailAt`'s arithmetic, `tableTail` at `2^30` tiles with `coinTail ≤ 2^−42`, and the node memory past which
  it fails.
- `fibers`: the per-pair constants `coinTail_le` rests on, at every coin tile and row-block pair of the points: the
  `G` fiber of a pair's tag differences (`7^−8` a column block claimed), `a₁`'s per row (`1/2`, by its first digit),
  the coin lanes' per column (`2^−7`), and how many coin tiles share a pair's tag differences (one `G` value serves
  them all, `G` being shared across coin tiles and steps).
- `slack`: the least slack `t` that the proof's decomposition (`Tail.Repeats`: across blocks, then the column blocks
  with a `G` agreement, then each one's agreements by `a₁` and `b`, each through `count_tail`) needs for `2^−42`, at
  the proof's constants and at the exact ones, against `coinSlack` (0.08% of stock).
- `count_tail`: `Tail.count_tail` on small product spaces, exhaustively, random and adversarial events.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction as Q

import numpy as np

import counter_pad as cp
import counter_statement as cs

HEAD = "5ced1a0e88"
MEM = cp.MEM
COIN_SLACK = Q(8, 10000)
MEAN_RATE = Q(4, 7 ** 8) + Q(1, 2 ** 16)
P_WITHIN, P_AB, P_ACROSS = 7.0 ** -8, 2.0 ** -16, 2.0 ** -56
FIXED_AT = {
    "pt512x64": (512, 64, Q(162, 100000), Q(486, 100000)),
    "pt512x512": (512, 512, Q(162, 100000), Q(486, 100000)),
    "pt8192": (8192, 8192, Q(162, 100000), Q(486, 100000)),
}
DECODE_AT = ((32, Q(206, 10000), Q(129, 100000)), (320, Q(238, 10000), Q(149, 100000)))
POINT_ROWS = {"decode": 1, "pt512x64": 64, "pt512x512": 512, "pt8192": 8192}


def layout(t: int, m: int, n: int, k: int, cap: int = 32) -> dict:
    return cp.layout(t, m, n, k, cap) | {"cols": n}


def blocks(lay: dict) -> tuple[int, int, int, int]:
    """Row blocks, column blocks, steps and coin tiles."""
    return lay["m"] // 16, lay["n"] // 8, lay["S"], (lay["m"] + 63) // 64


def gap_fixed(lay: dict) -> dict:
    """`Gap.lean`'s `gapFixed` with `ChunksOf`, exact, on `counter_pad`'s port of the honest work and the network."""
    g = cp.gap(lay)
    rb, cb, s, _ = blocks(lay)
    mean = 128 * rb * cb * s * MEAN_RATE
    terms = {"head": g["terms"]["head"], "network": g["terms"]["network"], "coinBranchMean": mean,
             "coinSlack": COIN_SLACK * g["stock"], "pStar": g["terms"]["pStar"]}
    fixed = sum(terms.values())
    served = Q(lay["rows"] * lay["cols"] * lay["K"], 32)
    return {"stock": g["stock"], "served": served, "terms": terms, "gapFixed": fixed,
            "share": fixed / g["stock"], "served_share": fixed / served}


def closed_fixed(m: int, cap: int, rows: int | None = None) -> Q:
    """An upper bound on `gapFixedShare` for every `n, K ≥ 4096` at padded rows `m`: `counter_pad.closed_share`'s
    network and head, `coinBranchMean / stock = 32 · S/K · (4/7^8 + 2^−16)` at `S/K ≤ (1 + 34/4096)/28`, the slack
    and `p*`."""
    rows = m if rows is None else rows
    sk = Q(1, 28) + Q(34, 28 * 4096)
    honest = (32 * sk * (1 + Q(351538, 3200)) + Q(351538, 100) * 32 * (sk / 65536 + Q(1, m * 4096 * 4096))
              + Q(344905, 100) * Q(rows, 4 * 4096 * m))
    network = cp.N_MAX * Q(cap * rows, m * cp.FWD_MACS) * honest
    head = Q(23 * 32, 65536) * sk + Q(23 * 32, m * 4096 * 4096)
    return COIN_SLACK + Q(35, 1000000) + network + head + 32 * sk * MEAN_RATE


def _scan(fn, calls):
    worst, arg = None, None
    for n, k in calls:
        v = fn(n, k)
        if worst is None or v > worst:
            worst, arg = v, (n, k)
    return worst, arg


def scan_calls(n_span: int, k_span: int) -> list[tuple[int, int]]:
    """Two strips crossing where the shares peak (padding's `4104/4097` and `S/K`'s `148/4110`) and large calls: every
    `n` in `[4096, 4096 + n_span)` at `K` within three steps of 4,096, every `K` in `[4096, 4096 + k_span)` at `n` up
    to 4,112, and both up to `2^24`."""
    ns, ks = range(4096, 4096 + n_span), range(4096, 4096 + k_span)
    near_k, near_n = range(4096, 4096 + 28 * 3 + 1), range(4096, 4113)
    big = (8192, 16384, 65536, 2 ** 20, 2 ** 24)
    calls = {(n, k) for n in ns for k in near_k} | {(n, k) for n in near_n for k in ks}
    return sorted(calls | {(n, k) for n in big + (4096, 4097) for k in big + (4096, 4110)})


def gapfixed_at(calls: list[tuple[int, int]]) -> dict:
    out = {}
    for name, (t, m, lim32, lim320) in FIXED_AT.items():
        for cap, lim in ((32, lim32), (320, lim320)):
            worst, arg = _scan(lambda n, k: gap_fixed(layout(t, m, n, k, cap))["share"], calls)
            closed = closed_fixed(cp.pad_rows(m), cap)
            out[f"{name}/cap{cap}"] = {
                "claimed": float(lim), "scan_max": float(worst), "at_n_K": arg, "closed_form": float(closed),
                "holds": bool(worst <= lim and closed <= lim), "margin_scan": float(lim - worst),
                "margin_closed": float(lim - closed),
                "terms_at_max": {k: float(v / gap_fixed(layout(t, m, *arg, cap))["stock"])
                                 for k, v in gap_fixed(layout(t, m, *arg, cap))["terms"].items()}}
    for cap, lim_served, lim_share in DECODE_AT:
        ws, a_s = _scan(lambda n, k: gap_fixed(layout(1, 1, n, k, cap))["served_share"], calls)
        wp, a_p = _scan(lambda n, k: gap_fixed(layout(1, 1, n, k, cap))["share"], calls)
        closed = closed_fixed(16, cap, 1)
        closed_served = closed * 16 * Q(4104, 4097)
        out[f"decode/cap{cap}"] = {
            "claimed_served": float(lim_served), "claimed_share": float(lim_share),
            "served_scan_max": float(ws), "served_at": a_s, "share_scan_max": float(wp), "share_at": a_p,
            "served_closed": float(closed_served), "share_closed": float(closed),
            "holds_scan": bool(ws <= lim_served and wp <= lim_share),
            "holds_closed": bool(closed_served <= lim_served and closed <= lim_share),
            "margin_served_scan": float(lim_served - ws), "margin_served_closed": float(lim_served - closed_served),
            "margin_share_closed": float(lim_share - closed)}
    return out


def table_tail(tiles: int, mem: int = MEM) -> Q:
    return tiles * (Q(mem + 1, 2 ** 168) + Q(mem, 2 ** 105))


def tail_at() -> dict:
    """`TailAt` from `coinTail ≤ 2^−42` at `2^30` tiles, and the largest memory (in 512-byte tiles) it allows, with
    `coinTail` at its bound and at 0."""
    tiles = 2 ** 30
    eta = table_tail(tiles) + Q(1, 2 ** 42)
    per = tiles * (Q(1, 2 ** 168) + Q(1, 2 ** 105))
    mem_at = {name: int((Q(1, 2 ** 40) - c - Q(tiles, 2 ** 168)) / per)
              for name, c in (("coinTail_2^-42", Q(1, 2 ** 42)), ("coinTail_0", Q(0)))}
    return {"tiles": tiles, "tableTail": float(table_tail(tiles)), "log2_tableTail": math.log2(table_tail(tiles)),
            "etaLead_bound": float(eta), "holds": bool(eta <= Q(1, 2 ** 40)),
            "margin_log2": math.log2(Q(1, 2 ** 40) / eta),
            "mem_max_tiles": mem_at, "mem_max_TiB": {k: v * 512 / 2 ** 40 for k, v in mem_at.items()},
            "mem_TiB": MEM * 512 / 2 ** 40}


G_BOX = np.array(list(itertools.product(range(-3, 4), repeat=4)), dtype=np.int64).T
A_DIFF = np.array([np.subtract(a, b) for a in itertools.product((0, 1), repeat=4)
                   for b in itertools.product((0, 1), repeat=4)], dtype=np.int64)
TAGS = np.array([cs.row_tag(i) for i in range(8192)], dtype=np.int64)


def tag_diffs(m: int, c: int, rb: int, rb2: int) -> np.ndarray:
    rows = [(64 * c + 16 * rb + r, 64 * c + 16 * rb2 + r) for r in range(16)]
    return np.array([TAGS[i] - TAGS[i2] for i, i2 in rows if i < m and i2 < m], dtype=np.int64)


def g_fiber(d: np.ndarray) -> int:
    """The largest set of a column's `G[:, j] ∈ [−3, 3]^4` with one value of `(d(r) · G[:, j])_r`."""
    _, counts = np.unique((d @ G_BOX).T, axis=0, return_counts=True)
    return int(counts.max())


def pair_fibers(m: int) -> dict:
    """Every coin tile and row-block pair at `m` rows: its tag differences' rank, `G` fiber a column, and the coin
    tiles that share its tag differences (a `G` value meeting one meets them all, given aligned data)."""
    tiles = (m + 63) // 64
    rank_hist, fiber_hist, classes = {}, {}, {}
    worst = None
    for c in range(tiles):
        rbs = [rb for rb in range(4) if 64 * c + 16 * rb < m]
        for rb, rb2 in itertools.combinations(rbs, 2):
            d = tag_diffs(m, c, rb, rb2)
            key = (rb2 - rb, d.tobytes())
            if key not in classes:
                rank = int(np.linalg.matrix_rank(d.astype(float)))
                classes[key] = {"rank": rank, "fiber": g_fiber(d), "tiles": []}
            cl = classes[key]
            cl["tiles"].append(c)
            rank_hist[cl["rank"]] = rank_hist.get(cl["rank"], 0) + 1
            fiber_hist[cl["fiber"]] = fiber_hist.get(cl["fiber"], 0) + 1
            if worst is None or cl["fiber"] > worst["fiber"]:
                worst = {"fiber": cl["fiber"], "rank": cl["rank"], "coin_tile": c, "pair": [rb, rb2],
                         "d": d.tolist()}
    pairs = sum(rank_hist.values())
    out = {"pairs": pairs, "rank_hist": rank_hist, "fiber_hist": fiber_hist}
    if pairs:
        p_col = worst["fiber"] / 7 ** 4
        out |= {"worst": worst, "p_column": p_col, "p_block": p_col ** 8,
                "log7_p_block": 8 * math.log(p_col, 7), "claimed_log7": -8.0,
                "holds": bool(p_col ** 8 <= 7.0 ** -8 * (1 + 1e-12)),
                "distinct_classes": len(classes),
                "largest_class": max(len(cl["tiles"]) for cl in classes.values())}
    return out


def a1_rows(n_max: int) -> dict:
    """The largest chance over a row pair's `a₁` (independent bits, 4 a row) of one value of
    `(Δa₁ · C[:, j])_j` across a column block, over the column blocks up to `n_max` columns."""
    worst, arg, hist = 0.0, None, {}
    for cb in range(n_max // 8):
        code = np.array([cs.col_code(8 * cb + c) for c in range(8)], dtype=np.int64)
        _, counts = np.unique(A_DIFF @ code.T, axis=0, return_counts=True)
        p = counts.max() / 256
        hist[p] = hist.get(p, 0) + 1
        if p > worst:
            worst, arg = p, cb
    return {"n_max": n_max, "p_row_max": worst, "at_column_block": arg, "claimed": 0.5,
            "holds": bool(worst <= 0.5), "hist": {str(k): v for k, v in sorted(hist.items())},
            "log2_p_tile": 16 * math.log2(worst)}


def lane_column() -> dict:
    """`(s₀, s₁, s₂) ↦ (127 s₀ + s₁ + s₂, s₀ + 127 s₁ + s₂)` on the odd 7-bit coins: its largest fiber."""
    s = np.arange(-127, 128, 2, dtype=np.int64)
    s0, s1, s2 = np.meshgrid(s, s, s, indexing="ij")
    key = (127 * s0 + s1 + s2 + 20000) * 40000 + (s0 + 127 * s1 + s2 + 20000)
    _, counts = np.unique(key.ravel(), return_counts=True)
    fib = int(counts.max())
    return {"fiber": fib, "log2_p_column": math.log2(fib / 128 ** 3), "claimed_log2": -7.0,
            "log2_p_tile": 8 * math.log2(fib / 128 ** 3), "holds": bool(fib / 128 ** 3 <= 2.0 ** -7)}


def _log2_tail(mu: float, r: int) -> float:
    if r <= 0:
        return 0.0
    if mu <= 0:
        return -math.inf
    return (r * math.log(mu) - math.lgamma(r + 1)) / math.log(2)


def least_r(mu: float, log2_budget: float, cap: int | None = None) -> int:
    """The least `r ≥ 1` with `μ^r / r! ≤ 2^budget` (`count_tail`), or `cap + 1` if fewer: the count is at most `cap`."""
    lo = max(1, math.floor(mu))
    if _log2_tail(mu, lo) <= log2_budget:
        r = lo
        while r > 1 and _log2_tail(mu, r - 1) <= log2_budget:
            r -= 1
    else:
        hi = max(2 * lo, 2)
        while _log2_tail(mu, hi) > log2_budget:
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if _log2_tail(mu, mid) > log2_budget else (lo, mid)
        r = hi
    return r if cap is None else min(r, cap + 1)


SPLITS = ((1 / 3, 1 / 3, 1 / 3), (0.05, 0.475, 0.475), (0.475, 0.05, 0.475), (0.475, 0.475, 0.05),
          (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9))


def slack_needed(lay: dict, p_within: float, p_ab: float, p_across: float, log2_eta: float = -42.0) -> dict:
    """The least slack, as a share of stock, at which the decomposition's three `count_tail` bounds sum to `2^η`:
    `N_a` tiles agreeing with an earlier block's (`p_across` a pair), `N_cb` column blocks with a `G` agreement
    (`p_within` a pair and step, union over pairs, coin tiles and steps), and in each column block its agreements by
    `a₁` and `b` (`p_ab` each). Repeats are at most `N_a + N_cb · max N_in`."""
    rb, cb, s, ct = blocks(lay)
    tiles = rb * cb * s
    stock = lay["m"] * lay["n"] * lay["K"] / 32
    mean = float(128 * tiles * MEAN_RATE)
    pairs = sum(math.comb(len([r for r in range(4) if 64 * c + 16 * r < lay["m"]]), 2) for c in range(ct))
    events = pairs * s
    mu_a = p_across * 8 * (tiles // 4) * (tiles // 4 - 1)
    best = None
    for fa, f1, f2 in SPLITS:
        r_a = least_r(mu_a, log2_eta + math.log2(fa), tiles)
        if events:
            p_cb = min(1.0, events * p_within)
            r1 = least_r(cb * p_cb, log2_eta + math.log2(f1), cb)
            r2 = least_r(events * p_ab, log2_eta + math.log2(f2) - math.log2(cb), 3 * ct * s)
            within = (r1 - 1) * (r2 - 1)
        else:
            r1 = r2 = 1
            within = 0
        rep = (r_a - 1) + within
        if best is None or rep < best["repeats"]:
            best = {"repeats": rep, "r_a": r_a, "r1": r1, "r2": r2, "split": [fa, f1, f2]}
    need = max(0.0, 128 * best["repeats"] - mean)
    return best | {"tiles": tiles, "log2_tiles": math.log2(tiles), "mu_a": mu_a, "events_per_column_block": events,
                   "slack_share": need / stock, "coinSlack_share": float(COIN_SLACK),
                   "ratio": need / stock / float(COIN_SLACK), "holds": bool(need <= float(COIN_SLACK) * stock)}


def _k_for_tiles(rb: int, cb: int, tiles: int) -> int:
    s = tiles // (rb * cb)
    return 28 * (s - 1) - 6 if s > 1 else 4096


def slack_grid(exact: dict) -> dict:
    out = {}
    for point, rows in POINT_ROWS.items():
        m = cp.pad_rows(rows)
        for n in (4096, 16384, 2 ** 20):
            rb, cb = m // 16, n // 8
            ks = {4096, 4110, 65536, 2 ** 20} | {_k_for_tiles(rb, cb, 2 ** e) for e in (30, 40)}
            for k in sorted(ks):
                lay = layout(512, rows, n, k)
                rb_, cb_, s, _ = blocks(lay)
                if k < 4096 or rb_ * cb_ * s > 2 ** 40:
                    continue
                pe = exact.get(point, {})
                out[f"{point}/n{n}/K{k}"] = {
                    "proof": slack_needed(lay, P_WITHIN, P_AB, P_ACROSS),
                    "exact": slack_needed(lay, pe.get("p_block", 0.0), P_AB, exact["lane_tile"])}
    worst = {key: max(v[key]["slack_share"] for v in out.values()) for key in ("proof", "exact")}
    return {"calls": out, "worst_slack_share": worst, "coinSlack_share": float(COIN_SLACK),
            "holds": all(v["proof"]["holds"] for v in out.values())}


def _count_check(sizes: tuple, events: list, ps: list) -> float:
    """The largest `Pr[N ≥ r] / ((Σ p)^r / r!)` over `r ≥ 1` on the product space `sizes`."""
    pts = list(itertools.product(*[range(x) for x in sizes]))
    counts = np.array([sum(e[pt[: i + 1]] for i, e in enumerate(events)) for pt in pts])
    tot = sum(ps)
    worst = 0.0
    for r in range(1, len(sizes) + 1):
        pr = float((counts >= r).mean())
        bound = tot ** r / math.factorial(r)
        if pr > 0:
            worst = max(worst, pr / bound if bound > 0 else math.inf)
    return worst


def _dominating(sizes: tuple, events: list) -> list[float]:
    """Each event's `p i`: its largest share of a cylinder fixing the coordinates before it."""
    ps = []
    for i, e in enumerate(events):
        best = 0.0
        for pre in itertools.product(*[range(x) for x in sizes[:i]]):
            best = max(best, sum(e[pre + (v,)] for v in range(sizes[i])) / sizes[i])
        ps.append(best)
    return ps


def count_tail_search(trials: int, seed: int) -> dict:
    """Random and hill-climbed events on small product spaces, each event a function of the coordinates up to its
    own: the largest ratio of `Pr[N ≥ r]` to `count_tail`'s bound."""
    rng = np.random.default_rng(seed)
    worst, arg = 0.0, None
    for _ in range(trials):
        b = int(rng.integers(2, 5))
        sizes = tuple(int(x) for x in rng.integers(2, 4, size=b))
        dens = rng.random()
        events = []
        for i in range(b):
            keys = list(itertools.product(*[range(x) for x in sizes[: i + 1]]))
            events.append({k: bool(rng.random() < dens) for k in keys})
        ratio = _count_check(sizes, events, _dominating(sizes, events))
        for _ in range(20):
            i = int(rng.integers(b))
            k = list(events[i])[int(rng.integers(len(events[i])))]
            events[i][k] = not events[i][k]
            new = _count_check(sizes, events, _dominating(sizes, events))
            if new >= ratio:
                ratio = new
            else:
                events[i][k] = not events[i][k]
        if ratio > worst:
            worst, arg = ratio, {"sizes": sizes, "events": [sorted(k for k, v in e.items() if v) for e in events]}
    return {"trials": trials, "seed": seed, "max_ratio": worst, "at": arg, "holds": bool(worst <= 1 + 1e-12)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", nargs="+", default=["gapfixed", "tail", "fibers", "slack", "count_tail"])
    ap.add_argument("--n-span", type=int, default=520)
    ap.add_argument("--k-span", type=int, default=1960)
    ap.add_argument("--a1-n-max", type=int, default=2 ** 17)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cp.use_head("d5263f864b")
    rep: dict = {"head": HEAD, "config": vars(args)}
    if "gapfixed" in args.parts:
        calls = scan_calls(args.n_span, args.k_span)
        rep["gapfixed"] = gapfixed_at(calls)
        rep["gapfixed"]["scan"] = {"calls": len(calls), "n_span": args.n_span, "k_span": args.k_span}
        print(json.dumps(rep["gapfixed"], indent=1, default=str), flush=True)
    if "tail" in args.parts:
        rep["tail"] = tail_at()
        print(json.dumps(rep["tail"], indent=1), flush=True)
    exact: dict = {}
    if "fibers" in args.parts or "slack" in args.parts:
        rep["fibers"] = {p: pair_fibers(cp.pad_rows(r)) for p, r in POINT_ROWS.items()}
        rep["fibers"]["a1"] = a1_rows(args.a1_n_max)
        rep["fibers"]["lanes"] = lane_column()
        exact = {p: v for p, v in rep["fibers"].items() if p in POINT_ROWS}
        exact["lane_tile"] = 2.0 ** rep["fibers"]["lanes"]["log2_p_tile"]
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "worst"} for k, v in rep["fibers"].items()},
                         indent=1, default=str), flush=True)
    if "slack" in args.parts:
        rep["slack"] = slack_grid(exact)
        print(json.dumps({"worst_slack_share": rep["slack"]["worst_slack_share"], "holds": rep["slack"]["holds"]}),
              flush=True)
    if "count_tail" in args.parts:
        rep["count_tail"] = count_tail_search(args.trials, args.seed)
        print(json.dumps({k: v for k, v in rep["count_tail"].items() if k != "at"}), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
