"""COUNTER (LEAN program): the lead statement from dd30c5c347, whole tiles (`padRows`, `padCols`) and a `CoinStream`.

`--head` picks the statement on `cursor/compute-research-lean-statement-4457`. ebe1fcd968 (the network restatement)
differs from dd30c5c347 in step 0, which holds 21 data lanes and the three coin lanes (`stepsFor K = 1 + ⌈(K − 21)/28⌉`),
in the network term (`betaLead · callBits / 32`, the call's MACs' share of Llama-3.2-1B's forward at the cap) and in
`GapAtDecode` (0.232% and 3.72% at cap 32, 0.252% and 4.04% at cap 320, against 0.266% and 4.25%). d5263f864b puts
`tableGap` at `(2 mem + 1) · 2^-100` a tile and step at every step (the coin lanes alone), `GapAt320` at 0.513%,
0.491% and 0.488%, and `GapAtDecode` against served compute (`gapServed`, the unpadded `rows · cols · K / 32`) at
3.71% and 4.03%, with the padded share at 0.231% and 0.252%.

At dd30c5c347 the statement commits every call over whole MMA tiles: `lead T m n K` has
`m := padRows m` (16 ⌈m/16⌉), `rows := m` and `n := padCols n` (8 ⌈n/8⌉). A padding row has zero data and its own tag,
coin lanes and coins. Stock, words and honest work are the padded call's, and `aBlocks` counts A's unpadded rows.
The stream is a `CoinStream` (each coin at one position at most). `tableGap` is, a tile of each row and column block,
`128 · ((2 mem + 1)/49^8 + 32/7^8 + (S − 1)(2 mem + 1)/2^100)`. `G` has no `DataInRange` and no `16 ∣ m ∧ 8 ∣ n`.

This module ports the padded layout and gap (exact, `Fraction`) and checks:

- **cover**: `counter_tile`'s cover strategy at m = n = 1, m 1 × n 9 and decode. Every tile is whole, so a step-0
  tile reads two tag digits of `G` in each of its 8 columns and `a₁`'s two digits in each of its 16 rows. A pre-coin
  table of `2 · mem + 1` values then holds it with chance at most `(2 mem + 1) · p_max`.
- **step-0 bound** (`step0_support`): a step-0 tile is a linear map of its block's 32 `G` coins given the rest, and
  a rank-`ρ` map on the box has chance at most `7^-ρ` a value. A constant tile has `ρ = 8 · rank T` (`49^-8` at
  rank 2). A stream tile holding some of the block's `G` loses a row of a column for each, at most 32: the least
  rank it can leave is `7^-8`'s exponent when it is at least 8.
- **later steps** (`lane_injective`): `(s₀, s₁, s₂) ↦ (127 s₀ + s₁ + s₂, s₀ + 127 s₁ + s₂)` is injective on the odd
  7-bit coins, so a column with an even and an odd position free of the tile's own 24 lane coins pins its 21 bits.
  At most 3 columns lack one (each needs 8 lane-coin positions), so at least 5 are pinned: `2^-105 ≤ 2^-100`.
- **GapAt, GapAt320, GapAtDecode** exact over a scan and in closed form, padded `n` included; `Nontrivial`.
- **pStar_eq** (130 of `2 · 128³`) and **word_exact** at K = 65,536 under `DataInRange`.
- **repeats** (`counter_tile.block_dedup` on the padded call): committed tiles equal whole to another of the call.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction as Q

import numpy as np

import counter_statement as cs
import counter_tile as ct

MEM = 2 ** 34
P0 = Q(2 * MEM + 1, 49 ** 8) + Q(32, 7 ** 8)
FWD_MACS = 973078528
N_MAX = Q(256, 3)

HEADS = {
    "dd30c5c347": {"step0_data": 24, "step0_lanes": False, "network": "stock", "sk_num": 31,
                   "decode": ((32, Q(266, 100000), Q(425, 10000)),)},
    "ebe1fcd968": {"step0_data": 21, "step0_lanes": True, "network": "bits", "sk_num": 34,
                   "decode": ((32, Q(232, 100000), Q(372, 10000)), (320, Q(252, 100000), Q(404, 10000)))},
    "d5263f864b": {"step0_data": 21, "step0_lanes": True, "network": "bits", "sk_num": 34, "table": "lanes",
                   "lim320": (Q(513, 100000), Q(491, 100000), Q(488, 100000)),
                   "decode": ((32, Q(231, 100000), Q(371, 10000)), (320, Q(252, 100000), Q(403, 10000)))},
}
HEAD: dict = {}
_CS_WORDS_AT, _CS_WORD_SCALAR = cs.words_at, cs.word_scalar


def _words_at(lay: dict, om: dict, rows: np.ndarray, cols: np.ndarray, kappa: int = 0) -> np.ndarray:
    """`counter_statement.words_at` with step 0's coin lanes (`laneTerm` at `t = 0`), which every later word sums."""
    wts = np.array([[cs.rot_weight(int(i), ell) for ell in range(3)] for i in rows], dtype=np.int64)
    s0 = om["s"][0][rows // lay["tile"]][:, :, cols].astype(np.int64)
    return _CS_WORDS_AT(lay, om, rows, cols, kappa) + np.einsum("il,ilj->ij", wts, s0)[:, None, :]


def _word_scalar(lay: dict, om: dict, i: int, t: int, j: int, kappa: int = 0) -> int:
    """`word` at ebe1fcd968 straight from the Lean: `coinTerm` is step 0's tag and code or a later step's tag lane,
    plus `laneTerm` at every step."""
    total, k = -cs.bias(lay["S"], j), i // lay["tile"]
    for tau in range(t + 1):
        if tau == 0:
            ct = sum(cs.row_tag(i)[q] * int(om["G"][q, j]) for q in range(4)) + sum(
                int(om["a1"][i, q]) * cs.col_code(j)[q] for q in range(4))
        else:
            ct = 1 + int(om["b"][i, tau])
        lane = sum(cs.rot_weight(i, ell) * (int(om["s"][tau, k, ell, j]) if k < cs.tiles(lay) else 0)
                   for ell in range(3))
        total += kappa + ct + lane
    return total


def use_head(name: str) -> None:
    """Port the statement at `name`: its `stepsFor` and, from ebe1fcd968, step 0's coin lanes."""
    HEAD.clear()
    HEAD.update(HEADS[name] | {"name": name})
    d = HEAD["step0_data"]
    cs.steps_for = lambda k: 1 + (max(k - d, 0) + 27) // 28
    cs.words_at = _words_at if HEAD["step0_lanes"] else _CS_WORDS_AT
    cs.word_scalar = _word_scalar if HEAD["step0_lanes"] else _CS_WORD_SCALAR


def pad_rows(m: int) -> int:
    return 16 * ((m + 15) // 16)


def pad_cols(n: int) -> int:
    return 8 * ((n + 7) // 8)


def layout(t: int, m: int, n: int, k: int, cap: int = 32) -> dict:
    """`Layout.lean`'s `lead T m n K` (`lead320` at cap 320)."""
    return cs.layout(t, pad_rows(m), pad_cols(n), k, cap) | {"rows": m, "mem": MEM}


def table_gap(lay: dict) -> Q:
    """From d5263f864b every step, step 0 included, is at `(2 mem + 1) · 2^-100` from its coin lanes alone."""
    rb, cb = lay["m"] // 16, lay["n"] // 8
    if HEAD.get("table") == "lanes":
        return rb * cb * cs.TILE_WORDS * lay["S"] * Q(2 * MEM + 1, 2 ** 100)
    return rb * cb * cs.TILE_WORDS * (P0 + (lay["S"] - 1) * Q(2 * MEM + 1, 2 ** 100))


def gap(lay: dict) -> dict:
    """`Gap.lean` with `ChunksOf`, exact: padded stock and words, `aBlocks` over `rows`. The network term is 0.036% of
    stock a 32 bits of cap at dd30c5c347, and `betaLead · callBits / 32` from ebe1fcd968."""
    m, n, k, s, cap = lay["m"], lay["n"], lay["K"], lay["S"], lay["cap"]
    chunks = cs.word_chunks(lay)
    stock = Q(m * n * k, 32)
    words = Q(m * n * s)
    sha = Q(351538, 100) * (4 * words / 128 + chunks) + Q(344905, 100) * Q(lay["rows"] * k, 128)
    honest = words + sha
    if HEAD["network"] == "stock":
        network = Q(36, 100000) * Q(cap, 32) * stock
    else:
        beta = N_MAX * 32 * honest / (m * n * k)
        network = beta * Q(cap * lay["rows"] * n * k, FWD_MACS) / 32
    terms = {"head": 23 * Q(chunks), "hiding": Q(16064, 1000000) * Q(n * k, 32), "coinBranch": Q(84, 100000) * stock,
             "pStar": Q(35, 1000000) * stock, "network": network, "table": table_gap(lay)}
    g = sum(terms.values())
    return {"stock": stock, "words": words, "sha": sha, "honest": honest, "gap": g, "terms": terms,
            "gapShare": g / stock, "chunks": chunks}


def closed_share(m: int, cap: int, rows: int | None = None) -> Q:
    """An upper bound on `gapShare` for every n, K ≥ 4096 at padded rows `m` (A's rows `rows`, `m` if None):
    `28 S ≤ K + 31` (dd30c5c347) or `K + 34` (ebe1fcd968), at most one partial chunk, and
    `tableGap / stock = 32 · (P0 + (S − 1)(2 mem + 1)/2^100) / K`. From ebe1fcd968 the network term over stock is
    `n_max · cap · (rows/m) / fwdMacs` times `honest / stock`, bounded the same way."""
    rows = m if rows is None else rows
    sk = Q(1, 28) + Q(HEAD["sk_num"], 28 * 4096)
    if HEAD["network"] == "stock":
        network = Q(36, 100000) * Q(cap, 32)
    else:
        honest = (32 * sk * (1 + Q(351538, 3200)) + Q(351538, 100) * 32 * (sk / 65536 + Q(1, m * 4096 * 4096))
                  + Q(344905, 100) * Q(rows, 4 * 4096 * m))
        network = N_MAX * Q(cap * rows, m * FWD_MACS) * honest
    p0 = 0 if HEAD.get("table") == "lanes" else 32 * P0 / 4096
    return (Q(84, 100000) + Q(35, 1000000) + network + Q(16064, 1000000) / m
            + Q(23 * 32, 65536) * sk + Q(23 * 32, m * 4096 * 4096)
            + p0 + 32 * sk * Q(2 * MEM + 1, 2 ** 100))


GAP_POINTS = {"pt512x64": (512, 64, Q(190, 100000), Q(514, 100000)),
              "pt512x512": (512, 512, Q(168, 100000), Q(492, 100000)),
              "pt8192": (8192, 8192, Q(165, 100000), Q(489, 100000))}


def gap_at(ns: list[int], ks: list[int]) -> dict:
    out = {}
    for p, (name, (t, m, lim32, lim320)) in enumerate(GAP_POINTS.items()):
        lim320 = HEAD.get("lim320", (None,) * 3)[p] or lim320
        for cap, lim in ((32, lim32), (320, lim320)):
            worst, arg = Q(0), None
            for n in ns:
                for k in ks:
                    sh = gap(layout(t, m, n, k, cap))["gapShare"]
                    if sh > worst:
                        worst, arg = sh, (n, k)
            closed = closed_share(pad_rows(m), cap)
            out[f"{name}/cap{cap}"] = {"claimed": float(lim), "scan_max": float(worst), "at_n_K": arg,
                                       "closed_form": float(closed), "holds": bool(worst <= lim and closed <= lim),
                                       "margin_closed": float(lim - closed), "margin_scan": float(lim - worst)}
    return out


def gap_at_decode(ns: list[int], ks: list[int]) -> dict:
    """`GapAtDecode` at each of the head's caps: the share of the padded stock and gapRegs over the unpadded
    `n K / 32` against the claimed bounds. T is not read."""
    out = {}
    for cap, lim_share, lim_unpad in HEAD["decode"]:
        worst_share, worst_unpad, arg_s, arg_u = Q(0), Q(0), None, None
        for n in ns:
            for k in ks:
                g = gap(layout(1, 1, n, k, cap))
                un = g["gap"] / Q(n * k, 32)
                if g["gapShare"] > worst_share:
                    worst_share, arg_s = g["gapShare"], (n, k)
                if un > worst_unpad:
                    worst_unpad, arg_u = un, (n, k)
        closed = closed_share(16, cap, 1)
        closed_un = closed * 16 * Q(4104, 4097)
        out[f"cap{cap}"] = {
            "claimed_share": float(lim_share), "claimed_unpadded": float(lim_unpad),
            "share_scan_max": float(worst_share), "share_at": arg_s, "unpadded_scan_max": float(worst_unpad),
            "unpadded_at": arg_u, "share_closed": float(closed), "unpadded_closed": float(closed_un),
            "holds_share": bool(worst_share <= lim_share and closed <= lim_share),
            "holds_unpadded": bool(worst_unpad <= lim_unpad and closed_un <= lim_unpad),
            "margin_share_closed": float(lim_share - closed), "margin_unpadded_closed": float(lim_unpad - closed_un)}
    return out


def decode_failures(ns: range, ks: range) -> dict:
    """At each cap, the calls `lead T 1 n K` whose gap exceeds the claimed share of the unpadded `n K / 32`, exact.
    A column count whose closed bound is under the claim by 0.01 points is skipped."""
    out = {}
    for cap, _, lim in HEAD["decode"]:
        fail, worst, arg = [], Q(0), None
        c16 = 16 * float(closed_share(16, cap, 1))
        skipped = 0
        for n in ns:
            if c16 * pad_cols(n) / n <= float(lim) - 1e-4:
                skipped += len(ks)
                continue
            for k in ks:
                g = gap(layout(1, 1, n, k, cap))
                un = g["gap"] / Q(n * k, 32)
                if un > lim:
                    fail.append([n, k, float(g["gap"]), float(lim * Q(n * k, 32))])
                if un > worst:
                    worst, arg = un, (n, k)
        residues = sorted({n % 8 for n, _, _, _ in fail})
        out[f"cap{cap}"] = {
            "claimed": float(lim), "n_range": [ns.start, ns.stop - 1], "K_range": [ks.start, ks.stop - 1],
            "calls": len(ns) * len(ks), "skipped_below_bound": skipped, "failing_count": len(fail),
            "n_mod_8_failing": residues, "n_max_failing": max((f[0] for f in fail), default=None),
            "worst": float(worst), "worst_at": arg, "worst_exact": str(worst), "failing": fail[:40]}
    return out


def nontrivial(ns: list[int], ks: list[int]) -> dict:
    out = {}
    for name, (t, m, _, _) in GAP_POINTS.items():
        ratio = max(gap(layout(t, m, n, k))["gap"] / gap(layout(t, m, n, k))["honest"] for n in ns for k in ks)
        out[name] = {"max_gap_over_honest": float(ratio), "holds": bool(ratio < 1)}
    return out


def _rank(rows: list[list[int]]) -> int:
    return int(np.linalg.matrix_rank(np.array(rows, dtype=float))) if rows else 0


def step0_support(rb: int, budget: int = 32) -> dict:
    """The 16-row block `rb` at step 0. Its tile's column `j` is `T · G_j + a₁ · C_j` over the block's tag matrix `T`
    (16 × 4). Given every coin but the block's 32 `G` coins, the tile is a linear map of them, and on the box
    `[−3, 3]^32` a map of rank `ρ` has fibers of at most `7^(32 − ρ)`: a value has chance at most `7^-ρ`.

    - A constant tile (a pre-coin table's, zero, or a stream tile holding none of the block's `G`): `ρ = 8 rank T`.
    - A stream tile holding some of them: each position it gives a `G` coin drops that row from the column's
      equations, at most 32 positions. `ρ` is at least the least `Σ_j rank T[R_j]` over the rows `R_j` left in each
      column, which `min_rank` finds (by how many rows each column loses to fall to rank `r`)."""
    tags = [cs.row_tag(16 * rb + r) for r in range(16)]
    rank = _rank(tags)
    keep = [sum(1 for t in tags if not any(t))]
    for r in range(1, rank):
        best = keep[0]
        nz = [t for t in tags if any(t)]
        for span in itertools.combinations(range(len(nz)), r):
            base = [nz[i] for i in span]
            if _rank(base) == r:
                best = max(best, sum(1 for t in tags if _rank(base + [t]) == r))
        keep.append(best)
    keep.append(16)
    cost = [16 - k for k in keep]
    reach = {0: 0}
    for _ in range(8):
        nxt: dict[int, int] = {}
        for spent, tot in reach.items():
            for r, c in enumerate(cost):
                if spent + c <= budget:
                    nxt[spent + c] = min(nxt.get(spent + c, 10 ** 9), tot + r)
        reach = nxt
    min_rank = min(reach.values())
    return {"rb": rb, "rank": rank, "rows_to_drop_to_rank": cost, "const_log7": 8 * rank, "stream_log7": min_rank,
            "ok_const": rank >= 2, "ok_stream": min_rank >= 8}


def step0_scan(rb_lo: int, rb_hi: int) -> dict:
    bad, ranks, stream = [], {}, {}
    for rb in range(rb_lo, rb_hi):
        r = step0_support(rb)
        ranks[r["rank"]] = ranks.get(r["rank"], 0) + 1
        stream[r["stream_log7"]] = stream.get(r["stream_log7"], 0) + 1
        if not (r["ok_const"] and r["ok_stream"]):
            bad.append(r)
    return {"blocks": [rb_lo, rb_hi], "rows": [16 * rb_lo, 16 * rb_hi], "failures": len(bad), "first_failures": bad[:6],
            "rank_counts": ranks, "stream_log7_counts": stream}


def lane_injective(e: int = 7) -> dict:
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2, dtype=np.int64)
    s0, s1, s2 = np.meshgrid(odd, odd, odd, indexing="ij")
    key = (127 * s0 + s1 + s2) * 70000 + (s0 + 127 * s1 + s2)
    return {"triples": int(key.size), "distinct": int(np.unique(key).size), "injective": bool(np.unique(key).size == key.size)}


def step0_pmax_block0() -> dict:
    """Tile (0, 0) at step 0 reads `G[0:2, 0:8]` and `a₁[0:16, 0:2]` (2^76.9 configurations). A second preimage
    `(G + ΔG, a₁ + Δa)` of a value has `ΔG[0, :] = 0` and `Δa = 0` on rows 0–11 (rows with z = ±2 force it at
    columns 0 and 6), and on rows 12–15 `Δa = (δ, −δ, δ, −δ)` for one `δ ∈ {−1, 0, 1}²` with `ΔG[1, j] = −δ · C_j`
    (`C_0` and `C_6` span ℤ²). A bit takes one sign of `δ_q` only, and `G[1, j]` stays in `[−3, 3]` for a set of
    shifts iff their `δ · C_j` spread at most 6. The largest such set is the most preimages a value has."""
    code = [cs.col_code(j) for j in range(8)]
    worst, arg = 1, None
    for s0, s1 in itertools.product((-1, 1), repeat=2):
        cand = [(0, 0), (s0, 0), (0, s1), (s0, s1)]
        for size in range(4, 1, -1):
            for sub in itertools.combinations(cand[1:], size - 1):
                deltas = [(0, 0), *sub]
                if all(max(d[0] * code[j][0] + d[1] * code[j][1] for d in deltas)
                       - min(d[0] * code[j][0] + d[1] * code[j][1] for d in deltas) <= 6 for j in range(8)):
                    if size > worst:
                        worst, arg = size, deltas
    bits = 16 * math.log2(7) + 32
    return {"configurations_log2": bits, "max_preimages": worst, "at_shifts": arg,
            "p_max_log2": math.log2(worst) - bits, "bound_49_8_log2": -8 * math.log2(49),
            "slack_log2": math.log2(worst) - bits + 8 * math.log2(49)}


def p_star() -> dict:
    """`pStar_eq`: the largest count of `127 l₀ + l₁ + l₂ + 1 + b` over the `2 · 128³` draws, exact."""
    lane = np.arange(128, dtype=np.int64) * 2 - 127
    x = np.zeros(127 * 254 + 1, dtype=np.int64)
    np.add.at(x, 127 * (lane + 127), 1)
    y = np.bincount(lane + 127)
    tot = np.convolve(np.convolve(x, np.convolve(y, y)), np.array([1, 1], dtype=np.int64))
    return {"max_count": int(tot.max()), "draws": int(tot.sum()), "holds": bool(tot.max() == 130),
            "values_at_max": int((tot == tot.max()).sum())}


def word_exact(k: int = 65536) -> dict:
    """|word| at `K` under `DataInRange`, any m and n: β, step 0's |T·G| + |a₁·C| ≤ 72 + 24 (balanced digits ≤ 6),
    and `S − 1` later steps of at most 2 + 129 · 127, each with |κ| ≤ 28 · 127²."""
    s = cs.steps_for(k)
    beta = max(abs(cs.bias(s, j)) for j in range(28561))
    kap = 28 * 127 ** 2
    lanes0 = 129 * 127 if HEAD["step0_lanes"] else 0
    worst = beta + s * kap + 72 + 24 + lanes0 + (s - 1) * (2 + 129 * 127)
    return {"K": k, "S": s, "worst": worst, "log2": math.log2(worst), "holds": bool(worst < 2 ** 31)}


def cover(m: int, n: int, k: int) -> dict:
    """The cover strategy on the padded call. No tile has fewer than 128 coin-carrying words, so its tables (and any
    stream) hold a committed tile with chance at most `(2 mem + 1) · 7^(−8 |D|)` at step 0 (`step0_support`) and
    `(2 mem + 1) · 2^-105` later: its saving, in registers averaged over the coins, is at most that times 128 a tile.
    From ebe1fcd968 step 0 carries the coin lanes too, so a step-0 tile is also at most `2^-105` (`lane_injective`
    given every coin but the block's 24 step-0 lane coins, which a stream tile holds at 24 positions at most)."""
    lay = layout(1, m, n, k)
    g = gap(lay)
    rb_n, cb_n = lay["m"] // 16, lay["n"] // 8
    per_block = Q(0)
    for rb in range(rb_n):
        sup = step0_support(rb)
        assert sup["ok_const"] and sup["ok_stream"], sup
        pc, ps = Q(1, 7 ** sup["const_log7"]), Q(1, 7 ** sup["stream_log7"])
        if HEAD["step0_lanes"]:
            pc, ps = min(pc, Q(1, 2 ** 105)), min(ps, Q(1, 2 ** 105))
        per_block += cb_n * ((2 * MEM + 1) * pc + 32 * ps)
    saved = cs.TILE_WORDS * (per_block + rb_n * cb_n * (lay["S"] - 1) * Q(2 * MEM + 1, 2 ** 105))
    work = cs.PRICE * (g["honest"] - saved)
    bound = cs.PRICE * (g["honest"] - g["gap"])
    return {"padded": [lay["m"], lay["n"]], "tiles_a_step": rb_n * cb_n, "words_a_tile_with_coins": 128,
            "cover_saving_bound_regs": float(saved), "table_gap_regs": float(g["terms"]["table"]),
            "gap_regs": float(g["gap"]), "work_lower": float(work), "G_bound": float(bound),
            "breaks": bool(work < bound), "gapShare": float(g["gapShare"])}


def sample_tile(m: int, n: int, k: int, t: int, draws: int, seed: int) -> dict:
    lay = layout(1, m, n, k)
    rng = np.random.default_rng(seed)
    seen = set()
    for _ in range(draws):
        om = cs.draw(lay, rng)
        w = cs.words_at(lay, om, np.arange(16), np.arange(8))
        seen.add(w[:, t, :].tobytes())
    return {"padded": [lay["m"], lay["n"]], "step": t, "draws": draws, "distinct": len(seen)}


def repeats(m: int, n: int, k: int, seeds: list[int]) -> dict:
    lay = layout(1, m, n, k)
    out = [ct.block_dedup(lay, s) for s in seeds]
    return {"padded": [lay["m"], lay["n"]], "K": k, "S": lay["S"], "tiles": [o["tiles"] for o in out],
            "repeats": [o["repeats"] for o in out]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rb-max", type=int, default=1785)
    ap.add_argument("--wrap-rb", type=int, default=7200)
    ap.add_argument("--samples", type=int, default=2 ** 16)
    ap.add_argument("--decode-n-max", type=int, default=6100)
    ap.add_argument("--repeat-calls", nargs="*", default=["1x9x4096", "17x4097x220", "1x4096x4096", "64x4096x4096",
                                                          "512x512x4096"])
    ap.add_argument("--head", choices=sorted(HEADS), default="ebe1fcd968")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    use_head(args.head)
    rep: dict = {"head": dict(HEAD, decode=[[c, float(a), float(b)] for c, a, b in HEAD["decode"]]),
                 "port": cs.check_port()}
    print("head:", rep["head"], "port:", rep["port"], flush=True)
    assert rep["port"]["mismatches"] == 0

    rep["cover"] = {f"m{m}n{n}K{k}": cover(m, n, k) for m, n, k in
                    ((1, 1, 24), (1, 1, 700), (1, 1, 4096), (1, 1, 8192), (1, 9, 4096), (2, 1, 4096), (1, 4096, 4096),
                     (1, 4097, 4113), (17, 4097, 4096))}
    for key, v in rep["cover"].items():
        print(f"cover {key}: padded {v['padded']}, saves ≤ {v['cover_saving_bound_regs']:.4g} regs (tableGap "
              f"{v['table_gap_regs']:.4g}), gap {v['gap_regs']:.4g}, breaks={v['breaks']}", flush=True)

    rep["step0_pmax_block0"] = step0_pmax_block0()
    print("step-0 p_max at block 0:", rep["step0_pmax_block0"], flush=True)
    rep["step0_support"] = step0_scan(0, args.rb_max)
    print("step-0 bound, rows below 28,560:", rep["step0_support"], flush=True)
    wrap = step0_scan(args.rb_max, args.wrap_rb)
    rep["step0_support_wrapped"] = wrap
    print(f"step-0 bound, rows {wrap['rows']}: failures {wrap['failures']}, first "
          f"{[(f['rb'], f['rank'], f['stream_log7']) for f in wrap['first_failures']]}, ranks {wrap['rank_counts']}",
          flush=True)
    rep["lane_injective"] = lane_injective()
    print("lane map injective:", rep["lane_injective"], flush=True)
    rep["sample"] = [sample_tile(1, 1, 4096, t, args.samples, 11 + t) for t in (0, 1)]
    print("tile (0,0) samples at m 1 n 1:", rep["sample"], flush=True)

    ns = list(range(4096, 4096 + 17)) + [4103, 4104, 4105, 8191, 8192, 8193, 16384]
    ks = list(range(4096, 4096 + 57)) + [4113, 4141, 5000, 8192, 8193, 16384, 65536]
    rep["gap_at"] = gap_at(ns, ks)
    for key, v in rep["gap_at"].items():
        print(f"GapAt {key}: claimed {100 * v['claimed']:.3f}%, scan {100 * v['scan_max']:.5f}% at {v['at_n_K']}, "
              f"closed {100 * v['closed_form']:.5f}%, holds={v['holds']}", flush=True)
    rep["gap_at_decode"] = gap_at_decode(ns, ks)
    for cap, v in rep["gap_at_decode"].items():
        print(f"GapAtDecode {cap}:", v, flush=True)
    rep["decode_failures"] = decode_failures(range(4096, args.decode_n_max), range(4096, 4096 + 28 * 12))
    for cap, v in rep["decode_failures"].items():
        print(f"GapAtDecode {cap}'s unpadded conjunct, failing calls:",
              {k: x for k, x in v.items() if k != "failing"}, flush=True)
    rep["nontrivial"] = nontrivial([4096, 4097], [4096, 4113])
    print("Nontrivial:", rep["nontrivial"], flush=True)
    rep["p_star"] = p_star()
    print("pStar_eq:", rep["p_star"], flush=True)
    rep["word_exact"] = word_exact()
    print("word_exact:", rep["word_exact"], flush=True)
    for name, (t, m, _, _) in GAP_POINTS.items():
        g = gap(layout(t, m, 4096, 4096))
        rep[f"terms_{name}"] = {k: float(v) for k, v in g["terms"].items()} | {"share": float(g["gapShare"])}
        print(f"terms {name} n=K=4096:", {k: f"{float(v / g['stock']) * 100:.6f}%" for k, v in g["terms"].items()},
              flush=True)

    rep["repeats"] = {}
    for call in args.repeat_calls:
        m, n, k = (int(x) for x in call.split("x"))
        rep["repeats"][call] = repeats(m, n, k, args.seeds)
        print(f"repeats m{m} n{n} K{k}:", rep["repeats"][call], flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
