"""N-EGRESS: coin layouts whose network term is at most 1% of stock, priced on N-CARD's basis.

DitherInstance's guarantee (Security/Definitions/Pouw/Guarantees/DitherInstance.lean) charges n_max W_mac E in its
slack: E the bits the network certifier counts (capped at served traffic, the coin stream excluded), W_mac the honest
block's priced cost a MAC (its hashing and answers included) and n_max the most MACs one exchanged bit carries, through
OutsourceWithinEgress at beta = n_max W_mac. Here beta is the most priced work one bit carries over every step interval
of every call (a word costs the same whatever its data lanes, so it peaks where the words a bit do), the coin bits
counted as the words read them; as a share of stock (W1: a MAC 1, a register write 32) the term is beta E / MACs.
Each layout is priced on Llama-3.2-1B's linears at each point's rows: its coin bits, n_max, beta, the
term at each cap and hashing price, the outsourcing provers that realize it (a call set when the served path is
coin-free, a prefix of the forward when it is noisy), the coins' forming on the GPU, the certifier's stream, T2 and the
served noise for T3, with whether N-EXACT's epilogue fold serves it exactly, and N-RED's A2c / A8 (one chosen data lane
cancelling a step's likeliest coin increment) under model P at each point, with N-ADAPT's walk on a small tile's runs
of steps without lane coins. Inputs: ``egress_inputs.json`` beside
``alt_roads_inputs.json``.
Writes ``egress.json`` and ``table.txt`` to ``--out``; exits non-zero if a self-check fails.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
from fractions import Fraction
from pathlib import Path

import alt_roads

HERE = Path(__file__).resolve().parent
POINTS = {"decode1": None, "decode16": None, "t64": "t512", "t512": "t512r512", "t8192": "t8192"}
LEAD_T2 = "ssn-1l-tag/served/s8/k32/w/r256k/cert/min"
COORDINATOR_LEAD = "blc4a-t64-e7"


def steps(k: int, d1: int, dl: int = 31) -> int:
    return 1 + math.ceil(max(0, k - d1) / dl)


def row_bits(k: int, d1: int, c: int) -> int:
    return 32 * math.ceil((steps(k, d1) - 1 + c) / 32)


def balanced_digits(z: int, base: int, count: int) -> list[int]:
    h, out = base // 2, []
    for _ in range(count):
        d = (z + h) % base - h
        out.append(d)
        z = (z - d) // base
    if z:
        raise ValueError("tag out of range")
    return out


def tag_rank(m: int, lead: dict) -> int:
    """The rank of the row tags T(i), the balanced base-b digits of z_i = (floor(i/2) + 1)(-1)^i, over a call's m rows:
    the digits up to the largest |z_i| = floor((m - 1)/2) + 1. Step 1's words read G[., j] only through T(i) G[., j],
    an injective function of G's first rank digits, so they read rank log2(digit values) bits a column."""
    if not lead["tag_lanes"]:
        return 0
    zmax, b, r = (m - 1) // 2 + 1, lead["tag_row_digit_base"], 1
    while (b ** r - 1) // 2 < zmax:
        r += 1
    if r > lead["tag_lanes"]:
        raise ValueError(f"{m} rows: T isn't injective in {lead['tag_lanes']} digits")
    return r


def exact_rank(rows: list[list[int]]) -> int:
    basis: list[list[Fraction]] = []
    for row in {tuple(r) for r in rows}:
        v = [Fraction(x) for x in row]
        for b in basis:
            p = next(i for i, x in enumerate(b) if x)
            if v[p]:
                v = [x - v[p] / b[p] * y for x, y in zip(v, b)]
        if any(v):
            basis.append(v)
    return len(basis)


def tag_bits(m: int, n: int, lead: dict) -> float:
    return n * tag_rank(m, lead) * math.log2(lead["tag_digit_values"]) if lead["tag_lanes"] else 0.0


def odd_mean_square(e: int) -> float:
    """E[s^2] for s uniform over the odd values in +-(2^e - 1)."""
    return (4 ** e - 1) / 3.0


def uniform_mean_square(bits: int) -> float:
    """E[r^2] for r uniform over the signed bits-bit integers."""
    h = 2 ** (bits - 1)
    return sum(v * v for v in range(-h, h)) / (2 * h)


def spread(steps_: int, coined: int) -> list[bool]:
    """Which of a call's steps carry lane coins when coined of them do: step 1, then evenly, so a run of steps without
    them is at most ceil(steps_ / coined) - 1 long."""
    return [t * coined // steps_ > (t - 1) * coined // steps_ for t in range(steps_)]


def lane_counts(design: dict, steps_: int, frac: float, la: int) -> list[int]:
    """blc1a / blc4a's lane coins a step: ceil(steps_ frac) steps' worth at la a step (a tile of frac T rows), on step
    1 and spread evenly, one lane or two a step where la > 1 so that no run of steps without them is longer than it
    must be. Mutations: interim 2's layout (step1_coins False), the coined steps first ("front"), la lanes a coined
    step ("packed"). pack_rows: la lanes a coined step on a tile of at least that many rows (the row count is
    public), spread below it."""
    if not design.get("step1_coins", True):
        return [0] + [la * p for p in coin_plan(steps_ - 1, math.ceil((steps_ - 1) * frac))]
    coined = math.ceil(steps_ * frac)
    if design.get("plan") == "front":
        return [la * (t < coined) for t in range(steps_)]
    if design.get("lanes") == "packed" or frac * design["tile"] >= design.get("pack_rows", math.inf):
        return [la * p for p in spread(steps_, coined)]
    base, extra = divmod(la * coined, steps_)
    return [base + p for p in spread(steps_, extra)]


def stepped(k: int, d1: int, dl: int, bits: list[float], lane: list[bool]) -> dict:
    d, cum = 0, []
    for t in range(len(bits)):
        d = min(k, d + (d1 if t == 0 else dl))
        cum.append(d)
    return {"D": cum, "sb": bits, "lane": lane, "bits": sum(bits)}


def call(design: dict, m: int, k: int, n: int, lead: dict, lay: dict, reg: float) -> dict:
    """One call's coins. Per step: its data lanes (cumulative, D), the coin bits its words add (sb: row coins at the
    bits the words read, lane coins at e bits an entry, the window coin, drawn once a window, at none) and whether it
    carries lane coins; with the certifier's stream in bytes (row coins in whole 32-bit words), forming in W1 units,
    words against the lead's and the served noise rms. off_lane and off_elem are the fewest bits the accumulators' coin
    state takes after a lane-coined step: e bits a (tile, column), a sum of at least one e-bit odd coin, and an init
    coin's e bits an element. design["step1_coins"] False is interim 2's layout: step 1 without lane coins, a small
    tile's coined later steps spread evenly; "front" puts them on its first steps. The tag lead's step 1 holds
    step1_data_lanes data lanes, tag_lanes of the row tag T(i) against the call's column coin G (one coin word a column
    in the certifier's stream, read by the words at tag_bits) and code_lanes of the row's code bits; code_bits, the
    state step 1 leaves, holds both."""
    c, d1 = lead["code_lanes"], lead["step1_data_lanes"]
    s, rb = steps(k, d1), row_bits(k, d1, c)
    g, gw = tag_bits(m, n, lead), (n if lead["tag_lanes"] else 0)
    kind = design["kind"]
    out = {"bytes": m * rb / 8.0 + gw * lead["tag_coin_word_bytes"], "forming": (m * rb / 32.0 + gw) * reg,
           "words": 1.0, "noise_rms": 0.0, "lead_steps": s, "code_bits": m * c + g, "tag_bits": g,
           "off_lane": 0.0, "off_elem": 0.0}
    if kind == "rows":
        return {**out, **stepped(k, d1, 31, [m * c + g] + [m] * (s - 1), [False] * s),
                "later": s - 1, "coin_steps": s - 1, "plan": [True] * (s - 1)}
    if kind == "blc1":
        tile = design["tile"] or m
        tiles, coin_steps = math.ceil(m / tile), math.ceil((s - 1) * min(1.0, m / tile))
        plan = coin_plan(s - 1, coin_steps)
        entries = tiles * n * coin_steps
        return {**out, **stepped(k, d1, 31, [m * c + g] + [m + (tiles * n * design["e"] if p else 0) for p in plan],
                                 [False] + plan),
                "bytes": out["bytes"] + entries, "forming": out["forming"] + entries / 4.0 * reg, "entries": entries,
                "later": s - 1, "coin_steps": coin_steps, "plan": plan, "off_lane": tiles * n * design["e"],
                "noise_rms": math.sqrt(2.5 * coin_steps * odd_mean_square(design["e"]))}
    if kind in ("blc1a", "blc4a"):
        la, dl = (1, lay["lane_add"]["data_lanes"]) if kind == "blc1a" else \
            (lay["lanes4"]["lanes"] - 1, lay["lanes4"]["data_lanes"])
        d1 = d1 - la if design.get("step1_coins", True) else d1
        sa = steps(k, d1, dl)
        tiles = math.ceil(m / design["tile"])
        lc = lane_counts(design, sa, min(1.0, m / design["tile"]), la)
        lane = [x > 0 for x in lc]
        plan, coined = lane[1:], sum(lane)
        coin_steps = sum(plan)
        fill, ef = sa - 1 - coin_steps, lay["lane_add"]["fill_row_coin_bits"]
        rba = 32 * math.ceil((coin_steps + fill * max(ef, 1) + c) / 32)
        cells, entry_bits = tiles * n * coined, tiles * n * design["e"]
        corr = 4.0 * tiles * n if coined > 1 else 0.0
        lane_bytes = cells if kind == "blc1a" else cells * lay["lanes4"]["lanes"]
        sb = [m * c + g + lc[0] * entry_bits] + [m + x * entry_bits if x else m * max(ef, 1) for x in lc[1:]]
        return {**out, **stepped(k, d1, dl, sb, lane),
                "bytes": m * rba / 8.0 + gw * lead["tag_coin_word_bytes"] + lane_bytes + corr,
                "forming": (m * rba / 32.0 + gw) * reg + (cells / 4.0 * reg if kind == "blc1a" else 0.0),
                "words": sa / s, "entries": lane_bytes, "later": sa - 1, "coin_steps": coin_steps, "plan": plan,
                "fill": fill, "off_lane": entry_bits, "min_lanes": min(x for x in lc if x),
                "word_noise_rms": math.sqrt(sum(lc) * odd_mean_square(design["e"]) + (
                    fill * (uniform_mean_square(ef) + odd_mean_square(lay["lane_add"]["fill_window_coin_bits"]))
                    if ef else 0.0))}
    if kind == "blc4":
        l4, d4 = lay["lanes4"]["lanes"], lay["lanes4"]["data_lanes"]
        s4 = steps(k, d1, d4)
        tiles, coin_steps = math.ceil(m / design["tile"]), math.ceil((s4 - 1) * min(1.0, m / design["tile"]))
        plan = coin_plan(s4 - 1, coin_steps)
        entries = tiles * n * l4 * coin_steps
        return {**out, **stepped(k, d1, d4,
                                 [m * c + g] + [m + (tiles * n * l4 * design["e"] if p else 0) for p in plan],
                                 [False] + plan),
                "bytes": out["bytes"] + entries, "words": s4 / s, "entries": entries,
                "off_lane": tiles * n * design["e"],
                "noise_rms": math.sqrt((2.5 + (l4 - 1)) * (s4 - 1) * odd_mean_square(design["e"]))}
    if kind == "cinit":
        e = math.ceil(k / lay["cinit_n_target"])
        return {**out, **stepped(k, d1, 31, [m * c + g + m * n * e] + [m] * (s - 1), [False] * s),
                "bytes": out["bytes"] + m * n * math.ceil(e / 8), "forming": out["forming"] + m * n * reg,
                "off_elem": m * n * e, "noise_rms": math.sqrt((4 ** e - 1) / 12.0)}
    if kind == "dither_a":
        b = m * k * design["bits_per_entry"]
        return {**out, **stepped(k, k, k, [b], [False]), "bytes": b / 8.0, "forming": float("nan"),
                "words": design.get("words", 1.0), "wd": design.get("words", 1.0), "noise_rms": float("nan")}
    raise ValueError(kind)


def intervals(c: dict, m: int, n: int, k: int):
    """Every step interval [t0, t1] of a call, as (t0, t1, MACs, coin bits, words a MAC against the lead's): a word of
    step t is a k32 partial sum, so the words of [t0, t1] read the interval's coins and the accumulators' coin state
    before t0, which is none at t0 = 1 and otherwise at least the rows' code bits, off_elem, and off_lane once a step
    before t0 carried lane coins (a lower count, so the outsourcer's side). Whole calls and step prefixes are among
    them; a set of rows or columns reads a tile's or a row's coins whole, so it carries no more a bit."""
    d0, p, seen = [0] + c["D"], [0.0], False
    for b in c["sb"]:
        p.append(p[-1] + b)
    for t0 in range(len(c["D"])):
        off = 0.0 if t0 == 0 else c["code_bits"] + c["off_elem"] + (c["off_lane"] if seen else 0.0)
        for t1 in range(t0, len(c["D"])):
            lanes = d0[t1 + 1] - d0[t0]
            yield (t0 + 1, t1 + 1, m * n * lanes, off + p[t1 + 1] - p[t0],
                   c.get("wd") or (t1 - t0 + 1) * k / (lanes * c["lead_steps"]))
        seen |= c["lane"][t0]


def designs(lay: dict) -> list[dict]:
    out = [{"id": "rows", "kind": "rows",
            "title": "the lead's row coins streamed raw (R1b): ssn-1l-tag's S - 1 + c bits a row in whole words"},
           {"id": "dither-a1", "kind": "dither_a", "bits_per_entry": 1,
            "title": "the locked Dither instance's layout: one coin bit an A entry (n_max = N), for reference"},
           {"id": "os2", "kind": "dither_a", "bits_per_entry": 3, "words": 2.0,
            "title": "SW-NOISE's os2: three coin bits an A entry, alpha 2, for reference"},
           {"id": "cinit", "kind": "cinit",
            "title": f"an accumulator-init coin an output element, ceil(K/{lay['cinit_n_target']}) bits, one init write"},
           {"id": "blc1-call-e1", "kind": "blc1", "tile": None, "e": 1,
            "title": "the B' lane's coins a (step, column) a call, shared by every row (ssn2-like two-sided)"}]
    for t in lay["tile_rows"]:
        for e in lay["lane_entropy_bits"]:
            out.append({"id": f"blc1-t{t}-e{e}", "kind": "blc1", "tile": t, "e": e,
                        "title": f"the lead's B' lane, a {e}-bit odd coin a ({t}-row tile, later step, column), "
                                 "patched"})
    for t in lay["tile_rows"]:
        for e in lay["lane_entropy_bits"][1:]:
            out.append({"id": f"blc1a-t{t}-e{e}", "kind": "blc1a", "tile": t, "e": e,
                        "title": f"a coin lane against A' = 1 beside the lead's row lane (30 data lanes; 23 on step 1, "
                                 f"beside its tag and code lanes), a {e}-bit odd coin a ({t}-row tile, step, column), on step 1 "
                                 "and spread evenly, patched, served exactly"})
    for e in (2, 7):
        out.append({"id": f"blc4-t64-e{e}", "kind": "blc4", "tile": 64, "e": e,
                    "title": f"four B' coin lanes a later step (28 data lanes), {e}-bit odd coins a (64-row tile, "
                             "step, column), cp.async"})
    for t, e in ((64, 4), (64, 7), (128, 7)):
        out.append({"id": f"blc4a-t{t}-e{e}", "kind": "blc4a", "tile": t, "e": e,
                    "title": f"three coin lanes against A' = 1 beside the lead's row lane (28 data lanes; 21 on step "
                             f"1), {e}-bit odd coins a ({t}-row tile, step, column), on step 1 and spread evenly, one "
                             "lane or two a step on a small tile, one 4-byte cp.async with the public weight, served "
                             "exactly"})
    out.append({"id": "blc4a-t64-e7-packed", "kind": "blc4a", "tile": 64, "e": 7, "lanes": "packed",
                "title": "blc4a-t64-e7 with all three lanes on each of a small tile's coined steps (fewer cp.asyncs at "
                         "decode, longer runs without lane coins)"})
    out.append({"id": "blc4a-t64-e7-pack16", "kind": "blc4a", "tile": 64, "e": 7, "pack_rows": 16,
                "title": "blc4a-t64-e7 with three lanes on each coined step of a tile of 16 rows or more, and spread "
                         "below 16 rows"})
    return out


SERVED = {"rows": "either", "blc1a": "exact", "blc4a": "exact"}


def lane_sum(lanes: int, e: int) -> dict[int, float]:
    odd = range(-(2 ** e - 1), 2 ** e, 2)
    dist = {0: 1.0}
    for _ in range(lanes):
        nxt: dict[int, float] = {}
        for v, p in dist.items():
            for s in odd:
                nxt[v + s] = nxt.get(v + s, 0.0) + p / len(odd)
        dist = nxt
    return dist


def step_dists(design: dict, lay: dict, w: int, lanes: int | None = None) -> dict | None:
    """N-RED's A2c / A8 against a layout: the adversary fixes, before the coins, one data lane a later step (A = 1,
    B = -v*) so that a word repeats its previous step's whenever the step's coin increment is v*, the likeliest value
    of the step's type. The increments' laws: "coin", a step carrying lanes of the layout's lane coins (uniform odd in
    +-(2^e - 1)) beside the row lane (1 + b) w, b a uniform row bit and w public; "fill", a step of a tile with fewer
    rows than T that carries none (the lead's lane, or blc1a / blc4a's uniform row coin r_t(i) w plus the window coin
    c_t[j])."""
    kind, e = design["kind"], design.get("e", 1)
    if kind not in ("rows", "blc1", "blc1a", "blc4a"):
        return None
    lead = {w: 0.5, 2 * w: 0.5}
    if lanes is None:
        lanes = {"rows": 0, "blc1": 0, "blc1a": 1, "blc4a": lay["lanes4"]["lanes"] - 1}[kind]
    coin: dict[int, float] = {}
    for b in (0, 1):
        for v, p in lane_sum(lanes, e).items():
            for s, q in (lane_sum(1, e) if kind == "blc1" else {w: 1.0}).items():
                coin[(1 + b) * s + v] = coin.get((1 + b) * s + v, 0.0) + 0.5 * p * q
    ef = lay["lane_add"]["fill_row_coin_bits"]
    if kind in ("blc1a", "blc4a") and ef:
        h = 2 ** (ef - 1)
        fill = {}
        for c, q in lane_sum(1, lay["lane_add"]["fill_window_coin_bits"]).items():
            for r in range(-h, h):
                fill[r * w + c] = fill.get(r * w + c, 0.0) + q / (2 * h)
    else:
        fill = lead
    return {"coin": coin, "fill": fill}


def coin_plan(later: int, coin_steps: int) -> list[bool]:
    """Which later steps carry coins when a tile carries them in coin_steps of its later steps, spread evenly."""
    return [(u + 1) * coin_steps // later > u * coin_steps // later for u in range(later)]


def a2c_point(design: dict, lay: dict, w: int, calls: list, m: int, reg: float) -> dict | None:
    """A8 under model P at a point, N-RED's count (nred_round3.at_scale) in expectation: each kept entry-step saves a
    word at reg units, one pointer a row-step is charged anyway, over the stock MACs m k n. Per step type the adversary
    uses the cancelling lane only where it gains. For blc1a / blc4a also N-ADAPT's walk: at CopyGrain's chunks on a
    small tile's runs, and, for reference, at one word if a one-word copy were free (each entry's later increments
    are a walk of about the 8-bit row coin's spread), which only copy-floor excludes. A layout whose coined steps carry
    different numbers of lane coins is counted at its fewest, every coined step at the likeliest increment's law."""
    lanes = [c["min_lanes"] for _, _, c in calls if "min_lanes" in c]
    dists = step_dists(design, lay, w, min(lanes) if lanes else None)
    if dists is None:
        return None
    ps = {t: max(d.values()) for t, d in dists.items()}
    saved = macs = 0.0
    seen = set()
    for k, n, c in calls:
        counts = {"coin": c["coin_steps"], "fill": c["later"] - c["coin_steps"]}
        for t, cnt in counts.items():
            saved += max(0.0, reg * m * cnt * (n * ps[t] - 1.0))
            if cnt:
                seen.add(t)
        macs += m * k * n
    out = {"pstar": max(ps[t] for t in seen), "pstar_by_type": ps, "types": sorted(seen),
           "model_p_share": saved / macs}
    if design["kind"] in ("blc1a", "blc4a") and lay["lane_add"]["fill_row_coin_bits"]:
        out["walk_share"] = walk_share(design, lay, calls, m, reg, window=True)
        out["walk_share_without_window_coin"] = walk_share(design, lay, calls, m, reg, window=False)
        out["walk_share_grain1_free_copies"] = sum(
            reg * m * n * walk_revisits(c["later"], lay["lane_add"]["fill_row_coin_bits"])
            for k, n, c in calls) / sum(m * k * n for k, n, _ in calls)
    return out


def fill_runs(plan: list[bool]) -> list[int]:
    runs, cur = [], 0
    for coined in plan:
        if coined:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    return runs + [cur] if cur else runs


@functools.lru_cache(maxsize=None)
def walk_revisits(length: int, bits: int, trials: int = 3000, seed: int = 79) -> float:
    """N-ADAPT's walk on a run of row-coined steps: each step adds the row coin r (uniform over the signed bits-bit
    integers) and the adversary's data a constant alternating 0 / 1 that cancels r's mean, so a row's run is a
    driftless walk W. Returns the expected count of steps whose W equals an earlier one (the run's start included)."""
    rng = random.Random(seed + length)
    h = 2 ** (bits - 1)
    tot = 0
    for _ in range(trials):
        x, seen = 0, {0}
        for u in range(length):
            x += rng.randrange(-h, h) + (u & 1)
            tot += x in seen
            seen.add(x)
    return tot / trials


def walk_share(design: dict, lay: dict, calls: list, m: int, reg: float, window: bool) -> float:
    """The walk's words as a share of stock: on a revisiting row-step a row's vector repeats an earlier one on every
    column whose increment is row-constant, and those words are copied, not written (free at CopyGrain's chunks). With
    the window coin only the aligned chunks inside the columns the call's data cancels (a data lane's worth a step,
    the data chosen after the window coin) are row-constant; without it every column is."""
    dl = lay["lane_add"]["data_lanes"] if design["kind"] == "blc1a" else lay["lanes4"]["data_lanes"]
    g = lay["lane_add"]["copy_grain_words"]
    saved = macs = 0.0
    for k, n, c in calls:
        cols = (dl // g) * g if window else n
        saved += reg * m * cols * sum(walk_revisits(L, lay["lane_add"]["fill_row_coin_bits"])
                                      for L in fill_runs(c["plan"]))
        macs += m * k * n
    return saved / macs


def a8_words(design: dict, lay: dict, w: int, m: int, n: int, later: int, seed: int, d1: int,
             plan_shift: int = 0) -> dict:
    """A8 under model P word for word on a small instance (nred_round3.build_entries' adversary on this layout): chosen
    data, zero past step 1 except each later step's first lane with A = 1 and B = -v* of the step's type; honest
    words are the k32 partial sums of A B plus the step's coins (step 1's lane coins included); the adversary keeps an
    entry's previous word when the coin increment is v* and otherwise adds its data lane and the coins. plan_shift > 0
    mutates the adversary's coin plan. Returns word equality, the kept entry-steps, their exact mean and standard
    deviation."""
    rng = random.Random(seed)
    kind, tile, e = design["kind"], design["tile"], design["e"]
    la, dl = (1, lay["lane_add"]["data_lanes"]) if kind == "blc1a" else \
        (lay["lanes4"]["lanes"] - 1, lay["lanes4"]["data_lanes"])
    d1, ef = d1 - la, lay["lane_add"]["fill_row_coin_bits"]
    lc = lane_counts(design, later + 1, min(1.0, m / tile), la)
    lanes_u, plan = lc[1:], [x > 0 for x in lc[1:]]
    dists = step_dists(design, lay, w)
    vfill = max(dists["fill"], key=dists["fill"].get)
    vcoin = {x: max(dd, key=dd.get) for x in set(lanes_u) if x for dd in [step_dists(design, lay, w, x)["coin"]]}
    k = d1 + dl * later
    a = [[rng.randint(-60, 60) if x < d1 else 0 for x in range(k)] for _ in range(m)]
    bcols = [[rng.randint(-60, 60) if x < d1 else 0 for x in range(k)] for _ in range(n)]
    for u in range(later):
        lane = d1 + dl * u
        for i in range(m):
            a[i][lane] = 1
        for j in range(n):
            bcols[j][lane] = -(vcoin[lanes_u[u]] if plan[u] else vfill)
    odd = list(range(-(2 ** e - 1), 2 ** e, 2))
    h = 2 ** (max(ef, 1) - 1)
    s = [[[sum(rng.choice(odd) for _ in range(lanes_u[u])) for _ in range(n)] for u in range(later)]
         for _ in range(math.ceil(m / tile))]
    s1 = [[sum(rng.choice(odd) for _ in range(lc[0])) for _ in range(n)] for _ in range(math.ceil(m / tile))]
    bt = [[rng.randint(0, 1) for _ in range(later)] for _ in range(m)]
    r = [[rng.randrange(-h, h) for _ in range(later)] for _ in range(m)]
    wodd = list(lane_sum(1, lay["lane_add"]["fill_window_coin_bits"]))
    cw = [[rng.choice(wodd) for _ in range(n)] for _ in range(later)]

    def coin(i: int, j: int, u: int, pl: list[bool]) -> int:
        if pl[u]:
            return (1 + bt[i][u]) * w + s[i // tile][u][j]
        return r[i][u] * w + cw[u][j] if ef else (1 + bt[i][u]) * w

    adv_plan = plan[plan_shift:] + plan[:plan_shift]
    adv_lanes = lanes_u[plan_shift:] + lanes_u[:plan_shift]
    equal, kept = True, 0
    for i in range(m):
        for j in range(n):
            hw = aw = sum(x * y for x, y in zip(a[i][:d1], bcols[j][:d1])) + s1[i // tile][j]
            for u in range(later):
                lo = d1 + dl * u
                hw += sum(x * y for x, y in zip(a[i][lo:lo + dl], bcols[j][lo:lo + dl])) + coin(i, j, u, plan)
                c_adv = coin(i, j, u, adv_plan)
                v_adv = vcoin[adv_lanes[u]] if adv_plan[u] else vfill
                if c_adv == v_adv:
                    kept += 1
                else:
                    aw += a[i][lo] * bcols[j][lo] + c_adv
                equal &= aw == hw
    rows_tile = min(m, tile)
    mean = var = 0.0
    for u in range(later):
        if plan[u]:
            x_law = {(1 + b) * w: 0.5 for b in (0, 1)}
            ps_ = lane_sum(lanes_u[u], e)
            q = {sv: sum(p for x, p in x_law.items() if x + sv == vcoin[lanes_u[u]]) for sv in ps_}
            eq = sum(ps_[sv] * q[sv] for sv in ps_)
            eq2 = sum(ps_[sv] * q[sv] ** 2 for sv in ps_)
            cells = math.ceil(m / tile) * n
            mean += cells * rows_tile * eq
            var += cells * (rows_tile * eq + rows_tile * (rows_tile - 1) * eq2 - (rows_tile * eq) ** 2)
        else:
            v = vfill
            p = dists["fill"][v]
            pr = {x * w: 1.0 / (2 * h) for x in range(-h, h)}
            pc = lane_sum(1, lay["lane_add"]["fill_window_coin_bits"])
            same_row = sum(q * pc.get(v - x, 0.0) ** 2 for x, q in pr.items())
            same_col = sum(q * pr.get(v - c, 0.0) ** 2 for c, q in pc.items())
            e2 = (m * n * p + m * n * (n - 1) * same_row + n * m * (m - 1) * same_col
                  + m * (m - 1) * n * (n - 1) * p * p)
            mean += m * n * p
            var += e2 - (m * n * p) ** 2
    return {"equal": equal, "kept": kept, "mean": mean, "sd": math.sqrt(var), "entry_steps": m * n * later,
            "coin_steps": sum(plan), "later": later}


def outsource(calls: list, ivs: list, m: int, layers: int, served: str, wfun, alpha: float, out: dict) -> dict:
    """Priced work an outsourcing prover saves a bit it exchanges, under R1b (it keeps A_j for the certifier, so it
    computes the served path; the outside machine computes the coin-dependent words and hashes them, returning a root a
    call). Coin-free served path: any set of call types, every layer, beside the token (saving W - 1 a MAC); or one
    call type's step interval, every layer, the outside machine returning the interval's root (a subtree when the
    tree holds a call's steps in order). Noisy served path: a prefix of the forward, since each call's input carries
    every earlier call's coins (W - alpha words a MAC: the prover still runs the noisy MMA)."""
    tok, root = out["token_bits"], out["root_bits"]
    res = {}
    if served in ("exact", "either"):
        per = [(k * n * (wfun(c["words"]) - 1.0), c["bits"] / m + root / m) for k, n, c in calls]
        res["subset"] = max(layers * sum(per[i][0] for i in range(len(per)) if mask >> i & 1)
                            / (layers * sum(per[i][1] for i in range(len(per)) if mask >> i & 1) + tok)
                            for mask in range(1, 2 ** len(per)))
        res["interval"] = max(layers * macs / m * (wfun(wd) - 1.0) / (layers * (b + root) / m + tok)
                              for iv in ivs for _, _, macs, b, wd in iv)
    if served in ("noisy", "either"):
        sv = bt = best = 0.0
        for _ in range(layers):
            for k, n, c in calls:
                sv += k * n * (wfun(c["words"]) - alpha * c["words"])
                bt += c["bits"] / m + root / m
                best = max(best, sv / (bt + tok))
        res["prefix"] = best
    return res


def w_mac(base: dict, words: float, units_per_byte: float, answers: float) -> float:
    return base["alpha"] * words + base["hash_bytes_per_mac"] * words * units_per_byte + answers


def price_design(design: dict, inp: dict, einp: dict, ll: dict) -> dict:
    cfg = inp["llama_3_2_1b"]
    lead, lay, pr = einp["lead"], einp["layouts"], einp["prices"]
    caps = inp["egress"]["cap_bits_per_token"]
    sha = inp["sha512"]
    units = {"floor": sha["floor_units_per_byte"],
             "floor_chained": sha["floor_units_per_byte"] * pr["sha_registers_per_compression"]["chained"]
             / pr["sha_registers_per_compression"]["locked"],
             "bar": sha["bar_units_per_byte"]}
    base = {"alpha": inp["ssn_1l"]["alpha"], "hash_bytes_per_mac": inp["egress"]["credited_hash_bytes_per_mac"]}
    pts = {}
    for key, space_key in POINTS.items():
        m = {**inp["points"], **einp["points"]}[key]["m"]
        calls = [(k, n, call(design, m, k, n, lead, lay, pr["reg_units"])) for k, n in cfg["linears"]]
        ivs = [list(intervals(c, m, n, k)) for k, n, c in calls]
        macs = cfg["layers"] * sum(m * k * n for k, n, _ in calls)
        bits = cfg["layers"] * sum(c["bits"] for _, _, c in calls)
        byts = cfg["layers"] * sum(c["bytes"] for _, _, c in calls)
        words = cfg["layers"] * sum(m * k * n * c["words"] for k, n, c in calls) / macs
        forming = cfg["layers"] * sum(c["forming"] for _, _, c in calls) / macs
        entries = cfg["layers"] * sum(c.get("entries", 0) for _, _, c in calls)
        ans = pr["answers_units_per_mac"]
        wm = {p: w_mac(base, words, u, ans) for p, u in units.items()}
        at = [(macs_ / b, macs_ / b * wd, f"K {k} N {n} steps {t0}-{t1} of {len(c['D'])}")
              for (k, n, c), iv in zip(calls, ivs) for t0, t1, macs_, b, wd in iv]
        n_max, n_at = max((a[0], a[2]) for a in at)
        b_max, b_at = max((a[1], a[2]) for a in at)
        beta = {p: w_mac(base, 1.0, u, 0.0) * b_max + ans * n_max for p, u in units.items()}
        beta_call = {p: max(m * k * n / c["bits"] * w_mac(base, c["words"], u, ans) for k, n, c in calls)
                     for p, u in units.items()}
        served = SERVED.get(design["kind"], "noisy")
        osr = einp["outsource"]
        per_bit = {p: outsource(calls, ivs, m, cfg["layers"], served,
                                lambda wd, u=u: w_mac(base, wd, u, ans), base["alpha"], osr)
                   for p, u in units.items()}
        row = {"m": m, "coin_bits_per_token": bits / m, "stream_bytes_per_token": byts / m, "n_max": n_max,
               "n_max_at": n_at, "beta": beta, "beta_at": b_at, "beta_call": beta_call,
               "n_eff": {p: beta[p] / wm[p] for p in wm},
               "macs_per_coin_bit_mean": macs / bits, "words_vs_lead": words, "w_mac": wm,
               "forming_share_of_stock": forming, "served": served,
               "noise_rms_lsb_by_k": {str(k): c["noise_rms"] for k, _, c in calls},
               "word_noise_rms_lsb_by_k": {str(k): c.get("word_noise_rms", c["noise_rms"]) for k, _, c in calls},
               "outsource_units_per_bit": per_bit,
               "caps": {str(cap): {"stated": {p: beta[p] * cap / ll["macs_per_token"] for p in wm},
                                   "honest_share": n_max * cap / ll["macs_per_token"],
                                   "attack": {p: {a: {"whole_cap": r * cap / ll["macs_per_token"],
                                                      "slack": r * (cap - osr["honest_io_bits_per_token"])
                                                      / ll["macs_per_token"],
                                                      "of_stated": r / beta[p]}
                                                  for a, r in per_bit[p].items()} for p in wm}}
                        for cap in caps}}
        if space_key:
            row["t2"] = t2_at(inp, einp, ll, space_key, byts, entries, words)
        if key in einp["nkernel_t2"]:
            row["t2_nkernel_scaled"] = [v * words for v in einp["nkernel_t2"][key]]
        row["a2c"] = a2c_point(design, lay, einp["nred"]["public_lane_weight"], calls, m, pr["reg_units"])
        pts[key] = row
    a2c = None
    if all(p["a2c"] for p in pts.values()):
        def resid(q: str) -> float:
            return pts[q]["a2c"]["model_p_share"] + pts[q]["a2c"].get("walk_share", 0.0)
        worst = max(pts, key=resid)
        a2c = {"pstar": max(p["a2c"]["pstar"] for p in pts.values()),
               "model_p_share": max(p["a2c"]["model_p_share"] for p in pts.values()),
               "residual": resid(worst), "worst_point": worst}
    return {"id": design["id"], "title": design["title"], "points": pts, "a2c": a2c}


def t2_at(inp: dict, einp: dict, ll: dict, space_key: str, step_bytes: float, step_entries: float,
          words: float) -> dict:
    """The coin stream beside N-CARD's honest step for the lead (hashing-bound, scaled by the words), the lane patch's
    byte stores, and R1b's arrival-order coins (alt_roads.r1b_arrival_t2)."""
    v = inp["space_t2"]["designs"][LEAD_T2][space_key]
    h = inp["h200"]
    patch_ms = step_entries / (h["sms"] * einp["prices"]["lane_store_bytes_per_sm_clk"] * h["fmax_hz"]) * 1e3
    honest_ms = v["t_h_ms"] * words + patch_ms
    arrival = alt_roads.r1b_arrival_t2(inp, ll)["designs"][LEAD_T2][space_key]["arrival_t2"]
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"]}
    out = {"honest_ms": honest_ms, "patch_ms": patch_ms, "stock_ms": v["stock_ms"],
           "lead_judged": v["judged"], "stream_bytes_per_s": step_bytes / (honest_ms * 1e-3),
           "certifier_cores": [step_bytes / (honest_ms * 1e-3) / r
                               for r in einp["certifier"]["sha512_bytes_per_s_core"]][::-1]}
    for ln, rate in links.items():
        link_ms = step_bytes / rate * 1e3
        out[ln] = {"port_share": step_bytes / (honest_ms * 1e-3) / rate, "link_ms": link_ms,
                   "t2": max(honest_ms, link_ms) / v["stock_ms"],
                   "t2_arrival": [max(a * words + patch_ms / v["stock_ms"], link_ms / v["stock_ms"])
                                  for a in arrival[ln]]}
    return out


def lower_bound(inp: dict, ll: dict, res: dict) -> dict:
    """Any layout: the stated term needs n_max <= 0.01 MACs / (W_mac cap), and n_max >= MACs / coin bits, so coin bits
    a token of at least 100 W_mac cap. A B-shared lane: a coin entry is read by the tile's T rows over a step's d data
    lanes, so n_max >= T d / (L e)."""
    rows = res["designs"][0]["points"]["t8192"]
    out = {}
    for cap in inp["egress"]["cap_bits_per_token"]:
        out[str(cap)] = {}
        for p, w in rows["w_mac"].items():
            need = 0.01 * ll["macs_per_token"] / (w * cap)
            out[str(cap)][p] = {"n_max_at_most": need, "coin_bits_per_token_at_least": 100.0 * w * cap,
                                "coin_bits_per_mac_at_least": 100.0 * w * cap / ll["macs_per_token"],
                                "blc_tile_e_lanes_that_meet_it": [
                                    {"tile": t, "lanes": lanes, "e": e, "n": t * d / (lanes * e)}
                                    for t in (16, 64, 128) for lanes, d in ((1, 31), (4, 28)) for e in (1, 2, 4, 7)
                                    if t * d / (lanes * e) <= need]}
    return out


def statement_check(inp: dict, ll: dict, res: dict) -> dict:
    """The term under the statements as locked: Accel.ext and CertifierSeesImports exclude only the coin stream, so
    R1b's A_j to the certifier (arrival order), or a 512-bit root(A_j) a call in its place, and the locked Dither
    layout's n_max = N, are counted as they read."""
    cfg = inp["llama_3_2_1b"]
    calls_per_forward = cfg["layers"] * len(cfg["linears"])
    out = {}
    for did in dict.fromkeys(("rows", res["closing"]["id"], COORDINATOR_LEAD)):
        out[did] = {}
        for key, p in res["designs_by_id"][did]["points"].items():
            extra = {"a_j_to_the_certifier": 8.0 * ll["a_bytes_per_token"],
                     "root_a_j_a_call": 512.0 * calls_per_forward / p["m"]}
            out[did][key] = {name: {p_: b * (32 + bits) / ll["macs_per_token"]
                                    for p_, b in p["beta"].items()} | {"bits_per_token": bits}
                             for name, bits in extra.items()}
    return out


def per_period(inp: dict, shares: list[tuple[str, float]]) -> list[dict]:
    acc = inp["accounting"]
    periods = {"window": acc["window_s"], "week": acc["week_s"], "month": acc["month_s"], "year": acc["year_s"]}
    h200 = alt_roads.peak(inp) * acc["mfu_serve"]
    rates = {"one H200": h200, "datacenter": h200 * acc["dc_gpus"],
             "world": acc["world_flop_per_year"] / 2.0 / acc["year_s"]}
    return [{"bound": name, "share_of_stock": share,
             "and_gates": {sc: {p: share * r * s * acc["and_per_int8_mac"] for p, s in periods.items()}
                           for sc, r in rates.items()}} for name, share in shares]


def distinct_check(s_zero: bool, seed: int) -> bool:
    """ssn-1l-tag's words with blc1's lane as affine functions of a row's coin bits (step 1's four code bits against
    the column code, each later step's bit b_t against the lane's coin s_t[j]): pairwise distinct coefficient vectors
    over rows, columns and steps for any nonzero s. With s_zero one coin is 0, so that word's step adds no bit."""
    rng = random.Random(seed)
    m, n, steps_ = 3, 5, 4
    code = [[(j // 13 ** d) % 13 - 6 for d in range(4)] for j in range(1, n + 1)]
    s = [[rng.choice([v for v in range(-127, 128, 2)]) for _ in range(n)] for _ in range(steps_)]
    if s_zero:
        s[2][1] = 0
    seen = set()
    for i in range(m):
        for j in range(n):
            for t in range(steps_):
                coef = [0] * (m * (4 + steps_))
                for d in range(4):
                    coef[i * (4 + steps_) + d] = code[j][d]
                for u in range(1, t + 1):
                    coef[i * (4 + steps_) + 4 + u] = s[u][j]
                key = tuple(coef)
                if key in seen or not any(coef):
                    return False
                seen.add(key)
    return True


def fold_check(kind: str, seed: int) -> bool:
    """A tile's final accumulators with the lead's row lane (1 + b_t(i)) against public w_t and blc1's coin in that lane,
    or blc1a's coin on its own lane against A' = 1 on alternate steps and its 8-bit row coin r_t(i) in the row lane on
    the others (a tile with r < T rows): True when the noise is r_i + c_j and subtracting r_i = sum_t a_t(i) w_t and
    c_j = sum_t s_t[j] (N-EXACT's IADD3 sources) leaves A B^T exactly. A noise whose second differences
    N[i,j] - N[i,0] - N[0,j] + N[0,0] aren't all 0 is no r_i + c_j, so no such fold removes it."""
    rng = random.Random(seed)
    m, n, k, st = 6, 5, 40, 4
    a = [[rng.randint(-127, 127) for _ in range(k)] for _ in range(m)]
    b = [[rng.randint(-127, 127) for _ in range(n)] for _ in range(k)]
    bits = [[rng.randint(0, 1) for _ in range(st)] for _ in range(m)]
    rowc = [[rng.randrange(-128, 128) for _ in range(st)] for _ in range(m)]
    w = [rng.randint(-127, 127) for _ in range(st)]
    s = [[rng.choice([v for v in range(-15, 16, 2)]) for _ in range(n)] for _ in range(st)]
    cw = [[rng.choice([v for v in range(-127, 128, 2)]) for _ in range(n)] for _ in range(st)]
    coined = [t % 2 == 0 or kind == "blc1" for t in range(st)]
    exact = [[sum(a[i][x] * b[x][j] for x in range(k)) for j in range(n)] for i in range(m)]
    def a31(i: int, t: int) -> int:
        return 1 + bits[i][t] if coined[t] else rowc[i][t]

    def lane(i: int, j: int, t: int) -> int:
        if kind == "blc1":
            return (1 + bits[i][t]) * s[t][j]
        return a31(i, t) * w[t] + (s[t][j] if coined[t] else cw[t][j])
    acc = [[exact[i][j] + sum(lane(i, j, t) for t in range(st)) for j in range(n)] for i in range(m)]
    noise = [[acc[i][j] - exact[i][j] for j in range(n)] for i in range(m)]
    if any(noise[i][j] - noise[i][0] - noise[0][j] + noise[0][0] for i in range(m) for j in range(n)):
        return False
    r = [sum(a31(i, t) * w[t] for t in range(st)) for i in range(m)]
    c = [sum(s[t][j] if coined[t] else cw[t][j] for t in range(st)) for j in range(n)]
    return all(acc[i][j] - r[i] - c[j] == exact[i][j] for i in range(m) for j in range(n))


def walk_words(n: int, length: int, dl: int, grain: int, window: bool, seed: int) -> dict:
    """N-ADAPT's walk word for word on a one-row tile's run of row-coined steps: each step adds r_u (the row coin), the
    window coin c_u[j] when the layout has one, and the call's data, which is chosen after the window coin and so
    cancels it on (dl // grain) aligned chunks, plus the walk's alternating 0 / 1. A chunk-step whose words equal that
    chunk's at an earlier step is a copy. Returns the copied words, all words, and the row-steps on which the scalar
    walk revisited."""
    rng = random.Random(seed)
    cols = (dl // grain) * grain
    vec = [0] * n
    seen = [{tuple(vec[q:q + grain])} for q in range(0, n, grain)]
    walk, wseen, copied, revisits = 0, {0}, 0, 0
    for u in range(length):
        r = rng.randrange(-128, 128)
        cw = [rng.choice(range(-127, 128, 2)) if window else 0 for _ in range(n)]
        data = [(-cw[j] if j < cols else 0) + (u & 1) for j in range(n)]
        vec = [vec[j] + r + cw[j] + data[j] for j in range(n)]
        walk += r + (u & 1)
        revisits += walk in wseen
        wseen.add(walk)
        for q in range(0, n, grain):
            key = tuple(vec[q:q + grain])
            copied += grain * (key in seen[q // grain])
            seen[q // grain].add(key)
    return {"copied": copied, "words": n * length, "revisits": revisits, "cols": cols}


def noise_mc(e: int, s_count: int, trials: int, seed: int) -> float:
    rng = random.Random(seed)
    odd = list(range(-(2 ** e - 1), 2 ** e, 2))
    tot = 0.0
    for _ in range(trials):
        x = sum((1 + rng.randint(0, 1)) * rng.choice(odd) for _ in range(s_count))
        tot += x * x
    return math.sqrt(tot / trials)


def checks(inp: dict, einp: dict, ll: dict, res: dict) -> list[tuple[str, bool, str]]:
    d = {x["id"]: x for x in res["designs"]}
    ar = alt_roads.egress_outsourcing(inp, ll)
    lay, w, reg = einp["layouts"], einp["nred"]["public_lane_weight"], einp["prices"]["reg_units"]
    lead, d1 = einp["lead"], einp["lead"]["step1_data_lanes"]
    pre = {**einp, "lead": {**lead, "step1_data_lanes": 32 - lead["code_lanes"], "tag_lanes": 0}}
    by_id = {x["id"]: x for x in designs(lay)}
    rows8 = d["rows"]["points"]["t8192"]
    pre8 = price_design(by_id["rows"], inp, pre, ll)["points"]["t8192"]
    best = d[res["closing"]["id"]]["points"]
    blc = call({"kind": "blc1", "tile": 128, "e": 2}, 8192, 2048, 16384, lead, lay, 32)
    brute = 8192 * 70 + 16384 * 4 * math.log2(7) + sum(16384 * 66 * 2 for _ in range(0, 8192, 128))
    ratio = {p: pre8["caps"]["32"]["stated"][p] / ar["share_of_stock_at_cap"]["32"]["r1b_live_coins"][
        f"{p}_units_per_byte"] for p in ("floor", "bar")}
    cols = inp["llama_3_2_1b"]["layers"] * sum(n for _, n in inp["llama_3_2_1b"]["linears"])
    lead_a2c, nred_pt = d["rows"]["points"]["t8192"]["a2c"], einp["nred"]["model_p_saving_at_pstar"][0]
    muts = {name: price_design({**by_id[did], **over, "id": name}, inp, e_, ll)
            for name, did, over, e_ in (("interim 2's step 1", "blc1a-t128-e7", {"step1_coins": False}, pre),
                                        ("interim 2's step 1 on blc4a", "blc4a-t64-e7", {"step1_coins": False}, pre),
                                        ("tag lead, interim 2's step 1 on blc4a", "blc4a-t64-e7",
                                         {"step1_coins": False}, einp),
                                        ("front-loaded", "blc1a-t128-e7", {"plan": "front"}, einp))}
    hole = muts["tag lead, interim 2's step 1 on blc4a"]["points"]
    hole8 = 8192 * 16384 * d1 / (8192 * 4 + 16384 * 4 * math.log2(7))
    ranks = {m: (tag_rank(m, lead), exact_rank([balanced_digits((i // 2 + 1) * (-1) ** i, 13, 4) for i in range(m)]))
             for m in (1, 12, 13, 64, 168, 169, 512, 2196, 2197, 8192, 28560)}
    try:
        over = f"rank {tag_rank(28561, lead)}"
    except ValueError as err:
        over = str(err)
    packed = d["blc4a-t64-e7-packed"]["points"]["decode1"]
    lay0 = json.loads(json.dumps(lay))
    lay0["lane_add"]["fill_row_coin_bits"] = 0
    old = {}
    for key in ("t64", "decode1"):
        m = inp["points"][key]["m"]
        calls = [(k, n, call(by_id["blc1a-t128-e7"], m, k, n, einp["lead"], lay0, reg))
                 for k, n in inp["llama_3_2_1b"]["linears"]]
        old[key] = a2c_point(by_id["blc1a-t128-e7"], lay0, w, calls, m, reg)["model_p_share"]
    a8 = {}
    for c in ("closing", "closing_320"):
        des = by_id[res[c]["id"]]
        t = des["tile"]
        a8[f"{des['id']} m {2 * t}"] = a8_words(des, lay, w, 2 * t, 128, 11, 71, d1)
        a8[f"{des['id']} m {t // 2}"] = a8_words(des, lay, w, t // 2, 32, 128, 73, d1)
    half = by_id[res["closing"]["id"]]
    a8_bad = a8_words(half, lay, w, half["tile"] // 2, 32, 128, 73, d1, plan_shift=1)
    g = lay["lane_add"]["copy_grain_words"]
    ww0 = walk_words(64, 128, lay["lane_add"]["data_lanes"], g, False, 83)
    ww1 = walk_words(64, 128, lay["lane_add"]["data_lanes"], g, True, 83)
    clo = [(res[c]["id"], k_, p["a2c"]) for c in ("closing", "closing_320")
           for k_, p in d[res[c]["id"]]["points"].items()]
    return [
        ("The lead before its tag (28 data lanes on step 1, no G): alt_roads' 9,216 streamed bits a token (whole "
         "32-bit words), of which its words read 7,648 (S - 1 + c a row); its work a bit peaks at the whole gate-up "
         "call, 2,048 x 16,384 / 70 MACs a bit at its words",
         pre8["stream_bytes_per_token"] * 8 == ar["coin_bits_per_token"] and pre8["coin_bits_per_token"] == 7648
         and abs(pre8["n_eff"]["floor"] / (2048 * 16384 / 70) - 1) < 1e-12 and pre8["beta_at"].endswith("1-67 of 67")
         and "K 2048 N 16384" in pre8["beta_at"],
         f"{pre8['stream_bytes_per_token'] * 8:.0f}, {pre8['coin_bits_per_token']:.0f}, n_eff "
         f"{pre8['n_eff']['floor']:,.1f} at {pre8['beta_at']}; n_max {pre8['n_max']:,.1f} at {pre8['n_max_at']}"),
        ("The lead before its tag: its stated term at a 32-bit cap is alt_roads' (137.6% floor, 267.8% bar, at the 96 "
         "bits a gate-up row streams) x 96 / 70, the bits its words read",
         all(abs(r - 96 / 70) < 1e-9 for r in ratio.values()),
         f"{pre8['caps']['32']['stated']['floor']:.4f}, {pre8['caps']['32']['stated']['bar']:.4f}"),
        ("The tag lead as built: S = 67 and 265 at Llama's K with 24 data lanes on step 1; at T 8,192 its stream adds "
         "G's coin word a column a call to the 9,216 bits, and its words read G's four digits, 4 log2 7 bits a column a "
         "call",
         steps(2048, d1) == 67 and steps(8192, d1) == 265
         and rows8["stream_bytes_per_token"] * 8 == ar["coin_bits_per_token"] + 32 * cols / 8192
         and abs(rows8["coin_bits_per_token"] - 7648 - cols * 4 * math.log2(7) / 8192) < 1e-9,
         f"{rows8['stream_bytes_per_token'] * 8:.0f}, {rows8['coin_bits_per_token']:.1f}; term at cap 32 "
         f"{rows8['caps']['32']['stated']['floor']:.4f} / {rows8['caps']['32']['stated']['bar']:.4f}"),
        ("The row tags' rank by exact elimination equals the digit count tag_rank reads (1 digit to 12 rows, 2 to 168, "
         "3 to 2,196, 4 to 28,560) and T stops being injective in four digits past 28,560 rows",
         all(a == b for a, b in ranks.values()) and over == "28561 rows: T isn't injective in 4 digits", ", ".join(f"{m} {a}/{b}" for m, (a, b) in ranks.items())),
        ("blc1's coin bits at gate-up, 8,192 rows, 128-row tiles, 2 bits: the formula equals a tile-by-tile count",
         abs(blc["bits"] - brute) < 1e-6, f"{blc['bits']:,.1f}"),
        ("Mutation: interim 2's blc1a-t128-e7 and blc4a-t64-e7 on the lead before its tag, step 1 without lane coins: "
         "gate-up's step 1 carries 28 x 16,384 / 4 MACs a bit, the term passes 1% at cap 32 at every point, and the "
         "interval outsourcer, its 512-bit root a call counted, saves more than 5% of stock at cap 32 (term, "
         "outsourcer, floor)",
         all(p["n_max"] == 28 * 16384 / 4 and p["caps"]["32"]["stated"]["floor"] > 0.01
             and p["caps"]["32"]["attack"]["floor"]["interval"]["whole_cap"] > 0.05
             for mu in ("interim 2's step 1", "interim 2's step 1 on blc4a") for p in muts[mu]["points"].values()),
         "; ".join(f"{mu}: " + ", ".join(f"{k} {p['caps']['32']['stated']['floor']:.3f} "
                                          f"{p['caps']['32']['attack']['floor']['interval']['whole_cap']:.3f}"
                                          for k, p in muts[mu]["points"].items())
                   for mu in ("interim 2's step 1", "interim 2's step 1 on blc4a"))),
        ("Mutation: the same on the tag lead: G closes step 1's hole at decode and 64 rows (term under 1% at cap 32) "
         "but not at T 8,192, where gate-up's step 1 carries 8,192 x 16,384 x 24 / (8,192 x 4 + 16,384 x 4 log2 7) "
         "MACs a bit, the term passes 1% and the interval outsourcer saves more than 5% of stock",
         all(hole[k]["caps"]["32"]["stated"]["bar"] < 0.01 for k in ("decode1", "t64"))
         and abs(hole["t8192"]["n_max"] / hole8 - 1) < 1e-12 and "steps 1-1 of" in hole["t8192"]["n_max_at"]
         and hole["t8192"]["caps"]["32"]["stated"]["floor"] > 0.01
         and hole["t8192"]["caps"]["32"]["attack"]["floor"]["interval"]["whole_cap"] > 0.05,
         ", ".join(f"{k} {p['caps']['32']['stated']['floor']:.4f}-{p['caps']['32']['stated']['bar']:.4f} n_max "
                   f"{p['n_max']:,.0f}" for k, p in hole.items())),
        ("Mutation: blc4a-t64-e7 with three lanes on each coined step: at decode a run between coined steps carries "
         "more than twice the call's work a bit and the term passes 1% at cap 320; one or two lanes a step it doesn't",
         packed["beta"]["floor"] > 2 * packed["beta_call"]["floor"]
         and packed["caps"]["320"]["stated"]["bar"] > 0.01
         and d["blc4a-t64-e7"]["points"]["decode1"]["caps"]["320"]["stated"]["bar"] <= 0.01,
         f"x{packed['beta']['floor'] / packed['beta_call']['floor']:.3f} at {packed['beta_at']}, "
         f"{packed['caps']['320']['stated']['bar']:.4f} vs "
         f"{d['blc4a-t64-e7']['points']['decode1']['caps']['320']['stated']['bar']:.4f}"),
        ("blc4a-t64-e7 packed from 16 rows and spread below: its term is at most 1% at cap 320 at both prices and A2c "
         "plus the walk at most 1%, at every point; at 16 rows it streams under half what the spread does, and from "
         "64 rows and at one row it is the spread",
         all(p["caps"]["320"]["stated"][q] <= 0.01 and p["a2c"]["model_p_share"] + p["a2c"]["walk_share"] <= 0.01
             for p in d["blc4a-t64-e7-pack16"]["points"].values() for q in ("floor", "bar"))
         and d["blc4a-t64-e7-pack16"]["points"]["decode16"]["stream_bytes_per_token"]
         < 0.5 * d["blc4a-t64-e7"]["points"]["decode16"]["stream_bytes_per_token"]
         and all(d["blc4a-t64-e7-pack16"]["points"][k]["stream_bytes_per_token"]
                 == d["blc4a-t64-e7"]["points"][k]["stream_bytes_per_token"]
                 and d["blc4a-t64-e7-pack16"]["points"][k]["beta"] == d["blc4a-t64-e7"]["points"][k]["beta"]
                 for k in ("decode1", "t64", "t512", "t8192")),
         ", ".join(f"{k} {p['caps']['320']['stated']['bar']:.4f} {p['stream_bytes_per_token'] / 1e6:.2f} MB"
                   for k, p in d["blc4a-t64-e7-pack16"]["points"].items())),
        ("Mutation: the same lane coins on a small tile's first steps leave a run of coin-free steps whose interval "
         "carries more a bit than the call; the term passes 1% at T 512 with 64 rows, where spread evenly it doesn't",
         muts["front-loaded"]["points"]["t64"]["caps"]["32"]["stated"]["bar"] > 0.01
         and d["blc1a-t128-e7"]["points"]["t64"]["caps"]["32"]["stated"]["bar"] <= 0.01,
         f"{muts['front-loaded']['points']['t64']['caps']['32']['stated']['bar']:.4f} at "
         f"{muts['front-loaded']['points']['t64']['beta_at']} vs "
         f"{d['blc1a-t128-e7']['points']['t64']['caps']['32']['stated']['bar']:.4f}"),
        ("Both closing layouts' work a bit over every step interval is within 5% of their best whole call's at every "
         "point (lane coins on step 1, spread evenly)",
         all(p["beta"]["floor"] <= 1.05 * p["beta_call"]["floor"]
             for c in ("closing", "closing_320") for p in d[res[c]["id"]]["points"].values()),
         "; ".join(f"{res[c]['id']} {k} x{p['beta']['floor'] / p['beta_call']['floor']:.4f} at {p['beta_at']}"
                   for c in ("closing", "closing_320") for k, p in d[res[c]["id"]]["points"].items())),
        ("Every layout's n_max is at least its mean MACs a coin bit (the lower bound's step)",
         all(p["n_max"] >= p["macs_per_coin_bit_mean"] * (1 - 1e-12) for x in res["designs"]
             for p in x["points"].values()), "ok"),
        ("The closing layout's stated term is at most 1% of stock at a 32-bit cap, at both prices, at every point",
         all(p["caps"]["32"]["stated"][q] <= 0.01 for p in best.values() for q in ("floor", "bar")),
         ", ".join(f"{k} {p['caps']['32']['stated']['bar']:.4f}" for k, p in best.items())),
        ("The closing layout's stream is under 10% of a 400 Gb/s port at the honest pace at every timed point",
         all(p["t2"]["gpu_400g"]["port_share"] < 0.10 for p in best.values() if "t2" in p),
         ", ".join(f"{p['t2']['gpu_400g']['port_share']:.4f}" for p in best.values() if "t2" in p)),
        ("blc1's served noise rms matches a seeded Monte Carlo within 3% (e 2, 66 steps)",
         abs(noise_mc(2, 66, 20000, 7) / math.sqrt(2.5 * 66 * odd_mean_square(2)) - 1) < 0.03,
         f"{noise_mc(2, 66, 20000, 7):.2f} vs {math.sqrt(2.5 * 66 * odd_mean_square(2)):.2f}"),
        ("blc1 keeps the tag lead's words pairwise distinct as functions of the row's coin bits for nonzero lane coins",
         all(distinct_check(False, sd) for sd in range(20)), "20 seeds"),
        ("Mutation: a zero lane coin makes two words of one element coincide, and the check catches it",
         not any(distinct_check(True, sd) for sd in range(20)), "20 seeds"),
        ("blc1a's served noise, its coin steps and a small tile's row-coined steps, is r_i + c_j and N-EXACT's two fold "
         "sources return A B^T exactly",
         all(fold_check("blc1a", sd) for sd in range(20)), "20 seeds"),
        ("Mutation: blc1's coin in the row lane isn't r_i + c_j, so no free fold removes it, and the check says so",
         not any(fold_check("blc1", sd) for sd in range(20)), "20 seeds"),
        ("The de-noised lead's call-set outsourcer saves at least 95% a bit of what the stated term charges at T 8,192",
         rows8["caps"]["32"]["attack"]["floor"]["subset"]["of_stated"] >= 0.95,
         f"{rows8['caps']['32']['attack']['floor']['subset']['of_stated']:.4f}"),
        ("No outsourcer saves more a bit than the stated term charges, at every design, point, price and cap",
         all(v["of_stated"] <= 1.0 + 1e-12 for x in res["designs"] for p in x["points"].values()
             for c in p["caps"].values() for a in c["attack"].values() for v in a.values()), "ok"),
        ("The closing layout serves exactly (its served noise folds)",
         all(p["served"] == "exact" for p in best.values()), res["closing"]["id"]),
        ("The lead's A2c by N-RED's count: p* = 1/2 and model P 51.6% at T 8,192, as N-RED measured",
         lead_a2c["pstar"] == nred_pt[0] and abs(lead_a2c["model_p_share"] - nred_pt[1]) < 0.001,
         f"{lead_a2c['pstar']}, {lead_a2c['model_p_share']:.4f}"),
        ("Both closing layouts' A2c / A8 (model P, N-RED's count) plus N-ADAPT's walk is at most 1% of stock at every "
         "point",
         all(a["model_p_share"] + a["walk_share"] <= 0.01 for _, _, a in clo),
         ", ".join(f"{i} {k_} {a['model_p_share']:.4f}+{a['walk_share']:.4f}" for i, k_, a in clo)),
        ("N-ADAPT's walk on a small tile's row-coined runs: without the window coin it exceeds 1% of stock at decode "
         "for both closing layouts, so the window coin is needed",
         all(a["walk_share_without_window_coin"] > 0.01 for _, k_, a in clo if k_ == "decode1"),
         ", ".join(f"{i} {a['walk_share_without_window_coin']:.4f}" for i, k_, a in clo if k_ == "decode1")),
        ("The walk word for word on a one-row run of 128 row-coined steps, 64 columns: without the window coin every "
         "4-word chunk repeats exactly when the row's walk revisits; with it only the 28 columns the data cancels do",
         ww0["copied"] == 64 * ww0["revisits"] and ww1["copied"] == ww1["cols"] * ww1["revisits"]
         and ww0["revisits"] > 0 and ww1["revisits"] > 0,
         f"{ww0['revisits']} / {ww1['revisits']} revisits; copied {ww0['copied']} / {ww1['copied']} of "
         f"{ww0['words']}"),
        ("Interim 2's blc1a-t128-e7 (the lead's public lane on a small tile's coin-less steps) leaves A2c above 25% "
         "at T 512 with 64 rows and above 50% at decode",
         old["t64"] > 0.25 and old["decode1"] > 0.5, f"{old['t64']:.4f}, {old['decode1']:.4f}"),
        ("A8 word for word on both closing layouts, two tiles of T rows and one of T/2 (half its steps with the row and "
         "window coins in place of lane coins): "
         "the adversary's words equal the honest words, its kept entry-steps within 4 sd of the expectation",
         all(v["equal"] and abs(v["kept"] - v["mean"]) <= 4 * v["sd"] for v in a8.values()),
         "; ".join(f"{name} kept {v['kept']} (mean {v['mean']:.0f}, sd {v['sd']:.0f}) of {v['entry_steps']:,}"
                   for name, v in a8.items())),
        ("Mutation: an A8 adversary whose coin plan is one step off on the half tile no longer matches the honest words",
         not a8_bad["equal"], f"kept {a8_bad['kept']}"),
        ("The cap-320 closing layout serves exactly, its stated term at most 1% at cap 320 at both prices, every point",
         all(p["served"] == "exact" and p["caps"]["320"]["stated"][q] <= 0.01
             for p in d[res["closing_320"]["id"]]["points"].values() for q in ("floor", "bar")),
         ", ".join(f"{k} {p['caps']['320']['stated']['bar']:.4f}"
                   for k, p in d[res["closing_320"]["id"]]["points"].items())),
        ("Per period: N-CARD's stock work reproduces (one H200 window 3.09e19 AND gates)",
         abs(res["period"][0]["and_gates"]["one H200"]["window"] / res["period"][0]["share_of_stock"] / 3.086e19 - 1)
         < 1e-3, f"{res['period'][0]['and_gates']['one H200']['window'] / res['period'][0]['share_of_stock']:.4e}"),
    ]


def table(res: dict) -> str:
    lines = ["N-EGRESS: OutsourceWithinEgress's term n_max W_mac E / stock MACs on Llama-3.2-1B (T 8,192 unless stated)",
             "  design: coin bits a token, n_max, stream B a token; term at cap 32 / 320, floor / floor chained / bar; "
             "outsourcers at 32 (floor; a call set or one call's step interval if the served path is coin-free, else a "
             "prefix of the forward): saving at the whole cap / at its slack over 17 bits, and a bit against the stated"
             " term; forming (T1 uncredited); words; served"]
    for x in res["designs"]:
        p = x["points"]["t8192"]
        c32, c320 = p["caps"]["32"], p["caps"]["320"]
        def f(c: dict) -> str:
            return " / ".join(f"{c['stated'][q]:.4%}" for q in ("floor", "floor_chained", "bar"))
        att = "; ".join(f"{a} {v['whole_cap']:.4%} / {v['slack']:.4%} ({v['of_stated']:.3f})"
                        for a, v in c32["attack"]["floor"].items())
        a2c = ("A2c p* " + " / ".join(f"{q['a2c']['pstar']:.5f}" for q in x["points"].values()) + " model P "
               + " / ".join(f"{q['a2c']['model_p_share']:.3%}" for q in x["points"].values()) + " (" +
               " / ".join(x["points"]) + ")" if x["a2c"] else "A2c n/a")
        lines.append(f"  {x['id']:<16} {p['coin_bits_per_token']:>12,.0f} {p['n_max']:>12,.1f} "
                     f"{p['stream_bytes_per_token']:>11,.0f}  32: {f(c32)}  320: {f(c320)}  outsourcers {att}  "
                     f"forming {p['forming_share_of_stock']:.4%}  words x{p['words_vs_lead']:.3f}  {p['served']}  "
                     f"{a2c}")
    for label, cl in (("Closing layout", res["closing"]), ("Closing layout at cap 320", res["closing_320"])):
        lines += closing_lines(res, label, cl)
    lines.append("Lower bound (any layout): coin bits a token >= 100 W_mac cap; n_max <= 0.01 MACs / (W_mac cap)")
    for cap, ps in res["lower_bound"].items():
        for p, v in ps.items():
            ok = ", ".join(f"T{r['tile']}/L{r['lanes']}/e{r['e']} ({r['n']:.0f})"
                           for r in v["blc_tile_e_lanes_that_meet_it"]) or "none"
            lines.append(f"  cap {cap} {p}: n_max <= {v['n_max_at_most']:,.1f}, coin bits a token >= "
                         f"{v['coin_bits_per_token_at_least']:,.0f} ({v['coin_bits_per_mac_at_least']:.2e} a MAC); "
                         f"B-shared lanes that meet it: {ok}")
    lines.append("Under the statements as locked (only the coin stream excluded from E), cap 32, floor / bar:")
    for did, pts in res["statement_check"].items():
        for key, v in pts.items():
            lines.append(f"  {did} {key}: " + "; ".join(
                f"{name} ({x['bits_per_token']:,.0f} bits a token) {x['floor']:.4%} / {x['bar']:.4%}"
                for name, x in v.items()))
    lines.append("Per period (N-CARD's basis, stock work held fixed), AND gates window / week / month / year:")
    for r in res["period"]:
        cells = "; ".join(f"{sc} " + " / ".join(f"{v:.2e}" for v in ps.values()) for sc, ps in r["and_gates"].items())
        lines.append(f"  {r['bound']} ({r['share_of_stock']:.4%}): {cells}")
    return "\n".join(lines)


def closing_lines(res: dict, label: str, cl: dict) -> list[str]:
    lines = [f"{label}: {cl['id']} ({cl['title']})"]
    for k, p in res["designs_by_id"][cl["id"]]["points"].items():
        t2 = p.get("t2")
        s = (f"  {k:<8} m {p['m']:>5}: n_max {p['n_max']:,.1f} ({p['n_max_at']}), work a bit beta / W_mac "
             f"{p['n_eff']['floor']:,.1f} ({p['beta_at']}), term cap 32 {p['caps']['32']['stated']['floor']:.4%}"
             f"-{p['caps']['32']['stated']['bar']:.4%}, cap 320 {p['caps']['320']['stated']['floor']:.4%}-"
             f"{p['caps']['320']['stated']['bar']:.4%}; forming {p['forming_share_of_stock']:.4%}; words "
             f"x{p['words_vs_lead']:.3f}; served noise rms "
             + ", ".join(f"K {k_}: {v:.1f}" for k_, v in p["noise_rms_lsb_by_k"].items())
             + "; committed words' coin rms "
             + ", ".join(f"K {k_}: {v:.1f}" for k_, v in p["word_noise_rms_lsb_by_k"].items())
             + f"; A2c model P {p['a2c']['model_p_share']:.3%} (p* {p['a2c']['pstar']:.5f}), walk "
             f"{p['a2c']['walk_share']:.3%} ({p['a2c']['walk_share_without_window_coin']:.3%} without the window coin; "
             f"{p['a2c']['walk_share_grain1_free_copies']:.2%} at one word if a one-word copy were free)")
        if t2:
            s += (f"; T2 {t2['gpu_400g']['t2']:.2f} (lead {t2['lead_judged']:.2f}), arrival 400G "
                  f"{t2['gpu_400g']['t2_arrival'][0]:.2f}-{t2['gpu_400g']['t2_arrival'][1]:.2f}, 100G "
                  f"{t2['census_100g']['t2_arrival'][0]:.2f}-{t2['census_100g']['t2_arrival'][1]:.2f}; stream "
                  f"{t2['stream_bytes_per_s'] / 1e9:.2f} GB/s ({t2['gpu_400g']['port_share']:.2%} of 400G, "
                  f"{t2['census_100g']['port_share']:.2%} of 100G), certifier {t2['certifier_cores'][0]:.2f}-"
                  f"{t2['certifier_cores'][1]:.2f} Genoa cores, patch {t2['patch_ms']:.3f} ms; N-KERNEL's ssn_1l "
                  f"x words {p['t2_nkernel_scaled'][0]:.1f} beside / {p['t2_nkernel_scaled'][1]:.1f} serial")
        lines.append(s)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=HERE / "alt_roads_inputs.json")
    ap.add_argument("--egress-inputs", type=Path, default=HERE / "egress_inputs.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inp, einp = alt_roads.load(args.inputs), alt_roads.load(args.egress_inputs)
    ll = alt_roads.llama(inp)
    res = {"designs": [price_design(x, inp, einp, ll) for x in designs(einp["layouts"])]}
    res["designs_by_id"] = {x["id"]: x for x in res["designs"]}
    closing = [x for x in res["designs"] if x["id"].startswith(("blc1-t", "blc1a-t"))
               and all(p["caps"]["32"]["stated"]["bar"] <= 0.01 for p in x["points"].values())]
    def a2c_open(x: dict) -> bool:
        return x["a2c"] is None or x["a2c"]["residual"] > 0.01

    best = min(closing, key=lambda x: (x["points"]["t8192"]["served"] != "exact", a2c_open(x),
                                       x["points"]["t8192"]["stream_bytes_per_token"],
                                       x["points"]["t8192"]["caps"]["32"]["stated"]["bar"]))
    res["closing"] = {"id": best["id"], "title": next(d["title"] for d in designs(einp["layouts"])
                                                      if d["id"] == best["id"])}
    exact320 = [x for x in res["designs"] if x["points"]["t8192"]["served"] == "exact"
                and all(p["caps"]["320"]["stated"]["bar"] <= 0.01 for p in x["points"].values())]
    best320 = min(exact320, key=lambda x: (a2c_open(x), x["points"]["t8192"]["stream_bytes_per_token"],
                                           x["points"]["t8192"]["caps"]["320"]["stated"]["bar"]))
    res["closing_320"] = {"id": best320["id"], "title": next(d["title"] for d in designs(einp["layouts"])
                                                             if d["id"] == best320["id"])}
    res["lower_bound"] = lower_bound(inp, ll, res)
    res["statement_check"] = statement_check(inp, ll, res)
    lead8 = res["designs_by_id"]["rows"]["points"]["t8192"]["caps"]
    coord = res["designs_by_id"][COORDINATOR_LEAD]

    def worst(x: dict, cap: str, p: str) -> float:
        return max(q["caps"][cap]["stated"][p] for q in x["points"].values())
    res["period"] = per_period(inp, [
        ("stock work", 1.0),
        ("the lead's network term, cap 32, floor", lead8["32"]["stated"]["floor"]),
        ("the lead's network term, cap 32, bar", lead8["32"]["stated"]["bar"]),
        ("the de-noised lead's call-set outsourcer, cap 32's slack, floor",
         lead8["32"]["attack"]["floor"]["subset"]["slack"]),
        (f"{best['id']}'s network term, cap 32, floor, worst point", worst(best, "32", "floor")),
        (f"{best['id']}'s network term, cap 32, bar, worst point", worst(best, "32", "bar")),
        (f"{best['id']}'s lane forming, uncredited", best["points"]["t8192"]["forming_share_of_stock"]),
        (f"{best['id']}'s A2c / A8 (model P) and walk residual at its worst point ({best['a2c']['worst_point']})",
         best["a2c"]["residual"]),
        (f"{COORDINATOR_LEAD}'s network term, cap 32, floor, worst point", worst(coord, "32", "floor")),
        (f"{COORDINATOR_LEAD}'s network term, cap 32, bar, worst point", worst(coord, "32", "bar")),
        (f"{COORDINATOR_LEAD}'s A2c / A8 (model P) and walk residual at its worst point "
         f"({coord['a2c']['worst_point']})", coord["a2c"]["residual"]),
        *([] if best320["id"] == COORDINATOR_LEAD else [
            (f"{best320['id']}'s A2c / A8 (model P) and walk residual at its worst point "
             f"({best320['a2c']['worst_point']})", best320["a2c"]["residual"])]),
        (f"{best320['id']}'s network term, cap 320, floor, worst point", worst(best320, "320", "floor")),
        (f"{best320['id']}'s network term, cap 320, bar, worst point", worst(best320, "320", "bar")),
        ("1% allowance", 0.01)])
    cks = checks(inp, einp, ll, res)
    res["checks"] = [{"name": n, "ok": ok, "value": v} for n, ok, v in cks]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "egress.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    text = table(res)
    (args.out / "table.txt").write_text(text + "\n")
    print(text)
    for n, ok, v in cks:
        print(f"{'ok  ' if ok else 'FAIL'} {n} {v}")
    bad = sum(1 for _, ok, _ in cks if not ok)
    print(f"{len(cks) - bad}/{len(cks)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
