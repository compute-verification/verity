"""blc4a with the epilogue de-noise: its lane budget, its words on the real lane layout, and A5's advice tables on it.

The coordinator's lead `ssn-1l-tag + blc4a-t64-e7` (egress.py's blc4a) carries three private coin lanes in every step at
m ≥ 64 (A′ = 1 against B′ = s_t[tile, j], odd in ±127, drawn per call after A_j): step 1 holds 21 data lanes, the row tag
T(i) against G_j, the code bits a₁ against C_j and the 3 coin lanes; every later step 28 data lanes, the walk 1 + a_t
against 1 and the 3 coin lanes. ssn_1l_tag_xu's de-noise adds to the last step a₁ against −C_j and κ − T(i) against G_j
(8 cancel lanes), u₀ against 1, u₁ against 127 and u₂ in e lanes against 127. The final word is
A·Bᵀ + ρ_i + κ·G_j + σ_{tile, j}, with ρ_i = u₀ + 127·u₁ + 127e·u₂ + w_i and σ the tile-column sum of every step's coin
lanes, which the IADD3 removes through its column source loaded per tile.

- `layout`: the last step's spare lanes at each K and the step count with the de-noise's 10 + e lanes, one step more
  where they don't fit. Putting some of them in step S − 1 frees no lane: every step before the last is full, so any
  placement leaves the same data capacity, 21 + 28(S − 1) − 10 − e.
- `sim_blc`: the words on the real lane layout over a salt grid, in 2-row tiles: exact after the epilogue, distinct and
  non-free on every data case. `exact_full`: the same exactness at Llama's K on 64-row tiles, the magic epilogue equal
  to stock's bit for bit.
- A5 (N-RED): advice tables the possible step-t values before the call's coins, so a hit is a copy, and a run of hits
  pays one write less than its length (the resume load). N-RED's row vectors (`a5_rows`, its `a5_step_greedy`) can't
  hold σ_t[tile, ·]'s n per-call coins, so blc4a leaves nothing to table from step 1. A word's own table (`Chain`) can:
  its address is a row index (a₁ and the walk) plus a column index (σ_t[tile, j], with G_j's digits when G is drawn per
  call), shared by the tile's rows, one load and no write, so under model P (copies of one word free) an entry pays
  when the word's coins land in it. The best top-set tables over steps 1..T by a Lagrangian DP, allotted over HBM and
  the call's chains; `linear_bound` bounds every table; under model C a copy moves 4 columns, whose coins its one
  address must hold jointly.

Output: JSON (`--json`); exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_sweep as es  # noqa: E402
import noise_sweep as ns  # noqa: E402

FIRST = 21
DATA = 28
COIN_LANES = 3
TILE = 64
KS = (2048, 7168, 8192)
FIXED = es.TAGX_FIXED
CANCEL = 8
HBM_BYTES = 141e9
ENTRY_BYTES = 4
MS = (64, 512, 8192)
CODES = 16
#: rows 0..11 carry tags ±1..±6 in digit 0 alone, so under a per-call G one column index (σ, G_0j) serves them
G_GROUP_ROWS = 12
#: rows below 168 have tags of at most two digits, so a tile of them shares the column index (σ, G_0j, G_1j)
G_TWO_DIGIT_ROWS = 168
CGRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536,
         2048, 2560, 3072)


def blc_steps(k: int) -> int:
    return 1 + math.ceil(max(0, k - FIRST) / DATA)


def spare(k: int, s: int) -> int:
    return FIRST + DATA * (s - 1) - k


def blcx_steps(k: int) -> int:
    s = blc_steps(k)
    while spare(k, s) < FIXED:
        s += 1
    return s


def blcx_extra(k: int) -> int:
    return min(es.TAGX_EXTRA_MAX, spare(k, blcx_steps(k)) - FIXED)


def data_lanes(k: int, s: int, added: int, split: int = 0) -> list[int] | None:
    """Data lanes a step for K entries on s steps with `added` de-noise lanes, `split` of them in step s − 1, or None
    where they don't fit."""
    free = [FIRST] + [DATA] * (s - 1)
    free[-1] -= added - split
    if split:
        free[-2] -= split
    if min(free) < 0 or sum(free) < k:
        return None
    out, left = [], k
    for f in free:
        out.append(min(f, left))
        left -= out[-1]
    return out


def walk_pmf(s: int) -> np.ndarray:
    return np.array([math.comb(s - 1, i) / 2 ** (s - 1) for i in range(s)])


def offset_q(k: int) -> float:
    """The final row offset's 1/Σp² on blc4a with the de-noise, as `es.tagx_offset_q` on the lead."""
    s, e, byte = blcx_steps(k), blcx_extra(k), np.arange(-128, 128)
    return es._q([(1, np.arange(128)), (es.LANE_MAX, byte)] + ([(es.LANE_MAX * e, byte)] if e else []), walk_pmf(s))


def layout() -> dict:
    out = {}
    for k in KS:
        s, sx = blc_steps(k), blcx_steps(k)
        sp = spare(k, s)
        rep_q = es._q([], walk_pmf(s - 1))
        out[str(k)] = {
            "S": s, "spare_at_S": sp, "denoise_fits_at_S": sp >= FIXED,
            "deficit_at_S_e0": max(0, FIXED - sp),
            "placements_fitting_at_S": [x for x in range(FIXED + 1) if data_lanes(k, s, FIXED, x) is not None],
            "S_x": sx, "e_x": blcx_extra(k), "added_steps": sx - s, "words_increment": sx / s - 1,
            "final_offset_q_log2": math.log2(offset_q(k)),
            "old_lead_final_offset_q_log2": math.log2(es.tagx_offset_q(k)),
            "chosen_rows_final_loss_m8192": (TILE - 1) / (2 * offset_q(k) * sx),
            # the alternative: the last step's walk and 3 coin lanes give way to the cancel lanes, and u goes
            "replace_lanes_fits": sp + 1 + COIN_LANES >= CANCEL,
            "replace_lanes_offset_q": rep_q,
            "replace_lanes_chosen_rows_final_loss": (TILE - 1) / (2 * rep_q * s),
        }
    return out


def price_added_step() -> dict:
    """One more step at K 8,192 (down): its word and its walk bit (estimate). Honest and adversary both write the word,
    so T1 moves only by the walk bit's forming; T2 by the step."""
    lins = ns.LLAMA
    words = {lin: blcx_steps(k) for lin, (n, k) in lins.items()}
    base = {lin: blc_steps(k) for lin, (n, k) in lins.items()}
    tot_words = sum(n * words[lin] for lin, (n, k) in lins.items())
    added = sum(n * (words[lin] - base[lin]) for lin, (n, k) in lins.items())
    macs = sum(n * k for n, k in lins.values())
    forming = sum(ns.H32 * (ns.ssn_1l_forming_writes(words[lin]) - ns.ssn_1l_forming_writes(base[lin]))
                  for lin in lins) / macs
    return {"added_words_share_of_linears_words": added / tot_words,
            "down_words_increment": words["down"] / base["down"] - 1,
            "t1_forming_share_of_stock": forming,
            "alpha_blcx": ns.H32 * tot_words / macs,
            "alpha_blc": ns.H32 * sum(n * base[lin] for lin, (n, k) in lins.items()) / macs}


# ---------------------------------------------------------------- the words on blc4a's lane layout

def operands(a, b, s, e, tile, a1, at, u, g, coins, kappa):
    """One salt's X (m × 32s) and Y (tiles × 32s × n) on blc4a with the de-noise in step s, the row offset ρ and the
    column source c (m × n: κ·G_j + σ_{tile(i), j})."""
    m, k = a.shape
    n = b.shape[1]
    tiles = -(-m // tile)
    dl = data_lanes(k, s, FIXED + e)
    assert dl is not None, (k, s, e)
    tag, code = es.tag_digits(m), ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    one, kap = np.ones((1, n), dtype=np.int64), np.asarray(kappa)
    u0, u1, u2 = u
    xs, ys, lo = [], [[] for _ in range(tiles)], 0
    for t in range(s):
        if t == 0:
            lx, ly = [tag, a1], [g, code]
        else:
            lx, ly = [at[:, t:t + 1]], [one]
        if t == s - 1:
            lx += [a1, kap - tag, u0, u1] + [u2] * e
            ly += [-code, g, one, es.LANE_MAX * one] + [es.LANE_MAX * one] * e
        lx, ly = np.concatenate(lx, axis=1), np.concatenate(ly, axis=0)
        pad = 32 - COIN_LANES - lx.shape[1] - dl[t]
        assert pad >= 0
        xs += [a[:, lo:lo + dl[t]], np.zeros((m, pad), np.int64), lx, np.ones((m, COIN_LANES), np.int64)]
        for q in range(tiles):
            ys[q] += [b[lo:lo + dl[t]], np.zeros((pad, n), np.int64), ly, coins[t, :, q, :]]
        lo += dl[t]
    assert lo == k
    x, y = np.concatenate(xs, axis=1), np.stack([np.concatenate(v, axis=0) for v in ys])
    assert x.min() >= -128 and x.max() <= es.LANE_MAX and np.abs(y).max() <= es.LANE_MAX
    rho = (u0 + es.LANE_MAX * u1 + es.LANE_MAX * e * u2)[:, 0] + at[:, 1:].sum(1)
    sigma = coins.sum(axis=(0, 1))
    col = (kap @ g)[None, :] + sigma[np.arange(m) // tile]
    return x, y, rho, col


def tile_words(x: np.ndarray, y: np.ndarray, tile: int) -> np.ndarray:
    """Committed words salts × m × n × steps, each row tile against its own B′."""
    sa, m, w = x.shape
    n = y.shape[-1]
    out = np.empty((sa, m, n, w // 32), dtype=np.int64)
    for q in range(y.shape[1]):
        r = slice(q * tile, (q + 1) * tile)
        prods = np.einsum("smk,skn->smnk", x[:, r], y[:, q])
        out[:, r] = prods.reshape(sa, -1, n, w // 32, 32).sum(-1).cumsum(-1)
    return out


def _coins(rng, s, tiles, n):
    return 2 * rng.integers(0, 128, (s, COIN_LANES, tiles, n)) - 127


def _signed(v):
    return ((v & 0xFF) ^ 0x80) - 0x80


SIM_CASES = ns.CASES + ["identical_rows_tiles", "zero_all", "identical_all"]


def sim_blc(case: str, k: int, s: int, e: int, salts: int = 24, m: int = 4, n: int = 6, tile: int = 2,
            seed: int = 1) -> dict:
    """blc4a + de-noise on s steps over a salt grid (raw coin words, G and the lane coins free; T(i) a constant)."""
    rng = np.random.default_rng(seed)
    a, b = ns._data(case if case in ns.CASES else "random", m, n, k, rng)
    if case == "identical_rows_tiles":
        a[2] = a[0]
    elif case == "zero_all":
        a[:] = 0
    elif case == "identical_all":
        a[:] = a[0]
    tiles = -(-m // tile)
    xs, ys, raw, rhos, cols = [], [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        hu = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        g = rng.integers(-3, 4, (4, n))
        coins = _coins(rng, s, tiles, n)
        a1 = np.stack([(hr >> q) & 1 for q in range(4)], axis=1)
        at = np.stack([np.zeros(m, dtype=np.int64)] + [1 + ((hr >> (4 + t)) & 1) for t in range(s - 1)], axis=1)
        u = (hu[:, None] & 0x7F, _signed(hu[:, None] >> 8), _signed(hu[:, None] >> 16))
        x, y, rho, col = operands(a, b, s, e, tile, a1, at, u, g, coins, es.TAGX_KAPPA)
        xs.append(x), ys.append(y), rhos.append(rho), cols.append(col)
        raw.append(np.concatenate([hr, hu, g.ravel(), coins.ravel()]))
    words = tile_words(np.stack(xs), np.stack(ys), tile)
    out = ns._distinct([words], [np.stack(raw)])
    resid = words[..., -1] - np.stack(rhos)[:, :, None] - np.stack(cols)
    out["exact_after_epilogue"] = bool((resid == (a @ b)[None]).all())
    out["final_bad"] = ns._distinct([words[..., -1]], [np.stack(raw)])["bad_share"]
    return out


def exact_full(k: int, m: int = 128, n: int = 8, salts: int = 2, seed: int = 5) -> dict:
    """blc4a + de-noise at Llama's K with its step count, on 64-row tiles: the final word minus ρ_i and the per-tile
    column source is A·Bᵀ, and the magic epilogue equals stock's bit for bit (22-bit column scales)."""
    rng = np.random.default_rng(seed)
    s, e = blcx_steps(k), blcx_extra(k)
    a, b = ns._data("random", m, n, k, rng)
    tiles = -(-m // TILE)
    exact, epi, peak = True, True, 0
    for _ in range(salts):
        a1 = rng.integers(0, 2, (m, 4))
        at = np.concatenate([np.zeros((m, 1), np.int64), 1 + rng.integers(0, 2, (m, s - 1))], axis=1)
        u = (rng.integers(0, 128, (m, 1)), rng.integers(-128, 128, (m, 1)), rng.integers(-128, 128, (m, 1)))
        g = rng.integers(-3, 4, (4, n))
        x, y, rho, col = operands(a, b, s, e, TILE, a1, at, u, g, _coins(rng, s, tiles, n), es.TAGX_KAPPA)
        w = tile_words(x[None], y[None], TILE)[0]
        peak = max(peak, int(np.abs(w).max()))
        p = a @ b
        exact &= bool((w[..., -1] - rho[:, None] - col == p).all())
        s_col = es.round_scale(es._f32(rng.uniform(1e-4, 1e-2, n)))
        s_row = es._f32(rng.uniform(1e-3, 1.0, m))
        epi &= bool((es.magic_epilogue(w[..., -1], rho, col, s_col, s_row) == es.stock_epilogue(p, s_col, s_row)).all())
    return {"S": s, "e": e, "exact_after_epilogue": exact, "magic_equals_stock": epi, "peak_word_log2": math.log2(peak),
            "max_abs_out_log2": math.log2(np.abs(a @ b).max()), "in_magic_range": bool(np.abs(a @ b).max() < es.MAGIC_RANGE)}


# ---------------------------------------------------------------- A5: advice tables

def a5_rows(vectors, steps_fn=ns.steps1l) -> dict:
    """N-RED's `a5_step_greedy` (nred_round2.py): every possible step-t vector of a row's steps 1..τ_ℓ tabled for each
    linear in one H200's HBM, a copy each; the row resumes with a 32-a-word load unless τ_ℓ = S. Best (τ_ℓ) by
    exhaustive search, as a share of stock."""
    out = {}
    lins = [(n, k, steps_fn(k)) for n, k in ns.LLAMA.values()]
    for m in MS:
        per = []
        for n, k, s in lins:
            unit = n * m * ns.LLAMA_LAYERS
            cost, opts = 0.0, [(0.0, 0.0, 0)]
            for t in range(1, s + 1):
                cost += vectors(t) * 4 * unit
                if cost > HBM_BYTES:
                    break
                opts.append((cost, ns.H32 * unit * (t if t == s else t - 1), t))
            per.append(opts)
        best = (0.0, (0,) * len(lins))
        last_costs = [o[0] for o in per[3]]
        for o0 in per[0]:
            for o1 in per[1]:
                if o0[0] + o1[0] > HBM_BYTES:
                    break
                for o2 in per[2]:
                    room = HBM_BYTES - o0[0] - o1[0] - o2[0]
                    if room < 0:
                        break
                    o3 = per[3][bisect.bisect_right(last_costs, room) - 1]
                    gain = o0[1] + o1[1] + o2[1] + o3[1]
                    if gain > best[0]:
                        best = (gain, (o0[2], o1[2], o2[2], o3[2]))
        stock = sum(n * k * m * ns.LLAMA_LAYERS for n, k, _ in lins)
        out[f"m{m}"] = {"share": best[0] / stock, "steps_tabled": dict(zip(ns.LLAMA, best[1]))}
    return out


def coin_pmf() -> np.ndarray:
    """An odd coin uniform on ±127, on the integers −127..127."""
    p = np.zeros(255)
    p[0::2] = 1 / 128
    return p


def top_order(p: np.ndarray) -> np.ndarray:
    idx = np.arange(len(p))
    return np.lexsort((np.abs(idx - (len(p) - 1) / 2), -p))


def top_mass(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.sort(p)[::-1])])


def pair_matrix(p: np.ndarray, inc: np.ndarray, grid: np.ndarray, grid1: np.ndarray) -> np.ndarray:
    """P(X ∈ top_c(X) ∧ X + I ∈ top_c'(X + I)) for c in grid, c' in grid1 (top_0 empty), X ~ p and I ~ inc independent."""
    p1 = np.convolve(p, inc)
    o, o1 = top_order(p), top_order(p1)
    out = np.zeros((len(grid), len(grid1)))
    for j, c1 in enumerate(grid1):
        if c1 == 0:
            continue
        ind = np.zeros(len(p1))
        ind[o1[:c1]] = 1.0
        h = np.correlate(ind, inc, "valid")
        cs = np.concatenate([[0.0], np.cumsum((p * h)[o])])
        out[:, j] = cs[np.minimum(grid, len(p))]
    return out


class Chain:
    """One word's chain (a row, a column, a layer) under per-word tables: at step t the row index is (a₁, w_t) and the
    column index σ_t (the sum of `lanes` odd coins a step), each a top set; T steps from step 1."""

    def __init__(self, lanes: int = COIN_LANES, steps: int = 8, cgrid=CGRID):
        coin = coin_pmf()
        inc_c = np.array([1.0])
        for _ in range(lanes):
            inc_c = np.convolve(inc_c, coin)
        inc_w = np.array([0.5, 0.5])
        self.lanes, self.T, self.inc_c, self.inc_w = lanes, steps, inc_c, inc_w
        self.pc, self.pw = [inc_c], [np.array([1.0])]
        for _ in range(1, steps):
            self.pc.append(np.convolve(self.pc[-1], inc_c))
            self.pw.append(np.convolve(self.pw[-1], inc_w))
        self.gc = [np.unique([0] + [c for c in cgrid if c < np.count_nonzero(p)] + [np.count_nonzero(p)])
                   for p in self.pc]
        self.gw = [np.arange(1, np.count_nonzero(p) + 1) for p in self.pw]
        self.SC = [top_mass(p)[np.minimum(g, len(p))] for p, g in zip(self.pc, self.gc)]
        self.SW = [top_mass(p)[g] for p, g in zip(self.pw, self.gw)]
        self.PC = [pair_matrix(self.pc[t], inc_c, self.gc[t], self.gc[t + 1]) for t in range(steps - 1)]
        self.PW = [pair_matrix(self.pw[t], inc_w, self.gw[t], self.gw[t + 1]) for t in range(steps - 1)]

    def pmax(self, t: int) -> float:
        """The likeliest (a₁, w_t, σ_t) at step t (0-based)."""
        return self.pc[t].max() * self.pw[t].max() / CODES

    def solve(self, lam: float, idx: float, grain: int = 1, gfac: float = 1.0, pairs: bool = True) -> dict:
        """max gain − λ·entries − idx·(steps tabled) over top sets at steps 1..T; gain in words a chain (a hit pair, or
        a hit with `pairs` False), entries CODES·w·c^grain a step, a hit's column factor gfac^grain."""
        T = self.T
        cs = [np.repeat(g, len(w)) for g, w in zip(self.gc, self.gw)]
        ws = [np.tile(w, len(g)) for g, w in zip(self.gc, self.gw)]
        ent = [CODES * w * c.astype(float) ** grain for c, w in zip(cs, ws)]
        used = [(c > 0).astype(float) for c in cs]
        fac = grain * gfac ** grain
        single = [fac * np.outer(sc ** grain, sw).ravel() for sc, sw in zip(self.SC, self.SW)]
        pair = [fac * np.einsum("ab,cd->acbd", pc ** grain, pw).reshape(len(cs[t]), len(cs[t + 1]))
                for t, (pc, pw) in enumerate(zip(self.PC, self.PW))]
        V, nxt = [None] * T, [None] * T
        for t in range(T - 1, -1, -1):
            base = (0.0 if pairs else single[t]) - lam * ent[t] - idx * used[t]
            if t == T - 1:
                V[t] = base
            else:
                tot = (pair[t] if pairs else np.zeros((len(cs[t]), len(cs[t + 1])))) + V[t + 1][None, :]
                nxt[t] = tot.argmax(1)
                V[t] = base + tot.max(1)
        path = [int(V[0].argmax())]
        for t in range(T - 1):
            path.append(int(nxt[t][path[-1]]))
        gain = sum(pair[t][path[t], path[t + 1]] for t in range(T - 1)) if pairs else \
            sum(single[t][path[t]] for t in range(T))
        steps_used = sum(used[t][path[t]] for t in range(T))
        return {"entries": float(sum(ent[t][path[t]] for t in range(T))), "gain": float(gain),
                "net": float(gain - idx * steps_used), "sets": [(int(cs[t][path[t]]), int(ws[t][path[t]]))
                                                               for t in range(T)]}

    def frontier(self, idx: float, grain: int = 1, gfac: float = 1.0, pairs: bool = True) -> list[dict]:
        pts = [self.solve(lam, idx, grain, gfac, pairs) for lam in np.logspace(-12, 0, 97)]
        return [p for p in pts if p["entries"] > 0 and p["net"] > 0]

    def best_ratio(self, idx: float, grain: int = 1, gfac: float = 1.0, pairs: bool = True) -> dict:
        """Dinkelbach: the tables of the highest net gain an entry."""
        lam, best = 0.0, None
        for _ in range(60):
            sol = self.solve(lam, idx, grain, gfac, pairs)
            if sol["entries"] == 0 or sol["net"] <= 0:
                break
            best = sol
            r = sol["net"] / sol["entries"]
            if r <= lam * (1 + 1e-12):
                break
            lam = r
        return {**best, "ratio": best["net"] / best["entries"]} if best else {"ratio": 0.0, "entries": 0.0,
                                                                              "gain": 0.0, "net": 0.0, "sets": []}


def allot(classes: list[tuple[float, list[dict]]], budget: float) -> float:
    """Words saved by the best mix of per-chain tables over chain classes (count, frontier) within `budget` entries:
    each class's upper concave hull from (0, 0), segments bought by slope."""
    segs = []
    for count, pts in classes:
        hull = [(0.0, 0.0)]
        for e, g in sorted({(p["entries"], p["net"]) for p in pts}):
            while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (e - hull[-2][0]) <= \
                    (g - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
                hull.pop()
            if g > hull[-1][1]:
                hull.append((e, g))
        for (e0, g0), (e1, g1) in zip(hull, hull[1:]):
            segs.append(((g1 - g0) / (e1 - e0), count * (e1 - e0), count * (g1 - g0)))
    saved, left = 0.0, budget
    for slope, cost, gain in sorted(segs, reverse=True):
        take = min(1.0, left / cost)
        saved += take * gain
        left -= take * cost
        if left <= 0:
            break
    return saved


def stock(m: int) -> float:
    return m * ns.LLAMA_MACS_TOKEN


def chains(rows: float, grain: int = 1) -> float:
    return rows * sum(n / grain for n, _ in ns.LLAMA.values()) * ns.LLAMA_LAYERS


def linear_bound(ch: Chain, m: int, grain: int = 1, gfac: float = 1.0, pairs: bool = True) -> float:
    """Any tables: a pair's second hit (a hit, with `pairs` False) is one entry holding the realized state, so the
    words saved are at most HBM's entries times the likeliest state at a step ≥ 2 (≥ 1), as a share of stock."""
    first = 1 if pairs else 0
    p = max(grain * (ch.pc[t].max() * gfac) ** grain * ch.pw[t].max() / CODES for t in range(first, ch.T))
    return ns.H32 * HBM_BYTES / ENTRY_BYTES * p / stock(m)


def a5_words(ch: Chain, old: Chain) -> dict:
    """Per-word tables on blc4a: G known to advice (a window coin, or advice after G), and G per call as built, where
    the column index carries G's digits (one digit on rows 0..11, two on a tile below row 168); gross (index writes
    free) and net (a column index a (tile or group, column, step), a row index a (row, step))."""
    budget = HBM_BYTES / ENTRY_BYTES
    out = {}
    for m in MS:
        n_avg = sum(n for n, _ in ns.LLAMA.values()) / len(ns.LLAMA)
        rec = {}
        for pairs in (True, False):
            tag = "" if pairs else "_resume_free"
            fr = ch.frontier(0.0, pairs=pairs)
            rec[f"P_G_known_gross{tag}"] = ns.H32 * allot([(chains(m), fr)], budget) / stock(m)
            rec[f"P_G_known_bound{tag}"] = linear_bound(ch, m, pairs=pairs)
            idx = 1 / min(m, TILE) + 1 / n_avg
            rec[f"P_G_known_net{tag}"] = ns.H32 * allot([(chains(m), ch.frontier(idx, pairs=pairs))], budget) / stock(m)
            g1, g2 = min(m, G_GROUP_ROWS), max(0, min(m, G_TWO_DIGIT_ROWS) - G_GROUP_ROWS)
            gross = [(chains(g1), ch.frontier(0.0, gfac=1 / 7, pairs=pairs)),
                     (chains(g2), ch.frontier(0.0, gfac=1 / 49, pairs=pairs))]
            rec[f"P_G_call_gross{tag}"] = ns.H32 * allot([c for c in gross if c[0] > 0], budget) / stock(m)
            rec[f"P_G_call_bound{tag}"] = linear_bound(ch, m, gfac=1 / 7, pairs=pairs)
            net = [(chains(g1), ch.frontier(1 / g1 + 1 / n_avg, gfac=1 / 7, pairs=pairs)),
                   (chains(g2), ch.frontier(1 / min(g2, TILE) + 1 / n_avg, gfac=1 / 49, pairs=pairs) if g2 else [])]
            rec[f"P_G_call_net{tag}"] = ns.H32 * allot([c for c in net if c[0] > 0], budget) / stock(m)
        rec["C_G_known_bound"] = linear_bound(ch, m, grain=4)
        rec["C_G_known_bound_resume_free"] = linear_bound(ch, m, grain=4, pairs=False)
        rec["old_lead_rows_model_P"] = ns.H32 * allot([(chains(m), old.frontier(0.0))], budget) / stock(m)
        out[f"m{m}"] = rec
    out["best_tables_G_known"] = ch.best_ratio(0.0)
    out["best_tables_G_known_resume_free"] = ch.best_ratio(0.0, pairs=False)
    out["best_tables_G_call_group"] = ch.best_ratio(0.0, gfac=1 / 7)
    out["pmax_state_step"] = [ch.pmax(t) for t in range(ch.T)]
    return out


def a5_monte_carlo(ch: Chain, sol: dict, chains_n: int = 400_000, s: int = 74, seed: int = 9) -> dict:
    """Draw the coins of `chains_n` chains, table `sol`'s top sets (every a₁ code), and count the adversary's saved writes
    directly: S − (misses + resumes), a resume after each run of hits that ends before step S."""
    rng = np.random.default_rng(seed)
    T = len(sol["sets"])
    sig = np.zeros(chains_n, np.int64)
    walk = np.zeros(chains_n, np.int64)
    hits = np.zeros((chains_n, s), bool)
    for t in range(T):
        sig += (2 * rng.integers(0, 128, (chains_n, ch.lanes)) - 127).sum(1)
        if t:
            walk += rng.integers(0, 2, chains_n)
        c, w = sol["sets"][t]
        if c == 0:
            continue
        oc, ow = top_order(ch.pc[t]), top_order(ch.pw[t])
        cset = oc[:c] - 127 * ch.lanes * (t + 1)
        wset = ow[:w]
        hits[:, t] = np.isin(sig, cset) & np.isin(walk, wset)
    misses = (~hits).sum(1)
    resumes = (hits[:, :-1] & ~hits[:, 1:]).sum(1)
    saved = s - misses - resumes
    pair_hits = (hits[:, :-1] & hits[:, 1:]).sum(1)
    se = saved.std() / math.sqrt(chains_n)
    return {"predicted": sol["gain"], "saved_mean": float(saved.mean()), "se": float(se),
            "pairs_mean": float(pair_hits.mean()), "within_5se": bool(abs(saved.mean() - sol["gain"]) < 5 * se + 1e-12),
            "saved_equals_pairs": bool((saved == pair_hits).all())}


# ---------------------------------------------------------------- report

def report() -> dict:
    checks = []

    def ok(name, cond, detail):
        checks.append({"check": name, "pass": bool(cond), "detail": detail})

    lay = layout()
    ok("blc4a's step counts are N-EGRESS's (21 data lanes in step 1, 28 later): 74 / 257 / 293 at K 2,048 / 7,168 / 8,192, "
       "with 17 / 21 / 5 spare lanes in the last step",
       [lay[str(k)]["S"] for k in KS] == [74, 257, 293] and [lay[str(k)]["spare_at_S"] for k in KS] == [17, 21, 5],
       {k: (v["S"], v["spare_at_S"]) for k, v in lay.items()})
    ok("the de-noise's 10 + e lanes fit the last step at K 2,048 (e 7) and 7,168 (e 11); at K 8,192 no placement of them "
       "across steps S − 1 and S fits (5 lanes short at e 0), and one step more (294) fits them at e 12",
       lay["2048"]["denoise_fits_at_S"] and lay["7168"]["denoise_fits_at_S"] and lay["8192"]["placements_fitting_at_S"] == []
       and lay["8192"]["deficit_at_S_e0"] == 5 and lay["8192"]["S_x"] == 294 and lay["8192"]["e_x"] == 12
       and lay["2048"]["e_x"] == 7 and lay["7168"]["e_x"] == 11,
       {k: {x: v[x] for x in ("placements_fitting_at_S", "deficit_at_S_e0", "S_x", "e_x")} for k, v in lay.items()})
    ok("giving the last step's walk and coin lanes to the cancel lanes fits at K 8,192 but leaves the final row offset "
       "the walk alone, Q < 64, against Q > 2^17 with one step more",
       lay["8192"]["replace_lanes_fits"] and lay["8192"]["replace_lanes_offset_q"] < 64
       and min(lay[str(k)]["final_offset_q_log2"] for k in KS) > 17,
       {k: (v["replace_lanes_offset_q"], v["final_offset_q_log2"]) for k, v in lay.items()})

    sims = {}
    for label, (k, s, e) in {"fits_e7": (FIRST + DATA * 4 - FIXED - 7, 5, 7),
                             "added_step_e12": (FIRST + DATA * 3 - 5, 5, 12)}.items():
        assert label != "added_step_e12" or (blcx_steps(k), blcx_extra(k)) == (s, e)
        sims[label] = {case: sim_blc(case, k, s, e) for case in SIM_CASES}
    ok("on blc4a's lane layout with the de-noise (2-row tiles, 24 salts, 10 data cases, the de-noise in the last step "
       "and in an added step): every word distinct and non-free, the final word included, and the final word minus "
       "ρ_i and the per-tile column source κ·G_j + σ_{tile, j} is A·Bᵀ",
       all(r["bad_share"] == 0 and r["final_bad"] == 0 and r["exact_after_epilogue"]
           for v in sims.values() for r in v.values()),
       {lab: {c: (r["bad_share"], r["final_bad"], r["exact_after_epilogue"]) for c, r in v.items()}
        for lab, v in sims.items()})
    full = {str(k): exact_full(k) for k in KS}
    ok("at Llama's K with S 74 / 257 / 294 on 64-row tiles: exact after the IADD3 with the per-tile column source, and "
       "the magic epilogue equals stock's bit for bit; no word nears 2^31",
       all(v["exact_after_epilogue"] and v["magic_equals_stock"] and v["in_magic_range"] and v["peak_word_log2"] < 30
           for v in full.values()), full)

    rows_old = a5_rows(lambda t: 16 * t)
    rows_blc = a5_rows(lambda t: float("inf"), blcx_steps)
    ok("N-RED's row-vector A5 reproduces its old-lead figure (ssn_1l, 16·t vectors: 14.87% at m 64) and is 0 at every m "
       "on blc4a, whose σ_t[tile, ·] puts n per-call coins in every row vector from step 1",
       abs(rows_old["m64"]["share"] - 0.1487) < 5e-4 and all(v["share"] == 0 for v in rows_blc.values()),
       {"old": rows_old, "blc4a": rows_blc})

    ch, old = Chain(steps=12), Chain(lanes=0, steps=24)
    words = a5_words(ch, old)
    ok("per-word tables on the old lead (no lane coins) save at least N-RED's row vectors (the same entries, partial "
       "walk sets allowed)", all(words[f"m{m}"]["old_lead_rows_model_P"] >= rows_old[f"m{m}"]["share"] - 1e-9 for m in MS),
       {m: (words[f"m{m}"]["old_lead_rows_model_P"], rows_old[f"m{m}"]["share"]) for m in MS})
    ok("the best tables found stay under the bound for every table, for each G timing and resume rule",
       all(words[f"m{m}"][f"P_G_{g}_gross{r}"] <= words[f"m{m}"][f"P_G_{g}_bound{r}"] * (1 + 1e-9)
           for m in MS for g in ("known", "call") for r in ("", "_resume_free")),
       {m: {key: v for key, v in words[f"m{m}"].items()} for m in MS})
    mc = {"pairs": a5_monte_carlo(ch, words["best_tables_G_known"])}
    ok("Monte Carlo of the best G-known tables (400k chains of 74 steps): the saved writes, counted as S − misses − "
       "resumes, equal the hit pairs chain by chain and match the computed gain within 5 standard errors",
       mc["pairs"]["within_5se"] and mc["pairs"]["saved_equals_pairs"], mc)
    ok("under model C (a copy moves 4 columns, one address) per-word tables save under 1e-6 of stock at every m, even "
       "with G known and resumes free",
       max(words[f"m{m}"]["C_G_known_bound_resume_free"] for m in MS) < 1e-6,
       {m: words[f"m{m}"]["C_G_known_bound_resume_free"] for m in MS})

    alpha = price_added_step()["alpha_blcx"]
    per = {f"{key}_m{m}": ns.periods(words[f"m{m}"][key], alpha)
           for m in MS for key in ("P_G_known_gross", "P_G_call_gross", "P_G_call_net")}
    return {"checks": checks, "layout": lay, "price_added_step": price_added_step(), "sim": sims, "full": full,
            "a5_rows_old": rows_old, "a5_rows_blc4a": rows_blc, "a5_words": words, "a5_monte_carlo": mc,
            "periods": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args(argv)
    rep = report()
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    for c in rep["checks"]:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"])
    print(json.dumps({"layout": rep["layout"], "price_added_step": rep["price_added_step"],
                      "a5_words": rep["a5_words"], "a5_monte_carlo": rep["a5_monte_carlo"]}, indent=1, default=float))
    return 0 if all(c["pass"] for c in rep["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
