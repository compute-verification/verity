"""Lemma 1's detection step on Pearl-C's noised operands (question #4): point masses and low-weight wrong polynomials.

Lemma 1 ("Three lemmas toward eps < 1% PoUW") needs, for every nonzero difference D of degree <= 2 that a restricted circuit
can commit (operands are protocol entries or constants, and a constant may depend on the committed A and B), that
Pr[D(A~, B~) = 0] <= delta_d < 1/3 over the verifier's noise.  For D = x_ik - c (a product that reads the constant c where
it should read the operand) that probability is the point mass Pr[A~_ik = c], so delta_d >= max over entries and c.

Pearl-C v1 on sm120 forms A~_ik = e4m3(atom(alpha*x_ik; E'_i, F_k)) with E'_i = e4m3(beta_i * E_i) (`pearl_c.eprime`),
E_i a norm-256 line and F_k a norm-16 line (`pearl_c.sample_line`), beta_i = rho*alpha*sqrt(32)/(256*16): the noise
N_ik = E'_i . F_k has standard deviation ~ rho*alpha, and alpha*x_ik and N_ik are all an entry's law depends on.  A
"line pool" holds lines from independent random seeds; an entry's law is sampled by pairing an E line with an F line.

The experiments: the noise's bound (an entry at alpha*x >= 432 + max|N| is +-448 whatever the seeds); point masses on an
honest Gaussian row; the worst live entry's point mass by rho*alpha; the largest entry's by s/rho; Pr[D = 0] for low-weight
differences on honest and floor rows; how often the bit-exact chain hides a changed product (absorption); and the
spike-column attack on `pearl_c.form_v1` (every row's spikes in the same columns: those codes are known before the seeds).

    python detection.py [--draws N] [--json FILE]
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter

from verity.core.protocols.pouw.schemes import pearl_c as PC
from verity.core.protocols.pouw.schemes import pearl_c_device as PC_DEV
from verity.core.protocols.pouw.schemes.pearl_c_device import SM120
from verity.ml.circuits.silicon import fp32

P = PC.P
ATOM = SM120.atom


def pool(n: int, side: int, factor: int, norm: float, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [PC.sample_line(PC.S512Seed(rng.randbytes(32)), side, factor, rng.randrange(1 << 20), norm) for _ in range(n)]


class Pools:
    def __init__(self, n: int, seed: int):
        self.e = pool(n, 0, 0, P.NOISE_NORM, seed)
        self.f = pool(n, 0, 1, PC.LINE_NORM_V1, seed + 1)


def entry(v_word: int, ep: list[int], f: list[int]) -> int:
    """One noised code: e4m3 of the sm120 atom with alpha*x as its accumulator (`pearl_c.form_v1`'s inner step)."""
    return P.f32_to_fp8(P.f32_of_bits(ATOM.step(v_word, ep, f)))


def value(code: int) -> float:
    return P.fp8_to_f32(code)


def beta_for(rho_alpha: float) -> int:
    """beta_row = (rho*alpha) * S5_LINE as `pearl_c.s5_stats` rounds it."""
    return P.f32_bits(P.f32(P.f32(rho_alpha) * P.f32_of_bits(PC.S5_LINE)))


def law(v: float, beta: int, pools: Pools, draws: int, rng: random.Random) -> Counter:
    vw = P.f32_bits(P.f32(v))
    eps = {}
    out = Counter()
    for _ in range(draws):
        ie, f = rng.randrange(len(pools.e)), pools.f[rng.randrange(len(pools.f))]
        ep = eps.get(ie)
        if ep is None:
            ep = eps[ie] = PC.eprime(beta, pools.e[ie])
        out[entry(vw, ep, f)] += 1
    return out


def summary(c: Counter) -> dict:
    n = sum(c.values())
    code, top = c.most_common(1)[0]
    zero = (c[0x00] + c[0x80]) / n
    return {"p_max": round(top / n, 4), "mode": value(code), "p_zero": round(zero, 5), "support": len(c)}


def noise_std(beta: int, pools: Pools, draws: int, rng: random.Random) -> float:
    xs = []
    for _ in range(draws):
        ep = PC.eprime(beta, pools.e[rng.randrange(len(pools.e))])
        f = pools.f[rng.randrange(len(pools.f))]
        xs.append(sum(value(a) * value(b) for a, b in zip(ep, f)))
    return math.sqrt(sum(x * x for x in xs) / len(xs))


# ---- the rows ---------------------------------------------------------------------------------------------------------

def gauss_row(k: int, rng: random.Random) -> list[int]:
    return [P.f32_bits(rng.gauss(0, 1)) for _ in range(k)]


def spike_row(k: int, rng: random.Random, spike: float, n_spikes: int, live: float = 0.0) -> list[int]:
    """rho's positions (every 8th) at +-1, so rho = 1; `n_spikes` entries off those positions at +-spike (alpha = 448 /
    (spike + 4)); every other entry ~ live * N(0, 1).  At spike = 443, rho*alpha = 448/447: the domain's noise floor."""
    x = [(rng.choice((-1.0, 1.0)) if p % 8 == 0 else live * rng.gauss(0, 1)) for p in range(k)]
    spots = [p for p in range(k) if p % 8 != 0]
    for p in rng.sample(spots, n_spikes):
        x[p] = rng.choice((-spike, spike))
    return [P.f32_bits(v) for v in x]


def admissible(row: list[int]) -> tuple[bool, float, float, int]:
    s, rho, alpha, beta = PC.s5_stats(row)
    ra = P.f32_of_bits(rho) * P.f32_of_bits(alpha)
    return ra >= P.SIGMA_MIN and PC.live_row(row), ra, P.f32_of_bits(alpha), beta


# ---- experiments ------------------------------------------------------------------------------------------------------

def honest_profile(pools: Pools, draws: int, rng: random.Random, k: int = 8192) -> dict:
    """A Gaussian row: each sampled entry's point mass, by |x|/rho."""
    row = gauss_row(k, rng)
    ok, ra, alpha, beta = admissible(row)
    vals = sorted(((abs(P.f32_of_bits(w)), P.f32_of_bits(w)) for w in row))
    rho = ra / alpha
    picks = [vals[int(q * (k - 1))] for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)] + [vals[-1]]
    out = {"admissible": ok, "rho_alpha": ra, "noise_std_measured": noise_std(beta, pools, 2000, rng), "entries": []}
    for _, x in picks:
        c = law(alpha * x, beta, pools, draws, rng)
        out["entries"].append({"x_over_rho": round(x / rho, 3), "alpha_x": round(alpha * x, 3), **summary(c)})
    sample = rng.sample(range(k), 400)
    pm = sorted(summary(law(alpha * P.f32_of_bits(row[i]), beta, pools, 600, rng))["p_max"] for i in sample)
    out["p_max_over_400_random_entries"] = {"median": pm[200], "p90": pm[360], "p99": pm[396], "max": pm[-1],
                                            "frac_above_1_6": round(sum(p > 1 / 6 for p in pm) / 400, 4),
                                            "frac_above_1_3": round(sum(p > 1 / 3 for p in pm) / 400, 4)}
    return out


def upper95(p: float, n: int) -> float:
    """Wilson's 95% upper bound on a proportion p observed over n draws."""
    z = 1.96
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(min(1.0, (centre + half) / (1 + z * z / n)), 4)


def live_max(sigma: float, pools: Pools, draws: int, rng: random.Random) -> dict:
    """The largest point mass of a live entry (|alpha*x| < 8 rho*alpha) at rho*alpha = sigma: a scan of v over
    [0, top) in 64 steps (the law is symmetric in v's sign), then the three best v again at 8x the draws.  top is
    min(8 sigma, 448 - 4 sigma): at rho*alpha = sigma a row's largest |alpha*x| is alpha*s = 448 - 4 sigma."""
    beta = beta_for(sigma)
    top = min(8 * sigma, 448 - 4 * sigma)
    scan = [(v, summary(law(v, beta, pools, draws, rng))["p_max"]) for v in (i * top / 64 for i in range(64))]
    best = sorted(scan, key=lambda t: -t[1])[:3]
    fine = []
    for v0, _ in best:
        for v in (v0 + d * top / 512 for d in range(-4, 5)):
            if 0 <= v < top:
                fine.append((v, summary(law(v, beta, pools, draws * 8, rng))["p_max"]))
    v, p = max(fine, key=lambda t: t[1])
    return {"rho_alpha": sigma, "worst_v": round(v, 4), "p_max": p, "p_max_upper95": upper95(p, draws * 8),
            "sz_degree2_bound (2p)": round(2 * p, 4), "linearisation (1-(1-p)^2)": round(1 - (1 - p) ** 2, 4)}


def sigma_scan(pools: Pools, draws: int, rng: random.Random) -> list[dict]:
    return [live_max(s, pools, draws, rng) for s in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 56.0)]


def spike_scan(pools: Pools, draws: int, rng: random.Random) -> list[dict]:
    """A row's largest entry x = s at s/rho = r: rho*alpha = 448/(r + 4) and alpha*s = 448 r/(r + 4) (`pearl_c.s5_stats`),
    so the cast's spacing at alpha*s (32 above 256) meets noise of std rho*alpha.  The domain (rho*alpha >= 1) allows
    r up to 444; the liveness rule allows up to k/64 entries per row at |x| >= 8 rho."""
    out = []
    for r in (8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 352, 444):
        sigma, v = 448 / (r + 4), 448 * r / (r + 4)
        s = summary(law(v, beta_for(sigma), pools, draws, rng))
        out.append({"s_over_rho": r, "rho_alpha": round(sigma, 4), "alpha_s": round(v, 3), **s,
                    "p_max_upper95": upper95(s["p_max"], draws)})
    return out


def row_worst(pools: Pools, draws: int, rng: random.Random) -> list[dict]:
    """The worst entry of a row whose largest entry is s = r rho with r <= 8 (every entry live): rho*alpha = 448/(r + 4),
    and v = alpha*x ranges over [0, 448 r/(r + 4)]; a 33-point scan, then the two best v again at 4x the draws."""
    out = []
    for r in (3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0):
        sigma, top = 448 / (r + 4), 448 * r / (r + 4)
        beta = beta_for(sigma)
        scan = [(v, summary(law(v, beta, pools, draws, rng))["p_max"]) for v in (i * top / 32 for i in range(33))]
        fine = [(v, summary(law(v, beta, pools, 4 * draws, rng))["p_max"])
                for v0, _ in sorted(scan, key=lambda t: -t[1])[:2]
                for v in (v0 + d * top / 128 for d in range(-2, 3)) if 0 <= v <= top]
        v, p = max(fine, key=lambda t: t[1])
        out.append({"s_over_rho": r, "rho_alpha": round(sigma, 3), "alpha_s": round(top, 2), "worst_v": round(v, 3),
                    "p_max": p, "p_max_upper95": upper95(p, 4 * draws)})
    return out


def noise_bound(pools: Pools, rng: random.Random, draws: int = 20000) -> dict:
    """max |N| / (rho*alpha) over random pairs, and Cauchy-Schwarz's ||E'|| ||F|| / (rho*alpha) over the pools: the noise
    an entry at alpha*x >= 432 + that bound cannot cross (E4M3 rounds [432, 448] and above to 448, RNE ties to even)."""
    beta = beta_for(1.0)
    eps = [PC.eprime(beta, e) for e in pools.e]
    ne = max(math.sqrt(sum(value(c) ** 2 for c in ep)) for ep in eps)
    nf = max(math.sqrt(sum(value(c) ** 2 for c in f)) for f in pools.f)
    seen = 0.0
    for _ in range(draws):
        ep, f = eps[rng.randrange(len(eps))], pools.f[rng.randrange(len(pools.f))]
        seen = max(seen, abs(sum(value(a) * value(b) for a, b in zip(ep, f))))
    return {"max_abs_noise_seen": round(seen, 4), "cauchy_schwarz_over_pools": round(ne * nf, 4),
            "max_norm_E'": round(ne, 4), "max_norm_F": round(nf, 4), "draws": draws}


def patterns(rows: tuple[list[int], list[int]], pools: Pools, draws: int, rng: random.Random, trials: int = 40) -> dict:
    """Pr[D = 0] for low-weight differences on one pair of rows (row i of A, column j of B as a row), at random
    positions k1 != k2, with the constants the cheater's best (each law's mode, read off the draws): the mean over
    positions and the worst position.  x = A~_ik, y = B~_kj; x and y use independent lines (E_A, F_A vs E_B, F_B)."""
    ra_, rb_ = rows
    k = len(ra_)
    _, sa, alpha_a, beta_a = admissible(ra_)
    _, sb, alpha_b, beta_b = admissible(rb_)
    names = ("x - c", "x*y - c", "(x - c1)*(y - c2)", "omit x*y", "swap x1*y1 - x2*y2")
    acc = {name: [] for name in names}
    per = max(draws // trials, 50)
    for _ in range(trials):
        k1, k2 = rng.sample(range(k), 2)
        va = [P.f32_bits(P.f32(alpha_a * P.f32_of_bits(ra_[t]))) for t in (k1, k2)]
        vb = [P.f32_bits(P.f32(alpha_b * P.f32_of_bits(rb_[t]))) for t in (k1, k2)]
        xs, ys, prods = Counter(), Counter(), Counter()
        omit = swap = 0
        pairs = []
        for _ in range(per):
            ea = PC.eprime(beta_a, pools.e[rng.randrange(len(pools.e))])
            eb = PC.eprime(beta_b, pools.e[rng.randrange(len(pools.e))])
            x = [entry(v, ea, pools.f[rng.randrange(len(pools.f))]) for v in va]
            y = [entry(v, eb, pools.f[rng.randrange(len(pools.f))]) for v in vb]
            p1, p2 = value(x[0]) * value(y[0]), value(x[1]) * value(y[1])
            omit += p1 == 0
            swap += p1 == p2
            xs[x[0]] += 1
            ys[y[0]] += 1
            prods[p1] += 1
            pairs.append((x[0], y[0]))
        cx, cy = xs.most_common(1)[0][0], ys.most_common(1)[0][0]
        acc["x - c"].append(xs[cx] / per)
        acc["x*y - c"].append(prods.most_common(1)[0][1] / per)
        acc["(x - c1)*(y - c2)"].append(sum(a == cx or b == cy for a, b in pairs) / per)
        acc["omit x*y"].append(omit / per)
        acc["swap x1*y1 - x2*y2"].append(swap / per)
    return {"rho_alpha_A": round(sa, 3), "rho_alpha_B": round(sb, 3), "positions": trials, "draws_per_position": per,
            **{name: {"mean": round(sum(v) / len(v), 4), "worst": round(max(v), 4)} for name, v in acc.items()}}


def floor_rows(k: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """A pair of rows at the domain's floor, rho*alpha ~ 1, with live entries spread over [0, 8 rho): rho's positions at
    +-1, a single spike at 443 sets alpha, and the rest uniform in (-8, 8) (still live: |x| < 8 rho)."""
    def row():
        x = [(rng.choice((-1.0, 1.0)) if p % 8 == 0 else rng.uniform(-7.99, 7.99)) for p in range(k)]
        x[rng.choice([p for p in range(k) if p % 8])] = 443.0
        return [P.f32_bits(v) for v in x]
    return row(), row()


def formed_pair(rows: tuple[list[int], list[int]], rng: random.Random) -> tuple[list[int], list[int]]:
    """One A row and one B row (as a row) formed by `pearl_c.form_v1` on sm120 under fresh seeds: the operands' codes."""
    seed_b = PC.S512Seed(rng.randbytes(32))
    k = len(rows[0])
    a = PC.form_v1([rows[0]], [PC.sample_line(PC.S512Seed(rng.randbytes(32)), 0, 0, 0)],
                   PC._basis(seed_b, 0, k, PC.LINE_NORM_V1), SM120).codes[0]
    b = PC.form_v1([rows[1]], [PC.sample_line(seed_b, 1, 0, 0)], PC._basis(seed_b, 1, k, PC.LINE_NORM_V1), SM120).codes[0]
    return a, b


def _windows(a: list[int], b: list[int], g: int) -> list[int]:
    out = []
    for ws in range(0, len(a), 32 * g):
        local = 0
        for s in range(ws, ws + 32 * g, 32):
            local = ATOM.step(local, a[s:s + 32], b[s:s + 32])
        out.append(local)
    return out


def _promote(locals_: list[int]) -> int:
    """`pearl_c.chain`'s promotion: one FP32 nearest-even add per window into the total, a zero total +0."""
    total = 0
    for w in locals_:
        total = fp32.add(total, w)
        if total == 0x80000000:
            total = 0
    return total


def _unpromoted_from(a: list[int], b: list[int], states: list[int], j: int) -> int:
    acc = states[j]
    for s in range(32 * j, len(a), 32):
        acc = ATOM.step(acc, a[s:s + 32], b[s:s + 32])
    return acc


def absorption(pair: tuple[list[int], list[int]], trials: int, rng: random.Random) -> dict:
    """Pr[the committed word is unchanged] when one product is changed in the bit-exact chain (`pearl_c.chain`, sm120 with
    G = 4 and unpromoted): the product omitted (A~_ik -> 0), or A~_ik moved one code step toward zero.  The polynomial
    picture counts both as caught whenever D != 0; the chain's truncations can hide a change below its grid."""
    a, b = pair
    g = SM120.g
    base_locals = _windows(a, b, g)
    base_p = _promote(base_locals)
    states = [0]
    for s in range(0, len(a), 32):
        states.append(ATOM.step(states[-1], a[s:s + 32], b[s:s + 32]))
    base_u = states[-1]
    assert base_p == PC.chain(a, b, SM120) and base_u == PC.chain(a, b, PC_DEV.SM120_UNPROMOTED)
    hits = Counter()
    n = 0
    for _ in range(trials):
        t = rng.randrange(len(a))
        c = a[t]
        if c & 0x7F == 0:
            continue
        n += 1
        for name, new in (("omit", 0x00), ("one_code_step", c - 1 if c & 0x7F else c)):
            a2 = list(a)
            a2[t] = new
            w = t // (32 * g)
            locals2 = list(base_locals)
            locals2[w] = _windows(a2[32 * g * w: 32 * g * (w + 1)], b[32 * g * w: 32 * g * (w + 1)], g)[0]
            hits[f"{name}_promoted"] += _promote(locals2) == base_p
            hits[f"{name}_unpromoted"] += _unpromoted_from(a2, b, states, t // 32) == base_u
    return {"trials": n, **{key: {"p_unchanged": round(v / n, 5), "upper95": upper95(v / n, n)} for key, v in
                            sorted(hits.items())}}


def s5_stats_rho_s4(x: list[int]) -> tuple[int, int, int, int]:
    """The proposed fix: `pearl_c.s5_stats` with rho' = max(rho, s/4) in rho's place, so alpha = 448/(s + 4 rho') keeps
    alpha*s + 4 rho'*alpha <= 448 and every row's noise is at least alpha*s/4 (the s/rho = 4 law at worst)."""
    s, rho, _, _ = PC.s5_stats(x)
    rho2 = max(rho, fp32.mul(s, 0x3E800000))        # non-negative words order as their values
    alpha = fp32.div(PC.S5_QMAX, max(fp32.add(s, fp32.mul(PC.S5_FOUR, rho2)), PC.S5_FLOOR))
    return s, rho2, alpha, fp32.mul(fp32.mul(rho2, alpha), PC.S5_LINE)


def form(rows, e_rows, f, stats=PC.s5_stats) -> list[list[int]]:
    """`pearl_c.form_v1`'s codes on sm120, with the row statistics `stats`."""
    if stats is PC.s5_stats:
        return PC.form_v1(rows, e_rows, f, SM120).codes
    out = []
    for x, e_row in zip(rows, e_rows):
        _, _, alpha, beta = stats(x)
        ep = PC.eprime(beta, e_row)
        out.append([P.f32_to_fp8(P.f32_of_bits(ATOM.step(fp32.mul(alpha, xt), ep, f[t]))) for t, xt in enumerate(x)])
    return out


def spike_attack(rng: random.Random, k: int = 1024, m: int = 32, seeds: int = 8, stats=PC.s5_stats) -> dict:
    """Rows at the noise floor with k/64 spikes in the SAME columns for every row (the cheater's choice of A): each spike's
    noised code is predicted before the seeds exist (+-448, the sign of the spike), so the spike columns' products could be
    shared by every row (accumulator fan-out) and a wrong polynomial using the constant +-448 is never caught.  `stats`
    other than `pearl_c.s5_stats` forms the same rows under that fix; the hit rate is then the predicted code's."""
    cols = sorted(rng.sample([p for p in range(k) if p % 8], k // 64))
    rows, signs = [], []
    for _ in range(m):
        base = spike_row(k, rng, 443.0, 0)
        sg = rng.choice((-1, 1))
        for p in cols:
            base[p] = P.f32_bits(sg * 443.0)
        rows.append(base)
        signs.append(sg)
    adm = [admissible(r) for r in rows]
    t0 = time.time()
    codes = Counter()
    for s in range(seeds):
        seed_b = PC.S512Seed(rng.randbytes(32))
        seed_a = PC.S512Seed(rng.randbytes(32))
        f = PC._basis(seed_b, 0, k, PC.LINE_NORM_V1)
        e = [PC.sample_line(seed_a, 0, 0, i) for i in range(m)]
        formed = form(rows, e, f, stats)
        for i in range(m):
            for p in cols:
                codes[formed[i][p] if signs[i] > 0 else formed[i][p] ^ 0x80] += 1
    total = sum(codes.values())
    _, rho, alpha, _ = stats(rows[0])
    return {"k": k, "rows": m, "spike_columns": len(cols), "all_rows_admissible": all(a[0] for a in adm),
            "rho_alpha": round(P.f32_of_bits(rho) * P.f32_of_bits(alpha), 4), "seeds": seeds, "spike_entries": total,
            "predicted_code_hit_rate": codes.most_common(1)[0][1] / total, "predicted_code": value(codes.most_common(1)[0][0]),
            "products_shareable_fraction": round(len(cols) / k, 5), "seconds": round(time.time() - t0, 1)}


def main(argv: list[str]) -> int:
    draws = int(argv[argv.index("--draws") + 1]) if "--draws" in argv else 4000
    if "--fix" in argv:
        res = {"fix": "rho' = max(rho, s/4)", "spike_attack_v1": spike_attack(random.Random(5)),
               "spike_attack_fixed": spike_attack(random.Random(5), stats=s5_stats_rho_s4)}
        print(json.dumps(res, indent=1))
        if "--json" in argv:
            open(argv[argv.index("--json") + 1], "w").write(json.dumps(res, indent=1))
        return 0
    rng = random.Random(20261015)
    t0 = time.time()
    pools = Pools(3000, 7)
    res = {"draws": draws}
    res["noise_bound"] = noise_bound(pools, rng)
    res["honest_gaussian_row"] = honest_profile(pools, draws, rng)
    res["live_point_mass_by_sigma"] = sigma_scan(pools, draws // 4, rng)
    res["worst_entry_by_s_over_rho_live_rows"] = row_worst(pools, draws // 4, rng)
    res["largest_entry_by_s_over_rho"] = spike_scan(pools, draws, rng)
    k = 8192
    honest = (gauss_row(k, rng), [P.f32_bits(rng.gauss(0, 0.02)) for _ in range(k)])
    floor = floor_rows(k, rng)
    res["patterns_honest"] = patterns(honest, pools, draws * 5, rng)
    res["patterns_floor"] = patterns(floor, pools, draws * 5, rng)
    res["absorption_honest"] = absorption(formed_pair(honest, rng), draws // 2, rng)
    res["absorption_floor"] = absorption(formed_pair(floor, rng), draws // 2, rng)
    res["spike_attack"] = spike_attack(rng)
    res["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(res, indent=1)
    if "--json" in argv:
        open(argv[argv.index("--json") + 1], "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
