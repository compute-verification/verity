"""An accounting subprotocol's target check over a stratified sampling design: missable amount, rate, and the cheapest
design that meets a target, from the partition's per-stratum public facts (template, count, amount per proof block).

A design is a list of strata. A stratum is `Sub(n, k, c)` (a uniform k-subset of its n proof blocks, independent across
strata) or `Bern(n, p, c)` (each proof block drawn with probability p), c the amount of each of its proof blocks.
`Pooled(k, strata)` is one uniform k-subset over the union of its strata (`rate:NUM/DEN`, `subset:K`). The miss
probability of a set T with t_s proof blocks in stratum s is prod_s C(n_s - t_s, k_s) / C(n_s, k_s) (prod_s (1 - p_s)^t_s),
and for `Pooled` C(N - t, k) / C(N, k), t = |T|: `Core/Draw.lean`'s `missProb` of these laws.

Error direction: every log is held as an interval (lo, hi) of -ln(miss); a bound that the guarantee charges
(`missable_ub`, the rate's theta) is computed from the end that makes it larger. `missable_lb` is a witness's amount and
only shows how tight the upper bound is.

Run: `python3 benchmarks/target_check/target_check.py [--census PATH]` prints the tables; `--selftest` the brute-force checks.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

EPS = 2.0 ** -52
INF = math.inf
# below this many factors -ln(miss) is summed term by term (log1p, relative error a few ulps); above, from lgamma, whose
# cancellation costs absolute error ~ ulp(lgamma(n)) and is charged at 2^10 ulps of each lgamma value
TERMS = 4096
LGAMMA_ULPS = 2.0 ** 10


@dataclass(frozen=True)
class Sub:
    n: int
    k: int
    c: float

    def g(self, t: int) -> tuple[float, float]:
        """-ln C(n-t, k)/C(n, k) as (lo, hi)."""
        n, k = self.n, self.k
        if t <= 0 or k <= 0:
            return 0.0, 0.0
        if t > n - k:
            return INF, INF
        m = min(k, t)
        if m <= TERMS:
            terms = ([-math.log1p(-t / (n - j)) for j in range(k)] if k <= t
                     else [-math.log1p(-k / (n - i)) for i in range(t)])
            s = math.fsum(terms)
            e = 8 * EPS * s
            return max(0.0, s - e), s + e
        lg = [math.lgamma(n + 1), math.lgamma(n - k + 1), math.lgamma(n - t + 1), math.lgamma(n - t - k + 1)]
        s = (lg[0] - lg[1]) - (lg[2] - lg[3])
        e = LGAMMA_ULPS * EPS * sum(abs(x) for x in lg) + 8 * EPS * abs(s)
        return max(0.0, s - e), s + e

    def inc(self, i: int) -> float:
        """-ln of the i-th factor, the increase of g from t = i to i + 1 (nondecreasing in i)."""
        return INF if i >= self.n - self.k else -math.log1p(-self.k / (self.n - i))

    @property
    def tmax(self) -> int:
        return self.n - self.k

    def t_at(self, mu: float) -> int:
        """The t maximizing c t - mu g(t) (g convex): the items whose increment is at most c / mu."""
        if self.k == 0:
            return self.n
        if self.c <= 0:
            return 0
        q = -math.expm1(-self.c / mu)
        if q <= 0:
            return 0
        x = self.n - self.k / q
        return max(0, min(self.tmax, math.floor(x) + 1)) if x > -1 else 0

    @property
    def log_single(self) -> float:
        """ln of one proof block's miss probability, ln(1 - k/n), rounded up (toward the larger miss)."""
        if self.k >= self.n:
            return -INF
        v = math.log1p(-self.k / self.n)
        return v * (1 - 8 * EPS)


@dataclass(frozen=True)
class Bern:
    n: int
    p: Fraction
    c: float

    def g(self, t: int) -> tuple[float, float]:
        if t <= 0 or self.p == 0:
            return 0.0, 0.0
        if self.p >= 1:
            return INF, INF
        a = -math.log1p(-float(self.p))
        return t * a * (1 - 8 * EPS), t * a * (1 + 8 * EPS)

    def inc(self, i: int) -> float:
        return INF if self.p >= 1 else -math.log1p(-float(self.p))

    @property
    def tmax(self) -> int:
        return 0 if self.p >= 1 else self.n

    def t_at(self, mu: float) -> int:
        if self.p >= 1:
            return 0
        if self.p == 0:
            return self.n
        return self.n if self.c > 0 and self.c >= mu * self.inc(0) else 0

    @property
    def log_single(self) -> float:
        return -INF if self.p >= 1 else math.log1p(-float(self.p)) * (1 - 8 * EPS)


@dataclass(frozen=True)
class Pooled:
    k: int
    strata: tuple  # ((n_s, c_s), ...)

    @property
    def n(self) -> int:
        return sum(n for n, _ in self.strata)


# ---------------------------------------------------------------- (b) the rate


def log_rate(design) -> float:
    """ln theta: the smallest theta with miss(T) <= theta^amount(T) for every T, rounded up.

    Stratified/Bernoulli: theta = max over strata with c_s > 0 of (1 - k_s/n_s)^(1/c_s). Proof: the factors of
    C(n-t, k)/C(n, k) = prod_{i<t} (1 - k/(n-i)) decrease in i, so a stratum's miss of t proof blocks is at most
    (1 - k/n)^t = ((1 - k/n)^(1/c))^(c t) <= theta^(c t); strata multiply and amounts add, so miss(T) <= theta^amount(T);
    a single proof block of the maximizing stratum attains it, so no smaller theta works. Pooled: the same with
    (1 - k/N)^(1/max c), the miss depending only on |T| and |T| >= amount/max c. A stratum with c_s = 0 constrains nothing."""
    if isinstance(design, Pooled):
        cmax = max((c for n, c in design.strata if n > 0), default=0)
        if cmax <= 0:
            return -INF
        return Sub(design.n, design.k, cmax).log_single / cmax
    return max((s.log_single / s.c for s in design if s.c > 0 and s.n > 0), default=-INF)


def k_at(log_theta: float, delta: float) -> float:
    """The amount K = ln(delta)/ln(theta): every set of amount >= K is missed with probability <= delta."""
    return INF if log_theta == 0 else math.log(delta) / log_theta


# ---------------------------------------------------------------- (a) the missable amount


def _inner(s, mu: float) -> float:
    """An upper bound on max_t (c t - mu g(t)) over 0 <= t <= tmax, from the lower end of g.

    t0 = s.t_at(mu) is moved to the float maximizer by its increments; a true maximizer differs from it only across
    increments within a few ulps of c/mu, and increments grow by a relative ~1/n per step (n < 2^40 here), so at most two
    such items lie on either side, each worth at most 8 eps (c + mu a); the slack charges four."""
    t0 = s.t_at(mu)
    while t0 < s.tmax and s.c - mu * s.inc(t0) > 0:
        t0 += 1
    while t0 > 0 and s.c - mu * s.inc(t0 - 1) < 0:
        t0 -= 1
    lo, _ = s.g(t0)
    a = s.inc(t0) if t0 < s.tmax else s.inc(max(t0 - 1, 0))
    slack = 4 * 8 * EPS * (s.c + (mu * a if a < INF else s.c))
    return s.c * t0 - mu * lo + slack


def missable_ub(design, delta: float) -> float:
    """A sound upper bound on missable(design, delta): Lagrangian duality, max sum c t s.t. sum g(t) < L is at most
    mu L + sum_s max_t (c_s t - mu g_s(t)) for every mu > 0 (L = -ln delta); minimized over mu by golden-section search,
    any mu giving a valid bound. Capped by the rate's K, also sound. Cost: O(strata) per mu, 80 values of mu."""
    if isinstance(design, Pooled):
        return float(missable_pooled(design, delta)[1])
    L = -math.log(delta)
    K = k_at(log_rate(design), delta)
    ratios = [s.c / s.inc(0) for s in design if s.c > 0 and 0 < s.inc(0) < INF]
    if not ratios:
        return min(K, sum(s.c * s.tmax for s in design))
    a, b = math.log(min(ratios)) - 40, math.log(max(ratios)) + 40

    def ub(lm: float) -> float:
        mu = math.exp(lm)
        return mu * L + sum(_inner(s, mu) for s in design if s.c > 0)

    phi = (math.sqrt(5) - 1) / 2
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = ub(x1), ub(x2)
    best = min(f1, f2)
    for _ in range(80):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = ub(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = ub(x2)
        best = min(best, f1, f2)
    return min(best, K)


def missable_lb(design, delta: float) -> float:
    """A witness: a set T (greedy by increment, prefix in each stratum) with sum of g's upper ends below -ln delta."""
    if isinstance(design, Pooled):
        return float(missable_pooled(design, delta)[0])
    L = -math.log(delta)
    live = [s for s in design if s.c > 0]
    ratios = [s.c / s.inc(0) for s in live if 0 < s.inc(0) < INF]
    free = sum(s.c * s.tmax for s in live if s.inc(0) == 0)
    if not ratios:
        return free
    lo, hi = math.log(min(ratios)) - 40, math.log(max(ratios)) + 40

    def ts(lm):
        return [s.t_at(math.exp(lm)) for s in live]

    def cost(t):
        return math.fsum(s.g(x)[1] for s, x in zip(live, t))

    for _ in range(100):  # larger mu, fewer items
        mid = (lo + hi) / 2
        if cost(ts(mid)) < L:
            hi = mid
        else:
            lo = mid
    t = ts(hi)
    if cost(t) >= L:
        t = [0] * len(live)
    for i, s in sorted(enumerate(live), key=lambda e: -e[1].c / max(e[1].inc(t[e[0]]), 1e-300)):
        while t[i] < s.tmax and cost(t[:i] + [t[i] + 1] + t[i + 1:]) < L:
            t[i] += 1
            break
    return sum(s.c * x for s, x in zip(live, t))


def missable_pooled(design: Pooled, delta: float) -> tuple[float, float]:
    """Exact up to g's interval, as (witness, bound): the miss depends only on |T|, so the largest set is the t* proof
    blocks of largest amount, t* the largest t with C(N-t, k)/C(N, k) > delta (binary search, O(log N) evaluations)."""
    L = -math.log(delta)
    N, k = design.n, design.k
    S = Sub(N, k, 1)

    def tstar(end: int) -> int:
        lo, hi = 0, N - k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if S.g(mid)[end] < L:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def top(t: int) -> float:
        out = 0.0
        for n, c in sorted(design.strata, key=lambda e: -e[1]):
            take = min(n, t)
            out += take * c
            t -= take
            if t == 0:
                break
        return out

    return top(tstar(1)), top(tstar(0))


# ---------------------------------------------------------------- (2) the picker


def pick(sizes, amounts, costs, log_theta: float, floor: int = 0):
    """The cheapest stratified design with rate <= theta: k_s = ceil(n_s (1 - theta^c_s)), at least `floor`.

    Optimal within the stratified family for any positive per-proof-block costs: the rate's constraint is one per
    stratum, (1 - k_s/n_s) <= theta^c_s, and each stratum's cost grows with k_s, so each k_s at its least is the optimum.
    Returns (ks, expected proofs, expected proved amount, expected cost)."""
    ks = []
    for n, c in zip(sizes, amounts):
        if c <= 0:
            k = min(n, floor)
        else:
            k = min(n, max(floor, math.ceil(n * -math.expm1(c * log_theta))))
            while k < n and Sub(n, k, c).log_single / c > log_theta:
                k += 1
        ks.append(k)
    return (ks, sum(ks), sum(k * c for k, c in zip(ks, amounts)), sum(k * p for k, p in zip(ks, costs)))


def pick_bernoulli(sizes, amounts, costs, log_theta: float, den: int = 2 ** 62):
    """The cheapest Bernoulli design with rate <= theta: p_s = 1 - theta^c_s rounded up to a multiple of 1/den (no
    per-stratum integer floor, so the expected proofs are sum_s n_s (1 - theta^c_s), first order W ln(1/theta)).
    Returns (ps, expected proofs, expected proved amount, expected cost)."""
    ps = []
    for c in amounts:
        if c <= 0:
            ps.append(Fraction(0))
            continue
        p = Fraction(math.ceil(-math.expm1(c * log_theta) * den) + 1, den)
        p = min(Fraction(1), p)
        while p < 1 and Bern(1, p, c).log_single / c > log_theta:
            p = min(Fraction(1), p + Fraction(1, den) * 1024)
        ps.append(p)
    return (ps, sum(float(p) * n for p, n in zip(ps, sizes)), sum(float(p) * n * c for p, n, c in zip(ps, sizes, amounts)),
            sum(float(p) * n * q for p, n, q in zip(ps, sizes, costs)))


# ---------------------------------------------------------------- the repository's rules, restated


def rate_count(num, den, n):  # verity/commitments/rate.py rate_count, one table
    return -(-(num * n) // den)


def stratum_k(k, size, population):  # spot/verifier/one_stage/draw.py stratum_k
    return max(1, (2 * k * size + population) // (2 * population))


def rate_need(num, den, size, n):  # draw.py rate_need
    c = rate_count(num, den, size)
    return 1 if c <= 1 else (2 * (c * n) - n + 2 * size - 1) // (2 * size)


def strat_k(num, den, sizes, n):  # draw.py strat_k
    return min(n, max([1, *(rate_need(num, den, s, n) for s in sizes)]))


# ---------------------------------------------------------------- self-test: brute force against the definitions


def _miss_exact(design, T) -> Fraction:
    """missProb of the set T of (stratum, index) pairs, exactly."""
    out = Fraction(1)
    if isinstance(design, Pooled):
        N, t = design.n, len(T)
        return Fraction(math.comb(N - t, design.k), math.comb(N, design.k))
    for si, s in enumerate(design):
        t = sum(1 for x in T if x[0] == si)
        if isinstance(s, Sub):
            out *= Fraction(math.comb(s.n - t, s.k), math.comb(s.n, s.k))
        else:
            out *= (1 - s.p) ** t
    return out


def _units(design):
    if isinstance(design, Pooled):
        return [((si, i), c) for si, (n, c) in enumerate(design.strata) for i in range(n)]
    return [((si, i), s.c) for si, s in enumerate(design) for i in range(s.n)]


def _brute(design, delta: Fraction):
    """`Scheme.missable` and the rate by enumerating every subset of proof blocks."""
    us = _units(design)
    best, rate = 0, 0.0
    for r in range(len(us) + 1):
        for T in itertools.combinations(us, r):
            m = _miss_exact(design, [u for u, _ in T])
            amt = sum(c for _, c in T)
            if delta < m:
                best = max(best, amt)
            if amt > 0 and m > 0:
                rate = max(rate, float(m) ** (1 / amt))
    return best, rate


def selftest() -> None:
    import random
    rng = random.Random(7)
    checked = 0
    for _ in range(300):
        kind = rng.choice(["sub", "sub", "bern", "pooled"])
        S = rng.randint(1, 3)
        if kind == "pooled":
            strata = tuple((rng.randint(1, 4), rng.randint(1, 5)) for _ in range(S))
            N = sum(n for n, _ in strata)
            if N > 11:
                continue
            design = Pooled(rng.randint(0, N), strata)
        else:
            design = []
            for _ in range(S):
                n = rng.randint(1, 5)
                c = rng.randint(0, 6)
                design.append(Sub(n, rng.randint(0, n), c) if kind == "sub"
                              else Bern(n, Fraction(rng.randint(0, 4), 4), c))
            if sum(s.n for s in design) > 11:
                continue
        delta = Fraction(rng.choice([1, 1, 2, 5, 10, 30]), 100)
        bm, br = _brute(design, delta)
        ub = missable_ub(design, float(delta))
        lb = missable_lb(design, float(delta))
        assert lb <= bm + 1e-9, (design, delta, lb, bm)
        assert ub >= bm - 1e-9, (design, delta, ub, bm)
        lr = log_rate(design)
        theta = math.exp(lr) if lr > -INF else 0.0
        assert theta >= br * (1 - 1e-12), (design, theta, br)
        assert theta <= br * (1 + 1e-9) or br == 0.0, (design, theta, br)
        checked += 1
    # the picker is the cheapest stratified design meeting the rate, against every k vector
    for _ in range(200):
        S = rng.randint(1, 3)
        sizes = [rng.randint(1, 6) for _ in range(S)]
        amounts = [rng.randint(0, 5) for _ in range(S)]
        costs = [rng.randint(1, 9) for _ in range(S)]
        lt = math.log(rng.uniform(0.3, 0.99))
        ks, _, _, cost = pick(sizes, amounts, costs, lt)
        best = min(sum(k * p for k, p in zip(kv, costs))
                   for kv in itertools.product(*[range(n + 1) for n in sizes])
                   if log_rate([Sub(n, k, c) for n, k, c in zip(sizes, kv, amounts)]) <= lt)
        assert cost == best, (sizes, amounts, costs, lt, ks, cost, best)
    # the large-n path (lgamma) agrees with the term-by-term sum where both apply
    for n, k, t in [(10 ** 9, 5000, 6000), (10 ** 10, 4097, 10 ** 6), (3 * 10 ** 9, 10 ** 5, 10 ** 5)]:
        lo, hi = Sub(n, k, 1).g(t)
        s = math.fsum(-math.log1p(-t / (n - j)) for j in range(k))
        assert lo <= s <= hi, (n, k, t, lo, s, hi)
    print(f"selftest: {checked} random designs against brute force over every subset; picker optimal on 200; lgamma path ok")


# ---------------------------------------------------------------- the census and the tables

CENSUS = Path(__file__).resolve().parents[1] / "spot_figures" / "cards" / "workload.glm-5.3.json"


def load(path: Path):
    d = json.loads(path.read_text())["value"]
    tok = d["tokens"]
    return [dict(block=s["block"], n=round(s["per_token"] * tok), macs=s["macs"], out_bits=s["out_bits"], ands=s["ands"])
            for s in d["strata"]], tok


def fmt(x: float) -> str:
    if x == INF:
        return "inf"
    if x == 0:
        return "0"
    return f"{x:.3e}" if abs(x) >= 1e5 or abs(x) < 1e-2 else f"{x:.4g}"


def today(strata, amount: str):
    sizes = [s["n"] for s in strata]
    N = sum(sizes)
    amts = [s[amount] for s in strata]
    K_app = rate_count(1, 3481648, N)
    pooled = Pooled(K_app, tuple(zip(sizes, amts)))
    K_srv = strat_k(1, 2097152, sizes, N)
    strat = [Sub(n, stratum_k(K_srv, n, N), c) for n, c in zip(sizes, amts)]
    return {"rate:1/3481648 (pooled subset)": pooled, "stratified-rate:1/2097152": strat}


def report(path: Path) -> None:
    strata, tok = load(path)
    N = sum(s["n"] for s in strata)
    print(f"census {path.name}: {len(strata)} strata, {tok} tokens, N = {N} proof blocks, "
          f"W = {fmt(sum(s['n'] * s['macs'] for s in strata))} MACs, "
          f"out bits = {fmt(sum(s['n'] * s['out_bits'] for s in strata))}")
    D40 = 2.0 ** -40
    print("\n(i) today's designs")
    print("| amount | design | proofs | ln theta | K@1% | missable@1% [lb, ub] | K@2^-40 | missable@2^-40 [lb, ub] |")
    print("|---|---|---|---|---|---|---|---|")
    base = {}
    for amount in ("macs", "out_bits"):
        for name, des in today(strata, amount).items():
            lr = log_rate(des)
            proofs = des.k if isinstance(des, Pooled) else sum(s.k for s in des)
            row = [amount, name, str(proofs), f"{lr:.4e}"]
            for d in (0.01, D40):
                row += [fmt(k_at(lr, d)), f"[{fmt(missable_lb(des, d))}, {fmt(missable_ub(des, d))}]"]
            print("| " + " | ".join(row) + " |")
            base[(amount, name)] = (des, lr)

    print("\n(ii) the cheapest design per target (rate theta, stated as K at 2^-40), beside today's stratified-rate design;"
          " proofs and proved amount are expected values; missable at 2^-40 is the sound upper bound")
    print("| amount | target | K@2^-40 | stratified: proofs | strata k_s>0 | proved amount | proved share | proved ANDs "
          "| missable@2^-40 | Bernoulli: proofs | proved amount |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for amount in ("macs", "out_bits"):
        des, lr = base[(amount, "stratified-rate:1/2097152")]
        sizes = [s["n"] for s in strata]
        amts = [s[amount] for s in strata]
        ands = [s["ands"] for s in strata]
        tot = sum(n * c for n, c in zip(sizes, amts))
        ks = [s.k for s in des]
        pr = sum(k * c for k, c in zip(ks, amts))
        mis_today = missable_ub(des, D40)
        print(f"| {amount} | today | {fmt(k_at(lr, D40))} | {sum(ks)} | {sum(1 for k in ks if k)} | {fmt(pr)} | "
              f"{fmt(pr / tot)} | {fmt(sum(k * a for k, a in zip(ks, ands)))} | {fmt(mis_today)} | - | - |")
        mis_today = missable_ub(des, D40)
        for label, f in (("10x tighter", 0.1), ("same K", 1.0), ("K = today's missable", mis_today / k_at(lr, D40)),
                         ("10x looser", 10.0)):
            K = k_at(lr, D40) * f
            lt = math.log(D40) / K
            ks2, proofs, proved, cost = pick(sizes, amts, ands, lt)
            mis = missable_ub([Sub(n, k, c) for n, k, c in zip(sizes, ks2, amts)], D40)
            _, bproofs, bproved, _ = pick_bernoulli(sizes, amts, ands, lt)
            print(f"| {amount} | {label} | {fmt(K)} | {proofs} | {sum(1 for k in ks2 if k)} | {fmt(proved)} | "
                  f"{fmt(proved / tot)} | {fmt(cost)} | {fmt(mis)} | {fmt(bproofs)} | {fmt(bproved)} |")

    print("\n(iii) re-partitioning the same work (MACs) at today's stratified-rate K@2^-40, cheapest design")
    des, lr = base[("macs", "stratified-rate:1/2097152")]
    K = k_at(lr, D40)
    lt = math.log(D40) / K
    work = [(s["n"], s["macs"]) for s in strata if s["macs"] > 0]
    W = sum(n * c for n, c in work)
    big = max(range(len(work)), key=lambda i: work[i][0] * work[i][1])
    print(f"K@2^-40 = {fmt(K)} MACs; break-even amount K/ln(1/delta) = {fmt(K / math.log(1 / D40))}; largest proof block "
          f"today {max(c for _, c in work)} MACs; largest stratum {fmt(work[big][0] * work[big][1])} MACs; "
          f"{len(work)} strata with work")
    print("| partition | proof blocks | max amount | stratified: proofs | proved MACs | proved share | strata fully proved "
          "| Bernoulli: proofs | proved MACs |")
    print("|---|---|---|---|---|---|---|---|---|")
    variants = [("split 64", [(n * 64, c / 64) for n, c in work]), ("split 8", [(n * 8, c / 8) for n, c in work]),
                ("split 2", [(n * 2, c / 2) for n, c in work]), ("today", work),
                ("merge 8", [(-(-n // 8), c * 8) for n, c in work]), ("merge 64", [(-(-n // 64), c * 64) for n, c in work]),
                ("merge 2^20", [(-(-n // 2 ** 20), c * 2 ** 20) for n, c in work]),
                ("largest stratum one proof block", [w if i != big else (1, w[0] * w[1]) for i, w in enumerate(work)])]
    for name, parts in variants:
        sizes = [n for n, _ in parts]
        amts = [c for _, c in parts]
        ks, proofs, proved, _ = pick(sizes, amts, [1] * len(parts), lt)
        _, bproofs, bproved, _ = pick_bernoulli(sizes, amts, [1] * len(parts), lt)
        full = sum(1 for k, n in zip(ks, sizes) if k == n)
        print(f"| {name} | {fmt(sum(sizes))} | {fmt(max(amts))} | {fmt(proofs)} | {fmt(proved)} | {fmt(proved / W)} | {full} "
              f"| {fmt(bproofs)} | {fmt(bproved)} |")
    print(f"first-order prediction: proofs = W ln(1/delta)/K = {fmt(W * math.log(1 / D40) / K)}, "
          f"proved amount = proofs x amount per proof block")

    print("\n(1c) the check's cost: wall seconds of log_rate and missable_ub against the number of strata (census strata repeated)")
    import time
    base_des, _ = base[("macs", "stratified-rate:1/2097152")]
    for rep in (1, 10, 100):
        d = list(base_des) * rep
        t0 = time.perf_counter(); log_rate(d); t1 = time.perf_counter(); missable_ub(d, D40); t2 = time.perf_counter()
        print(f"strata {len(d)}: log_rate {t1 - t0:.2e} s, missable_ub {t2 - t1:.2e} s")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        report(Path(sys.argv[sys.argv.index("--census") + 1]) if "--census" in sys.argv else CENSUS)
