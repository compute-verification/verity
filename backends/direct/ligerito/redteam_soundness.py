"""red-team-ligerito: independent soundness recomputation for Ligerito parameter set B and the variants the live lanes
are actually building.  Small pure-python arithmetic (laptop-safe); ``params.py`` is used ONLY for its byte model.

    python -m backends.direct.ligerito.redteam_soundness

Per committed level i (tall_i = RS message length k_i, n_i = block length m_i, k'_i folded column variables, |S_i|):

* proximity (unique decoding; BCIKS20/23 Thm 4.1 lines with eps = n up to e = floor((d-1)/2), Diamond-Gruen CiC 1(4)
  Thm 2 + AER24 Thm 3 for the tensor fold): a query passes w.p. <= 1 - (floor((m-k)/2) + 1)/m  ("tight");
  params.py books (1 + k/m)/2 ("params", >= tight); the paper's (4)/(17) print ((m - k - 1)/(2m))^|S|, the rejection
  fraction with the "1 -" missing -- read as 1 - (m-k-1)/(2m) = (m + k + 1)/(2m) ("paper"; also the Rust verifier's form).
* tensor false witness k'_i m_i / |F| (DG24: theta eps / q with eps = m_i).
* partial sumcheck 2 k'_i / |F| (degree 2 per column variable).
* claim batching (|S_{i-1}| + 1) / |F| for i >= 2 (powers-of-alpha form; fresh coefficients would give 1/|F|).
"""
from __future__ import annotations

import math
from dataclasses import replace

from backends.direct.ligerito import params as PM

P = 2013265921
LOG2F6 = 6 * math.log2(P)
LOG2F8 = 8 * math.log2(P)
TARGET = -128.0
T_PAD = 256                       # ligerito-zk zk.py T_PAD (per-column RS padding, every committed level)


# ---------------------------------------------------------------------------------------------------------------------
# 1. the challenge field
# ---------------------------------------------------------------------------------------------------------------------

def _pmod(a, f, p):
    a = [x % p for x in a]
    df = len(f) - 1
    inv = pow(f[-1], p - 2, p)
    while len(a) - 1 >= df and any(a):
        if a[-1] == 0:
            a.pop()
            continue
        c = a[-1] * inv % p
        s = len(a) - 1 - df
        for j in range(df + 1):
            a[s + j] = (a[s + j] - c * f[j]) % p
        a.pop()
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a or [0]


def _pmul(a, b, f, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % p
    return _pmod(out, f, p)


def _ppow(a, e, f, p):
    r, b = [1], a
    while e:
        if e & 1:
            r = _pmul(r, b, f, p)
        b = _pmul(b, b, f, p)
        e >>= 1
    return r


def _pgcd(a, b, p):
    a, b = a[:], b[:]
    while any(b):
        a, b = b, _pmod(a, b, p)
    return a


def rabin_irreducible(f: list[int], p: int = P) -> bool:
    """Rabin's test: f (coefficients low -> high, monic) of degree n is irreducible over F_p iff x^{p^n} = x mod f and
    gcd(x^{p^{n/q}} - x, f) = 1 for every prime q | n."""
    n = len(f) - 1
    x = [0, 1]

    def frob(k):
        y = x
        for _ in range(k):
            y = _ppow(y, p, f, p)
        return y

    if _pmod(frob(n), f, p) != _pmod(x, f, p):
        return False
    for q in {q for q in range(2, n + 1) if n % q == 0 and all(q % r for r in range(2, q))}:
        y = frob(n // q)
        d = [(y[i] if i < len(y) else 0) - (x[i] if i < len(x) else 0) for i in range(max(len(y), len(x)))]
        g = _pgcd(f, _pmod(d, f, p) if any(v % p for v in d) else [0], p)
        if len(g) > 1:
            return False
    return True


def field_checks() -> dict:
    out = {"p": P, "p-1 factors": "2^27 * 3 * 5", "log2|F_p^6|": LOG2F6, "log2|F_p^8|": LOG2F8}
    out["x^6-31 irreducible (Rabin)"] = rabin_irreducible([-31 % P, 0, 0, 0, 0, 0, 1])
    out["31 non-square, non-cube (binomial criterion)"] = (pow(31, (P - 1) // 2, P) != 1 and pow(31, (P - 1) // 3, P) != 1)
    out["x^8-11 irreducible (Rabin)"] = rabin_irreducible([-11 % P, 0, 0, 0, 0, 0, 0, 0, 1])
    # x^7 - a: gcd(7, p - 1) = 1, so a -> a^{1/7} exists for every a: x^7 - a has a linear factor
    inv7 = pow(7, -1, P - 1)
    out["x^7-a never irreducible (7 does not divide p-1)"] = all(pow(pow(a, inv7, P), 7, P) == a for a in (2, 3, 11, 31))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. per-level terms
# ---------------------------------------------------------------------------------------------------------------------

def per_query_log2(k: int, m: int, form: str) -> float:
    if form == "tight":
        return math.log2(1 - (((m - k) // 2) + 1) / m)
    if form == "params":
        return math.log2((1 + k / m) / 2)
    if form == "paper":
        return math.log2((m + k + 1) / (2 * m))
    if form == "rate":            # a code of EXACT rate k/m used with the query count of a lower-rate code: same formula
        return math.log2((1 + k / m) / 2)
    raise ValueError(form)


def log2sum(xs):
    m = max(xs)
    return m + math.log2(sum(2.0 ** (x - m) for x in xs))


def pcs_terms(levels, form="params", log2F=LOG2F6, fs_q_log2: float | None = None) -> dict:
    """``levels``: list of (k_i message length, m_i block length, k'_i, |S_i|).  ``fs_q_log2``: the PCS partial-sumcheck
    rounds (= the tensor-fold challenges) made Fiat-Shamir with Q = 2^fs_q_log2 oracle queries: round-by-round error
    per fold coordinate is m_i/|F| (line gap, eps = m_i) + 2/|F| (sumcheck), times Q."""
    t = {}
    for i, (k, m, kp, S) in enumerate(levels):
        t[f"L{i + 1}.prox"] = S * per_query_log2(k, m, form)
        t[f"L{i + 1}.tensor k'm/|F|"] = math.log2(kp * m) - log2F
        t[f"L{i + 1}.pcs-sumcheck 2k'/|F|"] = math.log2(2 * kp) - log2F
        if i:
            t[f"L{i + 1}.batch (|S|+1)/|F|"] = math.log2(levels[i - 1][3] + 1) - log2F
        if fs_q_log2 is not None:
            t[f"L{i + 1}.FS Q.(m+2)/|F|"] = fs_q_log2 + math.log2(m + 2) - log2F
    return t


def relation_terms(n_zc_vars: int, n_rows: int, n_shift: int, n_claims: int, n_constraints: int, K_coins: int,
                   log2F=LOG2F6, fs_q_log2: float | None = None) -> dict:
    """ligerito-sumcheck's structure (report sections 3-4): zero-check over (k, c) with one bivariate opening round
    (degree <= 2 per variable after Gruen, x eq -> 6/|F| by SZ incl. the eq zeros) then univariate (3/|F| each); rows and
    shift degree-2 sumchecks; claim batching; constraint batching booked as params.py does (conservative); Libra G-lie
    1/|F| per mask (3); committed-coin hiding K . 2 . 2^64 / 2^256 (PROTOCOL.md 8c)."""
    t = {
        "ZC.eq(tau) SZ nvars/|F|": math.log2(n_zc_vars) - log2F,
        "ZC.constraint batching (Q+Lin)/|F|": math.log2(n_constraints) - log2F,
        "ZC.sumcheck (6 + 3 (nvars-2))/|F|": math.log2(6 + 3 * (n_zc_vars - 2)) - log2F,
        "rows sumcheck 2 n_i/|F| + gamma 2/|F|": math.log2(2 * n_rows + 2) - log2F,
        "shift sumcheck 2 n_c/|F| + g3 links/|F|": math.log2(2 * n_shift + 3) - log2F,
        "claim batching into PCS": math.log2(n_claims) - log2F,
        "Libra G-lie 3/|F|": math.log2(3) - log2F,
        "coin-commit hiding K 2^-191": math.log2(K_coins) - 191.0,
    }
    if fs_q_log2 is not None:
        # round-by-round: max per-round error (6/|F| opening round), times Q
        t["ZC/rows/shift FS Q.6/|F| (RBR)"] = fs_q_log2 + math.log2(6) - log2F
    return t


# ---------------------------------------------------------------------------------------------------------------------
# 3. set B and the variants
# ---------------------------------------------------------------------------------------------------------------------

SET_B = {  # the design note's recommended B (tall1, wide_log2, rate_log2, queries)
    "fp8": dict(rel="fp8-ada", tall1=3 << 22, wide=(6, 4, 4, 3, 3), rates=(1, 2, 2, 2, 2), S=(239, 173, 174, 174, 176)),
    "bf16": dict(rel="bf16-hopper", tall1=3 << 22, wide=(7, 4, 4, 4), rates=(1, 2, 2, 2), S=(239, 173, 173, 175)),
    "fp4": dict(rel="fp4-nvf4", tall1=3 << 20, wide=(6, 4, 4, 3), rates=(1, 2, 2, 2), S=(239, 173, 173, 175)),
}
REL_SIZES = {  # (zero-check vars n_k + n_c incl. ternary, row vars, column vars, claims, constraints) -- sumcheck report
    "fp8": (13 + 17, 12, 17, 4, 3532 + 589 + 13),
    "bf16": (12 + 18, 12, 18, 4, 3164 + 339 + 13),
    "fp4": (11 + 16, 11, 16, 4, 1389 + 306 + 13),
}


def nextpow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def shape_levels(tall1, wide, rates, S, t_pad=0, pow2_tall=False):
    """(k_i, m_i, k'_i, |S_i|): m_i = nextpow2(tall_i) << rate_i as params.n(); k_i = tall_i (+ t_pad under ZK).
    ``pow2_tall``: the same level shapes on a power-of-two layout (cells padded to 2^e, tall_i = nextpow2)."""
    out, t = [], tall1
    for i, kp in enumerate(wide):
        if i:
            t //= 1 << kp
        tt = nextpow2(t) if pow2_tall else t
        m = nextpow2(t) << rates[i]
        out.append((tt + t_pad, m, kp, S[i]))
    return out


def lp_for(name, S=None) -> PM.LigeritoParams:
    b = SET_B[name]
    return PM.LigeritoParams(tall1=b["tall1"], wide_log2=b["wide"], rate_log2=b["rates"], queries=tuple(S or b["S"]),
                             relation=PM.RELATIONS[b["rel"]])


def proof_bytes(name, S=None, t_pad=0) -> float:
    """params.py byte model + the ZK padding combination ybar_i (t_pad ext per committed level) it does not count."""
    p = lp_for(name, S)
    return p.proof_bytes()["total_bytes"] + (t_pad * 24 * p.L if t_pad else 0)


def fix_queries(levels, form="params", target=TARGET, other_log2=None, cost=None):
    """Smallest per-level bumps (greedy, cheapest bit per byte) of |S_i| so the union meets ``target``."""
    S = [lv[3] for lv in levels]
    per = [per_query_log2(k, m, form) for (k, m, _, _) in levels]
    cost = cost or [1.0] * len(levels)

    def union(S_):
        xs = [s * a for s, a in zip(S_, per)] + ([other_log2] if other_log2 is not None else [])
        return log2sum(xs)

    while union(S) > target:
        # the bump that lowers the union most per byte
        best = None
        for i in range(len(S)):
            T = S.copy()
            T[i] += 1
            gain = (union(S) - union(T)) / cost[i]
            if best is None or gain > best[0]:
                best = (gain, i)
        S[best[1]] += 1
    return S, union(S)


def per_query_cost(name) -> list[float]:
    p = lp_for(name)
    base = p.proof_bytes()["total_bytes"]
    out = []
    for i in range(p.L):
        S = list(p.queries)
        S[i] += 1
        out.append(replace(p, queries=tuple(S)).proof_bytes()["total_bytes"] - base)
    return out


def report():
    print("== 1. challenge field")
    for k, v in field_checks().items():
        print(f"  {k}: {v}")

    print("\n== 2. set B, independent recomputation (log2)")
    rows = {}
    for name, b in SET_B.items():
        lv = shape_levels(b["tall1"], b["wide"], b["rates"], b["S"])
        rel = relation_terms(*REL_SIZES[name], K_coins=64)
        res = {}
        for form in ("tight", "params", "paper"):
            t = pcs_terms(lv, form)
            res[form] = log2sum(list(t.values()) + list(rel.values()))
        pb = lp_for(name).soundness()["total_log2"]
        t = pcs_terms(lv, "params")
        field_max = max(v for k, v in list(t.items()) + list(rel.items()) if "prox" not in k)
        prox = {k: round(v, 2) for k, v in t.items() if "prox" in k}
        rows[name] = res
        print(f"  {name}: cells {b['tall1'] << b['wide'][0]:,} = 3*2^{int(math.log2((b['tall1'] << b['wide'][0]) // 3))}; levels (k, m, k', S) = {lv}")
        print(f"     proximity {prox}")
        print(f"     max non-proximity term 2^{field_max:.1f}; relation union 2^{log2sum(list(rel.values())):.1f}")
        print(f"     UNION: tight 2^{res['tight']:.3f} | params-form 2^{res['params']:.3f} | paper/Rust-form 2^{res['paper']:.3f}"
              f" | params.py soundness() 2^{pb:.3f}; bytes {proof_bytes(name) / 1024:.1f} KiB")

    print("\n== 3. variants")
    for name, b in SET_B.items():
        rel = relation_terms(*REL_SIZES[name], K_coins=64)
        relu = log2sum(list(rel.values()))
        cost = per_query_cost(name)
        base_bytes = proof_bytes(name)
        print(f"  -- {name}  (set-B bytes {base_bytes / 1024:.1f} KiB; +1 query costs {[round(c) for c in cost]} B per level)")
        # (b) ZK per-column RS padding t_pad = 256 at every level, set B's |S|
        lvz = shape_levels(b["tall1"], b["wide"], b["rates"], b["S"], t_pad=T_PAD)
        tz = pcs_terms(lvz, "params")
        uz = log2sum(list(tz.values()) + [relu])
        worst = max((v, k) for k, v in tz.items() if "prox" in k)
        Sz, uzf = fix_queries(lvz, "params", other_log2=relu, cost=cost)
        bz = proof_bytes(name, Sz, t_pad=T_PAD)
        print(f"     (b) + ZK t_pad={T_PAD}: union 2^{uz:.2f} (worst {worst[1]} 2^{worst[0]:.2f}); corrected |S| = {Sz} -> 2^{uzf:.3f};"
              f" bytes {bz / 1024:.1f} KiB (incl. {T_PAD * 24 * len(Sz) / 1024:.0f} KiB ybar; +{(bz - base_bytes) / 1024:.1f} KiB vs B)")
        # (c) set B's |S| on the power-of-two layout ligerito-sumcheck / proto use (cols padded to 2^e)
        lvp = shape_levels(b["tall1"], b["wide"], b["rates"], b["S"], pow2_tall=True)
        tp = pcs_terms(lvp, "params")
        up = log2sum(list(tp.values()) + [relu])
        Sp, upf = fix_queries(lvp, "params", other_log2=relu, cost=cost)
        print(f"     (c) set-B |S| on the pow2 layout (rates exactly {['1/%d' % (m // k) for k, m, _, _ in lvp]}): union 2^{up:.2f};"
              f" corrected |S| = {Sp} -> 2^{upf:.3f}")
        # (d) margin: the paper/Rust-verifier reading of (17)
        lv = shape_levels(b["tall1"], b["wide"], b["rates"], b["S"])
        Sd, udf = fix_queries(lv, "paper", other_log2=relu, cost=cost)
        print(f"     (d) paper/Rust-verifier base (m+k+1)/(2m): corrected |S| = {Sd} -> 2^{udf:.3f}; bytes {proof_bytes(name, Sd) / 1024:.1f} KiB")
        # (e) B + ZK, sized under the paper/Rust form too (what a batch verifier with --target-bits 128 recomputes)
        Se, uef = fix_queries(lvz, "paper", other_log2=relu, cost=cost)
        print(f"     (e) + ZK t_pad, paper/Rust form: |S| = {Se} -> 2^{uef:.3f}; bytes {proof_bytes(name, Se, t_pad=T_PAD) / 1024:.1f} KiB")
        # (f) job split into N proofs: per-proof must be <= 2^-(128 + log2 N)
        for N in (2, 4):
            Sn, unf = fix_queries(lv, "params", target=TARGET - math.log2(N), other_log2=relu, cost=cost)
            print(f"     (f) job split into N = {N} proofs: per-proof |S| = {Sn} (2^{unf:.2f} each); bytes/proof {proof_bytes(name, Sn) / 1024:.1f} KiB")

    print("\n== 4. Fiat-Shamir readings (fp8 set B)")
    b = SET_B["fp8"]
    lv = shape_levels(b["tall1"], b["wide"], b["rates"], b["S"])
    n_rounds = 66
    for Q in (60, 64):
        design = Q + math.log2(n_rounds * 3) - LOG2F6
        rel_rbr = Q + math.log2(6) - LOG2F6
        pcs_rbr = max(v for k, v in pcs_terms(lv, "params", fs_q_log2=Q).items() if "FS" in k)
        pcs_rbr8 = max(v for k, v in pcs_terms(lv, "params", log2F=LOG2F8, fs_q_log2=Q).items() if "FS" in k)
        print(f"  Q = 2^{Q}: design's Q.66.3/|F| = 2^{design:.1f};  relation sumchecks only, RBR Q.6/|F| = 2^{rel_rbr:.1f};"
              f"  + PCS partial-sumcheck rounds (fold challenges), RBR Q.m_1/|F| = 2^{pcs_rbr:.1f}  [F_p^8: 2^{pcs_rbr8:.1f}]")


if __name__ == "__main__":
    report()
