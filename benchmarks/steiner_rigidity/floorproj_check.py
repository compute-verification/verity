"""Checks for Claim P (the factor-2 projection) and Claim R (the salt rate) (worker A7, CMP-8).

Note: `note:20261015T1624Z-draft-floor-projection`. Builds on A3's Theorems B and C
(`note:20261015T0436Z-draft-merges-and-reuse`) and on `cmix_check.py` (Proposition 5's tree).

A word is a polynomial in the inputs x (X coordinates) and y (Y coordinates), held as {monomial: coefficient} with a
monomial a sorted tuple of variable indices, over Z (p = 0) or GF(p). Its projection is its x-y bilinear block: the
nX x nY matrix of the coefficients of the monomials x_i y_j. No 1/2 enters it, so it is defined in characteristic 2.

Parts:
  1  The projection of a product: for operands u = a.x + b.y + u0 + (quadratic part q_u) and v likewise,
     proj(u v) = a d^T + c b^T + u0 proj(v) + v0 proj(u), over Z, GF(2), GF(3), GF(10007). Its rank is at most
     1 + [both operands mixed] when no operand with a bilinear part meets a constant (a copy edge); products of two
     pure-x or two pure-y forms project to 0. Over GF(2) read as functions (x^2 = x) the block is unchanged.
  2  Random accumulate-and-merge computations whose words carry xx and yy junk (mixed products, merges of junk-carrying
     words, copy edges): the projection commutes with every step, and a tensor step's increment of k products has
     rank <= (k - |K_m|) + min(r_a, r_d) + min(r_b, r_c) <= k + min(delta(V_A), delta(V_B)) (K_m: the positions where
     both operands are mixed; r_a, r_b the dimensions of the A-side x- and y-parts over K_m, r_c, r_d the B side's;
     delta(V) = dim pi_X V + dim pi_Y V - dim V).
  3  Proposition 5's family (merge-free, junk on every intermediate node): the projected terminals are the targets,
     and sum rank = L |G_T| exactly (2 x 2 at L = 2: 7 products, ranks 0 + 2 + 2 + 1 + 1 + 1 + 1 = 8), against
     Claim P's count #unmixed + 2 #mixed = 13 and #products + T_mix = 9.
  4  The plain-metric Steiner number of the 2 x 2 block at L = 2 over GF(2) is 8 (Dreyfus-Wagner on the Cayley graph
     of 4 x 4 matrices with rank-1 generators): Conjecture A in characteristic 2 at this size.
  5  Claim R's numbers. s8 inputs uniform on [-128, 127]: a form with two or more nonzero coefficients misses every
     256-point window {o + s e} (s8, or u8 with any offset) with probability >= 1/4, with equality at +-z1 +- z2;
     a tile's forms with private variables are all in range with probability (3/4)^e, forms sharing one variable
     with probability ~1/(m+1); E4M3 and BF16 exactness. The data-dependent dispatch counterexample to the
     per-product form of R: one reused mixed form per operand makes a tile cheap with probability 9/16.

Run: python floorproj_check.py [part ...] | --mutations
"""

import itertools
import random
import sys
from fractions import Fraction

import numpy as np

from cmix_check import rank, winograd_tree


def red(c, p):
    return c % p if p else c


def padd(f, g, p, s=1):
    h = dict(f)
    for m, c in g.items():
        h[m] = red(h.get(m, 0) + s * c, p)
        if not h[m]:
            del h[m]
    return h


def pmul(f, g, p):
    h = {}
    for m1, c1 in f.items():
        for m2, c2 in g.items():
            m = tuple(sorted(m1 + m2))
            h[m] = red(h.get(m, 0) + c1 * c2, p)
    return {m: c for m, c in h.items() if c}


def pscale(f, s, p):
    return {m: red(s * c, p) for m, c in f.items() if red(s * c, p)}


def affine(a, b, c0=0):
    """The form a.x + b.y + c0 as a polynomial (a over X, b over Y)."""
    nX = len(a)
    f = {(i,): int(v) for i, v in enumerate(a) if v}
    f.update({(nX + j,): int(v) for j, v in enumerate(b) if v})
    if c0:
        f[()] = c0
    return f


def proj(f, nX, nY, p):
    """The x-y bilinear block of f: coefficient of x_i y_j at [i, j]."""
    B = np.zeros((nX, nY), dtype=object)
    for m, c in f.items():
        if len(m) == 2 and m[0] < nX <= m[1]:
            B[m[0], m[1] - nX] = red(B[m[0], m[1] - nX] + c, p)
    return B


def mrank(B, p):
    q = p if p else 1000003
    return rank(np.array(B, dtype=np.int64) % q, q)


def parts(f, nX, n):
    """Linear x-part, linear y-part and constant of f."""
    a, b = np.zeros(nX, dtype=object), np.zeros(n - nX, dtype=object)
    for m, c in f.items():
        if len(m) == 1:
            if m[0] < nX:
                a[m[0]] = c
            else:
                b[m[0] - nX] = c
    return a, b, f.get((), 0)


def mixed(f, nX, n):
    a, b, _ = parts(f, nX, n)
    return bool(np.any(a != 0)) and bool(np.any(b != 0))


def coef(rng, p, lo=-3, hi=3):
    return rng.randrange(p) if p else rng.randint(lo, hi)


def rand_form(rng, nX, nY, p, kind, const=False):
    """A random affine form: kind 'x' (pure x), 'y' (pure y) or 'm' (mixed)."""
    while True:
        a = [coef(rng, p) if kind in "xm" and rng.random() < 0.6 else 0 for _ in range(nX)]
        b = [coef(rng, p) if kind in "ym" and rng.random() < 0.6 else 0 for _ in range(nY)]
        f = affine(a, b, coef(rng, p) if const else 0)
        if (kind != "x" or any(red(v, p) for v in a)) and (kind != "y" or any(red(v, p) for v in b)) and \
                (kind != "m" or (any(red(v, p) for v in a) and any(red(v, p) for v in b))):
            return {m: red(c, p) for m, c in f.items() if red(c, p)}


def rand_quad(rng, n, p, terms=4):
    """Random quadratic junk-and-bilinear terms."""
    f = {}
    for _ in range(terms):
        i, j = sorted((rng.randrange(n), rng.randrange(n)))
        f = padd(f, {(i, j): coef(rng, p) or 1}, p)
    return f


def proj_product(u, v, nX, nY, p, broken=False):
    """proj(u v) by the formula a d^T + c b^T + u0 proj(v) + v0 proj(u)."""
    n = nX + nY
    a, b, u0 = parts(u, nX, n)
    c, d, v0 = parts(v, nX, n)
    B = np.outer(a, d) + (0 if broken == "cb" else np.outer(c, b))
    if broken != "const":
        B = B + u0 * proj(v, nX, nY, p) + v0 * proj(u, nX, nY, p)
    return np.vectorize(lambda t: red(t, p), otypes=[object])(B)


def reduce_gf2(f):
    """Read f over GF(2) as a function: x^2 = x."""
    g = {}
    for m, c in f.items():
        g = padd(g, {tuple(sorted(set(m))): c}, 2)
    return g


# ---------------------------------------------------------------------------------------------------------------
# Part 1: the projection of a product.


def part1(rng, trials=400, broken=None):
    nX, nY = 4, 4
    n = nX + nY
    seen = {}
    for p in (0, 2, 3, 10007):
        for _ in range(trials):
            ku, kv = rng.choice("xym"), rng.choice("xym")
            cu, cv = rng.random() < 0.3, rng.random() < 0.3
            u, v = rand_form(rng, nX, nY, p, ku, cu), rand_form(rng, nX, nY, p, kv, cv)
            qu, qv = rng.random() < 0.3, rng.random() < 0.3
            if qu:
                u = padd(u, rand_quad(rng, n, p), p)
            if qv:
                v = padd(v, rand_quad(rng, n, p), p)
            B = proj(pmul(u, v, p), nX, nY, p)
            F = proj_product(u, v, nX, nY, p, broken)
            assert (B == F).all(), f"p={p}: proj(uv) is not a d^T + c b^T + u0 proj(v) + v0 proj(u)"
            copy = (parts(u, nX, n)[2] and proj(v, nX, nY, p).any()) or \
                (parts(v, nX, n)[2] and proj(u, nX, nY, p).any())
            r = mrank(B, p)
            if not copy:
                bound = 1 + (mixed(u, nX, n) and mixed(v, nX, n))
                assert r <= bound, f"p={p}: rank {r} > {bound} without a copy edge"
                if ku == kv and ku in "xy" and not qu and not qv:
                    assert r == 0, "a pure-x (or pure-y) product must project to 0"
                seen[p, bound, r] = seen.get((p, bound, r), 0) + 1
            if p == 2 and not qu and not qv:
                assert (proj(reduce_gf2(pmul(u, v, 2)), nX, nY, 2) == B).all(), "GF(2) as functions changed the block"
    assert any(b == 2 and r == 2 for (_, b, r) in seen), "no rank-2 mixed product drawn"
    assert any(p == 2 and b == 2 and r == 2 for (p, b, r) in seen), "no rank-2 mixed product over GF(2)"
    print(f"part 1: proj(uv) formula and rank <= 1 + [both mixed] over Z, GF(2), GF(3), GF(10007); "
          f"{sum(seen.values())} products without copy edges, {sum(v for (_, b, r), v in seen.items() if r == 2)} "
          f"of rank 2: ok")


# ---------------------------------------------------------------------------------------------------------------
# Part 2: random accumulate-and-merge computations with junk.


def span_dim(vecs, p):
    if not vecs:
        return 0
    return mrank(np.array(vecs, dtype=object), p)


def delta(forms, nX, n, p):
    """dim pi_X V + dim pi_Y V - dim V for V the span of the forms' linear parts."""
    A = [list(parts(f, nX, n)[0]) for f in forms]
    B = [list(parts(f, nX, n)[1]) for f in forms]
    return span_dim(A, p) + span_dim(B, p) - span_dim([a + b for a, b in zip(A, B)], p)


def step_bounds(us, vs, nX, n, p):
    """The two bounds on rank(sum_k proj(u_k v_k)) for affine operands (no constants)."""
    Km = [k for k in range(len(us)) if mixed(us[k], nX, n) and mixed(vs[k], nX, n)]
    ra = span_dim([list(parts(us[k], nX, n)[0]) for k in Km], p)
    rb = span_dim([list(parts(us[k], nX, n)[1]) for k in Km], p)
    rc = span_dim([list(parts(vs[k], nX, n)[0]) for k in Km], p)
    rd = span_dim([list(parts(vs[k], nX, n)[1]) for k in Km], p)
    fine = len(us) - len(Km) + min(ra, rd) + min(rb, rc)
    dl = min(delta([us[k] for k in Km], nX, n, p), delta([vs[k] for k in Km], nX, n, p)) if Km else 0
    return fine, len(us) + dl, len(Km)


def part2(rng, comps=60, broken=None):
    nX, nY = 6, 6
    n = nX + nY
    steps = heads = junk = tight = 0
    for ci in range(comps):
        p = (0, 2, 3, 10007)[ci % 4]
        words = [{}]
        pw = [np.zeros((nX, nY), dtype=object)]
        for _ in range(12):
            op = rng.random()
            if op < 0.6:
                w = rng.randrange(len(words))
                k = rng.randint(1, 4)
                share = rng.random() < 0.3
                pool = [rand_form(rng, nX, nY, p, "m") for _ in range(2)]
                us, vs = [], []
                for _ in range(k):
                    if share:
                        us.append(rng.choice(pool))
                        vs.append(rng.choice(pool[::-1] + [rand_form(rng, nX, nY, p, "m")]))
                    else:
                        us.append(rand_form(rng, nX, nY, p, rng.choice("xmm")))
                        vs.append(rand_form(rng, nX, nY, p, rng.choice("ymm")))
                inc = {}
                for u, v in zip(us, vs):
                    inc = padd(inc, pmul(u, v, p), p)
                D = sum((proj_product(u, v, nX, nY, p, broken) for u, v in zip(us, vs)),
                        np.zeros((nX, nY), dtype=object))
                D = np.vectorize(lambda t: red(t, p), otypes=[object])(D)
                r = mrank(D, p)
                fine, coarse, km = step_bounds(us, vs, nX, n, p)
                if broken == "noslack":
                    fine = fine - (len(us) - km)
                assert r <= sum(mrank(proj_product(u, v, nX, nY, p), p) for u, v in zip(us, vs)) <= k + km
                assert r <= fine <= coarse, f"step rank {r} exceeds the bound {fine} (coarse {coarse})"
                tight += r == fine and km > 0
                words.append(padd(words[w], inc, p))
                pw.append(np.vectorize(lambda t: red(t, p), otypes=[object])(pw[w] + D))
                steps += 1
            elif op < 0.85:
                idx = rng.sample(range(len(words)), min(len(words), rng.randint(2, 3)))
                cs = [coef(rng, p) or 1 for _ in idx]
                m = {}
                B = np.zeros((nX, nY), dtype=object)
                for i, c in zip(idx, cs):
                    m = padd(m, pscale(words[i], c, p), p)
                    B = B + c * pw[i]
                words.append(m)
                pw.append(np.vectorize(lambda t: red(t, p), otypes=[object])(B))
                heads += 1
            else:
                w = rng.randrange(len(words))
                v = rand_form(rng, nX, nY, p, rng.choice("xym"), const=True)
                inc = pmul(words[w], v, p)
                words.append(padd(words[w], inc, p))
                pw.append(np.vectorize(lambda t: red(t, p), otypes=[object])(
                    pw[w] + proj_product(words[w], v, nX, nY, p, broken)))
                heads += 1
        for f, B in zip(words, pw):
            assert (proj(f, nX, nY, p) == B).all(), "projection does not commute with the computation"
            junk += any(len(m) == 2 and (m[1] < nX or m[0] >= nX) for m in f)
    assert tight > 0, "the fine step bound was never attained on a step with doubly-mixed products"
    print(f"part 2: {comps} computations over Z, GF(2), GF(3), GF(10007): projection commutes with {steps} tensor "
          f"steps and {heads} merges or copies; {junk} words carry xx/yy junk; step bound attained {tight} times: ok")


# ---------------------------------------------------------------------------------------------------------------
# Part 3: Proposition 5's family.


def part3(broken=None):
    for P, Q, L in ((2, 2, 2), (3, 2, 2), (2, 4, 2), (2, 2, 4), (3, 4, 4)):
        G, edges = winograd_tree(P, Q, L)
        nX = G.nX
        val = {"0": np.zeros((G.nX, G.nY), dtype=np.int64)}
        ranks, weighted, tmix, prods = [], 0, 0, 0
        pending = dict(edges)
        while pending:
            for (u, v), ps in list(pending.items()):
                if u not in val:
                    continue
                D = np.zeros((G.nX, G.nY), dtype=np.int64)
                for a, g, h in ps:
                    Dp = a * (np.outer(g[:nX], h[nX:]) + (0 if broken == "cb" else np.outer(h[:nX], g[nX:])))
                    gm, hm = g[:nX].any() and g[nX:].any(), h[:nX].any() and h[nX:].any()
                    ranks.append(mrank(Dp, 0))
                    weighted += 2 if (gm or hm) else 1
                    tmix += gm and hm
                    prods += 1
                    D = D + Dp
                val[v] = val[u] + D
                del pending[u, v]
        for r in range(P):
            for c in range(Q):
                T = np.zeros((G.nX, G.nY), dtype=np.int64)
                for k in range(L):
                    T[G.x(r, k), G.y(k, c) - nX] = 1
                assert (val["t", r, c] == T).all(), f"projected terminal ({r}, {c}) is not its target"
        floor = L * P * Q
        assert sum(ranks) == floor, f"sum rank {sum(ranks)} != L|G_T| = {floor}"
        assert prods + tmix >= floor and weighted >= floor
        if (P, Q, L) == (2, 2, 2):
            assert (prods, sorted(ranks), weighted, prods + tmix) == (7, [0, 1, 1, 1, 1, 2, 2], 13, 9)
        print(f"part 3: {P} x {Q} at L = {L}: {prods} products, sum rank {sum(ranks)} = L|G_T| = {floor}, "
              f"#products + T_mix = {prods + tmix}, Claim P's count {weighted}: ok")


# ---------------------------------------------------------------------------------------------------------------
# Part 4: plain-metric Steiner number over GF(2), 2 x 2 at L = 2.


def part4(broken=None):
    nX = nY = 4
    nb = nX * nY
    gens = sorted({int(sum(1 << (i * nY + j) for i in range(nX) for j in range(nY) if (a >> i) & 1 and (b >> j) & 1))
                   for a in range(1, 1 << nX) for b in range(1, 1 << nY)})
    assert len(gens) == 225
    if broken == "rank2":
        gens = sorted({a ^ b for a in gens for b in gens} - {0})
    # X coordinates (r, k) -> r*2 + k, Y coordinates (k, c) -> k*2 + c
    def target(r, c):
        return sum(1 << ((r * 2 + k) * nY + (k * 2 + c)) for k in range(2))
    terms = [0] + [target(r, c) for r in range(2) for c in range(2)]
    N = 1 << nb
    idx = np.arange(N, dtype=np.int64)
    INF = 99

    def relax(S):
        while True:
            best = S.copy()
            for g in gens:
                np.minimum(best, S[idx ^ g] + 1, out=best)
            if (best == S).all():
                return S
            S = best

    k = len(terms)
    St = {}
    for i, t in enumerate(terms):
        S = np.full(N, INF, dtype=np.int64)
        S[t] = 0
        St[1 << i] = relax(S)
    for D in range(1, 1 << k):
        if D in St:
            continue
        S = np.full(N, INF, dtype=np.int64)
        sub = (D - 1) & D
        while sub:
            if sub < D ^ sub:
                np.minimum(S, St[sub] + St[D ^ sub], out=S)
            sub = (sub - 1) & D
        St[D] = relax(S)
    full = int(St[(1 << k) - 1][0])
    assert full == 8, f"plain Steiner number over GF(2) is {full}, not 8"
    print("part 4: plain-metric Steiner number of the 2 x 2 block at L = 2 over GF(2) is 8 = L|G_T|: ok")


# ---------------------------------------------------------------------------------------------------------------
# Part 5: Claim R's numbers.


S8 = range(-128, 128)


def dist(coeffs):
    """Exact distribution of sum c_i z_i, z_i independent uniform on s8."""
    d = {0: Fraction(1)}
    for c in coeffs:
        e = {}
        for v, w in d.items():
            for z in S8:
                e[v + c * z] = e.get(v + c * z, 0) + w / 256
        d = e
    return d


def best_window(d, scales, size=256):
    """Largest mass of {o + s e : 0 <= e < size} over offsets o and the given scales s (size 256: s8 or u8)."""
    best = Fraction(0)
    vals = sorted(d)
    for s in scales:
        for r in {v % s for v in vals}:
            lat = {}
            for v in vals:
                if v % s == r:
                    lat[(v - r) // s] = d[v]
            keys = sorted(lat)
            arr = np.zeros(keys[-1] - keys[0] + size + 1, dtype=object)
            for kk in keys:
                arr[kk - keys[0]] = lat[kk]
            c = np.concatenate([[Fraction(0)], np.cumsum(arr)])
            for i in range(len(arr) - size + 1):
                acc = c[i + size] - c[i]
                if acc > best:
                    best = acc
    return best


def e4m3_values():
    vals = set()
    for s in (1, -1):
        for e in range(16):
            for m in range(8):
                if e == 15 and m == 7:
                    continue
                vals.add(s * (Fraction(m, 8) * Fraction(2) ** (1 - 7) if e == 0 else
                              (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)))
    return sorted(vals)


def part5(broken=None):
    out = {}
    for co in ((1, 1), (1, -1), (1, 2), (2, 3), (1, 4), (3, -1)):
        d = dist(co)
        g = max(abs(c) for c in co)
        m = best_window(d, range(1, 2 * g + 1), 512 if broken == "s9" else 256)
        out[co] = 1 - m
        assert 1 - m >= Fraction(1, 4), f"{co}: a window holds {m} > 3/4"
    assert out[1, 1] == out[1, -1] == Fraction(1, 4), "the extremal pair should miss exactly 1/4"
    d3 = dist((1, 1, 1))
    m3 = best_window(d3, (1, 2, 3))
    assert 1 - m3 > Fraction(1, 4)
    print("part 5: least miss probability of a 256-point window (s8, or u8 at any offset): "
          + ", ".join(f"{a}z1{'+' if b > 0 else '-'}{abs(b)}z2 {float(v):.4f}" for (a, b), v in out.items())
          + f", z1+z2+z3 {float(1 - m3):.4f}: ok")
    pin = Fraction(3, 4)
    star = {}
    for mm in (1, 2, 4, 8, 16, 32):
        tot = Fraction(0)
        for x in S8:
            ok = sum(1 for y in S8 if -128 <= x - y <= 127)
            tot += Fraction(ok, 256) ** mm / 256
        star[mm] = tot
        assert tot <= pin or mm == 1
        if mm >= 4:
            assert tot <= Fraction(3, mm + 1), f"star of {mm}: {float(tot)}"
    print("part 5: all in range, private variables (3/4)^e: " +
          ", ".join(f"e={e} {float(pin ** e):.2e}" for e in (1, 4, 16, 32)) +
          "; one shared x (x1 - y_j, j <= m): " + ", ".join(f"m={m} {float(v):.3f}" for m, v in star.items()) + ": ok")
    e4 = e4m3_values()
    e4set = set(e4)
    ints = [v for v in S8 if Fraction(v) in e4set]
    sums = sum(1 for a in e4 for b in e4 if a + b in e4set) / len(e4) ** 2
    assert len(ints) == 80 and sums < 0.5
    print(f"part 5: E4M3 holds {len(ints)}/256 s8 values; x+y of two uniform E4M3 values is E4M3 with probability "
          f"{sums:.3f}; BF16 (8-bit significand) holds every integer in [-256, 256], so every +-z1 +-z2: ok")
    # Data-dependent dispatch against the per-product form: every A entry is u = x1 - y1, every B entry v = x2 - y2.
    p_cheap = pin * pin
    price = 1 + (1 - p_cheap)
    assert price < 2, "per-product R would need 2 int8 slots a rank-2 product"
    us = [affine([1, 0], [-1, 0])] * 32
    vs = [affine([0, 1], [0, -1])] * 32
    D = sum((proj_product(u, v, 2, 2, 0) for u, v in zip(us, vs)), np.zeros((2, 2), dtype=object))
    r = mrank(D, 0)
    assert r == 2 and step_bounds(us, vs, 2, 4, 0)[1] == 33
    print(f"part 5: dispatch on one reused mixed form a side: tile cheap with probability {p_cheap}, expected "
          f"{float(price):.4f} int8 slots a rank-2 product even when a miss costs a whole extra tile, so the "
          f"per-product form of R is false; the step's projected rank is {r} against its 32 slots: ok")


MUTATIONS = (
    ("part 1 drops c b^T", lambda rng: part1(rng, trials=200, broken="cb")),
    ("part 1 drops the constant terms", lambda rng: part1(rng, trials=200, broken="const")),
    ("part 2 drops the k - |K_m| slack", lambda rng: part2(rng, comps=20, broken="noslack")),
    ("part 3 drops c b^T", lambda rng: part3(broken="cb")),
    ("part 4 counts a rank-2 step as one", lambda rng: part4(broken="rank2")),
    ("part 5 gives the operand a 512-point (s9) window", lambda rng: part5(broken="s9")),
)


def mutations(rng):
    for name, run in MUTATIONS:
        try:
            run(rng)
        except AssertionError:
            print(f"mutation caught: {name}")
            continue
        raise SystemExit(f"mutation NOT caught: {name}")


def main(argv):
    rng = random.Random(20261015)
    if "--mutations" in argv:
        mutations(rng)
        return
    want = [int(a) for a in argv if a.isdigit()] or [1, 2, 3, 4, 5]
    for part in want:
        if part == 1:
            part1(rng)
        elif part == 2:
            part2(rng)
        elif part == 3:
            part3()
        elif part == 4:
            part4()
        elif part == 5:
            part5()


if __name__ == "__main__":
    main(sys.argv[1:])
