"""Checks for the width-aware floor for exact results: Theorem W and Proposition L (worker A7, CMP-8).

Note: `note:20261015T1704Z-draft-width-aware-floor`. Builds on A3's Theorem B
(`note:20261015T0436Z-draft-merges-and-reuse`), A1a's Theorem 1 and Lemma 3
(`note:20261015T0416Z-draft-steiner-rigidity-laminar`) and the referee's witnesses
(`note:20261015T0740Z-draft-rigidity-referee`, "Round 2: projection").

An s8 input is x = sum_a w_a X^(a) over its bits, w_a = 2^a (a < 7), w_7 = -128. A word is a function of the inputs
into Z (read mod 2^32); its bit-pair block P_ab is the matrix of the coefficients of X_i^(a) Y_j^(b) in its
multilinear form, computed by inclusion-exclusion at the corner points e_ia + e_jb. r_k(M) is the number of nonzero
Smith invariants of M over Z/2^(k+1).

Parts:
  1  Proposition L. sum_k wrap8(x_k + y_(k+1)) wrap8(x_(k+1) + y_k) (k odd) has every bit-pair block of every prefix
     target right over Z/2^32, at all 64 pairs; with the row and column terms it is right mod 2^8 on random inputs
     and wrong mod 2^9 at the stated input.
  2  r_k: subadditive, <= 1 on an outer product, r_k(outer (x) N) <= r_k(N), and r_k(2^k M0) = rank_F2(M0).
  3  Lemma W1. For random full binary hierarchies with random x/y typings, the x- and y-constraint matrices have
     full row rank over F2 (a unimodular minor from the pivots t_U in D(U'')), the integer kernel basis solved through
     those pivots reduces to an F2 basis, and its labels give rank Lambda(S) <= 1 over Q at every node.
  4  Lemma W2. For plain operands (integer affine forms in the bits, wrapped or not), P_ab(uv) = alpha delta^T +
     gamma beta^T mod 2^(k+1) for k <= 7 (every k unwrapped); wrapping breaks it at some k >= 8 and an inexact E4M3
     rounding cast breaks it mod 2. Pair-rank grids of the witnesses' products.
  5  Lemma W3. An exact E2M1 operand uses at most 8 bits (exhaustive), so an FP4 product's pair ranks sum to <= 36
     over the 36 pairs a + b <= 7: <= 2 a slot on average, the constant c = 1/2; per pair alone it is 4 a slot.

Run: python widthfloor_check.py [part ...] | --mutations
"""

import itertools
import random
import sys
from fractions import Fraction

W = [1, 2, 4, 8, 16, 32, 64, -128]
MOD32 = 1 << 32
PAIRS = [(a, b) for a in range(8) for b in range(8)]
Q = [(a, b) for a, b in PAIRS if a + b <= 7]
E2M1 = sorted({s * v for v in (0, Fraction(1, 2), 1, Fraction(3, 2), 2, 3, 4, 6) for s in (1, -1)})


def wrap8(z):
    return ((z + 128) % 256) - 128


def corner(nx, ny, xa=None, yb=None):
    xs, ys = [0] * nx, [0] * ny
    if xa:
        xs[xa[0]] = W[xa[1]]
    if yb:
        ys[yb[0]] = W[yb[1]]
    return xs, ys


def block(f, nx, ny, a, b):
    f0 = f(*corner(nx, ny))
    fx = [f(*corner(nx, ny, xa=(i, a))) for i in range(nx)]
    fy = [f(*corner(nx, ny, yb=(j, b))) for j in range(ny)]
    return [[f(*corner(nx, ny, (i, a), (j, b))) - fx[i] - fy[j] + f0 for j in range(ny)] for i in range(nx)]


def val2(z):
    return (z & -z).bit_length() - 1


def rk(M, k):
    """Number of nonzero Smith invariants of M over Z/2^(k+1)."""
    mod = 1 << (k + 1)
    M = [[v % mod for v in row] for row in M]
    count = 0
    while M and M[0]:
        piv = min(((val2(v), r, c) for r, row in enumerate(M) for c, v in enumerate(row) if v), default=None)
        if piv is None:
            break
        v, r, c = piv
        inv = pow(M[r][c] >> v, -1, mod)
        M[r] = [x * inv % mod for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                q = M[i][c] >> v
                M[i] = [(x - q * y) % mod for x, y in zip(M[i], M[r])]
        for j in range(len(M[0])):
            if j != c and M[r][j]:
                q = M[r][j] >> v
                for i in range(len(M)):
                    M[i][j] = (M[i][j] - q * M[i][c]) % mod
        M = [row[:c] + row[c + 1:] for i, row in enumerate(M) if i != r]
        count += 1
    return count


def rank_mod2(rows):
    rows = [int("".join("1" if v % 2 else "0" for v in r), 2) if r else 0 for r in rows]
    rank = 0
    basis = []
    for r in rows:
        for bv in basis:
            r = min(r, r ^ bv)
        if r:
            basis.append(r)
            rank += 1
    return rank


def rank_q(rows):
    M = [[Fraction(v) for v in r] for r in rows]
    rank, col = 0, 0
    ncol = len(M[0]) if M else 0
    while rank < len(M) and col < ncol:
        p = next((i for i in range(rank, len(M)) if M[i][col] != 0), None)
        if p is None:
            col += 1
            continue
        M[rank], M[p] = M[p], M[rank]
        for i in range(len(M)):
            if i != rank and M[i][col] != 0:
                f = M[i][col] / M[rank][col]
                M[i] = [x - f * y for x, y in zip(M[i], M[rank])]
        rank += 1
        col += 1
    return rank


def part1(rng, broken=None):
    op = (lambda z: z) if broken == "nowrap" else wrap8
    for n in (4, 6, 8):
        def acc(s, xs, ys, corr):
            tot = 0
            for k in range(0, 2 * s, 2):
                u, v = op(xs[k] + ys[k + 1]), op(xs[k + 1] + ys[k])
                assert -128 <= u <= 127 and -128 <= v <= 127, "an operand left s8"
                tot += u * v - (xs[k] * xs[k + 1] + ys[k] * ys[k + 1] if corr else 0)
            return tot
        for s in range(1, n // 2 + 1):
            f = lambda xs, ys, s=s: acc(s, xs, ys, False)
            for a, b in PAIRS:
                P = block(f, n, n, a, b)
                want = [[W[a] * W[b] if i == j and i < 2 * s else 0 for j in range(n)] for i in range(n)]
                assert all((P[i][j] - want[i][j]) % MOD32 == 0 for i in range(n) for j in range(n)), (n, s, a, b)
            g = lambda xs, ys, s=s: acc(s, xs, ys, True)
            for _ in range(300):
                xs = [rng.randint(-128, 127) for _ in range(n)]
                ys = [rng.randint(-128, 127) for _ in range(n)]
                assert (g(xs, ys) - sum(x * y for x, y in zip(xs[:2 * s], ys[:2 * s]))) % 256 == 0
        xs, ys = [0] * n, [0] * n
        xs[0], ys[1], xs[1] = 64, 64, 2 if broken == "even" else 1
        err = g(xs, ys) - sum(x * y for x, y in zip(xs, ys))
        assert err % 512 != 0, f"error {err} is 0 mod 2^9"
    print(f"part 1: wrapped Winograd (n/2 products a target entry, n = 4, 6, 8, every prefix): all 64 bit-pair blocks "
          f"right over Z/2^32; with row and column terms right mod 2^8 on 300 random inputs a prefix; error {err} at "
          f"x_k = y_(k+1) = 64, x_(k+1) = 1: wrong mod 2^9: ok")


def part2(rng, broken=None):
    def r(M, k):
        return rank_mod2(M) if broken == "mod2" else rk(M, k)
    for _ in range(300):
        k = rng.randint(0, 9)
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        A = [[rng.randint(-40, 40) * 2 ** rng.randint(0, 3) for _ in range(n)] for _ in range(m)]
        B = [[rng.randint(-40, 40) * 2 ** rng.randint(0, 3) for _ in range(n)] for _ in range(m)]
        S = [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(A, B)]
        assert r(S, k) <= r(A, k) + r(B, k)
        al = [rng.randint(-9, 9) for _ in range(m)]
        be = [rng.randint(-9, 9) for _ in range(n)]
        assert r([[x * y for y in be] for x in al], k) <= 1
        al2, be2 = al[:2], be[:2]
        K = [[al2[i1] * be2[j1] * A[i2][j2] for j1 in range(len(be2)) for j2 in range(n)]
             for i1 in range(len(al2)) for i2 in range(m)]
        assert r(K, k) <= r(A, k)
        M0 = [[rng.randint(0, 1) for _ in range(n)] for _ in range(m)]
        assert r([[(1 << k) * v for v in row] for row in M0], k) == rank_mod2(M0), "r_k(2^k M0) != rank_F2 M0"
    print("part 2: r_k over Z/2^(k+1) (k = 0..9, 300 random cases): subadditive, <= 1 on outer products, "
          "r_k(outer (x) N) <= r_k(N), r_k(2^k M0) = rank_F2 M0: ok")


def rand_hierarchy(rng, m):
    nodes = [("leaf", t) for t in range(m)]
    while len(nodes) > 1:
        i, j = sorted(rng.sample(range(len(nodes)), 2))
        b = nodes.pop(j)
        a = nodes.pop(i)
        nodes.append(("x" if rng.random() < 0.5 else "y", a, b))
    return nodes[0]


def part3(rng, trials=300, broken=None):
    worst = 0
    for _ in range(trials):
        m = rng.randint(2, 9)
        root = rand_hierarchy(rng, m)
        cons = {"x": [], "y": []}
        allnodes = []

        def walk(node, cons=cons, allnodes=allnodes):
            if node[0] == "leaf":
                d = e = frozenset([node[1]])
                info = (frozenset([node[1]]), d, e)
            else:
                s1, d1, e1 = walk(node[1])
                s2, d2, e2 = walk(node[2])
                if node[0] == "x":
                    cons["x"].append((d1, d2))
                    info = (s1 | s2, d1, e1 | e2)
                else:
                    cons["y"].append((e1, e2))
                    info = (s1 | s2, d1 | d2, e1)
            allnodes.append(info[0])
            return info
        walk(root)
        labels = {}
        for side in ("x", "y"):
            rows = [[(t in p) - (t in q) for t in range(m)] for p, q in cons[side]]
            assert rank_mod2(rows) == len(rows), "a constraint matrix lost rank mod 2"
            # walk() appends a node's constraint after its children's, so reversed is top-down.
            order = list(reversed(rows))
            pivots = [min(q) for _, q in reversed(cons[side])]
            assert len(set(pivots)) == len(pivots)
            free = [t for t in range(m) if t not in pivots]
            basis = []
            for f in free:
                s = [0] * m
                s[f] = 1
                if broken == "naive":
                    basis.append(s)
                    continue
                for row, pv in zip(order, pivots):
                    s[pv] = 0
                    s[pv] = -sum(c * x for c, x in zip(row, s)) * row[pv]
                assert all(sum(c * x for c, x in zip(row, s)) == 0 for row in rows)
                basis.append(s)
            assert rank_mod2(basis) == m - len(rows)
            labels[side] = [[vec[t] for vec in basis] for t in range(m)]
        for S in allnodes:
            Lam = [[sum(labels["x"][t][p] * labels["y"][t][q] for t in S) for q in range(len(labels["y"][0]))]
                   for p in range(len(labels["x"][0]))] if labels["x"][0] and labels["y"][0] else [[0]]
            rq = rank_q(Lam)
            worst = max(worst, rq)
            assert rq <= 1, f"Lambda(S) has rank {rq} over Q"
    print(f"part 3: {trials} random hierarchies (2..9 leaves, random typings): constraint matrices full rank mod 2, "
          f"integer kernel bases through the pivots reduce to F2 bases, max rank Lambda(S) over Q = {worst}: ok")


def affine_bits(rng, nx, ny, nbits=3, lo=-3, hi=3):
    terms = {}
    for _ in range(nbits):
        side = rng.choice("xy")
        terms[(side, rng.randrange(nx if side == "x" else ny), rng.randrange(8))] = rng.choice(
            [c for c in range(lo, hi + 1) if c])
    return rng.randint(-5, 5), terms


def eval_affine(form, xs, ys):
    c0, terms = form
    tot = c0
    for (side, i, a), c in terms.items():
        z = xs[i] if side == "x" else ys[i]
        tot += c * (((z & 0xFF) >> a) & 1)
    return tot


def additive(f, nx, ny, a, b):
    f0 = f(*corner(nx, ny))
    al = [f(*corner(nx, ny, xa=(i, a))) - f0 for i in range(nx)]
    be = [f(*corner(nx, ny, yb=(j, b))) - f0 for j in range(ny)]
    return al, be


def e4m3_round(z):
    vals = sorted({s * m * 2 ** e for s in (1, -1) for e in range(9) for m in range(8, 16)} | set(range(-16, 17)))
    return min(vals, key=lambda v: (abs(v - z), v % 2))


def part4(rng, trials=150, broken=None):
    nx = ny = 3
    kmax = 8 if broken == "k8" else 7
    viol_hi = 0
    for _ in range(trials):
        fu, fv = affine_bits(rng, nx, ny, 6, -60, 60), affine_bits(rng, nx, ny, 6, -60, 60)
        for wrapped in (False, True):
            cast = wrap8 if wrapped else (lambda z: z)
            u = lambda xs, ys, cast=cast, fu=fu: cast(eval_affine(fu, xs, ys))
            v = lambda xs, ys, cast=cast, fv=fv: cast(eval_affine(fv, xs, ys))
            uv = lambda xs, ys, u=u, v=v: u(xs, ys) * v(xs, ys)
            for a, b in PAIRS:
                k = a + b
                P = block(uv, nx, ny, a, b)
                al, be = additive(u, nx, ny, a, b)
                ga, de = additive(v, nx, ny, a, b)
                mod = 1 << (k + 1)
                diff = [P[i][j] - al[i] * de[j] - ga[i] * be[j] for i in range(nx) for j in range(ny)]
                ok = all(d % mod == 0 for d in diff) if wrapped else not any(diff)
                if not wrapped or k <= kmax:
                    assert ok, f"Lemma W2 fails at {(a, b)} wrapped={wrapped}"
                elif not ok:
                    viol_hi += 1
    u = lambda xs, ys: wrap8(xs[0] + ys[1] + 100)
    v = lambda xs, ys: wrap8(xs[1] + ys[0] + 1)
    uv = lambda xs, ys: u(xs, ys) * v(xs, ys)
    for a, b in PAIRS:
        P = block(uv, 2, 2, a, b)
        al, be = additive(u, 2, 2, a, b)
        ga, de = additive(v, 2, 2, a, b)
        mod = 1 << (a + b + 1)
        if any((P[i][j] - al[i] * de[j] - ga[i] * be[j]) % mod for i in range(2) for j in range(2)):
            assert a + b > kmax, f"Lemma W2 fails at {(a, b)} wrapped"
            viol_hi += 1
    assert viol_hi > 0, "wrapped operands should break Lemma W2 at some k >= 8"
    u = lambda xs, ys: e4m3_round(xs[0] + ys[1] + 15)
    v = lambda xs, ys: 1 + xs[1] + ys[0]
    uv = lambda xs, ys: u(xs, ys) * v(xs, ys)
    P = block(uv, 2, 2, 0, 0)
    al, be = additive(u, 2, 2, 0, 0)
    ga, de = additive(v, 2, 2, 0, 0)
    J = [[P[i][j] - al[i] * de[j] - ga[i] * be[j] for j in range(2)] for i in range(2)]
    assert J[0][1] % 2 == 1, "an inexact rounding cast should leave an odd interaction"
    print(f"part 4: Lemma W2 on {trials} random pairs of plain operands (6 bits each, coefficients up to 60): exact at "
          f"all 64 pairs unwrapped, mod 2^(k+1) for k <= 7 wrapped; wrapped fails at {viol_hi} (pair, k >= 8) cases "
          f"(wrap8(x_1 + y_2 + 100) wrap8(x_2 + y_1 + 1) at k = 8 among them); "
          f"E4M3-rounded x_1 + y_2 + 15 leaves interaction J = {J[0][1]} (odd) at pair (0, 0): ok")
    grids = {}
    halves = lambda z: z >> 1
    prods = {
        "wrapped": (lambda xs, ys: wrap8(xs[0] + ys[1]), lambda xs, ys: wrap8(xs[1] + ys[0]), 1),
        "halves": (lambda xs, ys: halves(xs[0]) + halves(ys[1]), lambda xs, ys: halves(xs[1]) + halves(ys[0]), 1),
        "fp4 bit sum (2,3)": (lambda xs, ys: ((xs[0] & 0xFF) >> 2 & 1) + ((ys[1] & 0xFF) >> 3 & 1),
                              lambda xs, ys: ((xs[1] & 0xFF) >> 2 & 1) + ((ys[0] & 0xFF) >> 3 & 1), Fraction(1, 2)),
    }
    for name, (u, v, price) in prods.items():
        for xs in itertools.product(*(range(-128, 128, 37),) * 2):
            for ys in itertools.product(*(range(-128, 128, 41),) * 2):
                assert -128 <= u(list(xs), list(ys)) <= 127 and -128 <= v(list(xs), list(ys)) <= 127
        uv = lambda xs, ys, u=u, v=v: u(xs, ys) * v(xs, ys)
        g = {(a, b): rk(block(uv, 2, 2, a, b), a + b) for a, b in PAIRS}
        grids[name] = g
        sq = sum(g[p] for p in Q)
        assert sq <= 2 * price * len(Q), f"{name}: average pair rank above 2 a slot"
        if broken == "perpair":
            assert max(g.values()) <= 2 * price, f"{name}: a pair rank above 2 a slot"
    assert all(v == 2 for v in grids["wrapped"].values())
    assert sum(v == 2 for v in grids["halves"].values()) == 49
    assert all(grids["halves"][a, b] == 0 for a, b in PAIRS if a == 0 or b == 0)
    assert [p for p, v in grids["fp4 bit sum (2,3)"].items() if v] == [(2, 3)]
    print("part 4: pair ranks of one product: wrapped Winograd 2 at all 64 pairs (sum over Q 72, 2 a slot); 7-bit "
          "halves 2 at the 49 pairs a, b >= 1, 0 elsewhere; FP4 bit sum 2 at its pair (2, 3) only: ok")


def max_bits(vals):
    vs = set(vals)
    diffs = sorted({p - q for p in vals for q in vals if p != q})
    best = 0

    def dfs(sums, cnt, start):
        nonlocal best
        best = max(best, cnt)
        for idx in range(start, len(diffs)):
            new = sums | {s + diffs[idx] for s in sums}
            if new <= vs:
                dfs(new, cnt + 1, idx)
    for l0 in vals:
        dfs({l0}, 0, 0)
    return best


def part5(broken=None):
    vals = E2M1 + ([Fraction(5)] if broken == "nine" else [])
    mb = max_bits(vals)
    if broken != "nine":
        assert mb == 8, mb

    def tri(s, t, pairs):
        return sum(1 for a, b in pairs if a < s and b < t)
    for pairs, name in ((Q, "a + b <= 7"), (PAIRS, "all")):
        best = max(tri(s1, t2, pairs) + tri(s2, t1, pairs)
                   for s1 in range(9) for t1 in range(mb + 1 - s1) for s2 in range(9) for t2 in range(mb + 1 - s2)
                   if t1 <= 8 and t2 <= 8)
        assert best <= len(pairs), f"an FP4 product's pair ranks sum to {best} over {len(pairs)} pairs"
    e4 = sorted({s * (Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7))
                 for s in (1, -1) for e in range(16) for m in range(8) if not (e == 15 and m == 7)})
    e4i = [v for v in range(-128, 128) if Fraction(v) in set(e4)]
    assert len(e4i) == 80
    print(f"part 5: an exact E2M1 operand uses at most {mb} bits (exhaustive over offsets and coefficient multisets); "
          f"an FP4 product's pair ranks sum to <= 36 over the 36 pairs a + b <= 7 and <= 64 over all 64, so <= 1 a "
          f"product, 2 a slot, as s8 and E4M3 (E4M3 holds {len(e4i)} s8 values): c = 1/2 for every format: ok")


MUTATIONS = (
    ("part 1 drops the wrap", lambda rng: part1(rng, broken="nowrap")),
    ("part 1 sets x_(k+1) = 2 in the mod-2^9 witness", lambda rng: part1(rng, broken="even")),
    ("part 2 reads r_k mod 2", lambda rng: part2(rng, broken="mod2")),
    ("part 3 lifts the labels naively (0/1, no pivot solve)", lambda rng: part3(rng, trials=200, broken="naive")),
    ("part 4 extends Lemma W2 to k = 8 with wraps", lambda rng: part4(rng, trials=60, broken="k8")),
    ("part 4 prices FP4 per pair", lambda rng: part4(rng, trials=5, broken="perpair")),
    ("part 5 lets E2M1 hold 5 (9 bits)", lambda rng: part5(broken="nine")),
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
        (part1, part2, part3, part4, lambda r: part5())[part - 1](rng)


if __name__ == "__main__":
    main(sys.argv[1:])
