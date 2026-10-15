"""Exact flattening rank for a branch into three targets (the unrescued 3-sets of `label_frac.py`).

Theorem 6.4 charges a branch B (coefficient vector c on S) by the drop rank Φ − rank(Φ − Λ(c) ⊗ B) <= rank Λ(c) ·
rank B, subadditivity. For a rank-1 branch B = (u·x)(v·y) the drop is exact linear algebra. Write I for the rows u
touches (u_i ≠ 0), J for the columns v touches, Γ_ij = u_i · v_j, R_i (C_j) for the span of row i's a-labels (column
j's b-labels), and g_s(ij), h_s(ij) for the coefficient of a_ij in a_s over R_i (of b_ij in b_s over C_j). Then the
drop is 3 iff every a_s lies in every R_i (i in I), every b_s in every C_j (j in J), and
Σ_ij Γ_ij h(ij) g(ij)ᵀ = diag(1/c) (a generalized Schur complement; Φ = M ⊗ I_q with M of full rank |cells|), and
otherwise at most 2. A rank-r branch has drop 3r only if the (1,1) entries of its r×r blocks Γ_ij solve the same
system, so excluding drop 3 for rank 1 bounds every rank-r branch's drop by 3r − 1.
The solutions (Γ, 1/c) of all typings' systems form a subspace; the configuration is rescued for (I, J) iff some
coordinate of 1/c vanishes on it (no coefficients, over any extension field, make every typing's drop 3).

    python exact_branch.py LOG           (for each unrescued (S, H) in LOG: the (I, J) no typing rescues, if any)
    python exact_branch.py --sample SEED TRIALS (the same on 3-sets unrescued on fresh hierarchies of 3×3 and 3×4)
    python exact_branch.py --rank LOG N TRIALS (Theorem 6.7's bound ⌊w r⌋ per typing against direct ranks)
    python exact_branch.py --climb LOG Q STEPS RESTARTS (hill-climbs branches toward the largest drop every typing allows)
    python exact_branch.py --construct LOG Q (fact (c) at rank ρ <= Q against direct ranks, on built branches)
    python exact_branch.py --check LOG N (compares with rank Φ − rank(Φ − Λ ⊗ uvᵀ) for every rank-1 uvᵀ, q = 2, F2,
                                          on the first N configurations of LOG over F2, with c = 1)
"""

import itertools
import sys

import numpy as np

import fq
from scipy.optimize import linprog

from check_labels import label, nullspace, random_hierarchy, rk, rref
from label_frac import configs
from label_proj import _nest, all_typings, inter_dim, reachable


def coords(cells, lab, S, axis, p):
    """For each line (row if axis 0) holding all of S's labels in its span, the coordinates of each s in S over
    that line's labels: {line: {cell index t: vector over S}}."""
    out = {}
    for key in sorted({c[axis] for c in cells}):
        ts = [t for t in range(len(cells)) if cells[t][axis] == key]
        A = np.array([lab[t] for t in ts]).T % p
        cols = []
        for s in S:
            red, piv = rref(np.column_stack([A, lab[s]]), p)
            if len(ts) in piv:
                break
            g = np.zeros(len(ts), dtype=np.int64)
            for r, c in enumerate(piv):
                g[c] = red[r, len(ts)]
            cols.append(g)
        else:
            out[key] = {t: np.array([g[k] for g in cols]) for k, t in enumerate(ts)}
    return out


def bad_system(cells, G, H, S, I, J, p):
    """The homogeneous equations Σ Γ_t h(t) g(t)ᵀ − diag(e) = 0 in (Γ_t for the cells t in I×J, e), or None if a
    containment fails (drop <= 2); G, H are `coords` of the a- and b-labels."""
    if any(i not in G for i in I) or any(j not in H for j in J):
        return None
    unknowns = [t for t, c in enumerate(cells) if c[0] in I and c[1] in J]
    rows = []
    for x, y in itertools.product(range(len(S)), repeat=2):
        rows.append([H[cells[t][1]][t][x] * G[cells[t][0]][t][y] % p for t in unknowns]
                    + [(p - 1) * (x == y == s) for s in range(len(S))])
    return unknowns, rows


def hypothesis(cells, G, H, S, p):
    """Theorem 6.7's hypothesis for one typing, through its sufficient conditions: the cells t of R × C (R, C the
    lines holding S's labels) are at most one; or K(c)_{tt'} = Σ_s g_s(t) c_s h_s(t') has rank <= 1 for every c (the
    g_s or the h_s span a line); or there are two cells and no c with all c_s ≠ 0 (over any extension field) makes
    K(c) diagonal and invertible."""
    ts = [t for t, c in enumerate(cells) if c[0] in G and c[1] in H]
    if len(ts) <= 1:
        return True
    g = {t: G[cells[t][0]][t] for t in ts}
    h = {t: H[cells[t][1]][t] for t in ts}
    if min(rk([[g[t][s] for t in ts] for s in range(len(S))], p, len(ts)),
           rk([[h[t][s] for t in ts] for s in range(len(S))], p, len(ts))) <= 1:
        return True
    if len(ts) > 2:
        return False
    C0 = nullspace([[g[t][s] * h[u][s] % p for s in range(len(S))] for t, u in itertools.permutations(ts, 2)],
                   p, len(S))
    forms = [np.eye(len(S), dtype=np.int64)[s] for s in range(len(S))]
    forms += [np.array([g[t][s] * h[t][s] % p for s in range(len(S))]) for t in ts]
    return any(not (C0 @ f % p).any() for f in forms)


def worst(lam, d, d2):
    """max over 0 <= r1, r2 <= 1 of min(3 − d r1 − d2 r2, lam (1 − max(r1, r2)) + 3 (r1 + r2) − max(d r1, d2 r2)):
    Theorem 6.7's bound on the drop per unit of rank, with B′'s drop at most lam · rank B′ and every line outside
    R (C) meeting W_a (W_b) in codimension at least d (d2). Four LPs, one per ordering of the two maxima."""
    best = -np.inf
    for big, big2 in itertools.product((0, 1), repeat=2):
        m = [1, 0] if big == 0 else [0, 1]
        m2 = [d, 0] if big2 == 0 else [0, d2]
        A = [[d, d2, 1], [-3 + lam * m[0] + m2[0], -3 + lam * m[1] + m2[1], 1],
             [-1 + 2 * big, 1 - 2 * big, 0], [-d + 2 * big2 * d, d2 - 2 * big2 * d2, 0]]
        res = linprog([0, 0, -1], A_ub=A, b_ub=[3, lam, 0, 0], bounds=[(0, 1), (0, 1), (None, None)],
                      method="highs")
        if res.status == 0:
            best = max(best, -res.fun)
    return best


def codims(cells, lab, S, axis, p):
    """The least codimension of W = span of S's labels in a line (row if axis 0) not holding it, 3 if none."""
    width = len(lab[0])
    W = [lab[s] for s in S]
    out = [3 - inter_dim(W, [lab[t] for t in range(len(cells)) if cells[t][axis] == key], p, width)
           for key in {c[axis] for c in cells}]
    return min([x for x in out if x] or [3])


def certificate(cells, a, b, S, p):
    """Theorem 6.7's bound on one typing's drop per unit of rank, for any coefficients (B′'s drop at most its rank
    when `hypothesis` holds, else at most k times it, k the cells of R × C)."""
    G, H = coords(cells, a, S, 0, p), coords(cells, b, S, 1, p)
    k = sum(c[0] in G and c[1] in H for c in cells)
    lam = 1 if hypothesis(cells, G, H, S, p) else k
    return worst(lam, codims(cells, a, S, 0, p), codims(cells, b, S, 1, p))


def lines(keys):
    return [set(x) for k in range(1, len(keys) + 1) for x in itertools.combinations(keys, k)]


def exact(cells, nodes, root, S, p):
    """The (I, J) for which some coefficients make every typing's drop 3, the number of (I, J) where every typing
    passes the containments, the number of typings, the number meeting Theorem 6.7's hypothesis, and the best
    `certificate` over typings (drop per unit of rank; the half charge needs < 5/2)."""
    labs = list(all_typings(cells, nodes, root, p))
    typings = [(coords(cells, a, S, 0, p), coords(cells, b, S, 1, p)) for a, b in labs]
    one = sum(hypothesis(cells, G, H, S, p) for G, H in typings)
    cert = min(certificate(cells, a, b, S, p) for a, b in labs)
    out, contained = [], 0
    for I in lines(sorted({c[0] for c in cells})):
        for J in lines(sorted({c[1] for c in cells})):
            unknowns, stacked = None, []
            for G, H in typings:
                sysm = bad_system(cells, G, H, S, I, J, p)
                if sysm is None:
                    break
                unknowns, rows = sysm
                stacked += rows
            else:
                contained += 1
                n = len(unknowns) + len(S)
                basis = nullspace(stacked, p, n)
                if all(basis[:, len(unknowns) + s].any() for s in range(len(S))):
                    out.append((I, J))
    return out, contained, len(typings), one, cert


def main_exact(path):
    total = bad = ones = proved = 0
    for p, S, cells, nodes, root, Si in configs(path):
        hits, contained, ntyp, one, cert = exact(cells, nodes, root, Si, p)
        total += 1
        bad += bool(hits)
        ones += bool(one)
        proved += cert < 2.5
        print(f"p={p} S={S}: {ntyp} typings ({one} meeting Theorem 6.7's hypothesis; best drop per rank "
              f"{cert:.4f}); (I, J) where every typing passes the containments: {contained}; where some "
              f"coefficients make every typing's drop 3: {len(hits)}"
              + (f", e.g. I={sorted(hits[0][0])} J={sorted(hits[0][1])}" if hits else ""), flush=True)
    print(f"{bad} of {total} configurations have a rank-1 branch that every typing charges 3; {ones} have a typing "
          f"meeting Theorem 6.7's hypothesis; {proved} have drop per rank < 5/2 (the half charge at every rank)",
          flush=True)


def main_sample(seed, trials):
    """`label_frac.py`'s search for unrescued 3-sets on fresh hierarchies, each then given to `exact`."""
    rng = np.random.default_rng(seed)
    for p in (2, 10007):
        for shape in ((3, 3), (3, 4)):
            cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
            count = bad = ones = proved = 0
            for trial in range(trials):
                nodes, root = random_hierarchy(cells, rng, "random" if trial % 2 else "spread")
                a, b, _, _, _ = label(cells, nodes, root, p)
                typings = None
                for S in itertools.combinations(range(len(cells)), 3):
                    if reachable(cells, a, b, S, p):
                        continue
                    if typings is None:
                        typings = list(all_typings(cells, nodes, root, p))
                    if any(reachable(cells, a2, b2, S, p) for a2, b2 in typings):
                        continue
                    count += 1
                    hits, _, _, one, cert = exact(cells, nodes, root, S, p)
                    bad += bool(hits)
                    ones += bool(one)
                    proved += cert < 2.5
                    if hits or not one:
                        print(f"p={p} S={tuple(cells[t] for t in S)}: drop 3 under every typing for "
                              f"{[(sorted(I), sorted(J)) for I, J in hits]}; typings meeting Theorem 6.7's "
                              f"hypothesis: {one}; best drop per rank {cert:.4f}; "
                              f"H={_nest(nodes, root, cells)}", flush=True)
            print(f"p={p} grid={shape[0]}x{shape[1]} seed={seed}: {count} 3-sets unrescued by labels, {bad} of them "
                  f"with a rank-1 branch every typing charges 3, {ones} with a typing meeting Theorem 6.7's "
                  f"hypothesis, {proved} with drop per rank < 5/2", flush=True)


def direct_drops(cells, a, b, S, q, p):
    """rank Φ − rank(Φ − Λ ⊗ uvᵀ) for every u, v over F2, with the targets of ⟨rows, q, cols⟩ on `cells`."""
    nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
    T = fq.matmul_targets(nr, nc, q)
    Phi = sum(np.kron(np.outer(a[t], b[t]), T[c]) for t, c in enumerate(cells)) % p
    Lam = sum(np.outer(a[s], b[s]) for s in S) % p
    us = [np.array(x) for x in itertools.product(range(2), repeat=nr * q)][1:]
    vs = [np.array(x) for x in itertools.product(range(2), repeat=q * nc)][1:]
    r0 = int(fq.rank(Phi, p)[0])
    out = {}
    for u in us:
        batch = np.array([(Phi - np.kron(Lam, np.outer(u, v))) % p for v in vs])
        for v, r in zip(vs, fq.rank(batch, p)):
            out[(tuple(u), tuple(v))] = r0 - int(r)
    return out


def main_rank(path, n, trials):
    """Theorem 6.7 directly, per typing of the first N configurations over F2, against ⌊w r⌋ with w that typing's
    `certificate`: every rank-1 branch at q = 2, then branches of rank r = 1..5 at q = 3 with part of their x-side
    on the rows R holding S's labels and part of their y-side on the columns C."""
    q, done = 3, 0
    rng = np.random.default_rng(1)
    for p, S, cells, nodes, root, Si in configs(path):
        if p != 2 or done >= n:
            continue
        done += 1
        nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
        T = fq.matmul_targets(nr, nc, q)
        for a, b in all_typings(cells, nodes, root, p):
            G, H = coords(cells, a, Si, 0, p), coords(cells, b, Si, 1, p)
            w = certificate(cells, a, b, Si, p)
            bound = {r: int(w * r + 1e-9) for r in range(1, 6)}
            one = max(direct_drops(cells, a, b, Si, 2, p).values())
            assert one <= bound[1], (S, one, w)
            Phi = sum(np.kron(np.outer(a[t], b[t]), T[c]) for t, c in enumerate(cells)) % p
            Lam = sum(np.outer(a[s], b[s]) for s in Si) % p
            r0 = int(fq.rank(Phi, p)[0])
            on_rows = np.array([[i // q in G] for i in range(nr * q)])
            on_cols = np.array([[i % nc in H] for i in range(q * nc)])
            worst = {}
            for r in range(1, 6):
                Bs = []
                while len(Bs) < trials:
                    k, k2 = rng.integers(0, r + 1, 2)
                    U = rng.integers(0, 2, (nr * q, r))
                    V = rng.integers(0, 2, (q * nc, r))
                    U[:, :k] *= on_rows
                    V[:, :k2] *= on_cols
                    B = U @ V[:, rng.permutation(r)].T % p
                    if int(fq.rank(B, p)[0]) == r:
                        Bs.append(B)
                drops = r0 - fq.rank(np.array([(Phi - np.kron(Lam, B)) % p for B in Bs]), p)
                worst[r] = int(drops.max())
                assert worst[r] <= bound[r], (S, r, worst[r], w)
            print(f"p={p} S={S} w={w:.4f}: largest rank-1 drop {one} (q = 2, every branch); largest drop at rank r "
                  f"(bound ⌊w r⌋): { {r: (d, bound[r]) for r, d in worst.items()} }", flush=True)


def main_climb(path, q, steps, restarts):
    """Hill-climbs branches B = U Vᵀ of rank r (c = 1, ⟨rows, q, cols⟩) toward the largest drop that every typing
    allows, the adversary of Theorem 6.7, and prints the best found against r."""
    rng = np.random.default_rng(2)
    for p, S, cells, nodes, root, Si in configs(path):
        nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
        T = fq.matmul_targets(nr, nc, q)
        labs = list(all_typings(cells, nodes, root, p))
        Phis = [sum(np.kron(np.outer(a[t], b[t]), T[c]) for t, c in enumerate(cells)) % p for a, b in labs]
        Lams = [sum(np.outer(a[s], b[s]) for s in Si) % p for a, b in labs]
        r0 = [int(fq.rank(F, p)[0]) for F in Phis]

        def drop(U, V):
            B = U @ V.T % p
            return min(r0[k] - int(fq.rank((Phis[k] - np.kron(Lams[k], B)) % p, p)[0]) for k in range(len(labs)))

        a, b = labs[0]
        rows, cols = coords(cells, a, Si, 0, p), coords(cells, b, Si, 1, p)
        on_rows = np.array([[i // q in rows] for i in range(nr * q)])
        on_cols = np.array([[i % nc in cols] for i in range(q * nc)])
        best = {}
        for r in range(1, min(q, 4) + 1):
            top = -99
            for k in range(restarts):
                while True:
                    U, V = rng.integers(0, p, (nr * q, r)), rng.integers(0, p, (q * nc, r))
                    if k % 2 == 0 and rows and cols:
                        U, V = U * on_rows, V * on_cols
                    if int(fq.rank(U @ V.T % p, p)[0]) == r:
                        break
                cur = drop(U, V)
                for _ in range(steps):
                    U2, V2 = U.copy(), V.copy()
                    M = U2 if rng.random() < 0.5 else V2
                    i, j = rng.integers(0, M.shape[0]), rng.integers(0, r)
                    M[i, j] = 0 if rng.random() < 0.5 else rng.integers(0, p)
                    if int(fq.rank(U2 @ V2.T % p, p)[0]) < r:
                        continue
                    new = drop(U2, V2)
                    if new >= cur:
                        U, V, cur = U2, V2, new
                top = max(top, cur)
            best[r] = top
        print(f"p={p} S={S} q={q}: largest drop found at rank r: {best}", flush=True)


def main_construct(path, q):
    """(c) at rank ρ: branches with x-side in the one row i0 holding S's labels, y-side on the columns holding them,
    and Γ_t = γ_t I_ρ with γ_t = 1/K_tt (c = 1), so that K diagonal makes the drop kρ; predicted against direct."""
    for p, S, cells, nodes, root, Si in configs(path):
        nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
        T = fq.matmul_targets(nr, nc, q)
        for a, b in all_typings(cells, nodes, root, p):
            G, H = coords(cells, a, Si, 0, p), coords(cells, b, Si, 1, p)
            if len(G) != 1:
                continue
            (i0,) = G
            ts = [t for t, c in enumerate(cells) if c[0] == i0 and c[1] in H]
            K = np.array([[int(G[i0][t] @ H[cells[u][1]][u] % p) for u in ts] for t in ts])
            gam = [pow(int(K[k, k]), p - 2, p) if K[k, k] else 1 for k in range(len(ts))]
            Phi = sum(np.kron(np.outer(a[t], b[t]), T[c]) for t, c in enumerate(cells)) % p
            Lam = sum(np.outer(a[s], b[s]) for s in Si) % p
            out = []
            for rho in range(1, q + 1):
                B = np.zeros((nr * q, q * nc), dtype=np.int64)
                for l in range(rho):
                    u = np.zeros(nr * q, dtype=np.int64)
                    u[i0 * q + l] = 1
                    v = np.zeros(q * nc, dtype=np.int64)
                    for k, t in enumerate(ts):
                        v[l * nc + cells[t][1]] = gam[k]
                    B = (B + np.outer(u, v)) % p
                D = np.kron(np.diag(gam), np.eye(rho, dtype=np.int64))
                M = (np.eye(len(ts) * rho, dtype=np.int64) - D @ np.kron(K, np.eye(rho, dtype=np.int64))) % p
                predicted = len(ts) * rho - int(fq.rank(M, p)[0])
                direct = int(fq.rank(Phi, p)[0]) - int(fq.rank((Phi - np.kron(Lam, B)) % p, p)[0])
                assert predicted == direct, (S, rho, predicted, direct)
                out.append(direct)
            print(f"p={p} S={S} cells {[cells[t] for t in ts]} K={K.tolist()}: drop at ρ = 1..{q}: {out} "
                  f"(predicted by (c), equal)", flush=True)


def width3(cells, rng, p):
    """Random labels of width 3 with (I), so every line spans everything and only the Γ condition decides."""
    while True:
        a = {t: rng.integers(0, p, 3) for t in range(len(cells))}
        b = {t: rng.integers(0, p, 3) for t in range(len(cells))}
        if all(rk([lab[t] for t in range(len(cells)) if cells[t][axis] == key], p, 3)
               == sum(cells[t][axis] == key for t in range(len(cells)))
               for lab, axis in ((a, 0), (b, 1)) for key in {c[axis] for c in cells}):
            return a, b


def main_check(path, n):
    q, done = 2, 0
    rng = np.random.default_rng(0)
    for p, S, cells, nodes, root, Si in configs(path):
        if p != 2 or done >= n:
            continue
        done += 1
        nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
        labels = list(itertools.islice(all_typings(cells, nodes, root, p), 3))
        if nr <= 3 and nc <= 3:
            labels += [width3(cells, rng, p) for _ in range(2)]
        for a, b in labels:
            G, H = coords(cells, a, Si, 0, p), coords(cells, b, Si, 1, p)
            agree = threes = 0
            drops = direct_drops(cells, a, b, Si, q, p)
            for (u, v), d in drops.items():
                I = {i for i in range(nr) if any(u[i * q:(i + 1) * q])}
                J = {j for j in range(nc) if any(v[k * nc + j] for k in range(q))}
                gam = {t: sum(u[cells[t][0] * q + k] * v[k * nc + cells[t][1]] for k in range(q)) % p
                       for t in range(len(cells))}
                sysm = bad_system(cells, G, H, Si, I, J, p)
                three = sysm is not None and all(
                    (sum(c * gam[t] for c, t in zip(row, sysm[0])) + sum(row[len(sysm[0]):])) % p == 0
                    for row in sysm[1])
                assert d <= 3 and (d == 3) == three, (S, u, v, d, three)
                agree += 1
                threes += three
            hist = {d: list(drops.values()).count(d) for d in sorted(set(drops.values()))}
            print(f"p={p} S={S} label width {len(a[0])}: {agree} rank-1 c agree with the direct rank "
                  f"({threes} with drop 3; drops {hist})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        main_check(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "--sample":
        main_sample(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "--rank":
        main_rank(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "--construct":
        main_construct(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "--climb":
        main_climb(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        main_exact(sys.argv[1])
