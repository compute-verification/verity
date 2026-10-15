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
    python exact_branch.py --rank LOG N TRIALS (Theorem 6.7's bound against direct ranks, rank-1..5 branches, q = 3)
    python exact_branch.py --check LOG N (compares with rank Φ − rank(Φ − Λ ⊗ uvᵀ) for every rank-1 uvᵀ, q = 2, F2,
                                          on the first N configurations of LOG over F2, with c = 1)
"""

import itertools
import sys

import numpy as np

import fq
from check_labels import label, nullspace, random_hierarchy, rk, rref
from label_frac import configs
from label_proj import _nest, all_typings, reachable


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
    """Theorem 6.7's hypothesis for one typing: the cells of R × C (R, C the lines holding S's labels) are at most
    one, or two whose matrix K(c)_{tt'} = Σ_s g_s(t) c_s h_s(t') no c with all c_s ≠ 0 (over any extension field)
    makes diagonal and invertible."""
    ts = [t for t, c in enumerate(cells) if c[0] in G and c[1] in H]
    if len(ts) <= 1:
        return True
    if len(ts) > 2:
        return False
    g = {t: G[cells[t][0]][t] for t in ts}
    h = {t: H[cells[t][1]][t] for t in ts}
    C0 = nullspace([[g[t][s] * h[u][s] % p for s in range(len(S))] for t, u in itertools.permutations(ts, 2)],
                   p, len(S))
    forms = [np.eye(len(S), dtype=np.int64)[s] for s in range(len(S))]
    forms += [np.array([g[t][s] * h[t][s] % p for s in range(len(S))]) for t in ts]
    return any(not (C0 @ f % p).any() for f in forms)


def lines(keys):
    return [set(x) for k in range(1, len(keys) + 1) for x in itertools.combinations(keys, k)]


def exact(cells, nodes, root, S, p):
    """The (I, J) for which some coefficients make every typing's drop 3, the number of (I, J) where every typing
    passes the containments, the number of typings, and the number of typings meeting Theorem 6.7's hypothesis."""
    typings = [(coords(cells, a, S, 0, p), coords(cells, b, S, 1, p)) for a, b in all_typings(cells, nodes, root, p)]
    one = sum(hypothesis(cells, G, H, S, p) for G, H in typings)
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
    return out, contained, len(typings), one


def main_exact(path):
    total = bad = ones = 0
    for p, S, cells, nodes, root, Si in configs(path):
        hits, contained, ntyp, one = exact(cells, nodes, root, Si, p)
        total += 1
        bad += bool(hits)
        ones += bool(one)
        print(f"p={p} S={S}: {ntyp} typings ({one} meeting Theorem 6.7's hypothesis); (I, J) where every typing "
              f"passes the containments: {contained}; where some coefficients make every typing's drop 3: {len(hits)}"
              + (f", e.g. I={sorted(hits[0][0])} J={sorted(hits[0][1])}" if hits else ""), flush=True)
    print(f"{bad} of {total} configurations have a rank-1 branch that every typing charges 3; "
          f"{ones} have a typing meeting Theorem 6.7's hypothesis", flush=True)


def main_sample(seed, trials):
    """`label_frac.py`'s search for unrescued 3-sets on fresh hierarchies, each then given to `exact`."""
    rng = np.random.default_rng(seed)
    for p in (2, 10007):
        for shape in ((3, 3), (3, 4)):
            cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
            count = bad = ones = 0
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
                    hits, _, _, one = exact(cells, nodes, root, S, p)
                    bad += bool(hits)
                    ones += bool(one)
                    if hits or not one:
                        print(f"p={p} S={tuple(cells[t] for t in S)}: drop 3 under every typing for "
                              f"{[(sorted(I), sorted(J)) for I, J in hits]}; typings meeting Theorem 6.7's "
                              f"hypothesis: {one}; "
                              f"H={_nest(nodes, root, cells)}", flush=True)
            print(f"p={p} grid={shape[0]}x{shape[1]} seed={seed}: {count} 3-sets unrescued by labels, {bad} of them "
                  f"with a rank-1 branch every typing charges 3, {ones} with a typing meeting Theorem 6.7's "
                  f"hypothesis", flush=True)


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
    """Theorem 6.7 directly: under a typing with one row i0 and one column j0 holding S's labels, branches B of rank
    r = 1..5 (q = 3, F2), part of their x-side in row i0 and part of their y-side in column j0, against ⌊17r/7⌋."""
    q, done = 3, 0
    rng = np.random.default_rng(1)
    for p, S, cells, nodes, root, Si in configs(path):
        if p != 2 or done >= n:
            continue
        lab = next(((a, b) for a, b in all_typings(cells, nodes, root, p)
                    if len(coords(cells, a, Si, 0, p)) == 1 and len(coords(cells, b, Si, 1, p)) == 1), None)
        if lab is None:
            continue
        done += 1
        a, b = lab
        (i0,), (j0,) = coords(cells, a, Si, 0, p), coords(cells, b, Si, 1, p)
        nr, nc = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
        T = fq.matmul_targets(nr, nc, q)
        Phi = sum(np.kron(np.outer(a[t], b[t]), T[c]) for t, c in enumerate(cells)) % p
        Lam = sum(np.outer(a[s], b[s]) for s in Si) % p
        r0 = int(fq.rank(Phi, p)[0])
        worst = {}
        for r in range(1, 6):
            Bs = []
            while len(Bs) < trials:
                k, k2 = rng.integers(0, min(r, q) + 1, 2)
                U = rng.integers(0, 2, (nr * q, r))
                V = rng.integers(0, 2, (q * nc, r))
                U[:, :k] *= np.array([[i // q == i0] for i in range(nr * q)])
                V[:, :k2] *= np.array([[i % nc == j0] for i in range(q * nc)])
                B = U @ V[:, rng.permutation(r)].T % p
                if int(fq.rank(B, p)[0]) == r:
                    Bs.append(B)
            drops = r0 - fq.rank(np.array([(Phi - np.kron(Lam, B)) % p for B in Bs]), p)
            worst[r] = int(drops.max())
            assert worst[r] <= 17 * r // 7, (S, r, worst[r])
        print(f"p={p} S={S}: largest drop at rank r (bound ⌊17r/7⌋): "
              f"{ {r: (d, 17 * r // 7) for r, d in worst.items()} }", flush=True)


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
    else:
        main_exact(sys.argv[1])
