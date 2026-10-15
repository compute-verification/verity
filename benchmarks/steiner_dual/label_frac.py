"""Rank-w labels for the branches that no rank-1 labeling rescues (`label_proj.py`'s unrescued 3-sets).

Labels of rank w, A_t = Σ_i a_t^(i) b_t^(i)ᵀ, give rank Φ = w Σ_t rank T_t when the w·|row| vectors of each grid row
(and each column) are independent, (L_w) rank Λ(X) <= w on the family, and then #products >= Σ_t rank T_t − Σ_j
(rank Λ(c_j)/w − 1) r_j. The direct sum of w typings' labels, quotiented by a generic subspace of the span of the a's
on S, reaches rank Λ(S) = max_r Σ_i dim(W_i ∩ R_{i,r}) (the same with b's and columns). A distribution μ over typings
reaches v = min(max_r E_μ dim(W ∩ R_r), max_c E_μ dim(W ∩ C_c)) per unit of w, an LP; a branch into k targets then
saves at most (v − 1) r, and the half charge needs v <= 1 + k/2.

    python label_frac.py [seed] [trials] [min]   (prints each unrescued (H, S) with v and the best pair of typings;
                                                  with min, also a minimal set of cells on which S stays unrescued)
"""

import itertools
import sys
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

from check_labels import label, random_hierarchy
from label_proj import _nest, all_typings, inter_dim, reachable


def profile(cells, lab, S, p, axis):
    width = len(next(iter(lab.values())))
    W = [lab[t] for t in S]
    groups = {}
    for t, c in enumerate(cells):
        groups.setdefault(c[axis], []).append(lab[t])
    return tuple(inter_dim(W, groups[key], p, width) for key in sorted(groups))


def game(profiles):
    """min over distributions on the typings of max over lines of the expected intersection dimension."""
    n, m = len(profiles), len(profiles[0])
    c = np.zeros(n + 1)
    c[-1] = 1
    A = np.hstack([np.array(profiles, dtype=float).T, -np.ones((m, 1))])
    res = linprog(c, A_ub=A, b_ub=np.zeros(m), A_eq=[[1.0] * n + [0.0]], b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    return res.fun


def best_pair(profiles):
    return min(max(x + y for x, y in zip(P, Q)) for P, Q in itertools.combinations_with_replacement(profiles, 2))


def restrict(cells, nodes, root, keep):
    """The hierarchy on the cells `keep` (indices), with emptied nodes removed and unary nodes contracted."""
    keep = sorted(keep)
    index = {t: i for i, t in enumerate(keep)}
    out = []

    def build(v):
        if nodes[v][0] == "leaf":
            if nodes[v][1] not in index:
                return None
            out.append(("leaf", index[nodes[v][1]]))
            return len(out) - 1
        left, right = build(nodes[v][1]), build(nodes[v][2])
        if left is None or right is None:
            return right if left is None else left
        out.append(("int", left, right))
        return len(out) - 1

    return [cells[t] for t in keep], out, build(root), index


def unrescued(cells, nodes, root, S, p):
    return not any(reachable(cells, a, b, S, p) for a, b in all_typings(cells, nodes, root, p))


def minimal(cells, nodes, root, S, p):
    """Greedily drop cells outside S while S stays unrescued; returns the remaining cells and hierarchy."""
    keep = set(range(len(cells)))
    for t in range(len(cells)):
        if t in S:
            continue
        sub = restrict(cells, nodes, root, keep - {t})
        if unrescued(*sub[:3], tuple(sub[3][s] for s in S), p):
            keep.discard(t)
    sub_cells, sub_nodes, sub_root, _ = restrict(cells, nodes, root, keep)
    return sub_cells, _nest(sub_nodes, sub_root, sub_cells)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    minimize = len(sys.argv) > 3 and sys.argv[3] == "min"
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p in (2, 10007):
        for shape in ((3, 3), (3, 4)):
            cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
            count = 0
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
                    rows = sorted({profile(cells, a2, S, p, 0) for a2, _ in typings})
                    cols = sorted({profile(cells, b2, S, p, 1) for _, b2 in typings})
                    v = min(game(rows), game(cols))
                    pair = min(best_pair(rows), best_pair(cols))
                    worst = max(worst, v)
                    print(f"p={p} grid={shape[0]}x{shape[1]} S={tuple(cells[t] for t in S)} v={Fraction(v).limit_denominator(64)} "
                          f"pair={pair}/2 row profiles={rows} column profiles={cols} H={_nest(nodes, root, cells)}",
                          flush=True)
                    if minimize:
                        print(f"   minimal: {minimal(cells, nodes, root, S, p)}", flush=True)
            print(f"p={p} grid={shape[0]}x{shape[1]}: {count} unrescued 3-sets", flush=True)
    print(f"largest v: {Fraction(worst).limit_denominator(64)} (the half charge for k = 3 needs v <= 5/2)")


if __name__ == "__main__":
    main()
