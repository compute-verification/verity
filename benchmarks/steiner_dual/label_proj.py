"""Can one branch's output set S get rank Λ(S) <= floor(|S|/2) + 1 under labels valid for a laminar family?

Lemma 3's labels for a binary hierarchy H (check_labels.label) satisfy (I) and (L). Quotienting every a_t by a
subspace U of W_a = span{a_t : t in S} keeps (L) and lowers rank Λ(S) to at most dim W_a - dim U; it keeps (I) iff
U meets no row's label space R_r, which a generic U of that dimension does over an infinite extension field iff
dim(W_a ∩ R_r) <= dim W_a - dim U for every row r. The same holds for the b's and columns. So S can reach
g = floor(|S|/2) + 1 iff dim W_a <= g, or max_r dim(W_a ∩ R_r) <= g, or the same for b and columns.

    python label_proj.py [seed]   (prints, per grid and |S|, the number of (H, S) pairs and of failures)
"""

import itertools
import sys

import numpy as np

from check_labels import label, members, nullspace, random_hierarchy, rk


def inter_dim(U, V, p, width):
    return rk(U, p, width) + rk(V, p, width) - rk(U + V, p, width)


def reachable(cells, a, b, S, p):
    k = len(S)
    g = k // 2 + 1
    for lab, axis in ((a, 0), (b, 1)):
        width = len(next(iter(lab.values())))
        W = [lab[t] for t in S]
        dW = rk(W, p, width)
        if dW <= g:
            return True
        groups = {}
        for t, c in enumerate(cells):
            groups.setdefault(c[axis], []).append(lab[t])
        if max(inter_dim(W, R, p, width) for R in groups.values()) <= g:
            return True
    return False


def typed_labels(cells, nodes, root, p, typing):
    """Labels for H with every internal node's type given (x: children's a-aggregates parallel, y: b's), the most
    generic ones with those relations; None if (I) fails."""
    P = len(cells)
    eye = np.eye(P, dtype=np.int64)
    xi, eta, Vx, Vy = {}, {}, [], []

    def walk(v):
        if nodes[v][0] == "leaf":
            xi[v] = eta[v] = eye[nodes[v][1]]
            return
        l, r = nodes[v][1], nodes[v][2]
        walk(l)
        walk(r)
        if typing[v] == "x":
            xi[v], eta[v] = xi[l], (eta[l] + eta[r]) % p
            Vx.append((xi[l] - xi[r]) % p)
        else:
            xi[v], eta[v] = (xi[l] + xi[r]) % p, eta[l]
            Vy.append((eta[l] - eta[r]) % p)

    walk(root)
    A, B = nullspace(Vx, p, P), nullspace(Vy, p, P)
    a = {t: A[:, t] for t in range(P)}
    b = {t: B[:, t] for t in range(P)}
    for axis, lab, M in ((0, a, A), (1, b, B)):
        for key in {c[axis] for c in cells}:
            ts = [t for t in range(P) if cells[t][axis] == key]
            if rk([lab[t] for t in ts], p, M.shape[0]) < len(ts):
                return None
    for v, nd in enumerate(nodes):
        if nd[0] == "int":
            L = sum(np.outer(a[t], b[t]) for t in members(nodes, v)) % p
            assert rk(list(L), p, L.shape[1]) <= 1, "(L) fails"
    return a, b


def all_typings(cells, nodes, root, p):
    internal = [v for v, nd in enumerate(nodes) if nd[0] == "int"]
    for bits in itertools.product("xy", repeat=len(internal)):
        lab = typed_labels(cells, nodes, root, p, dict(zip(internal, bits)))
        if lab is not None:
            yield lab


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    for p in (2, 10007):
        for shape in ((2, 3), (3, 3), (2, 4), (3, 4)):
            cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
            stats = {}
            examples = []
            for trial in range(12):
                style = "random" if trial % 2 else "spread"
                nodes, root = random_hierarchy(cells, rng, style)
                a, b, _, _, _ = label(cells, nodes, root, p)
                typings = None
                for k in range(3, min(6, len(cells)) + 1):
                    for S in itertools.combinations(range(len(cells)), k):
                        n, bad, rescued = stats.get(k, (0, 0, 0))
                        ok = reachable(cells, a, b, S, p)
                        saved = False
                        if not ok:
                            if typings is None:
                                typings = list(all_typings(cells, nodes, root, p))
                            saved = any(reachable(cells, a2, b2, S, p) for a2, b2 in typings)
                            if not saved and len(examples) < 2:
                                examples.append((tuple(cells[t] for t in S), _nest(nodes, root, cells)))
                        stats[k] = (n + 1, bad + (not ok), rescued + saved)
            print(f"p={p} grid={shape[0]}x{shape[1]} |S|: (pairs, Lemma 3 typing fails, rescued by another "
                  f"typing) {stats}", flush=True)
            for ex in examples:
                print("   unrescued:", ex, flush=True)


def _nest(nodes, v, cells):
    if nodes[v][0] == "leaf":
        return cells[nodes[v][1]]
    return (_nest(nodes, nodes[v][1], cells), _nest(nodes, nodes[v][2], cells))


if __name__ == "__main__":
    main()
