"""Independent check of the labelled-flattening certificate for block rigidity (CMP-8, worker A1b).

Written from the statement of A1a's Lemma 3 (note 20261015T0416Z-draft-steiner-rigidity-laminar), not from A1a's code.
For a binary hierarchy H on a set P of grid cells it types every internal node x or y by the handle rule, computing
the handles rho, gamma by linear algebra over F_p (not by the recursion), builds the labels a_t, b_t as bases of the
annihilators of V_x and V_y, and checks
  (L) rank Lambda(S) <= 1 for every node S of H, Lambda(S) = sum_{t in S} a_t b_t^T;
  (I) {a_t : t in row r} independent for every r, {b_t : t in column c} independent for every c;
  validity of every node, and that the typing rule never meets a node whose children share a row handle and a column
  handle.
End to end (matmul <n,k,m>, all cells): random Steiner trees with hierarchy H and dense random Steiner points; checks
  Phi = sum_t (a_t b_t^T) (x) T_t = sum_S Lambda(S) (x) W_S,  rank Phi = k*|P|,  #edges = sum_S rank W_S >= rank Phi.
Usage: check_labels.py [seed]   (a few seconds)
"""

import sys

import numpy as np

import fq


def rref(m, p):
    """Row-reduced echelon form over F_p; returns (rows of the reduced basis, pivot columns)."""
    a = np.array(m, dtype=np.int64) % p
    if a.size == 0:
        return a.reshape(0, a.shape[1] if a.ndim == 2 else 0), []
    rows, cols = a.shape
    piv = []
    r = 0
    for c in range(cols):
        nz = [i for i in range(r, rows) if a[i, c]]
        if not nz:
            continue
        i = nz[0]
        a[[r, i]] = a[[i, r]]
        a[r] = a[r] * pow(int(a[r, c]), p - 2, p) % p
        for j in range(rows):
            if j != r and a[j, c]:
                a[j] = (a[j] - a[j, c] * a[r]) % p
        piv.append(c)
        r += 1
        if r == rows:
            break
    return a[:r], piv


def rk(vecs, p, width):
    if len(vecs) == 0:
        return 0
    return len(rref(np.array(vecs).reshape(-1, width), p)[1])


def nullspace(vecs, p, width):
    """Basis (as rows) of {s : v.s = 0 for all v in vecs}."""
    if len(vecs) == 0:
        return np.eye(width, dtype=np.int64)
    red, piv = rref(np.array(vecs).reshape(-1, width), p)
    free = [c for c in range(width) if c not in piv]
    out = []
    for f in free:
        s = np.zeros(width, dtype=np.int64)
        s[f] = 1
        for i, c in enumerate(piv):
            s[c] = (-red[i, f]) % p
        out.append(s)
    return np.array(out, dtype=np.int64).reshape(-1, width)


def random_hierarchy(cells, rng, style):
    """Binary hierarchy as a list of nodes: ('leaf', t) or ('int', left, right); returns (nodes, root)."""
    nodes = [("leaf", i) for i in range(len(cells))]
    alive = list(range(len(cells)))
    while len(alive) > 1:
        if style == "random":
            i, j = rng.choice(len(alive), 2, replace=False)
        else:
            # prefer merging components whose cells avoid sharing rows and columns (diagonals, L-shapes)
            best, bi, bj = None, 0, 1
            for _ in range(8):
                i, j = rng.choice(len(alive), 2, replace=False)
                ri = {cells[t][0] for t in members(nodes, alive[i])}
                rj = {cells[t][0] for t in members(nodes, alive[j])}
                ci = {cells[t][1] for t in members(nodes, alive[i])}
                cj = {cells[t][1] for t in members(nodes, alive[j])}
                score = len(ri & rj) + len(ci & cj) + rng.random()
                if best is None or score < best:
                    best, bi, bj = score, i, j
            i, j = bi, bj
        a, b = alive[i], alive[j]
        nodes.append(("int", a, b))
        alive = [x for k, x in enumerate(alive) if k not in (i, j)] + [len(nodes) - 1]
    return nodes, alive[0]


def members(nodes, v):
    if nodes[v][0] == "leaf":
        return [nodes[v][1]]
    return members(nodes, nodes[v][1]) + members(nodes, nodes[v][2])


def label(cells, nodes, root, p):
    P = len(cells)
    rows = sorted({c[0] for c in cells})
    cols = sorted({c[1] for c in cells})
    rowvec = {r: [np.eye(P, dtype=np.int64)[t] for t in range(P) if cells[t][0] == r] for r in rows}
    colvec = {c: [np.eye(P, dtype=np.int64)[t] for t in range(P) if cells[t][1] == c] for c in cols}
    xi, eta, Vx, Vy, rho, gam, typ = {}, {}, {}, {}, {}, {}, {}

    def handles(v):
        rho[v] = {r for r in rows
                  if rk(Vx[v] + rowvec[r] + [xi[v]], p, P) == rk(Vx[v] + rowvec[r], p, P)}
        gam[v] = {c for c in cols
                  if rk(Vy[v] + colvec[c] + [eta[v]], p, P) == rk(Vy[v] + colvec[c], p, P)}

    def valid(v):
        ok = all(rk(Vx[v] + rowvec[r], p, P) == rk(Vx[v], p, P) + len(rowvec[r]) for r in rows)
        return ok and all(rk(Vy[v] + colvec[c], p, P) == rk(Vy[v], p, P) + len(colvec[c]) for c in cols)

    order = []

    def post(v):
        if nodes[v][0] == "int":
            post(nodes[v][1])
            post(nodes[v][2])
        order.append(v)

    post(root)
    for v in order:
        if nodes[v][0] == "leaf":
            e = np.eye(P, dtype=np.int64)[nodes[v][1]]
            xi[v], eta[v], Vx[v], Vy[v] = e, e, [], []
        else:
            a, b = nodes[v][1], nodes[v][2]
            if not rho[a] & rho[b]:
                typ[v] = "x"
                xi[v] = xi[a]
                eta[v] = (eta[a] + eta[b]) % p
                Vx[v] = Vx[a] + Vx[b] + [(xi[a] - xi[b]) % p]
                Vy[v] = Vy[a] + Vy[b]
            else:
                assert not gam[a] & gam[b], "typing rule stuck: children share a row and a column handle"
                typ[v] = "y"
                xi[v] = (xi[a] + xi[b]) % p
                eta[v] = eta[a]
                Vx[v] = Vx[a] + Vx[b]
                Vy[v] = Vy[a] + Vy[b] + [(eta[a] - eta[b]) % p]
        handles(v)
        assert valid(v), "invalid node"
    A = nullspace(Vx[root], p, P)  # rows s^(i); a_t = column t
    B = nullspace(Vy[root], p, P)
    a = {t: A[:, t] for t in range(P)}
    b = {t: B[:, t] for t in range(P)}
    for r in rows:
        ts = [t for t in range(P) if cells[t][0] == r]
        assert rk([a[t] for t in ts], p, A.shape[0]) == len(ts), "(I) fails in a row"
    for c in cols:
        ts = [t for t in range(P) if cells[t][1] == c]
        assert rk([b[t] for t in ts], p, B.shape[0]) == len(ts), "(I) fails in a column"
    lam = {}
    for v in order:
        ms = members(nodes, v)
        L = sum(np.outer(a[t], b[t]) for t in ms) % p
        assert fq.rank(L, p)[0] <= 1, "(L) fails"
        lam[v] = L
    return a, b, lam, order, typ


def matmul_cells(n, k, m):
    T = fq.matmul_targets(n, m, k)
    cells = sorted(T)
    return cells, [T[c] for c in cells]


def end_to_end(n, k, m, p, rng, style, steiner):
    cells, Ts = matmul_cells(n, k, m)
    nodes, root = random_hierarchy(cells, rng, style)
    a, b, lam, order, _ = label(cells, nodes, root, p)
    R, C = Ts[0].shape
    # node values: leaves are targets; internal nodes are dense random, low-rank shifts, or partial target sums
    val = {}
    for v in order:
        if nodes[v][0] == "leaf":
            val[v] = Ts[nodes[v][1]]
        elif steiner == "dense":
            val[v] = rng.integers(0, p, (R, C))
        elif steiner == "lowrank":
            val[v] = (rng.integers(0, p, (R, 2)) @ rng.integers(0, p, (2, C))) % p
        else:
            ms = members(nodes, v)
            val[v] = sum(Ts[t] * int(rng.integers(0, p)) for t in ms) % p
    parent = {}
    for v in order:
        if nodes[v][0] == "int":
            parent[nodes[v][1]] = v
            parent[nodes[v][2]] = v
    W = {v: (val[v] - (val[parent[v]] if v in parent else 0)) % p for v in order}
    edges = sum(int(fq.rank(W[v], p)[0]) for v in order)
    Phi = sum(np.kron(np.outer(a[t], b[t]), Ts[t]) for t in range(len(cells))) % p
    Phi2 = sum(np.kron(lam[v], W[v]) for v in order) % p
    assert np.array_equal(Phi, Phi2), "flattening identity fails"
    rphi = int(fq.rank(Phi, p)[0])
    assert rphi == k * len(cells), ("rank Phi", rphi, k * len(cells))
    assert edges >= rphi
    return edges, rphi


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    count = 0
    for p in (2, 3, 7):
        for (pr, qc) in ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (2, 6), (5, 5)):
            for style in ("random", "spread"):
                for _ in range(6 if pr * qc <= 9 else 2):
                    allc = [(r, c) for r in range(pr) for c in range(qc)]
                    keep = rng.random(len(allc)) < (0.75 if rng.random() < 0.5 else 1.0)
                    cells = [x for x, kk in zip(allc, keep) if kk] or allc
                    if len(cells) < 2:
                        continue
                    nodes, root = random_hierarchy(cells, rng, style)
                    label(cells, nodes, root, p)
                    count += 1
    print("Lemma 3 (labels): passed on", count, "hierarchies")
    e2e = 0
    for p in (2, 3, 5):
        for (n, k, m) in ((2, 2, 2), (2, 3, 2), (3, 2, 3), (2, 1, 3)):
            for style in ("random", "spread"):
                for steiner in ("dense", "lowrank", "partial"):
                    edges, rphi = end_to_end(n, k, m, p, rng, style, steiner)
                    e2e += 1
    print("end to end (random trees, dense Steiner points): passed on", e2e, "trees")


if __name__ == "__main__":
    main()
