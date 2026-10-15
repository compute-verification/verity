#!/usr/bin/env python3
"""Attacks on low-rank masks (RS-D2F, CMP-13), run exactly over F_p.

With A = B = 0 and masks E = U V^T, F = W Z^T of rank r, the committed prefix targets are
T_s = U G_s Z^T with G_s = sum_{k <= sL} v_k w_k^T (r x r). Any rank-metric Steiner tree in F_p^{r x r} spanning 0 and
G_1..G_S computes every T_s with n^2 products per tree edge (edge c d^T gives (U c)(Z d)^T). The naive tree is the
chain (S min(r, L) edges). Two cheaper trees, both built by linear algebra on r x r matrices:

- hub J: hub h is the previous hub plus a rank-J step C D^T and serves r // J targets at rank distance <= r - J, where
  D solves D^T X_t = E_t with X_t = (G_t - P_{h-1})^{-1} C E_t. Needs r <= L; J = r/2 costs 3r/4 edges per target.
- median: for each triple (G_{2h}, G_{2h+1}, G_{2h+2}) a Steiner point b + P, where b = G_{2h+1} and P is a rank-k
  piece of both X = G_{2h} - b and Z = G_{2h+2} - b that rank(X) and rank(Z) subtract (P = Gam R The^T over the
  intersections of their column and row spaces, R a common outer inverse of The^T X^- Gam and The^T Z^- Gam).
  It costs (2 min(r, L) + min(r, 2L)) / 4 edges per target: 3r/4 for r <= L, (2L + r)/4 for L <= r <= 2L.

P < 2^25 (int64 products).
Usage: lowrank_attack.py N L R P SEED [J | median]   (prints one JSON line; exit 1 if any target is wrong)
"""

import json
import sys

import numpy as np


def rref(a, p):
    """Reduced row echelon form of a mod p (nonzero rows) and its pivot columns."""
    m = a % p
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = next((i for i in range(r, rows) if m[i, c]), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        m[r] = (m[r] * pow(int(m[r, c]), p - 2, p)) % p
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] = (m[i] - m[i, c] * m[r]) % p
        pivots.append(c)
        r += 1
    return m[:r], pivots


def inv_mod(a, p):
    n = a.shape[0]
    red, pivots = rref(np.concatenate([a % p, np.eye(n, dtype=np.int64)], axis=1), p)
    if pivots[:n] != list(range(n)):
        return None
    return red[:, n:]


def rank_factor(a, p):
    """Return (C, D) with a = C D^T mod p and C, D of width rank(a)."""
    a = a % p
    dt, pivots = rref(a, p)
    cmat = a[:, pivots]
    assert np.array_equal((cmat @ dt) % p, a)
    return cmat, dt.T.copy()


def null_space(a, p):
    """Columns spanning {x : a x = 0} mod p."""
    red, pivots = rref(a, p)
    free = [c for c in range(a.shape[1]) if c not in pivots]
    basis = np.zeros((a.shape[1], len(free)), dtype=np.int64)
    for t, f in enumerate(free):
        basis[f, t] = 1
        for row, pc in enumerate(pivots):
            basis[pc, t] = (-red[row, f]) % p
    return basis


def left_inverse(a, p, rng):
    """Some L with L a = I, for a of full column rank (None if the random try fails)."""
    q = rng.integers(0, p, size=(a.shape[1], a.shape[0]))
    inv = inv_mod((q @ a) % p, p)
    return None if inv is None else (inv @ q) % p


class Tree:
    """Rank-one steps from node 0 (the zero matrix): edges (parent, c, d) and node values."""

    def __init__(self, r, p):
        self.p = p
        self.nodes = [np.zeros((r, r), dtype=np.int64)]
        self.edges = []

    def walk(self, start, delta):
        node = start
        if not (delta % self.p).any():
            return node
        cm, dm = rank_factor(delta, self.p)
        for e in range(cm.shape[1]):
            self.edges.append((node, cm[:, e].copy(), dm[:, e].copy()))
            self.nodes.append((self.nodes[node] + np.outer(cm[:, e], dm[:, e])) % self.p)
            node = len(self.nodes) - 1
        return node


def hub_tree(gs, r, j, p, rng):
    k = max(1, r // j)
    tree = Tree(r, p)
    target_node = {}
    hub = 0
    t = 0
    while t < len(gs):
        group = list(range(t, min(t + k, len(gs))))
        step = None
        for _ in range(20):
            cmat = rng.integers(0, p, size=(r, j))
            xs, es = [], []
            for g in group:
                mt = inv_mod(gs[g] - tree.nodes[hub], p)
                if mt is None:
                    break
                et = rng.integers(0, p, size=(j, j))
                xs.append((((mt @ cmat) % p) @ et) % p)
                es.append(et)
            else:
                xmat = np.concatenate(xs, axis=1)
                emat = np.concatenate(es, axis=1)
                if xmat.shape[1] < r:
                    pad = rng.integers(0, p, size=(r, r - xmat.shape[1]))
                    xmat = np.concatenate([xmat, pad], axis=1)
                    emat = np.concatenate([emat, rng.integers(0, p, size=(j, pad.shape[1]))], axis=1)
                xinv = inv_mod(xmat, p)
                if xinv is not None:
                    step = (cmat @ ((emat @ xinv) % p)) % p
                    break
        if step is None:
            raise RuntimeError("hub construction failed")
        hub = tree.walk(hub, step)
        for g in group:
            target_node[g] = tree.walk(hub, (gs[g] - tree.nodes[hub]) % p)
        t += len(group)
    return tree, target_node


def median_piece(x, z, p, rng):
    """A largest P that both rank(x) and rank(z) subtract, with col(P), row(P) in both (None on a failed try)."""
    a1, b1 = rank_factor(x, p)
    a2, b2 = rank_factor(z, p)
    na = null_space(np.concatenate([a1, (-a2) % p], axis=1), p)
    nb = null_space(np.concatenate([b1, (-b2) % p], axis=1), p)
    gam = (a1 @ na[: a1.shape[1]]) % p
    the = (b1 @ nb[: b1.shape[1]]) % p
    k = min(gam.shape[1], the.shape[1]) // 2
    if k == 0:
        return np.zeros_like(x)
    gam, the = gam[:, : 2 * k], the[:, : 2 * k]
    invs = [left_inverse(m, p, rng) for m in (a1, b1, a2, b2)]
    if any(i is None for i in invs):
        return None
    xg = (invs[1].T @ invs[0]) % p
    zg = (invs[3].T @ invs[2]) % p
    s1 = (((the.T @ xg) % p) @ gam) % p
    s2 = (((the.T @ zg) % p) @ gam) % p
    f = rng.integers(0, p, size=(2 * k, k))
    g = null_space((((s1 - s2) @ f) % p).T, p)[:, :k]
    if g.shape[1] < k:
        return None
    h = inv_mod((((g.T @ s1) % p) @ f) % p, p)
    if h is None:
        return None
    rmat = (((f @ h) % p) @ g.T) % p
    return (((gam @ rmat) % p) @ the.T) % p


def median_tree(gs, r, p, rng):
    tree = Tree(r, p)
    target_node = {}
    prev = 0
    s = 0
    while s < len(gs):
        if s + 1 == len(gs):
            target_node[s] = tree.walk(prev, gs[s] - tree.nodes[prev])
            break
        a, b, c = tree.nodes[prev], gs[s], gs[s + 1]
        piece = None
        for _ in range(20):
            piece = median_piece((a - b) % p, (c - b) % p, p, rng)
            if piece is not None:
                break
        if piece is None:
            raise RuntimeError("median construction failed")
        med = tree.walk(prev, (b + piece - a) % p)
        target_node[s] = tree.walk(med, (b - tree.nodes[med]) % p)
        target_node[s + 1] = tree.walk(med, (c - tree.nodes[med]) % p)
        prev = target_node[s + 1]
        s += 2
    return tree, target_node


def main():
    n, big_l, r, p, seed = (int(x) for x in sys.argv[1:6])
    mode = sys.argv[6] if len(sys.argv) > 6 else str(max(1, r // 2))
    if p >= 1 << 25 or (mode != "median" and r > big_l):
        sys.exit("need P < 2^25, and R <= L for the hub")
    rng = np.random.default_rng(seed)
    s_total = n // big_l
    u, v, w, z = (rng.integers(0, p, size=(n, r)) for _ in range(4))
    gs = []
    acc = np.zeros((r, r), dtype=np.int64)
    for s in range(s_total):
        blk = slice(s * big_l, (s + 1) * big_l)
        acc = (acc + v[blk].T @ w[blk]) % p
        gs.append(acc.copy())
    if mode == "median":
        tree, target_node = median_tree(gs, r, p, rng)
    else:
        tree, target_node = hub_tree(gs, r, int(mode), p, rng)
    edges = tree.edges
    alpha = [(u @ c) % p for _, c, _ in edges]
    beta = [(z @ d) % p for _, _, d in edges]
    e_mat = (u @ v.T) % p
    f_mat = (w @ z.T) % p
    prefix_of = {node: s for s, node in target_node.items()}
    children = [0] * (len(edges) + 1)
    for parent, _, _ in edges:
        children[parent] += 1
    wrong = 0
    value = {0: np.zeros((n, n), dtype=np.int64)}
    for e, (parent, _, _) in enumerate(edges):
        value[e + 1] = (value[parent] + np.outer(alpha[e], beta[e])) % p
        if e + 1 in prefix_of:
            k = (prefix_of[e + 1] + 1) * big_l
            wrong += not np.array_equal(value[e + 1], (e_mat[:, :k] @ f_mat[:k, :]) % p)
        children[parent] -= 1
        for node in (parent, e + 1):
            if not children[node]:
                del value[node]
    rho = min(r, big_l)
    n_edges = len(edges)
    cell_macs = n * n * n_edges
    prep = 2 * n * r * n_edges
    gram = n * r * r
    predicted = (2 * rho - (2 * rho - min(r, 2 * big_l)) // 2) / 2 if mode == "median" else None
    rec = {"n": n, "L": big_l, "r": r, "p": p, "mode": mode, "S": s_total, "edges_per_cell": n_edges,
           "edges_per_target": n_edges / s_total, "predicted_per_target": predicted, "chain_edges": s_total * rho,
           "cells_over_chain": n_edges / (s_total * rho), "products_cells": cell_macs,
           "products_operand_prep": prep, "products_gram": gram, "products_total": cell_macs + prep + gram,
           "ratio_to_honest": (cell_macs + prep + gram) / n**3, "targets_wrong": wrong}
    print(json.dumps(rec), flush=True)
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
