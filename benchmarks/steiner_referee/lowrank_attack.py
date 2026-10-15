#!/usr/bin/env python3
"""The hub attack on low-rank masks (RS-D2F, CMP-13), run exactly over F_p.

With A = B = 0 and masks E = U V^T, F = W Z^T of rank r, the committed prefix targets are
T_s = U G_s Z^T with G_s = sum_{k <= sL} v_k w_k^T (r x r). Any rank-metric Steiner tree in F_p^{r x r} spanning 0 and
G_1..G_S computes every T_s with n^2 products per tree edge (edge c d^T gives (U c)(Z d)^T). The naive tree is the
chain (S*r edges, n^3 r/L products). The hub chain below is built by linear algebra only: hub h is the previous hub
plus a rank-j step C D^T, and serves k = r // j targets at rank distance <= r - j, where D solves D^T X_t = E_t with
X_t = (G_t - P_{h-1})^{-1} C E_t. With j = r/2 it costs about 3r/4 edges per target.

Needs r <= L (the first hub inverts G_1, of rank min(r, L)) and P < 2^25 (int64 products).
Usage: lowrank_attack.py N L R P SEED [J]   (prints one JSON line; exit 1 if any target is wrong)
"""

import json
import sys

import numpy as np


def inv_mod(a, p):
    n = a.shape[0]
    m = np.concatenate([a % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        piv = next((i for i in range(c, n) if m[i, c] % p), None)
        if piv is None:
            return None
        m[[c, piv]] = m[[piv, c]]
        m[c] = (m[c] * pow(int(m[c, c]), p - 2, p)) % p
        for i in range(n):
            if i != c and m[i, c]:
                m[i] = (m[i] - m[i, c] * m[c]) % p
    return m[:, n:]


def rank_factor(a, p):
    """Return (C, D) with a = C D^T mod p and C, D of width rank(a)."""
    a = a % p
    rows, cols = a.shape
    m = a.copy()
    pivots = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if m[i, c] % p), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        m[r] = (m[r] * pow(int(m[r, c]), p - 2, p)) % p
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] = (m[i] - m[i, c] * m[r]) % p
        pivots.append(c)
        r += 1
    dt = m[:r]
    cmat = a[:, pivots]
    assert np.array_equal((cmat @ dt) % p, a)
    return cmat, dt.T.copy()


def hub_tree(gs, r, j, p, rng):
    """Edges as (parent node, c, d); nodes: 0 is the zero matrix; returns edges, node values, target nodes."""
    k = max(1, r // j)
    nodes = [np.zeros((r, r), dtype=np.int64)]
    edges = []
    target_node = {}

    def walk(start, delta):
        node = start
        if not delta.any():
            return node
        cm, dm = rank_factor(delta, p)
        for e in range(cm.shape[1]):
            edges.append((node, cm[:, e].copy(), dm[:, e].copy()))
            nodes.append((nodes[node] + np.outer(cm[:, e], dm[:, e])) % p)
            node = len(nodes) - 1
        return node

    hub = 0
    t = 0
    s_total = len(gs)
    while t < s_total:
        group = list(range(t, min(t + k, s_total)))
        ok = False
        for _ in range(20):
            cmat = rng.integers(0, p, size=(r, j))
            xs, es = [], []
            for g in group:
                mt = inv_mod(gs[g] - nodes[hub], p)
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
                    dt = (emat @ xinv) % p
                    step = (cmat @ dt) % p
                    ok = True
                    break
        if not ok:
            raise RuntimeError("hub construction failed")
        hub = walk(hub, step)
        for g in group:
            target_node[g] = walk(hub, (gs[g] - nodes[hub]) % p)
        t += len(group)
    return edges, nodes, target_node


def main():
    n, big_l, r, p, seed = (int(x) for x in sys.argv[1:6])
    j = int(sys.argv[6]) if len(sys.argv) > 6 else max(1, r // 2)
    if r > big_l or p >= 1 << 25:
        sys.exit("need R <= L and P < 2^25")
    rng = np.random.default_rng(seed)
    s_total = n // big_l
    u, v, w, z = (rng.integers(0, p, size=(n, r)) for _ in range(4))
    gs = []
    acc = np.zeros((r, r), dtype=np.int64)
    for s in range(s_total):
        blk = slice(s * big_l, (s + 1) * big_l)
        acc = (acc + v[blk].T @ w[blk]) % p
        gs.append(acc.copy())
    edges, nodes, target_node = hub_tree(gs, r, j, p, rng)
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
    n_edges = len(edges)
    cell_macs = n * n * n_edges
    prep = 2 * n * r * n_edges
    gram = n * r * r
    rec = {"n": n, "L": big_l, "r": r, "p": p, "j": j, "S": s_total, "edges_per_cell": n_edges,
           "edges_per_target": n_edges / s_total, "chain_edges": s_total * min(r, big_l),
           "products_cells": cell_macs, "products_operand_prep": prep, "products_gram": gram,
           "products_total": cell_macs + prep + gram, "honest_n3": n**3, "naive_n3r_over_L": n**3 * r // big_l,
           "ratio_to_honest": (cell_macs + prep + gram) / n**3, "ratio_to_naive": (cell_macs + prep + gram) /
           (n**3 * r / big_l + 2 * n * n * r + gram), "targets_wrong": wrong}
    print(json.dumps(rec), flush=True)
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
