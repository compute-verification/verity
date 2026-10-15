"""Upper bounds on the bidirected cut LP of <2,2,2> over F2 from subgraphs (CMP-8 worker A1b).

The BCR restricted to the nodes V_k = {M : rank M + min_t rank(C_t - M) <= 2 + k} (detour at most k from a geodesic
0 -> C_t) and the rank-1 steps among them is an upper bound on the full BCR (a flow in a subgraph is a flow), and
increases to it with k. No symmetry reduction: one flow per terminal. Usage: bcr_restricted.py K
"""

import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import merges

def ranks():
    """Rank over F2 of every 4x4 matrix (16-bit masks, bit 4*row + col)."""
    return np.array([merges.f2rank([x >> (4 * i) & 15 for i in range(4)]) for x in range(1 << 16)], dtype=np.int64)


def main():
    k = int(sys.argv[1])
    rk = ranks()
    assert rk[0] == 0 and all(rk[g] == 1 for g in merges.RANK1) and all(rk[t] == 2 for t in merges.TGT)
    m = np.arange(1 << 16, dtype=np.int64)
    detour = rk[m] + np.min(np.stack([rk[m ^ t] for t in merges.TGT]), axis=0) - 2
    nodes = np.flatnonzero(detour <= k)
    idx = np.full(1 << 16, -1, dtype=np.int64)
    idx[nodes] = np.arange(len(nodes))
    g = np.array(merges.RANK1, dtype=np.int64)
    tail = np.repeat(nodes, len(g))
    head = tail ^ np.tile(g, len(nodes))
    keep = idx[head] >= 0
    tail, head = idx[tail[keep]], idx[head[keep]]
    na, nv, T = len(tail), len(nodes), len(merges.TGT)
    # variables: x_a (na), then f^t_a (T * na)
    rows, cols, vals = [], [], []
    beq = np.zeros(T * nv)
    for t in range(T):
        off = na * (1 + t)
        rows += [t * nv + head, t * nv + tail]
        cols += [off + np.arange(na)] * 2
        vals += [np.ones(na), -np.ones(na)]
        beq[t * nv + idx[merges.TGT[t]]] += 1
        beq[t * nv + idx[0]] -= 1
    Aeq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(T * nv, na * (1 + T)))
    r2 = np.concatenate([np.arange(T * na)] * 2)
    c2 = np.concatenate([na + np.arange(T * na), np.tile(np.arange(na), T)])
    v2 = np.concatenate([np.ones(T * na), -np.ones(T * na)])
    Aub = sp.csr_matrix((v2, (r2, c2)), shape=(T * na, na * (1 + T)))
    cost = np.concatenate([np.ones(na), np.zeros(T * na)])
    res = linprog(cost, A_ub=Aub, b_ub=np.zeros(T * na), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    print(f"detour <= {k}: {nv} nodes, {na} arcs; restricted BCR = {res.fun:.6f} ({res.message})")


if __name__ == "__main__":
    main()
