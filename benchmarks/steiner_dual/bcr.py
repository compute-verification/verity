"""Exact Steiner cut-LP values for <n,k,m> matmul over F2 on the full rank-1 graph (CMP-8 worker A1b).

Nodes: all (n k) x (k m) matrices over F2 as bitmasks (bit row*(k m) + col; row x_{r j} = r*k + j, col y_{j c} = j*m + c).
Arcs: M -> M + u v^T for every rank-1 u v^T. Root 0, terminals the full outputs C_(r,c) = sum_j x_rj y_jc.

  BCR (bidirected cut relaxation): min sum_a x_a  s.t. for every terminal t, x supports a unit 0 -> t flow.
  UCR (undirected cut relaxation): the same with x_a = x_{reverse a}, objective sum over edges.

Symmetry reduction (exact): G = <inner GL_k, row permutations of X, column permutations of Y, transpose if n = m>
acts on nodes and arcs and permutes the terminals transitively. An optimal x is G-invariant and then a flow to one
terminal t0 suffices, which may be taken Stab(t0)-invariant. Variables: x per G-orbit of arcs, f per Stab-orbit.
With K, only the nodes within detour K of a geodesic 0 -> terminal (rank M + min_t rank(C_t - M) <= rank C_t + K)
and the steps among them: a G-invariant subgraph, so the value is an upper bound on the full one, exactly computed.
Usage: bcr.py n k m [ucr|bcr] [K]
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def main():
    n, k, m = (int(a) for a in sys.argv[1:4])
    mode = sys.argv[4] if len(sys.argv) > 4 else "bcr"
    detour_max = int(sys.argv[5]) if len(sys.argv) > 5 else None
    R, C = n * k, k * m
    nb = R * C
    N = 1 << nb
    t0 = time.time()

    def bit(row, col):
        return row * C + col

    gens = sorted({sum(1 << bit(r, c) for r in range(R) if u >> r & 1 for c in range(C) if v >> c & 1)
                   for u in range(1, 1 << R) for v in range(1, 1 << C)})
    gens = np.array(gens, dtype=np.int64)
    ng = len(gens)
    gidx = {int(g): i for i, g in enumerate(gens)}

    def target(r, c):
        return sum(1 << bit(r * k + j, j * m + c) for j in range(k))

    terms = [target(r, c) for r in range(n) for c in range(m)]

    # linear maps on matrices, given by the images of the nb basis bits
    def apply(img, masks):
        masks = np.asarray(masks, dtype=np.int64)
        out = np.zeros_like(masks)
        for b in range(nb):
            out ^= np.where((masks >> b) & 1, img[b], 0)
        return out

    def from_entry_map(f):
        # f(row, col) -> list of (row', col') with coefficient 1 (F2)
        img = []
        for row in range(R):
            for col in range(C):
                v = 0
                for (r2, c2) in f(row, col):
                    v ^= 1 << bit(r2, c2)
                img.append(v)
        return tuple(img)

    gl = []
    if k == 2:
        gl = [((1, 1), (0, 1)), ((0, 1), (1, 0))]
    elif k > 2:
        raise SystemExit("inner GL_k generators only for k <= 2")
    genmaps = []
    for B in gl:
        B = np.array(B) % 2
        Binv = np.round(np.linalg.inv(B)).astype(int) % 2
        # f'(X,Y) = f(XB, B^{-1}Y): M'[(r,l),(l',c)] = sum B[l,kk] M[(r,kk),(kk',c)] Binv[kk',l']
        def fm(row, col, B=B, Binv=Binv):
            r, kk = divmod(row, k)
            kk2, c = divmod(col, m)
            return [(r * k + l, l2 * m + c) for l in range(k) for l2 in range(k) if B[l, kk] and Binv[kk2, l2]]
        genmaps.append(from_entry_map(fm))
    for sigma in ([1, 0] + list(range(2, n)),) if n > 1 else ():
        genmaps.append(from_entry_map(lambda row, col, s=sigma: [(s[row // k] * k + row % k, col)]))
    if n > 2:
        cyc = list(range(1, n)) + [0]
        genmaps.append(from_entry_map(lambda row, col, s=cyc: [(s[row // k] * k + row % k, col)]))
    for sigma in ([1, 0] + list(range(2, m)),) if m > 1 else ():
        genmaps.append(from_entry_map(lambda row, col, s=sigma: [(row, (col // m) * m + s[col % m])]))
    if n == m:
        # f'(X,Y) = f(Y^T, X^T): M'[(c,k'),(kk,r)] = M[(r,kk),(k',c)]
        def tr(row, col):
            r, kk = divmod(row, k)
            k2, c = divmod(col, m)
            return [(c * k + k2, kk * m + r)]
        genmaps.append(from_entry_map(tr))

    def compose(a, b):  # a after b
        return tuple(int(x) for x in apply(a, np.array(b)))

    ident = tuple(1 << b for b in range(nb))
    group = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for g in frontier:
            for h in genmaps:
                gh = compose(h, g)
                if gh not in group:
                    group.add(gh)
                    nxt.append(gh)
        frontier = nxt
    group = sorted(group)
    tset = set(terms)
    for g in group:
        assert set(int(x) for x in apply(g, terms)) == tset
        assert set(int(x) for x in apply(g, gens)) == set(int(x) for x in gens)
    t_0 = terms[0]
    stab = [g for g in group if int(apply(g, [t_0])[0]) == t_0]
    print(f"<{n},{k},{m}> nodes {N} rank-1 steps {ng} |G| {len(group)} |Stab| {len(stab)}", flush=True)

    nodes = np.arange(N, dtype=np.int64)
    gpos = np.full(1 << nb, -1, dtype=np.int64) if nb <= 24 else None
    gpos[gens] = np.arange(ng)
    if detour_max is None:
        keep_node = np.ones(N, dtype=bool)
        arc_v = np.repeat(nodes, ng)
        arc_r = np.tile(np.arange(ng, dtype=np.int64), N)
        aid = None
    else:
        # nodes within detour K of a geodesic 0 -> terminal; G-invariant, so the reduction stays exact for the subgraph
        def rank_all(masks):
            rk = np.zeros(len(masks), dtype=np.int64)
            rows = [(masks >> (row * C)) & ((1 << C) - 1) for row in range(R)]
            basis = []
            for r_ in rows:
                x = r_.copy()
                for b_ in basis:
                    x = np.minimum(x, x ^ b_)
                rk += x != 0
                basis.append(x)
            return rk
        rk = rank_all(nodes)
        trank = int(rank_all(np.array([terms[0]]))[0])
        detour = rk + np.min(np.stack([rank_all(nodes ^ t) for t in terms]), axis=0) - trank
        keep_node = detour <= detour_max
        kv = np.flatnonzero(keep_node)
        arc_v = np.repeat(kv, ng)
        arc_r = np.tile(np.arange(ng, dtype=np.int64), len(kv))
        ok = keep_node[arc_v ^ gens[arc_r]]
        arc_v, arc_r = arc_v[ok], arc_r[ok]
        aid = arc_v * ng + arc_r
    A = len(arc_v)

    def arc_orbits(elems):
        rep = np.arange(A, dtype=np.int64)
        for g in elems:
            gv = apply(g, nodes)
            gr = gpos[apply(g, gens)]
            img = gv[arc_v] * ng + gr[arc_r]
            if aid is not None:
                img = np.searchsorted(aid, img)
            np.minimum(rep, img, out=rep)
        return rep

    def node_orbits(elems):
        rep = nodes.copy()
        for g in elems:
            np.minimum(rep, apply(g, nodes), out=rep)
        return rep

    def pos(ids):
        return ids if aid is None else np.searchsorted(aid, ids)

    repG = arc_orbits(group)
    if mode == "ucr":
        rev = pos((arc_v ^ gens[arc_r]) * ng + arc_r)
        # merge the orbit of a with the orbit of reverse(a): label = min of the two orbit reps
        repG = np.minimum(repG, repG[rev])
        # the merged relation is an equivalence: a and rev(a) share a class, orbits are closed under rev up to G
        repG = np.minimum(repG, repG[rev])
    repS = arc_orbits(stab)
    uG, Ocls, Osize = np.unique(repG, return_inverse=True, return_counts=True)
    uS, Scls = np.unique(repS, return_inverse=True)
    nO, nS = len(uG), len(uS)
    O_of_S = np.zeros(nS, dtype=np.int64)
    O_of_S[Scls] = Ocls
    print(f"arcs {A}  G-classes {nO}  Stab-orbits {nS}  ({time.time() - t0:.1f}s)", flush=True)

    nrep = node_orbits(stab)
    reps = np.unique(nrep[keep_node])
    rowid = np.full(N, -1, dtype=np.int64)
    rowid[reps] = np.arange(len(reps))
    # conservation at each representative node v: sum_in f - sum_out f = [v = t0] - [v = 0]
    rr = np.repeat(reps, ng)
    gg = np.tile(np.arange(ng, dtype=np.int64), len(reps))
    ex = keep_node[rr ^ gens[gg]]
    rr, gg = rr[ex], gg[ex]
    out_arc = pos(rr * ng + gg)
    in_arc = pos((rr ^ gens[gg]) * ng + gg)
    rows = np.concatenate([rowid[rr], rowid[rr]])
    cols = np.concatenate([Scls[in_arc], Scls[out_arc]])
    vals = np.concatenate([np.ones(len(rr)), -np.ones(len(rr))])
    Aeq = sp.coo_matrix((vals, (rows, cols)), shape=(len(reps), nS + nO)).tocsr()
    beq = np.zeros(len(reps))
    beq[rowid[t_0]] += 1
    beq[rowid[0]] -= 1
    # coupling f_s - x_{O(s)} <= 0
    Aub = sp.coo_matrix((np.concatenate([np.ones(nS), -np.ones(nS)]),
                         (np.concatenate([np.arange(nS), np.arange(nS)]),
                          np.concatenate([np.arange(nS), nS + O_of_S]))), shape=(nS, nS + nO)).tocsr()
    bub = np.zeros(nS)
    cost = np.concatenate([np.zeros(nS), Osize.astype(float) * (0.5 if mode == "ucr" else 1.0)])
    print(f"LP: {nS + nO} variables, {len(reps)} equalities, {nS} inequalities ({time.time() - t0:.1f}s)", flush=True)
    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    print(res.status, res.message)
    print(f"{mode.upper()} value {res.fun:.9f}  ({time.time() - t0:.1f}s)", flush=True)
    # support of the optimal x: how many arc classes are used, and with which values
    x = res.x[nS:]
    used = x > 1e-9
    vals_used = np.round(x[used], 6)
    uv, cnt = np.unique(vals_used, return_counts=True)
    print("x values on used classes (value: number of classes):",
          dict(zip(uv.tolist(), cnt.tolist())), " arcs used:", int(Osize[used].sum()))
    dump = os.environ.get("BCR_SUPPORT")
    if dump:
        xa = x[Ocls]
        sel = xa > 1e-9
        tails, heads = arc_v[sel], arc_v[sel] ^ gens[arc_r[sel]]
        with open(dump, "w") as fh:
            json.dump({"instance": [n, k, m], "mode": mode, "detour_max": detour_max, "value": res.fun,
                       "terminals": [int(t) for t in terms],
                       "arcs": [[int(a), int(b), float(v)] for a, b, v in zip(tails, heads, xa[sel])]}, fh)
        print("support written to", dump)


if __name__ == "__main__":
    main()
