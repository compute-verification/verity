"""The cheapest (s, k)-covering of a toy graph, exactly, as a MILP (HiGHS through scipy).

The covering is the one `Nci.Covering.Covers` states: blocks are subcircuits that may overlap and recompute, every
computed node is computed by some block, and a block of at most `s` gates may read at most `k` bits besides theta. A wire
from another block is an input like any other, at its width. Variables, per block `u`: `x[v, u]` (computes op `v`),
`y[w, u]` (reads node `w` as an input), `z[u]` (block used).

    x[v, u] <= x[w, u] + y[w, u]       for every unregistered predecessor w of v (y alone if w is a source)
    sum_u x[v, u] >= 1                  for every op v
    sum_w bits(w) y[w, u] <= k
    sum_v gates(v) x[v, u] <= s z[u],   x[v, u] <= z[u],   z[u] >= z[u + 1]
    minimise sum_u z[u]

`local` lists groups of nodes that must be held by one block (computed or read there): the locality rule a split
variant is compared under.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_array

from toy import Graph


def min_blocks(g: Graph, k: int, s: int, n_max: int, time_limit: float = 120.0, local=None) -> dict:
    nodes = g.nodes
    ops = [i for i, nd in enumerate(nodes) if nd.kind == "op"]
    readable = sorted({p for i in ops for p in nodes[i].preds if nodes[p].kind != "reg"})
    nx, ny = len(ops), len(readable)
    xi = {v: j for j, v in enumerate(ops)}
    yi = {w: j for j, w in enumerate(readable)}
    per = nx + ny
    nvar = per * n_max + n_max

    def X(v, u):
        return u * per + xi[v]

    def Y(w, u):
        return u * per + nx + yi[w]

    def Z(u):
        return per * n_max + u

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def row(entries, lb, ub):
        nonlocal r
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)
        r += 1

    for u in range(n_max):
        for v in ops:
            for w in set(nodes[v].preds):
                if nodes[w].kind == "reg":
                    continue
                e = [(X(v, u), 1.0), (Y(w, u), -1.0)]
                if nodes[w].kind == "op":
                    e.append((X(w, u), -1.0))
                row(e, -np.inf, 0.0)
            row([(X(v, u), 1.0), (Z(u), -1.0)], -np.inf, 0.0)
        row([(Y(w, u), float(nodes[w].bits)) for w in readable], -np.inf, float(k))
        row([(X(v, u), float(nodes[v].gates)) for v in ops] + [(Z(u), -float(s))], -np.inf, 0.0)
        if u + 1 < n_max:
            row([(Z(u), 1.0), (Z(u + 1), -1.0)], 0.0, np.inf)
    for v in ops:
        row([(X(v, u), 1.0) for u in range(n_max)], 1.0, np.inf)
    hold_vars = 0
    if local:
        # h[q, u] = 1 if block u holds every node of group q; each group is held somewhere.
        nq = len(local)
        base = nvar
        nvar += nq * n_max
        for q, grp in enumerate(local):
            for u in range(n_max):
                for w in grp:
                    e = [(base + q * n_max + u, 1.0)]
                    if nodes[w].kind == "op":
                        e.append((X(w, u), -1.0))
                    if w in yi:
                        e.append((Y(w, u), -1.0))
                    row(e, -np.inf, 0.0)
            row([(base + q * n_max + u, 1.0) for u in range(n_max)], 1.0, np.inf)
        hold_vars = nq * n_max
    A = coo_array((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    c = np.zeros(nvar)
    c[per * n_max: per * n_max + n_max] = 1.0
    res = milp(
        c,
        constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
        integrality=np.ones(nvar),
        bounds=Bounds(np.zeros(nvar), np.ones(nvar)),
        options={"time_limit": time_limit, "disp": False},
    )
    out = {"status": int(res.status), "message": res.message, "vars": nvar, "rows": r, "n_max": n_max,
           "hold_vars": hold_vars}
    if res.x is not None:
        out["blocks"] = int(round(res.fun))
        out["dual_bound"] = float(getattr(res, "mip_dual_bound", np.nan))
        out["optimal"] = res.status == 0
        z = res.x[per * n_max:per * n_max + n_max]
        used = [u for u in range(n_max) if z[u] > 0.5]
        out["block_inputs"] = [
            sorted(nodes[w].name for w in readable if res.x[Y(w, u)] > 0.5) for u in used
        ]
        out["block_bits"] = [int(sum(nodes[w].bits for w in readable if res.x[Y(w, u)] > 0.5)) for u in used]
        out["block_gates"] = [int(sum(nodes[v].gates for v in ops if res.x[X(v, u)] > 0.5)) for u in used]
    else:
        out["blocks"] = None
        out["optimal"] = False
    return out


class Closure:
    """What a block computes from its inputs: every op all of whose predecessors it reads, holds or recomputes."""

    def __init__(self, g: Graph):
        self.g = g
        self.ops = [i for i, nd in enumerate(g.nodes) if nd.kind == "op"]

    def __call__(self, inputs: set[int], s: int | None = None) -> set[int]:
        nodes = self.g.nodes
        have = set(inputs)
        comp: set[int] = set()
        gates = 0
        for v in self.ops:
            if all(nodes[p].kind == "reg" or p in have for p in nodes[v].preds):
                if s is not None and gates + nodes[v].gates > s:
                    continue
                gates += nodes[v].gates
                comp.add(v)
                have.add(v)
        return comp


def _price(g: Graph, pi: dict[int, float], mu: list[float], local, k: int, s: int, time_limit: float):
    """One block maximising sum pi_v over the ops it computes plus mu_q over the groups it holds."""
    nodes = g.nodes
    ops = [i for i, nd in enumerate(nodes) if nd.kind == "op"]
    readable = sorted({p for i in ops for p in nodes[i].preds if nodes[p].kind != "reg"}
                      | ({w for grp in local for w in grp} if local else set()))
    xi = {v: j for j, v in enumerate(ops)}
    yi = {w: len(ops) + j for j, w in enumerate(readable)}
    nq = len(local) if local else 0
    hb = len(ops) + len(readable)
    nvar = hb + nq
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def row(entries, lb, ub):
        nonlocal r
        for c_, v_ in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v_)
        lo.append(lb)
        hi.append(ub)
        r += 1

    for v in ops:
        for w in set(nodes[v].preds):
            if nodes[w].kind == "reg":
                continue
            e = [(xi[v], 1.0), (yi[w], -1.0)]
            if nodes[w].kind == "op":
                e.append((xi[w], -1.0))
            row(e, -np.inf, 0.0)
    row([(yi[w], float(nodes[w].bits)) for w in readable], -np.inf, float(k))
    row([(xi[v], float(nodes[v].gates)) for v in ops], -np.inf, float(s))
    for q in range(nq):
        for w in local[q]:
            e = [(hb + q, 1.0)]
            if nodes[w].kind == "op":
                e.append((xi[w], -1.0))
            e.append((yi[w], -1.0))
            row(e, -np.inf, 0.0)
    c = np.zeros(nvar)
    for v in ops:
        c[xi[v]] = -pi.get(v, 0.0)
    for w in readable:
        c[yi[w]] = 1e-6 * nodes[w].bits  # prefer fewer input bits among equal prices
    for q in range(nq):
        c[hb + q] = -mu[q]
    A = coo_array((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, np.array(lo), np.array(hi)), integrality=np.ones(nvar),
               bounds=Bounds(np.zeros(nvar), np.ones(nvar)), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None, 0.0, False
    inputs = {w for w in readable if res.x[yi[w]] > 0.5}
    return inputs, -float(res.fun), res.status == 0


def colgen(g: Graph, k: int, s: int = 10**9, local=None, max_iter: int = 400, price_limit: float = 30.0,
           ilp_limit: float = 120.0, budget: float | None = None, log=None, alpha_smooth: float = 0.7) -> dict:
    """Cheapest (s, k)-covering by column generation.

    Columns are blocks given by their inputs (each computes its closure under `s`). The set-cover LP over all blocks
    is solved by pricing with `_price`; its optimum, rounded up, is a lower bound on the number of blocks (`lp_bound`,
    valid once pricing finds no improving block). The integer set cover over the generated blocks is an upper bound
    (`blocks`). `optimal` when they meet.
    """
    nodes = g.nodes
    cl = Closure(g)
    ops = cl.ops
    oi = {v: j for j, v in enumerate(ops)}
    local = local or []
    cols: list[tuple[frozenset, frozenset, frozenset]] = []  # (inputs, computed, groups held)
    seen = set()

    def add_col(inputs):
        comp = cl(inputs, s if s < 10**9 else None)
        have = comp | set(inputs)
        held = frozenset(q for q, grp in enumerate(local) if all(w in have for w in grp))
        key = (frozenset(comp), held)
        if key in seen:
            return False
        seen.add(key)
        cols.append((frozenset(inputs), key[0], held))
        return True

    for v in ops:  # one op per block, reading its unregistered predecessors
        inp = {p for p in nodes[v].preds if nodes[p].kind != "reg"}
        if sum(nodes[p].bits for p in inp) <= k:
            add_col(inp)
    for grp in local:  # a block that holds a group by reading its pieces
        if sum(nodes[w].bits for w in grp) <= k:
            add_col(set(grp))
    lp_val = math.inf
    converged = False
    it = 0
    src_bits = sum(nd.bits for nd in nodes if nd.kind == "src")
    trivial = math.ceil(src_bits / k)  # every source bit is read by some block
    # Farley's bound: for any duals pi >= 0, pi / (max over all blocks of pi(block)) is dual feasible, so the full LP
    # is at least sum(pi) / max(pi(block), 1). Pricing is exact, so every exactly priced iteration gives one. Duals are
    # smoothed toward the best bound's (Wentges), which is what makes the bound close on these degenerate LPs.
    farley = 0.0
    center = None
    m_rows = len(ops) + len(local)
    t0 = time.monotonic()
    for it in range(max_iter):
        if budget is not None and time.monotonic() - t0 > budget:
            break
        nc = len(cols)
        A = np.zeros((m_rows, nc))
        for j, (_, comp, held) in enumerate(cols):
            for v in comp:
                A[oi[v], j] = 1.0
            for q in held:
                A[len(ops) + q, j] = 1.0
        if (A.sum(axis=1) == 0).any():
            missing = [nodes[ops[i]].name for i in np.where(A[:len(ops)].sum(axis=1) == 0)[0]]
            raise ValueError(f"no single-op block covers {missing[:5]}: k too small")
        lp = linprog(np.ones(nc), A_ub=-A, b_ub=-np.ones(m_rows), bounds=(0, None), method="highs")
        lp_val = lp.fun
        if lp_val <= trivial + 1e-9 or math.ceil(farley - 1e-6) >= math.ceil(lp_val - 1e-6):
            converged = True
            break
        duals = np.maximum(-lp.ineqlin.marginals, 0.0)
        for alpha in ((alpha_smooth, 0.0) if center is not None else (0.0,)):
            d = duals if alpha == 0.0 else alpha * center + (1 - alpha) * duals
            pi = {ops[i]: float(d[i]) for i in range(len(ops))}
            mu = [float(d[len(ops) + q]) for q in range(len(local))]
            inputs, val, exact = _price(g, pi, mu, local, k, s, price_limit)
            if exact:
                bound = float(d.sum()) / max(val, 1.0)
                if bound > farley:
                    farley, center = bound, d / max(val, 1.0)
            if inputs is None:
                break
            comp = cl(inputs, s if s < 10**9 else None)
            gain = sum(duals[oi[v]] for v in comp) + sum(
                duals[len(ops) + q] for q, grp in enumerate(local) if all(w in comp | inputs for w in grp))
            if gain > 1.0 + 1e-7 and add_col(inputs):
                break
        if log is not None:
            log(f"it {it} lp {lp_val:.3f} price {val:.3f} farley {farley:.3f} cols {len(cols)} "
                f"t {time.monotonic() - t0:.1f}")
        if len(cols) == nc:  # no improving block at the current duals: the restricted LP is the full LP
            converged = exact
            if exact:
                farley = max(farley, lp_val)
            break
    nc = len(cols)
    A = np.zeros((len(ops) + len(local), nc))
    for j, (_, comp, held) in enumerate(cols):
        for v in comp:
            A[oi[v], j] = 1.0
        for q in held:
            A[len(ops) + q, j] = 1.0
    res = milp(np.ones(nc), constraints=LinearConstraint(A, np.ones(A.shape[0]), np.inf),
               integrality=np.ones(nc), bounds=Bounds(0, 1), options={"time_limit": ilp_limit, "disp": False})
    chosen = [j for j in range(nc) if res.x is not None and res.x[j] > 0.5]
    lb = max(trivial, math.ceil(farley - 1e-6))
    blocks = len(chosen) if res.x is not None else None
    return {
        "k": k, "s": s, "blocks": blocks, "lp": lp_val, "lp_bound": lb, "converged": converged, "trivial": trivial,
        "farley": farley,
        "optimal": blocks is not None and lb is not None and blocks == lb, "columns": nc, "iterations": it + 1,
        "block_inputs": [sorted(nodes[w].name for w in cols[j][0]) for j in chosen],
        "block_bits": [sum(nodes[w].bits for w in cols[j][0]) for j in chosen],
    }
