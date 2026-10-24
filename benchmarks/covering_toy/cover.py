"""Coverings of a `toygraph.Graph` by `(s, k)`-inference blocks, and the solvers that find them.

The definitions are Lean's (`Security/Definitions/Nci/Covering.lean`): a block `S` is a set of gates with inputs
`in(S) ⊆ S`; every gate of `S \\ in(S)` has all its predecessors in `S`, and every source in `S` is an input. It is an
`(s, k)`-inference block when `|S| <= s` and `|in(S) \\ θ| <= k`; registered inputs count toward `s`, not `k`. A covering
is a family of blocks, overlaps and recomputation allowed, that together compute every non-source gate. Here a block is
`Block(computed, inputs)`, `S = computed ∪ inputs`; `s = None` means no gate budget.

Solvers:

- `greedy`: builds one block at a time by choosing its unregistered inputs, singly or in structured bundles (a token's
  data, a layer's activations or deltas at one token, a row of a step's weights), by new gates covered per input word,
  within `s`; the block computes what those inputs and `θ` determine and is still needed.
- `lns`: large-neighbourhood search over `greedy`: drop a few blocks, re-cover what only they covered, keep the result
  when it is no larger.
- `partition_topo`, `partition_bisect`: partitions (no recomputation), by a topological sweep and by recursive
  Kernighan-Lin bisection (a METIS-style partitioner), as baselines.
- `exact` (OR-tools CP-SAT): the minimum covering, proved optimal or bracketed by CP-SAT's bound.
- `block_capacity` (CP-SAT): the most gates of a target set one block can compute, and `min_inputs_for_slots`, the
  fewest unregistered inputs that complete products of `q` distinct weight slots (parameter-hardness's `c_w`).

`validate` checks a covering against the definitions and is what every solver's output goes through.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from toygraph import Graph


@dataclass(frozen=True)
class Block:
    computed: frozenset[int]
    inputs: frozenset[int]

    def size(self) -> int:
        return len(self.computed) + len(self.inputs)

    def unregistered(self, G: Graph) -> int:
        return sum(1 for g in self.inputs if not G.theta[g])


def validate(G: Graph, blocks: list[Block], k: int, s: int | None, targets: list[int] | None = None) -> None:
    """Raise unless `blocks` are `(s, k)`-blocks of `G` that compute every gate of `targets` (default: every non-source
    gate, which makes them an `(s, k)`-covering)."""
    covered: set[int] = set()
    for i, B in enumerate(blocks):
        assert not (B.computed & B.inputs), f"block {i}: a gate both computed and input"
        for g in B.computed:
            assert G.preds[g], f"block {i}: computes source {g}"
            for p in G.preds[g]:
                assert p in B.computed or p in B.inputs, f"block {i}: gate {g} reads {p} outside the block"
        assert B.unregistered(G) <= k, f"block {i}: {B.unregistered(G)} unregistered inputs > k = {k}"
        assert s is None or B.size() <= s, f"block {i}: size {B.size()} > s = {s}"
        covered |= B.computed
    missing = [g for g in (G.inner() if targets is None else targets) if g not in covered]
    assert not missing, f"{len(missing)} gates uncovered, e.g. {missing[:5]}"


def trim(G: Graph, B: Block) -> Block:
    """Drop inputs no computed gate reads."""
    used = {p for g in B.computed for p in G.preds[g] if p not in B.computed}
    return Block(B.computed, frozenset(B.inputs & used))


def drop_redundant(G: Graph, blocks: list[Block], targets: set[int] | None = None) -> list[Block]:
    """Remove blocks every one of whose target gates another block also computes, smallest first."""
    count: dict[int, int] = {}
    for B in blocks:
        for g in B.computed:
            count[g] = count.get(g, 0) + 1
    keep = []
    for B in sorted(blocks, key=lambda b: len(b.computed)):
        mine = B.computed if targets is None else B.computed & targets
        if all(count[g] > 1 for g in mine):
            for g in B.computed:
                count[g] -= 1
        else:
            keep.append(B)
    return keep


class Closure:
    """Vectorized closure and need sweeps over `G` (gates in topological order, at most two predecessors)."""

    def __init__(self, G: Graph):
        self.G = G
        N = G.N
        self.N = N
        self.src = np.array([not p for p in G.preds])
        self.theta = np.array(G.theta)
        self.p0 = np.array([p[0] if p else g for g, p in enumerate(G.preds)])
        self.p1 = np.array([p[-1] if p else g for g, p in enumerate(G.preds)])
        assert all(len(p) <= 2 for p in G.preds)
        level = np.zeros(N, dtype=int)
        for g in range(N):
            if G.preds[g]:
                level[g] = 1 + max(level[p] for p in G.preds[g])
        self.levels = [np.flatnonzero(level == L) for L in range(1, level.max() + 1)]

    def computable(self, Z: np.ndarray) -> np.ndarray:
        """`Z`: (C, N) or (N,) bool inputs. Returns what `Z ∪ θ` determine, inputs included."""
        A = Z | self.theta
        for idx in self.levels:
            A[..., idx] |= A[..., self.p0[idx]] & A[..., self.p1[idx]]
        return A

    def need(self, A: np.ndarray, Z: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The computed gates a block with availability `A` and inputs `Z` needs to compute every gate of `U` it can,
        and the inputs those read. Batched over a leading axis like `computable`."""
        comp = A & ~Z & ~self.src
        need = comp & U
        batched = need.ndim == 2
        for idx in reversed(self.levels):
            sel = need[..., idx]
            if not sel.any():
                continue
            for p in (self.p0, self.p1):
                pi = p[idx]
                if batched:
                    np.logical_or.at(need.T, pi, (sel & comp[:, pi]).T)
                else:
                    np.logical_or.at(need, pi, sel & comp[pi])
        reads = np.zeros_like(need)
        for p in (self.p0, self.p1):
            if batched:
                rows, cols = np.nonzero(need)
                reads[rows, p[cols]] = True
            else:
                reads[p[np.flatnonzero(need)]] = True
        reads &= ~need
        return need, reads


def _bundles(G: Graph) -> list[list[int]]:
    """Structured candidate input sets: what natural attacks read."""
    groups: dict[tuple, list[int]] = {}
    for g in range(G.N):
        tg = G.tag[g]
        tok = tg.get("token")
        if G.op[g] == "data":
            groups.setdefault(("data", tok), []).append(g)
            groups.setdefault(("data", tok, tg["kind"]), []).append(g)
            groups.setdefault(("stepdata", tg["step"]), []).append(g)
        elif tg.get("act"):
            groups.setdefault(("act", tok, tg["layer"]), []).append(g)
        elif "value" in tg:
            groups.setdefault(("pre", tok, tg["layer"]), []).append(g)
        elif G.op[g] == "drelu" or (G.op[g] == "sub" and tg.get("phase") == "bwd"):
            groups.setdefault(("delta", tok, tg["layer"]), []).append(g)
        elif tg.get("kind") == "w" and G.preds[g]:
            l, r, c = tg["slot"]
            groups.setdefault(("wrow", tg["step"], l, r), []).append(g)
            groups.setdefault(("wcol", tg["step"], l, c), []).append(g)
            groups.setdefault(("wlayer", tg["step"], l), []).append(g)
            groups.setdefault(("wall", tg["step"]), []).append(g)
    return [v for v in groups.values() if len(v) > 1]


def _truncate(G: Graph, comp: set[int], targets: list[int], budget: int, theta_ok) -> tuple[set[int], set[int]]:
    """Choose targets in order, each with the computed ancestors it needs, while computed + inputs fit `budget`."""
    chosen: set[int] = set()
    ins: set[int] = set()
    for t in targets:
        if t in chosen:
            continue
        stack, add = [t], set()
        while stack:
            g = stack.pop()
            if g in chosen or g in add:
                continue
            add.add(g)
            for p in G.preds[g]:
                if p in comp and p not in chosen and p not in add:
                    stack.append(p)
        new_ins = {p for g in add for p in G.preds[g] if p not in comp and p not in ins}
        if len(chosen) + len(add) + len(ins) + len(new_ins) <= budget:
            chosen |= add
            ins |= new_ins
    return chosen, ins


def greedy(G: Graph, k: int, s: int | None, *, U: set[int] | None = None, rng: random.Random | None = None,
           noise: float = 0.0, cl: Closure | None = None) -> list[Block]:
    """Cover `U` (default: every non-source gate) block by block; see the module docstring."""
    cl = cl or Closure(G)
    rng = rng or random.Random(0)
    N = G.N
    Umask = np.zeros(N, dtype=bool)
    Umask[list(G.inner() if U is None else U)] = True
    nontheta = ~cl.theta
    has_succ = np.zeros(N, dtype=bool)
    for ps in G.preds:
        for p in ps:
            has_succ[p] = True
    bundles = [np.array(b) for b in _bundles(G)]
    blocks: list[Block] = []
    while Umask.any():
        Z = np.zeros(N, dtype=bool)
        A = cl.computable(Z.copy())
        while True:
            left = k - int((Z & nontheta).sum())
            if left <= 0:
                break
            base_need, _ = cl.need(A, Z, Umask)
            base_gain = int((base_need & Umask).sum())
            cands: list[np.ndarray] = []
            single = np.flatnonzero(nontheta & ~A & has_succ)
            cands.extend(np.array([g]) for g in single)
            for b in bundles:
                miss = b[~A[b]]
                if 1 < len(miss) <= left:
                    cands.append(miss)
            first = np.flatnonzero(Umask & ~A)
            if len(first):
                g0 = int(first[0])
                miss = np.array([p for p in set(G.preds[g0]) if not A[p] and not G.theta[p]])
                if 0 < len(miss) <= left:
                    cands.append(miss)
            if not cands:
                break
            C = len(cands)
            Zc = np.repeat(Z[None, :], C, axis=0)
            for i, c in enumerate(cands):
                Zc[i, c] = True
            Ac = cl.computable(Zc)
            needc, readsc = cl.need(Ac, Zc, Umask)
            gain = (needc & Umask).sum(1) - base_gain
            size = needc.sum(1) + readsc.sum(1)
            if s is not None:
                eff = np.where(size <= s, gain, gain * s / np.maximum(size, 1) * 0.5)
            else:
                eff = gain.astype(float)
            cost = np.array([len(c) for c in cands], dtype=float)
            score = eff / cost
            if noise:
                score = score * (1 + noise * np.array([rng.random() for _ in range(C)]))
            best = int(np.argmax(score))
            if eff[best] <= 0:
                break
            Z = Zc[best]
            A = Ac[best]
            if s is not None and size[best] >= s:
                break
        need, reads = cl.need(A, Z, Umask)
        comp_set = set(np.flatnonzero(A & ~Z & ~cl.src).tolist())
        if s is not None and need.sum() + reads.sum() > s:
            targets = [int(g) for g in np.flatnonzero(need & Umask)]
            chosen, ins = _truncate(G, comp_set, targets, s, None)
        else:
            chosen = set(np.flatnonzero(need).tolist())
            ins = set(np.flatnonzero(reads).tolist())
        if not (Umask[list(chosen)].any() if chosen else False):
            g0 = int(np.flatnonzero(Umask)[0])
            chosen, ins = {g0}, set(G.preds[g0])
            if sum(1 for p in ins if not G.theta[p]) > k or (s is not None and len(ins) + 1 > s):
                raise RuntimeError(f"greedy: no block covers gate {g0} ({G.op[g0]}, {G.tag[g0]}) at k={k}, s={s}")
        B = trim(G, Block(frozenset(chosen), frozenset(ins)))
        blocks.append(B)
        Umask[list(B.computed)] = False
    return blocks


def lns(G: Graph, k: int, s: int | None, blocks: list[Block], *, iters: int = 200, drop: int = 3, seed: int = 0,
        noise: float = 0.3, targets: list[int] | None = None) -> list[Block]:
    """Large-neighbourhood search: remove `drop` blocks, re-cover greedily, keep if no larger."""
    rng = random.Random(seed)
    cl = Closure(G)
    inner = set(G.inner() if targets is None else targets)
    best = drop_redundant(G, list(blocks), inner)
    for _ in range(iters):
        if len(best) <= 1:
            break
        r = min(drop, len(best) - 1)
        if rng.random() < 0.5:
            out = set(rng.sample(range(len(best)), r))
        else:
            i = rng.randrange(len(best))
            ov = sorted(range(len(best)), key=lambda j: -len(best[i].computed & best[j].computed) - rng.random())
            out = set(ov[:r]) | {i}
        rest = [B for j, B in enumerate(best) if j not in out]
        cov = set().union(*(B.computed for B in rest)) if rest else set()
        U = inner - cov
        try:
            new = greedy(G, k, s, U=U, rng=rng, noise=noise, cl=cl)
        except RuntimeError:
            continue
        cand = drop_redundant(G, rest + new, inner)
        if len(cand) <= len(best):
            best = cand
    return best


def partition_topo(G: Graph, k: int, s: int | None, order: list[int] | None = None) -> list[Block]:
    """A partition by a sweep in topological order: a gate joins the current part while the part stays a block."""
    order = order or G.inner()
    parts: list[Block] = []
    cur: set[int] = set()
    ins: set[int] = set()

    def flush():
        if cur:
            parts.append(Block(frozenset(cur), frozenset(ins)))

    for g in order:
        new_ins = {p for p in G.preds[g] if p not in cur and p not in ins}
        unreg = sum(1 for p in ins | new_ins if not G.theta[p])
        size = len(cur) + 1 + len(ins | new_ins)
        if cur and (unreg > k or (s is not None and size > s)):
            flush()
            cur, ins = set(), set()
            new_ins = set(G.preds[g])
        cur.add(g)
        ins |= new_ins
        ins.discard(g)
    flush()
    return [trim(G, B) for B in parts]


def partition_bisect(G: Graph, k: int, s: int | None, seed: int = 0) -> list[Block]:
    """A partition by recursive Kernighan-Lin bisection of the inner gates' undirected graph, until every part is a
    block; then adjacent parts are merged while the merge is a block."""
    import networkx as nx

    inner = G.inner()
    H = nx.Graph()
    H.add_nodes_from(inner)
    inset = set(inner)
    for g in inner:
        for p in G.preds[g]:
            if p in inset:
                H.add_edge(g, p)

    def block_of(part: set[int]) -> Block:
        ins = {p for g in part for p in G.preds[g] if p not in part}
        return Block(frozenset(part), frozenset(ins))

    def ok(B: Block) -> bool:
        return B.unregistered(G) <= k and (s is None or B.size() <= s)

    out: list[Block] = []
    stack = [set(inner)]
    while stack:
        part = stack.pop()
        B = block_of(part)
        if ok(B) or len(part) == 1:
            out.append(B)
            continue
        a, b = nx.algorithms.community.kernighan_lin_bisection(H.subgraph(part), seed=seed)
        stack += [set(a), set(b)]
    merged = True
    while merged:
        merged = False
        out.sort(key=lambda B: min(B.computed))
        for i in range(len(out) - 1):
            M = block_of(set(out[i].computed | out[i + 1].computed))
            if ok(M):
                out[i : i + 2] = [M]
                merged = True
                break
    return [trim(G, B) for B in out]


def best_heuristic(G: Graph, k: int, s: int | None, *, restarts: int = 4, iters: int = 150, seed: int = 0,
                   targets: list[int] | None = None) -> dict:
    """Every heuristic at `(k, s)`, each output validated; the best covering (of `targets`, default every non-source
    gate) and each method's count. Partitions only for a whole covering."""
    res: dict = {}
    cands: list[tuple[str, list[Block]]] = []
    tset = None if targets is None else set(targets)
    parts = () if targets is not None else (("partition_topo", lambda: partition_topo(G, k, s)),
                                            ("partition_bisect", lambda: partition_bisect(G, k, s, seed)))
    for name, fn in parts:
        try:
            P = fn()
            validate(G, P, k, s)
            res[name] = len(P)
            cands.append((name, P))
        except (AssertionError, RuntimeError) as e:
            res[name] = None
            res[name + "_error"] = str(e)[:200]
    cl = Closure(G)
    best = None
    for r in range(restarts):
        try:
            B = greedy(G, k, s, U=tset, rng=random.Random(seed + r), noise=0.0 if r == 0 else 0.3, cl=cl)
            B = drop_redundant(G, B, tset)
        except RuntimeError as e:
            res["greedy_error"] = str(e)[:200]
            break
        validate(G, B, k, s, targets)
        if best is None or len(B) < len(best):
            best = B
    if best is not None:
        res["greedy"] = len(best)
        best = lns(G, k, s, best, iters=iters, seed=seed, targets=targets)
        validate(G, best, k, s, targets)
        res["lns"] = len(best)
        cands.append(("lns", best))
    name, cover = min(cands, key=lambda c: len(c[1])) if cands else (None, None)
    res["best"] = len(cover) if cover else None
    res["best_method"] = name
    res["cover"] = cover
    return res


# --- exact models (OR-tools CP-SAT) --------------------------------------------------------------------------------


def _block_vars(model, G: Graph, k: int, s: int | None, tag: str, succs: list[list[int]]):
    """One block's variables and constraints: `x[g]` computed, `z[g]` input."""
    x, z = {}, {}
    for g in range(G.N):
        if G.preds[g]:
            x[g] = model.NewBoolVar(f"x{tag}_{g}")
        if succs[g]:
            z[g] = model.NewBoolVar(f"z{tag}_{g}")
        if g in x and g in z:
            model.AddBoolOr([x[g].Not(), z[g].Not()])
    for g in x:
        for p in set(G.preds[g]):
            if p in x:
                model.AddBoolOr([x[g].Not(), x[p], z[p]])
            else:
                model.AddImplication(x[g], z[p])
    for p, zp in z.items():
        model.AddBoolOr([zp.Not()] + [x[g] for g in succs[p]])
    model.Add(sum(z[g] for g in z if not G.theta[g]) <= k)
    if s is not None:
        model.Add(sum(x.values()) + sum(z.values()) <= s)
    return x, z


def _solve(model, time_limit: float, workers: int):
    from ortools.sat.python import cp_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(model)
    return solver, st, cp_model


def exact(G: Graph, k: int, s: int | None, *, n_max: int, lower: int = 1, hint: list[Block] | None = None,
          time_limit: float = 60.0, workers: int = 8) -> dict:
    """The minimum covering by CP-SAT over `n_max` block slots. Returns `n` (best found), `bound` (CP-SAT's proven lower
    bound, at least `lower`), `optimal`, and the covering."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    succs = G.succs()
    X, Zs, used = [], [], []
    for u in range(n_max):
        x, z = _block_vars(model, G, k, s, str(u), succs)
        X.append(x)
        Zs.append(z)
        used.append(model.NewBoolVar(f"u{u}"))
        for v in x.values():
            model.AddImplication(v, used[u])
        if u:
            model.AddImplication(used[u], used[u - 1])
    order = list(reversed(G.inner()))
    for i, g in enumerate(order):
        model.AddBoolOr([X[u][g] for u in range(min(i + 1, n_max))])
    model.Add(sum(used) >= lower)
    model.Minimize(sum(used))
    if hint:
        _hint(model, G, X, Zs, used, hint, order)
    solver, st, _ = _solve(model, time_limit, workers)
    out = {"status": solver.StatusName(st), "bound": max(lower, math.ceil(solver.BestObjectiveBound() - 1e-6)),
           "wall": solver.WallTime()}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        blocks = []
        for u in range(n_max):
            if solver.Value(used[u]):
                comp = frozenset(g for g, v in X[u].items() if solver.Value(v))
                ins = frozenset(g for g, v in Zs[u].items() if solver.Value(v))
                if comp:
                    blocks.append(trim(G, Block(comp, ins)))
        validate(G, blocks, k, s)
        out.update(n=len(blocks), optimal=st == cp_model.OPTIMAL, cover=blocks)
    else:
        out.update(n=None, optimal=False, cover=None)
    return out


def _hint(model, G, X, Zs, used, hint: list[Block], order: list[int]) -> None:
    """Relabel `hint`'s blocks to meet the symmetry-breaking order, then hint every variable."""
    pos = {g: i for i, g in enumerate(order)}
    rem = list(hint)
    lab: list[Block] = []
    for g in order:
        if any(g in B.computed for B in lab):
            continue
        j = next(j for j, B in enumerate(rem) if g in B.computed)
        lab.append(rem.pop(j))
    if len(lab) > len(X):
        return
    for u in range(len(X)):
        B = lab[u] if u < len(lab) else Block(frozenset(), frozenset())
        model.AddHint(used[u], u < len(lab))
        for g, v in X[u].items():
            model.AddHint(v, g in B.computed)
        for g, v in Zs[u].items():
            model.AddHint(v, g in B.inputs)


def colgen(G: Graph, k: int, s: int | None, columns: list[Block], *, iters: int = 300, price_time: float = 20.0,
           ip_time: float = 120.0, workers: int = 8, tol: float = 1e-7, log=None,
           targets: list[int] | None = None, time_budget: float | None = None) -> dict:
    """Column generation for the minimum covering: the set-cover LP over every `(s, k)`-block, priced by CP-SAT (the
    most dual weight one block computes). `lb` is a proven lower bound on the optimum (the LP optimum when pricing
    closes, else the Farley bound `z / max price` of the last round); `n` and `cover` come from the integer program over
    the generated columns, solved by HiGHS."""
    from ortools.sat.python import cp_model
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix

    inner = G.inner() if targets is None else sorted(targets)
    row = {g: i for i, g in enumerate(inner)}
    seen: set[frozenset] = set()
    cols: list[Block] = []

    def add(B: Block) -> bool:
        if B.computed in seen:
            return False
        seen.add(B.computed)
        cols.append(B)
        return True

    for B in columns:
        add(B)
    model = cp_model.CpModel()
    x, z = _block_vars(model, G, k, s, "", G.succs())
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = price_time
    solver.parameters.num_workers = workers
    SCALE = 10**6
    lb, z_lp, closed, rounds = 0.0, None, False, 0

    def matrix():
        r, c = [], []
        for j, B in enumerate(cols):
            for g in B.computed:
                if g in row:
                    r.append(row[g])
                    c.append(j)
        return csc_matrix((np.ones(len(r)), (r, c)), shape=(len(inner), len(cols)))

    import time

    t0 = time.monotonic()
    for rounds in range(1, iters + 1):
        if time_budget is not None and time.monotonic() - t0 > time_budget:
            break
        A = matrix()
        lp = linprog(np.ones(len(cols)), A_ub=-A, b_ub=-np.ones(len(inner)), bounds=(0, None), method="highs")
        z_lp = lp.fun
        pi = np.maximum(-lp.ineqlin.marginals, 0.0)
        w = {g: int(round(pi[row[g]] * SCALE)) + 1 for g in inner}
        model.Maximize(sum(w[g] * x[g] for g in inner))
        st = solver.Solve(model)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        ub_int = solver.BestObjectiveBound()
        price_ub = (ub_int + 0.5 * len(inner)) / SCALE
        lb = max(lb, z_lp / max(price_ub, 1.0))
        comp = frozenset(g for g, v in x.items() if solver.Value(v))
        ins = frozenset(g for g, v in z.items() if solver.Value(v))
        price = sum(pi[row[g]] for g in comp if g in row)
        if log and rounds % 10 == 0:
            log(f"colgen round {rounds}: lp {z_lp:.3f} price {price:.3f} (ub {price_ub:.3f}) lb {lb:.3f} cols {len(cols)}")
        if st == cp_model.OPTIMAL and price <= 1 + tol:
            lb = max(lb, z_lp)
            closed = True
            break
        if not add(trim(G, Block(comp, ins))):
            break
    A = matrix()
    res = milp(np.ones(len(cols)), constraints=LinearConstraint(A, lb=np.ones(len(inner)), ub=np.inf),
               integrality=np.ones(len(cols)), bounds=Bounds(0, 1), options={"time_limit": ip_time})
    out = {"lp": z_lp, "lb": math.ceil(lb - 1e-6), "lb_real": lb, "lp_closed": closed, "rounds": rounds,
           "columns": len(cols)}
    if res.x is not None:
        cover = [cols[j] for j in np.flatnonzero(res.x > 0.5)]
        cover = drop_redundant(G, cover, set(inner))
        validate(G, cover, k, s, inner)
        out.update(n=len(cover), cover=cover)
    else:
        out.update(n=None, cover=None)
    return out


def block_capacity(G: Graph, k: int, s: int | None, targets: list[int], *, time_limit: float = 30.0,
                   workers: int = 8) -> dict:
    """The most gates of `targets` one `(s, k)`-block computes. `ub` is CP-SAT's proven upper bound (a valid cap for
    bounds), `n` the best found."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x, z = _block_vars(model, G, k, s, "", G.succs())
    model.Maximize(sum(x[g] for g in targets))
    solver, st, _ = _solve(model, time_limit, workers)
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    block = None
    if ok:
        B = trim(G, Block(frozenset(g for g, v in x.items() if solver.Value(v)),
                          frozenset(g for g, v in z.items() if solver.Value(v))))
        validate(G, [B], k, s, [])
        block = [sorted(B.computed), sorted(B.inputs)]
    return {"n": int(solver.ObjectiveValue()) if ok else None, "ub": math.floor(solver.BestObjectiveBound() + 1e-6),
            "optimal": st == cp_model.OPTIMAL, "status": solver.StatusName(st), "block": block}


def min_inputs_for_slots(G: Graph, s: int | None, q: int, *, time_limit: float = 30.0, workers: int = 8) -> dict:
    """The fewest unregistered inputs of one block of at most `s` gates that completes multiplication blocks of at
    least `q` distinct weight slots. `lb` is CP-SAT's proven lower bound."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x, z = _block_vars(model, G, G.N, s, "", G.succs())
    by_slot: dict[tuple, list[int]] = {}
    for (slot, _tok), g in G.mul.items():
        by_slot.setdefault(slot, []).append(g)
    ys = []
    for slot, gs in by_slot.items():
        y = model.NewBoolVar(f"y{slot}")
        model.AddBoolOr([y.Not()] + [x[g] for g in gs])
        ys.append(y)
    model.Add(sum(ys) >= q)
    model.Minimize(sum(z[g] for g in z if not G.theta[g]))
    solver, st, _ = _solve(model, time_limit, workers)
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {"n": int(solver.ObjectiveValue()) if ok else None, "lb": math.ceil(solver.BestObjectiveBound() - 1e-6),
            "optimal": st == cp_model.OPTIMAL, "status": solver.StatusName(st)}
