"""Low-merge schedules for a bilinear scheme over F2, and their Corollary 4 / Question 6.3 accounting.

A schedule computes every target with each product on exactly one arc u -> u + P_i, and merges N = A + B. Here it
is a graph G: each product i has a parent (the root 0, a product, or a merge node), each merge node two inputs
(products or merges), acyclic; then each target is the XOR of r_t nodes of G, by r_t - 1 final merges. Every DAG is
of this form (with r_t = 1), and its merge count is M + sum_t (r_t - 1). Over F2 a node's value is the set of
products on an odd number of root paths into it; the product nodes' values form a triangular basis, so r_t is
min over subsets Z of merge nodes of |Z| + |coords(X_t) + sum_Z coords(merge)|.

`anneal` searches G by simulated annealing; `weight` builds the merges in order and charges each
min(rank A, rank B), with ranks of the bilinear forms over F2, choosing each target's final XOR order to minimise.
"""

import itertools
import math
import random
import sys

from flip import is_scheme, walk


def f2rank(rows):
    basis = []
    for x in rows:
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    return len(basis)


class Scheme:
    def __init__(self, terms, p, q, r):
        self.terms = terms
        self.p, self.q, self.r = p, q, r
        self.m = len(terms)
        self.k = p * r
        sets = [sum(1 << i for i, t in enumerate(terms) if t[2] >> c & 1) for c in range(self.k)]
        self.X = [x for x in sets if x]

    def form(self, prods):
        """Rows (over A's entries) of the bilinear form sum_{i in prods} u_i v_i^T, as bitmasks over B's entries."""
        rows = [0] * (self.p * self.q)
        for i in _bits(prods):
            u, v, _ = self.terms[i]
            for a in _bits(u):
                rows[a] ^= v
        return rows

    def rank(self, prods):
        cache = self.__dict__.setdefault("_rank", {})
        if prods not in cache:
            cache[prods] = f2rank(self.form(prods))
        return cache[prods]


def _bits(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


class Graph:
    """par[i] for products i < m: -1 (root) or a node; merges[j] = (a, b), node m + j."""

    def __init__(self, sch, par=None, merges=None):
        self.s = sch
        self.par = list(par) if par is not None else [-1] * sch.m
        self.merges = [tuple(x) for x in merges] if merges else []

    def copy(self):
        return Graph(self.s, self.par, self.merges)

    def nodes(self):
        return self.s.m + len(self.merges)

    def preds(self, v):
        m = self.s.m
        if v < m:
            return [] if self.par[v] < 0 else [self.par[v]]
        return list(self.merges[v - m])

    def topo(self):
        n = self.nodes()
        order, state = [], [0] * n
        for v0 in range(n):
            if state[v0]:
                continue
            stack = [(v0, iter(self.preds(v0)))]
            state[v0] = 1
            while stack:
                v, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    state[v] = 2
                    order.append(v)
                elif state[nxt] == 1:
                    return None
                elif state[nxt] == 0:
                    state[nxt] = 1
                    stack.append((nxt, iter(self.preds(nxt))))
        return order

    def values(self, order):
        m = self.s.m
        val = [0] * self.nodes()
        for v in order:
            if v < m:
                val[v] = (1 << v) ^ (val[self.par[v]] if self.par[v] >= 0 else 0)
            else:
                a, b = self.merges[v - m]
                val[v] = val[a] ^ val[b]
        return val

    def cost(self):
        order = self.topo()
        if order is None:
            return None
        m = self.s.m
        val = self.values(order)
        prod_rev = [v for v in reversed(order) if v < m]

        def coords(x):
            c = 0
            for i in prod_rev:
                if x >> i & 1:
                    c |= 1 << i
                    x ^= val[i]
            assert x == 0
            return c

        g = [coords(val[m + j]) for j in range(len(self.merges))]
        gz = [0] * (1 << len(g))
        for z in range(1, len(gz)):
            low = z & -z
            gz[z] = gz[z ^ low] ^ g[low.bit_length() - 1]
        pz = [z.bit_count() for z in range(len(gz))]
        total = len(self.merges)
        reps = []
        for x in self.s.X:
            c = coords(x)
            best, bz = min((pz[z] + (c ^ gz[z]).bit_count(), z) for z in range(len(gz)))
            total += best - 1
            reps.append((bz, c ^ gz[bz]))
        return total, reps, val

    def wcost(self):
        """Merge weight with each final XOR chained greedily (largest rank first, then increasing rank)."""
        res = self.cost()
        if res is None:
            return None
        _, reps, val = res
        m, rank = self.s.m, self.s.rank
        w = sum(min(rank(val[a]), rank(val[b])) for a, b in self.merges)
        for z, c in reps:
            elems = sorted([val[m + j] for j in _bits(z)] + [val[i] for i in _bits(c)], key=rank)
            if len(elems) > 1:
                acc = elems.pop()
                for e in elems:
                    w += min(rank(acc), rank(e))
                    acc ^= e
        return w, reps, val

    def descendants(self, v):
        m = self.s.m
        kids = [[] for _ in range(self.nodes())]
        for i in range(m):
            if self.par[i] >= 0:
                kids[self.par[i]].append(i)
        for j, (a, b) in enumerate(self.merges):
            kids[a].append(m + j)
            kids[b].append(m + j)
        seen, stack = {v}, [v]
        while stack:
            for w in kids[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen


def mutate(G, rng, max_merges):
    H = G.copy()
    m = H.s.m
    move = rng.random()
    if move < 0.1 and len(H.merges) >= max_merges:
        return None
    if move < 0.1:
        a, b = rng.sample(range(H.nodes()), 2)
        H.merges.append((a, b))
    elif move < 0.7 or not H.merges:
        i = rng.randrange(m)
        bad = H.descendants(i)
        choices = [-1] + [v for v in range(H.nodes()) if v not in bad]
        H.par[i] = rng.choice(choices)
    elif move < 0.8:
        j = rng.randrange(len(H.merges))
        node = m + j
        a, _ = H.merges[j]
        H.par = [a if x == node else x for x in H.par]
        H.merges = [tuple(a if y == node else y for y in mg) for mg in H.merges]
        del H.merges[j]
        H.par = [x - 1 if x > node else x for x in H.par]
        H.merges = [tuple(y - 1 if y > node else y for y in mg) for mg in H.merges]
        if any(a == b for a, b in H.merges):
            return None
    elif H.merges:
        j = rng.randrange(len(H.merges))
        a, b = H.merges[j]
        bad = H.descendants(m + j)
        choices = [v for v in range(H.nodes()) if v not in bad]
        if rng.random() < 0.5:
            a = rng.choice(choices)
        else:
            b = rng.choice(choices)
        if a == b:
            return None
        H.merges[j] = (a, b)
    else:
        return None
    return H


def anneal(sch, rng, steps=20000, max_merges=8, t0=1.0, t1=0.05, start=None, objective="weight"):
    G = start.copy() if start else Graph(sch)
    f = Graph.wcost if objective == "weight" else Graph.cost
    cur = f(G)[0]
    best, bestG = cur, G.copy()
    for step in range(steps):
        T = t0 * (t1 / t0) ** (step / steps)
        H = mutate(G, rng, max_merges)
        if H is None:
            continue
        res = f(H)
        if res is None:
            continue
        c = res[0]
        if c <= cur or rng.random() < math.exp((cur - c) / T):
            G, cur = H, c
            if c < best:
                best, bestG = c, H.copy()
    return best, bestG


def weight(sch, G):
    """Merges in topological order; each target's final XOR in the order (over permutations, <= 5 terms) and
    chain that minimises its charge. Returns (merges, weight, sum_t rank T_t) and checks every target value."""
    total, reps, val = G.cost()
    m = sch.m
    w = 0
    for a, b in G.merges:
        w += min(sch.rank(val[a]), sch.rank(val[b]))
    count = len(G.merges)
    for t, (z, c) in enumerate(reps):
        elems = [val[m + j] for j in _bits(z)] + [val[i] for i in _bits(c)]
        acc_check = 0
        for e in elems:
            acc_check ^= e
        assert acc_check == sch.X[t]
        if len(elems) == 1:
            continue
        best = None
        perms = itertools.permutations(elems) if len(elems) <= 5 else [elems]
        for perm in perms:
            acc, cw = perm[0], 0
            for e in perm[1:]:
                cw += min(sch.rank(acc), sch.rank(e))
                acc ^= e
            best = cw if best is None else min(best, cw)
        w += best
        count += len(elems) - 1
    assert count == total
    sum_rank = sum(sch.rank(x) for x in sch.X)
    return count, w, sum_rank


def main():
    p, q, r, target = (int(s) for s in sys.argv[1:5])
    nschemes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 20000
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    restarts = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    rng = random.Random(seed)
    seen = set()
    for terms in walk(p, q, r, target, rng, every=5000):
        key = frozenset(terms)
        if key in seen:
            continue
        seen.add(key)
        assert is_scheme(terms, p, q, r)
        sch = Scheme(terms, p, q, r)
        runs = [anneal(sch, rng, steps=steps) for _ in range(restarts)]
        merges, w, sum_rank = min(weight(sch, G) for _, G in runs)
        G = min(runs, key=lambda bg: weight(sch, bg[1]))[1]
        saving = sum_rank - sch.m
        print(f"<{p},{q},{r}> m={sch.m} saving={saving} merges={merges} weight={w} "
              f"cor4_ok={sch.m >= sum_rank - w} half_ok={2 * sch.m >= 2 * sum_rank - w} "
              f"par={G.par} merges={G.merges} terms={sorted(terms)}", flush=True)
        if len(seen) >= nschemes:
            break


if __name__ == "__main__":
    main()
