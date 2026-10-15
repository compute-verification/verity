"""Schemes whose output sets are laminar after deleting a few products, and the schedules they give.

If the output sets of the kept products are laminar, the kept products form a tree and each deleted product (or
a group of deleted products with one output set, chained) is its own branch from 0, merged into each target it
feeds: |S| merges of charge at most the group's rank. `walk_min` anneals over flips, reductions and plus
transitions from the standard algorithm to minimise that charge minus twice the saving; a negative value is a
schedule that beats Question 6.3's half charge, and `schedule` builds it as a `sched.Graph` so that
`sched.weight` recomputes it exactly.

    python crossing.py p q r steps [seed] [rank|-] [gamma]

With a rank, the walk starts from a flip-graph scheme of that rank and makes no plus transitions. The annealed
objective is charge - gamma * saving (default gamma = 2); a larger gamma pushes the walk to larger savings.
"""

import math
import random
import sys

from flip import _plus, _reduce, is_scheme, standard, walk
from sched import Graph, Scheme, weight


def crosses(a, b):
    return a & b and a & ~b and b & ~a


def min_deletion(ws, cap):
    """Minimum of sum over deleted output sets S (distinct) of |S| * multiplicity, making the rest laminar: a
    minimum-weight vertex cover of the crossing graph, as total weight minus a maximum-weight independent set.
    Returns (charge, deleted set of distinct masks), or (None, None) if the charge is >= cap."""
    mult = {}
    for w in ws:
        mult[w] = mult.get(w, 0) + 1
    sets = sorted(mult)
    n = len(sets)
    cost = [s.bit_count() * mult[s] for s in sets]
    nbr = [sum(1 << j for j in range(n) if crosses(sets[i], sets[j])) for i in range(n)]
    memo = {}

    def mwis(alive):
        if alive in memo:
            return memo[alive]
        isolated = 0
        rest = alive
        x = alive
        while x:
            low = x & -x
            i = low.bit_length() - 1
            if not nbr[i] & alive:
                isolated |= low
            x ^= low
        rest = alive & ~isolated
        base = sum(cost[i] for i in _idx(isolated))
        if not rest:
            res = (base, isolated)
        else:
            v = max(_idx(rest), key=lambda i: (nbr[i] & rest).bit_count())
            w_in, s_in = mwis(rest & ~nbr[v] & ~(1 << v))
            w_out, s_out = mwis(rest & ~(1 << v))
            if w_in + cost[v] >= w_out:
                res = (base + w_in + cost[v], isolated | s_in | (1 << v))
            else:
                res = (base + w_out, isolated | s_out)
        memo[alive] = res
        return res

    kept_w, kept = mwis((1 << n) - 1)
    charge = sum(cost) - kept_w
    if charge >= cap:
        return None, None
    return charge, frozenset(sets[i] for i in range(n) if not kept >> i & 1)


def _idx(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def schedule(sch, deleted):
    """The tree-plus-branches schedule for a deletion set (distinct output masks)."""
    m = sch.m
    w = [t[2] for t in sch.terms]
    keep = [i for i in range(m) if w[i] not in deleted]
    order = sorted(keep, key=lambda i: -w[i].bit_count())
    par = [-1] * m
    placed = []
    for i in order:
        cands = [j for j in placed if w[j] & w[i] == w[i]]
        if cands:
            par[i] = min(cands, key=lambda j: (w[j].bit_count(), -placed.index(j)))
        placed.append(i)
    for d in deleted:
        group = [i for i in range(m) if w[i] == d]
        for a, b in zip(group, group[1:]):
            par[b] = a
    return Graph(sch, par)


def flip_pairs(terms):
    out = []
    for pos in range(3):
        by = {}
        for i, t in enumerate(terms):
            by.setdefault(t[pos], []).append(i)
        for idx in by.values():
            out += [(x, y, pos) for x in idx for y in idx if x != y]
    return out


def walk_min(p, q, r, steps, rng, t0=2.0, t1=0.2, plus=0.01, start=None, gamma=2):
    """Anneal charge - gamma * saving. Returns the best state with saving > 0 by charge - 2 * saving, and, for each
    saving, the number of accepted states and the least charge among them."""
    terms = list(start) if start else standard(p, q, r)
    cur = min_deletion([t[2] for t in terms], 10 ** 6)[0] - gamma * (p * q * r - len(terms))
    best = (10 ** 6, None)
    visits = {}
    for step in range(steps):
        T = t0 * (t1 / t0) ** (step / steps)
        new = list(terms)
        pairs = flip_pairs(new)
        if rng.random() < plus or not pairs:
            before = len(new)
            _plus(new, rng)
            if len(new) == before:
                continue
        else:
            x, y, pos = rng.choice(pairs)
            p1, p2 = (pos + 1) % 3, (pos + 2) % 3
            if rng.random() < 0.5:
                p1, p2 = p2, p1
            tx, ty = list(new[x]), list(new[y])
            tx[p1] ^= ty[p1]
            ty[p2] ^= tx[p2]
            new[x], new[y] = tuple(tx), tuple(ty)
        _reduce(new)
        s_new = p * q * r - len(new)
        cap = cur + gamma * s_new + int(10 * T) + 1
        charge, _ = min_deletion([t[2] for t in new], cap)
        if charge is None:
            continue
        c = charge - gamma * s_new
        if c <= cur or rng.random() < math.exp((cur - c) / T):
            terms, cur = new, c
            n_s, low = visits.get(s_new, (0, charge))
            visits[s_new] = (n_s + 1, min(low, charge))
            v = charge - 2 * s_new
            if s_new > 0 and (v, -s_new) < (best[0], -(p * q * r - len(best[1] or ()))):
                best = (v, list(new))
    return best + (visits,)


def main():
    p, q, r, steps = (int(s) for s in sys.argv[1:5])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    target = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    gamma = float(sys.argv[7]) if len(sys.argv) > 7 else 2
    rng = random.Random(seed)
    if target is None:
        val, terms, visits = walk_min(p, q, r, steps, rng, gamma=gamma)
    else:
        start = next(walk(p, q, r, target, rng, every=1))
        val, terms, visits = walk_min(p, q, r, steps, rng, plus=0.0, start=start, gamma=gamma)
    print(f"<{p},{q},{r}> seed={seed} gamma={gamma} saving: (accepted states, least charge) "
          f"{dict(sorted(visits.items()))}", flush=True)
    if terms is None:
        return
    assert is_scheme(terms, p, q, r)
    sch = Scheme(terms, p, q, r)
    charge, deleted = min_deletion([t[2] for t in terms], 10 ** 6)
    G = schedule(sch, deleted)
    merges, w, sum_rank = weight(sch, G)
    saving = sum_rank - sch.m
    print(f"<{p},{q},{r}> seed={seed} m={sch.m} saving={saving} deletion_charge={charge} merges={merges} "
          f"weight={w} weight-2s={w - 2 * saving} deleted={sorted(deleted)} par={G.par} terms={sorted(terms)}",
          flush=True)


if __name__ == "__main__":
    main()
