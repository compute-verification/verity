"""Schemes whose output sets are laminar after deleting a few products, and the schedules they give.

If the output sets of the kept products are laminar, the kept products form a tree and each deleted product (or
a group of deleted products with one output set, chained) is its own branch from 0, merged into each target it
feeds: |S| merges of charge at most the group's rank. `walk_min` anneals over flips, reductions and plus
transitions from the standard algorithm to minimise that charge minus twice the saving; a negative value is a
schedule that beats Question 6.3's half charge, and `schedule` builds it as a `sched.Graph` so that
`sched.weight` recomputes it exactly.

    python crossing.py p q r steps [seed]
"""

import math
import random
import sys

from flip import _reduce, is_scheme, standard
from sched import Graph, Scheme, weight


def crosses(a, b):
    return a & b and a & ~b and b & ~a


def min_deletion(ws, cap):
    """Minimum of sum over deleted output sets S (distinct) of |S| * multiplicity, making the rest laminar.
    Returns (charge, deleted set of distinct masks), or (None, None) if every deletion costs >= cap."""
    mult = {}
    for w in ws:
        mult[w] = mult.get(w, 0) + 1
    sets = sorted(mult)
    cost = {s: s.bit_count() * mult[s] for s in sets}
    best = [cap, None]

    def rec(alive, deleted, acc):
        if acc >= best[0]:
            return
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                if crosses(a, b):
                    for d in (a, b):
                        rec([x for x in alive if x != d], deleted | {d}, acc + cost[d])
                    return
        best[0], best[1] = acc, deleted

    rec(sets, frozenset(), 0)
    return (best[0], best[1]) if best[1] is not None else (None, None)


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


def walk_min(p, q, r, steps, rng, t0=2.0, t1=0.2, plus=0.01, start=None):
    terms = list(start) if start else standard(p, q, r)
    cur = min_deletion([t[2] for t in terms], 10 ** 6)[0] - 2 * (p * q * r - len(terms))
    best = (10 ** 6, None)
    visits = {}
    for step in range(steps):
        T = t0 * (t1 / t0) ** (step / steps)
        new = list(terms)
        pairs = flip_pairs(new)
        if rng.random() < plus or not pairs:
            x, y = rng.sample(range(len(new)), 2)
            (a, b, c), a2 = new[x], new[y][0]
            if a == a2:
                continue
            new[x] = (a ^ a2, b, c)
            new.append((a2, b, c))
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
        cap = cur + 2 * s_new + int(10 * T) + 1
        c, _ = min_deletion([t[2] for t in new], cap)
        if c is None:
            continue
        c -= 2 * s_new
        if c <= cur or rng.random() < math.exp((cur - c) / T):
            terms, cur = new, c
            visits[s_new] = visits.get(s_new, 0) + 1
            if s_new > 0 and (c, -s_new) < (best[0], -(p * q * r - len(best[1] or ()))):
                best = (c, list(new))
    return best + (visits,)


def main():
    p, q, r, steps = (int(s) for s in sys.argv[1:5])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = random.Random(seed)
    val, terms, visits = walk_min(p, q, r, steps, rng)
    print(f"<{p},{q},{r}> seed={seed} accepted states by saving: {dict(sorted(visits.items()))}", flush=True)
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
