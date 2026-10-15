"""Flip-graph random walks (Kauers–Moosbauer) for <p,q,r> over F2.

A scheme is a list of rank-1 terms (u, v, w) of bitmasks: u over A's p*q entries (index i*q+j), v over B's q*r
entries (j*r+k), w over C's p*r cells (i*r+k). w is the term's output set. `walk` starts from the standard
algorithm, flips, takes every reduction it meets, and after `plateau` flips without one makes a plus transition.
It yields each scheme it reaches at rank <= `target`, every `every` flips.

    python flip.py p q r target [count] [seed]
"""

import random
import sys


def standard(p, q, r):
    return [(1 << (i * q + j), 1 << (j * r + k), 1 << (i * r + k))
            for i in range(p) for j in range(q) for k in range(r)]


def tensor(terms):
    out = {}
    for u, v, w in terms:
        for a in _bits(u):
            for b in _bits(v):
                for c in _bits(w):
                    out[(a, b, c)] = out.get((a, b, c), 0) ^ 1
    return {key for key, bit in out.items() if bit}


def _bits(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def is_scheme(terms, p, q, r):
    return tensor(terms) == tensor(standard(p, q, r))


def _reduce(terms):
    """Merge any two terms that agree in two factors, and drop zero terms, until none do."""
    changed = True
    while changed:
        changed = False
        seen = {}
        for idx, t in enumerate(terms):
            if 0 in t:
                terms.pop(idx)
                changed = True
                break
            for pos in range(3):
                key = (pos, t[(pos + 1) % 3], t[(pos + 2) % 3])
                if key in seen:
                    j = seen[key]
                    s = list(terms[j])
                    s[pos] ^= t[pos]
                    terms[j] = tuple(s)
                    terms.pop(idx)
                    changed = True
                    break
                seen[key] = idx
            if changed:
                break
    return terms


def walk(p, q, r, target, rng, plateau=20000, every=500, max_flips=10 ** 8):
    terms = standard(p, q, r)
    since = 0
    flips = 0
    while flips < max_flips:
        n = len(terms)
        x = rng.randrange(n)
        pos = rng.randrange(3)
        partners = [y for y in range(n) if y != x and terms[y][pos] == terms[x][pos]]
        if not partners:
            continue
        y = rng.choice(partners)
        p1, p2 = (pos + 1) % 3, (pos + 2) % 3
        if rng.random() < 0.5:
            p1, p2 = p2, p1
        tx, ty = list(terms[x]), list(terms[y])
        tx[p1] ^= ty[p1]
        ty[p2] ^= tx[p2]
        terms[x], terms[y] = tuple(tx), tuple(ty)
        flips += 1
        before = len(terms)
        _reduce(terms)
        if len(terms) < before:
            since = 0
        else:
            since += 1
        if len(terms) <= target and flips % every == 0:
            yield list(terms)
        if since > plateau and len(terms) > target:
            _plus(terms, rng)
            since = 0


def _plus(terms, rng):
    """a⊗b⊗c + a'⊗b'⊗c' = (a+a')⊗b⊗c + a'⊗b⊗c + a'⊗b'⊗c': one more term, with a shared factor."""
    n = len(terms)
    x, y = rng.sample(range(n), 2)
    a, b, c = terms[x]
    a2, _, _ = terms[y]
    if a == a2:
        return
    terms[x] = (a ^ a2, b, c)
    terms.append((a2, b, c))


def main():
    p, q, r, target = (int(s) for s in sys.argv[1:5])
    count = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = random.Random(seed)
    got = set()
    for terms in walk(p, q, r, target, rng):
        key = frozenset(terms)
        if key in got:
            continue
        assert is_scheme(terms, p, q, r)
        got.add(key)
        print(len(terms), sorted(terms), flush=True)
        if len(got) >= count:
            break


if __name__ == "__main__":
    main()
