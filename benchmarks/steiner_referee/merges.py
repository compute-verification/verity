#!/usr/bin/env python3
"""Referee checks of Theorem B, Corollary 7 and Lemmas C / C-sym (note:20261015T0436Z-draft-merges-and-reuse,
note:20261015T0416Z-draft-steiner-rigidity-laminar), over GF(p).

Model: n x K x n' matrix multiplication with stride L; a-variable (r, k) is X coordinate r*K + k, b-variable (k, c) is
Y coordinate k*n' + c; T_{pi,s} = sum_{k < sL} a_{rk} b_{kc}. A computation is a list of words: the zero word, a
product step (one parent plus a rank-1 form: one tensor-core product), or a head (a merge: a sum of earlier words).
Commits map (pi, s) to words; a commit is correct when the word's value is T_{pi,s}.

For each computation it computes:
  * Theorem B: per merge-free tree, #products >= L*sum hi (tau_0) or L*(sum hi - min lo) (head trees), and the total
    #products >= L*(|G_T| - H), H = heads whose tree holds a correct target;
  * Corollary 7: #products + sum_v d(v) >= L * sum_pi s*_pi, with d(v) = min over earlier words q of rank(w_v - w_q);
  * the whole-floor form #products >= L * sum_pi s*_pi - L * #heads, which is not claimed and is reported, not asserted.

Modes (one JSON line each; exit 1 if Theorem B, Corollary 7 or a lemma fails):
  lift    inner-lifted Strassen on <2, 2m, 2>: 7m products, 2 merges, each at distance m from every earlier word.
  cover   the covering lemma exhaustively: every assignment of correct targets to tau_0 and head trees.
  random  random efficient computations with far merges, restarts, shared heads and wrong commits on <n, n, n>.
  lemmac  Lemma C (bilinear) and Lemma C-sym (polar forms of quadratic forms) on row stars and cell patterns.

    python3 merges.py lift --mmax 6 --primes 2,3,1000003
    python3 merges.py cover --positions 3 --levels 3 --heads 3
    python3 merges.py random --n 4 --L 1 --primes 2,3 --count 50 --seed 0
    python3 merges.py lemmac --primes 2,3,5 --count 500 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys

import numpy as np

from lemma3 import rank_mod


def rank1_decomp(d: np.ndarray, p: int) -> list[np.ndarray]:
    """rank(d) rank-1 forms summing to d over GF(p) (Wedderburn rank reduction)."""
    r = np.array(d, dtype=np.int64) % p
    out = []
    while True:
        nz = np.argwhere(r)
        if len(nz) == 0:
            return out
        i, j = nz[0]
        inv = pow(int(r[i, j]), p - 2, p)
        piece = np.outer(r[:, j], r[i, :] * inv) % p
        out.append(piece)
        r = (r - piece) % p


class Comp:
    def __init__(self, shape, p):
        self.shape, self.p = shape, p
        n, k, n2 = shape
        self.vals = [np.zeros((n * k, k * n2), dtype=np.int64)]
        self.kind = ["zero"]
        self.parents = [()]
        self.commits = {}

    def prod(self, parent, form):
        self.vals.append((self.vals[parent] + form) % self.p)
        self.kind.append("prod")
        self.parents.append((parent,))
        return len(self.vals) - 1

    def chain(self, parent, forms):
        w = parent
        for f in forms:
            w = self.prod(w, f)
        return w

    def merge(self, *ws):
        self.vals.append(sum(self.vals[w] for w in ws) % self.p)
        self.kind.append("head")
        self.parents.append(tuple(ws))
        return len(self.vals) - 1

    def commit(self, pos, s, w):
        self.commits[(pos, s)] = w


def target(shape, pos, s, L, p):
    n, k, n2 = shape
    r, c = pos
    m = np.zeros((n * k, k * n2), dtype=np.int64)
    for kk in range(s * L):
        m[r * k + kk, kk * n2 + c] = 1
    return m % p


def analyze(comp: Comp, L: int):
    p, shape = comp.p, comp.shape
    nw = len(comp.vals)
    root = [0] * nw
    for w in range(1, nw):
        root[w] = w if comp.kind[w] == "head" else root[comp.parents[w][0]]
    prods = {}
    for w in range(nw):
        if comp.kind[w] == "prod":
            prods[root[w]] = prods.get(root[w], 0) + 1
    correct = [(pos, s, w) for (pos, s), w in comp.commits.items()
               if np.array_equal(comp.vals[w], target(shape, pos, s, L, p))]
    trees = {}
    for pos, s, w in correct:
        t = trees.setdefault(root[w], {})
        lo, hi = t.get(pos, (s, s))
        t[pos] = (min(lo, s), max(hi, s))
    fails = []
    trees_tight = 0
    for tr, t in trees.items():
        if tr == 0:
            bound = L * sum(hi for lo, hi in t.values())
        else:
            bound = L * (sum(hi for lo, hi in t.values()) - min(lo for lo, hi in t.values()))
        trees_tight += prods.get(tr, 0) == bound
        if prods.get(tr, 0) < bound:
            fails.append({"tree": tr, "products": prods.get(tr, 0), "bound": bound})
    heads = [w for w in range(nw) if comp.kind[w] == "head"]
    H = sum(1 for tr in trees if tr != 0)
    total = sum(prods.values())
    gt = len(correct)
    sstar = {}
    for pos, s, w in correct:
        sstar[pos] = max(sstar.get(pos, 0), s)
    floor = L * sum(sstar.values())
    d = {}
    for v in heads:
        d[v] = min(rank_mod((comp.vals[v] - comp.vals[q]) % p, p) for q in range(v))
    res = {
        "products": total, "heads": len(heads), "H": H, "correct": gt, "floor": floor,
        "trees": len(trees), "trees_tight": trees_tight,
        "theorem_b": L * (gt - H), "cor7_lhs": total + sum(d.values()),
        "d": sorted(d.values()), "whole_floor": floor - L * len(heads),
    }
    if total < L * (gt - H):
        fails.append({"total": total, "bound": L * (gt - H)})
    if total + sum(d.values()) < floor:
        fails.append({"cor7": total + sum(d.values()), "floor": floor})
    res["whole_floor_fails"] = total < floor - L * len(heads)
    res["failures"] = fails
    return res


# Strassen, C = sum_i W_i <U_i, A> <V_i, B>.
STRASSEN = [
    ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
    ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
    ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
    ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
    ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
    ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
    ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
]
# Output supports the accumulator tree needs, as sets of (r, c): P1..P7 of the schedule below.
WANT = [{(0, 0), (1, 0), (1, 1)}, {(1, 1)}, {(0, 0), (0, 1), (1, 0), (1, 1)}, {(0, 0), (1, 0)}, {(0, 1)},
        {(1, 0), (1, 1)}, {(0, 0)}]


def is_matmul(dec) -> bool:
    t = np.zeros((2, 2, 2, 2, 2, 2), dtype=np.int64)
    for u, v, w in dec:
        t += np.einsum("ab,cd,ef->abcdef", np.array(u), np.array(v), np.array(w))
    want = np.zeros_like(t)
    for r, k, c in itertools.product(range(2), repeat=3):
        want[r, k, k, c, r, c] = 1
    return np.array_equal(t, want)


def strassen_forms():
    """A rank-7 decomposition (Strassen moved by C -> X^-1 C Z^-1) whose outputs W_i are exactly the indicators WANT."""
    assert is_matmul(STRASSEN)
    mats = [np.array(m) for m in itertools.product([-1, 0, 1], repeat=4)]
    mats = [m.reshape(2, 2) for m in mats if round(abs(np.linalg.det(m.reshape(2, 2)))) == 1]
    inv = lambda m: np.round(np.linalg.inv(m)).astype(np.int64)
    for x, z in itertools.product(mats, mats):
        xi, zi = inv(x), inv(z)
        dec = [(x.T @ np.array(u), np.array(v) @ z.T, xi @ np.array(w) @ zi) for u, v, w in STRASSEN]
        out = []
        for u, v, w in dec:
            vals = {int(e) for e in w.flat if e}
            if len(vals) != 1:
                break
            sgn = vals.pop()
            if sgn not in (1, -1):
                break
            out.append((sgn * u, v, sgn * w))
        else:
            supports = [frozenset((r, c) for r in range(2) for c in range(2) if w[r, c]) for u, v, w in out]
            order = []
            for want in WANT:
                cands = [i for i, s in enumerate(supports) if s == frozenset(want) and i not in order]
                if not cands:
                    break
                order.append(cands[0])
            else:
                dec = [out[i] for i in order]
                assert is_matmul([(u.tolist(), v.tolist(), w.tolist()) for u, v, w in dec])
                return dec
    raise SystemExit("no X, Z puts Strassen's outputs on the schedule's supports")


def lifted_strassen(m: int, L: int, p: int, walk_down: bool):
    """<2, 2m, 2>: Strassen on each of the m inner 2x2 blocks, accumulated by one tree with two merges."""
    shape = (2, 2 * m, 2)
    dec = strassen_forms()
    comp = Comp(shape, p)

    def P(i, j):
        u, v, _ = dec[i]
        a = np.zeros(2 * 2 * m, dtype=np.int64)
        b = np.zeros(2 * m * 2, dtype=np.int64)
        for r, kk in itertools.product(range(2), repeat=2):
            a[r * 2 * m + 2 * j + kk] = u[r, kk]
            b[(2 * j + kk) * 2 + r] = v[kk, r]
        return np.outer(a, b) % p

    run = lambda start, i: comp.chain(start, [P(i, j) for j in range(m)])
    n3 = run(0, 2)
    c12 = run(n3, 4)
    n13 = run(n3, 0)
    n134 = run(n13, 3)
    c11 = run(n134, 6)
    n132 = run(n13, 1)
    n6 = run(0, 5)
    c21 = comp.merge(n134, n6)
    c22 = comp.merge(n132, n6)
    top = 2 * m // L
    for pos, w in (((0, 0), c11), ((0, 1), c12), ((1, 0), c21), ((1, 1), c22)):
        comp.commit(pos, top, w)
        if walk_down:
            for s in range(top, 1, -1):
                r, c = pos
                for kk in range((s - 1) * L, s * L):
                    f = np.zeros_like(comp.vals[0])
                    f[r * 2 * m + kk, kk * 2 + c] = p - 1
                    w = comp.prod(w, f)
                comp.commit(pos, s - 1, w)
    return comp


def mode_lift(args):
    rows, bad = [], 0
    for p in args.primes:
        for m in range(1, args.mmax + 1):
            for L in sorted({l for l in (1, 2, m, 2 * m) if (2 * m) % l == 0}):
                for walk in (False, True):
                    comp = lifted_strassen(m, L, p, walk)
                    res = analyze(comp, L)
                    assert res["correct"] == (4 * 2 * m // L if walk else 4), res
                    bad += len(res["failures"])
                    rows.append({"p": p, "m": m, "L": L, "prefixes": walk, **res})
    first = [r for r in rows if r["whole_floor_fails"]][:3]
    return {"mode": "lift", "computations": len(rows), "failures": bad,
            "whole_floor_fails": sum(r["whole_floor_fails"] for r in rows),
            "first_whole_floor_fails": first, "rows": rows}


def cover_bound(assign, positions, levels, heads, mutate=False):
    per = {}
    for (pos, s), t in assign.items():
        if t >= 0:
            lo, hi = per.setdefault(t, {}).get(pos, (s, s))
            per[t][pos] = (min(lo, s), max(hi, s))
    total = 0
    for t, d in per.items():
        if t == 0:
            total += sum(hi for lo, hi in d.values())
        elif mutate:
            total += sum(hi - lo for lo, hi in d.values())
        else:
            total += sum(hi for lo, hi in d.values()) - min(lo for lo, hi in d.values())
    used = sum(1 for t in per if t != 0)
    gt = sum(1 for t in assign.values() if t >= 0)
    return total, gt - used


def mode_cover(args):
    cells = [(pos, s) for pos in range(args.positions) for s in range(1, args.levels + 1)]
    opts = range(-1, args.heads + 1)
    n = tight = fails = 0
    first = []
    for i, choice in enumerate(itertools.product(opts, repeat=len(cells))):
        if i % args.shards != args.shard:
            continue
        assign = dict(zip(cells, choice))
        total, want = cover_bound(assign, args.positions, args.levels, args.heads, args.mutate)
        n += 1
        tight += total == want
        if total < want:
            fails += 1
            if len(first) < 3:
                first.append({"assign": [[list(c), t] for c, t in assign.items()], "sum": total, "want": want})
    return {"mode": "cover", "positions": args.positions, "levels": args.levels, "heads": args.heads,
            "shard": [args.shard, args.shards], "mutate": args.mutate, "assignments": n, "tight": tight,
            "failures": fails, "first_failures": first}


def random_comp(n, L, p, rng):
    shape = (n, n, n)
    comp = Comp(shape, p)
    order = []
    left = {(r, c): 1 for r in range(n) for c in range(n)}
    while left:
        pos = rng.choice(sorted(left))
        order.append((pos, left[pos]))
        left[pos] += 1
        if left[pos] > n // L:
            del left[pos]
    last = {}
    partials = []
    for pos, s in order:
        t = target(shape, pos, s, L, p)
        nearest = lambda: min(range(len(comp.vals)), key=lambda q: (rank_mod((t - comp.vals[q]) % p, p), rng.random()))
        strat = rng.choices(["cont", "near", "merge", "far", "cheap"], weights=[3, 3, 2, 2, 3])[0]
        if strat == "cheap":
            prev = last.get(pos, 0)
            if partials and rng.random() < 0.5:
                z = rng.choice(partials)
            else:
                pieces = rank1_decomp((t - comp.vals[prev]) % p, p)
                rng.shuffle(pieces)
                z = comp.chain(0, pieces[:rng.randint(1, len(pieces))])
                partials.append(z)
            v = comp.merge(prev, z)
            w = comp.chain(v, rank1_decomp((t - comp.vals[v]) % p, p))
        elif strat == "cont":
            start = last.get(pos, 0)
            pieces = rank1_decomp((t - comp.vals[start]) % p, p)
            w = comp.chain(start, pieces)
        elif strat == "near":
            start = nearest()
            w = comp.chain(start, rank1_decomp((t - comp.vals[start]) % p, p))
        else:
            q1 = rng.randrange(len(comp.vals))
            r0 = rng.choice([0, rng.randrange(len(comp.vals))])
            pieces = rank1_decomp((t - comp.vals[q1] - comp.vals[r0]) % p, p)
            rng.shuffle(pieces)
            cut = rng.randint(0, len(pieces)) if strat == "merge" else len(pieces)
            q2 = comp.chain(r0, pieces[:cut])
            v = comp.merge(q1, q2)
            w = comp.chain(v, pieces[cut:])
        if rng.random() < 0.1:
            junk = np.outer(np.array([rng.randrange(p) for _ in range(n * n)]),
                            np.array([rng.randrange(p) for _ in range(n * n)])) % p
            w = comp.prod(w, junk)
        comp.commit(pos, s, w)
        last[pos] = w
    return comp


def mode_random(args):
    rng = random.Random(args.seed)
    rows, bad, wf = 0, [], 0
    tight_b = tight_7 = trees = trees_tight = 0
    for p in args.primes:
        for _ in range(args.count):
            comp = random_comp(args.n, args.L, p, rng)
            res = analyze(comp, args.L)
            rows += 1
            trees += res["trees"]
            trees_tight += res["trees_tight"]
            wf += res["whole_floor_fails"]
            tight_b += res["products"] == res["theorem_b"]
            tight_7 += res["cor7_lhs"] == res["floor"]
            if res["failures"] and len(bad) < 3:
                bad.append(res)
            elif res["failures"]:
                bad.append(None)
    return {"mode": "random", "n": args.n, "L": args.L, "primes": args.primes, "seed": args.seed,
            "computations": rows, "trees": trees, "trees_tight": trees_tight, "theorem_b_tight": tight_b, "cor7_tight": tight_7, "whole_floor_fails": wf,
            "failures": len(bad), "first_failures": [b for b in bad if b][:3]}


def polar(q: np.ndarray, p: int) -> np.ndarray:
    """Matrix of q(u+v) - q(u) - q(v): no 1/2, so it is the symmetric matrix of q in every characteristic."""
    return (q + q.T) % p


def bilinear_as_quadratic(d: np.ndarray) -> np.ndarray:
    nx, ny = d.shape
    q = np.zeros((nx + ny, nx + ny), dtype=np.int64)
    q[:nx, nx:] = d
    return q


def mode_lemmac(args):
    rng = random.Random(args.seed)
    n = fails = 0
    tight_c = tight_sym = 0
    first = []
    for p in args.primes:
        for it in range(args.count):
            L = rng.randint(1, 3)
            k = rng.randint(2, 4)
            P, Q = (1, k) if rng.random() < 0.6 else (rng.randint(1, 3), rng.randint(2, 4))
            nx, ny = P * L, Q * L
            cells = [(0, t) for t in range(k)] if (P, Q) == (1, k) else \
                rng.sample([(a, b) for a in range(P) for b in range(Q)], min(k, P * Q))
            k = len(cells)
            ds = []
            for a, b in cells:
                d = np.zeros((nx, ny), dtype=np.int64)
                for kk in range(L):
                    d[a * L + kk, b * L + kk] = 1
                ds.append(d)
            beta = [rng.randrange(1, p) for _ in range(k)]
            kind = rng.choice(["partials", "random", "lowrank"])
            if kind == "partials":
                z = sum(rng.randrange(p) * np.outer(np.eye(nx, dtype=np.int64)[a * L + kk], np.eye(ny, dtype=np.int64)[b * L + kk])
                        for a, b in cells for kk in range(L)) % p
            elif kind == "random":
                z = np.array([[rng.randrange(p) for _ in range(ny)] for _ in range(nx)])
            else:
                z = sum(np.outer([rng.randrange(p) for _ in range(nx)], [rng.randrange(p) for _ in range(ny)])
                        for _ in range(rng.randint(1, L))) % p
            zq = np.triu(np.array([[rng.randrange(p) for _ in range(nx + ny)] for _ in range(nx + ny)]))
            zq[:nx, nx:] = (zq[:nx, nx:] + z) % p
            for step in range(args.search):
                lhs = sum(rank_mod((ds[t] - beta[t] * z) % p, p) for t in range(k))
                rz = rank_mod(z, p)
                want_c = max(L * (k - 1), L * k - rz) if (P, Q) == (1, k) else L * (k - 1)
                pz = polar(zq, p)
                lhs_s = sum(rank_mod((polar(bilinear_as_quadratic(ds[t]), p) - beta[t] * pz) % p, p) for t in range(k))
                want_s = max(2 * L * (k - 1), 2 * L * k - rank_mod(pz, p))
                n += 1
                tight_c += lhs == want_c
                tight_sym += lhs_s == want_s
                if lhs < want_c or lhs_s < want_s:
                    fails += 1
                    if len(first) < 3:
                        first.append({"p": p, "L": L, "k": k, "cells": cells, "lemma_c": [lhs, want_c],
                                      "lemma_c_sym": [lhs_s, want_s]})
                i, j = rng.randrange(nx), rng.randrange(ny)
                z2 = z.copy()
                z2[i, j] = rng.randrange(p)
                if sum(rank_mod((ds[t] - beta[t] * z2) % p, p) for t in range(k)) <= lhs:
                    z = z2
                a, b = rng.randrange(nx + ny), rng.randrange(nx + ny)
                zq2 = zq.copy()
                zq2[min(a, b), max(a, b)] = rng.randrange(p)
                pz2 = polar(zq2, p)
                if sum(rank_mod((polar(bilinear_as_quadratic(ds[t]), p) - beta[t] * pz2) % p, p) for t in range(k)) <= lhs_s:
                    zq = zq2
    return {"mode": "lemmac", "primes": args.primes, "seed": args.seed, "instances": n, "lemma_c_tight": tight_c,
            "lemma_c_sym_tight": tight_sym, "failures": fails, "first_failures": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lift", "cover", "random", "lemmac"])
    ap.add_argument("--primes", type=lambda s: [int(x) for x in s.split(",")], default=[2, 3])
    ap.add_argument("--mmax", type=int, default=4)
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--search", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {"lift": mode_lift, "cover": mode_cover, "random": mode_random, "lemmac": mode_lemmac}[args.mode](args)
    print(json.dumps(out, default=int))
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
