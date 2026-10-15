#!/usr/bin/env python3
"""Round-2 referee checks of Claim P (projection onto the x-y bilinear block) and Claim R (int8 slots of mixed operands).

Claim P: in an accumulate-and-merge computation whose products multiply linear forms u = a.x + b.y and v = c.x + d.y,
the x-y block of every word is a linear image of the computation, a product maps to a d^T + c b^T (rank <= 2, <= 1
when an operand is pure), and a commit equal to its bilinear target as a full form is still equal after projection.
So #unmixed + 2 #mixed is at least Theorem B's bounds (per tree, and L(|G| - H) in total), G the commits that are right
as full quadratic forms.

Words are quadratic forms on Z = X + Y over GF(p), stored as upper-triangular coefficient matrices (Q[i, j] is the
coefficient of z_i z_j, i <= j); X coordinate r*K + k and Y coordinate k*n' + c, as in merges.py. A product adds
(u.z)(v.z). The projected computation is a merges.Comp in which a mixed product becomes two rank-1 steps a d^T, c b^T
and any other product one step, so its product count is the weighted cost and merges.analyze checks Theorem B and
Corollary 7 on it directly.

Modes (one JSON line each; exit 1 when any step of P or any bound fails):
  single  products u, v (all of them over GF(2), a sample over GF(p) and Z/2^32): the x-y block of the literal product
          equals a d^T + c b^T with its rank bound, and evaluating at 0, e_i, e_j, e_i + e_j reads the same block.
  search  mixed computations: honest random computations (far merges, restarts, shared heads, wrong commits) with
          chained products paired into mixed ones, junk cancelled in the chain, left in, or mixed; Winograd pairs with
          junk in the chain, shared through merges, or left in; inner-lifted Strassen paired; then products dropped
          and operands perturbed at random.
  prop5   A7's Proposition 5 family (7 products on the 2 x 2 block): P's count is the floor plus one per column pair.
  r       Claim R: the s8 range of integer linear operands, and one-slot mixed products (the wrapping cast, 7-bit
          halves, FP4 bit sums) checked exactly on random salted s8 matrices.

    python3 mixed.py single --shape 2,2,3 --primes 2,3 --count 200000 --seed 0
    python3 mixed.py search --shape 2,2,3 --L 1 --primes 2,3 --count 400 --seed 0
    python3 mixed.py prop5 --primes 2,3,5 --pmax 4
    python3 mixed.py r --n 64 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys

import numpy as np

from lemma3 import rank_mod
from merges import Comp, analyze, lifted_strassen, rank1_decomp, target


def qform(u: np.ndarray, v: np.ndarray, p: int) -> np.ndarray:
    m = np.outer(u, v)
    return (np.triu(m + m.T, 1) + np.diag(np.diag(m))) % p


class Full:
    def __init__(self, shape, p):
        n, k, n2 = shape
        self.shape, self.p = shape, p
        self.nx, self.ny = n * k, k * n2
        z = self.nx + self.ny
        self.vals = [np.zeros((z, z), dtype=np.int64)]
        self.kind = ["zero"]
        self.parents = [()]
        self.ops = [None]
        self.commits = {}

    def prod(self, parent, u, v):
        u, v = np.asarray(u, dtype=np.int64) % self.p, np.asarray(v, dtype=np.int64) % self.p
        self.vals.append((self.vals[parent] + qform(u, v, self.p)) % self.p)
        self.kind.append("prod")
        self.parents.append((parent,))
        self.ops.append((u, v))
        return len(self.vals) - 1

    def merge(self, *ws):
        self.vals.append(sum(self.vals[w] for w in ws) % self.p)
        self.kind.append("head")
        self.parents.append(tuple(ws))
        self.ops.append(None)
        return len(self.vals) - 1

    def commit(self, pos, s, w):
        self.commits[(pos, s)] = w

    def xy(self, w):
        return self.vals[w][:self.nx, self.nx:]

    def full_target(self, pos, s, L):
        t = np.zeros_like(self.vals[0])
        t[:self.nx, self.nx:] = target(self.shape, pos, s, L, self.p)
        return t

    def xvec(self, r, k, coef=1):
        e = np.zeros(self.nx + self.ny, dtype=np.int64)
        e[r * self.shape[1] + k] = coef % self.p
        return e

    def yvec(self, k, c, coef=1):
        e = np.zeros(self.nx + self.ny, dtype=np.int64)
        e[self.nx + k * self.shape[2] + c] = coef % self.p
        return e


def mixed_op(f: Full, u, v) -> bool:
    return bool(u[:f.nx].any() and u[f.nx:].any() and v[:f.nx].any() and v[f.nx:].any())


MIXED_WEIGHT = 2


def project(f: Full):
    """The projected computation, with every step of P checked on the way."""
    p = f.p
    c = Comp(f.shape, p)
    img = [0]
    nmix = nun = 0
    fails = []
    for w in range(1, len(f.vals)):
        if f.kind[w] == "prod":
            u, v = f.ops[w]
            a, b, cc, d = u[:f.nx], u[f.nx:], v[:f.nx], v[f.nx:]
            p1, p2 = np.outer(a, d) % p, np.outer(cc, b) % p
            step = (p1 + p2) % p
            lit = (f.vals[w] - f.vals[f.parents[w][0]]) % p
            if not np.array_equal(lit[:f.nx, f.nx:], step):
                fails.append({"block_not_adT_plus_cbT": w})
            if mixed_op(f, u, v) and MIXED_WEIGHT == 2:
                nmix += 1
                img.append(c.prod(c.prod(img[f.parents[w][0]], p1), p2))
            elif mixed_op(f, u, v):
                nmix += 1
                img.append(c.prod(img[f.parents[w][0]], step))
            else:
                nun += 1
                if rank_mod(step, p) > 1:
                    fails.append({"unmixed_rank_gt_1": w})
                img.append(c.prod(img[f.parents[w][0]], step))
        else:
            img.append(c.merge(*[img[q] for q in f.parents[w]]))
    for (pos, s), w in f.commits.items():
        c.commit(pos, s, img[w])
    for w in range(len(f.vals)):
        if not np.array_equal(c.vals[img[w]], f.xy(w)):
            fails.append({"projection_not_linear": w})
    return c, img, nmix, nun, fails


def tree_bounds(comp: Comp, L: int, correct):
    nw = len(comp.vals)
    root = [0] * nw
    for w in range(1, nw):
        root[w] = w if comp.kind[w] == "head" else root[comp.parents[w][0]]
    prods = {}
    for w in range(nw):
        if comp.kind[w] == "prod":
            prods[root[w]] = prods.get(root[w], 0) + 1
    trees = {}
    for pos, s, w in correct:
        t = trees.setdefault(root[w], {})
        lo, hi = t.get(pos, (s, s))
        t[pos] = (min(lo, s), max(hi, s))
    fails = []
    for tr, t in trees.items():
        his = sum(hi for lo, hi in t.values())
        bound = L * his if tr == 0 else L * (his - min(lo for lo, hi in t.values()))
        if prods.get(tr, 0) < bound:
            fails.append({"tree": tr, "weighted": prods.get(tr, 0), "bound": bound})
    H = sum(1 for tr in trees if tr != 0)
    return fails, L * (len(correct) - H)


def check(f: Full, L: int):
    proj, img, nmix, nun, fails = project(f)
    weighted = nun + MIXED_WEIGHT * nmix
    assert weighted == sum(1 for k in proj.kind if k == "prod")
    full_ok = [(pos, s, img[w]) for (pos, s), w in f.commits.items()
               if np.array_equal(f.vals[w], f.full_target(pos, s, L))]
    proj_ok = {(pos, s) for (pos, s), w in proj.commits.items()
               if np.array_equal(proj.vals[w], target(f.shape, pos, s, L, f.p))}
    if not {(pos, s) for pos, s, _ in full_ok} <= proj_ok:
        fails.append({"full_right_but_projection_wrong": True})
    tfails, bound = tree_bounds(proj, L, full_ok)
    fails += tfails
    if weighted < bound:
        fails.append({"weighted": weighted, "bound_full_G": bound})
    res = analyze(proj, L)
    fails += res["failures"]
    return {"weighted": weighted, "mixed": nmix, "unmixed": nun, "right_full": len(full_ok),
            "right_projected": len(proj_ok), "accidental": len(proj_ok) - len(full_ok), "bound_full_G": bound,
            "bound_projected_G": res["theorem_b"], "heads": res["heads"], "slack": weighted - bound,
            "tight": weighted == bound and bound > 0, "failures": fails}


def rank1_split(form: np.ndarray, p: int):
    nz = np.argwhere(form % p)
    if len(nz) == 0:
        return np.zeros(form.shape[0], dtype=np.int64), np.zeros(form.shape[1], dtype=np.int64)
    i, j = nz[0]
    inv = pow(int(form[i, j]), p - 2, p)
    a, d = form[:, j] % p, (form[i, :] * inv) % p
    assert np.array_equal(np.outer(a, d) % p, form % p)
    return a, d


def to_full(comp: Comp, rng, pair_prob: float, junk: str, wrong_prob: float) -> Full:
    """Lift a bilinear computation; chained product pairs a1 d1^T, a2 d2^T become (a1.x + d2.y)(a2.x + d1.y)."""
    p = comp.p
    f = Full(comp.shape, p)
    zx, zy = np.zeros(f.ny, dtype=np.int64), np.zeros(f.nx, dtype=np.int64)
    nk = len(comp.vals)
    children = [[] for _ in range(nk)]
    for w in range(1, nk):
        for q in comp.parents[w]:
            children[q].append(w)
    committed = set(comp.commits.values())
    img = [0] + [None] * (nk - 1)
    pending = {}
    for w in range(1, nk):
        if comp.kind[w] != "prod":
            img[w] = f.merge(*[img[q] for q in comp.parents[w]])
            continue
        par = comp.parents[w][0]
        a, d = rank1_split((comp.vals[w] - comp.vals[par]) % p, p)
        if w in pending:
            w0, a1, d1 = pending.pop(w)
            u, v = np.concatenate([a1, d]), np.concatenate([a, d1])
            if rng.random() < wrong_prob:
                (u if rng.random() < 0.5 else v)[rng.randrange(len(u))] += rng.randrange(1, p)
            t = f.prod(img[w0], u, v)
            if junk == "chain" or (junk == "mixed" and rng.random() < 0.5):
                t = f.prod(t, np.concatenate([a1, zx]), np.concatenate([(-a) % p, zx]))
                t = f.prod(t, np.concatenate([zy, d]), np.concatenate([zy, (-d1) % p]))
            img[w] = t
            continue
        ch = children[w]
        if len(ch) == 1 and comp.kind[ch[0]] == "prod" and w not in committed and rng.random() < pair_prob:
            pending[ch[0]] = (par, a, d)
            continue
        img[w] = f.prod(img[par], np.concatenate([a, zx]), np.concatenate([zy, d]))
    for (pos, s), w in comp.commits.items():
        f.commit(pos, s, img[w])
    return f


def drop(f: Full, rng, q: float) -> Full:
    """The same computation with each product skipped with probability q (its children hang from its parent)."""
    g = Full(f.shape, f.p)
    img = [0]
    for w in range(1, len(f.vals)):
        if f.kind[w] == "prod":
            if rng.random() < q:
                img.append(img[f.parents[w][0]])
            else:
                img.append(g.prod(img[f.parents[w][0]], *f.ops[w]))
        else:
            img.append(g.merge(*[img[x] for x in f.parents[w]]))
    for key, w in f.commits.items():
        g.commit(key[0], key[1], img[w])
    return g


def rebuild(f: Full, edit) -> Full:
    """A copy of f with edit(w, (u, v)) applied to each product: new operands, or None to skip the product."""
    g = Full(f.shape, f.p)
    img = [0]
    for w in range(1, len(f.vals)):
        if f.kind[w] == "prod":
            ops = edit(w, f.ops[w])
            img.append(img[f.parents[w][0]] if ops is None else g.prod(img[f.parents[w][0]], *ops))
        else:
            img.append(g.merge(*[img[x] for x in f.parents[w]]))
    for key, w in f.commits.items():
        g.commit(key[0], key[1], img[w])
    return g


def recommit(f: Full, L: int):
    """Commit each target to a word equal to it as a full form, preferring tree 0 and trees already used."""
    nw = len(f.vals)
    root = [0] * nw
    for w in range(1, nw):
        root[w] = w if f.kind[w] == "head" else root[f.parents[w][0]]
    vals = np.stack(f.vals)
    n, K, n2 = f.shape
    f.commits, used = {}, set()
    for pos in itertools.product(range(n), range(n2)):
        for s in range(1, K // L + 1):
            hits = np.nonzero((vals == f.full_target(pos, s, L)).all(axis=(1, 2)))[0]
            if len(hits):
                w = int(min(hits, key=lambda h: (root[h] != 0, root[h] not in used)))
                f.commits[(pos, s)] = w
                used.add(root[w])


def climb(f: Full, L: int, rng, iters: int):
    """Local search for weighted < bound: drop a product, perturb an operand entry, or zero one operand's x or y part."""
    recommit(f, L)
    cur = check(f, L)
    best, fails = cur["slack"], []
    for _ in range(iters):
        prods = [w for w in range(len(f.vals)) if f.kind[w] == "prod"]
        if not prods:
            break
        t = rng.choice(prods)
        move = rng.choice(["drop", "perturb", "perturb", "unmix"])
        if move == "drop":
            g = rebuild(f, lambda w, ops: None if w == t else ops)
        elif move == "perturb":
            i = rng.randrange(2 * (f.nx + f.ny))
            delta = rng.randrange(1, f.p)

            def edit(w, ops, i=i, delta=delta):
                if w != t:
                    return ops
                u, v = ops[0].copy(), ops[1].copy()
                (u if i < len(u) else v)[i % len(u)] += delta
                return u, v
            g = rebuild(f, edit)
        else:
            side, half = rng.randrange(2), rng.randrange(2)

            def edit(w, ops, side=side, half=half):
                if w != t:
                    return ops
                ops = [ops[0].copy(), ops[1].copy()]
                if half == 0:
                    ops[side][:f.nx] = 0
                else:
                    ops[side][f.nx:] = 0
                return tuple(ops)
            g = rebuild(f, edit)
        recommit(g, L)
        res = check(g, L)
        if res["failures"]:
            fails.append(res)
        if res["slack"] <= cur["slack"]:
            f, cur = g, res
            best = min(best, cur["slack"])
    return best, cur, fails


def random_bilinear(shape, L, p, rng):
    """merges.random_comp on a rectangular shape: cont, near, merge, far and cheap steps, 10% junk products."""
    n, K, n2 = shape
    comp = Comp(shape, p)
    order = []
    left = {(r, c): 1 for r in range(n) for c in range(n2)}
    while left:
        pos = rng.choice(sorted(left))
        order.append((pos, left[pos]))
        left[pos] += 1
        if left[pos] > K // L:
            del left[pos]
    last, partials = {}, []
    for pos, s in order:
        t = target(shape, pos, s, L, p)
        strat = rng.choices(["cont", "near", "merge", "far", "cheap"], weights=[3, 3, 2, 2, 3])[0]
        if strat == "cheap":
            prev = last.get(pos, 0)
            if partials and rng.random() < 0.5:
                z = rng.choice(partials)
            else:
                pieces = rank1_decomp((t - comp.vals[prev]) % p, p)
                rng.shuffle(pieces)
                z = comp.chain(0, pieces[:rng.randint(1, max(1, len(pieces)))])
                partials.append(z)
            v = comp.merge(prev, z)
            w = comp.chain(v, rank1_decomp((t - comp.vals[v]) % p, p))
        elif strat == "cont":
            start = last.get(pos, 0)
            w = comp.chain(start, rank1_decomp((t - comp.vals[start]) % p, p))
        elif strat == "near":
            start = min(range(len(comp.vals)), key=lambda q: (rank_mod((t - comp.vals[q]) % p, p), rng.random()))
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
            junk = np.outer(np.array([rng.randrange(p) for _ in range(n * K)]),
                            np.array([rng.randrange(p) for _ in range(K * n2)])) % p
            w = comp.prod(w, junk)
        comp.commit(pos, s, w)
        last[pos] = w
    return comp


def winograd(shape, L, p, rng, junk):
    """Winograd's pairs (x_{r,k0} + y_{k1,c})(x_{r,k1} + y_{k0,c}) per cell; junk in the chain, shared, or left in."""
    f = Full(shape, p)
    n, K, n2 = shape
    S = K // L
    cs = sorted({S} | {s for s in range(1, S) if rng.random() < 0.5})
    spans, k = [], 0
    for s in cs:
        ks = list(range(k, s * L))
        k = s * L
        spans.append((s, [(ks[2 * i], ks[2 * i + 1]) for i in range(len(ks) // 2)], ks[-1] if len(ks) % 2 else None))
    jx, jy = {}, {}
    if junk == "shared":
        for r in range(n):
            w = 0
            for s, pairs, _ in spans:
                for k0, k1 in pairs:
                    w = f.prod(w, f.xvec(r, k0), f.xvec(r, k1, -1))
                jx[(r, s)] = w
        for c in range(n2):
            w = 0
            for s, pairs, _ in spans:
                for k0, k1 in pairs:
                    w = f.prod(w, f.yvec(k1, c), f.yvec(k0, c, -1))
                jy[(c, s)] = w
    for r in range(n):
        for c in range(n2):
            w = 0
            for s, pairs, odd in spans:
                for k0, k1 in pairs:
                    w = f.prod(w, f.xvec(r, k0) + f.yvec(k1, c), f.xvec(r, k1) + f.yvec(k0, c))
                    if junk == "chain":
                        w = f.prod(w, f.xvec(r, k0), f.xvec(r, k1, -1))
                        w = f.prod(w, f.yvec(k1, c), f.yvec(k0, c, -1))
                if odd is not None:
                    w = f.prod(w, f.xvec(r, odd), f.yvec(odd, c))
                if junk == "shared" and jx[(r, s)]:
                    f.commit((r, c), s, f.merge(w, jx[(r, s)], jy[(c, s)]))
                else:
                    f.commit((r, c), s, w)
    return f


def prop5(P: int, Q: int, p: int) -> Full:
    """note:20261015T0822Z-draft-conjecture-c-mix's Proposition 5 on a P x Q block at L = 2, per column pair (c, c'):
    W = y_0c' y_1c, V_r = W - (x_r0 - y_1c)(x_r1 - y_0c'), t_rc = V_r + x_r0 (y_0c + x_r1 - y_0c'),
    t_rc' = V_r + x_r1 (y_1c' + x_r0 - y_1c)."""
    f = Full((P, 2, Q), p)
    for c in range(0, Q, 2):
        c2 = c + 1
        w = f.prod(0, f.yvec(0, c2), f.yvec(1, c))
        for r in range(P):
            v = f.prod(w, f.xvec(r, 0) - f.yvec(1, c), -(f.xvec(r, 1) - f.yvec(0, c2)))
            f.commit((r, c), 1, f.prod(v, f.xvec(r, 0), f.yvec(0, c) + f.xvec(r, 1) - f.yvec(0, c2)))
            f.commit((r, c2), 1, f.prod(v, f.xvec(r, 1), f.yvec(1, c2) + f.xvec(r, 0) - f.yvec(1, c)))
    return f


def mode_prop5(args):
    rows, fails = [], 0
    for p in args.primes:
        for P in range(1, args.pmax + 1):
            for Q in (2, 4, 6):
                res = check(prop5(P, Q, p), 2)
                want_slack = Q // 2 if MIXED_WEIGHT == 2 else Q // 2 - (P * Q) // 2
                ok = res["right_full"] == P * Q and res["slack"] == want_slack
                fails += bool(res["failures"]) or not ok
                rows.append({"p": p, "P": P, "Q": Q, "products": res["mixed"] + res["unmixed"], **res})
    return {"mode": "prop5", "mixed_weight": MIXED_WEIGHT, "primes": args.primes, "computations": len(rows),
            "failures": fails, "rows": rows}


def mode_search(args):
    shape = tuple(args.shape)
    rng = random.Random(args.seed)
    stats = {"computations": 0, "failures": 0, "tight": 0, "with_mixed": 0, "with_accidental": 0,
             "min_slack": None, "by_kind": {}}
    first = []
    for p in args.primes:
        for i in range(args.count):
            kind = rng.choice(["paired", "paired", "winograd"])
            if kind == "paired":
                junk = rng.choice(["chain", "none", "mixed"])
                f = to_full(random_bilinear(shape, args.L, p, rng), rng, rng.choice([0.0, 0.5, 1.0]), junk,
                            rng.choice([0.0, 0.0, 0.1]))
            else:
                junk = rng.choice(["chain", "shared", "none"])
                f = winograd(shape, args.L, p, rng, junk)
            q = rng.choice([0.0, 0.0, 0.1, 0.3])
            if q:
                f = drop(f, rng, q)
            res = check(f, args.L)
            tag = f"{kind}/{junk}/drop{q}"
            if args.climb:
                best, end, cfails = climb(f, args.L, rng, args.climb)
                stats["climbs"] = stats.get("climbs", 0) + 1
                stats["climb_min_slack"] = min(stats.get("climb_min_slack", best), best)
                stats["climb_ends_tight"] = stats.get("climb_ends_tight", 0) + end["tight"]
                stats["climb_ends_mixed"] = stats.get("climb_ends_mixed", 0) + (end["mixed"] > 0 and end["tight"])
                if cfails:
                    res["failures"] = res["failures"] + cfails[:1]
            b = stats["by_kind"].setdefault(tag, {"n": 0, "tight": 0, "min_slack": None})
            b["n"] += 1
            b["tight"] += res["tight"]
            b["min_slack"] = res["slack"] if b["min_slack"] is None else min(b["min_slack"], res["slack"])
            stats["computations"] += 1
            stats["tight"] += res["tight"]
            stats["with_mixed"] += res["mixed"] > 0
            stats["with_accidental"] += res["accidental"] > 0
            stats["min_slack"] = res["slack"] if stats["min_slack"] is None else min(stats["min_slack"], res["slack"])
            if res["failures"]:
                stats["failures"] += 1
                if len(first) < 3:
                    first.append({"p": p, "kind": tag, **res})
    if args.lifted:
        for p in args.primes:
            for m in range(1, args.lifted + 1):
                for L in sorted({l for l in (1, 2, m, 2 * m) if (2 * m) % l == 0}):
                    for walk in (False, True):
                        for pair_prob, junk in ((1.0, "none"), (1.0, "chain"), (0.5, "mixed")):
                            f = to_full(lifted_strassen(m, L, p, walk), rng, pair_prob, junk, 0.0)
                            res = check(f, L)
                            tag = f"lifted/{junk}"
                            b = stats["by_kind"].setdefault(tag, {"n": 0, "tight": 0, "min_slack": None})
                            b["n"] += 1
                            b["tight"] += res["tight"]
                            b["min_slack"] = res["slack"] if b["min_slack"] is None else min(b["min_slack"],
                                                                                             res["slack"])
                            stats["computations"] += 1
                            stats["with_mixed"] += res["mixed"] > 0
                            stats["with_accidental"] += res["accidental"] > 0
                            if res["failures"]:
                                stats["failures"] += 1
                                if len(first) < 3:
                                    first.append({"p": p, "kind": tag, "m": m, "L": L, **res})
    return {"mode": "search", "shape": list(shape), "L": args.L, "primes": args.primes, "seed": args.seed, **stats,
            "first_failures": first}


def mode_single(args):
    shape = tuple(args.shape)
    n, K, n2 = shape
    nx, ny = n * K, K * n2
    z = nx + ny
    fails, checked = 0, 0
    rng = np.random.default_rng(args.seed)

    def batch(us, vs, p):
        nonlocal fails, checked
        m = np.einsum("bi,bj->bij", us, vs)
        lit = (m + m.transpose(0, 2, 1))[:, :nx, nx:] % p
        want = (np.einsum("bi,bj->bij", us[:, :nx], vs[:, nx:]) + np.einsum("bi,bj->bij", vs[:, :nx], us[:, nx:])) % p
        fails += int((lit != want).any(axis=(1, 2)).sum())
        ev = lambda pt: ((us @ pt) * (vs @ pt)) % p
        e = np.eye(z, dtype=np.int64)
        for i in range(nx):
            for j in range(nx, z):
                blk = (ev(e[i] + e[j]) - ev(e[i]) - ev(e[j])) % p
                fails += int((blk != lit[:, i, j - nx]).sum())
        checked += len(us)

    rows = {}
    for p in args.primes:
        if p == 2 and z <= 10:
            allv = np.array(list(itertools.product([0, 1], repeat=z)), dtype=np.int64)
            for chunk in range(0, len(allv), 64):
                us = np.repeat(allv[chunk:chunk + 64], len(allv), axis=0)
                vs = np.tile(allv, (len(allv[chunk:chunk + 64]), 1))
                batch(us, vs, 2)
            rows[str(p)] = "exhaustive"
        else:
            for _ in range(0, args.count, 4096):
                batch(rng.integers(0, p, (4096, z)), rng.integers(0, p, (4096, z)), p)
            rows[str(p)] = "sample"
    us = rng.integers(-128, 128, (4096, z))
    vs = rng.integers(-128, 128, (4096, z))
    batch(us, vs, 2 ** 32)
    rows["2^32"] = "sample, s8 coefficients"
    xs = np.arange(-128, 128, dtype=np.int64)
    luck = float((((xs[:, None] * xs[None, :]) * 2 ** 31) % 2 ** 32 == 0).mean())
    return {"mode": "single", "shape": list(shape), "rings": rows, "products": checked, "failures": fails,
            "luck_2^31_xy_vanishes": luck}


E2M1 = sorted({s * m for m in (0, 0.5, 1, 1.5, 2, 3, 4, 6) for s in (1, -1)})


def mode_r(args):
    out = {"mode": "r", "seed": args.seed}
    xs = np.arange(-128, 128, dtype=np.int64)
    worst = {}
    for a, b in itertools.product([c for c in range(-4, 5) if c], repeat=2):
        vals = a * xs[:, None] + b * xs[None, :]
        frac = float(((vals >= -128) & (vals <= 127)).mean())
        width = int(vals.max() - vals.min())
        if frac > worst.get("in_range", -1):
            worst = {"a": a, "b": b, "in_range": frac, "width": width}
        assert width >= 510
    out["integer_linear_two_inputs"] = {**worst, "min_width_at_least_510": True,
                                        "m16n8k32_A_fragment_in_range_log2": 512 * math.log2(worst["in_range"])}

    rng = np.random.default_rng(args.seed)
    n = args.n
    X = rng.integers(-128, 128, (n, n))
    Y = rng.integers(-128, 128, (n, n))
    T = X @ Y
    wrap8 = lambda t: ((t + 128) % 256) - 128
    a0, a1 = X[:, 0::2], X[:, 1::2]
    b0, b1 = Y[0::2, :], Y[1::2, :]
    ut = a0[:, :, None] + b1[None, :, :]
    vt = a1[:, :, None] + b0[None, :, :]
    u, v = wrap8(ut), wrap8(vt)
    assert u.min() >= -128 and u.max() <= 127
    jx = (a0 * a1).sum(axis=1)
    jy = (b0 * b1).sum(axis=0)
    W = (u * v).sum(axis=1) - jx[:, None] - jy[None, :]
    c1, c2 = (ut - u) // 256, (vt - v) // 256
    corr = 256 * (c1 * v + c2 * u).sum(axis=1) + 65536 * (c1 * c2).sum(axis=1)
    out["wrapped_winograd"] = {
        "n": n, "operands_in_s8": True, "mod_256_exact": bool(((W - T) % 256 == 0).all()),
        "mod_2^32_exact_without_corrections": float(((W - T) % 2 ** 32 == 0).mean()),
        "nonzero_carry_fraction": float((c1 != 0).mean()),
        "exact_with_corrections": bool(((W + corr - T) % 2 ** 32 == 0).all()),
        "int8_slots_mod_256": n ** 3 // 2 + n * n,
        "int8_slots_exact_carry_products_on_tensor_cores": n ** 3 // 2 + n ** 3 + n ** 3 // 2 + n * n,
        "int8_slots_exact_cc_on_cuda_cores": n ** 3 // 2 + n ** 3 + n * n,
        "plain_int8_slots": n ** 3}

    hx, lx = X >> 1, X & 1
    hy, ly = Y >> 1, Y & 1
    pair = (xs[:, None] >> 1) + (xs[None, :] >> 1)
    h0, h1 = hx[:, 0::2], hx[:, 1::2]
    g0, g1 = hy[0::2, :], hy[1::2, :]
    hu = h0[:, :, None] + g1[None, :, :]
    hv = h1[:, :, None] + g0[None, :, :]
    Hh = (hu * hv).sum(axis=1) - (h0 * h1).sum(axis=1)[:, None] - (g0 * g1).sum(axis=0)[None, :]
    S = X @ ly + lx @ (2 * hy)
    out["seven_bit_halves"] = {
        "operand_x_h_plus_y_h_range": [int(pair.min()), int(pair.max())],
        "exact": bool((4 * Hh + S == T).all()), "mixed_slots": n ** 3 // 2, "unmixed_slots": 2 * n ** 3,
        "junk_slots": n * n, "total_int8_slots": n ** 3 // 2 + 2 * n ** 3 + n * n,
        "P_weighted": 2 * (n ** 3 // 2) + 2 * n ** 3 + n * n, "floor": n ** 3}

    e2m1 = set(E2M1)
    bitsum_ok = all(float(i + j) in e2m1 for i in (0, 1) for j in (0, 1))
    pieces = None
    for t in (2, 3):
        for exps in itertools.product(range(-1, 7), repeat=t):
            sums = {0.0}
            for e in exps:
                sums = {s + (2.0 ** e) * d for s in sums for d in E2M1}
            if all(float(x) in sums for x in range(-128, 128)):
                pieces = {"pieces": t, "scales_log2": list(exps)}
                break
        if pieces:
            break
    out["fp4"] = {"e2m1_values": len(E2M1), "bit_sum_operand_exact": bitsum_ok,
                  "two_piece_bound": f"{len(E2M1)}^2 = {len(E2M1) ** 2} < 256 sums, so an s8 operand needs >= 3",
                  "s8_operand_pieces": pieces, "nvfp4_rate_over_s8": round(1614.0 / 807.5, 3),
                  "int8_slots_per_exact_s8_mac": (pieces["pieces"] ** 2) * 807.5 / 1614.0 if pieces else None}
    out["failures"] = int(not (out["wrapped_winograd"]["mod_256_exact"] and out["wrapped_winograd"][
        "exact_with_corrections"] and out["seven_bit_halves"]["exact"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["single", "search", "prop5", "r"])
    ap.add_argument("--shape", type=lambda s: [int(x) for x in s.split(",")], default=[2, 2, 2])
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--primes", type=lambda s: [int(x) for x in s.split(",")], default=[2, 3])
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--lifted", type=int, default=0, help="also inner-lifted Strassen, m = 1..LIFTED, paired")
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--climb", type=int, default=0, help="local-search steps from each computation")
    ap.add_argument("--mixed-weight", type=int, choices=[1, 2], default=2,
                    help="1 is the claim without P's factor 2, which Winograd's pairs must break (a self-test)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global MIXED_WEIGHT
    MIXED_WEIGHT = args.mixed_weight
    out = {"single": mode_single, "search": mode_search, "prop5": mode_prop5, "r": mode_r}[args.mode](args)
    print(json.dumps(out))
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
