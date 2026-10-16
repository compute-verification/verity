#!/usr/bin/env python3
"""Theorem E's one-bit-pair projection for the peeled low-rank P-noise edit, and its F_2 floor at real sizes
(RS-D2F round 2, CMP-13).

Part 1 (exact over Z/2^32). The prover commits P, V; coins give U in {+-1}^{s x r} (U = -1 + 2 beta), W, and
F in {+-1}^{s x d} (F = -1 + 2 beta'). Target T_t[i,l] = sum_{k <= tL} (P + U W^T)_ik (V + F)_kl mod 2^32. Its
projection at the pair (beta_{i'a}, beta'_{kl'}) is the second difference; Lemma 2's analogue says it equals
4 W_ka [i' = i][l' = l][k <= tL] for every P, V (wrapping values included). Mod 8 that is 4 (W_ka mod 2), so the
reduced targets carry W mod 2: rank 1 a block when W is in {+-1}, the rank of W's block when W is in {0, 1}.

Part 2 (F_2 bound at L = 32, S = 256). For W uniform on {0, 1} (no rejection), Theorem 1's bound with a singular
column map is sum_{t<S} dim(row W_t cap row W_{t+1}) + rank W_S per cell (restricted to the first L columns of W when
r > L), against the floor S min(r, L); with per-block rejection to full rank it is the floor exactly.

Part 3 (u1's holes at L = r = 32, S = 256). The same two fractions when each row k of an invertible-block W (the
stacked-identity W, or uniform invertible mod 2) is zeroed with probability 1/3, Proposition O's holes for one
column l of a {-1, 0, 1} noise.

Usage: peel_proj.py SEED
"""

import json
import sys

import numpy as np

M32 = 1 << 32


def targets(P, V, U, W, F, L):
    s, d = V.shape
    Pt = (P + U @ W.T) % M32
    Vt = (V + F) % M32
    out = np.zeros((s // L, s, d), dtype=object)
    acc = np.zeros((s, d), dtype=object)
    for t in range(s // L):
        blk = slice(t * L, (t + 1) * L)
        acc = (acc + Pt[:, blk].astype(object) @ Vt[blk, :].astype(object)) % M32
        out[t] = acc
    return out


def projection_check(rng, s, d, r, L, w_mode, inst):
    if inst == "zero":
        P = np.zeros((s, s), dtype=object)
        V = np.zeros((s, d), dtype=object)
    elif inst == "wrap":
        P = np.array(rng.integers(0, M32, size=(s, s)), dtype=object)
        V = np.array(rng.integers(0, M32, size=(s, d)), dtype=object)
    else:
        P = np.array(rng.integers(0, 64, size=(s, s)), dtype=object)
        V = np.array(rng.integers(-100, 101, size=(s, d)), dtype=object)
    W = (rng.integers(0, 2, size=(s, r)) if w_mode == "01" else 2 * rng.integers(0, 2, size=(s, r)) - 1).astype(object)
    U0 = -np.ones((s, r), dtype=object)
    F0 = -np.ones((s, d), dtype=object)
    base = targets(P, V, U0, W, F0, L)
    mism = 0
    checks = 0
    reduced = {}
    for i2 in range(s):
        for a in range(r):
            U1 = U0.copy()
            U1[i2, a] = 1
            tu = targets(P, V, U1, W, F0, L)
            for k in range(s):
                for l2 in range(d):
                    F1 = F0.copy()
                    F1[k, l2] = 1
                    tf = targets(P, V, U0, W, F1, L)
                    tuf = targets(P, V, U1, W, F1, L)
                    diff = (tuf - tu - tf + base) % M32
                    for t in range(s // L):
                        for i in range(s):
                            for l in range(d):
                                want = (4 * W[k, a] * (i == i2) * (l == l2) * (k < (t + 1) * L)) % M32
                                checks += 1
                                mism += int(diff[t, i, l] != want)
                                if i == i2 and l == l2:
                                    reduced[(t, i, l, a, k)] = (int(diff[t, i, l]) // 4) % 2
    ranks = []
    for t in range(s // L):
        m = np.array([[reduced[(t, 0, 0, a, k)] for k in range(s)] for a in range(r)], dtype=np.uint8)
        ranks.append(rank2_rows([int("".join(map(str, row[::-1])), 2) for row in m]))
    inc = []
    for t in range(s // L):
        m = np.array([[reduced[(t, 0, 0, a, k)] ^ (reduced[(t - 1, 0, 0, a, k)] if t else 0) for k in range(s)]
                      for a in range(r)], dtype=np.uint8)
        inc.append(rank2_rows([int("".join(map(str, row[::-1])), 2) for row in m]))
    return {"part": 1, "s": s, "d": d, "r": r, "L": L, "W": w_mode, "instance": inst, "checks": checks,
            "mismatches": mism, "cell00_prefix_ranks_F2": ranks, "cell00_increment_ranks_F2": inc}


def rank2_rows(rows):
    """Rank over F_2 of rows given as Python ints."""
    basis = []
    for v in rows:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def rowspace(rows):
    basis = []
    for v in rows:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
    return basis


def f2_bound(rng, r, L, S, trials):
    rr = min(r, L)
    vals, spiders, deltas = [], [], []
    for _ in range(trials):
        w = rng.integers(0, 2, size=(S * L, r))[:, :rr]
        blocks = [[int("".join(map(str, row[::-1])), 2) for row in w[t * L:(t + 1) * L]] for t in range(S)]
        bases = [rowspace(b) for b in blocks]
        rk = [len(b) for b in bases]
        inter = [rk[t] + rk[t + 1] - rank2_rows(bases[t] + bases[t + 1]) for t in range(S - 1)]
        vals.append((sum(inter) + rk[-1]) / (S * L))
        spiders.append(sum(rk) / (S * L))
        deltas.append(sum(rr - x for x in rk) / S)
    return {"part": 2, "r": r, "L": L, "S": S, "trials": trials, "floor_frac": rr / L,
            "bound_frac_mean": float(np.mean(vals)), "bound_frac_min": float(np.min(vals)),
            "spider_frac_mean": float(np.mean(spiders)), "mean_block_corank": float(np.mean(deltas))}


def holes_bound(rng, L, S, w_mode, trials):
    """r = L, invertible blocks (the stacked-identity W, or uniform invertible mod 2), each row k zeroed with
    probability 1/3 (u1's holes for one column l): the same bound and chain fractions as f2_bound."""
    vals, spiders = [], []
    for _ in range(trials):
        rows = []
        for t in range(S):
            if w_mode == "eye":
                b = [1 << a for a in range(L)]
            else:
                while True:
                    b = [int(x) for x in rng.integers(0, 1 << L, size=L, dtype=np.uint64)]
                    if rank2_rows(b) == L:
                        break
            keep = rng.random(L) >= 1 / 3
            rows.append([v for v, kp in zip(b, keep) if kp])
        bases = [rowspace(b) for b in rows]
        rk = [len(b) for b in bases]
        inter = [rk[t] + rk[t + 1] - rank2_rows(bases[t] + bases[t + 1]) for t in range(S - 1)]
        vals.append((sum(inter) + rk[-1]) / (S * L))
        spiders.append(sum(rk) / (S * L))
    return {"part": 3, "W": w_mode, "r": L, "L": L, "S": S, "trials": trials, "hole_prob": 1 / 3,
            "bound_frac_mean": float(np.mean(vals)), "bound_frac_min": float(np.min(vals)),
            "spider_frac_mean": float(np.mean(spiders))}


def main():
    rng = np.random.default_rng(int(sys.argv[1]))
    for w_mode in ("01", "pm"):
        for inst in ("random", "zero", "wrap"):
            print(json.dumps(projection_check(rng, 4, 2, 2, 2, w_mode, inst)), flush=True)
    for r in (16, 24, 28, 30, 31, 32, 33, 40, 64):
        print(json.dumps(f2_bound(rng, r, 32, 256, 20)), flush=True)
    for w_mode in ("eye", "full"):
        print(json.dumps(holes_bound(rng, 32, 256, w_mode, 20)), flush=True)


if __name__ == "__main__":
    main()
