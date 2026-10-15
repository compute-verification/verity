"""Pi_L's chunks on the int8 grid (round 3): the noise rank a chunk needs, and the relations across chunks the noise must
exclude.  Products are exact int64.  A rank is the largest of the ranks mod three primes below 2^31 (each is a lower
bound on the rank over Q, and a prime can divide a determinant: det(I - 2P) for a signed 31-cycle is 1 - 2^31).

    rank_cheat   R < L.  Noise E_A F_s^T of rank R per chunk lets a prover with A = 0 (or A of rank rho) compute every
                 committed prefix T_s at inner dimension R (+ rho), exactly.  Rank-L noise built as rank R plus a
                 sigma-sparse part is as cheap (R + sigma), so (R) read as a rank condition isn't sufficient.
    intervals    whether every interval sum Gamma_I = sum_{s in I} F_A,s^T F_B,s is nonsingular, per design
    pencils      chunk pairs (s, j): L - rank(G_s - lam G_j) for lam in {+-1, +-2}, and twin columns between chunks
    committed    adversarial inputs: rank(T_s - T_s') for every committed interval and rank(C_s - lam C_j), against the
                 rank of the matching Gamma
    cost         the rank price 4L/k + L/n per Llama-3.1-8B linear and MAC-weighted, and the peel's operand range

Designs: `perm` (one signed entry per row of F) and `pair` (two, grid.signed_pairs), each with F_A, F_B independent or
F_B = F_A ("shared").  Only full chunks of L columns are read.

    python chunks.py [--quick] [--peel] [--procs N] [--json FILE]      (--peel: only the peel's range and the cost)
    python chunks.py --L 128[,...] --procs N [--json FILE]              intervals, pencils, peel range and total at those L
"""
from __future__ import annotations

import itertools
import json
import multiprocessing
import sys

import numpy as np

from grid import DESIGNS, PRIMES, parallel_key, rank_mod_p

P3 = PRIMES + (2_147_483_587,)
U = 3
LLAMA = {"q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024), "o_proj": (4096, 4096),
         "gate_proj": (4096, 14336), "up_proj": (4096, 14336), "down_proj": (14336, 4096)}
WORK = 2e10


def modpow(a: np.ndarray, e: int, p: int) -> np.ndarray:
    r, a = np.ones_like(a), a % p
    while e:
        if e & 1:
            r = (r * a) % p
        a, e = (a * a) % p, e >> 1
    return r


def rank(a: np.ndarray) -> int:
    return max(rank_mod_p(a, p) for p in P3)


def nonsingular(x: np.ndarray) -> np.ndarray:
    """For a stack (b, L, L) of integer matrices, whether each is nonsingular mod one of two primes (so over Q)."""
    ok = nonsingular_mod(x, P3[0])
    ok[~ok] = nonsingular_mod(x[~ok], P3[1])
    return ok


def nonsingular_mod(x: np.ndarray, p: int) -> np.ndarray:
    b, n, _ = x.shape
    if not b:
        return np.zeros(0, dtype=bool)
    batch = max(16, int(4e6 // (n * n)))
    if b > batch:
        return np.concatenate([nonsingular_mod(x[i:i + batch], p) for i in range(0, b, batch)])
    x, ok, ar = np.asarray(x, dtype=np.int64) % p, np.ones(b, dtype=bool), np.arange(b)
    for c in range(n):
        nz = x[:, c:, c] != 0
        has, piv = nz.any(axis=1), c + nz.argmax(axis=1)
        ok &= has
        rows = x[ar, piv].copy()
        x[ar, piv] = x[:, c]
        x[:, c] = rows
        x[~has, c, c] = 1
        x[:, c] = (x[:, c] * modpow(x[:, c, c], p - 2, p)[:, None]) % p
        f = x[:, c + 1:, c].copy()
        x[:, c + 1:] = (x[:, c + 1:] - (f[:, :, None] * x[:, c, None, :]) % p) % p
    return ok


def noise_f(design: str, shared: bool, k: int, L: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fa = DESIGNS[design](k, L, rng)
    return fa, (fa if shared else DESIGNS[design](k, L, rng))


def grams(fa: np.ndarray, fb: np.ndarray, L: int) -> np.ndarray:
    c = fa.shape[0] // L
    return np.stack([fa[s * L:(s + 1) * L].T @ fb[s * L:(s + 1) * L] for s in range(c)])


def deficient(stack: np.ndarray, cap: int = 300) -> tuple[np.ndarray, list[int]]:
    """Indices singular mod the first prime, and the exact ranks of up to `cap` of them, evenly spread."""
    bad = np.nonzero(~nonsingular(stack))[0]
    pick = bad[np.linspace(0, len(bad) - 1, min(cap, len(bad))).astype(int)] if len(bad) else bad
    return bad, [rank(stack[i]) for i in pick]


def sample(n: int, L: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    cap = int(WORK // L ** 3)
    return (np.arange(n), False) if n <= cap else (np.sort(rng.choice(n, cap, replace=False)), True)


def intervals(design: str, shared: bool, k: int, L: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fa, fb = noise_f(design, shared, k, L, rng)
    g = grams(fa, fb, L)
    c = len(g)
    pre = np.concatenate([np.zeros((1, L, L), dtype=np.int64), np.cumsum(g, axis=0)])
    ab = np.array([(a, b) for a in range(c) for b in range(a + 1, c + 1)])
    idx, sampled = sample(len(ab), L, rng)
    ab = ab[idx]
    bad, ranks = deficient(pre[ab[:, 1]] - pre[ab[:, 0]])
    length = ab[:, 1] - ab[:, 0]
    out = {"design": design, "shared_F": shared, "k": k, "L": L, "chunks": c, "intervals": len(ab), "sampled": sampled,
           "singular": len(bad), "singular_share": round(len(bad) / len(ab), 5)}
    mask = np.zeros(len(ab), dtype=bool)
    mask[bad] = True
    for n in (1, 2, 3):
        sel = length == n
        out[f"singular_share_len{n}"] = round(float(mask[sel].mean()), 5) if sel.any() else None
    if ranks:
        out.update(min_rank=min(ranks), median_rank=int(np.median(ranks)), ranks_checked=len(ranks),
                   example={"chunks": [int(ab[bad[0], 0]), int(ab[bad[0], 1])], "rank": rank(pre[ab[bad[0], 1]]
                                                                                          - pre[ab[bad[0], 0]])})
    return out


def pencils(design: str, shared: bool, k: int, L: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fa, fb = noise_f(design, shared, k, L, rng)
    g = grams(fa, fb, L)
    c = len(g)
    pairs = np.array(list(itertools.combinations(range(c), 2)))
    out = {"design": design, "shared_F": shared, "k": k, "L": L, "chunks": c}
    for lam in (1, -1, 2, -2):
        idx, sampled = sample(len(pairs), L, rng)
        p = pairs[idx]
        bad, ranks = deficient(g[p[:, 1]] - lam * g[p[:, 0]])
        out[f"lam{lam:+d}"] = {"pairs": len(p), "sampled": sampled, "singular": len(bad),
                               "max_deficiency": (L - min(ranks)) if ranks else 0}
    keys: dict = {}
    for t in range(c * L):
        keys.setdefault((parallel_key(fa[t]), parallel_key(fb[t])), []).append(t // L)
    per_pair: dict = {}
    for ch in keys.values():
        for s, j in itertools.combinations(ch, 2):
            if s != j:
                per_pair[(s, j)] = per_pair.get((s, j), 0) + 1
    out["twin_columns"] = sum(per_pair.values())
    out["max_twins_one_chunk_pair"] = max(per_pair.values(), default=0)
    return out


def int8_inputs(m: int, k: int, L: int, rng: np.random.Generator) -> dict:
    top = 127 - 2 * U
    vec = rng.integers(-top, top + 1, k)
    return {"zero": np.zeros((m, k), dtype=np.int64),
            "rank1": np.outer(rng.choice((-1, 1), m), vec),
            "equal_columns": np.outer(rng.integers(-top, top + 1, m), np.ones(k, dtype=np.int64)),
            "repeated_chunks": np.tile(rng.integers(-top, top + 1, (m, L)), (1, k // L)),
            "gauss_int8": np.clip(np.rint(rng.standard_normal((m, k)) * 30), -top, top).astype(np.int64)}


def committed(design: str, shared: bool, m: int = 96, L: int = 32, c: int = 10, seed: int = 11) -> dict:
    """Ranks of every committed interval difference T_b - T_a and of C_s - lam C_j (lam = +-1), for each adversarial
    input on both sides, against rank(Gamma) of the same combination.  `below_gamma` counts combinations whose rank is
    below the Gamma rank, which Theorem X says happens with probability < 10^-20 here."""
    rng = np.random.default_rng(seed)
    k = c * L
    fa, fb = noise_f(design, shared, k, L, rng)
    g = grams(fa, fb, L)
    qa, qb = int8_inputs(m, k, L, rng), int8_inputs(m, k, L, rng)
    out = []
    for name in qa:
        ea, eb = rng.integers(-U, U + 1, (m, L)), rng.integers(-U, U + 1, (m, L))
        at, bt = qa[name] + ea @ fa.T, qb[name] + eb @ fb.T
        assert np.abs(at).max() <= 127 and np.abs(bt).max() <= 127
        ch = np.stack([at[:, s * L:(s + 1) * L] @ bt[:, s * L:(s + 1) * L].T for s in range(c)])
        pre = np.concatenate([np.zeros((1, m, m), dtype=np.int64), np.cumsum(ch, axis=0)])
        gpre = np.concatenate([np.zeros((1, L, L), dtype=np.int64), np.cumsum(g, axis=0)])
        rows = []
        for a in range(c):
            for b in range(a + 1, c + 1):
                rows.append(("interval", rank(pre[b] - pre[a]), rank(gpre[b] - gpre[a])))
        for j, s in itertools.combinations(range(c), 2):
            for lam in (1, -1):
                rows.append((f"chunk_lam{lam:+d}", rank(ch[s] - lam * ch[j]), rank(g[s] - lam * g[j])))
        for kind in ("interval", "chunk_lam+1", "chunk_lam-1"):
            sel = [r for r in rows if r[0] == kind]
            out.append({"design": design, "shared_F": shared, "input": name, "combination": kind, "count": len(sel),
                        "min_rank": min(r[1] for r in sel), "min_gamma_rank": min(r[2] for r in sel),
                        "below_gamma": sum(r[1] < r[2] for r in sel), "below_L": sum(r[1] < L for r in sel)})
        if design == "perm" and shared and name == "zero":
            out.append({"design": design, "shared_F": shared, "input": name, "attack": "T_s = s T_1",
                        "holds_exactly_every_s": bool(all((pre[s] == s * pre[1]).all() for s in range(1, c + 1)))})
    return {"design": design, "shared_F": shared, "m": m, "L": L, "chunks": c, "rows": out}


def rank_cheat(m: int = 128, n: int = 128, L: int = 32, c: int = 8, seed: int = 13) -> list[dict]:
    """Exact prefix sums T_s from a cheap factorization, for noise of rank R < L per chunk (A = 0 and A of rank 1), and
    for rank-L noise that is rank R plus sigma-sparse.  MACs are per chunk: the honest m n L against the cheat's
    m n (R + rho) for E_A H_s, L n for H_s = F_s^T B~_s^T (one signed entry per row of F_s), rho L n for the input's
    factor, and sigma m n for the sparse part."""
    rng = np.random.default_rng(seed)
    k = c * L
    fb = DESIGNS["pair"](k, L, rng)
    qb = np.clip(np.rint(rng.standard_normal((n, k)) * 30), -121, 121).astype(np.int64)
    bt = qb + rng.integers(-U, U + 1, (n, L)) @ fb.T
    out = []
    for R, sigma, rho in ((8, 0, 0), (16, 0, 0), (16, 0, 1), (31, 0, 0), (8, 4, 0), (16, 8, 0)):
        e = rng.integers(-U, U + 1, (m, R))
        f = np.zeros((k, R), dtype=np.int64)
        for s in range(c):
            col = np.concatenate([rng.permutation(R), rng.integers(0, R, L - R)])[rng.permutation(L)]
            f[s * L + np.arange(L), col] = rng.choice((-1, 1), L)
        sp = np.zeros((m, k), dtype=np.int64)
        if sigma:
            for s in range(c):
                for i in range(m):
                    sp[i, s * L + rng.choice(L, sigma, replace=False)] = rng.choice((-1, 1), sigma)
        u, v = rng.choice((-1, 1), (m, rho)), rng.integers(-100, 101, (k, rho))
        at = u @ v.T + e @ f.T + sp
        honest = np.cumsum(np.stack([at[:, s * L:(s + 1) * L] @ bt[:, s * L:(s + 1) * L].T for s in range(c)]), axis=0)
        cheat, t, macs = [], np.zeros((m, n), dtype=np.int64), 0
        for s in range(c):
            sl = slice(s * L, (s + 1) * L)
            h = f[sl].T @ bt[:, sl].T
            t = t + e @ h + u @ (v[sl].T @ bt[:, sl].T)
            nz = np.nonzero(sp[:, sl])
            for i, tt in zip(*nz):
                t[i] += sp[i, s * L + tt] * bt[:, s * L + tt]
            cheat.append(t.copy())
            macs = m * n * (R + rho) + (1 + rho) * L * n + len(nz[0]) * n
        out.append({"R": R, "sigma": sigma, "input_rank": rho, "L": L, "m": m, "n": n, "chunks": c,
                    "chunk_rank_of_A_tilde": min(rank(at[:, s * L:(s + 1) * L]) for s in range(c)),
                    "prefix_sums_exact": bool((np.stack(cheat) == honest).all()),
                    "honest_macs_per_chunk": m * n * L, "cheat_macs_per_chunk": macs,
                    "cheat_share": round(macs / (m * n * L), 4)})
    return out


def peel_range(L: int, k: int, shared: bool = True, m: int = 64, seed: int = 17) -> dict:
    """The pair design: the peel's wide operands on Gaussian int8 rows, their worst case over all int8 inputs, and
    U = Q(A) Q(B)^T exactly."""
    rng = np.random.default_rng(seed)
    fa, fb = noise_f("pair", shared, k, L, rng)
    qa, qb = (np.clip(np.rint(rng.standard_normal((m, k)) * 30), -121, 121).astype(np.int64) for _ in range(2))
    ea, eb = rng.integers(-U, U + 1, (m, L)), rng.integers(-U, U + 1, (m, L))
    at, bt = qa + ea @ fa.T, qb + eb @ fb.T
    g = fb.T @ fa
    pa, qq = at @ fb, bt @ fa - eb @ g
    u_ = at @ bt.T - pa @ eb.T - ea @ qq.T
    worst = 127 * int(max(np.abs(fa).sum(axis=0).max(), np.abs(fb).sum(axis=0).max())) + U * int(np.abs(g).sum(axis=0).max())
    return {"L": L, "k": k, "shared_F": shared, "U_exact": bool((u_ == qa @ qb.T).all()), "max_abs_operand": int(max(np.abs(pa).max(),
            np.abs(qq).max())), "worst_case_operand": worst, "two_limbs_hold": 32767,
            "limbs_measured": 1 + int(max(np.abs(pa).max(), np.abs(qq).max()) > 127) + int(
                max(np.abs(pa).max(), np.abs(qq).max()) > 32767), "limbs_worst": 1 + int(worst > 127) + int(worst > 32767)}


def cost(L: int) -> dict:
    """Extra tensor work per useful MAC at R = L with two-limb peel operands: 4L/k (peel) + L/n (A~ F on the tensor
    cores); each linear an m x n x k product with k its input width."""
    per = {name: round(4 * L / k + L / n, 5) for name, (k, n) in LLAMA.items()}
    w = {name: k * n for name, (k, n) in LLAMA.items()}
    return {"L": L, "square_4096": round(4 * L / 4096 + L / 4096, 5), "per_linear": per,
            "llama8b_mac_weighted": round(sum(per[x] * w[x] for x in per) / sum(w.values()), 5)}


#: duo fused into the INT8 mma.sync loop, serial accumulators (note:20261015T0432Z-draft-receipt-phi): the loop's MMA rate
#: over the bare loop's, measured at each stride, and the cost in slots of tensor time a word (the median over L = 128-1024,
#: 1.68 with ping-pong accumulators; 0.4 is the Merkle term, the receipt page's B200 figure, unmeasured on sm_120)
DUO_RATE = {32: 0.686, 64: 0.797, 128: 0.923, 256: 0.960, 512: 0.980, 1024: 0.989}
DUO_H, DUO_H_PINGPONG, MERKLE_H = 1.33, 1.68, 0.4


def total(L: int) -> dict:
    """Extra tensor time per useful MAC at stride L: the peel (`cost`) plus the receipts, 1/rate - 1 for duo's measured rate
    at L (one receipt word per L MACs, so 8h/L for h slots a word at 1024 MACs over 128 slots), with the 8h/L model at the
    median and ping-pong h and the Merkle term beside it.  ``break_even_h`` is the h above which L beats L/2 on that linear."""
    peel = cost(L)
    receipts = 1 / DUO_RATE[L] - 1
    lines = dict(peel["per_linear"], square_4096=peel["square_4096"], llama8b_mac_weighted=peel["llama8b_mac_weighted"])
    shapes = dict(LLAMA, square_4096=(4096, 4096))
    return {"L": L, "receipts_measured": round(receipts, 5), "receipts_model": round(8 * DUO_H / L, 5),
            "receipts_model_pingpong": round(8 * DUO_H_PINGPONG / L, 5), "merkle_unmeasured": round(8 * MERKLE_H / L, 5),
            "peel": lines, "total": {x: round(p + receipts, 5) for x, p in lines.items()},
            "total_with_merkle": {x: round(p + receipts + 8 * MERKLE_H / L, 5) for x, p in lines.items()},
            "break_even_h": {x: round(L * L * (4 / k + 1 / n) / 16, 4) for x, (k, n) in shapes.items()}}


def job(args: tuple) -> dict:
    kind, *rest = args
    return {"intervals": intervals, "pencils": pencils}[kind](*rest)


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    procs = int(argv[argv.index("--procs") + 1]) if "--procs" in argv else 1
    strides = [int(x) for x in argv[argv.index("--L") + 1].split(",")] if "--L" in argv else None
    peels = [peel_range(L, k, sh) for sh in (False, True) for L in strides or (32, 64)
             for k in ((1024,) if quick else (4096, 14336))]
    if strides:
        jobs = [(kind, d, sh, k, L, 3) for kind in ("intervals", "pencils") for d in DESIGNS for sh in (False, True)
                for L in strides for k in ((256,) if quick else (4096, 14336))]
        with multiprocessing.Pool(procs) as pool:
            done = pool.map(job, jobs, chunksize=1)
        text = json.dumps({"intervals": [d for j, d in zip(jobs, done) if j[0] == "intervals"],
                           "pencils": [d for j, d in zip(jobs, done) if j[0] == "pencils"], "peel_range": peels,
                           "total": [total(L) for L in strides if L in DUO_RATE]}, indent=1)
        if "--json" in argv:
            open(argv[argv.index("--json") + 1], "w").write(text)
        print(text)
        return 0
    if "--peel" in argv:
        text = json.dumps({"peel_range": peels, "cost": [cost(L) for L in (32, 64, 170)],
                           "total": [total(L) for L in (32, 64, 128, 256)]}, indent=1)
        if "--json" in argv:
            open(argv[argv.index("--json") + 1], "w").write(text)
        print(text)
        return 0
    ls, ks = ((32,), (256,)) if quick else ((32, 64, 170), (4096, 14336))
    jobs = [(kind, d, sh, k, L, 3) for kind in ("intervals", "pencils") for d in DESIGNS for sh in (False, True)
            for L in ls for k in ks]
    if procs > 1:
        with multiprocessing.Pool(procs) as pool:
            done = pool.map(job, jobs, chunksize=1)
    else:
        done = [job(j) for j in jobs]
    res = {"rank_cheat": rank_cheat(c=4 if quick else 8),
           "intervals": [d for j, d in zip(jobs, done) if j[0] == "intervals"],
           "pencils": [d for j, d in zip(jobs, done) if j[0] == "pencils"],
           "committed": [committed(d, sh, m=48 if quick else 96, c=4 if quick else 10) for d in DESIGNS
                         for sh in (False, True)],
           "peel_range": peels,
           "cost": [cost(L) for L in (32, 64, 170)]}
    text = json.dumps(res, indent=1)
    if "--json" in argv:
        open(argv[argv.index("--json") + 1], "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
