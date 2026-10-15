"""Question #9: exact ranks of Pearl-C's noised chunks A~_{:,s} (m x L) and B~_{s,:} (L x n) on sm120, against condition (R).

Condition (R) ("Three lemmas", the KSW low-rank shortcut): min(rank A~_{:,s}, rank B~_{s,:}) >= L for every chunk s and every
(A, B).  Pearl-C v1 forms A~_ik = e4m3(atom(alpha_i x_ik; E'_i, F_k)) (`pearl_c.form_v1`): the noise N = E'F^T has rank
<= R = 32, and the atom's truncation and the E4M3 cast add a rounding term.  For each family of rows this reports, per L,
the rank of

    pearl_c         A~_{:,s} as formed on sm120;
    rounding_free   Q(alpha x) + E'F^T in exact arithmetic (Q the E4M3 cast): the KSW form A + UV^T the lemma's proof reads;
    noise           E'F^T alone;
    input           Q(alpha x) alone.

Ranks are over Q, computed mod two primes below 2^31 (a rank mod p is at most the rank over Q, and equal unless p divides
every maximal nonzero minor): every E4M3 code is an integer multiple of 2^-9, so 2^9 A~ and 2^18 (Q(alpha x) + E'F^T) are
integer matrices.  `spectrum` gives the formed chunk's singular values at 1, 32, 33 and L, over the first, in float64.

    python rank.py [--rows M] [--jobs J] [--json FILE]
"""
from __future__ import annotations

import json
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from verity.core.protocols.pouw.schemes import pearl_c as PC
from verity.core.protocols.pouw.schemes.pearl_c_device import SM120
from verity.ml.circuits.silicon import fp32

P = PC.P
PRIMES = (2147483629, 2147483587)
LS = (16, 32, 64, 128, 170, 256)
K = 8192
S0 = 1024                        # the chunk's first column: chunks start on a multiple of 32, rho's positions every 8th


def rank_mod(rows: list[list[int]], p: int) -> int:
    m = np.array([[v % p for v in r] for r in rows], dtype=np.int64)
    rank = 0
    for c in range(m.shape[1]):
        if rank == m.shape[0]:
            break
        piv = np.nonzero(m[rank:, c])[0]
        if piv.size == 0:
            continue
        i = rank + int(piv[0])
        if i != rank:
            m[[rank, i]] = m[[i, rank]]
        m[rank] = (m[rank] * pow(int(m[rank, c]), p - 2, p)) % p
        below = m[rank + 1:, c]
        nz = np.nonzero(below)[0]
        if nz.size:
            idx = rank + 1 + nz
            m[idx] = (m[idx] - (below[nz, None] * m[rank]) % p) % p
        rank += 1
    return rank


def exact_rank(rows: list[list[int]]) -> int | list[int]:
    r = [rank_mod(rows, p) for p in PRIMES]
    return r[0] if r[0] == r[1] else r


def code_int(c: int) -> int:
    """An E4M3 code's value as an integer multiple of 2^-9."""
    v = P.fp8_to_f32(c) * 512
    assert v == int(v)
    return int(v)


# ---- row families (FP32 words, k = K) ---------------------------------------------------------------------------------

def _w(xs) -> list[int]:
    return [P.f32_bits(float(v)) for v in xs]


def family_rows(name: str, m: int, rng: random.Random, side: int) -> list[list[int]]:
    g = rng.gauss
    if name == "gauss":
        return [_w(g(0, 1) for _ in range(K)) for _ in range(m)]
    if name == "weights":
        return [_w(g(0, 0.02) for _ in range(K)) for _ in range(m)]
    if name == "outliers":                                    # 1% of channels 30x, the same channels in every row
        return [_w(g(0, 1) * (30 if c % 100 == 7 else 1) for c in range(K)) for _ in range(m)]
    if name == "relu":
        return [_w(max(0.0, g(0, 1)) for _ in range(K)) for _ in range(m)]
    if name == "near_zero":                                   # A = 0 is outside the domain (rho = 0); this is its floor
        return [_w((rng.choice((-1.0, 1.0)) if p % 8 == 0 else 0.0) for p in range(K)) for _ in range(m)]
    if name in ("rank1", "rank8"):
        r = 1 if name == "rank1" else 8
        v = [[g(0, 1) for _ in range(K)] for _ in range(r)]
        out = []
        for _ in range(m):
            u = [g(0, 1) for _ in range(r)]
            out.append(_w(sum(u[j] * v[j][p] for j in range(r)) for p in range(K)))
        return out
    if name == "identical":
        row = _w(g(0, 1) for _ in range(K))
        return [list(row) for _ in range(m)]
    if name in ("dead_chunk", "dead_chunk_wide"):
        # rho's positions +-1 (rho = 1); entries at |x| = 443 >= 8 rho inside the chunk, in the same columns for every row
        # with a rank-1 sign pattern u_i w_t; every other entry 0.  dead_chunk: the liveness rule's k/64 = 128 per row at
        # k = 8192.  dead_chunk_wide: every one of the chunk's 224 non-rho columns, which the rule admits from k = 64 * 224
        # = 14336 on; s, rho, alpha and beta don't depend on k here, so its chunk is the one such a longer row forms
        n_dead = K // 64 if name == "dead_chunk" else 224
        cols = [S0 + t for t in range(256) if (S0 + t) % 8][:n_dead]
        w = {c: rng.choice((-1.0, 1.0)) for c in cols}
        out = []
        for _ in range(m):
            u = rng.choice((-1.0, 1.0))
            x = [(rng.choice((-1.0, 1.0)) if p % 8 == 0 else 0.0) for p in range(K)]
            for c in cols:
                x[c] = 443.0 * u * w[c]
            out.append(_w(x))
        return out
    raise ValueError(name)


# ---- forming the chunk ------------------------------------------------------------------------------------------------

def form_chunk(name: str, side: int, m: int, seed: int) -> dict:
    rng = random.Random(seed)
    rows = family_rows(name, m, rng, side)
    seed_b = PC.S512Seed(rng.randbytes(32))
    seed_e = PC.S512Seed(rng.randbytes(32)) if side == 0 else seed_b
    cols = range(S0, S0 + max(LS))
    f = {t: PC.sample_line(seed_b, side, 1, t, PC.LINE_NORM_V1) for t in cols}
    formed, free, noise, inp = [], [], [], []
    admissible = 0
    for i, x in enumerate(rows):
        s, rho, alpha, beta = PC.s5_stats(x)
        admissible += (P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN) and PC.live_row(x)
        ep = PC.eprime(beta, PC.sample_line(seed_e, side, 0, i))
        epi = [code_int(c) for c in ep]
        fr, fe, no, ip = [], [], [], []
        for t in cols:
            v = fp32.mul(alpha, x[t])
            fr.append(code_int(P.f32_to_fp8(P.f32_of_bits(SM120.atom.step(v, ep, f[t])))))
            n = sum(a * code_int(b) for a, b in zip(epi, f[t]))             # at 2^-18
            q = code_int(P.f32_to_fp8(P.f32_of_bits(v)))                    # at 2^-9
            fe.append(q * 512 + n)
            no.append(n)
            ip.append(q)
        formed.append(fr)
        free.append(fe)
        noise.append(no)
        inp.append(ip)
    out = {"family": name, "side": "A" if side == 0 else "B", "rows": m, "admissible_rows": admissible, "ranks": {}}
    for L in LS:
        sub = lambda mat: [r[:L] for r in mat]
        a = np.array(sub(formed), dtype=np.float64)
        sv = np.linalg.svd(a, compute_uv=False)
        top = sv[0] if sv[0] else 1.0
        out["ranks"][L] = {"pearl_c": exact_rank(sub(formed)), "rounding_free": exact_rank(sub(free)),
                           "noise": exact_rank(sub(noise)), "input": exact_rank(sub(inp)),
                           "spectrum": [round(float(sv[j] / top), 5) for j in (0, 31, 32, min(L, m) - 1) if j < len(sv)]}
    return out


CASES = [("gauss", 0), ("outliers", 0), ("relu", 0), ("near_zero", 0), ("rank1", 0), ("rank8", 0), ("identical", 0),
         ("dead_chunk", 0), ("dead_chunk_wide", 0), ("weights", 1), ("near_zero", 1), ("rank1", 1), ("dead_chunk", 1)]


def _run(args):
    name, side, m, seed = args
    t0 = time.time()
    r = form_chunk(name, side, m, seed)
    r["seconds"] = round(time.time() - t0, 1)
    return r


def main(argv: list[str]) -> int:
    m = int(argv[argv.index("--rows") + 1]) if "--rows" in argv else 512
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    cases = [(name, side, m, 1000 + i) for i, (name, side) in enumerate(CASES)]
    if "--only" in argv:
        keep = argv[argv.index("--only") + 1].split(",")
        cases = [c for c in cases if c[0] in keep]
    if jobs > 1:
        with Pool(jobs) as pool:
            res = pool.map(_run, cases)
    else:
        res = [_run(c) for c in cases]
    text = json.dumps({"k": K, "chunk_start": S0, "primes": PRIMES, "cases": res}, indent=1)
    if "--json" in argv:
        open(argv[argv.index("--json") + 1], "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
