"""Pi_L on an exact integer grid (round 2, question 2): int8 operands with s32 accumulation, noise N = E F^T with E's entries
uniform on {-u, ..., u} and F a signed permutation on every chunk of L columns (rank R = L, one nonzero per row of F).

    noise     the entry point mass 1/(2u+1) (and the dense +-1 alternative's C(T, T/2)/2^T), the range |N| <= u
    rank      (R) exactly, mod a prime: rank of a chunk Q(A)_s + E P_s^T for adversarial A (zero, rank 1, repeated
              chunks), and of two chunks together
    peel      U = C~ - (A~ F_B) E_B^T - E_A (B~ F_A - E_B G^T)^T with G = F_B^T F_A, in int64, against Q(A) Q(B)^T exactly,
              and the int8 limbs the two peel products need

    python grid.py [--json FILE]
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

PRIMES = (2_147_483_647, 2_147_483_629)


def rank_mod_p(a: np.ndarray, p: int) -> int:
    m = np.asarray(a, dtype=np.int64) % p
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(m[r:, c])[0]
        if not len(nz):
            continue
        piv = r + nz[0]
        m[[r, piv]] = m[[piv, r]]
        m[r] = (m[r] * pow(int(m[r, c]), p - 2, p)) % p
        col = m[:, c].copy()
        col[r] = 0
        m = (m - (col[:, None] * m[r][None, :]) % p) % p
        r += 1
        if r == rows:
            break
    return r


def rank(a: np.ndarray) -> int:
    """The rank over Q, as the rank mod two primes below 2^31 (which agree, or the call fails)."""
    r = {rank_mod_p(a, p) for p in PRIMES}
    assert len(r) == 1, r
    return r.pop()


def signed_perms(k: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """F (k x L), design `perm`: row t of chunk s has one nonzero, +-1, at column perm_s(t - s); each chunk's map injective."""
    f = np.zeros((k, L), dtype=np.int64)
    for s in range(0, k, L):
        n = min(L, k - s)
        f[np.arange(s, s + n), rng.permutation(L)[:n]] = rng.choice((-1, 1), n)
    return f


def signed_pairs(k: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """F (k x L), design `pair`: row t of chunk s is a_t e_{p1(t)} + b_t e_{p2(t)} for two independent permutations p1,
    p2 of the chunk with p1(t) != p2(t).  A full chunk is invertible iff no cycle C of p1^-1 p2 has prod_C a_t b_t =
    (-1)^|C| (its two permutation terms cancel), so one b per such cycle is flipped; a partial chunk is redrawn until its
    rows are independent.  Every chunk's rank is checked mod a prime."""
    f = np.zeros((k, L), dtype=np.int64)
    for s in range(0, k, L):
        n = min(L, k - s)
        while True:
            p1, p2 = rng.permutation(L)[:n], rng.permutation(L)[:n]
            if (p1 == p2).any():
                continue
            a, b = rng.choice((-1, 1), n), rng.choice((-1, 1), n)
            if n == L:
                tau, seen = np.argsort(p1)[p2], np.zeros(n, dtype=bool)
                for t0 in range(n):
                    if seen[t0]:
                        continue
                    cyc, t = [], t0
                    while not seen[t]:
                        seen[t] = True
                        cyc.append(t)
                        t = tau[t]
                    if np.prod(a[cyc] * b[cyc]) == (-1) ** len(cyc):
                        b[cyc[0]] *= -1
            blk = np.zeros((n, L), dtype=np.int64)
            blk[np.arange(n), p1], blk[np.arange(n), p2] = a, b
            if rank_mod_p(blk, PRIMES[0]) == n:
                break
            assert n < L, "a full chunk's cycle condition should make it invertible"
        f[s:s + n] = blk
    return f


DESIGNS = {"perm": signed_perms, "pair": signed_pairs}


def parallel_key(row: np.ndarray) -> tuple:
    nz = np.nonzero(row)[0]
    return tuple(nz) + tuple(row[nz] * row[nz[0]])


def twins(k: int, L: int, design: str, seed: int) -> dict:
    """Column pairs (t, t') whose noise is the same up to sign on both sides (F_A rows parallel and F_B rows parallel):
    with inputs whose columns t and t' agree, the two products are equal for every seed of E, so one can stand in for the
    other.  The share of products a cheater could skip this way is pairs / k."""
    rng = np.random.default_rng(seed)
    fa, fb = DESIGNS[design](k, L, rng), DESIGNS[design](k, L, rng)
    both: dict = {}
    for t in range(k):
        both.setdefault((parallel_key(fa[t]), parallel_key(fb[t])), []).append(t)
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in both.values())
    a_only: dict = {}
    for t in range(k):
        a_only.setdefault(parallel_key(fa[t]), []).append(t)
    return {"design": design, "k": k, "L": L, "twin_pairs": pairs, "skippable_share": round(pairs / k, 5),
            "a_side_parallel_pairs": sum(len(v) * (len(v) - 1) // 2 for v in a_only.values())}


def noise_table() -> dict:
    out = {"uniform_E": [], "pm1_E_T_sparse_F": []}
    for u in (1, 2, 3, 4, 8):
        p = 1 / (2 * u + 1)
        out["uniform_E"].append({"u": u, "range_perm": u, "range_pair": 2 * u, "p_E_variable": round(p, 4),
                                 "p_entry_perm": round(p, 4), "p_entry_pair": round(p, 4),
                                 "delta_d_SZ_degree2": round(2 * p, 4)})
    for t in (8, 16, 24, 32, 64):
        p = math.comb(t, t // 2) / 2 ** t
        out["pm1_E_T_sparse_F"].append({"T": t, "range": t, "p_entry": round(p, 4), "p_E_variable": 0.5,
                                        "delta_d_SZ_degree2_in_E": 1.0, "two_p_entry": round(2 * p, 4)})
    return out


def rank_table(m: int = 256, L: int = 170, u: int = 3, seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    k = 2 * L
    vec = rng.integers(-121, 122, k)
    inputs = {"zero": np.zeros((m, k), dtype=np.int64),
              "rank1": np.outer(rng.choice((-1, 1), m), vec),
              "equal_columns": np.outer(rng.integers(-121, 122, m), np.ones(k, dtype=np.int64)),
              "repeated_chunks": np.tile(rng.integers(-121, 122, (m, L)), (1, 2)),
              "gauss_int8": np.clip(np.rint(rng.standard_normal((m, k)) * 30), -121, 121).astype(np.int64)}
    for design, make in DESIGNS.items():
        f = make(k, L, rng)
        for name, a in inputs.items():
            e = rng.integers(-u, u + 1, (m, L))
            at = a + e @ f.T
            assert np.abs(at).max() <= 127
            out.append({"design": design, "input": name, "m": m, "L": L, "chunk_rank": rank(at[:, :L]),
                        "input_chunk_rank": rank(a[:, :L]), "two_chunk_rank": rank(at), "input_two_chunk_rank": rank(a)})
    return out


def peel_check(design: str, m: int = 64, n: int = 64, k: int = 4096, L: int = 170, u: int = 3, seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    top = 127 - u * (2 if design == "pair" else 1)
    qa = np.clip(np.rint(rng.standard_normal((m, k)) * 30), -top, top).astype(np.int64)
    qb = np.clip(np.rint(rng.standard_normal((n, k)) * 30), -top, top).astype(np.int64)
    fa, fb = DESIGNS[design](k, L, rng), DESIGNS[design](k, L, rng)
    ea, eb = rng.integers(-u, u + 1, (m, L)), rng.integers(-u, u + 1, (n, L))
    at, bt = qa + ea @ fa.T, qb + eb @ fb.T
    c = at @ bt.T
    pa = at @ fb
    qq = bt @ fa - eb @ (fb.T @ fa)
    u_ = c - pa @ eb.T - ea @ qq.T
    worst = 127 * int(max(np.abs(fb).sum(axis=0).max(), np.abs(fa).sum(axis=0).max())) + u * L * int(np.abs(fb.T @ fa).max())
    return {"design": design, "k": k, "L": L, "operands_within_int8": bool(max(np.abs(at).max(), np.abs(bt).max()) <= 127),
            "max_abs_C_tilde": int(np.abs(c).max()), "k_max_s32_no_overflow": (2 ** 31 - 1) // (127 * 127),
            "U_equals_QA_QB_exactly": bool((u_ == qa @ qb.T).all()),
            "max_abs_peel_operand": int(max(np.abs(pa).max(), np.abs(qq).max())), "worst_case_peel_operand": worst,
            "int8_limbs_worst_case": int(math.ceil((math.log2(worst + 1) + 1) / 8))}


def cost_table() -> list[dict]:
    """Tensor slots per useful MAC of an m x n x k unit, beyond the product itself (int8 = FP8 = 1 slot per MAC on sm120,
    BF16 = 2; the B side is formed once per job and left out):
      peel      two products of inner dimension R per output: BF16 (Pearl-C's) or exact in two int8 limbs, 4R/k either way
      A' F_B    m x k x R on the tensor cores, R/n
      forming   FP8: ceil(R/32) atoms of 32 per entry, 32 ceil(R/32)/n; the int8 grid: one gather-add per entry, no MMA."""
    out = []
    for r in (32, 64, 170):
        for k, n in ((4096, 4096), (4096, 14336), (14336, 4096)):
            out.append({"R": r, "k": k, "n": n,
                        "int8_grid": round(4 * r / k + r / n, 4),
                        "fp8_dense_lines": round(4 * r / k + (r + 32 * math.ceil(r / 32)) / n, 4)})
    return out


def main(argv: list[str]) -> int:
    res = {"noise": noise_table(), "rank": rank_table(), "peel": [peel_check(d) for d in DESIGNS],
           "twins": [twins(k, 170, d, 3) for d in DESIGNS for k in (4096, 14336)], "cost": cost_table()}
    text = json.dumps(res, indent=1)
    if "--json" in argv:
        open(argv[argv.index("--json") + 1], "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
