"""ACCEL PROTO-DESIGN (CMP-118): the receipt with no fold under R-ACCEL-6, on the H200, per matmul of the frozen model.

Every 4-byte word (a prefix sum of the noised product every L inner indices) is a SHA-512 Merkle leaf; binding is
collision resistance plus Theorem S's draws. Costs are in plain INT8 tensor time at η (`score.py`'s convention):

- hashing: 32 words a 1024-bit leaf block, one tree node a leaf (`score.commit`'s 2·⌈b/1024⌉), at the measured
  4.79 G compressions/s, so γ_hash = C/L with C = 2·η·MAC_RATE/(32·SHA) ≈ 11,000 (the leaves alone at the data-sheet
  rate are 6460/L, the coordinator's figure);
- the peel: noise of rank r ≥ L costs γ_rm = r/K, and 2r/K if both sides carry noise (HYB's U and V);
- serving skip: when the cheater knows the served output (its own workload, A = 0), one segment's words follow from
  the output and the other segments, so the longest segment, L MACs of every output's γ·K, is skippable unchecked;
  ε_skip = L/(γK). The peel must sit inside the words (extended inner dimension [Ã | −ÃΠ], [B̃; Uᵀ]); an unchecked
  peel adds γ_rm/γ to ε_skip.

Under an ε_skip budget e, γ solves γ = 1 + a + C/L + ρL/K with L = eγK; `solve` returns it. h = γ_hash/γ is the
share an r-times faster SHA-512 frees h·(1 − 1/r) of. The proof unit is one word's segment: 370 ANDs a MAC
(BET-COVERAGE's lane chunk) plus in-circuit SHA-512 for the operand leaves and five Merkle paths.

    uv run python benchmarks/pouw/accel_score/nofold.py
"""
from __future__ import annotations

import math

MAC_RATE = 1978.9e12 / 2
ETA = 0.85
SHA = 4.79e9
C_LEAF = MAC_RATE / (32 * SHA)
C = 2 * ETA * MAC_RATE / (32 * SHA)
A_COMMIT = 0.17            # operand A's commitments at n = 16384, common to every row
AND_PER_MAC = 48.2e6 / 131072
AND_PER_SHA = 5.8e4
UNIT_LIMIT = 2 ** 27
LN = {"1e-6": math.log(1e6), "1%": math.log(100)}

STATIC = [("qkv", 16384, 16512), ("o", 16384, 16384), ("up", 16384, 65536), ("down", 65536, 16384)]
STATIC_MACS = sum(k * n for _, k, n in STATIC)   # a token a layer
ATTN_WIDTH = 16384                                # heads × d_head on the query side


def gamma(K: float, L: float, rho: float = 1.0) -> float:
    return 1 + A_COMMIT + C / L + rho * L / K


def solve(K: float, e: float, rho: float = 1.0) -> tuple[float, float]:
    """(γ, L) with L = eγK, capped at the unconstrained optimum √(CK/ρ) and at K."""
    a = 1 + A_COMMIT
    q = (1 - rho * e)
    g = (a + math.sqrt(a * a + 4 * q * C / (e * K))) / (2 * q)
    L = e * g * K
    L_opt = min(math.sqrt(C * K / rho), K)
    if L > L_opt:
        L = L_opt
        g = gamma(K, L, rho)
    return g, L


def unit_ands(L: float) -> float:
    return AND_PER_MAC * L + (2 * L / 128 + 5 * 40) * AND_PER_SHA


def draws(eps: float, delta: str) -> int:
    return math.ceil(LN[delta] / eps)


def main() -> None:
    print(f"C = {C:.0f} (leaves only, data sheet: {C_LEAF:.0f}); words 4 B, tree x2, plain at eta {ETA}")
    print("\ncoordinator's points (unchecked peel), gamma = 1 + A + C/L + L/K:")
    for K, L in [(16384, 8192), (65536, 16384)]:
        gh, gr = C / L, L / K
        g = 1 + A_COMMIT + gh + gr
        print(f"  K {K} L {L}: gamma_hash {gh:.2f} (coord {6460 / L:.2f}) gamma_rm {gr:.2f} gamma {g:.2f} "
              f"h {gh / g:.2f} eps_skip unchecked peel {(gr + L / K) / g:.1%}, peel inside {L / (g * K):.1%}")

    print("\nper static shape, peel inside the words; e = eps_skip budget; rho = 1 (one side) | 2 (U and V)")
    print(f"{'shape':6} {'K':>6} {'N':>6} {'e':>5} {'rho':>3} {'L':>6} {'g_hash':>6} {'g_rm':>5} {'gamma':>6} "
          f"{'h':>5} {'unit ANDs':>9} {'fits':>4}")
    for name, K, N in STATIC:
        for e in (0.005, 0.01):
            for rho in (1, 2):
                g, L = solve(K, e, rho)
                u = unit_ands(L)
                print(f"{name:6} {K:6} {N:6} {e:5.3f} {rho:3} {L:6.0f} {C / L:6.2f} {rho * L / K:5.2f} {g:6.2f} "
                      f"{C / L / g:5.2f} {u:9.2e} {'yes' if u < UNIT_LIMIT else 'no':>4}")

    w = {K: sum(k * n for _, k, n in STATIC if k == K) / STATIC_MACS for K in (16384, 65536)}
    print(f"\nMAC shares: K=16384 {w[16384]:.1%}, K=65536 {w[65536]:.1%}")
    for e in (0.005, 0.01):
        gs = {K: solve(K, e)[0] for K in (16384, 65536)}
        print(f"  e {e}: MAC-weighted static gamma {sum(w[K] * gs[K] for K in gs):.2f}")

    print("\ndynamic matmuls (a token a layer: QK^T and PV each ATTN_WIDTH x T MACs):")
    for dh in (64, 128):
        L = dh
        g = gamma(dh, L)
        print(f"  QK^T d_head {dh}: L <= {dh}, gamma_hash {C / L:.0f} (coord leaves-only {6460 / L:.0f}), "
              f"gamma >= {g:.0f}, whole output one segment: eps_skip ~ 1/gamma")
    for T in (2048, 4096, 8192, 32768):
        g, L = solve(T, 0.01)
        print(f"  PV context {T}: gamma {g:.1f} at L {L:.0f} (e 1%)")

    print("\ncoverage of all matmul MACs (static credited; QK^T never; PV credited when gamma(PV) <= 10 at e 1%):")
    for T in (1024, 2048, 4096, 8192, 16384, 32768):
        dyn = 2 * ATTN_WIDTH * T
        pv_ok = solve(T, 0.01)[0] <= 10
        cov = (STATIC_MACS + (ATTN_WIDTH * T if pv_ok else 0)) / (STATIC_MACS + dyn)
        print(f"  avg context {T:6}: dynamic/static {dyn / STATIC_MACS:6.1%}  coverage {cov:6.1%}"
              f"  PV credited {pv_ok}  {'meets 95%' if cov >= 0.95 else 'below 95%'}")
    t_max = 0.05263 * STATIC_MACS / (2 * ATTN_WIDTH)
    print(f"  95% with no dynamic credit needs avg context <= {t_max:.0f}; with PV credited <= {2 * t_max:.0f}")

    print("\nTheorem S draws (one word a draw, a wrong word caught with certainty up to a collision):")
    for eps in (0.01, 0.005):
        print(f"  eps_S {eps}: K' = {draws(eps, '1e-6')} at delta 1e-6, {draws(eps, '1%')} at delta 1%")

    print("\nSHA-512 speed: freed share h(1 - 1/r)")
    for name, K, _ in STATIC[:1] + STATIC[3:]:
        g, L = solve(K, 0.01)
        h = C / L / g
        print(f"  K {K}: h {h:.2f}; r 1.1 frees {h * (1 - 1 / 1.1):.1%}, r 1.5 {h * (1 - 1 / 1.5):.1%}, "
              f"r 2 {h * 0.5:.1%}")


if __name__ == "__main__":
    main()
