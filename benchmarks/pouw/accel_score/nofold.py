"""ACCEL PROTO-DESIGN (CMP-118): the receipt with no fold under R-ACCEL-6, on the H200, per matmul of the frozen model.

Every 4-byte word (a prefix sum of the noised product every L inner indices) is a SHA-512 Merkle leaf; binding is
collision resistance plus Theorem S's draws. Costs are in plain INT8 tensor time at η (`score.py`'s convention):

- hashing: 32 words a 1024-bit leaf block, one tree node a leaf (`score.commit`'s 2·⌈b/1024⌉), at the measured
  4.79 G compressions/s, so γ_hash = C/L with C = 2·η·MAC_RATE/(32·SHA) ≈ 11,000 (the leaves alone at the data-sheet
  rate are 6460/L, the coordinator's figure);
- the peel: noise of rank r ≥ L costs γ_rm = r/K, and 2r/K if both sides carry noise (HYB's U and V);
- serving skip: a cheater that gets the served output O = (A+b)(B+c)ᵀ by a cheaper product (at A = B = 0 it is the
  bit product bcᵀ, on the live dither) derives one segment's word from O and the others, saving L − K/s_b MACs of
  every output's γ·K, s_b that route's speed over INT8. v1 bounds it for any s_b: ε_skip ≤ L/(γK) ≤ e. O is the
  last committed word, so the peels and wrap corrections that turn the words into O are checked, not skippable.

`v1()` scores protocol v1 on both cards: HYB at U ±80, V ±64 (wrap density 1.26%, BET-HYBRID-2 section H), the peel's
five products of inner size L, wraps on the ALU pipe, e = 0.75%, ε_S = 0.1%.

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


CARDS = {
    # name: (INT8 dense MAC/s, SHA-512 compressions/s measured, tensor MACs per ALU lane-op on the ALU pipe)
    "H200": (MAC_RATE, 4.79e9, 64),
    "RTX PRO 6000": (1.0e15 / 2, 7.13e9, 8),   # census rtx-pro-6000-server (INT8 at the FP8 rate); r20261016-110413-c5d1
}
HYB_RHO = 5          # U ±80, V ±64: two base-128 digits of Y_X and of Y_W, and V Uᵀ, each of inner size L
HYB_P = 0.0119 + 0.00071   # p_W at ±80 plus p_X at ±64 (BET-HYBRID-2 section H, Llama-3.1-8B): 1.26%
E_V1, EPS_S_V1 = 0.0075, 0.001


def v1_card(name: str, e: float = E_V1):
    mac, sha, c_op = CARDS[name]
    c = 2 * ETA * mac / (32 * sha)
    a = 1 + A_COMMIT * c / C + c_op * HYB_P / 2
    rows = []
    for shape, K, N in STATIC:
        q = 1 - HYB_RHO * e
        g = (a + math.sqrt(a * a + 4 * q * c / (e * K))) / (2 * q)
        L = max(128, 128 * math.floor(e * g * K / 128))
        g = a + c / L + HYB_RHO * L / K
        rows.append((shape, K, N, L, c / L, HYB_RHO * L / K, a - 1, g, c / L / g, L / (g * K)))
    return c, rows


def v1() -> None:
    print(f"\nprotocol v1: HYB (U ±80, V ±64, p {HYB_P:.2%}), no fold, e {E_V1:.2%}, eps_S {EPS_S_V1:.1%}, "
          f"L a multiple of 128 (64 for PV) at or below e*gamma*K; the last segment may be shorter")
    for name in CARDS:
        c, rows = v1_card(name)
        tot = sum(k * n for _, k, n in STATIC)
        gw = sum(r[1] * r[2] * r[7] for r in rows) / tot
        hw = sum(r[1] * r[2] * r[4] for r in rows) / tot / gw
        print(f"  {name}: C {c:.0f}; MAC-weighted static gamma {gw:.2f}, h {hw:.2f}")
        print(f"    {'shape':6} {'K':>6} {'L':>5} {'g_hash':>6} {'g_rm':>5} {'g_A+wrap':>8} {'gamma':>6} {'h':>5} "
              f"{'skip':>6} {'unit':>8}")
        for shape, K, N, L, gh, gr, ga, g, h, sk in rows:
            print(f"    {shape:6} {K:6} {L:5} {gh:6.2f} {gr:5.2f} {ga:8.2f} {g:6.2f} {h:5.2f} {sk:6.2%} "
                  f"{unit_ands(L):8.2e}")
        mac, sha, c_op = CARDS[name]
        a = 1 + A_COMMIT * c / C + c_op * HYB_P / 2
        g_qk = a + c / 64 + 3
        for T in (2048, 4096, 8192, 16384, 32768):
            _, pv = v1_card_pv(name, T)
            d = ATTN_WIDTH * T / STATIC_MACS
            best = (0.0, 0.0, "none")
            for credit_pv, credit_qk in ((True, True), (True, False), (False, False)):
                g = (gw + d * (pv if credit_pv else 1) + d * (g_qk if credit_qk else 1)) / (1 + 2 * d)
                cov = (1 + d * credit_pv + d * credit_qk) / (1 + 2 * d)
                if g <= 10 and cov > best[1]:
                    best = (g, cov, "PV+QK" if credit_qk else ("PV" if credit_pv else "static"))
            print(f"    avg context {T:6}: PV gamma {pv:5.1f}, QK^T gamma {g_qk:5.1f} (d_head 64); credit {best[2]:6}: "
                  f"step gamma {best[0]:5.2f}, coverage of all matmul MACs {best[1]:6.1%}")
    k6, k2 = draws(EPS_S_V1, "1e-6"), draws(EPS_S_V1, "1%")
    print(f"  draws: k = {k6} (eps_S {EPS_S_V1:.1%} at delta 1e-6), which gives eps_S {LN['1%'] / k6:.3%} at delta 1%"
          f" ({k2} draws would)")
    print(f"  O-unit (served output, recomputed over all K): {unit_ands(16384):.2e} at K 16384, "
          f"{unit_ands(65536):.2e} at K 65536 ANDs (limit {UNIT_LIMIT:.2e})")


def v1_card_pv(name: str, T: int, e: float = E_V1) -> tuple[int, float]:
    mac, sha, c_op = CARDS[name]
    c = 2 * ETA * mac / (32 * sha)
    a = 1 + A_COMMIT * c / C + c_op * HYB_P / 2
    q = 1 - HYB_RHO * e
    g = (a + math.sqrt(a * a + 4 * q * c / (e * T))) / (2 * q)
    L = max(64, 64 * math.floor(e * g * T / 64))
    return L, a + c / L + HYB_RHO * L / T


S_B = {"H200": 8, "RTX PRO 6000": 2}   # bit product over INT8: b1 wgmma .and.popc (RT-UNITS, measured), 0/1 NVFP4


def v1_leaks() -> None:
    """Routes v1's budget e misses, at A = B = 0 where O = bcᵀ (FINDING PD2-1).

    O-route: serve O by the bit product (K/s_b a cell) instead of the five peels (ρL), saving ρL − K/s_b.
    Zero digits: Y_X = Σ_s b and Y_W = Σ_s c are below 128, so the two high-digit peels are zero products (2L).
    """
    print("\nv1 leaks at A = B = 0 (each a share of the honest round, gamma K a cell)")
    for name in CARDS:
        _, rows = v1_card(name)
        tot = acc = 0.0
        for shape, K, N, L, _gh, _gr, _ga, g, _h, _sk in rows:
            o_route = max(0.0, HYB_RHO * L - K / S_B[name]) / (g * K)
            zero = 2 * L / (g * K)
            budgeted = max(0.0, L - K / S_B[name]) / (g * K)
            tot += K * N
            acc += K * N * max(o_route, zero)
            print(f"  {name:13} {shape:5} K {K:6} L {L:5}: O-route {o_route:6.2%}, zero digits {zero:6.2%}, "
                  f"v1's budgeted last-segment skip {budgeted:6.2%}")
        print(f"  {name}: MAC-weighted leak {acc / tot:.2%} (v1 budgeted e = {E_V1:.2%})")


LEAF_BLOCKS = 8                                             # SHA-512 blocks in one hiding leaf
LEAF_DATA_BITS = LEAF_BLOCKS * 1024 - 128 - 129             # one 128-bit salt, SHA-512's minimum padding
TREE = 1 + (16 * 512 + 1024) / 1024 / 15 / LEAF_BLOCKS      # 16-ary tree, node = SHA-512 of 16 child digests


def hash_c(name: str, word_bits: int) -> float:
    """γ_hash·L: compressions a word over the card's INT8 rate at η, with NF3's leaf, salt and node function."""
    mac, sha, _ = CARDS[name]
    return ETA * mac * TREE * word_bits * LEAF_BLOCKS / (LEAF_DATA_BITS * sha)


def v2_scan(budget: float = 0.004) -> None:
    """Route (b): O bound to the words; per shape the L (a multiple of 64) of least γ whose leak is within `budget`.

    leak = max(2L, ρL − K/s_b)/(γK): the zero high digits and the bit route to O (FINDING PD2-1).
    """
    print(f"\nv2 route (b): leaf {LEAF_BLOCKS} blocks + salt, 16-ary tree (x{TREE:.3f}), leak budget {budget:.2%}")
    for name in CARDS:
        mac, sha, c_op = CARDS[name]
        for wb in (32, 16, 8):
            c = hash_c(name, wb)
            a = 1 + A_COMMIT * c / C + c_op * HYB_P / 2
            tot = gw = lw = 0.0
            cells = []
            for shape, K, N in STATIC:
                best = None
                for L in range(64, K + 1, 64):
                    g = a + c / L + HYB_RHO * L / K
                    leak = max(2 * L, HYB_RHO * L - K / S_B[name]) / (g * K)
                    if leak <= budget and (best is None or g < best[1]):
                        best = (L, g, leak)
                L, g, leak = best
                tot += K * N
                gw += K * N * g
                lw += K * N * leak
                cells.append(f"{shape} L {L} g {g:.2f} leak {leak:.2%}")
            print(f"  {name:13} {wb:2}-bit words C {c:6.0f}: static gamma {gw / tot:5.2f}, leak {lw / tot:.2%}; "
                  + "; ".join(cells))


# CALIB's θ (#4513 at 188907341d), priced per SM-clock and stated at the tensor clock (SMs × MHz; the H200's at the
# 1980 MHz boost, M14, which leaves every per-SM-clock ratio as it was): ρ₈ INT8 MACs a
# SM-clock, and SHA-512 compressions a SM-clock at the paper ceiling with IMAD.HI (38.29 SM-clocks a compression on
# the H200, 38.11 on sm_120), today's kernel (0.018327, 0.018129 a SM-clock) and hashcat's structured mode (0.020717,
# 0.020646), so c_sha = today / hashcat is 0.885 and 0.878.
THETA = {"H200": (4096, 132, 1980e6, 1 / 38.29, 0.018327, 0.020717),
         "RTX PRO 6000": (1024, 188, 2617e6, 1 / 38.11, 0.018129, 0.020646)}
V2_CARDS = {
    # name: (INT8 dense MAC/s, SHA-512 compressions/s at the ceiling, tensor MACs per ALU op, s_b)
    # RT-MODEL M2's 59.2 ALU op on the H200. s_b: b1 wgmma 8× on the H200; on the RTX PRO 6000 ALGO-COST A7's depth-2
    # Strassen in exact 0/1 FP4, c_bit = 0.5·(7/8)² (0.26 with emulated b1 on the ALU alongside).
    "H200": (4096 * 132 * 1980e6, 132 * 1980e6 / 38.29, 59.2, 8),
    "RTX PRO 6000": (1024 * 188 * 2617e6, 188 * 2617e6 / 38.11, 8, 1 / (0.5 * (7 / 8) ** 2)),
}
SHA_BUILT = {c: sms * hz * today for c, (_, sms, hz, _, today, _) in THETA.items()}
C_SHA_TODAY = {c: today / hashcat for c, (*_, today, hashcat) in THETA.items()}
U_W, U_X = 80, 96              # BET-HYBRID-2 K (±80 both) and NAIVE-STRASSEN (V ±96 to depth 13): the larger of each
P_W, P_X = 0.0119, 0.0042      # wrap densities at those widths, Llama-3.1-8B (BET-HYBRID-2 H; 70B p_X 0.64%)
P_X_AT = {80: 0.0016, 96: 0.0042}   # p_X by V's width, Llama-3.1-8B (BET-HYBRID-2 table (c))
# v2 as frozen (coordinator, 04:13Z and 04:20Z): U ±80 and V ±80 on both cards. At ε_alg budget 0 every static shape's
# lane load (ALU without wraps over tensor) is at least 1.00 on both cards, which meets ALGO-COST A10's first fix (0.75
# at ±80), so neither V ±96 nor an honest Strassen kernel is needed; V ±96 on the RTX PRO 6000 is the margin. Noise
# tiles t = 2 (H200) and t = 4 (RTX PRO 6000).
V_WIDTH = {"H200": 80, "RTX PRO 6000": 80}
TILES = {"H200": 2, "RTX PRO 6000": 4}
# The tiles' cross-term route Σ_j (b + V)π_jᵀ Uᵀ + V(Σ_j c_j)ᵀ: one INT8 digit plus a sparse high part, so c_t = 2 + the
# best form (ALGO-COST A12, A13; floor 2). Unlisted tile counts take the least listed, the prover's cheapest.
C_T = {("H200", 2): 2.502, ("H200", 3): 2.548, ("RTX PRO 6000", 4): 2.539}
RHO_V2 = 4                     # Z_X Uᵀ and V Y_Wᵀ, two base-128 digits each; Z_X = Σ_s (X_s + V)π_sᵀ absorbs (VΠ)Uᵀ
M_PREFILL = 8192               # tokens a prefill call
B_STREAMED = True              # P2 (06:20Z): the certifier streams b (P's too); the other coins are seed-expanded here
EPS_ALG_V2 = 0.005             # budget on the deployed kernel's gap at A = B = 0, per credited shape
SHAPES = {                     # (K, N, layers): the frozen model, and COVER-ATTN's served static shapes
    "frozen": [(16384, 16512, 1), (16384, 16384, 1), (16384, 65536, 1), (65536, 16384, 1)],
    "Llama-3.1-8B": [(4096, 6144, 32), (4096, 4096, 32), (4096, 28672, 32), (14336, 4096, 32), (4096, 128256, 1)],
    "Llama-3.1-70B": [(8192, 10240, 80), (8192, 8192, 80), (8192, 57344, 80), (28672, 8192, 80), (8192, 128256, 1)],
    "DeepSeek-V4-Pro": [(7168, 7 * 6144, 61), (3072, 7 * 7168, 61), (7168, 1536, 61), (1536, 65536, 61),
                        (7168, 576, 61), (4096, 16 * 1024, 61), (16384, 7168, 61), (7168, 129280, 1)],
}


def v2_round(card: str, K: int, N: int, L: int, word_bits: int, m: int = M_PREFILL, commit_a: bool = False,
             built: bool = False, weight_coins: bool = True, tiles: int = 1, u_x: int | None = None) -> dict:
    """One prefill call of protocol v2, per MAC in plain INT8 tensor time at η.

    A segment (one word a cell) is L long and holds `tiles` noise tiles of period L' = L/tiles, each with a fresh
    permutation of U's columns. Tensor: the product (1) and the peels, O = T_last − Z_X Uᵀ − V Y_Wᵀ + 256(w_X Eᵀ +
    X w_Wᵀ), base-128 digits of Z_X (|X + V| ≤ 224, K/L' terms) and Y_W (|W| ≤ 128, K/L' terms), inner dimension L'.
    ALU: hashing (S segment words of `word_bits` and O at 32 bits a cell, NF3's leaf and tree), coin expansion (SHA-512
    counter mode, 512 bits a compression: the dither (m+N)K bits, the noise 8 bits a value of V and U, the K/L'
    permutations), and the wraps. A is the previous call's committed output under a public elementwise map plus
    committed per-row scalars, so it adds no leaves (`commit_a` prices committing it anyway).

    ε_alg is the cheapest route at A = B = 0 (or at low-rank A and B), where nothing wraps (|b + V| ≤ 97): PD2-1's
    O-route (O = bcᵀ at K/s_b instead of the peels) and zero high digits of Y_W = Σ c; the one-segment skip
    (L − K/s_b); and building every word from b_s c_sᵀ (L/s_b) plus its cross terms, Σ_j (b + V)π_jᵀ Uᵀ (one digit at
    one tile, two above) and V(Σ_j c_j)ᵀ (one digit), which also yields O without the peels.
    """
    mac, sha, c_op, s_b = V2_CARDS[card]
    per_comp = ETA * mac / (SHA_BUILT[card] if built else sha)
    per_bit = LEAF_BLOCKS * TREE / LEAF_DATA_BITS
    S = math.ceil(K / L)
    Lp = L / tiles
    n_terms = K / Lp
    u_x = V_WIDTH[card] if u_x is None else u_x
    dz, dy = (2 if (128 + u_x) * n_terms < 2 ** 14 else 3), (2 if 128 * n_terms < 2 ** 14 else 3)
    g_hash = per_comp * per_bit * (S * word_bits + 32) / K
    g_a = per_comp * per_bit * 8 / N if commit_a else 0.0
    side = m + N if weight_coins else m
    dither = (side - m if B_STREAMED else side) * K
    coin_bits = dither + 8 * side * Lp + (K * math.log2(Lp) if weight_coins else 0)
    g_coin = per_comp * coin_bits / (512 * m * N * K)
    g_rm = (dz + dy) * Lp / K
    g_wrap = c_op * (P_W + P_X_AT[u_x]) / 2
    tensor, alu = 1 + g_rm, g_hash + g_a + g_coin + g_wrap
    g = tensor + alu
    c_t = 2 if tiles == 1 else C_T.get((card, tiles), min(C_T.values()))
    saved = max(0.0, (dy - 1) * Lp, g_rm * K - K / s_b, L - K / s_b, K + g_rm * K - K / s_b - c_t * K / tiles)
    a_nw = alu - g_wrap
    return {"L": L, "tiles": tiles, "S": S, "digits": (dz, dy), "comps": (g_hash + g_a + g_coin) / per_comp,
            "gamma": g, "gamma_max": max(tensor, alu), "hash": g_hash, "A": g_a,
            "coin": g_coin, "rm": g_rm, "wrap": g_wrap, "h": (g_hash + g_a + g_coin) / g, "eps_alg": saved / (g * K),
            "T": tensor, "A_nw": a_nw, "saved": saved / K,
            "eps_ovl": min(saved / K, max(0.0, tensor - a_nw)) / max(tensor, a_nw)}


def v2_pick(card: str, K: int, N: int, word_bits: int, budget: float = EPS_ALG_V2, tile_set=(1,), overlap=False,
            **kw) -> dict:
    """The segment (L' a multiple of 64, L = tiles·L' at most K) of least γ with ε_alg within budget; else the least
    ε_alg. With `overlap`, γ is the round's pipe maximum and ε_alg the tensor saving that fits the tensor pipe's slack
    over the ALU without wraps (overlapped credit is priced without wraps, which vanish at small operands)."""
    rounds = [v2_round(card, K, N, t * Lp, word_bits, tiles=t, **kw)
              for t in tile_set for Lp in range(64, K // t + 1, 64)]
    eps, gam = ("eps_ovl", "gamma_max") if overlap else ("eps_alg", "gamma")
    ok = [r for r in rounds if r[eps] <= budget]
    return min(ok, key=lambda r: r[gam]) if ok else min(rounds, key=lambda r: r[eps])


def v2_static(card: str, model: str, word_bits: int, budget: float = EPS_ALG_V2, **kw) -> dict:
    """MAC-weighted γ (additive and overlapped), h; ε_alg weighted by honest cost; the worst shape's γ."""
    macs = cost = cost_max = hashed = saved = 0.0
    worst = (0.0, None)
    for K, N, reps in SHAPES[model]:
        r = v2_pick(card, K, N, word_bits, budget, **kw)
        w = K * N * reps
        macs += w
        cost += w * r["gamma"]
        cost_max += w * r["gamma_max"]
        hashed += w * r["gamma"] * r["h"]
        saved += w * r["gamma"] * r["eps_alg"]
        if r["gamma"] > worst[0]:
            worst = (r["gamma"], (K, N, r["L"]))
    return {"gamma": cost / macs, "gamma_max": cost_max / macs, "h": hashed / cost, "eps_alg": saved / cost,
            "worst": worst}


def v2() -> None:
    k6 = math.ceil(LN["1e-6"] / (EPS_S_V1 * (1 - 2 ** -16)))
    print(f"\nprotocol v2 (static shapes, prefill m {M_PREFILL}): HYB U ±{U_W} permuted per segment, V ±{U_X}, period L, "
          f"O bound to the words, NF3 leaf/tree, honest SHA-512 at the public best; eps_alg budget {EPS_ALG_V2:.1%} "
          f"a shape; k = {k6} draws (eps_S {EPS_S_V1:.1%} at 1e-6, {LN['1%'] / k6:.3%} at 1%)")
    print(f"  {'card':13} {'model':16} {'words':>5} {'budget':>6} {'gamma':>6} {'g_max':>6} {'h':>5} {'eps_alg':>7}"
          f"  worst shape")
    print(f"  SHA-512 compressions per 128 B of words, tree included: {1024 * LEAF_BLOCKS * TREE / LEAF_DATA_BITS:.3f}")
    for card in V2_CARDS:
        for model in SHAPES:
            for wb, budget, built in [(32, 0.01, False), (16, 0.005, False), (16, 0.01, False), (16, 0.02, False),
                                      (16, 0.01, True)]:
                s = v2_static(card, model, wb, budget, built=built)
                g, (K, N, L) = s["worst"]
                tag = "built" if built else ""
                print(f"  {card:13} {model:16} {wb:5} {budget:6.1%} {s['gamma']:6.2f} {s['gamma_max']:6.2f} "
                      f"{s['h']:5.2f} {s['eps_alg']:7.2%}  K {K} N {N} L {L}: {g:.2f} {tag}")


ATT_MODELS = {"Llama-3.1-8B": (128, 4), "Llama-3.1-70B": (128, 8), "DeepSeek-V4-Pro": (512, 128)}  # (d_v, q heads a KV head)


def v2_att(budget: float = 0.01) -> None:
    """ATT's PV with O bound: K = n attended keys, N = d_v, the weight side (Ṽ's dither and noise) streamed once a key
    and reused by every later query (COVER-ATTN), so a call expands only P̃'s side. At decode m is the KV group's query
    heads; QKᵀ (K = d_qk) has no L that meets PD2-1's bound, so it stays uncredited."""
    print(f"\nv2 ATT, PV with O bound (16-bit words, eps_alg budget {budget:.0%}): gamma (L) at n attended keys")
    for card in V2_CARDS:
        for model, (dv, group) in ATT_MODELS.items():
            cells = []
            for n in (2048, 8192, 32768):
                pre = v2_pick(card, n, dv, 16, budget, weight_coins=False)
                dec = v2_pick(card, n, dv, 16, budget, m=group, weight_coins=False)
                cells.append(f"n {n}: prefill {pre['gamma']:.2f} (L {pre['L']}), decode {dec['gamma']:.2f}")
            print(f"  {card:13} {model:16} " + "; ".join(cells))
    for card in V2_CARDS:
        r = v2_round(card, 128, 4096, 128, 16)
        print(f"  {card:13} QK^T at K 128 (one segment): eps_alg {r['eps_alg']:.0%} at gamma {r['gamma']:.2f}")


def v2_tiles(budget: float = 0.01) -> None:
    """v2 with several noise tiles a segment: words follow L, the peels and PD2-1's routes follow L' = L/tiles."""
    print(f"\nv2 with noise tiles (16-bit words, eps_alg budget {budget:.0%}): one tile a segment, then 1 to 6")
    for card in V2_CARDS:
        for model in SHAPES:
            one = v2_static(card, model, 16, budget)
            many = v2_static(card, model, 16, budget, tile_set=range(1, 7))
            g, (K, N, L) = many["worst"]
            print(f"  {card:13} {model:16} gamma {one['gamma']:5.2f} -> {many['gamma']:5.2f} (g_max {many['gamma_max']:.2f},"
                  f" h {many['h']:.2f}, eps_alg {many['eps_alg']:.2%}); worst K {K} N {N} L {L}: {g:.2f}")
        for model, (dv, _) in ATT_MODELS.items():
            cells = []
            for n in (2048, 8192, 32768):
                r = v2_pick(card, n, dv, 16, budget, tile_set=range(1, 7), weight_coins=False)
                cells.append(f"n {n}: {r['gamma']:.2f} (L {r['L']}, {r['tiles']} tiles)")
            print(f"  {card:13} {model:16} PV " + "; ".join(cells))


def _dense(name, layers, hidden, hq, hkv, d, ffn, vocab):
    static = layers * hidden * (hq * d + 2 * hkv * d + hq * d + 2 * ffn) + layers * ffn * hidden + hidden * vocab
    return {"name": name, "static": static, "hq": hq, "hkv": hkv, "dqk": d, "dv": d,
            "att": lambda t: [(layers, t)], "indexer": lambda t: 0}


V4_EMBED = 129_280 * 7_168    # V4's input embedding (vocab × hidden): inside its 49e9 active parameters, a lookup


def _dsv4():
    def att(t):
        w = min(t, 128)
        return [(30, w + min(1024, t // 4)), (31, w + t // 128)]

    # Static MACs a token: the 49e9 active parameters less the input embedding, which does no MACs (RT-PROTO-2's P24).
    return {"name": "DeepSeek-V4-Pro", "static": 49e9 - V4_EMBED, "hq": 128, "hkv": 1, "dqk": 512, "dv": 512,
            "att": att, "indexer": lambda t: 30 * 64 * 128 * (t // 4)}


# COVER-ATTN's world model and final ATT block (cover_attn.py at 27fc89418f on cursor/compute-research-accel-cover-attn-ea24):
# a 4,500-token request processed causally; ATT's P̃ carries 8 fresh bits, its K̃ and Ṽ noise is streamed once a key.
WORLD = [_dense("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256),
         _dense("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28672, 128256), _dsv4()]
WORLD_TOKENS, ATT_WRAP_P, SOFTMAX_PASS = 4500, 0.0126, 1.0
HBM = {"H200": 4.8e12, "RTX PRO 6000": 1.792e12}
LOOKUP_G_MAX = 8


def world_macs(m: dict) -> tuple[dict, float, float, float, float]:
    """{attended n: (token, layer) rows}; static, QKᵀ, PV and indexer MACs of one request."""
    hist, ix = {}, 0.0
    for t in range(1, WORLD_TOKENS + 1):
        for nl, n in m["att"](t):
            hist[n] = hist.get(n, 0) + nl
        ix += m["indexer"](t)
    qk = sum(c * m["hq"] * m["dqk"] * n for n, c in hist.items())
    pv = sum(c * m["hq"] * m["dv"] * n for n, c in hist.items())
    return hist, m["static"] * WORLD_TOKENS, qk, pv, ix


def static_pipes(card: str, model: str, budget: float, max_tiles: int = 6, frozen: bool = False, p1: bool = False,
                 picks: dict | None = None, **kw) -> dict:
    """v2's static rounds with tiles, picked under overlapped accounting: MAC-weighted tensor, ALU without wraps, wraps,
    the tensor saving at A = B = 0, and the worst shape's pipe maximum. `frozen` fixes the card's tile count (TILES)."""
    tot = T = A = W = S = 0.0
    worst = (0.0, None)
    tile_set = (TILES[card],) if frozen else range(1, max_tiles + 1)
    for K, N, reps in SHAPES[model]:
        if p1:
            # P1 (RT-PROTO; SKIP-FIX §4): O is not a drawn word, so one word a cell (L = K) and nothing cheap is credited
            # (ε_alg 0); the noise period L' = K/t is free and the peel to O is serving work.
            r = min((v2_round(card, K, N, K, 16, tiles=K // Lp, **kw) for Lp in range(64, K + 1, 64) if K % Lp == 0),
                    key=lambda r: r["gamma_max"])
            r = {**r, "saved": 0.0}
        elif picks and (card, K, N) in picks:
            t, Lp = picks[card, K, N]
            r = v2_round(card, K, N, t * Lp, 16, tiles=t, **kw)
        else:
            r = v2_pick(card, K, N, 16, budget, tile_set=tile_set, overlap=True, **kw)
        w = K * N * reps
        tot, T, A, W, S = tot + w, T + w * r["T"], A + w * r["A_nw"], W + w * r["wrap"], S + w * r["saved"]
        if r["gamma_max"] > worst[0]:
            worst = (r["gamma_max"], (K, N, r["L"], r["tiles"]))
    return {"T": T / tot, "A_nw": A / tot, "wrap": W / tot, "saved": S / tot, "worst": worst}


def pv_bucket(n: int) -> int:
    return max(64, 64 * round(n / 64))


def pv_pipes(card: str, m: dict, hist: dict, budget: float, built: bool = False,
             picks: dict | None = None) -> tuple[float, float, float, float]:
    """R-ACCEL-8's attention: PV credited as a v2 round with O bound (COVER-ATTN's PV design), K = n attended keys,
    N = d_v, the weight side (Ṽ's dither and noise) streamed once a key, so a call expands only P̃'s side. Picked
    like a static shape (overlapped) at each n rounded to 64; MAC-weighted (tensor, ALU without wraps, A = B = 0
    saving) a credited PV MAC, and the share of PV MACs left uncredited (a pick of None: served plain, P19)."""
    rounds, w, T, A, S, u = {}, 0.0, 0.0, 0.0, 0.0, 0.0
    for n, c in hist.items():
        nb = pv_bucket(n)
        if nb not in rounds:
            if picks and (card, nb, m["dv"]) in picks:
                p = picks[card, nb, m["dv"]]
                rounds[nb] = p and v2_round(card, nb, m["dv"], p[0] * p[1], 16, tiles=p[0], weight_coins=False,
                                            built=built)
            else:
                rounds[nb] = v2_pick(card, nb, m["dv"], 16, budget, overlap=True, weight_coins=False, built=built)
        r, x = rounds[nb], c * n
        if r is None:
            u += x
            continue
        w, T, A, S = w + x, T + x * r["T"], A + x * r["A_nw"], S + x * r["saved"]
    return (T / w, A / w, S / w, u / (u + w)) if w else (0.0, 0.0, 0.0, 1.0)


def lookup_alu(card: str, hist: dict) -> float:
    """COVER-ATTN's ℓ: ALU MAC-times of one PV round MAC moved to Four-Russians lookups on 8-bit P̃ (8/g adds a MAC
    over the bit-planes, 2^g-entry tables of Ṽ chunks built once for the n/2 later queries), g ≤ 8."""
    n = sum(c * n * n for n, c in hist.items()) / sum(c * n for n, c in hist.items())
    c_op = V2_CARDS[card][2]
    return min(c_op * 8 / g * (1 + 2 ** g / (n / 2)) for g in range(2, LOOKUP_G_MAX + 1))


def v2_world(budget: float = 0.01, built: bool = False, max_tiles: int = 6, cards=None, picks: dict | None = None,
             **kw) -> list[dict]:
    """v2 static (tiles) plus PV with O bound at the world model, the period's tensor and ALU pipes overlapped.

    R-ACCEL-8 (credited work is the useful product): static and PV rounds are credited; QKᵀ (both passes of the
    two-pass softmax) and V4's indexer are served plain and uncredited. Credit is priced at max(T, A without wraps);
    the honest period takes max(T, A). Where the period is tensor-bound the honest kernel moves PV round MACs to
    Four-Russians lookups on the ALU until the pipes meet (COVER-ATTN's RTX PRO 6000 kernel). ε_alg is what a prover at
    A = B = 0 (or low rank) saves within the tensor pipe's slack over the ALU without wraps: PD2-1's routes first, at
    no ALU cost, then lookups at ℓ."""
    rows = []
    for card in cards or V2_CARDS:
        a_wrap = V2_CARDS[card][2] * ATT_WRAP_P / 2
        for m in WORLD:
            hist, st, qk, pv, ix = world_macs(m)
            tot = st + qk + pv + ix
            s = static_pipes(card, m["name"], budget, max_tiles, built=built, picks=picks, **kw)
            pv_t, pv_a, pv_s, pv_u = pv_pipes(card, m, hist, budget, built=built, picks=picks)
            plain = qk * (1 + SOFTMAX_PASS) + ix + pv * pv_u
            pv, unc = pv * (1 - pv_u), qk + ix + pv * pv_u
            T = st * s["T"] + pv * pv_t + plain
            A_nw = st * s["A_nw"] + pv * pv_a
            Wr = st * s["wrap"] + pv * a_wrap
            g_add = (T + A_nw + Wr) / tot
            ell = lookup_alu(card, hist)
            x = min(pv, max(0.0, T - A_nw - Wr) / (1 + ell))
            T, A_nw = T - x, A_nw + ell * x
            honest = max(T, A_nw + Wr)
            # A prover after credit skips the uncredited QKᵀ, indexer and plain PV, so credit and its slack are the
            # credited rounds' own period.
            T_c = T - plain
            credit = max(T_c, A_nw)
            slack = max(0.0, T_c - A_nw)
            s1 = min(st * s["saved"] + pv * pv_s, slack)
            s2 = min(pv - x, (slack - s1) / (1 + ell))
            label = card + (" built" if built else "") + (" t=1" if max_tiles == 1 else "")
            label += " P1" if kw.get("p1") else ""
            label += f" t={TILES[card]}" if kw.get("frozen") else ""
            label += f" V±{kw.get('u_x', V_WIDTH[card])}" if kw.get("frozen") else ""
            rows.append({"card": label, "model": m["name"], "gamma_add": g_add, "gamma": honest / tot,
                         "util": T / honest,
                         "T": T_c / tot, "A": (A_nw + Wr) / tot, "eps_alg": (s1 + s2) / credit,
                         "wrap_uncredited": (honest - max(T, A_nw)) / honest, "lookup": x / pv if pv else 0.0,
                         "uncredited": unc / tot, "pv_uncredited": pv_u, "worst": s["worst"],
                         "static_gamma": max(s["T"], s["A_nw"] + s["wrap"]), "credit": credit / tot,
                         "honest": honest / tot, "T_all": T / tot, "A_nw": A_nw / tot, "Wr": Wr / tot,
                         "T_c": T_c / tot})
    return rows


def v2_world_print(budget: float = 0.01) -> None:
    print(f"\nv2 at the world model (static with tiles + PV with O bound, QKᵀ uncredited; 16-bit words, overlapped pipes, eps_alg budget {budget:.0%}"
          f" a shape)")
    print(f"  {'card':18} {'model':16} {'g_ovl':>6} {'g_add':>6} {'T':>5} {'A':>5} {'eps_alg':>7} {'wraps':>6} "
          f"{'lookup':>6} {'uncr':>5}  static, worst shape")
    for r in v2_world(budget) + v2_world(budget, built=True) + v2_world(budget, max_tiles=1):
        g, (K, N, L, t) = r["worst"]
        print(f"  {r['card']:18} {r['model']:16} {r['gamma']:6.2f} {r['gamma_add']:6.2f} {r['T']:5.2f} {r['A']:5.2f} "
              f"{r['eps_alg']:7.2%} {r['wrap_uncredited']:6.1%} {r['lookup']:6.1%} "
              f"{r['uncredited']:5.1%}  {r['static_gamma']:.2f}, K {K} N {N} L {L} t {t}: {g:.2f}")


def v2_frozen_print(budget: float = 0.0) -> list[dict]:
    """v2 as frozen: the card's tile count, A12/A13's c_t, V ±80, and V ±96 on the RTX PRO 6000 beside it (A10's margin). Then ε_alg at every registrable static shape on its own, the prover's cheapest instance (A = B = 0,
    its own model of one shape), not only the world mix. v2's budget is 0: at 1% a lone K 65536 shape leaks 0.50% (H200)
    and 0.76% (RTX PRO 6000); at 0 the world γ is unchanged and that shape alone costs 1.53 (from 1.46) and 1.09 (1.02)."""
    print(f"\nv2 as frozen at the world model (tiles H200 t={TILES['H200']}, RTX PRO 6000 t={TILES['RTX PRO 6000']}; "
          f"c_t A12/A13; 16-bit words; overlapped; eps_alg budget {budget:.0%} a shape)")
    print(f"  {'card':30} {'model':16} {'g_ovl':>6} {'g_add':>6} {'T':>5} {'A':>5} {'util':>5} {'eps_alg':>7} "
          f"{'wraps':>6} {'uncr':>5}  static worst shape")
    rows = (v2_world(budget, frozen=True) + v2_world(budget, built=True, frozen=True)
            + v2_world(budget, cards=["RTX PRO 6000"], frozen=True, u_x=96))
    for r in rows:
        g, (K, N, L, t) = r["worst"]
        print(f"  {r['card']:30} {r['model']:16} {r['gamma']:6.2f} {r['gamma_add']:6.2f} {r['T']:5.2f} {r['A']:5.2f} "
              f"{r['util']:5.1%} {r['eps_alg']:7.2%} {r['wrap_uncredited']:6.1%} "
              f"{r['uncredited']:5.1%}  K {K} N {N} L {L}: {g:.2f}")
    print("  each registrable static shape alone (prover's own model of that shape, A = B = 0): worst eps_alg, its L")
    for card in V2_CARDS:
        for u_x in sorted({V_WIDTH[card], 96}):
            worst, load = (0.0, None, None), math.inf
            for model, shapes in SHAPES.items():
                for K, N, _ in shapes:
                    r = v2_pick(card, K, N, 16, budget, tile_set=(TILES[card],), overlap=True, u_x=u_x)
                    if r["eps_ovl"] >= worst[0]:
                        worst = (r["eps_ovl"], (K, N, r["L"]), r["gamma_max"])
                    load = min(load, r["A_nw"] / r["T"])
            e, (K, N, L), g = worst
            print(f"  {card:13} V±{u_x}: worst eps_alg {e:.2%} at K {K} N {N} L {L} (gamma {g:.2f}); least lane load "
                  f"(A10: Strassen gains below 0.71 at ±80, 0.49 at ±96) {load:.2f}")
    return rows


def seg_coins() -> None:
    """SKIP-FIX's SEG-COINS points against PD2-1's O-route, which coin timing doesn't touch.

    At A = B = 0 a prover gets O = bcᵀ from the dither bits at K/s_b after the final coins arrive and skips the peel
    (ρL a cell), so ε_alg ≥ (ρL − K/s_b)/(γK) at every long-segment length L. γ is SKIP-FIX's convention.
    """
    print("\nSEG-COINS at SKIP-FIX's points (K 16384), PD2-1's O-route: eps_alg >= (rho L - K/s_b)/(gamma K)")
    K, C = 16384, 10970
    for label, L, rho, w in [("one long segment 7K/8, rho 1", 7 * K / 8, 1, 2),
                             ("two long segments 7K/16, rho 2", 7 * K / 16, 2, 3),
                             ("two long segments 7K/16, v2's rho 4", 7 * K / 16, RHO_V2, 3)]:
        g = 1 + 0.17 + C * w / K + rho * L / K
        cells = [f"{card} {max(0.0, rho * L - K / s_b) / (g * K):.1%}" for card, (_, _, _, s_b) in V2_CARDS.items()]
        print(f"  {label:36} gamma {g:4.2f}: " + ", ".join(cells))


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
    v1()
    v1_leaks()
    v2_scan()
    v2()
    v2_att()
    v2_tiles()
    v2_world_print()
    v2_frozen_print()
    seg_coins()
