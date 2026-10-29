"""Range-forced purity and the three-block exact stack: the CPU checks behind N-DENSE's design.

1. **Purity threshold.** For each noise alphabet and each tensor-core type of the bilinear accumulator model
   (`h200Types`), the most noise variables one lane can sum and still fit the type, with no data and beside full int7
   data. A lane sums two variables only if the alphabet's span is at most half the type's range (`Lane.pure_of_fitsOn`).
2. **Order intersections.** RFP-3 is `X = [A+E | -E·π | A]`, `Y = [B+F ; π·(B+F) ; -F]`; RFP-4 adds fresh coins `G`,
   `X = [A+E | -E·π | G-A | -G·π']`, `Y = [B+F ; π·(B+F) ; F ; π'·F]`, so every block is bilinear in fresh coins.
   Each commits running words of each block every `L` steps. With block 2 in a uniformly random order `π`, its `L`-chunks meet block 1's consecutive
   chunks in small pieces. Prices, per dot product in registers of 32, the honest program, the aligned-order shortcut
   (valid only for `π` the identity) and the intersection attack (atoms on every piece, merges of at most three
   registers). An attack family's price, not a lower bound. Once the pieces average 32 (`L² ≈ 32K`) the atoms fill
   their registers and the attack nears honest; from about twice that `L` it is cheaper.
3. **The stacks' terms on Llama-3.2-1B's linears.** Forming, coin entries and committed bytes a useful MAC, at T 512
   and T 8,192, from one 32-unit instruction per four s8 lanes formed (a permuted block reads its buffer in `π`'s
   order, and a block of coins as drawn is not formed) and SHA-512 at the floor's 908 units a byte.
4. **The projection bound on the repeated blocks** (a lower bound on pure lanes, argued, not in Lean). A coin pair
   produced by one lane only has its instruction's signature over the committed words, so an instruction's
   once-produced pairs lie in one piece. Projected onto those pairs the program is a linear straight-line program in
   one variable an instruction, which by the transposition principle needs at least `n + p − m'` additions for `n`
   such instructions, `p` pieces and `m'` words. The instructions supply `n` and a merge at most two, so a dot
   product's repeated pair of blocks costs at least
   `2K − (the piece (J−1, 0)) − (its 2J−1 largest other pieces) − Σ (s − 16)⁺ over the rest`.
5. **One-bit coins at zero data.** With coins in {±64} and A = B = 0 (data the prover chooses), every RFP-4 word is
   4096 times a sum of products of signs, so a b1 AND-popcount instruction (256 products at the price of 32 s8 ones)
   makes each word from the coins' bits and their per-segment popcounts. Checked word for word against the s8 words,
   with both programs' tensor units a dot product.
6. **Known words at zero activations.** At A = 0 a running word that is 0 for every coin draw is known, like the final
   word, so N-SAKU's middle-segment skip runs backward from it and skips a segment of L: in RFP-4's order 1, 2, 3, 4 the
   word at 2K is A·(B+F), so the skip is taken twice; the order 1, 3, 2, 4 (X = [A+E | G−A | −E·π | −G·π′]) leaves only
   the final word known.

    python3 benchmarks/pouw/theory/rfp_stack.py [--seeds 16] [--out PATH]
"""

import argparse
import json
import math
import random

TYPES = {
    "s8": (32, -128, 127),
    "e4m3": (32, -16, 16),
    "e5m2": (32, -8, 8),
    "f16": (16, -2048, 2048),
    "tf32": (8, -2048, 2048),
    "bf16": (16, -256, 256),
    "s4": (64, -8, 7),
    "b1": (256, 0, 1),
}

ALPHABETS = {
    "[-32,31] (H100Sem's noise)": list(range(-32, 32)),
    "[-64,63]": list(range(-64, 64)),
    "[-64,64] (RFP)": list(range(-64, 65)),
    "{-64,64} (RFP, one bit)": [-64, 64],
}

INT7 = 63
SHA_UNITS_PER_BYTE = 908

LLAMA_1B = [("q", 2048, 2048), ("k", 2048, 512), ("v", 2048, 512), ("o", 2048, 2048), ("gate", 2048, 8192),
            ("up", 2048, 8192), ("down", 8192, 2048)]


def purity(alphabets=ALPHABETS, types=TYPES):
    """The most unit-coefficient noise variables one lane sums and still fits, per alphabet and type."""
    out = {}
    for name, S in alphabets.items():
        span = max(S) - min(S)
        row = {}
        for t, (_k, lo, hi) in types.items():
            width = hi - lo
            free = width // span if span else None
            # Beside int7 data the lane is sum(a_i + e_i), each summand reaching 63 + max(S) above zero and
            # 63 - min(S) below it.
            with_data = min(hi // (INT7 + max(S)), -lo // (INT7 - min(S))) if lo < 0 else 0
            row[t] = {"free": free, "beside_int7": max(with_data, 0)}
        two_fit_s8 = 2 * span <= types["s8"][2] - types["s8"][1]
        out[name] = {"span": span, "values": len(S), "s8_two_variable_lane_fits": two_fit_s8, "max_vars": row}
    return out


def chunks_of(order, L):
    return [order[j * L:(j + 1) * L] for j in range(len(order) // L)]


def stack_costs(K, L, perm):
    """Registers of 32 a dot product: honest, the aligned shortcut, and the intersection attack for block 2 in `perm`."""
    J = K // L
    block = [i // L for i in range(K)]
    pieces = {}
    for jp, ch in enumerate(chunks_of(perm, L)):
        for i in ch:
            key = (block[i], jp)
            pieces[key] = pieces.get(key, 0) + 1
    d1 = [0] * J
    d2 = [0] * J
    for (j, jp) in pieces:
        d1[j] += 1
        d2[jp] += 1
    atoms = sum(math.ceil(s / 32) for s in pieces.values())
    honest = 3 * K / 32
    aligned = 2 * K / 32 + 2 * J
    # The T chain sum(a(b+f)) in consecutive order (K/32); the P2 = e(b+f) atoms on every piece; block 1 words
    # U_j + T_j with U_j = U_{j-1} + its pieces; block 2 words W_{j'-1} - its pieces; block 3 words W2_end - T_j + c_j.
    merges1 = sum(math.ceil(d / 2) + 1 for d in d1)
    merges2 = sum(math.ceil(d / 2) for d in d2)
    merges3 = J
    attack = K / 32 + atoms + merges1 + merges2 + merges3
    sizes = list(pieces.values())
    return {
        "pieces": len(pieces),
        "mean_piece": sum(sizes) / len(sizes),
        "max_piece": max(sizes),
        "mean_chunks_met": sum(d2) / J,
        "atoms": atoms,
        "honest": honest,
        "aligned_shortcut": aligned,
        "intersection_attack": attack,
        "attack_over_honest": attack / honest,
    }


def intersections(seeds, shapes=((2048, 64), (2048, 128), (2048, 256), (2048, 512), (2048, 1024), (8192, 64),
                                 (8192, 128), (8192, 256), (8192, 512), (8192, 1024), (7168, 64), (7168, 128),
                                 (7168, 512), (7168, 1024))):
    out = []
    for K, L in shapes:
        runs = []
        for s in range(seeds):
            rng = random.Random(1000 * K + 10 * L + s)
            perm = list(range(K))
            rng.shuffle(perm)
            runs.append(stack_costs(K, L, perm))
        ident = stack_costs(K, L, list(range(K)))
        keys = runs[0].keys()
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
        worst = min(r["intersection_attack"] for r in runs)
        out.append({
            "K": K, "L": L, "seeds": seeds, "random_mean": mean, "random_cheapest_attack": worst,
            "identity_order": ident,
            "aligned_gain_units_a_position": L - 64,
            "aligned_gain_share_of_stock": (3 * K / 32 - (2 * K / 32 + 2 * (K // L))) * 32 / K,
        })
    return out


def projection_bound(K, L, perm):
    """The most a pure-lane program can save, in products a dot product, on a repeated pair of blocks in `perm`."""
    J = K // L
    block = [i // L for i in range(K)]
    pieces = {}
    for jp, ch in enumerate(chunks_of(perm, L)):
        for i in ch:
            pieces[(block[i], jp)] = pieces.get((block[i], jp), 0) + 1
    corner = pieces.pop((J - 1, 0), 0)
    sizes = sorted(pieces.values(), reverse=True)
    free = sizes[:2 * J - 1]
    gain = corner + sum(free) + sum(max(0, s - 16) for s in sizes[2 * J - 1:])
    return {"gain": gain, "credited_share_of_pair": 1 - gain / (2 * K), "uncredited_share_of_stock": gain / K}


def projection_bounds(seeds, shapes=((2048, 64), (2048, 128), (2048, 256), (8192, 64), (8192, 128), (8192, 256),
                                     (8192, 512), (7168, 128), (7168, 256))):
    out = []
    for K, L in shapes:
        runs = []
        for s in range(seeds):
            rng = random.Random(1000 * K + 10 * L + s)
            perm = list(range(K))
            rng.shuffle(perm)
            runs.append(projection_bound(K, L, perm))
        worst = max(r["gain"] for r in runs)
        out.append({
            "K": K, "L": L, "seeds": seeds,
            "mean_credited_share_of_pair": sum(r["credited_share_of_pair"] for r in runs) / seeds,
            "worst_credited_share_of_pair": 1 - worst / (2 * K),
            # RFP-4 has two repeated pairs, so its uncredited tensor work is twice a pair's gain, against stock's K.
            "rfp4_uncredited_share_of_stock_worst": 2 * worst / K,
            "identity_order": projection_bound(K, L, list(range(K))),
        })
    return out


def rfp4_zero_data_one_bit(K, L, seed, m=2, n=2):
    """RFP-4's words at A = B = 0 with coins in {±64}: the s8 program's against the b1 program's, and their prices.

    A segment's sum is 4096·Σ s·t over its signs, and s·t = 2(u∧v + ū∧v̄) − 1 for the sign bits u, v, so a running word
    is 8192·popc([u, ū] ∧ [v, v̄]) − 4096·(its position): one b1 instruction a segment (2L bits, 256 to one) and one IMAD
    a word, against the s8 program's L/32 instructions a segment, in units of 32 a dot product."""
    rng = random.Random(seed)
    E = [[rng.choice((-1, 1)) for _ in range(K)] for _ in range(m)]
    G = [[rng.choice((-1, 1)) for _ in range(K)] for _ in range(m)]
    F = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(K)]
    pi, pi2 = list(range(K)), list(range(K))
    rng.shuffle(pi)
    rng.shuffle(pi2)
    agree = True
    for r in range(m):
        x = [64 * v for v in E[r] + [-E[r][k] for k in pi] + G[r] + [-G[r][k] for k in pi2]]
        for c in range(n):
            y = [64 * F[k][c] for k in list(range(K)) + pi + list(range(K)) + pi2]
            s8, acc, run = [], 0, 0
            for t in range(0, 4 * K, L):
                run += sum(a * b for a, b in zip(x[t:t + L], y[t:t + L]))
                s8.append(run)
                u = sum(1 << i for i, a in enumerate(x[t:t + L]) if a > 0)
                v = sum(1 << i for i, b in enumerate(y[t:t + L]) if b > 0)
                full = (1 << L) - 1
                acc += bin(u & v).count("1") + bin(~u & ~v & full).count("1")
                if s8[-1] != 8192 * acc - 4096 * (t + L):
                    agree = False
    honest = 4 * K
    b1 = (4 * K // L) * (32 * math.ceil(2 * L / 256) + 32)
    return {"K": K, "L": L, "words_agree": agree, "honest_units": honest, "b1_units": b1,
            "uncredited_share_of_stock": max(0, honest - b1) / K}


def rfp4_known_words(K, L, order, draws=2, m=2, n=2):
    """The running words of RFP-4 in block `order` that are 0 at zero activations (A = 0, B int7) for every draw.

    Each such word, like the final one, ends a chain an adversary runs backward from, so each skips a segment of L
    (N-SAKU's middle-segment skip): the skip costs (known words)·L/K of stock. In the order 1, 2, 3, 4 the word at 2K is
    A·(B+F), so 0; in 1, 3, 2, 4 the words at 2K and 3K are E·(B+F) + G·F and G·F."""
    zero = None
    for d in range(draws):
        rng = random.Random(7919 * K + 31 * L + d)
        E = [[rng.randint(-64, 64) for _ in range(K)] for _ in range(m)]
        G = [[rng.randint(-64, 64) for _ in range(K)] for _ in range(m)]
        F = [[rng.randint(-64, 64) for _ in range(n)] for _ in range(K)]
        B = [[rng.randint(-63, 63) for _ in range(n)] for _ in range(K)]
        pi, pi2 = list(range(K)), list(range(K))
        rng.shuffle(pi)
        rng.shuffle(pi2)
        for r in range(m):
            for c in range(n):
                blocks = {
                    1: ([E[r][i] for i in range(K)], [B[i][c] + F[i][c] for i in range(K)]),
                    2: ([-E[r][k] for k in pi], [B[k][c] + F[k][c] for k in pi]),
                    3: ([G[r][i] for i in range(K)], [F[i][c] for i in range(K)]),
                    4: ([-G[r][k] for k in pi2], [F[k][c] for k in pi2]),
                }
                x = sum((blocks[b][0] for b in order), [])
                y = sum((blocks[b][1] for b in order), [])
                run, words = 0, []
                for t in range(0, 4 * K, L):
                    run += sum(a * b for a, b in zip(x[t:t + L], y[t:t + L]))
                    words.append(run)
                z = {i for i, w in enumerate(words) if w == 0}
                zero = z if zero is None else zero & z
    known = sorted((i + 1) * L for i in zero)
    return {"K": K, "L": L, "order": list(order), "zero_words_at_position": known,
            "skip_share_of_stock": len(known) * L / K}


STACKS = {
    # blocks, X adds per entry of A, Y adds per entry of B, coin entries per entry of A and of B
    "RFP-3": {"blocks": 3, "x_adds": 1, "y_adds": 1, "coins_a": 1, "coins_b": 1},
    "RFP-4": {"blocks": 4, "x_adds": 2, "y_adds": 1, "coins_a": 2, "coins_b": 1},
}


def llama_terms(Ts=(512, 8192), Ls=(32, 64, 128, 256, 1024)):
    macs_tok = sum(K * n for _, K, n in LLAMA_1B)
    out = {"macs_a_token_a_layer": macs_tok, "stacks": {}}
    for name, st in STACKS.items():
        rows = []
        for T in Ts:
            stock = T * macs_tok
            form_x = sum(32 * st["x_adds"] * T * K / 4 for _, K, n in LLAMA_1B)
            form_y = sum(32 * st["y_adds"] * K * n / 4 for _, K, n in LLAMA_1B)
            coins = sum(st["coins_a"] * T * K + st["coins_b"] * K * n for _, K, n in LLAMA_1B)
            row = {
                "T": T,
                "alpha": st["blocks"],
                "forming_X_share": form_x / stock,
                "forming_Y_share": form_y / stock,
                "forming_share": (form_x + form_y) / stock,
                "coin_entries_a_mac": coins / stock,
                "coins_uncredited_share_one_bit": coins * SHA_UNITS_PER_BYTE / 8 / stock,
                "coins_uncredited_share_log2_129_bits": coins * math.log2(129) * SHA_UNITS_PER_BYTE / 8 / stock,
                "by_L": {},
            }
            for L in Ls:
                words = sum(st["blocks"] * T * n * (K // L) for _, K, n in LLAMA_1B if K % L == 0)
                a_bytes = sum(T * K for _, K, n in LLAMA_1B)
                b_per_mac = (4 * words + a_bytes) / stock
                row["by_L"][L] = {
                    "bytes_a_useful_mac": b_per_mac,
                    "hashing_over_stock_W1": b_per_mac * SHA_UNITS_PER_BYTE,
                    "shaspeed_delta_for_1pct": 0.01 / (b_per_mac * SHA_UNITS_PER_BYTE),
                }
            rows.append(row)
        out["stacks"][name] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"purity": purity(), "intersections": intersections(a.seeds), "projection_bounds": projection_bounds(a.seeds),
           "zero_data_one_bit": [rfp4_zero_data_one_bit(K, L, s) for K in (2048, 8192) for L in (64, 128, 256)
                                 for s in range(2)],
           "zero_activation_known_words": [rfp4_known_words(K, L, order) for K, L in ((2048, 128), (8192, 256))
                                           for order in ((1, 2, 3, 4), (1, 3, 2, 4))],
           "llama_3_2_1b": llama_terms()}
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
