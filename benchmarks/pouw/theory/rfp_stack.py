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
    res = {"purity": purity(), "intersections": intersections(a.seeds), "llama_3_2_1b": llama_terms()}
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
