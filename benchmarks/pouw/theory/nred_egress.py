"""N-RED on N-ALT's egress term (final): an independent recompute of DitherInstance's `n_max · W_mac · E`, the
adversary that realizes it (A-OUT, outsourcing within the egress budget), and what it costs T1.

The brief told N-RED to check N-ALT's egress arithmetic (138-268% of stock at a 32-bit-a-token cap,
compute-research/20261029T0903Z-report-np-alt) and to attack R1/R1b. This file recomputes the term from the
Llama-3.2-1B config and the DitherInstance statement WITHOUT importing `alt_roads.py`, so it is a cross-check, then
states the attack and its per-period saving on N-ALT's basis (so the rows line up with N-ALT's table).

The term, from the guarantee (`Security/Definitions/Pouw/Guarantees/DitherInstance.lean`): the slack ε_A carries
`n_max · W_mac · E / W_reg`, with E the bits exchanged across the replica's boundary (capped at served traffic,
excluding the challenger's signed coin stream), W_mac the honest block's priced cost a MAC (hashing included), and
n_max "the most MACs one exchanged coin bit carries: a node that forwards a call's coins and gets back only the root
has the outside machine do the call's MACs" (= N for a static call, in the file's words).

The attack A-OUT: a replica forwards one call's coin bits (and, within the served-traffic cap, the token id) to an
outside machine that already holds the weights; the outside machine runs the call and returns the root. The exchanged
bits carry the call's whole MAC count. With `ssn_1l_hw`'s per-ROW coins (96 bits a gate-up call, not one coin an
output element) the per-bit yield is K·N / row_bits ≫ N, so the granted slack is 138-268% of stock — larger than the
matmul itself, so the adversary may skip the entire matmul and still pass. T1 wants the gap ≤ 1%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Llama-3.2-1B, as served (fused qkv and gate-up), each linear (K in, N out); 16 identical layers.
LAYERS = 16
LINEARS = {"qkv": (2048, 3072), "o": (2048, 2048), "gate_up": (2048, 16384), "down": (8192, 2048)}
MACS_TOKEN = LAYERS * sum(k * n for k, n in LINEARS.values())

# The noise's coin budget (N-PREM's ssn_1l_hw): S steps + 15 bits a row, rounded to whole 32-bit words; 4 code lanes.
CODE_LANES = 4
COIN_BITS_OVER_STEPS = 15
SEED_BITS_PER_WINDOW = 512  # the seeded lead's window salt

# Prices, from the inputs N-ALT cites (SW-HASH / SW-PRICE / SW-NOISE).
ALPHA = 1.044                       # the matmul block's W1 cost a MAC (ssn_1l alpha), in bare-MAC units
CREDITED_HASH_BYTES_PER_MAC = 0.1307
SHA512_UNITS_PER_BYTE = {"floor": 908.0, "bar": 1774.85}
CAPS_BITS_PER_TOKEN = [32, 320]

# N-ALT's per-period basis (so these rows cross-check N-ALT's table directly).
STOCK_MAC_S = 3.96e14                # one H200 at tensor peak x 0.4
AND_PER_MAC = 2079 / 16             # 129.9375 AND gates an int8 MAC
DATACENTER_GPUS = 100_000
WORLD_MAC_YEAR = 9.36e30 / 2
PERIOD_S = {"window": 600, "week": 604_800, "month": 2_592_000, "year": 31_536_000}


def steps(k: int) -> int:
    """ssn_1l's k32 steps for a K-wide input: step 1 holds 32 − code_lanes data entries, each later step 31."""
    return 1 + math.ceil(max(0, k - (32 - CODE_LANES)) / 31)


def row_bits(k: int) -> int:
    """The call's coin bits a row, rounded to whole 32-bit words (coins ride in words)."""
    return 32 * math.ceil((steps(k) + COIN_BITS_OVER_STEPS) / 32)


def w_mac() -> dict:
    """The honest block's W1 cost a MAC, matmul plus credited hashing, at SHA-512's floor and HASH-38's bar."""
    return {p: ALPHA + CREDITED_HASH_BYTES_PER_MAC * u for p, u in SHA512_UNITS_PER_BYTE.items()}


def derive() -> dict:
    coin_bits = LAYERS * sum(row_bits(k) for k, _ in LINEARS.values())
    per_call = {name: k * n / row_bits(k) for name, (k, n) in LINEARS.items()}
    n_max = max(per_call.values())                       # the worst call's MACs a coin bit (A-OUT's yield)
    n_elem = max(n for _, n in LINEARS.values())         # the file's "N for a static call": a per-element coin
    wmac = w_mac()
    caps = {}
    for cap in CAPS_BITS_PER_TOKEN:
        granted = {p: cap * n_max * w / MACS_TOKEN for p, w in wmac.items()}       # the slack, share of stock
        mean = {p: cap * w / coin_bits for p, w in wmac.items()}                   # whole-forward mean
        n_req = {p: 0.01 * MACS_TOKEN / (cap * w) for p, w in wmac.items()}        # n_max that would give a 1% term
        row_cols = {p: min(nr * row_bits(k) / k for k, _ in LINEARS.values()) for p, nr in n_req.items()}
        elem_bits = {p: LAYERS * sum(n * math.ceil(k / nr) for k, n in LINEARS.values())
                     for p, nr in n_req.items()}
        caps[str(cap)] = {"granted_slack_share_of_stock": granted, "whole_forward_mean_share": mean,
                          "n_max_for_1pct": n_req, "row_coins_meet_it_up_to_columns": row_cols,
                          "element_coins_bits_per_token": elem_bits,
                          "r1_words_through_port_share": cap / MACS_TOKEN}
    return {"coin_bits_per_token": coin_bits, "macs_per_coin_bit_by_call": per_call, "n_max": n_max,
            "n_per_element_static": n_elem, "n_max_over_n_element": n_max / n_elem, "w_mac": wmac,
            "macs_per_coin_bit_mean": MACS_TOKEN / coin_bits, "share_of_stock_at_cap": caps,
            "seeded_lead_premise": f"false: {SEED_BITS_PER_WINDOW}-bit window salt carries every call's noise; "
                                   "one counted message outsources the whole window"}


def periods(share: float) -> dict:
    """AND gates on N-ALT's basis: a fixed `share` of the period's stock work at one H200, the 100k datacenter and
    the world, over window / week / month / year."""
    out = {}
    for scope, mac_s in (("one_h200", STOCK_MAC_S), ("datacenter", STOCK_MAC_S * DATACENTER_GPUS),
                         ("world", WORLD_MAC_YEAR / PERIOD_S["year"])):
        out[scope] = {p: share * mac_s * s * AND_PER_MAC for p, s in PERIOD_S.items()}
    return out


def row_used(k: int) -> int:
    """The coin bits a row actually reads (16-bit a₁ and one bit a later step): what an outsourcer must forward."""
    return steps(k) + COIN_BITS_OVER_STEPS


def summary_bits(k: int) -> int:
    """What a later call needs of an earlier call's coins when that call's noise is served and column-constant after
    step 1 (`ssn-1l-tag`: a₁·C + T(i)·G + Σ_t (1 + b_t)): a₁ and popcount(b), a row. G is a window's, amortized."""
    return 16 + math.ceil(math.log2(steps(k)))


def forward_calls() -> list:
    return [(layer, name, k, n) for layer in range(LAYERS) for name, (k, n) in LINEARS.items()]


def closure_max(bits_full, bits_summary) -> dict:
    """The best MACs a forwarded bit over call sets an outsider can run when the noise is served, so each call's input
    depends on every earlier call's noise (attention mixes tokens, so all rows of those calls): the outsider takes
    full coins for the calls it runs (T) and a summary for the rest of the prefix. Ratio (ΣT MACs)/(ΣT full +
    Σ rest summary), maximized per last call p by adding calls in decreasing MACs/(full − summary) while that beats
    the current ratio."""
    calls = forward_calls()
    best = {"ratio": 0.0}
    for p in range(len(calls)):
        _, _, kp, np_ = calls[p]
        num = kp * np_
        den = bits_full(kp) + sum(bits_summary(k) for _, _, k, _ in calls[:p])
        cands = sorted(((k * n, bits_full(k) - bits_summary(k), i) for i, (_, _, k, n) in enumerate(calls[:p])),
                       key=lambda c: -c[0] / c[1] if c[1] > 0 else -math.inf)
        taken = [p]
        for x, y, i in cands:
            if y > 0 and x / y <= num / den:
                break
            num += x
            den += y
            taken.append(i)
        if num / den > best["ratio"]:
            best = {"ratio": num / den, "last": f"layer {calls[p][0]} {calls[p][1]}",
                    "calls_run": sorted(f"{calls[i][0]}:{calls[i][1]}" for i in taken)}
    return best


LEAF_BYTES = {"r256k": 262_144, "r64k": 65_536, "r16k": 16_384, "r4k": 4_096}
DIGEST_BITS = 512        # a row digest x (hm96: b = x XOR M·y, leaf = SHA-512(prefix ‖ b ‖ c))
SALT_SIDE_BITS = 1024    # M·y and c, 128 B a leaf from the certifier (R1b's `cert` hiding)
WORD_BYTES_PER_MAC = 4 * ALPHA / 32
# N-CARD's space for R1b's cert designs at T 8,192 (art:14199aad…, run r20261029-084240-46c8; in N-ALT's inputs).
T8192 = {"r256k": {"m": 8192, "t_h_s": 1.874802, "judged_t2": (18.97, 58.77, 59.84)},
         "r16k": {"m": 8192, "t_h_s": 2.113547, "judged_t2": (34.73, 75.36, 67.46)}}
PORT_BYTES_S = 400e9 / 8


def corrections(d: dict) -> dict:
    """Three corrections to N-ALT's n_max, each a design-dependent max MACs a forwarded bit, and their terms."""
    wmac = d["w_mac"]

    def term(nmax: float, cap: int) -> dict:
        return {p: cap * nmax * w / MACS_TOKEN for p, w in wmac.items()}

    used = {name: k * n / row_used(k) for name, (k, n) in LINEARS.items()}
    n_used = max(used.values())
    hw = closure_max(row_used, row_used)                  # hw's later noise needs every b: no summary
    tag = closure_max(row_used, summary_bits)
    whole_used = MACS_TOKEN / (LAYERS * sum(row_used(k) for k, _ in LINEARS.values()))
    leaves = {}
    for name, lb in LEAF_BYTES.items():
        macs_leaf = lb / WORD_BYTES_PER_MAC
        # R1b's `cert` hiding: each hm96 leaf is SHA-512(prefix ‖ x ⊕ M·y ‖ c) with (M·y, c) issued to the node per
        # leaf by the certifier, so an outsider can't finish a leaf, let alone the root: it gets 1,024 bits a leaf
        # or returns the 512-bit row digest x. Prover-side salts (mask, L1/L2) give no such bound.
        k, n = LINEARS["gate_up"]
        bits_gu = row_used(k) + DIGEST_BITS * k * n / macs_leaf
        n_leaf = max(kk * nn / (row_used(kk) + DIGEST_BITS * kk * nn / macs_leaf) for kk, nn in LINEARS.values())
        leaves[name] = {"macs_a_leaf": macs_leaf, "n_max": n_leaf, "gate_up_bits_a_row": bits_gu,
                        "term": {str(c): term(n_leaf, c) for c in CAPS_BITS_PER_TOKEN},
                        "salt_side_share_of_word_bytes": (SALT_SIDE_BITS / 8) / lb}
        if name in T8192:
            t = T8192[name]
            words_b = t["m"] * MACS_TOKEN * WORD_BYTES_PER_MAC
            leaves[name]["salt_side_port_share_t8192"] = words_b / lb * (SALT_SIDE_BITS / 8) / t["t_h_s"] / PORT_BYTES_S
            leaves[name]["judged_t2"] = t["judged_t2"]
    need = {}
    for cap in CAPS_BITS_PER_TOKEN:
        need[str(cap)] = {p: 0.01 * MACS_TOKEN / (cap * w) * DIGEST_BITS * WORD_BYTES_PER_MAC
                          for p, w in wmac.items()}  # leaf bytes at which the leaf bound gives a 1% term
    return {"padding": {"per_call": used, "n_max": n_used, "factor": n_used / d["n_max"],
                        "term": {str(c): term(n_used, c) for c in CAPS_BITS_PER_TOKEN},
                        "whole_forward": {str(c): {p: c * w / (MACS_TOKEN / whole_used) for p, w in wmac.items()}
                                          for c in CAPS_BITS_PER_TOKEN}},
            "served_closure_hw": {**hw, "term": {str(c): term(hw["ratio"], c) for c in CAPS_BITS_PER_TOKEN}},
            "served_closure_tag": {**tag, "term": {str(c): term(tag["ratio"], c) for c in CAPS_BITS_PER_TOKEN},
                                   "summary_bits": {k: summary_bits(k) for k in (2048, 8192)}},
            "r1b_leaf_bound": leaves, "leaf_bytes_for_1pct": need}


def unbound_aj_saving() -> float:
    """R1b's arrival-order coins with A_j's bytes not bound into the proof (N-ALT's "a second A_j for one slot"):
    the prover sends a dummy A_j, gets the coins, then computes with an A whose data lane a step cancels the whole
    later noise (1 + b_t, now known), so every later word repeats its step-1 word. Model P, one pointer a row-step
    charged, as a share of stock over Llama's linears: all of the later words' H32 charge. Binding A_j's bytes (the
    certifier hashes what arrived; the proof opens A against it) leaves only A2c, which needs no coin foresight
    (it cancels the public 1 and repeats when b_t = 0: 51.6%), so arrival coins never close A2c."""
    save = sum(32 * (n - 1) * (steps(k) - 1) for k, n in LINEARS.values())
    return save / sum(n * k for k, n in LINEARS.values())


def attack(d: dict, cor: dict) -> dict:
    """A-OUT: outsource whole calls within the served-traffic cap. The outsider skips the calls' MACs and their
    hashing, so each forwarded bit removes n_max · W_mac of the honest's W1 work: 1.15% of the honest's MACs at a
    32-bit cap, worth 138-268% of stock because the honest's block costs W_mac ≈ 120-233 units a MAC."""
    slack32 = d["share_of_stock_at_cap"]["32"]["granted_slack_share_of_stock"]
    tag32 = cor["served_closure_tag"]["term"]["32"]
    return {"construction": "forward the coins of the calls to outsource (and the summaries of the calls they depend "
                            "on) to a machine that holds the weights and sees the request; it runs them, hashes "
                            "their words and returns the roots, all within the served-traffic cap",
            "saving_share_of_stock_cap32_n_alt": slack32,
            "saving_share_of_stock_cap32_tag_served": tag32,
            "share_of_honest_macs_cap32": 32 * d["n_max"] / MACS_TOKEN,
            "breaks_t1": True,
            "per_period_cap32_floor_tag_served": periods(tag32["floor"])}


def checks(d: dict) -> list:
    a = d["share_of_stock_at_cap"]["32"]["granted_slack_share_of_stock"]
    a320 = d["share_of_stock_at_cap"]["320"]["granted_slack_share_of_stock"]
    wmac = d["w_mac"]
    out = [
        ("coin bits a token = 9,216 (N-ALT)", d["coin_bits_per_token"] == 9216, str(d["coin_bits_per_token"])),
        ("n_max = gate-up's 2048*16384/96 = 349,525 (N-ALT)",
         abs(d["n_max"] - 2048 * 16384 / 96) < 1e-6, f"{d['n_max']:,.2f}"),
        ("per-call MACs a coin bit: qkv 65,536 / o 43,691 / down 58,254 (N-ALT)",
         abs(d["macs_per_coin_bit_by_call"]["qkv"] - 65536) < 1
         and abs(d["macs_per_coin_bit_by_call"]["o"] - 43691) < 1
         and abs(d["macs_per_coin_bit_by_call"]["down"] - 58254) < 1,
         ", ".join(f"{k} {v:,.0f}" for k, v in d["macs_per_coin_bit_by_call"].items())),
        ("mean MACs a coin bit = 105,586 (N-ALT)",
         abs(d["macs_per_coin_bit_mean"] - 105586) < 1, f"{d['macs_per_coin_bit_mean']:,.2f}"),
        ("W_mac 119.7 (floor) to 233.0 (bar) W1 units a MAC (N-ALT)",
         abs(wmac["floor"] - 119.7) < 0.1 and abs(wmac["bar"] - 233.0) < 0.1,
         f"{wmac['floor']:.2f} / {wmac['bar']:.2f}"),
        ("32-bit cap: granted slack 137.6% (floor) to 267.8% (bar) of stock (N-ALT)",
         abs(a["floor"] - 1.376) < 0.002 and abs(a["bar"] - 2.678) < 0.002,
         f"{100*a['floor']:.1f}% / {100*a['bar']:.1f}%"),
        ("320-bit cap: ten times the 32-bit term",
         abs(a320["floor"] / a["floor"] - 10) < 1e-6 and abs(a320["bar"] / a["bar"] - 10) < 1e-6,
         f"{100*a320['floor']:.0f}% / {100*a320['bar']:.0f}%"),
        ("whole-forward mean at 32 bits: 41.6% (floor) to 80.9% (bar) (N-ALT)",
         abs(d["share_of_stock_at_cap"]["32"]["whole_forward_mean_share"]["floor"] - 0.416) < 0.003
         and abs(d["share_of_stock_at_cap"]["32"]["whole_forward_mean_share"]["bar"] - 0.809) < 0.003,
         f"{100*d['share_of_stock_at_cap']['32']['whole_forward_mean_share']['floor']:.1f}% / "
         f"{100*d['share_of_stock_at_cap']['32']['whole_forward_mean_share']['bar']:.1f}%"),
        ("a 1% term needs n_max 1,305 (bar) to 2,540 (floor) MACs a bit at 32 bits (N-ALT)",
         abs(d["share_of_stock_at_cap"]["32"]["n_max_for_1pct"]["bar"] - 1305) < 2
         and abs(d["share_of_stock_at_cap"]["32"]["n_max_for_1pct"]["floor"] - 2540) < 2,
         f"{d['share_of_stock_at_cap']['32']['n_max_for_1pct']['bar']:,.0f} / "
         f"{d['share_of_stock_at_cap']['32']['n_max_for_1pct']['floor']:,.0f}"),
        ("row coins close it only up to 46-89 columns (N-ALT: no real linear)",
         45.5 <= d["share_of_stock_at_cap"]["32"]["row_coins_meet_it_up_to_columns"]["bar"] <= 46.5
         and 89 <= d["share_of_stock_at_cap"]["32"]["row_coins_meet_it_up_to_columns"]["floor"] <= 90,
         f"{d['share_of_stock_at_cap']['32']['row_coins_meet_it_up_to_columns']['bar']:.1f} / "
         f"{d['share_of_stock_at_cap']['32']['row_coins_meet_it_up_to_columns']['floor']:.1f}"),
        ("per-element coins: 917,504 (bar) and 475,136 (floor) bits a token (N-ALT's 475,136-917,504)",
         d["share_of_stock_at_cap"]["32"]["element_coins_bits_per_token"]["bar"] == 917504
         and d["share_of_stock_at_cap"]["32"]["element_coins_bits_per_token"]["floor"] == 475136,
         f"{d['share_of_stock_at_cap']['32']['element_coins_bits_per_token']['bar']:,} / "
         f"{d['share_of_stock_at_cap']['32']['element_coins_bits_per_token']['floor']:,}"),
        ("R1 words through the port: 3.3e-8 of stock at 32 bits (N-ALT)",
         abs(d["share_of_stock_at_cap"]["32"]["r1_words_through_port_share"] - 3.29e-8) < 1e-10,
         f"{d['share_of_stock_at_cap']['32']['r1_words_through_port_share']:.3e}"),
        # N-RED's own: the gap between the file's per-element n_max and the realized per-row n_max is K/row_bits.
        ("n_max / N (gate-up) = K/row_bits = 2048/96 = 21.33: per-row coins inflate the per-element premise",
         abs(d["n_max_over_n_element"] - 2048 / 96) < 1e-6, f"{d['n_max_over_n_element']:.4f}"),
    ]
    return out


def checks_cor(cor: dict) -> list:
    pad, hw, tag, lv = cor["padding"], cor["served_closure_hw"], cor["served_closure_tag"], cor["r1b_leaf_bound"]
    gu = 2048 * 16384
    return [
        ("padding: an outsourcer forwards the 82 used bits of a K-2,048 row, not 96: n_max = 2048*16384/82",
         abs(pad["n_max"] - gu / 82) < 1e-6, f"{pad['n_max']:,.1f} (x{pad['factor']:.4f})"),
        ("hw served: best closed set is layer 0's qkv + o + gate-up at 44,040,192 / 246 MACs a bit",
         abs(hw["ratio"] - (6291456 + 4194304 + gu) / 246) < 1e-6 and hw["last"] == "layer 0 gate_up",
         f"{hw['ratio']:,.1f} at {hw['last']}"),
        ("tag served: earlier calls need a₁ + popcount (23 bits a K-2,048 row); best is layer 0 gate-up at "
         "2048*16384/(82 + 2*23)", abs(tag["ratio"] - gu / 128) < 1e-6 and tag["last"] == "layer 0 gate_up",
         f"{tag['ratio']:,.1f} at {tag['last']}"),
        ("unbound A_j (R1b, coins on arrival, bytes not bound): every later word repeats, ~103% of stock",
         abs(unbound_aj_saving() - 62703168 / 60817408) < 1e-9, f"{100 * unbound_aj_saving():.1f}%"),
        ("R1b leaf bound at r256k: n_max ≈ 3,886 MACs a bit (gate-up: 82 coin bits + a digest a 256 KiB leaf)",
         3880 < lv["r256k"]["n_max"] < 3890, f"{lv['r256k']['n_max']:,.1f}"),
        ("R1b leaf bound at r16k: term < 0.2% of stock at 32 bits at both prices, < 1% at 320 bits at the floor",
         max(lv["r16k"]["term"]["32"].values()) < 0.002 and lv["r16k"]["term"]["320"]["floor"] < 0.01,
         f"{100*lv['r16k']['term']['32']['floor']:.3f}% / {100*lv['r16k']['term']['32']['bar']:.3f}% (32); "
         f"{100*lv['r16k']['term']['320']['floor']:.2f}% / {100*lv['r16k']['term']['320']['bar']:.2f}% (320)"),
        ("R1b salt sides at r256k are 0.05% of the words' bytes (N-ALT)",
         abs(lv["r256k"]["salt_side_share_of_word_bytes"] - 128 / 262144) < 1e-12,
         f"{100*lv['r256k']['salt_side_share_of_word_bytes']:.3f}%"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    d = derive()
    cor = corrections(d)
    a = attack(d, cor)
    ck = checks(d) + checks_cor(cor)
    ok = all(c[1] for c in ck)
    bounds = {"n_alt_cap32_floor": d["share_of_stock_at_cap"]["32"]["granted_slack_share_of_stock"]["floor"],
              "padding_cap32_floor": cor["padding"]["term"]["32"]["floor"],
              "tag_served_cap32_floor": cor["served_closure_tag"]["term"]["32"]["floor"],
              "hw_served_cap32_floor": cor["served_closure_hw"]["term"]["32"]["floor"],
              "r1b_r256k_cap32_floor": cor["r1b_leaf_bound"]["r256k"]["term"]["32"]["floor"],
              "r1b_r16k_cap32_floor": cor["r1b_leaf_bound"]["r16k"]["term"]["32"]["floor"],
              "r1b_r16k_cap320_floor": cor["r1b_leaf_bound"]["r16k"]["term"]["320"]["floor"],
              "unbound_aj": unbound_aj_saving()}
    rep = {"derive": d, "corrections": cor, "attack": a, "unbound_aj_saving": unbound_aj_saving(),
           "stock_per_period": periods(1.0), "per_period": {k: periods(v) for k, v in bounds.items()},
           "checks": [{"check": c[0], "pass": c[1], "got": c[2]} for c in ck], "all_pass": ok}
    if args.json:
        import pathlib
        pathlib.Path(args.json).write_text(json.dumps(rep, indent=1))
    print(f"MACS/token {MACS_TOKEN:,}; coin bits/token {d['coin_bits_per_token']:,}; n_max {d['n_max']:,.0f} "
          f"(= {d['n_max_over_n_element']:.1f}x the per-element N)")
    for cap in ("32", "320"):
        f = lambda t: f"{100*t['floor']:.3g}% / {100*t['bar']:.3g}%"
        print(f"cap {cap}: N-ALT {f(d['share_of_stock_at_cap'][cap]['granted_slack_share_of_stock'])}; "
              f"padding {f(cor['padding']['term'][cap])}; hw served {f(cor['served_closure_hw']['term'][cap])}; "
              f"tag served {f(cor['served_closure_tag']['term'][cap])}")
        print("   R1b leaf bound: " + "; ".join(f"{name} {f(v['term'][cap])}" for name, v in cor["r1b_leaf_bound"].items()))
        print(f"   leaf bytes for a 1% term: floor {cor['leaf_bytes_for_1pct'][cap]['floor']:,.0f} / "
              f"bar {cor['leaf_bytes_for_1pct'][cap]['bar']:,.0f}")
    for name in ("r256k", "r16k"):
        v = cor["r1b_leaf_bound"][name]
        print(f"{name}: salt sides {100*v['salt_side_port_share_t8192']:.2f}% of a 400 Gb/s port at T 8,192; "
              f"judged T2 {v['judged_t2']}")
    print(f"unbound A_j: {100*unbound_aj_saving():.1f}% of stock")
    for k, v in bounds.items():
        p = periods(v)
        print(f"periods {k} ({100*v:.4g}%): H200 " + " / ".join(f"{x:.2e}" for x in p["one_h200"].values())
              + "; DC " + " / ".join(f"{x:.2e}" for x in p["datacenter"].values())
              + "; world " + " / ".join(f"{x:.2e}" for x in p["world"].values()))
    for c in ck:
        print(f"{'PASS' if c[1] else 'FAIL'}  {c[0]}  [{c[2]}]")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
