"""The cost trade of aggregated words for Llama-3.2-1B int8 at T 8,192 (and T 512): the proof block each g implies, its
ANDs, proving GPU-s a window at the draws the profile needs, the serve-time bytes and their hashing, and J's two terms.
Every input is a cited measurement or an estimate, labelled in `INPUTS`; every output is an estimate.

A group is a g_r x g_c tile of one call's cells (g = g_r g_c) and one segment of L inner indices; its word is
W = sum f(T_s - T_(s-1)) mod 2^16 over the group's cells. A segment s < S's proof block is one group: g L MACs on the
noised operands, f and the sum, and the openings of the g_r A rows, g_c B rows and the word's row, at C-Flock's 81,402
ANDs a SHA-512 compression. The s = S block also checks O (see `block`), and the cap applies to the larger of the two.
Per-cell (g = 1) is Dither's blocks. Serve-time hashing is priced at GD-HASH's measured cost a byte beside the GEMM (no
free share), scaled to the word row's size, and also, for comparison, at p1a-levers' free-share model. J is against the
brief's stock step; `cap_knob` gives the best g and tile at each block cap 2^26..2^30."""
from __future__ import annotations

import json
import math
import sys
from math import ceil

INPUTS = {
    "cells_k2048_T8192": (16 * (3072 + 2048 + 16384) * 8192, "H1's 48 K=2048 linears (qkv, o, gate_up x 16 layers), exact"),
    "cells_down_T8192": (16 * 2048 * 8192, "the 16 down-projections at K 8,192, exact"),
    "macs_T8192": (7.9715e12, "declared int8 MACs a step (note:compute-research/20261028T1715Z-report-p1a-levers)"),
    "free_pouw_GB_at_80": (0.07, "hash bytes left for PoUW at 80% beside a 70 KB/token Commit (p1a-levers)"),
    "h_beside_16k": (1465, "MAC-times a byte hashed beside the GEMM at 16 KiB rows, T 8,192, up to 1.9 GB a forward "
                           "(note:compute-research/20261028T1945Z-report-gd-hash, r20261028-192103-b3eb)"),
    "h_grid_16k": (1910, "MAC-times a byte past the beside capacity, the hash on the whole grid (gd-hash)"),
    "beside_cap_GB_16k": (1.9, "the most bytes a forward hashed beside the GEMM at 16 KiB rows (gd-hash)"),
    "h_scale_4k": (521.0 / 398.0, "estimate: h at 4 KiB word rows over 16 KiB, as the whole-card SHA-512 bar's 521 GB/s over 398 (BENCH-34)"),
    "h_scale_1k": (((2 + 9 + 1) / 8) / ((2 + 129 + 1) / 128), "estimate: h at 1 KiB word rows over 16 KiB, as compressions a byte "
                                                             "(hm96's two hiding ones and one tree node a row included)"),
    "clock_Hz": (1.769e9, "the measured SM clock at T 8,192 (BENCH-34)"),
    "bar_GBps_16k": (521.0, "SHA-512 across the card at 16 KiB rows (BENCH-34 / commit-bytes)"),
    "step80_ms": (11.14, "the 80% step at 1,769 MHz (p1a-levers: 19.71 M SM-cycles)"),
    "ideal_ms_1769": (15.769e6 / 1.769e9 * 1e3, "ideal cycles 15.769 M at 1,769 MHz (p1a-levers)"),
    "stock_u": (0.2397, "stock vLLM tensor utilization at T 8,192 (BENCH-34, run b57f's comparison)"),
    "t_stock_ms": (35.88, "the brief's stock: BENCH-34's stock-n2048 arm, eager sync, 23.97% at 1,833 MHz, a T 8,192 step "
                          "(art:a5ea06dd0a7cb46424f76fa4b0524327ddbc57659f7dfa2888ddcf8c50acdcfa)"),
    "t_stock_fast_ms": (21.60, "the fastest unverified arm, stock-async-stack-n2048 (r20261028-075211-2cd6, GD-SMELL S4); "
                               "both J terms scale by t_stock_ms over it and the best g doesn't move"),
    "q": (math.log(2 ** 40) / 0.01, "draws a window, Lambda/eps at delta 2^-40, eps 1%, uniform weights (estimate)"),
    "P3": (2.8e9, "C-Flock --zk ANDs a GPU-s (modelled, estimate)"),
    "ands_per_mac": (146_345 / 1024, "PouwDitherBlock_v1 at L 1,024 (counted by agg_ands.py; MACs and noise, no openings)"),
    "ands_per_mac_cal": (350.0, "TOP-138's calibration: the served H1 tile, noise and openings included (inbox 19:16Z)"),
    "f_ands": (214, "mix4 on bits (counted by agg_ands.py)"),
    "sum_ands_per_word": (888 / 64, "sum of 64 words mod 2^16 (counted), per word"),
    "cap": (2 ** 27, "C-Flock's compose refuses a proof block above 2^27 ANDs (inbox 19:16Z)"),
    "cap_padded": (2 ** 26, "the same with compose's power-of-two padding, k_pad 1 (note:compute-research/20261028T1915Z-report-gd-model)"),
    "sha512_ands": (81_402, "C-Flock's SHA-512 circuit, rows a compression, 1.40x optimize.SHA512_ANDS's 58,120 (inbox 20:00Z, "
                            "note:compute-research/20261028T1938Z-report-gd-sis); hm96's hiding part is two of them"),
    "o_row": (16_384, "O's row in Dither's value root: 16 KiB hm96 rows of 4,096 u32 (dither PROTOCOL, the value root's layout)"),
    "c_O": (2.0, "estimate: the s = S block's O check over K at 32 bits and the binding identity's whole-row dots, in "
                 "noised-MAC equivalents a cell per inner index"),
    "f_instr_per_cell": (19, "mix4 in an epilogue: 4 x (SHF, LOP3, IMAD, LOP3) + SHF, LOP3 + IADD (estimate)"),
    "int_ops_per_s": (132 * 64 * 1.769e9, "H200 INT32 lanes: 132 SMs x 64 a clock at 1,769 MHz (estimate; 2x if IMAD on FMA)"),
}
V = {k: v for k, (v, _) in INPUTS.items()}


def row_ands(nbytes: int) -> int:
    return V["sha512_ands"] * (2 + ceil((nbytes + 17) / 128))


def tile(g: int, cA: float, cB: float) -> tuple[int, int]:
    """The g_r x g_c factorization of g (powers of two) with the cheapest openings."""
    best = None
    for k in range(int(math.log2(g)) + 1):
        gr, gc = 1 << k, g >> k
        c = gr * cA + gc * cB
        if best is None or c < best[0]:
            best = (c, gr, gc)
    return best[1], best[2]


def block(g: int, L: int, K: int, a_row: int, b_row: int | None, w_row: int, per_mac: float) -> dict:
    """A group's two proof blocks, Dither's segments s < S and s = S, and their mean over a cell's S segments (the draws
    taken uniform over segments). Per-cell (g = 1) is Dither's: its s = S block opens the whole rows, O's row and
    T_(S-1)'s, and checks segment S and O. An aggregated group keeps no T_(S-1) a cell, so its s = S block recomputes
    every segment over the whole K and checks the S - 1 words and O."""
    S = ceil(K / L)
    cA = row_ands(a_row)
    cB = 0 if b_row is None else row_ands(b_row)
    gr, gc = tile(g, cA, cB) if g > 1 else (1, 1)
    fsum = g * (V["f_ands"] + V["sum_ands_per_word"]) if g > 1 else 0
    word_rows = 1 if g > 1 else (1 + 2 * (S - 2)) / (S - 1) if S > 1 else 0
    inner = {"macs": g * L * per_mac, "f_and_sum": fsum, "open_A": gr * cA, "open_B": gc * cB,
             "open_word": word_rows * row_ands(w_row)}
    last_macs = g * (L if g == 1 else K) + g * V["c_O"] * K
    last = {"macs": last_macs * per_mac, "f_and_sum": fsum * (S - 1), "open_A": gr * cA, "open_B": gc * cB,
            "open_O": row_ands(V["o_row"]), "open_word": row_ands(w_row) if S > 1 else 0}
    if per_mac == V["ands_per_mac"]:
        a_in, a_last = sum(inner.values()), sum(last.values())
    else:
        a_in, a_last = inner["macs"] + inner["f_and_sum"], last["macs"] + last["f_and_sum"]
    if S == 1:
        a_in = a_last
    mean = ((S - 1) * a_in + a_last) / S
    top = max(a_in, a_last)
    return {"g": g, "tile": [gr, gc], "L": L, "K": K, "S": S, "w_row": w_row, "block_macs": g * L,
            "parts": inner, "parts_last": last, "ands_inner": a_in, "ands_last": a_last, "ands": mean, "ands_max": top,
            "fits_2^27": top <= V["cap"], "fits_2^26": top <= V["cap_padded"]}


def window(g: int, L: int, K: int, steps: float, blk: dict, words_per_cell: float, cells: float) -> dict:
    stock_ms = V["t_stock_ms"]
    prove_s = V["q"] * blk["ands"] / V["P3"]
    gb = cells * 2 * words_per_cell / g / 1e9
    exposed_free_share_ms = max(0.0, gb - V["free_pouw_GB_at_80"]) / V["bar_GBps_16k"] * 1e3
    h = V["h_beside_16k"] if gb <= V["beside_cap_GB_16k"] else V["h_grid_16k"]
    h *= {16384: 1.0, 4096: V["h_scale_4k"], 1024: V["h_scale_1k"]}[blk["w_row"]]
    exposed_ms = gb * 1e9 * h / (540_672 * V["clock_Hz"]) * 1e3
    f_ms = cells * V["f_instr_per_cell"] / V["int_ops_per_s"] * 1e3 if g > 1 else 0.0
    return {"pouw_GB_a_step": gb, "hash_exposed_ms": exposed_ms, "hash_exposed_ms_p1a_free_share": exposed_free_share_ms,
            "f_epilogue_ms_if_unhidden": f_ms,
            "prove_GPU_s_a_window": prove_s, "stock_ms_a_step": stock_ms,
            "J_serve_hidden_f": exposed_ms / stock_ms, "J_serve_unhidden_f": (exposed_ms + f_ms) / stock_ms,
            "J_prove": prove_s / (steps * stock_ms / 1e3)}


def main(out: str) -> None:
    K, L = 2048, 1024
    cells = V["cells_k2048_T8192"]
    res = {"inputs": {k: {"value": v, "source": s} for k, (v, s) in INPUTS.items()}, "rows": [], "breakeven": {}}
    defaults = {"a_row": 4096, "b_row": 2048, "w_row": 16384}
    for g in (1, 8, 16, 32, 64, 128, 256, 512):
        for per_mac_key in ("ands_per_mac", "ands_per_mac_cal"):
            b = block(g, L, K, defaults["a_row"], defaults["b_row"], defaults["w_row"], V[per_mac_key])
            for steps in (1, 100, 1000, 10000):
                res["rows"].append({"per_mac": per_mac_key, "steps_a_window": steps, **b,
                                    **window(g, L, K, steps, b, 1, cells)})
    for per_mac_key in ("ands_per_mac", "ands_per_mac_cal"):
        b1 = block(1, L, K, **defaults, per_mac=V[per_mac_key])
        w1 = window(1, L, K, 1, b1, 1, cells)
        for g in (8, 32, 64):
            bg = block(g, L, K, **defaults, per_mac=V[per_mac_key])
            wg = window(g, L, K, 1, bg, 1, cells)
            for f_key in ("J_serve_hidden_f", "J_serve_unhidden_f"):
                d_serve = w1["J_serve_hidden_f"] - wg[f_key]
                d_prove = wg["J_prove"] - w1["J_prove"]
                res["breakeven"][f"{per_mac_key}_g{g}_{f_key}"] = {
                    "steps_a_window_for_aggregation_to_win": (d_prove / d_serve) if d_serve > 0 else None,
                    "J_prove_one_step": [w1["J_prove"], wg["J_prove"]], "J_serve": [w1["J_serve_hidden_f"], wg[f_key]]}
    res["row_sensitivity"] = {}
    for a_row in (2048, 4096, 16384):
        for b_row in (2048, 16384, None):
            key = f"A{a_row}_B{b_row or 'public'}"
            b1 = block(1, L, K, a_row, b_row, defaults["w_row"], V["ands_per_mac"])
            bg = block(32, L, K, a_row, b_row, defaults["w_row"], V["ands_per_mac"])
            w1, wg = window(1, L, K, 1, b1, 1, cells), window(32, L, K, 1, bg, 1, cells)
            d_serve = w1["J_serve_hidden_f"] - wg["J_serve_hidden_f"]
            res["row_sensitivity"][key] = {
                "per_cell_ands": b1["ands"], "g32_ands": bg["ands"], "g32_tile": bg["tile"], "g32_fits": bg["fits_2^27"],
                "ratio": bg["ands"] / b1["ands"], "prove_s": [w1["prove_GPU_s_a_window"], wg["prove_GPU_s_a_window"]],
                "breakeven_steps_f_hidden": (wg["J_prove"] - w1["J_prove"]) / d_serve}
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["row_sensitivity"][key].items()})
    res["J_table"] = {}
    for steps in (1, 300, 1000, 10_000, 100_000):
        row = {}
        for k in range(0, 9):
            g = 1 << k
            b = block(g, L, K, **defaults, per_mac=V["ands_per_mac"])
            w = window(g, L, K, steps, b, 1, cells)
            row[g] = {"J_hidden_f": w["J_serve_hidden_f"] + w["J_prove"], "J_unhidden_f": w["J_serve_unhidden_f"] + w["J_prove"],
                      "fits_2^26": b["fits_2^26"]}
        res["J_table"][steps] = row
        print("J", steps, " ".join(f"{g}:{v['J_hidden_f']:.3g}/{v['J_unhidden_f']:.3g}" for g, v in row.items()))
    res["g_opt"] = {}
    for f_key in ("J_serve_hidden_f", "J_serve_unhidden_f"):
        for steps in (1, 100, 1000, 3000, 10_000, 30_000, 100_000):
            best = None
            for k in range(0, 10):
                g = 1 << k
                b = block(g, L, K, **defaults, per_mac=V["ands_per_mac"])
                if not b["fits_2^27"]:
                    continue
                w = window(g, L, K, steps, b, 1, cells)
                J = w[f_key] + w["J_prove"]
                if best is None or J < best[1]:
                    best = (g, J, w[f_key], w["J_prove"])
            res["g_opt"][f"{f_key}_steps{steps}"] = {"g": best[0], "J": best[1], "J_serve": best[2], "J_prove": best[3]}
            print("g_opt", f_key, steps, res["g_opt"][f"{f_key}_steps{steps}"])
    res["cap_knob"] = {}
    for k_log in (26, 27, 28, 29, 30):
        row = {"g_max": max(1 << k for k in range(0, 16)
                            if block(1 << k, L, K, **defaults, per_mac=V["ands_per_mac"])["ands_max"] <= 2 ** k_log)}
        for steps in (1, 1000, 10_000, 100_000, 1_000_000):
            cands = []
            for k in range(0, 16):
                g = 1 << k
                b = block(g, L, K, **defaults, per_mac=V["ands_per_mac"])
                w = window(g, L, K, steps, b, 1, cells)
                cands.append((w["J_serve_hidden_f"] + w["J_prove"], g, b["tile"], b["ands_max"] <= 2 ** k_log))
            free = min(cands)
            capped = min(c for c in cands if c[3])
            row[steps] = {"g_opt": capped[1], "tile": capped[2], "J": capped[0], "g_opt_uncapped": free[1],
                          "J_uncapped": free[0], "cap_binds": capped[1] != free[1]}
        res["cap_knob"][k_log] = row
        print("cap", k_log, row)
    res["w_row_sensitivity"] = {}
    for w_row in (1024, 4096, 16384):
        rows = {**defaults, "w_row": w_row}
        entry = {"g1_ands": block(1, L, K, **rows, per_mac=V["ands_per_mac"])["ands"],
                 "g128_ands": block(128, L, K, **rows, per_mac=V["ands_per_mac"])["ands"]}
        for steps in (1, 1000, 10_000, 100_000):
            cands = []
            for k in range(0, 10):
                b = block(1 << k, L, K, **rows, per_mac=V["ands_per_mac"])
                if b["fits_2^27"]:
                    w = window(1 << k, L, K, steps, b, 1, cells)
                    cands.append((w["J_serve_hidden_f"] + w["J_prove"], 1 << k, w["J_serve_hidden_f"], w["J_prove"]))
            J, g, js, jp = min(cands)
            entry[steps] = {"g_opt": g, "J": J, "J_serve": js, "J_prove": jp}
        res["w_row_sensitivity"][w_row] = entry
        print("w_row", w_row, entry)
    saved = V["c_O"]
    res["c_O_sensitivity"] = {}
    for c in (0.0, 1.0, 2.0, 3.0):
        V["c_O"] = c
        fits = {cap: max(1 << k for k in range(0, 12)
                         if block(1 << k, L, K, **defaults, per_mac=V["ands_per_mac"])["ands_max"] <= 2 ** cap)
                for cap in (26, 27, 28)}
        res["c_O_sensitivity"][c] = {"g_max": fits, "g1_mean": block(1, L, K, **defaults, per_mac=V["ands_per_mac"])["ands"]}
    V["c_O"] = saved
    print("c_O", res["c_O_sensitivity"])
    saved = V["sha512_ands"]
    res["sha_sensitivity"] = {}
    for sha in (58_120, 81_402):
        V["sha512_ands"] = sha
        b1, b128 = (block(g, L, K, **defaults, per_mac=V["ands_per_mac"]) for g in (1, 128))
        res["sha_sensitivity"][sha] = {"g1_ands": b1["ands"], "g128_ands": b128["ands"], "g128_tile": b128["tile"]}
    V["sha512_ands"] = saved
    print("sha", res["sha_sensitivity"])
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["rows"]:
        if r["steps_a_window"] in (1, 10000):
            print(f"g {r['g']:4d} {r['per_mac']:16s} tile {r['tile']} ands {r['ands'] / 1e6:7.2f}M last {r['ands_last'] / 1e6:7.2f}M fits27 {r['fits_2^27']!s:5s} "
                  f"fits26 {r['fits_2^26']!s:5s} "
                  f"GB {r['pouw_GB_a_step']:.3f} exp {r['hash_exposed_ms']:.2f}ms f {r['f_epilogue_ms_if_unhidden']:.2f}ms "
                  f"prove {r['prove_GPU_s_a_window']:.1f}s steps {r['steps_a_window']} J_prove {r['J_prove']:.3g}")
    print(json.dumps(res["breakeven"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/cost.json")
