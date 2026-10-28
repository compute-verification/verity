"""The cost trade of aggregated words for Llama-3.2-1B int8 at T 8,192 (and T 512): the proof block each g implies, its
ANDs, proving GPU-s a window at the draws the profile needs, the serve-time bytes and their hashing, and J's two terms.
Every input is a cited measurement or an estimate, labelled in `INPUTS`; every output is an estimate.

A group is a g_r x g_c tile of one call's cells (g = g_r g_c) and one segment of L inner indices; its word is
W = sum f(T_s - T_(s-1)) mod 2^16 over the group's cells. A proof block is one group: g L MACs on the noised operands, f
and the sum, and the openings of the g_r A rows, g_c B rows and the word's row. Per-cell (g = 1) is Dither's block. Serve-time
hashing is priced at GD-HASH's measured cost a byte beside the GEMM (no free share), and also, for comparison, at
p1a-levers' free-share model."""
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
    "clock_Hz": (1.769e9, "the measured SM clock at T 8,192 (BENCH-34)"),
    "bar_GBps_16k": (521.0, "SHA-512 across the card at 16 KiB rows (BENCH-34 / commit-bytes)"),
    "step80_ms": (11.14, "the 80% step at 1,769 MHz (p1a-levers: 19.71 M SM-cycles)"),
    "ideal_ms_1769": (15.769e6 / 1.769e9 * 1e3, "ideal cycles 15.769 M at 1,769 MHz (p1a-levers)"),
    "stock_u": (0.2397, "stock vLLM tensor utilization at T 8,192 (BENCH-34, run b57f's comparison)"),
    "q": (math.log(2 ** 40) / 0.01, "draws a window, Lambda/eps at delta 2^-40, eps 1%, uniform weights (estimate)"),
    "P3": (2.8e9, "C-Flock --zk ANDs a GPU-s (modelled, estimate)"),
    "ands_per_mac": (146_345 / 1024, "PouwDitherBlock_v1 at L 1,024 (counted by agg_ands.py; MACs and noise, no openings)"),
    "ands_per_mac_cal": (350.0, "TOP-138's calibration: the served H1 tile, noise and openings included (inbox 19:16Z)"),
    "f_ands": (214, "mix4 on bits (counted by agg_ands.py)"),
    "sum_ands_per_word": (888 / 64, "sum of 64 words mod 2^16 (counted), per word"),
    "cap": (2 ** 27, "C-Flock's compose refuses a proof block above 2^27 ANDs (inbox 19:16Z)"),
    "cap_padded": (2 ** 26, "the same with compose's power-of-two padding, k_pad 1 (note:compute-research/20261028T1915Z-report-gd-model)"),
    "f_instr_per_cell": (19, "mix4 in an epilogue: 4 x (SHF, LOP3, IMAD, LOP3) + SHF, LOP3 + IADD (estimate)"),
    "int_ops_per_s": (132 * 64 * 1.769e9, "H200 INT32 lanes: 132 SMs x 64 a clock at 1,769 MHz (estimate; 2x if IMAD on FMA)"),
}
V = {k: v for k, (v, _) in INPUTS.items()}
SHA512_ANDS, HM96_ANDS = 58_120, 116_240


def row_ands(nbytes: int) -> int:
    return HM96_ANDS + SHA512_ANDS * ceil((nbytes + 17) / 128)


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
    cA = row_ands(a_row)
    cB = 0 if b_row is None else row_ands(b_row)
    gr, gc = tile(g, cA, cB) if g > 1 else (1, 1)
    macs = g * L
    parts = {"macs": macs * per_mac, "f_and_sum": (g * (V["f_ands"] + V["sum_ands_per_word"]) if g > 1 else 0),
             "open_A": gr * cA, "open_B": gc * cB, "open_word": row_ands(w_row)}
    total = sum(parts.values()) if per_mac == V["ands_per_mac"] else macs * per_mac + parts["f_and_sum"]
    return {"g": g, "tile": [gr, gc], "L": L, "K": K, "block_macs": macs, "parts": parts, "ands": total,
            "fits_2^27": total <= V["cap"], "fits_2^26": total <= V["cap_padded"]}


def window(g: int, L: int, K: int, steps: float, blk: dict, words_per_cell: float, cells: float) -> dict:
    stock_ms = V["ideal_ms_1769"] / V["stock_u"]
    prove_s = V["q"] * blk["ands"] / V["P3"]
    gb = cells * 2 * words_per_cell / g / 1e9
    exposed_free_share_ms = max(0.0, gb - V["free_pouw_GB_at_80"]) / V["bar_GBps_16k"] * 1e3
    h = V["h_beside_16k"] if gb <= V["beside_cap_GB_16k"] else V["h_grid_16k"]
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
    defaults = {"a_row": 4096, "b_row": 2048, "w_row": 1024}
    for g in (1, 8, 16, 32, 64, 128, 256, 512):
        for per_mac_key in ("ands_per_mac", "ands_per_mac_cal"):
            b = block(g, L, K, defaults["a_row"], defaults["b_row"], defaults["w_row"], V[per_mac_key])
            for steps in (1, 100, 1000, 10000):
                res["rows"].append({"per_mac": per_mac_key, "steps_a_window": steps, **b,
                                    **window(g, L, K, steps, b, 1, cells)})
    for per_mac_key in ("ands_per_mac", "ands_per_mac_cal"):
        b1 = block(1, L, K, **defaults, per_mac=V[per_mac_key])
        w1 = window(1, L, K, 1, b1, 1, cells)
        for g in (128, 256):
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
            b1 = block(1, L, K, a_row, b_row, 1024, V["ands_per_mac"])
            bg = block(128, L, K, a_row, b_row, 1024, V["ands_per_mac"])
            w1, wg = window(1, L, K, 1, b1, 1, cells), window(128, L, K, 1, bg, 1, cells)
            d_serve = w1["J_serve_hidden_f"] - wg["J_serve_hidden_f"]
            res["row_sensitivity"][key] = {
                "per_cell_ands": b1["ands"], "g128_ands": bg["ands"], "g128_tile": bg["tile"], "g128_fits": bg["fits_2^27"],
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
                if not b["fits_2^26"]:
                    continue
                w = window(g, L, K, steps, b, 1, cells)
                J = w[f_key] + w["J_prove"]
                if best is None or J < best[1]:
                    best = (g, J, w[f_key], w["J_prove"])
            res["g_opt"][f"{f_key}_steps{steps}"] = {"g": best[0], "J": best[1], "J_serve": best[2], "J_prove": best[3]}
            print("g_opt", f_key, steps, res["g_opt"][f"{f_key}_steps{steps}"])
    res["g_min_bytes"] = {S: math.ceil(cells * 2 * (S - 1) / (V["free_pouw_GB_at_80"] * 1e9)) for S in (2, 4, 8)}
    res["g_max_cap"] = {f"L{L_}": {k: int(V["cap"] // (L_ * V[k])) for k in ("ands_per_mac", "ands_per_mac_cal")}
                        for L_ in (256, 512, 1024, 2048)}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["rows"]:
        if r["steps_a_window"] in (1, 10000):
            print(f"g {r['g']:4d} {r['per_mac']:16s} tile {r['tile']} ands {r['ands'] / 1e6:7.2f}M fits27 {r['fits_2^27']!s:5s} "
                  f"fits26 {r['fits_2^26']!s:5s} "
                  f"GB {r['pouw_GB_a_step']:.3f} exp {r['hash_exposed_ms']:.2f}ms f {r['f_epilogue_ms_if_unhidden']:.2f}ms "
                  f"prove {r['prove_GPU_s_a_window']:.1f}s steps {r['steps_a_window']} J_prove {r['J_prove']:.3g}")
    print(json.dumps({k: res[k] for k in ("breakeven", "g_min_bytes", "g_max_cap")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/cost.json")
