"""Aggregated words at the reframed parameters: epsilon 1e-4 as a fraction and delta 2^-40 a period, n -> infinity windows.
Draws are Lambda/epsilon a period whatever n is, so proving's share of the period (one prover's 1,000-GPU deployment-year)
is about 1e-5 or less and J is serve-time overhead. This step gives, by g: the blocks and the caps they fit, proving's
share of the period, the serve terms (words' hashing, f's epilogue hidden or exposed, mix4 or m32a), the best g under each
cap and each premise, and the scaling to a K = 7,168 linear (the 3-pager's d_model). Every output is an estimate."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agg_cost  # noqa: E402
from agg_cost import V, block, window  # noqa: E402

INPUTS = {
    "eps": (1e-4, "Daniel's decision: epsilon 1e-4 as a fraction of a period's work (inbox 22:35Z)"),
    "delta": (2.0 ** -40, "delta a period (inbox 22:35Z)"),
    "period_gpu_s": (1000 * 365.25 * 86400, "the proving term's period: one prover's 1,000-GPU deployment-year (BRIEF reframe, "
                                            "coordinator's pick 22:35Z)"),
    "planner_s_per_draw": (1.26, "GPU-s a draw today (GD-PLANNER, estimate, BRIEF reframe), scaled here by a block's ANDs "
                                 "over the per-cell block's"),
    "m32a_instr_per_cell": (8, "m32a in an epilogue: 3 x (SHF, LOP3) + 2 IMAD, one cell a 32-bit lane (estimate)"),
    "K_world": (7168, "the 3-pager's d_model, as a linear's K (BRIEF reframe; estimate of the 3-pager's model)"),
    "row_world": (8192, "A and B rows at K 7,168: K int8 rounded up to 8 KiB (estimate)"),
}
R = {k: v for k, (v, _) in INPUTS.items()}
CAPS = (26, 27, 28, 30)


def main(out: str) -> None:
    K, L = 2048, 1024
    cells = V["cells_k2048_T8192"]
    rows = {"a_row": 4096, "b_row": 2048, "w_row": 16384}
    draws = math.log(1 / R["delta"]) / R["eps"]
    g1 = block(1, L, K, **rows, per_mac=V["ands_per_mac"])
    res = {"inputs": {k: {"value": v, "source": s} for k, (v, s) in INPUTS.items()},
           "inherited_inputs": {k: {"value": v, "source": s} for k, (v, s) in agg_cost.INPUTS.items()},
           "draws_a_period": draws, "by_g": {}, "best_g": {}, "world": {}}
    print(f"draws a period {draws:.4g}")
    for k in range(0, 9):
        g = 1 << k
        b = block(g, L, K, **rows, per_mac=V["ands_per_mac"])
        w = window(g, L, K, 1, b, 1, cells)
        f_mix4 = w["f_epilogue_ms_if_unhidden"]
        f_m32a = f_mix4 * R["m32a_instr_per_cell"] / V["f_instr_per_cell"]
        prove_model = draws * b["ands"] / V["P3"]
        prove_planner = draws * R["planner_s_per_draw"] * b["ands"] / g1["ands"]
        res["by_g"][g] = {
            "tile": b["tile"], "ands_mean": b["ands"], "ands_max": b["ands_max"],
            "fits": {c: b["ands_max"] <= 2 ** c for c in CAPS},
            "pouw_GB_a_step": w["pouw_GB_a_step"], "hash_ms": w["hash_exposed_ms"],
            "f_ms_mix4": f_mix4, "f_ms_m32a": f_m32a,
            "J_serve_f_hidden": w["J_serve_hidden_f"],
            "J_serve_mix4_exposed": (w["hash_exposed_ms"] + f_mix4) / V["t_stock_ms"],
            "J_serve_m32a_exposed": (w["hash_exposed_ms"] + f_m32a) / V["t_stock_ms"],
            "prove_GPU_s_a_period_model": prove_model, "prove_GPU_s_a_period_planner": prove_planner,
            "J_prove_model": prove_model / R["period_gpu_s"], "J_prove_planner": prove_planner / R["period_gpu_s"],
            "budget_1pct_draw_GPU_s": 0.01 * R["period_gpu_s"] / draws,
        }
        r = res["by_g"][g]
        print(f"g {g:3d} tile {r['tile']} mean {r['ands_mean'] / 1e6:6.1f}M max {r['ands_max'] / 1e6:6.1f}M "
              f"fits {''.join('1' if r['fits'][c] else '0' for c in CAPS)} GB {r['pouw_GB_a_step']:.3f} "
              f"Jserve {r['J_serve_f_hidden']:.4f}/{r['J_serve_mix4_exposed']:.4f}/{r['J_serve_m32a_exposed']:.4f} "
              f"Jprove {r['J_prove_model']:.2e}/{r['J_prove_planner']:.2e}")
    for premise, allowed in (("allowed", lambda g: g == 1), ("AggWordMixing", lambda g: g == 1 or g >= 8)):
        for cap in CAPS:
            for serve in ("J_serve_f_hidden", "J_serve_mix4_exposed", "J_serve_m32a_exposed"):
                cands = [(r[serve] + r["J_prove_planner"], g) for g, r in res["by_g"].items()
                         if r["fits"][cap] and allowed(g)]
                J, g = min(cands)
                res["best_g"][f"{premise}_2^{cap}_{serve}"] = {"g": g, "J": J}
                print("best", premise, cap, serve, g, f"{J:.4f}")
    S2 = math.ceil(K / L)
    Kw = R["K_world"]
    Sw = math.ceil(Kw / L)
    res["world"]["bytes_a_MAC_ratio_Kworld_over_K2048"] = (2 * (Sw - 1) / Kw) / (2 * (S2 - 1) / K)
    res["world"]["words_a_cell"] = {"K2048": S2 - 1, "K7168": Sw - 1}
    res["world"]["credited_share_K_minus_L_over_K"] = {"K2048": (K - L) / K, "K7168": (Kw - L) / Kw}
    wr = {"a_row": R["row_world"], "b_row": R["row_world"], "w_row": 16384}
    res["world"]["g_max"] = {c: max((1 << k for k in range(0, 10)
                                     if block(1 << k, L, Kw, **wr, per_mac=V["ands_per_mac"])["ands_max"] <= 2 ** c),
                                    default=0) for c in CAPS}
    res["world"]["blocks"] = {1 << k: {"mean": (b := block(1 << k, L, Kw, **wr, per_mac=V["ands_per_mac"]))["ands"],
                                       "max": b["ands_max"], "tile": b["tile"]} for k in range(0, 8)}
    print("world", json.dumps({k: v for k, v in res["world"].items() if k != "blocks"}))
    draws_1pct = math.log(1 / R["delta"]) / 0.01
    res["draws_a_period_eps_S_1pct"] = draws_1pct
    res["fresh_u"] = {}
    for k in range(0, 9):
        g = 1 << k
        b = block(g, K, K, **rows, per_mac=V["ands_per_mac"])
        bits = 10 if g == 1 else 16 / g
        res["fresh_u"][g] = {"tile": b["tile"], "ands": b["ands_max"], "fits": {c: b["ands_max"] <= 2 ** c for c in CAPS},
                             "receipt_bits_a_cell": bits, "receipt_cut": 10 / bits,
                             "J_prove_planner_eps_S_1pct": draws_1pct * R["planner_s_per_draw"] * b["ands_max"] / g1["ands"]
                             / R["period_gpu_s"]}
        r = res["fresh_u"][g]
        print(f"fresh-U g {g:3d} tile {r['tile']} block {r['ands'] / 1e6:6.1f}M "
              f"fits {''.join('1' if r['fits'][c] else '0' for c in CAPS)} bits/cell {bits:.3g} cut {r['receipt_cut']:.3g} "
              f"Jprove(1%) {r['J_prove_planner_eps_S_1pct']:.2e}")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/reframe.json")
