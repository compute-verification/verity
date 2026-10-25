"""The fan-out block's readout: `python fan_table.py block_fp8t.json [block_nvfp4.json ...] > fan.md` (block.py's outputs).

Per block: l_w (the held layers' weights, real code), l_in (tokens at log2 V bits, the sent scales, the entering residual),
l_fix (the repairs that make every counted activation equal theta''s), n_I (counted activations, once per distinct cone),
and the bound on beta_x without the repairs (as substituted; the block is then not exact) and with them (exact), with a_dec
in both; "Q unbounded" drops l_w, a_dec and the scales, the limit as Q grows. (17 + F) / d is the substitution's rate at
that boundary, F being the entering residual's real code per token in the block.
"""
import json, sys

D = 1536


def main():
    print("| format | boundary l | held layers | entering residual at | Q fill | Q tokens (rollouts) | l_w (Mbit) | "
          "l_in: tokens + scales + residual (Mbit) | l_fix (Mbit) | n_I (M) | mismatched before repair | after repair | "
          "beta_x bound, substituted (no repair) | beta_x bound, exact | exact, Q unbounded | (17 + F)/d here | "
          "decode FLOPs / c | products per entering coordinate |")
    print("|" + "---|" * 18)
    for path in sys.argv[1:]:
        r = json.load(open(path))
        for b, B in r["boundaries"].items():
            for k in B["blocks"]:
                lin = k["l_tok"] + k["l_scales"] + k["l_bnd"]
                print(f"| {r['fmt']} | {b} | {k['held'][0]}–{k['held'][-1]} | {k['bfmt']} | {k['fill']} | "
                      f"{k['tokens']} ({k['rollouts']}) | {k['l_w'] / 1e6:.2f} | {lin / 1e6:.2f} | {k['l_fix'] / 1e6:.2f} | "
                      f"{k['n_I'] / 1e6:.1f} | {k['mismatch_naive_frac']:.3f} | {k['mismatch_after_repair']} | "
                      f"{k['beta_x_bound_substituted']:.3f} | {k['beta_x_bound_exact']:.3f} | "
                      f"{(lin + k['l_fix']) / k['n_I']:.3f} | {(17 + k['F_per_token']) / D:.2f} | "
                      f"{k['decode_flops_over_c']:.1e} | {k['products_per_entering_coord_with_attn']:,.0f} |")
    print()
    for path in sys.argv[1:]:
        r = json.load(open(path))
        print(f"{r['fmt']}: a_dec {r['a_dec_bits']:,} bits; tokens coded by theta instead of uniformly "
              f"{sum(r['nll_token_bits']) / sum(r['lens']):.2f} bits per token; weights per held layer: "
              + ", ".join(f"layer {i} {w['bits'] / 1e6:.2f} Mbit ({w['beta_w']:.4f} per weight, decoded exact "
                          f"{w['decoded_exact']})" for i, w in r["layer_weight_bits"].items()))
        print("F per token at the boundary (FMT): " + "; ".join(
            f"{k} real {v['real_ctx_per_token']:.0f}, analyzer ctx {v['est_ctx_per_token']:.0f}, analyzer min "
            f"{v['est_min_per_token']:.0f}" for k, v in r["F_compare"].items()))
        for b, B in r["boundaries"].items():
            for k in B["blocks"]:
                if k["bfmt"] == r["fmt"] and k["fill"] == "exact":
                    per = {f"{i}:{n}": round(v["naive"]["mismatch"] / v["naive"]["values"], 3)
                           for i, L in k["layers"].items() for n, v in L.items()}
                    fix = {f"{i}:{n}": round(v["repaired"]["l_fix"] / v["repaired"]["values"], 3)
                           for i, L in k["layers"].items() for n, v in L.items()}
                    print(f"  l={b} L_b={k['L_b']}: mismatched before repair {per}; repair bits per value {fix}")
        print()


if __name__ == "__main__":
    main()
