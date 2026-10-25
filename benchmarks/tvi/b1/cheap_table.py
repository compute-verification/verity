"""The cheap-input block's readout: `python cheap_table.py cheap_*.json > cheap.md` (cheap.py's outputs).

Per block (held layers 0..M-1 over the first m rollouts; "step" holds one GRPO step's worth of tokens, "k" the most rollouts
that fit k): the embedding's code (the rows the block's tokens use, or the whole table), the held layers' weights, the
tokens at log2 V bits each, the repairs (one flag bit per input, then the coded repairs), n_I (counted activations, once per
distinct cone), and the bound on beta_x, also with the tokens coded by theta instead.
"""
import json, sys


def main():
    rs = [json.load(open(p)) for p in sys.argv[1:]]
    for r in rs:
        lw = r["layer_weight_bits"]
        print(f"{r['fmt']} (embedding at {r['emb_fmt']}): {r['n_rollouts']} rollouts, {r['tokens']} tokens; embedding "
              f"whole table {r['emb_table_bits'] / 1e6:.3f} Mbit ({r['emb_codes_changed']:.5f} of its codes changed, "
              f"decoded exact {r['emb_table_decoded_exact']}); a_dec {r['a_dec_bits']:,} bits; tokens coded by theta "
              f"{sum(r['nll_token_bits']) / r['tokens']:.2f} bits each")
        print("  l_w per layer (Mbit): " + ", ".join(f"{i}: {w['bits'] / 1e6:.3f}" for i, w in lw.items())
              + f"; all decoded exact {all(w['decoded_exact'] for w in lw.values())}")
        pv = r["probe"]
        vals = sum(v["values"] for v in pv.values())
        print(f"  probe, half-length prefixes of {r['probe_rollouts']} rollouts: {sum(v['mismatched'] for v in pv.values())} "
              f"of {vals:,} counted values differ, repairs {sum(v['repair_bits'] for v in pv.values()):,.0f} bits, after "
              f"repair {sum(v['after_repair'] for v in pv.values())}")
        print()
    print("| format | held layers | fill | Q tokens (rollouts) | embedding rows | l_emb (Mbit) | l_w layers (Mbit) | "
          "l_tok (Mbit) | l_fix (bits) | total (Mbit) | fits k | n_I (M) | mismatched / after repair | beta_x bound | "
          "with theta-coded tokens | token floor | share of counted per token | decode FLOPs / c |")
    print("|" + "---|" * 18)
    for r in rs:
        for b in r["blocks"]:
            print(f"| {r['fmt']} | 0–{b['M'] - 1} | {b['fill']} | {b['tokens']:,} ({b['rollouts']}) | {b['emb_rows']:,} | "
                  f"{b['l_emb'] / 1e6:.3f} ({b['emb_desc']}; rows {b['l_emb_rows'] / 1e6:.3f}, table "
                  f"{b['l_emb_table'] / 1e6:.3f}) | {b['l_w_layers'] / 1e6:.3f} | {b['l_tok'] / 1e6:.3f} | "
                  f"{b['l_fix']:,.0f} | {b['total'] / 1e6:.3f} | {b['fits_k']} | {b['n_I'] / 1e6:,.1f} | "
                  f"{b['mismatch_before_repair']} / {b['mismatch_after_repair']} | {b['beta_x_bound']:.2e} | "
                  f"{b['beta_x_bound_theta_tokens']:.2e} | {b['token_floor']:.2e} | {b['share_of_counted_per_token']:.3f} | "
                  f"{b['decode_flops_over_c']:.1e} |")


if __name__ == "__main__":
    main()
