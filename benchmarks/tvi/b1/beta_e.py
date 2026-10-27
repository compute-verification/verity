"""beta_e, the token-entry rate, from block.py's exact blocks: `python beta_e.py OUT_DIR > beta_e.md` reads every
OUT_DIR/step*/block_*.json and writes OUT_DIR/beta_e.json.

Per exact block (boundary l, L_b held layers, the entering residual's format, the residual stream's precision): the bits
it takes per distinct entry, beta_e = (l_tok + l_bnd + l_fix) / entries, split into the token id (log2 V), theta''s entering
residual coded given theta's (F), and the repairs that make every counted activation theta''s (fix); beta_e / d is the
per-dimension rate. The best coder at a checkpoint, format and residual precision is its least beta_e over blocks.
"""
import glob, json, os, re, sys


def rows(path):
    r = json.load(open(path))
    step = int(re.search(r"step(\d+)", path).group(1))
    for b, B in r["boundaries"].items():
        for k in B["blocks"]:
            n = k["unique_rows"]
            yield {"step": step, "fmt": r["fmt"], "resid": r.get("resid", "fp32"), "boundary": int(b), "L_b": k["L_b"],
                   "entry_fmt": k["bfmt"], "fill": k["fill"], "tokens": k["tokens"], "entries": n, "beta_e": k["beta_e"],
                   "beta_x": k["beta_e"] / r["d"], "h_tok": k["l_tok"] / n, "F": k["l_bnd"] / n, "fix": k["l_fix"] / n,
                   "mismatch_before_repair": k["mismatch_naive_frac"], "mismatch_after_repair": k["mismatch_after_repair"],
                   "bnd_decoded_exact": k["bnd_decoded_exact"], "l_w": k["l_w"], "n_I": k["n_I"],
                   "beta_x_bound_exact": k["beta_x_bound_exact"], "d": r["d"], "vocab": r["vocab"],
                   "code_sha256": r["code_sha256"]}


def main():
    out = sys.argv[1]
    got = [x for p in sorted(glob.glob(os.path.join(out, "step*", "block_*.json"))) for x in rows(p)]
    best = {}
    for x in got:
        key = f"{x['step']}:{x['fmt']}:{x['resid']}"
        if key not in best or x["beta_e"] < best[key]["beta_e"]:
            best[key] = x
    json.dump({"blocks": got, "best": best}, open(os.path.join(out, "beta_e.json"), "w"), indent=1)
    print("| step | format | residual | boundary | L_b | entry at | entries | beta_e (bit/entry) | beta_e/d | h_tok | F | fix | "
          "mismatched before repair | after | per counted activation, exact (with l_w) |")
    print("|" + "---|" * 15)
    for x in sorted(got, key=lambda x: (x["step"], x["fmt"], x["resid"], x["boundary"], x["L_b"], x["entry_fmt"])):
        print(f"| {x['step']} | {x['fmt']} | {x['resid']} | {x['boundary']} | {x['L_b']} | {x['entry_fmt']} | "
              f"{x['entries']} | {x['beta_e']:.1f} | {x['beta_x']:.3f} | {x['h_tok']:.2f} | {x['F']:.1f} | {x['fix']:.1f} | "
              f"{x['mismatch_before_repair']:.4f} | {x['mismatch_after_repair']} | {x['beta_x_bound_exact']:.3f} |")
    print()
    for key, x in sorted(best.items(), key=lambda kv: (kv[1]["step"], kv[1]["fmt"], kv[1]["resid"])):
        print(f"best at step {x['step']} {x['fmt']} (residual {x['resid']}): beta_e {x['beta_e']:.1f} bits an entry, "
              f"beta_x {x['beta_x']:.3f} at d = {x['d']} (boundary {x['boundary']}, L_b {x['L_b']}, "
              f"entry at {x['entry_fmt']})")


if __name__ == "__main__":
    main()
