"""Render b2_<TAG>.json as Markdown: per target step, each predictor's residual fresh bits per value (best of context
entropy, lzma on the middle layer, low-rank MDL) for weights and counted activations, at bf16 / FP8 / NVFP4, and its FLOPs."""
import json, sys

FMTS = ("bf16", "fp8t", "nvfp4")


def e(x):
    return "0" if x == 0 else f"{x:.1e}".replace("e+", "e")


def main(path):
    r = json.load(open(path))
    m = r["meta"]
    print(f"## {m['model']}  (one main step = {e(m['F_step'])} FLOPs; activations: {m['eval_tokens']} held-out tokens, "
          f"outputs of {', '.join(m['linear_maps'])} in {m['n_layers']} layers; forward of a predicted model "
          f"= {e(m['F_eval'])} FLOPs)\n")
    for s, T in r["targets"].items():
        th = T["preds"]["theta"]
        print(f"### theta'_{s}  (training to step {s}: {e(T['F_target'])} FLOPs; held-out loss {T['loss_target']:.3f}, "
              f"theta {th['eval_loss']:.3f})\n")
        print("| predictor | FLOPs | budget | W bf16 | W FP8 | W NVFP4 | A bf16 | A FP8 | A NVFP4 | loss |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, p in T["preds"].items():
            if name.startswith("prevlayer"):
                base = p["w_same_layers"], p["a_same_layers"]
                w = " | ".join(f"{p['w'][f]['best']:.3f} ({base[0][f]['best']:.3f})" for f in FMTS)
                a = " | ".join(f"{p['a'][f]['best']:.3f} ({base[1][f]['best']:.3f})" for f in FMTS)
                print(f"| {name} (given layer l-1's fresh update; in parentheses the same predictor without it, "
                      f"layers 1+) | +0 | cond. | {w} | {a} | |")
                continue
            if name.startswith("inputs"):
                a = " | ".join(f"{p['a'][f]['best']:.3f}" for f in FMTS)
                print(f"| {name} (given each map's fresh inputs) | +0 | cond. | - | - | - | {a} | |")
                continue
            fl = p["flops"]
            bud = "-" if fl == 0 else f"{T['F_target'] / fl:.0f}x"
            w = " | ".join(f"{p['w'][f]['best']:.3f}" for f in FMTS)
            a = " | ".join(f"{p['a'][f]['best']:.3f}" for f in FMTS)
            print(f"| {name} | {e(fl)} | {bud} | {w} | {a} | {p['eval_loss']:.3f} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
