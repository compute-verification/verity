"""Cross-step summary of b2_<TAG>.json: per target step and format, theta's fresh bits per value, the best predictor whose
FLOPs are at most 1/10 of the training to that step (and at most 1/80), the share of bits it removes, and the conditionals."""
import json, sys

FMTS = ("bf16", "fp8t", "nvfp4")
NAMES = {"bf16": "bf16", "fp8t": "FP8", "nvfp4": "NVFP4"}


def best(T, side, fmt, min_budget):
    out = None
    for name, p in T["preds"].items():
        if "conditional" in p or name.startswith(("prevlayer", "inputs")) or side not in p:
            continue
        if p["flops"] and T["F_target"] / p["flops"] < min_budget * 0.79:
            continue
        v = p[side][fmt]["best"]
        if out is None or v < out[1]:
            out = (name, v, p["flops"], p[side][fmt].get("rel_err"))
    return out


def main(path):
    r = json.load(open(path))
    print(f"## {r['meta']['model']}\n")
    print("| step | side | format | theta | best at >=10x (FLOPs) | saved | best at >=80x | saved | prev layer (layers 1+) "
          "| fresh inputs, theta weights | fresh inputs, best LS weights |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, T in r["targets"].items():
        P = T["preds"]
        for side in ("w", "a"):
            for fmt in FMTS:
                th = P["theta"][side][fmt]["best"]
                row = [s, "weights" if side == "w" else "acts", NAMES[fmt], f"{th:.3f}"]
                for mb in (10, 80):
                    b = best(T, side, fmt, mb)
                    if b is None or b[0] == "theta":
                        row += ["-", "-"]
                    else:
                        row += [f"{b[1]:.3f} {b[0]} ({b[2]:.1e})", f"{100 * (1 - b[1] / th):.0f}%"]
                pl = P.get("prevlayer | theta")
                row.append(f"{pl[side][fmt]['best']:.3f} vs {pl[side + '_same_layers'][fmt]['best']:.3f}" if pl else "-")
                for key in ("inputs | theta", "inputs | pertensor LS(snaps+proxies)"):
                    q = P.get(key)
                    row.append(f"{q['a'][fmt]['best']:.3f}" if (q and side == "a") else "-")
                print("| " + " | ".join(row) + " |")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
