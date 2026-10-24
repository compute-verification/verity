"""One table per target step: each predictor family's best member (by FP8 weight bits) with its FLOPs and budget, and the
conditionals; bits per value, best of context entropy / lzma / low-rank MDL."""
import json, sys

FMTS = ("bf16", "fp8t", "nvfp4")
FAMILIES = [
    ("theta (registered, sweep baseline)", lambda n: n == "theta", 0),
    ("proxy: batch 1 (8x)", lambda n: n.startswith("proxy b1"), 6),
    ("proxy: batch 1 (80x)", lambda n: n.startswith("proxy b1"), 60),
    ("proxy: 10x fewer steps (10x)", lambda n: n.startswith("proxy sub10"), 8),
    ("proxy: 100x fewer steps (100x)", lambda n: n.startswith("proxy sub10"), 80),
    ("extrapolation (10x)", lambda n: n.startswith("extrap"), 8),
    ("extrapolation (100x)", lambda n: n.startswith("extrap"), 80),
    ("per-tensor LS on snapshots + proxies", lambda n: n.startswith("pertensor"), 6),
    ("learned elementwise map, half/half", lambda n: n.startswith("learned"), 4),
]


def pick(T, match, min_budget):
    best = None
    for n, p in T["preds"].items():
        if not match(n) or "conditional" in p or "w" not in p:
            continue
        bud = T["F_target"] / p["flops"] if p["flops"] else float("inf")
        if min_budget and (bud < min_budget or (min_budget >= 60 and bud > 2000)):
            continue
        if min_budget and min_budget < 60 and bud >= 60 and any(x in n for x in ("b1", "sub10")):
            continue
        if best is None or p["w"]["fp8t"]["best"] < best[1]["w"]["fp8t"]["best"]:
            best = (n, p, bud)
    return best


def fmt_row(label, p, bud, side_keys=("w", "a")):
    cells = []
    for side in side_keys:
        for f in FMTS:
            cells.append(f"{p[side][f]['best']:.3f}" if side in p else "-")
    return cells


def main(path):
    r = json.load(open(path))
    print(f"## {r['meta']['model']}\n")
    for s, T in r["targets"].items():
        th = T["preds"]["theta"]
        print(f"### theta'_{s}: training {T['F_target']:.1e} FLOPs\n")
        print("| predictor (member) | FLOPs | budget | W bf16 | W FP8 | W NVFP4 | A bf16 | A FP8 | A NVFP4 | W FP8 saved | A FP8 saved |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, match, mb in FAMILIES:
            b = pick(T, match, mb)
            if b is None:
                continue
            n, p, bud = b
            if label.startswith("theta"):
                bud_s, n_s = "-", ""
            else:
                bud_s, n_s = f"{bud:.0f}x", f" ({n})"
            sw = 100 * (1 - p["w"]["fp8t"]["best"] / th["w"]["fp8t"]["best"])
            sa = 100 * (1 - p["a"]["fp8t"]["best"] / th["a"]["fp8t"]["best"])
            print(f"| {label}{n_s} | {p['flops']:.1e} | {bud_s} | " + " | ".join(fmt_row(label, p, bud)) +
                  f" | {sw:.0f}% | {sa:.0f}% |")
        pl = T["preds"].get("prevlayer | theta")
        if pl:
            b = pl["w_same_layers"], pl["a_same_layers"]
            cells = [f"{pl[sd][f]['best']:.3f}" for sd in ("w", "a") for f in FMTS]
            sw = 100 * (1 - pl["w"]["fp8t"]["best"] / b[0]["fp8t"]["best"])
            sa = 100 * (1 - pl["a"]["fp8t"]["best"] / b[1]["fp8t"]["best"])
            print(f"| given layer l-1's fresh update (cond., layers 1+) | ~0 | cond. | " + " | ".join(cells) +
                  f" | {sw:.0f}% | {sa:.0f}% |")
        for key, label in (("inputs | theta", "given each map's fresh inputs, theta's weights (cond.)"),
                           ("inputs | pertensor LS(snaps+proxies)", "given fresh inputs, LS-predicted weights (cond.)")):
            q = T["preds"].get(key)
            if q:
                cells = ["-"] * 3 + [f"{q['a'][f]['best']:.3f}" for f in FMTS]
                sa = 100 * (1 - q["a"]["fp8t"]["best"] / th["a"]["fp8t"]["best"])
                print(f"| {label} | ~0 | cond. | " + " | ".join(cells) + f" | - | {sa:.0f}% |")
        print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
