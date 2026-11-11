"""Figure 4 of the SPOT post, 'Randomness is powerful': the sampling rate that keeps the served model's FP4 weights from
leaving, against a proof block's output bits, beside the rate a total cost of `GOAL` buys.

An incorrect proof block exports at most its b output bits, so T bits leave only through T/b incorrect blocks, each
escaping a sample at rate q with probability 1 - q: at most δ once q >= b ln(1/δ)/T, the simple form, whatever the
scope.  The planner's exact form (`optimize.leak_count`, `leak_draws`) also charges which of the scope's n blocks are the
incorrect ones, locBits(n, K), so its rate ln(1/δ)/(K + 1) rises with the scope; n is the scope's tokens times a token's
proof-block output bits (the `partition` card's) over b.  A line's bought rate is β/h at its marginal proof overhead h,
β being what `GOAL` leaves after its serve tax: proving alone, so at most what that total cost buys.

    uv run python benchmarks/spot_figures/figure4.py DIR/figure1.json [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from math import log, log2, log10
from pathlib import Path

import figure1 as f1
from verity.protocols.spot.optimize import Grain, leak_count
from verity.circuits.partition_v2 import UNIT_CLASSES

SCHEMA = "spot-figure4/v1"
#: the evasion threshold the post's sampling argument is stated at
DELTA = 2.0 ** -40
#: the total cost, serve tax plus β, a share of serving compute, the post's claim is made at
GOAL = 0.10
#: the lines whose bought rate the figure plots: Figure 1's configurations, by name
LINES = ("tokens", "per_expert")
#: the marginal proof overhead the proving work targets for every block, plotted at the first line's serve tax
TARGET_H = 1e7
#: the reference line the rate keeps in, by label prefix
TARGET = "{model}'s weights at FP4"
#: a proof block's output bits on the x-axis, 16 bits to past the target
BITS = tuple(2.0 ** k for k in range(4, 43))


def exact(bits: float, tokens: float, bits_per_token: float, target: float) -> dict:
    """The planner's one-stage design at `bits` a block over a scope's `tokens`: its blocks n, count K and rate."""
    n = max(1, round(tokens * bits_per_token / bits))
    k = leak_count(Grain("b", n, 1.0, bits, 0.0, 0.0), target)
    return {"blocks": n, "count": k, "rate": 0.0 if k >= n else min(1.0, log(1 / DELTA) / (k + 1))}


def simple(bits: float, target: float) -> float:
    return min(1.0, bits * log(1 / DELTA) / target)


def largest(rate: float, tokens: float, bits_per_token: float, target: float) -> float | None:
    """The widest block whose exact rate over the scope is at most `rate`, to a part in 2^20 of its log2; None when not
    even a 1-bit block's is."""
    lo, hi = 0.0, log2(target)
    if exact(2.0 ** lo, tokens, bits_per_token, target)["rate"] > rate:
        return None
    while hi - lo > 2.0 ** -20 * hi:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if exact(2.0 ** mid, tokens, bits_per_token, target)["rate"] <= rate else (lo, mid)
    return 2.0 ** lo


def run(model: str, one: dict) -> dict:
    """The figure's JSON from `model`'s cards and Figure 1's (`total_cost.run`'s) configurations and serve taxes, each
    scope's tokens at the reading of serving's rate `one` was planned at."""
    c = f1.cards(model, serving=one.get("serving", {}).get("reading", f1.SERVING[0]))
    sc = f1.slot_cards(model)
    part = f1.TokenBlocks.of(c["partition"].value["strata"], c["partition"].value.get("served_macs_per_token"))
    per_token = part.blocks * part.bits
    prefix = TARGET.format(model=model)
    target = next(line for card in sc.get("references", []) for line in card["value"]["lines"]
                  if line["label"].startswith(prefix))
    t = target["value"]
    tokens = {s.name: s.requests * c["tokens_per_request"].value for s in f1.scopes(c, part)}
    scopes = (f1.PRIMARY, f1.SECONDARY)
    marks = [{"label": f"{UNIT_CLASSES[0]:,} bits: the smallest size class", "bits": float(UNIT_CLASSES[0])},
             {"label": f"{part.bits:.0f} bits: {model}'s mean proof block", "bits": part.bits},
             {"label": f"{one['served_bits_request']:,} bits: a request's tokens",
              "bits": float(one["served_bits_request"])},
             {"label": f"{UNIT_CLASSES[-1]:,} bits: the largest size class", "bits": float(UNIT_CLASSES[-1])}]
    need = {s: {m["label"]: exact(m["bits"], tokens[s], per_token, t)["rate"] for m in marks} for s in scopes}
    lines = [(one["configurations"][n]["label"], one["configurations"][n]["overhead"], one["taxes"][n]["tax"], n)
             for n in LINES]
    lines.append((f"a proof overhead of h = {_g(TARGET_H)}, the target", TARGET_H, one["taxes"][LINES[0]]["tax"], None))
    bought = []
    for label, h, tax, name in lines:
        rate = (GOAL - tax) / h
        bought.append({"config": name, "label": label, "h": h, "tax": tax, "beta": GOAL - tax, "rate": rate,
                       "largest_bits": {s: largest(rate, tokens[s], per_token, t) for s in scopes},
                       "leeway": {s: {k: rate / q for k, q in need[s].items()} for s in scopes},
                       "h_for_smallest": {s: (GOAL - tax) / need[s][marks[0]["label"]] for s in scopes}})
    return ({"serving": one["serving"]} if "serving" in one else {}) | {
            "schema": SCHEMA, "model": model, "delta": DELTA, "goal": GOAL,
            "target": {"label": target["label"], "bits": t}, "bits_per_token": per_token, "tokens": tokens,
            "bits": list(BITS), "simple": [simple(b, t) for b in BITS],
            "exact": {s: [exact(b, tokens[s], per_token, t) for b in BITS] for s in scopes},
            "primary": f1.PRIMARY, "secondary": f1.SECONDARY, "marks": marks, "need": need, "bought": bought}


def _g(v: float) -> str:
    return f"{v:.2g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _times(ratio: float) -> str:
    return f"{ratio:.3g}x the rate" if ratio >= 1 else f"{1 / ratio:.3g}x short of the rate"


def caption(out: dict) -> str:
    first, p, s = out["marks"][0]["label"], out["primary"], out["secondary"]
    q = out["need"][p][first]
    lead = (f"An incorrect proof block exports at most its b output bits, so the {out['target']['bits'] / 8e9:.1f} GB "
            f"leave only through T/b incorrect blocks, each escaping a sample at rate q with probability 1 - q: at most "
            f"δ = 2^-40 once q >= b ln(1/δ)/T (thin, black), whatever the scope. The planner's exact form also charges "
            f"which of the scope's blocks are the incorrect ones, so its rate rises with the scope: over a {p} (thick) "
            f"and a {s} (dashed). At 16-bit blocks over a {p} it is {_g(q)}: proving one block in "
            f"{_g(1 / q)} keeps the weights in, {log10(1 / q):.1f} orders of magnitude below proving every block "
            f"({_g(out['need'][s][first])} over a {s}). ")
    rows = "; ".join(f"{b['label']}" + (f" (h = {_g(b['h'])})" if b["config"] else "") + f" buys q = {_g(b['rate'])} "
                     f"at a total cost of {out['goal']:.0%}, β = {b['beta']:.1%} after a {b['tax']:.1%} serve tax and "
                     f"proving alone: {_times(b['leeway'][p][first])} 16-bit blocks need over a {p}"
                     for b in out["bought"])
    need_h = out["bought"][0]["h_for_smallest"][p]
    return (lead + rows[0].upper() + rows[1:] + f". At that serve tax, h = {_g(need_h)} buys exactly the rate 16-bit "
            f"blocks need over a {p}.")


def render(out: dict, outdir: Path) -> list[Path]:
    """Sample the figure into `outdir` as figure4.png and .svg; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render import _save

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 18, "axes.labelsize": 19, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.subplots_adjust(left=.11, right=.97, bottom=.34, top=.9)
    xs = out["bits"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(xs, out["simple"], color="#222222", lw=1.6, label="q = b ln(1/δ)/T", zorder=4)
    for scope, ls, lw, color in ((out["primary"], "-", 3.4, "#08306B"), (out["secondary"], (0, (5, 2)), 2.0, "#6A51A3")):
        ax.plot(xs, [r["rate"] for r in out["exact"][scope]], color=color, ls=ls, lw=lw, zorder=3,
                label=f"The planner's exact rate, over a {scope}")
    for b, color in zip(out["bought"], ("#D95F02", "#1B9E77", "#888888")):
        ax.axhline(b["rate"], color=color, lw=2.2, ls=(0, (1.5, 1.5)), zorder=2,
                   label=f"{out['goal']:.0%} of serving compute buys, {b['label']}"
                         + (f" (h = {_g(b['h'])})" if b["config"] else ""))
    lo = min([b["rate"] for b in out["bought"]] + out["simple"]) / 10
    ax.set_ylim(lo, 3.0)
    ax.set_xlim(xs[0] / 2, xs[-1])
    for i, m in enumerate(out["marks"]):
        ax.axvline(m["bits"], color="#C8C8C8", lw=1.0, zorder=0)
        ax.text(m["bits"], 1.6e-3 if i % 2 else 2e-5, m["label"], rotation=90, va="center", ha="center", fontsize=12,
                color="#555555", bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1.5}, zorder=5)
    ax.set_xlabel("A proof block's output bits, b")
    ax.set_ylabel("Sampling rate q: the share of proof blocks proved")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(which="major", color="#E2E2E2", lw=.45)
    ax.legend(loc="lower right", frameon=True, framealpha=.92, edgecolor="none", fontsize=14)
    fig.text(.11, .94, f"The sampling rate that keeps {out['target']['label']} in, at δ = 2^-40", ha="left",
             va="bottom", fontsize=20)
    ax.text(0, -.12, textwrap.fill(caption(out), 128), transform=ax.transAxes, ha="left", va="top", fontsize=13,
            color="#444444")
    return _save(fig, outdir, "figure4")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json", type=Path, help="Figure 1's figure1.json, for its configurations and serve taxes")
    ap.add_argument("--out", type=Path, help="default: the JSON's directory")
    args = ap.parse_args(argv)
    one = json.loads(args.json.read_text())
    out = run(one["model"], one)
    outdir = args.out or args.json.parent
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figure4.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(*render(out, outdir), sep="\n")


if __name__ == "__main__":
    main()
