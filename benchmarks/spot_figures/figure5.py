"""Figure 5 of the SPOT post, where 10% comes from: the total cost, serve tax plus β, at which the bound reaches the
FP4 weights over the headline scope, from the path's first step through each lever in turn, beside the bound at a
total cost of `GOAL`.

It reads break-even surfaces (`breakeven_surface.json`: each case's serve tax, its bound at the reported βs and its β
at each weight size, per scope) and a path, the steps in order, each adding one lever to the steps before it, with
its status (measured, modeled or assumed), its basis (`cards.BASES`: the served cut measured, the served cut at card
prices, or a design not served) and the surface case that prices it in each series. A step whose case no
surface has is pending: it has no number yet, and the steps after it leave it out. A lever off the path (`off`, with
its owner, or none) is drawn in grey beside the step it would follow. A term no step's total includes (`unpriced`) is
drawn as a named line at its share of serving compute (`total`), or as an arrow while it has none; one whose status is
`charged` is stated in the caption as a condition every total holds under. The bound at `GOAL` is interpolated on
log-log axes between the surface's two nearest βs, which it fits within a part in 10^3 where the count binds. The
caption calls the first step today's measured system only when its basis is `served-measured`, names where the path
turns to a design not served, and states the model's `headline` card's rows, each with its basis.

    uv run python benchmarks/spot_figures/figure5.py SURFACE.json... [--path PATH.json] [--headline CARD.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from math import ceil, exp, floor, log, log10
from pathlib import Path

import cards as spot_cards
import figure1 as f1

SCHEMA = "spot-figure5/v1"
#: the total cost, serve tax plus β, a share of serving compute, the post's claim is made at
GOAL = 0.10
#: the weight size the bound must reach, by the surface's `weights_bits` key, and its label
TARGET, TARGET_LABEL = "FP4", "GLM-5.3's weights at FP4"
STATUSES = ("measured", "modeled", "assumed")
#: the default path: each step's label, status, basis, what the status rests on, and its case in each series; what
#: makes a step a design (`design`)
PATH = {"design": "every step prices the per-expert MoE cut (Figure 1's (A′)), which no serving path commits yet; the "
                  "served cut is the headline card's",
        "steps": [
    {"label": "(A′) per-expert cut at 128k, as Figure 1 prices its proving", "status": "measured", "basis": "design",
     "cases": {"two-stage": "R0"},
     "note": "the cards' measured proving, 600 s replay blocks, at 128k over vLLM's 128k serve plan (the indexer "
             "modeled); the verifier's CPU joins at its own step"},
    {"label": "the fused per-expert cut", "status": "measured", "basis": "design",
     "cases": {"two-stage": "R1", "one-stage": "R1-1s"},
     "note": "each expert's table read fused into its GEMM at k_log 25, priced on one H200 with the cache (512 and "
             "1024 blocks) on synthetic statements; its fixed cost before the cache is modeled"},
    {"label": "the statement cache", "status": "measured", "basis": "design", "cases": {"two-stage": "R2"},
     "note": "fixed seconds a statement 8-10x lower on the H200 (experts measured; 5 types priced from H200 splits)"},
    {"label": "4 provers on one H200, verify-ahead", "status": "measured", "basis": "design",
     "cases": {"two-stage": "R3"},
     "note": "on the expert types: the fused blocks measured at 128 and 256 blocks a statement, a quarter-statement "
             "each"},
    {"label": "a 3.5x GPU prover", "status": "modeled", "basis": "design", "cases": {"two-stage": "R4"},
     "note": "device parts only; the host witness stays"},
    {"label": "lifetime replay", "status": "modeled", "basis": "design", "cases": {"two-stage": "R5"},
     "note": "458 GPU-s a sampled request at 128k, no edge pause, pages committed when written"},
    {"label": "a 4x GPU prover", "status": "modeled", "basis": "design", "cases": {"two-stage": "R5x4"},
     "note": "device parts only"},
    {"label": "the Lean verifier's CPU", "status": "measured", "basis": "design",
     "cases": {"two-stage": "R6x4", "one-stage": "R6x4-1s"},
     "note": "each statement's Lean verify (measured on 9 types; families scaled; the median 681 s elsewhere) at 81.8 "
             "core-seconds a GPU-second, Nebius's on-demand list prices; total compute = serve tax + proving + "
             "verifying"},
    {"label": "4 provers on every type, four provers per H200", "status": "assumed", "basis": "design",
     "cases": {"two-stage": "R6x4+all+4perH200"},
     "note": "the experts' overlap factor taken for every type, and full statements under four provers once each "
             "prover's peak is about 33 GiB (its H200 run is pending)"},
    {"label": "the dense types' host witness 3.5x lower", "status": "assumed", "basis": "design",
     "cases": {"two-stage": "R6x4+all+4perH200+host"}, "note": "CPU's cache cut the witness 2-4x; the H200 run is pending"},
], "off": [
    {"label": "hashing cut 4x", "status": "assumed", "basis": "design", "owner": None,
     "after": "the dense types' host witness 3.5x lower", "cases": {"two-stage": "R6x4+all+4perH200+host+hash4"},
     "note": "in-proof hashing at a half share of the marginal, cut 4x"},
], "unpriced": [
    {"label": "compile-slice proving: under 1% with prefix caching off or hits charged (modeled)", "status": "modeled",
     "note": "the network audit proves every window's compile slice, never sampled, so its share adds to every step's "
             "total"},
    {"label": "the advice charge", "status": "charged",
     "note": "until StopRule_v1 checks length and finish in-circuit, the advice line charges log2(max_tokens)+1 bits a "
             "request, which no β covers at either scope; every break-even here holds once StopRule_v1 lands"},
]}
COLORS = ("#08306B", "#D95F02", "#1B9E77", "#7570B3")


def rows(surfaces: list[dict]) -> dict[tuple[str, str], dict]:
    """Every surface's rows by (case, scope), a later surface's row in place of an earlier one's."""
    return {(r["case"], r["scope"]): r for s in surfaces for r in s["rows"]}


def bound_at(row: dict, beta: float) -> float:
    """The row's bound at `beta`, on the log-log line through its two reported βs nearest it."""
    pts = sorted((float(b), v) for b, v in row["bound"].items())
    (b0, v0), (b1, v1) = min(zip(pts, pts[1:]), key=lambda p: abs(log(beta) - (log(p[0][0]) + log(p[1][0])) / 2))
    slope = (log(v1) - log(v0)) / (log(b1) - log(b0))
    return exp(log(v0) + slope * (log(beta) - log(b0)))


def point(row: dict, target_bits: float) -> dict:
    beta = row["beta"][TARGET]
    at = GOAL - row["tax"]
    bound = bound_at(row, at) if at > 0 else None
    return {"case": row["case"], "tax": row["tax"], "h": row.get("h"), "beta": beta,
            "total": None if beta is None else row["tax"] + beta,
            "bound_at_goal": bound, "times_target": None if bound is None else bound / target_bits}


def run(surfaces: list[dict], path: dict = PATH, headline: dict | None = None) -> dict:
    """The figure's JSON; `headline`, the model's `headline` card, adds its rows by what each rests on."""
    by = rows(surfaces)
    weights = {k: v for s in surfaces for k, v in s["meta"]["weights_bits"].items()}
    target_bits = float(weights[TARGET])
    scopes = (f1.PRIMARY, f1.SECONDARY)
    series = list(dict.fromkeys(n for st in path["steps"] for n in st["cases"]))
    labels = [st["label"] for st in path["steps"]]

    def priced(st: dict) -> dict:
        if st["status"] not in STATUSES:
            raise ValueError(f"step {st['label']!r}: status {st['status']!r} is not one of {STATUSES}")
        if st.get("basis") not in spot_cards.BASES:
            raise ValueError(f"step {st['label']!r}: basis {st.get('basis')!r} is not one of {spot_cards.BASES}; "
                             "say whether it prices the served cut, measured or at card prices, or a design")
        at = {n: {sc: point(by[(c, sc)], target_bits) if (c, sc) in by else None for sc in scopes}
              for n, c in st["cases"].items()}
        return st | {"at": at, "pending": all(v[f1.PRIMARY] is None for v in at.values())}

    off = []
    for st in path.get("off", []):
        if st["after"] not in labels:
            raise ValueError(f"off-path step {st['label']!r}: no step {st['after']!r} on the path to branch from")
        off.append(priced(st) | {"after_index": labels.index(st["after"])})
    steps = [priced(st) for st in path["steps"]]
    if any(st["basis"] == "design" for st in steps) and not path.get("design"):
        raise ValueError("a step's basis is design, but the path's `design` doesn't say what makes it one")
    rows_ = [{k: r[k] for k in ("label", "pricing", "basis", "cells")}
             for r in spot_cards.check(headline)["value"]["rows"]] if headline else []
    return {"schema": SCHEMA, "goal": GOAL, "target": {"label": TARGET_LABEL, "key": TARGET, "bits": target_bits},
            "primary": f1.PRIMARY, "secondary": f1.SECONDARY, "series": series, "steps": steps, "off": off,
            "unpriced": list(path.get("unpriced", [])), "design": path.get("design", ""), "headline": rows_,
            "headline_card": (headline or {}).get("art")}


def _pct(x: float) -> str:
    return f"{x:,.0%}" if x >= 1 else f"{x:.1%}" if x >= 0.01 else f"{x:.2%}"


def _bytes(bits: float) -> str:
    for unit, name in ((1e24, "YB"), (1e21, "ZB"), (1e18, "EB"), (1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if bits / 8 >= unit:
            return f"{bits / 8 / unit:.3g} {name}"
    return f"{bits / 8e3:.3g} kB"


#: how the caption names each basis
BASIS_TEXT = {"served-measured": "served and measured", "served-card-priced": "the served cut at card prices",
              "design": "a design not served"}


def headline_text(rows: list[dict], scope: str) -> str:
    """The headline card's rows over `scope`, each by what it rests on, at each S and verifier step it has."""
    if not rows:
        return ""
    cells = "; ".join(f"{r['label']}, {r['pricing']} ({BASIS_TEXT[r['basis']]}): " + ", ".join(
        f"{s} " + " and ".join(f"{_pct(by_scope[scope])} {m}" for m, by_scope in by_m.items() if scope in by_scope)
        for s, by_m in r["cells"].items()) for r in rows)
    served = [r for r in rows if r["basis"] == "served-measured"]
    return (f" The headline rows over a {scope}, each by what it rests on: {cells}. "
            + ("No row is served and measured." if not served else
               f"Served and measured: {', '.join(r['label'] for r in served)}."))


def caption(out: dict) -> str:
    p, s0 = out["primary"], out["series"][0]
    done = [st for st in out["steps"] if st["at"][s0][p]]
    first, last = done[0]["at"][s0][p], done[-1]["at"][s0][p]
    goal = out["goal"]
    start = ("Today's measured system" if done[0]["basis"] == "served-measured" else
             f"{done[0]['label'][0].upper()}{done[0]['label'][1:]}, {BASIS_TEXT[done[0]['basis']]},")
    lead = (f"Each step adds one lever to the steps before it, and every number is the planner's design over a {p}: the "
            f"serve tax plus the β at which the bound reaches {out['target']['label']} "
            f"({out['target']['bits'] / 8e9:.1f} GB), against the goal of {goal:.0%} (top), and the bound at a total "
            f"cost of {goal:.0%} (bottom, interpolated between the surface's β). {start} breaks even at "
            f"{_pct(first['total'])}, {first['total'] / goal:,.0f}x the goal, and at {goal:.0%} bounds a {p} at "
            f"{_bytes(first['bound_at_goal'])}, {first['times_target']:,.0f}x the weights. With the steps priced so far "
            f"it breaks even at {_pct(last['total'])}"
            + (f" and bounds {_bytes(last['bound_at_goal'])} at {goal:.0%}" if last["bound_at_goal"] else "") + ". ")
    design = next((st for st in out["steps"] if st["basis"] == "design"), None)
    if design:
        lead += (f"From {design['label']} on, every total is {BASIS_TEXT['design']}: {out['design']}. ")
    marks = "A filled marker is measured, a half-filled one modeled and a hollow one assumed. "
    later = [any(not s["pending"] for s in out["steps"][i + 1:]) for i in range(len(out["steps"]))]
    notes = " ".join(f"{st['label'][0].upper()}{st['label'][1:]} ({st['status']}"
                     + (", pending: no number yet" + (", and the steps after it leave it out" if after else "")
                        if st["pending"] else "")
                     + f"): {st['note']}." for st, after in zip(out["steps"], later))
    second = (f" The dashed line is the same path over a {out['secondary']}." if any(
        st["at"][s0][out["secondary"]] for st in done) else "")
    for name in out["series"][1:]:
        at = [st["at"][name][p] for st in out["steps"] if st["at"].get(name, {}).get(p)]
        if at and all(a["total"] is None for a in at):
            second += (f" The {name} design never breaks even: its serve tax alone is {_pct(min(a['tax'] for a in at))}"
                       f" of serving compute, and no β brings its bound to {out['target']['label']}.")
    off = "".join(f" Off the path, grey: {st['label']} ({st['owner'] or 'unowned'}, {st['status']}): {st['note']}"
                  + (f"; after {st['after']} it would break even at {_pct(st['at'][s0][p]['total'])}"
                     if st["at"].get(s0, {}).get(p) else "") + "." for st in out["off"])
    unpriced = "".join(f" {a['label'][0].upper()}{a['label'][1:]} is charged, in no total: {a['note']}."
                       if a["status"] == "charged" else
                       f" Not in any step, {a['label']}: {a['note']}"
                       + ("" if a.get("total") is None else f", {_pct(a['total'])} of serving compute") + "."
                       for a in out["unpriced"])
    return lead + marks + notes + second + off + unpriced + headline_text(out.get("headline", []), p)


def render(out: dict, outdir: Path) -> list[Path]:
    """Draw the figure into `outdir` as figure5.png and .svg; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render import _save

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 18, "axes.labelsize": 19, "xtick.labelsize": 14,
                         "ytick.labelsize": 16, "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(17, 15), sharex=True, height_ratios=(3, 2))
    fig.subplots_adjust(left=.11, right=.97, bottom=.33, top=.92, hspace=.08)
    xs = range(len(out["steps"]))
    p, s2 = out["primary"], out["secondary"]
    fill = {"measured": "full", "modeled": "bottom", "assumed": "none"}
    for k, name in enumerate(out["series"]):
        color = COLORS[k % len(COLORS)]
        for ax, key, fmt in ((top, "total", _pct), (bottom, "bound_at_goal", _bytes)):
            for scope, ls, lw in ((s2, (0, (5, 2)), 1.4), (p, "-", 2.6)):
                pts = [(x, st["at"][name][scope][key]) for x, st in zip(xs, out["steps"])
                       if st["at"].get(name, {}).get(scope) and st["at"][name][scope][key]]
                for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts, pts[1:])):
                    ax.plot([x0, x1], [y0, y1], color=color, ls=ls if x1 == x0 + 1 else (0, (1, 2)), lw=lw, zorder=2,
                            label=f"the {name} design, over a {scope}" if ax is top and i == 0 else None)
                if scope != p:
                    continue
                for x, y in pts:
                    st = out["steps"][x]
                    ax.plot([x], [y], marker="o", ms=13, mew=2.2, color=color, mfc=color if fill[st["status"]] != "none"
                            else "white", fillstyle=fill[st["status"]], markerfacecoloralt="white", zorder=4)
                    ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=13,
                                color=color)
    for x, st in zip(xs, out["steps"]):
        if st["pending"]:
            for ax in (top, bottom):
                ax.axvline(x, color="#BBBBBB", lw=1.0, ls=(0, (2, 3)), zorder=0)
                ax.text(x, .5, "pending", transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="center",
                        fontsize=13, color="#888888", style="italic",
                        bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1.5})
    s0, grey = out["series"][0], "#8C8C8C"
    for st in out["off"]:
        here, base = st["at"].get(s0, {}).get(p), out["steps"][st["after_index"]]["at"].get(s0, {}).get(p)
        if not here:
            continue
        x = st["after_index"] + .42
        for ax, key, fmt in ((top, "total", _pct), (bottom, "bound_at_goal", _bytes)):
            if base and base[key] and here[key]:
                ax.plot([st["after_index"], x], [base[key], here[key]], color=grey, ls=(0, (1, 2)), lw=1.4, zorder=2)
            if here[key]:
                ax.plot([x], [here[key]], marker="o", ms=11, mew=1.8, color=grey, mfc="white", zorder=4)
                ax.annotate(f"{st['label']}: {fmt(here[key])}\n({st['owner'] or 'unowned'}, {st['status']})",
                            (x, here[key]), textcoords="offset points", xytext=(0, -14), ha="center", va="top",
                            fontsize=11, color="#6E6E6E")
    priced = [(x, st["at"][s0][p]["total"]) for x, st in zip(xs, out["steps"]) if st["at"].get(s0, {}).get(p)]
    for a in (a for a in out["unpriced"] if a["status"] != "charged"):
        if a.get("total") is not None:
            top.axhline(a["total"], color=grey, ls=(0, (6, 2, 1, 2)), lw=1.4, zorder=1)
            top.text(len(out["steps"]) - .6, a["total"] * 1.1, a["label"], ha="right", va="bottom", fontsize=13,
                     color="#6E6E6E")
        elif priced:
            x, y = priced[-1][0] - .3, priced[-1][1]
            top.annotate("", (x, y * 3), (x, y * 1.15), arrowprops={"arrowstyle": "-|>", "color": grey, "lw": 1.6,
                                                                    "ls": (0, (6, 2, 1, 2))})
            top.text(x, y * 3.3, textwrap.fill(f"+ {a['label']}", 26), ha="center", va="bottom", fontsize=12,
                     color="#6E6E6E")
    top.axhline(out["goal"], color="#E69F00", lw=1.6, zorder=1)
    top.text(-.4, out["goal"] / 1.12, f"the goal: {out['goal']:.0%} of serving compute", ha="left", va="top",
             fontsize=14, color="#B07800")
    bottom.axhline(out["target"]["bits"], color="#909090", lw=1.4, zorder=1)
    bottom.text(-.4, out["target"]["bits"] / 1.15, f"{out['target']['label']} ({_bytes(out['target']['bits'])})",
                ha="left", va="top", fontsize=14, color="#555555")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    for ax in (top, bottom):
        ax.set_yscale("log")
        ax.grid(which="major", axis="y", color="#E2E2E2", lw=.45)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / (8 if ax is bottom and out["off"] else 3), hi * 3)
    top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _pct(y)))
    lo, hi = bottom.get_ylim()
    bottom.yaxis.set_major_locator(FixedLocator([8 * 10.0 ** e for e in range(floor(log10(lo / 8)),
                                                                              ceil(log10(hi / 8)) + 1)]))
    bottom.yaxis.set_minor_locator(FixedLocator([]))
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _bytes(y)))
    top.set_ylabel("Break-even total cost:\nserve tax + β at the FP4 weights")
    bottom.set_ylabel(f"Exfiltration bound at a\ntotal cost of {out['goal']:.0%}")
    bottom.set_xticks(list(xs), [textwrap.fill(f"+ {st['label']}" if i else st["label"][0].upper() + st["label"][1:], 14) for i, st in
                                 enumerate(out["steps"])], fontsize=12)
    last = any(st["after_index"] == len(out["steps"]) - 1 for st in out["off"])
    bottom.set_xlim(-.5, len(out["steps"]) - .5 + (.45 if last else 0))
    top.legend(loc="upper right", frameon=False, fontsize=14)
    fig.text(.11, .945, f"Where {out['goal']:.0%} comes from: the levers that bring GLM-5.3's bound over a {p} under "
             "its FP4 weights", ha="left", va="bottom", fontsize=20)
    fig.canvas.draw()
    ticks = min(t.get_window_extent().y0 for t in bottom.get_xticklabels())
    below = bottom.transAxes.inverted().transform((0, ticks))[1] - .05
    bottom.text(0, min(-.42, below), textwrap.fill(caption(out), 160), transform=bottom.transAxes, ha="left",
                va="top", fontsize=13, color="#444444")
    return _save(fig, outdir, "figure5")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("surfaces", type=Path, nargs="+", help="break-even surfaces, a later one's rows replacing an earlier's")
    ap.add_argument("--path", type=Path, help="the path's steps as JSON (default: the module's PATH)")
    ap.add_argument("--headline", type=Path, help="a headline card (default: the model's in cards/, if any)")
    ap.add_argument("--out", type=Path, help="default: the first surface's directory")
    args = ap.parse_args(argv)
    headline = (json.loads(args.headline.read_text()) if args.headline
                else spot_cards.load("headline", "GLM-5.3", required=False))
    out = run([json.loads(s.read_text()) for s in args.surfaces],
              json.loads(args.path.read_text()) if args.path else PATH, headline)
    outdir = args.out or args.surfaces[0].parent
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figure5.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(*render(out, outdir), sep="\n")


if __name__ == "__main__":
    main()
