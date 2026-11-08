"""The bounds against proof overhead as PNG and SVG for one scope, sampled from `figure1_overhead.json` alone, so anything
holding the JSON samples the same figure: three panels (exfiltration, infiltration, skipped work) against proof overhead,
a colour per share of compute
spent on verification, the panel's headline design solid and thick, the uniform two-stage design solid and thin, the
capped two-stage designs dash-dotted (`CAPPED`, `PATTERNS`; uniform thin), and the other comparisons dashed or dotted,
the measured overhead as a vertical line, the references as grey horizontal ones in the style the JSON gives each, and
each panel's caption from the JSON.  The main figure plots the series and references the JSON marks `main`; the
appendix figure (`appendix_stem`) is the exfiltration panel with every design and the references the main figure
leaves out.  A panel's y-axis spans its headline curves and the references up to `OFF_SCALE` above them, so a
comparison curve may leave it at the top; a reference further up is named at the axis's top, and one at 0 only in the
caption.

    uv run python benchmarks/spot_figures/render.py DIR/figure1_overhead.json [--scope "world year"] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from math import floor, log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, NullLocator
from matplotlib.transforms import offset_copy

from figure1 import PRIMARY

COLORS = ("#9ECAE1", "#3182BD", "#08306B")
CAPPED, PATTERNS = (0, (7, 2.5, 1.5, 2.5)), (0, (7, 2.5, 1.5, 2.5, 1.5, 2.5))
COMPARISON = {"one-stage": (0, (6, 3)), "request": (0, (1.2, 2.4)), "two-stage-uniform": "-",
              "two-stage-capped": CAPPED, "two-stage-capped-uniform": CAPPED, "two-stage-patterns": PATTERNS,
              "two-stage-patterns-uniform": PATTERNS}
#: a comparison's line width; the headline's is `HEADLINE_LW`
THIN = {"two-stage-uniform": 1.2, "two-stage-capped": 2.6, "two-stage-capped-uniform": 1.2,
        "two-stage-patterns": 2.6, "two-stage-patterns-uniform": 1.2}
HEADLINE_LW, COMPARISON_LW = 3.2, 1.8
NAMES = {"two-stage": "Two-stage, width-stratified", "two-stage-uniform": "Two-stage, uniform",
         "two-stage-capped": "Capped at served bits, width-stratified",
         "two-stage-capped-uniform": "Capped at served bits, uniform",
         "two-stage-patterns": "Served patterns, width-stratified",
         "two-stage-patterns-uniform": "Served patterns, uniform",
         "one-stage": "One-stage, dot-product blocks", "request": "Request-level",
         "layer_of_token": "Deferred interiors, a layer of a token"}
REFERENCE_STYLE = {"solid": "-", "dashed": (0, (4, 3))}
VERTICAL = "#E69F00"
#: a reference line this far above the headline curves' top is named at the top of the axis instead of stretching it
OFF_SCALE = 1e3
#: reference lines within this relative distance coincide on the plot
COINCIDENT = 0.02
#: the least gap, in the axis's height, between two labels on one side
LABEL_GAP = 0.045
#: how far below its axis, in the axis's height, a caption starts; the appendix's budget legend starts there beside it
CAPTION_DROP = 0.17
PANELS = (("exfiltration", "Exfiltration capacity"), ("infiltration", "Infiltration"),
          ("skipped_work", "Skipped work (H100-hours)"))


def stem(scope: str) -> str:
    return "figure1_overhead" if scope == PRIMARY else "figure1_overhead_" + scope.replace(" ", "_")


def appendix_stem(scope: str) -> str:
    return "figure1_overhead_appendix" if scope == PRIMARY else "figure1_overhead_appendix_" + scope.replace(" ", "_")


def _bytes(v: float, _pos=None) -> str:
    for unit, name in ((1e24, "YB"), (1e21, "ZB"), (1e18, "EB"), (1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if v >= unit:
            return f"{v / unit:g} {name}"
    return f"{v / 1e3:g} kB"


def _power(v: float, _pos=None) -> str:
    return "1" if v == 1 else rf"$10^{{{round(log10(v))}}}$"


def _sci(v: float) -> str:
    e = floor(log10(v))
    m = v / 10 ** e
    return rf"$10^{{{e}}}$" if round(m, 1) == 1 else rf"${m:.2g}\times10^{{{e}}}$"


def _decades(lo: float, hi: float, step: int) -> list[float]:
    a, b = floor(log10(lo)), floor(log10(hi))
    return [10.0 ** e for e in range(a + (-a) % step, b + 1, step)]


def _style(design: str, headline: str):
    return "-" if design == headline else COMPARISON[design]


def _width(design: str, headline: str) -> float:
    return HEADLINE_LW if design == headline else THIN.get(design, COMPARISON_LW)


def _lines(references: list[dict], scope: str, scale: float) -> list[tuple[float, str, str]]:
    """The panel's reference lines in `scope`, scaled and ascending, each with its style: a line at 0 is in the
    caption, not here, and lines within `COINCIDENT` of one another are plotted once, under the first one's label."""
    rows = sorted(((r["value"] if "value" in r else r["by_scope"][scope]) * scale, r["label"], r.get("style", "dashed"))
                  for r in references if "value" in r or scope in r["by_scope"])
    out: list[tuple[float, str, str]] = []
    for amount, label, style in rows:
        if amount > 0 and not (out and amount <= out[-1][0] * (1 + COINCIDENT)):
            out.append((amount, label, style))
    return out


def _panel(fig, ax, out: dict, scope: str, panel: str, ylabel: str, series: list[dict], references: list[dict],
           caption: str, wrap: int = 62) -> None:
    """One panel of `scope`'s figure: `series` and `references`, the panel's that the figure plots, with `caption`
    under it."""
    v, p = out["vertical"], out["panels"][panel]
    colors = dict(zip(out["budgets"], COLORS))
    lo_x, hi_x = out["inputs"]["overheads"]["value"]
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.xaxis.set_major_locator(FixedLocator(_decades(lo_x, hi_x, 3) + [hi_x]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Marginal proof overhead", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0, -CAPTION_DROP, textwrap.fill(caption, wrap), transform=ax.transAxes, ha="left", va="top", fontsize=15,
            color="#444444")
    ax.axvline(v["x"], color=VERTICAL, lw=2.2, zorder=2)
    if not series:
        ax.set_yticks([])
        ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center", va="center", fontsize=26, color="#909090")
        return
    scale = 1 / 8 if p["unit"] == "bit" else 1 / out["inputs"]["h100_hour_flop"]["value"]
    fmt = _bytes if p["unit"] == "bit" else _power
    refs = _lines(references, scope, scale)
    ys = [y * scale for s in series for y in s["y"]]
    head = [y * scale for s in series if s["design"] == p["headline"] for y in s["y"]] or ys
    above = [r for r in refs if r[0] > max(head) * OFF_SCALE]
    refs = [r for r in refs if r not in above]
    top = head + [r[0] for r in refs]
    lo, hi = 10.0 ** floor(log10(min(ys + top))), 10.0 ** (floor(log10(max(top))) + 1)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(_decades(lo, hi, 3 if hi / lo > 1e12 else 2)))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(which="major", color="#E2E2E2", lw=.45)
    last = {"left": -1.0, "right": -1.0}
    for amount, label, style in refs:
        ax.axhline(amount, color="#909090", ls=REFERENCE_STYLE[style], lw=1.3, zorder=1)
        at = log10(amount / lo) / log10(hi / lo)
        gap = {k: at - v for k, v in last.items()}
        side = "left" if gap["left"] >= LABEL_GAP else max(gap, key=gap.get)
        last[side] = at
        ax.text(lo_x * 1.5 if side == "left" else hi_x / 1.5, amount, label, ha=side, va="bottom", fontsize=15,
                color="#555555", transform=offset_copy(ax.transData, fig=fig, y=3, units="points"), zorder=5,
                bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": .5})
    if above:
        ax.text(lo_x * 1.5, hi, "Above the axis: " + "; ".join(f"{label} ({fmt(a)})" for a, label, _ in above),
                ha="left", va="top", fontsize=14, color="#555555", wrap=True,
                transform=offset_copy(ax.transData, fig=fig, y=-4, units="points"))
    ax.text(v["x"] * 2.5, lo * 1.6, v["label"], color=VERTICAL, fontsize=15, ha="left", va="bottom", rotation=90)
    for s in series:
        headline = s["design"] == p["headline"]
        ax.plot(s["x"], [y * scale for y in s["y"]], color=colors[s["budget"]], ls=_style(s["design"], p["headline"]),
                lw=_width(s["design"], p["headline"]), zorder=4 if headline else 3)
    designs = list(dict.fromkeys(s["design"] for s in series))
    rests = p.get("capped", {})
    ax.legend([plt.Line2D([], [], color="#333333", ls=_style(d, p["headline"]), lw=THIN.get(d, 2.4)) for d in designs],
              [NAMES[d] + (f" ({rests[d]['status']})" if d in rests else "") for d in designs],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=min(len(designs), 2),
              frameon=False, fontsize=15, handlelength=2.4, columnspacing=1.2, borderaxespad=0)


def _figure(out: dict, scope: str, panels: int, width: float, left: float):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 20, "axes.labelsize": 22, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "savefig.facecolor": "white", "figure.facecolor": "white"})
    top, bottom = .84 if panels > 1 else .8, .27
    fig, axes = plt.subplots(1, panels, figsize=(width, 10), squeeze=False)
    fig.subplots_adjust(left=left, right=.99, bottom=bottom, top=top, wspace=.3)
    info = out["scopes"][scope]
    colors = dict(zip(out["budgets"], COLORS))
    where = "the world's compute" if info["who"] == "world" else "a datacenter's compute"
    fig.text(left, .985, f"{out['model']} served on all of {where} for a {info['period']}: {_sci(info['gpus'])} H200s, "
             f"{_sci(info['requests'])} requests", ha="left", va="center", fontsize=23)
    one = panels == 1
    fig.legend([plt.Line2D([], [], color=colors[b], lw=3.2) for b in out["budgets"]],
               [f"{b * 100:g}%" for b in out["budgets"]],
               title="Share of compute for\nreplays and proofs:" if one else "Share of compute for replays and proofs:",
               loc="upper right" if one else "center right",
               bbox_to_anchor=(.99, bottom - CAPTION_DROP * (top - bottom)) if one else (.99, .985),
               ncol=1 if one else 4, frameon=False, fontsize=18, title_fontsize=18, alignment="left")
    return fig, list(axes[0])


def _save(fig, outdir: Path, name: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.{ext}" for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", pad_inches=.1)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.1, metadata={"Date": None})
    plt.close(fig)
    return paths


def render(out: dict, scope: str, outdir: Path) -> list[Path]:
    """Sample `scope`'s main figure from `out` into `outdir` as `stem(scope)`.png and .svg; returns their paths."""
    fig, axes = _figure(out, scope, 3, 27, .06)
    captions = out["scopes"][scope]["captions"]
    for ax, (panel, ylabel) in zip(axes, PANELS):
        p = out["panels"][panel]
        _panel(fig, ax, out, scope, panel, ylabel,
               [s for s in p.get("series", []) if s["scope"] == scope and s.get("main", True)],
               [r for r in p["references"] if r.get("main", True)], captions[panel])
    return _save(fig, outdir, stem(scope))


def render_appendix(out: dict, scope: str, outdir: Path) -> list[Path]:
    """Sample `scope`'s appendix figure, the exfiltration panel with every design and the references the main figure
    leaves out, into `outdir` as `appendix_stem(scope)`.png and .svg; returns their paths."""
    fig, (ax,) = _figure(out, scope, 1, 15, .1)
    p = out["panels"]["exfiltration"]
    _panel(fig, ax, out, scope, "exfiltration", PANELS[0][1], [s for s in p["series"] if s["scope"] == scope],
           [r for r in p["references"] if not r.get("main", True)],
           out["scopes"][scope]["captions"]["exfiltration_appendix"], wrap=90)
    return _save(fig, outdir, appendix_stem(scope))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json", type=Path)
    ap.add_argument("--scope", action="append",
                    help="a scope's name, as figure1_overhead.json's `scopes` has it; default all")
    ap.add_argument("--out", type=Path, help="default: the JSON's directory")
    args = ap.parse_args(argv)
    out = json.loads(args.json.read_text())
    for scope in args.scope or out["scopes"]:
        for path in render(out, scope, args.out or args.json.parent) + render_appendix(out, scope,
                                                                                         args.out or args.json.parent):
            print(path)


if __name__ == "__main__":
    main()
