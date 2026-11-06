"""Figure 1's PNG and SVG for one scope, drawn from `figure1.json` alone, so anything holding the JSON draws the same
figure: three panels (exfiltration, infiltration, skipped work) against proof overhead, a colour per share of compute
spent on verification, the panel's headline design solid and its comparisons dashed or dotted, the measured overhead as
a vertical line, the references as dashed horizontal ones, and each panel's caption from the JSON.  A panel's y-axis
spans its headline curves and references, so a comparison curve may leave it at the top.

    uv run python benchmarks/spot_figures/render.py DIR/figure1.json [--scope "world year"] [--out DIR]
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

COLORS = ("#9ECAE1", "#3182BD", "#08306B")
COMPARISON = {"one-stage": (0, (6, 3)), "request": (0, (1.2, 2.4))}
NAMES = {"two-stage": "Two-stage", "one-stage": "One-stage, dot-product blocks", "request": "Request-level",
         "layer_of_token": "Deferred interiors, a layer of a token"}
VERTICAL = "#E69F00"
PRIMARY = "world year"
PANELS = (("exfiltration", "Exfiltration capacity"), ("infiltration", "Infiltration"),
          ("skipped_work", "Skipped work (H100-hours)"))


def stem(scope: str) -> str:
    return "figure1" if scope == PRIMARY else "figure1_" + scope.replace(" ", "_")


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


def render(out: dict, scope: str, outdir: Path) -> list[Path]:
    """Draw `scope`'s figure from `out` into `outdir` as `stem(scope)`.png and .svg; returns their paths."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 20, "axes.labelsize": 22, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(27, 10))
    fig.subplots_adjust(left=.06, right=.99, bottom=.27, top=.86, wspace=.3)
    info, v = out["scopes"][scope], out["vertical"]
    colors = dict(zip(out["budgets"], COLORS))
    lo_x, hi_x = out["inputs"]["overheads"]["value"]
    for ax, (panel, ylabel) in zip(axes, PANELS):
        p = out["panels"][panel]
        series = [s for s in p.get("series", []) if s["scope"] == scope]
        ax.set_xscale("log")
        ax.set_xlim(lo_x, hi_x)
        ax.xaxis.set_major_locator(FixedLocator(_decades(lo_x, hi_x, 3) + [hi_x]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Proof overhead", labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0, -.17, textwrap.fill(info["captions"][panel], 62), transform=ax.transAxes, ha="left", va="top",
                fontsize=15, color="#444444")
        ax.axvline(v["x"], color=VERTICAL, lw=2.2, zorder=2)
        if not series:
            ax.set_yticks([])
            ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center", va="center", fontsize=26, color="#909090")
            continue
        scale = 1 / 8 if p["unit"] == "bit" else 1 / out["inputs"]["h100_hour_flop"]["value"]
        shown = [r for r in p["references"] if "value" in r or scope in r["by_scope"]]
        refs = [((r["value"] if "value" in r else r["by_scope"][scope]) * scale, r["label"]) for r in shown]
        ys = [y * scale for s in series for y in s["y"]]
        top = [y * scale for s in series if s["design"] == p["headline"] for y in s["y"]] + [r for r, _ in refs]
        lo, hi = 10.0 ** floor(log10(min(ys + top))), 10.0 ** (floor(log10(max(top))) + 1)
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator(_decades(lo, hi, 3 if hi / lo > 1e12 else 2)))
        ax.yaxis.set_major_formatter(FuncFormatter(_bytes if p["unit"] == "bit" else _power))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(which="major", color="#E2E2E2", lw=.45)
        for amount, label in refs:
            ax.axhline(amount, color="#909090", ls=(0, (4, 3)), lw=1.3, zorder=1)
            ax.text(lo_x * 1.5, amount, label, ha="left", va="bottom", fontsize=15, color="#555555",
                    transform=offset_copy(ax.transData, fig=fig, y=3, units="points"))
        ax.text(v["x"] * 2.5, lo * 1.6, v["label"], color=VERTICAL, fontsize=15, ha="left", va="bottom", rotation=90)
        for s in series:
            headline = s["design"] == p["headline"]
            ax.plot(s["x"], [y * scale for y in s["y"]], color=colors[s["budget"]],
                    ls=_style(s["design"], p["headline"]), lw=3.2 if headline else 1.8, zorder=4 if headline else 3)
        designs = list(dict.fromkeys(s["design"] for s in series))
        ax.legend([plt.Line2D([], [], color="#333333", ls=_style(d, p["headline"]), lw=2.4) for d in designs],
                  [NAMES[d] for d in designs], loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(designs),
                  frameon=False, fontsize=15, handlelength=2.4, columnspacing=1.2, borderaxespad=0)
    where = "the world's compute" if info["who"] == "world" else "a datacenter's compute"
    fig.text(.06, .985, f"{out['model']} served on all of {where} for a {info['period']}: {_sci(info['gpus'])} H200s, "
             f"{_sci(info['requests'])} requests", ha="left", va="center", fontsize=23)
    fig.legend([plt.Line2D([], [], color=colors[b], lw=3.2) for b in out["budgets"]],
               [f"{b * 100:g}%" for b in out["budgets"]], title="Share of compute for replays and proofs:",
               loc="center right", bbox_to_anchor=(.99, .985), ncol=4, frameon=False, fontsize=18, title_fontsize=18,
               alignment="left")
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem(scope)}.{ext}" for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", pad_inches=.1)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=.1, metadata={"Date": None})
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json", type=Path)
    ap.add_argument("--scope", action="append", help="a scope's name, as figure1.json's `scopes` has it; default all")
    ap.add_argument("--out", type=Path, help="default: the JSON's directory")
    args = ap.parse_args(argv)
    out = json.loads(args.json.read_text())
    for scope in args.scope or out["scopes"]:
        for path in render(out, scope, args.out or args.json.parent):
            print(path)


if __name__ == "__main__":
    main()
