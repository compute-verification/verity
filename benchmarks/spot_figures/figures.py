"""The SPOT post's Figures 1-3 for one model, from its cards: each figure's JSON (every plotted series beside every
card it read) and its PNG and SVG, Figure 1's appendix figure too, into one directory, which `research data put --kind
spot-figure/v1 --tree DIR` publishes for the console.

    uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR [--points N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import figure1
import figure2
import figure3
from cards import MODELS


def write(out: dict, path: Path) -> None:
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, default=MODELS[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", type=int, default=57, help="Figure 1's proof overheads, the vertical line's besides")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from render import render, render_appendix

    one = figure1.run(args.model, args.points)
    one["at_overhead"] = figure1.at_overhead(one)
    write(one, args.out / "figure1.json")
    for name in one["scopes"]:
        render(one, name, args.out)
        render_appendix(one, name, args.out)
    two = figure2.run(args.model)
    write(two, args.out / "figure2.json")
    figure2.render(two, args.out)
    three = figure3.run(args.model)
    write(three, args.out / "figure3.json")
    figure3.render(three, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
