"""The SPOT post's Figures 1-4 for one model, from its cards: each figure's JSON (every plotted series beside every
card it read) and its PNG and SVG into one directory, which `research data put --kind spot-figure/v1 --tree DIR`
publishes for the console.  Figure 1 is the total-cost form (`total_cost.py`, `figure1.json`), its skipped-work figure
beside it; the bounds against proof overhead (`figure1.py`, `figure1_overhead.json`) and their appendix come with it.
Figure 4 (`figure4.py`) reads Figure 1's configurations and serve taxes.

    uv run python benchmarks/spot_figures/figures.py --model GLM-5.3 --out DIR [--points N] [--request workload]
        [--workers N] [--no-lean]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import figure1
import figure2
import figure3
import figure4
import total_cost
from cards import MODELS


def write(out: dict, path: Path) -> None:
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, default=MODELS[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--points", type=int, default=57,
                    help="the proof overheads of the bounds against overhead, the vertical line's besides")
    ap.add_argument("--request", choices=figure1.REQUESTS, default="census",
                    help="a request's tokens: the census's mean, or the workload card's request")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes planning Figure 1's curves")
    ap.add_argument("--no-lean", dest="lean", action="store_false",
                    help="plan Figure 1 without Lean's check of its designs and without the break-even's certificate")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from render import render, render_appendix

    one = total_cost.run(args.model, request=args.request, workers=args.workers, lean=args.lean)
    one["at_budgets"] = total_cost.at_budgets(one)
    one["breakeven"] = total_cost.breakeven(args.model, request=args.request, workers=args.workers, lean=args.lean)
    (args.out / "breakeven.txt").write_text(total_cost.breakeven_table(one["breakeven"]))
    write(one, args.out / "figure1.json")
    for name in one["scopes"]:
        total_cost.render(one, name, args.out)
    over = figure1.run(args.model, args.points, request=args.request, workers=args.workers, lean=args.lean)
    over["at_overhead"] = figure1.at_overhead(over)
    write(over, args.out / "figure1_overhead.json")
    for name in over["scopes"]:
        render(over, name, args.out)
        render_appendix(over, name, args.out)
    two = figure2.run(args.model)
    write(two, args.out / "figure2.json")
    figure2.render(two, args.out)
    three = figure3.run(args.model)
    write(three, args.out / "figure3.json")
    figure3.render(three, args.out)
    four = figure4.run(args.model, one)
    write(four, args.out / "figure4.json")
    figure4.render(four, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
