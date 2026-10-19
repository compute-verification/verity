"""The dither-hb kernel rows as one figure: per-layer gamma (1 + receipts / the plain int8 GEMM, over Llama-3.1-8B's four
linears) against M, a panel a card and a line an arm, from h1ubench.py's h1u.json files (a `research fetch`'s
out/h1u.json). An arm in several files takes the last file's value; the noise floor (h1noise.so's midstate kernel) is
dashed.

    python hb_plot.py --panel H200=A/h1u.json,B/h1u.json --panel B200=... [--arms h1u,h1u-ws,h1u-sp4-ws] --out dither-hb.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def series(doc: dict) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for m, d in doc.get("per_layer", {}).items():
        m = int(m)
        out.setdefault("h1u", {})[m] = d["gamma"]
        if d.get("floor_us"):
            out.setdefault("floor", {})[m] = d["gamma_floor"]
        for arm, v in d.get("variants", {}).items():
            if v.get("gamma") is not None:
                out.setdefault(arm, {})[m] = v["gamma"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True, metavar="CARD=H1U_JSON[,...]")
    ap.add_argument("--arms", help="comma list (default every arm)")
    ap.add_argument("--title", default="H1 + fx1 receipts beside the int8 GEMM, per layer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    want = args.arms.split(",") if args.arms else None
    fig, axes = plt.subplots(1, len(args.panel), figsize=(6.5 * len(args.panel), 4.8), squeeze=False)
    for ax, spec in zip(axes[0], args.panel):
        card, _, paths = spec.partition("=")
        arms: dict[str, dict[int, float]] = {}
        for p in paths.split(","):
            for arm, pts in series(json.loads(Path(p).read_text())).items():
                arms[arm] = pts
        for arm in sorted(arms, key=lambda a: (a != "floor", a)):
            if want and arm not in want and arm != "floor":
                continue
            ms = sorted(arms[arm])
            ax.plot(ms, [arms[arm][m] for m in ms], "--" if arm == "floor" else "-o", ms=3, label=arm)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("M (rows a call)")
        ax.set_ylabel("gamma = 1 + receipts / plain int8 GEMM")
        ax.set_title(card)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
