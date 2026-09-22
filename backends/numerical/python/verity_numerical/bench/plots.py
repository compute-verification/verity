"""Overhead-over-time plots from the ledger (``bench.ledger``): one PNG per lane and one overview, into
``backends/numerical/reports/plots/``.

    python -m verity_numerical.bench.plots [--dir LEDGER_DIR] [--out PLOTS_DIR]

Log-scale overhead against wall-clock time; filled markers are whole-VU measurements, hollow ones are transition-unit
extrapolations (x96), crosses are component entries without an overhead (drawn at the lane's last overhead, greyed);
colour is the proof class; a star is a ``breakthrough`` entry, and every point carries its label.  Reference lines:
the 1e6 goal, the SP1 authenticated baseline and the TensorZKP HyperPlonk calibration.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from verity_numerical.bench import ledger

__all__ = ["OUT", "REFERENCES", "main", "plot_lane", "plot_overview"]

OUT = ledger.DIR.parent / "plots"
REFERENCES = {"goal 1e6": 1e6, "SP1 authenticated core, 1 VU per proof (4090, baselines.json)": 3.6e11, "TensorZKP HyperPlonk calibration (H100)": 5.0e7}
COLOURS = {"ARITHMETIC_DIAGNOSTIC": "#9e9e9e", "NON_ZK_PROOF_DIAGNOSTIC": "#1f77b4", "NON_ZK_PROOF": "#17becf",
           "COMPLETE_HVZK_BACKEND": "#ff7f0e", "COMPLETE_ZK_BACKEND": "#2ca02c", "NO_PROOF": "#7f7f7f", "COMPONENT": "#bcbd22"}


def _ts(e: dict[str, Any]) -> _dt.datetime:
    return _dt.datetime.fromisoformat(e["ts"])


def _draw(ax, entries: list[dict[str, Any]], annotate: bool = True, prefix: str = "") -> None:
    last = None
    for e in entries:
        y = e["overhead"]
        colour = COLOURS.get(e["proof_class"], "black")
        if y is None:
            if last is None:
                continue
            ax.plot(_ts(e), last, marker="x", color="#bbbbbb", linestyle="none", markersize=7)
            if annotate:
                ax.annotate(prefix + e["label"], (_ts(e), last), textcoords="offset points", xytext=(4, -12), fontsize=7, color="#888888")
            continue
        last = y
        if e["breakthrough"]:
            ax.plot(_ts(e), y, marker="*", color=colour, linestyle="none", markersize=14, markeredgecolor="black")
        else:
            fill = colour if e["scope"] == "vu" else "none"
            ax.plot(_ts(e), y, marker="o", markerfacecolor=fill, markeredgecolor=colour, linestyle="none", markersize=8)
        if annotate:
            ax.annotate(prefix + e["label"], (_ts(e), y), textcoords="offset points", xytext=(5, 5), fontsize=7,
                        fontweight="bold" if e["breakthrough"] else "normal")
    xs = [_ts(e) for e in entries if e["overhead"] is not None]
    ys = [e["overhead"] for e in entries if e["overhead"] is not None]
    if xs:
        ax.plot(xs, ys, color="#cccccc", linewidth=1, zorder=0)


def _finish(ax, title: str) -> None:
    for name, y in REFERENCES.items():
        ax.axhline(y, color="#dddddd", linestyle="--", linewidth=1)
        ax.annotate(name, (0.01, y), fontsize=6, color="#999999", xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylabel("overhead vs native A100 BF16 peak (K=1536 VU)")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=_dt.UTC))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    for c, name in COLOURS.items():
        ax.plot([], [], marker="o", color=name, linestyle="none", label=c)
    ax.plot([], [], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="none", label="filled: whole VU")
    ax.plot([], [], marker="o", markerfacecolor="none", markeredgecolor="black", linestyle="none", label="hollow: unit x96")
    ax.plot([], [], marker="*", color="black", linestyle="none", label="star: breakthrough")
    ax.legend(fontsize=6, loc="upper right")


def plot_lane(lane: str, entries: list[dict[str, Any]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw(ax, entries)
    hw = sorted({e["hardware"] for e in entries})
    _finish(ax, f"{lane} (track {entries[0]['track']}) -- overhead over time; hardware: {', '.join(hw)}")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / f"overhead_{lane}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overview(lanes: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 7))
    for lane, entries in lanes.items():
        _draw(ax, entries, annotate=True, prefix=f"{lane}: ")
    _finish(ax, "all lanes -- overhead over time (labels every breakthrough)")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / "overhead_all.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ledger.DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ns = ap.parse_args(argv)
    lanes = ledger.load_all(ns.dir)
    ns.out.mkdir(parents=True, exist_ok=True)
    written = [plot_lane(lane, entries, ns.out) for lane, entries in lanes.items() if entries]
    if lanes:
        written.append(plot_overview(lanes, ns.out))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
