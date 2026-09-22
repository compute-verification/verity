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

__all__ = ["OUT", "REFERENCES", "main", "plot_frontier", "plot_gate4", "plot_lane", "plot_overview"]

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
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False)


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
        _draw(ax, [e for e in entries if e["overhead"] is not None], annotate=False, prefix=f"{lane}: ")
        for e in entries:
            if e["breakthrough"] and e["overhead"] is not None:
                ax.annotate(f"{lane}: {e['label']}", (_ts(e), e["overhead"]), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, fontweight="bold")
    _finish(ax, "all lanes -- overhead over time (labels on breakthroughs only; per-lane plots carry every label)")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / "overhead_all.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


TRACK_COLOURS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "shared": "#7f7f7f", "sp1": "#7f7f7f", "0": "#7f7f7f"}
MEASURED_CLASSES = {"NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND"}


def plot_frontier(lanes: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    """Per track, the running best whole-VU overhead over time: solid = an actual proof was produced and verified
    (``NON_ZK_PROOF_DIAGNOSTIC`` and stronger), dashed = a re-pricing with at least one modelled bucket
    (``ARITHMETIC_DIAGNOSTIC``).  Only entries that set a new best for their track are labelled (staggered)."""
    fig, ax = plt.subplots(figsize=(13, 7))
    by_track: dict[str, list[dict[str, Any]]] = {}
    for lane, entries in lanes.items():
        for e in entries:
            if e["scope"] == "vu" and e["overhead"] is not None:
                by_track.setdefault(e["track"], []).append({**e, "_lane": lane})
    for track, entries in sorted(by_track.items()):
        colour = TRACK_COLOURS.get(track, "black")
        for measured, style in ((True, "-"), (False, "--")):
            sel = sorted((e for e in entries if (e["proof_class"] in MEASURED_CLASSES) == measured), key=_ts)
            if not sel:
                continue
            best = float("inf")
            xs, ys, firsts = [], [], []
            for e in sel:
                if e["overhead"] < best:
                    best = e["overhead"]
                    firsts.append(e)
                xs.append(_ts(e))
                ys.append(best)
            ax.step(xs, ys, where="post", color=colour, linestyle=style, linewidth=2 if measured else 1.2,
                    label=f"track {track} -- {'proved and verified' if measured else 're-priced, partly modelled'}")
            for i, e in enumerate(firsts):
                ax.plot(_ts(e), e["overhead"], marker="*" if e["breakthrough"] else "o", color=colour, linestyle="none",
                        markersize=12 if e["breakthrough"] else 6, markeredgecolor="black",
                        markerfacecolor=colour if measured else "white")
                dy = (12 + 11 * (i % 3)) * (1 if i % 2 == 0 else -1)
                ax.annotate(f"{e['_lane']}: {e['label']}"[:70], (_ts(e), e["overhead"]), textcoords="offset points",
                            xytext=(6, dy), fontsize=6.5, color=colour, arrowprops={"arrowstyle": "-", "color": "#cccccc", "lw": 0.6})
    for name, y in REFERENCES.items():
        ax.axhline(y, color="#dddddd", linestyle="--", linewidth=1)
        ax.annotate(name, (0.01, y), fontsize=6, color="#999999", xycoords=("axes fraction", "data"), xytext=(0, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylabel("best whole-VU overhead so far (vs native A100 BF16 peak, K=1536)")
    ax.set_title("frontier per track -- running best K=1536 VU overhead (labels where the best improves)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=_dt.UTC))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out / "frontier.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


GATE4_LANE = "gate4-h100"
DESIGN_ORDER = ("A", "B", "SP1", "C")


def _design(e: dict[str, Any]) -> str:
    """The design a gate-4 entry belongs to: the first token of its label (``A ...``, ``B ...``, ``SP1 ...``, ``C ...``)."""
    head = e["label"].split()[0].rstrip(":") if e["label"].split() else ""
    return head if head in DESIGN_ORDER else "other"


def plot_gate4(entries: list[dict[str, Any]], out: Path, name: str = "gate4_h100.png") -> Path:
    """Gate 4 (one machine, full B=4096 K=1536 batch): one bar per whole-VU entry, log overhead, grouped by design
    (A, B, SP1, C), coloured by proof class, hatched when the number is a sum of separately measured buckets
    (``ARITHMETIC_DIAGNOSTIC``) rather than an end-to-end proof; a hollow/pale bar is a cited number from other
    hardware (its ``hardware`` field is drawn under the label)."""
    rows = [e for e in entries if e["scope"] == "vu" and e["overhead"] is not None]
    rows.sort(key=lambda e: (DESIGN_ORDER.index(_design(e)) if _design(e) in DESIGN_ORDER else 99, e["overhead"]))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(rows) + 3), 6.5))
    x = 0.0
    ticks, labels = [], []
    for design in DESIGN_ORDER + ("other",):
        group = [e for e in rows if _design(e) == design]
        if not group:
            continue
        x0 = x
        for e in group:
            colour = COLOURS.get(e["proof_class"], "black")
            buckets = e["proof_class"] == "ARITHMETIC_DIAGNOSTIC"
            cited = "H100" not in e["hardware"]
            ax.bar(x, e["overhead"], width=0.8, color=colour, alpha=0.45 if cited else 0.95, edgecolor="black",
                   hatch="//" if buckets else None, linestyle="--" if cited else "-")
            ax.annotate(f"{e['overhead']:.2e}", (x, e["overhead"]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
            ticks.append(x)
            labels.append(e["label"].replace(f"{design} ", "", 1)[:34] + ("\n[" + e["hardware"][:22] + "]" if cited else ""))
            x += 1.0
        ax.annotate(design, ((x0 + x - 1.0) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("overhead vs native BF16 peak (K=1536 VU, 9.85 ps)")
    ax.set_title("gate 4 on one H100: end-to-end proofs (solid) vs measured-bucket sums (hatched); pale = cited, other hardware", fontsize=9)
    ax.axhline(REFERENCES["goal 1e6"], color="#dddddd", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    for c, name_ in COLOURS.items():
        ax.bar([], [], color=name_, label=c)
    ax.bar([], [], color="white", edgecolor="black", hatch="//", label="hatched: buckets summed, not a proof")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False)
    fig.tight_layout()
    path = out / name
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
        written.append(plot_frontier(lanes, ns.out))
    if lanes.get(GATE4_LANE):
        written.append(plot_gate4(lanes[GATE4_LANE], ns.out))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
