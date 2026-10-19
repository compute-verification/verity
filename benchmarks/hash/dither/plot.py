"""Dither's loop plot: each served row's standard number (decode B32 TPOT and saturation throughput, medians over its reps)
per arm against the row, read from the runs themselves: a run's `row` label (`research data labels RUN --json`) and its
out/summary.json (sbench.py's; a run cut before it, its reps.jsonl), fetched into ~/.research/runs/RUN when it isn't
there.  An arm that serves the dithered
product is drawn hollow, a peeled arm and the references filled, and the beside peel (z a fresh A B^T, rows 15-16: not
credit) with crosses.

    uv run --with matplotlib python3 benchmarks/hash/dither/plot.py --out DIR RUN [RUN ...]

Writes DIR/dither-loop.png and DIR/dither-loop.json (the points, each with its run id and row).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

RUNS = Path.home() / ".research" / "runs"
DITHERED = ("dither-1s", "dither-1s-warp", "dither-1s-split", "dither-1s-percall0", "dither-1s-percall-rtt",
            "dither-1s-credit", "dither", "dither-side")
BESIDE = ("dither-1s-peel", "dither-1s-split-peel")
MEASURES = (("tpot_b32_ms", "decode B32 TPOT (ms)"), ("tokens_per_s", "saturation throughput (tok/s)"))


def row_of(run: str) -> int:
    labels = json.loads(subprocess.run(["research", "data", "labels", run, "--json"], check=True, capture_output=True,
                                       text=True).stdout)
    rows = [lab["value"] for lab in labels if lab["key"] == "row"]
    if not rows:
        raise SystemExit(f"plot.py: {run} has no row label")
    return int(rows[-1])


def summary_of(run: str) -> dict:
    """The run's per-arm medians: its summary.json, else (a run cut before its summary) its reps.jsonl's."""
    p, reps = RUNS / run / "out" / "summary.json", RUNS / run / "out" / "reps.jsonl"
    if not p.exists() and not reps.exists():
        subprocess.run(["research", "fetch", run, "--all"], check=False)
    if p.exists():
        return json.loads(p.read_text())
    if not reps.exists():
        raise SystemExit(f"plot.py: no {p} or {reps} (the run timed nothing, or its outputs aren't fetched)")
    rows = [json.loads(x) for x in reps.read_text().splitlines() if x.strip()]
    per: dict = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        tp = [r["tpot_b32_ms"] for r in rows if r["arm"] == arm and "tpot_b32_ms" in r]
        th = [r["tokens_per_s"] for r in rows if r["arm"] == arm and "tokens_per_s" in r]
        per[arm] = {"tpot_b32_ms": statistics.median(tp) if tp else None, "tpot_all": tp,
                    "tokens_per_s": statistics.median(th) if th else None, "tokens_per_s_all": th}
    return {"per_arm": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    points = []
    for run in a.runs:
        row, s = row_of(run), summary_of(run)
        for arm, p in s["per_arm"].items():
            points.append({"run": run, "row": row, "arm": arm, **{m: p.get(m) for m, _ in MEASURES},
                           "tpot_all": p.get("tpot_all"), "tokens_per_s_all": p.get("tokens_per_s_all"),
                           "serves": "the dithered product" if arm in DITHERED else
                           "today's y, z computed beside (fresh A B^T); not credit" if arm in BESIDE else "today's y"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(13, 5))
    arms = list(dict.fromkeys(p["arm"] for p in points))
    for ax, (m, title) in zip(axes, MEASURES):
        for i, arm in enumerate(arms):
            xs = sorted((p["row"], p[m]) for p in points if p["arm"] == arm and p[m] is not None)
            if not xs:
                continue
            c = f"C{i % 10}"
            ax.plot([x for x, _ in xs], [y for _, y in xs], "-x" if arm in BESIDE else "-o", color=c,
                    label=arm + (" (beside, not credit)" if arm in BESIDE else ""),
                    markerfacecolor="none" if arm in DITHERED else c)
        ax.set_yscale("log")
        ax.set_xlabel("row (iteration)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dither one-stream on the RTX PRO 6000, int8 Llama-3.1-8B, T = 64 (hollow: serves the dithered product; x: not credit)")
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dither-loop.png", dpi=120)
    (out / "dither-loop.json").write_text(json.dumps(points, indent=1) + "\n")
    print(json.dumps({"png": str(out / "dither-loop.png"), "points": len(points)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
