"""Perplexity against the dither's size b, per side choice: one panel per (model, set, context), a line per side.

    dither_plot.py --png OUT.png --table OUT.md [--json OUT.json] RESULTS.json...

Arms are read by name: `adopted-bd-lm` (both operands dithered everywhere), `side-lin-act` / `side-lin-w` (the linears'
and lm_head's weights, or activations, rounded), `side-qk-q` / `side-qk-k` (K, or Q, rounded in Q Kᵀ), each with an
optional `-dB` suffix for a b-bit dither (no suffix: b = 1), and `adopted` (the record, at b = 1). The top row varies
the linears' side with Q Kᵀ both-sided, the bottom row Q Kᵀ's with the linears both-sided. Each point is the mean over
seeds of the arm's increase over its run's bf16, the bar the seeds' range. The table sets each one-sided arm against
the both-sided arm at its b by the significance rule (`methodology.compare`; every run of a point has the same
windows). When an arm appears in several results, the last one given wins."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import methodology
import report

SIDES = {"adopted-bd-lm": ("both", "both"), "side-lin-act": ("act", "both"), "side-lin-w": ("w", "both"),
         "side-qk-q": ("both", "q"), "side-qk-k": ("both", "k")}
ROWS = (("linears", "linears and lm_head", (("act", "activations only"), ("w", "weights only"), ("both", "both"))),
        ("qk", "Q Kᵀ", (("q", "Q only"), ("k", "K only"), ("both", "both"))))


def classify(arm: str) -> tuple[str, str, int] | None:
    """(linears' side, Q Kᵀ's side, b) of an arm this plot knows, `record` for the adopted design, else None."""
    if arm == "adopted":
        return ("record", "record", 1)
    m = re.fullmatch(r"(.+?)(?:-d(\d))?", arm)
    if m is None or m.group(1) not in SIDES:
        return None
    return (*SIDES[m.group(1)], int(m.group(2) or 1))


def rows(paths: list[str]) -> list[dict]:
    """One row per (point, arm) this plot knows, from passing results files; the last results file wins."""
    out: dict = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        if not r["passed"]:
            continue
        bf = report.arm_stats(r, "bf16")
        for arm in r["arms"]:
            c = classify(arm)
            s = report.arm_stats(r, arm)
            if c is None or s is None:
                continue
            inc = [100 * (x / bf["ppl"] - 1) for x in s["seeds"]]
            out[(r["model"], r["set"], r["ctx"], arm)] = {
                "model": r["model"], "set": r["set"], "ctx": r["ctx"], "arm": arm, "linears": c[0], "qk": c[1],
                "b": c[2], "seeds": len(s["seeds"]), "ppl": s["ppl"], "ppl_min": s["min"], "ppl_max": s["max"],
                "ppl_sd": statistics.stdev(s["seeds"]) if len(s["seeds"]) > 1 else 0.0, "bf16": bf["ppl"],
                "inc": sum(inc) / len(inc), "inc_min": min(inc), "inc_max": max(inc),
                "run": r.get("env", {}).get("run", str(p)), "seed_ppl": s["seeds"], "windows": s["windows"]}
    return list(out.values())


def against_both(rs: list[dict], r: dict) -> tuple[float, str] | None:
    """A one-sided arm against the both-sided arm at its point and b: the difference of their increases over bf16
    (percentage points) and the significance rule's verdict (`<`: the one-sided arm is better)."""
    if (r["linears"], r["qk"]) in (("both", "both"), ("record", "record")):
        return None
    both = [o for o in rs if (o["model"], o["set"], o["ctx"], o["b"], o["linears"], o["qk"])
            == (r["model"], r["set"], r["ctx"], r["b"], "both", "both")]
    if not both:
        return None
    o = both[0]
    return r["inc"] - o["inc"], methodology.compare(r["seed_ppl"], o["seed_ppl"], r["windows"], o["windows"])


def points(rs: list[dict]) -> list[tuple[str, str, int]]:
    return sorted({(r["model"], r["set"], r["ctx"]) for r in rs}, key=lambda k: (k[0] != "llama-3.1-8b", k))


def series(rs: list[dict], point, row: str, side: str) -> list[dict]:
    """The rows of one line: `side` on `row`'s component, the other component both-sided."""
    other = "qk" if row == "linears" else "linears"
    got = [r for r in rs if (r["model"], r["set"], r["ctx"]) == point and r[row] == side and r[other] == "both"]
    return sorted(got, key=lambda r: r["b"])


def table(rs: list[dict]) -> str:
    lines = ["| model · set · context | arm | linears | Q Kᵀ | b | seeds | perplexity mean (range) | sd | Δ% vs bf16 (range) "
             "| vs both-sided (points, verdict) | run |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for pt in points(rs):
        for r in sorted((r for r in rs if (r["model"], r["set"], r["ctx"]) == pt),
                        key=lambda r: (r["b"], r["linears"] == "record", r["qk"] != "both", r["linears"], r["qk"])):
            vs = against_both(rs, r)
            lines.append(f"| {pt[0]} · {pt[1]} · {pt[2] // 1024}K | {r['arm']} | {r['linears']} | {r['qk']} | {r['b']} "
                         f"| {r['seeds']} | {r['ppl']:.4f} ({r['ppl_min']:.4f}–{r['ppl_max']:.4f}) | {r['ppl_sd']:.4f} "
                         f"| {r['inc']:+.2f} ({r['inc_min']:+.2f} to {r['inc_max']:+.2f}) "
                         f"| {'-' if vs is None else f'{vs[0]:+.2f} {vs[1]}'} | {r['run']} |")
    return "\n".join(lines) + "\n"


def plot(rs: list[dict], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = points(rs)
    colors = {"act": "tab:blue", "q": "tab:blue", "w": "tab:orange", "k": "tab:orange", "both": "tab:green"}
    fig, axes = plt.subplots(len(ROWS), len(pts), figsize=(4.6 * len(pts), 3.8 * len(ROWS)), squeeze=False)
    for j, pt in enumerate(pts):
        rec = [r for r in rs if (r["model"], r["set"], r["ctx"]) == pt and r["linears"] == "record"]
        for i, (row, title, sides) in enumerate(ROWS):
            ax = axes[i][j]
            drawn = False
            for side, label in sides:
                s = series(rs, pt, row, side)
                if not s:
                    continue
                drawn = True
                xs, ys = [r["b"] for r in s], [r["inc"] for r in s]
                err = [[r["inc"] - r["inc_min"] for r in s], [r["inc_max"] - r["inc"] for r in s]]
                ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, color=colors[side], label=label)
            if rec:
                r = rec[0]
                ax.errorbar([1], [r["inc"]], yerr=[[r["inc"] - r["inc_min"]], [r["inc_max"] - r["inc"]]],
                            marker="*", markersize=11, capsize=3, color="black", linestyle="none",
                            label="record (one-sided + U)")
            if not drawn:
                ax.set_axis_off()
                continue
            ax.set_title(f"{pt[0]} · {pt[1]} · {pt[2] // 1024}K\n{title}", fontsize=10)
            ax.set_xlabel("dither size b (coin entries in {0..2^b−1})")
            ax.set_ylabel("perplexity increase over bf16 (%)")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlim(0.7, 4.3)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("One-sided against both-sided dither (int8, Llama-3.1 / Qwen2.5); bars: range over 3 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=140)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--png", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--json")
    ap.add_argument("results", nargs="+")
    a = ap.parse_args(argv)
    rs = rows(a.results)
    Path(a.table).write_text(table(rs))
    if a.json:
        Path(a.json).write_text(json.dumps(rs, indent=1) + "\n")
    plot(rs, a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
