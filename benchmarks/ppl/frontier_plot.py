"""The perplexity–core-ε frontier: each design's perplexity increase against its core ε under each reading of the table
route, one panel per (model, set, context).

    frontier_plot.py eps --out EPS.json                      # core ε and γ of every design under every reading
    frontier_plot.py plot --eps EPS.json --png OUT.png --table OUT.md [--json OUT.json] RESULTS.json...

`eps` runs `machine-model eps --json` (benchmarks/pouw/machine_model) with the decided settings: Π the current vector
with UNI on issue at 0.92 (decision 73), the floor credit (decision 74), operands at the noise's floor, q* = 8 on the
lane route and 100% checking (the core ε, never mixed with ε_sample). The readings differ only in the table route each
design's zero-instance words leave open (`READINGS`):
- `closed`: no table route (q*_T at or above the threshold);
- `measured`: QSTAR's fr16_c32 as timed, 1.906 issue slots and 8 B of L2 a word at 6 TB/s, table builds free. It
  times 1-bit coins only, so b = 2 is left closed here, which flatters a design that fails on perplexity anyway;
- `builds charged`: tables over a coin drawn fresh each instance are rebuilt each instance and shared only by its rows,
  which prices a 16-bit table at ≥ 2¹⁷/M adds a word (≥ 16 at M = 8,192). One-sided Dither's tables are of its resident
  U, so it keeps the restated floor. Design A's best is SPEC's packed shared-memory byte route (§4.2a), at its
  optimistic 2.05 slots and 4 B a word against shared memory's 128 B/SM/clk. co32c's (c + U)·b sums overflow a byte
  for U ≥ ±8 (narrow_u.py's T), so its packed route needs halfword fields, ≥ 2.5 slots before index and build, closed;
  so do b = 2 coins on both of A's operands (8 chunks a segment at an 8-bit index);
- `restated floor`: SCALAR-ROUTE as the coordinator restated it (1.22 slots and 3.5 B at 6 TB/s, builds and combining
  free); b-bit coins on the indexed operand move the multi-field floor only to ≈ 1.15 slots and 4.0 B, since a field's
  entries are bounded by the table side's width, not the index's.

`plot` reads arms by name (`DESIGNS`), each point the mean over seeds of the arm's increase over its run's bf16, the bar
the seeds' range; the last results file holding an arm wins. Each reading's Pareto frontier is a step line; the box
is the target (core ε ≤ 0.5%, perplexity ≤ +1.0% at 8B and ≤ +2.0% at 70B)."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import report

#: (label, ppl arm, machine-model design, table class). H1 is co32c's linears and lm_head with the peeled U on ±N
#: (weights capped at ±(127 − N), narrow_u.py) and design A's attention; its core ε is co-dither's at every N.
DESIGNS = [
    ("Dither-1S (record)", "adopted", "dither-duo", "resident"),
    ("Dither-2S (A)", "adopted-bd-lm", "dither-two-sided", "bit"),
    ("Dither-2S + pv (Ap)", "adopted-bd-lm-pv", "dither-two-sided-pv", "bit"),
    ("Dither-2S, b = 2", "adopted-bd-lm-d2", "dither-two-sided", "b2"),
    ("Dither-C (co32c)", "co32c", "co-dither", "wide"),
    ("Dither-C + pv", "co32c-pv", "co-dither-pv", "wide"),
    ("H1, U ±8", "co32c-u8", "co-dither", "wide"),
    ("H1, U ±16", "co32c-u16", "co-dither", "wide"),
    ("H1, U ±32", "co32c-u32", "co-dither", "wide"),
    ("H1, U ±64", "co32c-u64", "co-dither", "wide"),
]
FLOOR = ["--lookups", "--lookup-q", "1.22", "--lookup-bytes", "3.5"]
MEASURED = ["--lookups", "--lookup-q", "1.906", "--lookup-bytes", "8"]
#: reading -> table class -> machine-model flags.
READINGS = {
    "closed": {"resident": [], "bit": [], "b2": [], "wide": []},
    "measured": {"resident": MEASURED, "bit": MEASURED, "b2": [], "wide": MEASURED},
    "builds charged": {"resident": FLOOR, "bit": ["--lookups", "--lookup-q", "2.05", "--lookup-bytes", "4", "--l2", "5.0e13"],
                       "b2": [], "wide": []},
    "restated floor": {"resident": FLOOR, "bit": FLOOR, "b2": ["--lookups", "--lookup-q", "1.15", "--lookup-bytes", "4.0"],
                       "wide": FLOOR},
}
COMMON = ["--credit", "floor", "--operands", "floor", "--uni-issue", "0.92", "--q", "8", "--checking", "full", "--json"]
COLORS = {"closed": "tab:green", "measured": "tab:blue", "builds charged": "tab:orange", "restated floor": "tab:red"}
HOLLOW = {"builds charged", "restated floor"}
RECOMMENDED = "H1, U ±16"
MARKERS = "*osDv^<>pPhX"
NOTES = {"H1, U ±8": "narrow_u.py's H: at folds 1.5 and 1 a bit-sliced route gains below x = 11.86, above QSTAR's "
                     "measured 10.66, so U ±8's core ε holds only at the as-built fold 3 (±16 holds to x = 9.48)"}
TARGET = {"llama-3.1-8b": 1.0, "llama-3.1-70b": 2.0}


def eps(out: str) -> int:
    doc: dict = {"settings": " ".join(COMMON[:-1]), "readings": {}}
    for reading, by_class in READINGS.items():
        doc["readings"][reading] = {}
        for label, _, design, cls in DESIGNS:
            cmd = ["machine-model", "eps", design, *COMMON, *by_class[cls]]
            r = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
            doc["readings"][reading][label] = {"core": r["core"], "gamma": r["gamma"], "skip": r["terms"]["skip"],
                                               "vector": r["vector"], "cmd": " ".join(cmd)}
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def rows(paths: list[str]) -> dict:
    """{(model, set, ctx, arm): row} for the arms DESIGNS names, from passing results files; the last one wins."""
    arms = {a for _, a, _, _ in DESIGNS}
    out: dict = {}
    for p in paths:
        r = json.loads(Path(p).read_text())
        if not r["passed"]:
            continue
        bf = report.arm_stats(r, "bf16")
        for arm in r["arms"]:
            s = report.arm_stats(r, arm)
            if arm not in arms or s is None:
                continue
            inc = [100 * (x / bf["ppl"] - 1) for x in s["seeds"]]
            out[(r["model"], r["set"], r["ctx"], arm)] = {
                "model": r["model"], "set": r["set"], "ctx": r["ctx"], "arm": arm, "seeds": len(inc), "ppl": s["ppl"],
                "inc": sum(inc) / len(inc), "inc_min": min(inc), "inc_max": max(inc),
                "run": r.get("env", {}).get("run", str(p))}
    return out


def points(rs: dict) -> list[tuple]:
    return sorted({k[:3] for k in rs}, key=lambda k: (k[0] != "llama-3.1-8b", k))


def table(rs: dict, ep: dict) -> str:
    pts = points(rs)
    rd = list(READINGS)
    head = ["design", "arm"] + [f"{m} · {s} · {c // 1024}K" for m, s, c in pts] + ["γ"] + [f"core ε, {r}" for r in rd]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---:|" * (len(pts) + 1 + len(rd))]
    marks = {label: "†" * (n + 1) for n, label in enumerate(NOTES)}
    for label, arm, _, _ in DESIGNS:
        cells = [label + marks.get(label, ""), f"`{arm}`"]
        for pt in pts:
            r = rs.get((*pt, arm))
            cells.append("–" if r is None else f"{r['inc']:+.2f} ({r['inc_min']:+.2f} to {r['inc_max']:+.2f}), {r['run']}")
        cells.append(f"{ep['readings']['closed'][label]['gamma']:.3f}")
        for reading in rd:
            e = ep["readings"][reading][label]["core"]
            meets = e <= 0.005 and all((r := rs.get((*pt, arm))) is not None and r["inc"] <= TARGET[pt[0]] for pt in pts)
            cells.append(f"{100 * e:.3f}%" + (" ✓" if meets else ""))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""] + [f"{marks[label]} {text}." for label, text in NOTES.items()]
    return "\n".join(lines) + "\n"


def frontier(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The Pareto-minimal (ε, perplexity) points, by ε."""
    best, out = float("inf"), []
    for x, y in sorted(pairs):
        if y < best:
            out.append((x, y))
            best = y
    return out


def plot(rs: dict, ep: dict, png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    pts = points(rs)
    fig, axes = plt.subplots(1, len(pts), figsize=(6.4 * len(pts) + 2.6, 5.4), squeeze=False)
    for j, pt in enumerate(pts):
        ax = axes[0][j]
        here = [r for k, r in rs.items() if k[:3] == pt]
        lo, hi = min(r["inc_min"] for r in here), max(r["inc_max"] for r in here)
        ax.set_ylim(max(0.0, lo - 0.25), hi + 0.25)
        ax.add_patch(Rectangle((0.005, -1), 0.5 - 0.005, TARGET[pt[0]] + 1, color="tab:green", alpha=0.08, zorder=0))
        for reading, color in COLORS.items():
            pairs = []
            for n, (label, arm, _, _) in enumerate(DESIGNS):
                r = rs.get((*pt, arm))
                if r is None:
                    continue
                x = max(100 * ep["readings"][reading][label]["core"], 0.006)
                pairs.append((x, r["inc"]))
                ring = reading in HOLLOW
                ax.errorbar([x], [r["inc"]], yerr=[[r["inc"] - r["inc_min"]], [r["inc_max"] - r["inc"]]],
                            marker=MARKERS[n], markersize=(12 if MARKERS[n] == "*" else 7) + 3 * ring, color=color,
                            markerfacecolor="none" if ring else color, capsize=2, linestyle="none", alpha=0.85)
                if label == RECOMMENDED and reading == "closed":
                    ax.annotate(label, (x, r["inc"]), xytext=(-60, -28), textcoords="offset points", fontsize=8,
                                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8})
            f = frontier(pairs)
            if f:
                xs = [p[0] for p in f] + [10.0]
                ys = [p[1] for p in f] + [f[-1][1]]
                ax.step(xs, ys, where="post", color=color, alpha=0.5, linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlim(0.006, 6)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.axhline(TARGET[pt[0]], color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("core ε (100% checking: ε_skip + ε_mand + ε_machine), %")
        ax.set_ylabel("perplexity increase over bf16, %")
        seeds = "/".join(map(str, sorted({r["seeds"] for r in here})))
        ax.set_title(f"{pt[0]} · {pt[1]} · {pt[2] // 1024}K ({seeds} seeds; bars: range)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    g = sorted({round(v["gamma"], 3) for v in ep["readings"]["closed"].values()})
    hd = [Line2D([], [], color=c, marker="o", markerfacecolor="none" if k in HOLLOW else c, linestyle="-",
                 label=f"table route: {k}") for k, c in COLORS.items()]
    hd += [Line2D([], [], color="black", marker=MARKERS[n], linestyle="none", label=lab)
           for n, (lab, _, _, _) in enumerate(DESIGNS)]
    fig.legend(handles=hd, fontsize=8, loc="center right", title=f"γ = {', '.join(map(str, g))}\n(every design)",
               title_fontsize=9)
    fig.suptitle(f"Perplexity against core ε by table-route reading; Π {next(iter(ep['readings']['closed'].values()))['vector']}"
                 f", floor credit, q* = 8; shaded: core ε ≤ 0.5% and the perplexity target", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1 - 2.4 / fig.get_figwidth(), 1))
    fig.savefig(png, dpi=140)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eps")
    e.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--eps", required=True)
    p.add_argument("--png", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--json")
    p.add_argument("results", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "eps":
        return eps(a.out)
    rs = rows(a.results)
    ep = json.loads(Path(a.eps).read_text())
    Path(a.table).write_text(table(rs, ep))
    if a.json:
        Path(a.json).write_text(json.dumps({"ppl": list(rs.values()), "eps": ep}, indent=1) + "\n")
    plot(rs, ep, a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
