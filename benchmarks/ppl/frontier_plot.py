"""The perplexity–core-ε frontier: each design's perplexity increase against its core ε under each reading of the table
route, one panel per (model, set, context).

    frontier_plot.py eps --out EPS.json                      # core ε and γ of every design under every reading
    frontier_plot.py plot --eps EPS.json --png OUT.png --table OUT.md [--json OUT.json] RESULTS.json...

`eps` runs `machine-model eps --json` (benchmarks/pouw/machine_model) with the decided settings: Π the current vector
with UNI on issue at 0.92, the floor credit, operands at the noise's floor, q* = 8 on the
lane route and 100% checking (the core ε, never mixed with ε_sample). The readings differ only in the table route each
design's zero-instance words leave open (`READINGS`):
- `closed`: no table route (q*_T at or above the threshold);
- `measured`: QSTAR's fr16_c32 as timed, 1.906 issue slots and 8 B of L2 a word at 6 TB/s, table builds free. It
  times 1-bit coins only, so b = 2 is left closed here, which flatters a design that fails on perplexity anyway;
- `builds charged`: tables over a coin drawn fresh each instance are rebuilt each instance and shared only by its rows,
  which prices a 16-bit table at ≥ 2¹⁷/M adds a word (≥ 16 at M = 8,192). Tables of a resident U (one-sided Dither's
  words, co32c's digits, the adopted P·V's) keep the restated floor. A's c·b words' best is SPEC's packed byte route
  (§4.2a), at its optimistic 2.05 slots and 4 B a word against shared memory's 128 B/SM/clk (1.28% at 2.0 slots, the
  threshold being 2.07 two-sided). co32c's (c + U)·b sums overflow a byte
  for U ≥ ±8 (narrow_u.py's T), so its packed route needs halfword fields, ≥ 2.5 slots before index and build, closed;
  so do b = 2 coins on both of A's operands (8 chunks a segment at an 8-bit index);
- `restated floor`: the scalar table route's floor, restated (1.22 slots and 3.5 B at 6 TB/s, builds and combining
  free); b-bit coins on the indexed operand move the multi-field floor only to ≈ 1.15 slots and 4.0 B, since a field's
  entries are bounded by the table side's width, not the index's.

A reading prices each kind of committed word by its zero instance's table class, not each design: machine-model puts
every word of a design on one route, so a design's core ε under a reading is the mix Σ share × ε(design, the kind's
flags) over its word kinds (`DESIGNS`), with each kind's share of the committed words from the model's shapes at the
point's context (`shares`). The mix is linear in the shares, which holds while the table-routed share leaves the LP's
binding row unchanged. The kinds: `linears` (the linears' and lm_head's T words, and co32c's O′ and committed O),
`digits` (co32c's D_d, U·z at the zero instance: subset sums of the resident U, so Dither's tables), `qk` and `pv`
(the adopted P·V peels P with a resident U, so its tables are Dither's).

`plot` reads arms by name (`DESIGNS`), each point the mean over seeds of the arm's increase over its run's bf16, the bar
the seeds' range; the last results file holding an arm wins. Each reading's Pareto frontier is a step line; the box
is the target (core ε ≤ 0.5%, perplexity ≤ +1.0% at 8B and ≤ +2.0% at 70B)."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import report

_A = {"linears": "bit", "qk": "bit", "pv": "resident"}
_CO = {"linears": "wide", "digits": "resident", "qk": "wide", "pv": "resident"}
_H1 = {"linears": "wide", "digits": "resident", "qk": "bit", "pv": "resident"}
#: (label, ppl arm, machine-model design, table class of each word kind). H1 is co32c's linears and lm_head with the
#: peeled U on ±N (weights capped at ±(127 − N), narrow_u.py) and design A's attention; its machine-model design is
#: co-dither at every N (D_d's count depends on k, not on U). pv's P·V (P + c with a fresh bit, Ycum's digits
#: committed) is priced as co32c's linears, unchecked: both pv designs fail on perplexity whatever it costs.
DESIGNS = [
    ("Dither-1S (record)", "adopted", "dither-duo", {"linears": "resident", "qk": "resident", "pv": "resident"}),
    ("Dither-2S (A)", "adopted-bd-lm", "dither-two-sided", _A),
    ("Dither-2S + pv (Ap)", "adopted-bd-lm-pv", "dither-two-sided-pv", {**_A, "pv": "wide"}),
    ("Dither-2S, b = 2", "adopted-bd-lm-d2", "dither-two-sided", {**_A, "linears": "b2", "qk": "b2"}),
    ("Dither-C (co32c)", "co32c", "co-dither", _CO),
    ("Dither-C + pv", "co32c-pv", "co-dither-pv", {**_CO, "pv": "wide"}),
    ("H1, U ±8", "co32c-u8", "co-dither", _H1),
    ("H1, U ±16", "co32c-u16", "co-dither", _H1),
    ("H1, U ±32", "co32c-u32", "co-dither", _H1),
    ("H1, U ±64", "co32c-u64", "co-dither", _H1),
]
#: model -> (hidden, layers, KV width, MLP width, vocabulary), Llama-3.1's.
MODELS = {"llama-3.1-8b": (4096, 32, 1024, 14336, 128256), "llama-3.1-70b": (8192, 80, 1024, 28672, 128256)}
FLOOR = ["--lookups", "--lookup-q", "1.22", "--lookup-bytes", "3.5"]
MEASURED = ["--lookups", "--lookup-q", "1.906", "--lookup-bytes", "8"]
#: reading -> table class -> machine-model flags.
PACKED = ["--lookups", "--lookup-q", "2.05", "--lookup-bytes", "4", "--l2", "5.0e13"]
FLOOR_B2 = ["--lookups", "--lookup-q", "1.15", "--lookup-bytes", "4.0"]
READINGS = {
    "closed": {"resident": [], "bit": [], "b2": [], "wide": []},
    "measured": {"resident": MEASURED, "bit": MEASURED, "b2": [], "wide": MEASURED},
    "builds charged": {"resident": FLOOR, "bit": PACKED, "b2": [], "wide": []},
    "restated floor": {"resident": FLOOR, "bit": FLOOR, "b2": FLOOR_B2, "wide": FLOOR},
}
COMMON = ["--credit", "floor", "--operands", "floor", "--uni-issue", "0.92", "--q", "8", "--checking", "full", "--json"]
COLORS = {"closed": "tab:green", "measured": "tab:blue", "builds charged": "tab:orange", "restated floor": "tab:red"}
HOLLOW = {"builds charged", "restated floor"}
RECOMMENDED = "H1, U ±16"
MARKERS = "*osDv^<>pPhX"
NOTES = {"H1, U ±8": "narrow_u.py's H: at folds 1.5 and 1 a bit-sliced route gains below x = 11.86, above QSTAR's "
                     "measured 10.66, so U ±8's core ε holds only at the as-built fold 3 (±16 holds to x = 9.48)"}
TARGET = {"llama-3.1-8b": 1.0, "llama-3.1-70b": 2.0}


def n_digits(lo: int, hi: int) -> int:
    """checked_strip.py's: base-128 digits for values in [lo, hi]."""
    m = 1
    while not (lo >= -(128 ** m) and hi // 128 ** (m - 1) <= 127):
        m += 1
    return m


def shares(model: str, ctx: int, kinds: dict) -> dict:
    """Each word kind's share of a token's committed words: the linears' cells k/32 words each (co32c's add m − 1
    digits, O′ and O), Q Kᵀ's and P·V's 1/32 of their products at the mean causal context ctx/2."""
    d, layers, kv, ff, vocab = MODELS[model]
    co = "digits" in kinds
    lin = dig = 0.0
    for k, n, reps in [(d, d + 2 * kv, layers), (d, d, layers), (d, 2 * ff, layers), (ff, d, layers), (d, vocab, 1)]:
        lin += reps * n * (k / 32 + 2 * co)
        dig += reps * n * (n_digits(k // 32 * -128, k // 32 * 127) - 1) * co
    att = layers * (ctx / 2) * d / 32
    words = {"linears": lin, "digits": dig, "qk": att, "pv": att}
    total = sum(words[kind] for kind in kinds)
    return {kind: words[kind] / total for kind in kinds}


def flag_key(flags: list[str]) -> str:
    return " ".join(flags) or "closed"


def eps(out: str) -> int:
    """machine-model's core ε of each design at every flag set its kinds take under some reading."""
    doc: dict = {"settings": " ".join(COMMON[:-1]), "designs": {}}
    runs: dict = {}
    for label, _, design, kinds in DESIGNS:
        entry = doc["designs"][label] = {"design": design, "by_flags": {}}
        for by_class in READINGS.values():
            for cls in kinds.values():
                key = flag_key(by_class[cls])
                if (design, key) not in runs:
                    cmd = ["machine-model", "eps", design, *COMMON, *by_class[cls]]
                    runs[design, key] = json.loads(subprocess.run(cmd, check=True, capture_output=True,
                                                                  text=True).stdout)
                r = runs[design, key]
                entry["by_flags"][key] = r["core"]
                entry.update(gamma=r["gamma"], vector=r["vector"])
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def core(ep: dict, label: str, reading: str, model: str, ctx: int) -> float:
    kinds = next(k for lab, _, _, k in DESIGNS if lab == label)
    by_flags = ep["designs"][label]["by_flags"]
    return sum(s * by_flags[flag_key(READINGS[reading][kinds[kind]])] for kind, s in shares(model, ctx, kinds).items())


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
    at = " / ".join(f"{m.removeprefix('llama-3.1-').upper()} {c // 1024}K" for m, _, c in pts)
    head = (["design", "arm"] + [f"{m} · {s} · {c // 1024}K" for m, s, c in pts] + ["γ"]
            + [f"core ε, {r} ({at})" for r in rd])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---:|" * (len(pts) + 1 + len(rd))]
    marks = {label: "†" * (n + 1) for n, label in enumerate(NOTES)}
    for label, arm, _, _ in DESIGNS:
        cells = [label + marks.get(label, ""), f"`{arm}`"]
        for pt in pts:
            r = rs.get((*pt, arm))
            spread = f"{r['inc_min']:+.2f} to {r['inc_max']:+.2f}" if r and r["seeds"] > 1 else "1 seed"
            cells.append("–" if r is None else f"{r['inc']:+.2f} ({spread}), {r['run']}")
        cells.append(f"{ep['designs'][label]['gamma']:.3f}")
        for reading in rd:
            es = [core(ep, label, reading, m, c) for m, _, c in pts]
            meets = all(e <= 0.005 for e in es) and all(
                (r := rs.get((*pt, arm))) is not None and r["inc"] <= TARGET[pt[0]] for pt in pts)
            cells.append(" / ".join(f"{100 * e:.3f}" for e in es) + "%" + (" ✓" if meets else ""))
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
                x = max(100 * core(ep, label, reading, pt[0], pt[2]), 0.006)
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
    g = sorted({round(v["gamma"], 3) for v in ep["designs"].values()})
    hd = [Line2D([], [], color=c, marker="o", markerfacecolor="none" if k in HOLLOW else c, linestyle="-",
                 label=f"table route: {k}") for k, c in COLORS.items()]
    hd += [Line2D([], [], color="black", marker=MARKERS[n], linestyle="none", label=lab)
           for n, (lab, _, _, _) in enumerate(DESIGNS)]
    fig.legend(handles=hd, fontsize=8, loc="center right", title=f"γ = {', '.join(map(str, g))}\n(every design)",
               title_fontsize=9)
    fig.suptitle(f"Perplexity against core ε by table-route reading (mixed over word kinds); Π "
                 f"{next(iter(ep['designs'].values()))['vector']}, floor credit, q* = 8; shaded: core ε ≤ 0.5% and the "
                 f"perplexity target", fontsize=10)
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
