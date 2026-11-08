"""Figure 6 of the SPOT post, the certified tick's trade-off: (a) a timeline of continuous admission against tick
admission, with the network certifier stamping each arrival's release bucket as its own layer; (b) latency and
throughput against the tick period, beside the no-tick point at each load; (c) the scheduler advice's term in the
exfiltration bound, with and without the tick, over the headline and secondary scopes, against the model's FP4 weights.

(b) reads the sweep's points (`point.json`: `point`, `load`, `rate`, `tick` with `sat_ms` and `free_ms`, or null
without the tick, `ttft_p50_s`, `ttft_p99_s`, `itl_p50_s`, `output_tok_per_s`, `goodput_ttft_req_per_s`, and
`busy_cycle.mean_us` and `preemptions` where measured) and is pending without them. A tick point named by
`--old-rule` took τ_sat from the no-tick step at its own load, the old rule, and is drawn hollow, off its load's line.
(c) reads an advice file: `machine_windows` by scope and `arms`, each a `label`, `tick` (true or false) and `bits`, the
advice a machine window is charged, so its term is that times the scope's windows, and a `pending` reason where the
arm's figure isn't a result yet, drawn hatched.

    uv run python benchmarks/spot_figures/figure6.py --model GLM-5.3 --advice ADVICE.json [--points POINT.json...]
        [--old-rule POINT...] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from math import floor, log10
from pathlib import Path

import figure1 as f1

SCHEMA = "spot-figure6/v1"
TARGET = "{model}'s weights at FP4"
#: the schematic's timeline (ms): arrivals, the busy cycles a continuous scheduler runs back to back, the release
#: bucket and the tick period
ARRIVALS = (35.0, 150.0, 170.0, 420.0, 540.0, 735.0)
CYCLES = (95.0, 110.0, 100.0, 120.0, 90.0, 105.0, 115.0, 100.0, 95.0, 110.0)
BUCKET, TICK, SPAN = 100.0, 110.0, 1000.0
METRICS = (("ttft_p50_s", "TTFT p50 (s)"), ("ttft_p99_s", "TTFT p99 (s)"), ("itl_p50_s", "ITL p50 (ms)"),
           ("output_tok_per_s", "Throughput (output tok/s)"))
COLORS = ("#08306B", "#D95F02", "#1B9E77", "#7570B3", "#E7298A")


def schematic() -> dict:
    """The timeline's admissions: continuously, each arrival joins the next busy cycle's step; with the tick, steps start
    only at tick boundaries and each arrival joins the first one after its release bucket closes, in release order."""
    starts, t = [], 0.0
    for c in CYCLES:
        starts.append(t)
        t += c
    ticks = [k * TICK for k in range(int(SPAN // TICK) + 1)]
    rows = []
    for i, a in enumerate(ARRIVALS):
        bucket = floor(a / BUCKET)
        closes = (bucket + 1) * BUCKET
        rows.append({"request": f"R{i + 1}", "arrival_ms": a, "bucket": bucket, "bucket_closes_ms": closes,
                     "continuous_ms": min(s for s in starts if s >= a), "tick_ms": min(k for k in ticks if k >= closes)})
    return {"bucket_ms": BUCKET, "tick_ms": TICK, "span_ms": SPAN, "steps_ms": starts, "ticks_ms": ticks,
            "requests": rows}


def sweep(points: list[dict], old_rule: tuple[str, ...] = ()) -> list[dict]:
    """The points by load, each load's no-tick point first and its tick points by period, those in `old_rule` marked."""
    unknown = set(old_rule) - {p.get("point") for p in points if p.get("tick")}
    if unknown:
        raise ValueError(f"--old-rule names no tick point of the sweep: {sorted(unknown)}")
    loads: dict[str, dict] = {}
    for p in points:
        g = loads.setdefault(p["load"], {"load": p["load"], "rate": p["rate"], "none": None, "tick": []})
        step = (p.get("busy_cycle") or {}).get("mean_us")
        row = {m: p.get(m) for m, _ in METRICS} | {"goodput_ttft_req_per_s": p.get("goodput_ttft_req_per_s"),
                                                     "point": p.get("point"), "run": p.get("run_id"),
                                                     "step_ms": step / 1e3 if step is not None else None,
                                                     "preemptions": p.get("preemptions")}
        if p.get("tick"):
            g["tick"].append(row | {"sat_ms": p["tick"]["sat_ms"], "free_ms": p["tick"]["free_ms"],
                                    "old_rule": p.get("point") in old_rule})
        else:
            g["none"] = row
    for g in loads.values():
        g["tick"].sort(key=lambda r: r["free_ms"])
    return sorted(loads.values(), key=lambda g: -g["rate"])


def run(model: str, advice: dict, points: list[dict], old_rule: tuple[str, ...] = ()) -> dict:
    prefix = TARGET.format(model=model)
    target = next(line for card in f1.slot_cards(model).get("references", []) for line in card["value"]["lines"]
                  if line["label"].startswith(prefix))
    scopes = (f1.PRIMARY, f1.SECONDARY)
    windows = advice["machine_windows"]
    arms = [a | {"by_scope": {s: a["bits"] * windows[s] for s in scopes}} for a in advice["arms"]]
    return {"schema": SCHEMA, "model": model, "primary": f1.PRIMARY, "secondary": f1.SECONDARY,
            "target": {"label": target["label"], "bits": target["value"]}, "windows": {s: windows[s] for s in scopes},
            "arms": arms, "schematic": schematic(), "sweep": sweep(points, old_rule)}


def _bytes(bits: float) -> str:
    for unit, name in ((1e24, "YB"), (1e21, "ZB"), (1e18, "EB"), (1e15, "PB"), (1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if bits / 8 >= unit:
            return f"{bits / 8 / unit:.3g} {name}"
    return f"{bits / 8e3:.3g} kB"


def _times(ratio: float) -> str:
    return f"{ratio:,.0f}x" if ratio >= 100 else f"{ratio:.3g}x" if ratio >= 1 else f"{ratio:.2g}x"


def _sweep_caption(out: dict) -> str:
    sw = out["sweep"]
    loads = ", ".join(f"{g['load']} ({g['rate']:g} req/s)" for g in sw)
    taus = sorted({(r["sat_ms"], r["free_ms"]) for g in sw for r in g["tick"] if not r["old_rule"]})
    text = (f"(b) {out['model']} served at each load ({loads}), one run a point: the no-tick point on the production "
            "scheduler beside the tick at each period, τ_sat / τ_free "
            + ", ".join(f"{a:.0f} / {f:.0f} ms" for a, f in taus) + ". ")
    pairs = [(g, r) for g in sw if g["none"] for r in g["tick"]]

    def ratio(g, r, m):
        return r[m] / g["none"][m] if r[m] is not None and g["none"][m] else None

    rates = [x for g, r in pairs if (x := ratio(g, r, "output_tok_per_s")) is not None]
    if rates:
        text += f"The tick's throughput is {min(rates):.2f}–{max(rates):.2f}x the no-tick rate at the same load"
        costs = [f"at {g['load']} and τ_free {r['free_ms']:.0f} ms, ITL p50 {_times(i)} and TTFT p50 {_times(f)}"
                 for g, r in pairs if not r["old_rule"]
                 and (i := ratio(g, r, "itl_p50_s")) is not None and (f := ratio(g, r, "ttft_p50_s")) is not None]
        text += ("; the cost is latency: " + "; ".join(costs) if costs else "") + ". "
    text += "".join(f"{g['load']}'s no-tick point preempted {g['none']['preemptions']:,.0f} requests: that load is "
                    "past this configuration's capacity. " for g in sw if g["none"] and g["none"].get("preemptions"))
    for g, r in pairs:
        if r["old_rule"] and r["step_ms"] is not None and (f := ratio(g, r, "ttft_p50_s")) is not None:
            text += (f"{r['point']} took τ_sat from the no-tick step at its load, the old rule: {r['sat_ms']:.0f} ms "
                     f"against the tick arm's own {r['step_ms']:.0f} ms step, so its queue grew (TTFT p50 {_times(f)}); "
                     "it is drawn hollow, off its load's line. ")
    return text


def caption(out: dict) -> str:
    p, s, t = out["primary"], out["secondary"], out["target"]
    a = ("(a) Continuously, a request joins the next step after it arrives, so which step it joins is set by its exact "
         "arrival and is advice the scheduler sends. With the certified tick, steps start only at tick boundaries and a "
         "request joins the first one after its release bucket closes, in release order: the batch is then a function "
         "of what the network certifier, the auditor's device, stamped. ")
    b = _sweep_caption(out) if out["sweep"] else "(b) Pending: the sweep of the tick's periods and loads. "
    rows = []
    for arm in out["arms"]:
        v, w = arm["by_scope"][p], arm["by_scope"][s]
        bits = f"{arm['bits']:,.0f}" if arm["bits"] >= 100 else f"{arm['bits']:.4g}"
        rows.append(f"{arm['label'][0].lower()}{arm['label'][1:]}, {bits} bits a machine window: "
                    + ("0 at every scope" if v == 0 else
                       f"{_bytes(v)} over a {p} ({_times(v / t['bits'])} the weights) and {_bytes(w)} over a {s} "
                       f"({_times(w / t['bits'])})")
                    + (f" (pending, hatched: {arm['pending']})" if arm.get("pending") else ""))
    c = (f"(c) The advice's term in the bound is a window's bits times the scope's machine windows "
         f"({out['windows'][p]:.3g} over a {p}, {out['windows'][s]:.3g} over a {s}), against {t['label']}: "
         + "; ".join(rows) + ". The tick's arms take the certifier's release buckets as the arrivals: without them the "
         "server could hold a request back by whole ticks, which is advice again. The run with the real network "
         "certifier is pending: in the measured runs a client stood in for it. Until a circuit checks the stop rule, "
         "each response's length is advice with the tick too, charged today at log2(max_tokens) + 1 bits a request. "
         "The 0 holds with prefix caching off, once the length rule is checked in-circuit (pending); with prefix "
         "caching on, its alternative is each hit sent whole and charged as advice, also pending. Every arm charges "
         "the K/V cache 0, pending two fixes: each step's cache write recorded as a proof block, with the "
         "sparse-attention gather staged, and the K/V rows from earlier steps checked as history ports in each "
         "sampled block. Until both are in, staged attention reads cost advice.")
    return a + b + c


def _timeline(ax, sc: dict) -> None:
    from matplotlib.patches import Rectangle

    span = sc["span_ms"]
    rows = {"clients": 4.2, "certifier": 3.15, "continuous": 1.95, "tick": .75}
    names = {"clients": "Clients", "certifier": "Network certifier\n(the auditor's device)",
             "continuous": "Continuous admission", "tick": "Tick admission"}
    for key, y in rows.items():
        ax.text(-12, y, names[key], ha="right", va="center", fontsize=14,
                color="#6A3D9A" if key == "certifier" else "#222222")
    ax.add_patch(Rectangle((0, rows["certifier"] - .35), span, .7, facecolor="#EFE7F6", edgecolor="#6A3D9A", lw=1.2))
    for k in range(int(span // sc["bucket_ms"]) + 1):
        ax.plot([k * sc["bucket_ms"]] * 2, [rows["certifier"] - .35, rows["certifier"] + .35], color="#B9A3D3", lw=.8)
    for i, (start, nxt) in enumerate(zip(sc["steps_ms"], sc["steps_ms"][1:] + [span])):
        ax.add_patch(Rectangle((start, rows["continuous"] - .28), min(nxt, span) - start - 4, .56,
                               facecolor="#C6DBEF" if i % 2 else "#9ECAE1", edgecolor="none"))
    for tk in sc["ticks_ms"]:
        if tk < span:
            ax.add_patch(Rectangle((tk, rows["tick"] - .28), min(sc["tick_ms"] * .9, span - tk) - 4, .56,
                                   facecolor="#FDD0A2", edgecolor="none"))
        ax.plot([tk, tk], [rows["tick"] - .42, rows["tick"] + .42], color="#D95F02", lw=1.6)
    arrow = {"arrowstyle": "-|>", "lw": 1.1, "shrinkA": 2, "shrinkB": 2}
    for r in sc["requests"]:
        a = r["arrival_ms"]
        ax.plot([a], [rows["clients"]], marker="v", ms=11, color="#222222")
        ax.text(a, rows["clients"] + .3, r["request"], ha="center", va="bottom", fontsize=12)
        ax.annotate("", (a, rows["certifier"] + .2), (a, rows["clients"] - .1), arrowprops=arrow | {"color": "#444"})
        ax.plot([a], [rows["certifier"] + .1], marker="s", ms=7, color="#6A3D9A")
        ax.text(a, rows["certifier"] - .12, f"b{r['bucket']}", ha="center", va="top", fontsize=10, color="#6A3D9A")
        ax.annotate("", (r["continuous_ms"], rows["continuous"] + .3), (a, rows["certifier"] - .36),
                    arrowprops=arrow | {"color": "#2171B5", "connectionstyle": "arc3,rad=0"})
        ax.annotate("", (r["tick_ms"], rows["tick"] + .3), (r["bucket_closes_ms"], rows["certifier"] - .36),
                    arrowprops=arrow | {"color": "#D95F02", "ls": (0, (3, 2))})
    ax.text(span + 10, rows["continuous"], "the step each request joins\nis set by its exact arrival:\nadvice",
            ha="left", va="center", fontsize=12, color="#2171B5")
    ax.text(span + 10, rows["tick"], "steps start at ticks; a request\njoins the first after its bucket\ncloses: 0 bits",
            ha="left", va="center", fontsize=12, color="#D95F02")
    ax.annotate("", (span, .05), (0, .05), arrowprops={"arrowstyle": "-|>", "color": "#666", "lw": 1})
    ax.text(span / 2, -.15, f"time (ms); release buckets of {sc['bucket_ms']:g} ms, ticks every {sc['tick_ms']:g} ms",
            ha="center", va="top", fontsize=12, color="#555555")
    ax.set_xlim(-10, span * 1.27)
    ax.set_ylim(-.4, 4.8)
    ax.axis("off")


def _sweep(axes, sw: list[dict]) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

    taus = sorted({r["free_ms"] for g in sw for r in g["tick"]})
    none_x = taus[0] / 2.2 if taus else 1.0
    for ax, (metric, label) in zip(axes, METRICS):
        ax.set_title(label, fontsize=15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if not sw:
            ax.text(.5, .5, "pending: the sweep", transform=ax.transAxes, ha="center", va="center", fontsize=14,
                    color="#888888", style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        scale = 1e3 if metric == "itl_p50_s" else 1.0
        for k, g in enumerate(sw):
            color = COLORS[k % len(COLORS)]
            on = [r for r in g["tick"] if r[metric] is not None and not r["old_rule"]]
            ax.plot([r["free_ms"] for r in on], [r[metric] * scale for r in on], color=color, marker="o", ms=8, lw=2,
                    label=f"{g['load']}, {g['rate']:g} req/s")
            for r in g["tick"]:
                if r["old_rule"] and r[metric] is not None:
                    ax.plot([r["free_ms"]], [r[metric] * scale], color=color, marker="o", ms=9, mfc="white", mew=2,
                            ls="none")
                    ax.annotate(r["point"] or "", (r["free_ms"], r[metric] * scale), (6, 4),
                                textcoords="offset points", fontsize=11, color=color)
            if g["none"] and g["none"][metric] is not None:
                y0 = g["none"][metric] * scale
                ax.plot([none_x], [y0], color=color, marker="D", ms=9, mfc="white", mew=2)
                ax.axhline(y0, color=color, lw=.8, ls=(0, (2, 3)), zorder=0)
        ax.set_xscale("log")
        ys = [r[metric] for g in sw for r in g["tick"] + [g["none"] or {metric: None}] if r[metric]]
        if ys and max(ys) > 20 * min(ys):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_locator(FixedLocator([none_x] + taus))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["no\ntick"] + [f"{t:.0f}" for t in taus], fontsize=12)
        ax.set_xlim(none_x / 1.4, (taus[-1] if taus else 1) * 1.4)
        ax.set_xlabel("τ_free (ms)", fontsize=13)
        ax.grid(which="major", axis="y", color="#E2E2E2", lw=.45)
    if sw:
        from matplotlib.lines import Line2D

        handles, _ = axes[0].get_legend_handles_labels()
        if any(r["old_rule"] for g in sw for r in g["tick"]):
            handles.append(Line2D([], [], color="#555555", marker="o", ms=9, mfc="white", mew=2, ls="none",
                                  label="old τ rule"))
        axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=12)


def _advice(ax, out: dict) -> None:
    p, s, t = out["primary"], out["secondary"], out["target"]["bits"]
    arms = out["arms"]
    vals = [a["by_scope"][sc] for sc in (p, s) for a in arms if a["by_scope"][sc] > 0]
    lo = min(vals + [t]) / 30
    colors = ["#08306B" if not a["tick"] else "#D95F02" for a in arms]
    labels, y = [], 0
    for sc in (p, s):
        for a, color in zip(arms, colors):
            v = a["by_scope"][sc]
            if v > 0:
                ax.barh(y, v, left=lo, color=color, alpha=(1 if sc == p else .55) * (.6 if a.get("pending") else 1),
                        height=.7, log=True, hatch="//" if a.get("pending") else None, edgecolor="white", lw=0)
                ax.text(v * 1.15, y, _bytes(v), va="center", ha="left", fontsize=12, color="#333333")
            else:
                ax.text(lo * 1.3, y, "0 bits in every window", va="center", ha="left", fontsize=12, color=color,
                        style="italic")
            labels.append(f"{a['label']}, over a {sc}")
            y += 1
        y += .6
    ax.set_yticks([i + (.6 if i >= len(arms) else 0) for i in range(len(labels))], labels, fontsize=12)
    ax.set_ylim(y - .3, -1.2)
    ax.set_xscale("log")
    hi = max(vals + [t]) * 30
    ax.set_xlim(lo, hi)
    ax.axvline(t, color="#555555", lw=1.6)
    ax.text(t * 1.1, -.9, out["target"]["label"], ha="left", va="bottom", fontsize=13, color="#555555")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([8 * 10.0 ** e for e in range(floor(log10(lo / 8)) + 1,
                                                                          floor(log10(hi / 8)) + 1)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _bytes(v)))
    ax.set_xlabel("The advice's term in the exfiltration bound")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(which="major", axis="x", color="#E2E2E2", lw=.45)


def render(out: dict, outdir: Path) -> list[Path]:
    """Draw the figure into `outdir` as figure6.png and .svg; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render import _save

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 16, "axes.labelsize": 15, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(18, 22))
    rows = {"height_ratios": (1.25, 1, 1.1), "hspace": .42, "wspace": .32, "right": .97, "top": .94, "bottom": .2}
    grid, sweep_grid = fig.add_gridspec(3, 4, left=.26, **rows), fig.add_gridspec(3, 4, left=.09, **rows)
    _timeline(fig.add_subplot(grid[0, :]), out["schematic"])
    _sweep([fig.add_subplot(sweep_grid[1, k]) for k in range(4)], out["sweep"])
    _advice(fig.add_subplot(grid[2, :]), out)
    for y, tag in ((.935, "(a)"), (.655, "(b)"), (.43, "(c)")):
        fig.text(.03, y, tag, fontsize=20, weight="bold")
    fig.text(.03, .96, f"The certified tick: what it costs {out['model']}'s serving, and the advice it removes",
             fontsize=22, ha="left", va="bottom")
    fig.text(.03, .17, textwrap.fill(caption(out), 160), ha="left", va="top", fontsize=13, color="#444444")
    return _save(fig, outdir, "figure6")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="GLM-5.3")
    ap.add_argument("--advice", type=Path, required=True, help="the advice file: machine windows and arms")
    ap.add_argument("--points", type=Path, nargs="*", default=[], help="the sweep's point.json files")
    ap.add_argument("--old-rule", nargs="*", default=[], metavar="POINT",
                    help="tick points whose τ_sat came from the no-tick step at their own load")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = run(args.model, json.loads(args.advice.read_text()), [json.loads(p.read_text()) for p in args.points],
              tuple(args.old_rule))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figure6.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(*render(out, args.out), sep="\n")


if __name__ == "__main__":
    main()
