"""Formalism v1's scores of the search (note:training-20261102T1817Z-report-training-formalism-v1; the plots of
note:training-20261102T1715Z-draft-training-efficiency-experiments), from per-checkpoint v1 quantities.

Each row is one checkpoint of one valid record: `id`, `start`, `cp`, `schedule_end`, `direction`, `reference`, `bin`
(the record's class), `night`, `compute` (|C|, in the analysis's unit), `val_bpb`, `Delta`, `Delta_grad`, `u1` (bits),
`phi`, `fit_run`, `fit_train` and `rho` (the Slowdown theorem's ρ, a target until Lean proves it, per policy key). A
policy key names (c, k); `grid` lists them as (c in tokens per block s, k in bits, key).

- H(l*) is the least |C| over every row that ends a complete schedule and reaches l* (te_core.frontier.target).
- W*(l*; c, k) is the least |C| ρ over the same rows, and S = W*/H.
- A2's curve per start and target: the least |C| among rows whose Delta (or Delta_grad) is at most x, against x.
- P1 per start: bpb against compute after θ (over C0 where known): H's frontier, W*'s at the reference policy and the
  band W* spans over the grid.
- P3: S over (c, k) at the reference target, per start.
- P4: per record at the reference target and policy, p = |C|/H against s = ρ, with iso-S diagonals."""
import json
import math
from pathlib import Path

import scorer as S

TARGETS = S.TARGETS
REF_TARGET = 0.01


def frontier(on: list[dict], l_star: float, keys: list[str]) -> dict:
    """H, W* and S per key at one l*, by te_core 3.5.3's target over `compute` and the v1 ρ."""
    rows = [x | {"macs": x["compute"]} for x in on if x.get("compute")]
    t = S.te_pin("frontier").target(rows, l_star, keys, reference=lambda x: x["reference"])
    of = (t["honest_of"] or {})
    out = {"l_star": l_star, "H": t["honest"], "H_id": of.get("id"), "H_cp": of.get("checkpoint"), "bound": t["bound"],
           "cells": {}}
    for k, c in t["cells"].items():
        o = c["of"] or {}
        out["cells"][k] = {"W": c["w_star"], "S": c["effective_slowdown"], "id": o.get("id"), "cp": o.get("checkpoint"),
                           "rho": c["rho"]}
    return out


def scores(rows: list[dict], keys: list[str], l0: dict[str, float]) -> dict:
    return {s: [frontier([x for x in rows if x["start"] == s], v * (1 - f), keys) | {"drop": f}
                for f in TARGETS] for s, v in l0.items()}


def a2(rows: list[dict], l0: dict[str, float], field: str = "Delta") -> dict:
    """Per start and target, the step curve: (x, least |C| among complete schedules reaching l* with field <= x)."""
    out = {}
    for s, v in l0.items():
        out[s] = {}
        for f in TARGETS:
            ls = v * (1 - f)
            reach = sorted((x for x in rows if x["start"] == s and x.get("schedule_end") and x["val_bpb"] <= ls
                            and x.get(field) is not None and x.get("compute")), key=lambda x: x[field])
            curve, best = [], None
            for x in reach:
                if best is None or x["compute"] < best["compute"]:
                    best = x
                    curve.append({"x": x[field], "compute": x["compute"], "id": x["id"], "cp": x["cp"],
                                  "direction": x["direction"]})
            out[s][f"{f:g}"] = curve
    return out


def p4(rows: list[dict], l0: dict[str, float], ref: str, sc: dict) -> dict:
    """Per start, each record's cheapest complete schedule reaching the reference target: p, s and S = p s."""
    out = {}
    i = TARGETS.index(REF_TARGET)
    for s, v in l0.items():
        h = sc[s][i]["H"]
        ls = v * (1 - REF_TARGET)
        pts = {}
        for x in rows:
            if x["start"] == s and x.get("schedule_end") and x["val_bpb"] <= ls and x.get("compute") and h:
                r = (x.get("rho") or {}).get(ref)
                if r is None:
                    continue
                q = pts.get(x["id"])
                if q is None or x["compute"] * r < q["compute"] * q["s"]:
                    pts[x["id"]] = {"id": x["id"], "cp": x["cp"], "direction": x["direction"], "compute": x["compute"],
                                    "p": x["compute"] / h, "s": r, "S": x["compute"] * r / h, "full": x["reference"],
                                    "Delta": x.get("Delta")}
        out[s] = sorted(pts.values(), key=lambda q: q["S"])
    return out


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, best = [], math.inf
    for c, b in sorted(points):
        if b < best:
            out.append((c, b))
            best = b
    return out


def plot_a2(curves: dict, field: str, path: Path, unit_bits: float = 8.0, unit: str = "bytes") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    starts = list(curves)
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4), squeeze=False)
    for ax, s in zip(axes[0], starts):
        for f, curve in curves[s].items():
            if curve:
                xs = [c["x"] / unit_bits for c in curve] + [max(c["x"] for c in curve) / unit_bits * 10]
                ys = [c["compute"] for c in curve] + [curve[-1]["compute"]]
                ax.step(xs, ys, where="post", label=f"-{100 * float(f):g}%")
                ax.scatter(xs[:-1], ys[:-1], s=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel(f"{field} at most ({unit})", fontsize=8)
        ax.set_ylabel("least |C| to the target", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"A2: least compute to each target against {field} (complete schedules)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_p1(rows: list[dict], l0: dict[str, float], keys: list[str], ref: str, c0: dict[str, float], path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    starts = list(l0)
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4), squeeze=False)
    nums = {}
    for ax, s in zip(axes[0], starts):
        ends = [x for x in rows if x["start"] == s and x.get("schedule_end") and x.get("compute")]
        norm = c0.get(s) or 1.0
        h = pareto([(x["compute"] / norm, x["val_bpb"]) for x in ends])
        w = pareto([(x["compute"] * x["rho"][ref] / norm, x["val_bpb"]) for x in ends if ref in (x.get("rho") or {})])
        band = [pareto([(x["compute"] * x["rho"][k] / norm, x["val_bpb"]) for x in ends if k in (x.get("rho") or {})])
                for k in keys]
        nums[s] = {"H": h, "W_ref": w, "C0": c0.get(s)}
        for b in band:
            if b:
                ax.step(*zip(*b), where="post", c="tab:red", alpha=0.15, lw=0.8)
        if h:
            ax.step(*zip(*h), where="post", c="k", label="H (honest)")
        if w:
            ax.step(*zip(*w), where="post", c="tab:red", label=f"W* at {ref}")
        for f in TARGETS:
            ax.axhline(l0[s] * (1 - f), color="0.85", lw=0.7, zorder=0)
        ax.set_xscale("log")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("compute after θ" + (" / C0" if c0.get(s) else " (|C|)"), fontsize=8)
        ax.set_ylabel("bpb", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("P1: loss against compute after θ (band: W* over the (c, k) grid)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return nums


def plot_p3(sc: dict, grid: list[tuple[float, float, str]], ref: str, path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = TARGETS.index(REF_TARGET)
    cs, ks = sorted({g[0] for g in grid}), sorted({g[1] for g in grid})
    key = {(g[0], g[1]): g[2] for g in grid}
    starts = list(sc)
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4), squeeze=False)
    nums = {}
    for ax, s in zip(axes[0], starts):
        cells = sc[s][i]["cells"]
        z = [[math.log10(cells[key[(c, k)]]["S"]) if key.get((c, k)) in cells and cells[key[(c, k)]]["S"] else math.nan
              for c in cs] for k in ks]
        nums[s] = {key[(c, k)]: (cells.get(key.get((c, k))) or {}).get("S") for c in cs for k in ks if (c, k) in key}
        m = ax.pcolormesh(range(len(cs) + 1), range(len(ks) + 1), z, vmin=0, vmax=3, cmap="viridis", shading="flat")
        for yi, row in enumerate(z):
            for xi, v in enumerate(row):
                if not math.isnan(v):
                    ax.text(xi + 0.5, yi + 0.5, f"{10 ** v:.3g}", ha="center", va="center", fontsize=6, color="w")
        rc = next(g for g in grid if g[2] == ref)
        ax.scatter([cs.index(rc[0]) + 0.5], [ks.index(rc[1]) + 0.5], marker="*", s=80, c="tab:red")
        ax.set_xticks([x + 0.5 for x in range(len(cs))], [f"{c:g}" for c in cs], fontsize=6)
        ax.set_yticks([y + 0.5 for y in range(len(ks))], [f"{k / 8 / 1000:g} kB" for k in ks], fontsize=6)
        ax.set_xlabel("c (tokens per block, s = c/w)", fontsize=8)
        ax.set_ylabel("k", fontsize=8)
        ax.set_title(f"{s}, l* = -{100 * REF_TARGET:g}%", fontsize=9)
        fig.colorbar(m, ax=ax, label="log10 S")
    fig.suptitle("P3: S over the policy plane at the reference target (* the reference policy)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return nums


def plot_p4(pts: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    starts = list(pts)
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4), squeeze=False)
    for ax, s in zip(axes[0], starts):
        ps = pts[s]
        if ps:
            ax.scatter([q["p"] for q in ps if not q["full"]], [q["s"] for q in ps if not q["full"]], s=12,
                       label="red team")
            ax.scatter([q["p"] for q in ps if q["full"]], [q["s"] for q in ps if q["full"]], s=24, marker="s", c="k",
                       label="full fine-tuning")
        for lv in (1, 10, 100, 1000):
            xs = [10 ** (e / 10) for e in range(0, 41)]
            ax.plot(xs, [lv / x for x in xs], c="0.8", lw=0.7)
            ax.text(lv ** 0.5 if lv > 1 else 1.05, lv ** 0.5 if lv > 1 else 1.05, f"S={lv}", fontsize=6, color="0.5")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.9, 1e4)
        ax.set_ylim(0.9, 1e4)
        ax.set_xlabel("p = |C| / H", fontsize=8)
        ax.set_ylabel("s = ρ at the reference policy (target)", fontsize=8)
        ax.set_title(f"{s}, l* = -{100 * REF_TARGET:g}%", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("P4: inefficiency against slowdown, one point per record (iso-S diagonals)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write(rows: list[dict], grid: list[tuple[float, float, str]], ref: str, c0: dict[str, float], out: Path) -> dict:
    """Every score and plot into `out`: scores.json, a2.png/json, a2-grad.png, p1, p3 and p4 png/json."""
    out.mkdir(parents=True, exist_ok=True)
    keys = [g[2] for g in grid]
    l0 = {s: v for s in S.STARTS if (v := S.regime_start_bpb(rows, s)) is not None}
    sc = scores(rows, keys, l0)
    a, ag = a2(rows, l0, "Delta"), a2(rows, l0, "Delta_grad")
    pts = p4(rows, l0, ref, sc)
    (out / "scores.json").write_text(json.dumps({"l0": l0, "ref": ref, "ref_target": REF_TARGET, "grid": grid,
                                                 "scores": sc}, indent=1) + "\n")
    (out / "a2.json").write_text(json.dumps({"Delta": a, "Delta_grad": ag}, indent=1) + "\n")
    plot_a2(a, "Delta", out / "a2.png")
    plot_a2(ag, "Delta_grad", out / "a2-grad.png")
    (out / "p1.json").write_text(json.dumps(plot_p1(rows, l0, keys, ref, c0, out / "p1.png"), indent=1) + "\n")
    (out / "p3.json").write_text(json.dumps(plot_p3(sc, grid, ref, out / "p3.png"), indent=1) + "\n")
    plot_p4(pts, out / "p4.png")
    (out / "p4.json").write_text(json.dumps(pts, indent=1) + "\n")
    return {"l0": l0, "scores": sc, "a2": a, "p4": pts}
