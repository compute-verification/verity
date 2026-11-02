"""The headline plots P1-P4 (note:training-20261102T1715Z-draft-training-efficiency-experiments), from the ledger.

    python -m te_core.plots --out DIR [--root LEDGER] [--pin COMMIT ...] [--policy reference|inference|server|c1e14/k1e4]
                            [--drop 0.01] [--g 1] [--from-scratch F.json]

Every point is scored by the training formalism (te_core.frontier's `formalism_rows`: |C| in gates, ρ at the start's
policy grid) over the valid submissions at the pin, with honest H(l) the least |C| over complete schedules reaching l
(the 14:57Z rule; a checkpoint where none is complete, drawn dashed) and W*(l) the least |C|ρ over the same records.
Each plot writes DIR/P<n>.png and DIR/P<n>.json, the numbers drawn; each figure's footer names the stubs its
records' quantities rest on (`formalism.STUBS`), so a v2.3 record's (its old fields, no traced quantities) shows as
such.

- P1, per start (rows: size, columns: tokens per parameter): val bits per byte against compute after θ over C0, log x:
  H, W* at the reference policy (`--policy`, default te_core.policy's REFERENCE), W*'s band over the reference's
  neighbouring decades of c and k, and the from-scratch curve F (`--from-scratch`) as context.
- P2: S = W*/H at the reference policy against the gain g = F(l*)/C0, one curve per start, coloured by l0 and styled
  by saturation; without F, against the drop from l0 (the axis says so).
- P3, per start: S over the (c, k) grid at a fixed g (`--g`, with F) or drop (`--drop`), filled by decade, with the
  start's inference-block and server-block shapes and the reference policy marked.
- P4, at that l* and the reference policy: p_A = |C_A|/H against s_A = ρ_A, log-log, one point per algorithm (a
  direction and mechanism, its least-|C|ρ record), iso-S diagonals, the reference directions' full training marked.

F.json is {start: [[gates, bpb], ...]}, the from-scratch runs' (compute, loss) on the same data and evaluator (the
ladder's), read as F(l) = the least compute of a point at or below l, log-interpolated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import frontier, policy  # noqa: E402
from .ledger import Ledger  # noqa: E402

LEVELS = (1, 10, 100, 1000, 10000)
DROPS = tuple(round(0.0025 * i, 4) for i in range(1, 41))


# ---- the data ----------------------------------------------------------------------------------------------------------

def load(rows: list[dict], pin, shapes: dict[str, dict]) -> tuple[list[dict], dict]:
    """The valid rows scored by the formalism, each with `schedule_end`, and the stubs they rest on."""
    ends = frontier.schedule_ends(rows)
    scored, _ = frontier.formalism_rows(frontier.valid(rows, pin), shapes)
    stubs: dict[str, int] = {}
    for r in scored:
        r["schedule_end"] = r["id"] in ends
        for k, v in r["measured"].labels.items():
            if not v.startswith("measured") and not v.startswith("counted"):
                stubs[f"{k}: {v}"] = stubs.get(f"{k}: {v}", 0) + 1
    return scored, stubs


def reference_key(shape: dict, name: str) -> str:
    if "/" in name:
        return name
    s = policy.shapes(shape)[name]
    return policy.key(10 ** round(math.log10(s["c"])), 10 ** round(math.log10(s["k"])))


def neighbours(key: str, keys: set[str]) -> list[str]:
    c, k = policy.of_key(key)
    out = [policy.key(c * a, k * b) for a in (0.1, 1, 10) for b in (0.1, 1, 10)]
    return [x for x in out if x in keys]


def at(on: list[dict], l_star: float, keys: list[str]) -> dict:
    return frontier.target(on, l_star, keys, size="gates")


def from_scratch(points: list[list[float]] | None):
    """F(l): the least compute of a from-scratch point at or below l, log-interpolated between points; None outside."""
    if not points:
        return None
    env, best = [], math.inf
    for c, l in sorted(points, key=lambda p: p[0]):
        if l < best:
            env.append((c, l))
            best = l

    def f(l: float) -> float | None:
        if not env or l < env[-1][1] or l > env[0][1]:
            return None
        for (c0, l0), (c1, l1) in zip(env, env[1:]):
            if l1 <= l <= l0:
                t = (l0 - l) / (l0 - l1) if l0 != l1 else 0
                return math.exp(math.log(c0) + t * (math.log(c1) - math.log(c0)))
        return env[0][0]
    return f


def saturation(shape: dict) -> float:
    return shape.get("tokens_per_param") or shape["tokens"] / shape["parameters"]


def panels(shapes: dict[str, dict], starts: list[str]) -> dict[str, tuple[int, int]]:
    """Each start's panel: its size's row, and its saturation's rank among that size's starts."""
    sizes = sorted({shapes[s].get("parameters", 0) for s in starts})
    out = {}
    for i, size in enumerate(sizes):
        same = sorted((s for s in starts if shapes[s].get("parameters", 0) == size), key=lambda s: saturation(shapes[s]))
        out |= {s: (i, j) for j, s in enumerate(same)}
    return out


def footer(fig, stubs: dict[str, int], pin) -> None:
    lines = [f"pin {' + '.join(c[:12] for c in pin)}; |C| in Boolean gates (te_core.gates); rho: 1 where Fit holds, "
             "Slowdown's target where it fails; S = W*/H (Security)"] + [f"{n} rows, {s}" for s, n in sorted(stubs.items())]
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=6, va="bottom", ha="left", family="monospace")


# ---- P1-P4 -------------------------------------------------------------------------------------------------------------

def p1(scored, shapes, ref: str, fs: dict, out: Path, stubs, pin) -> dict:
    starts = sorted({r["start"] for r in scored})
    where = panels(shapes, starts)
    nr, nc = max(i for i, _ in where.values()) + 1, max(j for _, j in where.values()) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.6 * nr + 1.6), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            axes[i][j].set_visible((i, j) in where.values())
    numbers = {}
    for start in starts:
        sh = shapes[start]
        ax = axes[where[start][0]][where[start][1]]
        on = [r for r in scored if r["start"] == start]
        l0 = frontier.start_bpb(on)
        c0, _ = policy.c0(sh)
        rk = reference_key(sh, ref)
        keys = sorted(on[0]["rho"])
        band = neighbours(rk, set(keys))
        lmin = min(r["val_bpb"] for r in on if r.get("val_bpb") is not None)
        ls = list(np.linspace(l0, lmin, 60))
        pts = []
        for l in ls:
            t = at(on, l, band)
            ws = [t["cells"][k]["w_star"] for k in band if t["cells"][k]["w_star"]]
            pts.append({"l": l, "honest": t["honest"], "bound": t["bound"], "w_star": t["cells"][rk]["w_star"],
                        "band": [min(ws), max(ws)] if ws else None})
        f = from_scratch(fs.get(start))
        x = lambda v: v / c0 if v else np.nan  # noqa: E731
        ax.plot([x(p["honest"]) for p in pts], ls, "k-", lw=1.5, label="honest H")
        ax.plot([x(p["w_star"]) for p in pts], ls, "C3-", lw=1.5, label=f"covert W* at {rk}")
        ax.fill_betweenx(ls, [x(p["band"][0]) if p["band"] else np.nan for p in pts],
                         [x(p["band"][1]) if p["band"] else np.nan for p in pts], color="C3", alpha=0.15,
                         label="W* over c, k within a decade")
        if any(p["bound"] for p in pts):
            ax.plot([x(p["honest"]) if p["bound"] else np.nan for p in pts], ls, "k--", lw=1, label="H from a checkpoint")
        if f:
            ax.plot([x(f(l)) if f(l) else np.nan for l in ls], ls, "C0:", label="from scratch F")
        ax.set_xscale("log")
        ax.axhline(l0, color="grey", lw=0.5)
        ax.set_title(f"{start}: {sh.get('parameters', 0) / 1e6:.0f}M params, {saturation(sh):g} tok/param, l0 {l0:.4f}",
                     fontsize=9)
        ax.set_xlabel("compute after theta / C0")
        ax.set_ylabel("val bits per byte")
        ax.legend(fontsize=6)
        numbers[start] = {"l0": l0, "c0_gates": c0, "policy": rk, "band_cells": band, "points": pts}
    footer(fig, stubs, pin)
    fig.tight_layout(rect=(0, 0.06 + 0.012 * len(stubs), 1, 1))
    fig.savefig(out / "P1.png", dpi=130)
    plt.close(fig)
    return numbers


def gain_axis(f, c0: float, l0: float, l: float) -> float | None:
    return (f(l) / c0 if f and f(l) else None) if f else (l0 - l) / l0


def p2(scored, shapes, ref: str, fs: dict, out: Path, stubs, pin) -> dict:
    starts = sorted({r["start"] for r in scored})
    have_f = all(fs.get(s) for s in starts)
    l0s = {s: frontier.start_bpb([r for r in scored if r["start"] == s]) for s in starts}
    cmap = plt.get_cmap("viridis")
    lo, hi = min(l0s.values()), max(l0s.values())
    sat_rank = {v: i for i, v in enumerate(sorted({saturation(shapes[s]) for s in starts}))}
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    numbers = {}
    for start in starts:
        sh, on = shapes[start], [r for r in scored if r["start"] == start]
        c0, _ = policy.c0(sh)
        f = from_scratch(fs.get(start)) if have_f else None
        rk = reference_key(sh, ref)
        pts = []
        for d in DROPS:
            l = l0s[start] * (1 - d)
            t = at(on, l, [rk])
            s = t["cells"][rk]["effective_slowdown"]
            g = gain_axis(f, c0, l0s[start], l)
            if s is not None and g is not None:
                pts.append({"drop": d, "l_star": l, "x": g, "S": s, "bound": t["bound"], "rho": t["cells"][rk]["rho"]})
        if pts:
            ax.plot([p["x"] for p in pts], [p["S"] for p in pts], styles[sat_rank[saturation(sh)] % 4], marker=".",
                    color=cmap((l0s[start] - lo) / (hi - lo) if hi > lo else 0.5),
                    label=f"{start} (l0 {l0s[start]:.3f}, {saturation(sh):g} tok/param)")
        numbers[start] = {"policy": rk, "points": pts}
    ax.set_yscale("log")
    if have_f:
        ax.set_xscale("log")
    ax.set_xlabel("gain g = F(l*)/C0" if have_f else "drop from l0, (l0 - l*)/l0 (no from-scratch curve F yet)")
    ax.set_ylabel(f"S = W*/H at the {ref} policy")
    ax.set_title("P2: Security's S = W*/H against the gain (colour: l0, style: saturation)", fontsize=9)
    ax.legend(fontsize=7)
    footer(fig, stubs, pin)
    fig.tight_layout(rect=(0, 0.08 + 0.016 * len(stubs), 1, 1))
    fig.savefig(out / "P2.png", dpi=130)
    plt.close(fig)
    return {"x": "g" if have_f else "drop", "starts": numbers}


def l_star_of(on, shape, f, l0: float, g: float | None, drop: float) -> tuple[float, str]:
    if f and g is not None:
        c0, _ = policy.c0(shape)
        ls = np.linspace(l0, min(r["val_bpb"] for r in on), 400)
        for l in ls:
            if f(l) and f(l) >= g * c0:
                return float(l), f"g = {g:g}"
    return l0 * (1 - drop), f"drop {100 * drop:g}%"


def p3(scored, shapes, fs: dict, g, drop: float, out: Path, stubs, pin) -> dict:
    starts = sorted({r["start"] for r in scored})
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4.6 + 1.4), squeeze=False)
    numbers = {}
    for ax, start in zip(axes[0], starts):
        sh, on = shapes[start], [r for r in scored if r["start"] == start]
        l0 = frontier.start_bpb(on)
        l, what = l_star_of(on, sh, from_scratch(fs.get(start)), l0, g, drop)
        cells = policy.grid(sh)
        keys = [policy.key(c, k) for c, k in cells]
        t = at(on, l, keys)
        cs, ks = sorted({c for c, _ in cells}), sorted({k for _, k in cells})
        z = np.full((len(ks), len(cs)), np.nan)
        for (c, k), key in zip(cells, keys):
            s = t["cells"][key]["effective_slowdown"]
            if s is not None:
                z[ks.index(k), cs.index(c)] = math.log10(s)
        numbers[start] = {"l_star": l, "at": what, "honest": t["honest"], "bound": t["bound"],
                          "S": {key: t["cells"][key]["effective_slowdown"] for key in keys},
                          "marks": {m: {"c": s["c"], "k": s["k"]} for m, s in policy.shapes(sh).items()}}
        if np.isnan(z).all():
            ax.text(0.5, 0.5, f"no record reaches l* = {l:.4f}", ha="center", transform=ax.transAxes)
        else:
            lc, lk = np.log10(cs), np.log10(ks)
            top = max(math.log10(LEVELS[-1]), float(np.nanmax(z))) + 1e-9
            levels = [math.log10(v) for v in LEVELS] + [top]
            cf = ax.contourf(lc, lk, z, levels=levels, cmap="magma_r", extend="neither")
            ax.contour(lc, lk, z, levels=levels[1:-1], colors="k", linewidths=0.5)
            fig.colorbar(cf, ax=ax, label="log10 S")
        for m, s in policy.shapes(sh).items():
            ax.plot(math.log10(s["c"]), math.log10(s["k"]), {"inference": "c^", "reference": "co", "server": "cs"}[m],
                    mec="k", ms=8)
            ax.annotate(m, (math.log10(s["c"]), math.log10(s["k"])), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("log10 c (gates per block)")
        ax.set_ylabel("log10 k (bits per block beyond theta)")
        ax.set_title(f"{start}: S at {what} (l* {l:.4f})", fontsize=9)
    footer(fig, stubs, pin)
    fig.tight_layout(rect=(0, 0.1 + 0.02 * len(stubs), 1, 1))
    fig.savefig(out / "P3.png", dpi=130)
    plt.close(fig)
    return numbers


def p4(scored, shapes, ref: str, fs: dict, g, drop: float, out: Path, stubs, pin) -> dict:
    starts = sorted({r["start"] for r in scored})
    fig, axes = plt.subplots(1, len(starts), figsize=(4.6 * len(starts), 4.6 + 1.4), squeeze=False)
    numbers = {}
    for ax, start in zip(axes[0], starts):
        sh, on = shapes[start], [r for r in scored if r["start"] == start]
        l0 = frontier.start_bpb(on)
        l, what = l_star_of(on, sh, from_scratch(fs.get(start)), l0, g, drop)
        rk = reference_key(sh, ref)
        t = at(on, l, [rk])
        h = t["honest"]
        pool = [r for r in on if r.get("val_bpb") is not None and r["val_bpb"] <= l and (r["schedule_end"] or t["bound"])]
        best: dict[tuple, dict] = {}
        for r in pool:
            a = (r.get("direction", ""), r.get("mechanism", ""))
            if a not in best or frontier.cost(r, rk, "gates") < frontier.cost(best[a], rk, "gates"):
                best[a] = r
        pts = [{"algorithm": f"{d}/{m}", "id": r["id"], "p": r["gates"] / h, "s": r["rho"][rk],
                "full_training": d in frontier.REF_DIRECTIONS} for (d, m), r in sorted(best.items())] if h else []
        numbers[start] = {"l_star": l, "at": what, "policy": rk, "honest": h, "bound": t["bound"], "points": pts}
        if not pts:
            ax.text(0.5, 0.5, f"no record reaches l* = {l:.4f}", ha="center", transform=ax.transAxes)
        else:
            ps, ss = [p["p"] for p in pts], [p["s"] for p in pts]
            lo, hi = min(min(ps), min(ss), 1) / 2, max(max(ps), max(ss)) * 2
            xs = np.logspace(math.log10(lo), math.log10(hi), 50)
            for i in range(0, 2 * math.ceil(math.log10(hi)) + 1, 2):
                ax.plot(xs, 10 ** i / xs, color="grey", lw=0.4)
                x_top = 10 ** i / (hi / 1.6)
                if lo < x_top < hi:
                    ax.annotate(f"S=1e{i}", (x_top, hi / 1.6), fontsize=6, color="grey")
            for p in pts:
                ax.plot(p["p"], p["s"], "*" if p["full_training"] else "o", color="C3" if p["full_training"] else "C0",
                        ms=10 if p["full_training"] else 5)
                ax.annotate(p["algorithm"], (p["p"], p["s"]), fontsize=5, xytext=(3, 2), textcoords="offset points")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("p_A = |C_A| / H")
        ax.set_ylabel(f"s_A = rho (Slowdown's target) at {rk}")
        ax.set_title(f"{start}: {what} (l* {l:.4f}); star: full training", fontsize=9)
    footer(fig, stubs, pin)
    fig.tight_layout(rect=(0, 0.1 + 0.02 * len(stubs), 1, 1))
    fig.savefig(out / "P4.png", dpi=130)
    plt.close(fig)
    return numbers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None)
    ap.add_argument("--policy", default=policy.REFERENCE, help="a reference shape (te_core.policy.MARKS) or a cell key")
    ap.add_argument("--drop", type=float, default=0.01, help="P3 and P4's l*, as a drop from l0, without F")
    ap.add_argument("--g", type=float, default=None, help="P3 and P4's l*, as a gain F(l*)/C0 (needs --from-scratch)")
    ap.add_argument("--from-scratch", type=Path, default=None, help="JSON {start: [[gates, bpb], ...]}")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    a = ap.parse_args(argv)
    shapes = policy.starts(a.manifest)
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    scored, stubs = load(Ledger(a.root).index(), pin, shapes)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("P1", lambda: p1(scored, shapes, a.policy, fs, a.out, stubs, pin)),
                     ("P2", lambda: p2(scored, shapes, a.policy, fs, a.out, stubs, pin)),
                     ("P3", lambda: p3(scored, shapes, fs, a.g, a.drop, a.out, stubs, pin)),
                     ("P4", lambda: p4(scored, shapes, a.policy, fs, a.g, a.drop, a.out, stubs, pin))):
        numbers = fn()
        (a.out / f"{name}.json").write_text(json.dumps({"pin": list(pin), "policy": a.policy, "stubs": stubs,
                                                        "data": numbers}, indent=1, default=str) + "\n")
        print(f"{a.out / name}.png, {name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
