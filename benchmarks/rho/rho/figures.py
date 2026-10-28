"""Figures of the overhead: what `analysis` finds, drawn. Matplotlib is imported only when a figure is drawn.

Each figure writes `<name>.png` and `<name>.json` (the numbers it plots) into an output directory, never the repository
(`tests/test_repository.py` forbids tracked renders):

* `class-vs-k`: each class's per-block overhead against the block's novel input budget `k`, at the inference block's
  gates and at the serving block's (one server between two wipes), and where each becomes vacuous.
* `overheads`: the overhead per scenario and family: its lower bound from the bounds at each status, each bound alone,
  and each attack beside it.
* `dark`: the overhead's lower bound, and the best known attack's, against the dark share, per scenario.
* `phase`: the overhead over pairs of parameters (`PHASES`), with its contours and frontier at each target.
* `tornado`: each scenario's tornado.
* `vulnerability`: each parameter's plausible range against the overhead across a window around it, and where it crosses
  each target.
* `corner`: the overhead as the ranged parameters move to their adversarial ends one at a time, down to all at once.

Scenario names here are defaults: a figure falls back to the first scenario that has what it needs, and skips itself
(saying so in `render`'s result) when none does.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from rho import analysis, lean
from rho.analysis import LO, STATUSES, Metric
from rho.classes import Outside, per_block, weight_per_gate
from rho.params import Registry
from rho.scenario import build, evaluate
from rho.theorem import Budget

WINDOWED = "onsite-h200-kimi"
#: (default scenario, x, y): each axis a `(table, field)` of the scenario's choice of that table
PHASES = (
    ("onsite-h200-kimi", ("regime", "k_wipe"), ("regime", "wipe_s")),
    ("onsite-h200-kimi", ("regime", "side"), ("regime", "ingress")),
    ("remote-h200-kimi", ("regime", "residual"), ("fleet", "eps_dark")),
)
STATUS_COLORS = {"proved": "#1a9850", "conditional": "#2c7bb6", "draft": "#f28e2b", "absent": "#999999",
                 "outside": "#cccccc"}
TARGET_STYLES = ((":", "#7b3294"), ("--", "#d7191c"), ("-.", "#404040"))
CONTOUR_STYLES = (("k", "solid"), ("#08306b", "dashed"), ("#54278f", "dashdot"))
DPI = 130


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def jsonable(x: Any) -> Any:
    """`x` with every non-finite float spelled out (`"inf"`, `"-inf"`, None for NaN), so the JSON is standard."""
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def _save(fig: Any, out: Path, stem: str, data: dict) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    png, js = out / f"{stem}.png", out / f"{stem}.json"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    _plt().close(fig)
    js.write_text(json.dumps(jsonable(data), indent=1) + "\n")
    return [png, js]


def _logspace(a: float, b: float, n: int) -> list[float]:
    return [a * (b / a) ** (i / (n - 1)) for i in range(n)]


def _span(p: Any, v: float, n: int, widen: float) -> list[float]:
    a, b = analysis.window(p, widen)
    return _logspace(a, b, n) if a > 0 else [a + (b - a) * i / (n - 1) for i in range(n)]


def _finite(v: float) -> float:
    return v if math.isfinite(v) else math.nan


def _pick(names: Sequence[str], default: str, ok: Callable[[str], bool]) -> str | None:
    for n in ([default] if default in names else []) + list(names):
        if ok(n):
            return n
    return None


def _key(registry: Registry, scenario: str, axis: tuple[str, str]) -> str | None:
    table, field = axis
    choice = registry.get(f"scenario.{scenario}.{table}")
    key = f"{table}.{choice}.{field}"
    return key if choice is not None and key in registry else None


def _unit(registry: Registry, key: str) -> str:
    u = registry.param(key).unit
    return f" ({u})" if u else ""


def _short(key: str) -> str:
    """`regime.onsite.wipe_s` → `onsite.wipe_s`: the table's name and the field."""
    parts = key.split(".")
    return ".".join(parts[1:]) if len(parts) > 2 else key


def _targets(ax: Any, targets: Sequence[float], axis: str = "y") -> None:
    for t, (ls, c) in zip(targets, TARGET_STYLES * len(targets), strict=False):
        (ax.axhline if axis == "y" else ax.axvline)(t, ls=ls, c=c, lw=1, label=f"target overhead {t:g}")


def class_vs_k(registry: Registry, out: Path, names: Sequence[str] | None = None,
               targets: Sequence[float] = (10.0, 100.0), grid: int = 30, family: int = 0) -> list[Path]:
    """Each class's per-block overhead `w c / q_T(c, k)` for `k` from 1e4 to 1e13 bits (`6 grid + 1` points), at the
    inference block's gates `c` and (for a regime with serving blocks) at the serving block's `c_S`, with the declared
    policy's `k` and the serving block's input `k_S` marked. Below 1 a class is vacuous: it lets the block run
    unthrottled."""
    points = 6 * grid + 1
    plt = _plt()
    names = list(names or registry.names("scenario"))
    name = _pick(names, WINDOWED, lambda n: hasattr(build(registry, n).regime, "budget")) or (names[0] if names else None)
    if name is None:
        return []
    s = build(registry, name)
    fam = s.families[min(family, len(s.families) - 1)]
    p = fam.profile(s.model, s.run)
    c, k_policy, units = s.context.c, s.context.policy.k, s.context.units
    w = weight_per_gate(p, units)
    win = s.regime.budget(s.context) if hasattr(s.regime, "budget") else None
    panels = [("the inference block's gates c", c)]
    if win is not None:
        panels.append(("the serving block's gates c_S = ⌊Φ_s Δ_wipe⌋", win.gates))
    ks = _logspace(1e4, 1e13, points)
    data: dict = {"scenario": name, "family": fam.name, "k": ks, "declared_k": k_policy,
                  "window_k": win.inputs if win else None,
                  "overhead": "w c / q_T(c, k); 0 where the statement's premise fails (no cap), ∞ where q_T = 0",
                  "panels": []}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.2), sharey=True, squeeze=False)
    lines: dict[str, Any] = {}
    top = 10.0
    for ax, (title, gates) in zip(axes[0], panels, strict=True):
        full = w * gates
        panel: dict = {"title": title, "gates": gates, "classes": {}, "attacks": {}, "outside": {}}
        for i, arg in enumerate((*s.bounds, *s.attacks)):
            try:
                cap = per_block(arg.allowance(p), units)
            except Outside as e:
                panel["outside"][arg.name] = str(e)
                continue

            def overhead(k: float, cap: Any = cap, gates: float = gates, full: float = full) -> float:
                n = cap(Budget(gates, k))
                return full / n if n > 0 else math.inf

            ys = [overhead(k) for k in ks]
            status = lean.weakest(arg.cites)
            if arg.side == "attack":
                j = next((j for j, y in enumerate(ys) if math.isfinite(y)), None)
                first = None
                if j is not None:
                    first = ks[0] if j == 0 else analysis.crossing(
                        lambda k, f=overhead: float(math.isfinite(f(k))), 0.5, ks[j - 1], ks[j])
                panel["attacks"][arg.name] = {"status": status, "first_finite_k": first, "overhead": ys}
                if first is not None:
                    lines[f"attack {arg.name}"] = ax.axvline(
                        first, c="k", ls=(0, (1, 2)), lw=1.2,
                        label=f"attack {arg.name} ({status}): overhead in the block "
                              f"{overhead(first * 1.0001):.3g} from k = {first:.2g}")
                continue
            vacuous = next((j for j, y in enumerate(ys) if y <= 1), None)
            k_vac = None
            if vacuous is not None:
                k_vac = ks[0] if vacuous == 0 else analysis.crossing(overhead, 1.0, ks[vacuous - 1], ks[vacuous])
            j = next((j for j, y in enumerate(ys) if y == 0), None)
            fails = None if j is None else ks[0] if j == 0 else analysis.crossing(
                lambda k, f=overhead: float(f(k) == 0), 0.5, ks[j - 1], ks[j])
            panel["classes"][arg.name] = {"status": status, "overhead": ys, "vacuous_from_k": k_vac,
                                          "premise_fails_from_k": fails,
                                          "at_declared_k": overhead(k_policy),
                                          "at_window_k": overhead(win.inputs) if win else None}
            color = f"C{i % 10}"
            line, = ax.plot(ks, [y if y > 0 else math.nan for y in ys], c=color, lw=1.8,
                            ls="--" if status == "draft" else "-", label=f"{arg.name} ({status})")
            lines.setdefault(arg.name, line)
            if k_vac is not None:
                ax.plot([k_vac], [1.0], "o", c=color, ms=5, mec="k", mew=0.5)
            if fails is not None:
                last = max((k for k, y in zip(ks, ys, strict=True) if k < fails), default=None)
                if last is not None:
                    ax.plot([last], [overhead(last)], "x", c=color, ms=8, mew=2)
            if win is not None and gates == win.gates and overhead(win.inputs) > 1:
                ax.annotate(f"{arg.name}: {overhead(win.inputs):.3g} at K", (win.inputs, overhead(win.inputs)),
                            xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8,
                            color=color, weight="bold", bbox={"fc": "white", "alpha": 0.85, "lw": 0, "pad": 1})
            top = max([top, *(y for y in ys if math.isfinite(y))])
        ax.axhspan(1e-3, 1, color="#eeeeee", zorder=0)
        ax.text(1.3e4, 0.3, "vacuous: the block may run unthrottled", fontsize=8, color="#555555", va="center")
        ax.axhline(1, c="#888888", lw=0.8)
        lines["declared k"] = ax.axvline(k_policy, c="#2c7bb6", lw=1, ls="-.",
                                         label=f"declared policy k = {k_policy:.2g} bits")
        if win is not None:
            lines["window K"] = ax.axvline(win.inputs, c="#d7191c", lw=1, ls="-.",
                                           label=f"serving block k_S = {win.inputs:.2g} bits")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ks[0], ks[-1])
        ax.set_xlabel("block's novel input budget k (bits)")
        ax.set_title(f"at {title}\nc = {gates:.3g} gates", fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        data["panels"].append(panel)
    axes[0][0].set_ylim(1e-2, top * 3)
    axes[0][0].set_ylabel("per-block overhead w c / q_T(c, k)")
    fig.legend(lines.values(), [h.get_label() for h in lines.values()], loc="lower center",
               ncol=min(4, len(lines)), fontsize=8, bbox_to_anchor=(0.5, -0.02 - 0.035 * math.ceil(len(lines) / 4)))
    fig.suptitle(f"Training classes per block: {s.model.name}, family {fam.name} (× = premise fails, ○ = vacuous)",
                 fontsize=11)
    return _save(fig, out, "class-vs-k", data)


def overheads(registry: Registry, out: Path, names: Sequence[str] | None = None,
              targets: Sequence[float] = (10.0, 100.0), grid: int = 30) -> list[Path]:
    """The overhead for each scenario and family: its lower bound (every bound at once, the claim), the lower bound from
    the bounds and cuts at each status or stronger, each bound alone (colored by status, refuted ones hollow), and each
    attack below them, a construction beside the bound rather than an end of it."""
    plt = _plt()
    names = list(names or registry.names("scenario"))
    rows: list[dict] = []
    for name in names:
        r = analysis.result(registry, name)
        by_status = {st: evaluate(analysis.restrict(r.scenario, st)) for st in STATUSES[:3]}
        for i, f in enumerate(r.families):
            best = f.best_attack
            rows.append({"scenario": name, "family": f.family, "lo": f.lo,
                         "attack": {"argument": best.argument, "rho": best.rho} if best else None,
                         "lo_by_status": {st: by_status[st].families[i].lo for st in by_status},
                         "bounds": [{"argument": e.argument, "status": e.status, "premise": e.premise, "rho": e.rho}
                                    for e in f.bounds],
                         "excluded": list(f.combined.excluded),
                         "attacks": [{"argument": e.argument, "status": e.status, "rho": e.rho} for e in f.attacks],
                         "contradictions": list(f.contradictions)})
    finite = [v for row in rows for v in (row["lo"], *row["lo_by_status"].values(),
                                          *(e["rho"] for e in row["bounds"] + row["attacks"]))
              if v and math.isfinite(v) and v > 0]
    xmax = max(finite + list(targets)) * 3
    fig, ax = plt.subplots(figsize=(9, 1.2 + 1.1 * len(rows)))
    seen: set[str] = set()

    def label(text: str) -> str:
        if text in seen:
            return "_"
        seen.add(text)
        return text

    def edge(v: float) -> float:
        return v if math.isfinite(v) else xmax

    def num(v: float) -> str:
        return f"{v:.3g}" if math.isfinite(v) else ("∞" if v > 0 else "—")

    for y, row in enumerate(reversed(rows)):
        ax.plot(edge(row["lo"]), y, "|", ms=22, mew=3, c="k", zorder=5, label=label("the lower bound (every bound)"))
        for e in row["bounds"]:
            if not math.isnan(e["rho"]):
                refuted = e["premise"] == "refuted"
                hollow = {"mfc": "none"} if refuted else {}
                ax.plot(edge(e["rho"]), y + 0.25, "o", ms=5, c=STATUS_COLORS[e["status"]], alpha=0.8, zorder=3,
                        label=label("a bound alone, premises refuted" if refuted else f"a bound alone ({e['status']})"),
                        **hollow)
        for st, v in row["lo_by_status"].items():
            ax.plot(edge(v), y, "D", ms=9, c=STATUS_COLORS[st], mec="k", zorder=4,
                    label=label(f"the lower bound from {', '.join(STATUSES[:STATUSES.index(st) + 1])} bounds and "
                                "cuts"))
        for e in row["attacks"]:
            if not math.isnan(e["rho"]):
                ax.plot(edge(e["rho"]), y - 0.25, "v", ms=8, c=STATUS_COLORS[e["status"]], mec="k", zorder=3,
                        label=label(f"an attack ({e['status']})"))
        by = ", ".join(f"{st} {num(v)}" for st, v in row["lo_by_status"].items())
        attack = f"attack {row['attack']['argument']} {num(row['attack']['rho'])}" if row["attack"] else "no attack"
        ax.text(xmax * 1.05, y, f"overhead ≥ {num(row['lo'])}; {attack}\nby status: {by}", va="center", fontsize=8)
    _targets(ax, targets, "x")
    ax.set_yticks(range(len(rows)), [f"{r['scenario']}\n{r['family']}" for r in reversed(rows)], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(min([0.8, *finite]) / 2, xmax)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("the overhead on the target run (∞ drawn at the right edge)")
    ax.set_title("The overhead: its lower bound by status, each bound alone, and the attacks beside it", fontsize=11)
    ax.grid(True, axis="x", which="major", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.25 / len(rows)), ncol=3)
    return _save(fig, out, "overheads", {"rows": rows, "targets": list(targets)})


def dark(registry: Registry, out: Path, names: Sequence[str] | None = None, targets: Sequence[float] = (10.0, 100.0),
         grid: int = 30) -> list[Path]:
    """The overhead's lower bound (solid) and the best known attack's (dashed) against the dark share `ε_dark`
    (`grid + 11` points), per scenario, with `μ/ε_dark`, the most the dark pool alone allows."""
    points = grid + 11
    plt = _plt()
    names = [n for n in (names or registry.names("scenario")) if _key(registry, n, ("fleet", "eps_dark"))]
    if not names:
        return []
    p0 = registry.param(_key(registry, names[0], ("fleet", "eps_dark")))
    ds = _logspace(min(1e-4, p0.low / 10), 0.5, points)
    data: dict = {"d": ds, "scenarios": {}}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ms = set()
    for i, name in enumerate(names):
        key = _key(registry, name, ("fleet", "eps_dark"))
        lo = analysis.sweep(registry, name, key, ds, LO)
        attack = analysis.sweep(registry, name, key, ds, analysis.ATTACK)
        base = analysis.result(registry, name)
        m = min(f.profile.efficiency for f in base.families)
        ms.add(m)
        data["scenarios"][name] = {"key": key, "lo": [v for _, v in lo], "attack": [v for _, v in attack], "m": m,
                                   "base_d": registry[key], "base_lo": base.lo, "base_attack": base.attack}
        c = f"C{i % 10}"
        ax.plot(ds, [_finite(v) for _, v in lo], c=c, lw=2, label=f"{name}: lower bound")
        ax.plot(ds, [_finite(v) for _, v in attack], c=c, lw=1.2, ls="--", label=f"{name}: best known attack")
        ax.plot([registry[key]], [base.lo], "o", c=c, mec="k")
    for m in sorted(ms):
        ax.plot(ds, [m / d for d in ds], c="k", lw=0.8, ls=(0, (1, 1)), label=f"m/d (dark pool alone, m = {m:g})")
    ax.axvspan(p0.low, p0.high, color="#fde0dd", alpha=0.6, zorder=0, label=f"plausible d ({_short(p0.key)})")
    _targets(ax, targets)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dark share ε_dark (the dark leaves' share of the inventory's capacity)")
    ax.set_ylabel("overhead")
    ax.set_title("The overhead against the dark share, per scenario (• at today's ε_dark)", fontsize=11)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out, "dark", data)


def phase(registry: Registry, out: Path, names: Sequence[str] | None = None, targets: Sequence[float] = (10.0, 100.0),
          grid: int = 30, metric: Metric = LO, specs: Iterable = PHASES) -> list[Path]:
    """The overhead over each pair of parameters in `specs` (a window around both plausible ranges), with its
    contours at each target (from the grid) and the frontier (by bisection, dots), today's values (★) and the
    plausible box (dashed)."""
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    plt = _plt()
    names = list(names or registry.names("scenario"))
    panels = []
    for default, x, y in specs:
        def ok(n: str, x: tuple = x, y: tuple = y) -> bool:
            kx, ky = _key(registry, n, x), _key(registry, n, y)
            return bool(kx and ky and registry.param(kx).ranged and registry.param(ky).ranged)

        name = _pick(names, default, ok)
        if name is None:
            continue
        kx, ky = _key(registry, name, x), _key(registry, name, y)
        px, py = registry.param(kx), registry.param(ky)
        xs, ys = _span(px, px.value, grid, 0.5), _span(py, py.value, grid, 0.5)
        z = analysis.phase(registry, name, kx, xs, ky, ys, metric)
        fronts = {t: analysis.frontier(registry, name, kx, xs, ky, ys[0], ys[-1], t, metric) for t in targets}
        panels.append({"scenario": name, "x": kx, "y": ky, "xs": xs, "ys": ys, "rho": z,
                       "base": [px.value, py.value], "box": [[px.low, px.high], [py.low, py.high]],
                       "frontier": {str(t): f for t, f in fronts.items()}})
    if not panels:
        return []
    vals = [v for p in panels for row in p["rho"] for v in row if math.isfinite(v) and v > 0]
    norm = LogNorm(vmin=max(min(vals), 1.0) if vals else 1, vmax=max(vals + [max(targets) * 10]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0), squeeze=False, layout="constrained")
    for ax, p in zip(axes[0], panels, strict=True):
        z = [[_finite(v) for v in row] for row in p["rho"]]
        flat = [v for row in z for v in row if not math.isnan(v)]
        mesh = ax.pcolormesh(p["xs"], p["ys"], z, norm=norm, cmap="RdYlGn", shading="nearest")
        levels = [t for t in sorted(targets) if flat and min(flat) < t < max(flat)]
        handles = []
        for i, t in enumerate(targets):
            c, ls = CONTOUR_STYLES[i % len(CONTOUR_STYLES)]
            if t in levels:
                ax.contour(p["xs"], p["ys"], z, levels=[t], colors=c, linewidths=1.4, linestyles=ls)
            pts = [(xv, yv) for xv, yv in p["frontier"][str(t)] if yv is not None]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "o", c="white", mec=c, mew=1.2, ms=4)
            handles.append(Line2D([], [], c=c, ls=ls, marker="o", mfc="white", mec=c, ms=4, label=f"overhead {t:g}"))
        ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.85)
        p["not_reached"] = [t for t in targets if t not in levels]
        if p["not_reached"] and flat:
            ax.text(0.02, 0.98, "\n".join(f"overhead {'<' if max(flat) < t else '>'} {t:g} everywhere here "
                                          f"({min(flat):.3g} to {max(flat):.3g})" for t in p["not_reached"]),
                    transform=ax.transAxes, va="top", fontsize=8, bbox={"fc": "white", "alpha": 0.8, "lw": 0})
        (xl, xh), (yl, yh) = p["box"]
        ax.plot([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl], c="k", ls="--", lw=1)
        ax.plot([p["base"][0]], [p["base"][1]], "*", c="white", mec="k", ms=14)
        for axis, vals_ in (("x", p["xs"]), ("y", p["ys"])):
            if vals_[0] > 0:
                (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        ax.set_xlabel(_short(p["x"]) + _unit(registry, p["x"]))
        ax.set_ylabel(_short(p["y"]) + _unit(registry, p["y"]))
        ax.set_title(p["scenario"], fontsize=10)
    fig.colorbar(mesh, ax=axes[0].tolist(), label="overhead (lower bound)" if metric is LO else "overhead",
                 shrink=0.9)
    fig.suptitle("The overhead over pairs of parameters: contours from the grid and frontier dots by bisection at the "
                 "targets, ★ today's values, dashed box the plausible ranges", fontsize=10)
    return _save(fig, out, "phase", {"metric": getattr(metric, "__name__", str(metric)), "targets": list(targets),
                                     "panels": panels})


def tornado(registry: Registry, out: Path, names: Sequence[str] | None = None, targets: Sequence[float] = (10.0, 100.0),
            grid: int = 30, top: int = 10, metric: Metric = LO) -> list[Path]:
    """Each scenario's tornado (those where some parameter moves the overhead): the overhead at each parameter's low
    and high ends."""
    plt = _plt()
    names = list(names or registry.names("scenario"))
    swings = {n: analysis.tornado(registry, n, None, metric)[:top] for n in names}
    swings = {n: s for n, s in swings.items() if s}
    if not swings:
        return []
    fig, axes = plt.subplots(len(swings), 1, figsize=(9, 0.9 + 0.42 * sum(len(s) for s in swings.values())
                                                      + 0.8 * len(swings)), squeeze=False)
    for ax, (name, ss) in zip(axes[:, 0], swings.items(), strict=True):
        base = ss[0].base
        for y, s in enumerate(reversed(ss)):
            for v, c, lab in ((s.at_low, "#2c7bb6", "at the range's low end"), (s.at_high, "#d7191c",
                                                                               "at the range's high end")):
                v = _finite(v)
                ax.barh(y, v - base, left=base, height=0.6, color=c, alpha=0.8, label=lab if y == 0 else "_")
                if v > 0 and base > 0 and abs(math.log10(v / base)) > 0.02:
                    ax.annotate(f"{v:.3g}", (v, y), xytext=(3 if v > base else -3, 0), textcoords="offset points",
                                ha="left" if v > base else "right", va="center", fontsize=7)
        ax.axvline(base, c="k", lw=1)
        _targets(ax, targets, "x")
        ax.set_yticks(range(len(ss)), [f"{_short(s.key)}  [{s.low:.3g}, {s.high:.3g}]{_unit(registry, s.key)}"
                                       for s in reversed(ss)], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("overhead (lower bound)" if metric is LO else "overhead")
        ax.set_title(f"{name}: overhead {base:.3g} at today's values", fontsize=10)
        ax.grid(True, axis="x", which="major", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "tornado", {"scenarios": {n: [vars(s) for s in ss] for n, ss in swings.items()}})


def vulnerability(registry: Registry, out: Path, names: Sequence[str] | None = None,
                  targets: Sequence[float] = (10.0, 100.0), grid: int = 30, top: int = 12,
                  metric: Metric = LO) -> list[Path]:
    """For each parameter that moves the overhead: the overhead across a window around its plausible range (the
    colored strip of `grid + 1` points, on the plausible range's own scale: 0 its low end, 1 its high end), the range
    (black box), today's value (|) and where it crosses each target (filled inside the range)."""
    points = grid + 1
    from matplotlib.colors import LogNorm

    plt = _plt()
    names = list(names or registry.names("scenario"))
    found = {}
    for n in names:
        swings = analysis.tornado(registry, n, None, metric)[:top]
        if swings:
            found[n] = analysis.margins(registry, n, targets, metric=metric, points=points, swings=swings)
    if not found:
        return []
    from matplotlib.lines import Line2D

    vals = [v for ms in found.values() for m in ms for _, v in m.samples if math.isfinite(v) and v > 0]
    norm = LogNorm(vmin=max(min(vals), 1.0), vmax=max(vals + [max(targets) * 10]))
    fig, axes = plt.subplots(len(found), 1, squeeze=False, layout="constrained",
                             figsize=(11, 1.6 + 0.55 * sum(len(m) for m in found.values()) + 0.9 * len(found)))
    markers = ("D", "o", "s")
    xlo, xhi = -3.0, 4.0
    for ax, (name, ms) in zip(axes[:, 0], found.items(), strict=True):
        for y, m in enumerate(reversed(ms)):
            p = registry.param(m.key)
            us = [analysis.position(p, v) for v, _ in m.samples]
            edges = [us[0]] + [(a + b) / 2 for a, b in pairwise(us)] + [us[-1]]
            mesh = ax.pcolormesh(edges, [y - 0.32, y + 0.32], [[_finite(v) for _, v in m.samples]], norm=norm,
                                 cmap="RdYlGn")
            ax.plot([0, 1, 1, 0, 0], [y - 0.36, y - 0.36, y + 0.36, y + 0.36, y - 0.36], c="k", lw=1.2)
            ax.plot([analysis.position(p, m.value)] * 2, [y - 0.42, y + 0.42], c="k", lw=2.2)
            for i, t in enumerate(targets):
                for xv in m.crossings[t]:
                    if xv is None:
                        continue
                    u = analysis.position(p, xv)
                    shown = min(max(u, xlo + 0.08), xhi - 0.08)
                    marker = markers[i % 3] if shown == u else ("<" if u < xlo else ">")
                    ax.plot([shown], [y], marker, ms=7, mec="k", mew=1,
                            c=TARGET_STYLES[i % 3][1] if 0 <= u <= 1 else "white")
                    ax.annotate(f"overhead {t:g} at {xv:.2g}", (shown, y + (0.36 if i % 2 == 0 else -0.36)),
                                fontsize=6.5,
                                ha="center", va="bottom" if i % 2 == 0 else "top")
        ax.set_yticks(range(len(ms)), [f"{_short(m.key)}\n[{m.low:.3g}, {m.high:.3g}]{_unit(registry, m.key)}"
                                       for m in reversed(ms)], fontsize=7.5)
        ax.set_ylim(-0.6, len(ms) - 0.1)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("position relative to the plausible range (0 = low end, 1 = high end; log scale for positive "
                      "ranges; ◁ ▷ beyond the view)")
        ax.set_title(f"{name}: overhead {analysis.at(registry, name, None, metric):.3g} at today's values", fontsize=10)
    handles = [Line2D([], [], marker=markers[i % 3], ls="", mec="k", c=TARGET_STYLES[i % 3][1], ms=7,
                      label=f"overhead {t:g} crossed inside the plausible range") for i, t in enumerate(targets)]
    handles += [Line2D([], [], marker=markers[i % 3], ls="", mec="k", c="white", ms=7,
                       label=f"overhead {t:g} crossed outside it") for i, t in enumerate(targets)]
    handles += [Line2D([], [], c="k", lw=1.2, label="plausible range"), Line2D([], [], c="k", lw=2.2, ls="",
                                                                               marker="|", ms=12, label="today")]
    axes[0][0].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.colorbar(mesh, ax=axes[:, 0].tolist(), label="overhead (lower bound)" if metric is LO else "overhead",
                 location="bottom", shrink=0.5)
    fig.suptitle("How far each parameter is from where the overhead crosses a target: the overhead along a window "
                 "around its plausible range", fontsize=11)
    return _save(fig, out, "vulnerability", {
        "targets": list(targets),
        "scenarios": {n: [{"key": m.key, "value": m.value, "low": m.low, "high": m.high, "window": m.window,
                           "samples": m.samples, "crossings": {str(t): c for t, c in m.crossings.items()}}
                          for m in ms] for n, ms in found.items()}})


def corner(registry: Registry, out: Path, names: Sequence[str] | None = None, targets: Sequence[float] = (10.0, 100.0),
           grid: int = 30, metric: Metric = LO) -> list[Path]:
    """Each scenario's adversarial corner (`analysis.corner`): the overhead as its ranged parameters move to their
    worse ends one at a time, the worst alone first, down to all of them at once."""
    plt = _plt()
    names = list(names or registry.names("scenario"))
    found = {n: c for n in names if (c := analysis.corner(registry, n, metric=metric)).path}
    if not found:
        return []
    widths = [2.5 + 0.42 * len(c.path) for c in found.values()]
    fig, axes = plt.subplots(1, len(found), figsize=(sum(widths), 5.6), squeeze=False, layout="constrained",
                             gridspec_kw={"width_ratios": widths})
    for ax, (name, c) in zip(axes[0], found.items(), strict=True):
        ys = [c.base] + [v for _, v in c.path]
        ax.plot(range(len(ys)), ys, "o-", c="#d7191c", mec="k", ms=5)
        for i, v in enumerate(ys):
            ax.annotate(f"{v:.3g}", (i, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(range(len(ys)), ["today"] + [f"{_short(k)} → {c.values.get(k, math.nan):.3g}" for k, _ in c.path],
                      rotation=60, ha="right", fontsize=7.5)
        _targets(ax, targets)
        ax.set_yscale("log")
        ax.set_ylabel("overhead (lower bound)" if metric is LO else "overhead")
        note = "" if c.joint == c.rho else f" (all at once: {c.joint:.3g})"
        ax.set_title(f"{name}: {c.base:.3g} today → {c.rho:.3g} at the corner{note}", fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("The adversarial corner: each ranged parameter moved to the worse end of its plausible range, "
                 "cumulatively", fontsize=11)
    return _save(fig, out, "corner", {"targets": list(targets), "scenarios": {
        n: {"base": c.base, "rho": c.rho, "joint": c.joint, "values": dict(c.values), "path": c.path}
        for n, c in found.items()}})


FIGURES = {"class-vs-k": class_vs_k, "overheads": overheads, "dark": dark, "phase": phase, "tornado": tornado,
           "vulnerability": vulnerability, "corner": corner}


def render(registry: Registry, out: Path, names: Sequence[str] | None = None, targets: Sequence[float] = (10.0, 100.0),
           grid: int = 30, only: Iterable[str] | None = None) -> dict[str, list[Path]]:
    """Every figure (or those named in `only`) into `out`, `grid` setting how densely each samples: figure name → the
    files it wrote (none: skipped, because no scenario has what it needs)."""
    return {fig: FIGURES[fig](registry, Path(out), names, targets, grid) for fig in only or FIGURES}
