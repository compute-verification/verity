"""The headline plots P1-P4 (note:training-20261102T1715Z-draft-training-efficiency-experiments) and the A2 curve, from
the ledger, as the console's five panels (note:dashboard-20261102T1934Z-draft-training-plots-contract, columns per
note:training-20261102T1938Z-draft-training-plots-contract-reply), each with a PNG drawn from its rows.

    python -m te_core.plots --out DIR [--root LEDGER] [--pin COMMIT ...] [--policy reference|inference|server|c1e14/k1e4]
                            [--drop 0.01] [--g 1] [--from-scratch F.json]

Every point is scored by the training formalism (te_core.frontier's `formalism_rows`: |C| in gates, ρ at the start's
cells, te_core.policy's `cells`) over the valid submissions at the pin, with honest H(l) the least |C| over every
checkpoint reaching l and W*(l) the least |C|ρ over the same checkpoints (the coordinator's 19:50Z rule). A frozen tail
case is always charged (formalism v2). Every slowdown value carries its `basis`: `fit` (ρ = c/(c − d) by Fit for the
training) or `target` (Slowdown's bound, stated, not proved in Lean, from A1's and A1''s upper bounds on s* and x0;
P2 and P4 carry those bounds beside the attack side's best-known values, `x0_known` and `s_star_known`, which measure
how tight they are). A panel is DIR/<name>.json,
{id, title, kind: "table", updated_at, source, note, columns, rows}, the console's panels route's body (a value it
doesn't have is null), beside DIR/<name>.png; its note and the figure's footer name the stubs and bounds its rows rest
on (`formalism.STUBS`).

- `p1-loss-vs-compute`, per start: val bits per byte against compute after θ, each curve's staircase: `honest` (H),
  `covert` (W* at the reference policy, `--policy`, default te_core.policy's REFERENCE), `covert-policy` (W* at each
  grid cell within a decade of it in c and k, a band) and `scratch` (F, `--from-scratch`).
- `p2-slowdown-vs-gain`: S at the reference policy per start and target, against the gain g = F(l*)/C0 (null without
  F, when the figure plots S against the target), with W*'s x0 and s* bounds and best-known values.
- `p3-policy-plane`: S at every cell and reference shape (`mark`), per start and target (with F and `--g`, also at that
  gain, `target_pct` null); the figure at `--g`, else at `--drop`.
- `p4-decomposition`: p_A = |C_A|/H against s_A = ρ_A at the reference policy, one point per algorithm (a direction and
  mechanism, its least-|C|ρ checkpoint), per start and target (or gain), with its σ and Slowdown terms; the figure as
  P3's.
- `a2-efficiency-floor`, per start and target: the least compute reaching it among the checkpoints with Δ ≤ x, at each
  Δ that lowers it (Δ's bins, never caps), with that point's Δ_grad and the least k at which Fit holds at the reference
  c for running θ' (the start's served forward per token) and for training (formalism's `fit_k`).

F.json is {start: [[gates, bpb], ...]}, the from-scratch runs' (compute, loss) on the same data and evaluator (the
ladder's), read as F(l) = the least compute of a point at or below l, log-interpolated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import formalism, frontier, policy  # noqa: E402
from .ledger import Ledger  # noqa: E402

LEVELS = (1, 10, 100, 1000, 10000)
NAMES = ("p1-loss-vs-compute", "p2-slowdown-vs-gain", "p3-policy-plane", "p4-decomposition", "a2-efficiency-floor")


# ---- the data ----------------------------------------------------------------------------------------------------------

def load(rows: list[dict], pin, shapes: dict[str, dict], keep=None) -> tuple[list[dict], dict]:
    """The valid rows `keep` keeps (default all) scored by the formalism, and the stubs, assumptions and bounds they
    rest on."""
    scored, _ = frontier.formalism_rows([r for r in frontier.valid(rows, pin) if keep is None or keep(r)], shapes)
    stubs: dict[str, int] = {}
    for r in scored:
        for k, v in r["measured"].labels.items():
            if not v.startswith("measured") and not v.startswith("counted"):
                stubs[f"{k}: {v}"] = stubs.get(f"{k}: {v}", 0) + 1
    return scored, stubs


def decade(cells: dict[str, tuple[float, float]], key: str) -> str:
    """The grid cell nearest to `key`'s (c, k)."""
    c, k = cells[key]
    return policy.key(10 ** round(math.log10(c)), 10 ** round(math.log10(k)))


def neighbours(key: str, keys) -> list[str]:
    c, k = policy.of_key(key)
    out = [policy.key(c * a, k * b) for a in (0.1, 1, 10) for b in (0.1, 1, 10)]
    return [x for x in out if x in keys]


def policy_label(key: str) -> str:
    if key == "reference":
        return f"c={policy.SESSION_TOKENS}w k={policy.REFERENCE_K_BITS / 8000:g}kB"
    return f"{key} block" if key in policy.MARKS else key


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


def start_label(shape: dict) -> str:
    return f"{shape.get('parameters', 0) / 1e6:.0f}M · {saturation(shape):.3g} tok/p"


def staircase(points: list[tuple[float, float, dict]]) -> list[tuple[float, float, dict]]:
    """(compute, loss, row) sorted by compute with the loss strictly falling: the least compute reaching each loss."""
    out, best = [], math.inf
    for x, y, r in sorted(points, key=lambda p: (p[0], p[1])):
        if y < best:
            out.append((x, y, r))
            best = y
    return out


def deepest(l0: float, bpb: float) -> float | None:
    hit = [f for f in frontier.TARGETS if bpb <= l0 * (1 - f)]
    return -100 * max(hit) if hit else None


def method(r: dict | None) -> str | None:
    return r and f"{r.get('direction')}/{r.get('mechanism')}"


def basis(r: dict | None, key: str) -> str | None:
    return r and ("fit" if r["fit"].get(key) else "target")


class Start:
    """One start's scored rows and what its panels read: l0, C0 (None where it is a stub), the forward gates per token
    w, its cells, the reference key and the from-scratch curve."""

    def __init__(self, name: str, scored: list[dict], shape: dict, ref: str, fs: dict):
        self.name, self.shape, self.on = name, shape, [r for r in scored if r["start"] == name]
        self.l0 = frontier.start_bpb(self.on)
        self.c0, how = policy.c0(shape)
        self.c0_counted = None if how.startswith("stub") else self.c0
        self.w = policy.shapes(shape)["inference"]["c"]
        self.cells = policy.cells(shape)
        if ref not in self.cells:
            raise SystemExit(f"plots: no policy {ref!r} for {name}: one of {', '.join(policy.MARKS)} or a grid key")
        self.ref, self.f = ref, from_scratch(fs.get(name))
        self.head = [start_label(shape), name]
        self.by_id = {r["id"]: r for r in self.on}

    def frac(self, gates: float | None) -> float | None:
        return gates / self.c0_counted if gates is not None and self.c0_counted else None

    def gain(self, l: float) -> float | None:
        return self.f(l) / self.c0 if self.f and self.f(l) else None

    def levels(self, g: float | None = None, drop: float | None = None) -> list[tuple]:
        """(target_pct, g, l*) at each of `TARGETS` (and `drop`), and with F and `g`, at the l* whose gain is g."""
        drops = frontier.TARGETS + ((drop,) if drop is not None and drop not in frontier.TARGETS else ())
        out = [(-100 * d, self.gain(self.l0 * (1 - d)), self.l0 * (1 - d)) for d in drops]
        if self.f and g is not None:
            lo = min(r["val_bpb"] for r in self.on if r.get("val_bpb") is not None)
            for l in np.linspace(self.l0, lo, 400):
                if self.f(float(l)) and self.f(float(l)) >= g * self.c0:
                    out.append((None, g, float(l)))
                    break
        return out


def starts_of(scored, shapes, ref, fs) -> list[Start]:
    out = [Start(s, scored, shapes[s], ref, fs) for s in sorted({r["start"] for r in scored})]
    return [s for s in out if s.l0 is not None]


def picked(st: Start, g: float | None, drop: float) -> tuple[float | None, float | None, float]:
    """The level P3 and P4's figures draw: the gain `g`'s with F, else `drop`'s target."""
    lv = st.levels(g, drop)
    if g is not None and lv[-1][0] is None:
        return lv[-1]
    return next(x for x in lv if x[0] == -100 * drop)


def panel(name: str, title: str, columns: list[str], rows: list[list], note: str, ctx: dict) -> dict:
    return {"id": f"training/{name}", "title": title, "kind": "table", "updated_at": ctx["stamp"],
            "source": ctx["source"], "note": note + ctx["tail"], "columns": columns, "rows": rows}


def frame(p: dict) -> list[dict]:
    return [dict(zip(p["columns"], r)) for r in p["rows"]]


# ---- the panels --------------------------------------------------------------------------------------------------------

P1_COLUMNS = ["start", "start_id", "size", "saturation", "curve", "policy", "basis", "target_pct", "compute_gates",
              "compute_frac", "loss", "loss_lo", "loss_hi", "record", "checkpoint", "method", "family"]


def p1_panel(sts: list[Start], ctx) -> dict:
    rows = []
    for st in sts:
        size, sat = st.shape.get("parameters"), saturation(st.shape)

        def emit(curve, key, pts, st=st, size=size, sat=sat):
            for x, y, r in pts:
                rows.append([*st.head, size, sat, curve, policy_label(key) if key else "",
                             basis(r, key) if key and r else None, deepest(st.l0, y), x, st.frac(x), y, None, None,
                             r and r["id"], r and r.get("checkpoint_step"), method(r), r and r.get("family")])

        band = [k for k in neighbours(decade(st.cells, st.ref), st.cells) if k != st.ref]
        emit("honest", "", staircase([(r["gates"], r["val_bpb"], r) for r in st.on]))
        for key in [st.ref, *band]:
            emit("covert" if key == st.ref else "covert-policy", key,
                 staircase([(r["gates"] * r["rho"][key], r["val_bpb"], r) for r in st.on]))
        if st.f:
            emit("scratch", "", staircase([(c, l, None) for c, l in ctx["fs"][st.name]]))
    return panel("p1-loss-vs-compute", "Loss against compute spent after the start", P1_COLUMNS, rows,
                 "Each curve is the least compute reaching each loss (its staircase) over every checkpoint of every "
                 "valid record (19:50Z): honest = |C|, covert = |C| rho at the reference policy, covert-policy = at each "
                 "grid cell (c1e<i>/k1e<j>: gates, bits) within a decade of it, scratch = the from-scratch curve F. "
                 "loss = held-out bits per byte; target_pct = the deepest target the point reaches; compute_frac null "
                 "where C0 is a stub.", ctx)


def at_reference(st: Start, l_star: float, key: str) -> tuple[dict, dict]:
    t = frontier.target(st.on, l_star, [key], size="gates")
    return t, t["cells"][key]


def row_of(st: Start, of: dict | None) -> dict | None:
    return of and st.by_id[of["id"]]


def terms(r: dict | None, key: str) -> list:
    return [r and r["terms"][key][n] for n in frontier.TERMS]


def p2_panel(sts: list[Start], ctx) -> dict:
    columns = ["start", "start_id", "size", "saturation", "l0", "target_pct", "g", "S", "S_lo", "S_hi", "basis",
               "policy", "method", "family", "record", "checkpoint", "H_gates", "W_gates", "rho", "p", *frontier.TERMS]
    rows = []
    for st in sts:
        for tp, g, l in st.levels(None):
            t, c = at_reference(st, l, st.ref)
            w = row_of(st, c["of"])
            rows.append([*st.head, st.shape.get("parameters"), saturation(st.shape), st.l0, tp, g,
                         c["effective_slowdown"], None, None, basis(w, st.ref), policy_label(st.ref), method(w),
                         w and w.get("family"), w and w["id"], w and w.get("checkpoint_step"), t["honest"],
                         c["w_star"], c["rho"], w and t["honest"] and w["gates"] / t["honest"], *terms(w, st.ref)])
    return panel("p2-slowdown-vs-gain", "Slowdown S against the gain", columns, rows,
                 "S = W*/H at the reference policy, the best known attack over every checkpoint of every valid record "
                 "(19:50Z), so S >= 1; p = |C|/H and rho of the checkpoint that sets W*, with its x0 and s_star (the "
                 "claim's bounds, A1' and A1) and x0_known and s_star_known (the attack side: best-known values from "
                 "exhibited blocks, tightness). g = F(l*)/C0, null until the from-scratch curve F: plot S against "
                 "target_pct. One seed: S_lo and S_hi null.", ctx)


def p3_panel(sts: list[Start], g, drop, ctx) -> dict:
    columns = ["start", "start_id", "target_pct", "g", "c_tokens_per_block", "c_gates", "k_bits", "k_bytes", "S",
               "basis", "method", "family", "record", "checkpoint", "rho", "p", "mark"]
    rows = []
    for st in sts:
        keys = list(st.cells)
        for tp, gv, l in st.levels(g, drop):
            t = frontier.target(st.on, l, keys, size="gates")
            for key in keys:
                c, (cg, kb) = t["cells"][key], st.cells[key]
                w = row_of(st, c["of"])
                rows.append([*st.head, tp, gv, cg / st.w, cg, kb, kb / 8, c["effective_slowdown"], basis(w, key),
                             method(w), w and w.get("family"), w and w["id"], w and w.get("checkpoint_step"), c["rho"],
                             w and t["honest"] and w["gates"] / t["honest"],
                             policy_label(key) if key in policy.MARKS else ""])
    return panel("p3-policy-plane", "Slowdown S over the policy (c, k)", columns, rows,
                 "S = W*/H at each cell of the grid (every decade of c in gates and k in bits) and at the start's "
                 "inference block, reference policy and server block (mark; te_core.policy), over every checkpoint of "
                 "every valid record. c_tokens_per_block = c/w, w the start's forward gates per token.", ctx)


def p4_panel(sts: list[Start], g, drop, ctx) -> dict:
    columns = ["start", "start_id", "target_pct", "g", "algorithm", "family", "record", "checkpoint", "p", "s", "S",
               "basis", "seeds", "delta_bits", "delta_grad_bits", "compute_gates", "sigma", *frontier.TERMS]
    rows = []
    for st in sts:
        for tp, gv, l in st.levels(g, drop):
            h = frontier.target(st.on, l, [], size="gates")["honest"]
            reach = [r for r in st.on if r.get("val_bpb") is not None and r["val_bpb"] <= l]
            best: dict[str, dict] = {}
            for r in reach:
                a = method(r)
                if a not in best or r["gates"] * r["rho"][st.ref] < best[a]["gates"] * best[a]["rho"][st.ref]:
                    best[a] = r
            for a, r in sorted(best.items()):
                s = r["rho"][st.ref]
                rows.append([*st.head, tp, gv, a, r.get("family"), r["id"], r.get("checkpoint_step"), r["gates"] / h, s,
                             r["gates"] * s / h, basis(r, st.ref), 1, r["measured"].delta_bits,
                             r["measured"].delta_grad_bits, r["gates"], r["measured"].sigma, *terms(r, st.ref)])
    return panel("p4-decomposition", "p against s per algorithm at the reference policy", columns, rows,
                 "One point per algorithm (direction/mechanism) reaching the level: its checkpoint with the least |C| rho "
                 "at the reference policy; p = |C|/H with H over every checkpoint of every valid record, s = rho "
                 "(c/(c - d) by Fit, else Slowdown's bound floored at 1), S = p s; sigma = its token blocks' share of "
                 "|C|; x0, s_star the claim's bounds, x0_known, s_star_known the attack side's. One seed per point.", ctx)


def a2_panel(sts: list[Start], ctx) -> dict:
    columns = ["start", "start_id", "size", "saturation", "target_pct", "delta_bits", "delta_grad_bits",
               "compute_gates", "compute_frac", "method", "family", "record", "checkpoint", "fit_run_k_bits",
               "fit_train_k_bits"]
    rows = []
    for st in sts:
        c_ref = st.cells[st.ref][0]
        for d in frontier.TARGETS:
            l_star = st.l0 * (1 - d)
            reach = [r for r in st.on if r["measured"].delta_bits is not None and r.get("val_bpb") is not None
                     and r["val_bpb"] <= l_star]
            for delta, gates, r in staircase([(r["measured"].delta_bits, r["gates"], r) for r in reach]):
                run_k, train_k = formalism.fit_k(r["measured"], c_ref, st.w)
                rows.append([*st.head, st.shape.get("parameters"), saturation(st.shape), -100 * d, delta,
                             r["measured"].delta_grad_bits, gates, st.frac(gates), method(r), r.get("family"), r["id"],
                             r.get("checkpoint_step"), run_k, train_k])
    return panel("a2-efficiency-floor", "Efficiency floor: least compute to each target against the update's size",
                 columns, rows,
                 "Per start and target, the least compute (Boolean gates) reaching it among every checkpoint of every "
                 "valid record whose total update Delta (Fit's quantity) is at most delta_bits, at each Delta that "
                 "lowers it; delta_grad_bits = that point's gradient and optimizer state passed between blocks. "
                 "fit_run_k_bits: the least k at which Fit holds at the reference c for running theta' (Delta + s tau, "
                 "s the served tokens per block); fit_train_k_bits: for the training (the most of its three kinds of "
                 "block: compute, reduction, update application).", ctx)


# ---- the figures -------------------------------------------------------------------------------------------------------

def footer(fig, stubs: dict[str, int], pin) -> None:
    lines = [f"pin {' + '.join(c[:12] for c in pin)}; |C| in Boolean gates (te_core.gates); rho: c/(c - d) where Fit "
             "holds, Slowdown's target where it fails; S = W*/H (Security); H and W* over every checkpoint (19:50Z)"] + \
        [f"{n} rows, {s}" for s, n in sorted(stubs.items())]
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=6, va="bottom", ha="left", family="monospace")


def save(fig, out: Path, name: str, stubs, pin, room: float) -> Path:
    footer(fig, stubs, pin)
    fig.tight_layout(rect=(0, room + 0.014 * len(stubs), 1, 1))
    path = out / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def panels_by_size(sts: list[Start]) -> dict[str, tuple[int, int]]:
    """Each start's subplot: its size's row, and its saturation's rank among that size's starts."""
    sizes = sorted({s.shape.get("parameters", 0) for s in sts})
    out = {}
    for i, size in enumerate(sizes):
        same = sorted((s for s in sts if s.shape.get("parameters", 0) == size), key=lambda s: saturation(s.shape))
        out |= {s.name: (i, j) for j, s in enumerate(same)}
    return out


def draw_p1(p: dict, sts: list[Start], out: Path, stubs, pin) -> Path:
    rows, where = frame(p), panels_by_size(sts)
    nr, nc = max(i for i, _ in where.values()) + 1, max(j for _, j in where.values()) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.8 * nr + 1.6), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            axes[i][j].set_visible((i, j) in where.values())
    for st in sts:
        ax = axes[where[st.name][0]][where[st.name][1]]
        mine = [r for r in rows if r["start_id"] == st.name]

        def curve(sel, **kw):
            pts = [(r["compute_gates"], r["loss"]) for r in mine if sel(r)]
            if pts:
                ax.step(*zip(*pts, (pts[-1][0] * 10, pts[-1][1])), where="post", **kw)

        for pol in sorted({r["policy"] for r in mine if r["curve"] == "covert-policy"}):
            curve(lambda r, pol=pol: r["policy"] == pol, lw=0.6, alpha=0.2, color="C3")
        curve(lambda r: r["curve"] == "covert", lw=1.4, color="C3", label=f"W* at {policy_label(st.ref)}")
        curve(lambda r: r["curve"] == "honest", color="k", lw=1.6, label="honest H")
        curve(lambda r: r["curve"] == "scratch", color="C0", ls=":", label="from scratch F")
        for d in frontier.TARGETS:
            ax.axhline(st.l0 * (1 - d), color="0.85", lw=0.5, ls=":")
        ax.axhline(st.l0, color="grey", lw=0.5)
        ax.set_xscale("log")
        ax.set_title(f"{st.name} ({start_label(st.shape)}), l0 {st.l0:.4f}; C0 {st.c0:.3g} gates"
                     + ("" if st.c0_counted else " (stub)"), fontsize=9)
        ax.set_xlabel("compute after theta (Boolean gates)")
        ax.set_ylabel("val bits per byte")
        ax.legend(fontsize=6)
    fig.suptitle("P1: loss against compute after theta (light: W* at the grid cells within a decade of the policy)",
                 fontsize=9)
    return save(fig, out, p["id"].split("/")[1], stubs, pin, 0.06)


def draw_p2(p: dict, sts: list[Start], out: Path, stubs, pin) -> Path:
    rows = frame(p)
    have_g = all(r["g"] is not None for r in rows)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    cmap, l0s = plt.get_cmap("viridis"), [st.l0 for st in sts]
    lo, hi = min(l0s), max(l0s)
    sat_rank = {v: i for i, v in enumerate(sorted({saturation(st.shape) for st in sts}))}
    for st in sts:
        pts = [(r["g"] if have_g else r["target_pct"], r["S"]) for r in rows
               if r["start_id"] == st.name and r["S"] is not None]
        if pts:
            ax.plot(*zip(*pts), ["-", "--", ":", "-."][sat_rank[saturation(st.shape)] % 4], marker="o",
                    color=cmap((st.l0 - lo) / (hi - lo) if hi > lo else 0.5),
                    label=f"{st.name} ({start_label(st.shape)}, l0 {st.l0:.3f})")
    ax.set_yscale("log")
    if have_g:
        ax.set_xscale("log")
    ax.set_xlabel("gain g = F(l*)/C0" if have_g else "target, % of l0 (no from-scratch curve F yet)")
    ax.set_ylabel(f"S = W*/H at {policy_label(sts[0].ref)}")
    ax.legend(fontsize=7)
    fig.suptitle("P2: Security's S = W*/H against the gain (colour: l0, style: saturation)", fontsize=9)
    return save(fig, out, p["id"].split("/")[1], stubs, pin, 0.08)


def draw_p3(p: dict, sts: list[Start], g, drop, out: Path, stubs, pin) -> Path:
    rows = frame(p)
    fig, axes = plt.subplots(1, len(sts), figsize=(4.6 * len(sts), 4.4 + 1.4), squeeze=False)
    for ax, st in zip(axes[0], sts):
        tp, gv, l = picked(st, g, drop)
        mine = [r for r in rows if r["start_id"] == st.name and r["target_pct"] == tp and r["g"] == gv]
        grid = [r for r in mine if not r["mark"]]
        cs, ks = sorted({r["c_gates"] for r in grid}), sorted({r["k_bits"] for r in grid})
        z = np.full((len(ks), len(cs)), np.nan)
        for r in grid:
            if r["S"] is not None:
                z[ks.index(r["k_bits"]), cs.index(r["c_gates"])] = math.log10(r["S"])
        if np.isnan(z).all():
            ax.text(0.5, 0.5, f"no record reaches l* = {l:.4f}", ha="center", transform=ax.transAxes)
        else:
            lc, lk = np.log10(cs), np.log10(ks)
            top = max(math.log10(LEVELS[-1]), float(np.nanmax(z))) + 1e-9
            levels = [math.log10(v) for v in LEVELS] + [top]
            cf = ax.contourf(lc, lk, z, levels=levels, cmap="magma_r", extend="neither")
            ax.contour(lc, lk, z, levels=levels[1:-1], colors="k", linewidths=0.5)
            fig.colorbar(cf, ax=ax, label="log10 S")
        for r in mine:
            if r["mark"]:
                x, y = math.log10(r["c_gates"]), math.log10(r["k_bits"])
                ax.plot(x, y, {"inference block": "c^", "server block": "cs"}.get(r["mark"], "co"), mec="k", ms=8)
                s = "—" if r["S"] is None else f"{r['S']:.3g}"
                ax.annotate(f"{r['mark']}: S {s}", (x, y), fontsize=6, xytext=(4, 4), textcoords="offset points")
        what = f"g = {gv:g}" if tp is None else f"{tp:g}%"
        ax.set_xlabel("log10 c (gates per block)")
        ax.set_ylabel("log10 k (bits per block beyond theta)")
        ax.set_title(f"{st.name} at {what} (l* {l:.4f})", fontsize=8)
    fig.suptitle("P3: S over the policy plane", fontsize=9)
    return save(fig, out, p["id"].split("/")[1], stubs, pin, 0.08)


def draw_p4(p: dict, sts: list[Start], g, drop, out: Path, stubs, pin) -> Path:
    rows = frame(p)
    fig, axes = plt.subplots(1, len(sts), figsize=(4.6 * len(sts), 4.4 + 1.4), squeeze=False)
    for ax, st in zip(axes[0], sts):
        tp, gv, l = picked(st, g, drop)
        pts = [r for r in rows if r["start_id"] == st.name and r["target_pct"] == tp and r["g"] == gv]
        if not pts:
            ax.text(0.5, 0.5, f"no record reaches l* = {l:.4f}", ha="center", transform=ax.transAxes)
        else:
            ps, ss = [r["p"] for r in pts], [r["s"] for r in pts]
            lo, hi = min(min(ps), min(ss), 1) / 2, max(max(ps), max(ss)) * 2
            xs = np.logspace(math.log10(lo), math.log10(hi), 50)
            for e in range(0, 2 * math.ceil(math.log10(hi)) + 1, 2):
                ax.plot(xs, 10 ** e / xs, color="grey", lw=0.4)
            best = min(pts, key=lambda r: r["S"])
            for r in pts:
                full = r["algorithm"].split("/")[0] in frontier.REF_DIRECTIONS
                ax.plot(r["p"], r["s"], "*" if full else "o", color="C3" if full else "C0", ms=10 if full else 5,
                        mec="red" if r["basis"] == "fit" else None)
                ax.annotate(r["algorithm"], (r["p"], r["s"]), fontsize=5, xytext=(3, 2), textcoords="offset points")
            ax.plot(best["p"], best["s"], "o", mfc="none", mec="red", ms=14)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("p_A = |C_A| / H")
        ax.set_ylabel(f"s_A = rho at {policy_label(st.ref)}")
        what = f"g = {gv:g}" if tp is None else f"{tp:g}%"
        ax.set_title(f"{st.name} at {what} (l* {l:.4f}); star: full training, ring: sets W*", fontsize=7)
    fig.suptitle("P4: p against s per algorithm (grey: iso-S, S = p s)", fontsize=9)
    return save(fig, out, p["id"].split("/")[1], stubs, pin, 0.08)


def draw_a2(p: dict, sts: list[Start], out: Path, stubs, pin) -> Path:
    rows = frame(p)
    fig, axes = plt.subplots(1, len(sts), figsize=(4.6 * len(sts), 4.6 + 1.4), squeeze=False)
    for ax, st in zip(axes[0], sts):
        mine = [r for r in rows if r["start_id"] == st.name]
        x_end = max((r["delta_bits"] for r in mine), default=1.0) * 3
        for i, d in enumerate(frontier.TARGETS):
            pts = [(r["delta_bits"], r["compute_gates"]) for r in mine if r["target_pct"] == -100 * d]
            if pts:
                xs, ys = zip(*pts)
                ax.step([*xs, x_end], [*ys, ys[-1]], where="post", color=f"C{i}",
                        label=f"-{100 * d:g}% (l* {st.l0 * (1 - d):.4f})")
                ax.plot(xs, ys, "o", color=f"C{i}", ms=3)
        k_ref = st.cells[st.ref][1]
        ax.axvline(k_ref, color="grey", ls="--", lw=0.8, label=f"k at {policy_label(st.ref)}: Fit needs Δ + sτ ≤ k")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Δ, the update's bits (checkpoints with Δ ≤ x)")
        ax.set_ylabel("least compute after theta (gates)")
        ax.set_title(f"{st.name} ({start_label(st.shape)}, l0 {st.l0:.4f})", fontsize=9)
        ax.legend(fontsize=6)
    fig.suptitle("A2: the least compute reaching each target with an update of at most Δ bits, over every checkpoint",
                 fontsize=9)
    return save(fig, out, p["id"].split("/")[1], stubs, pin, 0.1)


def write(scored, shapes, stubs, pin, out: Path, ref: str = policy.REFERENCE, fs: dict | None = None,
          g: float | None = None, drop: float = 0.01, source: str = "", stamp: str | None = None) -> list[Path]:
    """The five panels into `out`, each DIR/<name>.json (the console's panel) and DIR/<name>.png. `source` names the run
    or command; `stamp` is `updated_at` (default now, UTC)."""
    fs = fs or {}
    out.mkdir(parents=True, exist_ok=True)
    sts = starts_of(scored, shapes, ref, fs)
    tail = (" Formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2): a frozen tail is always charged. "
            "basis: fit = rho c/(c - d) by Fit for the training; target = Slowdown's bound (stated, not proved in "
            "Lean) from A1's and A1''s upper bounds. Pin " + " + ".join(c[:12] for c in pin) + ". Stubs, assumptions "
            "and bounds: " + "; ".join(f"{s} ({n} rows)" for s, n in sorted(stubs.items())) + ".")
    ctx = {"stamp": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source, "tail": tail,
           "fs": fs}
    made = {"p1-loss-vs-compute": (p1_panel(sts, ctx), lambda p: draw_p1(p, sts, out, stubs, pin)),
            "p2-slowdown-vs-gain": (p2_panel(sts, ctx), lambda p: draw_p2(p, sts, out, stubs, pin)),
            "p3-policy-plane": (p3_panel(sts, g, drop, ctx), lambda p: draw_p3(p, sts, g, drop, out, stubs, pin)),
            "p4-decomposition": (p4_panel(sts, g, drop, ctx), lambda p: draw_p4(p, sts, g, drop, out, stubs, pin)),
            "a2-efficiency-floor": (a2_panel(sts, ctx), lambda p: draw_a2(p, sts, out, stubs, pin))}
    written = []
    for name in NAMES:
        p, draw = made[name]
        (out / f"{name}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
        written.append(draw(p))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None)
    ap.add_argument("--policy", default=policy.REFERENCE, help="a reference shape (te_core.policy.MARKS) or a grid key")
    ap.add_argument("--drop", type=float, default=0.01, help="P3 and P4's figures' target, as a drop from l0")
    ap.add_argument("--g", type=float, default=None, help="P3 and P4's gain F(l*)/C0 (needs --from-scratch)")
    ap.add_argument("--from-scratch", type=Path, default=None, help="JSON {start: [[gates, bpb], ...]}")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    a = ap.parse_args(argv)
    shapes = policy.starts(a.manifest)
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    scored, stubs = load(Ledger(a.root).index(), pin, shapes)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    for p in write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop, source="python -m te_core.plots"):
        print(f"{p}, {p.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
