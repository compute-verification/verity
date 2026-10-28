"""Figures for the slowdown D, the hardware multiple at MFU_0 (``statement.dampener``: the slowdown calculator's
``rho.theorem.multiple_bound`` on ``tiers.evaluate``'s terms, the residual at MFU_0), with the
ladder, its scenarios and its numbers from ``statement.py``: D per rung by status, the (r̂, R_wipe) phase diagram,
exposure tornados at the best PoUW design and at 6 GB streamed-label wipes, the memory cap by the fit bound and the
closed form, the cheating share against proving cost, years of delay, and the leaks and their plugs. Generated output:
it goes to the evidence store, never Git.

    python3 benchmarks/nci/figures.py OUTDIR    # NAME.png (1600x900) and NAME.svg per figure, and figures.json
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exposure as X  # noqa: E402
import fit as F  # noqa: E402
import statement as S  # noqa: E402
import tiers as T  # noqa: E402

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 100, "font.size": 22, "axes.labelsize": 24,
    "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 20, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4, "lines.linewidth": 3.5,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 4, "ytick.minor.size": 4, "mathtext.fontset": "dejavusans", "svg.hashsalt": "verity-figures",
})

#: one colour per leak of 1/D <= s + (min(r̂ + eps' + nu, mu) + chi) / MFU_0 (Okabe-Ito); the residual's two plugs in
#: two blues
TERM = {"unmonitored": "#CC79A7", "residual": "#0072B2", "memory": "#56B4E9", "declared": "#009E73",
        "cheat": "#D55E00"}
#: how far each rung is from real, as the statement page's ladder says
STATUS = {"deployed": "#1a1a1a", "specified": "#8c8c8c", "modelled": "#E69F00", "conditional": "#882255",
          "policy choice": "#332288"}
GRAY = "#6b6b6b"
#: the ladder's rungs: ``statement.RUNGS``' name, a short label, and the statement page's status
LADDER = [
    ("Today", "Today", "deployed"),
    ("PoUW's floor credit", "PoUW's\nfloor\ncredit", "specified"),
    ("Floor credit and attested clock", "+ attested\nclock", "specified"),
    ("Best plausible PoUW design", "Best\nPoUW\ndesign", "modelled"),
    ("Streamed-label wipes, 13 GB", "Streamed-\nlabel wipes,\n13 GB", "conditional"),
    ("Streamed-label wipes, 6 GB", "Streamed-\nlabel wipes,\n6 GB", "conditional"),
    ("Power removal, 1 GB", "Power\nremoval,\n1 GB", "conditional"),
    ("Floor (r̂ = 0, 4 GB)", "Floor,\none-hour\nlag", "policy choice"),
    ("Floor, one-minute verdict lag", "Floor,\none-minute\nlag", "policy choice"),
]
#: the rungs whose headline is the PoUW route, r̂ to 0 with no memory cap (``statement.ladder``'s closed-form D, which
#: caps nothing there); the fit bound's cap at their 4 GB is sampled beside it
FLOORS = ("Floor (r̂ = 0, 4 GB)", "Floor, one-minute verdict lag")
#: the Dot16 conjecture's (d, e), ``tiers.REFERENCE_THREADED``: conditional, as a run that trains its embedding reads
#: Layered's lookup columns as updated weights
CONJECTURE = {"d": T.REFERENCE_THREADED[0], "e": T.REFERENCE_THREADED[1]}


def one_line(label: str) -> str:
    return label.replace("-\n", "-").replace("\n", " ")


def words(label: str) -> str:
    """A mathtext label as plain text, for captions."""
    for tex, char in ((r"\hat r", "r̂"), (r"\alpha", "α"), (r"\Gamma", "Γ"), (r"\sigma", "σ"), (r"\kappa", "κ"),
                      (r"\tau", "τ"), (r"\delta", "δ"), ("$_0$", "_0"), ("N^*", "N*"), ("$", "")):
        label = label.replace(tex, char)
    return label


#: the routed-mixture dossier's gain on N*: defensible, and adversarial
ROUTED = 2.0
ROUTED_HIGH = 8.0
P = T.DEFAULTS
ROUNDS = S.num(S.STREAMED["pous_rounds"])
LAW, TARGET = S.FIT_TARGET
N_STAR = F.n_star(LAW, F.target_loss(LAW, TARGET))
W_E = F.BOUNDARY["central"]
#: the wipe residual, GB, up to which the fit bound holds: a one-period window holds fewer than N* weights
EDGE_GB = F.bite_gb(P, N_STAR, P["b_w"])
SHA, DIRTY = S.commit()
COMMIT = f"{SHA[:10]}{' with uncommitted changes' if DIRTY else ''}"
num = S.num
BASE = (f"D is the hardware multiple at MFU_0 = {num(P['mfu'])} (rho.theorem.multiple_bound on tiers.evaluate's terms, "
        f"the residual training at MFU_0 under the reference-efficiency assumption), at m = {num(P['m'])}, "
        f"kappa = {num(P['kappa'])}, the structural entry fee d = {num(P['d'])} (a run that trains its embedding reads "
        "Layered's lookup columns as updated weights; the Dot16 conjecture's (d, e) = "
        f"({num(CONJECTURE['d'])}, {num(CONJECTURE['e'])}) is conditional), e = {num(P['e'])}, a one-hour verdict lag "
        f"and r = r̂ + eps_W W_cert/C + eps' (tiers.residual_share, the residual DampenerRemote bounds) with PoUW's "
        f"deficit eps_W = {num(P['eps_w'])} and the aggregation slack eps' = {num(P['eps_agg'])} (r̂ = 1 - W_cert/C is "
        "tiers.audited_residual); the "
        f"memory rungs by the fit bound at central w_e = {num(W_E)} bits for a {TARGET.upper()}-class target "
        f"(N* = {num(N_STAR)}) and the floors by the PoUW route, as statement.py gives them. Rendered by "
        f"benchmarks/nci/figures.py at commit {COMMIT}.")


def plain(v: float, _=None) -> str:
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def sci(x: float, digits: int = 3) -> str:
    """``x`` as ``statement.num`` writes it, with a power of ten in mathtext."""
    s = num(x, digits)
    if "e" not in s:
        return s
    m, e = s.split("e")
    return f"${'' if m == '1' else m + r'\times'}10^{{{int(e)}}}$"


def title(fig, text: str) -> None:
    fig.text(0.015, 0.975, text, fontsize=28, va="top")


def save(fig, out: Path, name: str) -> list[str]:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    return [f"{name}.png", f"{name}.svg"]


def fit_mu(p: dict, N: float = N_STAR, w_e: float = W_E, b: float | None = None) -> float:
    """The fit bound's cap on monitored efficiency at ``p``'s wipe residual (``statement.fit_mu`` with ``p``'s intake,
    period and kappa, and any N*, w_e or stored bits per weight); ``inf`` where it doesn't hold."""
    return F.fit_mu(p, X.wipe_residual(p), N, w_e, b)


def parts(p: dict, tier: str, mu: float = math.inf) -> tuple[float, dict[str, float]]:
    """D at ``tier`` with ``mu`` as the memory cap, as ``statement.capped``, and the terms of 1/D: the monitored one as
    r = r̂ + eps' and nu, or as the cap."""
    r = T.evaluate(p, T.TIERS[tier] | {"wipes": False, "exhaust": False}, p["mfu"])
    free = r["residual"] + r["declared"]
    cap = p["m"] * p["g"] * (1 - min(p["unmonitored"], 0.999)) * mu
    k = p["g"] * p["mfu"]
    terms = {"unmonitored": r["unmonitored"] / p["g"], "residual": r["residual"] / k if free <= cap else 0.0,
             "memory": cap / k if free > cap else 0.0, "declared": r["declared"] / k if free <= cap else 0.0,
             "cheat": r["cheat"] / k}
    return S.capped(p, tier, mu), terms


def fit_parts(p: dict, tier: str, **kw: float) -> tuple[float, dict[str, float]]:
    """D at ``tier`` by the fit bound from T3 on, no memory cap where it doesn't hold (``statement.fit_dampener``)."""
    return parts(p, tier, fit_mu(p, **kw) if T.TIERS[tier]["wipes"] else math.inf)


def closed_mu(p: dict) -> float:
    """The closed-form memory cap at T3 as ``tiers.evaluate`` takes it: the windows bound at ``p``'s wipe residual."""
    H = p["years"] * T.YEAR
    per_server = T.windows_bound(p, H, X.wipe_residual(p), T.ingress(p, T.TIERS["T3"]), p["wipe_s"])
    return 2 / p["kappa"] * per_server / (p["node_flops"] * H)


def at_wipe(gb: float, **over: float) -> dict:
    """The defaults with ``over`` and streamed-label wipes that leave ``gb`` GB (``statement.streamed``)."""
    p = P | over | S.streamed(gb)
    assert abs(X.wipe_residual(p) / T.GB - gb) <= 1e-9 * gb
    return p


def rung_parts(name: str, p: dict, tier: str) -> tuple[float, dict[str, float]]:
    """D at ``tier`` by the rung's route: the PoUW route for ``FLOORS``, the fit bound for the rest."""
    return parts(p, tier) if name in FLOORS else fit_parts(p, tier)


@cache
def rungs() -> list[dict]:
    """Each rung of ``LADDER``: ``statement.ladder``'s row, its overrides, label and status, D at T3 and T5 by its
    route (``D_rung``), under the routed-mixture gain where its memory cap is the fit bound's, and the floor's under the
    Dot16 conjecture."""
    rows = {r["rung"]: r for r in S.ladder()}
    out = []
    for name, label, status in LADDER:
        row = rows[name]
        p = P | row["overrides"]
        D = {t: rung_parts(name, p, t)[0] for t in S.FIT_TIERS}
        for t in S.FIT_TIERS:
            assert math.isclose(fit_parts(p, t)[0], row["D_fit"][t], rel_tol=1e-4), (name, t)
            assert math.isclose(D[t], row["D" if name in FLOORS else "D_fit"][t], rel_tol=1e-4), (name, t)
        routed = None
        if name.startswith("Streamed-label"):
            routed = {t: fit_parts(p, t, N=N_STAR / ROUTED)[0] for t in S.FIT_TIERS}
            routed["void"] = math.isinf(fit_mu(p, N=N_STAR / ROUTED))
        conj = None
        if name == FLOORS[0]:
            conj = {t: parts(p | CONJECTURE, t)[0] for t in S.FIT_TIERS}
            assert all(math.isclose(conj[t], S.dampener(p | CONJECTURE, t), rel_tol=1e-4) for t in S.FIT_TIERS)
        out.append(row | {"name": name, "label": label, "status": status, "p": p, "routed": routed,
                          "D_rung": D, "years_rung": S.years(D), "conjecture": conj})
    return out


def ladder(out: Path) -> dict:
    """Figure 1: D per rung at T3 and T5, log y, coloured by status; the memory rungs by the fit bound, with the x2
    routed-mixture case, and the floors by the PoUW route, with the fit bound's cap at their 4 GB and the floor under
    the Dot16 conjecture."""
    rows = rungs()
    floors = [x for x, r in enumerate(rows) if r["name"] in FLOORS]
    fits = [rows[x]["D_fit"]["T3"] for x in floors]
    assert floors == [len(rows) - 2, len(rows) - 1] and math.isclose(*fits), (
        "the floors' fit-bound diamonds share one label between the last two rungs: label each if they differ")
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.17, top=0.8)
    conj = None
    for x, r in enumerate(rows):
        c = STATUS[r["status"]]
        d3, d5 = r["D_rung"]["T3"], r["D_rung"]["T5"]
        ax.plot([x, x], [d5, d3], color=c, lw=2.5, zorder=2)
        ax.plot(x, d3, "o", ms=22, color=c, mec="white", mew=2, zorder=4)
        ax.plot(x, d5, "o", ms=17, mfc="white", mec=c, mew=3, zorder=3)
        sx, ha = (0.17, "left") if x == floors[-1] else (-0.17, "right")
        ax.text(x + sx, d3, num(d3), ha=ha, va="center", fontsize=20, color=c, fontweight="bold")
        apart = abs(math.log10(d3 / d5))
        if apart > 0.25:
            ax.text(x + sx, d5, num(d5), ha=ha, va="center", fontsize=17, color=c)
        elif apart > 0.02:
            ax.text(x + 0.17, d5, num(d5), ha="left", va="center", fontsize=17, color=c)
        if r["routed"]:
            q3 = 1.0 if r["routed"]["void"] else r["routed"]["T3"]
            ax.plot(x + 0.25, q3, "^", ms=16, mfc="white", mec=c, mew=2.5, zorder=4)
            ax.text(x + 0.4, q3, "×2: void" if r["routed"]["void"] else f"×2: {num(q3)}", ha="left", va="center",
                    fontsize=16, color=c)
        if x in floors:
            ax.plot(x + (0.25 if x == floors[0] else -0.25), r["D_fit"]["T3"], "D", ms=13, mfc="white", mec=c,
                    mew=2.5, zorder=4)
        if r["conjecture"]:
            conj, cc = r["conjecture"]["T3"], STATUS["conditional"]
            ax.plot([x, x], [d3, conj], color=cc, lw=1.8, ls=(0, (2, 2)), zorder=1)
            ax.plot(x, conj, "o", ms=18, color=cc, mec="white", mew=2, zorder=4)
            ax.text(x + 0.17, conj, f"d = {num(CONJECTURE['d'])}: {num(conj, 4)}", ha="left", va="center",
                    fontsize=16, color=cc)
    ax.text(sum(floors) / 2, fits[0], num(fits[0]), ha="center", va="center", fontsize=16,
            color=STATUS[rows[floors[0]]["status"]])
    ax.set_yscale("log")
    ax.set_ylim(0.7, 10 ** math.ceil(math.log10(max(conj, *(r["D_rung"]["T3"] for r in rows))) + 0.3))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows], fontsize=17)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"compute dampener $\mathcal{D}$")
    ax.grid(axis="y", color="#e4e4e4", lw=1.2, zorder=0)
    marks = [plt.Line2D([], [], ls="", marker="o", ms=17, color=GRAY),
             plt.Line2D([], [], ls="", marker="o", ms=14, mfc="white", mec=GRAY, mew=3),
             plt.Line2D([], [], ls="", marker="^", ms=14, mfc="white", mec=GRAY, mew=2.5),
             plt.Line2D([], [], ls="", marker="D", ms=11, mfc="white", mec=GRAY, mew=2.5)]
    ax.add_artist(ax.legend(marks, ["T3, all blocks proved", "T5, 1% sampled, today's cost",
                                    r"×2 routed ($N^*/2$)", "fit bound at 4 GB"], loc="lower left",
                            bbox_to_anchor=(-0.01, 1.0), ncol=4, fontsize=17, handletextpad=0.2, columnspacing=1.2))
    status = [plt.Line2D([], [], ls="", marker="o", ms=17, color=c) for c in STATUS.values()]
    ax.legend(status, list(STATUS), loc="lower left", bbox_to_anchor=(-0.01, 1.07), ncol=5, fontsize=19,
              handletextpad=0.2, columnspacing=1.4)
    title(fig, r"The ladder: $\mathcal{D}$ per rung")
    quote = {r["name"]: {"status": r["status"], "r_hat": r["r_hat"], "R_wipe_gb": r["R_wipe_gb"],
                         "D_T3": r["D_rung"]["T3"], "D_T5": r["D_rung"]["T5"],
                         **({"routed_x2": r["routed"]} if r["routed"] else {}),
                         **({"fit_bound": r["D_fit"]} if r["name"] in FLOORS else {}),
                         **({"conjecture": CONJECTURE | r["conjecture"]} if r["conjecture"] else {})} for r in rows}
    floor = rows[floors[0]]
    mu4 = fit_mu(floor["p"])
    caption = (
        "The compute dampener D at each rung of the ladder (log y): filled at T3 (NCI, PoUW, network accounting and "
        "wipes, every proof block proved), hollow at T5 (1% of proof blocks sampled at today's proving cost, "
        f"{num(P['overhead'])}). "
        "Colour is how far the rung is from real: deployed, specified, modelled, conditional (on unproved links: F3, "
        "F4, NET-35 and WipeFullUse for the memory rungs) or a policy choice. The memory rungs are by the fit bound, "
        "the floors by the PoUW route (r̂ to 0, no memory cap; the closed form caps nothing at d = "
        f"{num(P['d'])}). D at T3 (T5): "
        + "; ".join(f"{r['name']} {num(r['D_rung']['T3'])} ({num(r['D_rung']['T5'])}), {r['status']}" for r in rows)
        + f". Triangles: the streamed-label rungs under routed mixtures at the defensible gain x{ROUTED:g} on N* "
        "(N*/2): "
        + "; ".join(f"{r['name'].split(', ')[1]} "
                    + ("void (one window holds N*/2 weights; no cap, D = 1)" if r["routed"]["void"] else
                       f"{num(r['routed']['T3'])} ({num(r['routed']['T5'])})") for r in rows if r["routed"])
        + f". Diamonds: both floors by the fit bound, whose cap at 4 GB (mu = {num(mu4)}) is below what the floor "
        f"leaves (eps', the verdict lag and nu), so the lag doesn't matter there: {num(fits[0])} "
        f"({num(floor['D_fit']['T5'])}). The dotted line rises to the floor under the Dot16 conjecture's (d, e) = "
        f"({num(CONJECTURE['d'])}, {num(CONJECTURE['e'])}), conditional: {num(floor['conjecture']['T3'], 4)} "
        f"({num(floor['conjecture']['T5'])}). " + BASE)
    return {"files": save(fig, out, "1_ladder"), "title": "The ladder", "caption": caption, "quote": quote}


def _phase_cell(rR: tuple[float, float]) -> float:
    r, R = rR
    p = at_wipe(R, residual=r, host_share=0.0, clock_share=0.0)
    mu = fit_mu(p)
    return parts(p, "T4", mu)[0] if math.isfinite(mu) else S.dampener(p, "T4")


def phase(out: Path) -> dict:
    """Figure 2: D at T4 over r̂ and R_wipe, by the fit bound below its edge and the closed form above it."""
    rs = np.geomspace(1e-3, 1.0, 91)
    lo = np.geomspace(0.5, EDGE_GB * (1 - 1e-9), 80)
    hi = np.geomspace(EDGE_GB * (1 + 1e-9), 200.0, 26)
    with ProcessPoolExecutor() as ex:
        cells = list(ex.map(_phase_cell, [(r, R) for R in (*lo, *hi) for r in rs], chunksize=64))
    D = np.log10(np.array(cells).reshape(len(lo) + len(hi), len(rs)))
    D_lo, D_hi = D[:len(lo)], D[len(lo):]
    p0 = at_wipe(6.0, residual=0.0, host_share=0.0, clock_share=0.0)
    free = {x: sum(v for k, v in parts(p0 | {"residual": x}, "T4")[1].items() if k in ("residual", "declared"))
            * P["mfu"] - P["mfu"] * x for x in (0.01, 0.1)}
    assert math.isclose(free[0.01], free[0.1], rel_tol=1e-6)
    offset = free[0.01]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.93, bottom=0.13, top=0.87)
    levels = np.arange(0, 4.01, 0.5)
    cmap = ListedColormap(plt.cm.Blues(np.linspace(0.02, 0.6, len(levels) - 1)))
    for Rs, Z in ((lo, D_lo), (hi, D_hi)):
        cf = ax.contourf(rs, Rs, Z, levels=levels, cmap=cmap)
        cs = ax.contour(rs, Rs, Z, levels=[1, 2, 3], colors="k", linewidths=2.6)
        ax.clabel(cs, fmt={1: "10", 2: "100", 3: "1000"}, fontsize=19, inline_spacing=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(rs[0], rs[-1])
    ax.set_ylim(0.5, 200)
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.035, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["1", "10", "100", "1000", r"$10^4$"])
    cb.set_label(r"$\mathcal{D}$", rotation=0, labelpad=16)
    cb.outline.set_visible(False)
    edge = np.geomspace(0.5, EDGE_GB * (1 - 1e-6), 300)
    x_edge = np.array([(fit_mu(at_wipe(R)) - offset) / P["mfu"] for R in edge])
    keep = x_edge >= rs[0]
    ax.plot(x_edge[keep], edge[keep], color="white", lw=2.6, ls=(0, (5, 3)), zorder=2)
    edge_hi = np.geomspace(EDGE_GB * (1 + 1e-6), 200, 100)
    ax.plot([(closed_mu(at_wipe(R)) - offset) / P["mfu"] for R in edge_hi], edge_hi, color="white", lw=2.6, ls=(0, (5, 3)),
            zorder=2)
    ax.axhline(EDGE_GB, color="k", lw=2.2, ls=(0, (2, 2)), zorder=3)
    ax.text(1.12e-3, EDGE_GB * 1.06, f"{EDGE_GB:.0f} GB: closed form above (needs window-scale threading)",
            fontsize=18, va="bottom", zorder=6)
    ax.text(1.12e-3, EDGE_GB / 1.06, f"fit bound below ({TARGET.upper()}-class target)", fontsize=18, va="top",
            zorder=6)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.set_xlabel(r"monitored compute residual $\hat r$")
    ax.set_ylabel(r"wipe residual $R_{\rm wipe}$ (GB per server)")
    pts = {}
    for r in rungs():
        if r["name"] in ("Floor, one-minute verdict lag",):
            continue
        d = r["D"]["T4"] if r["R_wipe_gb"] > EDGE_GB else fit_parts(r["p"], "T4")[0]
        if r["r_hat"] > 0:
            assert math.isclose(_phase_cell((r["r_hat"], r["R_wipe_gb"])), d, rel_tol=1e-6), r["name"]
        pts[r["name"]] = (r["r_hat"], r["R_wipe_gb"], d, STATUS[r["status"]], one_line(r["label"]))
    place = {"Today": (-14, -20), "PoUW's floor credit": (-14, -52), "Floor credit and attested clock":
             (-14, -84), "Best plausible PoUW design": (16, -16), "Streamed-label wipes, 13 GB": (-16, 0),
             "Streamed-label wipes, 6 GB": (-16, 0), "Power removal, 1 GB": (-16, 0), "Floor (r̂ = 0, 4 GB)": (16, 0)}
    for name, (x, y, d, c, label) in pts.items():
        dx, dy = place[name]
        if x == 0:
            ax.plot(rs[0], y, "<", ms=20, color=c, mec="white", mew=2, zorder=5, clip_on=False)
            text = f"Floor, $\\hat r \\to 0$: {num(d)} (fit bound)"
            x = rs[0]
        else:
            ax.plot(x, y, "o", ms=19, color=c, mec="white", mew=2.5, zorder=5, clip_on=False)
            text = f"{label}: {num(d)}"
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right",
                    va="top" if dy < 0 else "center", fontsize=17, zorder=6,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
                    arrowprops={"arrowstyle": "-", "color": c, "lw": 1.3} if dy < -30 else None)
    title(fig, r"Memory rescues a bad residual: $\mathcal{D}$ at T4")
    corner = 10 ** D_lo[0, 0]
    route = parts(next(r["p"] for r in rungs() if r["name"] == FLOORS[0]), "T4")[0]
    caption = (
        "The compute dampener D at T4 (colour in half-decades; black contours at D = 10, 100 and 1000) over the "
        "monitored compute residual r̂ (x, log, 0.001 to 1; the GPUs' residual, host CPUs and clock at 0) and the "
        f"wipe residual R_wipe per server (y, log, 0.5 to 200 GB), set by streamed-label wipes ({ROUNDS} rounds, no "
        "certificate floor) with the memory the fill doesn't reach as the rest. Below the dashed black line at "
        f"{EDGE_GB:.1f} GB the memory cap is the fit bound at central w_e ({TARGET.upper()}-class target), where it "
        "holds; above it the closed form, which needs (d, e)-threading at window scale (false today), so read the top "
        "band as the closed form's claim. Just under the edge a one-period window nearly holds N* weights and the fit "
        "bound weakens, so D dips there. The dashed white line is where the memory cap equals r̂ plus eps' and the "
        f"verdict lag at MFU_0, plus nu ({num(offset)} at r̂ = 0): left of it D is set by r̂ (vertical contours), right of it by "
        "memory (horizontal). "
        "Markers are the ladder's rungs: "
        + "; ".join(f"{name} (r̂ = {num(x)}, R_wipe = {num(y)} GB): D = {num(d)}" for name, (x, y, d, _, _) in
                    pts.items())
        + f". The floor's marker is by the fit bound, as everything below the edge is; by the ladder's PoUW route, "
        f"with no memory cap, it is {num(route)}. At r̂ = 0.001 and 0.5 GB, D = {num(corner)}. " + BASE)
    return {"files": save(fig, out, "2_phase"), "title": "The phase diagram", "caption": caption,
            "quote": {"edge_gb": EDGE_GB, "corner_D": corner, "offset": offset, "floor_pouw_route_T4": route,
                      "rungs": {k: {"r_hat": x, "R_wipe_gb": y, "D_T4": d} for k, (x, y, d, _, _) in pts.items()}}}


#: short labels and units for ``exposure.RANGES``' keys
LABELS = {
    "residual": (r"$\hat r$, the GPUs' residual", ""), "clock_share": ("GPU clock above the credited", ""),
    "host_share": ("host CPUs at peak", ""), "identities": ("identities a detection doesn't end", ""),
    "eps_agg": (r"$\varepsilon'$, aggregation slack", ""), "eps_w": (r"$\varepsilon_W$, PoUW's deficit", ""),
    "unmonitored": ("$s$, unmonitored share", ""), "mfu": (r"MFU$_0$", ""), "lag_s": ("verdict lag", " s"),
    "years": ("horizon", " yr"), "unfilled_gb": ("memory the fill doesn't reach", " GB"),
    "wipe_s": ("wipe period", " s"), "pous_rounds": ("proof-of-space rounds", ""),
    "pous_floor": ("store share rebuildable", ""), "intake_x": (r"declared intake, × $\alpha F$", ""),
    "disk_gb": ("disk left in place", " GB"), "Gamma": (r"$\Gamma$, gates per inference block", ""),
    "b_w": ("$b_w$, bits per weight", ""), "K": ("$K$, novel bits per block", ""),
    "sigma": (r"$\sigma$, side channels", " bit/s"), "m": ("$m$, recipe multiplier", ""),
    "kappa": (r"$\kappa$", ""), "pous_delta": (r"proof-of-space $\delta$", ""), "host_gb": ("host memory", " GB"),
    "node_flops": ("server compute", " FLOP/s"), "hbm_gb": ("HBM per server", " GB"),
    "tau": (r"$\tau$, certified timing", " bit/s"), "b_a": ("$b_a$, bits per activation", ""),
    "d": ("$d$, entry fee", ""), "e": ("$e$, fan-out", ""),
}
#: the fit bound's premises beside ``exposure.RANGES``: key, label, ``fit_mu``'s arguments at the low and high end,
#: their labels, and the source
FIT_ROWS = [
    ("w_e", "$w_e$, boundary bits (F4)", {"w_e": F.BOUNDARY["low"]}, {"w_e": F.BOUNDARY["high"]},
     num(F.BOUNDARY["low"]), num(F.BOUNDARY["high"]), "fit.BOUNDARY, low to high (central 35,840)"),
    ("routed", "routed mixtures, gain on $N^*$ (F3)", {}, {"N": N_STAR / ROUTED_HIGH}, "×1", f"×{ROUTED_HIGH:g}",
     f"none to the routed-mixture dossier's adversarial x{ROUTED_HIGH:g} (defensible x{ROUTED:g})"),
    ("b_train", "bits stored per trained weight", {}, {"b": F.B_TRAIN["central"]}, num(F.B_TRAIN["floor"]),
     num(F.B_TRAIN["central"]), "fit.B_TRAIN: the C1 floor b_w to 16-bit training storage"),
]
#: the smallest log10 swing in D a tornado draws (12%)
SHOWN = 0.05


def exposure(out: Path, rung: str, name: str) -> dict:
    """Figure 3: a tornado of D at T3 over each parameter's range in ``exposure.RANGES`` and the fit bound's premises,
    the others at ``rung``, with the move to T5 at each end of the proving cost's range."""
    p0 = next(r["p"] for r in rungs() if r["name"] == rung)
    D0, t0 = fit_parts(p0, "T3")
    assert math.isclose(D0, next(r["D_fit"]["T3"] for r in rungs() if r["name"] == rung), rel_tol=1e-4)

    def end(p: dict, tier: str = "T3", **kw: float) -> tuple[float, str]:
        d, t = fit_parts(p, tier, **kw)
        return d, max(t, key=lambda k: abs(t[k] - t0[k]))

    every = []
    for key, (lo, hi, src) in X.RANGES.items():
        if key in ("g", "overhead"):
            continue
        a, b = end(p0 | {key: lo}), end(p0 | {key: hi})
        label, unit = LABELS.get(key, (key, ""))
        every.append({"key": key, "label": label, "low": num(lo) + unit, "high": num(hi) + unit, "D_low": a[0],
                      "D_high": b[0], "via": (a[1], b[1]), "source": src})
    for key, label, kw_lo, kw_hi, tlo, thi, src in FIT_ROWS:
        a, b = end(p0, **kw_lo), end(p0, **kw_hi)
        every.append({"key": key, "label": label, "low": tlo, "high": thi, "D_low": a[0], "D_high": b[0],
                      "via": (a[1], b[1]), "source": src})
    lo, hi, src = X.RANGES["overhead"]
    a, b = end(p0 | {"overhead": lo}, "T5"), end(p0 | {"overhead": hi}, "T5")
    every.append({"key": "overhead", "label": "at T5: proving cost (1% sampled)", "low": num(lo), "high": num(hi),
                  "D_low": a[0], "D_high": b[0], "via": (a[1], b[1]), "source": src})

    def swing(r: dict) -> float:
        return abs(math.log10(r["D_low"] / r["D_high"]))

    rows = sorted((r for r in every if swing(r) >= SHOWN or r["key"] == "overhead"), key=swing)
    quiet = [r for r in every if r not in rows]
    routed = fit_parts(p0, "T3", N=N_STAR / ROUTED)[0]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.1, top=0.83)
    used = set()
    for y, r in enumerate(rows):
        for d, txt, hatch, via in ((r["D_low"], r["low"], "///", r["via"][0]), (r["D_high"], r["high"], None,
                                                                                 r["via"][1])):
            if abs(math.log10(d / D0)) < 2e-3:
                continue
            used.add(via)
            ax.barh(y, d - D0, left=D0, height=0.72, color=TERM[via], hatch=hatch, edgecolor="white", lw=0,
                    zorder=3)
            ax.text(d * (1.07 if d > D0 else 1 / 1.07), y, txt, va="center", ha="left" if d > D0 else "right",
                    fontsize=15, color="#333333")
        if r["key"] == "routed" and math.isfinite(fit_mu(p0, N=N_STAR / ROUTED)):
            ax.plot([routed] * 2, [y - 0.36, y + 0.36], color="white", lw=3, zorder=4)
            ax.text(routed / 1.08, y, f"×{ROUTED:g}: {num(routed)}", ha="right", va="center", fontsize=14,
                    color="white", fontweight="bold", zorder=5)
    ax.axvline(D0, color="k", lw=1.8, zorder=4)
    ax.set_xscale("log")
    ends = [d for r in rows for d in (r["D_low"], r["D_high"])]
    ax.set_xlim(min(*ends, D0) / 4, max(*ends, D0) * 5)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=17 if len(rows) > 12 else 19)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"compute dampener $\mathcal{D}$ at T3 (log)")
    ax.grid(axis="x", color="#e4e4e4", lw=1.2, zorder=0)
    via = {"unmonitored": "via $s$", "residual": r"via $\hat r$", "memory": r"via $\mu$", "declared": r"via $\nu$",
           "cheat": r"via $\chi$"}
    shown = [t for t in TERM if t in used]
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM[t]) for t in shown]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=GRAY, hatch="///", ec="white", lw=0),
                plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    ax.legend(handles, [via[t] for t in shown] + ["low end", "high end"], loc="lower left",
              bbox_to_anchor=(-0.3, 1.0), ncol=len(handles), fontsize=17, handlelength=1.1, columnspacing=1.2,
              handletextpad=0.4)
    phrase = {"Best plausible PoUW design": "the best PoUW design",
              "Streamed-label wipes, 6 GB": "6 GB streamed-label wipes"}[rung]
    title(fig, rf"Exposure of $\mathcal{{D}}$ around {phrase} ($\mathcal{{D}}$ = {num(D0)})")
    table = [{k: r[k] for k in ("label", "low", "high", "D_low", "D_high", "source")} | {"via": list(r["via"])}
             for r in reversed(rows)]
    caption = (
        f"A tornado of the compute dampener D at T3 (log x), centred on {phrase} (D = {num(D0)}"
        + (", by the fit bound" if math.isfinite(fit_mu(p0)) else ", where the fit bound doesn't hold and the closed "
           "form caps nothing") + "). Each bar moves one parameter across its range in exposure.RANGES, or one of the "
        "fit bound's premises, with the rest at the centre: the hatched segment is the range's low end, the solid one "
        "its high end, and the colour the term of 1/D that moves most there (a cap that stops holding lets r̂ "
        "through, so it shows as r̂). The proving-cost row moves to T5, 1% of proof blocks sampled, at each end of "
        "the cost's range. Rows sorted by swing: "
        + "; ".join(f"{words(t['label'])} {t['low']} to {t['high']}: D {num(t['D_low'])} to {num(t['D_high'])}"
                    for t in table)
        + (f". Routed mixtures at x{ROUTED:g}: D = {num(routed)}." if math.isfinite(fit_mu(p0, N=N_STAR / ROUTED))
           else ".")
        + " Every other parameter in exposure.RANGES, and premise, moves D by less than 12% over its range: "
        + ", ".join(r["key"] for r in quiet) + ". " + BASE)
    return {"files": save(fig, out, name), "title": f"Exposure around {phrase}", "caption": caption,
            "quote": {"D0": D0, "table": table, "quiet": [r["key"] for r in quiet],
                      **({"routed_x2": routed} if math.isfinite(fit_mu(p0, N=N_STAR / ROUTED)) else {})}}


def free_share(p: dict) -> float:
    """What memory has to cap at T3: r = r̂ + eps' and the verdict lag at MFU_0, and nu, as a share of compute."""
    r = T.evaluate(p, T.TIERS["T3"] | {"wipes": False, "exhaust": False}, p["mfu"])
    return r["residual"] + r["declared"]


def memory(out: Path) -> dict:
    """Figure 4: the memory cap mu against R_wipe, by the fit bound with its w_e band and by the closed form at the
    structural d and at the Dot16 conjecture's."""
    Rs = np.geomspace(0.5, 200, 241)
    held = Rs[Rs < EDGE_GB]
    closed = [closed_mu(at_wipe(R)) for R in Rs]
    closed_conj = [closed_mu(at_wipe(R, **CONJECTURE)) for R in Rs]
    fits = {k: [fit_mu(at_wipe(R), w_e=w) for R in held] for k, w in F.BOUNDARY.items()}
    routed_edge = F.bite_gb(P, N_STAR / ROUTED, P["b_w"])
    held2 = Rs[Rs < routed_edge]
    routed = [fit_mu(at_wipe(R), N=N_STAR / ROUTED) for R in held2]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.13, top=0.87)
    band = ax.fill_between(held, fits["high"], fits["low"], color=TERM["memory"], alpha=0.22, lw=0, zorder=1)
    central, = ax.plot(held, fits["central"], color=TERM["memory"], zorder=3)
    dashed, = ax.plot(held2, routed, color=TERM["memory"], lw=2.2, ls=(0, (5, 2)), zorder=3)
    form, = ax.plot(Rs, closed, color=TERM["residual"], zorder=3)
    form_conj, = ax.plot(Rs, closed_conj, color=TERM["residual"], lw=2.4, ls=(0, (7, 2, 1.5, 2)), zorder=3)
    ax.axvline(EDGE_GB, color="k", lw=1.8, ls=(0, (2, 2)), zorder=2)
    ax.text(EDGE_GB * 1.06, 1.6, f"fit bound void above {EDGE_GB:.1f} GB", fontsize=17, va="center")
    ax.legend([form, form_conj, central, band, dashed],
              [rf"closed form, $4R_{{\rm wipe}}\lambda_{{\rm in}}/(\kappa\, d\, b_w b_a F)$, $d$ = {num(P['d'])}",
               f"the same at the Dot16 conjecture's $d$ = {num(CONJECTURE['d'])}",
               f"fit bound, central $w_e$ ({num(W_E)} bits)",
               f"$w_e$ from {num(F.BOUNDARY['low'])} (top) to {num(F.BOUNDARY['high'])} bits (bottom)",
               rf"×{ROUTED:g} routed mixtures ($N^*/2$), void above {routed_edge:.1f} GB"],
              loc="lower right", fontsize=17, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    names = {"PoUW's floor credit": "PoUW's floor credit",
             "Floor credit and attested clock": "floor credit + attested clock",
             "Best plausible PoUW design": "best PoUW design", FLOORS[0]: "the floor"}
    levels = [(free_share(r["p"]), r["r_hat"], names[r["name"]]) for r in rungs() if r["name"] in names]
    for y, rh, txt in levels:
        ax.axhline(y, color=GRAY, lw=1.8, ls=(0, (6, 3)), zorder=1)
        ax.text(0.53, y * 1.1, rf"{num(y)}: {txt} ($\hat r$ = {num(rh)})", color=GRAY, fontsize=16, va="bottom")
    marks = {}
    for gb in (13.0, 6.0, 4.0, 1.0):
        p = at_wipe(gb)
        f = fit_mu(p)
        marks[f"{gb:g}"] = {"fit": f, "closed": closed_mu(p), "closed_conjecture": closed_mu(p | CONJECTURE),
                            "fit_low": fit_mu(p, w_e=F.BOUNDARY["low"]), "fit_high": fit_mu(p, w_e=F.BOUNDARY["high"])}
        ax.plot(gb, f, "o", ms=13, color=TERM["memory"], mec="white", mew=2, zorder=5)
        ax.annotate(f"{gb:g} GB{', the floor' if gb == 4.0 else ''}: {sci(f)}", (gb, f), xytext=(10, -10),
                    textcoords="offset points", va="top", fontsize=15, color="#333333", zorder=6,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 200)
    ax.set_ylim(2e-5, 3)
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.set_xlabel(r"wipe residual $R_{\rm wipe}$ (GB per server)")
    ax.set_ylabel(r"memory cap $\mu$ (share of compute)")
    title(fig, r"The memory cap $\mu$: the fit bound and the closed form")
    credit = levels[0][0]
    assert closed[0] > credit, "the closed form at d is below the floor credit's level on the chart: quote where"
    cross = float(np.exp(np.interp(np.log(credit), np.log(closed_conj), np.log(Rs))))
    assert Rs[0] < cross < Rs[-1]
    floor_cross = float(np.exp(np.interp(np.log(levels[-1][0]), np.log(fits["central"]), np.log(held))))
    caption = (
        "The memory cap mu, the most monitored compute can achieve as a share of server peak, against the wipe "
        f"residual R_wipe per server (log-log, 0.5 to 200 GB; streamed-label wipes, {ROUNDS} rounds). Dark blue: the "
        "closed form 4 R_wipe lambda_in / (kappa d b_w b_a F) as tiers.evaluate takes it (the windows bound: windows "
        "start at wipes and span whole periods), which needs (d, e)-threading at window scale: solid at the "
        f"structural d = {num(P['d'])}, where it is {num(closed[0])} already at 0.5 GB and caps nothing, dash-dot at "
        f"the Dot16 conjecture's d = {num(CONJECTURE['d'])} (conditional), which reaches the floor credit's "
        f"{num(credit)} at {num(cross)} GB. Light blue: the fit "
        f"bound (fit.py), which doesn't depend on d, for a {TARGET.upper()}-class target (N* = {num(N_STAR)} weights) "
        f"at central w_e = {num(W_E)} bits, with the band from high w_e ({num(F.BOUNDARY['high'])}, lower cap) to "
        f"low w_e ({num(F.BOUNDARY['low'])}, higher cap); it holds while a one-period window holds fewer than N* "
        f"weights, below {EDGE_GB:.1f} GB. Dashed: routed mixtures at x{ROUTED:g} (N*/2), void above "
        f"{routed_edge:.1f} GB. Memory caps where mu is below what a rung leaves, r = r̂ + eps' plus the verdict lag "
        "and nu (gray lines: the floor credit, the floor credit with an attested clock, the best PoUW "
        f"design, and the floor's r̂ = 0, where only eps', the lag and nu are left, {num(levels[-1][0])}); the fit "
        f"bound crosses the floor's line at {num(floor_cross)} GB. Markers: the fit bound at the streamed-label rungs' "
        "13 and 6 GB, the floor's 4 GB and power removal's 1 GB. Fit bound (closed form at d = "
        f"{num(P['d'])}, at d = {num(CONJECTURE['d'])}): "
        + "; ".join(f"{k} GB {num(v['fit'])} ({num(v['closed'])}, {num(v['closed_conjecture'])}), w_e band "
                    f"{num(v['fit_high'])} to {num(v['fit_low'])}" for k, v in marks.items())
        + ". " + BASE)
    return {"files": save(fig, out, "4_memory_cap"), "title": "The memory cap", "caption": caption,
            "quote": {"rungs": marks, "edge_gb": EDGE_GB, "routed_edge_gb": routed_edge,
                      "levels": {txt: {"r": y, "r_hat": rh} for y, rh, txt in levels},
                      "closed_at_0p5gb": closed[0], "closed_conjecture_reaches_floor_credit_gb": cross,
                      "fit_crosses_floor_gb": floor_cross}}


COSTS = [("today's sparse draws", P["overhead"]), ("512 draws per statement", S.COSTS["512 draws per statement"]),
         ("batched throughput", X.RANGES["overhead"][0])]
TOP = ["Best plausible PoUW design", "Streamed-label wipes, 6 GB", "Power removal, 1 GB", "Floor (r̂ = 0, 4 GB)",
       "Floor, one-minute verdict lag"]


def cheating(out: Path) -> dict:
    """Figure 5: the cheating share chi against proving cost, and D at T5 for the top rungs."""
    xs = np.geomspace(1e6, 1e14, 121)
    chi = [S.cheat(P | {"overhead": x}) for x in xs]
    fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.13, top=0.85, wspace=0.32)
    ax.plot(xs, chi, color=TERM["cheat"], zorder=3)
    quote = {}
    for (label, x), (dx, dy, ha, va) in zip(COSTS, ((-24, 18, "right", "center"), (-30, 40, "right", "center"),
                                                    (16, -14, "left", "top"))):
        c = S.cheat(P | {"overhead": x})
        quote[label] = {"overhead": x, "chi": c}
        ax.plot(x, c, "o", ms=14, color=TERM["cheat"], mec="white", mew=2.5, zorder=4)
        ax.annotate(f"{label}\n{sci(x)}: $\\chi$ = {sci(c)}", (x, c), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=16, zorder=5,
                    arrowprops={"arrowstyle": "-", "color": TERM["cheat"], "lw": 1.3} if dy > 20 else None)
    best = next(r for r in rungs() if r["name"] == TOP[0])["r_hat"]
    nu = X.nu(P)
    chord = T.chord(P, T.evaluate(P, T.TIERS["T5"])["cheat_bits"])
    ax.axhline(best, color=GRAY, lw=1.8, ls=(0, (6, 3)), zorder=2)
    ax.text(xs[0] * 1.4, best * 1.25, rf"best PoUW design's $\hat r$ = {num(best)}", color=GRAY, fontsize=16,
            va="bottom")
    ax.axhline(nu, color=TERM["declared"], lw=1.8, ls=(0, (6, 3)), zorder=2)
    ax.text(xs[0] * 1.4, nu / 1.3, rf"NCI's $\nu$ = {sci(nu)} at $d$ = {num(P['d'])}", color=TERM["declared"],
            fontsize=16, va="top")
    every_two = FuncFormatter(lambda v, _: sci(v) if round(np.log10(v)) % 2 == 0 else "")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(min(chi) / 3, 0.1)
    ax.xaxis.set_major_locator(LogLocator(numticks=10))
    ax.xaxis.set_major_formatter(every_two)
    ax.set_xlabel("proving cost (× step work)")
    ax.set_ylabel(r"cheating share $\chi$ at T5")
    D5 = {}
    for name in TOP:
        r = next(r for r in rungs() if r["name"] == name)
        ys = [rung_parts(name, r["p"] | {"overhead": x}, "T5")[0] for x in xs]
        c = STATUS[r["status"]]
        ls = (0, (5, 2)) if name.startswith(("Floor,", "Streamed")) else "-"
        bx.plot(xs, ys, color=c, ls=ls, zorder=3)
        below = name in (FLOORS[0], "Streamed-label wipes, 6 GB")
        bx.annotate(one_line(r["label"]), (xs[0], ys[0]), xytext=(6, -8 if below else 8), textcoords="offset points",
                    va="top" if below else "bottom", color=c, fontsize=15)
        D5[name] = {label: rung_parts(name, r["p"] | {"overhead": x}, "T5")[0] for label, x in COSTS}
        D5[name]["T3"] = r["D_rung"]["T3"]
        for label, x in COSTS:
            bx.plot(x, D5[name][label], "o", ms=10, color=c, zorder=4)
    for label, x in COSTS:
        for a in (ax, bx):
            a.axvline(x, color="#bbbbbb", lw=1.4, ls=":", zorder=1)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(xs[0], xs[-1])
    bx.set_ylim(10, 10 ** math.ceil(math.log10(max(v["T3"] for v in D5.values())) + 0.3))
    bx.xaxis.set_major_locator(LogLocator(numticks=10))
    bx.xaxis.set_major_formatter(every_two)
    bx.yaxis.set_major_formatter(FuncFormatter(plain))
    bx.set_xlabel("proving cost (× step work)")
    bx.set_ylabel(r"$\mathcal{D}$ at T5")
    title(fig, r"Proving cost sets the top of the ladder: $\chi$ and $\mathcal{D}$ at T5")
    caption = (
        "Left: the cheating share chi at T5 (1% of proof blocks sampled) against the proving cost per unit of step "
        "work (log-log, 1e6 to 1e14), from tiers.evaluate; chi doesn't depend on r̂. The blocks that inject a chosen "
        "output and pass number ln(1/delta) / q per identity, q = share / (overhead + replay), delta = "
        f"2^{math.log2(P['p_evade']):.0f}, 16 output bits each, each bit worth the chord of one inference block's "
        f"thread bound ({num(chord)} counted multiplications). "
        "Marked: " + "; ".join(f"{k} ({num(v['overhead'])}): chi = {num(v['chi'])}" for k, v in quote.items())
        + f". Reference lines: the best PoUW design's r̂ = {num(best)} and NCI's nu = {num(nu)} at the structural "
        f"d = {num(P['d'])}. Right: D at T5 for the top rungs against the same cost (memory rungs by the fit bound, the "
        "floors by the PoUW route); D at T3 is where each line flattens to the left. D at T5 at today's sparse "
        "draws, 512 draws per statement and batched throughput (and at T3): "
        + "; ".join(f"{k} " + ", ".join(num(v[label]) for label, _ in COSTS) + f" ({num(v['T3'])})"
                    for k, v in D5.items()) + ". " + BASE)
    return {"files": save(fig, out, "5_cheating_share"), "title": "The cheating share", "caption": caption,
            "quote": {"chi": quote, "D_T5": D5}}


def years(out: Path) -> dict:
    """Figure 6: years of frontier-compute growth each rung's D is worth, ln D / ln 4.5."""
    rows = rungs()
    g = math.log(S.GROWTH)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.265, right=0.97, bottom=0.12, top=0.78)
    quote = {}
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        c = STATUS[r["status"]]
        y3, y5 = r["years_rung"]["T3"], r["years_rung"]["T5"]
        quote[r["name"]] = {"T3": y3, "T5": y5, "D_T3": r["D_rung"]["T3"], "D_T5": r["D_rung"]["T5"]}
        ax.barh(y, y3, height=0.66, color=c, lw=0, zorder=3)
        if abs(y3 - y5) > 0.02:
            ax.plot([y5, y5], [y - 0.42, y + 0.42], color="white" if y5 > 0.3 else c, lw=3, zorder=4)
        ax.text(max(y3, 0) + 0.08, y, f"{y3:.1f} yr" + (f" (T5 {y5:.1f})" if abs(y3 - y5) > 0.05 else ""),
                va="center", fontsize=18, color=c)
        if r["routed"] and not r["routed"]["void"]:
            yr = math.log(r["routed"]["T3"]) / g
            quote[r["name"]]["routed_x2"] = yr
            ax.plot(yr, y, "^", ms=15, mfc="white", mec="k", mew=2, zorder=5)
            ax.text(yr - 0.14, y, f"×{ROUTED:g} routed: {yr:.1f}", ha="right", va="center", fontsize=15,
                    color="white", fontweight="bold", zorder=5)
    ax.set_yticks(range(len(rows)), [one_line(r["label"]) for r in reversed(rows)], fontsize=18)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    right = max(r["years_rung"]["T3"] for r in rows) + 1.9
    ax.set_xlim(0, right)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(rf"years of frontier-compute growth, $\ln\mathcal{{D}}/\ln {S.GROWTH:g}$ (T3)")
    ax.grid(axis="x", color="#e4e4e4", lw=1.2, zorder=0)
    top = ax.secondary_xaxis("top", functions=(lambda t: np.exp(t * g), lambda d: np.log(np.maximum(d, 1e-9)) / g))
    top.set_xticks([10.0 ** k for k in range(int(right * g / math.log(10)) + 1)])
    top.xaxis.set_major_formatter(FuncFormatter(plain))
    top.set_xlabel(r"$\mathcal{D}$", labelpad=8)
    status = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS.values()]
    ax.legend(status, list(STATUS), loc="upper right", ncol=1, fontsize=17, handlelength=1.0, handletextpad=0.4)
    title(fig, r"Years of delay: what each rung's $\mathcal{D}$ is worth")
    floor = next(r for r in rows if r["name"] == FLOORS[0])
    fit_years, conj_years = floor["years_fit"]["T3"], math.log(floor["conjecture"]["T3"]) / g
    caption = (
        f"Each rung's compute dampener D as years of frontier training-compute growth at {S.GROWTH:g}x a year "
        f"(Epoch AI): ln D / ln {S.GROWTH:g}, at T3 (bars; memory rungs by the fit bound, the floors by the PoUW "
        "route), with T5 at today's proving cost as the white tick, coloured by status. Years at T3 (T5): "
        + "; ".join(f"{k} {v['T3']:.1f} ({v['T5']:.1f}; D = {num(v['D_T3'])})" for k, v in quote.items())
        + f". Triangle: routed mixtures at x{ROUTED:g} on the 6 GB rung (13 GB is void under them). By the fit bound "
        f"both floors are {fit_years:.1f} years; under the Dot16 conjecture's d = {num(CONJECTURE['d'])} the floor "
        f"is {conj_years:.1f} (conditional). " + BASE)
    quote["floors_by_fit_bound"], quote["floor_conjecture"] = fit_years, conj_years
    return {"files": save(fig, out, "6_years"), "title": "Years of delay", "caption": caption, "quote": quote}


def schematic(out: Path) -> dict:
    """Figure 7: the four leaks of 1/D and what plugs each."""
    rows = {r["name"]: r for r in rungs()}
    six = rows["Streamed-label wipes, 6 GB"]["p"]
    mu6 = fit_mu(six)
    chi = {label: S.cheat(P | {"overhead": x}) for label, x in COSTS}
    r_floor, r_best = T.audited_residual(P), T.audited_residual(P | S.BEST)
    eps = P["eps_agg"]
    deficit = P["eps_w"] * (1 - r_floor)
    nu_conj = X.nu(P | CONJECTURE)
    s_lo, s_hi = min(min(v) for v in S.REGIMES.values()), max(max(v) for v in S.REGIMES.values())
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    title(fig, "The leaks and their plugs")
    pieces = [(r"$1/\mathcal{D}\ \leq\ $", "k"), ("$s$", TERM["unmonitored"]), (r"$\ +\ [\,\min($", "k"),
              (r"$\hat r$", TERM["residual"]), (r"$\ +\ $", "k"), (r"$\varepsilon_W W/C$", TERM["residual"]),
              (r"$\ +\ $", "k"), (r"$\varepsilon'$", TERM["residual"]),
              (r"$\ +\ $", "k"), (r"$\nu$", TERM["declared"]), ("$,\\ $", "k"),
              (r"$\mu$", TERM["memory"]), (r"$)\ +\ $", "k"), (r"$\chi$", TERM["cheat"]),
              (r"$\,]\,/\,\mathrm{MFU}_0$", "k")]
    eq = HPacker(children=[TextArea(t, textprops={"color": c, "fontsize": 30}) for t, c in pieces], pad=0, sep=0,
                 align="baseline")
    ax.add_artist(AnchoredOffsetbox(loc="center", child=eq, frameon=False, bbox_to_anchor=(8.0, 7.75),
                                    bbox_transform=ax.transData, borderpad=0))
    ax.text(3.3, 6.85, "plug", fontsize=19, color=GRAY, ha="center")
    ax.text(11.8, 6.85, "leak", fontsize=19, color=GRAY, ha="center")
    D6 = rows["Streamed-label wipes, 6 GB"]["D_fit"]["T3"]
    leaks = [
        ("s", 5.95, 1.1, "$s$", "unmonitored compute",
         [rf"{num(s_lo)}–{num(s_hi)} by regime; $\mathcal{{D}} \leq 1/s$"], TERM["unmonitored"]),
        ("r", 4.25, 1.95, r"$\hat r$", "monitored residual",
         [f"PoUW: 1 today → {num(r_floor)} floor credit → {num(r_best)} best design",
          rf"+ $\varepsilon_W W/C$ = {sci(deficit)} deficit, $\varepsilon'$ = {sci(eps)} wrong certificates",
          rf"memory: $\mu$ = {sci(mu6)} at 6 GB; $\mathcal{{D}}$ = {num(D6)} for any $\hat r > \mu$"],
         TERM["residual"]),
        ("nu", 2.42, 1.45, r"$\nu$", "declared computation",
         [rf"{sci(X.nu(P))} at Kimi K2's K and the structural $d$ = {num(P['d'])}",
          rf"{sci(nu_conj)} at the Dot16 conjecture's $d$ = {num(CONJECTURE['d'])}"], TERM["declared"]),
        ("chi", 1.0, 1.1, r"$\chi$", "faulty proof blocks (T5)",
         [f"{sci(chi[COSTS[0][0]])} today, {sci(chi[COSTS[1][0]])} at 512 draws"], TERM["cheat"]),
    ]
    plugs = [
        ("none", 5.95, "none: trains at full efficiency", "", "s", None),
        ("pouw", 4.67, "PoUW: credit certified work", "T1", "r", TERM["residual"]),
        ("memory", 3.83, "memory: wipes cap the residual", "T3", "r", TERM["memory"]),
        ("nci", 2.42, "NCI: novel-bit throttle", "T1", "nu", TERM["declared"]),
        ("detect", 1.4, "terminal detection", "T5", "chi", TERM["cheat"]),
        ("prove", 0.6, "cheaper proving per draw", "T5", "chi", TERM["cheat"]),
    ]
    px0, px1, lx0, lx1, h = 0.6, 6.0, 8.0, 15.6, 0.62
    ys = {}
    for key, y, height, sym, what, vals, color in leaks:
        ys[key] = y
        ax.add_patch(FancyBboxPatch((lx0, y - height / 2), lx1 - lx0, height,
                                    boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec=color, lw=3.2))
        ax.text(lx0 + 0.55, y, sym, fontsize=30, va="center", ha="center", color=color)
        top = y + 0.2 + 0.225 * (len(vals) - 1)
        ax.text(lx0 + 1.2, top, what, fontsize=21, va="center")
        for i, val in enumerate(vals):
            ax.text(lx0 + 1.2, top - 0.45 * (i + 1), val, fontsize=16, va="center", color="#444444")
    for key, y, text, tier, leak, color in plugs:
        dashed = color is None
        ax.add_patch(FancyBboxPatch((px0, y - h / 2), px1 - px0, h, boxstyle="round,pad=0,rounding_size=0.16",
                                    fc="white" if dashed else "#f4f4f4", ec=GRAY if dashed else "#9a9a9a", lw=1.6,
                                    ls="--" if dashed else "-"))
        ax.text(px0 + 0.25, y, text, fontsize=19, va="center", color=GRAY if dashed else "k")
        ax.text(px1 - 0.2, y, tier, fontsize=16, va="center", ha="right", color=GRAY)
        if not dashed:
            ax.add_patch(FancyArrowPatch((px1 + 0.08, y), (lx0 - 0.1, ys[leak] + (y - ys[leak]) * 0.35),
                                         arrowstyle="-|>,head_width=6,head_length=12", mutation_scale=1.6, lw=3,
                                         color=color, shrinkA=0, shrinkB=0))
    ax.text(px1 - 0.2, 6.85, "from", fontsize=16, color=GRAY, ha="right")
    caption = (
        "The bound 1/D <= s + (min(r̂ + eps_W W/C + eps' + nu, mu) + chi) / MFU_0 as four leaks and what plugs each. "
        f"s, unmonitored compute ({num(s_lo)} to {num(s_hi)} across the s dossier's regimes), has no plug: it trains at "
        "full efficiency, so D <= 1/s always. The monitored residual r = r̂ + eps_W W/C + eps' has two plugs, which "
        f"substitute: PoUW credits certified work (the audited residual r̂ = 1 today, {num(r_floor)} under the floor "
        f"credit, {num(r_best)} at the best plausible design, plus PoUW's deficit on the certified work, "
        f"eps_W W/C = {num(deficit)} under the floor credit, and the aggregation slack eps' = {num(eps)}, the share "
        "whose certificates may be wrong, which the theorem's delta needs), and memory caps whatever PoUW leaves "
        f"(mu = {num(mu6)} at 6 GB streamed-label wipes by the fit "
        f"bound, so D = {num(D6)} at T3 for any r̂ above it). NCI plugs the declared "
        f"computation (nu = {num(X.nu(P))} at Kimi K2's K and the structural entry fee d = {num(P['d'])}; "
        f"{num(nu_conj)} at the Dot16 conjecture's d = {num(CONJECTURE['d'])}, conditional), which is now the largest "
        "term the floor leaves. Faulty proof blocks at T5 are plugged by terminal detection, which ends "
        "the identity, and by the cost of proving a draw, which sets how many pass: chi = "
        f"{num(chi[COSTS[0][0]])} at today's sparse draws, {num(chi[COSTS[1][0]])} at 512 draws per statement. "
        "PoUW and NCI are complements: without PoUW r̂ = 1, without NCI nu = 1. Each plug's tag is the tier from "
        "which it is in force: PoUW and NCI from T1, wipes from T3, and sampling, which is where chi arises, at T5. "
        + BASE)
    return {"files": save(fig, out, "7_schematic"), "title": "Leaks and plugs", "caption": caption,
            "quote": {"s_range": [s_lo, s_hi], "r_hat": {"floor credit": r_floor, "best design": r_best},
                      "eps_agg": eps, "eps_w": P["eps_w"], "deficit": deficit,
                      "mu_6gb": mu6, "nu": X.nu(P), "nu_conjecture": nu_conj, "chi": chi}}


FIGURES = {
    "1_ladder": ladder, "2_phase": phase,
    "3_exposure_best": lambda out: exposure(out, "Best plausible PoUW design", "3_exposure_best"),
    "3b_exposure_6gb": lambda out: exposure(out, "Streamed-label wipes, 6 GB", "3b_exposure_6gb"),
    "4_memory_cap": memory, "5_cheating_share": cheating, "6_years": years, "7_schematic": schematic,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    results = {"commit": SHA, "uncommitted": DIRTY, "figures": {name: fn(out) for name, fn in FIGURES.items()}}
    (out / "figures.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    for name, r in results["figures"].items():
        print(f"{name}: {', '.join(r['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
