"""Figures for the compute dampener D = 1/eta, all computed by ``tiers``: D per tier and scenario, the (r, R_wipe) phase
diagram, exposure tornados, the memory cap mu, NCI's term nu, the cheating share chi, eta's four parts, and the map from
mechanisms to the terms they dampen. Generated output: it goes to the evidence store, never Git.

    python3 benchmarks/nci/figures.py OUTDIR    # NAME.png (1600x900) and NAME.svg per figure, and figures.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiers as T  # noqa: E402

plt.rcParams.update({
    "figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 100, "font.size": 22, "axes.labelsize": 24,
    "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 20, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4, "lines.linewidth": 3.5,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 4, "ytick.minor.size": 4, "mathtext.fontset": "dejavusans", "svg.hashsalt": "verity-figures",
})

#: one colour per term of eta <= s + min(r, mu) + nu + chi (Okabe-Ito), keyed as ``tiers.evaluate`` names the parts
TERM = {"unmonitored": "#CC79A7", "residual": "#0072B2", "declared": "#009E73", "cheat": "#D55E00"}
INTAKE = "#56B4E9"
GRAY = "#6b6b6b"

SCENARIOS = {
    "bounded": ("Design, host CPUs bounded", {"residual": 0.01, "host_share": 0.0}, "#1a1a1a"),
    "peak": ("Design, host CPUs at peak", {}, "#8c8c8c"),
    "today": ("Served today", {"residual": 0.9}, "#E69F00"),
    "wipes": ("Served today, better wipes",
              {"residual": 0.9, "pous_rounds": 12000.0, "unfilled_gb": 3.05, "pous_floor": 0.0}, "#882255"),
}
PHRASE = {"bounded": "the design with host CPUs bounded", "peak": "the design with host CPUs at peak",
          "today": "served today", "wipes": "served today with better wipes"}
TIER_LABEL = {"T0": "T0\nno checks", "T1": "T1\nNCI + PoUW", "T2": "T2\n+ network\naccounting", "T3": "T3\n+ wipes",
              "T4": "T4\n+ exhaustion", "T5": "T5\n+ sampled\nproofs 1%"}
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
P = T.DEFAULTS
BASE = (f"Other parameters at tiers.DEFAULTS: the Kimi K2 calibration (K = {P['K']:.3g} bit, Gamma = {P['Gamma']:.3g} "
        f"gates, alpha = K/Gamma = {P['K'] / P['Gamma']:.3g}; d = {P['d']:g}, e = {P['e']:.3g}, kappa = {P['kappa']}), "
        f"b_w = {P['b_w']} bit, b_a = {P['b_a']} bit, sigma = {P['sigma']:g} bit/s, tau = {P['tau']:g} bit/s, wipes "
        f"every {P['wipe_s']:g} s, {P['nodes']:.3g} servers of {P['node_flops']:.3g} FLOP/s for {P['years']:g} year.")


def sci(x: float, digits: int = 2) -> str:
    """``x`` as mathtext ``a x 10^b``, or plainly between 0.01 and 10^4."""
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1e4:
        return f"{float(f'{x:.{digits}g}'):g}"
    m, e = f"{x:.{digits - 1}e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"${'' if m == '1' else m + r'\times'}10^{{{int(e)}}}$"


def title(fig, text: str) -> None:
    fig.text(0.015, 0.975, text, fontsize=28, va="top")


def slope_deg(ax, p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """The on-screen angle of the segment ``p1``-``p2`` in data coordinates, once the axes' layout is final."""
    (x1, y1), (x2, y2) = ax.transData.transform([p1, p2])
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def dampener(p: dict, tier: str) -> float:
    return 1 / T.evaluate(p, T.TIERS[tier])["efficiency"]


def helpers_gb(p: dict) -> float:
    """What helpers refill within the read deadline during a wipe (``tiers.evaluate``'s ``help_bits``), in GB."""
    share = min(1.0, max(0.0, p["sigma"] * p["t_read"] - T.BLOCK_NAME_BITS) / T.BLOCK_BITS)
    return share * (p["hbm_gb"] + p["host_gb"])


def at_wipe(p: dict, R_gb: float) -> dict:
    """``p`` with the wipe residual at ``R_gb``: the slack all but freed (10^6 rounds, no floor), the rest unfilled
    memory."""
    q = p | {"pous_rounds": 1e6, "pous_floor": 0.0}
    unfilled = R_gb - T.slack(q) * (q["hbm_gb"] + q["host_gb"]) - helpers_gb(q) - q["disk_gb"]
    assert unfilled >= 0, f"R_wipe = {R_gb} GB is below what slack and helpers leave; raise it"
    return q | {"unfilled_gb": unfilled}


def memory_cap(p: dict, R_gb: float) -> float:
    """mu at T3: the share of compute the windows bound leaves the monitored side when a wipe leaves ``R_gb`` per
    server."""
    H = p["years"] * T.YEAR
    per_server = T.windows_bound(p, H, R_gb * T.GB, T.ingress(p, T.TIERS["T3"]), p["wipe_s"])
    return 2 / p["kappa"] * per_server / (p["node_flops"] * H)


def closed_cap(p: dict, R_gb: float) -> float:
    """mu's closed form, 4 R_wipe lambda_in / (kappa d b_w b_a F)."""
    lam_in = T.ingress(p, T.TIERS["T3"])
    return 4 * R_gb * T.GB * lam_in / (p["kappa"] * p["d"] * p["b_w"] * p["b_a"] * p["node_flops"])


def save(fig, out: Path, name: str) -> list[str]:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    return [f"{name}.png", f"{name}.svg"]


def stack(out: Path) -> dict:
    """Figure 1: D per tier, T0 (no checks, D = 1) to T5, for the four scenarios."""
    names = ["T0", *T.TIERS]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.09, right=0.71, bottom=0.2, top=0.87)
    quote = {}
    for key, (label, over, color) in SCENARIOS.items():
        p = P | over
        D = [1.0] + [dampener(p, t) for t in T.TIERS]
        quote[key] = dict(zip(names, D))
        ax.plot(range(6), D, drawstyle="steps-mid", color=color, ls=(0, (4, 2)) if key == "wipes" else "-", zorder=3)
        ax.plot(range(6), D, "o", color=color, ms=12, zorder=4)
        ax.text(5.65, D[-1], f"{label}\n$r$ = {T.residual_share(p):g}: $D$ = {D[-1]:.3g}", color=color, va="center",
                fontsize=20, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1500)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlim(-0.45, 5.5)
    ax.set_xticks(range(6), [TIER_LABEL[n] for n in names], fontsize=19)
    ax.set_ylabel(r"compute dampener $D = 1/\eta$")
    ax.grid(axis="y", color="#e4e4e4", lw=1.2, zorder=0)
    title(fig, "The stack: what each tier does to the compute dampener")
    caption = (
        "Compute dampener D = 1/eta at each cumulative tier for four scenarios, log scale. T0 has no checks (D = 1); "
        "T1 adds NCI, PoUW and proves every proof block; T2 adds network accounting; T3 wipes; T4 exhaustion; T5 "
        "replaces proving every block with sampled proofs at 1% of compute, which opens the cheating share chi. "
        "Scenarios: design with host CPUs bounded (residual 0.01, host_share 0, r = 0.01); design with host CPUs at "
        "datasheet peak (residual 0.01 + host_share 0.05, r = 0.06); served today (residual 0.9, r = 0.95); served "
        "today with better wipes (residual 0.9, pous_rounds 12000, unfilled_gb 3.05, pous_floor 0, so R_wipe = 4.07 GB "
        "instead of 203 GB). D at T1, T2, T3, T4, T5: " + "; ".join(
            f"{SCENARIOS[k][0]} " + ", ".join(f"{v:.3g}" for v in list(d.values())[1:]) for k, d in quote.items())
        + ". " + BASE)
    return {"files": save(fig, out, "1_stack"), "title": "The stack", "caption": caption,
            "quote": {k: {t: round(v, 3) for t, v in d.items()} for k, d in quote.items()}}


def _phase_cell(rR: tuple[float, float]) -> tuple[float, bool]:
    r, R = rR
    e = T.evaluate(at_wipe(P | {"residual": r, "host_share": 0.0}, R), T.TIERS["T4"])
    assert abs(e["wipe_state_bits"] / T.GB - R) <= 1e-9 * R
    return 1 / e["efficiency"], e["memory_caps_residual"]


def phase(out: Path) -> dict:
    """Figure 2: D at T4 over the monitored compute residual r and the wipe residual R_wipe."""
    rs, Rs = np.logspace(-3, 0, 91), np.logspace(-1, 3, 101)
    with ProcessPoolExecutor() as ex:
        cells = list(ex.map(_phase_cell, [(r, R) for R in Rs for r in rs], chunksize=64))
    D = np.array([c[0] for c in cells]).reshape(len(Rs), len(rs))
    caps = np.array([c[1] for c in cells]).reshape(D.shape)
    p0 = P | {"host_share": 0.0}
    edge = np.logspace(-1, 3, 400)
    mu_edge = np.array([memory_cap(p0, R) for R in edge])
    assert all(caps[i, j] == (rs[j] > memory_cap(p0, R)) for i, R in enumerate(Rs) for j in range(0, len(rs), 9)
               if abs(np.log10(rs[j] / memory_cap(p0, R))) > 0.05)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.93, bottom=0.13, top=0.87)
    levels = np.arange(0, 4.01, 0.5)
    cmap = ListedColormap(plt.cm.Blues(np.linspace(0.02, 0.6, len(levels) - 1)))
    cf = ax.contourf(rs, Rs, np.log10(D), levels=levels, cmap=cmap)
    cs = ax.contour(rs, Rs, np.log10(D), levels=[1, 2, 3], colors="k", linewidths=2.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(rs[0], rs[-1])
    ax.set_ylim(Rs[0], Rs[-1])
    cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.035, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["1", "10", "100", "1000", r"$10^4$"])
    cb.set_label("D", rotation=0, labelpad=16)
    cb.outline.set_visible(False)
    ax.clabel(cs, fmt={1: "D = 10", 2: "D = 100", 3: "D = 1000"}, fontsize=20, inline_spacing=14,
              manual=[(0.3, 135.0), (0.035, 13.4), (0.012, 1.34)])
    ax.plot(mu_edge, edge, color="white", lw=2.6, ls=(0, (5, 3)), zorder=2)
    i = np.searchsorted(edge, 60.0)
    ax.text(mu_edge[i] * 1.25, edge[i] / 1.25, r"$\mu(R_{\rm wipe}) = r$", color="white", fontsize=22,
            fontweight="bold", rotation=slope_deg(ax, (mu_edge[i - 20], edge[i - 20]), (mu_edge[i + 20], edge[i + 20])),
            ha="center", va="top", rotation_mode="anchor")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlabel(r"monitored compute residual $r$ (GPU residual + host CPU share)")
    ax.set_ylabel(r"wipe residual $R_{\rm wipe}$ (GB per server)")
    ax.text(1.3e-3, 40, r"$r$ binds: $D \approx 1/r$", fontsize=23, va="center")
    ax.text(0.6, 0.17, r"memory binds: $D \approx 1/\mu(R_{\rm wipe})$", fontsize=23, ha="right", va="center")
    pts = {}
    for key, (_, over, _) in SCENARIOS.items():
        p = P | over
        e = T.evaluate(p, T.TIERS["T4"])
        pts[key] = (T.residual_share(p), e["wipe_state_bits"] / T.GB, 1 / e["efficiency"])
    (xt, yt, _), (xw, yw, _) = pts["today"], pts["wipes"]
    ax.annotate("", xy=(xw, yw * 1.3), xytext=(xt, yt / 1.3),
                arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.9", "lw": 3.5,
                            "color": SCENARIOS["wipes"][2]})
    ax.text(xt / 1.12, (yt * yw) ** 0.5 * 1.5, "better\nwipes", ha="right", va="center", fontsize=21,
            color=SCENARIOS["wipes"][2])
    short = {"bounded": "Design,\nhost CPUs bounded", "peak": "Design,\nhost CPUs at peak", "today": "Served today",
             "wipes": "Today, better wipes"}
    for key, (x, y, d) in pts.items():
        ax.plot(x, y, "o", ms=20, color=SCENARIOS[key][2], mec="white", mew=2.5, zorder=5)
        ax.annotate(f"{short[key]}\nD = {d:.3g}", (x, y), xytext=(-14, 12), textcoords="offset points", ha="right",
                    va="bottom", fontsize=19, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none",
                                                    "alpha": 0.85}, zorder=6)
    title(fig, "Memory rescues a bad residual: D at T4")
    caption = (
        "Compute dampener D at T4 (NCI, PoUW, network accounting, wipes and exhaustion, every proof block proved) over "
        "the monitored compute residual r (x, log, 1e-3 to 1; residual = r, host_share = 0) and the wipe residual "
        "R_wipe per server (y, log, 0.1 GB to 1 TB). R_wipe is set through unfilled_gb with pous_rounds 1e6 and "
        "pous_floor 0 (slack 0.012 GB, no helpers at sigma = 1e3); the exhaustion envelope is then R_wipe + 38 GB and "
        "never binds before the wipe bound. Filled bands are half-decades of D; black lines D = 10, 100, 1000; the "
        "dashed white line is where the memory cap mu(R_wipe) equals r. Up-left of it D is about 1/r (vertical "
        "contours), down-right about 1/mu (horizontal contours): there memory caps the residual whatever r is. "
        "Markers: "
        + "; ".join(f"{SCENARIOS[k][0]} (r = {x:g}, R_wipe = {y:.3g} GB): D = {d:.3g}" for k, (x, y, d) in pts.items())
        + f". At r = 1e-3 and R_wipe = 0.1 GB, D = {D[0, 0]:.3g}: the one-hour wipe period floors mu. " + BASE)
    return {"files": save(fig, out, "2_phase"), "title": "The phase diagram", "caption": caption,
            "quote": {k: {"r": x, "R_wipe_gb": round(y, 3), "D_T4": round(d, 3)} for k, (x, y, d) in pts.items()}}


#: label, overrides at the range's low and high end, their labels, and where the range is from
EXPOSURE = [
    ("$s$, unmonitored share", {"unmonitored": 0.0}, {"unmonitored": 0.01}, "0", "1%",
     "assumed: up to 1% of the fleet's compute outside monitoring"),
    ("$r$, GPU residual", {"residual": 0.0074}, {"residual": 0.9}, "0.74%", "0.9",
     "PoUW's proved deficit epsilon_W = 0.74% to served windows today, 0.9 (SPEC residual)"),
    ("host CPUs, share of compute", {"host_share": 0.0}, {"host_share": 0.05}, "0", "5%",
     "bounded (0) to the AMX datasheet peak (SPEC host_share)"),
    (r"$R_{\rm wipe}$, all three parts", {"unfilled_gb": 3.05, "pous_rounds": 12000.0, "pous_floor": 0.0},
     {"unfilled_gb": 51.0, "pous_rounds": 86.0, "pous_floor": 0.0047}, "4.07 GB", "203 GB",
     "better wipes (unfilled 3.05 GB, 12,000 rounds, no floor) to today's wipes"),
    ("unfilled memory", {"unfilled_gb": 3.05}, {"unfilled_gb": 51.0}, "3.05 GB", "51 GB",
     "on-chip alone to the measured gaps (SPEC unfilled_gb)"),
    ("proof-of-space rounds $k$", {"pous_rounds": 86.0}, {"pous_rounds": 12000.0}, "86", "12,000",
     "today's 1/19 certificate to under 1 GB of slack (SPEC pous_rounds)"),
    ("certificate floor", {"pous_floor": 0.0}, {"pous_floor": 0.0047}, "0", "0.47%",
     "labels streamed from the challenger to the deployed D = 2 (SPEC pous_floor)"),
    ("disk left in place", {"disk_gb": 0.0}, {"disk_gb": 8000.0}, "0", "8 TB",
     "unplugged to the dark-compute red team's 8 TB of NVMe (test_tiers)"),
    (r"$\sigma$, side channels", {"sigma": 1e2}, {"sigma": 1e7}, "100 bit/s", "10 Mbit/s",
     "assumed, figure 4's range (SPEC allows 1 to 1e10 bit/s)"),
    ("$b_w$, bits per weight", {"b_w": 0.8}, {"b_w": 4.5}, "0.8", "4.5",
     "parameters.json b_w.pretrained/frontier-lossy, low to high"),
    ("$b_a$, bits per activation", {"b_a": 1.0}, {"b_a": 8.25}, "1", "8.25",
     "parameters.json b_a.activation, boundary-training low to pretraining-forward high"),
    (r"$\alpha = K/\Gamma$ via $K$", {"K": 2.2e6}, {"K": 9.1e7}, "2.2e6 bit", "9.1e7 bit",
     "calibrations.json kimi-k2-1m-coded to kimi-k2-11mb-raw's K, Gamma held at 6.4e16"),
    ("$e$, fan-out", {"e": 22297.0}, {"e": 165000.0}, "22,297", "165,000",
     "models.json dense-1b to dense-405b"),
    ("$d$, entry fee", {"d": 2027.0}, {"d": 15000.0}, "2027", "15,000",
     "models.json dense-1b to dense-405b"),
    ("proving overhead (acts at T5)", {"overhead": 1.6e8}, {"overhead": 4.8e12}, "1.6e8", "4.8e12",
     "batched throughput to today's sparse draws (SPEC overhead)"),
]


def exposure(out: Path, center: str = "bounded", name: str = "3_exposure") -> dict:
    """Figure 3: a tornado of D at T4 over each parameter's plausible range, the others at the ``center`` scenario."""
    over0 = SCENARIOS[center][1]
    p0 = P | over0
    e0 = T.evaluate(p0, T.TIERS["T4"])
    D0 = 1 / e0["efficiency"]

    def end(over: dict) -> tuple[float, str | None]:
        e = T.evaluate(p0 | over, T.TIERS["T4"])
        d = 1 / e["efficiency"]
        return d, max(TERM, key=lambda t: abs(e[t] - e0[t])) if abs(np.log10(d / D0)) >= 2e-3 else None

    rows = []
    for label, lo, hi, tlo, thi, src in EXPOSURE:
        (a, ta), (b, tb) = end(lo), end(hi)
        rows.append((abs(np.log10(a / b)), max(abs(np.log10(a / D0)), abs(np.log10(b / D0))), label, a, b, tlo, thi,
                     (ta, tb), src))
    rows.sort(key=lambda row: (row[0], row[1]))
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.11, top=0.83)
    for y, (_, _, label, a, b, tlo, thi, terms, _) in enumerate(rows):
        for d, txt, alpha, term in ((a, tlo, 0.45, terms[0]), (b, thi, 1.0, terms[1])):
            if abs(np.log10(d / D0)) < 2e-3:
                continue
            ax.barh(y, d - D0, left=D0, height=0.72, color=TERM[term], alpha=alpha, lw=0, zorder=3)
            ax.text(d * (1.08 if d > D0 else 1 / 1.08), y, txt, va="center", ha="left" if d > D0 else "right",
                    fontsize=16, color="#333333")
        if all(abs(np.log10(d / D0)) < 2e-3 for d in (a, b)):
            ax.text(D0 * 1.08, y, f"no effect ({tlo} to {thi})", va="center", fontsize=16, color=GRAY)
    ax.axvline(D0, color="k", lw=1.8, zorder=4)
    ax.set_xscale("log")
    lo = min(min(r[3], r[4]) for r in rows)
    hi = max(max(r[3], r[4]) for r in rows)
    ax.set_xlim(min(lo, D0) / 3.5, max(hi, D0) * 4.5)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}".strip("0.")[:1] == "3" else ""))
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=19)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"compute dampener $D$ at T4 (log)")
    ax.grid(axis="x", color="#e4e4e4", lw=1.2, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM[t]) for t in TERM]
    handles += [plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=GRAY)]
    ax.legend(handles, ["via $s$", r"via $r$ or $\mu$", r"via $\nu$", r"via $\chi$", "low end of range", "high end"],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=6, fontsize=17, handlelength=1.1,
              columnspacing=1.3, handletextpad=0.4)
    title(fig, f"Exposure of D at T4 around {PHRASE[center]} (D = {D0:.3g})")
    table = [{"parameter": r[2], "low": r[5], "high": r[6], "D_low": round(r[3], 4), "D_high": round(r[4], 4),
              "via": list(r[7]), "source": r[8]} for r in reversed(rows)]
    caption = (
        f"Tornado of the compute dampener D at T4, log axis, centered on {PHRASE[center]} "
        f"({', '.join(f'{k} {v:g}' for k, v in over0.items())}; D = {D0:.4g}). Each bar moves one parameter (or, for "
        "R_wipe, three together) across its plausible range with the rest at the center; the pale segment is the "
        "range's low end, the dark segment its high end, the colour the term of eta that moves most at that end, and "
        "'no effect' means D moves under 0.5%. Rows are sorted by swing. Proving overhead acts only at T5 (sampled "
        "proofs): every block is proved at T4. Ranges: "
        + "; ".join(f"{t['parameter']}: {t['low']} to {t['high']} ({t['source']})" for t in table) + ". " + BASE)
    return {"files": save(fig, out, name), "title": f"Exposure around {PHRASE[center]}", "caption": caption,
            "quote": {"D0": round(D0, 4), "table": table}}


def memory(out: Path) -> dict:
    """Figure 4: the memory cap mu against R_wipe, for four side-channel rates, against the residual levels."""
    Rs = np.logspace(-1, 3, 241)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.13, top=0.87)
    shades = plt.cm.Blues(np.linspace(0.45, 0.98, 4))
    quote = {}
    p0 = P | {"host_share": 0.0}
    for sigma, color in zip((1e3, 1e5, 1e6, 1e7), shades):
        p = p0 | {"sigma": sigma}
        slack_gb = T.slack(p | {"pous_rounds": 1e6, "pous_floor": 0.0}) * (p["hbm_gb"] + p["host_gb"])
        floor = max(Rs[0], helpers_gb(p) + slack_gb)
        R = Rs[Rs >= floor]
        ax.plot(R, [memory_cap(p, x) for x in R], color=color, zorder=3)
        ax.plot(R, [closed_cap(p, x) for x in R], color=color, lw=1.6, ls=(0, (3, 2)), zorder=3)
        if floor > Rs[0]:
            ax.plot(R[0], memory_cap(p, R[0]), "o", ms=11, mfc="white", mec=color, mew=2.5, zorder=4)
        if sigma != 1e5:
            name = r"$\sigma$ = $10^3$ or $10^5$ bit/s" if sigma == 1e3 else rf"$\sigma$ = {sci(sigma)} bit/s"
            va = {1e3: "top", 1e6: "bottom"}.get(sigma, "center")
            ax.text(1.08e3, memory_cap(p, 1e3), name, color=color, va=va, fontsize=19, clip_on=False)
        lam_in = T.ingress(p, T.TIERS["T3"])
        quote[f"{sigma:g}"] = {"lambda_in": lam_in, "helpers_floor_gb": round(helpers_gb(p), 3),
                               "closed_form_from_gb": round(lam_in * p["wipe_s"] / T.GB, 3),
                               "mu_at_4.07GB": memory_cap(p, 4.07) if 4.07 >= floor else None,
                               "mu_at_202.6GB": memory_cap(p, 202.6)}
    cross = {r: float(np.exp(np.interp(np.log(r), np.log([memory_cap(p0, x) for x in Rs]), np.log(Rs))))
             for r in (0.01, 0.06)}
    for r, txt in ((0.01, "$r$ = 0.01, host CPUs bounded"), (0.06, "$r$ = 0.06, host CPUs at peak")):
        ax.axhline(r, color=GRAY, lw=2, ls=(0, (6, 3)), zorder=2)
        ax.text(0.11, r * 1.12, txt, color=GRAY, fontsize=18, va="bottom")
        ax.plot(cross[r], r, "o", ms=11, color=GRAY, zorder=5)
        ax.annotate(f"{cross[r]:.3g} GB", (cross[r], r), xytext=(10, -8), textcoords="offset points", va="top",
                    fontsize=17, color=GRAY)
    for R, txt in ((4.07, "better wipes\n4.07 GB"), (202.6, "today's wipes\n203 GB")):
        ax.axvline(R, color="#bbbbbb", lw=1.6, ls=":", zorder=1)
        ax.text(R / 1.08, 3.2, txt, fontsize=17, color=GRAY, va="top", ha="right")
    lam = T.ingress(p0, T.TIERS["T3"])
    R_lo = lam * P["wipe_s"] / T.GB
    ax.annotate(r"dashed: closed form $\mu = 4R_{\rm wipe}\lambda_{\rm in}/(\kappa\, d\, b_w b_a F)$," "\n"
                rf"exact above $R_{{\rm wipe}} = \lambda_{{\rm in}} t_{{\rm wipe}}$ ≈ {R_lo:.2g} GB "
                "(one wipe period)",
                (R_lo, closed_cap(p0, R_lo)), xytext=(4.5, 1.25e-4), fontsize=17,
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 1.4})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e3)
    ax.set_ylim(1e-4, 5)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlabel(r"wipe residual $R_{\rm wipe}$ (GB per server)")
    ax.set_ylabel(r"memory cap $\mu$ (share of compute)")
    title(fig, r"The memory cap $\mu$ grows with what a wipe leaves")
    caption = (
        "The memory cap mu at T3 (wipes; at T4 the exhaustion envelope can bind instead) against the wipe residual "
        "R_wipe per server, log-log, for side channels sigma = 1e3, 1e5, 1e6 and 1e7 bit/s (1e5 coincides with 1e3). "
        "Solid: the model's windows bound (tiers.windows_bound; windows start at wipes and span whole periods). "
        "Dashed: the closed form mu = 4 R_wipe lambda_in / (kappa d b_w b_a F), lambda_in = alpha F + sigma + tau, "
        f"exact where the two coincide, above R_wipe = lambda_in t_wipe ({R_lo:.2g} GB at sigma = 1e3). Curves start "
        "where helpers within the 1 ms read deadline already refill that much memory (open circles: "
        + ", ".join(f"{v['helpers_floor_gb']:g} GB at sigma = {k}" for k, v in quote.items() if v["helpers_floor_gb"])
        + "). Horizontal lines: the residual levels r = 0.01 (host CPUs bounded) and r = 0.06 (at peak); memory caps r "
        f"where mu is under them, R_wipe under {cross[0.01]:.3g} GB and {cross[0.06]:.3g} GB at sigma = 1e3 (gray "
        "dots). Vertical lines: better wipes (4.07 GB, mu = 0.0030) and today's wipes (203 GB, mu = 0.15). " + BASE)
    return {"files": save(fig, out, "4_memory_cap"), "title": "The memory cap", "caption": caption,
            "quote": {"sigma": quote, "R_where_mu_equals_r_gb": {f"{k:g}": round(v, 3) for k, v in cross.items()}}}


def nci(out: Path) -> dict:
    """Figure 5: NCI's term nu = 2 alpha Lambda(K) / (kappa b_a) against K, linear below 2 d b_w e and quadratic
    above."""
    Ks = np.logspace(5, 10, 300)

    def nu(K: float) -> float:
        return 2 * (K / P["Gamma"]) * T.lam(P, K) / (P["kappa"] * P["b_a"])

    kink = 2 * P["d"] * P["b_w"] * P["e"]
    nu0, nu_kink = nu(P["K"]), nu(kink)
    model = T.evaluate(P | {"host_share": 0.0}, T.TIERS["T4"])["declared"]
    K_r = float(np.exp(np.interp(np.log(0.01), np.log([nu(K) for K in Ks]), np.log(Ks))))
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.13, top=0.87)
    ax.plot(Ks, [nu(K) for K in Ks], color=TERM["declared"], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e5, 1e10)
    ax.set_ylim(3e-8, 0.3)
    ax.axvline(kink, color="#bbbbbb", lw=1.6, ls=":", zorder=1)
    ax.plot(kink, nu_kink, "o", ms=14, mfc="white", mec=TERM["declared"], mew=3, zorder=4)
    ax.annotate(rf"kink at $K = 2\,d\,b_w e$ = {sci(kink)} bit ({kink / 8e6:.0f} MB)" "\n"
                rf"$\nu$ = {sci(nu_kink)}", (kink, nu_kink), xytext=(-24, 40), textcoords="offset points",
                ha="right", fontsize=19)
    ax.plot(P["K"], nu0, "o", ms=14, color=TERM["declared"], zorder=4)
    ax.annotate(rf"default $K$ = {sci(P['K'])} bit ({P['K'] / 8e6:.1f} MB): $\nu$ = {sci(nu0, 3)}", (P["K"], nu0),
                xytext=(24, -18), textcoords="offset points", fontsize=19, va="top")
    ax.text(3e5, nu(3e5) * 2.3, r"$\Lambda = e$: $\nu \propto K$", fontsize=21, color=TERM["declared"],
            rotation=slope_deg(ax, (1e5, nu(1e5)), (1e8, nu(1e8))), rotation_mode="anchor")
    ax.text(1.3e9, 1.2e-4, "$\\Lambda = K/(2\\,d\\,b_w)$:\n$\\nu \\propto K^2$", fontsize=21, color=TERM["declared"],
            va="top")
    ax.axhline(0.01, color=GRAY, lw=2, ls=(0, (6, 3)), zorder=2)
    ax.text(1.2e5, 0.0115, "design residual $r$ = 0.01", color=GRAY, fontsize=18, va="bottom")
    ax.plot(K_r, 0.01, "o", ms=10, color=GRAY, zorder=4)
    ax.annotate(rf"$\nu = r$ at $K$ = {sci(K_r)} bit", (K_r, 0.01), xytext=(-14, 14), textcoords="offset points",
                ha="right", fontsize=17, color=GRAY)
    ax.set_xlabel(r"$K$, novel bits per inference block (bit)")
    ax.set_ylabel(r"NCI's term $\nu = 2\alpha\Lambda(K)/(\kappa\, b_a)$")
    title(fig, r"NCI's term $\nu$: linear in $K$, then quadratic")
    caption = (
        "NCI's term nu = 2 alpha Lambda(K) / (kappa b_a) = 2 / (kappa rho_T), the declared computation's share at the "
        "thread bound, against K, the novel bits per inference block (log-log, 1e5 to 1e10 bit), with Gamma = 6.4e16 "
        "gates held fixed so alpha = K / Gamma grows with K. Lambda(K) = max(e, K / (2 d b_w)): below the kink at "
        f"K = 2 d b_w e = {kink:.3g} bit ({kink / 8e6:.0f} MB; d = 2048, b_w = 2.5, e = 4e4) nu is linear in K, "
        f"above it quadratic. At the default K = 2e7 bit, nu = {nu0:.3g} (tiers.evaluate's declared term at T4 with "
        f"host CPUs bounded: {model:.3g}); at the kink nu = {nu_kink:.3g}; nu reaches the design residual r = 0.01 at "
        f"K = {K_r:.3g} bit. b_a = 4, kappa = 0.655.")
    return {"files": save(fig, out, "5_nci_term"), "title": "NCI's term", "caption": caption,
            "quote": {"nu_default": nu0, "kink_bits": kink, "nu_kink": nu_kink, "K_where_nu_equals_0.01": K_r}}


def cheating(out: Path) -> dict:
    """Figure 6: the cheating share chi at T5 against the proving overhead per unit of work."""
    p0 = P | {"host_share": 0.0}
    xs = np.logspace(0, 14, 281)
    chi = [T.evaluate(p0 | {"overhead": x}, T.TIERS["T5"])["cheat"] for x in xs]
    marks = [(4.8e12, "sparse draws today", (-22, 12, "right", "bottom")),
             (1.7e10, "512 per statement", (22, -14, "left", "top")),
             (1.6e8, "batched throughput", (22, -14, "left", "top"))]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.13, top=0.87)
    ax.plot(xs, chi, color=TERM["cheat"], zorder=3)
    quote = {}
    for x, label, (dx, dy, ha, va) in marks:
        c = T.evaluate(p0 | {"overhead": x}, T.TIERS["T5"])["cheat"]
        quote[label] = {"overhead": x, "chi": c}
        ax.plot(x, c, "o", ms=15, color=TERM["cheat"], mec="white", mew=2.5, zorder=4)
        ax.annotate(f"{label}\n{sci(x)}: $\\chi$ = {sci(c)}", (x, c), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=19)
    nu = T.evaluate(p0, T.TIERS["T4"])["declared"]
    for y, txt, color in ((0.01, "design residual $r$ = 0.01", GRAY),
                          (nu, rf"NCI's term $\nu$ = {sci(nu)}", TERM["declared"])):
        ax.axhline(y, color=color, lw=2, ls=(0, (6, 3)), zorder=2)
        ax.text(1.5, y * 1.25, txt, color=color, fontsize=18, va="bottom")
    ax.axvline(P["replay"], color="#bbbbbb", lw=1.6, ls=":", zorder=1)
    ax.text(P["replay"] / 1.4, chi[0] * 6, f"below the replay cost ({sci(P['replay'])}),\nthe overhead stops mattering",
            fontsize=17, color=GRAY, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e14)
    ax.set_ylim(min(chi) / 5, 0.05)
    ax.xaxis.set_major_locator(LogLocator(numticks=15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: sci(v) if round(np.log10(v)) % 2 == 0 else ""))
    ax.set_xlabel("proving overhead per unit of step work (×)")
    ax.set_ylabel(r"cheating share $\chi$")
    title(fig, r"The cheating share $\chi$ falls with the proving overhead")
    caption = (
        "The cheating share chi at T5 (sampled proofs at share = 1% of compute) against the proving cost per unit of "
        "step work, log-log, from tiers.evaluate with host CPUs bounded (chi doesn't depend on r). The faulty blocks "
        "the sampling accepts number luck x ln(1/delta) / q with q = share / (overhead + replay), luck 8, "
        "delta = 2^-40, replay 1.1e5, step work 1.43e4 FLOP, 16 output bits per step. Marked: "
        + "; ".join(f"{k}, {v['overhead']:.2g}: chi = {v['chi']:.2g}" for k, v in quote.items())
        + f". Reference lines: the design residual r = 0.01 and NCI's term nu = {nu:.2g}. Below the replay cost the "
        "overhead stops mattering. " + BASE)
    return {"files": save(fig, out, "6_cheating_share"), "title": "The cheating share", "caption": caption,
            "quote": quote}


def decomposition(out: Path) -> dict:
    """Figure 7: eta's four parts per scenario at T5, as a dot plot on a log axis."""
    zero, lo, hi = 2e-8, 1e-8, 3.0
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, right=0.82, bottom=0.12, top=0.8)
    quote = {}
    keys = list(SCENARIOS)
    for i, key in enumerate(keys):
        y = len(keys) - 1 - i
        e = T.evaluate(P | SCENARIOS[key][1], T.TIERS["T5"])
        quote[key] = {t: e[t] for t in TERM} | {"eta": e["efficiency"], "D": 1 / e["efficiency"],
                                                 "memory_caps_residual": e["memory_caps_residual"]}
        ax.axhline(y, color="#ececec", lw=1.2, zorder=0)
        for t, color in TERM.items():
            v = e[t]
            mk = "D" if t == "residual" and e["memory_caps_residual"] else "o"
            ax.plot(v if v > 0 else zero, y, mk, ms=21, color=color, mfc=color if v > 0 else "white", mew=3, zorder=3)
        ax.plot([e["efficiency"]] * 2, [y - 0.3, y + 0.3], color="k", lw=4, zorder=4)
        ax.text(hi * 1.15, y, f"η = {e['efficiency']:.3g}\nD = {1 / e['efficiency']:.3g}", va="center", fontsize=19,
                clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.xaxis.set_major_locator(FixedLocator([zero] + [10.0 ** k for k in range(-7, 1)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v < 5e-8 else sci(v)))
    ax.text(3.6e-8, -0.6, "//", fontsize=20, ha="center", va="center", clip_on=False)
    ax.set_yticks(range(len(keys)), [f"{SCENARIOS[k][0]}\n$r$ = {T.residual_share(P | SCENARIOS[k][1]):g}"
                                     for k in reversed(keys)], fontsize=19)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"share of compute at T5 (log)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=17, color=TERM[t]) for t in TERM]
    handles += [plt.Line2D([], [], ls="", marker="D", ms=15, color=TERM["residual"]),
                plt.Line2D([], [], color="k", lw=4)]
    ax.legend(handles, ["$s$ unmonitored", "$r$ residual", r"$\nu$ NCI", r"$\chi$ cheating", r"$\mu$ (memory binds)",
                        r"$\eta$ total"], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize=18,
              handletextpad=0.3, columnspacing=1.2)
    title(fig, r"$\eta$'s four parts at T5")
    caption = (
        "eta's four parts at T5 for each scenario, on a log axis: s, the unmonitored share (0 in every scenario, drawn "
        "hollow at the axis break); the monitored residual r, or the memory cap mu where memory binds (diamond); NCI's "
        "term nu; the cheating share chi; the black bar is their sum eta, with D = 1/eta. A dot plot, not a stacked "
        "bar: on a log axis a stacked segment's length is set by the running sum, not its own value, so the small "
        "terms (nu about 1e-5, chi about 3e-4) would vanish or be exaggerated by stacking order. Values: "
        + "; ".join(f"{SCENARIOS[k][0]}: r or mu {q['residual']:.3g}, nu {q['declared']:.3g}, chi {q['cheat']:.3g}, "
                    f"eta {q['eta']:.3g}, D {q['D']:.3g}" for k, q in quote.items()) + ". " + BASE)
    return {"files": save(fig, out, "7_decomposition"), "title": "The four-term decomposition", "caption": caption,
            "quote": quote}


def schematic(out: Path) -> dict:
    """Figure 8: each mechanism, its tier, and the term of eta it dampens."""
    t4 = T.evaluate(P, T.TIERS["T4"])
    lam_in, nic = T.ingress(P, T.TIERS["T2"]), T.ingress(P, T.TIERS["T1"])
    R_wipe = t4["wipe_state_bits"] / T.GB
    mu = memory_cap(P, R_wipe)
    chi = T.evaluate(P | {"host_share": 0.0}, T.TIERS["T5"])["cheat"]
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    title(fig, "What each mechanism dampens")
    ax.text(3.15, 7.75, "mechanism (tier)", fontsize=19, color=GRAY, ha="center")
    ax.text(11.8, 7.75, "term of η it dampens", fontsize=19, color=GRAY, ha="center")
    mechs = [("NCI", "T1", 7.0), ("PoUW", "T1", 6.0), ("Network accounting", "T2", 5.0), ("Wipes", "T3", 4.0),
             ("Exhaustion", "T4", 3.0), ("Sampled proofs, 1%", "T5", 2.0)]
    terms = {
        "nu": (7.0, r"$\nu$", "NCI's term: declared work per bit read",
               rf"$2\alpha\Lambda(K)/(\kappa b_a)$ = {sci(t4['declared'])}", TERM["declared"]),
        "r": (6.0, "$r$", "monitored compute residual", "1% GPUs + 5% host CPUs at peak", TERM["residual"]),
        "lam": (5.0, r"$\lambda_{\rm in}$", "novel intake per server",
                rf"{sci(nic)} → {sci(lam_in)} bit/s ($\alpha F+\sigma+\tau$)", INTAKE),
        "mu": (3.5, r"$\mu$", "memory cap on $r$",
               rf"$4R_{{\rm wipe}}\lambda_{{\rm in}}/(\kappa d b_w b_a F)$ = {mu:.2g} at {R_wipe:.0f} GB",
               TERM["residual"]),
        "chi": (2.0, r"$\chi$", "cheating share", f"{sci(chi)} at today's overhead", TERM["cheat"]),
        "s": (0.95, "$s$", "unmonitored share", "no mechanism: trains at full efficiency", TERM["unmonitored"]),
    }
    edges = [(7.0, "nu"), (6.0, "r"), (5.0, "lam"), (4.0, "mu"), (3.0, "mu"), (2.0, "chi")]
    mx0, mx1, tx0, tx1, h = 0.6, 5.7, 8.6, 15.0, 0.72
    for name, tier, y in mechs:
        ax.add_patch(FancyBboxPatch((mx0, y - h / 2), mx1 - mx0, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="#f4f4f4", ec="#9a9a9a", lw=1.6))
        ax.text(mx0 + 0.3, y, name, fontsize=22, va="center")
        ax.text(mx1 - 0.3, y, tier, fontsize=20, va="center", ha="right", color=GRAY, fontweight="bold")
    for key, (y, sym, what, val, color) in terms.items():
        ax.add_patch(FancyBboxPatch((tx0, y - h / 2), tx1 - tx0, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="white", ec=color, lw=3.2, ls="--" if key == "s" else "-"))
        ax.text(tx0 + 0.55, y, sym, fontsize=27, va="center", ha="center", color=color)
        ax.text(tx0 + 1.15, y + 0.13, what, fontsize=19, va="center")
        ax.text(tx0 + 1.15, y - 0.2, val, fontsize=15, va="center", color="#444444")
    for y, key in edges:
        ty, color = terms[key][0], terms[key][4]
        ax.add_patch(FancyArrowPatch((mx1 + 0.08, y), (tx0 - 0.1, ty), arrowstyle="-|>,head_width=6,head_length=12",
                                     mutation_scale=1.6, lw=3, color=color, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((tx1 + 0.05, 5.0), (tx1 + 0.05, 3.5), connectionstyle="arc3,rad=-0.6",
                                 arrowstyle="-|>,head_width=6,head_length=12", mutation_scale=1.6, lw=2.6,
                                 color=INTAKE, shrinkA=0, shrinkB=0))
    ax.text(tx1 - 0.2, 4.25, r"$\mu \propto \lambda_{\rm in}$", fontsize=18, color=INTAKE, va="center", ha="right")
    pieces = [(r"$\eta\ \leq\ $", "k"), ("$s$", TERM["unmonitored"]), (r"$\ +\ \min($", "k"), ("$r$", TERM["residual"]),
              ("$,\\ $", "k"), (r"$\mu$", TERM["residual"]), (r"$)\ +\ $", "k"), (r"$\nu$", TERM["declared"]),
              (r"$\ +\ $", "k"), (r"$\chi$", TERM["cheat"]), (r"$,\quad D = 1/\eta$", "k")]
    eq = HPacker(children=[TextArea(t, textprops={"color": c, "fontsize": 27}) for t, c in pieces], pad=0, sep=0,
                 align="baseline")
    ax.add_artist(AnchoredOffsetbox(loc="lower left", child=eq, frameon=False, bbox_to_anchor=(0.6, 0.35),
                                    bbox_transform=ax.transData, borderpad=0))
    caption = (
        "Each mechanism on the left, with the tier that adds it, points to the term of the bound "
        "eta <= s + min(r, mu) + nu + chi it dampens (D = 1/eta). NCI bounds nu, the declared computation's share at "
        f"the thread bound ({t4['declared']:.2g} at the defaults); PoUW credits work and so bounds the monitored "
        "residual r (0.01 on the GPUs, plus 0.05 for host CPUs at peak); network accounting cuts the novel intake "
        f"lambda_in from line rate ({nic:.2g} bit/s) to alpha F + sigma + tau ({lam_in:.3g} bit/s); wipes and "
        "exhaustion bound the memory a window starts from, so the memory cap mu = 4 R_wipe lambda_in / "
        f"(kappa d b_w b_a F) ({mu:.2g} at R_wipe = {R_wipe:.0f} GB), which lambda_in also sets; sampled proofs at 1% "
        f"bound the cheating share chi ({chi:.2g} at today's 4.8e12 overhead). Nothing dampens s, the unmonitored "
        "share: it trains at full efficiency.")
    return {"files": save(fig, out, "8_schematic"), "title": "Mechanisms and the terms they dampen",
            "caption": caption, "quote": {"nu": t4["declared"], "lambda_in": lam_in, "line_rate": nic, "mu": mu,
                                          "chi": chi}}


FIGURES = {"1_stack": stack, "2_phase": phase, "3_exposure": exposure,
           "3b_exposure_memory": lambda out: exposure(out, "wipes", "3b_exposure_memory"), "4_memory_cap": memory,
           "5_nci_term": nci, "6_cheating_share": cheating, "7_decomposition": decomposition, "8_schematic": schematic}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    results = {name: fn(out) for name, fn in FIGURES.items()}
    (out / "figures.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    for name, r in results.items():
        print(f"{name}: {', '.join(r['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
