"""Figures A-E for the F/G policy model (``results/plots/``; PNG + the JSONL of points behind each figure).

Every figure is drawn from the analytic model (``analytic.py``: steady-state area-cap block = dashed, single-step block
= dotted grey, labelled *analytic*) and overlaid with measured ``L`` / ``U`` from ``coarse.jsonl`` (solid / filled,
labelled *measured*) wherever rows exist; cells marked ``status = "pod"`` are analytic fills.  The legend and the JSONL
``kind`` field say which is which.

A. ``Q_inf`` vs ``1/rho_cert`` and ``1/rho_ach`` (session-size frontier, log y).
B. Heatmap ``(F_hat, G_hat) -> log10(1/rho_cert)``: contours follow ``min(1.5 F_hat, G_hat)``; the one-RU honest-session
   region (``F_hat >= 1``, ``G_hat >= 1``) and the ``G_hat = 1.5 F_hat`` line (G irrelevant above it); the ``G`` axis is also
   labelled in parallel branches / batched sessions.
C. ``Q_step`` vs ``b_train`` (L and U bands) per ``G_hat``: flat steady state, step up where ``Q_step > u*`` (``c_W`` 2 -> 6).
D. Model scale vs ``1/rho_cert`` with the analytic ``sqrt(d P / Q_inf)`` trend.
E. ``K`` vs ``I(K) / (K Q_step)``: transient (rows only) up to the knee, then the steady-state slope ``r``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from accumulation.sweeps import analytic
from accumulation.sweeps.calibration import load_calibration
from accumulation.sweeps.coarse_report import _index, k_fits, load_rows, policy_record
from accumulation.sweeps.coarse_run import AXES, DEFAULT_POLICY, SYN_MODELS
from accumulation.sweeps.synthetic import model_cfg
from accumulation.sweeps.util import RESULTS_DIR

MB = 1_000_000
PLOTS = RESULTS_DIR / "plots"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _dump(path: Path, pts: list[dict]) -> None:
    with path.open("w") as fh:
        for p in pts:
            fh.write(json.dumps(p, default=str) + "\n")


def _measured(idx, model, Q_inf, F_hat, G_hat, Q_step) -> dict | None:
    rec = policy_record(idx, {}, model, Q_inf, float(F_hat), float(G_hat), Q_step)
    tr = rec["training"]
    if rec["status"].startswith("pod") or (rec["penalty_cert"] is None and rec["penalty_ach"] is None and tr["U_b"] is None and tr["L_b"] is None):
        return None
    return rec


def _est(model, Q, Fh, Gh, Qs, calib):
    return analytic.estimate_point(model, Q_inf=Q, F_hat=Fh, G_hat=Gh, Q_step=Qs, calib=calib)


# ------------------------------------------------------------------------------------------------------- A

def fig_a(idx, calib, out: Path, model="llama3-8b") -> list[dict]:
    d = DEFAULT_POLICY
    pts = []
    for Q in AXES["qinf"]:
        e = _est(model, Q, d["F_hat"], d["G_hat"], d["Q_step_mult"] * Q, calib)
        pts.append({"kind": "analytic", "model": model, "Q_inf": Q, "penalty": e["penalty_est"], "b_train": e["b_train_est"], "regime": e["regime"],
                    "penalty_single": e["penalty_single_est"], "b_single": e["b_single_est"]})
        for mdl in (model, "llama3-1b"):
            m = _measured(idx, mdl, Q, d["F_hat"], d["G_hat"], d["Q_step_mult"] * Q)
            if m:
                pts.append({"kind": "measured", "model": mdl, "Q_inf": Q, "penalty_cert": m["penalty_cert"], "penalty_ach": m["penalty_ach"],
                            "b_train_L": m["training"]["L_b"], "b_train_U": m["training"]["U_b"], "status": m["status"]})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    an = [p for p in pts if p["kind"] == "analytic"]
    ax.plot([p["Q_inf"] for p in an], [p["penalty"] for p in an], "o--", mfc="none", color="k", label=f"analytic steady-state block ({model})")
    ax.plot([p["Q_inf"] for p in an], [p["penalty_single"] for p in an], "o:", mfc="none", color="gray", label=f"analytic single-step block ({model})")
    for mdl, mk, col in ((model, "s", "C0"), ("llama3-1b", "^", "C2")):
        me = [p for p in pts if p["kind"] == "measured" and p["model"] == mdl]
        if me:
            xs = [p["Q_inf"] for p in me]
            if any(p["penalty_cert"] for p in me):
                ax.plot(xs, [p["penalty_cert"] for p in me], mk + "-", color=col, label=f"measured 1/rho_cert ({mdl}, certified L)")
            if any(p["penalty_ach"] for p in me):
                ax.plot(xs, [p["penalty_ach"] for p in me], mk + ":", color=col, mfc="none", label=f"measured 1/rho_ach ({mdl}, constructed U)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("honest session length Q_inf (tokens)\nF = 1.5 fwd(Q_inf), G = 4 fwd(Q_inf), Q_step = 16 Q_inf, seq = Q_inf")
    ax.set_ylabel("training penalty 1/rho\n(input bytes per MAC relative to honest inference)")
    ax.set_title("A. Session-size frontier")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figA_qinf_penalty.png", dpi=130)
    plt.close(fig)
    _dump(out / "figA_qinf_penalty.jsonl", pts)
    return pts


# ------------------------------------------------------------------------------------------------------- B

def fig_b(idx, calib, out: Path, model="llama3-8b") -> list[dict]:
    d = DEFAULT_POLICY
    Q, Qs = d["Q_inf"], d["Q_step_mult"] * d["Q_inf"]
    Fs = [10 ** (i / 6) for i in range(-3, 19)]      # 0.32 .. 1000
    Gs = [10 ** (i / 6) for i in range(-3, 19)]
    grid = [[_est(model, Q, f, g, Qs, calib) for f in Fs] for g in Gs]
    pts = [{"kind": "analytic", "F_hat": e["F_hat"], "G_hat": e["G_hat"], "penalty": e["penalty_est"], "regime": e["regime"], "b_train": e["b_train_est"],
            "A": e["A"], "penalty_single": e["penalty_single_est"]} for row in grid for e in row]
    meas = []
    for f in AXES["heat_F"]:
        for g in AXES["heat_G"]:
            m = _measured(idx, model, Q, f, g, Qs)
            if m and (m["penalty_cert"] or m["penalty_ach"]):
                meas.append({"kind": "measured", "F_hat": f, "G_hat": g, "penalty_cert": m["penalty_cert"], "penalty_ach": m["penalty_ach"], "status": m["status"]})
    pts += meas
    plt = _plt()
    import numpy as np
    Z = np.array([[math.log10(e["penalty_est"]) if e["penalty_est"] else np.nan for e in row] for row in grid])
    fig, ax = plt.subplots(figsize=(8, 6.2))
    pc = ax.pcolormesh(Fs, Gs, Z, shading="nearest", cmap="viridis")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("log10(1/rho_cert)  [analytic steady-state estimate; measured cells = white dots]")
    ax.contour(Fs, Gs, Z, levels=8, colors="white", linewidths=0.6, alpha=0.7)
    ax.plot(Fs, [1.5 * f for f in Fs], "r--", lw=1.2, label="G_hat = 1.5 F_hat (G irrelevant above: F-limited)")
    ax.axvline(1.0, color="orange", lw=1.5)
    ax.axhline(1.0, color="orange", lw=1.5, label="honest session = one RU (F_hat >= 1, G_hat >= 1)")
    for m in meas:
        ax.plot(m["F_hat"], m["G_hat"], "o", color="white", mec="k", ms=7)
        v = m["penalty_cert"] or m["penalty_ach"]
        ax.annotate(f"{math.log10(v):.2f}{'' if m['penalty_cert'] else ' (U)'}", (m["F_hat"], m["G_hat"]), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F_hat = F / fwd(Q_inf)  (upstream-work cap; an area cap ~1.5 F on dW-reducing units)")
    ax.set_ylabel("G_hat = G / fwd(Q_inf)\n(total-work cap = parallel branches / batched sessions per RU)")
    ax.set_title(f"B. Policy heatmap, {model}, Q_inf = {Q}, Q_step = {Qs}\ncontours follow min(1.5 F_hat, G_hat): G-limited below the red line, F-limited above",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figB_heatmap_penalty.png", dpi=130)
    plt.close(fig)
    _dump(out / "figB_heatmap_penalty.jsonl", pts)
    return pts


# ------------------------------------------------------------------------------------------------------- C

def fig_c(idx, calib, out: Path, model="llama3-8b") -> list[dict]:
    d = DEFAULT_POLICY
    Q, Fh = d["Q_inf"], d["F_hat"]
    pts = []
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    qs_fine = [2 ** k for k in range(12, 23)]
    for i, g in enumerate(AXES["heat_G"]):
        col = f"C{i}"
        an = [_est(model, Q, Fh, g, qs, calib) for qs in qs_fine]
        pts += [{"kind": "analytic", "G_hat": g, "Q_step": e["Q_step"], "b_train": e["b_train_est"], "b_subunit": e["b_subunit_est"], "u_star": e["u_star"],
                 "c_W": e["c_W"], "regime": e["regime"], "b_single": e["b_single_est"]} for e in an]
        ax.plot(qs_fine, [e["b_train_est"] / MB for e in an], "--", color=col, alpha=0.9, label=f"analytic steady state, G_hat={g:g}")
        if i == 0:
            ax.plot(qs_fine, [e["b_single_est"] / MB for e in an], ":", color="gray", alpha=0.8, label="analytic single-step block (G_hat=1)")
        me = []
        for mdl in (model, "llama3-1b"):
            for qs in AXES["qstep"]:
                m = _measured(idx, mdl, Q, Fh, g, qs)
                if m and (m["training"]["U_b"] or m["training"]["L_b"]):
                    me.append({"kind": "measured", "model": mdl, "G_hat": g, "Q_step": qs, "b_train_L": m["training"]["L_b"], "b_train_U": m["training"]["U_b"],
                               "status": m["status"]})
        pts += me
        for mdl, mk in ((model, "s"), ("llama3-1b", "^")):
            mm = [p for p in me if p["model"] == mdl]
            if mm:
                xs = [p["Q_step"] for p in mm]
                U = [p["b_train_U"] / MB if p["b_train_U"] else float("nan") for p in mm]
                L = [p["b_train_L"] / MB if p["b_train_L"] else float("nan") for p in mm]
                ax.plot(xs, U, mk + "-", color=col, label=f"measured U {mdl}, G_hat={g:g}")
                ax.plot(xs, L, mk + "-", color=col, mfc="none", label=f"measured L {mdl}, G_hat={g:g}")
                ax.fill_between(xs, L, U, color=col, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"tokens per optimiser step Q_step ({model}, Q_inf = {Q}, F_hat = {Fh:g}; measured = steady-state marginal of the K sweep)")
    ax.set_ylabel("b_train (MB / token)")
    ax.set_title("C. Step size vs training runtime input (flat steady state; step up where Q_step > u*, c_W 2 -> 6)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figC_qstep_btrain.png", dpi=130)
    plt.close(fig)
    _dump(out / "figC_qstep_btrain.jsonl", pts)
    return pts


# ------------------------------------------------------------------------------------------------------- D

def fig_d(idx, calib, out: Path) -> list[dict]:
    d = DEFAULT_POLICY
    Q, Qs = d["Q_inf"], d["Q_step_mult"] * d["Q_inf"]
    models = list(AXES["scale"]) + list(AXES["moe"]) + list(SYN_MODELS)
    pts = []
    for m in models:
        cfg = model_cfg(m)
        e = _est(m, Q, d["F_hat"], d["G_hat"], Qs, calib)
        fam = "dense" if m.startswith("llama") else ("moe" if cfg.experts != 1 else "synthetic")
        pts.append({"kind": "analytic", "model": m, "family": fam, "P_params": cfg.total_params(), "d": cfg.d, "layers": cfg.layers,
                    "penalty": e["penalty_est"], "b_train": e["b_train_est"], "trend": math.sqrt(cfg.d * e["P"] / Q), "penalty_single": e["penalty_single_est"]})
        mm = _measured(idx, m, Q, d["F_hat"], d["G_hat"], Qs)
        if mm:
            pts.append({"kind": "measured", "model": m, "family": fam, "P_params": cfg.total_params(), "d": cfg.d, "layers": cfg.layers,
                        "penalty_cert": mm["penalty_cert"], "penalty_ach": mm["penalty_ach"], "b_train_L": mm["training"]["L_b"],
                        "b_train_U": mm["training"]["U_b"], "status": mm["status"]})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ref = next(p for p in pts if p["kind"] == "analytic" and p["model"] == "llama3-8b")
    for fam, mk in (("dense", "o"), ("moe", "D"), ("synthetic", "^")):
        an = [p for p in pts if p["kind"] == "analytic" and p["family"] == fam]
        ax.scatter([p["P_params"] for p in an], [p["penalty"] for p in an], marker=mk, facecolors="none", edgecolors="k", s=60, label=f"analytic steady state ({fam})")
        for p in an:
            ax.annotate(p["model"], (p["P_params"], p["penalty"]), textcoords="offset points", xytext=(4, 3), fontsize=6, color="gray")
        me = [p for p in pts if p["kind"] == "measured" and p["family"] == fam]
        if any(p["penalty_cert"] for p in me):
            ax.scatter([p["P_params"] for p in me if p["penalty_cert"]], [p["penalty_cert"] for p in me if p["penalty_cert"]], marker=mk, color="C0", s=60,
                       label=f"measured 1/rho_cert ({fam})")
        if any(p["penalty_ach"] for p in me):
            ax.scatter([p["P_params"] for p in me if p["penalty_ach"]], [p["penalty_ach"] for p in me if p["penalty_ach"]], marker=mk,
                       facecolors="none", edgecolors="C0", s=60, label=f"measured 1/rho_ach ({fam})")
    dense = sorted([p for p in pts if p["kind"] == "analytic" and p["family"] == "dense"], key=lambda p: p["P_params"])
    xs = [p["P_params"] for p in dense]
    ax.plot(xs, [ref["penalty"] * p["trend"] / ref["trend"] for p in dense], "k:", lw=1, label="analytic trend  1/rho ∝ sqrt(d P / Q_inf)  (pinned at 8B)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"model parameters P (Q_inf = {Q}, F_hat = {d['F_hat']:g}, G_hat = {d['G_hat']:g}, Q_step = {Qs})")
    ax.set_ylabel("training penalty 1/rho")
    ax.set_title("D. Model scale vs certified penalty (70B+/MoE training = analytic pod fill)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figD_scale_penalty.png", dpi=130)
    plt.close(fig)
    _dump(out / "figD_scale_penalty.jsonl", pts)
    return pts


# ------------------------------------------------------------------------------------------------------- E

def fig_e(idx, calib, out: Path, model="llama3-8b") -> list[dict]:
    d = DEFAULT_POLICY
    Q, Fh, Gh = d["Q_inf"], d["F_hat"], d["G_hat"]
    pts = []
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks = list(range(1, 49))
    for j, (Qs, Kmeas) in enumerate(((Q, AXES["ksweep"]), (4 * Q, AXES["ksweep_32k"]))):
        e = _est(model, Q, Fh, Gh, Qs, calib)
        curve = analytic.k_curve(d=e["d"], layers=e["layers"], w_tl=e["w_tl"], F=e["F"], G=e["G"], Q_step=Qs, b_ss=e["b_train_est"], P=e["P"], Ks=Ks)
        pts += [{"kind": "analytic", "Q_step": Qs, **c, "K_knee_pred": e["K_knee"], "K_m2_pred": e["K_m2"], "b_steady": e["b_train_est"]} for c in curve]
        ax.plot(Ks, [c["I_per_token"] / MB for c in curve], "--", color=f"C{j}", alpha=0.9,
                label=f"analytic min(transient, steady) + P/(K Q_step), Q_step={Qs}; steady r = {e['b_train_est'] / MB:.2f} MB/tok")
        tr_m2 = [(K, 4 * e["d"] * e["layers"] / 2 + e["P"] / (K * Qs)) for K in Ks if K <= e["K_m2"]]
        if tr_m2:
            ax.plot([k for k, _ in tr_m2], [v / MB for _, v in tr_m2], "-.", color=f"C{j}", alpha=0.6, label=f"analytic 2-layer chunk transient (4dL/2 + P/(K Q_step)), Q_step={Qs}")
        ax.axvline(e["K_knee"], color=f"C{j}", ls=":", alpha=0.6)
        ax.annotate(f"pred. knee K≈{e['K_knee']:.1f}\n(2-layer chunk fails K>{e['K_m2']:.1f})", (e["K_knee"], 3.0 - 1.5 * j),
                    fontsize=7, color=f"C{j}")
        fits = k_fits(idx, model, Q, float(Fh), float(Gh), (Qs,), Ks=Kmeas)
        f = fits[0] if fits else None
        if f and f["K"]:
            for X, mk, fill in (("U", "s", True), ("L", "v", False)):
                me = [(K, v[f"{X}_avg"]) for K, v in sorted(f["K"].items()) if v.get(f"{X}_avg") is not None]
                if me:
                    lab = f"measured {X}(K)/(K Q_step), Q_step={Qs}"
                    ff = f["fits"].get(X)
                    if ff:
                        lab += f"; steady r_{X} = {ff['b_per_token'] / MB:.3g} MB/tok" + (f" (post-knee K>={ff['knee_K']})" if ff.get("knee_K") else "")
                    ax.plot([k for k, _ in me], [v / MB for _, v in me], mk + "-", color=f"C{j}", mfc=(f"C{j}" if fill else "none"), label=lab)
                    pts += [{"kind": "measured", "Q_step": Qs, "K": k, "bound": X, "I_per_token": v} for k, v in me]
            for X in ("L", "U"):
                if f["fits"].get(X):
                    ff = f["fits"][X]
                    pts.append({"kind": "fit", "Q_step": Qs, "bound": X, **{k: v for k, v in ff.items()}})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"chained SGD steps K ({model}, Q_inf = {Q}, F_hat = {Fh:g}, G_hat = {Gh:g})")
    ax.set_ylabel("I(K) / (K Q_step)  (MB / token)")
    ax.set_title("E. K sweep: transient (rows only) up to the knee, then the steady-state slope r", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "figE_k_steady_state.png", dpi=130)
    plt.close(fig)
    _dump(out / "figE_k_steady_state.jsonl", pts)
    return pts


# ------------------------------------------------------------------------------------------------------- F

def fig_f(idx, calib, out: Path) -> list[dict]:
    """Q_roll vs bytes/token of dynamic rollout generation (forward-nonfixed) for the four dense models: U (filled), L (open),
    the weight-loading transient P/Q_roll (dotted) and the 4 B/token inference reference."""
    from accumulation.sweeps.final_report import DENSE, SHORT, rollout_steady
    pts = []
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, m in enumerate(DENSE):
        col = f"C{i}"
        cur = rollout_steady(idx, m)["curve"]
        for c in cur:
            pts.append({"kind": "measured" if c["U_b"] is not None else "pending", "model": m, "Q_roll": c["Q_roll"], "U_b": c["U_b"], "L_b": c["L_b"],
                        "P_over_Q": c["P_over_Q"], "status": c["status"]})
        me = [c for c in cur if c["U_b"] is not None]
        if me:
            ax.plot([c["Q_roll"] for c in me], [c["U_b"] / MB for c in me], "s-", color=col, label=f"{SHORT[m]} U/Q (measured)")
            if any(c["L_b"] for c in me):
                ax.plot([c["Q_roll"] for c in me if c["L_b"]], [c["L_b"] / MB for c in me if c["L_b"]], "v--", color=col, mfc="none", label=f"{SHORT[m]} L/Q (certified)")
            P = next((c["row"]["P"] for c in me if c["row"] and c["row"].get("P")), None)
            if P:
                qs = [c["Q_roll"] for c in cur]
                ax.plot(qs, [P / q / MB for q in qs], ":", color=col, alpha=0.7, label=f"{SHORT[m]} weight transient P/Q_roll")
    ax.axhline(4 / MB, color="k", lw=1, ls="-.", label="inference reference 4 B/tok")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"rollout volume Q_roll (tokens; Q_inf = {DEFAULT_POLICY['Q_inf']}, F_hat = {DEFAULT_POLICY['F_hat']:g}, G_hat = {DEFAULT_POLICY['G_hat']:g})")
    ax.set_ylabel("runtime input per generated token (MB / token)")
    ax.set_title("F. Dynamic rollout (forward-nonfixed): P/Q transient, then a positive floor", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figF_rollout.png", dpi=130)
    plt.close(fig)
    _dump(out / "figF_rollout.jsonl", pts)
    return pts


def make_all(out_dir: Path = RESULTS_DIR, log=print) -> dict:
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out_dir) if (out_dir / "coarse.jsonl").exists() else []
    idx = _index(rows)
    calib = load_calibration(out_dir)
    res = {}
    for name, fn in (("A", fig_a), ("B", fig_b), ("C", fig_c), ("D", fig_d), ("E", fig_e), ("F", fig_f)):
        pts = fn(idx, calib, plots)
        n_meas = sum(1 for p in pts if p["kind"] == "measured")
        res[name] = {"n_analytic": sum(1 for p in pts if p["kind"].startswith("analytic")), "n_measured": n_meas}
        log(f"[plots] fig {name}: {res[name]['n_analytic']} analytic points, {n_meas} measured")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="figures A-E (analytic + measured overlays)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    make_all(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
