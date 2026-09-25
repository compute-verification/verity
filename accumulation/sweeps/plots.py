"""Figures from the sweep JSONL (``results/plots/*.png``).  Every plot skips missing values (rows whose bound is
``None``) and is skipped entirely -- with a message -- when no row has data for it.

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.plots [--out-dir accumulation/results]

1. ``p1_I_vs_Q_fit.png``                  P1: L and U vs Q at X = 5 MB, F = 1e12 with the fitted ``intercept + r Q`` lines,
                                          and the per-token view (always drawn)
2. ``p2_LU_per_token_vs_X_<model>.png``   L/token and U/token vs X (log-log), one colour per algorithm  (``--p2`` only)
3. ``p2_UL_vs_model_size.png``            U/L vs total parameters per algorithm (X = 5 MB)               (``--p2`` only)
4. ``p2_kappa_upper_<model>.png``         kappa_L / kappa_U per algorithm at X = 5 MB                      (``--p2`` only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accumulation.sweeps.run import finalize_row, row_key
from accumulation.sweeps.util import KAPPA_NATIVE_HBM
from accumulation.sweeps.util import PLOTS_DIR, RESULTS_DIR, dedupe, read_jsonl


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_rows(out_dir: Path, name: str) -> list[dict]:
    rows = [finalize_row(r) for r in read_jsonl(Path(out_dir) / name)]
    return [r for r in dedupe(rows, row_key) if not r.get("error")]


def _algs(rows):
    return sorted({r["alg"] for r in rows})


def plot_LU_vs_X(rows: list[dict], model: str, path: Path, F: int | None = None) -> bool:
    plt = _mpl()
    sel = [r for r in rows if r["model"] == model and (F is None or int(r["F"]) == F)]
    sel = [r for r in sel if r.get("L_per_token") is not None or r.get("U_per_token") is not None]
    if not sel:
        print(f"[plots] LU-vs-X: no data for {model}, skipped")
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, alg in enumerate(_algs(sel)):
        rr = sorted((r for r in sel if r["alg"] == alg), key=lambda r: r["X"])
        c = colors[i % len(colors)]
        xs = [r["X"] for r in rr if r.get("L_per_token")]
        ys = [r["L_per_token"] for r in rr if r.get("L_per_token")]
        if xs:
            ax.plot(xs, ys, "-o", color=c, label=f"{alg} L/token")
        xs = [r["X"] for r in rr if r.get("U_per_token")]
        ys = [r["U_per_token"] for r in rr if r.get("U_per_token")]
        if xs:
            ax.plot(xs, ys, "--s", color=c, mfc="none", label=f"{alg} U/token")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("X (bytes per replay unit)")
    ax.set_ylabel("bytes of runtime input per token")
    ax.set_title(f"L/token (solid) and U/token (dashed) vs X -- {model}" + (f", F={F:.0e}" if F else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_L_vs_Q(rows: list[dict], path: Path, X: int = int(5e6), F: int = int(1e12), model: str = "llama3-8b",
                fits: dict | None = None) -> bool:
    """P1: left panel ``I(Q)`` (bytes, log-log) for ``L`` and ``U`` of every algorithm at ``(model, X, F)`` with the
    fitted lines ``intercept + r Q`` (:func:`accumulation.sweeps.analysis.p1_fits`); right panel ``I/token`` vs ``Q``
    (which flattens to ``r``)."""
    plt = _mpl()
    sel = [r for r in rows if r["model"] == model and int(r["X"]) == X and int(r["F"]) == F
           and (r.get("L") is not None or r.get("U") is not None)]
    if not sel:
        print("[plots] P1 I-vs-Q: no data, skipped")
        return False
    fl = (fits or {}).get("fits", [])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {alg: f"C{i}" for i, alg in enumerate(_algs(sel))}
    for alg in _algs(sel):
        rr = sorted((r for r in sel if r["alg"] == alg), key=lambda r: r["tokens"])
        for bound, mk, ls in (("L", "o", "-"), ("U", "s", "--")):
            pts = [(r["tokens"], r[bound]) for r in rr if r.get(bound) is not None]
            if not pts:
                continue
            lab = f"{alg} {bound}" + (" (provisional)" if bound == "L" else "")
            ax.plot([q for q, _ in pts], [y for _, y in pts], marker=mk, ls="none", mfc="none" if bound == "U" else None,
                    color=colors[alg], label=lab)
            ax2.plot([q for q, _ in pts], [y / q for q, y in pts], marker=mk, ls=ls, mfc="none" if bound == "U" else None,
                     color=colors[alg], label=lab)
            f = next((f for f in fl if f["alg"] == alg and f["model"] == model and int(f["X"]) == X and int(f["F"]) == F
                      and f["bound"] == bound and f.get("seq") == rr[0]["seq"]), None)
            if f is not None:
                qs = [q for q, _ in pts]
                grid = [qs[0] * (qs[-1] / qs[0]) ** (i / 60) for i in range(61)]
                ax.plot(grid, [f["intercept"] + f["slope"] * q for q in grid], ls=ls, color=colors[alg], alpha=0.7, lw=1,
                        label=f"  fit: {f['intercept']:.3g} + {f['slope']:.4g} B/tok x Q")
                ax2.axhline(f["slope"], ls=":", color=colors[alg], alpha=0.5, lw=1)
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("tokens per weight version, Q")
        a.grid(True, which="both", alpha=0.3)
    ax.set_ylabel("runtime-input bound I(Q) [bytes]")
    ax.set_title(f"{model}: L and U vs Q at X = {X:.0e} B, F = {F:.0e} (lines = OLS fits)")
    ax2.set_ylabel("bytes per token (I / Q)")
    ax2.set_title("per-token view (dotted = fitted recurring term r)")
    ax.legend(fontsize=7)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_UL_vs_size(rows: list[dict], path: Path, X: int = int(5e6)) -> bool:
    plt = _mpl()
    sel = [r for r in rows if int(r["X"]) == X and r.get("U_over_L") is not None]
    if not sel:
        print("[plots] U/L-vs-size: no data, skipped")
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for alg in _algs(sel):
        rr = sorted((r for r in sel if r["alg"] == alg), key=lambda r: r["params_total"])
        ax.plot([r["params_total"] for r in rr], [r["U_over_L"] for r in rr], "-o", label=alg)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total parameters")
    ax.set_ylabel("U / L")
    ax.set_title(f"bound gap vs model size at X = {X:.0e} B")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_kappa_bar(rows: list[dict], model: str, path: Path, X: int = int(5e6), kappa_native: float = KAPPA_NATIVE_HBM) -> bool:
    plt = _mpl()
    sel = [r for r in rows if r["model"] == model and int(r["X"]) == X and r.get("kappa_L") is not None]
    if not sel:
        print(f"[plots] kappa bar: no data for {model}, skipped")
        return False
    sel = sorted(sel, key=lambda r: -r["kappa_L"])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["alg"] for r in sel]
    ax.bar(labels, [r["kappa_L"] for r in sel], color="tab:blue", label="kappa_L = ADW_matmul_macs / L (certified ceiling)")
    if any(r.get("kappa_U") for r in sel):
        ax.bar(labels, [r.get("kappa_U") or 0 for r in sel], width=0.4, color="tab:orange",
               label="kappa_U = ADW_matmul_macs / U")
    ax.axhline(kappa_native, color="k", ls="--", label=f"kappa_native (H100 HBM roofline) = {kappa_native:.0f} MAC/B")
    ax.set_yscale("log")
    ax.set_ylabel("MAC per byte of runtime input")
    ax.set_title(f"security efficiency ceiling per algorithm -- {model}, X = {X:.0e} B")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def make_all(out_dir: Path = RESULTS_DIR, plots_dir: Path = PLOTS_DIR, kappa_native: float = KAPPA_NATIVE_HBM,
             p2_plots: bool = False) -> list[Path]:
    """P1 figure always; the P2 figures only with ``p2_plots`` (on hold until the v2 lower bound is validated)."""
    import json
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    p1 = load_rows(out_dir, "sweep_p1.jsonl")
    fit_path = Path(out_dir) / "p1_fit.json"
    fits = json.loads(fit_path.read_text()) if fit_path.exists() else None
    made = []
    p = plots_dir / "p1_I_vs_Q_fit.png"
    if plot_L_vs_Q(p1, p, fits=fits):
        made.append(p)
    if p2_plots:
        p2 = load_rows(out_dir, "sweep_p2.jsonl")
        for model in sorted({r["model"] for r in p2}):
            p = plots_dir / f"p2_LU_per_token_vs_X_{model}.png"
            if plot_LU_vs_X(p2, model, p):
                made.append(p)
        p = plots_dir / "p2_UL_vs_model_size.png"
        if plot_UL_vs_size(p2, p):
            made.append(p)
        for model in sorted({r["model"] for r in p2}):
            p = plots_dir / f"p2_kappa_upper_{model}.png"
            if plot_kappa_bar(p2, model, p, kappa_native=kappa_native):
                made.append(p)
    print(f"[plots] wrote {len(made)} figure(s) to {plots_dir}")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--kappa-native", type=float, default=KAPPA_NATIVE_HBM)
    ap.add_argument("--p2", action="store_true", help="also draw the P2 figures (held until L v2 is validated)")
    a = ap.parse_args(argv)
    make_all(a.out_dir, a.plots_dir or (a.out_dir / "plots"), a.kappa_native, p2_plots=a.p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
