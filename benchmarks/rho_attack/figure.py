"""The figure: capability against S for each training goal, with rho_attack(S) and the bound's rho(S) on the same S axis,
for DeepSeek-V4-Pro and Kimi K2 at matched c.

    python3 figure.py RUN.json [RUN.json ...] --out DIR [--precision fp8]

reads the curves of `rho_attack.py` runs (any that hold the two configs) and writes `capability_vs_rho_<precision>.png`
and `.svg`. Capability is placed at S as each experiment counts its trained state (an adapter's own bits); "full" sits at
the target model's full fine-tune, its parameters at the measured beta_w, where the rho curves end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KB = 8_000
CONFIGS = ("deepseek-v4-pro", "kimi-k2")
#: 10 kB and the deck's calibration, k = 2.5 MB (1 MB sits 0.4 decades from it and overlaps its curves)
K_LINES = {"10kB": 10 * KB, "2.5MB": 2500 * KB}
K_COLORS = {"10kB": "tab:blue", "2.5MB": "tab:red"}
FULL = None  # placed at the target model's full-training S

#: goal -> [(series, [(S bits or FULL, capability)])]: the coordinator's capability data, fractions of the full-training
#: gain (A2, A3) or recall of H (B3).
GOALS = {
    "RL elicitation (A2, Qwen2.5-1.5B)": [
        ("GSM8K Instruct", [(272, 0.07), (9920, 0.62), (78400, 0.82), (802816, 0.84), (295436288, 0.83), (FULL, 1.0)]),
        ("GSM8K Base", [(272, 0.31), (9920, 0.87), (78400, 0.95), (FULL, 1.0)]),
        ("GSM8K Base, greedy", [(272, 0.81)]),
        ("MATH Instruct", [(272, 0.22), (9920, 0.86), (78400, 0.97), (802816, 1.03)]),
    ],
    "new skill: cipher (A3, Qwen2.5-0.5B)": [
        ("H = 88 bit", [(272, 0.0), (10080, 0.42), (80000, 1.0)]),
        ("H = 88 bit, RL", [(10080, 0.71)]),
        ("H = 2.4 kbit", [(80000, 0.06), (800000, 0.985)]),
        ("H = 2.4 kbit, RL", [(80000, 0.22)]),
        ("H = 62 kbit", [(80000, 0.0), (800000, 0.0), (144e6, 0.90), (FULL, 1.0)]),
    ],
    "facts (B3, Qwen2.5-0.5B, recall of H)": [
        ("N = 200, H = 8.9 kbit (90 epochs)", [(9 * 8.9e3, 0.13), (27 * 8.9e3, 0.99)]),
        ("N = 2,000, H = 89 kbit (12 epochs)", [(100 * 89e3, 0.1), (1580 * 89e3, 0.76), (FULL, 1.0)]),
    ],
}
#: the training run each goal's rho curves price: RL is GRPO (honest 4f per token), the others SFT (3f)
WORKLOAD = {goal: ("grpo" if goal.startswith("RL") else "sft") for goal in GOALS}
#: points that are upper bounds ("under 0.1") or a range midpoint
UPPER = {("N = 2,000, H = 89 kbit (12 epochs)", 100 * 89e3)}
RANGE = {("H = 2.4 kbit", 800000): (0.97, 1.0)}


def load(paths: list[str], precision: str) -> dict[tuple[str, str, str], dict]:
    curves = {}
    for p in paths:
        for cv in json.loads(Path(p).read_text())["curves"]:
            if cv["config"] in CONFIGS and cv["c_mode"] == "eta1" and cv["precision"] == precision:
                curves[(cv["config"], cv["k_label"], cv.get("workload", "sft"))] = cv
    return curves


def draw(curves: dict, out: Path, precision: str, inputs_note: str) -> list[Path]:
    """One column per goal on one S axis: capability on top, then rho for each config (the best attack found with the
    recipe kept solid, with recipe changes dashed, the bound's rho(S) dotted, its full-training count at beta_gx a filled
    star and at beta_v a hollow one), k dashed through all rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 1 + len(CONFIGS)
    fig, axes = plt.subplots(rows, len(GOALS), figsize=(6.0 * len(GOALS), 3.4 * rows), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [1.15] + [1.0] * len(CONFIGS)})
    nan = lambda xs: np.array([x if x is not None else np.nan for x in xs], dtype=float)  # noqa: E731
    markers = "osD^vP*X"
    s_full = {cfg: max(cv["S"][-1] for (c, _, _), cv in curves.items() if c == cfg) for cfg in CONFIGS}
    s_cap = max(s_full.values())
    for j, (goal, series) in enumerate(GOALS.items()):
        ax = axes[0][j]
        for n, (name, pts) in enumerate(series):
            xs = np.array([s_cap if s is FULL else s for s, _ in pts], dtype=float)
            ys = np.array([y for _, y in pts])
            line = ax.plot(xs, ys, marker=markers[n % len(markers)], ms=5, lw=1.4, color=f"C{(n + 3) % 10}",
                           label=name, zorder=5)[0]
            for (s, y), x in zip(pts, xs):
                if (name, s) in UPPER:
                    ax.annotate("", xy=(x, 0.0), xytext=(x, y),
                                arrowprops={"arrowstyle": "->", "color": line.get_color(), "lw": 0.8})
                if (name, s) in RANGE:
                    ax.plot([x, x], list(RANGE[(name, s)]), color=line.get_color(), lw=2.5)
                if s is FULL:
                    ax.annotate("full", (x, y), textcoords="offset points", xytext=(-16, 5), fontsize=6)
        ax.set_ylim(-0.05, 1.12)
        ax.set_ylabel("capability (fraction of full)")
        ax.set_title(f"{goal}", fontsize=10)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        for i, cfg in enumerate(CONFIGS, start=1):
            rx = axes[i][j]
            for k_label in K_LINES:
                cv = curves.get((cfg, k_label, WORKLOAD[goal]))
                if cv is None:
                    continue
                col = K_COLORS[k_label]
                S = np.array(cv["S"])
                rx.plot(S, nan(cv["attack_exact"]), "-", color=col, lw=2.0,
                        label=f"execution upper bound (best attack found, recipe kept), k = {k_label}")
                rx.plot(S, nan(cv["attack"]), "--", color=col, lw=1.0,
                        label=f"execution upper bound (best attack found, with recipe changes), k = {k_label}")
                rx.plot(S, cv["bound"], ":", color=col, lw=1.8,
                        label=f"slowdown lower bound (beta_gx count, conditional), k = {k_label}")
                rx.scatter([S[-1]], [cv["full_gx"]], marker="*", color=col, s=90, zorder=6,
                           label=f"slowdown lower bound at full training (beta_gx count), k = {k_label}")
                rx.scatter([S[-1]], [cv["full"]], marker="*", facecolors="none", edgecolors=col, s=90, zorder=6,
                           label=f"the same with the beta_v count, k = {k_label}")
            rx.set_yscale("log")
            rx.set_ylabel(f"rho, {cfg}")
            rx.set_title(f"{cfg}, matched c, {precision}, {WORKLOAD[goal].upper()}", fontsize=8)
        for i in range(rows):
            a = axes[i][j]
            a.set_xscale("log")
            a.set_xlim(1e2, s_cap * 2)
            a.grid(alpha=0.25, which="major", lw=0.4)
            for k_label, k in K_LINES.items():
                a.axvline(k, color=K_COLORS[k_label], lw=1.0, ls=(0, (6, 3)), alpha=0.7)
                if i == 0:
                    a.text(k, 1.05, f" k = {k_label}", color=K_COLORS[k_label], fontsize=7, va="bottom")
        axes[-1][j].set_xlabel("S (bits of trained state)")
    fig.suptitle(f"Capability against S (top; measured on Qwen2.5-0.5B/1.5B, S as the experiment counts it; facts placed "
                 f"at the 12-epoch measurement of N = 2,000, N = 200 at 90 epochs for reference)\nand rho for DeepSeek-V4-Pro and Kimi K2 (below), matched c = k phi, "
                 f"{precision}; k = 2.5 MB is the deck's calibration.\nF (fresh bits per token at a block boundary) is "
                 f"B1's measurement on Qwen2.5-1.5B scaled by d/1536 to each model. {inputs_note}", fontsize=9)
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(K_LINES), fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    paths = []
    for ext in ("png", "svg"):
        p = out / f"capability_vs_rho_{precision}.{ext}"
        fig.savefig(p, dpi=130)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="fp8")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves = load(args.runs, args.precision)
    missing = [(c, k, w) for c in CONFIGS for k in K_LINES for w in set(WORKLOAD.values()) if (c, k, w) not in curves]
    if missing:
        raise SystemExit(f"figure.py: no eta1 {args.precision} curves for {missing} in {args.runs}")
    for p in draw(curves, out, args.precision, args.note):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
