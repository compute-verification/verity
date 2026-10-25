"""The fan-out estimate (Proposition 3 with explicit weight-activation products) beside a `rho_attack.py` run's
full-training rows, at matched c (eta = 1):

    python3 fanout.py RUN.json --inputs inputs/measured_b1.json --out fanout.md

Each activation occurrence meets at most e products, so a block prices at most e * h_x / beta_x of them and, pricing every
product of the layers' maps (the LM head left out, so e is the largest fan-out among the layers' activations),
rho >= c * beta_x * N / (e * h_x * u) with N the products per token and u the run's honest FLOPs per token (3f SFT, 4f
GRPO). At h_x = k and c = k phi this is phi * beta_x / (kappa e), kappa = u / N. beta_x is the replay rate
beta_x^rep = (h_tok + F) / d: a block holding a token recomputes theta's activations and adds the F fresh bits of the
residual. F is taken two ways: B1's Ff_min (the cheapest boundary) and its mean over boundaries (each layer's own input),
both scaled by d / d_model as rho_attack scales F.

The frozen-prefix attack's rho is against its own recipe's honest FLOPs (a forward, and a backward through the trained
top layers and the head) and its blocks hold the trained layers' maps only, so its estimate prices those layers' products
against that denominator, with theta's own activations (h_tok / d per coordinate) at the first trained layer's attention
input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import configs  # noqa: E402
import rho_attack as ra  # noqa: E402

H_TOK = 17.0
CONFIGS = {"deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k2": "Kimi K2"}
PRECISIONS = ("fp8", "nvfp4")
KS = ("10kB", "1MB", "2.5MB")
WORKLOADS = ("grpo", "sft")


def fanouts(cfg: configs.Config) -> dict[str, tuple[float, float]]:
    """Per activation type: (products one occurrence feeds, occurrences per token per layer). Maps reading the normed
    residual share one input per half (attention, MLP); every other map has its own input, one per active copy."""
    out = {"attention input (normed residual)": (sum(m.d_out * m.active for m in cfg.maps if m.resid_in and not m.mlp), 1),
           "MLP input (normed residual)": (sum(m.d_out * m.active for m in cfg.maps if m.resid_in and m.mlp), 1)}
    for m in cfg.maps:
        if not m.resid_in:
            out[f"{m.name} input ({m.d_in} wide, x{m.active:g})"] = (m.d_out, m.active)
    return out


def occurrences(cfg: configs.Config) -> float:
    """Activation coordinates the layers' maps read per token: two normed residuals per layer and every interior input."""
    return cfg.layers * (2 * cfg.d + sum(m.d_in * m.active for m in cfg.maps if not m.resid_in))


def products(cfg: configs.Config, layers: float | None = None) -> float:
    return (cfg.layers if layers is None else layers) * sum(m.d_in * m.d_out * m.active for m in cfg.maps)


def e_layers(cfg: configs.Config) -> float:
    return max(e for e, _ in fanouts(cfg).values())


def beta_rep(inputs: ra.Inputs, doc: dict, cfg: configs.Config, label: str, precision: str) -> dict[str, float]:
    F_min = inputs.F_for(label, precision, cfg)
    F_mean = doc["F_layer_mean"][label][precision] * cfg.d / inputs.d_model
    return {"Ff_min": (H_TOK + F_min) / cfg.d, "F_mean": (H_TOK + F_mean) / cfg.d}


def estimate(cfg: configs.Config, beta: float, workload: str) -> float:
    u = (4 if workload == "grpo" else 3) * cfg.f
    return cfg.phi() * beta * products(cfg) / (e_layers(cfg) * u)


def frozen_estimate(cfg: configs.Config, beta: float, m: int, workload: str) -> float:
    f_top = cfg.f_layer * m + cfg.f_head
    u = cfg.f + 2 * f_top + (cfg.f if workload == "grpo" else 0.0)
    cheap = sum(mp.d_in * mp.d_out * mp.active for mp in cfg.maps if mp.resid_in and not mp.mlp)
    priced = (products(cfg, m) - cheap) * beta + cheap * H_TOK / cfg.d
    return cfg.phi() * priced / (e_layers(cfg) * u)


def sig(x: float) -> str:
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.2e}"


def span(xs: list[float]) -> str:
    lo, hi = sig(min(xs)), sig(max(xs))
    return lo if lo == hi else f"{lo}-{hi}"


def names(xs: list[str]) -> str:
    return " / ".join(dict.fromkeys(xs))


def report(run: dict, inputs: ra.Inputs, doc: dict) -> str:
    rows = sorted((r for r in run["rows"] if r["c_mode"] == "eta1" and r["S_label"] == "full" and r["workload"] in WORKLOADS),
                  key=lambda r: WORKLOADS.index(r["workload"]))
    out = ["## e per activation type (configs.py)", "",
           "| config | activation | products per occurrence (e) |", "|---|---|---|"]
    for name, label in CONFIGS.items():
        cfg = configs.get(name)
        for act, (e, _) in fanouts(cfg).items():
            out.append(f"| {label} | {act} | {e:,.0f} |")
        out.append(f"| {label} | LM head input (final normed residual) | {cfg.vocab:,} |")
        out.append(f"| {label} | max over the layers' activations (e used) | {e_layers(cfg):,.0f} |")
        out.append(f"| {label} | layers' products per activation occurrence, averaged | "
                   f"{products(cfg) / occurrences(cfg):,.0f} |")
    out += ["", "## Replay rate and fan-out estimate per config (eta = 1, kappa = u / N)", "",
            "| config | precision | label | F (bits/token, scaled) | beta_x^rep (Ff_min) | beta_x^rep (mean F) | phi | "
            "kappa (GRPO, SFT) | fan-out estimate (Ff_min) | fan-out estimate (mean F) |", "|" + "---|" * 10]
    for name, label in CONFIGS.items():
        cfg = configs.get(name)
        for prec in PRECISIONS:
            for lab in ("lora16", "full"):
                b = beta_rep(inputs, doc, cfg, lab, prec)
                kap = [(4 if w == "grpo" else 3) * cfg.f / products(cfg) for w in WORKLOADS]
                F = inputs.F_for(lab, prec, cfg)
                Fm = doc["F_layer_mean"][lab][prec] * cfg.d / inputs.d_model
                out.append(f"| {label} | {prec} | {lab} | Ff_min {sig(F)}, mean {sig(Fm)} | {sig(b['Ff_min'])} | "
                           f"{sig(b['F_mean'])} | {sig(cfg.phi())} | {sig(kap[0])}, {sig(kap[1])} | "
                           f"{span([estimate(cfg, b['Ff_min'], w) for w in WORKLOADS])} | "
                           f"{span([estimate(cfg, b['F_mean'], w) for w in WORKLOADS])} |")
    out += ["", "## Full training beside the fan-out estimate (eta = 1, absolute rho; GRPO and SFT as a range, strategies GRPO / "
            "SFT where they differ)", "",
            "| config | precision | k | fan-out estimate (Ff_min) | fan-out estimate (mean F) | bound (beta_gx count) | "
            "bound (beta_v count) | cheapest recipe-kept attack (strategy) | cheapest attack overall (strategy) | "
            "overall / fan-out (Ff_min) | overall / fan-out (mean F) |", "|" + "---|" * 11]
    below, frozen = [], []
    for name, label in CONFIGS.items():
        cfg = configs.get(name)
        for prec in PRECISIONS:
            b = beta_rep(inputs, doc, cfg, "full", prec)
            for k in KS:
                at = [r for r in rows if (r["config"], r["precision"], r["k_label"]) == (name, prec, k)]
                if not at:
                    continue
                est = {r["workload"]: {v: estimate(cfg, b[v], r["workload"]) for v in b} for r in at}
                out.append(
                    f"| {label} | {prec} | {k} | {span([est[r['workload']]['Ff_min'] for r in at])} | "
                    f"{span([est[r['workload']]['F_mean'] for r in at])} | {span([r['bound']['rho_gx'] for r in at])} | "
                    f"{span([r['bound']['rho'] for r in at])} | "
                    f"{span([r['attack_exact']['rho'] for r in at])} ({names([r['attack_exact']['strategy'] for r in at])}) | "
                    f"{span([r['attack']['rho'] for r in at])} ({names([r['attack']['strategy'] for r in at])}) | "
                    f"{span([r['attack']['rho'] / est[r['workload']]['Ff_min'] for r in at])} | "
                    f"{span([r['attack']['rho'] / est[r['workload']]['F_mean'] for r in at])} |")
                for r in at:
                    for s, a in r["strategies"].items():
                        if s == "frozen_prefix_tiling":
                            m = a["params"]["trained_top_layers"]
                            fe = {v: frozen_estimate(cfg, b[v], m, r["workload"]) for v in b}
                            frozen.append((label, prec, k, r["workload"], m, a["rho"], fe, est[r["workload"]]))
                            continue
                        for v in b:
                            if a["rho"] < est[r["workload"]][v]:
                                below.append((label, prec, k, r["workload"], s, a["rho"], v, est[r["workload"]][v]))
    out += ["", "## Modeled attacks below the full-model fan-out estimate (every strategy but frozen prefix)", ""]
    out += [f"- {lb} {p} k = {k} {w}: {s} rho {sig(x)} < estimate ({v}) {sig(e)}" for lb, p, k, w, s, x, v, e in below] or [
        "- none"]
    out += ["", "## Frozen prefix against the fan-out estimate over its trained layers' products (its own honest FLOPs)",
            "", "| config | precision | k | workload | trained top layers | frozen-prefix rho | its fan-out estimate (Ff_min) | "
            "its fan-out estimate (mean F) | full-model fan-out estimate (Ff_min) | rho / its estimate (Ff_min) | "
            "rho / its estimate (mean F) |", "|" + "---|" * 11]
    for lb, p, k, w, m, x, fe, fm in frozen:
        out.append(f"| {lb} | {p} | {k} | {w} | {m} | {sig(x)} | {sig(fe['Ff_min'])} | {sig(fe['F_mean'])} | "
                   f"{sig(fm['Ff_min'])} | {sig(x / fe['Ff_min'])} | {sig(x / fe['F_mean'])} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = json.loads(Path(args.inputs).read_text())
    text = report(json.loads(Path(args.run).read_text()), ra.Inputs.from_json(doc), doc)
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
