"""Loss against cost for learning-frontier runs: each restricted family against dense at loss targets every configuration reaches.

    python analyze.py --out DIR RUN_DIR...

A RUN_DIR is a fetched `research run` directory (`research fetch --all RUN`): its `out/*/` hold train.py's `config.json`,
`curve.json` and `result.json`, and its `gpu-lease-usage.jsonl` the GPU time it held. A configuration is a size, a family with
its rank or sync interval, and a token budget. At each size a configuration takes its best learning rate on the shared grid
(lowest final held-out loss); a size trained at one learning rate per configuration takes that one. Then, per size:

- the fit L(C) = E + A * C^-b to each configuration's checkpoints from a tenth of its run on, C its analytic training FLOPs;
- per comparison (dense with the per-window low-rank configurations; dense with the local-SGD ones), three loss targets every
  configuration in it reaches: the worst configuration's final loss, dense's loss a quarter of the way through its run (or,
  when that is no higher, halfway from the worst final loss to the highest first-checkpoint loss), and their midpoint;
- kappa at each target, the family's cost to reach it over dense's, in FLOPs, tokens and measured training seconds. The step at
  which a curve reaches a target is interpolated between checkpoints (log step, linear in loss, on the curve's running
  minimum); kappa from the fits is written beside it;
- the implied alpha at each target, dense's restricted quantity to reach it over the family's. Per-window low rank: fresh input,
  every value trained in a window at fp32, dense training every parameter in every window of the same length. Local SGD:
  bits communicated, each replica's delta at fp32 per sync, dense as data parallel over as many replicas sending a full
  gradient every step;
- for per-window low rank, alpha on the stock instead: the values in the departure from the initialization when the target is
  reached, every parameter for dense; low rank's merged delta of each matrix has rank at most the windows started times r, and
  is counted as factors or whole, whichever is smaller, with the norms in full. A base registered once, at the start, makes
  the stock what a server block must hold, not one window's delta;
- kappa at each configuration's own final loss (dense's cost to reach it), the late-training number;
- where shorter-horizon runs exist (dense's or a configuration's, at the learning rate its main run took), kappa and alpha
  between envelopes: each side's least cost to reach the target over its main run and those, since a checkpoint inside a
  cosine schedule has not annealed. A side with no shorter runs is its main run alone, and the table says which horizon won.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from model import SIZES, matrix_shapes
from train import OUTER_LR, OUTER_MOMENTUM

FIT_FROM = 0.1
QUARTER = 0.25
#: each comparison with dense: the configurations in it (a local-SGD run with other outer settings than train.py's is an anchor,
#: compared on its own so it moves no target of the main comparison)
GROUPS = {"lowrank": lambda r: r["family"] == "lowrank",
          "localsgd": lambda r: r["family"] == "localsgd" and " outer " not in r["label"],
          "localsgd-outer": lambda r: r["family"] == "localsgd" and " outer " in r["label"]}
RULES = ("dense at a quarter", "midpoint", "worst final")


def label(args: dict, tpp: float) -> str:
    base = {"dense": "dense", "lowrank": f"r{args['rank']}", "localsgd": f"H{args['sync']}"}[args["family"]]
    if args["family"] == "localsgd" and (args["outer_lr"], args["outer_momentum"]) != (OUTER_LR, OUTER_MOMENTUM):
        base += f" outer {args['outer_lr']:g}/{args['outer_momentum']:g}"
    return base if args["tokens_per_param"] == tpp else f"{base}@{args['tokens_per_param']:g}tpp"


def load(run_dirs: list[Path], tpp: float) -> tuple[list[dict], list[dict], dict]:
    runs, skipped, gpu = [], [], {}
    for rd in run_dirs:
        usage = rd / "gpu-lease-usage.jsonl"
        recs = [json.loads(x) for x in usage.read_text().splitlines() if x.strip()] if usage.exists() else []
        gpu[rd.name] = {"held_hours": sum(g["held_s"] for r in recs for g in r["gpus"]) / 3600,
                        "busy_hours": sum(g["busy_s"] for r in recs for g in r["gpus"]) / 3600, "leases": len(recs)}
        for cfg in sorted(rd.glob("out/*/config.json")):
            d = cfg.parent
            config = json.loads(cfg.read_text())
            args = config["args"]
            entry = {"run": rd.name, "name": d.name, "size": args["size"], "family": args["family"], "label": label(args, tpp),
                     "lr": args["lr"]}
            if args["max_steps"] or args["val_windows"]:
                skipped.append(entry | {"why": "smoke test (--max-steps or --val-windows)"})
                continue
            if not (d / "result.json").exists():
                skipped.append(entry | {"why": "no result.json: incomplete"})
                continue
            result = json.loads((d / "result.json").read_text())
            curve = [p for p in json.loads((d / "curve.json").read_text()) if not p.get("diverged")]
            runs.append(entry | {"config": config, "result": result, "curve": curve, "complete": result["complete"],
                                 "diverged": result["diverged"], "final": curve[-1]["val_nats_per_byte"] if curve else None})
    return runs, skipped, gpu


def best_runs(runs: list[dict]) -> tuple[dict, dict]:
    grid, best = {}, {}
    for r in runs:
        grid.setdefault(r["size"], {}).setdefault(r["label"], {})[f"{r['lr']:g}"] = (
            "diverged" if r["diverged"] else r["final"] if r["complete"] else "incomplete")
        if r["complete"] and not r["diverged"]:
            cur = best.setdefault(r["size"], {}).get(r["label"])
            if cur is None or r["final"] < cur["final"]:
                best[r["size"]][r["label"]] = r
    for size, labels in best.items():
        for lab, r in labels.items():
            lrs = sorted(float(x) for x in grid[size][lab])
            r["lr_at_grid_edge"] = len(lrs) > 1 and r["lr"] in (lrs[0], lrs[-1])
    return grid, best


def series(r: dict) -> dict:
    """The curve's columns, and every cost as a function of the step (each is linear in it)."""
    c, res = r["config"], r["result"]
    args, counts = c["args"], c["counts"]
    tokens_per_step = args["batch"] * c["ctx"]
    s = {"step": np.array([p["step"] for p in r["curve"]], float), "loss": np.array([p["val_nats_per_byte"] for p in r["curve"]]),
         "n_steps": c["n_steps"], "flops_per_step": c["flops_per_token"] * tokens_per_step, "tokens_per_step": tokens_per_step,
         "seconds_per_step": res["seconds_per_step"], "bits": c["bits_per_value"], "total": counts["total"],
         "trained": counts["trained_per_window"] if args["family"] == "lowrank" else counts["total"],
         "window": args["window"], "replicas": args["replicas"], "sync": args["sync"] if args["family"] == "localsgd" else 1,
         "tpp": args["tokens_per_param"]}
    if args["family"] == "lowrank":
        s |= {"shapes": matrix_shapes(SIZES[args["size"]], c["vocab"]), "rank": args["rank"], "norms": counts["norms"]}
    return s


def stock(s: dict, step: float) -> float:
    """The values in the departure from the initialization after `step` steps: every parameter, except under per-window low rank,
    where each matrix's merged delta has rank at most the windows started times r (stored as factors or whole, whichever is
    smaller) and the norms are trained in full."""
    if "shapes" not in s:
        return s["total"]
    w = math.ceil(step / s["window"])
    return sum(min(min(w * s["rank"], i, o) * (i + o), i * o) for i, o in s["shapes"]) + s["norms"]


def per_step(s: dict, group: str, like: dict) -> float:
    """The restricted quantity per step: fresh-input bits (every value trained in a window, over the window's steps) for low rank,
    bits communicated (every replica's full state per sync) for local SGD. Window length and replica count are `like`'s, the
    family's, so dense is counted on the same terms."""
    if group == "lowrank":
        return s["bits"] * s["trained"] / like["window"]
    return s["bits"] * like["replicas"] * s["total"] / s["sync"]


def step_at(s: dict, target: float) -> float | None:
    """The step at which the curve's running minimum first reaches `target`, interpolated; None if it never does or did before the
    first checkpoint."""
    m = np.minimum.accumulate(s["loss"])
    hit = np.nonzero(m <= target)[0]
    if not len(hit) or hit[0] == 0:
        return None
    i = hit[0]
    lo, hi = math.log(s["step"][i - 1]), math.log(s["step"][i])
    f = (m[i - 1] - target) / (m[i - 1] - m[i]) if m[i - 1] > m[i] else 1.0
    return math.exp(lo + f * (hi - lo))


def fit(s: dict) -> dict:
    keep = s["step"] >= FIT_FROM * s["n_steps"]
    c, y = s["step"][keep] * s["flops_per_step"], s["loss"][keep]
    best = None
    for e in np.linspace(0.0, y.min() - 1e-4, 400):
        slope, icept = np.polyfit(np.log(c), np.log(y - e), 1)
        sse = float(((e + math.exp(icept) * c ** slope - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, e, math.exp(icept), -slope)
    sse, e, a, b = best
    return {"E": e, "A": a, "b": b, "rmse": math.sqrt(sse / len(y)), "points": int(len(y))}


def flops_from_fit(f: dict, target: float) -> float | None:
    return (f["A"] / (target - f["E"])) ** (1 / f["b"]) if target > f["E"] and f["b"] > 0 else None


def targets(dense: dict, configs: dict[str, dict]) -> tuple[list[float], list[str]]:
    """The loss targets, and the configurations left out of choosing them: those never below the comparison's lowest
    first-checkpoint loss, where no curve's crossing could be interpolated."""
    first = min(float(s["loss"][0]) for s in configs.values())
    out = [lab for lab, s in configs.items() if float(s["loss"].min()) >= first - 1e-3]
    worst_final = max(float(s["loss"].min()) for lab, s in configs.items() if lab not in out) + 1e-6
    quarter = float(np.interp(QUARTER * dense["n_steps"], dense["step"], np.minimum.accumulate(dense["loss"])))
    hi = min(quarter, first - 1e-3)
    if hi <= worst_final + 1e-3:
        hi = (first + worst_final) / 2
    return [hi, (hi + worst_final) / 2, worst_final], out


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def seconds(s: dict, step: float) -> float | None:
    return step * s["seconds_per_step"] if s["seconds_per_step"] else None


def compare(fam: dict, dense: dict, fam_fit: dict, dense_fit: dict, target: float, group: str,
            envelopes: tuple[list[dict], list[dict]]) -> dict:
    sf, sd = step_at(fam, target), step_at(dense, target)
    row = {"target": target, "step": sf, "dense_step": sd}
    if sf is None or sd is None:
        return row | {"reached": False}
    omega, omega_dense = sf * per_step(fam, group, fam), sd * per_step(dense, group, fam)
    row |= {"reached": True,
            "kappa_flops": sf * fam["flops_per_step"] / (sd * dense["flops_per_step"]),
            "kappa_tokens": sf / sd,
            "kappa_time": ratio(seconds(fam, sf), seconds(dense, sd)),
            "kappa_flops_fit": ratio(flops_from_fit(fam_fit, target), flops_from_fit(dense_fit, target)),
            "omega": omega, "omega_dense": omega_dense, "alpha": omega_dense / omega}
    if group == "lowrank":
        row |= {"stock": stock(fam, sf), "alpha_stock": stock(dense, sd) / stock(fam, sf)}
    return row | at_envelope(fam, sf, dense, sd, target, group, envelopes)


def least(s: dict, step: float | None, shorter: list[dict], target: float) -> tuple[float, dict] | None:
    """The run that reaches `target` at the least FLOPs, the main one (`s`, at `step`, None if it can't place the target) or a
    shorter-horizon one, and its step; None if none does."""
    reach = [*([(step, s)] if step is not None else []), *((st, e) for e in shorter if (st := step_at(e, target)) is not None)]
    return min(reach, key=lambda p: p[0] * p[1]["flops_per_step"]) if reach else None


def at_envelope(fam: dict, sf: float, dense: dict, sd: float | None, target: float, group: str,
                envelopes: tuple[list[dict], list[dict]]) -> dict:
    """kappa and alpha between the family's envelope and dense's (`envelopes`: each one's shorter-horizon runs)."""
    if not any(envelopes) or (fam_least := least(fam, sf, envelopes[0], target)) is None or (
            dense_least := least(dense, sd, envelopes[1], target)) is None:
        return {}
    (fs, f), (ds, d) = fam_least, dense_least
    row = {"envelope_tokens_per_param": [f["tpp"], d["tpp"]],
           "kappa_flops_envelope": fs * f["flops_per_step"] / (ds * d["flops_per_step"]),
           "kappa_tokens_envelope": fs * f["tokens_per_step"] / (ds * d["tokens_per_step"]),
           "alpha_envelope": ds * per_step(d, group, fam) / (fs * per_step(f, group, fam))}
    return row | ({"alpha_stock_envelope": stock(d, ds) / stock(f, fs)} if group == "lowrank" else {})


def own_final(fam: dict, dense: dict, group: str, envelopes: tuple[list[dict], list[dict]]) -> dict:
    """kappa and alpha at the family's final loss; where dense never gets that low, at dense's final loss instead. Where dense's
    main run is already there at its first checkpoint, only its shorter runs can place the target, and kappa in tokens is at
    least the family's steps over that checkpoint's."""
    target, at = float(fam["loss"].min()), "family's final loss"
    if float(dense["loss"].min()) > target:
        target, at = float(dense["loss"].min()), "dense's final loss"
    target += 1e-9
    sd, sf = step_at(dense, target), step_at(fam, target)
    row = {"target": target, "at": at, "reached": sd is not None and sf is not None}
    if sf is None:
        return row
    if sd is None:
        row["kappa_tokens_at_least"] = sf * fam["tokens_per_step"] / (dense["step"][0] * dense["tokens_per_step"])
    else:
        row |= {"kappa_flops": sf * fam["flops_per_step"] / (sd * dense["flops_per_step"]), "kappa_tokens": sf / sd,
                "kappa_time": ratio(seconds(fam, sf), seconds(dense, sd)),
                "alpha": sd * per_step(dense, group, fam) / (sf * per_step(fam, group, fam)),
                "alpha_stock": stock(dense, sd) / stock(fam, sf) if group == "lowrank" else None}
    return row | at_envelope(fam, sf, dense, sd, target, group, envelopes)


def plot(out: Path, size: str, group: str, curves: dict[str, dict], shorter: dict[str, list[dict]], tgts: list[float]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab, s in curves.items():
        line, = ax.plot(s["step"] * s["flops_per_step"], s["loss"], marker=".", label=lab)
        for e in shorter.get(lab, []):
            ax.plot(e["step"] * e["flops_per_step"], e["loss"], color=line.get_color(), lw=0.7, ls=":")
    for t in tgts:
        ax.axhline(t, color="grey", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs (analytic)")
    ax.set_ylabel("held-out loss (nats per byte)")
    ax.set_title(f"size {size}, dense and {group} at their best rates; dotted: shorter horizons", fontsize=10)
    ax.legend(fontsize=8)
    name = f"loss-flops-{size}-{group}.png"
    fig.tight_layout()
    fig.savefig(out / name, dpi=120)
    plt.close(fig)
    return name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens-per-param", type=float, default=8.0, help="the main token budget; other budgets are dense's envelope")
    ap.add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    runs, skipped, gpu = load(a.run_dirs, a.tokens_per_param)
    grid, best = best_runs(runs)
    report = {"schema": "learning-frontier/analysis/v1", "tokens_per_param": a.tokens_per_param, "lr_grid": grid,
              "skipped": skipped, "gpu": gpu, "gpu_held_hours": sum(g["held_hours"] for g in gpu.values()),
              "non_embedding": {size: next(iter(labels.values()))["config"]["counts"]["non_embedding"]
                                for size, labels in best.items()},
              "best": {}, "fits": {}, "envelopes": {}, "comparisons": {}, "throughput": {}, "plots": []}
    for size, labels in sorted(best.items()):
        ser = {lab: series(r) for lab, r in labels.items()}
        fits = {lab: fit(s) for lab, s in ser.items()}
        report["best"][size] = {lab: {"lr": r["lr"], "run": r["run"], "name": r["name"], "final": r["final"],
                                      "lr_at_grid_edge": r["lr_at_grid_edge"]} for lab, r in labels.items()}
        report["fits"][size] = fits
        report["throughput"][size] = {lab: {k: r["result"].get(k) for k in ("tokens_per_second", "achieved_flop_per_second", "mfu",
                                                                             "seconds_per_step", "max_memory_gib")}
                                      for lab, r in labels.items()}
        if "dense" not in ser:
            continue
        shorter = {lab: [r for r in runs if r["size"] == size and r["label"].startswith(lab + "@") and r["lr"] == best_r["lr"]
                         and r["complete"] and not r["diverged"]] for lab, best_r in labels.items() if "@" not in lab}
        report["envelopes"][size] = {lab: sorted(r["config"]["args"]["tokens_per_param"] for r in rs) for lab, rs in shorter.items()}
        shorter = {lab: [series(r) for r in rs] for lab, rs in shorter.items()}
        for group, member in GROUPS.items():
            members = {"dense": ser["dense"]} | {lab: ser[lab] for lab, r in labels.items() if "@" not in lab and member(r)}
            if len(members) < 2:
                continue
            tgts, out_of_targets = targets(ser["dense"], members)
            rows = {lab: [compare(s, ser["dense"], fits[lab], fits["dense"], t, group, (shorter[lab], shorter["dense"]))
                          for t in tgts] for lab, s in members.items() if lab != "dense"}
            finals = {lab: own_final(s, ser["dense"], group, (shorter[lab], shorter["dense"]))
                      for lab, s in members.items() if lab != "dense"}
            report["comparisons"].setdefault(size, {})[group] = {"targets": tgts, "never_below_first_checkpoint": out_of_targets,
                                                                 "rows": rows, "own_final": finals}
            report["plots"].append(plot(a.out, size, group, members, shorter, tgts))
    (a.out / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    (a.out / "tables.md").write_text(tables(report))
    print(tables(report))
    return 0


def fmt(x: float | None, spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


MAIN = ("kappa_flops", "kappa_tokens", "kappa_time", "alpha", "alpha_stock")
ENVELOPES = ("kappa_flops_envelope", "kappa_tokens_envelope", "alpha_envelope", "alpha_stock_envelope")
HEAD = ("kappa FLOPs | kappa tokens | kappa time | alpha | alpha (stock) | kappa FLOPs (envelopes) | kappa tokens (envelopes) |"
        " alpha (envelopes) | alpha stock (envelopes) | horizons won, tpp (family/dense)")


def horizons(r: dict) -> str:
    h = r.get("envelope_tokens_per_param")
    return f"{h[0]:g}/{h[1]:g}" if h else "-"


def cell(r: dict, keys: tuple[str, str, str], group: str) -> str:
    c = f"{fmt(r.get(keys[0]))} / {fmt(r.get(keys[1]), '.1f')}"
    return c + (f" / {fmt(r.get(keys[2]))}" if group == "lowrank" else "")


def tables(report: dict) -> str:
    out = ["## Learning rates: final held-out nats per byte at each", "", "| size | config | lr: final loss |", "|---|---|---|"]
    for size, labels in sorted(report["lr_grid"].items()):
        for lab, by_lr in sorted(labels.items()):
            cells = ", ".join(f"{lr}: {v if isinstance(v, str) else format(v, '.4f')}"
                              for lr, v in sorted(by_lr.items(), key=lambda kv: float(kv[0])))
            out.append(f"| {size} | {lab} | {cells} |")
    out += ["", "## Measured throughput of each configuration's best run (training steps only, one GPU)", "",
            "| size | config | tokens/s | achieved FLOP/s | MFU | seconds/step | peak memory GiB |", "|---|---|---|---|---|---|---|"]
    for size, labels in sorted(report["throughput"].items()):
        out += [f"| {size} | {lab} | {fmt(t['tokens_per_second'], ',.0f')} | {fmt(t['achieved_flop_per_second'], '.3g')} | "
                f"{fmt(t['mfu'], '.3f')} | {fmt(t['seconds_per_step'], '.3f')} | {fmt(t['max_memory_gib'], '.1f')} |"
                for lab, t in sorted(labels.items()) if "@" not in lab]
    out += ["", "## Envelopes: each configuration's shorter-horizon runs at its main run's learning rate (tokens per parameter)", "",
            "| size | config | horizons |", "|---|---|---|"]
    for size, labs in sorted(report["envelopes"].items()):
        out += [f"| {size} | {lab} | {', '.join(f'{t:g}' for t in tpps) or '-'} |" for lab, tpps in sorted(labs.items())]
    for size, groups in sorted(report["comparisons"].items()):
        for group, comp in groups.items():
            never = comp["never_below_first_checkpoint"]
            out += ["", f"## Size {size}, {group}: targets " + ", ".join(f"{t:.4f}" for t in comp["targets"]), "",
                    *([f"Never below the comparison's first checkpoint, so left out of the targets: {', '.join(never)}.", ""]
                      if never else []),
                    f"| config | best lr | target | kappa FLOPs (fit) | {HEAD} |", "|---|---|---|" + "---|" * 11]
            for lab, rows in comp["rows"].items():
                lr = report["best"][size][lab]["lr"]
                for r in rows:
                    out.append(f"| {lab} | {lr:g} | {r['target']:.4f} | {fmt(r.get('kappa_flops_fit'))} | "
                               + " | ".join(fmt(r.get(k)) for k in (*MAIN, *ENVELOPES)) + f" | {horizons(r)} |")
            out += ["", f"| config | at | loss | {HEAD} |", "|---|---|---|" + "---|" * 10]
            for lab, f in comp["own_final"].items():
                low = f.get("kappa_tokens_at_least")
                out.append(f"| {lab} | {f['at']} | {f['target']:.4f} | " + " | ".join(
                    f">{low:.1f}" if k == "kappa_tokens" and low is not None else fmt(f.get(k)) for k in (*MAIN, *ENVELOPES))
                           + f" | {horizons(f)} |")
    sizes = sorted(report["comparisons"], key=lambda s: report["non_embedding"][s])
    for title, keys in (("main runs", ("kappa_flops", "alpha", "alpha_stock")),
                        ("envelopes", ("kappa_flops_envelope", "alpha_envelope", "alpha_stock_envelope"))):
        out += ["", f"## Across sizes, {title}: kappa (FLOPs) / alpha at each target rule (low rank: then alpha on the stock)", "",
                "| group | config | target | " + " | ".join(sizes) + " |", "|---|---|---|" + "---|" * len(sizes)]
        for group in GROUPS:
            labs = sorted({lab for s in sizes for lab in report["comparisons"][s].get(group, {}).get("rows", {})})
            for lab in labs:
                for i, rule in enumerate((*RULES, "own final")):
                    cells = []
                    for s in sizes:
                        comp = report["comparisons"][s].get(group, {})
                        r = (comp.get("own_final", {}).get(lab) if rule == "own final"
                             else (comp.get("rows", {}).get(lab) or [None] * 3)[i]) or {}
                        cells.append(cell(r, keys, group) + (f" ({horizons(r)})" if title == "envelopes" else ""))
                    out.append(f"| {group} | {lab} | {rule} | " + " | ".join(cells) + " |")
    out += ["", f"GPU hours held: {report['gpu_held_hours']:.2f}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
