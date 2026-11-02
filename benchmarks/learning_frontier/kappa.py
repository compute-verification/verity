"""The speedrun's κ(Δ) per class per start, and each class's ceiling, from speedrun.py's witness records.

    python kappa.py --out DIR RUN_DIR...

A RUN_DIR is a fetched `research run` directory (`research fetch --all RUN`), whose `out/*/witness.json` are speedrun.py's
records and whose `gpu-lease-usage.jsonl` is the GPU time it held, or a directory holding `witness.json` itself. Per start (a
size and a registered θ, by digest):

- each configuration (a method with its structural knob) takes its best learning rate: the lowest held-out loss within the
  budget. The sweep's cost (every rate's FLOPs and GPU-seconds) is written beside it;
- ℓ₀ is the start's held-out loss, as the runs measured it at step 0;
- the reference is S0's best: at each loss, the least FLOPs AdamW's or Muon's best run takes to first reach it;
- the targets are ℓ₀ - Δ for Δ = q * Δ_max, q in QS, with Δ_max what the reference gains in half the budget, so the budget is
  twice the reference's cost to reach the deepest target. Where the reference gains nothing in half the budget, Δ_max is its
  gain in the whole budget, and the table says so;
- κ(Δ) = C_class(ℓ₀ - Δ) / C_ref(ℓ₀ - Δ), in analytic FLOPs, tokens and measured training seconds. C is the cost of first
  reaching the loss on the curve's running minimum, interpolated linearly between checkpoints (every 5% of the budget, and the
  start at 0). A configuration that never reaches it within the budget has κ > budget / C_ref;
- the ceiling: the lowest loss each configuration reaches within the budget, and its gain ℓ₀ - ceiling, also as a share of the
  reference's gain in the whole budget;
- S3 (no training) is its own table: each evaluation setting's loss, its gain over ℓ₀, and its forward FLOPs per scored target
  over the trained-context evaluation's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QS = (0.25, 0.5, 0.75, 1.0)
CLASS_ORDER = ("S0", "S1", "S2", "S3")


def human_bytes(b: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= scale:
            return f"{b / scale:.3g} {unit}"
    return f"{b:.0f} B"


def label(w: dict) -> str:
    m, cfg = w["method"], w["config"]
    if m == "lora":
        knob = human_bytes(cfg["update_bytes"]).replace(" ", "") if cfg["update_bytes"] else (
            f"r{cfg['rank']}" + (f" last{cfg['lora_blocks']}" if cfg["lora_blocks"] else ""))
        return f"S1 lora {knob}"
    return f"{w['class']} {m}"


def load(run_dirs: list[Path], include_smoke: bool) -> tuple[list[dict], list[dict], dict]:
    runs, skipped, gpu = [], [], {}
    for rd in run_dirs:
        usage = rd / "gpu-lease-usage.jsonl"
        recs = [json.loads(x) for x in usage.read_text().splitlines() if x.strip()] if usage.exists() else []
        gpu[rd.name] = {"held_hours": sum(g["held_s"] for r in recs for g in r["gpus"]) / 3600, "leases": len(recs)}
        for f in sorted({*rd.glob("out/*/witness.json"), *rd.glob("*/witness.json"), *rd.glob("witness.json")}):
            w = json.loads(f.read_text())
            entry = {"run": rd.name, "dir": str(f.parent), "label": label(w), "witness": w}
            smoke = w["config"].get("max_steps") or w["config"].get("val_windows")
            if smoke and not include_smoke:
                skipped.append(entry | {"why": "smoke test (--max-steps or --val-windows)"})
            elif w["method"] != "icl" and (not w.get("complete") or w.get("diverged")):
                skipped.append(entry | {"why": "diverged" if w.get("diverged") else "incomplete"})
            else:
                runs.append(entry)
    return runs, skipped, gpu


def series(w: dict) -> dict:
    pts = [p for p in w["trajectory"] if not p.get("diverged")]
    f = np.array([p["flops"] for p in pts], float)
    loss = np.minimum.accumulate(np.array([p["val_nats_per_byte"] for p in pts], float))
    steps = np.array([p["step"] for p in pts], float)
    spt = w.get("seconds_per_step")
    return {"flops": f, "loss": loss, "raw": np.array([p["val_nats_per_byte"] for p in pts], float),
            "tokens": np.array([p["tokens"] for p in pts], float), "seconds": steps * spt if spt else None,
            "fraction": f / w["budget_flops"]}


def crossing(s: dict, target: float) -> dict | None:
    """FLOPs, tokens and seconds at which the running minimum first reaches `target`, linear between checkpoints; None if
    it never does. At the start (index 0) the cost is 0."""
    hit = np.nonzero(s["loss"] <= target)[0]
    if not len(hit):
        return None
    i = int(hit[0])
    if i == 0:
        return {"flops": 0.0, "tokens": 0.0, "seconds": 0.0}
    lo, hi = s["loss"][i - 1], s["loss"][i]
    t = (lo - target) / (lo - hi) if lo > hi else 1.0
    out = {k: float(s[k][i - 1] + t * (s[k][i] - s[k][i - 1])) for k in ("flops", "tokens")}
    out["seconds"] = float(s["seconds"][i - 1] + t * (s["seconds"][i] - s["seconds"][i - 1])) if s["seconds"] is not None else None
    return out


def at_flops(s: dict, flops: float) -> float:
    return float(np.interp(flops, s["flops"], s["loss"]))


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def analyze_start(entries: list[dict]) -> dict:
    by_label: dict[str, list[dict]] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    w0 = entries[0]["witness"]
    l0s = [e["witness"]["trajectory"][0]["val_nats_per_byte"] for e in entries if e["witness"]["trajectory"]]
    l0 = float(np.median(l0s))
    budget = max(e["witness"]["budget_flops"] for e in entries if e["witness"]["method"] != "icl") if any(
        e["witness"]["method"] != "icl" for e in entries) else None
    configs, s3 = {}, None
    for lab, es in sorted(by_label.items()):
        if lab.startswith("S3"):
            w = min(es, key=lambda e: e["witness"]["min_val_nats_per_byte"])["witness"]
            s3 = {"settings": [{k: p.get(k) for k in ("ctx", "stride", "rope", "val_nats_per_byte", "inference_multiplier",
                                                      "least_context", "fresh_input_bits_per_window")}
                               | {"gain": l0 - p["val_nats_per_byte"]} for p in w["trajectory"]],
                  "best": min(w["trajectory"], key=lambda p: p["val_nats_per_byte"])}
            s3["best_gain"] = l0 - s3["best"]["val_nats_per_byte"]
            continue
        best = min(es, key=lambda e: e["witness"]["min_val_nats_per_byte"])
        lrs = sorted(e["witness"]["lr"] for e in es)
        configs[lab] = {"run": best["run"], "dir": best["dir"], "lr": best["witness"]["lr"], "lrs": lrs,
                        "lr_at_grid_edge": len(lrs) > 1 and best["witness"]["lr"] in (lrs[0], lrs[-1]),
                        "grid": {f"{e['witness']['lr']:g}": e["witness"]["min_val_nats_per_byte"] for e in es},
                        "class": best["witness"]["class"], "method": best["witness"]["method"],
                        "update_bits": best["witness"]["update_bits"], "update": best["witness"]["update"],
                        "ceiling": best["witness"]["min_val_nats_per_byte"],
                        "final": best["witness"]["final_val_nats_per_byte"],
                        "merged_val_nats_per_byte": best["witness"].get("merged_val_nats_per_byte"),
                        "steps": best["witness"]["steps"], "tokens": best["witness"]["tokens"],
                        "flops": best["witness"]["flops"], "seconds_per_step": best["witness"].get("seconds_per_step"),
                        "sweep_flops": sum(e["witness"]["flops"] for e in es),
                        "sweep_gpu_seconds": sum(e["witness"]["gpu_seconds"]["wall"] for e in es),
                        "series": series(best["witness"])}
    out = {"size": w0["size"], "theta_sha256": w0["start"]["theta_sha256"], "fraction": w0["start"]["fraction"],
           "l0": l0, "l0_recorded": w0["start"]["val_nats_per_byte_recorded"], "l0_spread": float(np.ptp(l0s)),
           "budget_flops": budget, "s3": s3, "configs": {}, "targets": [], "kappa": {}}
    refs = [lab for lab, c in configs.items() if c["class"] == "S0"]
    if not refs or budget is None:
        out["configs"] = {lab: {k: v for k, v in c.items() if k != "series"} for lab, c in configs.items()}
        out["note"] = "no S0 run at this start: no reference, no kappa"
        return out
    half = min(at_flops(configs[lab]["series"], budget / 2) for lab in refs)
    whole = min(at_flops(configs[lab]["series"], budget) for lab in refs)
    d_max, basis = l0 - half, "half the budget"
    if d_max <= 0:
        d_max, basis = l0 - whole, "the whole budget (the reference gains nothing in half of it)"
    out |= {"reference": refs, "reference_gain_half": l0 - half, "reference_gain_whole": l0 - whole, "delta_max": d_max,
            "delta_basis": basis}
    if d_max <= 0:
        out["note"] = "the reference never gets below the start's loss within the budget: no target"
    else:
        out["targets"] = [{"q": q, "delta": q * d_max, "loss": l0 - q * d_max} for q in QS]
    for lab, c in configs.items():
        row = []
        for t in out["targets"]:
            ref = min((x for r in refs if (x := crossing(configs[r]["series"], t["loss"])) is not None),
                      key=lambda x: x["flops"], default=None)
            mine = crossing(c["series"], t["loss"])
            cell = {"q": t["q"], "delta": t["delta"], "ref_flops": ref and ref["flops"]}
            if ref is None or not ref["flops"]:
                row.append(cell | {"reached": mine is not None, "why": "the reference does not place this target"})
            elif mine is None:
                row.append(cell | {"reached": False, "kappa_flops_more_than": budget / ref["flops"]})
            else:
                row.append(cell | {"reached": True, "kappa_flops": mine["flops"] / ref["flops"],
                                   "kappa_tokens": ratio(mine["tokens"], ref["tokens"]),
                                   "kappa_time": ratio(mine["seconds"], ref["seconds"])})
        out["kappa"][lab] = row
        c["gain"] = l0 - c["ceiling"]
        c["gain_share_of_reference"] = c["gain"] / (l0 - whole) if l0 > whole else None
    out["configs"] = {lab: {k: v for k, v in c.items() if k != "series"} for lab, c in configs.items()}
    out["_series"] = {lab: c["series"] for lab, c in configs.items()}
    return out


def fmt(x: float | None, spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


def kcell(c: dict) -> str:
    if c.get("kappa_flops") is not None:
        return fmt(c["kappa_flops"])
    if c.get("kappa_flops_more_than") is not None:
        return f">{c['kappa_flops_more_than']:.1f}"
    return "-"


def order_key(lab: str) -> tuple:
    """Class, then method, then the update size a label names (`S1 lora 100kB` before `S1 lora 1MB`)."""
    parts = lab.split()
    cls = parts[0]
    size = 0.0
    if len(parts) > 2:
        units = {"GB": 1e9, "MB": 1e6, "kB": 1e3, "B": 1.0}
        unit = next((u for u in units if parts[2].endswith(u)), None)
        try:
            size = float(parts[2][: -len(unit)]) * units[unit] if unit else 0.0
        except ValueError:
            size = 0.0
    return (CLASS_ORDER.index(cls) if cls in CLASS_ORDER else 9, parts[1] if len(parts) > 1 else "", size, lab)


def tables(report: dict) -> str:
    out = []
    for st in report["starts"]:
        head = (f"## Size {st['size']}, start at {round(100 * (st['fraction'] or 0))}% (θ {st['theta_sha256'][:12]}), "
                f"ℓ₀ = {st['l0']:.4f} (recorded {fmt(st['l0_recorded'], '.4f')}, spread across runs {st['l0_spread']:.1e})")
        out += [head, ""]
        if st.get("note"):
            out += [st["note"], ""]
        if st["targets"]:
            out += [f"Budget {st['budget_flops']:.3e} FLOPs. Reference: the better of {', '.join(st['reference'])} at each loss. "
                    f"Δ_max = {st['delta_max']:.4f} nats per byte, the reference's gain in {st['delta_basis']} "
                    f"(in the whole budget: {st['reference_gain_whole']:.4f}).", "",
                    "| config | best lr | " + " | ".join(f"κ FLOPs Δ={t['delta']:.4f}" for t in st["targets"])
                    + " | κ tokens Δ_max | κ time Δ_max |", "|---|---|" + "---|" * (len(st["targets"]) + 2)]
            for lab in sorted(st["kappa"], key=order_key):
                row, c = st["kappa"][lab], st["configs"][lab]
                last = row[-1] if row else {}
                out.append(f"| {lab} | {c['lr']:g}{' (edge)' if c['lr_at_grid_edge'] else ''} | "
                           + " | ".join(kcell(x) for x in row) + f" | {fmt(last.get('kappa_tokens'))} | {fmt(last.get('kappa_time'))} |")
        out += ["", "| config | update | ceiling | gain ℓ₀ - ceiling | share of S0's gain | lrs: lowest loss | steps | tokens | "
                    "sweep GPU-s |", "|---|---|---|---|---|---|---|---|---|"]
        for lab in sorted(st["configs"], key=order_key):
            c = st["configs"][lab]
            grid = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(c["grid"].items(), key=lambda kv: float(kv[0])))
            out.append(f"| {lab} | {human_bytes(c['update_bits'] / 8)} | {c['ceiling']:.4f} | {fmt(c.get('gain'), '.4f')} | "
                       f"{fmt(c.get('gain_share_of_reference'))} | {grid} | {c['steps']} | {c['tokens']:.3g} | "
                       f"{c['sweep_gpu_seconds']:.0f} |")
        if st["s3"]:
            out += ["", "S3, no training (θ fixed): each evaluation setting", "",
                    "| ctx | stride | rope | loss | gain over ℓ₀ | least context | inference FLOPs ×  | fresh input bits per window |",
                    "|---|---|---|---|---|---|---|---|"]
            for p in st["s3"]["settings"]:
                out.append(f"| {p['ctx']} | {p['stride']} | {p['rope']} | {p['val_nats_per_byte']:.4f} | {p['gain']:+.4f} | "
                           f"{p['least_context']} | {p['inference_multiplier']:.2f} | {p['fresh_input_bits_per_window']} |")
        out.append("")
    out += ["## Across starts: κ (FLOPs) at Δ_max, and the ceiling's gain (nats per byte)", "",
            "| config | " + " | ".join(f"{s['size']} {round(100 * (s['fraction'] or 0))}%" for s in report["starts"]) + " |",
            "|---|" + "---|" * len(report["starts"])]
    labs = sorted({lab for s in report["starts"] for lab in s["configs"]}, key=order_key)
    for lab in labs:
        cells = []
        for s in report["starts"]:
            c, k = s["configs"].get(lab), s["kappa"].get(lab)
            cells.append("-" if c is None else f"{kcell(k[-1]) if k else '-'} / {fmt(c.get('gain'), '.4f')}")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("| S3 icl best | " + " | ".join(
        "-" if not s["s3"] else f"free / {s['s3']['best_gain']:.4f} (×{s['s3']['best']['inference_multiplier']:.1f})"
        for s in report["starts"]) + " |")
    out += ["", f"GPU hours held: {report['gpu_held_hours']:.2f}; witness GPU-seconds: {report['witness_gpu_seconds']:.0f}", ""]
    return "\n".join(out)


def plots(out: Path, report: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for size in sorted({s["size"] for s in report["starts"]}):
        starts = sorted((s for s in report["starts"] if s["size"] == size), key=lambda s: s["fraction"] or 0)
        labs = sorted({lab for s in starts for lab in s["configs"]}, key=order_key)
        colors = {lab: plt.cm.tab10(i % 10) for i, lab in enumerate(labs)}
        fig, axes = plt.subplots(1, len(starts), figsize=(5.2 * len(starts), 4.2), squeeze=False)
        for ax, s in zip(axes[0], starts):
            for lab in labs:
                if lab in s.get("_series", {}):
                    ser = s["_series"][lab]
                    ax.plot(ser["fraction"], ser["raw"], marker=".", ms=3, lw=1, color=colors[lab], label=lab)
            ax.axhline(s["l0"], color="k", lw=0.8, label="ℓ₀")
            for t in s["targets"]:
                ax.axhline(t["loss"], color="grey", lw=0.5, ls="--")
            if s["s3"]:
                ax.axhline(s["s3"]["best"]["val_nats_per_byte"], color="purple", lw=0.8, ls=":", label="S3 best (no training)")
            lo = min([s["l0"] - 1.6 * (s.get("reference_gain_whole") or 0.01)] + [t["loss"] for t in s["targets"]])
            ax.set_ylim(lo - 0.005, s["l0"] + 0.4 * (s["l0"] - lo) + 0.005)
            ax.set_xlabel("FLOPs / budget")
            ax.set_ylabel("held-out nats per byte")
            ax.set_title(f"{size}, start at {round(100 * (s['fraction'] or 0))}%", fontsize=10)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        name = f"loss-budget-{size}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        names.append(name)

        fig, axes = plt.subplots(1, len(starts), figsize=(5.2 * len(starts), 4.2), squeeze=False)
        for ax, s in zip(axes[0], starts):
            for lab in labs:
                row = s["kappa"].get(lab)
                if not row:
                    continue
                xs = [c["delta"] for c in row if c.get("kappa_flops") is not None]
                ys = [c["kappa_flops"] for c in row if c.get("kappa_flops") is not None]
                ax.plot(xs, ys, marker="o", ms=4, color=colors[lab], label=lab)
                lx = [c["delta"] for c in row if c.get("kappa_flops_more_than") is not None]
                ly = [c["kappa_flops_more_than"] for c in row if c.get("kappa_flops_more_than") is not None]
                ax.scatter(lx, ly, marker="^", facecolors="none", edgecolors=colors[lab])
            ax.set_yscale("log")
            ax.axhline(1, color="k", lw=0.6)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.set_xlabel("Δ (nats per byte below ℓ₀)")
            ax.set_ylabel("κ (FLOPs over the reference's)")
            ax.set_title(f"{size}, start at {round(100 * (s['fraction'] or 0))}%; hollow: never reached (>)", fontsize=9)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        name = f"kappa-{size}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        names.append(name)

        if not any(c.get("gain") is not None for s in starts for c in s["configs"].values()):
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for s in starts:
            pts = sorted((c["update_bits"], c["gain"], lab) for lab, c in s["configs"].items() if c.get("gain") is not None)
            line, = ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                            label=f"start at {round(100 * (s['fraction'] or 0))}%")
            for b, g, lab in pts:
                ax.annotate(lab.split(" ", 1)[1], (b, g), fontsize=6, xytext=(2, 2), textcoords="offset points")
            if s["s3"]:
                ax.axhline(s["s3"]["best_gain"], color=line.get_color(), lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("update size (bits)")
        ax.set_ylabel("ceiling's gain ℓ₀ - ceiling (nats per byte)")
        ax.set_title(f"{size}: what each update size buys within the budget; dotted: S3 (no update)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        name = f"ceilings-{size}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        names.append(name)
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-smoke", action="store_true", help="read smoke runs too (the CPU smoke test)")
    ap.add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    runs, skipped, gpu = load(a.run_dirs, a.include_smoke)
    by_start: dict[tuple, list[dict]] = {}
    for r in runs:
        w = r["witness"]
        by_start.setdefault((w["size"], w["start"]["theta_sha256"], w["seed"]), []).append(r)
    starts = sorted((analyze_start(es) for es in by_start.values()), key=lambda s: (s["size"], s["fraction"] or 0))
    report = {"schema": "covert-speedrun/analysis/v1", "qs": QS, "skipped": skipped, "gpu": gpu,
              "gpu_held_hours": sum(g["held_hours"] for g in gpu.values()),
              "witness_gpu_seconds": sum(r["witness"]["gpu_seconds"]["wall"] for r in runs), "starts": starts}
    report["plots"] = plots(a.out, report)
    for s in starts:
        s.pop("_series", None)
    (a.out / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    (a.out / "tables.md").write_text(tables(report))
    print(tables(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
