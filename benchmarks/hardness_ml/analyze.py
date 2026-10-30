"""Loss gaps and effective-compute penalties of constrained runs against the baseline runs.

For a constrained run of size N trained on D tokens to validation loss L:

- ``gap``: L − L_base(N, D), the same-size, same-token baseline's loss.
- ``token_mult``: D / D_eq, where D_eq is the token count at which the same-size baseline reaches L, read off its token
  ladder by linear interpolation in (log D, L); past the ladder's ends it extrapolates from the two nearest points and
  says so (``token_mult_extrapolated``).
- ``compute_penalty``: C / C*(L), where C is the run's training FLOPs and C*(L) the least baseline training compute
  that reaches L, read off the lower envelope of every baseline run (all sizes, all token counts) in (log C, L).
  ``compute_penalty_dense`` uses the dense-shape FLOPs 6·(dense MACs)·D instead of the run's own.
- ``size_penalty``: C / C_base(N, L), the same-size baseline's compute to reach L: ``token_mult`` times the run's training
  FLOPs per token over the baseline's. It leaves out how far size N itself is from compute-optimal at D, which
  ``compute_penalty`` includes.

Baselines are runs with design ``dense``, no noise, no int8 and ``baseline_lr`` (default: each size's best final loss
at the largest token count picks its lr). The ladder and the frontier are seed 0's. A run of another seed is compared
with the baseline of its own seed where one exists: its gap is to that baseline, and its loss is moved by that
baseline's offset from seed 0 before it is read off seed 0's ladder and frontier (seeds share most of their scatter).
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def load(dirs, partial=False):
    """Finished runs from runs/*.json; with ``partial``, also runs still going, from their logs (curve only)."""
    runs = {}
    for d in dirs:
        for p in sorted(Path(d).glob("runs/*.json")):
            r = json.loads(p.read_text())
            runs[r["cfg"]["name"]] = r
        if not partial:
            continue
        for p in sorted(Path(d).glob("logs/*.log")):
            lines = p.read_text().splitlines()
            if p.stem in runs or not lines or not lines[0].startswith("{"):
                continue
            head = json.loads(lines[0])
            curve = []
            for ln in lines[1:]:
                f = ln.split()
                if f and f[0] == "step":
                    curve.append({"step": int(f[1].split("/")[0]), "tokens": int(f[3]), "val": float(f[7])})
            runs[p.stem] = {"cfg": head["cfg"], "tokens": head["steps"] * head["cfg"]["batch"] * head["cfg"]["ctx"],
                            "curve": curve, "partial": True}
    return runs


def is_baseline(r):
    c = r["cfg"]
    return c["design"] == "dense" and not c["int8"] and c["noise_k"] == 0


def interp(xs, ys, y):
    """x at which the piecewise-linear (xs, ys), ys decreasing in xs, equals y; (x, extrapolated)."""
    pts = sorted(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(y0, y1) <= y <= max(y0, y1) and y0 != y1:
            return x0 + (y - y0) * (x1 - x0) / (y1 - y0), False
    if len(pts) < 2:
        return None, True
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if y > pts[0][1] else (pts[-2], pts[-1])
    if y0 == y1:
        return None, True
    return x0 + (y - y0) * (x1 - x0) / (y1 - y0), True


def envelope(points):
    """Lower envelope of (C, L): keep points no cheaper point beats."""
    pts = sorted(points)
    out, best = [], math.inf
    for c, l in pts:
        if l < best:
            out.append((c, l))
            best = l
    return out


def analyze(runs, baseline_lr=None, best=False):
    base = [r for r in runs.values() if is_baseline(r)]
    by_size = defaultdict(list)
    for r in base:
        by_size[r["cfg"]["size"]].append(r)
    lr_of = {}
    for size, rs in by_size.items():
        top = max(r["tokens"] for r in rs)
        cands = [r for r in rs if r["tokens"] == top]
        lr_of[size] = baseline_lr or min(cands, key=lambda r: r["final_val"])["cfg"]["lr"]
    on_lr = [r for r in base if r["cfg"]["lr"] == lr_of[r["cfg"]["size"]]]
    ladder = {s: sorted((r["tokens"], r["final_val"]) for r in on_lr if r["cfg"]["size"] == s and r["cfg"]["seed"] == 0)
              for s in by_size}
    base_at = {(r["cfg"]["size"], r["tokens"], r["cfg"]["seed"]): r["final_val"] for r in on_lr}
    base_lr = {(r["cfg"]["size"], r["tokens"], r["cfg"]["seed"], r["cfg"]["lr"]): r["final_val"] for r in base}
    best0 = {}
    for (s, t, seed, _), v in base_lr.items():
        if seed == 0:
            best0[(s, t)] = min(best0.get((s, t), math.inf), v)
    base_fpt = {s: rs[0]["train_flops_per_token"] for s, rs in by_size.items()}
    frontier = envelope([(r["train_flops"], r["final_val"]) for r in base if r["cfg"]["seed"] == 0])
    rows = []
    for name, r in sorted(runs.items()):
        c = r["cfg"]
        L = r["final_val"]
        row = {"name": name, "size": c["size"], "design": c["design"], "harden": c["harden"],
               "rank_frac": c["rank_frac"] if c["design"] in ("A", "lowrank") else None, "diag": c["diag"],
               "rho": c["rho"] if c["design"] == "B" else None, "int8": c["int8"], "noise_k": c["noise_k"],
               "lr": c["lr"], "uv_lr_mult": c["uv_lr_mult"], "tokens": r["tokens"], "final_val": L,
               "hardened_frac_fwd": r["hardened_frac_fwd"], "train_flops": r["train_flops"],
               "trainable_params": r["trainable_params"], "baseline": is_baseline(r)}
        lad = ladder.get(c["size"], [])
        b0 = base_at.get((c["size"], r["tokens"], 0))
        bs = base_at.get((c["size"], r["tokens"], c["seed"]), b0)
        if bs is not None:
            row["gap"] = L - bs
        bl = base_lr.get((c["size"], r["tokens"], c["seed"], c["lr"]))
        if bl is not None and c["lr"] != lr_of.get(c["size"]):
            row["gap_same_lr"] = L - bl
        L0 = L - (bs - b0) if bs is not None and b0 is not None else L  # on seed 0's ladder, paired by seed
        Lf = L0  # the frontier already holds every lr's baselines
        if best and b0 is not None and (c["size"], r["tokens"]) in best0:
            shift = best0[(c["size"], r["tokens"])] - b0  # the lr-tuned baseline's offset from the ladder's lr
            row["gap"] = row["gap"] - shift
            L0 -= shift
        if len(lad) >= 2:
            ld, ext = interp([math.log(t) for t, _ in lad], [l for _, l in lad], L0)
            if ld is not None:
                row["token_mult"] = r["tokens"] / math.exp(ld)
                row["token_mult_extrapolated"] = ext
                row["size_penalty"] = row["token_mult"] * r["train_flops_per_token"] / base_fpt[c["size"]]
        if len(frontier) >= 2:
            lc, ext = interp([math.log(x) for x, _ in frontier], [l for _, l in frontier], Lf)
            if lc is not None:
                row["compute_penalty"] = r["train_flops"] / math.exp(lc)
                row["compute_penalty_dense"] = r["dense_train_flops"] / math.exp(lc)
                row["compute_penalty_extrapolated"] = ext
        for k, v in r.get("diag_summary", {}).items():
            row["diag." + k] = v
        rows.append(row)
    return {"baseline_lr": lr_of, "ladder": ladder, "frontier": frontier, "rows": rows}


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "y" if v else ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(res):
    cols = ["name", "design", "harden", "rank_frac", "diag", "rho", "noise_k", "tokens", "final_val", "gap",
            "token_mult", "size_penalty", "compute_penalty", "hardened_frac_fwd", "diag.fc1.r_share_out",
            "diag.fc2.r_share_out"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in sorted(res["rows"], key=lambda r: (r["size"], not r["baseline"], r["design"], r["name"])):
        cells = []
        for k in cols:
            v = row.get(k)
            if k in ("token_mult", "size_penalty", "compute_penalty") and v is not None:
                flag = "compute_penalty_extrapolated" if k == "compute_penalty" else "token_mult_extrapolated"
                v = f"{v:.2f}" + ("*" if row.get(flag) else "")
            cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pivot(res, best=False):
    """Arms (run names without their size prefix) by size: gap / size_penalty for each finished run. With ``best``,
    an arm's lr variants (a trailing ``-lr<x>``) fold into the arm, keeping the lowest loss."""
    sizes = sorted({r["size"] for r in res["rows"]}, key=lambda s: int("".join(ch for ch in s if ch.isdigit())))
    cells = defaultdict(dict)
    rows = [r for r in res["rows"] if not r["baseline"] and "gap" in r]
    if best:
        keep = {}
        for r in rows:
            k = (r["size"], re.sub(r"-lr[0-9.e-]+$", "", r["name"]))
            if k not in keep or r["final_val"] < keep[k]["final_val"]:
                keep[k] = {**r, "name": k[1]}
        rows = [{k: v for k, v in r.items() if k != "gap_same_lr"} for r in keep.values()]
    for r in rows:
        arm = r["name"].split("-", 1)[1]
        pen = r.get("size_penalty")
        cells[arm][r["size"]] = f"{r['gap']:+.3f} / " + (f"{pen:.2f}×" if pen is not None else "-") + (
            f" (same lr {r['gap_same_lr']:+.3f})" if "gap_same_lr" in r else "")
    lines = ["| arm | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
    for arm in sorted(cells):
        lines.append(f"| {arm} | " + " | ".join(cells[arm].get(s, "") for s in sizes) + " |")
    return "\n".join(lines)


def curves(runs, partial_dir_logs=()):
    """Gap to the same-size baseline at each eval step, for runs sharing the baseline's token count and schedule."""
    out = {}
    for name, r in runs.items():
        c = r["cfg"]
        base = runs.get(f"{c['size']}-base-D{c['tokens']}")
        if base is None or name == base["cfg"]["name"]:
            continue
        bv = {p["step"]: p["val"] for p in base["curve"]}
        out[name] = [(p["tokens"], round(p["val"] - bv[p["step"]], 4)) for p in r["curve"] if p["step"] in bv]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="sweep out dirs (each with runs/*.json)")
    ap.add_argument("--baseline-lr", type=float, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--curves", action="store_true", help="print each run's gap to its baseline at every eval")
    ap.add_argument("--pivot", action="store_true", help="print arms by size: gap / size_penalty")
    ap.add_argument("--best", action="store_true",
                    help="against each size's best baseline over the lrs run (the ladder shifted by its offset), "
                         "and with --pivot each arm at its best lr")
    a = ap.parse_args()
    if a.curves:
        for name, pts in sorted(curves(load(a.dirs, partial=True)).items()):
            print(name, " ".join(f"{t / 1e6:.0f}M:{g:+.3f}" for t, g in pts[::2] + pts[-1:]))
        return
    res = analyze(load(a.dirs), a.baseline_lr, a.best)
    if a.pivot:
        print(pivot(res, a.best))
        return
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print("baseline lr:", res["baseline_lr"])
    print("frontier (C, L):", [(f"{c:.2e}", round(l, 4)) for c, l in res["frontier"]])
    print(table(res))
    print("\n* extrapolated past the baseline ladder or frontier")


if __name__ == "__main__":
    main()
