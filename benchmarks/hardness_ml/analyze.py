"""Loss gaps and effective-compute penalties of constrained runs against the baseline runs.

For a constrained run of size N trained on D tokens to validation loss L:

- ``gap``: L − L_base(N, D), the same-size, same-token baseline's loss.
- ``token_mult``: D / D_eq, where D_eq is the token count at which the same-size baseline reaches L, read off its token
  ladder by linear interpolation in (log D, L); past the ladder's ends it extrapolates from the two nearest points and
  says so (``token_mult_extrapolated``).
- ``compute_penalty``: C / C*(L), where C is the run's training FLOPs and C*(L) the least baseline training compute
  that reaches L, read off the lower envelope of every baseline run (all sizes, all token counts) in (log C, L).
  ``compute_penalty_dense`` uses the dense-shape FLOPs 6·(dense MACs)·D instead of the run's own.

Baselines are runs with design ``dense``, no noise, no int8 and ``baseline_lr`` (default: each size's best final loss
at the largest token count picks its lr).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(dirs):
    runs = {}
    for d in dirs:
        for p in sorted(Path(d).glob("runs/*.json")):
            r = json.loads(p.read_text())
            runs[r["cfg"]["name"]] = r
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


def analyze(runs, baseline_lr=None):
    base = [r for r in runs.values() if is_baseline(r)]
    by_size = defaultdict(list)
    for r in base:
        by_size[r["cfg"]["size"]].append(r)
    lr_of = {}
    for size, rs in by_size.items():
        top = max(r["tokens"] for r in rs)
        cands = [r for r in rs if r["tokens"] == top]
        lr_of[size] = baseline_lr or min(cands, key=lambda r: r["final_val"])["cfg"]["lr"]
    ladder = {s: sorted(((r["tokens"], r["final_val"]) for r in rs if r["cfg"]["lr"] == lr_of[s]))
              for s, rs in by_size.items()}
    frontier = envelope([(r["train_flops"], r["final_val"]) for r in base])
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
        same = [l for t, l in lad if t == r["tokens"]]
        if same:
            row["gap"] = L - same[0]
        if len(lad) >= 2:
            ld, ext = interp([math.log(t) for t, _ in lad], [l for _, l in lad], L)
            if ld is not None:
                row["token_mult"] = r["tokens"] / math.exp(ld)
                row["token_mult_extrapolated"] = ext
        if len(frontier) >= 2:
            lc, ext = interp([math.log(x) for x, _ in frontier], [l for _, l in frontier], L)
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
            "token_mult", "compute_penalty", "hardened_frac_fwd", "diag.fc1.r_share_out", "diag.fc2.r_share_out"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in sorted(res["rows"], key=lambda r: (r["size"], not r["baseline"], r["design"], r["name"])):
        cells = []
        for k in cols:
            v = row.get(k)
            if k in ("token_mult", "compute_penalty") and v is not None:
                v = f"{v:.2f}" + ("*" if row.get(k.replace("_mult", "_mult_extrapolated").replace(
                    "penalty", "penalty_extrapolated")) else "")
            cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="sweep out dirs (each with runs/*.json)")
    ap.add_argument("--baseline-lr", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = analyze(load(a.dirs), a.baseline_lr)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print("baseline lr:", res["baseline_lr"])
    print("frontier (C, L):", [(f"{c:.2e}", round(l, 4)) for c, l in res["frontier"]])
    print(table(res))
    print("\n* extrapolated past the baseline ladder or frontier")


if __name__ == "__main__":
    main()
