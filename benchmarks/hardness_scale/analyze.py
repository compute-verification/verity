"""The effective-compute penalty P(κ, N) of each arrangement against the dense int8-QAT baselines.

For a run of size N (the dense model's non-embedding parameters) on D tokens reaching validation loss L:

- ``gap``: L − L_base(N, D) at the size's baseline lr, paired by seed. lr* is the lr of the size's token ladder (the
  most baseline token counts, then the best loss at D): the lr its arms run at. The envelope reads every lr.
- ``P_env``: C / C_env(L). C is the run's dense-shape training FLOPs (6 a forward MAC, adapters included, no discount
  for frozen R), C_env(L) the least baseline compute reaching L, on the lower envelope of every seed-0 baseline run (all
  sizes, token counts and lrs), interpolated linearly in (log C, L). ``P_env_real`` uses the run's own FLOPs (4 a frozen
  R MAC). Past the envelope's ends it extrapolates from the end segment and marks it (*).
- ``P_size``: D / D_eq · (FLOPs a token over the baseline's), with D_eq the tokens the same-size baseline at lr* needs to
  reach L, read off its token ladder in (log D, L). It leaves out how far N is from compute-optimal.
- ``P_fit``: C / C*(L) on the frontier of L(N, D) = E + A/N^α + B/D^β fitted to each size's seed-0 ladder at its lr*
  with D ≥ 10N, with C*(L) minimized over the width of the 12-block shape at its own FLOPs a token.
- ``best``: with --best, each arm's lr variants fold into the arm, keeping its lowest loss.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

VOCAB, CTX, LAYERS = 49152, 1024, 12


def load(paths):
    runs = {}
    for p in paths:
        p = Path(p)
        files = [p] if p.suffix == ".json" else list(p.glob("**/runs/*.json"))
        for f in files:
            r = json.loads(f.read_text())
            if r["cfg"].get("synthetic") or r["cfg"].get("max_steps"):
                continue
            runs[r["cfg"]["name"]] = r
    return runs


def arm_of(r):
    m = re.match(r"^[0-9]+m-(.+?)-lr", r["cfg"]["name"])
    return m.group(1) if m else r["cfg"]["name"]


def size_of(r):
    return r["cfg"]["name"].split("-", 1)[0]


def is_base(r):
    c = r["cfg"]
    return c["arr"] == "none" and c["qat"]


def interp(xs, ys, y):
    """x where the piecewise-linear (xs, ys), ys falling in xs, reaches y: (x, extrapolated)."""
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
    out, best = [], math.inf
    for c, l in sorted(points):
        if l < best:
            out.append((c, l))
            best = l
    return out


def flops_per_token(d):
    return 6 * (LAYERS * 12 * d * d + LAYERS * CTX * d + VOCAB * d)


def nonemb(d):
    return LAYERS * (12 * d * d + 2 * d + 128) + d


def fit(base):
    """L = E + A/N^α + B/D^β by a grid over (α, β) with nonnegative least squares for (E, A, B)."""
    if len(base) < 6:
        return None
    N = np.array([r["nonemb_dense"] for r in base], float)
    D = np.array([r["tokens"] for r in base], float)
    L = np.array([r["final_val"] for r in base], float)
    best = None
    for a in np.arange(0.1, 1.01, 0.01):
        for b in np.arange(0.1, 1.01, 0.01):
            X = np.stack([np.ones_like(N), N ** -a, D ** -b], 1)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            if (coef < 0).any():
                continue
            err = float(((X @ coef - L) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, a, b, *coef)
    if best is None:
        return None
    err, a, b, E, A, B = best
    return {"E": E, "A": A, "B": B, "alpha": a, "beta": b, "rmse": math.sqrt(err)}


def c_star(f, L):
    best = math.inf
    for d in np.exp(np.linspace(math.log(64), math.log(8192), 400)):
        rest = L - f["E"] - f["A"] / nonemb(d) ** f["alpha"]
        if rest <= 0:
            continue
        D = (f["B"] / rest) ** (1 / f["beta"])
        best = min(best, flops_per_token(d) * D)
    return best


def analyze(runs, best_lr=False):
    base = [r for r in runs.values() if is_base(r)]
    lr_star = {}
    for size in {size_of(r) for r in base}:
        cands = [r for r in base if size_of(r) == size and r["cfg"]["seed"] == 0]
        rungs = defaultdict(set)
        for r in cands:
            rungs[r["cfg"]["lr"]].add(r["tokens"])
        d1 = {r["cfg"]["lr"]: r["final_val"] for r in cands if "-Dx" not in r["cfg"]["name"]}
        lr_star[size] = max(rungs, key=lambda lr: (len(rungs[lr]), -d1.get(lr, math.inf)))
    ladder = defaultdict(list)
    base_at = {}
    for r in base:
        s = size_of(r)
        if r["cfg"]["lr"] == lr_star[s]:
            base_at[(s, r["tokens"], r["cfg"]["seed"])] = r
            if r["cfg"]["seed"] == 0:
                ladder[s].append((r["tokens"], r["final_val"]))
    env = envelope([(r["train_flops_dense"], r["final_val"]) for r in base if r["cfg"]["seed"] == 0])
    f = fit([r for r in base if r["cfg"]["seed"] == 0 and r["cfg"]["lr"] == lr_star[size_of(r)]
             and r["tokens"] >= 10 * r["nonemb_dense"]])
    rows = []
    for name, r in runs.items():
        c, s, L = r["cfg"], size_of(r), r["final_val"]
        row = {"name": name, "size": s, "arm": arm_of(r), "lr": c["lr"], "seed": c["seed"], "tokens": r["tokens"],
               "kappa": r["kappa"], "kappa_lin": r["kappa_lin"], "val": L, "base": is_base(r)}
        b0, bs = base_at.get((s, r["tokens"], 0)), base_at.get((s, r["tokens"], c["seed"]))
        if bs or b0:
            row["gap"] = L - (bs or b0)["final_val"]
        L0 = L - (bs["final_val"] - b0["final_val"]) if bs and b0 else L  # on seed 0's curves, paired by seed
        if len(ladder[s]) >= 2:
            x, ext = interp([math.log(t) for t, _ in ladder[s]], [v for _, v in ladder[s]], L0)
            if x is not None:
                bfpt = next(v for v in base if size_of(v) == s)["train_flops_dense_per_token"]
                row["P_size"] = r["tokens"] / math.exp(x) * r["train_flops_dense_per_token"] / bfpt
                row["P_size_ext"] = ext
        if len(env) >= 2:
            x, ext = interp([math.log(v) for v, _ in env], [v for _, v in env], L0)
            if x is not None:
                row["P_env"] = r["train_flops_dense"] / math.exp(x)
                row["P_env_real"] = r["train_flops"] / math.exp(x)
                row["P_env_ext"] = ext
        if f:
            cs = c_star(f, L0)
            if math.isfinite(cs):
                row["P_fit"] = r["train_flops_dense"] / cs
        a = r.get("ablations") or {}
        if a:
            row["d_perm"] = a["val_r_permuted"] - a["val_eager"]
            row["d_removed"] = a["val_r_removed"] - a["val_eager"]
            row["r_share"] = a.get("r_share_out")
        rows.append(row)
    if best_lr:
        keep = {}
        for row in rows:
            k = (row["size"], row["arm"], row["tokens"], row["seed"])
            if k not in keep or row["val"] < keep[k]["val"]:
                keep[k] = row
        rows = list(keep.values())
    return {"lr_star": lr_star, "envelope": env, "fit": f, "rows": rows}


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(res):
    cols = ["name", "kappa", "kappa_lin", "val", "gap", "P_size", "P_env", "P_env_real", "P_fit", "d_perm", "d_removed"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(res["rows"], key=lambda r: (int(r["size"][:-1]), not r["base"], r["arm"], r["name"])):
        cells = []
        for k in cols:
            v = r.get(k)
            if k in ("P_size", "P_env", "P_env_real") and v is not None:
                v = f"{v:.2f}" + ("*" if r.get(k.replace("_real", "") + "_ext") else "")
            elif k == "P_fit" and v is not None:
                v = f"{v:.2f}"
            cells.append(fmt(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pivot(res, key="P_env"):
    sizes = sorted({r["size"] for r in res["rows"]}, key=lambda s: int(s[:-1]))
    cells = defaultdict(dict)
    for r in res["rows"]:
        if r["base"] or r["seed"] or "-Dx" in r["name"]:
            continue
        v = r.get(key)
        cells[r["arm"]][r["size"]] = (f"κ {r['kappa']:.2f}: {r['gap']:+.3f} / "
                                      + (f"{v:.2f}×" + ("*" if r.get("P_env_ext") else "") if v else "-"))
    out = ["| arm | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
    for arm in sorted(cells):
        out.append(f"| {arm} | " + " | ".join(cells[arm].get(s, "") for s in sizes) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run dirs, out dirs or result files")
    ap.add_argument("--best", action="store_true", help="each arm at its best lr")
    ap.add_argument("--pivot", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = analyze(load(a.paths), a.best)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print("lr*:", res["lr_star"])
    print("envelope (C, L):", [(f"{c:.2e}", round(l, 4)) for c, l in res["envelope"]])
    print("fit:", res["fit"])
    print(pivot(res) if a.pivot else table(res))
    print("\n* extrapolated past the baseline ladder or envelope")


if __name__ == "__main__":
    main()
