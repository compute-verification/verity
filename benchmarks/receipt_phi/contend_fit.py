"""contend.sh's results: each filler class's cost to the tensor pipe, by regime.

c is the loop's cycles per MMA per scheduler (16 at the tensor pipe's peak), measured on one SM; the bare loop's c0 is
16.000 from W = 2 warps per scheduler. A filler's cost is w = (c(ρ) - c0) / ρ scheduler cycles, ρ fillers per MMA, over
W = 2, 4, 8 (W = 1 is reported apart: one warp leaves the pipe idle at the loop branch). Cycle counts repeat exactly
across reps, passes and SMs, so a w's error is its spread over W and ρ within its regime, not noise:

- dilute, ρ = 0.25-2: the least-squares w through the origin, and the range of the per-point w;
- ρ = 4, and saturated, ρ = 8 and 16, where the class's own pipe (`PIPE`, cycles per instruction on one scheduler)
  can bind: there c is compared with max(16 + w_dilute·ρ, p·ρ);
- free: the largest ρ whose loss (1 - c0/c) is under 0.2%, at every W >= 2.

Split rows (fillers in their own warps) use the MMA warps' span. `python3 contend_fit.py OUTDIR > fit.json` writes the
fit and prints the Markdown table on stderr.
"""

import collections
import json
import os
import statistics
import sys

DILUTE = (0.25, 0.5, 1, 2)
PIPE = {"iadd3": 2, "lop3": 2, "shf": 2, "prmt": 2, "iadd3r2": 2, "iadd3c": 2, "iadd3i": 2, "rot1": 2, "imad": 2,
        "idp4a": 2, "ffma": 1, "ffma3": 1.5, "lds": 8, "mix_iadd3_shf": 2, "mix_iadd3_lds": 4}


def load(path):
    if not os.path.exists(path):
        return []
    return [r for r in (json.loads(x) for x in open(path) if x.startswith("{")) if r.get("ok")]


def table(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[(r["type"], r["cls"], r["F"], r["W"])].append(r["cyc_per_mma"])
    return {k: statistics.median(v) for k, v in d.items()}


def fit(c, t, cls, Ws, nfill):
    """nfill[F]: the filler instructions the SASS has for F (ptxas can emit more than F); w is per instruction."""
    pts, act = {}, {}
    for W in Ws:
        c0 = c.get((t, "none", 0, W))
        for F in (2, 4, 8, 16, 32, 64, 128, 256):
            v = c.get((t, cls, F, W))
            if c0 is not None and v is not None:
                pts[(F / 16, W)] = (v, c0)
                act[F / 16] = nfill.get(F, F) / 16
    if not pts:
        return None
    dil = [(act[r], v - c0) for (r, W), (v, c0) in pts.items() if r in DILUTE]
    w = sum(r * d for r, d in dil) / sum(r * r for r, _ in dil)
    each = [d / r for r, d in dil]
    by = lambda rho: [(v - c0) / act[rho] for (r, W), (v, c0) in pts.items() if r == rho]
    free = 0.0
    for rho in sorted({r for r, _ in pts}):
        if all(1 - c0 / v < 0.002 for (r, W), (v, c0) in pts.items() if r == rho):
            free = rho
        else:
            break
    sat = {}
    for rho in (4, 8, 16):
        got = [v for (r, W), (v, c0) in pts.items() if r == rho]
        if got:
            pred = max(16 + w * act[rho], PIPE.get(cls, 0) * act[rho])
            sat[rho] = {"c": [min(got), max(got)], "w": [min(by(rho)), max(by(rho))], "model": pred,
                        "model_err": [min(got) / pred - 1, max(got) / pred - 1]}
    return {"w": w, "w_range": [min(each), max(each)], "free_rho": free, "regimes": sat,
            "loss_at_smallest": {f"rho={r}": round(1 - c0 / v, 5) for (r, W), (v, c0) in sorted(pts.items())
                                 if r == min(x for x, _ in pts)}}


def main():
    d = sys.argv[1]
    one = load(os.path.join(d, "sweep-one-a.jsonl")) + load(os.path.join(d, "sweep-one-b.jsonl"))
    allsm = load(os.path.join(d, "sweep-all.jsonl"))
    split = load(os.path.join(d, "split.jsonl"))
    c, call, cs = table(one), table(allsm), table(split)
    res = {"base": {f"{t}/W{W}": v for (t, cl, F, W), v in sorted(c.items()) if cl == "none"},
           "all_sms_max_dev": max((abs(call[k] - v) for k, v in c.items() if k in call), default=None),
           "fused": {}, "split": {}, "w1": {}}
    sass_path = os.path.join(d, "contend-sass.json")
    sass = json.load(open(sass_path)) if os.path.exists(sass_path) else []
    res["sass_inexact"] = [(s["type"], s["cls"], s["F"], s["fillers"]) for s in sass if not s["exact"]]
    nfill = collections.defaultdict(dict)
    for s in sass:
        nfill[(s["type"], s["cls"])][s["F"]] = s["fillers"]
    nsplit = collections.defaultdict(dict)
    for s in sass:
        if "fill_only" in s["split"]:
            nsplit[(s["type"], s["cls"])][s["F"]] = s["split"]["fill_only"]["fillers"]
    classes = sorted({k[1] for k in c if k[1] != "none"})
    for t in ("fp8", "s8"):
        for cls in classes:
            f = fit(c, t, cls, (2, 4, 8), nfill[(t, cls)])
            if f:
                res["fused"][f"{t}/{cls}"] = f
            f1 = fit(c, t, cls, (1,), nfill[(t, cls)])
            if f1:
                res["w1"][f"{t}/{cls}"] = {"w": f1["w"], "w_range": f1["w_range"]}
            if split:
                # split W counts both groups: W = 4 is 2 MMA warps + 2 filler warps a scheduler, against the bare
                # loop with the filler warps idle (F = 0 in split mode)
                fs = fit(cs, t, cls, (4, 8), nsplit[(t, cls)])
                if fs:
                    res["split"][f"{t}/{cls}"] = fs
    json.dump(res, sys.stdout, indent=1)
    md = ["| class | FP8 w, ρ ≤ 2 | INT8 w, ρ ≤ 2 | FP8 / INT8 w at ρ = 4 | at ρ = 16 (model err) | free ρ FP8 / INT8 |"
          + (" split FP8 / INT8 w |" if split else ""),
          "|---|---|---|---|---|---|" + ("---|" if split else "")]
    for cls in classes:
        a, b = res["fused"].get(f"fp8/{cls}"), res["fused"].get(f"s8/{cls}")
        if not a or not b:
            continue
        r4 = lambda x: "{:.2f}-{:.2f}".format(*x["regimes"][4]["w"]) if 4 in x["regimes"] else "-"
        r16 = lambda x: ("{:.2f}-{:.2f} ({:+.0%})".format(*x["regimes"][16]["w"], x["regimes"][16]["model_err"][1])
                         if 16 in x["regimes"] else "-")
        row = (f"| {cls} | {a['w']:.2f} [{a['w_range'][0]:.2f}, {a['w_range'][1]:.2f}] | {b['w']:.2f} "
               f"[{b['w_range'][0]:.2f}, {b['w_range'][1]:.2f}] | {r4(a)} / {r4(b)} | {r16(a)} / {r16(b)} | "
               f"{a['free_rho']} / {b['free_rho']} |")
        if split:
            sa, sb = res["split"].get(f"fp8/{cls}"), res["split"].get(f"s8/{cls}")
            row += f" {sa['w']:.2f} / {sb['w']:.2f} |" if sa and sb else " - |"
        md.append(row)
    print("\n".join(md), file=sys.stderr)


if __name__ == "__main__":
    main()
