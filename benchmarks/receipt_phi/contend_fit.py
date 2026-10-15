"""contend.sh's results: each filler class's cost to the tensor pipe, and which model explains it.

For every type (FP8, INT8), class, ρ (fillers per MMA) and W (warps per scheduler), the cost of a filler is
w = (c(ρ) - c(0)) / ρ, in scheduler cycles per filler instruction, where c is the loop's cycles per MMA per scheduler
(16 at the tensor pipe's peak). Its error is half the gap between the two one-SM passes plus the larger rep spread.
Models compared on every point:

- additive: c = c0 + w·ρ;
- dispatch slot: c = max(c0, k + o + ρ), each instruction one issue slot of 1 per cycle, an MMA k slots, o the loop's
  overhead instructions per MMA (from the SASS);
- pipe: c = max(c0, ρ·p), p the class's pipe cycles per instruction on one scheduler.

    python3 contend_fit.py OUTDIR > fit.json   (and a Markdown summary on stderr)
"""

import csv
import collections
import json
import os
import statistics
import sys

RHO_ASKED = (0.25, 0.5, 1, 2, 4)


def load(path):
    return [r for r in (json.loads(x) for x in open(path) if x.startswith("{")) if r.get("ok")]


def by_key(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[(r["type"], r["cls"], r["F"], r["W"])].append(r)
    return d


def ncu_rows(path):
    lines = [x for x in open(path, errors="replace") if x.startswith('"')]
    if len(lines) < 3:
        return None
    rows = list(csv.reader(lines))
    head, vals = rows[0], rows[2]
    out = {}
    for k, v in zip(head, vals):
        try:
            out[k] = float(v.replace(",", ""))
        except ValueError:
            out[k] = v
    return out


def main():
    d = sys.argv[1]
    a, b = load(os.path.join(d, "sweep-one-a.jsonl")), load(os.path.join(d, "sweep-one-b.jsonl"))
    al = load(os.path.join(d, "sweep-all.jsonl")) if os.path.exists(os.path.join(d, "sweep-all.jsonl")) else []
    sass = json.load(open(os.path.join(d, "contend-sass.json")))
    exact = {(s["type"], s["cls"], s["F"]): s for s in sass}
    ka, kb, kal = by_key(a), by_key(b), by_key(al)

    def cpm(rows):
        return statistics.median(r["cyc_per_mma"] for r in rows)

    def spread(rows):
        return max((r["cycles_hi"] - r["cycles_lo"]) / r["cycles"] * r["cyc_per_mma"] for r in rows)

    types, classes = ("fp8", "s8"), sorted({r["cls"] for r in a if r["cls"] != "none"})
    Ws = sorted({r["W"] for r in a})
    res = {"base": {}, "points": [], "sass_inexact": [], "fit": {}}
    for t in types:
        for W in Ws:
            ra, rb = ka[(t, "none", 0, W)], kb[(t, "none", 0, W)]
            res["base"][f"{t}/W{W}"] = {"cyc_per_mma_a": [r["cyc_per_mma"] for r in ra],
                                        "cyc_per_mma_b": [r["cyc_per_mma"] for r in rb],
                                        "all_sms": [r["cyc_per_mma"] for r in kal.get((t, "none", 0, W), [])]}
    for s in sass:
        if not s["exact"]:
            res["sass_inexact"].append({k: s[k] for k in ("type", "cls", "F", "mma", "fillers", "other")})
    ovh = {t: sum(exact[(t, "none", 0)]["other"].values()) / 16 for t in types if (t, "none", 0) in exact}
    for t in types:
        for c in classes:
            for W in Ws:
                c0a, c0b = cpm(ka[(t, "none", 0, W)]), cpm(kb[(t, "none", 0, W)])
                for F in sorted({r["F"] for r in a if r["cls"] == c}):
                    ra, rb = ka.get((t, c, F, W)), kb.get((t, c, F, W))
                    if not ra or not rb:
                        continue
                    rho = F / 16
                    da, db = cpm(ra) - c0a, cpm(rb) - c0b
                    err = abs(da - db) / 2 + max(spread(ra), spread(rb), spread(ka[(t, "none", 0, W)]))
                    rall = kal.get((t, c, F, W))
                    dall = cpm(rall) - cpm(kal[(t, "none", 0, W)]) if rall and kal.get((t, "none", 0, W)) else None
                    s = exact.get((t, c, F), {})
                    res["points"].append({
                        "type": t, "cls": c, "F": F, "rho": rho, "W": W, "c0": (c0a + c0b) / 2,
                        "c": (cpm(ra) + cpm(rb)) / 2, "dc": (da + db) / 2, "dc_err": err, "w": (da + db) / 2 / rho,
                        "w_err": err / rho, "dc_all_sms": dall, "sass_exact": s.get("exact"),
                        "tensor_eff": 16 / ((cpm(ra) + cpm(rb)) / 2)})
    # Per type, class and W: w over the asked ρ range (least squares through the origin of dc on ρ), and the largest ρ
    # whose dc is within its error of zero (the free budget).
    for t in types:
        for c in classes:
            for W in Ws:
                pts = [p for p in res["points"] if p["type"] == t and p["cls"] == c and p["W"] == W]
                asked = [p for p in pts if p["rho"] in RHO_ASKED]
                if not asked:
                    continue
                sxx = sum(p["rho"] ** 2 for p in asked)
                w = sum(p["rho"] * p["dc"] for p in asked) / sxx
                resid = max(abs(p["dc"] - w * p["rho"]) for p in asked)
                free = 0.0
                for p in sorted(pts, key=lambda p: p["rho"]):
                    if abs(p["dc"]) <= max(2 * p["dc_err"], 0.01 * p["c0"]):
                        free = p["rho"]
                    else:
                        break
                res["fit"][f"{t}/{c}/W{W}"] = {"w": w, "max_resid": resid, "free_rho": free,
                                               "w_by_rho": {p["rho"]: round(p["w"], 4) for p in pts}}
    res["loop_overhead_per_mma"] = ovh
    ncu = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("ncu-") and f.endswith(".csv"):
            r = ncu_rows(os.path.join(d, f))
            if r:
                keep = {k: v for k, v in r.items() if isinstance(v, float) and (
                    "issue_active" in k or "pipe_tensor" in k or "inst_executed.sum" in k or "inst_issued" in k
                    or ("issue_stalled" in k and "per_issue_active" in k and v >= 0.05))}
                ncu[f[4:-4]] = keep
    res["ncu"] = ncu
    json.dump(res, sys.stdout, indent=1)
    md = ["| type | class | W | w (cyc/instr) | w at ρ=0.25 0.5 1 2 4 8 16 | free ρ |", "|---|---|---|---|---|---|"]
    for key, v in res["fit"].items():
        t, c, W = key.split("/")
        wr = " ".join(f"{v['w_by_rho'].get(r, float('nan')):.2f}" for r in (0.25, 0.5, 1, 2, 4, 8, 16))
        md.append(f"| {t} | {c} | {W[1:]} | {v['w']:.3f} ± {v['max_resid']:.3f} | {wr} | {v['free_rho']} |")
    print("\n".join(md), file=sys.stderr)


if __name__ == "__main__":
    main()
