"""p2b.py RUNS_DIR NOW_S RUN_A RUN_B: CIR-272's second pass under four runs at a time (the coordinator, 07:38Z 13 Nov). Reads
p2-plan.json (p2.py's) and the types runs A and B (types-p2a.tsv, types-p2b.tsv) have ended, fetched under RUNS_DIR/RUN/out;
NOW_S is each run's seconds since its start, now. Writes beside this file:

    types-p2c.tsv  a 64 GB run's list: the types of A's and B's lists past what A and B can still reach (each run starts no
                   type once fewer than its limit and 300 s of its 21,000 s are left, so it reaches at most the types whose
                   0.85 x estimated time fits before then, plus 3), cheapest first, so no type is staged twice
    types-p2e.tsv  the 18 extras, in p2.py's order, for the first run slot A or B frees
    types-p2d.tsv  the 160 GB run's list: the 14 types the 64 GB rule left out in v3 (13 Lean, 1 stager), each estimated at
                   128 GB or under, cheapest first
    p2-plan.json   updated: DsaMqaLogits_v3's estimates refit on A's and B's measured N (time, stager, circuit, linear in N),
                   which puts N past 45 over the 60 GB Lean gate; each tag's list and limit"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
GIB, RATIO, BUDGET = 2**30, 4.42, 21000
runs_dir, now = Path(sys.argv[1]), int(sys.argv[2])
RA, RB = sys.argv[3], sys.argv[4]
P = json.loads((HERE / "p2-plan.json").read_text())
lists = {w: [l.split("\t") for l in (HERE / f"types-p2{w}.tsv").read_text().splitlines() if l] for w in "ab"}


def fit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - k * mx, k


pts = []
for run in (RA, RB):
    for d in (runs_dir / run / "out").glob("t*/done"):
        tag = d.parent.name
        if not P[tag]["block"].startswith("DsaMqaLogits_v3"):
            continue
        st = json.loads((d.parent / "B" / "stage.json").read_text())
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", (d.parent / "B" / "stage.time").read_text())[1]) / 2**20
        pts.append((int(re.search(r"N=(\d+)", P[tag]["block"])[1]), json.loads(d.read_text())["seconds"], rss, st["circuit_bytes"] / GIB, run, tag))
assert len(pts) >= 5, pts
Ns = [p[0] for p in pts]
FT, FG, FC = (fit(Ns, [p[i] for p in pts]) for i in (1, 2, 3))
src = f"DsaMqaLogits_v3 at N={min(Ns)}..{max(Ns)}, measured in {RA} and {RB} ({len(pts)} types), each linear in N"
for t, p in P.items():
    if p["block"].startswith("DsaMqaLogits_v3"):
        n = int(re.search(r"N=(\d+)", p["block"])[1])
        tot, gb, gib = (a + k * n for a, k in (FT, FG, FC))
        p["estimate"] = {"basis": src, "kind": "inferred", "total_s": round(tot), "stage_gb": round(gb, 1), "circuit_gib": round(gib, 2),
                         "lean_gb": round(RATIO * gib, 1)}
        p["run"] = gb <= 62 and RATIO * gib <= 60

# what A and B can still reach: their lists from the first type not ended, at 0.85 x the estimate, before each stops starting
reach = {}
for w, run in (("a", RA), ("b", RB)):
    out = runs_dir / run / "out"
    ended = {d.parent.name for d in out.glob("t*/done")} | {d.parent.name for d in out.glob("x*/done")}
    rest = [r for r in lists[w] if r[0] not in ended]
    t, k = now, 0
    for i, r in enumerate(rest):
        lim = int(r[4])
        if t + lim + 300 > BUDGET:
            break
        e = P[r[0]]["estimate"].get("total_s") or 120
        t += 0.85 * e
        k = i + 1
    reach[w] = {"ended": sorted(ended), "may_reach": [r[0] for r in rest[: k + 3]], "beyond": [r[0] for r in rest[k + 3:]]}

D14 = ["t032", "t099", "t105", "t106", "t107", "t108", "t109", "t110", "t111", "t112", "t114", "t116", "t118", "t120"]
c = [t for w in "ab" for t in reach[w]["beyond"][2:] if t not in D14 and P[t]["run"] and not P[t]["extra"]]
c.sort(key=lambda t: (P[t]["estimate"].get("total_s") or 1580, t))
ex = sorted((t for t in P if t.startswith("x") and P[t]["run"]), key=lambda t: P[t]["order"])
for w, ts in (("c", c), ("e", ex)):
    for i, t in enumerate(ts):
        e = P[t]["estimate"].get("total_s")
        P[t].update(list=w, order=i + 1, limit_s=min(3600, max(900, 2 * e)) if e else (2700 if not P[t]["extra"] else 1800))


def d_est(t):
    p = P[t]
    e = dict(p["estimate"])
    if t == "t032":
        e = {"basis": "3 x Fp8GroupQuantUe8m0_v1{K=2048,G=128} (t033, accepted: 8724721893 bytes, 21.38 GB, 550 s staging, 1,391 s in all), "
                      "linear in K; its own run r20261112-215921-f101 stopped its stager past 62 GB at 613 s", "kind": "inferred",
             "stage_s": 1650, "stage_gb": 64.1, "circuit_gib": round(3 * 8724721893 / GIB, 2), "lean_gb": round(RATIO * 3 * 8724721893 / GIB, 1),
             "total_s": 1650 + round(99 * 3 * 8724721893 / GIB)}
    elif "circuit_gib" in e and "stage_s" in e:
        e["total_s"] = round(e["stage_s"] + 99 * e["circuit_gib"])
        e["basis"] += "; at 160 GB: time = its staging + 99 s per circuit GiB, t104's Lean and session (930 s on 9.39 GiB)"
    return e


d = []
for t in D14:
    e = d_est(t)
    P[t]["estimate_160"] = e
    if max(e.get("stage_gb") or 0, e.get("lean_gb") or 0) <= 128:
        d.append(t)
d.sort(key=lambda t: P[t]["estimate_160"]["total_s"])
for i, t in enumerate(d):
    P[t].update(run=True, list="d", order=i + 1, limit_s=min(6000, 2 * P[t]["estimate_160"]["total_s"]), mem_gb=160)
for w, ts in (("c", c), ("d", d), ("e", ex)):
    (HERE / f"types-p2{w}.tsv").write_text("".join(f"{t}\tcensus:{P[t]['block']}\t1\tsession\t{P[t]['limit_s']}\n" for t in ts))
P["_split"] = {"at_s": now, "runs": {"a": RA, "b": RB}, "reach": reach, "dsa_fit": {"points": pts, "total_s": FT, "stage_gb": FG, "circuit_gib": FC}}
(HERE / "p2-plan.json").write_text(json.dumps(P, indent=1) + "\n")
print({w: (len(reach[w]["ended"]), len(reach[w]["may_reach"]), len(reach[w]["beyond"])) for w in "ab"})
print("c", len(c), sum(P[t]["estimate"].get("total_s") or 0 for t in c), c)
print("d", len(d), [(t, P[t]["estimate_160"]["total_s"], P[t]["estimate_160"].get("lean_gb")) for t in d])
print("dsa not run:", sorted(int(re.search(r"N=(\d+)", p["block"])[1]) for t, p in P.items() if t[0] in "tx" and p["block"].startswith("DsaMqa") and not p["run"]))
