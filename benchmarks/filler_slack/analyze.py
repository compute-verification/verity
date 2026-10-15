"""ν three ways from a run's outputs: `python3 analyze.py OUT > summary.json`.

ν is free issue slots over total issue slots per scheduler per cycle (an SM has 4 schedulers, each issuing at most one
warp-instruction a cycle). Rates are warp-MMAs per SM-clock (peak 0.25: one m16n8k32 per SMSP every 16 cycles).

- (a) probe: a probe point's ν is its warp-instructions per SM-clock over 4, scaled by the probe loop's SASS count over its
  256 ops (the loop's own counter, flag load and branch). Per rep, ν(thr) is the largest ν among points whose filler rate
  is within thr of the filler-alone launch run just before it.
- (b) in-loop: a point's ν is k extra instructions per MMA times the filler's MMAs per SMSP-cycle. Per rep, ν(thr) is the
  largest such ν whose filler rate is within thr of its adjacent base.
- (c) ncu: 1 − smsp__issue_active, and the per-pipe idle fractions, over the NCU_REPS profiles of the filler alone.

Error bars: mean ± sample standard deviation over reps (and min/max).
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

THR = (0.001, 0.005, 0.01)
PEAK = 0.25


def ms(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"mean": st.fmean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs), "n": len(xs)}


def load(o: Path) -> list[dict]:
    rows = []
    for line in (o / "sweep.jsonl").read_text().splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def paired(rows: list[dict], phase: str, base_phase: str) -> list[tuple[dict, dict]]:
    """Each `phase` row with the `base_phase` row launched just before it."""
    out, last = [], None
    for r in rows:
        if r["phase"] == base_phase:
            last = r
        elif r["phase"] == phase and last is not None and last["fw"] == r["fw"] and last["rep"] == r["rep"]:
            out.append((r, last))
    return out


OPS = ("none", "iadd3", "lop3", "imad", "prmt", "mix", "imma")


def probe_overhead(sass: dict) -> dict[str, float]:
    """Instructions per probe body over its PB ops, keyed by kernel name: the smallest loop holding a NANOSLEEP (a
    backward branch to the kernel's exit path spans the filler's loop and the probe's)."""
    out = {}
    for key, v in sass.items():
        f = key.split(",")
        loops = [lp for lp in v["loops"] if "NANOSLEEP" in lp["ops"]]
        if len(f) == 5 and f[0] != "0" and loops:
            out[f"p-{OPS[int(f[0])]}-b{f[4]}"] = min(loops, key=lambda lp: lp["n"])["n"] / int(f[4])
    return out


def frontier(points: list[tuple[int, float, float]], thr: float) -> list[float]:
    """points: (rep, nu, rel). Per rep, the largest nu with rel >= 1 - thr (0 if none)."""
    reps = sorted({p[0] for p in points})
    return [max([nu for r, nu, rel in points if r == rep and rel >= 1 - thr] + [0.0]) for rep in reps]


def method_a(rows: list[dict], over: dict[str, float]) -> dict:
    """rel: the filler's rate over the same kernel's with no probe warps (the co-resident effect alone); rel_plain: over
    the plain filler (base-u1) launched at the start of the kernel's group (what the probe kernel costs in all). For the
    IMMA probe, nu is instead the covert MMAs' share of the tensor peak."""
    pts: dict[tuple, list] = defaultdict(list)
    pts_plain: dict[tuple, list] = defaultdict(list)
    detail: dict[tuple, list] = defaultdict(list)
    base_noise = defaultdict(list)
    plain, base = None, None
    for r in rows:
        if r["phase"] == "probe-plain":
            plain = r
        elif r["phase"] == "probe-base":
            base = r
        elif r["phase"] == "probe" and base and plain and base["kernel"] == r["kernel"] and base["rep"] == r["rep"]:
            rel = r["mma_per_sm_clk_med"] / base["mma_per_sm_clk_med"]
            rel_plain = r["mma_per_sm_clk_med"] / plain["mma_per_sm_clk_med"]
            if r["pop"] == "imma":
                nu = r["probe_inst_per_sm_clk_med"] / PEAK
            else:
                nu = r["probe_inst_per_sm_clk_med"] / 4 * over.get(r["kernel"], 1.0)
            fam = "covert-imma" if r["pop"] == "imma" else "any-int"
            for key in ((r["kernel"],), (fam,)):
                pts[key].append((r["rep"], nu, rel))
                pts_plain[key].append((r["rep"], nu, rel_plain))
            detail[(r["kernel"], r["pw"], r["sleep_ns"])].append((nu, rel, rel_plain, r["mma_per_sm_clk_med"]))
    bases = [r for r in rows if r["phase"] in ("probe-base", "probe-plain")]
    for k in {r["kernel"] for r in bases}:
        for rep in {r["rep"] for r in bases}:
            v = [r["mma_per_sm_clk_med"] for r in bases if r["kernel"] == k and r["rep"] == rep]
            if len(v) > 1:
                m = st.fmean(v)
                base_noise[k].extend(x / m - 1 for x in v)
    plain_vs = defaultdict(list)
    for r in rows:
        if r["phase"] == "probe-base":
            p = [x for x in rows if x["phase"] == "probe-plain" and x["rep"] == r["rep"]]
            if p:
                plain_vs[r["kernel"]].append(r["mma_per_sm_clk_med"] / st.fmean(x["mma_per_sm_clk_med"] for x in p))
    out = {"frontier": {}, "frontier_vs_plain": {}, "points": {},
           "base_noise_sd": {k: st.pstdev(v) for k, v in base_noise.items()},
           "probe_kernel_alone_vs_plain": {k: ms(v) for k, v in plain_vs.items()}}
    for key, p in sorted(pts.items(), key=str):
        out["frontier"]["/".join(map(str, key))] = {f"{t:g}": ms(frontier(p, t)) for t in THR}
    for key, p in sorted(pts_plain.items(), key=str):
        out["frontier_vs_plain"]["/".join(map(str, key))] = {f"{t:g}": ms(frontier(p, t)) for t in THR}
    for key, v in sorted(detail.items(), key=str):
        out["points"][f"{key[0]}/pw{key[1]}/s{key[2]}"] = {
            "nu": ms([x[0] for x in v]), "rel": ms([x[1] for x in v]), "rel_plain": ms([x[2] for x in v]),
            "filler_share": ms([x[3] / PEAK for x in v])}
    return out


def method_b(rows: list[dict]) -> dict:
    """rel: the filler's rate with k extras per MMA over the full-rate filler (base-u1 at 16 filler warps, the occupancy
    maximum) of the same rep; rel_own: over the adjacent base of the same unroll and occupancy."""
    full = {r["rep"]: r["mma_per_sm_clk_med"] for r in rows if r["phase"] == "occupancy" and r["kernel"] == "base-u1"
            and r["fw"] == 16}
    pts: dict[tuple, list] = defaultdict(list)
    detail: dict[tuple, list] = defaultdict(list)
    for r, b in paired(rows, "inloop", "inloop-base"):
        if b["unr"] != r["unr"] or r["rep"] not in full:
            continue
        rel = r["mma_per_sm_clk_med"] / full[r["rep"]]
        rel_own = r["mma_per_sm_clk_med"] / b["mma_per_sm_clk_med"]
        nu = r["k"] * r["mma_per_sm_clk_med"] / 4
        cost = (1 / rel_own - 1) * (4 / b["mma_per_sm_clk_med"]) / r["k"]  # SMSP-cycles lost per extra instruction
        pts[(r["xop"], r["fw"])].append((r["rep"], nu, rel))
        pts[("any", r["fw"])].append((r["rep"], nu, rel))
        detail[(r["xop"], r["unr"], r["x"], r["fw"])].append((r["k"], nu, rel, rel_own, cost))
    out = {"frontier": {}, "points": {}}
    for key, p in sorted(pts.items(), key=str):
        out["frontier"][f"{key[0]}/fw{key[1]}"] = {f"{t:g}": ms(frontier(p, t)) for t in THR}
    for key, v in sorted(detail.items(), key=str):
        out["points"][f"{key[0]}/u{key[1]}/x{key[2]}/fw{key[3]}"] = {
            "k": v[0][0], "nu": ms([x[1] for x in v]), "rel": ms([x[2] for x in v]), "rel_own": ms([x[3] for x in v]),
            "cycles_lost_per_inst": ms([x[4] for x in v])}
    return out


def occupancy(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        if r["phase"] == "occupancy":
            by[(r["kernel"], r["fw"])].append(r["mma_per_sm_clk_med"])
    return {f"{k}/fw{fw}": {"rate": ms(v), "rel_peak": st.fmean(v) / PEAK} for (k, fw), v in sorted(by.items())}


def ncu_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    out = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        name, val = row.get("Metric Name"), row.get("Metric Value")
        if name and val:
            try:
                out[name] = float(val.replace(",", ""))
            except ValueError:
                pass
    return out


def method_c(o: Path) -> dict:
    by = defaultdict(list)
    for p in sorted((o / "ncu").glob("m-*.csv")):
        cfg = re.sub(r"-r\d+\.csv$", "", p.name)[2:]
        m = ncu_metrics(p)
        if m:
            by[cfg].append(m)
    out = {}
    for cfg, ms_ in by.items():
        names = sorted(set().union(*ms_))
        d = {n: ms([m.get(n) for m in ms_]) for n in names}
        ia = [m.get("smsp__issue_active.avg.pct_of_peak_sustained_active") for m in ms_]
        d["nu_issue"] = ms([1 - x / 100 for x in ia if x is not None])
        ipc = [m["smsp__inst_issued.sum"] / m["smsp__cycles_active.sum"] for m in ms_
               if m.get("smsp__cycles_active.sum")]
        d["nu_ipc"] = ms([1 - x for x in ipc])
        for pipe in ("alu", "fma", "fmaheavy", "fmalite", "tensor"):
            v = [m.get(f"sm__pipe_{pipe}_cycles_active.avg.pct_of_peak_sustained_active") for m in ms_]
            d[f"idle_{pipe}"] = ms([1 - x / 100 for x in v if x is not None])
        out[cfg] = d
    return out


def clocks(o: Path, rows: list[dict]) -> dict:
    out = {"launch_mhz": ms([r["mhz"] for r in rows])}
    smi = o / "smi.csv"
    if smi.exists():
        clk, pw = [], []
        for ln in smi.read_text().splitlines():
            f = [x.strip() for x in ln.split(",")]
            if len(f) >= 4:
                try:
                    clk.append(float(f[1].split()[0]))
                    pw.append(float(f[3].split()[0]))
                except (ValueError, IndexError):
                    pass
        out["smi_sm_mhz"] = ms(clk)
        out["smi_power_w"] = ms(pw)
    return out


def main() -> int:
    o = Path(sys.argv[1])
    rows = load(o)
    sass = json.loads((o / "sass-loops.json").read_text()) if (o / "sass-loops.json").exists() else {}
    over = probe_overhead(sass)
    summary = {
        "rows": len(rows),
        "distinct_sms": sorted({r["distinct_sms"] for r in rows}),
        "occupancy": occupancy(rows),
        "probe_overhead": over,
        "a_probe": method_a(rows, over),
        "b_inloop": method_b(rows),
        "c_ncu": method_c(o),
        "clocks": clocks(o, rows),
        "sass_base_loops": {k: v["loops"] for k, v in sass.items() if k in ("0,0,0,1,1", "0,0,0,4,1", "1,0,0,1,256")},
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
