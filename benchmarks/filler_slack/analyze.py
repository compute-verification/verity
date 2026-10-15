"""ν three ways from a run's outputs: `python3 analyze.py OUT > summary.json`.

ν is free issue slots over total issue slots per scheduler per cycle (an SM has 4 schedulers, each issuing at most one
warp-instruction a cycle). Rates are warp-MMAs per SM-clock (peak 0.25: one m16n8k32 per SMSP every 16 cycles).

- (a) probe: a probe point's ν is its warp-instructions per SM-clock over 4, scaled by the probe loop's SASS count over its
  PB ops (the loop's own counter, flag load and branch). Per rep, ν(thr) is the largest ν among points whose filler rate
  is within thr of a reference (method_a: the same filler code alone, the plain filler, or the probes resident but
  idle), for probe warps in the filler's kernel and for a probe kernel on a second stream beside the plain filler.
- (b) in-loop: a point's ν is k extra instructions per MMA times the filler's MMAs per SMSP-cycle. Per rep, ν(thr) is the
  largest such ν whose filler rate is within thr of its adjacent base.
- (c) ncu: 1 − smsp__issue_active, and the per-pipe idle fractions, over the NCU_REPS profiles of the filler alone.

Error bars: mean ± sample standard deviation over reps (and min/max); a frontier also carries its grid bracket (the largest
passing point and the smallest failing point above it), which is wider than the rep-to-rep spread, and the bracket's
linear interpolation to the threshold.
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
    """Instructions per probe body over its PB ops, keyed by kernel name: the smallest loop holding a NANOSLEEP and at
    least PB instructions (a backward branch to the kernel's exit path spans the filler's loop and the probe's, and the
    probe-first kernel's start-flag wait is a NANOSLEEP loop of its own)."""
    out = {}
    for key, v in sass.items():
        if key.startswith("pk:"):
            pop, pb = key[3:].split(",")
            name = f"s-{OPS[int(pop)]}-b{pb}"
        elif len(f := key.split(",")) == 5 and f[0] != "0":
            pop, pb, name = f[0], f[4], f"p-{OPS[int(f[0])]}-b{f[4]}"
        else:
            continue
        loops = [lp["n"] for lp in v["loops"] if "NANOSLEEP" in lp["ops"] and lp["n"] >= int(pb)]
        if loops:
            out[name] = min(loops) / int(pb)
    return out


def frontier(points: list[tuple[int, float, float]], thr: float) -> list[float]:
    """points: (rep, nu, rel). Per rep, the largest nu with rel >= 1 - thr (0 if none)."""
    reps = sorted({p[0] for p in points})
    return [max([nu for r, nu, rel in points if r == rep and rel >= 1 - thr] + [0.0]) for rep in reps]


def bracket(points: list[tuple[int, float, float]], thr: float) -> dict:
    """The frontier's grid bracket: per rep, the largest passing nu (lo) and the smallest failing nu above it (hi; None
    if every larger point passes), and the nu at which the loss reaches thr on the line between them (interp; None
    without a failing point), each as mean ± sd over reps."""
    los, his, interps = [], [], []
    for rep in sorted({p[0] for p in points}):
        pts = [(nu, rel) for r, nu, rel in points if r == rep]
        lo = max([(nu, rel) for nu, rel in pts if rel >= 1 - thr] + [(0.0, 1.0)])
        hi = min([(nu, rel) for nu, rel in pts if rel < 1 - thr and nu > lo[0]], default=None)
        los.append(lo[0])
        his.append(hi[0] if hi else None)
        if hi:
            f = (lo[1] - (1 - thr)) / (lo[1] - hi[1])
            interps.append(lo[0] + (hi[0] - lo[0]) * min(max(f, 0.0), 1.0))
    return {"lo": ms(los), "hi": ms(his), "interp": ms(interps)}


FAMILY = {"probe": ("probe-base", "in-kernel"), "ss-probe": ("ss-base", "second-stream"),
          "ssf-probe": ("ss-base", "second-stream-first")}
IDLE_NS = 1000000


def method_a(rows: list[dict], over: dict[str, float]) -> dict:
    """Per probe point, three references for the filler's rate:
    rel        the filler alone launched just before it with the same filler code (in-kernel: the same kernel with no
               probe warps, since ptxas compiles the filler loop differently once a probe path is in the kernel;
               second stream, the probe kernel launched after the filler or resident before it: base-u1 at the
               second-stream shared memory size)
    rel_plain  the plain filler (base-u1 at 16 warps, one block an SM) of the same rep: what the probe costs in all
    rel_idle   the same kernel and probe warps sleeping 1 ms between bodies (probes resident but nearly idle)
    nu: probe warp-instructions per SM-clock over 4 schedulers, scaled by the loop's SASS count over its PB ops; for the
    IMMA probes instead the covert MMAs' share of the tensor peak."""
    plain = {r["rep"]: r["mma_per_sm_clk_med"] for r in rows if r["phase"] == "occupancy" and r["kernel"] == "base-u1"
             and r["fw"] == 16}
    idle = {(r["phase"], r["kernel"], r["pw"], r["rep"]): r["mma_per_sm_clk_med"] for r in rows
            if r["phase"] in FAMILY and r["sleep_ns"] == IDLE_NS}
    pts: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    detail: dict[tuple, list] = defaultdict(list)
    base = None
    for r in rows:
        if r["phase"] in ("probe-base", "ss-base"):
            base = r
            continue
        if r["phase"] not in FAMILY or base is None or base["phase"] != FAMILY[r["phase"]][0] or base["rep"] != r["rep"]:
            continue
        fam = FAMILY[r["phase"]][1]
        rate = r["mma_per_sm_clk_med"]
        rels = {"rel": rate / base["mma_per_sm_clk_med"], "rel_plain": rate / plain[r["rep"]],
                "rel_idle": rate / idle[(r["phase"], r["kernel"], r["pw"], r["rep"])]}
        imma = r["pop"] == "imma"
        nu = r["probe_inst_per_sm_clk_med"] / PEAK if imma else r["probe_inst_per_sm_clk_med"] / 4 * over.get(r["kernel"], 1.0)
        groups = (f"{fam}/{r['kernel']}", f"{fam}/{'covert-imma' if imma else 'any-int'}")
        if r["sleep_ns"] != IDLE_NS:
            for ref, rel in rels.items():
                for g in groups:
                    pts[ref][g].append((r["rep"], nu, rel))
        detail[(fam, r["kernel"], r["pw"], r["sleep_ns"])].append(
            (nu, rels["rel"], rels["rel_plain"], rels["rel_idle"], rate / PEAK, r.get("paired_sms"), r.get("probe_distinct_sms")))
    bases = defaultdict(list)
    for r in rows:
        if r["phase"] in ("probe-base", "ss-base") and r["rep"] in plain:
            bases[f"{r['phase']}/{r['kernel']}"].append(r["mma_per_sm_clk_med"] / plain[r["rep"]])
    out = {"frontier": {}, "points": {}, "base_vs_plain": {k: ms(v) for k, v in sorted(bases.items())}}
    for ref, groups in pts.items():
        out["frontier"][ref] = {g: {f"{t:g}": {"nu": ms(frontier(p, t)), **bracket(p, t)} for t in THR}
                                for g, p in sorted(groups.items())}
    for key, v in sorted(detail.items(), key=str):
        out["points"][f"{key[0]}/{key[1]}/pw{key[2]}/s{key[3]}"] = {
            "nu": ms([x[0] for x in v]), "rel": ms([x[1] for x in v]), "rel_plain": ms([x[2] for x in v]),
            "rel_idle": ms([x[3] for x in v]), "filler_share": ms([x[4] for x in v]),
            "paired_sms": sorted({x[5] for x in v if x[5] is not None}),
            "probe_distinct_sms": sorted({x[6] for x in v if x[6] is not None})}
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
        out["frontier"][f"{key[0]}/fw{key[1]}"] = {f"{t:g}": {"nu": ms(frontier(p, t)), **bracket(p, t)} for t in THR}
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
