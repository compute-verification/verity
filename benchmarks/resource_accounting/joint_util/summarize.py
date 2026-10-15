"""The table, from one run's outputs (a directory holding `run.sh`'s OUT): per resource, its utilization under Pearl-C's
call loop alone, the audit's passes alone and both at once, and the joint run's idle fraction; each protocol's
slowdown beside the other.

Utilization is ncu's busy cycles per call or pass (`ncu_summary.py`) times the run's measured rate, over the unit's
clock (the SMs' from NVML in that run; L2's and DRAM's from ncu); in the joint run, the two protocols' sum. Where nsys's
GPU-metric sampler measures the same resource (`nsys_summary.py`: SMs active, tensor active, warps in flight, DRAM
bandwidth), its device-wide reading is beside it, and it is the joint run's measured value.

    python summarize.py OUT_DIR [--m 32,128] [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: (row, ncu busy base, clock domain)
RESOURCES = [
    ("tensor pipe (all)", "sm__pipe_tensor_cycles_active", "sm"),
    ("tensor pipe, IMMA/QMMA subpipe", "sm__pipe_tensor_subpipe_imma_cycles_active", "sm"),
    ("tensor pipe, HMMA subpipe", "sm__pipe_tensor_subpipe_hmma_cycles_active", "sm"),
    ("FMA pipe", "sm__pipe_fma_cycles_active", "sm"),
    ("FMA heavy", "sm__pipe_fmaheavy_cycles_active", "sm"),
    ("FMA lite", "sm__pipe_fmalite_cycles_active", "sm"),
    ("ALU pipe", "sm__pipe_alu_cycles_active", "sm"),
    ("LSU issue", "sm__inst_executed_pipe_lsu", "sm"),
    ("XU issue", "sm__inst_executed_pipe_xu", "sm"),
    ("uniform issue", "sm__inst_executed_pipe_uniform", "sm"),
    ("TMA pipe", "sm__pipe_tma_cycles_active", "sm"),
    ("SM issue (all inst)", "sm__inst_executed", "sm"),
    ("SM active cycles", "sm__cycles_active", "sm"),
    ("warp occupancy", "sm__warps_active", "sm"),
    ("L1/LSU wavefronts", "l1tex__data_pipe_lsu_wavefronts", "sm"),
    ("L2 sectors", "lts__t_sectors", "lts"),
    ("L2 reductions (sectors)", "lts__t_sectors_op_red", "lts"),
    ("L2 atomics (sectors)", "lts__t_sectors_op_atom", "lts"),
    ("L2 atomic unit busy", "lts__d_atomic_input_cycles_active", "lts"),
    ("DRAM bytes", "dram__bytes", "dram"),
]
#: rows nsys's gb20x set measures device-wide: (row, metric-name prefix)
NSYS = {
    "SM active cycles": "SMs Active",
    "SM issue (all inst)": "SM Issue",
    "tensor pipe (all)": "Tensor Active",
    "warp occupancy": "Compute Warps In Flight",
}
NSYS_DRAM = ("DRAM Read Bandwidth", "DRAM Write Bandwidth")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def nsys_metric(g: dict | None, prefix: str) -> float | None:
    if not g or "metrics" not in g:
        return None
    for name, v in g["metrics"].items():
        if name.startswith(prefix):
            return v["mean"] / 100.0
    return None


def clock(rate: dict, ncu: dict, domain: str) -> float:
    if domain == "sm":
        return rate["nvml"]["sm_mhz_median"] * 1e6
    return ncu["clocks_median"][f"{domain}__cycles_elapsed.avg.per_second"]


def util(busy: dict, ncu: dict, rate_json: dict, per_s: float, base: str, domain: str) -> float | None:
    b = busy.get(base)
    if b is None:
        return None
    return b * per_s / clock(rate_json, ncu, domain)


def table(d: Path, m: int) -> dict:
    pr, ar, jr = load(d / f"pouw{m}.json"), load(d / "audit.json"), load(d / f"joint{m}.json")
    pn, an = load(d / f"ncu_pouw{m}.sum.json"), load(d / "ncu_audit.sum.json")
    pg, ag, jg = (load(d / f"nsys_{x}.gpu.json") for x in (f"pouw{m}", "audit", f"joint{m}"))
    rows = []
    for name, base, dom in RESOURCES:
        row = {"resource": name}
        if pn and pr:
            row["pouw_alone"] = util(pn["busy_cycles_per_unit"], pn, pr, pr["pouw"]["calls_per_s"], base, dom)
        if an and ar:
            row["audit_alone"] = util(an["busy_cycles_per_unit"], an, ar, ar["audit"]["passes_per_s"], base, dom)
        if pn and an and jr:
            a = util(pn["busy_cycles_per_unit"], pn, jr, jr["pouw"]["calls_per_s"], base, dom)
            b = util(an["busy_cycles_per_unit"], an, jr, jr["audit"]["passes_per_s"], base, dom)
            row["joint_pouw_part"], row["joint_audit_part"] = a, b
            row["joint"] = None if a is None or b is None else a + b
        if name in NSYS:
            row["nsys"] = {k: nsys_metric(g, NSYS[name]) for k, g in (("pouw_alone", pg), ("audit_alone", ag), ("joint", jg))}
        if base == "dram__bytes":
            row["nsys"] = {k: sum(x for x in (nsys_metric(g, p) for p in NSYS_DRAM) if x is not None) if g else None
                           for k, g in (("pouw_alone", pg), ("audit_alone", ag), ("joint", jg))}
        meas = (row.get("nsys") or {}).get("joint")
        best = meas if meas is not None else row.get("joint")
        row["idle_joint"] = None if best is None else 1.0 - best
        rows.append(row)
    rates = {}
    if pr:
        rates["pouw_alone_calls_per_s"] = pr["pouw"]["calls_per_s"]
    if ar:
        rates["audit_alone_passes_per_s"] = ar["audit"]["passes_per_s"]
    if jr:
        rates["joint_calls_per_s"], rates["joint_passes_per_s"] = jr["pouw"]["calls_per_s"], jr["audit"]["passes_per_s"]
        if pr:
            rates["pouw_slowdown"] = pr["pouw"]["calls_per_s"] / jr["pouw"]["calls_per_s"]
        if ar:
            rates["audit_slowdown"] = ar["audit"]["passes_per_s"] / jr["audit"]["passes_per_s"]
    pp, pa = load(d / f"procs{m}_pouw.json"), load(d / f"procs{m}_audit.json")
    if pp and pa:
        rates["procs_calls_per_s"], rates["procs_passes_per_s"] = pp["pouw"]["calls_per_s"], pa["audit"]["passes_per_s"]
        if pr:
            rates["procs_pouw_slowdown"] = pr["pouw"]["calls_per_s"] / pp["pouw"]["calls_per_s"]
        if ar:
            rates["procs_audit_slowdown"] = ar["audit"]["passes_per_s"] / pa["audit"]["passes_per_s"]
    clocks = {k: (j or {}).get("nvml", {}).get("sm_mhz_median") for k, j in (("pouw", pr), ("audit", ar), ("joint", jr))}
    return {"m": m, "rows": rows, "rates": rates, "sm_mhz": clocks}


def fmt(x) -> str:
    return "-" if x is None else f"{100 * x:5.1f}%"


def show(t: dict) -> str:
    out = [f"m = {t['m']} (n = k = 8192); SM MHz {t['sm_mhz']}", f"rates {json.dumps({k: round(v, 3) for k, v in t['rates'].items()})}",
           "| resource | PoUW alone | audit alone | joint (ncu sum) | joint (nsys) | idle joint |", "|---|---|---|---|---|---|"]
    for r in t["rows"]:
        ns = r.get("nsys") or {}
        alone = lambda k: fmt(r.get(k)) + (f" (nsys {fmt(ns[k])})" if ns.get(k) is not None else "")  # noqa: E731
        out.append(f"| {r['resource']} | {alone('pouw_alone')} | {alone('audit_alone')} | {fmt(r.get('joint'))} "
                   f"| {fmt(ns.get('joint'))} | {fmt(r.get('idle_joint'))} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--m", default="32,128")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ts = [table(a.out, int(m)) for m in a.m.split(",")]
    print(json.dumps(ts, indent=1) if a.json else "\n\n".join(show(t) for t in ts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
