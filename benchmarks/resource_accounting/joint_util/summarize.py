"""The table, from one run's outputs (a directory holding `run.sh`'s OUT): per resource, its utilization under Pearl-C's
call loop alone, the audit's passes alone and both at once, and the joint run's idle fraction; each protocol's
slowdown beside the other.

Utilization is ncu's busy cycles per call or pass (`ncu_summary.py`) times the run's measured rate, over the unit's
clock (the SMs' from NVML in that run; L2's and DRAM's from ncu); in the joint run, the two protocols' sum. Where nsys's
GPU-metric sampler measures the same resource (`nsys_summary.py`: SMs active, tensor active, warps in flight, DRAM
bandwidth), its device-wide reading is beside it, and it is the joint run's measured value.

    python summarize.py OUT_DIR [--m 32,128] [--variant _h9] [--json]
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
    ("ALU heavy", "sm__pipe_aluheavy_cycles_active", "sm"),
    ("SM sub-unit max (sm__throughput)", "sm__throughput", "sm"),
    ("SM issue (all inst)", "sm__inst_executed", "sm"),
    ("SM active cycles", "sm__cycles_active", "sm"),
    ("warp occupancy", "sm__warps_active", "sm"),
    ("L1/LSU wavefronts", "l1tex__data_pipe_lsu_wavefronts", "sm"),
    ("L2 sectors", "lts__t_sectors", "lts"),
    ("L2 sub-unit max (lts__throughput)", "lts__throughput", "lts"),
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
    "warp occupancy": "Compute Warps in Flight [Throughput",
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


def table(d: Path, m: int, v: str) -> dict:
    pr, ar, jr = load(d / f"pouw{m}.json"), load(d / f"audit{v}.json"), load(d / f"joint{m}{v}.json")
    pn, an = load(d / f"ncu_pouw{m}.sum.json"), load(d / f"ncu_audit{v}.sum.json")
    pg, ag, jg = (load(d / f"nsys_{x}.gpu.json") for x in (f"pouw{m}", f"audit{v}", f"joint{m}{v}"))
    ns = nsys_rows({"pouw_alone": pg, "audit_alone": ag, "joint": jg})
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
        if name in ns:
            row["nsys"] = ns[name]
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
    power = {k: (j or {}).get("nvml", {}).get("power_w_median") for k, j in (("pouw", pr), ("audit", ar), ("joint", jr))}
    return {"m": m, "audit_variant": v or "as shipped", "rows": rows, "rates": rates, "sm_mhz": clocks, "power_w": power}


def nsys_rows(gs: dict) -> dict:
    """Each nsys-measured row's device-wide mean in each run of `gs` (label: gpu.json)."""
    out = {row: {k: nsys_metric(g, p) for k, g in gs.items()} for row, p in NSYS.items()}
    out["DRAM bytes"] = {k: sum(x for x in (nsys_metric(g, p) for p in NSYS_DRAM) if x is not None) if g else None
                         for k, g in gs.items()}
    return out


def stretch(d: Path, m: int, v: str) -> dict | None:
    """Pearl-C's kernels under nsysk (each graph node traced), alone and beside the audit (and with Pearl-C's stream at
    the greatest priority, _q): each kernel's mean duration, and a call's summed kernel time against its wall time."""
    runs = {"alone": f"pouw{m}", "joint": f"joint{m}{v}", "joint_q": f"joint{m}{v}_q"}
    got = {k: (load(d / f"nsysk_{x}.gpu.json"), load(d / f"nsysk_{x}.json")) for k, x in runs.items()}
    got = {k: (g, j) for k, (g, j) in got.items() if g and j and "kernels" in g}
    if "alone" not in got:
        return None
    per_call = {}
    for k, (g, j) in got.items():
        calls = j["pouw"]["calls"] * g["kernels"]["window_ns"] / (j["pouw"]["gpu_ms"] * 1e6)
        p = g["kernels"]["by_protocol"].get("pouw", {})
        per_call[k] = {"call_ms": j["pouw"]["call_ms"], "kernel_ms": p.get("sum_ns", 0) / calls / 1e6,
                       "kernels_per_call": p.get("kernels", 0) / calls}
    kernels = {}
    for name, a in got["alone"][0]["kernels"]["top"].items():
        if a["protocol"] != "pouw":
            continue
        row = {"alone_us": a["sum_ns"] / a["count"] / 1e3}
        for k in ("joint", "joint_q"):
            b = got.get(k, (None,))[0]
            b = b and b["kernels"]["top"].get(name)
            if b:
                row[f"{k}_us"] = b["sum_ns"] / b["count"] / 1e3
        kernels[name] = row
    return {"m": m, "per_call": per_call, "kernels": kernels}


def gemm_table(d: Path, v: str, dtype: str) -> dict | None:
    s = "" if dtype == "fp8" else f"_{dtype}"
    g, j, ar = load(d / f"gemm{s}.json"), load(d / f"gemmjoint{v}{s}.json"), load(d / f"audit{v}.json")
    if not g:
        return None
    out = {"dtype": dtype, "shape": g["gemm"]["shape"], "tflops_alone": g["gemm"]["tflops"],
           "power_w": {"alone": g["nvml"].get("power_w_median"), "joint": (j or {}).get("nvml", {}).get("power_w_median")},
           "sm_mhz": {"alone": g["nvml"].get("sm_mhz_median"), "joint": (j or {}).get("nvml", {}).get("sm_mhz_median")}}
    if j:
        out.update({"tflops_joint": j["gemm"]["tflops"], "passes_per_s_joint": j["audit"]["passes_per_s"],
                    "gemm_slowdown": g["gemm"]["tflops"] / j["gemm"]["tflops"]})
        if ar:
            out["audit_slowdown"] = ar["audit"]["passes_per_s"] / j["audit"]["passes_per_s"]
    out["nsys"] = nsys_rows({"alone": load(d / f"nsys_gemm{s}.gpu.json"), "joint": load(d / f"nsys_gemmjoint{v}{s}.gpu.json")})
    return out


def fmt(x) -> str:
    return "-" if x is None else f"{100 * x:5.1f}%"


def show_stretch(t: dict) -> str:
    out = [f"Pearl-C's kernels at m = {t['m']} (nsysk): per call " + json.dumps(
        {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in t["per_call"].items()}),
           "| kernel | alone us | joint us | joint, Pearl-C first us |", "|---|---|---|---|"]
    for name, r in sorted(t["kernels"].items(), key=lambda kv: -kv[1]["alone_us"]):
        cell = lambda k: "-" if k not in r else f"{r[k]:.1f} ({r[k] / r['alone_us']:.2f}x)"  # noqa: E731
        out.append(f"| {name} | {r['alone_us']:.1f} | {cell('joint_us')} | {cell('joint_q_us')} |")
    return "\n".join(out)


def show_gemm(t: dict) -> str:
    head = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k != "nsys"}
    out = [f"GEMM {json.dumps(head)}", "| resource (nsys) | GEMM alone | GEMM + audit |", "|---|---|---|"]
    out += [f"| {row} | {fmt(r['alone'])} | {fmt(r['joint'])} |" for row, r in t["nsys"].items()]
    return "\n".join(out)


def show(t: dict) -> str:
    out = [f"m = {t['m']} (n = k = 8192), audit {t['audit_variant']}; SM MHz {t['sm_mhz']}; power W {t['power_w']}",
           f"rates {json.dumps({k: round(v, 3) for k, v in t['rates'].items()})}",
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
    ap.add_argument("--variant", default="_h9", help="the audit's phase suffix (run.sh's _h9, _p0; '' as shipped)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ms = [int(m) for m in a.m.split(",")]
    ts = [table(a.out, m, a.variant) for m in ms]
    ks = [k for k in (stretch(a.out, m, a.variant) for m in ms) if k]
    gs = [g for g in (gemm_table(a.out, a.variant, dt) for dt in ("fp8", "bf16")) if g]
    if a.json:
        print(json.dumps({"tables": ts, "stretch": ks, "gemm": gs}, indent=1))
    else:
        print("\n\n".join([show(t) for t in ts] + [show_stretch(k) for k in ks] + [show_gemm(g) for g in gs]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
