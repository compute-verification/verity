#!/usr/bin/env python3
"""The η study's tables from fetched runs: job_pearlc.sh's run dir (eta.py's rows, run.py's split, ncu by kernel and by graph)
and job_base.sh's (lt_probe.py's rows, ncu on cuBLASLt, the harness's bench.json); or, given a lease_pearlc2.sh run alone,
its SHA-512 probe, par<c> and power tables.  Markdown on stdout; no GPU.

  python3 analyze.py ~/.research/runs/<pearlc run> ~/.research/runs/<base run>
  python3 analyze.py ~/.research/runs/<second-lease run>
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SMS, MACS_PER_SM_CLOCK, CAP_MHZ = 188, 1024, 2430
T_ACT = "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"
T_SUM = "sm__pipe_tensor_subpipe_hmma_cycles_active.sum"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.startswith("{")] if p.exists() else []


#: ncu's per-column units, to µs, MHz and MB
SCALE = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6,
         "hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3, "byte": 1e-6, "kbyte": 1e-3, "mbyte": 1.0, "gbyte": 1e3}


def ncu(p: Path) -> list[dict]:
    """ncu's raw CSV, one dict a result, durations in µs, frequencies in MHz, bytes in MB."""
    if not p.exists():
        return []
    rows = list(csv.reader(p.open()))
    head, units = rows[0], rows[1]
    out = []
    for r in rows[2:]:
        d = {}
        for k, unit, v in zip(head, units, r):
            try:
                d[k] = float(v.replace(",", "")) * SCALE.get(unit.lower(), 1.0)
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def tree_launches(count: int) -> int:
    """run.py's `levels`: s512_nodes a level down to 128 nodes, then one s512_tree_small."""
    width, n = 1 << max(count - 1, 0).bit_length(), 0
    while width > 128:
        n, width = n + 1, width // 2
    return n + (width > 1)


def launches(step: str, p: dict) -> int:
    m_real, n = p["shape"][0], p["shape"][1]
    if step in ("hash_rows_a", "tree_leaves_t"):
        return 2
    if step == "tree_a":
        return tree_launches(m_real)
    if step == "tree_t":
        return tree_launches(-(-m_real // 64) * (n // 64))
    return 1


def attach(results: list[dict], profiles: list[dict], graphs: bool = False) -> list[tuple[dict, str, list[dict]]]:
    """ncu's results in launch order, grouped by eta.py's profile lines' (shape, step) order: a step's launches by
    `launches`, a graph's one result."""
    out, i = [], 0
    for p in profiles:
        for name in p["order"]:
            k = 1 if graphs else launches(name, p)
            out.append((p, name, results[i:i + k]))
            i += k
    return out


def merged(ds: list[dict]) -> dict:
    """A step's launches as one: durations, cycles, ops and bytes summed; percentages weighted by duration."""
    if len(ds) == 1:
        return ds[0]
    t = sum(d.get("gpu__time_duration.sum", 0.0) for d in ds) or 1.0
    out = {"Kernel Name": "+".join(sorted({d.get("Kernel Name", "") for d in ds})), "Grid Size": ds[0].get("Grid Size", ""),
           "Block Size": ds[0].get("Block Size", "")}
    for key in ds[0]:
        vals = [d.get(key) for d in ds]
        if not all(isinstance(v, float) for v in vals):
            continue
        if key.endswith(".sum") or key.startswith(("dram__bytes",)):
            out[key] = sum(vals)
        else:
            out[key] = sum(v * d.get("gpu__time_duration.sum", 0.0) for v, d in zip(vals, ds)) / t
    return out


def f(x, nd=3):
    return "" if x is None or x == "" else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def second(pc: Path) -> None:
    """lease_pearlc2.sh's tables: the SHA-512 probe, par<c>, the power holds."""
    sha = [r for r in jsonl(pc / "eta_sha.jsonl") if r["kind"] == "sha"]
    if sha:
        print("## the SHA-512 probe (CUDA graphs of 10 launches; compressions/s)\n")
        print("| kernel | m or rows | threads a CTA | ms | compressions | compressions/s | digests = served |")
        print("|---|---|---|---|---|---|---|")
        for r in sha:
            print(f"| {r['kernel']} | {r['shape'][0] if 'shape' in r else r['rows']} | {r['block']} | {f(r['ms'], 4)} | "
                  f"{r['compressions']} | {r['compressions_per_s']:.3g} | {r.get('same_digests', '')} |")
    par = [r for r in jsonl(pc / "eta_par.jsonl") if r["kind"] == "eta"]
    if par:
        print("\n## c independent calls at once (W = 8)\n")
        print("| m | schedule | ms/call | MHz | η | η_cap |")
        print("|---|---|---|---|---|---|")
        for r in par:
            print(f"| {r['shape'][0]} | {r['schedule']} | {f(r.get('per_call_ms'), 4)} | {f(r.get('sm_mhz'), 0)} | "
                  f"{f(r.get('eta'), 4)} | {f(r.get('eta_cap'), 4)} |")
    pw = [r for r in jsonl(pc / "eta_power.jsonl") if r["kind"] == "power"]
    if pw:
        print("\n## each schedule held 2 s under NVML (W = 4; the middle half's medians)\n")
        print("| m | schedule | ms/call | SM MHz | power W | throttle | mJ a call | η | η_cap |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in pw:
            print(f"| {r['shape'][0]} | {r['schedule']} | {f(r['per_call_ms'], 4)} | {f(float(r['sm_mhz_median'] or 0), 0)} | "
                  f"{f(r['power_w_median'], 1)} | {','.join(r['throttle_seen'])} | {f(r.get('energy_mj_per_call'), 1)} | "
                  f"{f(r.get('eta'), 4)} | {f(r.get('eta_cap'), 4)} |")


def main(argv: list[str]) -> int:
    pc = Path(argv[1]) / "out"
    if (pc / "eta_sha.jsonl").exists() or (pc / "eta_power.jsonl").exists():
        second(pc)
        return 0
    base = Path(argv[2]) / "out" if len(argv) > 2 else None
    eta = [r for r in jsonl(pc / "eta_time.jsonl") if r["kind"] == "eta"]
    print("## eta.py: per call, graphs of 2W calls\n")
    print("| m | tile | W | schedule | ms/call | MHz | η | η_cap |")
    print("|---|---|---|---|---|---|---|---|")
    for r in eta:
        tile = f"{r['tile'][0]}" + ("(forced)" if r.get("bm_forced") else "")
        print(f"| {r['shape'][0]} | {tile} | {r['w']} | {r['schedule']} | {f(r.get('per_call_ms'), 4)} | {f(r.get('sm_mhz'), 0)} | "
              f"{f(r.get('eta'), 4)} | {f(r.get('eta_cap'), 4)} |")
    kprof = [r for r in jsonl(pc / "eta_prof_kernels.log") if r["kind"] == "profile"]
    gprof = [r for r in jsonl(pc / "eta_prof_graphs.log") if r["kind"] == "profile"]
    kn = attach(ncu(pc / "ncu_kernels.csv"), kprof)
    print("\n## ncu by kernel (one serial call, W = 1, --clock-control none --cache-control none)\n")
    print("| m | kernel | µs | MHz | tensor active % | FP8 ops / m n k | BF16 ops / m n k | DRAM % | SM thr % | issue % | ALU % "
          "| FMA % | warps % | grid | block |")
    print("|" + "---|" * 15)
    tensor_sum: dict[int, float] = {}
    for p, name, ds in kn:
        d = merged(ds) if ds else {}
        m = p["shape"][0]
        mnk = p["macs_computed"]
        if name not in ("gemm_ct", "gemm_fast", "gemm_plain", "tree_leaves_t", "tree_t"):
            tensor_sum[m] = tensor_sum.get(m, 0.0) + d.get(T_ACT, 0.0) / 100 * d.get("gpu__time_duration.sum", 0.0)
        print(f"| {m} | {name} ({d.get('Kernel Name', '')}) | {f(d.get('gpu__time_duration.sum'), 1)} | "
              f"{f(d.get('sm__cycles_elapsed.avg.per_second'), 0)} | {f(d.get(T_ACT), 1)} | "
              f"{f(d.get('sm__ops_path_tensor_src_fp8_dst_fp32.sum', 0) / mnk, 3)} | "
              f"{f(d.get('sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum', 0) / mnk, 4)} | "
              f"{f(d.get('dram__throughput.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('sm__throughput.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('smsp__issue_active.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('sm__warps_active.avg.pct_of_peak_sustained_active'), 1)} | {d.get('Grid Size', '')} | {d.get('Block Size', '')} |")
    gn = attach(ncu(pc / "ncu_graphs.csv"), gprof, graphs=True)
    print("\n## ncu by graph (2W = 6 calls, W = 3, --graph-profiling graph)\n")
    print("| m | schedule | ms/call | MHz | tensor active % | FP8 ops / useful MACs | η from ops | DRAM % | issue % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p, name, ds in gn:
        d = ds[0] if ds else {}
        calls, m = p["calls_per_graph"], p["shape"][0]
        dur = d.get("gpu__time_duration.sum", 0) / 1e3 / calls
        mhz = d.get("sm__cycles_elapsed.avg.per_second", 0)
        ops = d.get("sm__ops_path_tensor_src_fp8_dst_fp32.sum", 0) / calls
        print(f"| {m} | {name} | {f(dur, 4)} | {f(mhz, 0)} | {f(d.get(T_ACT), 2)} | {f(ops / p['macs_useful'], 3)} | "
              f"{f(p['macs_useful'] / (dur * 1e-3) / (SMS * MACS_PER_SM_CLOCK * mhz * 1e6), 4) if dur and mhz else ''} | "
              f"{f(d.get('dram__throughput.avg.pct_of_peak_sustained_elapsed'), 1)} | "
              f"{f(d.get('smsp__issue_active.avg.pct_of_peak_sustained_elapsed'), 1)} |")
    print("\n## the served call's tensor-pipe utilization: its kernels' tensor-busy time (active % x duration) over the timed "
          "serial call (W = 3)\n")
    for m, s in tensor_sum.items():
        row = next((r for r in eta if r["shape"][0] == m and r["schedule"] == "serial" and r["w"] == 3 and not r.get("bm_forced")), None)
        if row:
            print(f"- m = {m}: tensor busy {s:.1f} µs over {1e3 * row['per_call_ms']:.1f} µs = {100 * s / (1e3 * row['per_call_ms']):.2f}%")
    for p, name, ds in kn:
        d = merged(ds) if ds else {}
        if name == "gemm_pearlc" and d.get(T_ACT):
            cyc = d[T_ACT] / 100 * SMS * d["sm__cycles_elapsed.avg.per_second"] * d["gpu__time_duration.sum"]
            print(f"- m = {p['shape'][0]} gemm_pearlc: {T_SUM} / (active x SMs x cycles) = {d.get(T_SUM, 0) / cyc:.2f} "
                  "(the sub-pipes an SM counts)")
    split = [r for r in jsonl(pc / "run_py.jsonl") if r["kind"] == "timing"]
    print("\n## run.py's split (ms, each kernel timed alone)\n")
    served = [nm for p, nm, _ in kn if p["shape"][0] == 8192]
    for r in split:
        print(f"- {r['shape']} served steps: " + ", ".join(f"{nm} {r['ms'][nm]:.4f}" for nm in served if nm in r["ms"]))
    for r in split:
        print(f"- {r['shape']} m_real {r['m_real']}: split {r['split_ms']}, call e2e {r['call_e2e_ms']}, no hash "
              f"{r['call_e2e_no_hash_ms']}, gemm_pearlc {r['ms'].get('gemm_pearlc')}, plain {r.get('plain_tflops')} / fast "
              f"{r.get('fast_tflops')} TFLOPS")
    for r in jsonl(pc / "eta_clock.jsonl"):
        if r["kind"] == "clock":
            print(f"- clock hold {r['kernel']} {r['shape']}: {r['launches']} launches, {r['tflops']:.1f} TFLOPS, SM {r['sm_mhz_median']} "
                  f"MHz, power median {r['power_w_median']} W max {r['power_w_max']} W, limit {r.get('enforced_power_limit_w')} W, "
                  f"max SM {r.get('max_sm_mhz')} MHz, apps {r.get('applications_sm_mhz')} MHz, throttle {r['throttle_seen']}, "
                  f"temp max {r['temp_c_max']} C")
    if base:
        print("\n## cuBLASLt via torch._scaled_mm\n")
        print("| m | W | ms/call | TFLOPS | MHz | η | η_cap |")
        print("|---|---|---|---|---|---|---|")
        for r in jsonl(base / "lt_time.jsonl"):
            if r["kind"] == "lt":
                print(f"| {r['shape'][0]} | {r['w']} | {f(r['per_call_ms'], 4)} | {r['tflops']} | {f(r['sm_mhz'], 0)} | {f(r['eta'], 4)} | "
                      f"{f(r['eta_cap'], 4)} |")
        lprof = [r for r in jsonl(base / "lt_prof.log") if r["kind"] == "profile"]
        print("\n| m | kernel | µs | MHz | tensor active % | FP8 ops / m n k | DRAM % | issue % | grid |")
        print("|---|---|---|---|---|---|---|---|---|")
        for p, d in zip(lprof, ncu(base / "ncu_lt.csv")):
            mnk = p["m"] * p["n"] * p["k"]
            print(f"| {p['m']} | {d.get('Kernel Name', '')[:60]} | {f(d.get('gpu__time_duration.sum'), 1)} | "
                  f"{f(d.get('sm__cycles_elapsed.avg.per_second'), 0)} | {f(d.get(T_ACT), 1)} | "
                  f"{f(d.get('sm__ops_path_tensor_src_fp8_dst_fp32.sum', 0) / mnk, 3)} | "
                  f"{f(d.get('dram__throughput.avg.pct_of_peak_sustained_elapsed'), 1)} | "
                  f"{f(d.get('smsp__issue_active.avg.pct_of_peak_sustained_elapsed'), 1)} | {d.get('Grid Size', '')} |")
        bj = base / "bench.json"
        if bj.exists():
            b = json.loads(bj.read_text())
            print("\n## the harness's autotuned FP8 baselines (bench.json)\n")
            for sh in b.get("shapes", []):
                mnk = sh["m"] * sh["n"] * sh["k"]
                best = sh["baselines"]["fp8-e4m3"]["graph"]
                print(f"- {sh['label']}: fastest {best['kernel']} ({best['source']}) {best['ms']:.4f} ms, {best['tflops']:.1f} TFLOPS")
                for name, v in sh["summary"].items():
                    if name.startswith("baseline/"):
                        print(f"  - {name.split('/')[-1]}: graph median {v['graph']['median']:.4f} ms "
                              f"({2 * mnk / v['graph']['median'] / 1e9:.1f} TFLOPS), items with a bad throttle "
                              f"{v.get('items_with_bad_throttle')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
