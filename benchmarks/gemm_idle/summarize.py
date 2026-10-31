"""`python3 summarize.py OUT`: gemm_idle's passes (OUT/passes.jsonl, OUT/info.json) as tensor utilization.

Each clock label's bound is SMs x 1,024 MACs a clock (dense int8 and e4m3 mma.sync on sm_120; the zero-operand peak's
`macs_per_sm_clk` checks it) x the clock: `clk` at the label's nominal clock (node-<MHz>: the node's lock as it reads idle;
lock-<MHz>: the lock asked for), and `unif` the uniform-operand peak measured at that label (the power-capped bound where
the cap binds). A GEMM's utilization is its MACs over the pass's wall time against each; beside a co-runner, `co` is the
co-runner's MACs over the same window against `clk`, and `keep` the GEMM's rate beside it over its rate alone. Prints the
tables and writes OUT/summary.json. Stdlib only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

PER_SM_CLK = 1024
MODELS = {"1b": {(3072, 2048): "qkv", (2048, 2048): "o", (16384, 2048): "gate_up", (2048, 8192): "down"},
          "8b": {(6144, 4096): "qkv", (4096, 4096): "o", (28672, 4096): "gate_up", (4096, 14336): "down"}}


def linear(n: int, k: int) -> str:
    for m, shapes in MODELS.items():
        if (n, k) in shapes:
            return f"{m}.{shapes[(n, k)]}"
    return f"{n}x{k}"


def main(out: str) -> int:
    info = json.load(open(f"{out}/info.json"))
    sms = info["sms"]
    rows = [json.loads(ln) for ln in open(f"{out}/passes.jsonl") if ln.startswith("{")]
    peak: dict = defaultdict(dict)   # (clock, kind) -> class -> best peak row (nacc 8)
    co_alone: dict = {}              # (clock, kind, warps) -> row
    for r in rows:
        if r["pass"] != "peak":
            continue
        key = (r["clock"], r["kind"])
        if r["nacc"] == 4 and r["class"] == "uniform":
            co_alone[(r["clock"], r["kind"], r["warps"])] = r
        elif r["nacc"] == 8 and r["macs_per_s"] > peak[key].get(r["class"], {}).get("macs_per_s", 0):
            peak[key][r["class"]] = r
    nominal = {c: float(c.split("-")[1]) for c, _ in peak}
    bound = {k: sms * PER_SM_CLK * nominal[k[0]] * 1e6 for k in peak}
    summary: dict = {"sms": sms, "l2_mb": info["l2_bytes"] / 2**20, "per_sm_clk": PER_SM_CLK, "peaks": [], "co_alone": [],
                     "gemms": []}
    print(f"{info['name']}: {sms} SMs, L2 {info['l2_bytes'] / 2**20:.0f} MiB, cuBLASLt {info['cublaslt']}")
    print("\npeaks (mma.sync m16n8k32 from registers, best of 8 and 16 warps an SM, 8 accumulators)")
    print(f"{'clock':<12}{'kind':<6}{'class':<9}{'MAC/s':>11}{'clk64 MHz':>10}{'MAC/SM/clk':>11}{'of clk':>8}{'W':>6}"
          f"{'NVML MHz':>9}  reasons")
    for (c, k), byc in sorted(peak.items()):
        for cls in ("zero", "uniform"):
            r = byc.get(cls)
            if not r:
                continue
            nv = r.get("nvml") or {}
            f = r["macs_per_s"] / bound[(c, k)]
            print(f"{c:<12}{k:<6}{cls:<9}{r['macs_per_s']:>11.4e}{r['sm_mhz_clock64']:>10.1f}{r['macs_per_sm_clk']:>11.1f}"
                  f"{f:>8.4f}{nv.get('energy_w', 0):>6.0f}{nv.get('sm_mhz_mean', 0):>9.0f}  {nv.get('reasons')}")
            summary["peaks"].append({"clock": c, "kind": k, "class": cls, "macs_per_s": r["macs_per_s"],
                                     "of_clock_bound": f, "sm_mhz_clock64": r["sm_mhz_clock64"],
                                     "macs_per_sm_clk": r["macs_per_sm_clk"], "power_w": nv.get("energy_w"),
                                     "reasons": nv.get("reasons")})
    for (c, k, w), r in sorted(co_alone.items()):
        if (c, k) in bound:
            summary["co_alone"].append({"clock": c, "kind": k, "warps": w, "of_clock_bound": r["macs_per_s"] / bound[(c, k)]})
    if summary["co_alone"]:
        print("\nco-runner alone (uniform, 4 accumulators): " + ", ".join(
            f"{x['clock']} {x['kind']} {x['warps']}w {x['of_clock_bound']:.3f}" for x in summary["co_alone"]))

    alone = {}
    beside = defaultdict(dict)
    tune = {}
    for r in rows:
        key = (r.get("clock"), r.get("kind"), r.get("lib"), r.get("m"), r.get("n"), r.get("k"), r.get("wclass"))
        if r["pass"] == "alone":
            alone[key] = r
        elif r["pass"] == "beside":
            beside[key][r["co_warps"]] = r
        elif r["pass"] == "tune":
            tune[key] = r
    warps = sorted({w for b in beside.values() for w in b})
    hdr = "".join(f"{'co' + str(w):>7}{'keep' + str(w):>7}{'sum' + str(w):>7}" for w in warps)
    print("\nGEMMs: MACs over wall time; clk = against SMs x 1024 x the label's clock, unif = against the uniform peak there;"
          "\nco<w> = the co-runner's rate beside it against clk, keep<w> = the GEMM's rate beside it / alone, sum<w> = both"
          " against clk")
    print(f"{'clock':<11}{'kind':<5}{'lib':<4}{'w':<8}{'linear':<11}{'T':>5}{'MAC/s':>11}{'clk':>7}{'unif':>7}{hdr}"
          f"{'MHz':>6}{'W':>5}  algo, check")
    for key in sorted(alone, key=lambda x: (x[0], x[1], x[2], x[6], linear(x[4], x[5]), x[3])):
        c, k, lib, m, n, kk, wc = key
        r = alone[key]
        if (c, k) not in bound:
            continue
        b = bound[(c, k)]
        u = peak[(c, k)].get("uniform", {}).get("macs_per_s")
        nv = r.get("nvml") or {}
        row = {"clock": c, "kind": k, "lib": lib, "wclass": wc, "linear": linear(n, kk), "m": m, "n": n, "k": kk,
               "macs_per_s": r["macs_per_s"], "of_clock_bound": r["macs_per_s"] / b,
               "of_uniform_peak": r["macs_per_s"] / u if u else None, "algo": r["algo"], "check_bad": r["check"]["bad"],
               "sm_mhz": nv.get("sm_mhz_mean"), "power_w": nv.get("energy_w"), "beside": {}}
        cells = ""
        for w in warps:
            s = beside[key].get(w)
            if not s:
                cells += f"{'-':>7}{'-':>7}{'-':>7}"
                continue
            co = s["co_macs_per_s"] / b
            ok = s["co_saw_end"] == s["co_warps_ran"]
            row["beside"][w] = {"co_of_clock_bound": co, "keep": s["gemm_vs_alone"], "sum_of_clock_bound":
                                (s["macs_per_s"] + s["co_macs_per_s"]) / b, "co_sms": s["co_sms"], "co_saw_all": ok,
                                "power_w": (s.get("nvml") or {}).get("energy_w"),
                                "sm_mhz": (s.get("nvml") or {}).get("sm_mhz_mean")}
            cells += f"{co:>7.3f}{s['gemm_vs_alone']:>7.3f}{row['beside'][w]['sum_of_clock_bound']:>7.3f}" + ("" if ok else "!")
        summary["gemms"].append(row)
        print(f"{c:<11}{k:<5}{lib:<4}{wc:<8}{linear(n, kk):<11}{m:>5}{r['macs_per_s']:>11.4e}{row['of_clock_bound']:>7.3f}"
              f"{(row['of_uniform_peak'] or 0):>7.3f}{cells}{nv.get('sm_mhz_mean', 0):>6.0f}{nv.get('energy_w', 0):>5.0f}"
              f"  {r['algo']}, bad {r['check']['bad']}")
    missing = [k for k in tune if k not in alone]
    for k in missing:
        print(f"no timing: {k}: {tune[k].get('error')}")
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
