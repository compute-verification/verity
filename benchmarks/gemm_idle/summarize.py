"""`python3 summarize.py OUT`: gemm_idle's passes (OUT/passes.jsonl, OUT/info.json) as tensor utilization.

Each clock label's bound `clk` is SMs x 1,024 MACs a clock (dense int8 and e4m3 mma.sync on sm_120; the zero-operand peak's
`macs_per_sm_clk` checks it) x the label's nominal clock (node-<MHz>: the node's lock as it reads idle; lock-<MHz>: the lock
asked for); `unif` is the uniform-operand peak measured at that label (the power-capped bound where the cap binds). A GEMM's
utilization is its MACs over the pass's wall time against each. Beside a co-runner entry W:SLEEP, `co` is the co-runner's
MACs over the same window against `clk` and `keep` the GEMM's rate there over the fastest candidate's alone (the mean of
its runs before and after the co-runner passes; `drift` is after over before). `x`: no candidate of the four fastest could
run beside it; `<`: the co-runner hit its deadline before the GEMM ended, so it has no count (it adds its MMAs when it
sees the end marker) and its keep overstates.
`co@99` and `co@96` are the most `co` of the entries (neither `x` nor `<`) whose keep is at least 0.99 and 0.96: what an
undeclared kernel takes while the GEMM loses at most 1% or 4% of its rate. Prints the tables and writes OUT/summary.json.
Stdlib only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median

PER_SM_CLK = 1024
MODELS = {"1b": {(3072, 2048): "qkv", (2048, 2048): "o", (16384, 2048): "gate_up", (2048, 8192): "down"},
          "8b": {(6144, 4096): "qkv", (4096, 4096): "o", (28672, 4096): "gate_up", (4096, 14336): "down"}}
KEEPS = (0.99, 0.96)


def linear(n: int, k: int) -> str:
    for m, shapes in MODELS.items():
        if (n, k) in shapes:
            return f"{m}.{shapes[(n, k)]}"
    return f"{n}x{k}"


def regimes(summary: dict) -> None:
    """Per clock, kind, weight class and regime (decode: T <= 32; prefill: each T), the ranges over the linears of each
    library and of the faster library per linear (`best`): clk, the idle share 1 - clk, co@99, co@96, and `fill96`, co@96
    over the idle share. Rows where nothing fit beside the GEMM count in `fit` as misses."""
    groups: dict = defaultdict(list)
    per_linear: dict = defaultdict(list)
    for g in summary["gemms"]:
        reg = "decode" if g["m"] <= 32 else f"T{g['m']}"
        groups[(g["clock"], g["kind"], g["wclass"], reg, g["lib"])].append(g)
        per_linear[(g["clock"], g["kind"], g["wclass"], reg, g["linear"], g["m"])].append(g)
    for (c, k, wc, reg, _, _), gs in per_linear.items():
        groups[(c, k, wc, reg, "best")].append(max(gs, key=lambda g: g["of_clock_bound"]))

    def span(xs: list) -> str:
        return f"{min(xs):.3f}-{max(xs):.3f} ({median(xs):.3f})" if xs else "-"

    def rank(key: tuple) -> tuple:
        c, k, wc, reg, lib = key
        return c, k, wc, 0 if reg == "decode" else int(reg[1:]), lib == "best", lib

    print("\nby regime (ranges over the linears, median in parentheses; best = the faster library per linear;"
          " fill96 = co@96 / (1 - clk); fit = linears some co-runner fit beside)")
    print(f"{'clock':<11}{'kind':<5}{'w':<8}{'regime':<8}{'lib':<5}{'n':>3}{'clk':>21}{'idle':>21}{'co@99':>21}"
          f"{'co@96':>21}{'fill96':>21}{'fit':>6}")
    summary["regimes"] = []
    for key in sorted(groups, key=rank):
        gs = groups[key]
        c, k, wc, reg, lib = key
        clk = [g["of_clock_bound"] for g in gs]
        fill = [g["co_at_keep_96"] / (1 - g["of_clock_bound"]) for g in gs if g["of_clock_bound"] < 1]
        fit = sum(any(b.get("co_resident") for b in g["beside"].values()) for g in gs)
        rec = {"clock": c, "kind": k, "wclass": wc, "regime": reg, "lib": lib, "n": len(gs), "clk": [min(clk), max(clk)],
               "co99": [min(g["co_at_keep_99"] for g in gs), max(g["co_at_keep_99"] for g in gs)],
               "co96": [min(g["co_at_keep_96"] for g in gs), max(g["co_at_keep_96"] for g in gs)],
               "fill96": [min(fill), max(fill)] if fill else None, "fit": fit}
        summary["regimes"].append(rec)
        print(f"{c:<11}{k:<5}{wc:<8}{reg:<8}{lib:<5}{len(gs):>3}{span(clk):>21}{span([1 - x for x in clk]):>21}"
              f"{span([g['co_at_keep_99'] for g in gs]):>21}{span([g['co_at_keep_96'] for g in gs]):>21}"
              f"{span(fill):>21}{fit:>4}/{len(gs)}")


def main(out: str) -> int:
    info = json.load(open(f"{out}/info.json"))
    sms = info["sms"]
    rows = [json.loads(ln) for ln in open(f"{out}/passes.jsonl") if ln.startswith("{")]
    peak: dict = defaultdict(dict)   # (clock, kind) -> class -> best peak row (8 accumulators, no sleep)
    co_alone: dict = {}              # (clock, kind, warps, sleep) -> rate
    for r in rows:
        if r["pass"] != "peak":
            continue
        key = (r["clock"], r["kind"])
        if r["class"] == "uniform" and r["nacc"] == 8:
            co_alone[(r["clock"], r["kind"], r["warps"], r.get("sleep_ns", 0))] = r["macs_per_s"]
        if r["nacc"] == 8 and not r.get("sleep_ns") and r["macs_per_s"] > peak[key].get(r["class"], {}).get("macs_per_s", 0):
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
                                     "sm_mhz_nvml": nv.get("sm_mhz_mean"), "reasons": nv.get("reasons")})
    for (c, k, w, s), v in sorted(co_alone.items()):
        if (c, k) in bound:
            summary["co_alone"].append({"clock": c, "kind": k, "warps": w, "sleep_ns": s, "of_clock_bound": v / bound[(c, k)]})
    print("\nco-runner alone (uniform, 8 accumulators), of clk: " + ", ".join(
        f"{x['clock']} {x['kind']} {x['warps']}:{x['sleep_ns']} {x['of_clock_bound']:.3f}" for x in summary["co_alone"]))

    alone, after, beside = {}, {}, defaultdict(dict)
    tune = {}
    for r in rows:
        key = (r.get("clock"), r.get("kind"), r.get("lib"), r.get("m"), r.get("n"), r.get("k"), r.get("wclass"))
        if r["pass"] == "alone":
            alone[key] = r
        elif r["pass"] == "alone_after":
            after[key] = r
        elif r["pass"] == "beside":
            beside[key][f"{r['co_warps']}:{r.get('co_sleep_ns', 0)}"] = r
        elif r["pass"] == "tune":
            tune[key] = r
    entries = sorted({e for b in beside.values() for e in b}, key=lambda e: tuple(map(int, e.split(":"))))
    hdr = "".join(f"{'co' + e:>10}{'keep':>6}" for e in entries) + "".join(f"{'co@' + str(round(k * 100)):>7}" for k in KEEPS)
    print("\nGEMMs: MACs over wall time; clk = against SMs x 1024 x the label's clock, unif = against the uniform peak there;"
          "\ndrift = the fastest alone after the co-runner passes / before; co<W:SLEEP> = the co-runner's rate beside it"
          " against clk, keep = the GEMM's rate beside it / the fastest alone (x: nothing fit beside it; <: the co-runner"
          " ended first); co@99, co@96 = the most co at keep >= 0.99, 0.96")
    print(f"{'clock':<11}{'kind':<5}{'lib':<4}{'w':<8}{'linear':<11}{'T':>5}{'MAC/s':>11}{'clk':>7}{'unif':>7}"
          f"{'drift':>7}{hdr}{'MHz':>6}{'W':>5}  algo, check")
    for key in sorted(alone, key=lambda x: (x[0], x[1], x[2], x[6], linear(x[4], x[5]), x[3])):
        c, k, lib, m, n, kk, wc = key
        r = alone[key]
        if (c, k) not in bound:
            continue
        b = bound[(c, k)]
        u = peak[(c, k)].get("uniform", {}).get("macs_per_s")
        nv = r.get("nvml") or {}
        ra = after.get(key)
        ref = (r["macs_per_s"] + ra["macs_per_s"]) / 2 if ra else r["macs_per_s"]
        row = {"clock": c, "kind": k, "lib": lib, "wclass": wc, "linear": linear(n, kk), "m": m, "n": n, "k": kk,
               "macs_per_s": r["macs_per_s"], "of_clock_bound": r["macs_per_s"] / b,
               "of_uniform_peak": r["macs_per_s"] / u if u else None,
               "drift": ra["macs_per_s"] / r["macs_per_s"] if ra else None, "algo": r["algo"],
               "check_bad": r["check"]["bad"], "sm_mhz": nv.get("sm_mhz_mean"), "power_w": nv.get("energy_w"), "beside": {}}
        cells = ""
        for e in entries:
            s = beside[key].get(e)
            if not s:
                cells += f"{'-':>10}{'-':>6}"
                continue
            if not s.get("co_resident", True) or "co_macs_per_s" not in s:
                row["beside"][e] = {"co_resident": False}
                cells += f"{'x':>10}{'x':>6}"
                continue
            snv = s.get("nvml") or {}
            outlasted = s.get("co_outlasted_gemm", True)
            row["beside"][e] = {"co_resident": True, "co_outlasted_gemm": outlasted,
                                "co_of_clock_bound": s["co_macs_per_s"] / b, "keep": s["macs_per_s"] / ref,
                                "algo": s["algo"], "sum_of_clock_bound": (s["macs_per_s"] + s["co_macs_per_s"]) / b,
                                "co_sms": s["co_sms"], "power_w": snv.get("energy_w"), "sm_mhz": snv.get("sm_mhz_mean")}
            if not outlasted:
                cells += f"{'<':>10}{row['beside'][e]['keep']:>6.3f}"
                continue
            mark = "" if s["algo"] == r["algo"] else "*"
            cells += f"{row['beside'][e]['co_of_clock_bound']:>9.3f}{mark or ' '}{row['beside'][e]['keep']:>6.3f}"
        for kp in KEEPS:
            got = [x for x in row["beside"].values()
                   if x.get("co_resident") and x.get("co_outlasted_gemm") and x["keep"] >= kp]
            best = max(got, key=lambda x: x["co_of_clock_bound"]) if got else None
            row[f"co_at_keep_{round(kp * 100)}"] = best["co_of_clock_bound"] if best else 0.0
            row[f"sum_at_keep_{round(kp * 100)}"] = best["sum_of_clock_bound"] if best else row["of_clock_bound"]
            cells += f"{row[f'co_at_keep_{round(kp * 100)}']:>7.3f}"
        summary["gemms"].append(row)
        dr = f"{row['drift']:>7.3f}" if ra else f"{'-':>7}"
        print(f"{c:<11}{k:<5}{lib:<4}{wc:<8}{linear(n, kk):<11}{m:>5}{r['macs_per_s']:>11.4e}{row['of_clock_bound']:>7.3f}"
              f"{(row['of_uniform_peak'] or 0):>7.3f}{dr}{cells}{nv.get('sm_mhz_mean', 0):>6.0f}"
              f"{nv.get('energy_w', 0):>5.0f}  {r['algo']}, bad {r['check']['bad']}")
    print("(* beside: a slower candidate than the fastest alone, the fastest that fit beside the co-runner)")
    regimes(summary)
    for k in tune:
        if k not in alone:
            print(f"no timing: {k}: {tune[k].get('error')}")
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
