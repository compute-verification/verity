"""The repriced break-even's tables from sweep.py's surfaces: the served line step by step over all six scopes, its
extras and its advice, then the earlier main line, its extras, the one-stage and plain columns.

    python benchmarks/spot_figures/breakeven/reprice_table.py SURFACE.json... [--json OUT]
"""
import json
import sys
from pathlib import Path

SCOPES = ("world year", "datacenter year")
ALL_SCOPES = ("world week", "world month", "world year", "datacenter week", "datacenter month", "datacenter year")
#: compiling one sampled request's Programs, the core-hours the line's total takes and its upper reading
COMPILE = ("3.8", "14.0")
#: the long-running proof server, the statement cache warm, each lever in turn
SERVED = [
    ("N0", "(A′) at 128k on the proof server today: one prover an H200, 600 s replay blocks, main's Lean verifier, the "
     "quant K6144 at its sibling's floor", "measured: 6 light types and the fused blocks on the server; the rest the "
     "cards × the cache's factor", "STATIONARY, @kernels"),
    ("N1", "+ the Lean verifier with its proved twins", "measured on 3 types (×0.21, ×0.22, ×0.58); ×0.21–0.23 "
     "assumed on 6; the median elsewhere", "VERIFYFAST"),
    ("N2", "+ lifetime replay, no edge pause, pages committed when written", "modeled from the 128k serve's trace",
     "LIFETIME"),
    ("N3", "+ four provers an H200 on full statements, each type by its device's busy share",
     "measured: light types 2.3–3.9× at a quarter statement, the fused blocks 1.12×; host-bound types one prover",
     "STATIONARY, @kernels"),
    ("N4", "+ the next session's front beside this one's device work", "modeled", "@kernels"),
    ("N5", "+ a GPU prover 4× on device parts, the witness build undivided", "modeled", "@kernels"),
]
SERVED_EXTRAS = [
    ("host", "the dense types' host witness 3.5× lower", "assumed (CPU's 2–4×)", "HILLCLIMB"),
    ("hash4", "in-proof hashing 4× (half the marginal)", "assumed", "no owner"),
    ("plain", "Flock without --zk", "fused blocks' shares measured on H200; the rest CPU's", "HILLCLIMB"),
    ("tmpl", "the templated session check where cheaper (RMSNorm types flat)", "estimate (quant K6144's ratio)",
     "VERIFYFAST"),
    ("vfloor", "the types the median prices at the fast column's floor (8.7 s)", "a lower bound", ""),
    ("fusedfront", "the fused blocks' front overlap at the kernels' split (device 0.55 of the session)",
     "modeled, unmeasured under contention", "@kernels"),
    ("card", "the quant K6144 at its card's 0.233 s a block (load)", "superseded", "HASHCUT"),
    ("host+tmpl", "host + tmpl", "assumed", ""),
    ("host+hash4+tmpl", "host + hash4 + tmpl", "assumed", ""),
]
#: the advice when it is not closed: every prefix-cache hit sent whole, or caching off
ADVICE = [
    ("hits", "+ the prefix-cache hits as advice, 5,181 bits a machine's window", "measured on 2k traces"),
    ("prefixoff", "+ prefix caching off: the hits' prefill recomputed, +13.86% in the serve tax", "measured at 128k"),
]
#: each step's lever, how it is priced, and its owner
MAIN = [
    ("R0", "(A′) the per-expert cut as Figure 1 prices it, at 128k over vLLM's 128k plan, 600 s replay blocks",
     "measured cards; 128k modeled", "proofs"),
    ("R1", "+ each expert's table read fused into its GEMM, staged at k_log 25",
     "measured on one H200 with the cache (512 and 1024 blocks); before the cache, its fixed over the cache's factor",
     "STATIONARY, HILLCLIMB"),
    ("R2", "+ the statement cache (#7875)",
     "measured on H200 (experts), priced from H200 splits (5 types) and CPU shares (2); the rest assumed ×0.1",
     "HILLCLIMB, STATIONARY"),
    ("R3", "+ 4 provers on one H200 with verify-ahead, on the expert types",
     "measured on the fused blocks at 128 and 256 blocks a statement", "STATIONARY"),
    ("R4", "+ the GPU prover 3.5× on device parts", "modeled (host witness untouched)", "@kernels (PRF-459)"),
    ("R4x4", "+ the GPU prover 4× instead", "modeled", "@kernels (PRF-459)"),
    ("R5", "R4 + lifetime replay at 458 GPU-s a sample, no edge pause, pages committed when written",
     "modeled from the 128k serve's trace", "LIFETIME"),
    ("R5x4", "R4x4 + lifetime replay", "modeled", "LIFETIME"),
    ("R6", "R5 + the Lean verifier's CPU a statement, at 81.8 core-seconds a GPU-second (Nebius list prices)",
     "measured on 9 types, scaled within families, the median (681 s) elsewhere", "VERIFYCOST"),
    ("R6x4", "R5x4 + the Lean verifier's CPU", "as R6", "VERIFYCOST"),
]
#: named lines no case prices yet
UNPRICED = [
    ("compile-slice proving", "every window's CompileSlice_v3 proved, never sampled", "unmeasured",
     "@network-accounting"),
]
EXTRAS = [
    ("all", "4 provers on every type, the experts' factor beyond the cache", "assumed", "STATIONARY"),
    ("4perH200", "four provers per H200 at full statements: each prover's peak about 33 GiB", "assumed",
     "@kernels (four-provers-per-h200)"),
    ("host", "the dense types' host witness 3.5× lower", "assumed (CPU's 2–4×)", "HILLCLIMB"),
    ("hash4", "in-proof hashing 4× (half the marginal)", "assumed", "no owner"),
    ("plain", "Flock without --zk", "fused blocks' shares measured on H200 (21%, 24%); the rest CPU's",
     "HILLCLIMB, STATIONARY"),
    ("all+4perH200", "all + four provers per H200 (the closing pair)", "assumed", ""),
    ("all+4perH200+host", "all + four provers per H200 + host", "assumed", ""),
    ("all+4perH200+host+plain", "all + four provers per H200 + host + plain", "assumed", ""),
    ("all+4perH200+host+hash4", "all + four provers per H200 + host + hash4", "assumed", ""),
]
SIDE = [
    ("R6x4+vflat", "R6x4 with the verifier's flat fixes (0.4 of today's)", "estimate", "VERIFYCOST"),
    ("R6x4+vtmpl", "R6x4 with the templated, served verifier: quant K6144's estimate (125 s, 0.06 of today's) for "
     "every type", "assumed", "VERIFYCOST"),
    ("R6x4+vmedian", "R6x4 with the expert and fused types' verifier at the median (681 s), not the MoE GEMM's 15 s",
     "assumed", ""),
    ("R6x4+vfloor", "R6x4 with the types the median prices at the verifier's 10 s floor instead", "a lower bound",
     ""),
    ("R6x4+all+4perH200+vflat", "the closing pair with the verifier's flat fixes", "estimate", ""),
    ("R6x4+all+4perH200+vtmpl", "the closing pair with the templated verifier for every type", "assumed", ""),
    ("R6x4+all+4perH200+host+vtmpl", "the closing pair + host, with the templated verifier", "assumed", ""),
    ("R6x4+all+4perH200+host+hash4+vtmpl", "the closing pair + host + hash4, with the templated verifier", "assumed",
     ""),
    ("R6x4+all+4perH200+vmedian", "the closing pair with the expert types' verifier at the median", "assumed", ""),
    ("R6x4+all+4perH200+vfloor", "the closing pair with the median's types at the 10 s floor", "a lower bound", ""),
    ("R1-cpuratio", "R1 with the fused blocks at the CPU ratio (0.75 w13, 0.59 w2) on the pair's prices", "CPU", ""),
    ("R6x4-cpuratio", "R6x4 with the fused blocks at the CPU ratio", "CPU", ""),
    ("R1-1s", "one-stage design on R1's blocks, (C)'s hashing tax", "", ""),
    ("R6x4-1s", "one-stage design on R6x4's blocks, (C)'s hashing tax", "", ""),
    ("R6x4-served", "R6x4 on the 128k plan as served (seven longest rows aborted)", "", ""),
]


def load(paths):
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            out[(r["case"], r["scope"])] = r
    return out


def pct(x):
    return "never" if x is None else (f"{x:.3g}×" if x >= 1 else f"{x:.2%}")


def cells(rows, case, scopes=SCOPES, beta=True):
    out, data = [], {}
    for s in scopes:
        r = rows.get((case, s))
        if r is None:
            out += ["–", "–"] if beta else ["–"]
            continue
        b = r["beta"]["FP4"]
        out += ([pct(b)] if beta else []) + [pct(None if b is None else r["tax"] + b)]
        data[s] = {"beta_fp4": b, "beta_fp8": r["beta"]["FP8"], "tax": r["tax"],
                   "total_fp4": None if b is None else r["tax"] + b, "bound_at_10pct_bits": r["bound"]["0.1"],
                   "advice_bits": r.get("advice_bits", 0.0)}
        if "compile" in r:
            data[s]["compile"] = r["compile"]
    return out, data


def served(rows, data):
    """The served line's tables over all six scopes: the total FP4 compute a step, then the compiling line at the
    last step's plan, the extras on it and the advice."""
    head = ("| case | lever | priced | owner | " + " | ".join(ALL_SCOPES) + " |\n|---|---|---|---|"
            + "---:|" * len(ALL_SCOPES))
    print("Served line, total FP4 compute (serve tax + proving + verifying; 376.7 GB):\n" + head)
    for case, lever, label, owner in SERVED:
        c, d = cells(rows, case, ALL_SCOPES, beta=False)
        data[case] = d | {"lever": lever, "priced": label, "owner": owner}
        print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    last = SERVED[-1][0]
    for h in COMPILE:
        c = [pct(rows[(last, s)]["compile"]["share"][h]) if "compile" in rows.get((last, s), {}) else "–"
             for s in ALL_SCOPES]
        print(f"| compile | + compiling each sampled request's Programs, {h} core-h on 8 ranks, at {last}'s plan | "
              f"{'measured a step; 256-step span modeled' if h == COMPILE[0] else 'upper reading'} | @vllm, @circuits "
              "| " + " | ".join(c) + " |")
    c = []
    for s in ALL_SCOPES:
        r = rows.get((last, s))
        c.append("–" if r is None or "compile" not in r or r["beta"]["FP4"] is None
                 else pct(r["tax"] + r["beta"]["FP4"] + r["compile"]["share"][COMPILE[0]]))
    data["total"] = {"case": last, "compile_core_h": COMPILE[0], "cells": dict(zip(ALL_SCOPES, c))}
    print(f"| total | {last} + compiling at {COMPILE[0]} core-h | | | " + " | ".join(c) + " |")
    print(f"\nExtras on {last}:\n" + head)
    for e, lever, label, owner in SERVED_EXTRAS:
        case = f"{last}+{e}"
        c, d = cells(rows, case, ALL_SCOPES, beta=False)
        data[case] = d | {"lever": lever, "priced": label, "owner": owner}
        print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    print("\nAdvice not closed (each scope's FP4 bits less the advice it charges):\n" + head)
    for base in ("N2", "N3", last):
        for e, lever, label in ADVICE:
            case = f"{base}+{e}"
            c, d = cells(rows, case, ALL_SCOPES, beta=False)
            data[case] = d | {"lever": lever, "priced": label}
            print(f"| {case} | {lever} | {label} | | " + " | ".join(c) + " |")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_json = None
    if "--json" in argv:
        k = argv.index("--json")
        out_json, argv = Path(argv[k + 1]), argv[:k] + argv[k + 2:]
    rows = load(argv)
    data = {}
    if any(c == SERVED[0][0] for c, _ in rows):
        served(rows, data)
    if not any(c == MAIN[0][0] for c, _ in rows):
        if out_json:
            out_json.write_text(json.dumps(data, indent=1) + "\n")
        return
    head = "| case | lever | priced | owner | world year β | total | datacenter year β | total |\n|---|---|---|---|---:|---:|---:|---:|"
    print("\nMain line (FP4, 376.7 GB; total = serve tax + β):\n" + head)
    for case, lever, label, owner in MAIN:
        c, d = cells(rows, case)
        data[case] = d | {"lever": lever, "priced": label, "owner": owner}
        print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    for name, lever, label, owner in UNPRICED:
        data[name] = {"lever": lever, "priced": label, "owner": owner}
        print(f"| {name} | {lever} | {label} | {owner} | – | – | – | – |")
    for base in ("R6", "R6x4"):
        print(f"\nExtras on {base}:\n" + head)
        for e, lever, label, owner in EXTRAS:
            case = f"{base}+{e}"
            c, d = cells(rows, case)
            data[case] = d | {"lever": lever, "priced": label, "owner": owner}
            print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    print("\nSensitivities, columns and comparisons:\n" + head)
    for case, what, label, owner in SIDE:
        c, d = cells(rows, case)
        data[case] = d | {"lever": what, "priced": label, "owner": owner}
        print(f"| {case} | {what} | {label} | {owner} | " + " | ".join(c) + " |")
    if out_json:
        out_json.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
