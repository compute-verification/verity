"""The repriced break-even's tables from sweep.py's surfaces: the main line step by step, the extras on its last two
steps, the one-stage and plain columns, and the advice's charge.

    python benchmarks/spot_figures/breakeven/reprice_table.py SURFACE.json... [--json OUT]
"""
import json
import sys
from pathlib import Path

SCOPES = ("world year", "datacenter year")
#: each step's lever, how it is priced, and its owner
MAIN = [
    ("R0", "(A′) the per-expert cut as Figure 1 prices it, at 128k over vLLM's 128k plan, 600 s replay blocks",
     "measured cards; 128k modeled", "proofs"),
    ("R1", "+ each expert's table read fused into its GEMM (#7852)", "modeled until its price lands", "@circuits"),
    ("R2", "+ the statement cache (#7875)",
     "measured on H200 (experts), priced from H200 splits (5 types) and CPU shares (2); the rest assumed ×0.1",
     "HILLCLIMB, STATIONARY"),
    ("R3", "+ 4 provers on one H200 with verify-ahead, on the expert types",
     "measured on the unfused GEMMs, applied to the fused blocks", "STATIONARY"),
    ("R4", "+ the GPU prover 3.5× on device parts", "modeled (host witness untouched)", "@kernels (PRF-459)"),
    ("R4x4", "+ the GPU prover 4× instead", "modeled", "@kernels (PRF-459)"),
    ("R5", "R4 + lifetime replay at 458 GPU-s a sample, no edge pause, pages committed when written",
     "modeled from the 128k serve's trace", "LIFETIME"),
    ("R5x4", "R4x4 + lifetime replay", "modeled", "LIFETIME"),
]
EXTRAS = [
    ("all", "4 provers on every type, the experts' factor beyond the cache", "assumed", "STATIONARY"),
    ("arena", "the prover frees its arena between sessions: full statements under 4 provers", "assumed",
     "@kernels (scoping)"),
    ("host", "the dense types' host witness 3.5× lower", "assumed (CPU's 2–4×)", "HILLCLIMB"),
    ("hash4", "in-proof hashing 4× (half the marginal)", "assumed", "no owner"),
    ("plain", "Flock without --zk at the CPU shares", "measured on CPU; H200 pending", "HILLCLIMB, STATIONARY"),
    ("all+arena", "all + arena", "assumed", ""),
    ("all+arena+host", "all + arena + host", "assumed", ""),
    ("all+arena+host+plain", "all + arena + host + plain", "assumed", ""),
    ("all+arena+host+hash4", "all + arena + host + hash4", "assumed", ""),
]
SIDE = [
    ("R1-1s", "one-stage design on R1's blocks, (C)'s hashing tax"),
    ("R5x4-1s", "one-stage design on R5x4's blocks, (C)'s hashing tax"),
    ("R0-census", "R0 on the census request (4,500 tokens, the figure's replay)"),
    ("R4x4-census", "R4x4 on the census request"),
    ("R5x4-served", "R5x4 on the 128k plan as served (seven longest rows aborted)"),
]


def load(paths):
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            out[(r["case"], r["scope"])] = r
    return out


def pct(x):
    return "never" if x is None else (f"{x:.3g}×" if x >= 1 else f"{x:.2%}")


def cells(rows, case):
    out, data = [], {}
    for s in SCOPES:
        r = rows.get((case, s))
        if r is None:
            out += ["–", "–"]
            continue
        b = r["beta"]["FP4"]
        out += [pct(b), pct(None if b is None else r["tax"] + b)]
        data[s] = {"beta_fp4": b, "beta_fp8": r["beta"]["FP8"], "tax": r["tax"],
                   "total_fp4": None if b is None else r["tax"] + b, "bound_at_10pct_bits": r["bound"]["0.1"]}
    return out, data


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_json = None
    if "--json" in argv:
        k = argv.index("--json")
        out_json, argv = Path(argv[k + 1]), argv[:k] + argv[k + 2:]
    rows = load(argv)
    head = "| case | lever | priced | owner | world year β | total | datacenter year β | total |\n|---|---|---|---|---:|---:|---:|---:|"
    data = {}
    print("Main line (FP4, 376.7 GB; total = serve tax + β):\n" + head)
    for case, lever, label, owner in MAIN:
        c, d = cells(rows, case)
        data[case] = d | {"lever": lever, "priced": label, "owner": owner}
        print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    for base in ("R5", "R5x4"):
        print(f"\nExtras on {base}:\n" + head)
        for e, lever, label, owner in EXTRAS:
            case = f"{base}+{e}"
            c, d = cells(rows, case)
            data[case] = d | {"lever": lever, "priced": label, "owner": owner}
            print(f"| {case} | {lever} | {label} | {owner} | " + " | ".join(c) + " |")
    print("\nColumns and comparisons:\n| case | what | world year β | total | datacenter year β | total |\n|---|---|---:|---:|---:|---:|")
    for case, what in SIDE:
        c, d = cells(rows, case)
        data[case] = d | {"lever": what}
        print(f"| {case} | {what} | " + " | ".join(c) + " |")
    if out_json:
        out_json.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
