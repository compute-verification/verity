"""The repriced break-even's tables from sweep.py's surfaces: the served line step by step over all six scopes with
each line's status, owner and the theorem form that prices it from `reprice_path.json`, its totals, the path from it, a
pool per window's lost hits, its extras, the per-window budget form and the advice, then the earlier main line, its
extras, the one-stage and plain columns.

    python benchmarks/spot_figures/breakeven/reprice_table.py SURFACE.json... [--json OUT]
"""
import json
import sys
from pathlib import Path

SCOPES = ("world year", "datacenter year")
ALL_SCOPES = ("world week", "world month", "world year", "datacenter week", "datacenter month", "datacenter year")
#: compiling one sampled request's Programs, the core-hours the line's total takes and its upper reading
COMPILE = ("3.8", "14.0")
#: the served line's levers, statuses and owners, which OPTMODES reads too
PATH = json.loads((Path(__file__).parent / "reprice_path.json").read_text())
#: the memory-checked KV pool beside a step: sampled like the other units, as built (every position paid for) and with
#: the ragged token row, each walked and with the head-of-line memo (not built); and proved every window for comparison
POOL = [("sampled", "built walked", "sampled like the other units, as built, every lookup walking (at least)"),
        ("sampled", "built memo", "sampled like the other units, as built with the head-of-line memo (at least)"),
        ("sampled", "ragged walked", "sampled like the other units, with the ragged token row, every lookup walking"),
        ("sampled", "ragged memo", "sampled like the other units, with the ragged token row and the memo"),
        ("every", "built walked", "proved every window, as built (superseded)"),
        ("every", "ragged walked", "proved every window, with the ragged token row (superseded)")]
#: the two pool readings a total takes until the pool's unit is settled
TOTAL_POOLS = (("sampled", "built walked", "the pool sampled as built"),
               ("sampled", "ragged walked", "the pool sampled with the ragged row"))
#: each theorem form a row is priced under, as the table names it
FORM = {"v5": "V5 per block: one-stage proof only", "tk": "T·K_δ: two-stage, proved",
        "v5-1s": "V5 per block: one-stage, proved"}


def form_of(case: str) -> str:
    """The earlier main line's rows: the one-stage design's under V5's form, which it proves, the rest two-stage."""
    return FORM["v5-1s" if case.endswith("-1s") else "v5"]
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


def total(rows, case, pool=TOTAL_POOLS[0][:2], delta=None):
    """Each scope's total FP4 compute at `case`: serve tax + proving + verifying + compiling at COMPILE[0] + the pool,
    plus `delta`'s points where given; None where the case never breaks even, and no entry where it wasn't run."""
    out = {}
    for s in ALL_SCOPES:
        r = rows.get((case, s))
        if r is None:
            continue
        if "pool" not in r or r["beta"]["FP4"] is None:
            out[s] = None
            continue
        out[s] = (r["tax"] + r["beta"]["FP4"] + r["compile"]["share"][COMPILE[0]] + r["pool"][pool[0]][pool[1]]
                  + (delta["delta"][s] if delta else 0.0))
    return out


def entries(rows, data, items, head, title):
    print(f"\n{title}\n{head}")
    for x in items:
        case = x["cases"]["two-stage"]
        c, d = cells(rows, case, ALL_SCOPES, beta=False)
        data[case] = d | {k: x[k] for k in ("label", "status", "owner", "form", "note")}
        print(f"| {case} | {x['label']} | {x['status']} | {x['owner']} | {FORM[x['form']]} | " + " | ".join(c) + " |")


def totals(rows, data, cases, what=""):
    for case in cases:
        for how, load, name in TOTAL_POOLS:
            t = total(rows, case, (how, load))
            data[f"{case} total, {name}"] = {"case": case, "compile_core_h": COMPILE[0], "pool": f"{how} {load}",
                                             "cells": t}
            print(f"| {case} total | {case}{what} + compiling at {COMPILE[0]} core-h + {name} | | | {FORM['v5']} | "
                  + " | ".join(pct(t[s]) if s in t else "–" for s in ALL_SCOPES) + " |")


def served(rows, data):
    """The served line's tables over all six scopes, as total FP4 compute, each row named by the theorem form that
    prices it: each step, the measured levers alone, what sits beside the line (compiling, the pool, #8086's delta) and
    the totals today, with the pool's two readings; the path from the last step and its total; a pool per window's lost
    hits and their totals; the extras on the last step, the per-window budget form and the advice."""
    head = ("| case | lever | status | owner | form | " + " | ".join(ALL_SCOPES) + " |\n|---|---|---|---|---|"
            + "---:|" * len(ALL_SCOPES))
    last = PATH["served"][-1]["cases"]["two-stage"]
    print("Forms: " + "; ".join(f"{FORM[k]}: {v}" for k, v in PATH["forms"].items()) + ".")
    entries(rows, data, PATH["served"] + PATH["served_measured"], head,
            "Served line, total FP4 compute (serve tax + proving + verifying; FP4 weights 376.7 GB):")
    for h in COMPILE:
        c = [pct(rows[(last, s)]["compile"]["share"][h]) if "compile" in rows.get((last, s), {}) else "–"
             for s in ALL_SCOPES]
        print(f"| compile | + compiling each sampled request's Programs, {h} core-h on 8 ranks, at {last}'s plan | "
              f"{'measured a step; span modeled' if h == COMPILE[0] else 'upper reading'} | @vllm, @circuits | "
              f"{FORM['v5']} | " + " | ".join(c) + " |")
    for how, load, what in POOL:
        c = [pct(rows[(last, s)]["pool"][how][load]) if "pool" in rows.get((last, s), {}) else "–" for s in ALL_SCOPES]
        print(f"| pool | + the memory-checked KV pool, {what} | measured ANDs, --zk rate; verifier assumed | MEMPOOL | "
              f"{FORM['v5']} | " + " | ".join(c) + " |")
    for delta in PATH["deltas"]:
        print(f"| delta | {delta['label']}: points on {delta['base']}, not in the totals | {delta['status']} | "
              f"{delta['owner']} | {FORM[delta['form']]} | "
              + " | ".join(f"{delta['delta'][s] * 100:+.2f} pt" for s in ALL_SCOPES) + " |")
    measured = PATH["served_measured"][0]["cases"]["two-stage"]
    totals(rows, data, [last, measured] + [x["cases"]["two-stage"] for x in PATH["path"]])
    entries(rows, data, PATH["path"], head, f"The path from {last} to 10% (total FP4; each step adds to the one above "
            "it):")
    entries(rows, data, PATH["pool_window"], head, "One pool per window: the hits it loses recomputed (total FP4 before "
            "compiling and the pool):")
    totals(rows, data, [x["cases"]["two-stage"] for x in PATH["pool_window"]])
    entries(rows, data, PATH["extras"], head, f"Extras on {last}, each alone (total FP4 before compiling and the pool):")
    entries(rows, data, PATH["zk"], head, "The per-window budget form, T windows of K_delta (total FP4):")
    print("\nAdvice not closed (each scope's FP4 bits less the advice it charges):\n" + head)
    for e, lever, label in ADVICE:
        case = f"{last}+{e}"
        c, d = cells(rows, case, ALL_SCOPES, beta=False)
        data[case] = d | {"label": lever, "status": label, "form": "v5"}
        print(f"| {case} | {lever} | {label} | | {FORM['v5']} | " + " | ".join(c) + " |")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_json = None
    if "--json" in argv:
        k = argv.index("--json")
        out_json, argv = Path(argv[k + 1]), argv[:k] + argv[k + 2:]
    rows = load(argv)
    data = {}
    if any(c == PATH["served"][0]["cases"]["two-stage"] for c, _ in rows):
        served(rows, data)
    if not any(c == MAIN[0][0] for c, _ in rows):
        if out_json:
            out_json.write_text(json.dumps(data, indent=1) + "\n")
        return
    head = ("| case | lever | priced | owner | form | world year β | total | datacenter year β | total |\n"
            "|---|---|---|---|---|---:|---:|---:|---:|")
    print("\nMain line (FP4, 376.7 GB; total = serve tax + β):\n" + head)
    for case, lever, label, owner in MAIN:
        c, d = cells(rows, case)
        data[case] = d | {"lever": lever, "priced": label, "owner": owner, "form": form_of(case)}
        print(f"| {case} | {lever} | {label} | {owner} | {form_of(case)} | " + " | ".join(c) + " |")
    for name, lever, label, owner in UNPRICED:
        data[name] = {"lever": lever, "priced": label, "owner": owner}
        print(f"| {name} | {lever} | {label} | {owner} | – | – | – | – | – |")
    for base in ("R6", "R6x4"):
        print(f"\nExtras on {base}:\n" + head)
        for e, lever, label, owner in EXTRAS:
            case = f"{base}+{e}"
            c, d = cells(rows, case)
            data[case] = d | {"lever": lever, "priced": label, "owner": owner, "form": form_of(case)}
            print(f"| {case} | {lever} | {label} | {owner} | {form_of(case)} | " + " | ".join(c) + " |")
    print("\nSensitivities, columns and comparisons:\n" + head)
    for case, what, label, owner in SIDE:
        c, d = cells(rows, case)
        data[case] = d | {"lever": what, "priced": label, "owner": owner, "form": form_of(case)}
        print(f"| {case} | {what} | {label} | {owner} | {form_of(case)} | " + " | ".join(c) + " |")
    if out_json:
        out_json.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
