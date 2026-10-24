#!/usr/bin/env python3
# Round-6 combiner: for each kind, confirm the sm_120 lowering from SASS, then report the two numbers a JI3 joint row needs
# (MMA·a_M + op·a_X <= cycles), both as fractions in JI3 units:
#   (1) displacement of IMMA per op = a_X/a_M = least-squares slope of (1 - IMMA-MAC-frac-of-alone) vs ops-a-MMA at
#       OPM <= 8 (k_mixu; k_spec corroborates). A fraction of one IMMA.
#   (2) own sustained rate = ops / SMSP-cycle standalone = 1/a_X. In JI3 units a vector instruction issues at 1/cycle, so
#       1.0 == a normal vector kind. Reported at 8-way and 16-way ILP; the max is the dispatch-bound rate (plateau => not
#       latency-bound).
# a_X in JI3 units (normal-vector-instruction slots an op costs) = slope·a_M (from displacement) and 1/rate (from the
# standalone rate); the two should agree, and a joint row takes the max. a_M = measured IMMA-s8 cycles a MMA a SMSP.
# Usage: issue_wide.py OUTDIR
import json, re, sys

OUT = sys.argv[1]
ROWS = [json.loads(l) for l in open(f"{OUT}/issue_wide.jsonl") if l.strip()]

# enum order in issue_wide.cu
KINDS = ["iadd3_32", "imad_32", "iadd_64", "imad_wide", "dadd", "dfma", "red64", "redv4_f32"]
KIND_IDX = {k: i for i, k in enumerate(KINDS)}
# the SASS mnemonic each op lowers to on sm_120 (confirmed by cuobjdump): add.u32 -> IADD3 (adjacent independent adds do
# not fuse here); mad.lo.u32 -> IMAD; add.u64 -> single IADD.64; mad.wide.u32 -> single IMAD.WIDE.U32; add.f64 -> DADD;
# fma.rn.f64 -> DFMA; red.global.add.u64 -> REDG.E.ADD.64; red.global.add.v4.f32 -> REDG.E.ADD.F32x4.
OP_MATCH = {
    "iadd3_32":  lambda o: o == "IADD3",
    "imad_32":   lambda o: o == "IMAD",
    "iadd_64":   lambda o: o == "IADD.64",
    "imad_wide": lambda o: o.startswith("IMAD.WIDE"),
    "dadd":      lambda o: o == "DADD",
    "dfma":      lambda o: o == "DFMA",
    "red64":     lambda o: o.startswith("REDG.E.ADD.64") or o.startswith("RED.E.ADD.64"),
    "redv4_f32": lambda o: o.startswith("REDG.E.ADD.F32x4") or o.startswith("RED.E.ADD.F32x4"),
}
OPS_PER_ITER = 64  # k_alone unrolls 64 ops an iteration


def load_funcs(path):
    """split a cuobjdump -sass dump into {function_name: [instruction mnemonics]}"""
    funcs, name, lines = {}, None, []
    for ln in open(path, errors="replace"):
        m = re.search(r"Function : (\S+)", ln)
        if m:
            if name:
                funcs[name] = lines
            name, lines = m.group(1), []
            continue
        m = re.search(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Za-z0-9_.]*)", ln)
        if m and name is not None:
            lines.append(m.group(1))
    if name:
        funcs[name] = lines
    return funcs


def alone_func(funcs, xidx, ilp=8):
    """k_alone<xidx, ilp> mangled name contains ILi<xidx>ELi<ilp>E"""
    for nm, ins in funcs.items():
        if "k_alone" in nm and f"ILi{xidx}ELi{ilp}E" in nm:
            return nm, ins
    return None, []


def sass_per_op(funcs, kind):
    xidx = KIND_IDX[kind]
    nm, ins = alone_func(funcs, xidx, 8)
    if not ins:
        return None, {}, nm
    hist = {}
    for o in ins:
        hist[o] = hist.get(o, 0) + 1
    total = sum(1 for o in ins if OP_MATCH[kind](o))
    return total / OPS_PER_ITER, hist, nm


def slope(pairs):
    # least-squares slope of y vs x through (x = ops a MMA, y = 1 - mac_frac_of_alone)
    n = len(pairs)
    sx = sum(x for x, _ in pairs); sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs); sxy = sum(x * y for x, y in pairs)
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else 0.0


def disp(kind, mode):
    # The issue-contention model is exact: mac_frac = 1/(1 + (a_X/a_M)·OPM), i.e. 1/mac_frac = 1 + (a_X/a_M)·OPM. So the
    # displacement a_X/a_M is the slope of 1/mac_frac vs OPM (intercept ~1), NOT the slope of (1 - mac_frac), which
    # saturates and understates displacement badly for slow ops. Fit over OPM <= 8 (the co-issue regime; OPM=16 bends).
    pts = [(float(r["ops_per_mma"]), r["mma_macs_frac_of_alone"]) for r in ROWS
           if r.get("mode") == mode and r.get("op") == kind and r["mma_macs_frac_of_alone"] > 0]
    recip = [(x, 1.0 / f) for x, f in pts]
    lin = [p for p in recip if p[0] <= 8] or recip
    return slope(lin), pts


funcs = load_funcs(f"{OUT}/sass/issue_wide.sass")
alone = {}
for r in ROWS:
    if r.get("mode") == "alone":
        alone.setdefault(r["op"], {})[r.get("ilp", 8)] = r["ops_per_smsp_per_cyc"]
macs_alone = next((r["mma_macs_per_sm_per_clk"] for r in ROWS if r.get("mode") == "specmma_alone"), 0)
# a_M: IMMA-s8 SMSP-cycles a MMA = 1 / (MMAs a SMSP a cycle) = 4 (SMSPs) * 4096 (MACs a MMA) / (MAC a SM a cyc)
a_M = (4.0 * 4096.0 / macs_alone) if macs_alone else 0.0

report = {"macs_alone": macs_alone, "a_M_cyc_per_mma_smsp": a_M, "kinds": {}}
lines = []
lines.append("# r6 joint-row inputs: displacement of IMMA an op and own sustained rate, in JI3 units (sm_120, RTX PRO 6000)\n")
lines.append(f"IMMA s8 alone: {macs_alone:.1f} MAC/SM/cyc -> a_M = {a_M:.2f} SMSP-cycles a MMA. JI3 prices a normal vector "
             "instruction at 1 cycle; an op's JI3 cost a_X (cycles) is slope·a_M (from displacement) and 1/rate (from the "
             "standalone rate), which agree for a dispatch-bound kind; a joint row takes the max.\n")
lines.append("| kind | SASS instr/op | disp. of IMMA an op (a_X/a_M; mixu, spec) | sustained rate ops/cyc (ilp8, ilp16) | "
             "a_X cyc (from disp / from rate) |")
lines.append("|---|---|---|---|---|")
for kind in KINDS:
    spo, hist, nm = sass_per_op(funcs, kind)
    rates = alone.get(kind, {})
    r8, r16 = rates.get(8, 0.0), rates.get(16, 0.0)
    rate = max(r8, r16)  # dispatch-bound = the higher of the two ILP levels
    ds, sp = disp(kind, "spec")
    dm, mp = disp(kind, "mixu")
    fracs = " ".join(f"{int(x)}:{y:.3f}" for x, y in sorted(mp))  # y is mac_frac of alone
    aX_disp = dm * a_M
    aX_rate = (1.0 / rate) if rate else 0.0
    opcs = ",".join(f"{o}x{c}" for o, c in sorted(hist.items(), key=lambda kv: -kv[1]) if OP_MATCH[kind](o))[:40]
    lines.append(f"| {kind} | {spo:.2f} ({opcs}) | {dm:.4f}, {ds:.4f} | {r8:.4f}, {r16:.4f} | {aX_disp:.2f} / {aX_rate:.2f} |")
    report["kinds"][kind] = {
        "sass_instrs_per_op": spo, "sass_hist": hist, "func": nm,
        "disp_per_op_mixu": dm, "disp_per_op_spec": ds,           # a_X/a_M, fraction of one IMMA
        "sustained_rate_ops_per_cyc_ilp8": r8, "sustained_rate_ops_per_cyc_ilp16": r16,
        "sustained_rate_ops_per_cyc": rate,                        # dispatch-bound (max over ILP), 1/a_X in JI3 units
        "a_X_cyc_from_disp": aX_disp, "a_X_cyc_from_rate": aX_rate,
        "mixu_frac_of_alone_by_opm": fracs}
lines.append("\n_IMMA MAC frac of alone by ops-a-MMA (k_mixu):_")
for kind in KINDS:
    lines.append(f"- {kind}: {report['kinds'][kind]['mixu_frac_of_alone_by_opm']}")

lines.append("\n_dispatch-bound check (standalone ops/cyc at 8-way vs 16-way ILP; equal => the rate is the throughput "
             "limit, not latency):_")
for kind in KINDS:
    k = report["kinds"][kind]
    lines.append(f"- {kind}: ilp8 {k['sustained_rate_ops_per_cyc_ilp8']:.4f}, ilp16 "
                 f"{k['sustained_rate_ops_per_cyc_ilp16']:.4f}")

open(f"{OUT}/wide.md", "w").write("\n".join(lines) + "\n")
json.dump(report, open(f"{OUT}/wide.json", "w"), indent=2)
print("\n".join(lines))
