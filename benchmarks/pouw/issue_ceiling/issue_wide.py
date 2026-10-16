#!/usr/bin/env python3
# Combine issue_wide.jsonl + the SASS: per kind, how ptxas lowers it (SASS instrs an op), standalone dispatch throughput,
# IMMA-s8 MAC displacement an op (spec + unrolled), and the bit-exact two-word verdict. Emits uniform.md / uniform.json
# style report (wide.md / wide.json). Usage: issue_wide.py OUTDIR
import json, re, sys

OUT = sys.argv[1]
ROWS = [json.loads(l) for l in open(f"{OUT}/issue_wide.jsonl") if l.strip()]

# enum order in issue_wide.cu
KINDS = ["iadd3_32", "imad_32", "iadd_64", "mov_64", "cs2r_64", "imad_wide"]
KIND_IDX = {k: i for i, k in enumerate(KINDS)}
# the SASS opcode(s) that implement each op (besides shared loop control). ptxas lowers add.u64 to IADD3 + IADD3.X.
OP_OPCODES = {
    "iadd3_32":  ["IADD3"],
    "imad_32":   ["IMAD"],
    "iadd_64":   ["IADD3", "IADD3.X"],
    "mov_64":    ["MOV"],
    "cs2r_64":   ["CS2R"],
    "imad_wide": ["IMAD.WIDE"],
}
OPS_PER_ITER = 64  # k_alone unrolls 64 ops an iteration

def load_funcs(path):
    """split a cuobjdump -sass dump into {function_name: [instruction mnemonics]}"""
    funcs, name, lines = {}, None, []
    for ln in open(path, errors="replace"):
        m = re.search(r"Function : (\S+)", ln)
        if m:
            if name: funcs[name] = lines
            name, lines = m.group(1), []
            continue
        m = re.search(r"/\*[0-9a-f]+\*/\s+@?\S*\s*([A-Z][A-Z0-9_.]*)", ln)
        if m and name is not None:
            lines.append(m.group(1))
    if name: funcs[name] = lines
    return funcs

def alone_func(funcs, xidx):
    """k_alone<xidx> mangled name contains ILi<xidx>E"""
    for nm, ins in funcs.items():
        if "k_alone" in nm and f"ILi{xidx}E" in nm:
            return nm, ins
    return None, []

def count(ins, opcode):
    # exact-prefix match: IADD3 counts IADD3 but not IADD3.X; IADD3.X counts only the carry form
    if opcode.endswith(".X"):
        return sum(1 for o in ins if o == opcode)
    return sum(1 for o in ins if o == opcode or o.startswith(opcode + "."))

def sass_per_op(funcs, kind):
    xidx = KIND_IDX[kind]
    nm, ins = alone_func(funcs, xidx)
    if not ins:
        return None, {}, nm
    hist = {}
    for o in ins:
        base = o.split(".")[0]
        hist[o] = hist.get(o, 0) + 1
    # instrs an op = total op-specific opcodes / 64. For iadd_64 that is IADD3(non-.X) + IADD3.X.
    total = 0
    for opc in OP_OPCODES[kind]:
        if opc == "IADD3":  # count IADD3 but exclude the .X variant and loop control is tiny
            total += sum(1 for o in ins if o == "IADD3")
        elif opc == "IADD3.X":
            total += sum(1 for o in ins if o == "IADD3.X")
        else:
            total += count(ins, opc)
    return total / OPS_PER_ITER, hist, nm

def slope(pairs):
    # least-squares slope of y vs x through the points (x = ops a MMA, y = 1 - mac_frac_of_alone)
    n = len(pairs)
    sx = sum(x for x, _ in pairs); sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs); sxy = sum(x * y for x, _ in pairs)
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else 0.0

def disp(kind, mode):
    pts = [(r["ops_per_mma"], 1.0 - r["mma_macs_frac_of_alone"]) for r in ROWS
           if r.get("mode") == mode and r.get("op") == kind]
    pts = [(float(x), y) for x, y in pts]
    return slope(pts), pts

funcs = load_funcs(f"{OUT}/sass/issue_wide.sass")
alone = {r["op"]: r for r in ROWS if r.get("mode") == "alone"}
tw = next((r for r in ROWS if r.get("mode") == "twoword"), {})
macs_alone = next((r["mma_macs_per_sm_per_clk"] for r in ROWS if r.get("mode") == "specmma_alone"), 0)

# per-kind "does one 64-bit instruction perform two word-ops?"
def two_word_ops(kind, spo, hist):
    if kind in ("iadd3_32", "imad_32"):
        return "n/a (32-bit dest)"
    if kind == "iadd_64":
        c = "couples" if tw.get("carry_couples") else "no carry"
        swar = tw.get("swar_two_independent_31bit")
        # lowered to 2 instrs => the 2 words ARE 2 instructions; a single add can only do 2 INDEPENDENT lanes with a guard bit
        n = f"{spo:.0f} instrs/op; carry {c}; guard-bit SWAR 2x31b independent = {swar}"
        return n
    if kind == "mov_64":
        return f"{spo:.0f} instr(s)/op; moves 2 words (yes, data movement)"
    if kind == "cs2r_64":
        return f"{spo:.0f} instr(s)/op; reads a 64-bit clock (not attack compute)"
    if kind == "imad_wide":
        return f"{spo:.0f} instr/op; one 32x32->64 multiply (no, single word-op)"
    return "?"

report = {"macs_alone": macs_alone, "twoword": tw, "kinds": {}}
lines = []
lines.append("# VF3c 64-bit word-op probe: do 64-bit-destination instructions do two word-ops? (sm_120, RTX PRO 6000)\n")
lines.append(f"IMMA s8 MAC rate alone: {macs_alone:.1f} MAC/SM/cyc. Standalone = 8 warps a SMSP, ILP 8; displacement = "
             "least-squares slope of (1 - IMMA MAC frac of alone) vs ops a MMA.\n")
lines.append("| kind | SASS instrs/op (opcodes) | standalone ops/cyc/SMSP | slots (=1/thru) | disp/op spec | disp/op unrolled | two word-ops? |")
lines.append("|---|---|---|---|---|---|---|")
for kind in KINDS:
    spo, hist, nm = sass_per_op(funcs, kind)
    thru = alone.get(kind, {}).get("ops_per_smsp_per_cyc", 0.0)
    slots = (1.0 / thru) if thru else 0.0
    ds, _ = disp(kind, "spec")
    dm, _ = disp(kind, "mixu")
    opcs = ",".join(f"{o}x{c}" for o, c in sorted(hist.items(), key=lambda kv: -kv[1])
                    if o.split(".")[0] in ("IADD3", "IMAD", "MOV", "CS2R", "LOP3") or ".WIDE" in o)[:80]
    tword = two_word_ops(kind, spo or 0, hist)
    lines.append(f"| {kind} | {spo:.2f} ({opcs}) | {thru:.4f} | {slots:.2f} | {ds:.4f} | {dm:.4f} | {tword} |")
    report["kinds"][kind] = {"sass_instrs_per_op": spo, "sass_hist": hist, "standalone_ops_per_smsp_per_cyc": thru,
                             "slots_from_throughput": slots, "disp_per_op_spec": ds, "disp_per_op_mixu": dm,
                             "two_word_ops": tword, "func": nm}

lines.append("\n## Two-word bit-exact test (one add.u64)\n")
lines.append(f"- carry coupling: 0xFFFFFFFF + 1 = {tw.get('add64_carry')} -> carry crosses into high word: "
             f"**{tw.get('carry_couples')}**")
lines.append(f"- naive packed 2x32 (low lane overflows): {tw.get('packed_naive')} -> high lane corrupted by low carry: "
             f"**{tw.get('naive_corrupts')}** (so a full-32-bit 2x add is NOT two independent word-ops)")
lines.append(f"- guard-bit SWAR (two 31-bit lanes): {tw.get('packed_swar')} -> two independent 31-bit sums: "
             f"**{tw.get('swar_two_independent_31bit')}**")

open(f"{OUT}/wide.md", "w").write("\n".join(lines) + "\n")
json.dump(report, open(f"{OUT}/wide.json", "w"), indent=2)
print("\n".join(lines))
