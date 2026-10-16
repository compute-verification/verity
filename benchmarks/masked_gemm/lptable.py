"""The table of lpgemm.cu's bench: the one-side-peeled noise's 1 + gamma over cuBLASLt's plain int8 GEMM, per shape, for
each route (the peel by cuBLASLt's K extension, or the custom mainloop's tail stage with A' from a pass or added in shared
memory; B's edit and S's digits as two passes or one), with no receipts and with duo at L = 32, gamma_gen (U and F), the
peel's own cost, and every arm; the gate's verdict for each arm.

python3 lptable.py BENCH.jsonl GATE.jsonl > table.md
"""

import json
import math
import re
import sys

LLAMA = ["qkv", "o", "gate_up", "down"]
BS = ["bsep", "bfus"]
NONE = [f"{r}_{b}" for r in ("lt", "t", "f", "o") for b in BS]
DUO = [f"{r}_{b}" for r in ("td", "fd", "od") for b in BS]
LABEL = {"lt": "cuBLASLt on the extended operands (A' + ext pass)", "t": "tail peel, A' by a pass", "f": "tail peel, A' fused (serialized)",
         "o": "tail peel, A' fused (overlapped)", "td": "duo 32 + tail, A' by a pass", "fd": "duo 32 + tail, A' fused (serialized)",
         "od": "duo 32 + tail, A' fused (overlapped)"}
BLABEL = {"bsep": "B' and S two passes", "bfus": "B' and S one pass"}
GLOBAL = ["sigma", "Kp", "inputs_range", "inputs_saturated_rows", "U_in_S", "U_uniform", "F_pm1", "Ap_Bp_range", "S_range", "PB_digits",
          "ext_digits", "AB_exact_range", "peel_identity", "ext_identity", "every_arm_checked"]
# (label, arms summed with sign)
PARTS = [("U's expansion (genU)", [("genU", 1)]),
         ("F's expansion alone (genF)", [("genF", 1)]),
         ("A' pass, first k columns (formA)", [("formA", 1)]),
         ("A' pass with the extension (formAx)", [("formAx", 1)]),
         ("B' pass, F's SHA included (formB)", [("formB", 1)]),
         ("S pass, the tail's digits (speel)", [("speel", 1)]),
         ("S pass with the extension's digits (speelx)", [("speelx", 1)]),
         ("B' and S in one pass (bsp)", [("bsp", 1)]),
         ("B' and S in one pass with the extension (bspx)", [("bspx", 1)]),
         ("cuBLASLt at k + X (ltK)", [("ltK", 1)]),
         ("the peel's GEMM increment, cuBLASLt (ltK - plain)", [("ltK", 1), ("plain", -1)]),
         ("custom mainloop, no tail (cplain)", [("cplain", 1)]),
         ("custom mainloop with the tail (tK)", [("tK", 1)]),
         ("the tail's increment (tK - cplain)", [("tK", 1), ("cplain", -1)]),
         ("tail + A' added in shared memory, serialized (fK)", [("fK", 1)]),
         ("tail + A' added in shared memory, overlapped (oK)", [("oK", 1)]),
         ("duo 32 on A, B alone, no tail (cduo)", [("cduo", 1)])]


def gate_verdicts(path):
    checks = {}
    for line in open(path):
        for key, v in json.loads(line).items():
            if isinstance(v, bool) and key not in ("ok", "receipts"):
                checks.setdefault(key, []).append(v)
    return checks


def arm_gate(name, checks):
    keys = [k for k in checks if re.fullmatch(rf"(U|R|XA|XB|PA|PB|G|cover)_{re.escape(name)}", k)]
    if not keys:
        return "unchecked"
    keys += GLOBAL
    return "pass" if all(k in checks and all(checks[k]) for k in keys) else "FAIL"


def value(arms, parts):
    if any(a not in arms for a, _ in parts):
        return None
    return sum(s * arms[a]["ms"] for a, s in parts) / arms["plain"]["ms"]


def geo(vs):
    return math.exp(sum(math.log(v) for v in vs) / len(vs)) if vs and all(v and v > 0 for v in vs) else None


def main(bench, gate):
    rows = [json.loads(line) for line in open(bench) if line.strip()]
    checks = gate_verdicts(gate)
    names = sorted({a for j in rows for a in j["arms"]})
    verdict = {a: arm_gate(a, checks) for a in names}
    by = {j["shape"]: j for j in rows}
    cols = [j["shape"] for j in rows]
    llama = [s for s in LLAMA if s in by]
    out = ["# One-side-peeled noise (RS-D2F): 1 + gamma on sm_120 against cuBLASLt int8", ""]
    if rows:
        out.append(f"GPU {rows[0]['gpu']}, {rows[0]['sms']} SMs. Y = A B'^T served as (A + U W^T)(B + F)^T - U (W^T (B + F)^T), "
                   "U (m x 32, {±1..±8}) and F (±1) from SHA-512 in counter mode, fresh every call; L = 32. Every route includes "
                   "U's expansion, F's (in B's edit pass) and the peel.")
    out.append("Gate (lpref.py against hashlib and numpy, every arm bit-exact, the peel included): "
               + ("every timed arm passed" if all(v == "pass" for v in verdict.values())
                  else ", ".join(f"{a} {v}" for a, v in verdict.items() if v != "pass")) + ".")
    hdr = ["| | " + " | ".join(cols) + (" | Llama geomean |" if llama else " |"), "|---|" + "---|" * (len(cols) + bool(llama))]

    def row(label, f, fmt="{:.3f}", geom=True):
        vals = [f(by[c]) for c in cols]
        cells = ["–" if v is None else (v if isinstance(v, str) else fmt.format(v)) for v in vals]
        if llama:
            g = geo([f(by[c]) for c in llama]) if geom else None
            cells.append("–" if g is None or isinstance(g, str) else fmt.format(g))
        return f"| {label} | " + " | ".join(cells) + " |"

    def shape_desc(j):
        return f"{j['m']}×{j['n']}×{j['k']}" + (f" ×{j['batch']}" if j["batch"] > 1 else "")

    def best(j, pats):
        b = None
        for p in pats:
            if verdict.get(p) != "pass":
                continue
            v = value(j["arms"], [(p, 1)])
            if v is not None and (b is None or v < b[0]):
                b = (v, p)
        return b

    out += ["", "## 1 + gamma (route time over plain cuBLASLt int8)", ""] + hdr
    out.append(row("m × n × k (× problems)", shape_desc, geom=False))
    out.append(row("plain cuBLASLt int8, ms", lambda j: j["arms"]["plain"]["ms"], "{:.4f}", geom=False))
    out.append(row("k + X (the extension's columns, mx parts)", lambda j: f"{j['Kp']} ({j['mx']})", geom=False))
    out.append(row("gamma_gen (genU + genF)", lambda j: value(j["arms"], [("genU", 1), ("genF", 1)])))
    for p in NONE + DUO:
        r, b = p.split("_")
        out.append(row(f"{LABEL[r]}, {BLABEL[b]}", lambda j, p=p: (value(j["arms"], [(p, 1)])
                                                                   if verdict.get(p) == "pass" else None)))
    for lab, pats in [("cheapest, no receipts", NONE), ("cheapest, duo 32", DUO)]:
        out.append(row(f"**{lab}**", lambda j, pats=pats: (best(j, pats) or (None,))[0]))
        out.append(row(f"({lab}: route)", lambda j, pats=pats: (best(j, pats) or (None, "–"))[1], geom=False))
    out += ["", "## The parts, over plain", ""] + hdr
    for label, parts in PARTS:
        out.append(row(label, lambda j, parts=parts: value(j["arms"], parts), geom=False))
    out += ["", "## Every arm, ms (median of interleaved repetitions) and over plain", "",
            "| shape | arm | ms | / plain | MHz | gate |", "|---|---|---|---|---|---|"]
    for j in rows:
        arms = j["arms"]
        for a in sorted(arms):
            out.append(f"| {j['shape']} | {a} | {arms[a]['ms']:.4f} | {arms[a]['ms'] / arms['plain']['ms']:.3f} | {arms[a]['mhz']:.0f} | "
                       f"{verdict[a]} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
