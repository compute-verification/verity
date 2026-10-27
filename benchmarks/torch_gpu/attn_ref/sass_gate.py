"""build.sh's gate on attn_ref.sm_90a.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr)
a ptxas warning, a kernel compiled for another arch, a stack frame, a spill or local memory, any tensor-core instruction but
HMMA.16816.F32.BF16, and any floating-point instruction but that HMMA and the cvt's F2FP.BF16(.F32).PACK_AB.  The one
exception is HFMA2.MMA of -RZ and RZ, the compiler's way of loading the +0 accumulator: it adds nothing.  Each kernel must
hold at least one HMMA and one F2FP.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

ARCH = "sm_90a"
KERNELS = ("attn_ref_qk", "attn_ref_pv")
HMMA = "HMMA.16816.F32.BF16"
CVT = re.compile(r"^F2FP\.BF16(\.F32)?\.PACK_AB$")
MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
FLOAT = re.compile(r"\b(FADD|FMUL|FFMA|FMNMX|FSETP|FSET|FSEL|FCHK|FRND|FSWZADD|MUFU|F2F|F2I|I2F|DADD|DMUL|DFMA|DSETP|DMNMX|"
                   r"HADD2|HMUL2|HFMA2|HMNMX2|HSETP2|HSET2|F2FP)(\.[\w.]+)?\b")
CONSTANT = re.compile(r"\bHFMA2\.MMA R\d+, -RZ, RZ\b")


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    problems = [line.strip() for line in ptxas.splitlines() if re.search(r"warning|error|serializ", line, re.I)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for '(\w+)'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        if m.group(2) != ARCH:
            problems.append(f"{m.group(1)}: compiled for {m.group(2)}, want {ARCH}")
        body = m.group(3)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        regs = re.search(r"Used (\d+) registers", body)
        funcs[m.group(1)] = {"regs": int(regs.group(1)) if regs else None, "stack": int(frame.group(1)) if frame else None,
                             "spill_stores": int(frame.group(2)) if frame else None,
                             "spill_loads": int(frame.group(3)) if frame else None}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            funcs.setdefault(cur, {}).update({"HMMA": 0, "F2FP": 0, "local": 0, "other_mma": [], "float": []})
            continue
        if cur is None:
            continue
        r = funcs[cur]
        r["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
        for mm in MMA.finditer(line):
            if mm.group(0) == HMMA:
                r["HMMA"] += 1
            else:
                r["other_mma"].append(mm.group(0))
        for fl in FLOAT.finditer(line):
            if CVT.match(fl.group(0)):
                r["F2FP"] += 1
            elif not CONSTANT.search(line):
                r["float"].append(fl.group(0))
    for name in KERNELS:
        r = funcs.get(name)
        if r is None or "HMMA" not in r:
            problems.append(f"{name}: missing from the ptxas report or the SASS")
            continue
        if r.get("stack") is None:
            problems.append(f"{name}: no ptxas resource line")
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r["local"]:
            problems.append(f"{name}: local memory (stack {r.get('stack')}, spills {r.get('spill_stores')}/{r.get('spill_loads')}, "
                            f"{r['local']} LDL/STL)")
        if r["other_mma"]:
            problems.append(f"{name}: tensor-core forms other than {HMMA}: {sorted(set(r['other_mma']))}")
        if r["float"]:
            problems.append(f"{name}: floating-point instructions other than {HMMA} and the cvt: {sorted(set(r['float']))}")
        if not r["HMMA"]:
            problems.append(f"{name}: no {HMMA}")
        if not r["F2FP"]:
            problems.append(f"{name}: no F2FP.BF16 cvt")
    extra = sorted(set(funcs) - set(KERNELS))
    if extra:
        problems.append(f"kernels the gate doesn't know: {extra}")
    return funcs, problems


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(open(ptxas).read(), open(sass).read())
    with open(out, "w") as f:
        json.dump({"arch": ARCH, "kernels": funcs, "problems": problems}, f, indent=1)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("attn_ref.sm_90a.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs {funcs[n]['HMMA']} HMMA {funcs[n]['F2FP']} F2FP"
                                                for n in KERNELS) + "; no other floating-point instruction, no local memory")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
