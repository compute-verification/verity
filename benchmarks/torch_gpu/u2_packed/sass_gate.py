"""run.sh's gate on packed.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a ptxas
warning or serialized wgmma, a stack frame, a spill or local memory in any kernel, on a GEMM kernel (pk_*) whose
tensor-core instructions aren't the atom alone, IGMMA.64x256x32 on s8 (u8 for p2u) operands, in a multiple of 4 (a
128-deep k-block), or none at all where it should multiply (none in p2x and p2ux), or that issues no TMA load, and on a
GEMM kernel over 168 registers or hash_k over 80 (so a hash_k CTA fits beside it).  Records each kernel's opcode counts,
from which the unpacking's instructions are read.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
#: GEMM kernel -> the operand type of its atoms (None: no MMAs)
GEMMS = {"pk_i8": "S8", "pk_p2": "S8", "pk_p2u": "U8", "pk_res": "S8", "pk_p2x": None, "pk_p2ux": None}
WANT = set(GEMMS) | {"pk_layout", "hash_k", "stamp_k", "wait_k"}
REG_CAP = {**{g: 168 for g in GEMMS}, "hash_k": 80}
OPCODE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[\w.]+)?")


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    problems = [line for line in ptxas.splitlines() if re.search(r"warning|error|C751\d|serializ", line, re.I)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for 'sm_90a'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        body = m.group(2)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        funcs[m.group(1)] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)), "stack": int(frame.group(1)),
                             "spill_stores": int(frame.group(2)), "spill_loads": int(frame.group(3))}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            funcs.setdefault(cur, {}).update({"LDL": 0, "STL": 0, "UTMALDG": 0, "IGMMA": 0, "forms": [], "other_mma": [],
                                              "ops": Counter()})
            continue
        if not cur:
            continue
        r = funcs[cur]
        om = OPCODE.search(line)
        if om:
            r["ops"][om.group(1)] += 1
        r["LDL"] += len(re.findall(r"\bLDL\b", line))
        r["STL"] += len(re.findall(r"\bSTL\b", line))
        r["UTMALDG"] += len(re.findall(r"\bUTMALDG\b", line))
        for mm in MMA.finditer(line):
            form = mm.group(0)
            want = GEMMS.get(cur)
            if want and form.startswith("IGMMA.64x256x32") and f"{want}.{want}" in form:
                r["IGMMA"] += 1
                if form not in r["forms"]:
                    r["forms"].append(form)
            else:
                r["other_mma"].append(form)
    for name, r in funcs.items():
        if max(r.get("stack", 0), r.get("spill_stores", 0), r.get("spill_loads", 0)) or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than its atom: {sorted(set(r['other_mma']))}")
        if name in GEMMS:
            atoms = r.get("IGMMA", 0)
            if (GEMMS[name] and (atoms < 4 or atoms % 4)) or (not GEMMS[name] and atoms) or not r.get("UTMALDG"):
                problems.append(f"{name}: {atoms} atoms (want {'a multiple of 4' if GEMMS[name] else 'none'}), "
                                f"{r.get('UTMALDG')} TMA loads")
        if name in REG_CAP and r.get("regs", 0) > REG_CAP[name]:
            problems.append(f"{name}: {r['regs']} registers, over {REG_CAP[name]}")
    missing = sorted(WANT - set(funcs))
    if missing:
        problems.append(f"kernels missing from the cubin: {missing}")
    return funcs, problems


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    funcs, problems = gate(open(ptxas).read(), open(sass).read())
    json.dump({"kernels": funcs, "problems": problems}, open(out, "w"), indent=1)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("packed.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs" for n in sorted(funcs) if n in WANT and "regs" in funcs[n]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
