"""run.sh's gate on u2_gemm.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a ptxas
warning or serialized wgmma, a stack frame, a spill or local memory in any kernel, on a GEMM kernel (u2mm_*) whose
tensor-core instructions aren't the s8 atom alone, IGMMA.64x256x32 on s8 operands, in a multiple of 4 (a 128-deep stage),
or that issues no TMA load, and on ci1 or ci2 (ci2l) over 168 registers or hash_k over 80 (so a hash_k CTA fits beside
them).

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
WANT = ({f"u2mm_{s}_m{c}" for s in ("ws", "ci1", "ci2", "ci3") for c in (1, 2)}
        | {"u2mm_ci2l_m2", "u2mm_ci3l_m1", "u2mm_ci3l_m2", "hash_k", "stamp_k", "wait_k"})
REG_CAP = {"u2mm_ci1_m1": 168, "u2mm_ci1_m2": 168, "u2mm_ci2_m1": 168, "u2mm_ci2_m2": 168, "u2mm_ci2l_m2": 168, "hash_k": 80}


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
            funcs.setdefault(cur, {}).update({"LDL": 0, "STL": 0, "UTMALDG": 0, "IGMMA": 0, "MEMBAR_GPU": 0, "CCTL_IVALL": 0,
                                              "forms": [], "other_mma": []})
            continue
        if not cur:
            continue
        r = funcs[cur]
        r["LDL"] += len(re.findall(r"\bLDL\b", line))
        r["STL"] += len(re.findall(r"\bSTL\b", line))
        r["UTMALDG"] += len(re.findall(r"\bUTMALDG\b", line))
        r["MEMBAR_GPU"] += len(re.findall(r"\bMEMBAR\.\w*\.?GPU\b", line))
        r["CCTL_IVALL"] += len(re.findall(r"\bCCTL\.IVALL\b", line))
        for mm in MMA.finditer(line):
            form = mm.group(0)
            if form.startswith("IGMMA.64x256x32") and "U8" not in form:
                r["IGMMA"] += 1
                if form not in r["forms"]:
                    r["forms"].append(form)
            else:
                r["other_mma"].append(form)
    for name, r in funcs.items():
        if max(r.get("stack", 0), r.get("spill_stores", 0), r.get("spill_loads", 0)) or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than the s8 atom: {sorted(set(r['other_mma']))}")
        if name.startswith("u2mm_") and name != "u2mm_layout":
            if r.get("IGMMA", 0) < 4 or r["IGMMA"] % 4 or not r.get("UTMALDG"):
                problems.append(f"{name}: {r.get('IGMMA', 0)} s8 atoms (want a multiple of 4), {r.get('UTMALDG')} TMA loads")
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
    print("u2_gemm.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs" for n in sorted(funcs) if n in WANT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
