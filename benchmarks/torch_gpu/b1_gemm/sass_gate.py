"""run.sh's gate on b1.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a ptxas
warning or serialized wgmma, a stack frame, a spill or local memory in any kernel, on a probe kernel (b1_res*) whose
tensor-core instructions aren't the b1 atom alone, BGMMA.64x256x256.AND.POPC, in a multiple of 4 (a 128-byte-deep stage),
and on a probe kernel over 168 registers or hash_k over 80 (so two hash_k CTAs fit beside b1_res).

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

MMA = re.compile(r"\b([QHOIUB]MMA|[QHIB]GMMA|UTC\w*MMA)(\.[\w.]+)?")
ATOM = "BGMMA.64x256x256.AND.POPC"
PROBES = ("b1_res", "b1_res_w4", "b1_res_w8")
WANT = set(PROBES) | {"hash_k", "stamp_k", "wait_k", "b1_layout"}
REG_CAP = {**dict.fromkeys(PROBES, 168), "hash_k": 80}


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
            funcs.setdefault(cur, {}).update({"LDL": 0, "STL": 0, "BGMMA": 0, "other_mma": []})
            continue
        if not cur:
            continue
        r = funcs[cur]
        r["LDL"] += len(re.findall(r"\bLDL\b", line))
        r["STL"] += len(re.findall(r"\bSTL\b", line))
        for mm in MMA.finditer(line):
            if mm.group(0) == ATOM:
                r["BGMMA"] += 1
            else:
                r["other_mma"].append(mm.group(0))
    for name, r in funcs.items():
        if max(r.get("stack", 0), r.get("spill_stores", 0), r.get("spill_loads", 0)) or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than the b1 atom: {sorted(set(r['other_mma']))}")
        if name in PROBES and (r.get("BGMMA", 0) < 4 or r["BGMMA"] % 4):
            problems.append(f"{name}: {r.get('BGMMA', 0)} b1 atoms (want a multiple of 4)")
        if name not in PROBES and r.get("BGMMA"):
            problems.append(f"{name}: {r['BGMMA']} b1 atoms outside a probe kernel")
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
    print("b1.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs, {funcs[n].get('BGMMA', 0)} BGMMA"
                                   for n in sorted(funcs) if n in WANT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
