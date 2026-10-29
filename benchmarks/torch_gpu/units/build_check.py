"""build.sh's check of one arch's units cubin (units.sm_90a.cubin or units.sm_120a.cubin), from ptxas's report and
cuobjdump's SASS.  The floating-point policy is the harness SASS gate's (benchmarks/pouw/harness/sass_gate.py, whose verdict
build.sh records beside this one, harness-gate.ARCH.json); this holds what is particular to the units.  Refused (exit 1, the
reasons on stderr):
- a ptxas warning, a kernel compiled for another arch, a kernel missing or one this check doesn't know (the arch's
  `KERNELS`: sm_120a's adds units_sweep_words);
- a stack frame, a spill, or local memory (LDL/STL);
- any tensor-core instruction;
- MUFU.EX2 outside units_softmax and the sweep kernels, or none in units_softmax;
- any MUFU but EX2, RCP and RSQ (RCP and RSQ are div.rn's and sqrt.rn's sequences, and the sm_120a sweep's bare RSQ);
- any floating-point instruction in the integer kernels (units_argmax, units_relu), but HFMA2 of -RZ, RZ and two
  immediates (HFMA2.MMA on sm_90a), the compiler's way of loading a constant, and the compiler's unsigned integer
  division (`DIVISION`), exact by its correction steps, which sm_120a's ptxas uses for units_relu's unrolled trip count.
Each kernel's registers and its floating-point opcodes with their counts are recorded. ptxas's report and the SASS listing
are read by s8_linear's sass_gate.py (`ptxas_report`, `sass_functions`).

  python3 build_check.py PTXAS_TXT SASS_TXT OUT_JSON [ARCH]   (ARCH sm_90a or sm_120a, default sm_90a)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s8_sass_gate", Path(__file__).resolve().parents[1] / "s8_linear" / "sass_gate.py")
SG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SG)

BASE = ("units_rmsnorm", "units_quant", "units_quant4", "units_softmax", "units_argmax", "units_relu", "units_add",
        "units_sweep")
#: each arch's kernels (lib.KERNELS and lib.ARCH_KERNELS)
KERNELS = {"sm_90a": BASE, "sm_120a": BASE + ("units_sweep_words",)}
EX2 = ("units_softmax", "units_sweep", "units_sweep_words")
INTEGER = ("units_argmax", "units_relu")
MUFU_OK = {"EX2", "RCP", "RSQ"}
FLOAT = re.compile(r"\b(FADD|FMUL|FFMA|FMNMX|FSETP|FSET|FSEL|FCHK|FRND|FSWZADD|MUFU|F2F|F2I|I2F|F2IP|I2FP|DADD|DMUL|DFMA|"
                   r"DSETP|DMNMX|HADD2|HMUL2|HFMA2|HMNMX2|HSETP2|HSET2|F2FP)(\.[\w.]+)?\b")
IMM = r"[-+]?[\d.]+(?:e[-+]?\d+)?"
CONSTANT = re.compile(rf"\bHFMA2(\.MMA)? R\d+, -RZ, RZ, {IMM}, {IMM}\s*;")
DIVISION = {"I2F.U32.RP", "I2F.U64.RP", "MUFU.RCP", "F2I.FTZ.U32.TRUNC.NTZ", "F2I.U64.TRUNC"}


def check(ptxas: str, sass: str, arch: str = "sm_90a") -> tuple[dict, list[str]]:
    kernels = KERNELS[arch]
    funcs, problems = SG.ptxas_report(ptxas, arch)
    ops: dict[str, Counter] = {}
    for cur, lines in SG.sass_functions(sass).items():
        r = funcs.setdefault(cur, {})
        r.update({"local": 0, "mma": []})
        ops[cur] = Counter()
        for line in lines:
            r["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
            r["mma"] += [mm.group(0) for mm in SG.MMA.finditer(line) if not CONSTANT.search(line)]
            for fl in FLOAT.finditer(line):
                if not CONSTANT.search(line):
                    ops[cur][fl.group(0)] += 1
    for name in kernels:
        r = funcs.get(name)
        if r is None or "local" not in r:
            problems.append(f"{name}: missing from the ptxas report or the SASS")
            continue
        r["float"] = dict(sorted(ops[name].items()))
        if r.get("stack") is None:
            problems.append(f"{name}: no ptxas resource line")
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r["local"]:
            problems.append(f"{name}: local memory (stack {r.get('stack')}, spills {r.get('spill_stores')}/{r.get('spill_loads')}, "
                            f"{r['local']} LDL/STL)")
        if r["mma"]:
            problems.append(f"{name}: tensor-core instructions {sorted(set(r['mma']))}")
        mufu = {op.split(".")[1] if "." in op else "" for op in r["float"] if op.startswith("MUFU")}
        if mufu - MUFU_OK:
            problems.append(f"{name}: MUFU forms other than {sorted(MUFU_OK)}: {sorted(mufu - MUFU_OK)}")
        if "EX2" in mufu and name not in EX2:
            problems.append(f"{name}: MUFU.EX2 outside {list(EX2)}")
        if name == "units_softmax" and "EX2" not in mufu:
            problems.append(f"{name}: no MUFU.EX2")
        other = {op: n for op, n in r["float"].items() if op not in DIVISION}
        if name in INTEGER and other:
            problems.append(f"{name}: floating-point instructions in an integer kernel: {other}")
    extra = sorted(set(funcs) - set(kernels))
    if extra:
        problems.append(f"kernels the check doesn't know: {extra}")
    return funcs, problems


def main(argv: list[str]) -> int:
    ptxas, sass, out = argv[1:4]
    arch = argv[4] if len(argv) > 4 else "sm_90a"
    if arch not in KERNELS:
        print(f"REFUSED: no units cubin for {arch} (only {sorted(KERNELS)})", file=sys.stderr)
        return 1
    funcs, problems = check(open(ptxas).read(), open(sass).read(), arch)
    with open(out, "w") as f:
        json.dump({"arch": arch, "kernels": funcs, "problems": problems}, f, indent=1)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"units.{arch}.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs" for n in KERNELS[arch]) + "; no local memory, "
          "MUFU.EX2 only in " + ", ".join(n for n in EX2 if n in KERNELS[arch]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
