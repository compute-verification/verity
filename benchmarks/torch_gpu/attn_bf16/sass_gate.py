"""build.sh's gate on one attn_bf16 cubin (sm_90a or sm_120a): ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the
reasons on stderr) on a ptxas warning, a kernel compiled for another arch, a stack frame, a spill or local memory, and on
any kernel whose floating-point instructions aren't the Definition's: every tensor-core instruction an HMMA.16816.F32.BF16
(D/16 of them in QK^T's unrolled k loop, 4 D/16 in _s16's, at least one in PV's and in the _ref kernels'), the cvt an
F2FP.BF16(.F32).PACK_AB, and no other floating-point instruction (an HFMA2.MMA of -RZ and RZ, the compiler's way of loading
a constant, adds nothing, and an FSEL whose operands carry no negation or absolute value moves a word unchanged).
ptxas's report and the SASS listing are read by s8_linear's sass_gate.py (`ptxas_report`, `sass_functions`).

  python3 sass_gate.py ARCH PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("s8_sass_gate", Path(__file__).resolve().parents[1] / "s8_linear" / "sass_gate.py")
SG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SG)

HMMA = "HMMA.16816.F32.BF16"
CVT = re.compile(r"^F2FP\.BF16(\.F32)?\.PACK_AB$")
FLOAT = re.compile(r"\b(FADD|FMUL|FFMA|FMNMX|FSETP|FSET|FSEL|FCHK|FRND|FSWZADD|MUFU|F2F|F2I|I2F|DADD|DMUL|DFMA|DSETP|DMNMX|"
                   r"HADD2|HMUL2|HFMA2|HMNMX2|HSETP2|HSET2|F2FP)(\.[\w.]+)?\b")
CONSTANT = re.compile(r"\bHFMA2\.MMA R\d+, -RZ, RZ\b")
SELECT = re.compile(r"\bFSEL R\d+, [^;|\-]*;")
#: kernel: (exact HMMA count, or None for at least one); _s16's QK^T unrolls its 4 key tiles, _ref's loops
WANT = {f"attn_{op}_d{d}{tile}": ((4 if tile == "_s16" else 1) * d // 16 if op == "qk" and tile != "_ref" else None)
        for tile in ("", "_h8", "_h16", "_s16", "_ref") for op in ("qk", "pv") for d in (64, 128)}


def gate(arch: str, ptxas: str, sass: str) -> tuple[dict, list[str]]:
    funcs, problems = SG.ptxas_report(ptxas, arch)
    for cur, lines in SG.sass_functions(sass).items():
        r = funcs.setdefault(cur, {})
        r.update({"HMMA": 0, "F2FP": 0, "LDL": 0, "STL": 0, "other_mma": [], "float": []})
        for line in lines:
            for op in ("LDL", "STL"):
                r[op] += bool(re.search(r"\b" + op + r"\b", line))
            for mm in SG.MMA.finditer(line):
                if mm.group(0) == HMMA:
                    r["HMMA"] += 1
                else:
                    r["other_mma"].append(mm.group(0))
            for fl in FLOAT.finditer(line):
                if CVT.match(fl.group(0)):
                    r["F2FP"] += 1
                elif not CONSTANT.search(line) and not (fl.group(1) == "FSEL" and SELECT.search(line)):
                    r["float"].append(fl.group(0))
    for name, r in funcs.items():
        if r.get("regs") is None or r.get("stack") is None:
            problems.append(f"{name}: no ptxas resource line")
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than {HMMA}: {sorted(set(r['other_mma']))}")
        if r.get("float"):
            problems.append(f"{name}: floating-point instructions other than {HMMA} and {CVT.pattern}: {sorted(set(r['float']))}")
        want = WANT.get(name, 0)
        if (r.get("HMMA", 0) != want) if want is not None else not r.get("HMMA"):
            problems.append(f"{name}: {r.get('HMMA')} {HMMA}, want {want if want is not None else 'at least one'}")
        if name in WANT and not r.get("F2FP"):
            problems.append(f"{name}: no {CVT.pattern}")
    missing = sorted(set(WANT) - set(funcs))
    if missing:
        problems.append(f"kernels missing from the cubin: {missing}")
    return funcs, problems


def main(argv: list[str]) -> int:
    arch, ptxas, sass, out = argv[1:5]
    funcs, problems = gate(arch, open(ptxas).read(), open(sass).read())
    json.dump({"arch": arch, "want_hmma": WANT, "kernels": funcs, "problems": problems}, open(out, "w"), indent=1)
    if problems:
        print(f"REFUSED ({arch}): " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"attn_bf16.{arch}.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs {funcs[n]['HMMA']} HMMA {funcs[n]['F2FP']} F2FP" for n in WANT)
          + "; no other floating-point instruction, no local memory")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
