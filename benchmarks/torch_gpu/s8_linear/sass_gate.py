"""build.sh's gate on s8_wgmma.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a ptxas
warning or serialized wgmma, a stack frame, a spill or local memory, and on any GEMM kernel whose tensor-core instructions
aren't its s8 atom alone, IGMMA.64xBNx32 on s8 operands at the kernel's BN, in a multiple of 4 (a 128-deep stage), or whose
epilogue isn't the Definition's instructions: an s32-to-f32 conversion, FMUL and an F2FP to BF16 present, and no FFMA (a
fused multiply-add would round once) and no .FTZ anywhere but integer division's (`INT_DIV`).

`ptxas_report` and `sass_functions` read ptxas's report and cuobjdump's SASS for torch_gpu's other gates too (attn_bf16's
sass_gate.py, units' build_check.py); this file stays self-contained, since j10r_sm90's job ships it alone.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
KERNEL = re.compile(r"s8mm2?_(64|128)x(64|128|256)$")
#: the compiler's unsigned integer division (the raster's, by a runtime group and tile count) truncates a reciprocal it
#: computed from the divisor with this F2I; it never sees a value, so it is counted apart from every other .FTZ
INT_DIV = r"\bF2I\.FTZ\.U32\.TRUNC\.NTZ\b"
COUNTS = {"LDL": r"\bLDL\b", "STL": r"\bSTL\b", "FFMA": r"\bFFMA\b", "INT_DIV": INT_DIV, "FMUL": r"\bFMUL\b",
          "I2F": r"\bI2FP?\.F32\.S32\b", "F2FP_BF16": r"\bF2FP\.BF16\S*", "UTMALDG": r"\bUTMALDG\b"}


def ptxas_report(ptxas: str, arch: str) -> tuple[dict[str, dict], list[str]]:
    """ptxas -v's report: each entry function's registers, stack frame and spills (None where the report gives none), and
    what it refuses: a line with a warning, an error or a serialized wgmma (C751x), and an entry compiled for another arch."""
    problems = [line for line in ptxas.splitlines() if re.search(r"warning|error|C751\d|serializ", line, re.I)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for '(\w+)'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        if m.group(2) != arch:
            problems.append(f"{m.group(1)}: compiled for {m.group(2)}, want {arch}")
        regs = re.search(r"Used (\d+) registers", m.group(3))
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", m.group(3))
        funcs[m.group(1)] = {"regs": int(regs.group(1)) if regs else None, "stack": int(frame.group(1)) if frame else None,
                             "spill_stores": int(frame.group(2)) if frame else None,
                             "spill_loads": int(frame.group(3)) if frame else None}
    return funcs, problems


def sass_functions(sass: str) -> dict[str, list[str]]:
    """cuobjdump -sass's listing: each function's lines after its `Function :` header, in the listing's order (a header
    seen again starts that function's lines over)."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def atom_ok(form: str, bn: int) -> bool:
    """IGMMA.64x<bn>x32 with both operands s8 (cuobjdump spells the types after the shape; an unsigned operand reads U8)."""
    return form.startswith(f"IGMMA.64x{bn}x32") and "U8" not in form


def gate(ptxas: str, sass: str) -> tuple[dict, list[str]]:
    funcs, problems = ptxas_report(ptxas, "sm_90a")
    for cur, lines in sass_functions(sass).items():
        r = funcs.setdefault(cur, {})
        r.update({k: 0 for k in COUNTS} | {"FTZ": 0, "IGMMA": 0, "forms": [], "other_mma": []})
        km = KERNEL.match(cur)
        for line in lines:
            for k, pat in COUNTS.items():
                r[k] += len(re.findall(pat, line))
            r["FTZ"] += len(re.findall(r"\.FTZ\b", line)) - len(re.findall(INT_DIV, line))
            for mm in MMA.finditer(line):
                form = mm.group(0)
                if km and atom_ok(form, int(km.group(2))):
                    r["IGMMA"] += 1
                    if form not in r["forms"]:
                        r["forms"].append(form)
                else:
                    r["other_mma"].append(form)
    want = {f"s8mm{c}_{tm}x{bn}" for c in ("", "2") for tm in (64, 128) for bn in (64, 128, 256)}
    for name, r in funcs.items():
        if "regs" in r and (r["regs"] is None or r["stack"] is None):
            problems.append(f"{name}: no ptxas resource line")
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than its s8 atom: {sorted(set(r['other_mma']))}")
        if r.get("FFMA") or r.get("FTZ"):
            problems.append(f"{name}: {r.get('FFMA')} FFMA, {r.get('FTZ')} .FTZ")
        if name in want:
            if r.get("IGMMA", 0) < 4 or r["IGMMA"] % 4:
                problems.append(f"{name}: {r.get('IGMMA', 0)} s8 atoms (want a multiple of 4)")
            if not (r.get("I2F") and r.get("FMUL") and r.get("F2FP_BF16") and r.get("UTMALDG")):
                problems.append(f"{name}: epilogue or loads missing: {({k: r.get(k) for k in ('I2F', 'FMUL', 'F2FP_BF16', 'UTMALDG')})}")
    missing = sorted(want - set(funcs))
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
    print("s8_wgmma.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs {funcs[n]['IGMMA']} {funcs[n]['forms']}"
                                         for n in sorted(funcs) if KERNEL.match(n)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
