"""build.sh's gate on s8_wgmma.cubin: ptxas's report and cuobjdump's SASS.  Refuses (exit 1, the reasons on stderr) on a ptxas
warning or serialized wgmma, a stack frame, a spill or local memory, and on any GEMM kernel whose tensor-core instructions
aren't its s8 atom alone, IGMMA.64xBNx32 on s8 operands at the kernel's BN, in a multiple of 4 (a 128-deep stage), or whose
epilogue isn't the Definition's instructions: an s32-to-f32 conversion, FMUL and an F2FP to BF16 present, and no FFMA (a
fused multiply-add would round once) and no .FTZ anywhere but integer division's (`INT_DIV`).  A reduction kernel
(s8mr*_128xBN_rR) may also run the reduction's atom, IGMMA.64xRx32 on s8 operands, BN / 8 of them a tile.  Each kernel's
record holds its highest register (`max_reg`): a consumer warpgroup's budget after setmaxnreg, which ptxas's count doesn't show.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import json
import re
import sys

MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
KERNEL = re.compile(r"s8mm2?_(64|128)x(64|128|256)$")
REDUCE = re.compile(r"s8mrb?_(128)x(128|256)_r(8|16|32|64|128)$")
#: the reduction kernels s8_wgmma.cu doesn't build (wgmma.py's UNBUILT)
UNBUILT = {"s8mr_128x256_r8", "s8mr_128x256_r16"}
REG = re.compile(r"\bR(\d+)\b")
#: the compiler's unsigned integer division (the raster's, by a runtime group and tile count) truncates a reciprocal it
#: computed from the divisor with this F2I; it never sees a value, so it is counted apart from every other .FTZ
INT_DIV = r"\bF2I\.FTZ\.U32\.TRUNC\.NTZ\b"
COUNTS = {"LDL": r"\bLDL\b", "STL": r"\bSTL\b", "FFMA": r"\bFFMA\b", "INT_DIV": INT_DIV, "FMUL": r"\bFMUL\b",
          "I2F": r"\bI2FP?\.F32\.S32\b", "F2FP_BF16": r"\bF2FP\.BF16\S*", "UTMALDG": r"\bUTMALDG\b"}


def atom_ok(form: str, bn: int) -> bool:
    """IGMMA.64x<bn>x32 with both operands s8 (cuobjdump spells the types after the shape; an unsigned operand reads U8)."""
    return form.startswith(f"IGMMA.64x{bn}x32") and "U8" not in form


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
            funcs.setdefault(cur, {}).update({k: 0 for k in COUNTS} | {"FTZ": 0, "IGMMA": 0, "forms": [], "other_mma": [],
                                                                       "max_reg": 0})
            continue
        if not cur:
            continue
        for k, pat in COUNTS.items():
            funcs[cur][k] += len(re.findall(pat, line))
        funcs[cur]["FTZ"] += len(re.findall(r"\.FTZ\b", line)) - len(re.findall(INT_DIV, line))
        funcs[cur]["max_reg"] = max([funcs[cur]["max_reg"]] + [int(v) + 1 for v in REG.findall(line)])
        km, rm = KERNEL.match(cur), REDUCE.match(cur)
        widths = (int(km.group(2)),) if km else (int(rm.group(2)), int(rm.group(3))) if rm else ()
        for mm in MMA.finditer(line):
            form = mm.group(0)
            if any(atom_ok(form, n) for n in widths):
                funcs[cur]["IGMMA"] += 1
                if form not in funcs[cur]["forms"]:
                    funcs[cur]["forms"].append(form)
            else:
                funcs[cur]["other_mma"].append(form)
    want = {f"s8mm{c}_{tm}x{bn}" for c in ("", "2") for tm in (64, 128) for bn in (64, 128, 256)}
    reduce = {f"{f}_128x{bn}_r{r}" for f in ("s8mr", "s8mrb") for bn in (128, 256) for r in (8, 16, 32, 64, 128)} - UNBUILT
    for name, r in funcs.items():
        if max(r.get("stack", 0), r.get("spill_stores", 0), r.get("spill_loads", 0)) or r.get("LDL") or r.get("STL"):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than its s8 atom: {sorted(set(r['other_mma']))}")
        if r.get("FFMA") or r.get("FTZ"):
            problems.append(f"{name}: {r.get('FFMA')} FFMA, {r.get('FTZ')} .FTZ")
        if name in want | reduce:
            least = 4 + (int(REDUCE.match(name).group(2)) // 8 if name in reduce else 0)
            if r.get("IGMMA", 0) < least or r["IGMMA"] % 4:
                problems.append(f"{name}: {r.get('IGMMA', 0)} s8 atoms (want a multiple of 4, at least {least})")
            if not (r.get("I2F") and r.get("FMUL") and r.get("F2FP_BF16") and r.get("UTMALDG")):
                problems.append(f"{name}: epilogue or loads missing: {({k: r.get(k) for k in ('I2F', 'FMUL', 'F2FP_BF16', 'UTMALDG')})}")
    missing = sorted((want | reduce) - set(funcs))
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
    print("s8_wgmma.cubin: " + ", ".join(f"{n} {funcs[n]['regs']} regs (max R{funcs[n]['max_reg'] - 1}) {funcs[n]['IGMMA']} "
                                         f"{funcs[n]['forms']}" for n in sorted(funcs) if KERNEL.match(n) or REDUCE.match(n)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
