#!/usr/bin/env python3
"""pearl_c4.cubin's offline gate, from ptxas's -v log and cuobjdump -sass: no warning, spill, stack frame or local-memory
access in any kernel, no FP32 FADD, FFMA or FMUL with .FTZ (it breaks the scalar model on subnormals; FP32 division and
square root's slow paths hold them), and the tensor-core and cast instructions each kernel is meant to issue.

  python3 sass.py PTXAS_LOG SASS OUT_JSON      exit 1 (with the reasons) when a kernel fails

Per kernel: OMMA (block-scaled FP4 m16n8k64: `.UE4M3.4X` for NVFP4, `.E8` for MXFP4, never the other format's), HMMA
(BF16 m16n8k16), the E2M1 casts (F2FP ... E2M1), LDL/STL.  The counts are static instructions, so an unrolled loop counts
each copy: a GEMM's mainloop issues 32 OMMA per 128-deep block per copy (a warp's two m16 x n64 regions, two k64 steps),
the peel 32 HMMA per region, forming 8 OMMA (the noise atoms of one k64 chunk) and 16 E2M1 casts per copy.
"""

from __future__ import annotations

import json
import re
import sys

OPS = {"OMMA_NV": r"OMMA\.SF\.16864\.F32\.E2M1\.E2M1\.UE4M3\.4X", "OMMA_MX": r"OMMA\.SF\.16864\.F32\.E2M1\.E2M1\.E8\b",
       "HMMA": r"HMMA\.16816\.F32\.BF16", "E2M1": r"F2FP\.SATFINITE\.E2M1", "LDL": r"\bLDL\b", "STL": r"\bSTL\b",
       "FTZ": r"\bF(ADD|FMA|MUL)(\.\w+)*\.FTZ\b"}


def census(ptxas: str, sass: str) -> dict[str, dict]:
    funcs: dict[str, dict] = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for 'sm_120a'(.*?)(?=Compiling entry function|\Z)", ptxas, re.S):
        body = m.group(2)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        funcs[m.group(1)] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)), "stack": int(frame.group(1)),
                             "spill_stores": int(frame.group(2)), "spill_loads": int(frame.group(3))}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            funcs.setdefault(cur, {}).update({op: 0 for op in OPS})
        elif cur:
            for op, pat in OPS.items():
                if re.search(pat, line):
                    funcs[cur][op] += 1
    return funcs


def want(name: str) -> dict[str, tuple[int, int]] | None:
    """(multiple, minimum) per counted op, or None for a kernel with no tensor-core or cast work to hold."""
    fmt = "NV" if "_nv" in name else "MX"
    other = "MX" if fmt == "NV" else "NV"
    if name.startswith("gemm_") and not name.endswith("_smem"):
        peel = (0, 0) if "_plain" in name else (32, 32)
        return {f"OMMA_{fmt}": (32, 32), f"OMMA_{other}": (0, 0), "HMMA": peel, "E2M1": (0, 0)}
    if name.startswith("chain32_") and not name.endswith("_smem"):
        return {f"OMMA_{fmt}": (8, 8), f"OMMA_{other}": (0, 0), "HMMA": (0, 0), "E2M1": (0, 0)}
    if name.startswith("form_") and not name.endswith("_smem"):
        return {"OMMA_NV": (8, 8), "OMMA_MX": (0, 0), "HMMA": (0, 0), "E2M1": (16, 16)}
    return None


def problems(ptxas: str, funcs: dict[str, dict]) -> list[str]:
    out = [ln for ln in ptxas.splitlines() if re.search(r"warning|error|C7514|serializ", ln, re.I)]
    for name, r in funcs.items():
        if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r.get("LDL") or r.get("STL"):
            out.append(f"{name}: local memory {r}")
        if r.get("FTZ"):
            out.append(f"{name}: {r['FTZ']} FP32 FADD, FFMA or FMUL with .FTZ (server.md 09:34Z: none in a shipped kernel)")
        for op, (mult, least) in (want(name) or {}).items():
            n = r.get(op, 0)
            if (mult == 0 and n) or (mult and (n < least or n % mult)):
                out.append(f"{name}: {n} {op} (want " + (f"a multiple of {mult}, at least {least})" if mult else "none)"))
    expected = {f"gemm_{f}_{m}{h}" for f in ("nv", "mx") for m in ("hash", "nohash", "plain") for h in ("", "_m64", "_m64n64")}
    expected |= {f"{k}_{f}{r}" for k, r in (("form", ""), ("chain32", ""), ("chain32", "_r2")) for f in ("nv", "mx")}
    expected |= {"lines", "transpose", "commit_rows", "pb", "leaves", "form_nv_v0", "form_nv_v1", "form_nv_smem", "form_mx_smem"}
    out += [f"{name}: missing" for name in sorted(expected - set(funcs))]
    return out


def main(argv: list[str]) -> int:
    ptxas, sass = open(argv[0]).read(), open(argv[1]).read()
    funcs = census(ptxas, sass)
    bad = problems(ptxas, funcs)
    with open(argv[2], "w") as f:
        json.dump({"kernels": funcs, "problems": bad}, f, indent=1)
    if bad:
        print("REFUSED: " + "; ".join(bad), file=sys.stderr)
        return 1
    g = sorted((n, r) for n, r in funcs.items() if n.startswith("gemm_") and not n.endswith("_smem"))
    per = [f"{n} {r['regs']} regs {r['OMMA_NV'] + r['OMMA_MX']} OMMA {r['HMMA']} HMMA" for n, r in g]
    print(f"pearl_c4.cubin: {len(funcs)} kernels, no local memory; " + ", ".join(per))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
