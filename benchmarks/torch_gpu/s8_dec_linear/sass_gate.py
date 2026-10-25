"""build.sh's gate on a libp2stage.so: ptxas's report and cuobjdump's SASS of its kernels.  Refuses (exit 1, the reasons on
stderr) on a ptxas warning, a stack frame, a spill or any local-memory instruction in either kernel, or a kernel missing; records
each kernel's registers, static shared memory and instruction mix (the integer pipe's multiplies, the logic and shift ops of the
key expansion), which the co-residency budget beside a GEMM works from.

  python3 sass_gate.py ARCH PTXAS_TXT SASS_TXT OUT_JSON
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

KERNELS = ("p2st_keyed_kernel", "p2st_plain_kernel")
MIX = ("IMAD.WIDE.U32", "IMAD.WIDE", "IMAD.HI.U32", "IMAD.X", "IMAD.MOV", "IMAD", "IADD3", "LOP3", "SHF", "PRMT", "SHFL", "LDS", "STS",
       "LDG", "STG", "BAR", "WARPSYNC", "VOTE", "ATOMG", "RED", "DFMA", "LDL", "STL")


def short(mangled: str) -> str:
    return next((k for k in KERNELS if k in mangled), mangled)


def gate(arch: str, ptxas: str, sass: str) -> tuple[dict, list[str]]:
    problems = [line for line in ptxas.splitlines() if re.search(r"warning|error", line, re.IGNORECASE)]
    funcs: dict[str, dict] = {}
    for m in re.finditer(rf"Compiling entry function '(\w+)' for '{arch}'(.*?)(?=Compiling entry function|\Z)", ptxas, re.DOTALL):
        body = m.group(2)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        smem = re.search(r"(\d+) bytes smem", body)
        funcs[short(m.group(1))] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)),
                                    "stack": int(frame.group(1)), "spill_stores": int(frame.group(2)),
                                    "spill_loads": int(frame.group(3)), "static_smem": int(smem.group(1)) if smem else 0}
    cur, mix = None, {}
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = short(fm.group(1))
            mix[cur] = Counter()
            continue
        if cur is None:
            continue
        op = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if op:
            name = op.group(1)
            mix[cur][next((k for k in MIX if name == k or name.startswith(k + ".")), name.split(".")[0])] += 1
    for name in KERNELS:
        r = funcs.get(name)
        if r is None:
            problems.append(f"{name}: not in ptxas's report for {arch}")
            continue
        m = mix.get(name, Counter())
        r["sass"] = dict(sorted(m.items(), key=lambda kv: -kv[1]))
        r["instructions"] = sum(m.values())
        if r["stack"] or r["spill_stores"] or r["spill_loads"] or m["LDL"] or m["STL"]:
            problems.append(f"{name}: local memory (stack {r['stack']}, spills {r['spill_stores']}/{r['spill_loads']}, "
                            f"LDL {m['LDL']}, STL {m['STL']})")
        if not m:
            problems.append(f"{name}: no SASS")
    return funcs, problems


def main(argv: list[str]) -> int:
    arch, ptxas, sass, out = argv[1:5]
    funcs, problems = gate(arch, Path(ptxas).read_text(), Path(sass).read_text())
    Path(out).write_text(json.dumps({"arch": arch, "kernels": funcs, "problems": problems}, indent=1))
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"libp2stage ({arch}): " + ", ".join(
        f"{n} {funcs[n]['regs']} regs {funcs[n]['static_smem']} B smem {funcs[n]['instructions']} instructions" for n in KERNELS)
        + "; no local memory")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
