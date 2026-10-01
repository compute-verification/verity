"""pouw_hash_bench's SASS, per kernel: ptxas's registers and local memory, and the opcode census.

    python sass.py PTXAS.txt SASS.txt OUT.json     exit 1 (REFUSED) on a gate

Gates: no stack frame, spill or local-memory instruction in any timed kernel (the check kernels may use a stack); each
`k_op<OP>` holds at least OP_PER_ITER of its opcode; `k_mma` at least MMA_PER_ITER FP8 mma.sync (QMMA); `k_epi` holds 16 FP8
and 64 BF16 mma.sync per tile. IADD3 is split in two:
`IADD3.RZ` (a zero third source: what add.u32 becomes) and `IADD3` (three sources), since Blackwell's doubled integer rate is
said to cover the first and not the second.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

OP_PER_ITER = 128
MMA_PER_ITER = 16
OPS = ["IADD3.RZ", "IADD3", "LOP3", "SHF", "PRMT", "IMAD", "IDP", "FFMA",   # k_op<0 … 9>, as pouw_hash_bench.cu's enum
       ("LOP3", "IMAD"), ("LOP3", "SHF")]


def opcode(line: str) -> str | None:
    m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;", line)
    if not m:
        return None
    op, args = m.group(2), m.group(3)
    base = op.split(".")[0]
    if base == "IADD3" and args.split(",")[-1].strip() in ("RZ", "RZ.reuse"):
        return "IADD3.RZ"
    return base


def ptxas(text: str) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r"Compiling entry function '(\w+)' for '(\w+)'(.*?)(?=Compiling entry function|\Z)", text, re.S):
        body = m.group(3)
        regs = re.search(r"Used (\d+) registers", body)
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        smem = re.search(r"(\d+) bytes smem", body)
        out[m.group(1)] = {"regs": int(regs.group(1)) if regs else None,
                           "stack": int(frame.group(1)) if frame else 0,
                           "spill_stores": int(frame.group(2)) if frame else 0,
                           "spill_loads": int(frame.group(3)) if frame else 0,
                           "static_smem": int(smem.group(1)) if smem else 0}
    return out


def census(text: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    cur = None
    for line in text.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = out.setdefault(fm.group(1), Counter())
        elif cur is not None and (op := opcode(line)):
            cur[op] += 1
    return out


def gate(kernels: dict[str, dict]) -> list[str]:
    problems = []
    for name, k in kernels.items():
        ops = k["ops"]
        if not name.startswith("_Z7k_check") and not name.startswith("_Z12k_check") and \
                (k.get("stack") or k.get("spill_stores") or k.get("spill_loads") or ops.get("LDL") or ops.get("STL")):
            problems.append(f"{name}: local memory (stack {k.get('stack')}, LDL {ops.get('LDL', 0)}, STL {ops.get('STL', 0)})")
        if m := re.search(r"k_opILi(\d+)E", name):
            want = OPS[int(m.group(1))]
            for op in (want,) if isinstance(want, str) else want:
                n = OP_PER_ITER // (1 if isinstance(want, str) else 2)
                if ops.get(op, 0) < n:
                    problems.append(f"{name}: {ops.get(op, 0)} {op}, want at least {n}")
        if "k_epiILi" in name and (ops.get("QMMA", 0) < 16 or ops.get("HMMA", 0) < 64):
            problems.append(f"{name}: {ops.get('QMMA', 0)} QMMA and {ops.get('HMMA', 0)} HMMA, want 16 and 64 per tile")
        if name.startswith("_Z5k_mma") and ops.get("QMMA", 0) < MMA_PER_ITER:
            problems.append(f"{name}: {ops.get('QMMA', 0)} QMMA, want at least {MMA_PER_ITER}")
    return problems


def main(argv: list[str]) -> int:
    regs = ptxas(open(argv[0]).read())
    ops = census(open(argv[1]).read())
    kernels = {n: {**regs.get(n, {}), "ops": dict(sorted(c.items(), key=lambda kv: -kv[1])), "total": sum(c.values())}
               for n, c in ops.items()}
    problems = gate(kernels)
    json.dump({"kernels": kernels, "problems": problems}, open(argv[2], "w"), indent=1)
    for n, k in sorted(kernels.items()):
        top = ", ".join(f"{op} {c}" for op, c in list(k["ops"].items())[:7])
        print(f"{n}: {k.get('regs')} regs, {k['total']} instructions: {top}")
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
