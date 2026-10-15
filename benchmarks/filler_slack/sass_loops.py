"""Opcode counts of every `fk` and `pk` kernel in `cuobjdump -sass` output: the whole function and each loop (a backward
BRA), and for each loop its instruction shapes and vector-register reads.

    python3 sass_loops.py filler.sass > sass-loops.json

Keys are "pop,xop,x,unr,pb" for `fk` and "pk:pop,pb" for `pk`, from the mangled template arguments (filler.cu's Op
numbering). A shape is the instruction with register numbers dropped (`IADD3 R, PT, PT, R, R.reuse, RZ`). A loop's
`vreads` counts its source operands read from the vector register file: every `R<n>` source (not RZ, not the
destination), less those the reuse cache serves (the previous instruction flagged the same register `.reuse` in the
same operand slot). `ureads` and `creads` count uniform-register (`UR<n>`, not URZ) and constant-bank (`c[..][..]`)
sources.
"""

import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"Function : (\S+)")
INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
ARGS = re.compile(r"_Z2fkILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE")
PARGS = re.compile(r"_Z2pkILi(\d+)ELi(\d+)EE")
VREG = re.compile(r"^[-~|!]*(R\d+)(\.reuse)?")
# Opcodes whose first operand is a source, not a destination.
NODEST = ("BRA", "NANOSLEEP", "STG", "STS", "ST", "RED", "EXIT", "BAR", "WARPSYNC", "BSYNC", "BSSY", "NOP", "RET")


def shape(rest: str) -> str:
    s = re.sub(r"\bUR\d+\b", "UR", rest)
    s = re.sub(r"\bR\d+\b", "R", s)
    s = re.sub(r"\bUP\d\b", "UP", s)
    s = re.sub(r"\bP\d\b", "P", s)
    s = re.sub(r"c\[(0x[0-9a-f]+)\]\[0x[0-9a-f]+\]", r"c[\1][.]", s)
    return re.sub(r"(?<![\[\w])-?0x[0-9a-f]+", "imm", s)


def reads(body: list[tuple[str, str]]) -> dict[str, int]:
    """Vector, reuse-cache, uniform and constant source reads over the loop body, in program order."""
    v = cached = u = c = 0
    prev: dict[int, str] = {}
    for op, rest in body:
        ops = [o.strip() for o in rest.split(",")] if rest else []
        srcs = ops if op.split(".")[0] in NODEST else ops[1:]
        cur: dict[int, str] = {}
        for i, o in enumerate(srcs):
            m = VREG.match(o)
            if m:
                if prev.get(i) == m.group(1):
                    cached += 1
                else:
                    v += 1
                if m.group(2):
                    cur[i] = m.group(1)
            elif re.match(r"^[-~|!]*UR\d+", o):
                u += 1
            elif "c[" in o:
                c += 1
        prev = cur
    return {"vreads": v, "reuse_hits": cached, "ureads": u, "creads": c}


def main() -> int:
    funcs: dict[str, list[tuple[int, str, str]]] = {}
    cur = None
    for line in open(sys.argv[1]):
        m = FUNC.search(line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        m = INST.search(line)
        if m and cur is not None:
            toks = m.group(2).split()
            if toks and toks[0].startswith("@"):
                toks = toks[1:]
            if toks:
                cur.append((int(m.group(1), 16), toks[0], " ".join(toks[1:])))
    out = {}
    for name, insts in funcs.items():
        a, p = ARGS.search(name), PARGS.search(name)
        key = ",".join(a.groups()) if a else "pk:" + ",".join(p.groups()) if p else name
        loops = []
        for addr, op, rest in insts:
            if op.split(".")[0] == "BRA":  # the filler's own loop closes with a uniform `BRA.U UP0`
                t = re.search(r"0x([0-9a-f]+)", rest)
                if t and int(t.group(1), 16) < addr:
                    lo = int(t.group(1), 16)
                    body = [(o, r) for x, o, r in insts if lo <= x <= addr]
                    loops.append({"start": lo, "end": addr, "n": len(body),
                                  "ops": dict(Counter(o for o, _ in body).most_common()),
                                  "shapes": dict(Counter(f"{o} {shape(r)}" for o, r in body).most_common()),
                                  **reads(body)})
        out[key] = {"total": dict(Counter(o for _, o, _ in insts).most_common()), "loops": loops}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
