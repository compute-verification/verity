"""Opcode counts of every `fk` kernel in `cuobjdump -sass` output: the whole function and each loop (a backward BRA).

    python3 sass_loops.py filler.sass > sass-loops.json

Keys are "pop,xop,x,unr" from the mangled template arguments (filler.cu's Op numbering).
"""

import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"Function : (\S+)")
INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
ARGS = re.compile(r"_Z2fkILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE")


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
        a = ARGS.search(name)
        key = ",".join(a.groups()) if a else name
        loops = []
        for addr, op, rest in insts:
            if op == "BRA":
                t = re.search(r"0x([0-9a-f]+)", rest)
                if t and int(t.group(1), 16) < addr:
                    lo = int(t.group(1), 16)
                    body = Counter(o for x, o, _ in insts if lo <= x <= addr)
                    loops.append({"start": lo, "end": addr, "n": sum(body.values()), "ops": dict(body.most_common())})
        out[key] = {"total": dict(Counter(o for _, o, _ in insts).most_common()), "loops": loops}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
