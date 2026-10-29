"""SASS gate for sw_gpu's kernels: every loop's opcodes, pinned.

    python sass_loops.py SASS.txt [PINS.json]   one JSON line a kernel: its loops (backward branches), each loop's opcode
                                                histogram, and the verdict; exit 1 on a spill (LDL/STL) or a pinned count
                                                that differs

A pin is {"kernel substring": {"loop": i, "ops": {"OPCODE": n, ...}}}: loop i (innermost first, by body length) must hold
exactly n of each pinned opcode (the major opcode, before the first dot). The unit prices come from these counts, so a
build whose loop changed fails here instead of pricing the wrong code.
"""

import collections
import json
import re
import sys

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def kernels(text):
    for chunk in re.split(r"\n\s+Function : ", text)[1:]:
        name = chunk.split("\n", 1)[0].strip()
        ins = [(int(m.group(1), 16), m.group(3), m.group(4)) for m in LINE.finditer(chunk)]
        yield name, ins


def loops(ins):
    out = []
    for i, (addr, op, args) in enumerate(ins):
        if op.split(".")[0] != "BRA":
            continue
        t = re.search(r"0x([0-9a-f]+)", args)
        if not t or int(t.group(1), 16) > addr:
            continue
        lo = int(t.group(1), 16)
        body = [o for a, o, _ in ins if lo <= a <= addr]
        out.append(collections.Counter(o.split(".")[0] for o in body))
    return sorted(out, key=lambda c: sum(c.values()))


def main():
    text = open(sys.argv[1]).read()
    pins = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    bad = 0
    for name, ins in kernels(text):
        ops = collections.Counter(o.split(".")[0] for _, o, _ in ins)
        ls = loops(ins)
        why = []
        if ops["LDL"] or ops["STL"]:
            why.append(f"spills: LDL {ops['LDL']}, STL {ops['STL']}")
        for key, pin in pins.items():
            if key not in name:
                continue
            if pin["loop"] >= len(ls):
                why.append(f"pin {key}: loop {pin['loop']} absent ({len(ls)} loops)")
                continue
            got = ls[pin["loop"]]
            for op, n in pin["ops"].items():
                if got[op] != n:
                    why.append(f"pin {key}: loop {pin['loop']} has {got[op]} {op}, pinned {n}")
        bad += bool(why)
        print(json.dumps({"kernel": name, "instructions": len(ins), "loops": [dict(c.most_common()) for c in ls],
                          "ok": not why, "why": why}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
