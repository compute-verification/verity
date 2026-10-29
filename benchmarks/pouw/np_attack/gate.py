"""SASS gate for np_attack's probe kernels: no spills, and each mode's hot loop runs the op it is meant to.

    python gate.py SASS.txt    one JSON line a kernel: its hot loop's opcode histogram and the verdict; exit 1 on a spill
                               (LDL/STL), a missing required opcode, or a present forbidden one.

The price a word a mode reports is only the op its loop runs, so a build whose loop lost that op (or gained a cheaper
one) fails here before it is timed. Counts are floors, not exact, so a different unroll still passes while a changed
datapath does not.
"""

import collections
import json
import re
import sys

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")

# mode index (ILi{m}E in the mangled name) -> (tag, required ops with a floor count, forbidden ops) in the hot loop.
# A required entry "A+B" is a floor on the two counts' sum: ptxas spreads integer adds over the IMAD pipe (IMAD.IADD)
# and the ALU pipe (IADD3) to co-issue them, which is the realistic adversary, so neither alone is pinned.
RULES = {
    0: ("advadd", {"IADD3+IMAD": 24}, {"FADD", "FFMA", "FMUL", "REDG", "LDG", "LDSM", "ATOMG"}),
    1: ("advmix", {"IMAD": 1, "FADD": 1}, {"REDG", "LDSM", "LDG"}),
    2: ("widemul", {"IMAD": 24}, {"FADD", "REDG", "LDSM", "LDG"}),
    3: ("carry", {"IADD3": 24}, {"FADD", "REDG", "LDSM", "LDG"}),
    4: ("redg", {"REDG": 24}, {"STG", "LDSM", "ATOMG"}),
    5: ("move", {"LDG": 1}, {"REDG", "LDSM", "STS", "FADD"}),
}


def kernels(text):
    for chunk in re.split(r"\n\s+Function : ", text)[1:]:
        name = chunk.split("\n", 1)[0].strip()
        ins = [(int(m.group(1), 16), m.group(3), m.group(4)) for m in LINE.finditer(chunk)]
        yield name, ins


def loops(ins):
    out = []
    for addr, op, args in ins:
        if op.split(".")[0] != "BRA":
            continue
        t = re.search(r"0x([0-9a-f]+)", args)
        if not t or int(t.group(1), 16) > addr:
            continue
        lo = int(t.group(1), 16)
        out.append(collections.Counter(o.split(".")[0] for a, o, _ in ins if lo <= a <= addr))
    return sorted(out, key=lambda c: sum(c.values()))


def main():
    bad = 0
    for name, ins in kernels(open(sys.argv[1]).read()):
        m = re.search(r"ILi(\d+)E", name)
        if not m or int(m.group(1)) not in RULES:
            continue
        tag, req, forb = RULES[int(m.group(1))]
        allops = collections.Counter(o.split(".")[0] for _, o, _ in ins)
        ls = loops(ins)
        hot = ls[-1] if ls else collections.Counter()
        why = []
        if allops["LDL"] or allops["STL"]:
            why.append(f"spills: LDL {allops['LDL']}, STL {allops['STL']}")
        for op, n in req.items():
            have = sum(hot[o] for o in op.split("+"))
            if have < n:
                why.append(f"required {op} >= {n}, hot loop has {have}")
        why += [f"forbidden {op} in hot loop ({hot[op]})" for op in forb if hot[op]]
        bad += bool(why)
        print(json.dumps({"kernel": name, "mode": tag, "instructions": len(ins),
                          "hot_loop": dict(hot.most_common()), "ok": not why, "why": why}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
