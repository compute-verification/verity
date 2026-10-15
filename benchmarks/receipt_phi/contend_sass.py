"""Exact instruction counts in each `contend_k` kernel's timed loop, from a cuobjdump -sass listing.

The loop is `#pragma unroll 1` over U = 16 MMAs and F fillers, so its body is one iteration. For each kernel this counts
the MMAs, the filler class's opcode net of the F = 0 kernel's (the loop's own overhead can use the same opcode), and
everything else, and says whether the body is exactly 16 MMAs, F fillers and the F = 0 kernel's overhead. `pattern` is
the body in issue order: M an MMA, x a filler, o anything else.

    python3 contend_sass.py LISTING.sass > contend-sass.json
"""

import collections
import json
import re
import sys

from sass_mix import loop_body, parse

NAME = re.compile(r"_Z9contend_kILi(\d)ELi(\d+)ELi(\d+)E")
TYPES = ("fp8", "s8")
CLASSES = ("iadd3", "lop3", "shf", "imad", "prmt", "ffma", "lds", "idp4a", "iadd3r2", "ffma3")
OPCODE = {"iadd3": "IADD3", "lop3": "LOP3", "shf": "SHF", "imad": "IMAD", "prmt": "PRMT", "ffma": "FFMA", "lds": "LDS",
          "idp4a": "IDP4A", "iadd3r2": "IADD3", "ffma3": "FFMA"}
MMA = ("QMMA", "IMMA", "HMMA", "OMMA")


def opcode_of(full):
    """IMAD.MOV and IMAD.SHL are moves and shifts the compiler spells as IMAD; they don't count as IMAD."""
    return "IMAD.MOV" if full.startswith(("IMAD.MOV", "IMAD.SHL")) else full.split(".")[0]


def counts(body):
    return collections.Counter(opcode_of(full) for _, _, full, _ in body)


def main():
    kernels, base = [], {}
    for fn, ins in parse(sys.argv[1]).items():
        m = NAME.search(fn)
        if not m:
            continue
        t, c, f = TYPES[int(m.group(1))], int(m.group(2)), int(m.group(3))
        body = loop_body(ins)
        k = {"type": t, "cls": CLASSES[c] if f else "none", "F": f, "function": fn, "body": body}
        kernels.append(k)
        if f == 0:
            base[t] = counts(body)
    out = []
    for k in kernels:
        body = k.pop("body")
        cnt = counts(body)
        b = base.get(k["type"], collections.Counter())
        mma = sum(cnt[o] for o in MMA)
        op = OPCODE.get(k["cls"])
        fill = cnt[op] - b[op] if op else 0
        other = {o: n for o, n in cnt.items() if o not in MMA and o != op and n}
        base_other = {o: n for o, n in b.items() if o not in MMA and o != op and n}
        pattern = "".join("M" if opcode_of(full) in MMA else "x" if op and opcode_of(full) == op else "o"
                          for _, _, full, _ in body)
        k.update(total=len(body), mma=mma, fillers=fill, other=other,
                 exact=mma == 16 and fill == k["F"] and other == base_other, pattern=pattern,
                 forms=sorted({full for _, _, full, _ in body if op and opcode_of(full) == op}),
                 text=[line.strip() for *_, line in body] if k["F"] in (0, 16) else None)
        out.append(k)
    out.sort(key=lambda r: (r["type"], r["cls"], r["F"]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
