"""Exact instruction counts in each `contend_k` kernel's timed loops, from a cuobjdump -sass listing.

Each loop is `#pragma unroll 1`, so its body is one iteration: the fused loop (U = 16 MMAs and F fillers), and for the
split mode the MMA-only loop (16 MMAs) and the filler-only loop (F fillers). For each kernel this counts, in the fused
loop, the MMAs, the filler class's opcodes net of the F = 0 kernel's (the loop's own overhead can use the same opcode),
the NOPs and MOVs ptxas adds, and everything else, and says whether the body is exactly 16 MMAs, F fillers and the
F = 0 kernel's overhead, NOPs and MOVs aside. ptxas puts a NOP after an MMA that no other instruction follows, so the
NOPs fall as F rises. `pattern` is the fused body in issue order: M an MMA, x a filler, n a NOP, o anything else.

    python3 contend_sass.py LISTING.sass > contend-sass.json
"""

import collections
import json
import re
import sys

from sass_mix import BRA, parse

NAME = re.compile(r"_Z9contend_kILi(\d)ELi(\d+)ELi(\d+)E")
TYPES = ("fp8", "s8")
CLASSES = ("iadd3", "lop3", "shf", "imad", "prmt", "ffma", "lds", "idp4a", "iadd3r2", "ffma3", "iadd3c", "iadd3i",
           "uiadd3", "rot1", "mix_iadd3_shf", "mix_iadd3_lds")
OPCODE = {"iadd3": ("IADD3",), "lop3": ("LOP3",), "shf": ("SHF",), "imad": ("IMAD",), "prmt": ("PRMT",),
          "ffma": ("FFMA",), "lds": ("LDS",), "idp4a": ("IDP",), "iadd3r2": ("IADD3",), "ffma3": ("FFMA",),
          "iadd3c": ("IADD3",), "iadd3i": ("IADD3",), "uiadd3": ("UIADD3", "IADD3"), "rot1": ("SHF",),
          "mix_iadd3_shf": ("IADD3", "SHF"), "mix_iadd3_lds": ("IADD3", "LDS")}
MMA = ("QMMA", "IMMA", "HMMA", "OMMA")
PAD = ("NOP", "MOV")


def opcode_of(full):
    """IMAD.MOV and IMAD.SHL are moves and shifts the compiler spells as IMAD; they don't count as IMAD."""
    return "IMAD.MOV" if full.startswith(("IMAD.MOV", "IMAD.SHL")) else full.split(".")[0]


def loops(ins):
    """Every loop body, from a backward branch's target to the branch."""
    out = []
    for addr, _, _, line in ins:
        m = BRA.search(line)
        if not m:
            continue
        tgt = re.search(r"/\*->(0x[0-9a-f]+)\*/", line)
        target = int(m.group(1) or tgt.group(1), 16) if (m.group(1) or tgt) else None
        if target is not None and target < addr:
            out.append([i for i in ins if target <= i[0] <= addr])
    return out


def counts(body):
    return collections.Counter(opcode_of(full) for _, _, full, _ in body)


def nmma(body):
    return sum(1 for _, _, full, _ in body if opcode_of(full) in MMA)


def main():
    kernels, base = [], {}
    for fn, ins in parse(sys.argv[1]).items():
        m = NAME.search(fn)
        if not m:
            continue
        t, c, f = TYPES[int(m.group(1))], int(m.group(2)), int(m.group(3))
        bodies = loops(ins)
        with_mma = sorted((b for b in bodies if nmma(b)), key=len)
        no_mma = sorted((b for b in bodies if not nmma(b)), key=len)
        k = {"type": t, "cls": CLASSES[c] if f else "none", "F": f, "function": fn,
             "body": with_mma[-1] if with_mma else [], "mma_only": with_mma[0] if len(with_mma) > 1 else None,
             "fill_only": no_mma[-1] if no_mma else None}
        kernels.append(k)
        if f == 0:
            base[t] = counts(k["body"])
    out = []
    for k in kernels:
        body, mo, fo = k.pop("body"), k.pop("mma_only"), k.pop("fill_only")
        cnt = counts(body)
        b = base.get(k["type"], collections.Counter())
        ops = OPCODE.get(k["cls"], ())
        fill = sum(cnt[o] - b[o] for o in ops)
        other = {o: n for o, n in cnt.items() if o not in MMA + PAD + ops and n}
        base_other = {o: n for o, n in b.items() if o not in MMA + PAD + ops and n}
        pattern = "".join("M" if o in MMA else "x" if o in ops else "n" if o == "NOP" else "o"
                          for o in (opcode_of(full) for _, _, full, _ in body))
        split = {}
        if mo is not None:
            split["mma_only"] = {"mma": nmma(mo), "total": len(mo)}
        if fo is not None:
            fc = counts(fo)
            split["fill_only"] = {"fillers": sum(fc[o] for o in ops), "total": len(fo),
                                  "other": {o: n for o, n in fc.items() if o not in ops}}
        k.update(total=len(body), mma=nmma(body), fillers=fill, nop=cnt["NOP"], mov=cnt["MOV"], other=other,
                 exact=nmma(body) == 16 and fill == k["F"] and other == base_other, pattern=pattern,
                 forms=sorted({full for _, _, full, _ in body if opcode_of(full) in ops}), split=split,
                 text=[line.strip() for *_, line in body] if k["F"] in (0, 16) else None)
        out.append(k)
    out.sort(key=lambda r: (r["type"], r["cls"], r["F"]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
