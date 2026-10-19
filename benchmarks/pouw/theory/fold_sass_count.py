"""Count each fold kernel's SASS opcodes by census class from `cuobjdump -sass` (`fold_sass.cu`), with
facet_break/loads.py's vector rules (`IMAD`: IMAD, IMUL; `ALU`: IADD3, IADD, LOP3, LOP, SHF, PRMT, LEA), per absorbed word.
fx1 absorbs L = 128 words, li and li heavy 2L = 256 (a pair a half step). Prints one JSON object per kernel."""
from __future__ import annotations

import collections
import json
import re
import sys

IMAD = re.compile(r"\bIMAD\b|\bIMUL\b")
ALU = re.compile(r"\bIADD3\b|\bIADD\b|\bLOP3\b|\bLOP\b|\bSHF\b|\bPRMT\b|\bLEA\b")
WORDS = {"fx1_chain": 128, "li_chain": 256, "li_heavy_chain": 256}


def cls(op: str) -> str:
    base = op.split(".")[0]
    if IMAD.search(base):
        return "IMAD"
    if ALU.search(base):
        return "ALU"
    return base


def count(path: str) -> dict[str, dict]:
    kernel = None
    classes: dict[str, collections.Counter] = {}
    ops: dict[str, collections.Counter] = {}
    for line in open(path, encoding="utf-8"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            kernel = m.group(1)
            classes[kernel] = collections.Counter()
            ops[kernel] = collections.Counter()
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and kernel:
            op = m.group(2)
            ops[kernel][op] += 1
            classes[kernel][cls(op)] += 1
    out = {}
    for k, c in classes.items():
        w = WORDS.get(k)
        if w is None:
            continue
        out[k] = {"words": w, "per_word": {x: round(c[x] / w, 4) for x in ("ALU", "IMAD")},
                  "lop3": ops[k].get("LOP3.LUT", 0), "classes": dict(c.most_common()),
                  "opcodes": dict(ops[k].most_common())}
    return out


if __name__ == "__main__":
    print(json.dumps(count(sys.argv[1]), indent=1))
