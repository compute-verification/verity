"""Opcode counts of the built row kernels (``cuobjdump -sass``), by variant: what each compression form issues on which pipe.

``python sass.py LIB`` prints {variant: {opcode: count}} as JSON; counts are static (a rolled loop's body once).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter


def variant_name(v: int, roll: int) -> str:
    """The name engine.variant_flags reads back as ``-DSX_V=v -DSX_ROLL=roll``."""
    digits = "".join(str((v >> (2 * i) & 3) + ((v >> 2 * i & 3) == 3 and v >> 9 & 1)) for i in range(4))
    return digits + ("w" if v >> 8 & 1 else "") + ("" if roll else "s")


def counts(lib: str) -> dict[str, dict[str, int]]:
    out = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=True).stdout
    res: dict[str, dict[str, int]] = {}
    for fn in out.split("Function : ")[1:]:
        name = fn.split("\n", 1)[0].strip()
        if "k_rows" not in name:
            continue
        m = re.search(r"SxILi(\d+)ELi(\d)E", name)
        if m:
            key = variant_name(int(m.group(1)), int(m.group(2)))
        else:
            key = "pouw" if "Pouw" in name else name
        ops = Counter()
        for line in fn.splitlines():
            ins = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)", line)
            if ins:
                ops[ins.group(1)] += 1
        res[key] = dict(ops.most_common())
    return res


if __name__ == "__main__":
    print(json.dumps(counts(sys.argv[1]), indent=1))
