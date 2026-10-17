"""Static SASS opcode counts per kernel of a cuobjdump -sass dump, as one JSON object (kernel -> opcode -> count).

The receipt's cost model is its ops a word: a duo kernel's LOP3 + IMAD + LEA.HI (or SHF + IADD3) over its g0 twin's,
divided by the words it absorbs in one stage body, is what the table checks against duo's 3 (duo2's 6).
python3 sass_mix.py DUMP
"""

import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"Function : (\S+)")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def main(path: str) -> int:
    out: dict[str, Counter] = {}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        f = FUNC.search(line)
        if f:
            cur = f.group(1)
            out[cur] = Counter()
            continue
        if cur is None:
            continue
        i = INSN.search(line)
        if i:
            op = i.group(1)
            base = op.split(".")[0]
            out[cur][base] += 1
            if op.startswith("LEA.HI"):
                out[cur]["LEA.HI"] += 1
            if op.startswith("IMMA"):
                out[cur][op] += 1
    keep = {"IMMA", "LOP3", "IMAD", "LEA", "LEA.HI", "SHF", "IADD3", "LDSM", "SYNCS", "BAR", "STG", "LDG", "SHFL", "PRMT"}
    res = {k: {o: c for o, c in sorted(v.items()) if o in keep or o.startswith("IMMA")} | {"total": sum(v.values())}
           for k, v in out.items()}
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0 if any("IMMA" in v for v in res.values()) else 5


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
