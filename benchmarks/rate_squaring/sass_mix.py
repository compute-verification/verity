"""The SASS opcode mix of each `chain` and `probe` kernel in a cuobjdump -sass listing, whole function and the timed loop's body.

The loop body is the instructions between the loop's head (the back edge's target) and its back edge (`BRA.U UP0, head`
or `@P BRA head`, the last backward branch), so a squaring's static count is body / ILP. The rare second fold's carry
(`add_small`'s branch) is inside the body; it runs about once in 2^13 squarings.

    python3 sass_mix.py LISTING.sass > mix.json
"""

import collections
import json
import re
import sys

FN = re.compile(r"Function : (\S+)")
INS = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?(.*)$")
BRA = re.compile(r"BRA(?:\.U)?\s+(?:!?U?P\w+,\s*)?(0x[0-9a-f]+)")


def parse(path):
    fns, cur = {}, None
    for line in open(path):
        m = FN.search(line)
        if m:
            cur = fns.setdefault(m.group(1), [])
            continue
        m = INS.match(line)
        if m and cur is not None:
            cur.append((int(m.group(1), 16), m.group(2), m.group(2) + (m.group(3) or ""), line))
    return fns


def mix(ins):
    base, full = collections.Counter(i[1] for i in ins), collections.Counter(i[2] for i in ins)
    srt = lambda c: dict(sorted(c.items(), key=lambda kv: -kv[1]))
    return {"total": len(ins), "base": srt(base), "full": srt(full)}


def loop_body(ins):
    back = None
    for addr, _, _, line in ins:
        m = BRA.search(line)
        if m and int(m.group(1), 16) < addr:
            back = (int(m.group(1), 16), addr)
    if back is None:
        return []
    head, tail = back
    return [i for i in ins if head <= i[0] <= tail]


def main():
    out = {}
    for fn, ins in parse(sys.argv[1]).items():
        if "chain" not in fn and "probe" not in fn:
            continue
        body = loop_body(ins)
        out[fn] = {"function": mix(ins), "loop_body": mix(body)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
