"""The SASS opcode mix of each `pure_k`, `probe_k` and `fused_k` kernel's timed loop, from a cuobjdump -sass listing.

The timed loops are `#pragma unroll 1`, so the loop body (from the last backward branch's target to that branch) is one
iteration exactly: per thread, PU chains of PW words (pure), 8 steps of PU chains (probe), or C MMAs into FT tiles plus
one absorb of 4 FT words (fused).

    python3 sass_mix.py LISTING.sass > sass-mix.json
"""

import collections
import json
import re
import sys

FN = re.compile(r"Function : (\S+)|^\.text\.(\S+):")
LABEL = re.compile(r"^(\.L_x_\d+):")
INS = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?(.*)$")
BRA = re.compile(r"BRA(?:\.U)?\s+(?:!?U?P\w+,\s*)?(?:(0x[0-9a-f]+)|`\((\.L_x_\d+)\))")
NAME = re.compile(r"_Z\d+(pure_k|probe_k|fused_k)ILi(n?\d+)E(?:Li(\d+)E)?(?:Li(\d+)E)?")


def parse(path):
    """Instructions per function, from cuobjdump -sass (hex branch targets) or nvdisasm (label targets)."""
    fns, cur, pending, labels = {}, None, [], {}
    for line in open(path):
        m = FN.search(line)
        if m:
            cur = fns.setdefault(m.group(1) or m.group(2), [])
            labels[id(cur)] = {}
            continue
        m = LABEL.match(line)
        if m and cur is not None:
            pending.append(m.group(1))
            continue
        m = INS.match(line)
        if m and cur is not None:
            addr = int(m.group(1), 16)
            for name in pending:
                labels[id(cur)][name] = addr
            pending = []
            cur.append((addr, m.group(2), m.group(2) + (m.group(3) or ""), line.rstrip()))
    for ins in fns.values():
        names = labels.get(id(ins), {})
        for k, (addr, base, full, line) in enumerate(ins):
            m = BRA.search(line)
            if m and m.group(2) in names:
                ins[k] = (addr, base, full, line + f"  /*->{names[m.group(2)]:#06x}*/")
    return fns


def mix(ins):
    base, full = collections.Counter(i[1] for i in ins), collections.Counter(i[2] for i in ins)
    srt = lambda c: dict(sorted(c.items(), key=lambda kv: -kv[1]))
    return {"total": len(ins), "base": srt(base), "full": srt(full)}


def loop_body(ins):
    back = None
    for addr, _, _, line in ins:
        m = BRA.search(line)
        if not m:
            continue
        tgt = re.search(r"/\*->(0x[0-9a-f]+)\*/", line)
        target = int(m.group(1) or tgt.group(1), 16) if (m.group(1) or tgt) else None
        if target is not None and target < addr:
            back = (target, addr)
    if back is None:
        return []
    head, tail = back
    return [i for i in ins if head <= i[0] <= tail]


def ident(fn):
    m = NAME.search(fn)
    if not m:
        return None
    first = int(m.group(2).replace("n", "-"))
    return {"mode": m.group(1)[:-2], "index": first, "C": int(m.group(3)) if m.group(3) else None,
            "acc": int(m.group(4)) if m.group(4) else None}


def main():
    out = []
    for fn, ins in parse(sys.argv[1]).items():
        who = ident(fn)
        if who is None:
            continue
        body = loop_body(ins)
        out.append({**who, "function": fn, "loop_body": mix(body), "loop_text": [i[3].strip() for i in body]})
    out.sort(key=lambda r: (r["mode"], r["acc"] or 0, r["index"], r["C"] or 0))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
