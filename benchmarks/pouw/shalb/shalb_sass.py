"""SHA-512's registers written a compression, from a probe's SASS, with the probe's own loop overhead taken out: stdlib.

    python3 shalb_sass.py OUT_JSON NAME=PROBE_SASS ...

Each PROBE_SASS is `cuobjdump -sass` of a probe whose loop trip is one compression (sha512_probe.cu, the committer's;
sha512_min_probe.cu, the minimal one). `sha512_sass.py` prices the trip as the committer's count does: every
instruction of the loop body but its loads. This adds the harness split for a straight-line body: the harness is the
backward slice, around the loop, of the body's global-memory addresses and of its branch's predicate (the block
pointer, the loop counter and the trip test); the compression is the rest. A rolled body (an inner loop) is reported
as `sha512_sass.py` prices it, without a split, since its inner loop's control is the compression's own.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import sha512_sass as ss

REG = re.compile(r"(?<![\w.])!?(U?[RP](?:\d+))(\.64|\.128)?")
PRED = re.compile(r"^!?U?P(?:\d+|T)$")
MEM = ("LDG", "STG", "LD", "ST", "ATOM", "RED")


def regs(text: str) -> set[str]:
    out = set()
    for name, width in REG.findall(text):
        n = 4 if width == ".128" else 2 if width == ".64" else 1
        kind, num = re.match(r"(U?[RP])(\d+)", name).groups()
        out |= {f"{kind}{int(num) + k}" for k in range(n)}
    return out


def operands(i: dict) -> list[str]:
    return [o.strip() for o in i["ops"].split(",")] if i["ops"] else []


def defs_uses(i: dict) -> tuple[set[str], set[str], set[str]]:
    """Registers an instruction writes, the registers it reads, and those of its reads that form a memory address."""
    op, ops = i["op"], operands(i)
    base = op.split(".")[0]
    uses = regs(i["pred"])
    addr: set[str] = set()
    for o in ops:
        if "[" in o:
            addr |= regs(o[o.rfind("["):])
    if base in ss.NO_WRITE or not ops:
        return set(), uses | set().union(*(regs(o) for o in ops)) if ops else uses, addr
    d = ops[0]
    if base == "LDG" or base == "LD":
        n = 4 if ".128" in op else 2 if ".64" in op else 1
        m = re.match(r"(U?R)(\d+)", d)
        defs = {f"{m.group(1)}{int(m.group(2)) + k}" for k in range(n)} if m else set()
    else:
        n = 2 if ".WIDE" in op or ".64" in op else 1
        defs = {f"{k}{int(v) + j}" for k, v in [re.match(r"(U?[RP])(\d+)", d).groups()] for j in range(n)} \
            if re.match(r"U?[RP]\d+", d) else set()
    rest = ops[1:]
    extra = 0
    while rest and extra < 2 and PRED.match(rest[0]):
        defs |= regs(rest[0])
        rest, extra = rest[1:], extra + 1
    for o in rest:
        uses |= regs(o)
    return defs, uses, addr


def harness(body: list[dict]) -> set[int]:
    """Addresses of the body's instructions in the backward slice of its memory addresses and its branch predicate."""
    marked: set[int] = set()
    live: set[str] = set()
    for _ in range(8):
        before = set(marked)
        for i in reversed(body):
            defs, uses, addr = defs_uses(i)
            if i["op"].split(".")[0] in MEM:
                live |= addr
            if i["op"].startswith("BRA"):
                live |= regs(i["pred"])
            if defs & live:
                marked.add(i["addr"])
                live -= defs
                live |= uses
        if marked == before:
            break
    return marked


def count(path: Path) -> dict:
    funcs = ss.parse(path)
    (name, ins), = [(k, v) for k, v in funcs.items() if "probe" in k]
    r = {"sha512_sass": ss.per_compression(ins, 4)}
    a, b = ss.loops(ins)[0]
    body = ins[a:b + 1]
    if r["sha512_sass"]["inner_loops"]:
        r["compression"] = r["sha512_sass"]
        r["harness"] = None
        return r
    h = harness(body)
    r["harness"] = [f"{i['pred']} {i['op']} {i['ops']}".strip() for i in body if i["addr"] in h]
    r["compression"] = ss.price([i for i in body if i["addr"] not in h])
    return r


def main() -> None:
    out_path = Path(sys.argv[1])
    out = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        r = out[name] = count(Path(path))
        c, s = r["compression"], r["sha512_sass"]
        print(f"{name}: {c['registers_written']:.0f} registers written a compression ({c['instructions']:.0f} "
              f"instructions; the trip as sha512_sass.py prices it {s['registers_written']:.0f} / {s['instructions']:.0f}; "
              f"harness {r['harness'] if r['harness'] is not None else 'not split (rolled)'}); "
              f"opcodes {c['by_opcode']}")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
