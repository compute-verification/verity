"""Count each kernel's SASS loop in `cuobjdump -sass` output, and gate sm90_clock's kernels on it.

A kernel's loop is the instructions from the target of its last backward BRA to that BRA. Counts are by opcode with its
modifiers (`LOP3.LUT`, `IMAD.WIDE.U32`, `REDUX.XOR`), and by class as DS-90's census names them where `loads.class_of`
has one (ALU, IMAD, SHFL and so on), else the opcode's first word.

The gate (`--gate`): no local-memory access (LDL, STL) in any kernel; each `clk<...>` kernel's loop holds the IGMMA its
load issues (4 a trip for the three wgmma loads, none for li_alone); each `li_exact<...>` loop holds li's 64 LOP3 and 64
IMAD a unit (256 and 256 a trip of four units), the checkpoint's REDUX (4 a trip a segment, 16 a set; for CP_RED, 4 RED
and no REDUX) and its per-lane XOR (4 LOP3 a checkpoint). Exit 1 when a check fails.

Run: `python sass_count.py SASS [--json OUT] [--gate]`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"^\s*Function : (\S+)")
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)(.*?);")
BRA = re.compile(r"BRA\s+(?:`\(\S+\)\s*)?(0x[0-9a-f]+)")


def kernels(text: str) -> dict[str, list[tuple[int, str, str]]]:
    out: dict[str, list[tuple[int, str, str]]] = {}
    cur = None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = INS.search(line)
        if m and cur is not None:
            out[cur].append((int(m.group(1), 16), m.group(3), m.group(4)))
    return out


def loop(ins: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """The kernel's timed loop: of its backward branches' bodies, the one with the most IGMMA, then the longest (a wgmma
    loop is shorter than the tile fill's)."""
    best: list[tuple[int, str, str]] = []
    rank = (-1, -1)
    for addr, op, rest in ins:
        if op.split(".")[0] != "BRA":
            continue
        m = BRA.search(op + rest)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt < addr:
            body = [x for x in ins if tgt <= x[0] <= addr]
            r = (sum(1 for _, o, _ in body if o.startswith("IGMMA")), len(body))
            if r > rank:
                best, rank = body, r
    return best


def demangle_tag(name: str) -> str:
    m = re.match(r"_Z\d+(\w+?)ILi(\d+)E", name)
    return f"{m.group(1)}<{m.group(2)}>" if m else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--json")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    ks = kernels(open(args.sass).read())
    rows = {}
    fails = []
    for name, ins in ks.items():
        tag = demangle_tag(name)
        body = loop(ins)
        ops = Counter(op for _, op, _ in body)
        base = Counter(op.split(".")[0] for _, op, _ in body)
        local = sum(1 for _, op, _ in ins if op.split(".")[0] in ("LDL", "STL"))
        rows[tag] = {"mangled": name, "instructions": len(ins), "loop": len(body), "local": local,
                     "loop_ops": dict(sorted(ops.items())), "loop_base": dict(sorted(base.items()))}
        if not (tag.startswith("clk<") or tag.startswith("li_exact<")):
            continue
        if local:
            fails.append(f"{tag}: {local} local-memory accesses (a spill)")
        if tag.startswith("clk<"):
            want = 0 if tag == "clk<3>" else 4
            got = sum(n for op, n in ops.items() if op.startswith("IGMMA"))
            if got != want:
                fails.append(f"{tag}: {got} IGMMA in the loop, want {want}")
        if tag.startswith("li_exact<"):
            cp = int(tag[len("li_exact<"):-1])
            ncp = 16 if cp == 2 else 4
            redux, red = (0, ncp) if cp == 3 else (ncp, 0)
            got_red = base["RED"] + base["REDG"]
            if base["REDUX"] != redux or got_red != red:
                fails.append(f"{tag}: {base['REDUX']} REDUX and {got_red} RED a trip, want {redux} and {red}")
            if base["LOP3"] < 256 + 4 * ncp:
                fails.append(f"{tag}: {base['LOP3']} LOP3 a trip, want at least {256 + 4 * ncp}")
            imad = sum(n for op, n in ops.items() if op == "IMAD")
            if imad < 256:
                fails.append(f"{tag}: {imad} IMAD a trip, want at least 256")
    for tag, r in rows.items():
        if tag.startswith(("clk<", "li_exact<")):
            print(f"{tag}: {r['instructions']} instructions, loop {r['loop']}: {r['loop_ops']}")
    for f in fails:
        print(f"REFUSED {f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"kernels": rows, "refused": fails}, f, indent=1)
    if args.gate:
        print("GATE", "REFUSED" if fails else "PASS")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
