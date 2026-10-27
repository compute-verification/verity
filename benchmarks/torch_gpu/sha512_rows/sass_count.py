"""SASS instruction counts of one kernel, by opcode and by issue pipe, per iteration of its main loop.

    python3 sass_count.py BINARY FUNCTION [--trip N] [--per K] [--sass TEXT]    (one JSON line on stdout)

FUNCTION is a substring of the kernel's mangled name. The main loop is the backward branch whose body holds the most
instructions; a loop nested in it counts N times (--trip, default 1: the straight-line forms have none), so the count is
dynamic per main-loop iteration, divided by K (--per: compressions an iteration). Pipes, at the issue rates the derived
ceiling assumes (16 lanes a clock a sub-partition for `alu` and `fma`, one warp instruction a clock a sub-partition
issued in all): `alu` the integer ALU (shifts, LOP3, adds, moves, selects, compares), `fma` the multiply-add pipe (IMAD,
FFMA), `uniform` the uniform datapath (U*), `mem`, `ctrl` and `other`.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import subprocess

ALU = {"SHF", "LOP3", "LOP", "IADD3", "IADD", "IADD.64", "LEA", "SEL", "ISETP", "PRMT", "MOV", "IMNMX", "IABS", "SHL",
       "SHR", "FSEL", "PLOP3", "P2R", "R2P", "VIADD", "VIMNMX", "BMSK", "BREV", "FLO", "SGXT", "IDP", "ICMP", "CSET"}
FMA = {"IMAD", "IMAD.WIDE", "IMAD.HI", "IMUL", "FFMA", "FMUL", "FADD", "HFMA2", "IMAD.64"}
MEM = {"LDG", "STG", "LDS", "STS", "LDC", "LD", "ST", "LDL", "STL", "ATOM", "ATOMG", "ATOMS", "RED", "LDSM", "MEMBAR", "CCTL"}
CTRL = {"BRA", "EXIT", "BAR", "NOP", "S2R", "CS2R", "BSSY", "BSYNC", "WARPSYNC", "CALL", "RET", "YIELD", "DEPBAR", "BPT",
        "JMP", "BRX", "ACQBULK", "ELECT", "VOTE", "VOTEU"}

INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
TARGET = re.compile(r"0x([0-9a-f]+)\s*$")


def opcode(tok: str) -> str:
    parts = tok.split(".")
    if parts[0] in ("IADD", "IMAD") and len(parts) > 1 and parts[1] in ("64", "WIDE", "HI"):
        return parts[0] + "." + parts[1]
    return parts[0]


def pipe(op: str) -> str:
    if op in ALU:
        return "alu"
    if op in FMA:
        return "fma"
    if op.startswith("U") and op not in ("UNKNOWN",):
        return "uniform"
    if op in MEM:
        return "mem"
    if op in CTRL:
        return "ctrl"
    return "other"


def function_sass(binary: str, name: str, text: str | None = None) -> list[str]:
    out = text if text is not None else subprocess.run(["cuobjdump", "-sass", binary], check=True, capture_output=True,
                                                       text=True).stdout
    lines, keep, found = [], False, []
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            keep = name in m.group(1)
            if keep:
                found.append(m.group(1))
            continue
        if keep:
            lines.append(line)
    if len(found) != 1:
        raise SystemExit(f"{len(found)} functions match {name!r}: {found}")
    return lines


def count(binary: str, name: str, trip: int, per: float, text: str | None = None) -> dict:
    insts = []
    for line in function_sass(binary, name, text):
        m = INST.search(line)
        if m:
            insts.append((int(m.group(1), 16), opcode(m.group(3)), m.group(4).strip()))
    index = {a: i for i, (a, _, _) in enumerate(insts)}
    loops = []
    for i, (addr, op, args) in enumerate(insts):
        t = TARGET.search(args)
        if op == "BRA" and t and int(t.group(1), 16) < addr and int(t.group(1), 16) in index:
            loops.append((index[int(t.group(1), 16)], i))
    static = collections.Counter(op for _, op, _ in insts)
    if not loops:
        body, inner = (0, len(insts) - 1), []
    else:
        body = max(loops, key=lambda lp: lp[1] - lp[0])
        inner = [lp for lp in loops if lp != body and body[0] <= lp[0] and lp[1] <= body[1]]
        inner = [lp for lp in inner if not any(o != lp and o[0] <= lp[0] and lp[1] <= o[1] for o in inner)]
    dyn = collections.Counter()
    for i in range(body[0], body[1] + 1):
        dyn[insts[i][1]] += trip if any(a <= i <= b for a, b in inner) else 1
    pipes = collections.Counter()
    for op, c in dyn.items():
        pipes[pipe(op)] += c
    total = sum(dyn.values())
    # a warp's cycles a block on its sub-partition: each half-rate pipe instruction holds the pipe 2 cycles, and every
    # instruction takes one issue slot
    cycles = max(2 * pipes["alu"], 2 * pipes["fma"], total) / per
    return {
        "binary": binary, "function": name, "instructions_static": sum(static.values()), "loops": len(loops),
        "main_loop_static": body[1] - body[0] + 1, "inner_loops": [b - a + 1 for a, b in inner], "trip": trip, "per": per,
        "per_block": {op: round(c / per, 2) for op, c in sorted(dyn.items(), key=lambda kv: -kv[1])},
        "pipes_per_block": {p: round(c / per, 2) for p, c in sorted(pipes.items())},
        "total_per_block": round(total / per, 2), "warp_cycles_per_block_bound": round(cycles, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("function")
    ap.add_argument("--trip", type=int, default=1)
    ap.add_argument("--per", type=float, default=1.0)
    ap.add_argument("--sass", help="read this cuobjdump -sass text (gzip if .gz) instead of running cuobjdump on BINARY")
    a = ap.parse_args()
    text = None
    if a.sass:
        text = (gzip.open(a.sass, "rt") if a.sass.endswith(".gz") else open(a.sass)).read()
    print(json.dumps(count(a.binary, a.function, a.trip, a.per, text)))


if __name__ == "__main__":
    main()
