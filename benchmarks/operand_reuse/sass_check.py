"""Per-function SASS summary of `probe.cu`: opcode counts, and for every tensor-core instruction the instruction that last
wrote each of its source registers, so that "this QMMA reads the F2FP's register" or "this HMMA reads the QMMA's D" is
read off the listing rather than inferred.

Usage: python3 sass_check.py probe_sm120a.sass > sass_check.json   (stdlib only; `cuobjdump -sass` output on stdin or argv)
"""

import json
import re
import sys

MMA = ("HMMA", "QMMA", "IMMA", "OMMA", "UTCHMMA", "UTCQMMA", "UTCIMMA", "UTCOMMA")
NO_DEST = ("ST", "BRA", "EXIT", "BAR", "RET", "NOP", "BSYNC", "BSSY", "WARPSYNC", "CALL", "RED", "MEMBAR", "ERRBAR", "DEPBAR")
LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\s*(.*?)\s*;")
REG = re.compile(r"\bR(\d+)\b")


def width(op: str, role: str) -> int:
    """Registers spanned by an operand of a tensor-core instruction (the base register names a contiguous group)."""
    if role == "A":
        return 2 if ".1684" in op or ".8816" in op else 4
    if role == "B":
        return 1 if ".1684" in op or ".1688.F16" in op else 2
    parts = op.split(".")
    return 2 if len(parts) > 2 and parts[2] == "F16" else 4


def dest_width(op: str) -> int:
    if any(f".{w}" in op for w in ("128",)):
        return 4
    if ".64" in op:
        return 2
    return 1


def functions(text: str):
    name, body = None, []
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            if name:
                yield name, body
            name, body = m.group(1), []
        elif name:
            mm = LINE.search(line)
            if mm:
                body.append((mm.group(1), (mm.group(2) or "").strip(), mm.group(3), mm.group(4)))
    if name:
        yield name, body


def analyse(body):
    counts, writer, mmas = {}, {}, []
    for idx, (addr, guard, op, args) in enumerate(body):
        base = op.split(".")[0]
        counts[base] = counts.get(base, 0) + 1
        regs = [int(r) for r in REG.findall(args)]
        if base in MMA:
            d, a, b, c = (regs + [None] * 4)[:4]
            roles = {"A": a, "B": b, "C": c}
            sources = {}
            for role, r0 in roles.items():
                if r0 is None:
                    continue
                w = width(op, role)
                sources[role] = {
                    f"R{r}": (lambda x: f"{x[1]}@{x[0]}" if x else "entry")(writer.get(r)) for r in range(r0, r0 + w)
                }
            mmas.append({"addr": addr, "op": op, "args": args, "sources": sources})
            if d is not None:
                for r in range(d, d + width(op, "D")):
                    writer[r] = (addr, op)
        elif not base.startswith(NO_DEST) and regs and REG.match(args.split(",")[0].strip()):
            d = regs[0]
            for r in range(d, d + dest_width(op)):
                writer[r] = (addr, op)
    return {"counts": dict(sorted(counts.items())), "mma": mmas}


def main() -> None:
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    out = {name: analyse(body) for name, body in functions(text)}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
