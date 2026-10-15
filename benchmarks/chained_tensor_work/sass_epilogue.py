"""What the chain's GEMM kernels do to an accumulator between the IMMA that last writes it and its store: for each plain
kernel of a `cuobjdump -sass` dump, a forward pass from the last IMMA over the rest of the function, tracking the registers
an IMMA wrote until another instruction overwrites them, and counting by opcode every instruction that reads one. Zero
readers other than stores means X_{s+1}'s bytes are the raw s32 words, with no conversion between the tensor core and
memory. One JSON object over every plain kernel; exit 1 if a reader of one the chain launches (CHAIN) is not a store. The
other plain kernels' counts include their loop's address and barrier registers, which the pass can't tell from accumulators.

    python3 sass_epilogue.py DUMP
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"Function : (\S+)")
INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
REG = re.compile(r"\bR(\d+)\b")
STORES = ("STG", "STS", "ST", "STL")
CHAIN = ("p_plain", "w_plain")


def regs(operand: str, width: int = 1) -> set[int]:
    m = REG.search(operand)
    return set(range(int(m.group(1)), int(m.group(1)) + width)) if m else set()


def width_of(op: str) -> int:
    """Registers an instruction's first operand spans: an IMMA.16832's D is 4, a 64- or 128-bit store's data 2 or 4."""
    if op.startswith("IMMA"):
        return 4
    return 4 if ".128" in op else 2 if ".64" in op or ".WIDE" in op else 1


def analyse(lines: list[tuple[str, str]]) -> dict:
    last = max((i for i, (op, _) in enumerate(lines) if op.startswith("IMMA")), default=None)
    if last is None:
        return {"imma": 0}
    acc: set[int] = set()
    for op, args in lines[: last + 1]:
        if op.startswith("IMMA"):
            acc |= regs(args.split(",")[0], 4)
    readers, stored, examples = Counter(), 0, []
    for op, args in lines[last + 1:]:
        parts = [p.strip() for p in args.split(",")]
        if not parts or not parts[0]:
            continue
        base = op.split(".")[0]
        if base in STORES:
            data = regs(parts[-1], width_of(op))
            if data & acc:
                readers[op] += 1
                stored += len(data & acc)
            continue
        srcs = set().union(*(regs(p) for p in parts[1:])) if len(parts) > 1 else set()
        if srcs & acc:
            readers[op] += 1
            if op.split(".")[0] not in STORES:
                examples.append(f"{op} {args}")
        acc -= regs(parts[0], width_of(op))
    return {"imma": sum(op.startswith("IMMA") for op, _ in lines), "readers": dict(readers), "accumulator_regs_stored": stored,
            "non_store_readers": sum(c for op, c in readers.items() if op.split(".")[0] not in STORES), "non_store_examples": examples[:8]}


def main(path: str) -> int:
    funcs: dict[str, list[tuple[str, str]]] = {}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        f = FUNC.search(line)
        if f:
            cur = f.group(1)
            funcs[cur] = []
            continue
        i = INSN.search(line)
        if cur is not None and i:
            funcs[cur].append((i.group(1), i.group(2)))
    res = {name: analyse(body) for name, body in funcs.items() if "plain" in name}
    print(json.dumps(res, indent=1, sort_keys=True))
    chain = [r for name, r in res.items() if any(c in name for c in CHAIN)]
    return 0 if len(chain) == len(CHAIN) and all(r.get("non_store_readers", 1) == 0 for r in chain) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
