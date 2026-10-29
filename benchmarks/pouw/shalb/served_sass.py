"""The served GEMM's tensor-core instructions in SASS: whether each writes a fresh register set (D != C) or accumulates
in place (D == C), the form honest's xor-m write count reads.

    python3 served_sass.py SASS [--json OUT]

SASS is cuobjdump -sass of np_kernel's ssn.cu. Per function: its IGMMA and IMMA instructions, each one's D and C base
registers, and its local-memory instructions. Honest's xor-m count (1 + ceil((m - 1)/2)/m writes a word) folds two
words a LOP3, so it needs two consecutive words alive at once: a fresh-register IMMA. An in-place accumulator leaves one
word alive at a time, and honest's fold is then one LOP3 a word (2 writes a word at every m).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INSTR = re.compile(r"\b(IGMMA|IMMA)(\.[\w.]+)?\s+(R\d+|RZ)\s*,\s*([^;]*);")


def operands(rest: str) -> list[str]:
    return [o.strip() for o in rest.split(",")]


def main() -> int:
    args = sys.argv[1:]
    funcs: dict = {}
    cur = None
    for line in Path(args[0]).read_text().splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = {"igmma": [], "imma": [], "local": 0}
            continue
        if cur is None:
            continue
        funcs[cur]["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
        m = INSTR.search(line)
        if m:
            op, shape, d, rest = m.group(1), m.group(2) or "", m.group(3), operands(m.group(4))
            regs = [o for o in rest if re.fullmatch(r"R\d+|RZ", o)]
            c = regs[-1] if regs else None
            funcs[cur][op.lower()].append({"shape": shape, "d": d, "c": c, "in_place": c == d, "text": line.strip()})
    gemms = {k: v for k, v in funcs.items() if "ssn_k" in k}
    tc = [i for v in gemms.values() for i in v["igmma"] + v["imma"]]
    summary = {
        "gemm_functions": len(gemms),
        "tensor_instructions": len(tc),
        "igmma": sum(len(v["igmma"]) for v in gemms.values()),
        "imma": sum(len(v["imma"]) for v in gemms.values()),
        "in_place": sum(i["in_place"] for i in tc),
        "fresh": sum(not i["in_place"] and i["c"] not in (None, "RZ") for i in tc),
        "zero_c": sum(i["c"] == "RZ" for i in tc),
        "local": sum(v["local"] for v in gemms.values()),
    }
    for k, v in gemms.items():
        print(f"{k}: IGMMA {len(v['igmma'])} (in place {sum(i['in_place'] for i in v['igmma'])}), "
              f"IMMA {len(v['imma'])}, local {v['local']}")
        for i in (v["igmma"] + v["imma"])[:3]:
            print(f"    {i['text']}")
    print(json.dumps(summary))
    ok = summary["gemm_functions"] > 0 and summary["tensor_instructions"] > 0 and summary["fresh"] == 0
    print("served GEMM: every tensor-core instruction accumulates in place (D == C) or from zero; none writes a fresh "
          "register set" if ok else "served GEMM: a fresh-register tensor-core instruction exists, or none was found")
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps({"summary": summary, "functions": funcs}, indent=1)
                                                        + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
