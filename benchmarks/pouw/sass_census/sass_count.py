"""The SASS opcodes a binary holds, counted by architecture: the census diff_menu.py diffs against A7's menu.

`count` reads `cuobjdump -sass` (or `nvdisasm`) text on stdin and writes one JSON object: for each architecture a
section names (`arch = sm_90a`, `code for sm_100a`), its functions, its instructions, and each dotted opcode's count
with its first instruction as an example (`IMMA.16832.S8.S8`: `R4, R8, R12, R4`). `merge` folds those objects into one
census an architecture: every dotted opcode, its count, the sources it appears in, an example, and its base opcode.

    cuobjdump -sass lib.so | python sass_count.py count --source libcublasLt > lt.json
    python sass_count.py merge counts/*.json > census.json
    python sass_count.py summary census.json

Stdlib only, so it runs on a node's system Python beside the toolkit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: an instruction line: its address, an optional guard predicate, the dotted opcode and its operands up to the `;`
INS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(?:\{\s*)?(@!?U?P(?:[T0-9]+)\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Za-z0-9_]+)*)"
                 r"\s*([^;]*);")
ARCH = re.compile(r"^\s*(?:arch = |code for )(sm_\d+[a-z]?)\b")
FUNC = re.compile(r"^\s*Function : (\S+)")


def family(arch: str) -> str:
    """The device an architecture's SASS runs on: sm_90 and sm_90a cubins both run on the H200, sm_100, sm_100a and
    sm_100f on the B200, sm_120, sm_120a and sm_120f on the RTX PRO 6000. SASS doesn't run across minor versions."""
    m = re.fullmatch(r"sm_(\d+)[a-z]?", arch)
    return f"sm_{m.group(1)}" if m else arch


def count(lines, source: str, only: str | None = None) -> dict:
    """Each architecture's functions, instructions and dotted opcodes in a listing; with `only`, that architecture's
    sections alone (cuobjdump's `-arch` filter may also pass a sibling's)."""
    archs: dict[str, dict] = {}
    cur = None
    skip = False
    for line in lines:
        if "/*" in line and (m := INS.match(line)):
            if skip:
                continue
            if cur is None:
                cur = archs.setdefault(only or "unknown", {"funcs": 0, "instrs": 0, "ops": {}})
            op, operands = m.group(3), " ".join(m.group(4).split())
            cur["instrs"] += 1
            e = cur["ops"].get(op)
            if e is None:
                cur["ops"][op] = [1, operands[:160]]
            else:
                e[0] += 1
            continue
        if m := ARCH.match(line):
            skip = only is not None and m.group(1) != only
            if not skip:
                cur = archs.setdefault(m.group(1), {"funcs": 0, "instrs": 0, "ops": {}})
        elif FUNC.match(line) and cur is not None and not skip:
            cur["funcs"] += 1
    return {"source": source, "archs": archs}


def merge(objs: list[dict]) -> dict:
    """One census a device family: every dotted opcode's count, its sources and an example, and each base opcode's."""
    out: dict[str, dict] = {}
    for o in objs:
        src = o["source"]
        for arch, a in o["archs"].items():
            fam = out.setdefault(family(arch), {"archs": {}, "sources": {}, "funcs": 0, "instrs": 0, "ops": {}})
            fam["archs"][arch] = fam["archs"].get(arch, 0) + a["instrs"]
            fam["sources"][src] = fam["sources"].get(src, 0) + a["instrs"]
            fam["funcs"] += a["funcs"]
            fam["instrs"] += a["instrs"]
            for op, (n, ex) in a["ops"].items():
                e = fam["ops"].setdefault(op, {"count": 0, "sources": {}, "example": ex, "base": op.split(".")[0]})
                e["count"] += n
                e["sources"][src] = e["sources"].get(src, 0) + n
    for fam in out.values():
        bases: dict[str, dict] = {}
        for op, e in fam["ops"].items():
            b = bases.setdefault(e["base"], {"count": 0, "forms": 0, "sources": set()})
            b["count"] += e["count"]
            b["forms"] += 1
            b["sources"] |= set(e["sources"])
        fam["bases"] = {b: {"count": v["count"], "forms": v["forms"], "sources": sorted(v["sources"])}
                        for b, v in sorted(bases.items())}
    return out


def summary(census: dict) -> str:
    rows = ["family  instrs        funcs     sources  bases  dotted"]
    for fam, f in sorted(census.items()):
        rows.append(f"{fam:7s} {f['instrs']:>13,} {f['funcs']:>9,} {len(f['sources']):>8} {len(f['bases']):>6} "
                    f"{len(f['ops']):>7}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--source", required=True)
    c.add_argument("--arch", help="count only this architecture's sections")
    m = sub.add_parser("merge")
    m.add_argument("files", type=Path, nargs="+")
    s = sub.add_parser("summary")
    s.add_argument("census", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "count":
        json.dump(count(sys.stdin, a.source, a.arch), sys.stdout)
    elif a.cmd == "merge":
        objs = []
        for f in a.files:
            if f.stat().st_size:
                objs.append(json.loads(f.read_text()))
        json.dump(merge(objs), sys.stdout, indent=1, sort_keys=True)
    else:
        print(summary(json.loads(a.census.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
