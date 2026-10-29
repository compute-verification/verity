"""The SASS opcodes a binary holds, counted by architecture: the census RT-ISA diffs against A7's menu.

`count` reads `cuobjdump -sass` (or `nvdisasm`) text on stdin and writes one JSON object: for each architecture a
section names (`arch = sm_90a`, `code for sm_100a`, nvdisasm's `.target sm_90a`), its functions, its instructions, and
each dotted opcode's count with its first instruction as an example (`IMMA.16832.S8.S8`: `R4, R8, R12, R4`). `merge`
folds those objects into one census an architecture: every dotted opcode, its count, the sources it appears in, an
example, and its base opcode.

On `nvdisasm -plr -lrm narrow` text (a cubin's listing with its register life ranges) `count` also records what each
instruction writes, from the compiler's own dataflow: the registers its life-range columns mark `^` (written) or `x`
(read and written), counted in each section the function's header names (`GPR`, `PRED`, `UGPR`, `UPRED`). Each dotted
opcode then carries `writes`, its instructions' histogram over those counts (`GPR=2 PRED=1`: `n`), which `merge` sums.
A write the compiler never reads (a dead half, `PT`, `RZ`) has no column, so a form's counts are what its uses wrote.

    cuobjdump -sass lib.so | python sass_count.py count --source libcublasLt > lt.json
    nvdisasm -plr -lrm narrow k.sm_90a.cubin | python sass_count.py count --source k > k.json
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
ARCH = re.compile(r"^\s*(?:arch = |code for |\.target\s+)(sm_\d+[a-z]?)\b")
FUNC = re.compile(r"^\s*(?:Function : (\S+)|\.text\.(\S+):)")
#: the register sections an `nvdisasm -plr` header names, in its column order
SECTIONS = ("GPR", "PRED", "UGPR", "UPRED")


def life_writes(line: str, names: list[str]) -> str | None:
    """`GPR=2 PRED=1`: the registers an instruction's life-range columns mark written, by section (zeros left out)."""
    _, sep, ranges = line.partition("// |")
    if not sep or not names:
        return None
    cols = ranges.split("|")[:len(names)]
    counts = [(n, sum(c.count(ch) for ch in "^x")) for n, c in zip(names, cols)]
    return " ".join(f"{n}={k}" for n, k in counts if k) or "-"


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
    names: list[str] = []
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
                e = cur["ops"][op] = [0, operands[:160]]
            e[0] += 1
            if (w := life_writes(line, names)) is not None:
                if len(e) == 2:
                    e.append({})
                e[2][w] = e[2].get(w, 0) + 1
            continue
        if "// |" in line and "GPR" in line:
            names = [s.strip() for s in line.partition("// |")[2].split("|")[:-1]]
            unknown = set(names) - set(SECTIONS)
            if unknown:
                raise ValueError(f"life-range sections {sorted(unknown)}: SECTIONS names {SECTIONS}")
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
            for op, (n, ex, *hist) in a["ops"].items():
                e = fam["ops"].setdefault(op, {"count": 0, "sources": {}, "example": ex, "base": op.split(".")[0]})
                e["count"] += n
                e["sources"][src] = e["sources"].get(src, 0) + n
                for w, k in (hist[0] if hist else {}).items():
                    writes = e.setdefault("writes", {})
                    writes[w] = writes.get(w, 0) + k
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


#: 32-bit registers a warp instruction writes for each register a `writes` key counts: a GPR or a predicate a lane, a
#: uniform register or uniform predicate a warp
WARP_BITS = {"GPR": 32, "PRED": 1, "UGPR": 1, "UPRED": 1 / 32}


def lane_regs(key: str) -> float:
    """A `writes` key's 32-bit registers a warp instruction writes, over 32: a lane's registers' worth."""
    if key == "-":
        return 0.0
    return sum(WARP_BITS[n] * int(k) for n, k in (kv.split("=") for kv in key.split())) / 32


def writes(census: dict, fam: str, dispatch: float, unit: float) -> list[dict]:
    """A form a row: its count, sources, write histogram, its most registers a lane, and its dispatch floor, the least
    price per register it could have at `dispatch` warp instructions a SM a clock (`unit` over its registers a clock);
    None where its instructions carried no life ranges."""
    out = []
    for op, e in sorted(census[fam]["ops"].items()):
        hist = e.get("writes")
        most = max(map(lane_regs, hist), default=0.0) if hist else None
        floor = unit / (dispatch * 32 * most) if most else None
        out.append({"op": op, "base": e["base"], "count": e["count"], "sources": len(e["sources"]),
                    "writes": hist, "lane_regs": most, "dispatch_floor": floor, "example": e["example"]})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--source", required=True)
    c.add_argument("--arch", help="count only this architecture's sections")
    m = sub.add_parser("merge")
    m.add_argument("files", type=Path, nargs="+", help="count files, or directories of them (*.json)")
    s = sub.add_parser("summary")
    s.add_argument("census", type=Path)
    w = sub.add_parser("writes", help="a form a TSV line: its writes and dispatch floor")
    w.add_argument("census", type=Path)
    w.add_argument("--family", default="sm_90")
    w.add_argument("--dispatch", type=float, default=4, help="warp instructions a SM a clock (4 SMSPs, 1 each)")
    w.add_argument("--unit", type=float, default=4096, help="the price unit's rate a SM a clock (dense int8 MACs)")
    a = p.parse_args(argv)
    if a.cmd == "writes":
        print("op\tbase\tcount\tsources\tlane_regs\tdispatch_floor\twrites\texample")
        for r in writes(json.loads(a.census.read_text()), a.family, a.dispatch, a.unit):
            hist = ";".join(f"{k}:{n}" for k, n in sorted((r["writes"] or {}).items(), key=lambda kv: -kv[1]))
            fl = "" if r["dispatch_floor"] is None else f"{r['dispatch_floor']:.3f}"
            lr = "" if r["lane_regs"] is None else f"{r['lane_regs']:.4f}"
            print(f"{r['op']}\t{r['base']}\t{r['count']}\t{r['sources']}\t{lr}\t{fl}\t{hist}\t{r['example']}")
        return 0
    if a.cmd == "count":
        json.dump(count(sys.stdin, a.source, a.arch), sys.stdout)
    elif a.cmd == "merge":
        objs = []
        for f in (g for f in a.files for g in (sorted(f.glob("*.json")) if f.is_dir() else [f])):
            if f.stat().st_size:
                objs.append(json.loads(f.read_text()))
        json.dump(merge(objs), sys.stdout, indent=1, sort_keys=True)
    else:
        print(summary(json.loads(a.census.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
