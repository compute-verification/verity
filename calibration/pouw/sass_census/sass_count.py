"""The SASS opcodes a binary holds, counted by architecture: the census diff_menu.py diffs against A7's menu.

`count` reads `cuobjdump -sass` (or `nvdisasm`) text on stdin and writes one JSON object: for each architecture a
section names (`arch = sm_90a`, `code for sm_100a`), its functions, its instructions, and each dotted opcode's count
with its first instruction as an example (`IMMA.16832.S8.S8`: `R4, R8, R6, R4`). `merge` folds those objects into one
census an architecture: every dotted opcode, its count, the sources it appears in, an example, and its base opcode.
`units`, `join`, `version` and `key` are dump.sh's: a binary split into one disassembly an architecture, its parts
joined into the whole binary's count, and where that count is kept between runs.

    cuobjdump -sass lib.so | python sass_count.py count --source libcublasLt > lt.json
    python sass_count.py merge counts/*.json > census.json
    python sass_count.py summary census.json

Stdlib only, so it runs on a node's system Python beside the toolkit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
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


def count(lines: Iterable[str], source: str, only: str | None = None) -> dict:
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


def elfs(lelf: Iterable[str]) -> list[tuple[str, int]]:
    """The architectures a `cuobjdump -lelf` listing names, in the order of their first ELF, each with its ELF count."""
    n: dict[str, int] = {}
    for line in lelf:
        for arch in re.findall(r"sm_[0-9]*[a-z]?", line):
            n[arch] = n.get(arch, 0) + 1
    return list(n.items())


def units(size: int, lelf: Iterable[str], allowed: set[str]) -> list[tuple[int, str | None]]:
    """How dump.sh disassembles one binary of `size` bytes: one `cuobjdump -sass -arch A` a listed architecture, each
    weighed by its share of the ELFs, or the whole binary at once (None) when it lists one architecture or one that
    `cuobjdump -arch` doesn't take (`allowed`)."""
    archs = elfs(lelf)
    if len(archs) < 2 or any(a not in allowed for a, _ in archs):
        return [(size, None)]
    total = sum(k for _, k in archs)
    return [(size * k // total, a) for a, k in archs]


def join(parts: list[dict], source: str, order: list[str]) -> dict:
    """One binary's `count` from its architectures' (each `count --arch A` of `cuobjdump -sass -arch A`), the
    architectures in `order`, the order a whole-binary dump meets them, so it equals the whole binary's `count`."""
    archs: dict[str, dict] = {}
    for p in parts:
        for arch, a in p["archs"].items():
            if arch in archs:
                raise ValueError(f"{source}: {arch} is in two parts")
            archs[arch] = a
    rank = {a: i for i, a in enumerate(order)}
    return {"source": source, "archs": dict(sorted(archs.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])))}


def version(files: list[Path], tool: str) -> str:
    """The dump's version: SHA-256 over the counting code's files and the disassembler's `--version` text, so a count
    made by other code or another cuobjdump is never read back from the cache."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    h.update(tool.encode())
    return h.hexdigest()


def key(ver: str, sha256: str, source: str) -> str:
    """A binary's count's path in the cache: the dump's version, the binary's SHA-256 and the source name its count
    carries."""
    if not re.fullmatch(r"[0-9a-f]{64}", ver) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("version and sha256 are 64 hex digits each")
    return f"{ver}/{sha256}/{source.replace('/', '_')}.json"


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
    u = sub.add_parser("units", help="a binary's units of disassembly from its `cuobjdump -lelf` listing on stdin")
    u.add_argument("--size", type=int, required=True)
    u.add_argument("--allowed", type=Path, required=True, help="the architectures `cuobjdump -arch` takes, one a line")
    j = sub.add_parser("join", help="one binary's count from its architectures' counts")
    j.add_argument("--source", required=True)
    j.add_argument("--lelf", type=Path, required=True, help="the binary's `cuobjdump -lelf` listing")
    j.add_argument("files", type=Path, nargs="+")
    v = sub.add_parser("version", help="the dump's version")
    v.add_argument("--tool", required=True, help="cuobjdump --version's text")
    v.add_argument("files", type=Path, nargs="+")
    k = sub.add_parser("key", help="a count's path in the cache")
    k.add_argument("version")
    k.add_argument("sha256")
    k.add_argument("source")
    m = sub.add_parser("merge")
    m.add_argument("files", type=Path, nargs="+")
    s = sub.add_parser("summary")
    s.add_argument("census", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "count":
        json.dump(count(sys.stdin, a.source, a.arch), sys.stdout)
    elif a.cmd == "units":
        allowed = set(a.allowed.read_text().split())
        for w, arch in units(a.size, sys.stdin, allowed):
            print(f"{w}\t{arch or '-'}")
    elif a.cmd == "join":
        order = [arch for arch, _ in elfs(a.lelf.read_text().splitlines())]
        json.dump(join([json.loads(f.read_text()) for f in a.files], a.source, order), sys.stdout)
    elif a.cmd == "version":
        print(version(a.files, a.tool))
    elif a.cmd == "key":
        print(key(a.version, a.sha256, a.source))
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
