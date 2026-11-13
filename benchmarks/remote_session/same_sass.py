"""Which kernels of one build of `session.cu` are, instruction for instruction, kernels of another: `cuobjdump -sass` of
each library, split at its `Function :` lines. A kernel is the lines under its own, so one that only changed its name
(a kernel made a template's instance) still counts as the same.

    python same_sass.py BASE.so NEW.so [--out FILE] [--listings DIR] [--ptxas LOG]

Prints how many of BASE's kernels NEW holds and names the ones it does not; FILE gets those, the kernels NEW holds
under another name, and NEW's kernels BASE has none equal to. DIR gets, for each of BASE's kernels NEW does not hold
(the --listings-max shortest), its listing beside NEW's kernel of that name (a template's instance at 16 rows first)
and their diff, and the listing of each kernel BASE has none equal to. With LOG, what nvcc said building NEW (`-Xptxas -v`), FILE also gets the registers and the spilled
bytes of each kernel BASE has none equal to, and a line names them. Exits 0 only when NEW holds every kernel of
BASE's, 1 when it does not or, with LOG, when a kernel only NEW has spills or is not in LOG, 2 if cuobjdump fails on a
library (with what it said, on one line) or a library lists no kernel: the job that built them (`host_read.sh`)
records what it prints and runs nothing unless it exits 0.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def kernels(lib: str) -> dict[str, list[str]]:
    """`lib`'s kernels by mangled name: each one's disassembly, a line an entry, its runs of blanks made one (the
    column cuobjdump puts the encodings in is as wide as the library's longest instruction)."""
    done = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True)
    if done.returncode:
        said = " ".join(((done.stderr or "") + " " + (done.stdout or "")[-2000:]).split())
        raise RuntimeError(f"cuobjdump -sass {lib} exits {done.returncode}: {said[:600] or 'it said nothing'}")
    text = done.stdout
    out: dict[str, list[str]] = {}
    body = None
    for line in text.splitlines():
        if "Function :" in line:
            body = out.setdefault(line.split("Function :", 1)[1].strip(), [])
        elif body is not None:
            body.append(" ".join(line.split()))
    return out


def built(log: str) -> dict[str, dict]:
    """Each kernel's registers and spilled bytes (stores and loads) from nvcc's `-Xptxas -v` output, by mangled
    name."""
    out: dict[str, dict] = {}
    entry = of = None
    for line in log.splitlines():
        if "Compiling entry function '" in line:
            entry = line.split("'")[1]
        elif "Function properties for " in line:
            of = line.split("Function properties for ", 1)[1].strip()
        elif "bytes spill stores" in line and of is not None:
            stack, stores, loads = (int(x) for x in re.findall(r"(\d+) bytes", line))
            out.setdefault(of, {})["spill_bytes"] = stores + loads
        elif "Used " in line and " registers" in line and entry is not None:
            out.setdefault(entry, {})["registers"] = int(line.split("Used ", 1)[1].split()[0])
    return out


def stem(name: str) -> str:
    """A kernel's own name in its mangled one: the last length-prefixed identifier before its arguments."""
    i, last = (3 if name.startswith("_ZN") else 2), name
    while i < len(name) and name[i].isdigit():
        j = i
        while name[j].isdigit():
            j += 1
        last, i = name[j:j + int(name[i:j])], j + int(name[i:j])
    return last


def shown(name: str) -> str:
    """A kernel's own name, with its argument if it is a template's instance at one integer."""
    own = stem(name)
    at = re.match(r"ILi(\d+)E", name[name.index(own) + len(own):])
    return f"{own}<{at[1]}>" if at else own


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--listings", type=Path, metavar="DIR")
    ap.add_argument("--listings-max", type=int, default=6)
    ap.add_argument("--ptxas", type=Path, metavar="LOG")
    a = ap.parse_args()
    try:
        base_text, new_text = kernels(a.base), kernels(a.new)
    except RuntimeError as e:
        print(f"no comparison: {e}", file=sys.stderr)
        return 2
    if not base_text or not new_text:
        print(f"no kernel listed: {len(base_text)} in {a.base}, {len(new_text)} in {a.new}", file=sys.stderr)
        return 2
    base, new = ({name: hashlib.sha256("\n".join(lines).encode()).hexdigest() for name, lines in text.items()}
                 for text in (base_text, new_text))
    held = {digest: name for name, digest in new.items()}
    same = {name: name if new.get(name) == digest else held[digest] for name, digest in base.items() if digest in held}
    doc = {"base_kernels": len(base), "new_kernels": len(new), "same": len(same),
           "renamed": {name: to for name, to in sorted(same.items()) if name != to},
           "changed": sorted(name for name in base if name not in same),
           "only_new": sorted(name for name, digest in new.items() if digest not in set(base.values()))}
    bad = []
    if a.ptxas:
        said = built(a.ptxas.read_text())
        doc["only_new_built"] = {name: said.get(name, {}) for name in doc["only_new"]}
        bad = sorted(name for name, b in doc["only_new_built"].items()
                     if b.get("spill_bytes") != 0 or "registers" not in b)
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1))
    if a.listings and doc["changed"]:
        a.listings.mkdir(parents=True, exist_ok=True)
        for name in sorted(doc["changed"], key=lambda k: len(base_text[k]))[:a.listings_max]:
            twins = sorted((k for k in new if stem(k) == stem(name)), key=lambda k: (k != name, "Li16E" not in k, k))
            was, now = base_text[name], new_text[twins[0]] if twins else []
            at = a.listings / f"{stem(name)}-{base[name][:8]}"
            at.with_suffix(".base.sass").write_text("\n".join(was) + "\n")
            at.with_suffix(".new.sass").write_text("\n".join(now) + "\n")
            at.with_suffix(".diff").write_text(
                "\n".join(difflib.unified_diff(was, now, name, twins[0] if twins else "(none)", lineterm="")) + "\n")
    if a.listings and doc["only_new"]:
        a.listings.mkdir(parents=True, exist_ok=True)
        for name in doc["only_new"]:
            (a.listings / (shown(name).replace("<", "-").replace(">", "") + ".only-new.sass")).write_text(
                "\n".join(new_text[name]) + "\n")
    print(f"{len(same)} of BASE's {len(base)} kernels are in NEW as they were ({len(doc['renamed'])} under a new name); "
          f"changed: {', '.join(doc['changed']) or 'none'}")
    if a.ptxas:
        print("only NEW's: " + ("; ".join(
            f"{shown(name)} {b.get('registers', 'no')} registers, "
            f"{b.get('spill_bytes', 'unknown')} spilled bytes" for name, b in doc["only_new_built"].items()) or "none")
              + (f"; refused: {', '.join(bad)}" if bad else ""))
    return 1 if doc["changed"] or bad else 0


if __name__ == "__main__":
    sys.exit(main())
