"""Which kernels of one build of `session.cu` are, instruction for instruction, kernels of another: `cuobjdump -sass` of
each library, split at its `Function :` lines. A kernel is the lines under its own, so one that only changed its name
(a kernel made a template's instance) still counts as the same.

    python same_sass.py BASE.so NEW.so [--out FILE] [--listings DIR]

Prints how many of BASE's kernels NEW holds and names the ones it does not; FILE gets those, the kernels NEW holds
under another name, and NEW's kernels BASE has none equal to. DIR gets, for each of BASE's kernels NEW does not hold
(the --listings-max shortest), its listing beside NEW's kernel of that name (a template's instance at 16 rows first)
and their diff. Exits 0 whatever it finds, 2 if a library lists no kernel: `rows.sh` records what it prints.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def kernels(lib: str) -> dict[str, list[str]]:
    """`lib`'s kernels by mangled name: each one's disassembly, a line an entry."""
    text = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=True).stdout
    out: dict[str, list[str]] = {}
    body = None
    for line in text.splitlines():
        if "Function :" in line:
            body = out.setdefault(line.split("Function :", 1)[1].strip(), [])
        elif body is not None:
            body.append(line.rstrip())
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--listings", type=Path, metavar="DIR")
    ap.add_argument("--listings-max", type=int, default=6)
    a = ap.parse_args()
    base_text, new_text = kernels(a.base), kernels(a.new)
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
    print(f"{len(same)} of BASE's {len(base)} kernels are in NEW as they were ({len(doc['renamed'])} under a new name); "
          f"changed: {', '.join(doc['changed']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
