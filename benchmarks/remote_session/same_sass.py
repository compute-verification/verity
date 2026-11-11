"""Which kernels of one build of `session.cu` are, instruction for instruction, kernels of another: `cuobjdump -sass` of
each library, split at its `Function :` lines. A kernel is the lines under its own, so one that only changed its name
(a kernel made a template's instance) still counts as the same.

    python same_sass.py BASE.so NEW.so [--out FILE]

Prints how many of BASE's kernels NEW holds and names the ones it does not; FILE gets those, the kernels NEW holds
under another name, and NEW's kernels BASE has none equal to. Exits 0 whatever it finds, 2 if a library lists no
kernel: `rows.sh` records what it prints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def kernels(lib: str) -> dict[str, str]:
    """`lib`'s kernels by mangled name: the sha256 of each one's disassembly."""
    text = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=True).stdout
    out: dict[str, list[str]] = {}
    body = None
    for line in text.splitlines():
        if "Function :" in line:
            body = out.setdefault(line.split("Function :", 1)[1].strip(), [])
        elif body is not None:
            body.append(line.rstrip())
    return {name: hashlib.sha256("\n".join(lines).encode()).hexdigest() for name, lines in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    base, new = kernels(a.base), kernels(a.new)
    if not base or not new:
        print(f"no kernel listed: {len(base)} in {a.base}, {len(new)} in {a.new}", file=sys.stderr)
        return 2
    held = {digest: name for name, digest in new.items()}
    same = {name: name if new.get(name) == digest else held[digest] for name, digest in base.items() if digest in held}
    doc = {"base_kernels": len(base), "new_kernels": len(new), "same": len(same),
           "renamed": {name: to for name, to in sorted(same.items()) if name != to},
           "changed": sorted(name for name in base if name not in same),
           "only_new": sorted(name for name, digest in new.items() if digest not in set(base.values()))}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1))
    print(f"{len(same)} of BASE's {len(base)} kernels are in NEW as they were ({len(doc['renamed'])} under a new name); "
          f"changed: {', '.join(doc['changed']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
