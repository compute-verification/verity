#!/usr/bin/env python3
"""`merge_map.py BASE OURS THEIRS [PATH]`: git's merge driver for `fixtures/artifacts.json` (`.gitattributes`: `merge=fixture-map`),
the replica map every fixture PR adds keys to, so two PRs that each register a fixture merge.

The map is merged key by key, `previous` (each path's former ids) one level further: a key either side added, changed or removed
takes that side's change, and a key both sides set to different values, or that one side removed and the other changed, refuses
the merge.  So does a merge that leaves two ids naming one path (both sides refreshed the same fixture).  Exit 0: merged, written
to OURS as `research data refresh-fixtures` writes the map.  Exit 1: refused, the keys named on stderr and OURS left as it was,
which git reports as a conflict.  A merged map is a candidate: the repository's suite checks every entry hashes to its id
(`test_repo_replicas.py`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PREVIOUS = "previous"
MISSING = object()


def one(path: str, o, a, b, why: list[str]):
    if a == b:
        return a
    if a == o:
        return b
    if b == o:
        return a
    why.append(f"{path}: " + ("removed on one side and changed on the other" if MISSING in (a, b) else
                              "added on both sides, differently" if o is MISSING else "changed on both sides, differently"))
    return a


def entries(path: str, o: dict, a: dict, b: dict, why: list[str], nested: frozenset = frozenset()) -> dict:
    out = {}
    for k in list(a) + [k for k in b if k not in a]:
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if k in nested and all(isinstance(v, dict) for v in (y, z)) and (x is MISSING or isinstance(x, dict)):
            v = entries(f"{path}{k}.", {} if x is MISSING else x, y, z, why)
        else:
            v = one(f"{path}{k}", x, y, z, why)
        if v is not MISSING:
            out[k] = v
    return out


def merge(base: dict, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """(the merged map, why it is refused: empty when it merged)."""
    why: list[str] = []
    out = entries("", base, ours, theirs, why, frozenset({PREVIOUS}))
    seen: dict[str, str] = {}
    for k, v in out.items():
        if isinstance(v, str):
            if v in seen:
                why.append(f"{v}: named by both {seen[v]} and {k}")
            seen[v] = k
    return out, why


def dump(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != PREVIOUS}
    if doc.get(PREVIOUS):
        body[PREVIOUS] = doc[PREVIOUS]
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    base, ours, theirs = (Path(p) for p in argv[:3])
    path = argv[3] if len(argv) == 4 else ours
    try:
        docs = [json.loads(p.read_text() or "{}") for p in (base, ours, theirs)]
        if not all(isinstance(d, dict) for d in docs):
            raise ValueError("a side is not a JSON object")
        merged, why = merge(*docs)
    except ValueError as e:
        why = [str(e)]
    if why:
        print(f"merge_map.py: {path}: refused:", file=sys.stderr)
        print("\n".join(f"  {w}" for w in why), file=sys.stderr)
        return 1
    ours.write_text(dump(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
