#!/usr/bin/env python3
"""`merge_allowlist.py BASE OURS THEIRS [PATH]`: git's merge driver for the integration's shrink-only lists, the rule lints'
`allowlists/*.json` and `tests/dead_code_keep.json` (`.gitattributes`: `merge=vllm-allowlist`), so two PRs that each retire an
entry merge even when the entries sit on neighbouring lines.

An allowlist's entries are merged by key, `(file, kind, symbol, detail)` as `_ratchet.load_allowlist` reads them, and the keep
map's `modules` by module: an entry either side added, changed (its `count`) or removed takes that side's change, and one both
sides set to different values, or that one side removed and the other changed, refuses the merge.  The repository's lists
(`tests/lint/allowlists/*.json`, `.gitattributes` names this driver for them too) count and allow more than a tree finds: the
ruff list's findings per `(file, rule)`, an entry `{file, rule, count}`, and the environment-read list's reads per file,
`{file, count}`.  A count both sides changed takes the larger (a removed entry counts 0), which allows at least what either
side's tree finds.  The environment-read list's settings registry marks, `{file, registry: true}`, merge like any entry, and
P16's `public` table (each command run from outside the integration, with the file that runs it) by command, as the keep map
by module.  Entries keep OURS' order, with THEIRS' additions after them.
Exit 0: merged, written to OURS in the file's own format.  Exit 1: refused, the entries named on stderr and OURS left as it
was, which git reports as a conflict.  A merged list is a candidate: the lints judge it on the merged tree, so an entry kept
wrongly fails as stale and one dropped wrongly as a new violation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KEY = ("file", "kind", "symbol", "detail")
#: the repository lists' keys, each entry a count: the ruff list's per (file, rule), the environment-read list's per file
COUNTED = (("file", "rule"), ("file",))
#: the tables merged by key: the keep map's ``modules``, P16's ``public`` (each command run from outside the integration, and
#: the file that runs it)
TABLES = ("modules", "public")
MISSING = object()


def one(label: str, o, a, b, why: list[str], counted: bool = False):
    if a == b:
        return a
    if a == o:
        return b
    if b == o:
        return a
    if counted:
        return {**(b if a is MISSING else a), "count": max(0 if x is MISSING else x["count"] for x in (a, b))}
    why.append(f"{label}: " + ("removed on one side and changed on the other" if MISSING in (a, b) else
                               "added on both sides, differently" if o is MISSING else "changed on both sides, differently"))
    return a


def entry_key(e) -> tuple | None:
    """An entry's key: ``KEY``'s fields; ``("count", *fields)`` for a count keyed by one of ``COUNTED``; ``("registry",
    file)`` for a settings registry mark; None for none."""
    if not isinstance(e, dict):
        return None
    if all(isinstance(e.get(f), str) for f in KEY):
        return tuple(e[f] for f in KEY)
    for fields in COUNTED:
        if set(e) == {*fields, "count"} and all(isinstance(e[f], str) for f in fields) and type(e["count"]) is int:
            return ("count", *(e[f] for f in fields))
    if set(e) == {"file", "registry"} and isinstance(e["file"], str) and e["registry"] is True:
        return ("registry", e["file"])
    return None


def is_count(k: tuple) -> bool:
    return len(k) < len(KEY) and k[0] == "count"


def label(k: tuple) -> str:
    if len(k) == len(KEY):
        return f"{k[0]} [{k[1]}] {k[2]}: {k[3]}"
    return k[1] + "".join(f" [{x}]" for x in k[2:]) + (" (a settings registry)" if k[0] == "registry" else "")


def entries(side: str, value) -> dict:
    """An allowlist's entries by key."""
    if value is MISSING:
        return {}
    if not isinstance(value, list):
        raise ValueError(f"{side}: `entries` is not a list")
    out = {}
    for e in value:
        if (k := entry_key(e)) is None:
            raise ValueError(f"{side}: malformed entry {e}")
        if k in out:
            raise ValueError(f"{side}: duplicate entry {e}")
        out[k] = e
    return out


def by_key(o: dict, a: dict, b: dict, label, why: list[str], counted=lambda k: False) -> dict:
    out = {}
    for k in list(a) + [k for k in b if k not in a]:
        v = one(label(k), o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING), why, counted(k))
        if v is not MISSING:
            out[k] = v
    return out


def merge(base: dict, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """(the merged document, why it is refused: empty when it merged)."""
    why: list[str] = []
    out = {}
    for f in list(ours) + [f for f in theirs if f not in ours]:
        o, a, b = (d.get(f, MISSING) for d in (base, ours, theirs))
        if f == "entries" and MISSING not in (a, b):
            m = by_key(entries("base", o), entries("ours", a), entries("theirs", b), label, why, counted=is_count)
            out[f] = list(m.values())
        elif f in TABLES and all(isinstance(x, dict) for x in (a, b)) and (o is MISSING or isinstance(o, dict)):
            out[f] = by_key({} if o is MISSING else o, a, b, lambda k, f=f: f"{f}.{k}", why)
        else:
            v = one(f, o, a, b, why)
            if v is not MISSING:
                out[f] = v
    return out, why


def dump(doc: dict) -> str:
    """An allowlist as the lints' files are written (one `_ratchet.entry_line` per entry, a table such as P16's `public` one
    item a line), the keep map at indent 1."""
    if "entries" not in doc:
        return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    lines, keys = ["{"], list(doc)
    for i, k in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        if k == "entries" and doc[k]:
            es = doc[k]
            lines += ['  "entries": [', *(f"    {json.dumps(e, ensure_ascii=False)}{',' if j < len(es) - 1 else ''}"
                                          for j, e in enumerate(es)), f"  ]{comma}"]
        elif k != "entries" and isinstance(doc[k], dict) and doc[k]:
            rows = [f"    {json.dumps(n, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}" for n, v in doc[k].items()]
            lines += [f"  {json.dumps(k)}: {{", *(f"{r}," for r in rows[:-1]), rows[-1], f"  }}{comma}"]
        else:
            lines.append(f"  {json.dumps(k)}: {json.dumps(doc[k], ensure_ascii=False)}{comma}")
    return "\n".join(lines) + "\n}\n"


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    base, ours, theirs = (Path(p) for p in argv[:3])
    path = argv[3] if len(argv) == 4 else ours
    try:
        docs = [json.loads(p.read_text(encoding="utf-8") or "{}") for p in (base, ours, theirs)]
        if not all(isinstance(d, dict) for d in docs):
            raise ValueError("a side is not a JSON object")
        merged, why = merge(*docs)
    except ValueError as e:
        why = [str(e)]
    if why:
        print(f"merge_allowlist.py: {path}: refused:", file=sys.stderr)
        print("\n".join(f"  {w}" for w in why), file=sys.stderr)
        return 1
    ours.write_text(dump(merged), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
