#!/usr/bin/env python3
"""`tools/move/adapt_map.py MAP LAYOUT_MAP...` (STAGING ONLY): make a map written for the tree before the layout's moves
run after them. Its `[roots]` become the last layout map's `[roots] new` (old and new alike: the map renames no root),
and each path it names between quotes (an include, a `keep_in` or `word_ends` key) goes through the layout maps in
order. Writes MAP in place and prints each edit, for the map's owner. A glob stays as written."""
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L  # noqa: E402


def through(p, layouts):
    for lay in layouts:
        q = lay.file(p)
        if q is None:
            q, _ = lay.directory(p)
        p = q or p
    return p


def main(argv):
    path = Path(argv[0])
    datas = [tomllib.loads(Path(m).read_text()) for m in argv[1:]]
    layouts = [L.Layout(d) for d in datas]
    roots = datas[-1]["roots"]["new"]
    inline = "{ " + ", ".join(f'{k} = "{v}"' for k, v in roots.items()) + " }"
    out, edits, section = [], [], ""
    for line in path.read_text().split("\n"):
        if line.startswith("["):
            section = line.strip()
        new = line
        if section == "[roots]" and re.match(r"^(old|new) = \{", line):
            new = f"{line.split(' = ', 1)[0]} = {inline}"
        elif not line.lstrip().startswith("#") and section not in ("[paths]", "[retired]", "[modules]"):
            def sub(m):
                s = m.group(1)
                return m.group(0) if any(c in s for c in "*?[{ ") or "/" not in s else f'"{through(s, layouts)}"'
            new = re.sub(r'"([^"\\]+)"', sub, line)
        if new != line:
            edits.append(f"-{line[:400]}\n+{new[:400]}")
        out.append(new)
    path.write_text("\n".join(out))
    print("\n".join(edits) or "no edit")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
