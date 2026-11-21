#!/usr/bin/env python3
"""The references to the protocol groups that `tools/move/layout.py` doesn't read, rewritten after it, from the root of a
moved checkout: names written from `verity/` or `verity/protocols/` (`protocols/compose/...`, `proofs/spot/...`,
`compose.schemes`, `resource_accounting.network`), pathlib and `os.path.join` parts, deploy and `../` prefixes, Python
names in Lean comments, and the docs' lone `proofs/`. The groups, the files kept and the strings a file keeps (`keep_in`)
are protocol_names_map.toml's.

    python3 hand.py MAP [--files FILE]    rewrite every tracked text file (or those FILE lists), print each it changed
"""
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

args = sys.argv[1:]
m = tomllib.loads(Path(args[0]).read_text())
only = Path(args[args.index("--files") + 1]).read_text().split() if "--files" in args else None
P = "verity/protocols/"
NAMES = {o[len(P):-1]: n[len(P):-1] for o, n in m["paths"].items() if o.startswith(P)}
KEEP = m["rewrite"]["exclude"] + [args[0], "tools/move/**"]
CHILD = {o: sorted(c.stem for c in Path(P + n).iterdir() if not c.name.startswith("_")) for o, n in NAMES.items()}
alt = "|".join(sorted(NAMES, key=len, reverse=True))
RULES = [
    # a directory tree's entry, its description kept at the column the tree's other entries use
    (re.compile(rf"(?m)^(    )({alt})/( {{2,}})(?=\S)"), lambda g: f"{g[1]}{NAMES[g[2]]}/".ljust(len(g[1]) + max(15, len(NAMES[g[2]]) + 3))),
    # verity.protocols.OLD, protocols/OLD, a regex's protocols\.OLD, after any prefix (/opt/vy-*/current/, ../../)
    (re.compile(rf"(?<![\w-])protocols(/|\.|\\\.)({alt})(?![\w-])"), lambda g: f"protocols{g[1]}{NAMES[g[2]]}"),
    # "protocols" / "OLD" and "protocols", "OLD"
    (re.compile(rf"""(["'])protocols\1(\s*[/,]\s*)(["'])({alt})\3"""), lambda g: f"{g[1]}protocols{g[1]}{g[2]}{g[3]}{NAMES[g[4]]}{g[3]}"),
    # OLD/child and OLD.child, written from verity/protocols/
    *[(re.compile(rf"(?<![\w/.-]){o}([./])({'|'.join(CHILD[o])})(?![\w-])"), (lambda o: lambda g: f"{NAMES[o]}{g[1]}{g[2]}")(o))
      for o in NAMES if CHILD[o]],
    # names that are only ever these directories'
    *[(re.compile(rf"(?<![\w-]){o}(?!\w)"), (lambda o: lambda g: NAMES[o])(o)) for o in NAMES if "_" in o],
    # a doc's lone `OLD/`
    (re.compile(rf"`({alt})/`"), lambda g: f"`{NAMES[g[1]]}/`"),
]

files = only or subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
for f in files:
    p = Path(f)
    if any(fnmatch.fnmatch(f, k) for k in KEEP) or not p.is_file() or p.is_symlink() or p.stat().st_size > 4 << 20:
        continue
    raw = p.read_bytes()
    if b"\0" in raw[:8192]:
        continue
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        print(f"hand.py: {f} is not UTF-8, so its lines are left as they are", file=sys.stderr)
        continue
    kept = [s for g, ss in m["rewrite"].get("keep_in", {}).items() if fnmatch.fnmatch(f, g) for s in ss]
    new = text
    for i, s in enumerate(kept):
        new = new.replace(s, f"\0{i}\0")
    for rx, to in RULES:
        new = rx.sub(lambda mt: to((mt.group(0), *mt.groups())), new)
    for i, s in enumerate(kept):
        new = new.replace(f"\0{i}\0", s)
    if new != text:
        p.write_text(new)
        print(f)

# its lists sort by path, as tests/test_interfaces_ratchet.py's dump writes them
ratchet = Path("tests/interfaces_ratchet.json")
text = ratchet.read_text()
new = json.dumps(json.loads(text), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
if new != text:
    ratchet.write_text(new)
    print(ratchet)
