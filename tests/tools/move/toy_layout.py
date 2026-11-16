"""A toy `tools/move/layout.py` with the real one's contract: run at the root of a checkout, it applies
`tools/move/layout_map.toml` to the tree, idempotently.

The map has `[paths]` (a file or directory, old = new), `[modules]` (a dotted import name, old = new) and `[create]` (a file and
its text, written when missing). Like the real one, it refuses a dirty checkout. Every tracked or untracked file under an old path moves; then every text file outside
`tools/move/` and `skip` (suffixes) has each old module name and old path replaced where it stands as a whole name. A file
named `LAYOUT_FAIL` at the root makes it fail (exit 1), for tests of a script that fails.
"""
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

if Path("LAYOUT_FAIL").exists():
    sys.exit("layout: LAYOUT_FAIL is present")
if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip():
    sys.exit("layout: the worktree isn't clean")
MAP = tomllib.loads(Path("tools/move/layout_map.toml").read_text())


def files() -> list[str]:
    p = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True, check=True)
    return [f for f in p.stdout.decode().split("\0") if f and os.path.lexists(f)]


for old, new in MAP.get("paths", {}).items():
    for f in files():
        if f == old or f.startswith(old + "/"):
            os.renames(f, new + f[len(old):])
for f, text in MAP.get("create", {}).items():
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
        Path(f).write_text(text)

subs = [(re.compile(rf"(?<![\w.]){re.escape(o)}(?![\w])"), n) for o, n in MAP.get("modules", {}).items()]
subs += [(re.compile(rf"(?<![\w./-]){re.escape(o)}(?![\w-]|\.\w)"), n) for o, n in MAP.get("paths", {}).items()]
skip = tuple(MAP.get("skip", []))
for f in files():
    if f.startswith("tools/move/") or f.endswith(skip) or os.path.islink(f):
        continue
    raw = Path(f).read_bytes()
    if b"\0" in raw:
        continue
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        continue
    new = text
    for pattern, repl in subs:
        new = pattern.sub(repl, new)
    if new != text:
        Path(f).write_bytes(new.encode())
