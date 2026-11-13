"""A screen for names a carried head still uses that the chain renamed elsewhere.

    python3 dangling.py MAIN OLD_HEAD CARRIED

`rename.py` renames a name only in the files a map's entry lists (`[[words_in]]`), so a file a branch adds keeps a call to a
declaration the move renamed where it is declared: the carry ends `clean`, the map's script changes nothing more, and the
file does not build. For each word move MAIN has and OLD_HEAD lacks, this takes every old name of the map that is one
identifier (no space), from `[words]` and every `[[words_in]]` entry whatever its files, and looks for it as a whole
identifier in the lines CARRIED adds over MAIN, outside the files that entry lists. It prints each one with how many lines of
MAIN's code still hold the old and the new identifier: an old name MAIN no longer holds, whose new name it does, is a call
that will not resolve. A screen only: the build and the suites are the check.
"""
import re
import subprocess
import sys
import tomllib
from fnmatch import fnmatch

IDENT = re.compile(r"^[A-Za-z_][\w.]*$")


def git(*a, check=True):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=check).stdout


def added(main, carried):
    f = None
    for raw in git("diff", "-U0", "--no-renames", "--no-color", main, carried).split("\n"):
        if raw.startswith("+++ "):
            f = raw[6:] if raw.startswith("+++ b/") else None
        elif raw.startswith("+") and f is not None:
            yield f, raw[1:]


def count(main, name):
    """Lines of MAIN's Lean and Python that hold ``name`` as a whole identifier (its last part, for a dotted one)."""
    last = name.rsplit(".", 1)[-1]
    out = git("grep", "-c", "-w", "-F", last, main, "--", "*.lean", "*.py", check=False)
    return sum(int(line.rsplit(":", 1)[-1]) for line in out.splitlines() if line.strip())


def main():
    main_sha, old, carried = sys.argv[1:4]
    lines = list(added(main_sha, carried))
    log = git("log", "--topo-order", "--reverse", "--no-merges", "-E", "--grep=^Move: tools/move/", "--format=%H%x00%B%x1e",
              main_sha, f"^{old}")
    flagged = 0
    seen = set()
    for rec in filter(None, (s.strip() for s in log.split("\x1e"))):
        move, _, body = rec.partition("\x00")
        maps = [ln.split()[1] for ln in body.splitlines() if ln.startswith("Move: tools/move/")]
        if not maps:
            continue
        data = tomllib.loads(git("show", f"{move}:{maps[0]}"))
        if data.get("scope") == "prose":
            print(f"MOVE {move[:12]} {maps[0]}: prose only, renames no name")
            continue
        tables = [(["**"], data.get("words", {}))] + [(e["files"], e["words"]) for e in data.get("words_in", [])]
        hits = []
        for globs, table in tables:
            for o, n in table.items():
                if not IDENT.match(o) or not IDENT.match(n):
                    continue
                last = o.rsplit(".", 1)[-1]
                pat = re.compile(rf"(?<![\w']){re.escape(last)}(?![\w'])")
                for f, text in lines:
                    if not f.endswith((".lean", ".py")) or any(fnmatch(f, g) for g in globs):
                        continue
                    if pat.search(text) and (o, f, text) not in seen:
                        seen.add((o, f, text))
                        hits.append((o, n, f, text.strip()[:150]))
        print(f"MOVE {move[:12]} {maps[0]}: {len(hits)} added line(s) outside an entry's files hold an old identifier of the map")
        for o, n in sorted({(o, n) for o, n, _, _ in hits}):
            co, cn = count(main_sha, o), count(main_sha, n)
            gone = co == 0 and cn > 0
            flagged += gone
            print(f"   `{o}` -> `{n}`: main's code holds the old one on {co} line(s), the new on {cn}"
                  f"{'  <== DANGLING: the old name is gone from main' if gone else ''}")
            for oo, _, f, text in hits:
                if oo == o and (gone or co < 5):
                    print(f"        {f}: {text}")
    print(f"TOTAL names that look dangling: {flagged}")


main()
