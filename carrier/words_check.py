"""The words check of a carried head, with the queue's own reading of a move's map.

    uv run python words_check.py MAIN OLD_HEAD CARRIED [--lines N]

For each move MAIN has and OLD_HEAD lacks (`research.queue.moves_between`), its map read at its base as the queue reads it
(`map_retires`): the names and words the queue finds in OLD_HEAD's own change (`lacked_moves`: what its refusal text
names), and those it finds in CARRIED's own change (`own_text`: the files and the added lines of `git diff MAIN...CARRIED`).
The queue matches a path or dotted module as written and a word in any case as a substring, so each hit in CARRIED is then
read again as the rename tool reads it: as a whole word, and only in a file the map's own entry names (`[[words_in]]`'s
`files`; `[words]` is every file), with the lines.
"""
import fnmatch
import re
import sys
import tomllib
from pathlib import Path

from research import queue as q

args = sys.argv[1:]
show = 6
if "--lines" in args:
    i = args.index("--lines")
    show = int(args[i + 1])
    del args[i:i + 2]
main, old, new = args
repo = Path(".")


def added(main, head):
    out = q._out(repo, "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", f"{main}...{head}")
    f = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            f = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and f is not None:
            yield f, line[1:]


moves = q.moves_between(repo, main, old)
print(f"moves {main[:12]} has and the old head {old[:12]} lacks: {len(moves)}; moves the carried head {new[:12]} lacks: "
      f"{len(q.moves_between(repo, main, new))}")
old_text, new_text = q.own_text(repo, main, old), q.own_text(repo, main, new)
new_lines = list(added(main, new))
new_files = q.own_files(repo, main, new)
print(f"the carried head's own change over main: {len(new_files)} files, {len(new_lines)} added lines")
total_q = total_w = 0
for commit, parents in moves:
    body = q._out(repo, "log", "-1", "--format=%B", commit)
    m = q.MOVE_LINE_RE.search(body)
    base = parents[0] if parents else f"{commit}^"
    got = q.map_retires(repo, base, m[1])
    name = m[1].rsplit("/", 1)[-1]
    if got is None:
        print(f"MOVE {commit[:12]} {name}: the map can't be read at its base")
        continue
    was, now = got.named(old_text), got.named(new_text)
    if not was and not now:
        continue
    total_q += len(now)
    print(f"MOVE {commit[:12]} {name}: the queue's reading names {len(was)} in the old head's change {was[:8]}; "
          f"{len(now)} in the carried head's change {now[:8]}")
    if not now:
        continue
    data = tomllib.loads(q._out(repo, "show", f"{base}:{m[1]}"))
    scopes = [(None, set(data.get("words") or {}))] + [(s.get("files") or [], set((s.get("words") or {})))
                                                        for s in data.get("words_in") or []]
    for hit in now:
        if hit in got.names:
            lines = [(f, t) for f, t in new_lines if hit in t] + [(f, "(the file's own path)") for f in new_files if hit in f]
            # a new path that begins with the old one is no old name
            news = [n for o, n in (data.get("paths") or {}).items() if isinstance(n, str) and o.rstrip("/") == hit]
            real = [(f, t) for f, t in lines
                    if not any(n.rstrip("/") in t or n.rstrip("/") in f for n in news if n.rstrip("/").startswith(hit))]
            # as a path or module standing by itself, not inside a longer one (FRAME's pattern)
            pat = re.compile(r"(?<![\w./-])" + re.escape(hit) + r"(?![\w-])")
            alone = [(f, t) for f, t in real if pat.search(t) or pat.search(f)]
            total_w += len(alone)
            print(f"   name `{hit}`: {len(lines)} lines or paths hold it as a substring; {len(alone)} hold it standing by itself")
            for f, t in alone[:show]:
                print(f"      {f}: {t.strip()[:170]}")
            continue
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(hit) + r"(?![A-Za-z0-9_])", re.I)
        whole = [(f, t) for f, t in new_lines if pat.search(t)]
        cased = [w for _, ws in scopes for w in ws if w.lower() == hit]
        scoped = []
        for f, t in whole:
            for files, ws in scopes:
                for w in ws:
                    if w.lower() != hit:
                        continue
                    exact = re.search(r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])", t)
                    if exact and (files is None or any(fnmatch.fnmatch(f, g) or f == g or f.startswith(g.rstrip("/") + "/") for g in files)):
                        scoped.append((f, t, w))
        total_w += len(scoped)
        print(f"   word `{hit}` (the map's spelling {sorted(set(cased))[:4]}): {sum(hit in t.lower() for _, t in new_lines)} added lines hold it as a "
              f"substring, {len(whole)} as a whole word in any case, {len(scoped)} as the map spells it in a file its entry names")
        for f, t, w in scoped[:show]:
            print(f"      IN SCOPE {f}: {t.strip()[:170]}")
        for f, t in [x for x in whole if x not in [(a, b) for a, b, _ in scoped]][:show]:
            print(f"      out of the entry's files, or another case: {f}: {t.strip()[:170]}")
print(f"TOTAL: {total_q} names or words by the queue's reading; {total_w} by the rename tool's reading (whole, as spelled, in an entry's files)")
