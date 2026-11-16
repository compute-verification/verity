#!/usr/bin/env python3
"""Every word pair (old = new) of every word map that ever ran: a map's [words] table and its [[words_in]] tables
(words applied only in listed files), each read at the commit that added the map; plus the maps in tools/move/ now,
read from the worktree (or from --ref REF for a pull request's head, whose added maps it lists with `gh`-less git diff).
Run from a checkout root: python3 maps_words.py OUT.tsv [--ref REF --base BASE]
Columns: map, commit, scope (words | words_in:<n files>), old, new, in_tree_now."""
import subprocess, sys, tomllib, re
from pathlib import Path

def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False).stdout

def parse(txt):
    try:
        return tomllib.loads(txt)
    except tomllib.TOMLDecodeError:
        # TOML 1.1 inline tables span lines; fold the lines of a brace that is still open
        out, depth = [], 0
        for line in txt.split("\n"):
            if depth > 0:
                out[-1] += " " + line.strip()
            else:
                out.append(line)
            code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line.split("#")[0] if not line.lstrip().startswith('"') else line)
            depth += code.count("{") - code.count("}")
        return tomllib.loads("\n".join(out))

def pairs(d):
    for old, new in (d.get("words") or {}).items():
        yield "words", old, str(new)
    for w in d.get("words_in") or []:
        files = w.get("files") or []
        scope = "words_in:" + ";".join(files)
        for old, new in (w.get("words") or {}).items():
            yield scope, old, str(new)

def main():
    args = sys.argv[1:]
    out = args[0]
    ref = base = None
    if "--ref" in args:
        ref = args[args.index("--ref") + 1]; base = args[args.index("--base") + 1]
    rows = []
    if ref:
        names = git("diff", "--name-only", "--diff-filter=A", f"{base}...{ref}", "--", "tools/move/*_map.toml").split()
        for p in names:
            for s, o, n in pairs(parse(git("show", f"{ref}:{p}"))):
                rows.append((Path(p).name, ref[:12], s, o, n, "pr"))
    else:
        added, c = [], None
        for l in git("log", "--diff-filter=A", "--name-only", "--format=C %H", "--", "tools/move/*_map.toml").split("\n"):
            if l.startswith("C "): c = l[2:]
            elif l.strip(): added.append((c, l.strip()))
        now = set(git("ls-files", "tools/move/*_map.toml").split())
        seen = set()
        for c, p in added:
            try: d = parse(git("show", f"{c}:{p}"))
            except Exception as e: print("unparsable", p, c[:10], e, file=sys.stderr); continue
            seen.add(p)
            for s, o, n in pairs(d):
                rows.append((Path(p).name, c[:12], s, o, n, "yes" if p in now else "no"))
        for p in sorted(now - seen):
            for s, o, n in pairs(parse(Path(p).read_text())):
                rows.append((Path(p).name, "worktree", s, o, n, "yes"))
    with open(out, "w") as f:
        f.write("map\tcommit\tscope\told\tnew\tin_tree_now\n")
        for r in rows: f.write("\t".join(r) + "\n")
    print(len(rows), "pairs from", len({r[0] for r in rows}), "maps;", len({r[3] for r in rows}), "distinct old")
if __name__ == "__main__":
    main()
