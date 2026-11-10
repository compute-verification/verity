"""names_check_globs.py WORKTREE BASE: for each map names_rehearse.sh ran in BASE..HEAD of WORKTREE, on the commit it
ran on: every glob it names (scoped tables' files, keep_in, keep_paths, the [rewrite] excludes and includes, retired
paths) with the number of tracked files it matches there, and every kept string with the number of files that spell
it. A glob or a kept string that matches nothing is printed as STALE: neither tool refuses one, so each is read by
hand."""

import fnmatch
import re
import subprocess
import sys
import tomllib


def git(*a: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", sys.argv[1], *a], capture_output=True, text=True, check=check).stdout


def match(path: str, glob: str) -> bool:
    return fnmatch.fnmatchcase(path, glob) or (glob.startswith("**/") and fnmatch.fnmatchcase(path, glob[3:]))


stale = 0
for line in git("log", "--reverse", "--format=%H\t%s\t%b%x00", f"{sys.argv[2]}..HEAD").split("\0"):
    sha, _, message = line.strip("\n").partition("\t")
    m = re.search(r"tools/move/(\w+_map)\.toml", message)
    if not m or message.startswith("tools/move:"):
        continue
    name, rev = m[1], git("rev-parse", f"{sha}^").strip()
    files = git("ls-tree", "-r", "--name-only", rev).split("\n")
    data = tomllib.loads(git("show", f"{rev}:tools/move/{name}.toml"))
    rows = [(f"words_in[{i}].files", g) for i, t in enumerate(data.get("words_in", [])) for g in t["files"]]
    rows += [(sec, g) for sec in ("keep_in", "keep_paths") for g in data.get(sec, {})]
    rw = data.get("rewrite", {})
    rows += [("rewrite." + k, g) for k in ("exclude", "include") for g in rw.get(k, [])]
    rows += [("rewrite.keep_in", g) for g in rw.get("keep_in", {})]
    rows += [("retired", g + "**") for g in data.get("retired", {}).get("paths", [])]
    print(f"== {name} on {rev[:12]}")
    for sec, g in rows:
        n = sum(match(p, g) for p in files if p)
        flag = "" if n else "   <-- STALE: matches no tracked file"
        stale += not n
        print(f"  {sec:<20} {g:<75} {n:>6}{flag}")
    for s in data.get("keep", {}).get("strings", []):
        found = git("grep", "-l", "-I", "-F", "-e", s, rev, "--", ".", ":!tools/move", check=False)
        n = len([x for x in found.split("\n") if x])
        flag = "" if n else "   <-- STALE: no file spells it"
        stale += not n
        print(f"  {'keep.strings':<20} {s[:75]:<75} {n:>6}{flag}")
print(f"\n{stale} stale")
