"""Each `[rewrite] exclude` entry of MAP that matches no tracked file of REPO, by rename.py's own matcher."""
import importlib.util
import subprocess
import sys
import tomllib

repo, path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("rename_mod", f"{repo}/tools/move/rename.py")
src = open(f"{repo}/tools/move/rename.py").read()
match_src = src[src.index("def _match"):]
ns: dict = {}
exec(compile("import fnmatch, re\n" + match_src[:match_src.index("\ndef ", 5)], "match", "exec"), ns)
files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
ex = tomllib.load(open(path, "rb"))["rewrite"]["exclude"]
dead = [g for g in ex if not any(ns["_match"](f, g) for f in files)]
print(len(ex), "entries;", len(dead), "match nothing")
for g in dead:
    print(" ", g)
