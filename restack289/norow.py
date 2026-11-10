#!/usr/bin/env python3
"""`norow.py BASE META.json [LABEL=COMMIT...]`: for each open pull request in META (and each LABEL=COMMIT), the files its
head adds over its merge base with BASE that BASE doesn't hold, which a layout map of BASE leaves under a path it retires: `layout.py` refuses such a
tree ("tracked files under a retired path with no row in the map"), so `restack.py` returns `script-failed`. Reads only
git objects; BASE's `tools/move/layout.py` is loaded for its `Layout`, and each map is read as BASE holds it.
The second map runs on the first's result, so a file is taken through the first before the second is asked."""
import json
import subprocess
import sys
import tomllib
import types

MAPS = ("tools/move/layout_one_map.toml", "tools/move/security_dir_map.toml")


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def main() -> int:
    base, meta, extra = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3:]
    mod = types.ModuleType("layout_at_base")
    mod.__file__ = "tools/move/layout.py"
    exec(compile(git("show", f"{base}:tools/move/layout.py"), "layout.py", "exec"), mod.__dict__)
    layouts = [mod.Layout(tomllib.loads(git("show", f"{base}:{m}"))) for m in MAPS]
    have = set(git("ls-tree", "-r", "--name-only", base).splitlines())
    heads = [(f"#{r['pr']}", r["head"]) for r in meta["rows"]] + [tuple(x.split("=", 1)) for x in extra]
    out = {}
    for label, head in heads:
        left = []
        fork = git("merge-base", head, base).strip()
        added = git("diff", "--no-renames", "--name-only", "--diff-filter=A", fork, head).splitlines()
        for p in sorted(set(added) - have):
            q = p
            for m, lay in zip(MAPS, layouts):
                new = lay.file(q)
                if new is None and q not in lay.drop and (q.startswith(lay.retired) or q in lay.retired):
                    left.append({"file": p, "map": m, "as": q})
                    break
                q = new or q
        if left:
            out[label] = left
            print(label, head[:12], *(f"\n    {x['file']}  [{x['map'].rsplit('/', 1)[1]}]" for x in left))
    json.dump(out, open("/tmp/rs/norow.json", "w"), indent=1)
    print(f"{len(out)} of {len(heads)} heads hold a file with no row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
