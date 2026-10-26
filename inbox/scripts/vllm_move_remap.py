#!/usr/bin/env python3
"""`vllm_move_remap.py MAP FILE...`: carry a branch's vLLM lint allowlists across the layout move (#3854), in place.

MAP is the move's map (`git show origin/cursor/vllm-arch-map-496a:tools/move/vllm_map.toml > /tmp/vllm_map.toml`). Each
FILE is an allowlist (`entries[].file`, relative to integrations/vllm/) or `tests/dead_code_keep.json` (`modules` keys,
dotted). Every pre-move path or module is rewritten to where the move put it; the file keeps its format
(`merge_allowlist.dump`). Run it from the repository root after `git merge origin/main`, on the allowlists your branch
added or changed, then run their lint test files.
"""
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

R = "integrations/vllm/"


def mod(p):
    p = p.rstrip("/")
    p = p[:-12] if p.endswith("/__init__.py") else p[:-3] if p.endswith(".py") else p
    return p.replace("/", ".")


def main(map_path, files):
    paths = tomllib.load(open(map_path, "rb"))["paths"]
    rel = {o[len(R):]: n[len(R):] for o, n in paths.items() if o.startswith(R)}
    dirs = sorted(((o, n) for o, n in rel.items() if o.endswith("/")), key=lambda x: -len(x[0]))
    mods = sorted(((mod(o), mod(n)) for o, n in rel.items() if o.startswith("verity_vllm/") and mod(o) != mod(n)),
                  key=lambda x: -len(x[0]))

    def path(p):
        if p in rel:
            return rel[p]
        if p + "/" in rel:
            return rel[p + "/"].rstrip("/")
        return next((n + p[len(o):] for o, n in dirs if p.startswith(o)), p)

    def module(m):
        return next((n + m[len(o):] for o, n in mods if m == o or m.startswith(o + ".")), m)

    spec = importlib.util.spec_from_file_location("merge_allowlist", "integrations/vllm/tests/lint/merge_allowlist.py")
    ma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ma)
    for f in files:
        doc = json.loads(Path(f).read_text())
        n = 0
        for e in doc.get("entries") or []:
            new = path(e["file"])
            n += new != e["file"]
            e["file"] = new
        if isinstance(doc.get("modules"), dict):
            new = {module(k): v for k, v in doc["modules"].items()}
            n += sum(a != b for a, b in zip(doc["modules"], new))
            doc["modules"] = new
        Path(f).write_text(ma.dump(doc) if hasattr(ma, "dump") else json.dumps(doc, indent=1) + "\n")
        print(f"{f}: {n} renamed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
