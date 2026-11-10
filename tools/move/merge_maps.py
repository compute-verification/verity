#!/usr/bin/env python3
"""`tools/move/merge_maps.py OUT.toml MAP...`: the layout's maps, in the order they run, as one map (STAGING ONLY, a test).

Each of the layout's maps names today's paths only, so their rows are one map's rows as they stand: `[paths]`, `[tokens]`,
`[split]`, `[retired]`, `[drop]` and `[rewrite]` are the union, `[roots]` is the first map's `old` and the last map's
`new`, and `[modules]` is what `layout.py` derives from the merged `[paths]` and `[roots]`. A row two maps give
differently stops it. The one map answers whether the layout stages as one move, and its tree against the chain's says
whether the order of the maps changes anything.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("layout", HERE / "layout.py")
LY = importlib.util.module_from_spec(spec)
sys.modules["layout"] = LY
spec.loader.exec_module(LY)


def q(s: str) -> str:
    return json.dumps(s)


def main(argv: list[str]) -> int:
    out, maps = Path(argv[0]), argv[1:]
    docs = [tomllib.loads(Path(m).read_text()) for m in maps]
    merged: dict = {"paths": {}, "tokens": {}, "split": {}, "create": {}}
    lists: dict = {"retired": [], "drop": []}
    rewrite: dict = {"exclude": [], "include": [], "keep": [], "keep_dotted": [], "keep_in": {}}
    for m, d in zip(maps, docs):
        for table in merged:
            for k, v in d.get(table, {}).items():
                if merged[table].setdefault(k, v) != v:
                    raise SystemExit(f"{m}: [{table}] {k!r} = {v!r}, and an earlier map gives {merged[table][k]!r}")
        for table in lists:
            lists[table] += [p for p in d.get(table, {}).get("paths", []) if p not in lists[table]]
        for k, v in d.get("rewrite", {}).items():
            if k == "keep_in":
                for f, ps in v.items():
                    rewrite["keep_in"].setdefault(f, [])
                    rewrite["keep_in"][f] += [p for p in ps if p not in rewrite["keep_in"][f]]
            elif k in rewrite:
                rewrite[k] += [x for x in v if x not in rewrite[k]]
            else:
                raise SystemExit(f"{m}: [rewrite] {k}: a key this script doesn't merge")
        if extra := set(d) - {"roots", "paths", "tokens", "split", "create", "retired", "drop", "rewrite", "modules"}:
            raise SystemExit(f"{m}: tables this script doesn't merge: {sorted(extra)}")
    roots = {"old": dict(docs[0]["roots"]["old"]), "new": dict(docs[0]["roots"]["old"])}
    for d in docs:
        roots["new"].update(d["roots"]["new"])
    modules = LY.derived_modules(LY.Layout({"paths": merged["paths"], "roots": roots}))
    given: dict = {}
    for d in docs:
        given.update(d.get("modules", {}))
    lines = [f"# STAGING ONLY, a test: the layout's {len(maps)} maps as one, written by tools/move/merge_maps.py from",
             f"# {maps[0]} .. {maps[-1]}.", "", "[roots]"]
    for side in ("old", "new"):
        lines.append(f"{side} = {{ " + ", ".join(f"{n} = {q(p)}" for n, p in sorted(roots[side].items())) + " }")
    for table in ("paths", "tokens", "split", "create"):
        if merged[table] or table == "paths":
            lines += ["", f"[{table}]"] + [f"{q(k)} = {q(v)}" for k, v in sorted(merged[table].items())]
    for table, paths in lists.items():
        lines += ["", f"[{table}]", "paths = [" + ", ".join(q(p) for p in paths) + "]"]
    lines += ["", "[rewrite]"]
    for k in ("exclude", "include", "keep", "keep_dotted"):
        if rewrite[k]:
            lines.append(f"{k} = [" + ", ".join(q(x) for x in rewrite[k]) + "]")
    if rewrite["keep_in"]:
        lines.append("keep_in = { " + ", ".join(f"{q(f)} = [{', '.join(q(p) for p in ps)}]"
                                                 for f, ps in sorted(rewrite["keep_in"].items())) + " }")
    lines += ["", "[modules]"] + [f"{q(k)} = {q(v)}" for k, v in sorted(modules.items())]
    out.write_text("\n".join(lines) + "\n")
    only_chain = {k: v for k, v in given.items() if modules.get(k) != v}
    only_one = {k: v for k, v in modules.items() if given.get(k) != v}
    print(f"{out}: {len(merged['paths'])} rows, {len(merged['tokens'])} tokens, {len(merged['split'])} split rows, "
          f"{len(modules)} module renames from {len(maps)} maps")
    print(f"module renames the maps give that the one map doesn't, or gives differently: {len(only_chain)}")
    for k, v in sorted(only_chain.items())[:40]:
        print(f"  {k} -> {v} (one map: {modules.get(k)})")
    print(f"module renames only the one map gives: {len(only_one)}")
    for k, v in sorted(only_one.items())[:40]:
        print(f"  {k} -> {v} (the maps: {given.get(k)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
