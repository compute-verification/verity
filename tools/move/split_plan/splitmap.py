"""Replay plan.py's operations and write them as a split map, in order, each with the layout it applies to: a split's
declarations, the modules they use, the modules that must import the new module and those that stop importing the
source; a merge's module and destination. Imports are by use (the constant graph), the shaken graph the plan assumed.

    python3 splitmap.py --repo R --deps deps.tsv.gz --times T.json --decls trace.json --floor F --plan PLAN.json \
        [--first N] --out MAP.toml
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
for f in ("--repo", "--deps", "--times", "--decls", "--plan", "--out"):
    ap.add_argument(f, required=True)
ap.add_argument("--floor", type=float, required=True)
ap.add_argument("--first", type=int)
ap.add_argument("--root", default="Proofs.Flock")
ap.add_argument("--base", default="", help="the commit the times were measured at, for the header")
a = ap.parse_args()
sys.argv = ["x", a.repo]
sys.path.insert(0, str(Path(__file__).parent))
import graph
from plan import Plan, load_deps, longest

mods, consts = load_deps(Path(a.deps))
mtime = json.loads(Path(a.times).read_text())
tr = json.loads(Path(a.decls).read_text())
dtime = {(m, n): sum(v.values()) for m, r in tr.items() for n, v in r["decls"].items()}
cl = graph.closure(a.root)
P = Plan(mods, consts, mtime, dtime, a.floor, graph.imports, cl)
P._all = defaultdict(list)
for k in P.unit:
    P._all[k[0]].append(k)
plan = json.loads(Path(a.plan).read_text())
ops = plan["splits"][:a.first]
order = defaultdict(dict)
for cid, mi, kind, name, deps in consts:
    order[mods[mi]].setdefault(name, cid)


def camel(n):
    last = n.split(".")[-1]
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", last) if w)


def frontier(srcs):
    """The untracked modules (Mathlib, Std, Lean, ArkLib) that the tracked modules of `srcs`' closure import directly:
    a superset of what a part of them needs beyond the constant graph, which shake then trims."""
    seen, stack, out = set(), list(srcs), set()
    while stack:
        m = stack.pop()
        if m in seen or m not in graph.MODS:
            continue
        seen.add(m)
        stack.extend(graph.imports(m))
        hs = graph.headers.imports(graph.MODS[m][0].read_text()) or []
        out |= {graph.headers.name(i) for i in hs if graph.headers.name(i) not in graph.MODS}
    return out


def q(xs):
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


names, used = {}, set(cl)
N = lambda u: names.get(u, u)
_, sh = longest(a.root, P.deps, P.weight)
body = []
for i, op in enumerate(ops):
    unit = op["module"]
    if op["op"] == "merge":
        A, B = unit, op["into"]
        body += ["[[op]]", f"# {i}: path {op['path_after_min']:.2f} min after", 'kind = "merge"', f'module = "{N(A)}"',
                 f'into = "{N(B)}"', f"saves_s = {op['saves_s']}", ""]
        P.merge(A, B)
        continue
    need = {k for k in P.members[unit] if k[1] in set(op["moves"])}
    assert need, op
    lo, rest, dl, dr, imps = P.try_split(unit, need)
    assert lo == op["new_part"], (lo, op["new_part"])
    P.apply(unit, need, lo, rest, dl, dr, imps)
    parent, last = P.base[unit].rsplit(".", 1)
    stem = f"{parent}.{last}{camel(sorted(op['moves'])[0])}"
    nm, j = stem, 2
    while nm in used:
        nm, j = f"{stem}{j}", j + 1
    used.add(nm)
    names[lo] = nm
    decls = sorted(need, key=lambda k: (k[0], order[k[0]].get(k[1], 1 << 30)))
    importers = sorted(N(u) for u, (x, _) in imps.items() if x)
    drop = sorted(u for u, (x, y) in imps.items() if x and not y)
    body += ["[[op]]", f"# {i}: path {op['path_after_min']:.2f} min after", 'kind = "split"', f'source = "{N(unit)}"',
             f'new = "{nm}"', f'for_importer = "{N(op["for_importer"])}"', f"saves_s = {op['saves_s']}",
             f"decls = {q(k[1] for k in decls)}", f"imports = {q(sorted(N(u) for u in dl))}",
             f"external_imports = {q(sorted(frontier(sorted({k[0] for k in need}))))}",
             f"source_imports_new = {'true' if lo in dr else 'false'}", f"importers = {q(importers)}",
             f"drop_source = {q(N(u) for u in drop)}"]
    if drop:
        body.append("[op.uses]")
        body += [f'"{N(u)}" = {q(sorted(N(d) for d in P.deps[u]))}' for u in drop]
    body.append("")
t, _ = longest(a.root, P.deps, P.weight)
print(f"replayed {len(ops)} operations: path {t / 60:.2f} min (plan: {ops[-1]['path_after_min']:.2f})")
splits = sum(o["op"] == "split" for o in ops)
head = [f"# The {a.root} split map: {len(ops)} operations ({splits} splits, {len(ops) - splits} merges), applied in order.",
        "# kind = \"split\": `decls` (each with its docstring, attributes and equation lemmas) leave `source` for `new`;",
        "# `imports` are the in-tree modules they use (the constant graph: shake's minimal imports) and `external_imports`",
        "# the untracked modules the source's closure imports, a superset for shake to trim. `importers` use one of `decls`",
        "# and import `new`; `drop_source` use nothing else of `source` and stop importing it, importing instead the",
        "# modules their constants use (`uses`) that they reached only through it. `source_imports_new` when the rest of",
        "# `source` uses `decls`. kind = \"merge\": `module` goes to the top of `into`, and its importers import `into`.",
        f"# Lake model{' at ' + a.base if a.base else ''}: path {plan['current_min']:.1f} min, {plan['shaken_min']:.1f} after"
        f" shake, {ops[-1]['path_after_min']:.1f} after these operations.", ""]
Path(a.out).write_text("\n".join(head + body))
print("wrote", a.out, len(head + body), "lines")
