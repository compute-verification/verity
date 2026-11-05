"""Write a split map: ordered operations, each with the layout it applies to (a split's declarations, the modules they use,
the modules that must import the new module and those that stop importing the source; a merge's module and destination).
Imports are by use (the constant graph), the shaken graph the plan assumed.

    python3 splitmap.py --repo R --deps deps.tsv.gz --plan PLAN.json --times T.json --decls trace.json --floor F \
        [--first N] --out MAP.toml
    python3 splitmap.py --repo R --deps deps.tsv.gz --refresh MAP.toml --out MAP.toml

From `--plan` (plan.py's operations) it names each new module; with `--refresh` it takes an existing map's operations and
names. Either way each split's declarations are rederived from R's constant graph (`--deps`, Extract.lean and refs.py
on R): what
`for_importer` uses of `source`, closed under uses inside `source`, as plan.py chose them. An operation whose module isn't in
R, or whose importer now needs none or all of its source, is left out and said so, with every later operation on what it
would have made. The model's path (`--times`, `--decls`) goes to stdout, never into the map."""
import argparse, json, re, sys, tomllib
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--deps", required=True)
ap.add_argument("--out", required=True)
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--plan")
src.add_argument("--refresh")
ap.add_argument("--times", help="JSON {module: seconds}: only for the model's path")
ap.add_argument("--decls", help="parse_trace.py's JSON: only for the model's path")
ap.add_argument("--floor", type=float, default=4.5)
ap.add_argument("--first", type=int)
ap.add_argument("--root", default="Proofs.Flock")
a = ap.parse_args()
sys.argv = ["x", a.repo]
sys.path.insert(0, str(Path(__file__).parent))
import graph
from plan import Plan, load_deps, longest

mods, consts = load_deps(Path(a.deps))
mtime = json.loads(Path(a.times).read_text()) if a.times else {}
tr = json.loads(Path(a.decls).read_text()) if a.decls else {}
dtime = {(m, n): sum(v.values()) for m, r in tr.items() for n, v in r["decls"].items()}
cl = graph.closure(a.root)
P = Plan(mods, consts, mtime, dtime, a.floor, graph.imports, cl)
P._all = defaultdict(list)
for k in P.unit:
    P._all[k[0]].append(k)
order = defaultdict(dict)
for cid, mi, kind, name, deps in consts:
    order[mods[mi]].setdefault(name, cid)

if a.plan:
    ops = [{"kind": o["op"], "source": o["module"], "for_importer": o.get("for_importer"), "into": o.get("into")}
           for o in json.loads(Path(a.plan).read_text())["splits"]]
else:
    ops = [{"kind": o["kind"], "source": o.get("source", o.get("module")), "for_importer": o.get("for_importer"),
            "into": o.get("into"), "new": o.get("new")} for o in tomllib.loads(Path(a.refresh).read_text())["op"]]
ops = ops[:a.first]


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
inv = {}
U = lambda n: inv.get(n, n)
imp = {m: set(graph.imports(m)) for m in cl}


def reach(m):
    """`m` and every module its imports reach, in `imp`: leansplit's imports after the operations so far."""
    seen, stack = set(), [m]
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            stack.extend(imp.get(x, ()))
    return seen
if mtime:
    t0, _ = longest(a.root, P.deps, P.weight)
body, kept, splits = [], 0, 0
for i, op in enumerate(ops):
    unit = U(op["source"])
    if op["kind"] == "merge":
        A, B = unit, U(op["into"])
        if not P.members.get(A) or not P.members.get(B):
            print(f"op {i}: merge {op['source']} into {op['into']}: {'source' if not P.members.get(A) else 'destination'} "
                  f"not in the tree: left out")
            continue
        body += ["[[op]]", 'kind = "merge"', f'module = "{N(A)}"', f'into = "{N(B)}"', ""]
        P.merge(A, B)
        a_, b_ = N(A), N(B)
        imp[b_] = (imp.get(b_, set()) | imp.pop(a_, set())) - {a_, b_}
        for m, s in imp.items():
            if a_ in s:
                s.discard(a_)
                if m != b_:
                    s.add(b_)
        kept += 1
        continue
    up = U(op["for_importer"])
    if not P.members.get(unit) or not P.members.get(up):
        print(f"op {i}: split {op['source']} for {op['for_importer']}: "
              f"{'source' if not P.members.get(unit) else 'importer'} not in the tree: left out")
        continue
    need = P.need(unit, up)
    if not need or need == P.members[unit]:
        print(f"op {i}: split {op['source']} for {op['for_importer']}: the importer needs "
              f"{'none' if not need else 'all'} of it: left out")
        continue
    lo, rest, dl, dr, imps = P.try_split(unit, need)
    P.apply(unit, need, lo, rest, dl, dr, imps)
    if op.get("new"):
        nm = op["new"]
    else:
        parent, last = P.base[unit].rsplit(".", 1)
        stem = f"{parent}.{last}{camel(sorted(n for _, n in need)[0])}"
        nm, j = stem, 2
        while nm in used:
            nm, j = f"{stem}{j}", j + 1
    used.add(nm)
    names[lo], inv[nm] = nm, lo
    decls = sorted(need, key=lambda k: (k[0], order[k[0]].get(k[1], 1 << 30)))
    importers = sorted(N(u) for u, (x, _) in imps.items() if x)
    drop = sorted(u for u, (x, y) in imps.items() if x and not y)
    body += ["[[op]]", 'kind = "split"', f'source = "{N(unit)}"', f'new = "{nm}"', f'for_importer = "{N(up)}"',
             f"decls = {q(k[1] for k in decls)}", f"imports = {q(sorted(N(u) for u in dl))}",
             f"external_imports = {q(sorted(frontier(sorted({k[0] for k in need}))))}",
             f"source_imports_new = {'true' if lo in dr else 'false'}", f"importers = {q(importers)}",
             f"drop_source = {q(N(u) for u in drop)}"]
    if drop:
        body.append("[op.uses]")
        body += [f'"{N(u)}" = {q(sorted(N(d) for d in P.deps[u]))}' for u in drop]
    body.append("")
    imp[nm] = {N(u) for u in dl}
    if lo in dr:
        imp.setdefault(N(unit), set()).add(nm)
    for i in importers:
        imp.setdefault(i, set()).add(nm)
    for u in drop:
        s = imp.setdefault(N(u), set())
        s.discard(N(unit))
        r = reach(N(u))
        for d in sorted(N(x) for x in P.deps[u]):
            if d not in r:
                s.add(d)
                r |= reach(d)
    kept += 1
    splits += 1
if mtime:
    t, _ = longest(a.root, P.deps, P.weight)
    print(f"model: shaken path {t0 / 60:.2f} min, {t / 60:.2f} after the {kept} operations")
repair = {}
for u, ds in P.deps.items():
    m = N(u)
    if m in imp and (miss := sorted({N(d) for d in ds} - reach(m) - {m})):
        repair[m] = miss
body += ["[repair]"] + [f'"{m}" = {q(ds)}' for m, ds in sorted(repair.items())] + [""]
head = [f"# The {a.root} split map: {kept} operations ({splits} splits, {kept - splits} merges), applied in order.",
        "# kind = \"split\": `decls` (each with its docstring, attributes and equation lemmas) leave `source` for `new`;",
        "# `imports` are the in-tree modules they use (the constant graph: shake's minimal imports) and `external_imports`",
        "# the untracked modules the source's closure imports, a superset for shake to trim. `importers` use one of `decls`",
        "# and import `new`; `drop_source` use nothing else of `source` and stop importing it, importing instead the",
        "# modules their constants use (`uses`) that they reached only through it. `source_imports_new` when the rest of",
        "# `source` uses `decls`. kind = \"merge\": `module` goes to the top of `into`, and its importers import `into`.",
        "# `[repair]`: the modules of `root`'s closure whose imports, after the operations, no longer reach some of the",
        "# modules their constants use, and those modules, which they import directly.",
        "# `python3 tools/move/split_plan/splitmap.py --refresh` rederives it on a tree (`split_plan/refresh.sh`).", "",
        f'root = "{a.root}"', ""]
Path(a.out).write_text("\n".join(head + body))
print(f"wrote {a.out}: {kept} of {len(ops)} operations, {len(repair)} modules to repair")
