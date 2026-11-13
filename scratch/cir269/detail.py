import ast, sys
from collections import Counter, defaultdict
from pathlib import Path
SRC = Path(sys.argv[1]); trees = {p.stem: ast.parse(p.read_text()) for p in sorted(SRC.glob("*.py"))}
def intra(n): return ([n.module] if n.module else [a.name for a in n.names]) if isinstance(n, ast.ImportFrom) and n.level == 1 else []
full = defaultdict(set)
lazy = []
for s, t in trees.items():
    def visit(node, inside):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.Import, ast.ImportFrom)):
                for m in intra(ch): full[s].add(m)
                if inside: lazy.append((s, ch))
            visit(ch, inside or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    visit(t, False)
def reaches(a, b, g):
    seen, todo = set(), [a]
    while todo:
        x = todo.pop()
        if x == b: return True
        if x in seen: continue
        seen.add(x); todo += g.get(x, ())
    return False
c = Counter(); mods = Counter(); where = defaultdict(Counter)
for s, n in lazy:
    ts = intra(n)
    if ts:
        k = "intra-cycle(full graph)" if any(reaches(t, s, full) for t in ts) else "intra-hoist"
        c[k] += 1; where[k][f"{s}->{','.join(ts)}"] += 1
    else:
        m = n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
        if m.startswith("verity.") or m == "verity": mods[(s == "cache", ".".join(m.split(".")[:3]))] += 1
print(c); print(dict(where))
for (iscache, m), k in sorted(mods.items()): print("cache" if iscache else "     ", m, k)

# greedy: hoist each lazy intra edge (cache.py's kept lazy by its rule) unless it closes a cycle with what is top-level so far
top = defaultdict(set)
for s, t in trees.items():
    for n in t.body:
        for m in intra(n): top[s].add(m)
kept, hoisted = Counter(), Counter()
for s, n in sorted(lazy, key=lambda x: (x[0], x[1].lineno)):
    for b in intra(n):
        if s == "cache" or reaches(b, s, top):
            kept[f"{s}->{b}"] += 1
        else:
            hoisted[f"{s}->{b}"] += 1
            top[s].add(b)
print("kept lazy", sum(kept.values()), dict(kept))
print("hoisted", sum(hoisted.values()), dict(hoisted))
