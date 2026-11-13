"""Item 6 (a), (b): per file of circuit_check's src, lines, function-level imports (classified), `Any` and capital aliases."""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

SRC = Path(sys.argv[1])
FILES = sorted(SRC.glob("*.py"))
CAP = re.compile(r"^[A-Z][A-Z0-9_]*$")
SERVED = ("verity.definitions.registry", "verity.definitions.numerics", "verity.definitions.boolean")
STDLIB = set(sys.stdlib_module_names)


def top_level_imports(tree):
    """Intra-package modules a file imports at its top level (not inside a def or class)."""
    out = set()
    for n in tree.body:
        for m in ast.walk(n) if isinstance(n, (ast.If, ast.Try, ast.With)) else [n]:
            if isinstance(m, ast.ImportFrom) and m.level == 1:
                out |= {m.module} if m.module else {a.name for a in m.names}
    return out


trees = {p.stem: ast.parse(p.read_text()) for p in FILES}
graph = {k: top_level_imports(t) for k, t in trees.items()}


def reaches(a, b):
    """Whether module a imports b at the top level, transitively."""
    seen, todo = set(), [a]
    while todo:
        x = todo.pop()
        if x == b:
            return True
        if x in seen:
            continue
        seen.add(x)
        todo += graph.get(x, ())
    return False


def func_imports(tree):
    """(node, enclosing function name) for every import inside a def or class body."""
    out = []

    def visit(node, inside):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.Import, ast.ImportFrom)) and inside:
                out.append((ch, inside))
            visit(ch, inside or (ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None))

    visit(tree, None)
    return out


def classify(stem, node):
    mods = ([node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names])
    if isinstance(node, ast.ImportFrom) and node.level:
        targets = [node.module] if node.module else [a.name for a in node.names]
        if any(reaches(t, stem) for t in targets):
            return "lazy:cycle"
        return "hoist:intra"
    m = mods[0]
    top = m.split(".")[0]
    if any(m == s or m.startswith(s + ".") for s in SERVED):
        return "cir269"
    if top == "verity_flock":
        return "verity_flock"
    if stem == "cache" and top not in STDLIB:
        return "lazy:cache-imports-nothing-of-the-tree"
    if m.startswith("verity.definitions") or m.startswith("verity.protocols") or m.startswith("verity.commitments"):
        return "lazy:registers-definitions"
    if top in STDLIB:
        return "hoist:stdlib"
    if top == "verity":
        return "hoist:verity-language"
    return f"other:{top}"


rows = {}
total = Counter()
classes = Counter()
by_class_file = {}
for p in FILES:
    text = p.read_text()
    tree = trees[p.stem]
    lines = text.count("\n")
    fi = func_imports(tree)
    anys = sum(1 for n in ast.walk(tree) if (isinstance(n, ast.Name) and n.id == "Any")
               or (isinstance(n, ast.Attribute) and n.attr == "Any"))
    caps = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names if a.asname and CAP.match(a.asname))
    cls = Counter(classify(p.stem, n) for n, _f in fi)
    for c, k in cls.items():
        by_class_file.setdefault(c, Counter())[p.name] += k
    classes += cls
    rows[p.name] = {"lines": lines, "func_imports": len(fi), "any": anys, "cap_aliases": caps}
    total += Counter(rows[p.name])
print(json.dumps({"files": rows, "total": dict(total), "classes": dict(classes),
                  "by_class_file": {c: dict(v) for c, v in by_class_file.items()}}, indent=1))
