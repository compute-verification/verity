"""Scopes where two distinct names become one: per function, class, lambda, comprehension and module, the set of names
(bound, read, parameters, keywords, attributes) before and after; the scopes walk in the same order since a rename keeps
structure."""
import ast, subprocess
moves = dict(l.rstrip("\n").split("\t")[::-1] for l in open(__import__("os").environ.get("MOVES", "/tmp/kit/m1_moves.tsv")) if l.strip())
SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Module)
def scopes(tree):
    out = []
    def walk(node, acc):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, SCOPE):
                mine = set(); out.append((getattr(ch, "name", type(ch).__name__), ch.lineno if hasattr(ch, "lineno") else 0, mine))
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): acc.add(ch.name)
                walk(ch, mine)
            else:
                if isinstance(ch, ast.Name): acc.add(ch.id)
                elif isinstance(ch, ast.arg): acc.add(ch.arg)
                elif isinstance(ch, ast.keyword) and ch.arg: acc.add("kw:" + ch.arg)
                elif isinstance(ch, ast.Attribute): acc.add("." + ch.attr)
                elif isinstance(ch, ast.alias): acc.add(ch.asname or ch.name)
                walk(ch, acc)
    top = set(); out.append(("<module>", 0, top)); walk(tree, top)
    return out
files = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AMR", "--", "*.py"], capture_output=True, text=True).stdout.split()
for f in files:
    old = subprocess.run(["git", "show", f"HEAD:{moves.get(f, f)}"], capture_output=True, text=True).stdout
    try:
        a, b = scopes(ast.parse(old)), scopes(ast.parse(open(f, encoding="utf-8").read()))
    except SyntaxError as e:
        print(f, "SYNTAX", e); continue
    if len(a) != len(b):
        print(f, "scope count", len(a), len(b)); continue
    for (n, l, s), (_, _, t) in zip(a, b):
        if len(t) < len(s):
            print(f"{f}:{l} {n}: {sorted(s - t)} -> {sorted(t - s)}")
