"""python3 importcheck.py TREE PREFIX: every `from PREFIX... import X` in TREE's tracked .py names a module file that
binds X (def, class, assignment, import, or a submodule), and every `import PREFIX.a.b` names a file. Prints breaks."""
import ast, subprocess, sys
from pathlib import Path

tree, prefix = Path(sys.argv[1]), sys.argv[2]
files = subprocess.run(["git", "-C", str(tree), "ls-files", "*.py"], capture_output=True, text=True).stdout.split()


def modfile(mod):
    p = tree / mod.replace(".", "/")
    for c in (p.with_suffix(".py"), p / "__init__.py"):
        if c.exists():
            return c
    return None


bound_cache = {}


def bound(f):
    if f not in bound_cache:
        names, star = set(), False
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
                for t in ast.walk(n):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        names.add(t.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    if a.name == "*":
                        star = True
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.Global):
                names.update(n.names)
        bound_cache[f] = (names, star)
    return bound_cache[f]


breaks = 0
for rel in files:
    try:
        t = ast.parse((tree / rel).read_text())
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        continue
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.startswith(prefix):
            f = modfile(n.module)
            if f is None:
                print(f"{rel}:{n.lineno}: no module {n.module}"); breaks += 1; continue
            names, star = bound(f)
            for a in n.names:
                if a.name != "*" and a.name not in names and not star and modfile(f"{n.module}.{a.name}") is None:
                    print(f"{rel}:{n.lineno}: {n.module} binds no {a.name}"); breaks += 1
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith(prefix) and modfile(a.name) is None:
                    print(f"{rel}:{n.lineno}: no module {a.name}"); breaks += 1
print(f"{breaks} breaks")
