"""python3 ownercheck.py TREE BASE HEAD: for each Python name a rename commit changed, its kind where it stands (a keyword
argument, an attribute, a declaration: def, parameter, class-level or self. assignment), and whether its owner or its
readers outside the commit's files still say the old name. Prints the suspects."""
import ast, re, subprocess, sys
from collections import defaultdict

tree, base, head = sys.argv[1:4]
git = lambda *a: subprocess.run(["git", "-C", tree, *a], capture_output=True, text=True).stdout
changed = [f for f in git("diff", "--name-only", base, head).split() if f.endswith(".py")]
piece = set(git("diff", "--name-only", base, head).split())


def occ(src):
    out = defaultdict(list)
    try:
        t = ast.parse(src)
    except SyntaxError:
        return out
    for n in ast.walk(t):
        if isinstance(n, ast.keyword) and n.arg:
            out[n.lineno].append((n.col_offset, "kwarg", n.arg))
        elif isinstance(n, ast.Attribute):
            kind = "self-assign" if isinstance(n.ctx, ast.Store) else "attr"
            out[n.end_lineno].append((n.end_col_offset, kind, n.attr))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.lineno].append((n.col_offset, "def", n.name))
        elif isinstance(n, ast.arg):
            out[n.lineno].append((n.col_offset, "param", n.arg))
    for cls in [n for n in ast.walk(t) if isinstance(n, ast.ClassDef)]:
        for s in cls.body:
            tgts = s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, ast.AnnAssign) else []
            for g in tgts:
                if isinstance(g, ast.Name):
                    out[g.lineno].append((g.col_offset, "field", g.id))
    return {k: sorted(v) for k, v in out.items()}


renamed = []
for f in changed:
    a, b = occ(git("show", f"{base}:{f}")), occ(git("show", f"{head}:{f}"))
    for line in set(a) | set(b):
        x, y = a.get(line, []), b.get(line, [])
        if len(x) != len(y):
            continue
        for (_, k1, n1), (_, k2, n2) in zip(x, y):
            if n1 != n2 and k1 == k2:
                renamed.append((f, line, k1, n1, n2))

files = git("ls-files", head, "--", "*.py").split() or git("ls-tree", "-r", "--name-only", head).split()
files = [f for f in git("ls-tree", "-r", "--name-only", head).split() if f.endswith(".py")]
outside = [f for f in files if f not in piece]


def grep(pattern, paths):
    if not paths:
        return []
    r = subprocess.run(["git", "-C", tree, "grep", "-n", "-P", pattern, head, "--", *paths], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if not re.match(r"^[^:]+:[^:]+:\d+:\s*#", l)]


sus = 0
seen = set()
for f, line, kind, old, new in sorted(renamed):
    key = (kind, old, new)
    if kind in ("kwarg", "attr"):
        hits = grep(rf"(def\s+{old}\b|\b{old}\s*[:=][^=]|self\.{old}\s*=|[(,]\s*{old}\s*[:=,)])", outside)
        why = f"old name still declared outside: {len(hits)}"
    elif kind in ("def", "param", "field", "self-assign"):
        hits = grep(rf"(\.{old}\b|[(,]\s*{old}\s*=[^=])", outside)
        why = f"outside reads .{old} or {old}=: {len(hits)}"
    else:
        continue
    print(f"{'SUSPECT' if hits else 'ok     '} {f}:{line} {kind} {old} -> {new}  ({why})")
    if hits:
        sus += 1
        if key not in seen:
            seen.add(key)
            for h in hits[:4]:
                print("        ", h.split(":", 1)[1][:170])
print(f"{len(renamed)} renamed names in kinds; {sus} suspects")
