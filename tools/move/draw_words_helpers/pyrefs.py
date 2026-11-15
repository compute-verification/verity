"""Where the tree's Python names a module's names: imports and `alias.name` reads, resolved to files by path suffix."""
import ast
from collections import defaultdict


class Refs:
    def __init__(self, repo, files):
        self.repo, self.files = repo, [f for f in files if f.endswith(".py")]
        self.fileset = set(self.files)
        self.by_dotted = defaultdict(list)
        for f in self.files:
            parts = f[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            for i in range(len(parts)):
                self.by_dotted[".".join(parts[i:])].append(f)
        self.trees, self.defs = {}, {}

    def tree(self, f):
        if f not in self.trees:
            try:
                self.trees[f] = ast.parse((self.repo / f).read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, FileNotFoundError, ValueError):
                self.trees[f] = None
        return self.trees[f]

    def defines(self, f):
        if f not in self.defs:
            out, star, t = set(), False, self.tree(f)
            for n in (t.body if t else []):
                for x in ([n] if not isinstance(n, (ast.If, ast.Try)) else list(ast.walk(n))):
                    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        out.add(x.name)
                    elif isinstance(x, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        for tg in (x.targets if isinstance(x, ast.Assign) else [x.target]):
                            out |= {nm.id for nm in ast.walk(tg) if isinstance(nm, ast.Name)}
                    elif isinstance(x, (ast.Import, ast.ImportFrom)):
                        for a in x.names:
                            star |= a.name == "*"
                            out.add((a.asname or a.name).split(".")[0])
            if f.endswith("__init__.py"):
                d = f[: -len("__init__.py")]
                out |= {p[len(d):].split("/")[0].removesuffix(".py") for p in self.files if p.startswith(d)}
            self.defs[f] = (out, star)
        return self.defs[f]

    def resolve(self, f, mod, level):
        if level:
            cand = "/".join(f.split("/")[:-level] + (mod.split(".") if mod else []))
            return [p for p in (cand + ".py", cand + "/__init__.py") if p in self.fileset]
        return self.by_dotted.get(mod, []) if mod else []

    def uses(self, f):
        """(kind, alias, module files, name, line): `from` imports ("from", bound name, targets, name) and attribute reads
        on an alias of an in-tree module ("attr", alias, targets, attr)."""
        t, out, aliases = self.tree(f), [], {}
        if t is None:
            return out
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom):
                targets = self.resolve(f, n.module, n.level)
                for a in n.names:
                    if a.name == "*":
                        continue
                    sub = self.resolve(f, (n.module + "." if n.module else "") + a.name, n.level)
                    if sub:
                        aliases[a.asname or a.name] = sub
                    elif targets:
                        out.append(("from", a.asname or a.name, targets, a.name, n.lineno))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    targets = self.resolve(f, a.name, 0)
                    if targets:
                        aliases[a.asname or a.name.split(".")[0]] = targets if a.asname else \
                            (self.resolve(f, a.name.split(".")[0], 0) or targets)
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                out.append(("attr", n.value.id, aliases[n.value.id], n.attr, n.lineno))
        return out
