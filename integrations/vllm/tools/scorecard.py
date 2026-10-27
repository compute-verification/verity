"""The vLLM integration's quality scorecard: the numbers its quality bar is measured by, read from the tree alone.

    python3 integrations/vllm/tools/scorecard.py [--json] [--top N] [--root REPO]

* **ratchets**: each shrink-only allowlist under ``integrations/vllm/tests`` (``**/allowlists/*.json`` and
  ``**/*allowlist*.json``, any JSON with an ``entries`` list) and its number of entries.
* **size**: library modules (``verity_vllm/**/*.py`` less ``tests/`` and ``test_*.py``, as the lints scan) over 800,
  600 and 400 lines, functions over 150 and 100 lines, and the largest modules.
* **outside imports**: the distinct modules of ``verity_vllm`` that code outside ``integrations/vllm/`` imports, counted
  as ``tests/test_vllm_public_api.py`` counts them (an import statement, an ``import_module``/``__import__``/
  ``importorskip`` string, or a ``"module:attr"`` string; ``from P import n`` is the module ``P.n`` when that is a module),
  and the private (``_name``) names it takes.
* **docs and types**: public top-level functions and classes without a docstring, modules without one, and public
  functions and methods with a parameter or return that has no annotation; the whole package, and the modules outside
  code imports.
* **settings**: uses of the process environment (``os.environ``, ``os.getenv``, ...) outside the settings module
  ``verity_vllm/settings.py``.

Nothing scanned is imported, so it needs only the standard library. Outside files are those ``git grep`` finds (tracked
and untracked, not ignored) when the root is a Git checkout, else every ``*.py`` under the root outside hidden
directories.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

INTEGRATION = "integrations/vllm"
PACKAGE = "verity_vllm"
SETTINGS = f"{PACKAGE}/settings.py"
SIZE_BANDS = (800, 600, 400)
FUNCTION_BANDS = (150, 100)
SKIP_DIRS = frozenset({"__pycache__", "tests"})
#: outside files whose ``verity_vllm`` strings are data, not imports (as tests/test_vllm_public_api.py's NOT_SCANNED)
NOT_SCANNED = frozenset({"tests/test_vllm_public_api.py", "tests/test_module_strings.py"})
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = frozenset({"import_module", "__import__", "importorskip"})
ENV_NAMES = frozenset({"os.environ", "os.environb", "os.getenv", "os.getenvb", "os.putenv", "os.unsetenv"})
ENV_WRITERS = frozenset({"os.putenv", "os.unsetenv"})
ENV_WRITE_METHODS = frozenset({"setdefault", "update", "pop", "popitem", "clear", "__setitem__", "__delitem__"})


@dataclass(frozen=True)
class Tree:
    root: Path

    @property
    def integration(self) -> Path:
        return self.root / INTEGRATION

    @property
    def package(self) -> Path:
        return self.integration / PACKAGE

    def library_files(self) -> list[Path]:
        out = []
        for p in sorted(self.package.rglob("*.py")):
            parts = p.relative_to(self.package).parts
            if not (any(d in SKIP_DIRS for d in parts[:-1]) or parts[-1].startswith("test_")):
                out.append(p)
        return out

    def rel(self, p: Path) -> str:
        return p.relative_to(self.integration).as_posix()

    def module_name(self, p: Path) -> str:
        parts = list(p.relative_to(self.integration).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def module_file(self, module: str) -> Path | None:
        base = self.integration.joinpath(*module.split("."))
        for p in (base.with_suffix(".py"), base / "__init__.py"):
            if p.is_file():
                return p
        return None


@lru_cache(maxsize=None)
def _source(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _parse(p: Path) -> ast.Module | None:
    try:
        return ast.parse(_source(p), filename=str(p))
    except SyntaxError:
        return None


# --- ratchets -----------------------------------------------------------------------------------------------------------

def ratchets(tree: Tree) -> dict[str, dict]:
    tests = tree.integration / "tests"
    paths = sorted({*tests.rglob("allowlists/*.json"), *tests.rglob("*allowlist*.json")})
    out = {}
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            out[p.stem] = {"file": tree.rel(p), "entries": len(data["entries"])}
    return out


# --- size ---------------------------------------------------------------------------------------------------------------

def _functions(stmts: list[ast.stmt]):
    """Every def among the statements and nested in their bodies (a def is a statement, so expressions need no walk)."""
    for s in stmts:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield s
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            sub = getattr(s, field, None)
            if isinstance(sub, list):
                yield from _functions(sub)


def size(tree: Tree, parsed: dict[Path, ast.Module | None], top: int) -> dict:
    lines = {tree.rel(p): len(_source(p).splitlines()) for p in parsed}
    spans = [f.end_lineno - f.lineno + 1 for mod in parsed.values() if mod for f in _functions(mod.body)]
    largest = sorted(lines.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return {
        "modules": len(lines),
        "lines": sum(lines.values()),
        **{f"modules_over_{n}": sum(v > n for v in lines.values()) for n in SIZE_BANDS},
        **{f"functions_over_{n}": sum(s > n for s in spans) for n in FUNCTION_BANDS},
        "largest": [{"file": f, "lines": n} for f, n in largest],
    }


# --- outside imports ----------------------------------------------------------------------------------------------------

def _outside_files(tree: Tree) -> list[str]:
    try:
        res = subprocess.run(["git", "grep", "-l", "-F", "--untracked", PACKAGE, "--", "*.py", f":!{INTEGRATION}/"],
                             cwd=tree.root, capture_output=True, text=True)
    except FileNotFoundError:
        res = None
    if res is not None and res.returncode in (0, 1) and "not a git repository" not in res.stderr:
        files = res.stdout.split()
    else:
        files = [p.relative_to(tree.root).as_posix() for p in tree.root.rglob("*.py")
                 if not any(part.startswith(".") for part in p.relative_to(tree.root).parts)
                 and not p.relative_to(tree.root).as_posix().startswith(f"{INTEGRATION}/")]
    return sorted(f for f in files if f not in NOT_SCANNED)


def import_sites(tree: Tree, path: str, text: str) -> list[tuple[str, str | None]]:
    """(module, name) for each import of the package in one file; name is None when a whole module is imported."""
    out: list[tuple[str, str | None]] = []
    ours = lambda m: m.split(".")[0] == PACKAGE   # noqa: E731
    try:
        mod = ast.parse(text, filename=path)
    except SyntaxError:
        return out
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            out += [(a.name, None) for a in node.names if ours(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and ours(node.module):
            for a in node.names:
                sub = f"{node.module}.{a.name}"
                out.append((sub, None) if tree.module_file(sub) else (node.module, a.name))
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS
              and ours(node.args[0].value)):
            out.append((node.args[0].value, None))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value) \
                and ours(node.value):
            module, _, name = node.value.partition(":")
            out.append((module, name))
    return out


def outside_imports(tree: Tree) -> dict:
    sites = []
    for f in _outside_files(tree):
        sites += [(f, m, n) for m, n in import_sites(tree, f, (tree.root / f).read_text(encoding="utf-8"))]
    modules = sorted({m for _, m, _ in sites})
    private = sorted({(m, n) for _, m, n in sites if n is not None and n.startswith("_")})
    return {
        "modules": len(modules),
        "private_names": len(private),
        "sites": len(sites),
        "files": len({f for f, _, _ in sites}),
        "module_list": modules,
        "private_list": [f"{m}.{n}" for m, n in private],
    }


# --- docs and types -----------------------------------------------------------------------------------------------------

def _missing_annotations(fn: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> list[str]:
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]
    static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in fn.decorator_list)
    if method and not static and params:
        params = params[1:]
    return [p.arg for p in params if p.annotation is None] + (["return"] if fn.returns is None else [])


def _private_module(module: str) -> bool:
    return any(part.startswith("_") for part in module.split("."))


def docs_types(tree: Tree, parsed: dict[Path, ast.Module | None], interface: set[str]) -> dict:
    """Per scope ("package", "interface"): public defs, those without a docstring, modules without one, and public
    functions/methods with a missing annotation."""
    scopes = {"package": Counter(), "interface": Counter()}
    for p, mod in parsed.items():
        name = tree.module_name(p)
        if mod is None or _private_module(name):
            continue
        counts = Counter(modules=1, modules_undocumented=int(ast.get_docstring(mod) is None))
        for node in mod.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name.startswith("_"):
                continue
            counts["public_defs"] += 1
            counts["undocumented"] += ast.get_docstring(node) is None
            if isinstance(node, ast.ClassDef):
                methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and (not m.name.startswith("_") or m.name in ("__init__", "__call__"))]
                counts["functions"] += len(methods)
                counts["untyped"] += sum(bool(_missing_annotations(m, method=True)) for m in methods)
            else:
                counts["functions"] += 1
                counts["untyped"] += bool(_missing_annotations(node, method=False))
        scopes["package"].update(counts)
        if name in interface:
            scopes["interface"].update(counts)
    keys = ("modules", "modules_undocumented", "public_defs", "undocumented", "functions", "untyped")
    return {scope: {k: c[k] for k in keys} for scope, c in scopes.items()}


# --- settings -----------------------------------------------------------------------------------------------------------

def _aliases(mod: ast.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
    return names


def _resolve(node: ast.AST, names: dict[str, str]) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in names:
        return None
    return ".".join([names[node.id], *reversed(parts)])


def env_uses(src: str, mod: ast.Module | None = None) -> list[tuple[str, str, int]]:
    """(variable or access form, "read" | "write", line) for each use of the process environment in one module."""
    if not any(s in src for s in ("environ", "getenv", "putenv", "unsetenv")):
        return []
    mod = mod or ast.parse(src)
    names = _aliases(mod)
    parents = {id(c): n for n in ast.walk(mod) for c in ast.iter_child_nodes(n)}
    out = []
    for node in ast.walk(mod):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        origin = _resolve(node, names)
        parent = parents.get(id(node))
        if origin not in ENV_NAMES or (isinstance(parent, ast.Attribute) and _resolve(parent, names) in ENV_NAMES):
            continue
        var, mode = origin, "write" if origin in ENV_WRITERS else "read"
        if isinstance(parent, ast.Subscript) and parent.value is node:
            var = parent.slice.value if isinstance(parent.slice, ast.Constant) else origin
            mode = "write" if isinstance(parent.ctx, (ast.Store, ast.Del)) else "read"
        elif isinstance(parent, ast.Attribute):
            call = parents.get(id(parent))
            literal = isinstance(call, ast.Call) and call.func is parent and call.args and isinstance(call.args[0], ast.Constant)
            var = call.args[0].value if literal else f"{origin}.{parent.attr}"
            mode = "write" if parent.attr in ENV_WRITE_METHODS else "read"
        elif isinstance(parent, ast.Call) and parent.func is node and parent.args and isinstance(parent.args[0], ast.Constant):
            var = parent.args[0].value
        elif isinstance(parent, ast.Compare) and isinstance(parent.left, ast.Constant):
            var = parent.left.value
        out.append((str(var), mode, node.lineno, node.col_offset))
    return [(v, m, line) for v, m, line, _ in sorted(out, key=lambda u: u[2:])]


def settings(tree: Tree, parsed: dict[Path, ast.Module | None]) -> dict:
    uses = []
    for p, mod in parsed.items():
        rel = tree.rel(p)
        if rel != SETTINGS and mod is not None:
            uses += [(rel, v, m) for v, m, _ in env_uses(_source(p), mod)]
    modes = Counter(m for _, _, m in uses)
    variables = Counter(v for _, v, _ in uses if not v.startswith("os."))
    by_module = Counter(f for f, _, _ in uses)
    return {
        "settings_module": SETTINGS if (tree.integration / SETTINGS).is_file() else None,
        "sites": len(uses),
        "reads": modes["read"],
        "writes": modes["write"],
        "modules": len(by_module),
        "variables": len(variables),
        "by_module": dict(sorted(by_module.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


# --- the card -----------------------------------------------------------------------------------------------------------

def _commit(root: Path) -> dict | None:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", INTEGRATION], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if head.returncode:
        return None
    return {"sha": head.stdout.strip(), "integration_dirty": bool(dirty.stdout.strip())}


def scorecard(root: Path, *, top: int = 15, outside: bool = True) -> dict:
    tree = Tree(root)
    parsed = {p: _parse(p) for p in tree.library_files()}
    imports = outside_imports(tree) if outside else None
    interface = {PACKAGE, *(imports["module_list"] if imports else ())}
    return {
        "schema": "vllm-scorecard/v1",
        "commit": _commit(root),
        "ratchets": ratchets(tree),
        "size": size(tree, parsed, top),
        "outside_imports": imports,
        "docs_types": docs_types(tree, parsed, interface),
        "settings": settings(tree, parsed),
    }


def render(card: dict) -> str:
    s, o, d, e = card["size"], card["outside_imports"], card["docs_types"], card["settings"]
    w = 52
    row = lambda label, value: f"  {label:<{w}}{value:>7}"   # noqa: E731
    at = f" at {card['commit']['sha'][:10]}{' (+ edits)' if card['commit']['integration_dirty'] else ''}" if card["commit"] else ""
    out = [f"verity_vllm quality scorecard{at}: {s['modules']} library modules, {s['lines']} lines", "",
           "ratchets (allowlist entries; each only shrinks)"]
    out += [row(name, r["entries"]) for name, r in card["ratchets"].items()]
    out += ["", "size"]
    out += [row(f"modules over {n} lines", s[f"modules_over_{n}"]) for n in SIZE_BANDS]
    out += [row(f"functions over {n} lines", s[f"functions_over_{n}"]) for n in FUNCTION_BANDS]
    out += ["  largest modules:"] + [f"    {m['lines']:>6}  {m['file']}" for m in s["largest"]]
    out += ["", f"outside imports (code outside {INTEGRATION}/)"]
    if o is None:
        out.append("  not scanned")
    else:
        out += [row("distinct modules imported", o["modules"]), row("private names imported", o["private_names"]),
                row("import sites", o["sites"]), row("importing files", o["files"])]
    out += ["", "docs and types (public top-level defs; public functions and methods)"]
    for scope, label in (("package", "whole package"), ("interface", "modules imported from outside")):
        c = d[scope]
        out += [f"  {label} ({c['modules']} modules)",
                row("  modules without a docstring", c["modules_undocumented"]),
                row(f"  public defs without a docstring (of {c['public_defs']})", c["undocumented"]),
                row(f"  functions with a missing annotation (of {c['functions']})", c["untyped"])]
    out += ["", f"settings (environment uses outside {SETTINGS}{'' if e['settings_module'] else ', which does not exist yet'})",
            row("uses", e["sites"]), row("  reads", e["reads"]), row("  writes", e["writes"]),
            row("modules", e["modules"]), row("distinct variables", e["variables"])]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the scorecard as JSON")
    ap.add_argument("--top", type=int, default=15, help="how many of the largest modules to list (default 15)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3],
                    help="the repository root (default: the checkout this file is in)")
    args = ap.parse_args(argv)
    card = scorecard(args.root.resolve(), top=args.top)
    print(json.dumps(card, indent=1) if args.json else render(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
