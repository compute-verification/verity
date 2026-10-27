"""Ratchet machinery shared by the rule lints in this directory (one rule per ``test_pNN_*.py``).

Each ``test_pNN_*.py`` scans the integration's library code (``verity_vllm/**/*.py`` and ``verity_vllm/**/*.sh``), and the
circuit Definitions that moved from it to ``verity/ml/definitions/`` (``LIBRARY``: circuits' since the move, held to the
ratchets as they stood), with ``ast`` and regexes.  Nothing scanned is imported, so the lints need neither torch nor vllm
and run in seconds.  What a scan finds is compared with a committed allowlist, ``allowlists/<name>.json``, of the
violations that existed when the rule was introduced.

A violation is keyed by ``(file, kind, symbol, detail)``: the file relative to ``integrations/vllm`` (to the repository
outside it), the rule's kind of violation, the enclosing ``Class.function`` (``<module>`` at top level, ``<script>`` for
shell) and a short detail (a name, an import edge, a token).  Line numbers are not part of the key, so edits elsewhere in a file do not churn the
list.  An entry allows ``count`` occurrences of its key (default 1).

* A key found more often than its entry allows fails: fix the new code.  If an existing violation only moved (a rename
  or a file move), move its entry in the same change.
* An entry allowing more than is found fails: the violation is gone, so delete the entry (or lower its count).

So an allowlist can only shrink, and the failure message prints the exact entry lines to delete or add.
"""
from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                     # integrations/vllm
REPO = ROOT.parents[1]
PKG = ROOT / "verity_vllm"
CORE_PKG = REPO / "verity"
DEFINITIONS = CORE_PKG / "ml" / "definitions"
#: the scanned packages: the integration's, and the Definitions that moved out of it
LIBRARY = (PKG, DEFINITIONS)
ALLOWLISTS = HERE / "allowlists"
SKIP_DIRS = frozenset({"__pycache__", "tests"})


@dataclass(frozen=True)
class Violation:
    file: str
    kind: str
    symbol: str
    detail: str
    line: int = 0

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.file, self.kind, self.symbol, self.detail)


# --- the scanned tree -------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def library_files(suffix: str = ".py", root: Path | None = None) -> tuple[Path, ...]:
    """Library files under ``root`` (default every ``LIBRARY`` package), excluding ``tests`` directories and ``test_*`` files;
    a root outside ``LIBRARY`` (core's) without the ``LIBRARY`` packages inside it."""
    if root is None:
        return tuple(p for r in LIBRARY for p in library_files(suffix, r))
    out = []
    for p in sorted(root.rglob(f"*{suffix}")):
        parts = p.relative_to(root).parts
        if any(d in SKIP_DIRS for d in parts[:-1]) or parts[-1].startswith("test_"):
            continue
        if root not in LIBRARY and any(p.is_relative_to(r) for r in LIBRARY):
            continue
        out.append(p)
    return tuple(out)


def rel(p: Path) -> str:
    return (p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.relative_to(REPO)).as_posix()


def module_name(p: Path) -> str:
    """``verity_vllm/a/b.py`` -> ``verity_vllm.a.b``, ``verity/ml/definitions/a.py`` -> ``verity.ml.definitions.a``; a
    package's ``__init__.py`` names the package."""
    parts = list(p.relative_to(ROOT if p.is_relative_to(ROOT) else REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@lru_cache(maxsize=None)
def source(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def parse_src(src: str) -> ast.Module:
    """One shared (read-only) tree per source text, so the rules parse each file once."""
    return ast.parse(src)


def parse(p: Path) -> ast.Module:
    return parse_src(source(p))


def text(node: ast.AST, limit: int = 80) -> str:
    """Normalized source text of an expression (quotes and spacing independent of the file's formatting)."""
    return ast.unparse(node)[:limit]


# --- AST helpers --------------------------------------------------------------------------------------------------------

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def walk(node: ast.AST, symbol: str = "<module>") -> tuple[tuple[ast.AST, str], ...]:
    """Every node below ``node``, in source order, with the qualified name of its enclosing def/class (``<module>`` at
    top level)."""
    return _walk(node, symbol)


@lru_cache(maxsize=None)
def _walk(node: ast.AST, symbol: str) -> tuple[tuple[ast.AST, str], ...]:
    out: list[tuple[ast.AST, str]] = []

    def visit(n: ast.AST, sym: str) -> None:
        for c in ast.iter_child_nodes(n):
            out.append((c, sym))
            visit(c, (c.name if sym == "<module>" else f"{sym}.{c.name}") if isinstance(c, _SCOPES) else sym)

    visit(node, symbol)
    return tuple(out)


@lru_cache(maxsize=None)
def docstring_ids(tree: ast.AST) -> frozenset[int]:
    """ids of string constants that are docstrings or bare string statements (prose, not code)."""
    return frozenset(id(node.value) for node in ast.walk(tree)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))


def marker_ids(tree: ast.Module) -> frozenset[int]:
    """ids of the strings a module's ``__models__`` names (``verity_vllm.glue``: the models it exists for, which decide nothing)."""
    return frozenset(id(c) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "__models__" for t in node.targets) for c in ast.walk(node.value))


def docstrings(tree: ast.Module) -> Iterator[tuple[str, str, int]]:
    """(symbol, docstring, line) for the module and every class and function."""
    doc = ast.get_docstring(tree, clean=False)
    if doc:
        yield "<module>", doc, 1
    for node, symbol in walk(tree):
        if isinstance(node, _SCOPES):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                yield (node.name if symbol == "<module>" else f"{symbol}.{node.name}"), doc, node.lineno


def dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


class Aliases:
    """Where the names bound by a file's imports come from: ``np`` -> ``numpy``, ``defs`` -> ``verity.ml.circuits.defs``.

    Also records ``name = importlib.import_module("pkg.mod")``, ``name = sys.modules["pkg.mod"]`` and
    ``name = __import__("pkg.mod", ...)`` with literal module names.  Scope-insensitive: one map per file.
    """

    def __init__(self, tree: ast.Module, module: str = "", is_package: bool = False):
        self.names: dict[str, str] = {}
        pkg = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.names[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.names.setdefault(top, top)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = pkg.split(".")[: max(0, len(pkg.split(".")) - node.level + 1)] if pkg else []
                    base = ".".join([*anchor, *([base] if base else [])])
                for a in node.names:
                    if a.name != "*":
                        self.names[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                mod = dynamic_module(node.value)
                if mod:
                    self.names[node.targets[0].id] = mod

    def resolve(self, node: ast.AST) -> str | None:
        """The dotted origin of a Name/Attribute chain (``np.random.default_rng`` -> ``numpy.random.default_rng``)."""
        mod = dynamic_module(node)
        if mod:
            return mod
        d = dotted(node)
        if d is None:
            return None
        head, _, rest = d.partition(".")
        origin = self.names.get(head)
        if origin is None:
            return None
        return f"{origin}.{rest}" if rest else origin


@lru_cache(maxsize=None)
def aliases(tree: ast.Module, module: str = "", is_package: bool = False) -> Aliases:
    return Aliases(tree, module, is_package)


def dynamic_module(node: ast.AST) -> str | None:
    """The literal module name of ``importlib.import_module("m")``, ``sys.modules["m"]``, ``sys.modules.get("m")`` or
    ``__import__("m", ...)``."""
    if isinstance(node, ast.Call):
        fn = dotted(node.func) or ""
        if fn in ("importlib.import_module", "import_module", "__import__", "sys.modules.get") and node.args \
                and is_str(node.args[0]):
            return node.args[0].value
    if isinstance(node, ast.Subscript) and dotted(node.value) == "sys.modules" and is_str(node.slice):
        return node.slice.value
    return None


# --- allowlists ---------------------------------------------------------------------------------------------------------

def load_allowlist(name: str) -> Counter:
    path = ALLOWLISTS / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and isinstance(data.get("entries"), list), f"{path.name}: {{'entries': [...]}} expected"
    allowed: Counter = Counter()
    for e in data["entries"]:
        assert set(e) <= {"file", "kind", "symbol", "detail", "count"} and {"file", "kind", "symbol", "detail"} <= set(e), \
            f"{path.name}: malformed entry {e}"
        k = (e["file"], e["kind"], e["symbol"], e["detail"])
        assert k not in allowed, f"{path.name}: duplicate entry {e}"
        n = e.get("count", 1)
        assert isinstance(n, int) and n >= 1, f"{path.name}: count must be a positive int: {e}"
        allowed[k] = n
    return allowed


def entry_line(key: tuple[str, str, str, str], count: int = 1) -> str:
    d = {"file": key[0], "kind": key[1], "symbol": key[2], "detail": key[3]}
    if count != 1:
        d["count"] = count
    return json.dumps(d, ensure_ascii=False)


def new_violations(name: str, violations: Iterable[Violation]) -> list[str]:
    violations = list(violations)
    found = Counter(v.key for v in violations)
    lines = defaultdict(list)
    for v in violations:
        lines[v.key].append(v.line)
    allowed = load_allowlist(name)
    out = []
    for k, n in sorted(found.items()):
        if n > allowed.get(k, 0):
            where = ",".join(str(ln) for ln in sorted(lines[k]) if ln)
            extra = f"  ({n} found, {allowed[k]} allowed)" if allowed.get(k) else ""
            out.append(f"  {k[0]}:{where} [{k[1]}] {k[2]}: {k[3]}{extra}")
    return out


def stale_entries(name: str, violations: Iterable[Violation]) -> list[str]:
    found = Counter(v.key for v in violations)
    out = []
    for k, n in sorted(load_allowlist(name).items()):
        have = found.get(k, 0)
        if have < n:
            out.append(("  delete " if have == 0 else f"  lower count to {have}: ") + entry_line(k, n))
    return out


def assert_no_new(name: str, violations: Iterable[Violation], rule: str, fix: str) -> None:
    new = new_violations(name, violations)
    assert not new, (f"{rule}: {len(new)} violation(s) not in allowlists/{name}.json. {fix} "
                     "(If an allowlisted violation only moved, move its entry in the same change.)\n" + "\n".join(new))


def assert_no_stale(name: str, violations: Iterable[Violation], rule: str) -> None:
    stale = stale_entries(name, violations)
    assert not stale, (f"{rule}: {len(stale)} entr(y/ies) in allowlists/{name}.json no longer match anything: the "
                       "violation is gone, so delete the entry (allowlists only shrink):\n" + "\n".join(stale))
