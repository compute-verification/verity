"""P13: no function body is written twice, inside ``verity_vllm`` or against ``verity.ml``.

Two functions are copies when their bodies are equal after normalizing: the docstring and every other bare string
statement, comments, formatting, annotations and type comments are dropped, and the names the function binds (its
parameters, assignment, loop, ``with``, ``except`` and comprehension targets, nested ``def``/``class`` names, import
aliases) are renamed in the order they first appear.  Free names (globals, builtins), attributes, keyword names and
constants are kept, so two bodies that call different functions or read different keys differ.  Only bodies over 8
lines (`MIN_LINES` or more lines of ``ast.unparse`` of the normalized body) are compared; nested functions and methods are
bodies of their own.

Every copy inside ``verity_vllm`` is a violation, keyed ``(file, "duplicate-body", symbol, "verity_vllm")``, so a group of
n copies has n entries in ``allowlists/p13_duplicate_bodies.json`` and the dedup that keeps one deletes them all.  A body
that ``verity.ml`` (its importable modules, tests aside) already has is keyed with detail ``"verity.ml"``, and only the
integration's copies are violations: ``verity.ml``'s is the one to keep.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import _ratchet as R

NAME = "p13_duplicate_bodies"
RULE = "P13 (no duplicate function bodies)"
KIND = "duplicate-body"
MIN_LINES = 9
ML = R.CORE_PKG / "ml"
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SKIP_FIELDS = frozenset({"ctx", "annotation", "returns", "type_comment", "type_params"})


@dataclass(frozen=True)
class Body:
    file: str
    symbol: str
    line: int
    digest: str
    node: ast.AST


def _is_doc(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _bound(fn: ast.AST) -> frozenset[str]:
    """The names ``fn`` binds: parameters, targets, nested defs and classes, except, import and match-capture names."""
    a = fn.args
    out = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if x is not None}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (*_DEFS, ast.ClassDef)) and n is not fn:
            out.add(n.name)
        elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
        elif isinstance(n, ast.alias):
            out.add(n.asname or n.name.split(".")[0])
    return frozenset(out)


def _bound_fields(n: ast.AST) -> tuple[str, ...]:
    """The string fields of ``n`` that hold a name the enclosing function binds."""
    if isinstance(n, ast.Name):
        return ("id",)
    if isinstance(n, ast.arg):
        return ("arg",)
    if isinstance(n, (*_DEFS, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return ("name",)
    if isinstance(n, ast.MatchMapping):
        return ("rest",)
    if isinstance(n, ast.alias):
        return ("asname",)
    return ()


def _statements(body: list[ast.stmt]) -> list[ast.stmt]:
    return [s for s in body if not _is_doc(s)]


def digest(fn: ast.AST) -> str:
    """The normalized body's digest: equal exactly when the bodies are copies (module docstring)."""
    names = _bound(fn)
    renamed: dict[str, int] = {}
    out: list[str] = []

    def emit(v: object) -> None:
        if isinstance(v, list):
            out.append("[")
            for x in v:
                if not _is_doc(x):
                    emit(x)
            out.append("]")
        elif isinstance(v, ast.AST):
            out.append(type(v).__name__)
            bound = _bound_fields(v)
            for f in v._fields:
                if f in _SKIP_FIELDS:
                    continue
                x = getattr(v, f, None)
                out.append(f)
                if f in bound and isinstance(x, str) and x in names:
                    out.append(f"_{renamed.setdefault(x, len(renamed))}")
                else:
                    emit(x)
        else:
            out.append(repr(v))

    emit(_statements(fn.body))
    return hashlib.sha512("\x00".join(out).encode()).hexdigest()


class _Normalize(ast.NodeTransformer):
    def __init__(self, names: frozenset[str]) -> None:
        self.names, self.renamed = names, {}

    def _r(self, x: str) -> str:
        return f"_{self.renamed.setdefault(x, len(self.renamed))}" if x in self.names else x

    def visit_Name(self, n: ast.Name) -> ast.Name:
        n.id = self._r(n.id)
        return n

    def visit_arg(self, n: ast.arg) -> ast.arg:
        n.arg, n.annotation = self._r(n.arg), None
        return n

    def visit_Expr(self, n: ast.Expr) -> ast.AST | None:
        return None if _is_doc(n) else self.generic_visit(n)


def lines(fn: ast.AST) -> int:
    """The normalized body's length: the lines ``ast.unparse`` gives it, equal for every copy."""
    fn = copy.deepcopy(fn)
    norm = _Normalize(_bound(fn))
    body = [s for s in (norm.visit(s) for s in _statements(fn.body)) if s is not None]
    return len(ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))).splitlines())


def bodies(tree: ast.Module, rel: str) -> list[Body]:
    out = []
    for node, symbol in R.walk(tree):
        if isinstance(node, _DEFS):
            qual = node.name if symbol == "<module>" else f"{symbol}.{node.name}"
            out.append(Body(rel, qual, node.lineno, digest(node), node))
    return out


def _is_module(p: Path, root: Path) -> bool:
    """``p`` is importable under ``root``'s package: every directory from ``root`` down to it has an ``__init__.py``."""
    return all((d / "__init__.py").is_file() for d in p.parents if d == root or root in d.parents)


def duplicates(pkg: Iterable[Body], ml: Iterable[Body]) -> list[tuple[Body, str, list[Body]]]:
    """(copy, where the other copies are, the other copies) for each integration body over the size floor that repeats."""
    groups: dict[str, list[Body]] = defaultdict(list)
    for b in pkg:
        groups[b.digest].append(b)
    ml_groups: dict[str, list[Body]] = defaultdict(list)
    for b in ml:
        if b.digest in groups:
            ml_groups[b.digest].append(b)
    out = []
    for d, group in groups.items():
        theirs = ml_groups.get(d, [])
        if len(group) + len(theirs) < 2 or lines(group[0].node) < MIN_LINES:
            continue
        for b in group:
            out.append((b, "verity.ml" if theirs else "verity_vllm", [x for x in (*theirs, *group) if x is not b]))
    return sorted(out, key=lambda t: (t[0].file, t[0].line))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[R.Violation, ...], dict[tuple[str, str, str, str], list[str]]]:
    """The violations, and for each key the other copies (``file:line symbol``) the failure message names."""
    pkg = [b for p in R.library_files() for b in bodies(R.parse(p), R.rel(p))]
    ml = [b for p in R.library_files(".py", ML) if _is_module(p, ML)
          for b in bodies(R.parse(p), p.relative_to(R.CORE_PKG.parent).as_posix())]
    found, others = [], {}
    for b, where, rest in duplicates(pkg, ml):
        v = R.Violation(b.file, KIND, b.symbol, where, b.line)
        found.append(v)
        others.setdefault(v.key, []).extend(f"{x.file}:{x.line} {x.symbol}" for x in rest)
    return tuple(found), others


def test_no_new_duplicates() -> None:
    found, others = scan()
    allowed = R.load_allowlist(NAME)
    counts = Counter(v.key for v in found)
    new_keys = [k for k, n in sorted(counts.items()) if n > allowed.get(k, 0)]
    detail = []
    for k, line in zip(new_keys, R.new_violations(NAME, found)):
        detail += [line, *(f"      same body as {o}" for o in sorted(set(others[k])))]
    assert not new_keys, (f"{RULE}: {len(new_keys)} function bod(y/ies) not in allowlists/{NAME}.json repeat another. Keep "
                          "one: call verity.ml's where it has the body, else keep the copy in the module that owns its job "
                          "and import it in every other copy's place (no re-export shim). (If an allowlisted copy only "
                          "moved, move its entry in the same change.)\n" + "\n".join(detail))


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan()[0], RULE)


def _defs(src: str, rel: str) -> list[Body]:
    return bodies(ast.parse(textwrap.dedent(src)), rel)


def test_detectors_on_examples() -> None:
    steps = "".join(f"    {{v}} = {{v}} + {i}\n" for i in range(MIN_LINES - 1))
    pkg = [
        *_defs('def f(x: int) -> int:\n    """one"""\n' + steps.format(v="x") + "    return x\n", "verity_vllm/a.py"),
        *_defs("def g(y):\n    # a comment\n" + steps.format(v="y") + "    return y\n", "verity_vllm/b.py"),
        *_defs("class C:\n" + textwrap.indent('def f(self, z):\n    """two"""\n' + steps.format(v="z") + "    return z\n",
                                              "    "), "verity_vllm/c.py"),
        *_defs("def h(z):\n    z = z + 1\n    return z\n\ndef k(w):\n    w = w + 1\n    return w\n", "verity_vllm/short.py"),
        *_defs("def m(x):\n" + steps.format(v="x") + "    return len(x)\n", "verity_vllm/m.py"),
    ]
    assert lines(pkg[0].node) == MIN_LINES
    got = [(b.file, b.symbol, where, [o.symbol for o in rest]) for b, where, rest in duplicates(pkg, [])]
    assert got == [("verity_vllm/a.py", "f", "verity_vllm", ["g", "C.f"]),
                   ("verity_vllm/b.py", "g", "verity_vllm", ["f", "C.f"]),
                   ("verity_vllm/c.py", "C.f", "verity_vllm", ["f", "g"])], got
    ml = _defs("def kept(x):\n" + steps.format(v="x") + "    return len(x)\n", "verity/ml/x.py")
    got = [(b.file, where, [o.file for o in rest]) for b, where, rest in duplicates(pkg, ml)]
    assert ("verity_vllm/m.py", "verity.ml", ["verity/ml/x.py"]) in got, got

    def same(a: str, b: str) -> bool:
        return digest(ast.parse(a).body[0]) == digest(ast.parse(b).body[0])

    assert same("def f(a):\n    b = a\n    return b\n", "def f(x):\n    y = x\n    return y\n")
    assert not same("def f(a):\n    return g(k=a)\n", "def f(a):\n    return g(j=a)\n")     # keyword names are kept
    assert not same("def f(a):\n    return a.x\n", "def f(a):\n    return a.y\n")         # attributes are kept
    assert not same("def f(a):\n    return b\n", "def f(b):\n    return b\n")             # a free name is not a bound one
    assert not same("def f(a):\n    return g(a)\n", "def f(a):\n    return h(a)\n")       # free names are kept
