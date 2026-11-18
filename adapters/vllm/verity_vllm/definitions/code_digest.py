"""The digest of a module's or a function's parsed code, which no comment, docstring, layout or sorting of imports moves.

The syntax tree is hashed with every statement that is a bare string (a docstring) removed.  A run of consecutive import
statements is hashed as what it does: the modules it first imports, in their order, and the names it binds, as a set.  So
splitting, merging and reordering the names of a run leave the digest, and reordering the modules a run imports moves it,
since a module's first import runs its body (a registration, say).  A name a ``from`` import takes counts as a module
where its package holds a submodule of that name, or where the package can't be found.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import textwrap
from functools import cache
from typing import Any

from verity.circuits.codec import canonical_json

DOMAIN = b"verity/code-digest/v1\x00"


def code_digest(obj: Any) -> str:
    """SHA-512 of `obj`'s parsed code (a module, class or function); `inspect.getsource`'s errors propagate."""
    module = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
    package = getattr(module, "__package__", None) or None
    return source_digest(textwrap.dedent(inspect.getsource(obj)), package)


def source_digest(source: str, package: str | None = None) -> str:
    """SHA-512 of the parsed code of `source`; `package` resolves its relative imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"code_digest: the source does not parse on its own ({e}); hash the enclosing definition") from e
    return hashlib.sha512(DOMAIN + canonical_json(_node(tree, package))).hexdigest()


def _node(node: Any, package: str | None) -> Any:
    if isinstance(node, ast.AST):
        fields = {}
        for name in node._fields:
            value = getattr(node, name, None)
            if value is None or (isinstance(value, list) and not value):
                continue
            if isinstance(value, list) and isinstance(value[0], ast.stmt):
                fields[name] = _statements(value, package)
            else:
                fields[name] = _node(value, package)
        return [type(node).__name__, fields]
    if isinstance(node, list):
        return [_node(v, package) for v in node]
    if isinstance(node, bool) or node is None:
        return ["const", node]
    if isinstance(node, str):
        return ["str", node]
    if isinstance(node, bytes):
        return ["bytes", node.hex()]
    if isinstance(node, int):
        return ["int", str(node)]
    if isinstance(node, float):
        return ["float", node.hex()]
    if isinstance(node, complex):
        return ["complex", node.real.hex(), node.imag.hex()]
    if node is Ellipsis:
        return ["ellipsis"]
    raise TypeError(f"code_digest: no canonical form for {type(node).__name__}")


def _statements(body: list[ast.stmt], package: str | None) -> list:
    out: list = []
    run: list[ast.stmt] = []
    for s in body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            continue
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            run.append(s)
            continue
        if run:
            out.append(_imports(run, package))
            run = []
        out.append(_node(s, package))
    if run:
        out.append(_imports(run, package))
    return out


def _imports(run: list[ast.stmt], package: str | None) -> list:
    """A run's first imports in order and its bindings as a set; the run as written where a name is bound twice or by `*`."""
    loads: list[str] = []
    binds: dict[str, list[str]] = {}

    def load(module: str) -> None:
        parts = module.split(".")
        for i in range(1, len(parts) + 1):
            if (m := ".".join(parts[:i])) not in loads:
                loads.append(m)

    for s in run:
        if isinstance(s, ast.Import):
            for a in s.names:
                load(a.name)
                target, value = (a.asname, ["module", a.name]) if a.asname else (a.name.split(".")[0], ["module", a.name.split(".")[0]])
                if target in binds:
                    return ["imports-as-written", _node(run, package)]
                binds[target] = value
        else:
            assert isinstance(s, ast.ImportFrom)
            module = _absolute(s.module, s.level, package)
            load(module)
            for a in s.names:
                target = a.asname or a.name
                if a.name == "*" or target in binds:
                    return ["imports-as-written", _node(run, package)]
                if _may_be_submodule(module, a.name):
                    load(f"{module}.{a.name}")
                binds[target] = ["from", module, a.name]
    return ["imports", {"loads": loads, "binds": sorted([t, *v] for t, v in binds.items())}]


def _absolute(module: str | None, level: int, package: str | None) -> str:
    if not level:
        return module or ""
    if package:
        return importlib.util.resolve_name("." * level + (module or ""), package)
    return "." * level + (module or "")


@cache
def _may_be_submodule(module: str, name: str) -> bool:
    dirs = _package_dirs(module)
    if dirs is None:
        return True
    return any(os.path.isdir(os.path.join(d, name)) or any(os.path.isfile(os.path.join(d, name + s)) for s in _SUFFIXES)
               for d in dirs)


_SUFFIXES = tuple(importlib.machinery.all_suffixes())


@cache
def _package_dirs(module: str) -> tuple[str, ...] | None:
    """The directories of package `module`, `()` for a module that is no package, None where it can't be found; found on
    the file system, importing nothing."""
    parts = module.split(".")
    if not parts[0] or module.startswith("."):
        return None
    loaded = sys.modules.get(parts[0])
    spec = loaded.__spec__ if loaded is not None else importlib.util.find_spec(parts[0])
    if spec is None:
        return None
    if spec.submodule_search_locations is None:
        return () if len(parts) == 1 else None
    dirs = list(spec.submodule_search_locations)
    for i, p in enumerate(parts[1:], start=2):
        sub = [os.path.join(d, p) for d in dirs if os.path.isdir(os.path.join(d, p))]
        if sub:
            dirs = sub
            continue
        if i == len(parts) and any(os.path.isfile(os.path.join(d, p + s)) for d in dirs for s in _SUFFIXES):
            return ()
        return None
    return tuple(dirs)


__all__ = ["DOMAIN", "code_digest", "source_digest"]
