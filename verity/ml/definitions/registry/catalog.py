"""The integration's registry, every module loaded: the one registry each consumer of a Program reads, and the one loader of
Verity's Definitions by package.

GP-01, the Match, the query views and the manifest with its word check all resolve Definitions and primitives through `load`, so a
Definition or primitive a new registry module registers resolves everywhere, not only where someone imported its module.  A reader
of Definitions beyond the registry names their packages as `load`'s scope (`SCOPES`).
"""
from __future__ import annotations

#: the packages `load` imports, by scope
SCOPES = {"registry": "verity.ml.definitions.registry", "ops": "verity.ml.circuits.ops"}
_LOADED: dict[str, list[str]] = {}


def load(scope: str | tuple[str, ...] = "registry"):
    """(REGISTRY, import errors) with every module of each package `scope` names (`SCOPES`) imported, its subpackages' too, in
    `pkgutil.walk_packages` order (`quarantine`, `tests` and this module aside), each package once a process.  A module that fails
    to import is named in the errors, never read as "not registered"."""
    import importlib
    import pkgutil

    from verity.ml.circuits.defs import REGISTRY
    errors: list[str] = []
    for s in (scope,) if isinstance(scope, str) else scope:
        if s not in _LOADED:
            found: list[str] = []
            todo = [(SCOPES[s], True)]
            while todo:
                name, is_package = todo.pop()
                try:
                    module = importlib.import_module(name)
                except Exception as e:  # noqa: BLE001  (recorded: the error names the module a lookup then misses)
                    found.append(f"{name}: {type(e).__name__}: {e}")
                    continue
                if not is_package:
                    continue
                for m in sorted(pkgutil.iter_modules(module.__path__), key=lambda m: m.name, reverse=True):  # popped in name order
                    # the literal prefix is how tests/dead_code_census.py sees every registry module reached from here
                    child = f"verity.ml.definitions.registry.{m.name}" if name == SCOPES["registry"] else f"{name}.{m.name}"
                    if m.name not in ("quarantine", "tests") and child != __name__:
                        todo.append((child, m.ispkg))
            _LOADED[s] = found
        errors += _LOADED[s]
    return REGISTRY, errors
