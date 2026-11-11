"""The integration's registry, every module loaded: the one registry each consumer of a Program reads, and the one loader of
Verity's Definitions by package.

The workload Program's driver, the Match, the query views and the manifest with its word check all resolve Definitions and
primitives through `load`, so a Definition or primitive a new registry module registers resolves everywhere, not only where someone
imported its module.  A reader of Definitions beyond the registry names their packages as `load`'s scope (`SCOPES`).
"""
from __future__ import annotations

#: the packages `load` imports, by scope
SCOPES = {"registry": "verity.definitions.registry", "ops": "verity.definitions.ops"}
_LOADED: dict[str, list[str]] = {}


def load(scope: str | tuple[str, ...] = "registry", _package=None, _errors: list[str] | None = None):
    """(REGISTRY, import errors) with every module of each package `scope` names (`SCOPES`) imported, its subpackages' too, in
    `pkgutil.walk_packages` order (`quarantine`, `tests` and this module aside), each package once a process.  A module that fails
    to import is named in the errors, never read as "not registered".

    ``_package`` and ``_errors`` are its own descent into a subpackage, which imports that package's modules in name order, each
    subpackage's right after it, and returns nothing: the descent stays in `load`, so circuit-check's cache holds what it lists by
    the modules' names (`cache._ENUMERATORS`), and each name a listing yields is only compared, imported or written into an
    error."""
    import importlib
    import pkgutil

    if _package is not None:
        errors = _errors
        for m in sorted(pkgutil.iter_modules(_package.__path__), key=lambda m: m.name):
            if m.name in ("quarantine", "tests") or f"{_package.__name__}.{m.name}" == __name__:
                continue
            try:
                # the literal prefix is how tests/dead_code_census.py sees every registry module reached from here
                child = importlib.import_module(f"verity.definitions.registry.{m.name}" if _package.__name__ == SCOPES["registry"]
                                                else f"{_package.__name__}.{m.name}")
            except Exception as e:  # noqa: BLE001  (recorded: the error names the module a lookup then misses)
                errors.append(f"{_package.__name__}.{m.name}: {type(e).__name__}: {e}")
                continue
            if hasattr(child, "__path__"):
                load(scope, child, errors)
        return None

    from verity.circuits.defs import REGISTRY
    out: list[str] = []
    for s in (scope,) if isinstance(scope, str) else scope:
        if s not in _LOADED:
            found: list[str] = []
            try:
                package = importlib.import_module(SCOPES[s])
            except Exception as e:  # noqa: BLE001  (recorded, as above)
                found.append(f"{SCOPES[s]}: {type(e).__name__}: {e}")
            else:
                load(s, package, found)
            _LOADED[s] = found
        out += _LOADED[s]
    return REGISTRY, out
