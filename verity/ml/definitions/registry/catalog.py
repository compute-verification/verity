"""The integration's registry, every module loaded: the one registry each consumer of a Program reads.

GP-01, the Match, the query views and the manifest with its word check all resolve Definitions and primitives through `load`, so a
Definition or primitive a new registry module registers resolves everywhere, not only where someone imported its module.
"""
from __future__ import annotations

_LOADED: list = []


def load():
    """(REGISTRY, import errors) with every module of `verity.ml.definitions.registry` imported (`quarantine` and this module aside).
    A module that fails to import is named in the errors, never read as "not registered"."""
    if not _LOADED:
        import importlib
        import pkgutil

        from verity.ml.circuits.defs import REGISTRY
        errors = []
        package = importlib.import_module(__name__.rsplit(".", 1)[0])
        for m in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if m.name in ("quarantine", "catalog"):
                continue
            try:
                # the literal prefix is how tests/dead_code_census.py sees every registry module reached from here
                importlib.import_module(f"verity.ml.definitions.registry.{m.name}")
            except Exception as e:  # noqa: BLE001  (recorded: the error names the module a lookup then misses)
                errors.append(f"{package.__name__}.{m.name}: {type(e).__name__}: {e}")
        _LOADED.append((REGISTRY, errors))
    return _LOADED[0]
