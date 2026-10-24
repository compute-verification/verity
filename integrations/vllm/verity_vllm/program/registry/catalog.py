"""The integration's registry, every module loaded: the one registry each consumer of a Program reads.

GP-01, the Match, the query views and the manifest with its word check all resolve Definitions and primitives through `load`, so a
Definition or primitive a new registry module registers resolves everywhere, not only where someone imported its module.
"""
from __future__ import annotations

_LOADED: list = []
#: the entry-point group of the modules outside this package whose Definitions the registry loads too: an integration's
#: Definitions that build on a protocol's (`pouw_rows`, on PoUW's `NcpDequant`), which `verity.ml` may not import
GROUP = "verity.definitions"


def load():
    """(REGISTRY, import errors) with every module of `verity_vllm.program.registry` imported (`quarantine` and this module aside),
    and every module an installed distribution names in `GROUP`.  A module that fails to import is named in the errors, never read
    as "not registered"."""
    if not _LOADED:
        import importlib
        import importlib.metadata
        import pkgutil

        from verity.ml.circuits.defs import REGISTRY
        errors = []
        package = importlib.import_module(__name__.rsplit(".", 1)[0])
        for m in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if m.name in ("quarantine", "catalog"):
                continue
            try:
                # the literal prefix is how tests/dead_code_census.py sees every registry module reached from here
                importlib.import_module(f"verity_vllm.program.registry.{m.name}")
            except Exception as e:  # noqa: BLE001  (recorded: the error names the module a lookup then misses)
                errors.append(f"{package.__name__}.{m.name}: {type(e).__name__}: {e}")
        for ep in sorted(importlib.metadata.entry_points(group=GROUP), key=lambda ep: ep.value):
            try:
                importlib.import_module(ep.value)
            except Exception as e:  # noqa: BLE001  (recorded, as above)
                errors.append(f"{ep.value}: {type(e).__name__}: {e}")
        _LOADED.append((REGISTRY, errors))
    return _LOADED[0]
