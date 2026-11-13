"""What circuit-check reads of the catalog (`circuit_check.targets.HOOKS`), handed by the module that loads it, so circuit-check
names no module of the catalog's.  circuit-check reads these before it imports any module with a sidecar, so this file
imports nothing at its top level: each function imports what it hands."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def enumerators() -> set[tuple[str, str]]:
    """The functions (module, qualified name) that list a package only to import each module listed (`cache.enumerators`)."""
    return {("verity.definitions.registry.catalog", "load"), ("verity.definitions.boolean", "_load"),
            (__name__, "load_catalog")}


def load_catalog() -> None:
    """Import every module of the registry and the kernels of the replay's rows and twins, which register with core."""
    import verity.definitions.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity.definitions.numerics import rows, twins  # noqa: F401


def preload() -> dict[str, Any]:
    from verity.definitions.numerics import twins

    return twins.preload()


def kernels() -> list[Any]:
    from verity.definitions.numerics import kernel_registry

    return kernel_registry.registered()


def tanh_rn() -> tuple[Any, Any, int]:
    from verity.definitions.registry import tables

    return tables.tanh_rn_table, tables.TANH_RN_REGIONS, tables.TANH_RN_LO


def row_families() -> set[str]:
    from verity.definitions.numerics import rows

    return set(rows.ROWS)


def boolean_table() -> tuple[dict[str, list], dict[str, list], list[str]]:
    from verity.definitions.boolean import _load

    return _load()


def made_on_demand(fid: str) -> Any:
    from verity.definitions.boolean import made_on_demand

    return made_on_demand(fid)


def boolean_version() -> tuple[Any, type[Exception]]:
    from verity.definitions.boolean import NoBooleanVersion, boolean_version

    return boolean_version, NoBooleanVersion


def lift_composite(defn: Any) -> Any:
    from verity.definitions.registry.lifted import lift_composite

    return lift_composite(defn)
