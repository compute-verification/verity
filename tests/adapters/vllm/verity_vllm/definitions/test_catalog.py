"""Each scope of `load` registers the Definitions of the walk it replaced (`REPLACED`), in the same order, whatever the registry
holds: a module only `load`'s walk reaches, a registry subpackage's, can't register on import or move one unnoticed.

Each check reads `catalog.load()` in an interpreter of its own (`fresh`): the registry and `sys.modules` are the process's, and
both grow as code runs (a constant or input gate registers the first time a body uses it, `prims.const` and
`program.input_prim`), so read in this process they would hold whatever any test before evaluated or imported."""
from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]

#: run in a fresh interpreter: what importing every registry module registers and imports, as JSON on its last line
_LOAD = """
import json, sys
from pathlib import Path
from verity_vllm.definitions.registry import catalog
registry, errors = catalog.load()
repo = Path(sys.argv[1])
own = Path(catalog.__file__).resolve().parents[1]
print(json.dumps({
    "ids": sorted(registry.defs),
    "errors": errors,
    "core": sorted(m for m in sys.modules if m.startswith("verity.protocols")),
    "integration": sorted(m for m, mod in list(sys.modules.items())
                          if (f := Path(getattr(mod, "__file__", None) or "/").resolve()).is_relative_to(repo / "adapters")
                          and not f.is_relative_to(own) and f != own.parent / "__init__.py"),
    "package": catalog.__name__.rsplit(".", 2)[0],
    "catalog_packages": sorted(sys.modules["verity.circuits.defs"].CATALOG_PACKAGES),
}))
"""


@cache
def fresh() -> dict:
    """`catalog.load()`'s registry (its ids, sorted), import errors and the modules it imports, in an interpreter that ran
    nothing before."""
    p = subprocess.run([sys.executable, "-c", _LOAD, str(REPO)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.splitlines()[-1])

#: by scope of `catalog.load`, the walk it replaced: the registry's own modules alone, none of a subpackage's, and replay's and
#: the red-team demo's every module of ops and of the registry
REPLACED = {
    "registry": ("import importlib, pkgutil\n"
                 "import verity_vllm.definitions.registry as package\n"
                 "for m in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):\n"
                 "    if m.name not in ('quarantine', 'catalog'):\n"
                 "        importlib.import_module(package.__name__ + '.' + m.name)\n"),
    ("ops", "registry"): ("import importlib, pkgutil\n"
                          "for name in ('verity.circuits.ops', 'verity_vllm.definitions.registry'):\n"
                          "    package = importlib.import_module(name)\n"
                          "    for m in pkgutil.walk_packages(package.__path__, name + '.'):\n"
                          "        importlib.import_module(m.name)\n"),
}


def _registered(code: str) -> list[str]:
    """The ids `REGISTRY` holds, in registration order, once ``code`` has run in a process of its own: the test modules beside
    this one import registry modules as they are collected, which reorders this process's registry."""
    p = subprocess.run([sys.executable, "-c", code + "from verity.circuits.defs import REGISTRY\nprint('\\n'.join(REGISTRY.defs))\n"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.split()


def test_each_scope_registers_what_the_walk_it_replaced_did_in_its_order():
    """Each scope of `catalog.load` registers the Definitions the walk it replaced (`REPLACED`) registered, in the same order:
    the registry scope's walk reaches its subpackages' modules, which the old one never imported, so one that registers on import
    fails here.  Nothing to update when a Definition is added: both walks register it."""
    for scope, walk in REPLACED.items():
        got = _registered(f"from verity_vllm.definitions.registry import catalog\nassert not catalog.load({scope!r})[1]\n")
        want = _registered(walk)
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        assert got == want, (f"catalog.load({scope!r}) registers {len(got)} Definitions, the walk it replaced {len(want)}, first apart "
                             f"at {at}: {got[at:at + 3]} against {want[at:at + 3]} (a module only load's walk imports registers on import)")


def test_the_catalog_reads_no_protocol():
    assert not fresh()["core"], f"a registry module imports verity.core: {fresh()['core']}"


def test_the_catalog_loads_nothing_of_the_integration():
    """No module under `adapters/` but the catalog's own package and the `__init__.py` of the package that holds it."""
    assert not fresh()["integration"], f"loading the catalog imports the integration: {fresh()['integration']}"


def test_the_package_names_itself_a_catalog_package():
    """The package that holds the registry names itself a catalog package as it loads (`verity.circuits.defs.catalog_package`),
    by its own name: a window takes its templates from a catalog package by that name, wherever the package is."""
    got = fresh()
    assert got["package"] in got["catalog_packages"], (
        f"{got['package']} is no catalog package ({got['catalog_packages']}): its __init__.py calls catalog_package(__name__)")

