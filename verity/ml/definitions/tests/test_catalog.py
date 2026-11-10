"""`catalog.load()` registers exactly the Definitions and primitives `catalog_ids.json` lists, by id: a module that stops
importing or registering, or one that registers more, fails here and names the ids.  Update the list in the change that adds or
removes a Definition (`uv run python verity/ml/definitions/tests/test_catalog.py --update`).
Each scope of `load` registers the Definitions of the walk it replaced (`REPLACED`), in the same order, whatever the registry
holds: a module only `load`'s walk reaches, a registry subpackage's, can't register on import or move one unnoticed.

The list is the registry's own: PoUW's circuit (the `Ncp*` and `Pouw*` Definitions `verity.ml.protocols.pouw.circuit`
registers) is not in it.

Each check reads `catalog.load()` in an interpreter of its own (`fresh`): the registry and `sys.modules` are the process's, and
both grow as code runs (a constant or input gate registers the first time a body uses it, `prims.const` and
`program.input_prim`), so read in this process they would hold whatever any test before evaluated or imported."""
from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

#: run in a fresh interpreter: what importing every registry module registers and imports, as JSON on its last line
_LOAD = """
import json, sys
from pathlib import Path
from verity.ml.definitions.registry import catalog
registry, errors = catalog.load()
repo = Path(sys.argv[1])
print(json.dumps({
    "ids": sorted(registry.defs),
    "errors": errors,
    "core": sorted(m for m in sys.modules if m.startswith("verity.core")),
    "integration": sorted(m for m, mod in list(sys.modules.items())
                          if Path(getattr(mod, "__file__", None) or "/").resolve().is_relative_to(repo / "integrations")),
}))
"""


@cache
def fresh() -> dict:
    """`catalog.load()`'s registry (its ids, sorted), import errors and the modules it imports, in an interpreter that ran
    nothing before."""
    p = subprocess.run([sys.executable, "-c", _LOAD, str(REPO)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.splitlines()[-1])

#: the ids `catalog.load()` registers, sorted, one a line
IDS = Path(__file__).with_name("catalog_ids.json")


def pinned() -> list[str]:
    return json.loads(IDS.read_text())


def test_the_catalog_loads_every_definition_and_no_other():
    got = fresh()
    assert not got["errors"], got["errors"]
    want = pinned()
    added, gone = sorted(set(got["ids"]) - set(want)), sorted(set(want) - set(got["ids"]))
    assert got["ids"] == want, (
        f"catalog.load() registers {len(added)} ids catalog_ids.json lacks, {added}, and lacks {len(gone)} it lists, {gone}. "
        "If the change adds or removes them, update the list: uv run python verity/ml/definitions/tests/test_catalog.py --update")


#: by scope of `catalog.load`, the walk it replaced: the registry's own modules alone, none of a subpackage's, and replay's and
#: the red-team demo's every module of ops and of the registry
REPLACED = {
    "registry": ("import importlib, pkgutil\n"
                 "import verity.ml.definitions.registry as package\n"
                 "for m in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):\n"
                 "    if m.name not in ('quarantine', 'catalog'):\n"
                 "        importlib.import_module(package.__name__ + '.' + m.name)\n"),
    ("ops", "registry"): ("import importlib, pkgutil\n"
                          "for name in ('verity.ml.circuits.ops', 'verity.ml.definitions.registry'):\n"
                          "    package = importlib.import_module(name)\n"
                          "    for m in pkgutil.walk_packages(package.__path__, name + '.'):\n"
                          "        importlib.import_module(m.name)\n"),
}


def _registered(code: str) -> list[str]:
    """The ids `REGISTRY` holds, in registration order, once ``code`` has run in a process of its own: the test modules beside
    this one import registry modules as they are collected, which reorders this process's registry."""
    p = subprocess.run([sys.executable, "-c", code + "from verity.ml.circuits.defs import REGISTRY\nprint('\\n'.join(REGISTRY.defs))\n"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.split()


def test_each_scope_registers_what_the_walk_it_replaced_did_in_its_order():
    """Each scope of `catalog.load` registers the Definitions the walk it replaced (`REPLACED`) registered, in the same order:
    the registry scope's walk reaches its subpackages' modules, which the old one never imported, so one that registers on import
    fails here.  Nothing to update when a Definition is added: both walks register it."""
    for scope, walk in REPLACED.items():
        got = _registered(f"from verity.ml.definitions.registry import catalog\nassert not catalog.load({scope!r})[1]\n")
        want = _registered(walk)
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        assert got == want, (f"catalog.load({scope!r}) registers {len(got)} Definitions, the walk it replaced {len(want)}, first apart "
                             f"at {at}: {got[at:at + 3]} against {want[at:at + 3]} (a module only load's walk imports registers on import)")


def test_the_ops_scope_imports_every_module_of_ops():
    """`load(("ops", "registry"))` imports every module of `verity.ml.circuits.ops` and of its subpackages besides the registry's,
    and registers Definitions the registry alone does not: in a process of its own, so the list above stays the registry's."""
    code = ("import pkgutil, sys\n"
            "from verity.ml.circuits import ops\n"
            "from verity.ml.definitions.registry import catalog\n"
            "alone = len(catalog.load()[0].defs)\n"
            "registry, errors = catalog.load(('ops', 'registry'))\n"
            "assert not errors, errors\n"
            "missing = [m.name for m in pkgutil.walk_packages(ops.__path__, 'verity.ml.circuits.ops.') if m.name not in sys.modules]\n"
            "assert not missing, missing\n"
            "print(alone, len(registry.defs))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    alone, both = map(int, p.stdout.split())
    assert alone == len(pinned()) and both > alone, (alone, both)


def test_the_catalog_reads_no_protocol():
    assert not fresh()["core"], f"a registry module imports verity.core: {fresh()['core']}"


def test_the_catalog_loads_nothing_of_the_integration():
    assert not fresh()["integration"], f"loading the catalog imports the integration: {fresh()['integration']}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        sys.exit("usage: test_catalog.py --update  (rewrites catalog_ids.json from a fresh catalog.load())")
    got = fresh()
    assert not got["errors"], got["errors"]
    IDS.write_text(json.dumps(got["ids"], indent=0) + "\n")
    print(f"{IDS.relative_to(REPO)}: {len(got['ids'])} ids")
