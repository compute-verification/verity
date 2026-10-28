"""`catalog.load()` registers exactly `COUNT` Definitions and primitives: a module that stops importing or registering, or
one that registers more, changes the count and fails here.  Change `COUNT` in the change that adds or removes a Definition.
Each scope of `load` registers the Definitions of the walk it replaced (`REPLACED`), in the same order, whatever the registry
holds: a module only `load`'s walk reaches, a registry subpackage's, can't register on import or move one unnoticed.

The count is the registry's own: PoUW's circuit (the `Ncp*` and `Pouw*` Definitions `verity.core.protocols.pouw.circuit`
registers) is not in it.

Each check reads `catalog.load()` in an interpreter of its own (`fresh`): the registry and `sys.modules` are the process's, and
both grow as code runs (a constant or input gate registers the first time a body uses it, `prims.const` and
`program.input_prim`), so counted in this process they would hold whatever any test before evaluated or imported."""
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
    "count": len(registry.defs),
    "ids": sorted(registry.defs),
    "errors": errors,
    "core": sorted(m for m in sys.modules if m.startswith("verity.core")),
    "integration": sorted(m for m, mod in list(sys.modules.items())
                          if Path(getattr(mod, "__file__", None) or "/").resolve().is_relative_to(repo / "integrations")),
    "in_verity": Path(catalog.__file__).resolve().is_relative_to(repo / "verity"),
}))
"""


@cache
def fresh() -> dict:
    """`catalog.load()`'s registry (its count and ids), import errors and the modules it imports, in an interpreter that ran
    nothing before."""
    p = subprocess.run([sys.executable, "-c", _LOAD, str(REPO)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(p.stdout.splitlines()[-1])

COUNT = 1261

COUNT -= 3  # `GeluTanhMulBf16_v2`, `GeluTanhMul_v2` and `GeluTanhMul_v3`, retired once `GeluTanhMul_v4` superseded them

COUNT += 1  # `dead_free`'s AttnBlockFirst_v2, a line of its own so it merges with any other change to the count

COUNT += 26  # the training kernels: `torch_attention`'s 7, `torch_loss`'s 13 and `torch_embedding`'s 6
COUNT += 6  # the step's blocks: `torch_attention`'s 2, `torch_loss`'s 2 rows, `torch_embedding`'s entry and tied

COUNT += 1  # `torch_embedding`'s forward, `EmbeddingTorch_v1`

COUNT += 11  # `torch_embedding`'s EmbeddingBackwardTied_v2, the ids sorted once, and its 10 parts

COUNT += 10  # `torch_clip`'s, clip_grad_norm_ on bf16 grads

COUNT += 1  # dead_free's TokenSelectGroups_v1, a line of its own so it merges with any other change to the count

COUNT += 2  # ops' `ScaledMmS8ResidualCoordinateRows_v1` and `_v2`, a line of its own

COUNT += 2  # `b1`'s RMSNormFusedCudaAdd_v1 and RMSNormFusedCudaTail_v1, a line of its own so it merges with any other change to the count

COUNT += 2  # ops' `PvRowPad_v1` and `_v2`, a line of its own so it merges with any other change to the count

COUNT += 3  # io_shell's `ScanSlots_v2` (boolean_io_shell's), `OutCollect_v1` and `SlotEntry_v1`, a line of its own so it merges with any other change to the count

COUNT -= 1  # io_shell's `SlotRow_v1`, which `ScanSlots_v2` gives the first position's slot in place of

COUNT -= 1  # io_shell's `SlotMask_v1`, since a slot's entry is all zero when it is empty

COUNT += 2  # `int8_exact`'s Int8RowQuant_v3 and `boolean_int8`'s Int8RowQuant_v4, its σ, a line of its own

COUNT += 14  # `torch_train`'s 7 (Transpose_v1, RmsNormTorchBackward_v1, the SiLU five) and `torch_optim`'s 7

COUNT += 1  # `torch_train`'s RmsNormTorchWeightGrad_v1

COUNT += 1  # `cublas_bf16`'s CublasSplitKGemmBf16PartialsAligned_v1

COUNT += 3  # `torch_train`'s RmsNormTorch_v1 with its In and Out parts

COUNT += 5  # `torch_step`'s TrainStep_v1 with its TrainStepConstants_v1, and F.linear's three GEMMs, LinearTorch_v1,
# LinearTorchDx_v1 and LinearTorchDw_v1

COUNT += 1  # ops' `HopperBF16WgmmaDot16_v3`, a line of its own so it merges with any other change to the count

COUNT += 4  # `cublas_bf16`'s `HOPPER_STEP_V3`, the four `HopperBf16Dot16Step*_v2`


def test_the_catalog_loads_every_definition_and_no_other():
    got = fresh()
    assert not got["errors"], got["errors"]
    assert got["count"] == COUNT, f"catalog.load() registers {got['count']} Definitions, not {COUNT}"


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
    and registers Definitions the registry alone does not: in a process of its own, so the count above stays the registry's."""
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
    assert alone == COUNT and both > alone, (alone, both)


def test_the_catalog_reads_no_protocol():
    assert not fresh()["core"], f"a registry module imports verity.core: {fresh()['core']}"


def test_the_catalog_in_verity_loads_nothing_of_the_integration():
    """Holds once the catalog sits in `verity` (the move); before it, the catalog is the integration's own."""
    if not fresh()["in_verity"]:
        return
    assert not fresh()["integration"], f"loading the catalog imports the integration: {fresh()['integration']}"
