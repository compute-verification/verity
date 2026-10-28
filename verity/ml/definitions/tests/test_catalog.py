"""`catalog.load()` registers exactly `COUNT` Definitions and primitives: a module that stops importing or registering, or
one that registers more, changes the count and fails here.  Change `COUNT` in the change that adds or removes a Definition.

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


def test_the_catalog_reads_no_protocol():
    assert not fresh()["core"], f"a registry module imports verity.core: {fresh()['core']}"


def test_the_catalog_in_verity_loads_nothing_of_the_integration():
    """Holds once the catalog sits in `verity` (the move); before it, the catalog is the integration's own."""
    if not fresh()["in_verity"]:
        return
    assert not fresh()["integration"], f"loading the catalog imports the integration: {fresh()['integration']}"
