"""`catalog.load()` registers exactly `COUNT` Definitions and primitives: a module that stops importing or registering, or
one that registers more, changes the count and fails here.  Change `COUNT` in the change that adds or removes a Definition.

The count is the registry's own: PoUW's circuit (the `Ncp*` and `Pouw*` Definitions `verity.core.protocols.pouw.circuit`
registers) is not in it."""
from __future__ import annotations

import sys
from pathlib import Path

from verity.ml.definitions.registry import catalog

REPO = Path(__file__).resolve().parents[4]

COUNT = 1261

COUNT -= 3  # `GeluTanhMulBf16_v2`, `GeluTanhMul_v2` and `GeluTanhMul_v3`, retired once `GeluTanhMul_v4` superseded them

COUNT += 26  # the training kernels: `torch_attention`'s 7, `torch_loss`'s 13 and `torch_embedding`'s 6

COUNT += 2  # ops' `ScaledMmS8ResidualCoordinateRows_v1` and `_v2` (CIR-107), a line of its own

COUNT += 2  # ops' `PvRowPad_v1` and `_v2`, a line of its own so it merges with any other change to the count

COUNT += 3  # io_shell's `ScanSlots_v1`, `OutCollect_v1` and `SlotEntry_v1`, a line of its own so it merges with any other change to the count

COUNT += 14  # `torch_train`'s 7 (Transpose_v1, RmsNormTorchBackward_v1, the SiLU five) and `torch_optim`'s 7

COUNT += 1  # `torch_train`'s RmsNormTorchWeightGrad_v1

COUNT += 1  # `cublas_bf16`'s CublasSplitKGemmBf16PartialsAligned_v1


def test_the_catalog_loads_every_definition_and_no_other():
    registry, errors = catalog.load()
    assert not errors, errors
    assert len(registry.defs) == COUNT, f"catalog.load() registers {len(registry.defs)} Definitions, not {COUNT}"


def test_the_catalog_reads_no_protocol():
    catalog.load()
    assert not [m for m in sys.modules if m.startswith("verity.core")], "a registry module imports verity.core"


def test_the_catalog_in_verity_loads_nothing_of_the_integration():
    """Holds once the catalog sits in `verity` (the move); before it, the catalog is the integration's own."""
    catalog.load()
    if not Path(catalog.__file__).resolve().is_relative_to(REPO / "verity"):
        return
    loaded = sorted(m for m, mod in list(sys.modules.items())
                    if Path(getattr(mod, "__file__", None) or "/").resolve().is_relative_to(REPO / "integrations"))
    assert not loaded, f"loading the catalog imports the integration: {loaded}"
