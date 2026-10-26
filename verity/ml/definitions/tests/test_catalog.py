"""`catalog.load()` registers exactly the Definitions and primitives the registry held when it moved into
`verity.ml.definitions` (circuits' condition on the move, CIR-93): a module that stops importing or registering, or one
that registers more, changes the count and fails here.  Change `COUNT` in the change that adds or removes a Definition.

The count is the registry's own: 1,287 on main before the move less PoUW's circuit (the 47 `Ncp*` and `Pouw*` Definitions
`verity.core.protocols.pouw.circuit` registers), which `pouw_rows` stopped importing once `NcpDequant` moved to
`verity.ml.circuits.ops.ncp`, 1,242 since `AttentionHead_v18` and `Attention_v18` (bcf1e1aade), and 1,250 since
`AttentionHead_v19`, `Attention_v19`, `AttnBlockFirst_v1` and `AttnHeadEnd_v1` (d37b243ac6) and ops' `GemmCoordinatePad_v1`,
`_v2`, `ScaledMmS8CoordinateRows_v1` and `_v2` (fa7065c758)."""
from __future__ import annotations

import sys
from pathlib import Path

from verity_vllm.program.registry import catalog

REPO = Path(__file__).resolve().parents[4]

COUNT = 1250


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
