"""`catalog.load()` registers exactly `COUNT` Definitions and primitives: a module that stops importing or registering, or
one that registers more, changes the count and fails here.  Change `COUNT` in the change that adds or removes a Definition.
Each scope registers them in the order `ORDER` pins, so a module that registers on import, a subpackage's among them, can't
move a digest while the count holds.  Change `ORDER` in the change that adds, removes or reorders a Definition.

The count is the registry's own: PoUW's circuit (the `Ncp*` and `Pouw*` Definitions `verity.core.protocols.pouw.circuit`
registers) is not in it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from verity.ml.definitions.registry import catalog

REPO = Path(__file__).resolve().parents[4]

COUNT = 1261

COUNT -= 3  # `GeluTanhMulBf16_v2`, `GeluTanhMul_v2` and `GeluTanhMul_v3`, retired once `GeluTanhMul_v4` superseded them

COUNT += 26  # the training kernels: `torch_attention`'s 7, `torch_loss`'s 13 and `torch_embedding`'s 6

COUNT += 2  # ops' `ScaledMmS8ResidualCoordinateRows_v1` and `_v2`, a line of its own

COUNT += 2  # ops' `PvRowPad_v1` and `_v2`, a line of its own so it merges with any other change to the count

COUNT += 3  # io_shell's `ScanSlots_v1`, `OutCollect_v1` and `SlotEntry_v1`, a line of its own so it merges with any other change to the count

COUNT += 14  # `torch_train`'s 7 (Transpose_v1, RmsNormTorchBackward_v1, the SiLU five) and `torch_optim`'s 7

COUNT += 1  # `torch_train`'s RmsNormTorchWeightGrad_v1

COUNT += 1  # `cublas_bf16`'s CublasSplitKGemmBf16PartialsAligned_v1

#: by scope of `catalog.load`, the SHA-512 of its Definitions' ids (`REGISTRY.defs`' keys) newline-joined in registration order:
#: the default, and ops with the registry (replay's and the red-team demo's)
ORDER = {
    "registry":
        "270213f4ab439d78deadf44941a1e1ced8b16401062ccddc58352a2311dfc56c2e8ff32e81c26628054bbce4a61f775cec8fa125876bc8724e9f00b32f29d73a",
    ("ops", "registry"):
        "33cc950b3fa175a02ce72288d12e66e1afae0b06125e272b16d571b903697ef7c60b640e593dca7183981df4622b5ab9787bc5d7da89ac207199684db0fbbd9d",
}


def test_the_catalog_loads_every_definition_and_no_other():
    registry, errors = catalog.load()
    assert not errors, errors
    assert len(registry.defs) == COUNT, f"catalog.load() registers {len(registry.defs)} Definitions, not {COUNT}"


def test_each_scope_registers_its_definitions_in_the_pinned_order():
    """Each scope of `ORDER` registers its Definitions in the order its digest pins: each in a process of its own, since the
    test modules beside this one import registry modules as they are collected, which reorders this process's registry."""
    for scope, want in ORDER.items():
        code = ("import hashlib\n"
                "from verity.ml.definitions.registry import catalog\n"
                f"registry, errors = catalog.load({scope!r})\n"
                "assert not errors, errors\n"
                "print(len(registry.defs), hashlib.sha512('\\n'.join(registry.defs).encode()).hexdigest())\n")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        n, got = p.stdout.split()
        assert got == want, (f"catalog.load({scope!r}) registers its {n} Definitions in another order or set: their ids hash to "
                             f"{got}.  If the change adds, removes or reorders a Definition, set ORDER[{scope!r}] to that digest")


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
