"""The primitives, composites and serving family's Definition modules import only `verity`, Definition modules, the standard
library and numpy: every import in the module, a function's lazy one included.  Twins (numpy/torch/CUDA references,
difftests, capture helpers) and vocabulary (which Definition a serving role binds) live outside the registry
(`tools/migrate/split_prims_serve.py`), so a catalog can take these modules without the integration."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[2] / "verity_vllm" / "program" / "registry"
REG = "verity_vllm.program.registry"
FAMILY = ("__init__", "catalog", "prims", "ref_prims", "pad_prims", "lifted", "tables", "value_inputs", "operand_widths", "b1",
          "b1_tp2", "serve3", "spec", "pouw_rows")
#: the other families' Definition modules this family reads (each family's own split judges its modules)
OTHER_FAMILIES = {"dense", "fa2_softcap", "fa3_check_inf", "sampling", "topp_geometry", "topp_word_gates"}
THIRD_PARTY = {"numpy"}
#: the catalog, where the layout move put core's Definitions and silicon models
CATALOG = "verity.ml.circuits"
#: (module, what it imports outside the rule) -> why it stays
EXCEPTIONS = {("pouw_rows", "verity.core.protocols.pouw.circuit.ncp2"): "NcpDequant is PoUW's own Definition, which the row binds"}


def _imports(name: str) -> list[tuple[int, str]]:
    """(line, absolute module) of every import in the module; `from REG import m` counts as `REG.m`."""
    path = REGISTRY / f"{name}.py"
    package = REG                                    # every family module, `__init__` too, resolves relative imports here
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                up = package.split(".")[: len(package.split(".")) - (n.level - 1)]
                base = ".".join(up + ([n.module] if n.module else []))
            if base == REG:
                out += [(n.lineno, f"{REG}.{a.name}") for a in n.names]
            else:
                out.append((n.lineno, base))
    return out


def _core(name: str) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, C-Flock, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.ml.commitments.multiproof", "verity.ml.private_circuits", "verity.ml.circuits.ops",
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock")
    return under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.core.claims"))))


def _allowed(mod: str) -> bool:
    top = mod.split(".")[0]
    if _core(mod) or top in sys.stdlib_module_names or top == "__future__" or top in THIRD_PARTY:
        return True
    if mod == REG or mod == CATALOG or mod.startswith(CATALOG + "."):
        return True
    if mod.startswith(REG + "."):
        return mod[len(REG) + 1:].split(".")[0] in set(FAMILY) | OTHER_FAMILIES
    return False


@pytest.mark.parametrize("name", FAMILY)
def test_definition_module_imports_only_definitions(name):
    bad = [f"{name}.py:{line}: {mod}" for line, mod in _imports(name) if not _allowed(mod) and (name, mod) not in EXCEPTIONS]
    assert not bad, ("a Definition module imports what is not a Definition module (move the code that needs it out of the "
                     "registry, as tools/migrate/split_prims_serve.py does):\n  " + "\n  ".join(bad))


def test_every_exception_is_still_needed():
    stale = [k for k in EXCEPTIONS if k[1] not in {mod for _, mod in _imports(k[0])}]
    assert not stale, f"exceptions whose import is gone; delete them: {stale}"


def test_the_family_is_in_the_registry():
    missing = [n for n in FAMILY if not (REGISTRY / f"{n}.py").exists()]
    assert not missing, f"Definition modules no longer in the registry (moved? update FAMILY): {missing}"
