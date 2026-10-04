"""The registry's Definition modules import only Definitions: the standard library, numpy, `verity` and other Definition
modules, function-local imports and `importlib.import_module` strings included.  They are what the catalog takes
(digest-addressed public parameters that import only `verity`); a twin, a vocabulary or a lifting rule that reads the rest of
the integration lives beside them, in `program/twins/`, `program/vocabulary/` or `program/lifting/`, and imports them, never
the other way round.

`DEFINITION_MODULES` lists the modules split so far, by family; `OUTSIDE_THE_FAMILIES` names the other modules a Definition
module may import while its own family is not split yet (each is another family's Definition module, or a protocol's).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REG = "verity_vllm.program.registry"

DEFINITION_MODULES = {
    # prims / serve (tools/migrate/split_prims_serve.py)
    **{f"{REG}.{m}": "prims-serve" for m in ("prims", "ref_prims", "pad_prims", "lifted", "tables", "value_inputs", "operand_widths",
                                              "b1", "b1_tp2", "serve3", "spec", "pouw_rows", "catalog")},
    REG: "prims-serve",
}
#: Definition modules of families not split yet, and a protocol's Definitions, that a split family's Definitions import
OUTSIDE_THE_FAMILIES = {
    f"{REG}.dense": "dense / softcap attention (value_inputs, tables)",
    f"{REG}.fa2_softcap": "softcap attention (value_inputs)",
    f"{REG}.fa3_check_inf": "FA3 (value_inputs)",
    f"{REG}.topp_split": "top-p sampling (lifted's split rule)",
    f"{REG}.topp_word_gates": "top-p sampling (the registry's lazy families)",
    f"{REG}.sampling": "sampling (the registry's lazy families)",
    "verity_pouw.circuit.ncp2": "PoUW's NcpDequant (pouw_rows)",
}
ALLOWED_TOP = {"numpy", "verity", "__future__"}


def _path(mod: str) -> Path:
    base = ROOT.joinpath(*mod.split("."))
    return base / "__init__.py" if (base / "__init__.py").is_file() else base.with_suffix(".py")


def _imports(mod: str) -> list[tuple[str, int]]:
    path = _path(mod)
    package = mod.split(".") if path.name == "__init__.py" else mod.split(".")[:-1]
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join((package[:len(package) - node.level + 1] if node.level else []) + ([node.module] if node.module else []))
            out += [(f"{base}.{a.name}" if f"{base}.{a.name}" in DEFINITION_MODULES or f"{base}.{a.name}" in OUTSIDE_THE_FAMILIES
                     else base, node.lineno) for a in node.names]
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out.append((node.args[0].value, node.lineno))
    return out


@pytest.mark.parametrize("mod", sorted(DEFINITION_MODULES))
def test_a_definition_module_imports_only_definitions(mod):
    bad = [f"{mod}:{line}: {name}" for name, line in _imports(mod)
           if name.split(".")[0] not in ALLOWED_TOP and name.split(".")[0] not in sys.stdlib_module_names
           and name not in DEFINITION_MODULES and name not in OUTSIDE_THE_FAMILIES]
    assert bad == [], "a Definition module imports outside the Definitions:\n" + "\n".join(bad)


def test_the_listed_modules_exist():
    assert [m for m in [*DEFINITION_MODULES, *OUTSIDE_THE_FAMILIES] if m.startswith("verity_vllm") and not _path(m).is_file()] == []
