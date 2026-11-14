"""The dense family's Definition modules (dense, activation, norms, RoPE, GEMM bias) import only `verity`, other Definition
modules and the evidence their conformance texts cite (`EVIDENCE`), so they can move to `verity/circuits/` as they are. Their vocabulary lives in the integration package:
`program.frontend.gemm_targets`."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.lint._imports import core_less_definitions as _core

REGISTRY = "verity.definitions.registry"
_SPEC = importlib.util.find_spec(REGISTRY)
assert _SPEC is not None and _SPEC.origin is not None
REGISTRY_DIR = Path(_SPEC.origin).parent
FAMILY = ("dense", "act", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_activation", "boolean_silu",
          "boolean_rope", "boolean_tanh", "cuda_rmsnorm", "gemm_bias", "layer_norm", "gelu_erf_table", "traced_parts")
#: other families' Definition modules the family reads: the primitives, the b1 composites, `sampling.bit_at` (`UnpackF32_v1`),
#: `value_inputs`' word views of the norms and scalar rows with their f32 word an input, and `tables`' `tanh_rn` region
READS = ("prims", "b1", "sampling", "value_inputs", "tables")
THIRD_PARTY = ("numpy",)
#: the catalog, where the layout move put core's Definitions, silicon models and targets: not all of `verity.circuits`,
#: which holds the kernels and the private circuits too
CATALOG = ("verity.circuits.ops", "verity.circuits.silicon", "verity.circuits.targets")
#: `evidence.json`'s ids, read with the standard library only
EVIDENCE = "verity.definitions.evidence"
MOVED = ("verity_vllm.program.frontend.gemm_targets",)


def _imports(path: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    """(line, module, names) of every import in the file, at any depth, relative ones resolved against the registry."""
    out: list[tuple[int, str, tuple[str, ...]]] = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name, ()) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = REGISTRY.rsplit(".", n.level - 1)[0] if n.level else ""
            mod = ".".join(p for p in (base, n.module or "") if p)
            out.append((n.lineno, mod, tuple(a.name for a in n.names)))
    return out


def _foreign(mod: str, names: tuple[str, ...]) -> list[str]:
    top = mod.partition(".")[0]
    allowed = set(FAMILY) | set(READS)
    if mod == REGISTRY:
        return [f"{REGISTRY}.{x}" for x in names if x not in allowed]
    if mod.startswith(REGISTRY + "."):
        return [] if mod[len(REGISTRY) + 1:].split(".")[0] in allowed else [mod]
    if top in sys.stdlib_module_names or top in THIRD_PARTY or _core(mod) or mod in (*CATALOG, EVIDENCE) or mod.startswith(tuple(c + "." for c in CATALOG)):
        return []
    return [mod]


@pytest.mark.parametrize("name", FAMILY)
def test_a_family_definition_module_imports_only_verity_and_definition_modules(name):
    path = REGISTRY_DIR / f"{name}.py"
    bad = [f"line {line}: {m}" for line, mod, names in _imports(path) for m in _foreign(mod, names)]
    assert not bad, f"{name}.py imports outside verity and the Definition modules: {bad}"


def test_the_family_twins_and_vocabulary_left_the_registry():
    for module in MOVED:
        name = module.rpartition(".")[2]
        assert not (REGISTRY_DIR / f"{name}.py").exists(), f"{name} is back in the registry"
        assert importlib.import_module(module).__name__ == module


def test_the_boundary_names_a_foreign_import():
    assert _foreign("verity_vllm.config", ("ROOT",)) == ["verity_vllm.config"]
    assert _foreign(REGISTRY, ("b1", "moe_difftest")) == [f"{REGISTRY}.moe_difftest"]
    assert _foreign(f"{REGISTRY}.gemm_targets", ("GemmTarget",)) == [f"{REGISTRY}.gemm_targets"]
    assert _foreign("torch", ()) == ["torch"]
    assert _foreign(f"{REGISTRY}.prims", ("BF16",)) == [] and _foreign("verity.circuits.defs", ("bind",)) == []


@pytest.mark.parametrize("mod", ["verity.circuits.ops", "verity.circuits.ops.prims", "verity.circuits.silicon", "verity.circuits.targets.target"])
def test_the_catalog_is_let_through(mod):
    assert _foreign(mod, ()) == []


@pytest.mark.parametrize("mod", ["verity.circuits.kernels", "verity.circuits.kernels.tc", "verity.circuits.private_circuits.universal"])
def test_the_kernels_and_the_private_circuits_are_not_the_catalog(mod):
    assert _foreign(mod, ()) == [mod]
