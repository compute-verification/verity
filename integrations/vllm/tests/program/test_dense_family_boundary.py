"""The dense family's Definition modules (dense, activation, norms, RoPE, GEMM bias) import only `verity` and other Definition
modules, so they can move to `catalog/` as they are. Their twins and vocabulary live in the integration package
(`tools/migrate/split_dense.py`): `properties.difftests.rope_difftest`, `properties.difftests.rope_gptj_difftest`,
`program.kernels.rmsnorm_fused_sweep`, `program.frontend.gemm_targets`."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REGISTRY = "verity_vllm.program.registry"
REGISTRY_DIR = Path(__file__).resolve().parents[2] / "verity_vllm" / "program" / "registry"
FAMILY = ("dense", "act", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_activation", "boolean_silu",
          "boolean_rope", "gemm_bias")
#: other families' Definition modules the family reads: the primitives, the b1 composites, and `sampling.bit_at` (`UnpackF32_v1`)
READS = ("prims", "b1", "sampling")
THIRD_PARTY = ("numpy",)
MOVED = {"rope_difftest": "verity_vllm.properties.difftests", "rope_gptj_difftest": "verity_vllm.properties.difftests",
         "rmsnorm_fused_sweep": "verity_vllm.program.kernels", "gemm_targets": "verity_vllm.program.frontend"}


def _imports(path: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    """(line, module, names) of every import in the file, at any depth, relative ones resolved against the registry."""
    out = []
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
    if top in sys.stdlib_module_names or top in THIRD_PARTY or top == "verity":
        return []
    allowed = set(FAMILY) | set(READS)
    if mod == REGISTRY:
        return [f"{REGISTRY}.{x}" for x in names if x not in allowed]
    if mod.startswith(REGISTRY + ".") and mod.split(".")[3] in allowed:
        return []
    return [mod]


@pytest.mark.parametrize("name", FAMILY)
def test_a_family_definition_module_imports_only_verity_and_definition_modules(name):
    path = REGISTRY_DIR / f"{name}.py"
    bad = [f"line {line}: {m}" for line, mod, names in _imports(path) for m in _foreign(mod, names)]
    assert not bad, f"{name}.py imports outside verity and the Definition modules: {bad}"


def test_the_family_twins_and_vocabulary_left_the_registry():
    for name, pkg in MOVED.items():
        assert not (REGISTRY_DIR / f"{name}.py").exists(), f"{name} is back in the registry"
        assert importlib.import_module(f"{pkg}.{name}").__name__ == f"{pkg}.{name}"


def test_the_boundary_names_a_foreign_import():
    assert _foreign("verity_vllm.config", ("ROOT",)) == ["verity_vllm.config"]
    assert _foreign(REGISTRY, ("b1", "moe_difftest")) == [f"{REGISTRY}.moe_difftest"]
    assert _foreign(f"{REGISTRY}.gemm_targets", ("GemmTarget",)) == [f"{REGISTRY}.gemm_targets"]
    assert _foreign("torch", ()) == ["torch"]
    assert _foreign(f"{REGISTRY}.prims", ("BF16",)) == [] and _foreign("verity.ir.defs", ("bind",)) == []
