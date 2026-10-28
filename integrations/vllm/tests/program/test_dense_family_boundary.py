"""The dense family's Definition modules (dense, activation, norms, RoPE, GEMM bias) import only `verity`, other Definition
modules and the evidence their conformance texts cite (`EVIDENCE`), so they can move to `verity/ml/circuits/` as they are. Their twins and vocabulary live in the integration package:
`properties.difftests.rope_difftest`, `properties.difftests.rope_gptj_difftest`,
`program.kernels.rmsnorm_fused_sweep`, `program.frontend.gemm_targets`."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REGISTRY = "verity.ml.definitions.registry"
REGISTRY_DIR = Path(importlib.util.find_spec(REGISTRY).origin).parent
FAMILY = ("dense", "act", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_activation", "boolean_silu",
          "boolean_rope", "boolean_tanh", "cuda_rmsnorm", "gemm_bias", "layer_norm", "gelu_erf_table", "traced_parts")
#: other families' Definition modules the family reads: the primitives, the b1 composites, `sampling.bit_at` (`UnpackF32_v1`),
#: `value_inputs`' word views of the norms and scalar rows with their f32 word an input, and `tables`' `tanh_rn` region
READS = ("prims", "b1", "sampling", "value_inputs", "tables")
THIRD_PARTY = ("numpy",)
#: the catalog, where the layout move put core's Definitions and silicon models
CATALOG = "verity.ml.circuits"
#: `evidence.json`'s ids, read with the standard library only
EVIDENCE = "verity.ml.definitions.evidence"
MOVED = ("verity_vllm.properties.difftests.rope_difftest", "verity_vllm.properties.difftests.rope_gptj_difftest",
         "verity_vllm.program.rmsnorm_fused_sweep", "verity_vllm.program.frontend.gemm_targets")


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


def _core(name: str) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, C-Flock, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.ml.commitments.multiproof", "verity.ml.private_circuits", "verity.ml.circuits.ops",
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock",
                "verity.ml.definitions")
    return under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.core.claims"))))


def _foreign(mod: str, names: tuple[str, ...]) -> list[str]:
    top = mod.partition(".")[0]
    allowed = set(FAMILY) | set(READS)
    if mod == REGISTRY:
        return [f"{REGISTRY}.{x}" for x in names if x not in allowed]
    if mod.startswith(REGISTRY + "."):
        return [] if mod[len(REGISTRY) + 1:].split(".")[0] in allowed else [mod]
    if top in sys.stdlib_module_names or top in THIRD_PARTY or _core(mod) or mod in (CATALOG, EVIDENCE) or mod.startswith(CATALOG + "."):
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
    assert _foreign(f"{REGISTRY}.prims", ("BF16",)) == [] and _foreign("verity.ml.circuits.defs", ("bind",)) == []
