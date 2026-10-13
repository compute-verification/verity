"""The MoE and FP8 family's Definition modules import only `verity`, the standard library and Definition modules: their own family's
(`FAMILY`) and the other families' they call (`OTHER_DEFINITION_MODULES`), never numpy, a twin, the frontend or a runtime.  That is
what lets them move to the catalog, which imports only `verity`, once the other families' modules are Definitions only too.

An import is any import statement in the module, a function-level one included, with a relative one resolved against the registry
package; `from verity_vllm.program.registry import m` imports the module `m`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REGISTRY = "verity_vllm.program.registry"
DIR = Path(__file__).resolve().parents[2] / "verity_vllm" / "program" / "registry"
FAMILY = ("moe", "moe_pad", "boolean_moe", "fp8", "fp8_moe", "fp8_moe_experts", "boolean_fp8", "boolean_fp8_moe", "nvfp4",
          "boolean_gather", "dsv4_moe")
#: other families' modules the family's Definitions call: the scalar and bit primitives (`prims`, `ref_prims`, `pad_prims`), the B1
#: composites (`b1`), Hopper's tensor-core step (`hopper`), the lifted-Definition makers (`lifted`), the Boolean bit views
#: (`boolean_attention._Bits`), and the SwiGLU, logf and ex2 that DeepSeek-V4's routers and activation call (`act`,
#: `boolean_silu`, `topp_word_gates`, `gumbel_boolean`, `mla`)
OTHER_DEFINITION_MODULES = ("b1", "prims", "ref_prims", "pad_prims", "hopper", "lifted", "boolean_attention", "act", "boolean_silu",
                            "topp_word_gates", "gumbel_boolean", "mla")
ALLOWED = {f"{REGISTRY}.{m}" for m in FAMILY + OTHER_DEFINITION_MODULES}
#: the catalog, where the layout move put core's Definitions and silicon models
CATALOG = "verity.ml.circuits"


def _imports(name: str) -> list[tuple[str, int]]:
    out = []
    for node in ast.walk(ast.parse((DIR / f"{name}.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            module = node.module or ""
            if node.level:
                base = REGISTRY.split(".")[:len(REGISTRY.split(".")) - node.level + 1]
                module = ".".join(base + ([module] if module else []))
            if module == REGISTRY:
                out += [(f"{REGISTRY}.{a.name}", node.lineno) for a in node.names]
            else:
                out.append((module, node.lineno))
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


def _allowed(module: str) -> bool:
    top = module.split(".")[0]
    return ((top in sys.stdlib_module_names and top != "__main__") or _core(module) or module in ALLOWED
            or module == CATALOG or module.startswith(CATALOG + "."))


@pytest.mark.parametrize("name", FAMILY)
def test_a_definition_module_imports_only_verity_and_definition_modules(name):
    bad = [f"{name}.py:{line}: {m}" for m, line in _imports(name) if not _allowed(m)]
    assert not bad, f"imports outside `verity`, the standard library and the Definition modules: {bad}"


def test_the_rule_sees_what_it_forbids(tmp_path, monkeypatch):
    (tmp_path / "x.py").write_text("import numpy as np\n\n\ndef f():\n    from ..kernels import fp8_twins\n"
                                   "    from . import fp8, targets\n")
    monkeypatch.setattr(sys.modules[__name__], "DIR", tmp_path)
    assert [(m, _allowed(m)) for m, _ in _imports("x")] == [
        ("numpy", False), ("verity_vllm.program.kernels", False), (f"{REGISTRY}.fp8", True), (f"{REGISTRY}.targets", False)]
