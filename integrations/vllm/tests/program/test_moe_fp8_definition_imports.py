"""The MoE and FP8 family's Definition modules import only `verity`, the standard library and Definition modules: their own family's
(`FAMILY`) and the other families' they call (`OTHER_DEFINITION_MODULES`), with the evidence their conformance texts cite
(`EVIDENCE`), never numpy, a twin, the frontend or a runtime.  That is what lets them move to the catalog, which imports only
`verity`, once the other families' modules are Definitions only too.

An import is any import statement in the module, a function-level one included, with a relative one resolved against the registry
package; `from verity.ml.definitions.registry import m` imports the module `m`.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REGISTRY = "verity.ml.definitions.registry"
DIR = Path(importlib.util.find_spec(REGISTRY).origin).parent
FAMILY = ("moe", "moe_pad", "boolean_moe", "fp8", "fp8_moe", "fp8_moe_experts", "boolean_fp8", "boolean_fp8_moe", "nvfp4",
          "boolean_gather", "dsv4_moe", "boolean_dsv4_moe")
#: other families' modules the family's Definitions call: the scalar and bit primitives (`prims`, `ref_prims`, `pad_prims`), the B1
#: composites (`b1`), Hopper's tensor-core step (`hopper`), the lifted-Definition makers (`lifted`), the Boolean bit views
#: (`boolean_attention._Bits`), the SwiGLU and logf that DeepSeek-V4's routers and activation call (`act`, `boolean_silu`,
#: `topp_word_gates`, `gumbel_boolean`), and the i32 gather `GatherI32x{V}_v1` that `boolean_gather.gather_i32` lowers
#: (`dsv4_decode_indices`, which also registers its row's numpy twin, so it is no Definitions-only module yet)
OTHER_DEFINITION_MODULES = ("b1", "prims", "ref_prims", "pad_prims", "hopper", "lifted", "boolean_attention", "act", "boolean_silu",
                            "topp_word_gates", "gumbel_boolean", "dsv4_decode_indices")
#: `evidence.json`'s ids, read with the standard library only
EVIDENCE = "verity.ml.definitions.evidence"
ALLOWED = {f"{REGISTRY}.{m}" for m in FAMILY + OTHER_DEFINITION_MODULES} | {EVIDENCE}
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
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock",
                "verity.ml.definitions")
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
    (tmp_path / "x.py").write_text("import numpy as np\n\n\ndef f():\n    from verity.ml.definitions.numerics import fp8_twins\n"
                                   "    from . import fp8, targets\n")
    monkeypatch.setattr(sys.modules[__name__], "DIR", tmp_path)
    assert [(m, _allowed(m)) for m, _ in _imports("x")] == [
        ("numpy", False), ("verity.ml.definitions.numerics", False), (f"{REGISTRY}.fp8", True), (f"{REGISTRY}.targets", False)]
