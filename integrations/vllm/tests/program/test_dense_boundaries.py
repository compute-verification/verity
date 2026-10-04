"""The dense, activation, norm and RoPE family's split (`tools/migrate/split_dense.py`) as import-boundary checks over the source.

Each of the family's Definition modules imports only `verity`, the standard library, numpy and the registry's Definition
modules (its own family's and `b1`, `prims`, `sampling`), so it can move to the catalog, which imports only `verity`. Its twins
(the RoPE difftest adapters and the fused-RMSNorm sweep) live in `verity_vllm.program.kernels`, not in the registry, and
register no Definition.

An import is an import statement, module level or lazy (a relative one resolved against its package; ``from p import n`` is
``p.n`` when that is a module, else ``p``), or a string handed to ``importlib.import_module`` or ``__import__``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "verity_vllm.program.registry"
DEFINITION_MODULES = ("act", "boolean_activation", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_rope",
                      "boolean_silu", "dense", "gemm_bias")
#: other families' Definition modules the family's Definitions call
OTHER_DEFINITION_MODULES = ("b1", "prims", "sampling")
TWINS = ("rope_difftest", "rope_gptj_difftest", "rmsnorm_fused_sweep")
TWIN_PACKAGE = "verity_vllm.program.kernels"
ALLOWED_TOP = {"verity", "numpy"} | set(sys.stdlib_module_names)
IMPORT_CALLS = {"import_module", "__import__"}


def _path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def _is_module(module: str) -> bool:
    return _path(module).is_file() or (ROOT / module.replace(".", "/") / "__init__.py").is_file()


def _imports(module: str) -> list[tuple[str, int]]:
    package = module.split(".")[:-1]
    out = []
    for node in ast.walk(ast.parse(_path(module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[:len(package) - node.level + 1] + ([node.module] if node.module else [])) if node.level \
                else node.module
            out += [(f"{base}.{a.name}" if _is_module(f"{base}.{a.name}") else base, node.lineno) for a in node.names]
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS):
            out.append((node.args[0].value, node.lineno))
    return out


@pytest.mark.parametrize("name", DEFINITION_MODULES)
def test_definition_module_imports_only_verity_and_definition_modules(name):
    allowed = {f"{REGISTRY}.{m}" for m in DEFINITION_MODULES + OTHER_DEFINITION_MODULES}
    bad = [f"line {line}: {mod}" for mod, line in _imports(f"{REGISTRY}.{name}")
           if mod.split(".")[0] not in ALLOWED_TOP and mod not in allowed]
    assert not bad, f"{name} imports a module that is not verity, the standard library, numpy or a Definition module: {bad}"


@pytest.mark.parametrize("name", TWINS)
def test_twin_is_out_of_the_registry_and_registers_nothing(name):
    assert not _is_module(f"{REGISTRY}.{name}"), f"{name} is back in the registry"
    assert _is_module(f"{TWIN_PACKAGE}.{name}")
    registers = []
    for node in ast.walk(ast.parse(_path(f"{TWIN_PACKAGE}.{name}").read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")) in ("composite", "primitive"):
                    registers.append(f"line {node.lineno}: {node.name}")
        if isinstance(node, ast.Call):
            f = node.func
            if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")) in ("CompositeDefinition", "PrimitiveDefinition"):
                if not any(k.arg == "register" and isinstance(k.value, ast.Constant) and k.value.value is False for k in node.keywords):
                    registers.append(f"line {node.lineno}: a registered {f.attr if isinstance(f, ast.Attribute) else f.id}")
    assert not registers, f"{name} registers Definitions, so it belongs with the Definitions: {registers}"


def test_the_checks_see_a_violation():
    """The boundary check is not vacuous: it reads the lazy imports and resolves the relative ones of a real module."""
    got = dict(_imports(f"{REGISTRY}.boolean_rope"))
    assert f"{REGISTRY}.b1" in got and f"{REGISTRY}.prims" in got
    assert any(m.startswith("verity_vllm.") and m not in {f"{REGISTRY}.{x}" for x in DEFINITION_MODULES + OTHER_DEFINITION_MODULES}
               for m, _ in _imports(f"{TWIN_PACKAGE}.rmsnorm_fused_sweep"))
