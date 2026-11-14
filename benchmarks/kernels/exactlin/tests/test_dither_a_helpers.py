"""dither_a.py's kernels inline `verity_vllm.engine.int8_fused`'s Triton helpers by name (`EL.rni_s32`, `EL.membar_gl`,
`EL.epilogue`). Triton resolves those names only when a kernel compiles on a GPU, so a helper int8_fused renamed or dropped
would pass every CPU suite; this holds each name dither_a reads off int8_fused to a public one int8_fused binds, and each
it reads inside a kernel to one of its `@triton.jit` functions."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DITHER_A = ROOT / "benchmarks/hash/dither/dither_a.py"
SERVED = ROOT / "adapters/vllm/verity_vllm/engine/int8_fused.py"


def _jit(fn: ast.FunctionDef) -> bool:
    return any(ast.unparse(d).split("(")[0] == "triton.jit" for d in fn.decorator_list)


def _alias(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "verity_vllm.engine":
            for a in node.names:
                if a.name == "int8_fused":
                    return a.asname or a.name
    raise AssertionError("dither_a.py does not import int8_fused from verity_vllm.engine")


def _reads(tree: ast.AST, alias: str) -> set[str]:
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias}


def test_dither_a_reads_only_names_int8_fused_binds() -> None:
    dither_a, served = ast.parse(DITHER_A.read_text()), ast.parse(SERVED.read_text())
    alias = _alias(dither_a)
    defined = {n.name for n in served.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in served.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    defined |= {n.target.id for n in served.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    defined |= {a.asname or a.name for n in served.body if isinstance(n, ast.ImportFrom) for a in n.names}
    jit = {n.name for n in served.body if isinstance(n, ast.FunctionDef) and _jit(n)}
    reads = _reads(dither_a, alias)
    in_kernels = set().union(*(_reads(f, alias) for f in dither_a.body if isinstance(f, ast.FunctionDef) and _jit(f)))
    assert {"rni_s32", "membar_gl", "epilogue"} <= in_kernels, "dither_a's kernels no longer inline int8_fused's three helpers"
    assert not {r for r in reads if r.startswith("_")}, "dither_a reads a private name of int8_fused: ask vllm for a public one"
    assert reads <= defined, f"int8_fused binds no {sorted(reads - defined)}: take its current name into dither_a.py"
    assert in_kernels <= jit, f"{sorted(in_kernels - jit)} are no @triton.jit function of int8_fused: dither_a's kernels inline them"
