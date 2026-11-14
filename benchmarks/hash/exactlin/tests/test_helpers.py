"""exactlin.py's three Triton helpers are `verity_vllm.engine.int8_fused`'s, definition for definition, and nothing else."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COPY = ROOT / "benchmarks/hash/exactlin/exactlin.py"
SERVED = ROOT / "adapters/vllm/verity_vllm/engine/int8_fused.py"
HELPERS = {"_rni_s32", "_membar_gl", "_epilogue"}


def _functions(path: Path) -> dict[str, str]:
    return {n.name: ast.unparse(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def test_the_helpers_are_the_served_kernels_own() -> None:
    copy, served = _functions(COPY), _functions(SERVED)
    assert set(copy) == HELPERS, "exactlin.py holds only the helpers int8_fused keeps private; take the rest from int8_fused"
    for name in sorted(HELPERS):
        assert copy[name] == served[name], f"{name} differs from int8_fused's: copy int8_fused's text into exactlin.py"
