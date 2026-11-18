"""Nothing that serves or commits loads `verity_vllm.flock`: no module of the adapter outside it imports it, by statement or
by a module name in a string (`import_module`, `load_catalog`)."""
from __future__ import annotations

import ast
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "verity_vllm"
FLOCK = "verity_vllm.flock"


def _loads_flock(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names] if node.level == 0 else []
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        else:
            continue
        if any(n == FLOCK or n.startswith(FLOCK + ".") for n in names):
            lines.append(node.lineno)
    return lines


def test_no_adapter_module_outside_flock_loads_it():
    bad = [f"{p.relative_to(ADAPTER.parent)}:{line}" for p in sorted(ADAPTER.rglob("*.py"))
           if (ADAPTER / "flock") not in p.parents
           for line in _loads_flock(ast.parse(p.read_text(), str(p)))]
    assert not bad, f"these load {FLOCK}, which Flock's prover alone loads: {bad}"


def test_the_scan_finds_a_load():
    assert _loads_flock(ast.parse("from verity_vllm.flock import topp_word\nimport_module('verity_vllm.flock')\n")) == [1, 2]
    assert _loads_flock(ast.parse("from verity_vllm import flock\n")) == [1]
    assert _loads_flock(ast.parse("import verity_vllm.definitions\n")) == []
