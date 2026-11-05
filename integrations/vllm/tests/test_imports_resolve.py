"""Every absolute import of a first-party package names a module of this tree, function-local imports included.

An import and collection check executes module bodies only, so a function-local import of a module that moved (`verity_capture`,
`verity_vllm_adapter` and `verity_vllm_numerics` are now `verity_vllm`'s) passes it and fails only when its function runs: a Commit
dies on `ModuleNotFoundError` before any engine work.  This lint reads the source with `ast` (torch-free) and resolves each such
module to a file of the tree.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY = ("verity_vllm", "verity_capture", "verity_vllm_adapter", "verity_vllm_numerics")


def _resolves(mod: str) -> bool:
    base = ROOT.joinpath(*mod.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


@pytest.mark.slow
def test_first_party_imports_resolve_to_modules_of_the_tree():
    unresolved = []
    for top in ("verity_vllm", "tests"):
        for f in sorted((ROOT / top).rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                else:
                    continue
                unresolved += [f"{f.relative_to(ROOT)}:{node.lineno}: {m}" for m in mods
                               if m.split(".")[0] in FIRST_PARTY and not _resolves(m)]
    assert unresolved == [], "first-party imports that name no module of the tree:\n" + "\n".join(unresolved)
