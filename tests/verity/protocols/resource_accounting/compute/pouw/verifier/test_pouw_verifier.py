"""PoUW's verifier (`verifier`): a module that imports only the standard library, so what decides an execution phase, or
flips its coins, never reaches `verity.ml`."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from verity.protocols.resource_accounting.compute.pouw.verifier import verifier as V


def test_the_verifier_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported and imported - {"__future__"} <= set(sys.stdlib_module_names), imported


def test_without_lake_the_refusal_names_the_path_it_searched(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(V, "BUILD_LOCK", tmp_path / "build.lock")
    try:
        V.executable.__wrapped__("pouw-verifier-difftest")
    except V.LeanError as e:
        assert f"lake is not on the PATH ({tmp_path})" in str(e) and "~/.elan/bin" in str(e)
    else:
        raise AssertionError("built without lake")
