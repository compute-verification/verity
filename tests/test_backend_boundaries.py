"""Backends never need an integration (README "Development": backends must be testable without installing a frontend or
runtime integration).  Today one does: C-Flock takes the vLLM integration's registered Definitions and kernels where core has no
equivalent yet (the numerical bench, which did too, is a benchmark now, and a benchmark may compose an integration).  ``KNOWN`` lists every such ``(file, module)`` site and may only shrink: a new site
fails, and so does a listed site that is gone, so the list stays exact.

An import is an import statement, a string handed to ``importlib.import_module``, ``__import__`` or ``pytest.importorskip``, or
a ``"module:attr"`` string naming what to load at run time.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The backends' tree: C-Flock, the backend in development.
BACKENDS = ROOT / "verity" / "ml" / "flock"
#: The integrations' import packages, from each distribution's wheel configuration.
INTEGRATIONS = frozenset(Path(p).name for pyproject in sorted((ROOT / "integrations").glob("*/pyproject.toml"))
                         for p in tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
SKIP_DIRS = {"target", "node_modules", "__pycache__"}
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}

#: Every backend file that imports an integration, with each integration module it imports.  Repoint a backend at core (or
#: drop the use) and delete its line; a new line needs a reason the dependency cannot go through core.
KNOWN = {
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity_vllm.pipeline.program_graph"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity_vllm.query"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity_vllm.query.word"),
    ("verity/ml/flock/python/verity_flock/class_statement.py", "verity_vllm.pipeline.program_graph"),
    ("verity/ml/flock/python/verity_flock/ir_lower.py", "verity_vllm.program.kernels"),
    ("verity/ml/flock/python/verity_flock/ir_lower.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/ir_sampling.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/partition_units.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/tail.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/tail_pieces.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/python/verity_flock/topp_word.py", "verity_vllm.program.registry.topp_keep_builder"),
    ("verity/ml/flock/python/verity_flock/topp_word.py", "verity_vllm.program.registry.topp_word_gates"),
    ("verity/ml/flock/tests/test_boolean_export.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_boolean_export.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_circuit.py", "verity_vllm.program.kernels"),
    ("verity/ml/flock/tests/test_circuit.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_class_statement.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_derive.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_flock_rows.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_ir_lowering.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_ir_lowering.py", "verity_vllm.program.registry.sampling"),
    ("verity/ml/flock/tests/test_ir_sampling.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_ir_sampling.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_lean_rope.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_lean_typed_statement.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_partition_units.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_stage_typed.py", "verity_vllm.program.registry.prims"),
    ("verity/ml/flock/tests/test_topp_word.py", "verity_vllm.program.registry"),
    ("verity/ml/flock/tests/test_typed_statement.py", "verity_vllm.program.registry.prims"),
}


def _modules(path: Path, text: str) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS):
            out.add(node.args[0].value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value):
            out.add(node.value.partition(":")[0])
    return {m for m in out if m.split(".")[0] in INTEGRATIONS}


@functools.cache
def _sites() -> frozenset[tuple[str, str]]:
    sites = set()
    for path in sorted(BACKENDS.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in INTEGRATIONS):
            sites.update((rel.as_posix(), m) for m in _modules(path, text))
    return frozenset(sites)


def test_the_integrations_are_found():
    assert "verity_vllm" in INTEGRATIONS


def test_the_scan_sees_the_backends():
    """The scan walks C-Flock's tree: a backend that moved out of it would leave its imports unchecked, and `KNOWN` would pass
    as sites of files that don't exist yet."""
    for name in ("verity_flock",):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(BACKENDS), f"{name} is at {origin}, outside the scanned {BACKENDS.relative_to(ROOT)}/"
    assert not KNOWN or any((ROOT / f).exists() for f, _ in KNOWN), "no file KNOWN lists exists: it names pre-move paths"


def test_no_backend_imports_an_integration_outside_known():
    new = sorted(_sites() - KNOWN)
    assert not new, ("a backend imports an integration (README: backends are testable without one); import from core instead, "
                     "or, if the dependency cannot go through core yet, add to KNOWN in tests/test_backend_boundaries.py:\n"
                     + "\n".join(f'    ("{f}", "{m}"),' for f, m in new))


def test_every_known_site_still_exists():
    # a listed file that does not exist yet is an open PR's site, listed ahead so both merge orders pass; once the file
    # exists, its entry must match what it imports
    gone = sorted((f, m) for f, m in KNOWN - _sites() if (ROOT / f).exists())
    assert not gone, ("these backend -> integration imports are gone; delete them from KNOWN in tests/test_backend_boundaries.py:\n"
                      + "\n".join(f'    ("{f}", "{m}"),' for f, m in gone))
