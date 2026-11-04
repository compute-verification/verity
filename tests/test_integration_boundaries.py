"""Nothing under ``verity/`` needs an integration (AGENTS.md "Design rules", "Integrations are frontends";
ARCHITECTURE.md "Integrations"): any serving stack plugs in without changing ``verity/``.
Today some files do: Flock takes the vLLM integration's registered Definitions and kernels where ``verity`` has no
equivalent yet, and core's difftests of the primitives and sigma hold the Lean to them (a benchmark may compose an
integration).  ``KNOWN`` lists every such ``(file, module)`` site and may only shrink: a new site fails, and so does a
listed site that is gone, so the list stays exact.

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
#: The tree that needs no integration: ``verity``, Flock and their tests.
SCANNED = ROOT / "verity"
#: The integrations' import packages, from each distribution's wheel configuration (a virtual member, such as the Slack
#: events service, has none to import).
INTEGRATIONS = frozenset(Path(p).name for pyproject in sorted((ROOT / "integrations").glob("*/pyproject.toml"))
                         for p in tomllib.loads(pyproject.read_text()).get("tool", {}).get("hatch", {}).get("build", {})
                         .get("targets", {}).get("wheel", {}).get("packages", []))
SKIP_DIRS = {"target", "node_modules", "__pycache__"}
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}

#: Every file under ``verity/`` that imports an integration, with each integration module it imports.  Repoint it at
#: ``verity`` (or drop the use) and delete its line; a new line needs a reason the dependency cannot go through ``verity``.
KNOWN = {
    ("verity/core/primitives/evaluate/tests/test_difftest.py", "verity.ml.definitions.registry"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.commit"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.program.predict.build"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.program.predict.phi"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.query.required"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.protocols.remote_bandwidth.interface"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.protocols.remote_bandwidth.registry"),
    ("verity/core/primitives/requirement/tests/test_requirement_difftest.py", "verity.ml.definitions.registry"),
    ("verity/core/service/sigma/tests/test_sigma_difftest.py", "verity.ml.definitions.boolean"),
    ("verity/core/service/sigma/tests/test_sigma_difftest.py", "verity.ml.definitions.registry"),
    ("verity/core/service/sigma/tests/test_sigma_difftest.py", "verity.ml.definitions.registry.catalog"),
    ("verity/core/service/sigma/tests/test_sigma_difftest.py", "verity.ml.definitions.replay"),
    ("verity/ml/definitions/tests/test_catalog.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity.ml.definitions.encoding"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/boolean_export.py", "verity.ml.definitions.word"),
    ("verity/ml/flock/python/verity_flock/class_statement.py", "verity.ml.definitions.encoding"),
    ("verity/ml/flock/python/verity_flock/ir_lower.py", "verity.ml.definitions.numerics"),
    ("verity/ml/flock/python/verity_flock/ir_lower.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/ir_sampling.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/partition_units.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/tail.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/python/verity_flock/tail_pieces.py", "verity.ml.definitions.registry"),
    # SiluMul_v2's unit is the registry's Boolean SiluMulBf16_v4, which only this import registers; core has no SiluMulBf16_v2
    ("verity/ml/flock/python/verity_flock/templates/silu_mul_v2.py", "verity.ml.definitions.registry.boolean_silu"),
    ("verity/ml/flock/python/verity_flock/topp_word.py", "verity.ml.definitions.registry.topp_keep_builder"),
    ("verity/ml/flock/python/verity_flock/topp_word.py", "verity.ml.definitions.registry.topp_word_gates"),
    ("verity/ml/flock/tests/test_boolean_export.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_boolean_export.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_circuit.py", "verity.ml.definitions.numerics"),
    ("verity/ml/flock/tests/test_circuit.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_class_statement.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_derive.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_flock_rows.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_ir_lowering.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_ir_lowering.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_ir_lowering.py", "verity.ml.definitions.registry.sampling"),
    ("verity/ml/flock/tests/test_ir_sampling.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_ir_sampling.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_lean_typed_statement.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_partition_units.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_stage_typed.py", "verity.ml.definitions.registry.prims"),
    ("verity/ml/flock/tests/test_topp_word.py", "verity.ml.definitions.registry"),
    ("verity/ml/flock/tests/test_typed_statement.py", "verity.ml.definitions.registry.prims"),
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
    for path in sorted(SCANNED.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in INTEGRATIONS):
            sites.update((rel.as_posix(), m) for m in _modules(path, text))
    return frozenset(sites)


def test_the_integrations_are_found():
    assert "verity_vllm" in INTEGRATIONS


def test_the_scan_sees_verity():
    """The scan walks ``verity/``: a package that moved out of it would leave its imports unchecked, and `KNOWN` would pass
    as sites of files that don't exist yet."""
    for name in ("verity", "verity_flock"):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(SCANNED), f"{name} is at {origin}, outside the scanned {SCANNED.relative_to(ROOT)}/"
    assert not KNOWN or any((ROOT / f).exists() for f, _ in KNOWN), "no file KNOWN lists exists: it names pre-move paths"


def test_nothing_in_verity_imports_an_integration_outside_known():
    new = sorted(_sites() - KNOWN)
    assert not new, ("a file under verity/ imports an integration (ARCHITECTURE.md \"Integrations\": verity meets a serving "
                     "stack at one interface); import from verity instead, or, if the dependency cannot go through verity "
                     "yet, add to KNOWN in tests/test_integration_boundaries.py:\n"
                     + "\n".join(f'    ("{f}", "{m}"),' for f, m in new))


def test_every_known_site_still_exists():
    # a listed file that does not exist yet is an open PR's site, listed ahead so both merge orders pass; once the file
    # exists, its entry must match what it imports.  An entry a layout move took out of the integrations (its module now
    # `verity`'s) is no site at all: the move leaves it for a later change to delete
    gone = sorted((f, m) for f, m in KNOWN - _sites() if (ROOT / f).exists() and m.split(".")[0] in INTEGRATIONS)
    assert not gone, ("these verity -> integration imports are gone; delete them from KNOWN in tests/test_integration_boundaries.py:\n"
                      + "\n".join(f'    ("{f}", "{m}"),' for f, m in gone))
