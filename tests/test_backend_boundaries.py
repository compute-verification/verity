"""Nothing under ``verity/`` needs an integration (DESIGN.md "Development": backends must be testable without installing a
frontend or runtime integration), the trees `tests/test_boundaries.py` leaves out (``SEPARATE``: C-Flock, the GPU hash, the
difftest harness, PoUS's check) included.  C-Flock took the vLLM integration's registered Definitions and kernels until they moved
to ``verity.ml.definitions``; what still reaches an integration is in ``KNOWN``, which lists every ``(file, module)`` site and may
only shrink: a new site fails, and so does a listed site that is gone, so the list stays exact.

An import is an import statement, a string handed to ``importlib.import_module``, ``__import__``, ``pytest.importorskip`` or
``importlib.resources.files``, or a ``"module:attr"`` string naming what to load at run time.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The tree that needs no integration: ``verity``, C-Flock and their tests.
SCANNED = ROOT / "verity"
#: The integrations' import packages, from each distribution's wheel configuration (a virtual member, such as the Slack
#: events service, has none to import).
INTEGRATIONS = frozenset(Path(p).name for pyproject in sorted((ROOT / "integrations").glob("*/pyproject.toml"))
                         for p in tomllib.loads(pyproject.read_text()).get("tool", {}).get("hatch", {}).get("build", {})
                         .get("targets", {}).get("wheel", {}).get("packages", []))
SKIP_DIRS = {"target", "node_modules", "__pycache__"}
MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip", "files"}

#: Every file under ``verity/`` that imports an integration, with each integration module it imports.  Repoint it at
#: ``verity`` (or drop the use) and delete its line; a new line needs a reason the dependency cannot go through ``verity``.
KNOWN = {
    # the memory-map test holds the Commit's builder to the request Programs the integration's predictor builds
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.commit"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.predict.build"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.predict.phi"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.query.required"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.store.interface"),
    ("verity/core/primitives/requirement/tests/test_census_and_memory_map.py", "verity_vllm.store.registry"),
    # the cuBLAS route check reads the device pins the integration measures (`verity_vllm/pins/`) until it hands them in
    ("verity/ml/definitions/registry/bf16_gemm_check.py", "verity_vllm"),
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


def test_the_scan_sees_the_backends():
    """The scan walks C-Flock's tree and the Definitions': one that moved out of ``verity/`` would leave its imports unchecked,
    and `KNOWN` would pass as sites of files that don't exist yet."""
    for name in ("verity_flock", "verity.ml.definitions"):
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(SCANNED), f"{name} is at {origin}, outside the scanned {SCANNED.relative_to(ROOT)}/"
    assert not KNOWN or any((ROOT / f).exists() for f, _ in KNOWN), "no file KNOWN lists exists: it names pre-move paths"


def test_no_backend_imports_an_integration_outside_known():
    new = sorted(_sites() - KNOWN)
    assert not new, ("a file under verity/ imports an integration (DESIGN.md: backends are testable without one); import from verity "
                     "instead, or, if the dependency cannot go through verity yet, add to KNOWN in tests/test_backend_boundaries.py:\n"
                     + "\n".join(f'    ("{f}", "{m}"),' for f, m in new))


def test_every_known_site_still_exists():
    # a listed file that does not exist yet is an open PR's site, listed ahead so both merge orders pass; once the file
    # exists, its entry must match what it imports.  An entry a layout move took out of the integrations (its module now
    # `verity`'s) is no site at all: the move leaves it for a later change to delete
    gone = sorted((f, m) for f, m in KNOWN - _sites() if (ROOT / f).exists() and m.split(".")[0] in INTEGRATIONS)
    assert not gone, ("these backend -> integration imports are gone; delete them from KNOWN in tests/test_backend_boundaries.py:\n"
                      + "\n".join(f'    ("{f}", "{m}"),' for f, m in gone))
