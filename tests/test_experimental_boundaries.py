"""Nothing outside `experimental/` depends on it (note:20261004T2058Z-draft-repo-organization-principles: `experimental/` holds
what no cited claim depends on).  Library code never imports an experimental package.  Some core tests still do: they check a
quarantine lane's re-exports, difftest adapters or collective from the integration's suite.  ``KNOWN`` lists every such
``(file, module)`` site and may only shrink: a new site fails, and so does a listed site that is gone, so the list stays exact.

An import is an import statement, or a string handed to ``importlib.import_module``, ``__import__`` or ``pytest.importorskip``.
"""

from __future__ import annotations

import ast
import functools
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The experimental distributions' import packages, from each one's wheel configuration.
EXPERIMENTAL = frozenset(Path(p).name for pyproject in sorted((ROOT / "experimental").rglob("pyproject.toml"))
                         for p in tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}

#: Every core test file that imports an experimental package, with each module it imports.  Move the test into the experimental
#: package's own suite (or drop the use) and delete its line.  A library file never belongs here.
KNOWN = {
    ("integrations/vllm/tests/collectives/collective_link.py", "verity_quarantine.collective.allreduce"),
    ("integrations/vllm/tests/collectives/test_tp_collective.py", "verity_quarantine"),
    ("integrations/vllm/tests/collectives/test_tp_collective.py", "verity_quarantine.collective"),
    ("integrations/vllm/tests/engine/test_gen_dense_gemma2_patterns.py", "verity_quarantine.dense"),
    ("integrations/vllm/tests/engine/test_gen_dense_softcap.py", "verity_quarantine.dense"),
    ("integrations/vllm/tests/engine/test_gen_ov_moe.py", "verity_quarantine.ov_moe"),
    ("integrations/vllm/tests/program/test_norm_chain.py", "verity_quarantine.dense"),
    ("integrations/vllm/tests/program/test_scalar_tensor_mul.py", "verity_quarantine.dense"),
    ("integrations/vllm/tests/program/test_silu_mul_v2.py", "verity_quarantine.act"),
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
    return {m for m in out if m.split(".")[0] in EXPERIMENTAL}


@functools.cache
def _sites() -> frozenset[tuple[str, str]]:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                           check=True, capture_output=True, text=True).stdout.split()
    sites = set()
    for rel in files:
        path = ROOT / rel
        if rel.startswith("experimental/") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in EXPERIMENTAL):
            sites.update((rel, m) for m in _modules(path, text))
    return frozenset(sites)


def test_the_experimental_packages_are_found():
    assert "verity_quarantine" in EXPERIMENTAL


def test_nothing_outside_experimental_imports_it_except_known_tests():
    new = sorted(_sites() - KNOWN)
    assert not new, ("a file outside experimental/ imports an experimental package; put the use in that package (its tests go in "
                     "its own suite), or, for a core test that cannot move yet, add to KNOWN in tests/test_experimental_boundaries.py:\n"
                     + "\n".join(f'    ("{f}", "{m}"),' for f, m in new))


def test_every_known_site_still_exists_and_is_a_test():
    gone = sorted(KNOWN - _sites())
    assert not gone, ("these imports of an experimental package are gone; delete them from KNOWN in "
                      "tests/test_experimental_boundaries.py:\n" + "\n".join(f'    ("{f}", "{m}"),' for f, m in gone))
    assert all("/tests/" in f for f, _ in KNOWN), "library code never imports an experimental package"
