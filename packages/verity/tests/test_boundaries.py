"""The core dependency rules (README "Development") as import-boundary checks over the source tree.

    ir <- evaluation <- ml,  ir <- commitments,  ir + ml + commitments + census <- proofs;  randomness, census and claims
    import no core module;  nothing depends back on a later layer, and no core module imports a backend, frontend, integration, or
    external runtime.

numpy is a core dependency (architecture decision: `verity.evaluation` batches over numpy arrays and kernels sit beside
their Definitions in `verity.ml`).  The plain-integer references stay numpy-free on purpose, because they are the
specification the Rust guest, the Lean vectors and the probes are checked against: `verity.ir` and `verity.ml.tc`.
`verity.randomness` is exact stdlib arithmetic too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "verity"

#: For each core module, the sibling core modules it may import.  ``errors`` is shared by all.
ALLOWED = {
    "ir": set(),
    "evaluation": {"ir"},
    "ml": {"ir", "evaluation"},
    "commitments": {"ir"},
    "proofs": {"ir", "ml", "commitments", "census"},
    "resource_accounting": {"ir", "proofs", "commitments"},
    "randomness": set(),
    "census": set(),
    "claims": set(),
}
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_torch", "sp1", "vllm", "torch", "jax", "onnx",
                      "backends", "frontends", "integrations", "research", "veritor")
#: Source subtrees that must stay stdlib-only: the scalar references.
NUMPY_FREE = ("ir", "ml/tc", "randomness", "census", "claims")


def _imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
    return out


def _core_modules() -> list[str]:
    return sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists())


@pytest.mark.parametrize("module", _core_modules())
def test_core_module_imports_only_inward(module: str) -> None:
    assert module in ALLOWED, f"new core module {module!r}: add it to ALLOWED with its permitted dependencies"
    offenders = []
    for path in (SRC / module).rglob("*.py"):
        for name, line in _imports(path):
            top = name.split(".")[0]
            if top in FORBIDDEN_EXTERNAL:
                offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
            if top == "verity":
                parts = name.split(".")
                target = parts[1] if len(parts) > 1 else None
                if target in (None, module, "errors"):
                    continue
                if target not in ALLOWED[module]:
                    offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("subtree", NUMPY_FREE)
def test_scalar_references_are_stdlib_only(subtree: str) -> None:
    offenders = [f"{path.relative_to(SRC)}:{line}: {name}"
                 for path in (SRC / subtree).rglob("*.py")
                 for name, line in _imports(path)
                 if name.split(".")[0] == "numpy" or (subtree == "ml/tc" and name.split(".")[:2] != ["verity", "errors"]
                                                      and name.startswith("verity."))]
    assert not offenders, "\n".join(offenders)
