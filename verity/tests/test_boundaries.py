"""The core dependency rules (README "Development") as import-boundary checks over ``verity/``.

    primitives.circuits <- primitives.commitments -> primitives.crypto;  primitives.circuits <- protocols.profile;
    each protocol imports the primitives, the profile and the protocols it names in ``ALLOWED``, and no third-party package;  claims, randomness and crypto import no
    core unit;  a primitive never imports a protocol, and no unit imports another distribution of the workspace (the
    catalog, the kernels, experimental, a backend, an integration, a benchmark, a tool) or an external runtime: core
    imports the standard library, its declared dependencies and itself.

An import is an import statement (a relative one resolved against its package), a string handed to
``importlib.import_module``, ``__import__`` or ``importorskip``, or a ``"module:attr"`` string naming what to load at run
time: a module named as data is still a dependency.  Tests (``tests/``) are suites of their own and declare what they read;
a ``lean/`` directory holds a Lean package and the scripts that write its vectors, run by path.

numpy is a core dependency (architecture decision: `verity.primitives.circuits.evaluation` batches over numpy arrays).  The
plain-integer references stay numpy-free on purpose, because they are the specification the Rust guest, the Lean vectors and
the probes are checked against: the rest of `verity.primitives.circuits`.  `verity.primitives.randomness` is exact stdlib
arithmetic too.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]

#: For each core unit (a package, or a module, under ``verity``), the other units it may import.  ``errors`` is shared by all.
ALLOWED = {
    "claims": set(),
    "primitives.circuits": set(),
    "primitives.randomness": set(),
    "primitives.crypto": set(),
    "primitives.commitments": {"primitives.circuits", "primitives.crypto"},
    "protocols.profile": {"primitives.circuits"},
    # The proof service's challenger draws through the one challenge derivation (`service.Stream.select`).
    "protocols.verification.sampled_proofs": {"primitives.circuits", "primitives.commitments", "primitives.randomness",
                                              "protocols.profile"},
    "protocols.accounting.space.pous": {"claims", "primitives.circuits", "primitives.commitments", "primitives.randomness"},
    # 2026-10-03 (Daniel, `.agents/skills/friction/SKILL.md`): PoUW's tile check is a sampled proof.
    "protocols.accounting.work.pouw": {"primitives.circuits", "primitives.commitments", "primitives.crypto",
                                       "primitives.randomness", "protocols.profile", "protocols.verification.sampled_proofs"},
    # N2 of the proof-service plan: a link-window's network certificates are registered through the proof service.
    "protocols.accounting.communication.certifier": {"primitives.circuits", "primitives.commitments",
                                                     "protocols.verification.sampled_proofs"},
}
#: Import names core never imports or names as a string: the other workspace distributions, external runtimes, and the
#: repository's own directories.
FORBIDDEN_EXTERNAL = ("verity_catalog", "verity_kernels", "verity_experimental", "verity_sp1", "verity_vllm", "verity_numerical",
                      "verity_flock", "verity_torch", "circuit_check", "research", "cluster", "sp1", "vllm", "torch", "jax",
                      "onnx", "catalog", "kernels", "experimental", "archive", "backends", "benchmarks", "frontends",
                      "integrations", "tools", "veritor")
#: The one edge out of core: PoUW's schemes and circuits are stated over the catalog's silicon models and Definitions (the
#: tensor-core steps, FP32 and the casts its verifier recomputes), which live in the catalog by ruling, while the scheme
#: stays in core.  Nothing else in core imports the catalog.
EXCEPTIONS = {"protocols.accounting.work.pouw": {"verity_catalog"}}
#: Core's third-party dependencies (the root's ``pyproject.toml``); anything else it imports is the stdlib or ``verity``.  A
#: protocol imports none: it is the standard library and core, and the application composes protocols.
DEPENDENCIES = {"numpy"}
#: Source subtrees that must stay stdlib-only, the scalar references, and within them what may use numpy.
NUMPY_FREE = ("primitives/circuits", "primitives/randomness", "claims")
NUMPY_USERS = ("primitives/circuits/evaluation", "primitives/circuits/boolean/forms.py")

MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


def _sources(root: Path = SRC) -> list[Path]:
    """Core's own Python files: every ``.py`` under ``root`` outside a ``tests/`` or ``lean/`` directory, or a Lake build
    directory (``verity/Security/.lake`` holds its dependencies' checkouts, Mathlib's scripts among them)."""
    return sorted(p for p in root.rglob("*.py")
                  if not {"tests", "lean", ".lake", "__pycache__"} & set(p.relative_to(SRC).parts))


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    """``(module, line, imported)`` for every module ``path`` names.  Statements and import-call strings are ``imported``; a
    ``"module:attr"`` string is not, since any ``word:word`` constant matches: only the forbidden and core-direction rules
    apply to it."""
    package = ["verity", *path.relative_to(SRC).parts[:-1]]
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno, True) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.module, node.lineno, True))
        elif isinstance(node, ast.ImportFrom) and node.level > len(package):
            out.append(("." * node.level + (node.module or ""), node.lineno, True))
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1]
            out += [(".".join([*base, head]), node.lineno, True) for head in ([node.module] if node.module else [a.name for a in node.names])]
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS):
            out.append((node.args[0].value, node.lineno, True))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value):
            out.append((node.value.partition(":")[0], node.lineno, False))
    return out


def _unit(dotted: str) -> str | None:
    """The unit a module under ``verity`` (``primitives.circuits.cut``) belongs to, or None for a grouping or a top-level file."""
    parts = dotted.split(".")
    return next((".".join(parts[:n]) for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in ALLOWED), None)


def _dotted(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _external(name: str, imported: bool, allowed: set[str]) -> bool:
    """Whether a non-core ``name`` breaks the rules: it is forbidden and not ``allowed``, or it is imported and is neither stdlib
    nor in ``allowed``."""
    top = name.split(".")[0]
    return (top in FORBIDDEN_EXTERNAL and top not in allowed) or (
        imported and top != "verity" and top not in sys.stdlib_module_names and top not in allowed)


@pytest.mark.parametrize("unit", sorted(ALLOWED))
def test_core_unit_imports_only_inward(unit: str) -> None:
    files = [p for p in _sources() if _unit(_dotted(p)) == unit]
    assert files, f"core unit {unit!r} has no source: remove it from ALLOWED"
    offenders = []
    for path in files:
        for name, line, imported in _imports(path):
            parts = name.split(".")
            target = _unit(".".join(parts[1:])) if parts[0] == "verity" and len(parts) > 1 else None
            third_party = set() if unit.startswith("protocols.") else DEPENDENCIES
            if (_external(name, imported, third_party | EXCEPTIONS.get(unit, set()))
                    or (parts[0] == "verity" and parts[1:2] != ["errors"] and target not in (None, unit, *ALLOWED[unit]))):
                offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


def test_every_source_belongs_to_a_unit_or_is_a_grouping() -> None:
    """A new package under ``verity`` is a new unit, listed in ``ALLOWED``; what is outside every unit is a grouping's
    ``__init__.py`` (``primitives``, ``protocols.accounting``) or a file directly under ``verity/``, and imports the standard
    library and nothing else: ``verity/__init__.py`` re-exports nothing and ``verity/errors.py`` is shared by every unit, the
    stdlib-only references included."""
    offenders = []
    for path in _sources():
        if _unit(_dotted(path)) is not None:
            continue
        if path.name != "__init__.py" and path.parent != SRC:
            offenders.append(f"{path.relative_to(SRC)}: outside every unit; add its package to ALLOWED")
        offenders += [f"{path.relative_to(SRC)}:{line}: {name}" for name, line, imported in _imports(path)
                      if _external(name, imported, set()) or name.split(".")[0] == "verity"]
    assert not offenders, "\n".join(offenders)


def test_every_source_directory_is_a_package() -> None:
    """``verity`` is one regular package: every directory of its own Python has an ``__init__.py``, so no part of it resolves
    as a namespace package from whatever else is on the path.  Tests and Lean scripts are run by path and stay out."""
    missing = sorted({str(p.parent.relative_to(SRC.parent)) for p in _sources() for d in [p.parent, *p.parent.parents]
                      if d.is_relative_to(SRC) and not (d / "__init__.py").exists()})
    assert not missing, missing


@pytest.mark.parametrize("subtree", NUMPY_FREE)
def test_scalar_references_are_stdlib_only(subtree: str) -> None:
    offenders = [f"{path.relative_to(SRC)}:{line}: {name}"
                 for path in _sources(SRC / subtree)
                 if not any(path.relative_to(SRC).as_posix().startswith(u) for u in NUMPY_USERS)
                 for name, line, _ in _imports(path)
                 if name.split(".")[0] == "numpy"]
    assert not offenders, "\n".join(offenders)


def _docstrings(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


CENSUS_PATH = re.compile(r"census[/\\]|\b(hardware|networks|workloads|instance_sets)(\.schema)?\.json\b")


def test_core_never_reads_the_census() -> None:
    """The census (``catalog/census/``, data only) is read by the renderer, benchmarks, tools and the website, never by core: core takes
    such numbers as parameters.  No string in core code (docstrings aside) names the directory or one of its files."""
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docs = _docstrings(tree)
        offenders += [f"{path.relative_to(SRC)}:{n.lineno}" for n in ast.walk(tree)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and CENSUS_PATH.search(n.value)]
    assert not offenders, offenders
