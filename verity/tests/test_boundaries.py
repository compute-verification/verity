"""The core dependency rules (README "Development") as import-boundary checks over the source tree.

    ir <- evaluation <- ml,  ir <- commitments,  ir + ml + commitments <- proofs;  randomness and claims import no
    core module;  nothing depends back on a later layer, and no core module imports a backend, frontend, integration,
    protocol, tool or external runtime: core imports the standard library, its declared dependencies and itself.

An import is an import statement (a relative one resolved against its package), a string handed to
``importlib.import_module``, ``__import__`` or ``importorskip``, or a ``"module:attr"`` string naming what to load at run
time: a module named as data is still a dependency.

numpy is a core dependency (architecture decision: `verity.primitives.circuits.evaluation` batches over numpy arrays and kernels sit beside
their Definitions in `verity_catalog.definitions`).  The plain-integer references stay numpy-free on purpose, because they are the
specification the Rust guest, the Lean vectors and the probes are checked against: `verity.primitives.circuits` and `verity_catalog.silicon`.
`verity.primitives.randomness` is exact stdlib arithmetic too.
"""

from __future__ import annotations

import ast
import importlib.util
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import verity

#: Each directory the import system finds `verity` in: one while it is a regular package, one per member that ships part of it
#: once it is a namespace (PEP 420).
PORTIONS = [Path(p).resolve() for p in verity.__path__]
#: Each module of `verity`, wherever the member that ships it lives: a package's directory, or a top-level module's file
#: (``errors.py``).
MODULES = {m.name: Path(importlib.util.find_spec(f"verity.{m.name}").origin).resolve().parent if m.ispkg
           else Path(importlib.util.find_spec(f"verity.{m.name}").origin).resolve() for m in pkgutil.iter_modules(verity.__path__)}

#: For each core module, the sibling core modules it may import.  ``errors`` is shared by all.
ALLOWED = {
    "ir": set(),
    "evaluation": {"ir"},
    "ml": {"ir", "evaluation"},
    "commitments": {"ir"},
    "proofs": {"ir", "ml", "commitments"},
    "resource_accounting": {"ir", "proofs", "commitments"},
    "randomness": set(),
    "claims": set(),
}
#: Import names core never imports or names as a string: the other workspace distributions (backends, integrations,
#: protocols, tools), external runtimes, and the repository's own directories.
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_numerical", "verity_flock", "verity.protocols.verification.sampled_proofs", "verity.protocols.accounting.space.pous",
                      "verity_torch", "circuit_check", "research", "sp1", "vllm", "torch", "jax", "onnx",
                      "backends", "frontends", "integrations", "protocols", "tools", "veritor")
#: Core's third-party dependencies (``verity/pyproject.toml``); anything else it imports is the stdlib or ``verity``.
DEPENDENCIES = {"numpy"}
#: Source subtrees that must stay stdlib-only: the scalar references.
NUMPY_FREE = ("ir", "ml/tc", "randomness", "claims")

MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


def _package(path: Path) -> list[str]:
    """The dotted package a core source file belongs to, as parts."""
    if path.parent in PORTIONS:
        return ["verity"]
    for name, root in MODULES.items():
        if root.is_dir() and path.is_relative_to(root):
            return ["verity", name, *path.relative_to(root).parts[:-1]]
    raise AssertionError(f"{path} is in no module of verity")


def _rel(path: Path) -> str:
    """``path`` as it lies under ``verity/``, whichever member ships it (``ml/tc/total.py``)."""
    return "/".join([*_package(path)[1:], path.name])


def _sources(module: str) -> list[Path]:
    """The Python files of ``module`` (``ml/tc`` names a subpackage), which must exist: a rule over nothing passes."""
    head, _, rest = module.partition("/")
    assert head in MODULES, f"verity.{head} is not importable: found {sorted(MODULES)} in {PORTIONS}"
    root = MODULES[head] / rest if rest else MODULES[head]
    files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    assert files, f"no source of verity.{module.replace('/', '.')} under {root}"
    return files


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    """``(module, line, imported)`` for every module ``path`` names.  Statements and import-call strings are ``imported``; a
    ``"module:attr"`` string is not, since any ``word:word`` constant matches: only the forbidden and core-direction rules
    apply to it."""
    package = _package(path)
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


def _external(name: str, imported: bool, allowed: set[str]) -> bool:
    """Whether a non-core ``name`` breaks the rules: it is forbidden, or it is imported and is neither stdlib nor in ``allowed``."""
    top = name.split(".")[0]
    return top in FORBIDDEN_EXTERNAL or (imported and top != "verity" and top not in sys.stdlib_module_names and top not in allowed)


def _core_modules() -> list[str]:
    return sorted(name for name, root in MODULES.items() if root.is_dir())


def _top_level_files() -> list[Path]:
    return sorted(p for portion in PORTIONS for p in portion.glob("*.py"))


def test_every_rule_is_over_modules_that_are_found() -> None:
    """The rules below are parametrized over what the import system finds, so a module that moved where they don't look would
    drop out of them silently: the modules every layout has, and the files directly under ``verity``, are found."""
    assert {"ir", "evaluation", "ml", "errors"} <= MODULES.keys(), f"found {sorted(MODULES)} in {PORTIONS}"
    assert MODULES["errors"] in _top_level_files(), _top_level_files()


@pytest.mark.parametrize("module", _core_modules())
def test_core_module_imports_only_inward(module: str) -> None:
    assert module in ALLOWED, f"new core module {module!r}: add it to ALLOWED with its permitted dependencies"
    offenders = []
    for path in _sources(module):
        for name, line, imported in _imports(path):
            parts = name.split(".")
            target = parts[1] if parts[0] == "verity" and len(parts) > 1 else None
            if _external(name, imported, DEPENDENCIES) or target not in (None, module, "errors", *ALLOWED[module]):
                offenders.append(f"{_rel(path)}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


def test_top_level_files_import_only_the_standard_library() -> None:
    """``verity/__init__.py`` re-exports nothing and ``verity/errors.py`` is shared by every module, the stdlib-only references
    included: the files directly under ``verity/`` import the standard library and nothing else."""
    offenders = [f"{path.name}:{line}: {name}" for path in _top_level_files() for name, line, imported in _imports(path)
                 if _external(name, imported, set()) or name.split(".")[0] == "verity"]
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("subtree", NUMPY_FREE)
def test_scalar_references_are_stdlib_only(subtree: str) -> None:
    offenders = [f"{_rel(path)}:{line}: {name}"
                 for path in _sources(subtree)
                 for name, line, _ in _imports(path)
                 if name.split(".")[0] == "numpy" or (subtree == "ml/tc" and name.startswith("verity.")
                                                      and name.split(".")[:2] != ["verity", "errors"]
                                                      and name.split(".")[:3] != ["verity", "ml", "tc"])]
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
    for path in [*_top_level_files(), *(p for module in _core_modules() for p in _sources(module))]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docs = _docstrings(tree)
        offenders += [f"{_rel(path)}:{n.lineno}" for n in ast.walk(tree)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and CENSUS_PATH.search(n.value)]
    assert not offenders, offenders
