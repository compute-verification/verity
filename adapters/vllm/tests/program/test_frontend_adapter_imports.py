"""`verity_vllm.program.frontend` imports the rest of the adapter only through the names its docstring lists.

The translator is to become an adapter of its own, so every name its modules take from `verity_vllm` outside their
directory is written down once, in the docstring of `verity_vllm.program.frontend.__init__`, with what it is for.
These tests hold that list to the code both ways. They read the source and import nothing, so an import inside a
function or under `TYPE_CHECKING` counts like any other.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "verity_vllm.program.frontend"
DIR = Path(__file__).resolve().parents[2].joinpath(*PACKAGE.split("."))
#: a training step's derivation sits on the translator and is not one of its modules
NOT_THE_TRANSLATORS = frozenset({"train_step.py"})
IMPORTERS = frozenset({"import_module", "__import__"})


def names_in(source: str) -> set[str]:
    """The dotted names a module of the package imports from `verity_vllm` outside the package: `from a.b import c` is
    `a.b.c`, `import a.b` and `import_module("a.b")` are `a.b`, and a relative import is resolved against the package."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                above = PACKAGE.split(".")[: len(PACKAGE.split(".")) - node.level + 1]
                base = ".".join([*above, *([base] if base else [])])
            found.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            first = node.args[0]
            if name in IMPORTERS and isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
    return {n for n in found if f"{n}.".startswith("verity_vllm.") and not f"{n}.".startswith(f"{PACKAGE}.")}


def imported() -> dict[str, list[str]]:
    """Each such name the translator's modules import, with the modules that import it."""
    places: dict[str, list[str]] = {}
    for path in sorted(DIR.glob("*.py")):
        if path.name not in NOT_THE_TRANSLATORS:
            for name in names_in(path.read_text()):
                places.setdefault(name, []).append(path.name)
    return places


def listed() -> list[str]:
    """The names the package's docstring lists: each indented line of it that starts with `verity_vllm.`."""
    doc = ast.get_docstring(ast.parse((DIR / "__init__.py").read_text()), clean=False) or ""
    return [line.split()[0] for line in doc.splitlines() if line.startswith("    verity_vllm.")]


def test_the_scan_reads_every_way_a_module_imports():
    source = (
        "from verity_vllm import settings\n"
        "from verity_vllm.engine import env as engine_env, hooks\n"
        "from verity_vllm.program.frontend.derive import derive\n"
        "from . import emit\n"
        "from .rules import base\n"
        "from .. import inputs\n"
        "from ..prune import kept_records\n"
        "import verity_vllm.config as config\n"
        "import torch\n"
        "def late():\n"
        "    from verity_vllm.log import init_logger\n"
        "    importlib.import_module('verity_vllm.errors')\n"
        "    importlib.import_module('vllm.envs')\n"
    )
    assert sorted(name.removeprefix("verity_vllm.") for name in names_in(source)) == [
        "config", "engine.env", "engine.hooks", "errors", "log.init_logger", "program.inputs",
        "program.prune.kept_records", "settings"]


def test_every_import_of_the_adapter_is_listed():
    names = listed()
    assert len(names) == len(set(names)), f"{PACKAGE}'s docstring lists a name twice: {names}"
    unlisted = {name: files for name, files in imported().items() if name not in names}
    assert not unlisted, (
        f"{PACKAGE} imports names of the adapter that its docstring does not list: {unlisted}. Add each to the list in "
        f"{DIR / '__init__.py'} with what it is for, or do without it: the translator moves to an adapter of its own.")


def test_every_listed_name_is_imported():
    stale = sorted(set(listed()) - set(imported()))
    assert not stale, f"{PACKAGE}'s docstring lists names that none of its modules imports: {stale}. Delete their lines."
