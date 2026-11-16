"""The task families' listers: each module here, `<family>.py` with a family's name's - written _, lists the tasks of the
tasks.toml table `[family.<family>]` as `tasks(root: Path, files: list[str]) -> list[dict]`. ``files`` is the tree's files
(`graph.tracked`: POSIX, relative, sorted), and each task is a graph/v2 task without its priority and key; the table fills
what the task leaves out (`graph.complete`)."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType


def name_of(module: str) -> str:
    return module.replace("_", "-")


def modules() -> dict[str, ModuleType]:
    """{family name: its lister module}, every module of this package."""
    return {name_of(m.name): importlib.import_module(f"{__name__}.{m.name}")
            for m in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name) if not m.name.startswith("_")}


def listers() -> dict[str, object]:
    """{family name: its `tasks(root, files)`}."""
    return {name: m.tasks for name, m in modules().items()}


def tasks(name: str, root: Path, files: list[str]) -> list[dict]:
    return listers()[name](root, files)
