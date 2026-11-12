"""`machine-model`'s commands, one module a command, which `cli` finds here: a new command is a new module and touches no
other file.

A command module `NAME.py` (`list_.py` for `list`, a builtin's name) holds its usage as its docstring, which `machine-model`
prints with no command, `add(sub)`, which adds its parser to `sub` (argparse's subparsers), and `run(args) -> int`. A
module whose name starts with `_` holds what several commands share.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def commands() -> dict[str, ModuleType]:
    """Each command's module by the command's name, in name order."""
    names = sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))
    return {n.rstrip("_"): importlib.import_module(f"{__name__}.{n}") for n in names}
