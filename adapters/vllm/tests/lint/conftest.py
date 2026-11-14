"""What the lints leave in a test process: nothing they parsed.

The rules read the library through `_ratchet`'s shared trees, which `functools` caches hold with what each rule derived
from them: every module of the library, parsed. A process that goes on to other tests would hold them to its end, and
the cyclic collector would walk them in every full collection a later test or the library asks for. So the caches are
emptied when the process leaves this directory. A lint that runs after that parses again.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import Any

import pytest


def lint_modules() -> list[ModuleType]:
    """This directory's modules that the process has imported."""
    return [m for name, m in sorted(sys.modules.items()) if name == __package__ or name.startswith(f"{__package__}.")]


def caches(modules: Iterable[ModuleType]) -> list[Any]:
    """Each `functools` cache that one of ``modules`` defines at its top level. One a module only imports is its definer's."""
    return [value for module in modules for value in vars(module).values()
            if getattr(value, "__module__", None) == module.__name__ and callable(getattr(value, "cache_clear", None))]


@pytest.fixture(scope="package", autouse=True)
def _nothing_parsed_outlives_the_lints() -> Iterator[None]:
    yield
    for held in caches(lint_modules()):
        held.cache_clear()
