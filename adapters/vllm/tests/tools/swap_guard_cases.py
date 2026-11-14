"""Cases for the suite's swap guard (`conftest.py`), which `test_swap_guard.py` runs in a pytest of their own.  Never collected
with the suite: the guard is meant to fail the second and the fourth of them, at their teardown, and the test after each to read
the module that was loaded."""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import verity.circuits
import verity.circuits.defs as loaded

NAME = "verity.circuits.defs"


def test_a_stand_in_module_set_with_monkeypatch_is_not_left(monkeypatch):
    monkeypatch.setitem(sys.modules, NAME, ModuleType(NAME))
    monkeypatch.setattr(verity.circuits, "defs", ModuleType(NAME))


def test_a_loaded_module_left_swapped_is_failed_by_name():
    sys.modules[NAME] = ModuleType(NAME)


def test_the_test_after_it_reads_the_module_that_was_loaded():
    assert sys.modules[NAME] is loaded


def test_a_package_attribute_left_swapped_is_failed_by_name():
    package: Any = verity.circuits
    package.defs = ModuleType(NAME)


def test_the_test_after_that_reads_the_attribute_that_was_there():
    assert verity.circuits.defs is loaded and sys.modules[NAME] is loaded
