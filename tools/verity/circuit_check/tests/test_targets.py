"""`targets._authored`: a Definition is this tree's by the file of the module that defines it, so a package that moves to a new
directory and import name stays in `REGISTERED`, and one whose module can't be judged fails by name.  `targets.root_functions`:
every root function a binding file defines runs, the listed ones first."""
from __future__ import annotations

import sys
import types

import pytest

from circuit_check import targets as TG


def _definition(module: str, id: str = "Probe_v1"):
    def evaluate(*args):
        return args

    evaluate.__module__ = module
    return types.SimpleNamespace(id=id, evaluate=evaluate)


def _module(monkeypatch, name: str, file) -> None:
    mod = types.ModuleType(name)
    mod.__file__ = None if file is None else str(file)
    monkeypatch.setitem(sys.modules, name, mod)


def test_a_definition_is_the_trees_by_its_modules_file(monkeypatch):
    _module(monkeypatch, "catalog.silicon", TG.REPO / "catalog" / "src" / "catalog" / "silicon.py")
    assert TG._authored(_definition("catalog.silicon"))
    _module(monkeypatch, "tests.test_probe", TG.REPO / "catalog" / "tests" / "test_probe.py")
    assert not TG._authored(_definition("tests.test_probe"))
    _module(monkeypatch, "conftest", TG.REPO / "catalog" / "conftest.py")
    assert not TG._authored(_definition("conftest"))
    _module(monkeypatch, "elsewhere", TG.REPO.parent / "other-checkout" / "verity" / "x.py")
    assert not TG._authored(_definition("elsewhere"))


def test_a_definition_whose_module_has_no_file_is_refused_by_id(monkeypatch):
    _module(monkeypatch, "fileless", None)
    with pytest.raises(LookupError, match="'Fileless_v1'"):
        TG._authored(_definition("fileless", "Fileless_v1"))
    with pytest.raises(LookupError, match="'Unloaded_v1'"):
        TG._authored(_definition("never.imported.anywhere", "Unloaded_v1"))
    with pytest.raises(LookupError, match="'Bare_v1'"):
        TG._authored(types.SimpleNamespace(id="Bare_v1"))


def _binding_files(monkeypatch, **sidecars: dict) -> None:
    """Stand-in sidecars, each a binding file of these names (`sidecars.binding_modules`), and no registries."""
    from circuit_check import sidecars as SC

    mods = []
    for name, ns in sidecars.items():
        m = types.ModuleType(f"{name}.circuit-check")
        m.__file__ = f"{name}.circuit-check.py"
        vars(m).update(ns)
        mods.append((name, m))
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(SC, "binding_modules", lambda: mods)


def test_a_sidecars_own_root_function_runs_after_the_listed_ones_by_name(monkeypatch):
    """A root function need not be listed in `ROOT_FUNCTIONS` (one file per binding, no shared list to edit): any ``_X_roots``
    a binding file defines runs, after the listed ones, by name."""
    roots = lambda: []  # noqa: E731
    _binding_files(monkeypatch, a={"_zeta_roots": roots, "_core_roots": roots}, b={"_alpha_roots": roots, "_alpha_bindings": roots})
    assert TG.root_functions() == (*TG.ROOT_FUNCTIONS, "_alpha_roots", "_zeta_roots")
    assert [units for units, _make in TG._root_bindings("_alpha_roots")] == [("b.circuit-check#_alpha_roots",
                                                                            "b.circuit-check#_alpha_bindings")]


def test_a_sidecars_bindings_without_their_root_function_is_refused(monkeypatch):
    _binding_files(monkeypatch, a={"_lone_bindings": lambda: []})
    with pytest.raises(ValueError, match="_lone_bindings is no binding circuit-check runs"):
        TG.binding_files()
