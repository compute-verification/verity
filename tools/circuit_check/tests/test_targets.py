"""`targets._authored`: a Definition is this tree's by the file of the module that defines it, so a package that moves to a new
directory and import name stays in `REGISTERED`, and one whose module can't be judged fails by name."""
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


def test_registered_is_what_the_import_name_rule_gave():
    """Today's tree: deciding by file keeps exactly the Definitions the old rule (the top-level import name is `verity`,
    `verity_vllm` or `verity_pouw`) kept. Drop this once a rename moves a package off those names."""
    from verity.ir.defs import REGISTRY

    TG.load_registries()
    old = sorted(i for i, d in REGISTRY.defs.items()
                 if getattr(getattr(d, "evaluate", None) or getattr(d, "body", None), "__module__", "").partition(".")[0]
                 in ("verity", "verity_vllm", "verity_pouw"))
    assert len(TG.REGISTERED) > 200
    assert TG.REGISTERED == old


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
