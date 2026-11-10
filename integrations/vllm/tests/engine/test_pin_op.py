"""`engine.pin_op`: one name a module or a ``torch.ops`` namespace is called through, replaced once and put back, over
modules written for the test and an op registered for it."""
from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from verity_vllm.engine import hooks
from verity_vllm.engine.pin_op import loads_global, pin_op

EXPERTS = '''
import types

def aligner(x):
    return ("vllm", x)

ops = types.SimpleNamespace(aligner=aligner)

def experts(x):
    def inner():
        return aligner(x)
    return inner()

def bypassing(x):
    return ops.aligner(x)
'''


@pytest.fixture
def site(tmp_path, monkeypatch, request):
    """A module importable by name and not yet imported, whose ``experts`` calls its global ``aligner``."""
    name = f"pin_op_{request.node.name.removeprefix('test_')[:40]}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(EXPERTS))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield name
    finally:
        sys.modules.pop(name, None)


def pinned(orig):
    return lambda x: ("pinned", orig(x))


def test_a_global_is_pinned_once_its_module_is_imported_and_uninstall_puts_it_back(site):
    pin = pin_op(site, "aligner", pinned, caller="experts")
    assert site not in sys.modules and not pin.installed and pin.original is hooks.ABSENT
    module = importlib.import_module(site)
    assert pin.installed and module.experts(1) == ("pinned", ("vllm", 1)) and pin.original(1) == ("vllm", 1)
    pin.uninstall()
    assert not pin.installed and module.experts(1) == ("vllm", 1) and hooks.live(f"pin-op:{site}:aligner") == ()


def test_a_global_of_an_imported_module_is_pinned_now_and_a_pin_still_waiting_disarms(site):
    pin = pin_op(site, "aligner", pinned)
    pin.uninstall()
    module = importlib.import_module(site)
    assert module.experts(1) == ("vllm", 1)
    pin = pin_op(site, "aligner", pinned, caller="experts")
    try:
        assert pin.installed and module.experts(1) == ("pinned", ("vllm", 1))
    finally:
        pin.uninstall()


def test_a_pin_its_caller_would_bypass_is_refused_and_installs_nothing(site):
    module = importlib.import_module(site)
    assert loads_global(module.experts, "aligner") and not loads_global(module.bypassing, "aligner")
    with pytest.raises(RuntimeError, match="bypassing doesn't load aligner as its global"):
        pin_op(site, "aligner", pinned, caller="bypassing")
    assert module.experts(1) == ("vllm", 1) and hooks.live(f"pin-op:{site}:aligner") == ()


def test_a_second_pin_and_a_pin_under_another_patch_are_refused(site):
    pin = pin_op(site, "aligner", pinned)
    with pytest.raises(ValueError, match="already pinned"):
        pin_op(site, "aligner", pinned)
    pin.uninstall()
    module = importlib.import_module(site)
    with hooks.patched(module, "aligner", lambda x: ("other", x)):
        with pytest.raises(RuntimeError, match="which is already on"):
            pin_op(site, "aligner", pinned)
    assert module.experts(1) == ("vllm", 1)


@pytest.fixture(scope="module")
def op_namespace():
    torch = pytest.importorskip("torch")
    lib = torch.library.Library("verity_pin_op_test", "DEF")
    lib.define("double(Tensor x) -> Tensor")
    lib.impl("double", lambda x: x * 2, "CPU")
    yield torch, "torch.ops.verity_pin_op_test"


def test_an_op_is_pinned_now_keeping_its_overloads_and_uninstall_puts_it_back(op_namespace):
    torch, site = op_namespace
    ns = torch.ops.verity_pin_op_test
    real = ns.double
    pin = pin_op(site, "double", lambda op: lambda x: op(x) + 1)
    try:
        assert pin.installed and pin.original is real
        assert ns.double(torch.ones(2)).tolist() == [3.0, 3.0]
        assert ns.double.default is real.default and ns.double.default(torch.ones(2)).tolist() == [2.0, 2.0]
        with pytest.raises(ValueError, match="op namespace"):
            pin_op(site, "triple", lambda op: op, caller="experts")
    finally:
        pin.uninstall()
    assert ns.double is real


def test_an_op_nothing_registered_is_refused(op_namespace):
    _, site = op_namespace
    with pytest.raises(RuntimeError, match="import the library that registers it"):
        pin_op(site, "missing", lambda op: op)
