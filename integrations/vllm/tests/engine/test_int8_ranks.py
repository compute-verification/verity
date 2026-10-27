"""`engine.int8_ranks`: the int8 parts registered in this process and exported to a tensor-parallel engine's ranks, which register
them from the environment when they import their `worker_extension_cls`.  CPU-only: the two vLLM registrations are stand-ins."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from verity_vllm.engine import int8_ranks as IR


@pytest.fixture
def registered(monkeypatch):
    """The parts whose vLLM registration ran, in order (`int8_vllm.register`, `int8_attention_vllm.register` stood in)."""
    import verity_vllm.engine as pkg
    seen: list[str] = []
    for part, mod in (("linears", "int8_vllm"), ("attention", "int8_attention_vllm")):
        stand_in = ModuleType(f"verity_vllm.engine.{mod}")
        stand_in.register = lambda part=part: seen.append(part)
        monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)
        monkeypatch.setattr(pkg, mod, stand_in, raising=False)
    monkeypatch.delenv(IR.ENV, raising=False)
    return seen


def test_register_registers_each_part_and_exports_them(registered, monkeypatch):
    assert IR.register(["linears", "attention", "linears"]) == ["attention", "linears"]
    assert registered == ["linears", "attention"]
    import os
    assert os.environ[IR.ENV] == "attention,linears"
    assert IR.register([]) == [] and IR.ENV not in os.environ and registered == ["linears", "attention"]


def test_register_refuses_a_part_it_does_not_serve(registered):
    with pytest.raises(ValueError, match=r"int8 parts \['kv'\] are not among"):
        IR.register(["linears", "kv"])
    assert registered == []


@pytest.mark.parametrize("env, want", [("linears", ["linears"]), ("attention,linears", ["linears", "attention"]), ("", [])])
def test_a_rank_registers_what_the_driver_exported_when_it_imports_the_module(registered, monkeypatch, env, want):
    monkeypatch.setenv(IR.ENV, env)
    importlib.reload(IR)
    assert registered == want


def test_the_rank_extension_imports_the_module():
    from verity_vllm.pipeline.tp import rank_worker as RW
    assert RW.int8_ranks is IR
    mod, _, cls = IR.EXTENSION.rpartition(".")
    assert mod == IR.__name__ and getattr(IR, cls) is IR.Int8Ranks
