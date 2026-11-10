"""`engine.stock`: an in-process engine built from ``EngineArgs``, vLLM's ``SamplingParams`` and its version, over stand-ins
of vLLM's modules; and importing the module imports no vLLM."""
from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from verity_vllm.engine import stock

VERSION = "0.28.1rc1.dev472+gd9105ea80"


@pytest.fixture
def fake_vllm(monkeypatch):
    built = []

    class LLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            built.append(args)
            return SimpleNamespace(args=args)

    root = types.ModuleType("vllm")
    root.EngineArgs = lambda **kw: SimpleNamespace(**kw)
    root.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    root.__version__ = VERSION
    core = types.ModuleType("vllm.v1.engine.llm_engine")
    core.LLMEngine = LLMEngine
    for name, mod in {"vllm": root, "vllm.v1": types.ModuleType("vllm.v1"),
                      "vllm.v1.engine": types.ModuleType("vllm.v1.engine"), "vllm.v1.engine.llm_engine": core}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return built


def test_the_engine_is_built_from_its_engine_args_and_the_rest_passes_through(fake_vllm):
    eng = stock.engine("/models/m", dtype="bfloat16", enforce_eager=True)
    assert fake_vllm == [eng.args]
    assert vars(eng.args) == {"model": "/models/m", "dtype": "bfloat16", "enforce_eager": True}
    assert vars(stock.sampling(temperature=0.8, seed=7)) == {"temperature": 0.8, "seed": 7}
    assert stock.version() == VERSION


def test_importing_it_imports_no_vllm(monkeypatch):
    class NoVllm:
        def find_spec(self, name, path=None, target=None):
            assert name.split(".")[0] != "vllm", f"importing engine.stock imported {name}"

    for name in [m for m in sys.modules if m.split(".")[0] == "vllm"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoVllm(), *sys.meta_path])
    importlib.reload(stock)
