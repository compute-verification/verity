"""A Program id's `SRC` static names its source by the source's declared, versioned `DERIVE_SOURCE`, never by a Python module path:
the 09-22 records' `SRC` moved (`609750e4714a` -> `bfb0f1dba532`, every row) only because the wrappers moved out of
`verity_capture/experimental/cb_a/`. Loading `vllm_meta.py` under that old module path must give the same Program id."""
from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from torch import nn

from verity.ml.circuits.codec import encode_program, program_digest
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend import vllm_meta as vm
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.registry import b1

PROF = dict(mode="profile", profile="b1-eager")
OLD_PATH = "verity_capture.experimental.cb_a.vllm_meta"


def _config() -> dict:
    C = dict(b1.B0_CONFIG)
    C.update(NL=1, V=64, MAX_POS=64)
    return C


class _ToyVllmLM(nn.Module):
    """vLLM's model interface as the wrappers call it: `forward(input_ids, positions)` -> hidden, `compute_logits(hidden)`."""

    def __init__(self, C: dict) -> None:
        super().__init__()
        self.lm = ex.CausalLM(C)

    def forward(self, input_ids, positions):
        return self.lm.model(input_ids, positions, False, None)

    def compute_logits(self, hidden):
        return self.lm.lm_head(hidden)


@pytest.fixture(scope="module")
def moved_vllm_meta():
    spec = importlib.util.spec_from_file_location(OLD_PATH, vm.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[OLD_PATH] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(OLD_PATH, None)


def _derive_step(wrapper_cls, C: dict):
    with torch.device("meta"):
        mod = wrapper_cls(_ToyVllmLM(C), 0, 3)
    rep = derive_report(mod, (torch.empty((3,), dtype=torch.int32, device="meta"),), **PROF, config=C, name="vLLMStep[Toy]")
    assert rep.ok, rep.unsupported
    return rep


def test_program_id_is_unchanged_when_the_wrapper_module_moves(moved_vllm_meta):
    assert moved_vllm_meta.EngineStep.__module__ == OLD_PATH != vm.EngineStep.__module__
    C = _config()
    here, moved = _derive_step(vm.EngineStep, C), _derive_step(moved_vllm_meta.EngineStep, C)
    assert here.program.fn.id == moved.program.fn.id
    assert program_digest(encode_program(here.program)) == program_digest(encode_program(moved.program))
    src = here.program.fn.bindings["SRC"]
    assert src["class"] == "verity-vllm/engine-step/v1"
    assert here.provenance["source"]["python_class"] == f"{vm.__name__}.EngineStep"


def test_an_undeclared_source_still_names_its_module(moved_vllm_meta):
    """The fallback for sources that declare nothing (derive's own tests, HF modules): the id follows the module, which is why every
    wrapper `make_wrapper` returns declares its source (next test)."""
    C = _config()

    def undeclared(cls):
        return type(cls.__name__, (cls,), {"__module__": cls.__module__, "DERIVE_SOURCE": None})

    here, moved = _derive_step(undeclared(vm.EngineStep), C), _derive_step(undeclared(moved_vllm_meta.EngineStep), C)
    assert here.program.fn.id != moved.program.fn.id
    assert here.program.fn.bindings["SRC"]["class"] == f"{vm.__name__}.EngineStep"


def test_every_row_wrapper_declares_a_distinct_versioned_source():
    wrappers = (vm.EngineStep, vm.GreedyRequest, vm.PaddedGreedyRequest, vm.StochasticEngineStep, vm.StochasticRequest)
    ids = [w.DERIVE_SOURCE for w in wrappers]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("verity-vllm/") and i.rsplit("/", 1)[-1].startswith("v") for i in ids)
    with torch.device("meta"):
        step = vm.EngineStep(_ToyVllmLM(_config()), 0, 1)
    assert vm.WithPeers(step, bus=None).DERIVE_SOURCE == "verity-vllm/with-peers/v1[verity-vllm/engine-step/v1]"
