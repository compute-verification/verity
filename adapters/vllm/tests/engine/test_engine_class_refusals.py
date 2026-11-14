"""`verity_vllm.engine.profiles.engine_classes.effective_refusals` for a class whose compiled execution names no compilation.

Class G's compiled execution names one (`breakable`), so a compiled class-G engine is held to its CUDA graphs.  A class may
serve a compiled execution at no target compilation (`EXECUTIONS[name]["compiled"]` is None, as every class's eager is): its
built engine is then held to the class's world, served limits, prefix cache, context limit and environment, and to no
CUDA-graph line, since the class states none.  No vLLM and no GPU: the engine's read-back is a dict written here."""
from __future__ import annotations

import pytest

from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.program.frontend.target_profile import TargetProfile

CTX = 8192
ENV = {"CLASS_P_SWITCH": "on"}


@pytest.fixture
def built(monkeypatch):
    """Class "P", registered for one test: class G's facts and limits, and a compiled execution at no compilation.  The
    fixture's value is a compiled engine's read-back that runs at all of them."""
    monkeypatch.setitem(EC.CLASSES, "P", EC.CLASSES["G"])
    monkeypatch.setitem(EC.EXECUTIONS, "P", {"eager": None, "compiled": None})
    monkeypatch.setitem(EC.ENVS, "P", ENV)
    return {"tensor_parallel_size": EC.CLASS_G["world"], "enable_prefix_caching": False, **EC.CLASS_G_SERVED,
            "max_model_len": CTX, "enforce_eager": False}


def test_a_compiled_engine_of_a_class_that_names_no_compilation_is_held_to_no_cuda_graph_line(built):
    assert EC.compilation("P", "compiled") is None
    assert EC.effective_refusals("P", built, ENV, CTX) == []
    # whatever graphs the engine read back, the class states none to hold them to
    assert EC.effective_refusals("P", {**built, "breakable_cudagraph": False, "capture_sizes": [1, 2, 4]}, ENV, CTX) == []


def test_it_is_still_held_to_the_classes_world_its_launchs_context_limit_and_the_classes_environment(built):
    assert EC.effective_refusals("P", {**built, "tensor_parallel_size": 4, "max_model_len": 4096}, {}, CTX) == [
        "the built engine's tensor_parallel_size is 4, not the class's 8",
        f"the built engine's max_model_len is 4096, not the launch's {CTX}",
        "the built engine's environment sets CLASS_P_SWITCH=None, not the class's 'on'"]


def test_class_g_compiled_is_held_to_its_breakable_graphs():
    sizes = list(TargetProfile.COMPILATIONS["breakable"]["capture_sizes"])
    eff = {"tensor_parallel_size": 8, "enable_prefix_caching": False, **EC.CLASS_G_SERVED, "max_model_len": CTX,
           "enforce_eager": False, "breakable_cudagraph": True, "capture_sizes": sizes}
    assert EC.effective_refusals("G", eff, EC.env("G"), CTX) == []
    assert EC.effective_refusals("G", {**eff, "breakable_cudagraph": False, "capture_sizes": [1]}, EC.env("G"), CTX) == [
        "the built engine's breakable_cudagraph is False, not the class's True",
        f"the built engine's capture_sizes is [1], not the class's {sizes}"]
