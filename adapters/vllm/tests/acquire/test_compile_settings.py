"""What `compile_settings` records of a built engine's compilation, and where it finds the engine's configuration.

The Commit hands it the engine (`pipeline.commit`).  It reads the configuration from the engine, else from the engine's in-process
model runner (`verity_vllm.engine.handle`), and records nothing but the compile cache's switch where it finds none.  The engines
here are stand-ins that hold only the attributes read.  CPU only."""
from __future__ import annotations

import inspect
import types
from typing import Any

import pytest
import torch

from verity_vllm.acquire.sources.compiled_source import compile_settings

ns = types.SimpleNamespace
NOTHING = {"compile_cache_disabled": False}


def _config(mode: int) -> Any:
    """A configuration whose compilation mode is `mode` and that states nothing else."""
    return ns(compilation_config=ns(mode=mode), model_config=ns())


def _stated(mode: int) -> dict[str, Any]:
    """The record of `_config(mode)`."""
    return {"compilation_mode": str(mode), "cudagraph_mode": "None", "cudagraph_capture_sizes": [], "max_cudagraph_capture_size": None,
            "splitting_ops": [], "use_inductor_graph_partition": False, "backend": "None", "custom_ops": [],
            "inductor_compile_config": {}, "emulate_precision_casts": False, "enforce_eager": False, "torch": torch.__version__,
            "compile_cache_disabled": False}


def _engine(config: Any) -> Any:
    """A built engine that states `config`."""
    return ns(llm_engine=ns(vllm_config=config))


def _in_process(config: Any, **own: Any) -> Any:
    """An engine with no `llm_engine` whose in-process model runner states `config`; `own` are its own attributes."""
    return ns(engine_core=ns(model_executor=ns(driver_worker=ns(model_runner=ns(vllm_config=config)))), **own)


class _CoreRaises:
    """An engine whose core cannot be read, for a reason that is not the handle's own refusal of a core in another process."""

    @property
    def engine_core(self) -> Any:
        raise ValueError("no core")


@pytest.fixture(autouse=True)
def _compile_cache_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)


def test_a_built_engines_compilation_is_recorded_field_by_field() -> None:
    cc = ns(mode=3, cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[4, 1, 2], max_cudagraph_capture_size=4, splitting_ops=("a.op",),
            use_inductor_graph_partition=1, backend="inductor", custom_ops=("none",), inductor_compile_config={"emulate_precision_casts": 1})
    assert compile_settings(_engine(ns(compilation_config=cc, model_config=ns(enforce_eager=1)))) == {
        "compilation_mode": "3", "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4], "max_cudagraph_capture_size": 4,
        "splitting_ops": ["a.op"], "use_inductor_graph_partition": True, "backend": "inductor", "custom_ops": ["none"],
        "inductor_compile_config": {"emulate_precision_casts": 1}, "emulate_precision_casts": True, "enforce_eager": True,
        "torch": torch.__version__, "compile_cache_disabled": False}


def test_a_configuration_that_states_only_its_mode_is_recorded_with_each_other_field_empty() -> None:
    assert compile_settings(_engine(_config(3))) == _stated(3)
    old = ns(compilation_config=ns(level=2, cudagraph_capture_sizes=None, splitting_ops=None, custom_ops=None, inductor_compile_config=None),
             model_config=ns())
    assert compile_settings(_engine(old)) == _stated(2)


def test_at_most_256_capture_sizes_are_recorded() -> None:
    config = ns(compilation_config=ns(mode=3, cudagraph_capture_sizes=range(300, 0, -1)), model_config=ns())
    assert compile_settings(_engine(config))["cudagraph_capture_sizes"] == list(range(45, 301))


@pytest.mark.parametrize(("llm", "mode"), [
    (_engine(_config(1)), 1),
    (ns(llm_engine=_in_process(_config(3), vllm_config=_config(1))), 1),
    (_in_process(_config(3)), 3),
    (ns(llm_engine=_in_process(_config(3))), 3),
    (ns(llm_engine=_in_process(_config(3), vllm_config=None)), 3),
])
def test_the_configuration_is_the_engines_then_the_in_process_runners(llm: Any, mode: int) -> None:
    assert compile_settings(llm) == _stated(mode)


@pytest.mark.parametrize("llm", [
    None,
    ns(),
    _engine(None),
    _in_process(None),
    ns(engine_core=ns(model_executor=ns(driver_worker=None))),
])
def test_an_engine_that_states_no_configuration_in_this_process_is_recorded_by_the_cache_switch_alone(llm: Any) -> None:
    assert compile_settings(llm) == NOTHING


def test_the_engine_is_all_it_takes() -> None:
    assert list(inspect.signature(compile_settings).parameters) == ["llm"]


def test_an_engine_is_found_as_the_handle_finds_it() -> None:
    """`verity_vllm.engine.handle.engine_of`: the `LLMEngine` itself, and the engine under a wrapper's `vllm`."""
    assert compile_settings(_in_process(_config(3), vllm_config=_config(1))) == _stated(1)
    assert compile_settings(ns(vllm=_engine(_config(1)))) == _stated(1)


def test_a_core_that_cannot_be_read_is_recorded_as_the_error() -> None:
    assert compile_settings(_CoreRaises()) == {"error": "ValueError('no core')", "compile_cache_disabled": False}


def test_a_configuration_that_cannot_be_read_is_recorded_as_the_error() -> None:
    out = compile_settings(_engine(ns()))
    assert out == {"error": out["error"], "compile_cache_disabled": False} and out["error"].startswith("AttributeError(")
    config = ns(compilation_config=ns(mode=3, cudagraph_capture_sizes=["x"]), model_config=ns())
    out = compile_settings(_engine(config))
    assert out == {"compilation_mode": "3", "cudagraph_mode": "None", "error": out["error"], "compile_cache_disabled": False}
    assert out["error"].startswith("ValueError(")


@pytest.mark.parametrize(("value", "disabled"), [("1", True), ("0", False), ("", False), ("true", False), (" 1", False)])
def test_the_compile_cache_is_disabled_by_the_one_value(monkeypatch: pytest.MonkeyPatch, value: str, disabled: bool) -> None:
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", value)
    assert compile_settings() == {"compile_cache_disabled": disabled}
    assert compile_settings(_engine(_config(3))) == {**_stated(3), "compile_cache_disabled": disabled}
