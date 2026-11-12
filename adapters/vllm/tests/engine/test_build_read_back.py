"""What `verity_vllm.engine.build` reads back from a built engine, and the kernel config it asks vLLM for, on stand-ins.

The effective-execution document whole, for an engine whose config holds every field, for one whose config holds none
of the optional ones, and for an object that is no engine; the compilation facts where the config names its mode
``level`` and where the config cannot be read; the FlashAttention versions a model's layers carry; vLLM's FlashInfer
verdict; and the ``kernel_config`` of `engine_kwargs_for`, whatever the target's engine args hold under that key.
"""
from __future__ import annotations

import enum
import json
import logging
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

from tests import fake_vllm
from verity_vllm.engine import build
from verity_vllm.engine import engine_profile as prof


class _Mode(enum.IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class _Graphs(enum.Enum):
    NONE = 0
    PIECEWISE = 1


def _engine(config: NS) -> NS:
    return NS(llm_engine=NS(vllm_config=config))


def _full() -> NS:
    """A config that holds every field the document reads, each as a value the reader has to convert."""
    return NS(
        cache_config=NS(enable_prefix_caching=1, block_size="16", gpu_memory_utilization="0.5"),
        scheduler_config=NS(enable_chunked_prefill=0, max_num_batched_tokens="2048", long_prefill_token_threshold=7.0,
                            max_num_seqs=True, async_scheduling=1),
        model_config=NS(max_model_len="288", dtype="torch.bfloat16", enforce_eager=0),
        parallel_config=NS(tensor_parallel_size="2"),
        compilation_config=NS(mode=_Mode.VLLM_COMPILE, cudagraph_mode=_Graphs.PIECEWISE, cudagraph_capture_sizes=(4, "1", 2),
                              custom_ops=("none",)),
        use_v2_model_runner=0, optimization_level="2")


def _bare() -> NS:
    """A config that holds only what the readers take without a default."""
    return NS(cache_config=NS(), scheduler_config=NS(), model_config=NS(enforce_eager=True), compilation_config=NS(),
              use_v2_model_runner=1)


class _Layer:
    def __init__(self, *version: Any) -> None:
        if version:
            self.impl = NS(vllm_flash_attn_version=version[0])


class _Model:
    def __init__(self, *layers: Any) -> None:
        self.layers = layers

    def named_modules(self) -> list[tuple[str, Any]]:
        return [(f"layers.{i}", layer) for i, layer in enumerate(self.layers)]


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> fake_vllm.FakeVllm:
    monkeypatch.setattr(prof, "breakable_cudagraph", lambda: False)
    return fake_vllm.install(monkeypatch)


def test_the_document_of_an_engine_whose_config_holds_every_field(stand_in: fake_vllm.FakeVllm) -> None:
    stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = 1
    doc = build.effective_execution_doc(_engine(_full()), _Model(_Layer(3), _Layer(), _Layer(3)))
    assert doc == {
        "mode": "3", "cudagraph_mode": "_Graphs.PIECEWISE", "enforce_eager": False, "breakable_cudagraph": False,
        "capture_sizes": [1, 2, 4], "model_runner": "v1", "custom_ops": ["none"], "optimization_level": 2,
        "enable_prefix_caching": True, "enable_chunked_prefill": False, "max_num_batched_tokens": 2048,
        "long_prefill_token_threshold": 7, "tensor_parallel_size": 2, "max_num_seqs": 1, "max_model_len": 288,
        "dtype": "torch.bfloat16", "block_size": 16, "gpu_memory_utilization": 0.5, "async_scheduling": True,
        "batch_invariant": True, "flash_attn_version": 3, "execution_label": "bi-compiled"}
    assert list(doc) == ["mode", "cudagraph_mode", "enforce_eager", "breakable_cudagraph", "capture_sizes", "model_runner",
                         "custom_ops", "optimization_level", "enable_prefix_caching", "enable_chunked_prefill",
                         "max_num_batched_tokens", "long_prefill_token_threshold", "tensor_parallel_size", "max_num_seqs",
                         "max_model_len", "dtype", "block_size", "gpu_memory_utilization", "async_scheduling",
                         "batch_invariant", "flash_attn_version", "execution_label"]
    assert [type(doc[k]) for k in ("max_num_batched_tokens", "long_prefill_token_threshold", "tensor_parallel_size",
                                   "max_num_seqs", "max_model_len", "block_size")] == [int] * 6
    assert type(doc["gpu_memory_utilization"]) is float
    json.dumps(doc)


def test_the_document_of_an_engine_whose_config_holds_none_of_the_optional_fields(stand_in: fake_vllm.FakeVllm) -> None:
    stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = 0
    doc = build.effective_execution_doc(_engine(_bare()))
    assert doc == {
        "mode": "None", "cudagraph_mode": "None", "enforce_eager": True, "breakable_cudagraph": False, "capture_sizes": [],
        "model_runner": "v2", "custom_ops": [], "optimization_level": -1,
        "enable_prefix_caching": None, "enable_chunked_prefill": None, "max_num_batched_tokens": None,
        "long_prefill_token_threshold": None, "tensor_parallel_size": None, "max_num_seqs": None, "max_model_len": None,
        "dtype": "None", "block_size": None, "gpu_memory_utilization": None, "async_scheduling": None,
        "batch_invariant": False, "execution_label": "eager"}
    assert "flash_attn_version" not in doc, "no model was given"


@pytest.mark.parametrize("field", ["enable_prefix_caching", "enable_chunked_prefill", "max_num_batched_tokens",
                                   "long_prefill_token_threshold", "max_num_seqs", "max_model_len", "block_size",
                                   "gpu_memory_utilization", "async_scheduling"])
def test_a_field_the_config_holds_as_none_is_none_in_the_document(stand_in: fake_vllm.FakeVllm, field: str) -> None:
    stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = True
    config = _full()
    holder = next(c for c in (config.cache_config, config.scheduler_config, config.model_config) if hasattr(c, field))
    setattr(holder, field, None)
    doc = build.effective_execution_doc(_engine(config))
    assert doc[field] is None and "config_error" not in doc
    whole = build.effective_execution_doc(_engine(_full()))
    assert {k: v for k, v in doc.items() if k != field} == {k: v for k, v in whole.items() if k != field}


def test_a_field_that_cannot_be_converted_is_an_error_in_the_document_and_ends_the_read(stand_in: fake_vllm.FakeVllm) -> None:
    stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = True
    config = _full()
    config.scheduler_config.max_num_seqs = "eight"
    doc = build.effective_execution_doc(_engine(config))
    assert doc["config_error"].startswith("ValueError(") and "eight" in doc["config_error"]
    assert doc["tensor_parallel_size"] == 2 and "max_num_seqs" not in doc and "max_model_len" not in doc
    assert "batch_invariant" not in doc and doc["execution_label"] is None, "the label needs the batch invariance"


def test_the_document_of_an_object_that_is_no_engine_holds_two_errors_and_no_label(stand_in: fake_vllm.FakeVllm) -> None:
    doc = build.effective_execution_doc(NS())
    assert list(doc) == ["compilation_error", "config_error", "execution_label"] and doc["execution_label"] is None
    assert doc["compilation_error"].startswith("AttributeError(") and doc["config_error"].startswith("AttributeError(")


def test_an_error_in_the_document_is_its_first_200_characters(stand_in: fake_vllm.FakeVllm) -> None:
    stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = True
    config = _full()
    config.compilation_config.cudagraph_capture_sizes = ["x" * 400]
    config.model_config.max_model_len = "y" * 400
    doc = build.effective_execution_doc(_engine(config))
    for key, letter in (("compilation_error", "x"), ("config_error", "y")):
        assert len(doc[key]) == 200 and doc[key].startswith("ValueError(") and doc[key].endswith(letter * 20), key
    assert "mode" not in doc and doc["max_num_seqs"] == 1, "the compilation's fields are all or none; the config's up to the error"


def test_the_batch_invariance_is_read_from_vllms_environment_module(stand_in: fake_vllm.FakeVllm) -> None:
    for held, want, label in ((True, True, "bi-eager"), (False, False, "eager"), ("", False, "eager"), ("0", True, "bi-eager")):
        stand_in["vllm.envs"].VLLM_BATCH_INVARIANT = held
        doc = build.effective_execution_doc(_engine(_bare()))
        assert (doc["batch_invariant"], doc["execution_label"]) == (want, label), held


def test_compilation_facts_where_the_config_names_its_mode_level_and_holds_little_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prof, "breakable_cudagraph", lambda: True)
    monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)
    facts = build.compilation_facts(_engine(NS(compilation_config=NS(level=_Mode.VLLM_COMPILE), model_config=NS(enforce_eager=0))))
    assert facts == {"mode": "VLLM_COMPILE", "mode_value": 3, "cudagraph_mode": "None", "capture_sizes": [],
                     "max_cudagraph_capture_size": None, "use_inductor_graph_partition": False, "backend": "inductor",
                     "enforce_eager": False, "breakable_cudagraph": True, "compile_cache": "enabled",
                     "inductor_compile_config": {}, "emulate_precision_casts": False}
    plain = build.compilation_facts(_engine(NS(compilation_config=NS(mode="2", backend="eager", cudagraph_mode=_Graphs.NONE,
                                                                       cudagraph_capture_sizes=range(300, 0, -1),
                                                                       inductor_compile_config=None),
                                               model_config=NS(enforce_eager=1))))
    assert (plain["mode"], plain["mode_value"], plain["backend"], plain["cudagraph_mode"]) == ("2", "2", "eager", "NONE")
    assert plain["capture_sizes"] == list(range(45, 301)), "the first 256 the config lists, sorted"
    assert plain["inductor_compile_config"] == {} and plain["enforce_eager"] is True


@pytest.mark.parametrize(("held", "want"), [("1", "disabled"), ("0", "enabled"), ("true", "enabled"), ("", "enabled")])
def test_compilation_facts_record_the_compile_cache_switch(monkeypatch: pytest.MonkeyPatch, held: str, want: str) -> None:
    monkeypatch.setattr(prof, "breakable_cudagraph", lambda: False)
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", held)
    facts = build.compilation_facts(_engine(NS(compilation_config=NS(mode=_Mode.NONE), model_config=NS(enforce_eager=True))))
    assert facts["compile_cache"] == want


def test_compilation_facts_of_a_config_that_cannot_be_read_hold_the_error_and_what_was_read() -> None:
    assert list(build.compilation_facts(NS())) == ["error"]
    assert build.compilation_facts(NS())["error"].startswith("AttributeError(")
    part = build.compilation_facts(_engine(NS(compilation_config=NS(mode=_Mode.NONE, cudagraph_capture_sizes=["x"]))))
    assert part["mode"] == "NONE" and part["mode_value"] == 0 and "capture_sizes" not in part
    assert part["error"].startswith("ValueError(") and len(part["error"]) <= 200


def test_the_flash_attention_versions_a_models_layers_carry() -> None:
    assert build.flash_attn_versions(_Model()) == set()
    assert build.flash_attn_versions(_Model(_Layer(), NS(impl=None), NS(impl=NS()))) == set()
    assert build.flash_attn_versions(_Model(_Layer(2), _Layer(), _Layer(2))) == {2}
    assert build.flash_attn_versions(_Model(_Layer(2), _Layer(3), _Layer(None))) == {2, 3, None}
    assert build.observed_flash_attn_version(_Model(_Layer("3"), _Layer("3"))) == 3
    assert build.observed_flash_attn_version(_Model(_Layer(2), _Layer(3))) is None
    assert build.observed_flash_attn_version(_Model(_Layer(None))) is None
    assert build.observed_flash_attn_version(_Model()) is None


def test_vllms_flashinfer_verdict_or_none_and_why_none_at_debug(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake = fake_vllm.install(monkeypatch)
    for held, want in ((True, True), (False, False), (1, True), ("", False)):
        fake.set("vllm.utils.flashinfer:has_flashinfer", lambda held=held: held)
        assert build.vllm_has_flashinfer() is want

    def fails() -> bool:
        raise OSError("no nvcc")

    fake.set("vllm.utils.flashinfer:has_flashinfer", fails)
    with caplog.at_level(logging.DEBUG, logger="verity_vllm"):
        assert build.vllm_has_flashinfer() is None
        monkeypatch.setitem(sys.modules, "vllm.utils.flashinfer", None)
        assert build.vllm_has_flashinfer() is None, "a vLLM with no such module"
    said = [r.getMessage() for r in caplog.records if r.name == "verity_vllm.engine.build"]
    assert len(said) == 2 and "OSError('no nvcc')" in said[0] and "ModuleNotFoundError" in said[1]
    assert {r.levelno for r in caplog.records if r.name == "verity_vllm.engine.build"} == {logging.DEBUG}


class _Target:
    """A declared target that sets nothing but its engine args."""

    int8_parts = ()

    def __init__(self, **engine_args: Any) -> None:
        self.args = engine_args

    def env(self) -> dict[str, str]:
        return {}

    def engine_args(self) -> dict[str, Any]:
        return dict(self.args)


@pytest.fixture
def checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(build.served, "apply", lambda execution: {})
    # unloaded for the test: one imported before the engine's pin, by another test in the same worker, is refused (`require_inductor`)
    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({"checkpoints": [{"repo": "r/m", "role": "M", "downloaded": True, "local_path": str(tmp_path)}]}))
    return path


@pytest.mark.parametrize(("target", "want"), [
    (None, {"enable_flashinfer_autotune": False}),
    (_Target(), {"enable_flashinfer_autotune": False}),
    (_Target(kernel_config=None), {"enable_flashinfer_autotune": False}),
    (_Target(kernel_config={}), {"enable_flashinfer_autotune": False}),
    (_Target(kernel_config={"moe_backend": "triton"}), {"moe_backend": "triton", "enable_flashinfer_autotune": False}),
    (_Target(kernel_config={"enable_flashinfer_autotune": True}), {"enable_flashinfer_autotune": True}),
])
def test_the_kernel_config_turns_the_flashinfer_autotune_off_unless_the_target_sets_it(checkpoints: Any, target: Any, want: dict) -> None:
    held = target.args.get("kernel_config") if target is not None else None
    before = dict(held) if held else held
    kw, _, _ = build.engine_kwargs_for(checkpoints, case="M", target=target)
    assert kw["kernel_config"] == want and list(kw["kernel_config"]) == list(want)
    assert kw["kernel_config"] is not held and held == before, "the target's own dict is not written"


def test_an_engine_arg_replaces_the_kernel_config_whole(checkpoints: Any) -> None:
    kw, _, _ = build.engine_kwargs_for(checkpoints, case="M", target=_Target(kernel_config={"moe_backend": "triton"}),
                                       engine_args={"kernel_config": {"gemm": "tuned"}})
    assert kw["kernel_config"] == {"gemm": "tuned"}
