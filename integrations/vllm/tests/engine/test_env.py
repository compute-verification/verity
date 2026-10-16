"""`engine.env`: the vLLM pins are defaults an operator's export overrides, a target profile's values replace the
process's, and every caller reads the one table."""
from __future__ import annotations

import os

import pytest

from verity_vllm.engine import env

PINNED = sorted({*env.ENGINE, *env.EXPORT, "HF_HOME", "HF_HUB_OFFLINE"})


@pytest.fixture
def clean(monkeypatch):
    for k in PINNED:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_pin_fills_what_is_unset_and_keeps_what_the_operator_exported(clean) -> None:
    clean.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    got = env.pin(env.EXPORT)
    assert got == {**env.EXPORT, "VLLM_LOGGING_LEVEL": "DEBUG"}
    assert {k: os.environ[k] for k in env.EXPORT} == got


def test_apply_engine_pins_the_profile_and_the_checkpoint_store(clean, tmp_path) -> None:
    got = env.apply_engine(str(tmp_path))
    assert got == env.ENGINE
    assert os.environ["HF_HOME"] == str(tmp_path) and os.environ["HF_HUB_OFFLINE"] == "1"


def test_apply_engine_without_a_store_leaves_hf_home_alone(clean) -> None:
    env.apply_engine(None)
    assert "HF_HOME" not in os.environ and os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_target_profile_replaces_the_pin(clean) -> None:
    clean.setenv("VLLM_BATCH_INVARIANT", "1")
    clean.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    assert env.apply_target({"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0"}) == {
        "VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0"}
    assert env.apply_engine(None)["VLLM_BATCH_INVARIANT"] == "0"


def test_hidden_devices_override_what_the_process_inherited(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env.hide_devices()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_set_since_is_what_the_process_set_or_changed_after_the_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("KEPT", "1")
    monkeypatch.delenv("CUBLASLT_WORKSPACE_SIZE", raising=False)
    before = env.snapshot()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("CUBLASLT_WORKSPACE_SIZE", "1")
    assert env.set_since(before) == {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}


def test_an_engine_records_what_its_build_wrote_into_the_environment() -> None:
    import dataclasses
    import inspect

    from verity_vllm.engine import build

    src = inspect.getsource(build.build_engine)
    assert src.index("env0 = engine_env.snapshot()") < src.index("engine_kwargs_for(") < src.index("llm = LLM(**kw)")
    assert "engine_env.set_since(env0))" in src.split("llm = LLM(**kw)")[1].split("eng.effective")[0]
    assert [f.name for f in dataclasses.fields(build.Engine)][-1] == "environ"


def test_the_engine_profile_reads_this_table() -> None:
    from verity_vllm.engine import engine_profile as prof

    assert prof.DECLARED_ENV is env.ENGINE
    assert env.ENGINE == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                          "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1"}
    assert env.EXPORT == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_LAYERNAME": "0", "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
                          "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"}


def test_a_checkpoint_not_on_this_host_is_refused_by_name(tmp_path) -> None:
    import json

    import pytest

    from verity_vllm.engine import engine_profile as prof

    entry = {"repo": "org/model", "role": "ROLE_X", "downloaded": True, "local_path": str(tmp_path / "gone")}
    (m := tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": [entry]}))
    with pytest.raises(SystemExit, match=f"org/model: local snapshot {tmp_path / 'gone'} missing"):
        prof.checkpoint_for("ROLE_X", m)
    m.write_text(json.dumps({"checkpoints": [dict(entry, downloaded=False)]}))
    with pytest.raises(SystemExit, match="org/model is not downloaded on this pod"):
        prof.checkpoint_for("ROLE_X", m)
    (tmp_path / "gone").mkdir()
    m.write_text(json.dumps({"checkpoints": [entry]}))
    assert prof.checkpoint_for("ROLE_X", m) == entry


def test_serving_and_the_build_put_a_shared_expert_on_one_stream() -> None:
    """A Build orders a shared expert on the main stream, so a served run must run it there too: on two streams, its log's order
    is not the execution order, and wiring (operand_provenance) fails."""
    assert env.ENGINE["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == env.EXPORT["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
