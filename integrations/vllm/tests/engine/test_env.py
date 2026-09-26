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


def test_the_engine_profile_reads_this_table() -> None:
    from verity_vllm.engine import engine_profile as prof

    assert prof.DECLARED_ENV is env.ENGINE
    assert env.ENGINE == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
                          "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    assert env.EXPORT == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_LAYERNAME": "0", "HF_HUB_OFFLINE": "1",
                          "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false"}
