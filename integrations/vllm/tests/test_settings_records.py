"""Records a reader fills from the environment, pinned at the bytes they had before the reader moved to `verity_vllm.settings`: the hot
worker's key (`hot.env_identity`, `hot.so_identity`) and the fast Match's declared configuration (`global_match_fast.config`, on every
result document's impl block)."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from verity_vllm.check.match import global_match as GM
from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.pipeline import hot


def _digest(doc: object) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in list(os.environ):
        monkeypatch.delenv(k)
    return monkeypatch


def test_the_hot_key_takes_the_environment_as_before(env: pytest.MonkeyPatch) -> None:
    for k, v in {"PATH": "/bin", "PYTHONPATH": "/src", "VLLM_BATCH_INVARIANT": "1", "VERITY_WINDOW_MB": "256", "VERITY_TIMELINE": "/t",
                 "VERITY_HOT_SUBMIT": "1", "HASH_THREADS": "4", "CUDA_HOME": "/cuda", "TORCH_X": "", "OMP_NUM_THREADS": "8",
                 "NATIVE_COLLECT_BUILD": "/no-such-build", "HIDDEN_GPU_BUILD": "", "HOME": "/h", "NCCL_P2P_DISABLE": "1"}.items():
        env.setenv(k, v)
    assert _digest([hot.env_identity(), hot.so_identity([])]) == "0019aeeec8c143005b7a8b4b19531a0c"


@pytest.mark.parametrize(("values", "digest"), [
    ({"MATCH_WORKERS": "1"}, "37c433ce7a6f3d0ef54af75f49134685"),
    ({"MATCH_WORKERS": " 3 ", "MATCH_COMPACT": " 0 "}, "aee01761c9e2b4dd6f984669223ef09f"),
    ({"MATCH_WORKERS": "2", "MATCH_COMPACT": "false"}, "07dfbb9661bbc2f54bfcceb284cf36ab"),
    ({"MATCH_WORKERS": "1", "MATCH_COMPACT": ""}, "37c433ce7a6f3d0ef54af75f49134685"),
])
def test_the_fast_match_declares_its_configuration_as_before(env: pytest.MonkeyPatch, values: dict[str, str], digest: str) -> None:
    for k, v in values.items():
        env.setenv(k, v)
    GMF.ops()
    config = {k: v for k, v in GMF.config.items() if k != "cpu_count"}
    assert _digest([config, GM.impl_banner(GMF)]) == digest
