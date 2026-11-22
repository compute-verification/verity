"""Records a reader fills from the environment, pinned at their bytes on `main` before the reader moves to `verity_vllm.settings`, so
the move can't change them: the fast Match's declared configuration (`global_match_fast.config`, on every result document's impl
block), the staging environment the Commit hands its committers (`commit.staging_environ`), the configuration a committer
takes from that environment (the collector's on `test_staging_defaults`' CPU fakes, its name among it), and the fields of the
engine's records the environment fills (the replay's engine facts, the engine profile, the compilation facts, the generated kernels)."""
from __future__ import annotations

import hashlib
import json
import os
import types
from collections.abc import Iterator
from unittest import mock

import pytest

from tests import fake_vllm
from verity_vllm.check.match import global_match as GM
from verity_vllm.check.match import global_match_fast as GMF


def _digest(doc: object) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


@pytest.fixture
def env() -> Iterator[pytest.MonkeyPatch]:
    """An empty environment, the process's restored whole after the test: the code under test writes settings straight into
    `os.environ` (`settings.put`, as `commit.staging_environ` does), which no `MonkeyPatch` records.  The test's own `MonkeyPatch`
    is undone inside that, since one undone after it (the function's `monkeypatch`) would delete what a test set that the process
    holds too (PATH, HOME)."""
    with mock.patch.dict(os.environ, clear=True), pytest.MonkeyPatch.context() as mp:
        yield mp




@pytest.fixture(scope="module")
def committers():
    pytest.importorskip("torch")
    from tests.commit import test_staging_defaults
    from verity_vllm.commit.committer import native_host

    return test_staging_defaults, native_host


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




@pytest.mark.parametrize(("retain", "values", "digest"), [
    ("none", {}, "ff0b0f64489f1687c56406d80f95c9b3"),
    ("host", {"VERITY_STAGING_BOUNDED": " yes ", "VERITY_HI_PRIO": "", "VERITY_RETAIN_PINNED": "false", "VERITY_INLINE_D2H": "1",
              "VERITY_PIN": "true", "VERITY_FAULT": " skip_copy:3", "VERITY_WINDOW_MB": " 16 ", "VERITY_WINDOW_SLOTS": "3",
              "NATIVE_GATHER_BLOCKS": "64", "NATIVE_GATHER_AUTO_BYTES": "4096", "VERITY_CONSUMER_DELAY_MS": "2.5",
              "VERITY_LEARN_HOST_BUDGET_MB": "1024", "VERITY_RETAIN_EXCLUDE": "fa3_hidden_m1", "VERITY_SKIP": "a,,b",
              "VERITY_WEIGHTS_HASH": "device"}, "1eb12e9bd19c559e217418921c4074b6"),
    ("none", {"VERITY_STAGING_BOUNDED": "on", "VERITY_FAULT": "skip_copy:2", "VERITY_HI_PRIO": "1", "VERITY_RETAIN_PINNED": " no",
              "VERITY_INLINE_D2H": "true"}, "cbc856bbf4739c6b70ec3bed50df6bbd"),
])
def test_a_committer_takes_its_configuration_as_before(committers, env: pytest.MonkeyPatch, retain: str, values: dict[str, str],
                                                       digest: str) -> None:
    from verity_vllm import settings

    staging, native_host = committers
    com = staging.committer(env, retain, **values)
    keys = ("win_bytes", "win_slots_req", "bounded", "learn_budget_bytes", "hi_prio", "consumer_delay_s", "fault", "retain", "retain_pinned",
            "inline_d2h", "pin", "name", "windowed")
    nc = com._nc
    doc = {k: getattr(com, k) for k in keys} | {
        "retain_exclude": sorted(com.retain_exclude), "footprint": com.footprint is not None,
        "blocks_per_desc": getattr(nc, "blocks_per_desc", None), "auto_bytes_per_block": getattr(nc, "auto_bytes_per_block", None),
        "fault_skip_copy_at": getattr(nc, "fault_skip_copy_at", None), "skip": sorted(native_host.skip_classes_of_env()),
        "weights_hash": settings.VERITY_WEIGHTS_HASH}
    assert _digest(doc) == digest




_ENGINE_VARS = ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS",
                "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_LAUNCH_MODE", "VLLM_DISABLE_COMPILE_CACHE", "RUNPOD_POD_ID",
                "TORCHINDUCTOR_CACHE_DIR", "VERITY_INDUCTOR_DIR_PER_PROCESS", "HF_HOME")




@pytest.fixture(scope="module")
def build_target():
    pytest.importorskip("torch")
    from verity_vllm.program.frontend import export_compat, vllm_meta

    return export_compat, vllm_meta


_BUILD_VARS = ("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL", "TOKENIZERS_PARALLELISM",
               "VLLM_ATTENTION_BACKEND", "VLLM_USE_DEEP_GEMM", "CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize(("values", "digest"), [
    ({}, "4be0a8acddaeaa879c13009a998f19d6"),
    ({"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_LAYERNAME": "0", "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING",
      "TOKENIZERS_PARALLELISM": "false", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN", "VLLM_USE_DEEP_GEMM": "0", "CUDA_VISIBLE_DEVICES": "0,1"},
     "8b88316c51537deda318b282d9873e03"),
    (dict.fromkeys(_BUILD_VARS, ""), "92678ab9dfe7c37a7ae9002006956c51"),
    ({"VLLM_BATCH_INVARIANT": " 1", "VLLM_ATTENTION_BACKEND": "", "TOKENIZERS_PARALLELISM": "true", "CUDA_VISIBLE_DEVICES": " "},
     "b41e0b68c73454c37d4cc0cbd7cf4e5e"),
])
def test_a_builds_target_facts_take_the_environment_as_before(build_target, env: pytest.MonkeyPatch, values: dict[str, str],
                                                              digest: str) -> None:
    """The Build's target facts the environment fills: the variables it set that the pinned vLLM consults, ignores or doesn't
    own (`vllm_meta.consulted_facts`), and the host's CUDA visibility (`export_compat.host_facts`).  vLLM's ``envs``, the host's
    name and CUDA are faked."""
    import socket

    import torch

    export_compat, vllm_meta = build_target
    for k, v in values.items():
        env.setenv(k, v)
    fake_vllm.install(env)["vllm.envs"].environment_variables = dict.fromkeys(
        ("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "VLLM_LOGGING_LEVEL", "VLLM_USE_DEEP_GEMM"))
    env.setattr(socket, "gethostname", lambda: "host-a")
    env.setattr(torch.cuda, "is_available", lambda: False)
    doc = {"consulted": vllm_meta.consulted_facts(types.SimpleNamespace()), "host": export_compat.host_facts()}
    assert _digest(doc) == digest, doc
