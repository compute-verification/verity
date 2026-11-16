"""The two fields of an engine profile that the environment fills beside the settings registry's:
`verity_vllm.engine.engine_profile.gpu_info`'s ``pod`` (``RUNPOD_POD_ID``, else the node's name) and
`verity_vllm.engine.engine_profile._backends_probe`'s ``nccl_env`` (the seven ``NCCL_*`` variables it names, where set).  vLLM is
the stand-in of ``tests/fake_vllm.py`` and CUDA's device properties are faked: no GPU, no vLLM, no TP group."""
from __future__ import annotations

import os
from types import SimpleNamespace as NS
from typing import Any

import pytest

from tests import fake_vllm
from verity_vllm.engine import engine_profile as EP

POD = "RUNPOD_POD_ID"
NCCL = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_LAUNCH_MODE")
TP_GROUP = "vllm.distributed.parallel_state:get_tp_group"


@pytest.fixture
def one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: NS(name="F", major=8, minor=9, total_memory=1 << 30, multi_processor_count=8))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(EP, "_run", lambda argv: "550.1")
    monkeypatch.setattr(EP, "_tuned_gemm_arch_family", lambda major, minor: "ada")


def test_a_gpu_records_pod_is_the_environments_string_as_held(one_gpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for held in ("pod-7", " p ", ""):
        monkeypatch.setenv(POD, held)
        assert EP.gpu_info()["pod"] == held


def test_a_gpu_records_pod_is_the_nodes_name_where_the_environment_names_none(one_gpu: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POD, raising=False)
    assert EP.gpu_info()["pod"] == os.uname().nodename


def _probe(fake: fake_vllm.FakeVllm, group: Any) -> dict[str, Any]:
    fake.set(TP_GROUP, group)
    runner = NS(get_model=lambda: NS(named_modules=lambda: []), sampler=None)
    return EP._backends_probe(NS(model_runner=runner))


def _group() -> NS:
    comm = NS(ca_comm=None, symm_mem_comm=None, qr_comm=None, fi_allreduce_fusion=None, pynccl_comm=None)
    return NS(rank_in_group=1, world_size=2, device_communicator=comm)


def test_the_probes_nccl_env_is_each_of_the_seven_that_is_set_as_held(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_vllm.install(monkeypatch)
    for name in (*NCCL, "NCCL_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    assert _probe(fake, _group)["all_reduce"]["nccl_env"] == {}

    monkeypatch.setenv("NCCL_BUFFSIZE", "")
    monkeypatch.setenv("NCCL_ALGO", " Tree ")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    held = _probe(fake, _group)["all_reduce"]["nccl_env"]
    assert held == {"NCCL_ALGO": " Tree ", "NCCL_BUFFSIZE": ""}, "one of the family that the probe does not name is left out"
    assert list(held) == ["NCCL_ALGO", "NCCL_BUFFSIZE"], "in the order the probe names them"

    for i, name in enumerate(NCCL):
        monkeypatch.setenv(name, str(i))
    held = _probe(fake, _group)["all_reduce"]["nccl_env"]
    assert held == {name: str(i) for i, name in enumerate(NCCL)} and list(held) == list(NCCL)


def test_the_probe_carries_the_tp_group_beside_the_runners_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_vllm.install(monkeypatch)
    for name in NCCL:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    doc = _probe(fake, _group)
    assert doc == {
        "attention": None, "linear": "UnquantizedLinearMethod._gemm_impl", "moe": None, "sampler": None,
        "tp_rank": 1, "tp_world_size": 2,
        "all_reduce": {"device_communicator": "types.SimpleNamespace", "custom_allreduce": None, "symm_mem": False,
                       "quick_allreduce": False, "flashinfer_allreduce": False, "pynccl": False, "nccl_env": {}},
    }

    def no_group() -> NS:
        raise AssertionError("tensor model parallel group is not initialized")

    doc = _probe(fake, no_group)
    assert doc == {"attention": None, "linear": "UnquantizedLinearMethod._gemm_impl", "moe": None, "sampler": None,
                   "tp_error": "AssertionError('tensor model parallel group is not initialized')"}
