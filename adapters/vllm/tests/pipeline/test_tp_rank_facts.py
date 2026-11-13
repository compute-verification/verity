"""`verity_vllm.pipeline.tp.rank_worker._tp_facts`: what a rank says of its own process, which both TP records keep under `ranks`:
the process and the devices it may use, the worker's and the runner's classes, the TP group vLLM made on it with each transport
its communicator holds, the NCCL variables it inherited (a credential by name only) and its parallel configuration.  A read that
fails is recorded by its error and the rest is still said.  CPU-only: vLLM is `tests.fake_vllm`, the group, the communicator
and the worker are stand-ins, and the test sets `torch.cuda`'s three reads."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests import fake_vllm
from verity_vllm import settings
from verity_vllm.pipeline.tp import rank_worker as RW

HERE = __name__
TRANSPORTS = ("ca_comm", "qr_comm", "symm_mem_comm", "pynccl_comm", "fi_allreduce_fusion", "all2all_manager")
DUMMY = "dummy"


class _Runner:
    """The rank's model runner: the facts read its class."""


class _Worker:
    """vLLM's Worker as far as the facts read it: its config and its model runner."""

    def __init__(self, parallel_config: Any) -> None:
        self.vllm_config = SimpleNamespace(parallel_config=parallel_config)
        self.model_runner = _Runner()


class _CustomAllReduce:
    """A transport that states all three of what the facts read from one."""
    disabled = True
    max_size = 8 << 20
    available = False


class _PyNccl:
    """A transport that states none of them."""


class _Communicator:
    def __init__(self, **transports: Any) -> None:
        self.__dict__.update(transports)


def _group(communicator: Any) -> SimpleNamespace:
    return SimpleNamespace(rank_in_group=1, world_size=2, rank=5, unique_name="tp:0", use_custom_op_call=True,
                           device_communicator=communicator)


def _rank(monkeypatch, *, group: Any = None, parallel_config: Any = None, environ: dict[str, str] | None = None) -> Any:
    """A worker with the extension mixed in, on a process whose TP group is ``group`` (a callable: what reading the group does)
    and whose only NCCL and device variables are ``environ``'s."""
    fake = fake_vllm.install(monkeypatch)
    read_group = group if callable(group) else lambda: group
    fake.set("vllm.distributed.parallel_state:get_tp_group", read_group)
    fake.set("vllm.distributed.parallel_state:get_pp_group", lambda: SimpleNamespace(world_size=3))
    for name in [k for k in os.environ if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"]:
        monkeypatch.delenv(name)
    for name, value in (environ or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "a stand-in device")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    w = _Worker(parallel_config if parallel_config is not None else SimpleNamespace())
    w.__class__ = type("Worker", (_Worker, RW.TP2CaptureWorkerExtension), {})
    return w


def test_a_rank_says_its_process_its_classes_its_group_and_each_transport_the_communicator_holds(monkeypatch):
    comm = _Communicator(ca_comm=_CustomAllReduce(), pynccl_comm=_PyNccl())
    facts = RW._tp_facts(_rank(monkeypatch, group=_group(comm), environ={"CUDA_VISIBLE_DEVICES": "0,1"}))
    assert (facts["pid"], facts["ppid"], facts["cuda_visible_devices"]) == (os.getpid(), os.getppid(), "0,1")
    assert facts["worker_mro"] == [f"{HERE}.Worker", f"{HERE}._Worker", "verity_vllm.pipeline.tp.rank_worker.TP2CaptureWorkerExtension",
                                   "builtins.object"]
    assert facts["worker_class"] == facts["worker_mro"][1] and facts["runner_class"] == f"{HERE}._Runner"
    assert facts["device"] == {"current": 1, "name": "a stand-in device", "count_visible": 2} and "device_error" not in facts
    tp = facts["tp"]
    assert {k: tp[k] for k in ("rank", "world_size", "global_rank", "group_name", "use_custom_op_call", "pp_world_size")} == {
        "rank": 1, "world_size": 2, "global_rank": 5, "group_name": "tp:0", "use_custom_op_call": True, "pp_world_size": 3}
    assert tp["device_communicator"] == f"{HERE}._Communicator" and "tp_error" not in facts
    assert tp["ca_comm"] == {"class": "_CustomAllReduce", "disabled": True, "max_size": 8 << 20, "available": False}
    assert tp["pynccl_comm"] == {"class": "_PyNccl", "disabled": False, "max_size": 0, "available": True}
    assert [tp[t] for t in TRANSPORTS if t not in ("ca_comm", "pynccl_comm")] == [None] * 4
    assert json.loads(json.dumps(facts)) == facts


def test_a_group_with_no_communicator_names_none_and_holds_no_transport(monkeypatch):
    tp = RW._tp_facts(_rank(monkeypatch, group=_group(None)))["tp"]
    assert tp["device_communicator"] is None and [tp[t] for t in TRANSPORTS] == [None] * len(TRANSPORTS)
    assert (tp["rank"], tp["world_size"]) == (1, 2)


@pytest.mark.parametrize("environ, said", [({}, None), ({"CUDA_VISIBLE_DEVICES": ""}, ""), ({"CUDA_VISIBLE_DEVICES": "3"}, "3")],
                         ids=["unset", "empty, which hides every device", "one device"])
def test_the_devices_a_rank_may_use_are_the_variables_string_and_none_when_it_is_unset(monkeypatch, environ, said):
    assert settings.setting("CUDA_VISIBLE_DEVICES").empty                       # declared: an empty string is a value, not unset
    assert RW._tp_facts(_rank(monkeypatch, group=_group(None), environ=environ))["cuda_visible_devices"] == said


def test_a_rank_records_the_nccl_variables_it_inherited_in_their_order_and_a_credential_by_name_only(monkeypatch):
    environ = {"NCCL_PROTO": "Simple", "NCCL_DUMMY_TOKEN": DUMMY, "NCCL_ALGO": "Ring", "CUDA_VISIBLE_DEVICES": "0"}
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")                             # another family's: not a rank's NCCL fact
    facts = RW._tp_facts(_rank(monkeypatch, group=_group(None), environ=environ))
    assert list(facts["nccl_env"].items()) == [("NCCL_PROTO", "Simple"), ("NCCL_DUMMY_TOKEN", settings.REDACTED), ("NCCL_ALGO", "Ring")]
    assert DUMMY not in json.dumps(facts)
    assert RW._tp_facts(_rank(monkeypatch, group=_group(None)))["nccl_env"] == {}


def test_a_device_or_a_group_that_cannot_be_read_is_recorded_by_its_error_and_the_rest_is_still_said(monkeypatch):
    def no_group():
        raise RuntimeError("no TP group on this process")

    def no_device():
        raise AssertionError("no CUDA here")

    w = _rank(monkeypatch, group=no_group, environ={"NCCL_ALGO": "Tree"})
    monkeypatch.setattr(torch.cuda, "current_device", no_device)
    facts = RW._tp_facts(w)
    assert facts["device_error"] == repr(AssertionError("no CUDA here")) and "device" not in facts
    assert facts["tp_error"] == repr(RuntimeError("no TP group on this process")) and "tp" not in facts
    assert facts["nccl_env"] == {"NCCL_ALGO": "Tree"} and facts["pid"] == os.getpid() and facts["runner_class"] == f"{HERE}._Runner"


def test_a_ranks_parallel_configuration_is_seven_fields_as_json_with_none_for_one_the_config_lacks(monkeypatch):
    class Backend:
        def __str__(self) -> str:
            return "a backend that is no JSON value"

    pc = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1, world_size=2, distributed_executor_backend=Backend(),
                         disable_custom_all_reduce=True, worker_extension_cls=RW.EXTENSION, not_one_of_them="left out")
    said = RW._tp_facts(_rank(monkeypatch, group=_group(None), parallel_config=pc))["parallel_config"]
    assert said == {"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "world_size": 2,
                    "distributed_executor_backend": "a backend that is no JSON value", "disable_custom_all_reduce": True,
                    "worker_cls": None, "worker_extension_cls": "verity_vllm.pipeline.tp.rank_worker.TP2CaptureWorkerExtension"}
