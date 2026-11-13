"""`verity_vllm.pipeline.tp.rank_worker`: a rank's picks for the cross-rank pass
(`TP2CaptureWorkerExtension.tp2_commit_xrank_dump`).  The rank hands `verity_vllm.pipeline.tp.xrank_dump.dump` its own Commit
state, the all-reduce switches its engine and its process's environment set, and its device as the capability and the SM count
the driver's routes are keyed on, with the driver's seed, pair and bounds as given.  CPU-only: `dump` is a stand-in that
keeps what it is called with, and the test sets the environment and `torch.cuda`'s two reads."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.pipeline.tp import xrank_dump
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension

SWITCHES = ("VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_BATCH_INVARIANT", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
DEFAULT = {"custom_all_reduce": True, "symmetric_memory": True, "flashinfer": True, "batch_invariant": False, "nccl_transport": None}


class _Rank(TP2CaptureWorkerExtension):
    """A rank after its Commit: its engine's all-reduce switch and the state the picks are read from."""

    def __init__(self, disable_custom_all_reduce: bool) -> None:
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(disable_custom_all_reduce=disable_custom_all_reduce))
        self._tp2_commit = {"rank": 1, "world": 2}


def _picks(monkeypatch, env: dict[str, str], *, disable: bool = False, **kw: Any) -> tuple[Any, _Rank, dict[str, Any]]:
    """(what the rank's method returns, the rank, what `dump` was called with) in a process whose all-reduce switches are ``env``."""
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(major=12, minor=i - 3, multi_processor_count=188))
    called: dict[str, Any] = {}

    def dump(st: dict[str, Any], **given: Any) -> dict[str, Any]:
        called.update(given, st=st)
        return {"rank": st["rank"], "picks": 4}

    monkeypatch.setattr(xrank_dump, "dump", dump)
    w = _Rank(disable)
    return w.tp2_commit_xrank_dump(out_dir="/driver/out", seed_hex="ab" * 32, pair=2, **kw), w, called


def test_the_picks_are_dumps_over_the_ranks_commit_state_switches_and_device(monkeypatch):
    res, w, called = _picks(monkeypatch, {})
    assert res == {"rank": 1, "picks": 4} and called.pop("st") is w._tp2_commit
    assert called == {"settings": DEFAULT, "out_dir": "/driver/out", "seed_hex": "ab" * 32, "pair": 2, "per_stratum": 2, "max_rows": 16,
                      "device": ("12.0", 188)}                                        # the current device's capability and SMs


def test_the_drivers_bounds_are_passed_as_given(monkeypatch):
    _, _, called = _picks(monkeypatch, {}, per_stratum=5, max_rows=64)
    assert (called["per_stratum"], called["max_rows"]) == (5, 64)


@pytest.mark.parametrize("env, disable, differs", [
    ({}, True, {"custom_all_reduce": False}),
    ({"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}, False, {"symmetric_memory": False}),
    ({"VLLM_ALLREDUCE_USE_FLASHINFER": "false"}, False, {"flashinfer": False}),
    ({"VLLM_BATCH_INVARIANT": "1"}, False, {"batch_invariant": True}),
    ({"NCCL_P2P_DISABLE": "1"}, False, {"nccl_transport": "shm"}),
    ({"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}, False, {}),
    ({"VLLM_ALLREDUCE_USE_SYMM_MEM": "0", "VLLM_BATCH_INVARIANT": "1", "NCCL_P2P_DISABLE": "1"}, True,
     {"custom_all_reduce": False, "symmetric_memory": False, "batch_invariant": True, "nccl_transport": "shm"}),
], ids=["the engine's switch", "symmetric memory off", "flashinfer off", "batch invariant", "NCCL over shared memory",
        "shared memory off too", "four at once"])
def test_the_switches_are_the_engines_and_the_environments(monkeypatch, env, disable, differs):
    _, _, called = _picks(monkeypatch, env, disable=disable)
    assert called["settings"] == {**DEFAULT, **differs}
