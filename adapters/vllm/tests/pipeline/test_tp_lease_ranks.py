"""`verity_vllm.pipeline.tp.lease` where it meets vLLM and the device: under a held lease vLLM spawns its ranks; the distributed
groups a rank destroys and the worker whose shutdown it replaces are vLLM's own; a rank that gave its GPU back ends at that
shutdown with its output written and no teardown run; and a rank whose CUDA context cannot be reset, or stays, says so and is
left as it was, which keeps every GPU to the Commit's exit.  CPU-only: vLLM is `tests.fake_vllm`; the committer, the model, the
driver and the held lease are `tests.pipeline.test_tp_gpu_lease`'s stand-ins."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests import fake_vllm
from tests.pipeline import test_tp_gpu_lease as L
from verity_vllm.engine import hooks
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline.tp import lease as TPL

lease = L.lease

STARTED = "/row/gpu_lease.started"


def _worker(rank: int = 1, **state: Any) -> Any:
    return SimpleNamespace(_tp2_commit={"rank": rank, **state})


# ------------------------------------------------------------------------------------------------------------------- the lease taken


@pytest.mark.parametrize("held", [True, False], ids=["a lease is held", "no lease"])
def test_under_a_held_lease_vllm_spawns_its_ranks_in_the_environment_the_lease_set(monkeypatch, held):
    """A forked rank would hold the lease's lock descriptors past the release; without a lease the environment is left alone."""
    environ = {"CUDA_VISIBLE_DEVICES": "4,5"}
    took: list[tuple] = []

    def acquire(*a: Any, **k: Any) -> None:
        took.append((a, k))
        if held:
            GL._HELD.append((None, None, [], STARTED, environ))

    monkeypatch.setattr(GL, "_HELD", [])
    monkeypatch.setattr(GL, "acquire", acquire)
    monkeypatch.delenv(TPL.SPAWN, raising=False)
    assert TPL.acquire() is held and took == [((), {})]
    assert environ == {"CUDA_VISIBLE_DEVICES": "4,5", **({"VLLM_WORKER_MULTIPROC_METHOD": "spawn"} if held else {})}
    assert TPL.SPAWN not in os.environ


# -------------------------------------------------------------------------------------------------------------- a rank off its device


def _off(monkeypatch, ev: list, com: Any, **groups: Any) -> tuple[dict[str, Any], L._Model]:
    """`rank_off` on rank 1 over vLLM's stand-in, whose distributed groups note their end on ``ev`` (or are ``groups``')."""
    torch = pytest.importorskip("torch")
    fake = fake_vllm.install(monkeypatch)
    model = L._Model()
    state = fake["vllm.distributed.parallel_state"]
    state.destroy_model_parallel = groups.get("model", lambda: ev.append(
        ("the model's groups destroyed, its tensors on the device:", sum(t.is_cuda for t in [*model.p, *model.b]))))
    state.destroy_distributed_environment = groups.get("world", lambda: ev.append("the world's group destroyed"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: ev.append("device waited for"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: ev.append("cache emptied"))
    return TPL.rank_off(_worker(com=com, eng=SimpleNamespace(model=model))), model


def test_the_groups_a_rank_destroys_are_vllms_own_and_go_once_its_model_is_on_the_host(monkeypatch):
    ev: list = []
    off, _ = _off(monkeypatch, ev, L._Com(1, ev))
    assert ev == [(1, "worker stopped"), "device waited for", "device waited for", ("the model's groups destroyed, its tensors on the device:", 0),
                  "the world's group destroyed", "device waited for", "cache emptied"]
    assert {k: off[k] for k in ("rank", "moved_bytes", "dropped_bytes", "teardown_error")} == {
        "rank": 1, "moved_bytes": (3 << 20) + 16, "dropped_bytes": 0, "teardown_error": None}
    assert sorted(off) == ["dropped_bytes", "moved_bytes", "rank", "teardown_error", "teardown_s", "to_host_s"]
    assert off["to_host_s"] >= 0 and off["teardown_s"] >= 0


def test_a_committer_with_no_worker_to_stop_leaves_its_device_the_same_way(monkeypatch):
    ev: list = []
    off, _ = _off(monkeypatch, ev, SimpleNamespace())
    assert ev[:2] == ["device waited for", "device waited for"] and ev[-1] == "cache emptied" and off["moved_bytes"] == (3 << 20) + 16


def test_a_group_that_cannot_be_destroyed_is_on_the_record_in_300_characters_and_the_cache_is_still_emptied(monkeypatch):
    def stuck() -> None:
        raise RuntimeError("nccl abort: " + "x" * 400)

    ev: list = []
    off, model = _off(monkeypatch, ev, L._Com(1, ev), world=stuck)
    assert off["teardown_error"] == ("RuntimeError('nccl abort: " + "x" * 400)[:300] and len(off["teardown_error"]) == 300
    assert ev[-3:] == [("the model's groups destroyed, its tensors on the device:", 0), "device waited for", "cache emptied"]
    assert not any(t.is_cuda for t in [*model.p, *model.b])


# ------------------------------------------------------------------------------------------------------ a rank's context reset, or not


@pytest.fixture
def rank(monkeypatch):
    """A rank's process: vLLM's stand-in with its `WorkerProc`, the CUDA driver's library opening one driver
    (`test_tp_gpu_lease._Cuda`), and the three things a rank that reset its context does next (the device hidden, the watch, its
    thread) noted on ``ev``, ``watched`` and ``threads``. The patch a rank puts on the worker comes off at the end."""
    ev: list = []
    watched: list = []
    threads: list[tuple[str | None, bool | None]] = []

    class WorkerProc:
        def shutdown(self) -> None:
            raise AssertionError("vLLM's worker teardown")

    class Thread:
        """`threading.Thread` that notes how it was made and runs its target at ``start``."""

        def __init__(self, target, args=(), daemon=None, name=None) -> None:
            self.target, self.args = target, args
            threads.append((name, daemon))

        def start(self) -> None:
            self.target(*self.args)

    fake = fake_vllm.install(monkeypatch)
    fake.set("vllm.v1.executor.multiproc_executor:WorkerProc", WorkerProc)
    cuda = L._Cuda(ev)

    def watch(drv, started) -> None:
        ev.append("watch")
        watched.append((drv, started, WorkerProc.shutdown is TPL._exit))

    for owner, name, value in [(GL, "DRIVER", lambda: cuda), (GL, "_no_device", lambda: ev.append("no device")), (GL, "_watch", watch),
                               (TPL, "threading", SimpleNamespace(Thread=Thread))]:
        monkeypatch.setattr(owner, name, value)
    yield SimpleNamespace(ev=ev, watched=watched, cuda=cuda, threads=threads, worker=WorkerProc, shutdown=WorkerProc.shutdown)
    for p in reversed(hooks.live()):
        if isinstance(p, hooks.Patch) and p.owner is WorkerProc:
            p.uninstall()


def test_a_rank_that_reset_its_context_hides_the_device_replaces_vllms_worker_shutdown_and_watches(rank):
    answer = TPL.rank_reset(_worker(), STARTED)
    assert rank.worker.shutdown is TPL._exit
    # the watch is on the driver that reset the context, started once the device is hidden and the worker's shutdown replaced
    assert rank.ev == ["context reset", "no device", "watch"] and rank.watched == [(rank.cuda, STARTED, True)]
    assert rank.threads == [("gpu-lease-watch", True)]
    assert sorted(answer) == ["rank", "reset_s", "why"] and (answer["rank"], answer["why"]) == (1, None) and answer["reset_s"] >= 0


def _raises(error: BaseException) -> Any:
    def call(*a: Any) -> Any:
        raise error
    return call


def _no_library(cuda: L._Cuda, monkeypatch) -> None:
    monkeypatch.setattr(GL, "DRIVER", _raises(OSError("libcuda.so.1: cannot open shared object file: No such file or directory")))


def _reset_fails(cuda: L._Cuda, monkeypatch) -> None:
    monkeypatch.setattr(cuda, "reset", _raises(RuntimeError("cuDevicePrimaryCtxReset: CUDA driver error 999")))


def _state_unread(cuda: L._Cuda, monkeypatch) -> None:
    monkeypatch.setattr(cuda, "active", _raises(RuntimeError("cuDevicePrimaryCtxGetState: " + "x" * 400)))


def _context_stays(cuda: L._Cuda, monkeypatch) -> None:
    cuda.looks.append([0, 3])


@pytest.mark.parametrize("fail, why, did", [
    (_no_library, "the CUDA context could not be reset: OSError('libcuda.so.1: cannot open shared object file: No such file or directory')", []),
    (_reset_fails, "the CUDA context could not be reset: RuntimeError('cuDevicePrimaryCtxReset: CUDA driver error 999')", []),
    (_state_unread, ("the CUDA context could not be reset: RuntimeError('cuDevicePrimaryCtxGetState: " + "x" * 400)[:300], ["context reset"]),
    (_context_stays, "a CUDA context is still active on device(s) [0, 3] after its reset", ["context reset"]),
], ids=["no driver library", "the reset fails", "its state cannot be read", "a context stays"])
def test_a_rank_whose_context_is_not_gone_says_why_and_is_left_as_it_was(rank, monkeypatch, fail, why, did):
    """It still sees its device, vLLM's own shutdown is still its worker's, and nothing watches it: the lease is kept."""
    fail(rank.cuda, monkeypatch)
    assert TPL.rank_reset(_worker(), STARTED) == {"rank": 1, "why": why} and len(why) <= 300
    assert rank.ev == did and rank.watched == [] and rank.threads == [] and rank.worker.shutdown is rank.shutdown


def test_an_error_the_driver_is_not_known_to_raise_is_not_a_ranks_answer(rank, monkeypatch):
    """`gpu_lease.give_back` keeps the lease on any error of the ranks' reset, by its own name for it."""
    monkeypatch.setattr(rank.cuda, "reset", _raises(ValueError("not the driver's")))
    with pytest.raises(ValueError, match="not the driver's"):
        TPL.rank_reset(_worker(), STARTED)
    assert rank.ev == [] and rank.watched == [] and rank.worker.shutdown is rank.shutdown


class _Stuck(L._Cuda):
    """The driver of every rank, whose reset number ``fail`` (from 0, in rank order) raises ``error``."""

    def __init__(self, ev: list, fail: int, error: BaseException) -> None:
        super().__init__(ev)
        self.fail, self.error, self.resets = fail, error, 0

    def reset(self) -> None:
        self.resets += 1
        if self.resets - 1 == self.fail:
            raise self.error
        super().reset()


@pytest.mark.parametrize("error, kept", [
    (RuntimeError("cuDevicePrimaryCtxReset: CUDA driver error 999"),
     "rank 1: the CUDA context could not be reset: RuntimeError('cuDevicePrimaryCtxReset: CUDA driver error 999')"),
    (ValueError("not the driver's"), "the ranks could not give their GPUs back: ValueError(\"not the driver's\")"),
], ids=["the driver's error", "any other"])
def test_one_rank_that_cannot_reset_its_context_keeps_every_gpu_to_the_exit(lease, monkeypatch, error, kept):
    """Rank 0 has reset and hidden its device by then; the driver's own context is not reset, no lock is let go, and the record
    names the rank that kept the lease beside what each rank did."""
    monkeypatch.setattr(GL, "DRIVER", _Stuck(lease.ev, 1, error))
    L._last_run(lease)
    note = lease.note()
    assert note["kept"] == kept and note["after"] == L.LAST and "released_at" not in note
    assert len(GL._HELD) == 1 and not GL._RELEASED.is_set() and lease.holder.poll() is None and not any(L.free(p) for p in lease.locks)
    assert [e for e in lease.ev if e in ("context reset", "no device")] == ["context reset", "no device"]          # rank 0's alone
    assert lease.watched == [str(lease.started)] and lease.smi == []
    assert L._rpcs(lease.ev)[-2:] == [("rank_reset", {}), ("tp2_commit_finalize", {})]             # the checks still run, on the device
    whys = [r.get("why", "no answer") for r in note["ranks"]]
    assert whys == ([None, kept.removeprefix("rank 1: ")] if isinstance(error, RuntimeError) else ["no answer", "no answer"])


# --------------------------------------------------------------------------------------------------------- the exit after the release


def test_the_exit_writes_out_both_streams_and_then_ends_the_process_with_0(monkeypatch):
    ev: list = []
    monkeypatch.setattr(TPL, "sys", SimpleNamespace(stdout=SimpleNamespace(flush=lambda: ev.append("stdout written out")),
                                                    stderr=SimpleNamespace(flush=lambda: ev.append("stderr written out"))))
    monkeypatch.setattr(TPL, "os", SimpleNamespace(_exit=lambda code: ev.append(("the process ends", code))))
    assert TPL._exit(object()) is None
    assert sorted(ev[:2]) == ["stderr written out", "stdout written out"] and ev[2:] == [("the process ends", 0)]


_CHILD = """
import atexit
import sys

from verity_vllm.pipeline.tp import lease as TPL


class Teardown:
    def __del__(self):
        print("an object's finalizer ran")


atexit.register(lambda: print("the interpreter's exit handlers ran"))
keep = Teardown()
sys.stdout.write("said before the end")
sys.stderr.write("an error before the end")
try:
    TPL._exit(None)
finally:
    print("the caller's cleanup ran")
"""


def test_a_rank_that_ends_at_vllms_shutdown_runs_no_teardown_and_loses_no_output():
    """In a Python of its own, its output a pipe (so buffered until written out): what it said before the end arrives, and neither
    the caller's cleanup, an exit handler nor a finalizer runs, any of which could open a CUDA context."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                       check=False)
    assert (r.returncode, r.stdout, r.stderr) == (0, "said before the end", "an error before the end")
