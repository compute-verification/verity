"""A leased TP Commit (`tp-commit` under ``VERITY_LEASE_STARTED``, `pipeline.gpu_lease`): the driver takes the task's GPUs
(``GPU_LEASE_N``, every one at once: the lease tool grants all or none) right before the target reads the device (``acquire``), and
vLLM spawns the ranks, which see only the leased devices and inherit none of the lease's lock descriptors.  After the last run's
finalize has fixed every rank's root and taken off its hooks, and before any check, the GPUs go back together (``finalize``):

1. every rank says what still needs its device after its root (``rank_kept``); a reason on any rank keeps the lease to the exit;
2. every rank leaves its device (``rank_off``): its committer's worker stopped, the model's parameters and buffers copied to host
   memory (the sampled replay, its weights self-check and the cross-rank embedding check read them there) but for those its weights view
   already holds on the host (`acquire.as_loaded`), which are emptied, its NCCL groups destroyed;
3. every rank resets its CUDA context (``rank_reset``); from then on it sees no device, ends at once if a context reappears, and
   exits without vLLM's worker teardown, which would open one;
4. the driver resets its own context, sees nvidia-smi list nothing on any leased GPU, and lets go of every lock at once.

The binding and manifest coverage, the attribution, the sampled replay and its boundary linkage, the openings and the cross-rank pass
that follow read host memory only, and no rank is reset for another run (``reset``).  A Commit that stops after a failed run gives
the GPUs back the same way there (``stop``).  A rank that keeps, or a step that fails or
leaves a context behind, keeps the lease to the Commit's exit and ``gpu_lease.released`` says why; the holder then ends the lease once
nothing is left on the leased GPUs.  Without a lease ``finalize`` is the ranks' finalize and nothing else changes."""
from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

from verity_vllm.engine import hooks
from verity_vllm.pipeline import gpu_lease as GL

SPAWN = "VLLM_WORKER_MULTIPROC_METHOD"
_TL = SimpleNamespace(start=lambda *a, **k: None, end=lambda *a, **k: None)
_OPENINGS_AFTER = SimpleNamespace(open_after_release=True)    # a TP Commit's openings follow its finalize, so its release
_OFF = threading.Event()                                      # the ranks were asked to leave their devices


def acquire() -> bool:
    """The Commit's GPUs before anything reads the device (`gpu_lease.acquire`); vLLM then spawns its ranks (a forked rank would hold
    the lease's lock descriptors past the release) with the environment the lease set its devices in (the CLI's).  Returns whether
    the lease is held."""
    GL.acquire()
    if GL._HELD:
        GL._HELD[0][4][SPAWN] = "spawn"
    return bool(GL._HELD)


def rank_kept(worker: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): why its checks after its root still need its device, or None."""
    st = worker._tp2_commit
    return {"rank": int(st["rank"]), "why": GL._kept_by(_OPENINGS_AFTER, st["com"])}


def _parallel_state() -> Any:
    from vllm.distributed import parallel_state

    return parallel_state


def _replaced(st: dict[str, Any]) -> set[int]:
    """The model's tensors that the rank's weights (``st["weights"]``, a `protocol_options.WeightsView` when `acquire.as_loaded` or an
    option replaced some) read from elsewhere: every reader after the release reads those names from the view's host tensors."""
    plain = getattr(st.get("weights"), "plaintext", None) or {}
    model = st["eng"].model
    return {id(t) for n, t in [*model.named_parameters(), *model.named_buffers()] if n in plain}


def _to_host(model: Any, drop: set[int] = frozenset()) -> tuple[int, int]:
    """Every parameter and buffer of ``model`` on the device moved to host memory in place (the same tensor objects, so any reader
    holding them reads the host copy), except those in ``drop``, which nothing reads after the release and which are emptied (a
    TP rank's MXFP4 Marlin experts, the size of its model, already on the host as loaded); returns the bytes moved and dropped."""
    seen: set[int] = set()
    moved = dropped = 0
    for t in [*model.parameters(), *model.buffers()]:
        if id(t) in seen or not t.is_cuda:
            continue
        seen.add(id(t))
        n = t.numel() * t.element_size()
        if id(t) in drop:
            t.data = t.data.new_empty(0, device="cpu")
            dropped += n
        else:
            t.data = t.data.cpu()
            moved += n
    return moved, dropped


def rank_off(worker: Any) -> dict[str, Any]:
    """On a rank: off its device, which its context still holds: the committer's worker thread stopped (its exit after the reset would
    open a context), the model to host memory but for what its weights view replaced (`_replaced`), the distributed groups (NCCL
    communicators, their threads) destroyed, the allocator's cache emptied."""
    import gc

    import torch

    st = worker._tp2_commit
    t0 = time.perf_counter()
    getattr(st["com"], "_stop_worker", lambda: None)()
    torch.cuda.synchronize()
    moved, dropped = _to_host(st["eng"].model, _replaced(st))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    err = None
    try:
        ps = _parallel_state()
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
    except Exception as e:  # noqa: BLE001 -- the reset frees the device either way; the error goes on the record
        err = repr(e)[:300]
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"rank": int(st["rank"]), "moved_bytes": moved, "dropped_bytes": dropped, "to_host_s": round(t1 - t0, 3),
            "teardown_s": round(time.perf_counter() - t1, 3), "teardown_error": err}


def _exit(_proc: Any) -> None:
    """vLLM's worker shutdown on a rank that gave its GPU back (its parent's exit, a signal): the process ends here, before the model
    runner's and the distributed groups' teardown, which would open a CUDA context."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _worker_proc() -> Any:
    from vllm.v1.executor.multiproc_executor import WorkerProc

    return WorkerProc


def rank_reset(worker: Any, started: str) -> dict[str, Any]:
    """On a rank: its CUDA context reset (every allocation in it gone, as at the process's exit); then no device for the rest of the
    process, a watch that ends it if a context reappears (noted beside the lease's start marker ``started``), and an exit instead of
    vLLM's teardown."""
    rank = int(worker._tp2_commit["rank"])
    t0 = time.perf_counter()
    try:
        drv = GL.DRIVER()
        drv.reset()
        left = drv.active()
    except (OSError, RuntimeError) as e:
        return {"rank": rank, "why": f"the CUDA context could not be reset: {e!r}"[:300]}
    if left:
        return {"rank": rank, "why": f"a CUDA context is still active on device(s) {left} after its reset"}
    GL._no_device()
    hooks.patch(_worker_proc(), "shutdown", _exit)
    threading.Thread(target=GL._watch, args=(drv, started), daemon=True, name="gpu-lease-watch").start()
    return {"rank": rank, "why": None, "reset_s": round(time.perf_counter() - t0, 3)}


class _Driver:
    """The driver process's CUDA driver (`gpu_lease.DRIVER`), read only once the process has initialised CUDA: before that it has no
    context (its ranks hold the model), and the watch after the release looks again each time.  Its reset resets an active context
    only."""

    def active(self) -> list[int]:
        return GL.DRIVER().active() if GL.cuda_initialised() else []

    def reset(self) -> None:
        if self.active():
            GL.DRIVER().reset()


class _Ranks:
    """A TP Commit's holders (`gpu_lease.give_back`): its ranks, each on its own leased GPU, over ``llm``'s `collective_rpc`."""

    failed = "the ranks could not give their GPUs back"
    fails: tuple[type[BaseException], ...] = (Exception,)

    def __init__(self, llm: Any, rpc_timeout: float, started: str) -> None:
        self.llm, self.rpc_timeout, self.started = llm, rpc_timeout, started
        self.ranks: dict[int, dict[str, Any]] = {}

    def kept(self) -> str | None:
        kept = [k for k in self.llm.collective_rpc(rank_kept, timeout=self.rpc_timeout) if k.get("why")]
        return "; ".join(f"rank {k['rank']}: {k['why']}" for k in kept) or None

    def off(self) -> str | None:
        _OFF.set()
        try:
            self.ranks = {int(o["rank"]): o for o in self.llm.collective_rpc(rank_off, timeout=self.rpc_timeout)}
        except Exception as e:  # noqa: BLE001 -- a rank that could not leave its device keeps the lease to the exit, named
            return f"{self.failed}: {e!r}"[:300]
        return None

    def reset(self) -> str | None:
        for o in self.llm.collective_rpc(rank_reset, timeout=self.rpc_timeout, args=(self.started,)):
            self.ranks[int(o["rank"])].update(o)
        return next((f"rank {r}: {o['why']}" for r, o in sorted(self.ranks.items()) if o.get("why")), None)

    def driver(self) -> _Driver:
        return _Driver()

    def error(self) -> str | None:
        return "; ".join(f"rank {r}: {o['teardown_error']}" for r, o in sorted(self.ranks.items()) if o.get("teardown_error")) or None

    def record(self) -> dict[str, Any]:
        return {"ranks": [self.ranks[r] for r in sorted(self.ranks)]}

    def done(self) -> str:
        moved, dropped = (sum(int(o.get(k) or 0) for o in self.ranks.values()) for k in ("moved_bytes", "dropped_bytes"))
        return (f"{len(self.ranks)} ranks off their GPUs ({moved >> 20} MiB of weights to host memory, {dropped >> 20} MiB their weights "
                f"view holds on the host emptied, NCCL groups destroyed), every CUDA context reset")


def release(llm: Any, rpc_timeout: float, after: str, *, wait: Callable[[float], Any] = time.sleep) -> bool:
    """Every rank's GPU back, and the driver's, then the lease let go; or, when a rank keeps or a step fails, the lease kept to the
    exit (`gpu_lease.give_back` over the ranks).  ``after``: the run whose finalize was the Commit's last device work."""
    return GL.give_back(_TL, _Ranks(llm, rpc_timeout, GL._HELD[0][3]), after=after, wait=wait)


def finalize(llm: Any, rpc_timeout: float, after: str, *, last: bool, wait: Callable[[float], Any] = time.sleep,
             defer: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The ranks' finalize of instrumented run ``after`` (`TP2CaptureWorkerExtension.tp2_commit_finalize`).  Under a lease, when it is
    the Commit's last device work (``last``), every rank's root is fixed first, then the GPUs go back (``release``), then the checks.
    ``defer`` ({"out", "pair"}): the ranks write their replay bundles with the checks, in place of their sampled replay."""
    kw = {"defer": defer} if defer else None
    if not (last and GL._HELD):
        return llm.collective_rpc("tp2_commit_finalize", timeout=rpc_timeout, **({"kwargs": kw} if kw else {}))
    llm.collective_rpc("tp2_commit_finalize", timeout=rpc_timeout, kwargs={"checks": False})
    release(llm, rpc_timeout, after, wait=wait)
    return llm.collective_rpc("tp2_commit_finalize", timeout=rpc_timeout, **({"kwargs": kw} if kw else {}))


def stop(llm: Any, rpc_timeout: float, after: str, *, wait: Callable[[float], Any] = time.sleep) -> None:
    """The Commit stops after run ``after`` (a failed run, `tp.commit`'s fail-fast), which was its last device work: under a lease
    whose ranks are still on their devices, the GPUs go back now (``release``), as after the last run's finalize."""
    if GL._HELD and not _OFF.is_set():
        release(llm, rpc_timeout, after, wait=wait)


def reset(llm: Any, rpc_timeout: float) -> None:
    """The ranks' committers recycled for the next run; not once the ranks were asked off their devices."""
    if not _OFF.is_set():
        llm.collective_rpc("tp2_commit_reset", timeout=rpc_timeout)


__all__ = ["acquire", "finalize", "release", "reset", "stop", "rank_kept", "rank_off", "rank_reset"]
