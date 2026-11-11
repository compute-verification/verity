"""The CUDA communicator's one seam for its all-reduce (`AllReduceSeam`): one replacement of the routes vLLM picks.

vLLM's tensor-parallel all-reduce (``tensor_model_parallel_all_reduce``, through ``GroupCoordinator.all_reduce`` and, under
torch.compile, its ``vllm.all_reduce`` custom op) reaches ``CudaCommunicator.all_reduce(input_)`` of the group's
communicator (``vllm.distributed.device_communicators.cuda_communicator`` at the pin), which picks one of its routes
(symmetric memory, FlashInfer, the custom all-reduce, the quick all-reduce, NCCL) for each tensor.

A plugin (a ``vllm.general_plugins`` entry point, whose ``register`` runs in each worker process) adds the replacement by
name, one of `ALL_REDUCES`, to `SEAM`, this process's seam, with a function that makes it for a communicator, or returns
None where that communicator keeps vLLM's routes (`verity_vllm.engine.makers.Makers`). One is added per process at most:
two replacements of one reduction don't compose. Adding it wraps ``CudaCommunicator.all_reduce`` once, through one
`verity_vllm.engine.hooks.Hooks`. On a communicator with a replacement, each call returns ``replacement(comm, input_,
vllm)``, where ``vllm(tensor)`` is vLLM's own all-reduce on that communicator, for the tensors it leaves on vLLM's routes;
on one without, vLLM's own. A wrap above it (of ``tensor_model_parallel_all_reduce`` or of the group coordinator) still sees
every call. This module imports vLLM only when it wraps.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from verity_vllm.engine.hooks import Hooks
from verity_vllm.engine.makers import Makers

__all__ = ["ALL_REDUCES", "SEAM", "AllReduce", "AllReduceSeam"]

#: the replacements' names, of which one is added per process at most
ALL_REDUCES = ("tree",)


class AllReduce(Protocol):
    def __call__(self, comm: Any, input_: Any, vllm: Callable[[Any], Any], /) -> Any:
        """``input_`` all-reduced on ``comm``: computed here, or ``vllm(input_)``, vLLM's own route for it."""


class AllReduceSeam:
    """One process's seam: the replacement added by name, made per communicator, and the wrap that runs it.
    ``communicator`` is the class it wraps (vLLM's ``CudaCommunicator`` unless given)."""

    def __init__(self, *, communicator: type | None = None) -> None:
        self._cls = communicator
        self._replacements: Makers[AllReduce] = Makers(ALL_REDUCES, "engine.collectives.all_reduce's ALL_REDUCES")
        self._hooks: Hooks | None = None

    def on_all_reduce(self, name: str, make: Callable[[Any], AllReduce | None]) -> None:
        """Add the replacement ``name``, one of `ALL_REDUCES`: ``make(comm)`` makes it for a communicator, or returns
        None."""
        if added := self._replacements.added:
            raise ValueError(f"{name!r} can't replace this process's all-reduce: {added[0]!r} already does, and two "
                             "replacements of one reduction don't compose")
        self._replacements.add(name, make)
        self._install()

    def uninstall(self) -> None:
        """Take the wrap off and forget the replacement and every communicator."""
        if self._hooks is not None:
            self._hooks.uninstall()
        self._hooks = None
        self._replacements.clear()

    def _install(self) -> None:
        if self._hooks is not None:
            return
        cls = self._cls
        if cls is None:
            from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator as cls
        hooks = Hooks(tag="all-reduce")
        hooks.wrap(cls, "all_reduce", self._all_reduce)
        self._hooks = hooks

    def _all_reduce(self, orig: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        def all_reduce(comm: Any, input_: Any) -> Any:
            replacements = self._replacements.made(comm)
            if not replacements:
                return orig(comm, input_)
            return replacements[0](comm, input_, lambda tensor: orig(comm, tensor))
        return all_reduce


#: this process's all-reduce seam, which a plugin adds its replacement to
SEAM = AllReduceSeam()
