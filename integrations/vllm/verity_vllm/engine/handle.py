"""A built engine's vLLM objects: the one module that knows vLLM's object graph (``llm_engine.engine_core[.engine_core]``
and the uniproc executor's driver worker).

Each accessor takes a :class:`verity_vllm.LLM`, a ``vllm.LLM`` or its ``LLMEngine``, and raises ``RuntimeError`` naming
the multiprocess engine core when the object lives in another process (build with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``,
`engine.env.ENGINE`).  Attribute reads only; nothing is patched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    import vllm
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class Wrapper(Protocol):
        """:class:`verity_vllm.LLM`, structurally (`engine` sits below it in the layers)."""

        @property
        def vllm(self) -> vllm.LLM: ...

    Engine = Union[Wrapper, vllm.LLM, LLMEngine]

__all__ = ["model_runner", "scheduler"]


def _engine_core(llm: Engine) -> Any:
    eng = getattr(llm, "vllm", llm)
    eng = getattr(eng, "llm_engine", eng)
    core = getattr(eng, "engine_core", None)
    return getattr(core, "engine_core", None) or core


def scheduler(llm: Engine) -> Scheduler:
    """The engine's in-process V1 scheduler."""
    core = _engine_core(llm)
    sched = getattr(core, "scheduler", None)
    if sched is None:
        raise RuntimeError(f"{type(llm).__name__}: no in-process V1 scheduler (engine core {type(core).__name__}); "
                           "build with VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine.env.ENGINE)")
    return sched


def model_runner(llm: Engine) -> GPUModelRunner:
    """The engine's in-process model runner (the uniproc executor's driver worker's)."""
    core = _engine_core(llm)
    worker = getattr(getattr(core, "model_executor", None), "driver_worker", None)
    for attr in ("worker", "model_runner"):
        if hasattr(worker, attr):
            worker = getattr(worker, attr)
            if attr == "model_runner":
                return worker
    raise RuntimeError(f"{type(llm).__name__}: no in-process model runner (engine core {type(core).__name__}); "
                       "build with VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine.env.ENGINE)")
