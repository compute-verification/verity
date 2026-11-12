"""A built engine's vLLM objects: the one module that knows vLLM's object graph (``llm_engine.engine_core[.engine_core]``,
its executor and the uniproc executor's driver worker).

Each accessor takes a :class:`verity_vllm.LLM`, a ``vllm.LLM`` or its ``LLMEngine``, and raises ``RuntimeError`` naming
the multiprocess engine core when the object lives in another process (build with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``,
`engine.env.ENGINE`); `scheduler_or_none` and `executor_or_none` return None there instead, `core_or_none` returns the
client the engine holds, and `structural_output_lag` says only which of vLLM's two architectures the engine is.
Attribute reads only; nothing is patched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import vllm
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class Wrapper(Protocol):
        """:class:`verity_vllm.LLM`, structurally (`engine` sits below it in the layers)."""

        @property
        def vllm(self) -> vllm.LLM: ...

    type Engine = Wrapper | vllm.LLM | LLMEngine

__all__ = ["core_or_none", "engine_of", "executor", "executor_or_none", "model_runner", "scheduler", "scheduler_or_none",
           "structural_output_lag"]


def engine_of(llm: Engine) -> LLMEngine:
    """The engine's ``LLMEngine``, which takes its requests and steps it."""
    eng = getattr(llm, "vllm", llm)
    return getattr(eng, "llm_engine", eng)


def _engine_core(llm: Engine) -> Any:
    core = getattr(engine_of(llm), "engine_core", None)
    return getattr(core, "engine_core", None) or core


def core_or_none(llm: Engine) -> Any:
    """The engine core as the ``LLMEngine`` holds it (a client, where the core runs in another process), else None: for a
    reader that records which it is."""
    return getattr(engine_of(llm), "engine_core", None)


def structural_output_lag(llm: Engine) -> int | None:
    """The output lag the engine's architecture implies: 1 for vLLM V1 (an ``engine_core``), 0 for V0 (the engine holds
    its ``model_executor``, no core), None when the engine is of no known kind (a caller then keeps the lag it measured)."""
    eng = engine_of(llm)
    if getattr(eng, "engine_core", None) is not None:
        return 1
    if getattr(eng, "model_executor", None) is not None:
        return 0
    return None


def scheduler(llm: Engine) -> Scheduler:
    """The engine's in-process V1 scheduler."""
    core = _engine_core(llm)
    sched = getattr(core, "scheduler", None)
    if sched is None:
        raise RuntimeError(f"{type(llm).__name__}: no in-process V1 scheduler (engine core {type(core).__name__}); "
                           "build with VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine.env.ENGINE)")
    return sched


def scheduler_or_none(llm: Engine) -> Scheduler | None:
    """The engine's in-process V1 scheduler where it holds its requests in a dict (``requests``), else None (a multiprocess
    engine core, a stub): for a reader that goes without it."""
    sched = getattr(_engine_core(llm), "scheduler", None)
    return sched if isinstance(getattr(sched, "requests", None), dict) else None


def executor(llm: Engine) -> Any:
    """The engine's in-process executor, whose ``collective_rpc`` reaches every worker."""
    core = _engine_core(llm)
    found = getattr(core, "model_executor", None)
    if found is None or not callable(getattr(found, "collective_rpc", None)):
        raise RuntimeError(f"{type(llm).__name__}: no in-process executor with collective_rpc (engine core "
                           f"{type(core).__name__}); build with VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine.env.ENGINE)")
    return found


def executor_or_none(llm: Engine) -> Any:
    """The engine's in-process executor, whatever it can call, else None (a multiprocess engine core, a stub): for a
    reader that records which it is."""
    return getattr(_engine_core(llm), "model_executor", None)


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
