"""`engine.handle`: the scheduler and the model runner of a `verity_vllm.LLM`, a `vllm.LLM` or its `LLMEngine`, through
vLLM's object graph (stand-ins of it here), and a refusal naming the engine core when they live in another process."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine import handle

SCHED, RUNNER = object(), object()


class MultiprocClient:
    """An engine core that is a client of another process: no scheduler, no executor."""


def _engines(inner: object) -> dict[str, object]:
    engine = NS(engine_core=NS(engine_core=inner))
    vllm_llm = NS(llm_engine=engine)
    return {"LLMEngine": engine, "vllm.LLM": vllm_llm, "verity_vllm.LLM": NS(vllm=vllm_llm)}


@pytest.mark.parametrize("kind", ["LLMEngine", "vllm.LLM", "verity_vllm.LLM"])
def test_each_kind_of_engine_reaches_its_scheduler_and_model_runner(kind):
    worker = NS(worker=NS(model_runner=RUNNER))
    llm = _engines(NS(scheduler=SCHED, model_executor=NS(driver_worker=worker)))[kind]
    assert handle.scheduler(llm) is SCHED
    assert handle.model_runner(llm) is RUNNER


def test_a_core_without_the_inner_wrapper_and_a_worker_without_the_wrapper_resolve_too():
    llm = NS(llm_engine=NS(engine_core=NS(scheduler=SCHED, model_executor=NS(driver_worker=NS(model_runner=RUNNER)))))
    assert handle.scheduler(llm) is SCHED
    assert handle.model_runner(llm) is RUNNER


@pytest.mark.parametrize("get", [handle.scheduler, handle.model_runner])
def test_a_multiprocess_engine_core_is_refused_by_name(get):
    llm = NS(llm_engine=NS(engine_core=MultiprocClient()))
    with pytest.raises(RuntimeError, match="engine core MultiprocClient.*VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        get(llm)
