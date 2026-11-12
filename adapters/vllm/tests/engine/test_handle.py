"""`verity_vllm.engine.handle`: the scheduler, the executor and the model runner of a `verity_vllm.LLM`, a `vllm.LLM` or
its `LLMEngine`, through vLLM's object graph (stand-ins of it here), a refusal naming the engine core when they live
in another process, and the output lag the engine's architecture implies."""
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


@pytest.mark.parametrize("get", [handle.scheduler, handle.model_runner, handle.executor])
def test_a_multiprocess_engine_core_is_refused_by_name(get):
    llm = NS(llm_engine=NS(engine_core=MultiprocClient()))
    with pytest.raises(RuntimeError, match="engine core MultiprocClient.*VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        get(llm)


def test_each_kind_of_engine_reaches_the_executor_whose_collective_rpc_it_can_call():
    found = NS(collective_rpc=lambda method: [method])
    for kind, llm in _engines(NS(model_executor=found)).items():
        assert handle.executor(llm) is found, kind
    assert handle.executor(NS(llm_engine=NS(engine_core=NS(model_executor=found)))) is found


@pytest.mark.parametrize("found", [None, NS(), NS(collective_rpc="not a call")])
def test_an_executor_that_cannot_reach_its_workers_is_refused_by_name(found):
    class Core:
        model_executor = found

    llm = NS(llm_engine=NS(engine_core=Core()))
    with pytest.raises(RuntimeError, match=r"SimpleNamespace: no in-process executor with collective_rpc \(engine core Core\)"):
        handle.executor(llm)


def test_scheduler_or_none_is_the_scheduler_that_holds_its_requests_in_a_dict_else_none():
    sched = NS(requests={})
    for kind, llm in _engines(NS(scheduler=sched)).items():
        assert handle.scheduler_or_none(llm) is sched, kind
    assert handle.scheduler_or_none(NS(llm_engine=NS(engine_core=NS(scheduler=sched)))) is sched
    assert handle.scheduler_or_none(NS()) is None
    assert handle.scheduler_or_none(NS(llm_engine=None)) is None
    assert handle.scheduler_or_none(NS(llm_engine=NS(engine_core=NS(scheduler=NS(requests=[]))))) is None
    assert handle.scheduler_or_none(NS(llm_engine=NS(engine_core=MultiprocClient()))) is None


def test_the_structural_output_lag_is_one_with_an_engine_core_and_zero_with_the_engines_own_executor():
    for kind, llm in _engines(NS()).items():
        assert handle.structural_output_lag(llm) == 1, kind
    v0 = NS(model_executor=object())
    for llm in (v0, NS(llm_engine=v0), NS(vllm=NS(llm_engine=v0))):
        assert handle.structural_output_lag(llm) == 0
    assert handle.structural_output_lag(NS(engine_core=object(), model_executor=object())) == 1


@pytest.mark.parametrize("engine", [NS(), NS(engine_core=None, model_executor=None)])
def test_an_engine_of_no_known_kind_has_no_structural_output_lag(engine):
    assert handle.structural_output_lag(engine) is None
    assert handle.structural_output_lag(NS(llm_engine=engine)) is None
