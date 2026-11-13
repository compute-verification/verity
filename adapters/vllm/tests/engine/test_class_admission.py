"""`engine.class_admission`: a served class-G window refuses a prompt over its engine's max_model_len (the launch's, whatever
length that is, or the one vLLM fitted to the KV pool) before the scheduler holds it, and a Commit window's close while a request
is in flight; a refusal is sticky, and uninstalling puts the scheduler back.

The scheduler here keeps what the admission reads of vLLM V1's: ``max_model_len`` (the model config's, after the engine's start
fitted it), ``add_request`` and ``requests``, the requests it holds, from which it deletes a request when it frees it
(`Scheduler._free_blocks`)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine.class_admission import Admission
from verity_vllm.errors import Refused


class _Scheduler:
    def __init__(self, max_model_len: object = 131072) -> None:
        self.max_model_len = max_model_len
        self.requests: dict[str, object] = {}

    def add_request(self, request) -> None:
        self.requests[request.request_id] = request

    def free(self, request_id: str) -> None:
        del self.requests[request_id]


def _engine(max_model_len: object = 131072):
    sched = _Scheduler(max_model_len)
    return sched, NS(llm_engine=NS(engine_core=NS(engine_core=NS(scheduler=sched))))


def _request(rid: str, n: int):
    return NS(request_id=rid, num_prompt_tokens=n, prompt_token_ids=list(range(n)))


@pytest.mark.parametrize("limit", [4032, 131072, 401408, 1048576])
def test_class_g_admits_a_prompt_at_its_engines_limit_and_refuses_one_over_it_before_the_scheduler_holds_it(limit):
    """The limit is the built engine's, at any length: from a few thousand tokens, where a small KV pool fitted it, to the
    checkpoint's own 1,048,576."""
    sched, eng = _engine(limit)
    adm = Admission("G")
    assert adm.limit == 0
    assert adm.install(eng).limit == limit
    sched.add_request(_request("a", limit))
    with pytest.raises(Refused, match=rf"engine class G refused: request b: a prompt of {limit + 1} tokens, over the {limit} a class-G "
                                      r"window admits \(its engine's max_model_len\)"):
        sched.add_request(_request("b", limit + 1))
    assert list(sched.requests) == ["a"]


@pytest.mark.parametrize("stated", [None, -1, 0, "auto", True])
def test_an_engine_whose_scheduler_states_no_context_limit_is_refused(stated):
    """-1 is the launch's request for a fit, never a built engine's limit: a scheduler still holding it wasn't fitted."""
    sched, eng = _engine(stated)
    add = sched.add_request
    with pytest.raises(Refused, match=rf"engine class G refused: the engine's scheduler states no context limit \(its max_model_len "
                                      rf"is {stated!r}\)"):
        Admission("G").install(eng)
    assert sched.add_request == add


def test_a_refusal_ends_the_window():
    sched, eng = _engine()
    adm = Admission("G").install(eng)
    with pytest.raises(Refused, match="a prompt of 200000 tokens"):
        sched.add_request(_request("long", 200000))
    for refused in (lambda: sched.add_request(_request("short", 8)), adm.close_window):
        with pytest.raises(Refused, match="request long: a prompt of 200000 tokens"):
            refused()
    assert sched.requests == {}


def test_a_window_closes_once_its_requests_are_freed():
    sched, eng = _engine()
    adm = Admission("G").install(eng)
    adm.close_window()
    sched.add_request(_request("a", 16))
    sched.free("a")
    adm.close_window()
    assert adm.refused is None


def test_a_window_close_with_a_request_in_flight_is_refused():
    sched, eng = _engine()
    adm = Admission("G").install(eng)
    sched.add_request(_request("a", 16))
    sched.add_request(_request("b", 16))
    with pytest.raises(Refused, match=r"2 requests in flight at a Commit window's close, 'a' first: a class-G window closes with "
                                      r"nothing in flight"):
        adm.close_window()
    sched.free("a")
    sched.free("b")
    with pytest.raises(Refused, match="in flight"):
        adm.close_window()


def test_uninstall_puts_the_schedulers_add_request_back():
    sched, eng = _engine()
    adm = Admission("G").install(eng)
    adm.uninstall()
    sched.add_request(_request("long", 200000))
    assert list(sched.requests) == ["long"]


def test_a_class_not_registered_is_refused():
    with pytest.raises(Refused, match=r"engine class 'H' is not registered \(registered: G\)"):
        Admission("H")


def test_an_engine_whose_scheduler_is_in_another_process_is_refused():
    with pytest.raises(RuntimeError, match="no in-process V1 scheduler"):
        Admission("G").install(NS(llm_engine=NS(engine_core=NS())))
