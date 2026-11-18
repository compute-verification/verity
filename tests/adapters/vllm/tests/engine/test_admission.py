from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity import hooks as H
from verity_vllm import messages
from verity_vllm.engine import admission, scheduler_step

from tests.pipeline.test_serve_admit_tick import Req, Sched


class Outputs(Sched):
    """`Sched` whose `update_from_output` returns one token for each running request, as vLLM's engine core outputs."""

    def schedule(self) -> SimpleNamespace:
        out = super().schedule()
        out.num_scheduled_tokens = {r.request_id: 1 for r in self.running}
        return out

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> dict:
        ran = [r for r in self.running if r.request_id in scheduler_output.num_scheduled_tokens]
        self.update()
        outs = [SimpleNamespace(request_id=r.request_id, new_token_ids=[7],
                                finish_reason=None if r in self.running else "length") for r in ran]
        return {0: SimpleNamespace(outputs=outs)}


def step(sched: Outputs) -> None:
    sched.update_from_output(sched.schedule(), None)


def test_the_hook_emits_each_point_as_a_steps_event():
    ex, seen = messages.Exchange(), []
    ex.subscribe("s", "steps", seen.append)
    sched = Outputs()
    admission.install(sched, ex)
    sched.add_request(Req("a", out=2))
    step(sched)
    step(sched)
    assert seen == [H.InputArrived("a", 0, 0),
                    H.StepBegan(0, ("a",), 1, 0), H.OutputReleased("a", 0, 0), H.StepEnded(0),
                    H.StepBegan(1, ("a",), 1, 0), H.OutputReleased("a", 1, 1), H.OutputsComplete("a", 1), H.StepEnded(1)]


def test_a_gated_request_finished_from_outside_never_reaches_the_scheduler():
    ex = messages.Exchange()
    starts: dict[str, Any] = {}
    gate = SimpleNamespace(add=starts.__setitem__, pending=lambda: bool(starts), ready=lambda due: True,
                           open=lambda why: None)
    ex.request("g", H.StartOnBoundary(gate))
    ended: list = []
    ex.subscribe("s", "steps", lambda e: isinstance(e, H.OutputsComplete) and ended.append(e.input))
    sched = Outputs()
    admission.install(sched, ex)
    sched.add_request(Req("a"))
    assert sched.added == [] and ex.pending()
    sched.finish_requests(["a"], None)
    starts.pop("a")()
    assert sched.added == [] and sched.handed == [[]] and ended == ["a"]


def test_a_held_release_is_refused_until_the_adapter_keeps_tokens_back():
    ex = messages.Exchange()
    ex.request("h", H.HoldRelease("a", until=5))
    sched = Outputs()
    admission.install(sched, ex)
    sched.add_request(Req("a"))
    with pytest.raises(NotImplementedError, match="vllm/adapter-names-no-scheme"):
        step(sched)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def due(active: int = 0, completing: int = 0) -> H.StepDue:
    return H.StepDue(0, active, 0, completing)


def test_window_groups_gather_one_group_and_wait_out_the_window_for_a_later_one():
    clock, started = Clock(), []
    g = admission.WindowGroups(0.01, clock)
    g.add("a", lambda: started.append("a"))
    clock.t = 0.005
    g.add("b", lambda: started.append("b"))
    assert started == ["a", "b"] and not g.ready(due(active=2))
    clock.t = 0.02
    assert g.ready(due(active=2))
    g.add("c", lambda: started.append("c"))
    assert g.ready(due(active=2)) and started == ["a", "b"]
    assert g.ready(due(completing=2)) and started == ["a", "b"]
    assert not g.ready(due()) and started == ["a", "b", "c"]
    clock.t = 0.04
    assert g.ready(due(active=1)) and not g.pending()


def test_window_groups_filed_as_start_on_boundary_start_each_group_at_a_window_through_the_hook():
    clock, ex = Clock(), messages.Exchange()
    ex.request("tie", H.StartOnBoundary(admission.WindowGroups(0.01, clock)))
    sched = Outputs()
    hook = admission.install(sched, ex)
    sched.add_request(Req("a", out=2))
    sched.add_request(Req("b", out=2))
    assert sched.added == ["a", "b"] and not hook.ready()
    clock.t = 0.02
    assert hook.ready()
    step(sched)
    sched.add_request(Req("c", out=1))
    assert sched.added == ["a", "b"] and hook.ready() and hook.pending()
    step(sched)
    assert hook.ready() and sched.added == ["a", "b"]           # a and b complete; the next step frees them
    step(sched)
    assert not hook.ready() and sched.added == ["a", "b", "c"]
    clock.t = 0.04
    assert hook.ready()
    step(sched)
    assert sched.steps == [["a", "b"], ["a", "b"], ["c"]]


def test_a_scheduler_has_one_step_emitter_however_often_the_hook_is_installed():
    ex, seen = messages.Exchange(), []
    ex.subscribe("s", "steps", lambda e: isinstance(e, H.StepBegan) and seen.append(e.step))
    sched = Outputs()
    hook = admission.install(sched, ex)
    assert admission.install(sched, ex) is hook
    assert [p for p in scheduler_step.parts(sched) if isinstance(p, admission.Hook)] == [hook]
    with pytest.raises(RuntimeError, match="has the admission hook already"):
        admission.install(sched, messages.Exchange())
    assert ex.next_step == 0
    sched.add_request(Req("a", out=2))
    step(sched)
    assert ex.next_step == 1
    step(sched)
    assert seen == [0, 1] and ex.step == 1 and ex.next_step == 2
