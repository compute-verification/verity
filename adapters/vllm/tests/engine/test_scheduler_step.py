"""`engine.scheduler_step` on the stand-in scheduler (`fake_scheduler`): which methods are patched and while what, the order
parts run in, what a part may do with the call it is handed, parts added and removed between calls and during one, and
the two parts of the engine package on one scheduler (an engine class's admission and the step trace)."""
from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, drive
from verity_vllm.engine import hooks, scheduler_step
from verity_vllm.engine import step_trace as ST
from verity_vllm.engine.class_admission import Admission
from verity_vllm.engine.runner_step import Rank
from verity_vllm.errors import Refused

FOUR = ["add_request", "schedule", "update_from_output", "finish_requests"]


def _req(rid: str, prompt: int = 3, outputs: tuple[int, ...] = (5, 2), max_tokens: int = 4) -> FakeRequest:
    return FakeRequest(rid, list(range(10, 10 + prompt)), list(outputs), max_tokens,
                       sampling_params=SimpleNamespace(temperature=0.0, max_tokens=max_tokens, ignore_eos=False, seed=None))


def wrapped(sched: Any) -> list[str]:
    """The scheduler's methods the registry has patched now, in the order the patches went on."""
    return [e.name for e in hooks.live(scheduler_step.TAG) if isinstance(e, hooks.Patch) and e.owner is sched]


class Recorder(scheduler_step.Part):
    """Acts at all four methods: logs entering and leaving each under its name."""

    def __init__(self, name: str, rank: Rank, log: list[str]) -> None:
        self.name, self.rank, self.log = name, rank, log
        self.then: dict[str, Callable[[], None]] = {}

    def _around(self, what: str, call: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        self.log.append(f"{self.name} enter {what}")
        self.then.get(what, lambda: None)()
        out = call(*a, **kw)
        self.log.append(f"{self.name} leave {what}")
        return out

    def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
        return self._around("add_request", call, request, *a, **kw)

    def around_schedule(self, call: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        return self._around("schedule", call, *a, **kw)

    def around_update_from_output(self, call: Callable[..., Any], scheduler_output: Any, model_runner_output: Any,
                                  *a: Any, **kw: Any) -> Any:
        return self._around("update_from_output", call, scheduler_output, model_runner_output, *a, **kw)

    def around_finish_requests(self, call: Callable[..., Any], request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
        return self._around("finish_requests", call, request_ids, finished_status, *a, **kw)


class Adds(scheduler_step.Part):
    """Acts at ``add_request`` alone, as an engine class's admission does."""

    rank = Rank.ADMIT

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
        self.log.append(f"adds {request.request_id}")
        return call(request, *a, **kw)


def test_parts_run_by_rank_nested_and_one_rank_in_the_order_added() -> None:
    log: list[str] = []
    sched = FakeScheduler()
    made = {name: Recorder(name, rank, log) for name, rank in (
        ("observe", Rank.OBSERVE), ("commit", Rank.COMMIT), ("pass", Rank.PASS), ("admit", Rank.ADMIT), ("plan", Rank.PLAN),
        ("commit2", Rank.COMMIT))}
    removals = [scheduler_step.add(sched, part) for part in made.values()]
    order = ["admit", "plan", "commit", "commit2", "observe", "pass"]
    assert scheduler_step.parts(sched) == tuple(made[n] for n in order)
    sched.add_request(_req("a"))
    so = sched.schedule()
    sched.update_from_output(so, sched.execute(so))
    assert sched.finish_requests("a", "FINISHED_ABORTED") == [sched.requests["a"]]
    assert log == [line for what in FOUR
                   for line in (*(f"{n} enter {what}" for n in order), *(f"{n} leave {what}" for n in reversed(order)))]
    for remove in removals:
        remove()


def test_a_method_is_patched_only_while_a_part_acts_at_it() -> None:
    log: list[str] = []
    sched, before = FakeScheduler(), hooks.live()
    assert scheduler_step.parts(sched) == () and wrapped(sched) == []
    adds, all_four = Adds(log), Recorder("observe", Rank.OBSERVE, log)
    off_adds = scheduler_step.add(sched, adds)
    assert wrapped(sched) == ["add_request"] and [n for n in FOUR if n in vars(sched)] == ["add_request"]
    off_four = scheduler_step.add(sched, all_four)
    assert wrapped(sched) == FOUR and scheduler_step.parts(sched) == (adds, all_four)
    off_four()
    off_four()                                                        # idempotent
    assert wrapped(sched) == ["add_request"] and [n for n in FOUR if n in vars(sched)] == ["add_request"]
    off_four = scheduler_step.add(sched, all_four)
    off_adds()                                                        # the first added goes first: `add_request` keeps its patch
    assert wrapped(sched) == FOUR and scheduler_step.parts(sched) == (all_four,)
    del log[:]
    sched.add_request(_req("a"))
    assert log == ["observe enter add_request", "observe leave add_request"]
    off_four()
    assert scheduler_step.parts(sched) == () and hooks.live() == before and not [n for n in FOUR if n in vars(sched)]
    off_again = scheduler_step.add(sched, adds)                       # and on again
    off_adds()                                                        # an old removal does not take off the part's new adding
    assert scheduler_step.parts(sched) == (adds,)
    off_again()
    assert hooks.live() == before


def test_a_patch_put_on_the_schedulers_class_later_is_called_where_no_part_acts() -> None:
    """The store's binder wraps ``schedule`` and ``update_from_output`` on vLLM's ``Scheduler`` class, and a protocol
    installed again on a built engine installs it after the engine class's admission is on."""
    log: list[str] = []

    class Scheduler(FakeScheduler):
        pass

    def on_schedule(inner: Callable[..., Any]) -> Callable[..., Any]:
        def schedule(s: Any, *a: Any, **kw: Any) -> Any:
            log.append("binder schedule")
            return inner(s, *a, **kw)
        return schedule

    sched = Scheduler()
    remove = scheduler_step.add(sched, Adds(log))
    with hooks.Hooks(tag="binder") as binder:
        binder.wrap(Scheduler, "schedule", on_schedule)
        sched.add_request(_req("a"))
        sched.schedule()
    assert log == ["adds a", "binder schedule"]
    remove()


def test_a_part_may_hand_the_call_other_arguments_and_they_reach_the_parts_inside_and_vllm() -> None:
    seen: list[Any] = []

    class Renames(scheduler_step.Part):
        rank = Rank.ADMIT

        def around_finish_requests(self, call: Callable[..., Any], request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
            return call([f"{r}-0" for r in request_ids], finished_status, *a, **kw)

    class Sees(scheduler_step.Part):
        rank = Rank.OBSERVE

        def around_finish_requests(self, call: Callable[..., Any], request_ids: Any, finished_status: Any, *a: Any, **kw: Any) -> Any:
            seen.append(list(request_ids))
            return call(request_ids, finished_status, *a, **kw)

    sched = FakeScheduler()
    removals = [scheduler_step.add(sched, Sees()), scheduler_step.add(sched, Renames())]
    sched.add_request(_req("a-0"))
    assert [q.request_id for q in sched.finish_requests(["a"], "FINISHED_ABORTED")] == ["a-0"] and seen == [["a-0"]]
    for remove in removals:
        remove()


def test_an_override_that_defers_to_its_base_passes_the_call_on_with_what_it_was_handed() -> None:
    log: list[str] = []

    class Defers(Recorder):
        def _around(self, what: str, call: Callable[..., Any], *a: Any, **kw: Any) -> Any:
            self.log.append(f"{self.name} at {what}")
            return getattr(scheduler_step.Part, f"around_{what}")(self, call, *a, **kw)

    sched, plain = FakeScheduler(), FakeScheduler()
    removals = [scheduler_step.add(sched, Defers("outer", Rank.ADMIT, log)), scheduler_step.add(sched, Recorder("inner", Rank.PASS, log))]
    for s in (sched, plain):
        s.add_request(_req("a"))
        so = s.schedule()
        s.update_from_output(so, s.execute(so))
    finished = sched.finish_requests("a", "FINISHED_ABORTED")
    assert [q.request_id for q in finished] == ["a"] and finished == plain.finish_requests("a", "FINISHED_ABORTED")
    assert log == [line for what in FOUR for line in (f"outer at {what}", f"inner enter {what}", f"inner leave {what}")]
    arrivals = {0: ["b"], 1: ["c"]}
    steps = [drive(s, {at: [_req(r) for r in rids] for at, rids in arrivals.items()}) for s in (sched, plain)]
    assert steps[0] == steps[1] > 0 and sched.requests == plain.requests
    for remove in removals:
        remove()


def test_a_part_may_call_vllm_later_or_not_at_all() -> None:
    log: list[str] = []
    held: list[Callable[[], Any]] = []

    class Gate(scheduler_step.Part):
        rank = Rank.ADMIT

        def around_add_request(self, call: Callable[..., Any], request: Any, *a: Any, **kw: Any) -> Any:
            if request.request_id == "never":
                raise Refused("not this one", what="the gate")
            held.append(lambda: call(request, *a, **kw))
            return None

    sched = FakeScheduler()
    removals = [scheduler_step.add(sched, Gate()), scheduler_step.add(sched, Recorder("observe", Rank.OBSERVE, log))]
    sched.add_request(_req("a"))
    with pytest.raises(Refused, match="the gate refused: not this one"):
        sched.add_request(_req("never"))
    assert sched.requests == {} and log == []                         # nothing inside the gate has run
    held.pop()()
    assert list(sched.requests) == ["a"] and log == ["observe enter add_request", "observe leave add_request"]
    for remove in removals:
        remove()


def test_a_part_added_during_a_call_joins_at_the_next_and_one_removed_is_not_entered_after() -> None:
    log: list[str] = []
    sched = FakeScheduler()
    commit, late, gone = Recorder("commit", Rank.COMMIT, log), Recorder("admit", Rank.ADMIT, log), Recorder("pass", Rank.PASS, log)
    removals = {"commit": scheduler_step.add(sched, commit), "pass": scheduler_step.add(sched, gone)}

    def swap() -> None:
        removals["admit"] = scheduler_step.add(sched, late)
        removals.pop("pass")()

    commit.then["schedule"] = swap
    sched.schedule()
    assert log == ["commit enter schedule", "commit leave schedule"]
    commit.then.clear()
    del log[:]
    sched.schedule()
    assert log == ["admit enter schedule", "commit enter schedule", "commit leave schedule", "admit leave schedule"]
    for remove in removals.values():
        remove()


def test_the_last_part_removed_during_a_call_takes_the_patches_off_and_the_call_ends_in_vllm() -> None:
    log: list[str] = []
    sched, before = FakeScheduler(), hooks.live()
    part = Recorder("commit", Rank.COMMIT, log)
    part.then["add_request"] = scheduler_step.add(sched, part)
    sched.add_request(_req("a"))
    assert log == ["commit enter add_request", "commit leave add_request"] and list(sched.requests) == ["a"]
    assert hooks.live() == before and not [n for n in FOUR if n in vars(sched)]
    sched.add_request(_req("b"))
    assert len(log) == 2 and list(sched.requests) == ["a", "b"]


def test_a_removal_is_refused_with_nothing_changed_while_a_later_patch_covers_a_patch_it_would_take_off() -> None:
    log: list[str] = []
    sched, before = FakeScheduler(), hooks.live()
    adds, all_four = Adds(log), Recorder("observe", Rank.OBSERVE, log)
    off_adds, off_four = scheduler_step.add(sched, adds), scheduler_step.add(sched, all_four)
    cover = hooks.wrap(sched, "finish_requests", lambda inner: lambda *a, **kw: inner(*a, **kw))
    with pytest.raises(RuntimeError, match=r"refused to uninstall Hooks\(tag='scheduler-step'\): the patch of a FakeScheduler"
                                           r"\.finish_requests .* was installed on its patch of a FakeScheduler\.finish_requests "
                                           r".*nothing was uninstalled"):
        off_four()                                                    # `hooks`' own refusal; `schedule` could come off, and stays
    assert scheduler_step.parts(sched) == (adds, all_four) and sorted(wrapped(sched)) == sorted(FOUR)
    sched.schedule()
    assert log == ["observe enter schedule", "observe leave schedule"]
    off_adds()                                                        # takes no patch off: nothing to refuse
    assert scheduler_step.parts(sched) == (all_four,)
    cover.uninstall()
    off_four()
    assert scheduler_step.parts(sched) == () and hooks.live() == before and not [n for n in FOUR if n in vars(sched)]


def test_removals_run_as_one_take_every_part_off_or_none() -> None:
    log: list[str] = []
    one, two, before = FakeScheduler(), FakeScheduler(), hooks.live()
    first, second, adds = Recorder("first", Rank.OBSERVE, log), Recorder("second", Rank.OBSERVE, log), Adds(log)
    both = [scheduler_step.add(one, first), scheduler_step.add(one, second), scheduler_step.add(two, first)]
    off_adds = scheduler_step.add(one, adds)
    cover = hooks.wrap(one, "schedule", lambda inner: lambda *a, **kw: inner(*a, **kw))
    both[1]()                                                         # alone it takes no patch off: `first` acts at all four
    assert scheduler_step.parts(one) == (adds, first)
    both[1] = scheduler_step.add(one, second)
    with pytest.raises(RuntimeError, match=r"refused to uninstall Hooks\(tag='scheduler-step'\): the patch of a FakeScheduler"
                                           r"\.schedule .*nothing was uninstalled"):
        scheduler_step.remove(*both)                                  # together they would take `schedule`'s patch off `one`
    assert scheduler_step.parts(one) == (adds, first, second) and scheduler_step.parts(two) == (first,)
    assert sorted(wrapped(one)) == sorted(FOUR) and wrapped(two) == FOUR
    cover.uninstall()
    scheduler_step.remove(*both)
    assert scheduler_step.parts(one) == (adds,) and wrapped(one) == ["add_request"]
    assert scheduler_step.parts(two) == () and not wrapped(two)
    again = scheduler_step.add(two, second)                           # a scheduler whose parts all went takes parts again
    scheduler_step.remove(*both, off_adds)                            # the removals that ran are skipped
    assert scheduler_step.parts(one) == () and scheduler_step.parts(two) == (second,) and wrapped(two) == FOUR
    again()
    assert hooks.live() == before and not [n for n in FOUR if n in vars(one) or n in vars(two)]


def test_a_scheduler_without_a_method_a_part_acts_at_is_refused_with_no_patch_left_on() -> None:
    bare = SimpleNamespace(add_request=lambda request: None)
    own, before = bare.add_request, hooks.live()
    with pytest.raises(AttributeError, match="schedule"):
        scheduler_step.add(bare, Recorder("observe", Rank.OBSERVE, []))
    assert hooks.live() == before and scheduler_step.parts(bare) == () and bare.add_request is own
    remove = scheduler_step.add(bare, Adds([]))                       # a part that acts at what it has is taken
    assert wrapped(bare) == ["add_request"]
    remove()
    assert hooks.live() == before


def test_a_scheduler_whose_parts_all_went_is_held_no_longer() -> None:
    sched = FakeScheduler()
    scheduler_step.add(sched, Adds([]))()
    gone = weakref.ref(sched)
    del sched
    gc.collect()
    assert gone() is None


def test_each_scheduler_has_its_own_parts() -> None:
    log: list[str] = []
    one, two = FakeScheduler(), FakeScheduler()
    part = Adds(log)
    off_one, off_two = scheduler_step.add(one, part), scheduler_step.add(two, part)
    one.add_request(_req("a"))
    off_one()
    one.add_request(_req("b"))
    two.add_request(_req("c"))
    assert log == ["adds a", "adds c"] and scheduler_step.parts(one) == () and scheduler_step.parts(two) == (part,)
    off_two()


def test_add_refuses_a_part_with_no_rank_and_a_part_that_is_already_on() -> None:
    sched, before = FakeScheduler(), hooks.live()
    with pytest.raises(TypeError, match="sets no rank"):
        scheduler_step.add(sched, scheduler_step.Part())
    assert hooks.live() == before
    part = Adds([])
    remove = scheduler_step.add(sched, part)
    with pytest.raises(ValueError, match="already on"):
        scheduler_step.add(sched, part)
    assert scheduler_step.parts(sched) == (part,)
    remove()
    assert hooks.live() == before


# --- the engine package's two parts ------------------------------------------------------------------------------------

def _engine(sched: Any) -> Any:
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=sched))))


class Limited(FakeScheduler):
    """The stand-in with the context limit an engine class's admission reads."""

    max_model_len = 16


def _limited() -> Limited:
    return Limited(budget=8, max_seqs=4)


def ranks(sched: Any) -> list[Rank]:
    return [part.rank for part in scheduler_step.parts(sched)]


def test_an_admission_alone_patches_add_request_alone_and_a_trace_all_four() -> None:
    sched, before = _limited(), hooks.live()
    adm = Admission("G").install(_engine(sched))
    assert wrapped(sched) == ["add_request"] and ranks(sched) == [Rank.ADMIT]
    adm.uninstall()
    trace = ST.StepTrace().install(sched)
    assert wrapped(sched) == FOUR and ranks(sched) == [Rank.OBSERVE]
    trace.uninstall()
    assert hooks.live() == before and not [n for n in FOUR if n in vars(sched)]


@pytest.mark.parametrize("first", ["admission", "trace"])
def test_an_admission_and_a_trace_share_four_patches_and_the_admission_is_outside_whichever_went_on_first(first: str) -> None:
    sched, before = _limited(), hooks.live()
    if first == "admission":
        adm = Admission("G").install(_engine(sched))
        trace = ST.StepTrace().install(sched)
    else:
        trace = ST.StepTrace().install(sched)
        adm = Admission("G").install(_engine(sched))
    assert wrapped(sched) == FOUR and ranks(sched) == [Rank.ADMIT, Rank.OBSERVE]
    alone = _limited()
    plain = ST.StepTrace().install(alone)
    for s in (sched, alone):
        drive(s, {0: [_req("a", 3, (5, 6, 2), 8)], 2: [_req("b", 12, (7, 7, 7), 3)]})
    assert trace.records == plain.records and len(trace.records) > 4
    with pytest.raises(Refused, match="request long: a prompt of 17 tokens, over the 16 a class-G window admits"):
        sched.add_request(_req("long", 17))
    assert trace.records == plain.records and "long" not in sched.requests       # refused before the trace saw it
    with pytest.raises(Refused, match="request long: a prompt of 17 tokens"):
        sched.add_request(_req("c"))                                  # sticky
    assert trace.records == plain.records
    plain.uninstall()
    if first == "admission":                    # either may go first
        adm.uninstall()
        assert wrapped(sched) == FOUR and ranks(sched) == [Rank.OBSERVE]
        trace.uninstall()
    else:
        trace.uninstall()
        assert wrapped(sched) == ["add_request"] and ranks(sched) == [Rank.ADMIT]
        adm.uninstall()
    assert hooks.live() == before and not [n for n in FOUR if n in vars(sched)]


def test_a_trace_or_an_admission_installed_more_than_once_is_refused_whole() -> None:
    """An uninstall that `hooks` refuses leaves the component as it was, wherever it is on: a trace installed twice on a
    scheduler still writes each record twice, one on two schedulers is still on both, an admission still has both parts."""
    def cover(sched: Any, name: str) -> hooks.Patch:
        return hooks.wrap(sched, name, lambda inner: lambda *a, **kw: inner(*a, **kw))

    def arrivals(trace: ST.StepTrace, sched: Any, rid: str) -> int:
        n = len(trace.records)
        sched.add_request(_req(rid))
        return sum(r["kind"] == "arrive" for r in trace.records[n:])

    one, two, before = _limited(), _limited(), hooks.live()
    twice = ST.StepTrace().install(one).install(one)
    assert arrivals(twice, one, "a") == 2
    over = cover(one, "schedule")
    with pytest.raises(RuntimeError, match="refused to uninstall .*nothing was uninstalled"):
        twice.uninstall()
    assert ranks(one) == [Rank.OBSERVE, Rank.OBSERVE] and arrivals(twice, one, "b") == 2
    over.uninstall()
    twice.uninstall()
    assert hooks.live() == before

    on_two = ST.StepTrace().install(one).install(two)
    over = cover(one, "finish_requests")
    with pytest.raises(RuntimeError, match="refused to uninstall .*nothing was uninstalled"):
        on_two.uninstall()
    assert wrapped(one) == FOUR and wrapped(two) == FOUR
    over.uninstall()
    on_two.uninstall()
    assert hooks.live() == before

    adm = Admission("G").install(_engine(one)).install(_engine(one))
    over = cover(one, "add_request")
    with pytest.raises(RuntimeError, match="refused to uninstall .*nothing was uninstalled"):
        adm.uninstall()
    assert ranks(one) == [Rank.ADMIT, Rank.ADMIT]
    over.uninstall()
    adm.uninstall()
    assert hooks.live() == before and not [n for n in FOUR if n in vars(one) or n in vars(two)]
