"""`engine.runner_step` on a stand-in runner: which calls are steps, the order parts run in, what an error leaves, and
parts added and removed between steps and during one."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.engine import hooks, runner_step
from verity_vllm.engine.runner_step import Rank, Step


class Runner:
    """vLLM's two methods as far as the wrap reads them: the arguments, what they return and an error on request."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.fails = ""
        self.calls: list[tuple[Any, ...]] = []

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None, dummy_run: bool = False) -> Any:
        self.log.append("vllm execute_model")
        self.calls.append((scheduler_output, intermediate_tensors, dummy_run))
        if self.fails == "execute_model":
            raise OSError("execute_model failed")
        return ("forward", scheduler_output)

    def sample(self, hidden_states: Any, input_batch: Any, grammar_output: Any) -> Any:
        self.log.append("vllm sample")
        if self.fails == "sample":
            raise OSError("sample failed")
        return ("sampled", hidden_states, input_batch, grammar_output)


class Recorder(runner_step.Part):
    """Logs each of its calls under its name, and keeps the steps it was handed."""

    def __init__(self, name: str, rank: Rank, log: list[str]) -> None:
        self.name, self.rank, self.log = name, rank, log
        self.steps: list[Step] = []
        self.then: dict[str, Callable[[], None]] = {}

    def _did(self, what: str, step: Step) -> None:
        self.log.append(f"{self.name} {what}")
        if not any(step is s for s in self.steps):
            self.steps.append(step)
        self.then.get(what, lambda: None)()

    @contextmanager
    def _around(self, what: str, step: Step) -> Iterator[None]:
        self._did(f"enter {what}", step)
        try:
            yield
        finally:
            self._did(f"leave {what}", step)

    def before_forward(self, step: Step) -> None:
        self._did("before_forward", step)

    def around_forward(self, step: Step) -> AbstractContextManager[None]:
        return self._around("forward", step)

    def after_forward(self, step: Step) -> None:
        self._did("after_forward", step)

    def around_sample(self, step: Step) -> AbstractContextManager[None]:
        return self._around("sample", step)

    def after_sample(self, step: Step) -> None:
        self._did("after_sample", step)


def wrapped(runner: Any) -> list[str]:
    """The runner's methods the registry has wrapped now."""
    return [e.name for e in hooks.live(runner_step.TAG) if isinstance(e, hooks.Patch) and e.owner is runner]


def tokens(n: int = 3) -> SimpleNamespace:
    return SimpleNamespace(total_num_scheduled_tokens=n)


def one_step(runner: Runner) -> Any:
    runner.execute_model(tokens())
    return runner.sample("hidden", "batch", None)


def test_parts_run_by_rank_nested_and_one_rank_in_the_order_added() -> None:
    log: list[str] = []
    runner = Runner(log)
    made = {name: Recorder(name, rank, log) for name, rank in (
        ("observe", Rank.OBSERVE), ("commit", Rank.COMMIT), ("pass", Rank.PASS), ("admit", Rank.ADMIT), ("plan", Rank.PLAN),
        ("commit2", Rank.COMMIT))}
    removals = [runner_step.add(runner, part) for part in made.values()]
    order = ["admit", "plan", "commit", "commit2", "observe", "pass"]
    assert runner_step.parts(runner) == tuple(made[n] for n in order)
    assert one_step(runner) == ("sampled", "hidden", "batch", None)
    assert log == [
        *(f"{n} before_forward" for n in order), *(f"{n} enter forward" for n in order), "vllm execute_model",
        *(f"{n} leave forward" for n in reversed(order)), *(f"{n} after_forward" for n in reversed(order)),
        *(f"{n} enter sample" for n in order), "vllm sample",
        *(f"{n} leave sample" for n in reversed(order)), *(f"{n} after_sample" for n in reversed(order))]
    for remove in removals:
        remove()


def test_a_step_carries_the_calls_arguments_and_results() -> None:
    log: list[str] = []
    runner = Runner(log)
    seen: list[tuple[Any, ...]] = []

    class Reader(runner_step.Part):
        rank = Rank.COMMIT

        def before_forward(self, step: Step) -> None:
            seen.append(("before_forward", step.runner, step.scheduler_output, step.forwarded))

        def after_forward(self, step: Step) -> None:
            seen.append(("after_forward", step.forwarded, step.input_batch))

        def around_sample(self, step: Step) -> None:
            seen.append(("around_sample", step.hidden_states, step.input_batch, step.grammar_output, step.sampled))

        def after_sample(self, step: Step) -> None:
            seen.append(("after_sample", step.sampled))

    remove = runner_step.add(runner, Reader())
    so = tokens()
    assert runner.execute_model(so, "intermediate") == ("forward", so) and runner.calls == [(so, "intermediate", False)]
    assert runner.sample("hidden", input_batch="batch", grammar_output="grammar") == ("sampled", "hidden", "batch", "grammar")
    assert seen == [("before_forward", runner, so, None), ("after_forward", ("forward", so), None),
                    ("around_sample", "hidden", "batch", "grammar", None), ("after_sample", ("sampled", "hidden", "batch", "grammar"))]
    remove()


def test_a_dummy_run_a_step_with_no_token_and_a_sample_with_no_step_call_no_part() -> None:
    log: list[str] = []
    runner = Runner(log)
    part = Recorder("commit", Rank.COMMIT, log)
    remove = runner_step.add(runner, part)
    assert runner.execute_model(tokens(), None, True) == ("forward", runner.calls[-1][0])            # the third argument
    assert runner.execute_model(tokens(), dummy_run=True) is not None                                # the keyword
    assert runner.execute_model(tokens(0)) is not None and runner.execute_model(SimpleNamespace()) is not None
    assert runner.sample("hidden", "batch", None) == ("sampled", "hidden", "batch", None)            # no step is open
    assert log == ["vllm execute_model"] * 4 + ["vllm sample"] and part.steps == []
    runner.execute_model(tokens())
    del log[:]
    runner.execute_model(tokens(0))                                                                  # leaves the open step open
    runner.execute_model(tokens(), dummy_run=True)
    runner.sample("hidden", "batch", None)
    assert log == ["vllm execute_model"] * 2 + ["commit enter sample", "vllm sample", "commit leave sample", "commit after_sample"]
    assert len(part.steps) == 1
    del log[:]
    runner.sample("hidden", "batch", None)                                                           # its sample closed the step
    assert log == ["vllm sample"]
    remove()


def test_no_after_runs_when_vllms_call_raises_and_the_step_stays_open() -> None:
    log: list[str] = []
    runner = Runner(log)
    outer, inner = Recorder("admit", Rank.ADMIT, log), Recorder("pass", Rank.PASS, log)
    removals = [runner_step.add(runner, inner), runner_step.add(runner, outer)]
    runner.fails = "execute_model"
    with pytest.raises(OSError, match="execute_model failed"):
        runner.execute_model(tokens())
    assert log == ["admit before_forward", "pass before_forward", "admit enter forward", "pass enter forward", "vllm execute_model",
                   "pass leave forward", "admit leave forward"]
    failed, = outer.steps
    assert failed.forwarded is None
    del log[:]
    runner.fails = "sample"
    with pytest.raises(OSError, match="sample failed"):
        runner.sample("hidden", "batch", None)                       # the failed step is still the open one
    assert log == ["admit enter sample", "pass enter sample", "vllm sample", "pass leave sample", "admit leave sample"]
    assert outer.steps == [failed] and failed.sampled is None
    del log[:]
    runner.fails = ""
    one_step(runner)                                                 # the next execute_model opens a step of its own
    assert len(outer.steps) == 2 and outer.steps[1] is not failed and inner.steps == outer.steps
    assert log[-2:] == ["pass after_sample", "admit after_sample"] and "pass after_forward" in log
    for remove in removals:
        remove()


def test_a_parts_error_goes_to_the_caller_and_the_later_parts_are_not_called() -> None:
    log: list[str] = []
    runner = Runner(log)
    first, second = Recorder("admit", Rank.ADMIT, log), Recorder("commit", Rank.COMMIT, log)
    removals = [runner_step.add(runner, first), runner_step.add(runner, second)]

    def refuse() -> None:
        raise ValueError("refused")

    first.then["before_forward"] = refuse
    with pytest.raises(ValueError, match="refused"):
        runner.execute_model(tokens())
    assert log == ["admit before_forward"]
    del first.then["before_forward"], log[:]
    second.then["after_sample"] = refuse
    with pytest.raises(ValueError, match="refused"):
        one_step(runner)
    assert log[-1] == "commit after_sample" and "admit after_sample" not in log
    del log[:]
    runner.sample("hidden", "batch", None)                           # vLLM's sample had returned: the step is closed
    assert log == ["vllm sample"]
    for remove in removals:
        remove()


def test_an_around_that_swallows_vllms_error_is_refused() -> None:
    runner = Runner([])

    class Swallows(runner_step.Part):
        rank = Rank.OBSERVE

        @contextmanager
        def around_forward(self, step: Step) -> Iterator[None]:
            try:
                yield
            except OSError:
                pass

    remove = runner_step.add(runner, Swallows())
    runner.fails = "execute_model"
    with pytest.raises(RuntimeError, match="around_forward swallowed"):
        runner.execute_model(tokens())
    remove()


def test_the_wraps_go_on_with_the_first_part_and_off_with_the_last_in_any_order() -> None:
    log: list[str] = []
    runner, before = Runner(log), hooks.live()
    assert runner_step.parts(runner) == () and wrapped(runner) == []
    a, b, c = (Recorder(n, r, log) for n, r in (("admit", Rank.ADMIT), ("commit", Rank.COMMIT), ("pass", Rank.PASS)))
    off_b = runner_step.add(runner, b)
    assert wrapped(runner) == ["execute_model", "sample"] and {"execute_model", "sample"} <= set(vars(runner))
    off_a, off_c = runner_step.add(runner, a), runner_step.add(runner, c)
    assert wrapped(runner) == ["execute_model", "sample"] and runner_step.parts(runner) == (a, b, c)
    one_step(runner)
    off_b()                                                           # the first added goes first: not last in, first out
    off_b()                                                           # idempotent
    assert runner_step.parts(runner) == (a, c)
    del log[:]
    one_step(runner)
    assert log and not [line for line in log if line.startswith("commit")]
    off_a()
    assert runner_step.parts(runner) == (c,) and wrapped(runner) == ["execute_model", "sample"]
    off_c()
    assert runner_step.parts(runner) == () and hooks.live() == before
    assert "execute_model" not in vars(runner) and "sample" not in vars(runner)
    del log[:]
    assert one_step(runner) == ("sampled", "hidden", "batch", None) and log == ["vllm execute_model", "vllm sample"]
    off_again = runner_step.add(runner, b)                            # and on again
    off_b()                                                           # an old removal does not take off the part's new adding
    assert runner_step.parts(runner) == (b,)
    off_again()
    assert hooks.live() == before


def test_a_part_added_during_a_step_joins_at_the_next_and_one_removed_is_not_called_again() -> None:
    log: list[str] = []
    runner = Runner(log)
    commit, late, gone = Recorder("commit", Rank.COMMIT, log), Recorder("admit", Rank.ADMIT, log), Recorder("pass", Rank.PASS, log)
    removals = {"commit": runner_step.add(runner, commit), "pass": runner_step.add(runner, gone)}
    commit.then["before_forward"] = lambda: removals.update(admit=runner_step.add(runner, late))
    commit.then["enter forward"] = lambda: removals.pop("pass")()
    one_step(runner)
    assert log == ["commit before_forward", "pass before_forward", "commit enter forward", "vllm execute_model",
                   "commit leave forward", "commit after_forward", "commit enter sample", "vllm sample", "commit leave sample",
                   "commit after_sample"]
    commit.then.clear()
    del log[:]
    one_step(runner)
    assert log[:2] == ["admit before_forward", "commit before_forward"] and not [line for line in log if line.startswith("pass")]
    assert late.steps == commit.steps[1:] and gone.steps == commit.steps[:1]
    for remove in removals.values():
        remove()


def test_the_last_part_removed_during_a_step_takes_the_wraps_off() -> None:
    log: list[str] = []
    runner, before = Runner(log), hooks.live()
    part = Recorder("commit", Rank.COMMIT, log)
    remove = runner_step.add(runner, part)
    part.then["enter forward"] = remove
    runner.execute_model(tokens())
    assert log == ["commit before_forward", "commit enter forward", "vllm execute_model", "commit leave forward"]
    assert hooks.live() == before and "execute_model" not in vars(runner)
    del log[:]
    runner.sample("hidden", "batch", None)
    assert log == ["vllm sample"]


def test_removing_the_last_part_is_refused_while_a_later_patch_covers_the_wraps() -> None:
    log: list[str] = []
    runner, before = Runner(log), hooks.live()
    part, other = Recorder("commit", Rank.COMMIT, log), Recorder("pass", Rank.PASS, log)
    remove, remove_other = runner_step.add(runner, part), runner_step.add(runner, other)
    cover = hooks.wrap(runner, "sample", lambda inner: lambda *a, **kw: inner(*a, **kw))
    remove_other()                                                    # not the last: the wraps stay, nothing to refuse
    with pytest.raises(RuntimeError, match="refused to uninstall"):
        remove()
    assert runner_step.parts(runner) == (part,) and wrapped(runner) == ["execute_model", "sample"]
    one_step(runner)
    assert log[-1] == "commit after_sample"
    cover.uninstall()
    remove()
    assert runner_step.parts(runner) == () and hooks.live() == before and "sample" not in vars(runner)


def test_a_runner_with_no_sample_has_execute_model_wrapped_alone() -> None:
    log: list[str] = []

    class Forward:
        def execute_model(self, scheduler_output: Any) -> str:
            log.append("vllm execute_model")
            return "forward"

    runner, part = Forward(), Recorder("commit", Rank.COMMIT, log)
    remove = runner_step.add(runner, part)
    assert wrapped(runner) == ["execute_model"]
    assert runner.execute_model(tokens()) == "forward" and runner.execute_model(tokens()) == "forward"
    assert log.count("commit before_forward") == 2 and len(part.steps) == 2
    remove()
    assert "execute_model" not in vars(runner)


def test_each_runner_has_its_own_parts_and_steps() -> None:
    log: list[str] = []
    one, two = Runner(log), Runner(log)
    part = Recorder("commit", Rank.COMMIT, log)
    off_one, off_two = runner_step.add(one, part), runner_step.add(two, part)
    one.execute_model(tokens())
    two.sample("hidden", "batch", None)                               # no step is open on the second runner
    assert log[-1] == "vllm sample" and [s.runner for s in part.steps] == [one]
    off_one()
    assert runner_step.parts(one) == () and runner_step.parts(two) == (part,) and "execute_model" not in vars(one)
    off_two()


def test_add_refuses_a_part_with_no_rank_and_a_part_that_is_already_on() -> None:
    runner, before = Runner([]), hooks.live()
    with pytest.raises(TypeError, match="sets no rank"):
        runner_step.add(runner, runner_step.Part())
    assert hooks.live() == before and "execute_model" not in vars(runner)
    part = Recorder("commit", Rank.COMMIT, [])
    remove = runner_step.add(runner, part)
    with pytest.raises(ValueError, match="already on"):
        runner_step.add(runner, part)
    assert runner_step.parts(runner) == (part,)
    remove()
    assert hooks.live() == before
