"""What the windowed committer (`native_collect`) does at an engine step, through the runner's own ``execute_model`` and
``sample``, on the CPU as in `test_native_collect_plan_cache`.

The committer begins a step when vLLM's ``execute_model`` runs the model and ends it when vLLM's ``sample`` returns, after
hashing, in this order: the sampled token ids, what each source of the sample's inputs commits (its ``on_sample``) and, where
the batch sends explicit seeds and no source owns them, each row's seed. A greedy workload reaches none of the seeds' lines,
so their order is held here: every step's root is compared with the root of the same tensors packed in that order into one
block. Also held: a dummy run and a call with no scheduled token begin no step, the next step ends a step whose ``sample``
never came, and vLLM's ``execute_model`` runs on the committer's high-priority stream, fenced both ways.

The committer patches neither method itself: it puts one part on the runner (`engine.runner_step`) at ``install`` and takes
it off first at ``uninstall``, which is refused whole while another component's later patch covers the runner's two wraps.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.commit.served_record import SAMPLED, SAMPLER_INPUTS, SamplerInputsSource
from verity_vllm.engine import hooks, runner_step

from tests.commit import test_native_collect_plan_cache as PC

SEED, POSITION = SAMPLER_INPUTS
#: the sampler's seed of each request state slot, and the state slot of each row of the batch
SEEDS, SLOTS = [11, 22, 33], [2, 0]
ROW_SEEDS = torch.tensor([33, 11], dtype=torch.int64)
EXTRA = "served.sampler/extra"

#: a host tensor that reports a CUDA device, as the sampler's output does where it is served
OnDevice: Any = type("OnDevice", (torch.Tensor,), {"device": property(lambda self: torch.device("cuda"))})


class SamplingStates:
    """vLLM's sampling states as far as the committer reads them: each state slot's seed, and whether a row sent one."""

    def __init__(self) -> None:
        self.seeds = SimpleNamespace(gpu=torch.tensor(SEEDS, dtype=torch.int64))
        self.explicit = True
        self.asked: list[list[int]] = []

    def any_explicit_seed(self, idx_mapping_np: Any) -> bool:
        self.asked.append([int(i) for i in idx_mapping_np])
        return self.explicit


class Runner(PC._Runner):
    """The model runner with vLLM's ``sample``: ``execute_model`` acquires the step's module tensors, as the wrapped forwards do,
    and ``sample`` returns the sampler's output on the device with vLLM's two counts."""

    def __init__(self) -> None:
        super().__init__()
        self.states = SamplingStates()
        self.sampler = SimpleNamespace(sampling_states=self.states)
        self.log: list[str] = []
        self.fails = False
        self.sampled = torch.empty(0, dtype=torch.int32)

    def execute_model(self, scheduler_output: Any, *a: Any, **kw: Any) -> None:
        self.log.append("execute_model")
        if self.fails:
            raise OSError("execute_model failed")
        super().execute_model(scheduler_output, *a, **kw)

    def sample(self, hidden_states: Any, input_batch: Any, grammar_output: Any) -> tuple[SimpleNamespace, int, int]:
        self.log.append("sample")
        return SimpleNamespace(sampled_token_ids=self.sampled.as_subclass(OnDevice)), int(self.sampled.shape[0]), 0


class Extra:
    """A source of the sample's inputs that owns no name of the committer's: at each sample it commits one tensor of its own."""

    short_name = "extra"

    def __init__(self, com: NC.NativeCollectCommitter) -> None:
        self.com = com
        self.uninstalls = 0

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        self.uninstalls += 1

    def on_step_begin(self) -> None:
        pass

    @staticmethod
    def tensor(n: int) -> torch.Tensor:
        return torch.arange(3 * n, dtype=torch.float32).reshape(n, 3)

    def on_sample(self, input_batch: Any, n_sampled: int, job: Any) -> None:
        self.com._acquire(EXTRA, self.tensor(n_sampled), job)


def committer(monkeypatch: pytest.MonkeyPatch) -> tuple[NC.NativeCollectCommitter, Runner]:
    monkeypatch.setattr(PC, "DTYPES", (*PC.DTYPES, torch.int64))          # the seeds and the positions are i64
    com = PC.committer(monkeypatch)
    runner = com.runner = Runner()
    runner.committer = com
    return com, runner


def batch(so: PC._Sched, kv: tuple[int, ...]) -> SimpleNamespace:
    """The step's input batch: each request decodes one token at its KV length, its row in the state slot `SLOTS` gives it."""
    n = len(kv)
    return SimpleNamespace(req_ids=so.req_ids, num_reqs=n, num_scheduled_tokens=None, idx_mapping=torch.tensor(SLOTS[:n], dtype=torch.int32),
                           idx_mapping_np=np.array(SLOTS[:n]), positions=torch.tensor(kv), logits_indices=torch.arange(n),
                           query_start_loc_np=np.arange(n + 1), num_computed_tokens_np=np.array(kv), prefill_len_np=np.full(n, 8))


def forward(runner: Runner, kv: tuple[int, ...]) -> tuple[list[tuple[str, torch.Tensor]], SimpleNamespace]:
    """vLLM's ``execute_model`` for one decode step; returns the module tensors it committed and the step's input batch."""
    so = PC._Sched(kv)
    *modules, (_name, runner.sampled) = PC.step_tensors(kv)
    runner.tensors = modules
    runner.input_batch = batch(so, kv)
    runner.execute_model(so)
    return modules, runner.input_batch


def step(runner: Runner, kv: tuple[int, ...]) -> list[tuple[str, torch.Tensor]]:
    """One engine step as vLLM runs it; returns the tensors the committer hashes without its sources, in their order."""
    modules, input_batch = forward(runner, kv)
    runner.sample(None, input_batch, None)
    return [*modules, (SAMPLED, runner.sampled)]


def commit(com: NC.NativeCollectCommitter, steps: int) -> tuple[list[bytes], dict[int, bytes]]:
    """The worker's side of the run's steps, then the run's end; returns each step's root and salt key."""
    for _ in range(steps):
        com._commit_step(com._q.get_nowait())
    rc = com.finalize()
    com.uninstall()
    assert com.errors == [], com.errors[:3]
    return [s.root for s in rc.steps], dict(com.row_keys)


def one_block(tensors: list[list[tuple[str, torch.Tensor]]], keys: dict[int, bytes]) -> list[bytes]:
    return [PC.one_block_root(i, t, keys[i]) for i, t in enumerate(tensors)]


SCHED = PC.schedule(40, 2)


@pytest.mark.parametrize("source, explicit, seeds_hashed", [
    (None, True, 2), (None, False, 0), (Extra, True, 2), (Extra, False, 0), (SamplerInputsSource, True, 0)],
    ids=["explicit seeds", "no explicit seed", "a source, then the seeds", "a source alone", "a source that owns the seed"])
def test_a_sample_hashes_the_sampled_ids_then_each_sources_inputs_then_the_seeds(monkeypatch, source, explicit, seeds_hashed):
    com, runner = committer(monkeypatch)
    runner.states.explicit = explicit
    if source:
        com.attach_source(source(com))
    com.install()
    want = []
    for kv in SCHED:
        tensors = step(runner, kv)
        if source is Extra:
            tensors.append((EXTRA, Extra.tensor(len(kv))))
        if source is SamplerInputsSource:
            tensors += [(SEED, ROW_SEEDS), (POSITION, torch.tensor(kv))]
        elif explicit:
            tensors.append((SEED, ROW_SEEDS))
        want.append(tensors)
    roots, keys = commit(com, len(SCHED))
    assert roots == one_block(want, keys)
    assert getattr(com, "seed_bound_steps", 0) == seeds_hashed
    assert runner.states.asked == ([] if source is SamplerInputsSource else [SLOTS] * len(SCHED))


def test_a_dummy_run_and_a_call_with_no_scheduled_token_begin_no_step(monkeypatch):
    com, runner = committer(monkeypatch)
    com.install()
    so = PC._Sched(SCHED[0])
    for call in (lambda: runner.execute_model(so, None, True), lambda: runner.execute_model(so, dummy_run=True),
                 lambda: runner.execute_model(PC._Sched(())), lambda: runner.sample(None, batch(so, SCHED[0]), None)):
        call()
        assert com._cur is None and com._q.empty() and com._step_index == 0 and com._nc.expected == 0
    assert runner.log == ["execute_model"] * 3 + ["sample"]
    want = [step(runner, kv) + [(SEED, ROW_SEEDS)] for kv in SCHED]            # the run's first steps are the ones that ran the model
    roots, keys = commit(com, len(SCHED))
    assert roots == one_block(want, keys)


def test_the_next_step_ends_a_step_whose_sample_never_came(monkeypatch):
    com, runner = committer(monkeypatch)
    com.install()
    first, _input_batch = forward(runner, SCHED[0])
    open_job = com._cur
    assert open_job is not None and not open_job.done_evt.is_set()
    second = step(runner, SCHED[1]) + [(SEED, ROW_SEEDS)]
    assert open_job.done_evt.is_set() and com._cur is None and com._step_index == 2
    roots, keys = commit(com, 2)
    assert roots == one_block([first, second], keys)


class Fence:
    """A CUDA stream that logs which stream it waits for."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    def wait_stream(self, other: Fence) -> None:
        self.log.append(f"{self.name} waits for {other.name}")


def test_vllms_execute_model_runs_on_the_high_priority_stream_fenced_both_ways(monkeypatch):
    com, runner = committer(monkeypatch)
    log = runner.log
    caller, hi = Fence("the caller's stream", log), Fence("the high-priority stream", log)

    @contextmanager
    def entered(stream: Fence) -> Iterator[None]:
        log.append(f"on {stream.name}")
        try:
            yield
        finally:
            log.append(f"off {stream.name}")

    def on_stream(stream: Any) -> AbstractContextManager[None]:
        return entered(stream) if isinstance(stream, Fence) else nullcontext()

    monkeypatch.setattr(PC._Cuda, "current_stream", staticmethod(lambda: caller))
    monkeypatch.setattr(PC._Cuda, "stream", staticmethod(on_stream))
    monkeypatch.setattr(com, "hi_prio", True)
    monkeypatch.setattr(com, "_hi", hi)
    begin = com._begin_step

    def begins() -> None:
        log.append("the step begins")
        begin()

    monkeypatch.setattr(com, "_begin_step", begins)
    com.install()
    step(runner, SCHED[0])
    assert log == ["the step begins", "the high-priority stream waits for the caller's stream", "on the high-priority stream",
                   "execute_model", "off the high-priority stream", "the caller's stream waits for the high-priority stream", "sample"]
    del log[:]
    runner.fails = True
    with pytest.raises(OSError, match="execute_model failed"):
        forward(runner, SCHED[1])
    assert log == ["the step begins", "the high-priority stream waits for the caller's stream", "on the high-priority stream",
                   "execute_model", "off the high-priority stream"]
    assert com._cur is not None                                              # the step stays open
    com.uninstall()
    assert com._cur is None


def patched(runner: Runner, entries: tuple[Any, ...]) -> list[str]:
    """The runner's attributes that ``entries`` patch."""
    return [e.name for e in entries if isinstance(e, hooks.Patch) and e.owner is runner]


def test_install_puts_one_part_on_the_runner_and_uninstall_takes_it_off(monkeypatch):
    com, runner = committer(monkeypatch)

    def gather_batch_req_state(*a: Any, **kw: Any) -> tuple[None, None]:
        return None, None

    runner.gather_batch_req_state = gather_batch_req_state
    before = hooks.live()
    com.install()
    part, = runner_step.parts(runner)
    assert part.rank is runner_step.Rank.COMMIT
    assert patched(runner, hooks.live(runner_step.TAG)) == ["execute_model", "sample"]
    assert patched(runner, com._hooks.entries) == ["gather_batch_req_state"]         # that wrap is still the committer's own
    com.install()
    assert runner_step.parts(runner) == (part,)
    step(runner, SCHED[0])
    com.uninstall()
    assert runner_step.parts(runner) == () and hooks.live() == before
    assert not {"execute_model", "sample"} & set(vars(runner)) and runner.gather_batch_req_state is gather_batch_req_state
    com.uninstall()


def test_uninstall_is_refused_whole_while_a_later_patch_covers_the_runners_wraps(monkeypatch):
    com, runner = committer(monkeypatch)
    source = Extra(com)
    com.attach_source(source)
    com.install()
    cover = hooks.wrap(runner, "execute_model", lambda inner: lambda *a, **kw: inner(*a, **kw))
    with pytest.raises(RuntimeError, match="refused to uninstall"):
        com.uninstall()
    assert com.installed and source.uninstalls == 0 and len(runner_step.parts(runner)) == 1
    want = [step(runner, kv) + [(EXTRA, Extra.tensor(len(kv))), (SEED, ROW_SEEDS)] for kv in SCHED]     # whole: it still commits
    cover.uninstall()
    roots, keys = commit(com, len(SCHED))
    assert roots == one_block(want, keys)
    assert not com.installed and source.uninstalls == 1 and runner_step.parts(runner) == ()
