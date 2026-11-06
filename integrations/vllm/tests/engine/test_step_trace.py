"""`engine.step_trace` on a stand-in scheduler (`fake_scheduler`): what it records of each step, and that it leaves the scheduler as
it found it."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, drive
from verity_vllm.engine import step_trace as ST


def _req(rid: str, prompt: int, outputs: list[int], max_tokens: int, **kw) -> FakeRequest:
    return FakeRequest(rid, list(range(10, 10 + prompt)), outputs, max_tokens,
                       sampling_params=SimpleNamespace(temperature=0.0, max_tokens=max_tokens, ignore_eos=False, seed=None), **kw)


class _Held:
    """A trace file whose writes wait for ``gate`` (then raise ``fail``, if given), refused on the thread that made it: the
    engine's, which must never write."""

    def __init__(self, fh, gate: threading.Event, fail: OSError | None = None) -> None:
        self.fh, self.gate, self.fail, self.engine = fh, gate, fail, threading.current_thread()

    def _wait(self) -> None:
        assert threading.current_thread() is not self.engine, "the engine thread wrote the step trace's file"
        self.gate.wait()
        if self.fail is not None:
            raise self.fail

    def write(self, s: str) -> int:
        self._wait()
        return self.fh.write(s)

    def flush(self) -> None:
        self._wait()
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class _HeldPath:
    def __init__(self, path, gate: threading.Event, fail: OSError | None = None) -> None:
        self.path, self.parent, self.gate, self.fail = path, path.parent, gate, fail

    def open(self, mode: str) -> _Held:
        return _Held(self.path.open(mode), self.gate, self.fail)


def _served(path, **kw) -> tuple[FakeScheduler, ST.StepTrace]:
    trace = ST.StepTrace(path.path if isinstance(path, _HeldPath) else path, **kw)
    if isinstance(path, _HeldPath):
        trace.path = path
    sched = FakeScheduler(budget=8, max_seqs=4)
    return sched, trace.install(sched)


def _plan() -> dict[int, list[FakeRequest]]:
    return {0: [_req("a", 3, [5, 6, 2], 8)], 2: [_req("b", 12, [7, 7, 7], 3)]}


def test_with_a_writer_thread_the_engine_never_writes_the_file_and_it_holds_the_same_bytes(tmp_path):
    """serve-openai's trace (`serve_commit.advice_trace`): while a seal writes its store, an append to the serve dir's disk can
    wait hundreds of ms, and the engine thread made it every step."""
    sched, plain = _served(tmp_path / "plain.jsonl")
    drive(sched, _plan())
    plain.window(0, "close")
    plain.uninstall()
    gate = threading.Event()
    sched, trace = _served(_HeldPath(tmp_path / "written.jsonl", gate), writer=True)
    drive(sched, _plan())                       # the writer is held at the header's write: every record still returns
    trace.window(0, "close")
    assert (tmp_path / "written.jsonl").read_bytes() == b"" and trace.records == plain.records
    gate.set()
    trace.drain()
    assert (tmp_path / "written.jsonl").read_bytes() == (tmp_path / "plain.jsonl").read_bytes()
    trace.uninstall()
    assert ST.steps(ST.load(tmp_path / "written.jsonl")) == ST.steps(plain.records)


def test_a_failed_trace_write_is_raised_on_the_engine_thread_at_its_next_record(tmp_path):
    gate = threading.Event()
    gate.set()
    _, trace = _served(_HeldPath(tmp_path / "t.jsonl", gate, fail=OSError(28, "No space left on device")), writer=True)
    trace.drain()                               # the header's write failed: the writer still drains
    with pytest.raises(OSError, match="No space left on device"):
        trace.window(0, "open")
    trace.uninstall()


def test_a_chunked_prefill_beside_decodes_is_recorded_row_by_row_in_the_runners_order(tmp_path):
    sched = FakeScheduler(budget=8, max_seqs=4)
    trace = ST.StepTrace(tmp_path / "step_trace.jsonl").install(sched)
    drive(sched, {0: [_req("a", 3, [5, 6, 2], 8)], 2: [_req("b", 12, [7, 7, 7], 3)]})
    trace.uninstall()
    recs = ST.load(tmp_path / "step_trace.jsonl")
    assert recs[0]["config"]["max_num_batched_tokens"] == 8 and recs[0]["config"]["scheduler"] == "FakeScheduler"
    steps = ST.steps(recs)
    assert [s["rows"] for s in steps[:5]] == [
        [["a", 0, 3]],                          # a's prompt
        [["a", 3, 1]],                          # a decodes
        [["a", 4, 1], ["b", 0, 7]],             # b arrived during a's decode: its prompt is chunked by the budget left
        [["b", 7, 5]],                          # a sampled its EOS at step 2; b's last chunk
        [["b", 12, 1]],
    ]
    assert all(s["order"] == "runner" for s in steps)
    assert steps[2]["new"] == ["b"] and steps[2]["finished"] == {"a": {"reason": "stop", "stop_reason": None}}
    assert steps[-1]["finished"] == {"b": {"reason": "length", "stop_reason": None}}
    arrive = {r["request"]: r for r in recs if r["kind"] == "arrive"}
    assert (arrive["a"]["before_step"], arrive["b"]["before_step"], arrive["b"]["prompt_len"]) == (0, 2, 12)
    assert arrive["b"]["sampling"] == {"temperature": 0.0, "max_tokens": 3, "ignore_eos": False, "seed": None}
    assert sum(sum(s["emitted"].values()) for s in steps) == 3 + 3


def test_a_preemption_and_its_resume_are_recorded(tmp_path):
    sched = FakeScheduler(budget=16, max_seqs=4, preempt_at={1: "a"})
    trace = ST.StepTrace().install(sched)
    drive(sched, {0: [_req("a", 4, [5, 6, 7, 8], 4), _req("b", 4, [5, 6, 2], 4)]})
    steps = ST.steps(trace.records)
    assert steps[1]["preempted"] == ["a"] and steps[1]["rows"] == [["b", 4, 1]]
    assert steps[2]["resumed"] == ["a"] and steps[2]["rows"] == [["b", 5, 1], ["a", 0, 5]]


def test_an_abort_from_outside_the_scheduler_is_recorded_as_a_finish(tmp_path):
    sched = FakeScheduler(budget=16)
    trace = ST.StepTrace().install(sched)
    drive(sched, {0: [_req("a", 4, [5] * 9, 9)]}, abort_at={3: "a"})
    fin = [r for r in trace.records if r["kind"] == "finish"]
    assert fin == [{"kind": "finish", "request": "a", "after_step": 2, "reason": "abort", "stop_reason": None, "outside": True}]


def test_uninstall_leaves_the_scheduler_as_it_was():
    sched = FakeScheduler()
    before = {n: getattr(sched, n) for n in ("add_request", "schedule", "update_from_output", "finish_requests")}
    ST.StepTrace().install(sched).uninstall()
    assert {n: getattr(sched, n) for n in before} == before
    assert not any(n in vars(sched) for n in before)


def test_a_multiprocess_engine_core_is_refused_by_name():
    with pytest.raises(RuntimeError, match="VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        ST.scheduler_of(SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace())))


def test_a_trace_without_its_header_is_refused(tmp_path):
    (tmp_path / "t.jsonl").write_text('{"kind": "step", "step": 0}\n')
    with pytest.raises(ValueError, match="not a verity-vllm/step-trace/v1 step trace"):
        ST.load(tmp_path / "t.jsonl")


def test_max_model_len_is_the_schedulers_where_its_scheduler_config_no_longer_holds_it():
    """vLLM 0.11 moved ``max_model_len`` off the scheduler config; the served workloads' statics read it."""
    sched = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
                            cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=100), max_model_len=512)
    assert ST.scheduler_config(sched)["max_model_len"] == 512
    sched.scheduler_config.max_model_len = 256
    assert ST.scheduler_config(sched)["max_model_len"] == 256
    del sched.max_model_len
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1024))
    del sched.scheduler_config.max_model_len
    assert ST.scheduler_config(sched)["max_model_len"] == 1024
