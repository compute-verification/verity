"""`engine.step_trace` on a stand-in scheduler (`fake_scheduler`): what it records of each step, and that it leaves the scheduler as
it found it."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from verity.protocols.certifier.verifier import replay_step as RS

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


@pytest.mark.parametrize("resend_resumed", [False, True])
def test_a_preemption_and_its_resume_are_recorded(tmp_path, resend_resumed):
    """The same record from a scheduler that names a resumed request among its cached requests (the pin) and from one
    that sends it again among the new ones and names none as resumed (vLLM 0.34): there it is no second first schedule."""
    sched = FakeScheduler(budget=16, max_seqs=4, preempt_at={1: "a"}, resend_resumed=resend_resumed)
    trace = ST.StepTrace().install(sched)
    drive(sched, {0: [_req("a", 4, [5, 6, 7, 8], 4), _req("b", 4, [5, 6, 2], 4)]})
    steps = ST.steps(trace.records)
    assert steps[0]["new"] == ["a", "b"] and steps[0]["resumed"] == []
    assert steps[1]["preempted"] == ["a"] and steps[1]["rows"] == [["b", 4, 1]]
    assert steps[2]["resumed"] == ["a"] and steps[2]["rows"] == [["b", 5, 1], ["a", 0, 5]]
    assert [s["new"] for s in steps[1:]] == [[]] * (len(steps) - 1) and [s["resumed"] for s in steps[3:]] == [[]] * (len(steps) - 3)


def test_a_resumed_request_sent_again_in_full_is_no_cache_hit_of_a_new_one_and_an_aborted_one_is_forgotten(tmp_path):
    """Under vLLM 0.34's scheduler: a resumed request that starts from cached blocks is recorded by its row's start, as the
    pin's is, and never under ``cache_hit``, which is a new request's; and a preempted request that is aborted before it
    runs again does not make a later request of its id a resume."""
    sched = FakeScheduler(budget=16, max_seqs=4, preempt_at={2: "a"}, prefix_block=4, resend_resumed=True)
    trace = ST.StepTrace().install(sched)
    drive(sched, {0: [_req("a", 9, [5, 6, 7, 8], 4), _req("b", 4, [5, 6, 7, 2], 4)]})
    steps = ST.steps(trace.records)
    back = next(s for s in steps if s["resumed"])
    assert back["resumed"] == ["a"] and back["new"] == [] and back["cache_hit"] == {}
    assert [row for row in back["rows"] if row[0] == "a"][0][1] > 0, "the resumed request starts from its cached blocks"

    assert ST.step_geometry(SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="x", num_computed_tokens=8)]),
                            preempted={"x"}) == ({"x": 8}, [], ["x"], {})
    sched = FakeScheduler(budget=16, max_seqs=4, preempt_at={1: "a"}, resend_resumed=True)
    trace = ST.StepTrace().install(sched)
    sched.add_request(_req("a", 4, [5, 6, 7, 8], 4))
    sched.add_request(_req("b", 4, [5, 6, 2], 4))
    for _ in range(2):
        so = sched.schedule()
        sched.update_from_output(so, sched.execute(so))
    assert ST.steps(trace.records)[1]["preempted"] == ["a"] and trace._preempted == {"a"}
    sched.finish_requests("a", "aborted")
    assert trace._preempted == set()


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
        ST.StepTrace().install(SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace())))


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


def test_an_arrival_after_an_empty_schedule_says_how_many():
    """A replay puts such a request back after an empty schedule or into an idle engine, so the step in flight has finished its
    requests and cached their blocks first; an arrival right after a step, or before any, says nothing more than its step."""
    class Sched:
        def __init__(self):
            self.requests, self.tokens = {}, 0

        def add_request(self, request):
            self.requests[request.request_id] = request

        def schedule(self):
            return SimpleNamespace(total_num_scheduled_tokens=self.tokens)

        def update_from_output(self, so, mro):
            return {}

        def finish_requests(self, request_ids, finished_status):
            return []

    sched = Sched()
    trace = ST.StepTrace().install(sched)

    def arrive(rid: str, schedules: tuple[int, ...] = ()) -> None:
        for sched.tokens in schedules:
            sched.schedule()
        sched.add_request(SimpleNamespace(request_id=rid, prompt_token_ids=[1, 2], sampling_params=None))

    arrive("a")
    arrive("b", (3, 0, 0))                      # a step, then two schedules that held every request back
    arrive("c", (2,))
    sched.requests.clear()
    arrive("d", (0,))                           # the engine drained: the schedule that held its last request back still counts
    assert [(r["request"], r["before_step"], r.get("after_empty")) for r in trace.records if r["kind"] == "arrive"] == [
        ("a", 0, None), ("b", 1, 2), ("c", 2, None), ("d", 2, 1)]


def test_a_request_added_again_under_an_id_the_scheduler_holds_is_one_arrival():
    sched = FakeScheduler()
    trace = ST.StepTrace().install(sched)
    sched.add_request(_req("a", 3, [5, 2], 4))
    sched.add_request(_req("a", 3, [5, 2], 4))                  # held: no second record
    sched.add_request(_req("b", 2, [2], 1))
    del sched.requests["a"]
    sched.add_request(_req("a", 3, [5, 2], 4))                  # no longer held: it arrives again
    assert [r["request"] for r in trace.records if r["kind"] == "arrive"] == ["a", "b", "a"]
    trace.uninstall()


def test_the_header_says_where_the_launch_left_the_context_limit_to_vllm():
    """vLLM keeps the launch's -1 beside the limit it fitted to the KV pool (its model config's ``original_max_model_len``).
    A replay's engine is then launched at -1 on the header's block count, so its model is built at the length the served one
    was; a launch that stated a length, and a vLLM without the entry, write a header as before."""
    sched = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
                            cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=64, cache_dtype="auto"), max_model_len=1008,
                            vllm_config=SimpleNamespace(speculative_config=None, model_config=SimpleNamespace(
                                max_model_len=1008, original_max_model_len=-1)))
    stated = {"scheduler": "SimpleNamespace", "max_num_batched_tokens": 2048, "max_num_seqs": 4, "block_size": 16,
              "num_gpu_blocks": 64, "max_model_len": 1008}
    assert ST.scheduler_config(sched) == {**stated, "max_model_len_auto": True}
    for launch in (None, 1008, 1048576):
        sched.vllm_config.model_config.original_max_model_len = launch
        assert ST.scheduler_config(sched) == stated
    del sched.vllm_config.model_config.original_max_model_len
    assert ST.scheduler_config(sched) == stated


class _Every(SimpleNamespace):
    """A stand-in that has every attribute: the ones it is given, and 1 for any other name."""

    def __getattr__(self, name):
        return 1


def test_every_key_the_header_can_hold_is_on_one_of_the_replay_steps_lists():
    """The replay step's binding refuses an engine key it does not know (`replay_step.statics`), and a committing serve
    hands it this header's config at start. So a key the recorder starts to write has to be placed on one of the
    binding's four lists in the same change, or that serve stops at start. The stand-in has every attribute, so a name
    the recorder starts to pick is in its dict; a key written on a condition needs the condition met here."""
    sched = _Every(scheduler_config=_Every(), cache_config=_Every(cache_dtype="fp8"), vllm_config=_Every(
        model_config=_Every(original_max_model_len=-1), speculative_config=_Every()))
    widest = ST.scheduler_config(sched)
    assert {"scheduler", "max_model_len", "max_model_len_auto", "kv_cache_dtype", "speculative", "policy"} <= set(widest)
    assert set(widest["speculative"]) == set(ST.SPECULATIVE)
    placed = {*RS.ENGINE_READ, *RS.ENGINE_DRAFT, *RS.ENGINE_HELD, *RS.ENGINE_INERT}
    assert sorted(set(widest) - placed) == [], (
        "step_trace.scheduler_config writes a key replay_step.statics refuses. Put it on one of replay_step's lists: "
        "ENGINE_READ with a static if a scheduler step depends on it, ENGINE_INERT with why if none does")


def test_the_header_holds_the_cache_format_and_speculative_settings_an_engine_was_built_with():
    """A replay's engine takes them as its ``kv_cache_dtype`` and ``speculative_config``; an engine with vLLM's default cache
    format and no draft writes neither entry, so its header is what it was."""
    sched = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
                            cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=100, cache_dtype="auto"),
                            max_model_len=512, vllm_config=SimpleNamespace(speculative_config=None))
    assert ST.scheduler_config(sched) == {"scheduler": "SimpleNamespace", "max_num_batched_tokens": 2048, "max_num_seqs": 4,
                                          "block_size": 16, "num_gpu_blocks": 100, "max_model_len": 512}
    sched.cache_config.cache_dtype = "fp8_ds_mla"
    assert ST.scheduler_config(sched)["kv_cache_dtype"] == "fp8_ds_mla"
    sched.vllm_config.speculative_config = SimpleNamespace(
        method="mtp", num_speculative_tokens=3, model="/a/checkpoint", draft_sample_method="greedy",
        rejection_sample_method="synthetic", synthetic_acceptance_rates=[0.8, 0.6, 0.4], synthetic_acceptance_length=None,
        index_share_for_mtp_iteration=None, num_speculative_tokens_per_batch_size=[(1, 8, 3)], disable_eagle_block_drop=False,
        draft_model_config=object())
    assert ST.scheduler_config(sched)["speculative"] == {
        "method": "mtp", "num_speculative_tokens": 3, "num_speculative_tokens_per_batch_size": [[1, 8, 3]],
        "draft_sample_method": "greedy", "rejection_sample_method": "synthetic", "synthetic_acceptance_rates": [0.8, 0.6, 0.4],
        "disable_eagle_block_drop": False}
