"""`verity_vllm.observe.capture`'s steps on a CPU: a `Capture` over a stand-in for the V2 runner and a two-tensor model takes
two requests through a prefill step and a decode step, and the log it writes is read back.

The runner is stood in as far as the capture reads one: `execute_model` leaves the step's input batch in
`execute_model_state`, runs the model's forward and copies its output, `sample` computes the logits and returns the step's
sampled ids, and the request states, the block tables and the KV-cache configuration are the arrays the capture copies its
step facts from.
Two things of the device are stood in too, since a CPU has neither: pinned host memory (`torch.empty(pin_memory=True)`
gives plain memory) and `torch.cuda.Event` (an event that counts what it is asked).  Where a test reads a time the capture
took, the capture's clock is stood in as well: each reading is twice the last, so no two spans between readings are equal."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.observe import capture as C
from verity_vllm.observe import events as E
from verity_vllm.observe.capture import Capture, ObserverConfig
from verity_vllm.observe.log import read_log

BLOCK_SIZE = 2
NUM_BLOCKS = 16
LAYER = "attn"
#: the two requests, as the engine names them (ours, and vLLM's suffix), each with its row of the request states
A, B = "a-0123abcd", "b-89abcdef"
STATE_ROW = {A: 3, B: 1}
PROMPT_LEN = {A: 2, B: 1}
#: the prefill step and the decode step: each request's tokens, the tokens it has in the KV cache before the step, its
#: blocks (a list a KV-cache group where the runner has several), and the id sampled for it
STEPS = [
    {"ids": {A: [3, 5], B: [7]}, "computed": {A: 0, B: 0}, "blocks": {A: [4], B: [6]}, "sampled": {A: 9, B: 4}},
    {"ids": {A: [9], B: [4]}, "computed": {A: 2, B: 1}, "blocks": {A: [4, 9], B: [6]}, "sampled": {A: 2, B: 6}},
]


class _Event:
    """A CUDA event, as far as the capture uses one."""

    made: list[_Event] = []

    def __init__(self) -> None:
        self.recorded = self.synchronized = 0
        _Event.made.append(self)

    def record(self) -> None:
        self.recorded += 1

    def synchronize(self) -> None:
        self.synchronized += 1


class _Attention(torch.nn.Module):
    """The model's one layer, which holds a KV cache as vLLM's attention layers do."""

    def __init__(self) -> None:
        super().__init__()
        self.kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE)

    def forward(self, hidden: Any) -> Any:
        return hidden + 1


class _Model(torch.nn.Module):
    """An embedding table, a buffer and one layer; the forward gathers the rows of the step's ids, and the logits are their
    products with the table."""

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(16, 2))
        self.register_buffer("scale", torch.ones(2))
        self.attn = _Attention()

    def forward(self, ids: Any) -> Any:
        return self.attn(self.table[ids] * self.scale)

    def compute_logits(self, hidden: Any) -> Any:
        return hidden @ self.table.T


def _batch(step: dict, buffers: SimpleNamespace) -> SimpleNamespace:
    """The step's input batch, as the runner prepares it: the requests in the order of `step["ids"]`, their ids and
    positions written into the runner's input buffers."""
    reqs = list(step["ids"])
    counts = [len(step["ids"][r]) for r in reqs]
    ids = [t for r in reqs for t in step["ids"][r]]
    positions = [step["computed"][r] + k for r in reqs for k in range(len(step["ids"][r]))]
    slots = [step["blocks"][r][p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
             for r in reqs for p in range(step["computed"][r], step["computed"][r] + len(step["ids"][r]))]
    buffers.input_ids[: len(ids)] = torch.tensor(ids)
    buffers.positions[: len(ids)] = torch.tensor(positions)
    return SimpleNamespace(
        num_reqs=len(reqs), req_ids=reqs, num_tokens=len(ids), num_tokens_after_padding=len(ids) + 1,
        query_start_loc_np=np.cumsum([0, *counts, 99]), num_computed_tokens_np=np.array([step["computed"][r] for r in reqs] + [99]),
        idx_mapping_np=np.array([STATE_ROW[r] for r in reqs] + [99]),
        input_ids=buffers.input_ids, positions=buffers.positions,
        logits_indices=torch.tensor(np.cumsum(counts) - 1), slots=torch.tensor(slots + [0]))


class _Runner:
    """The V2 runner, as far as the capture reads one.  `prepared` is handed each step's batch before the forward, and
    `sampled` each step's sampler output before `sample` returns it, so a test can change either."""

    def __init__(self, model: _Model, steps: list[dict], groups: tuple[list[str], ...] = ([LAYER],)) -> None:
        self.model, self.steps = model, steps
        self.calls: list[tuple] = []
        self.prepared: Any = lambda batch: None
        self.sampled: Any = lambda output: (output, "the rest")
        self.input_buffers = SimpleNamespace(input_ids=torch.zeros(8, dtype=torch.int64), positions=torch.zeros(8, dtype=torch.int64))
        self.kv_caches = [model.attn.kv_cache]
        spec = type("FullAttentionSpec", (), {"block_size": BLOCK_SIZE})
        self.kv_cache_config = SimpleNamespace(num_blocks=NUM_BLOCKS,
                                               kv_cache_groups=[SimpleNamespace(layer_names=layers, kv_cache_spec=spec()) for layers in groups])
        self.req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.zeros(4, dtype=np.int64)), req_id_to_index=dict(STATE_ROW))
        for r, row in STATE_ROW.items():
            self.req_states.prompt_len.np[row] = PROMPT_LEN[r]
        self.block_tables = SimpleNamespace(num_kv_cache_groups=len(groups), num_blocks=SimpleNamespace(np=np.zeros((len(groups), 4), dtype=np.int64)),
                                            input_block_tables=[torch.zeros(2, 4, dtype=torch.int32) for _ in groups],
                                            block_tables=[SimpleNamespace(gpu=torch.zeros(4, 4, dtype=torch.int32)) for _ in groups])
        self.execute_model_state: Any = None

    def execute_model(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> str:
        self.calls.append(("execute_model", scheduler_output, args, kwargs))
        if kwargs.get("dummy_run") or (len(args) >= 2 and args[1]) or not getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            return "passed through"
        step = self.steps[scheduler_output.step]
        batch = _batch(step, self.input_buffers)
        for g, blocks in enumerate([step["blocks"], *step.get("more blocks", [])]):
            for i, r in enumerate(batch.req_ids):
                self.block_tables.num_blocks.np[g, STATE_ROW[r]] = len(blocks[r])
                self.block_tables.input_block_tables[g][i, : len(blocks[r])] = torch.tensor(blocks[r], dtype=torch.int32)
                self.block_tables.block_tables[g].gpu[STATE_ROW[r], : len(blocks[r])] = torch.tensor(blocks[r], dtype=torch.int32)
        self.execute_model_state = SimpleNamespace(input_batch=batch, slot_mappings_by_layer={LAYER: batch.slots})
        self.prepared(batch)
        self.hidden = self.model(batch.input_ids[: batch.num_tokens]).clone()
        return f"forward {scheduler_output.step}"

    def sample(self, hidden_states: Any, input_batch: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("sample", hidden_states, input_batch, grammar_output, args, kwargs))
        if input_batch is None:
            return "passed through"
        self.model.compute_logits(self.hidden[input_batch.logits_indices])
        step = self.steps[grammar_output]
        return self.sampled(SimpleNamespace(sampled_token_ids=torch.tensor([[step["sampled"][r]] for r in input_batch.req_ids])))


def _scheduled(step: int, steps: list[dict] = STEPS) -> SimpleNamespace:
    counts = {r: len(ids) for r, ids in steps[step]["ids"].items()}
    return SimpleNamespace(step=step, num_scheduled_tokens=counts, total_num_scheduled_tokens=sum(counts.values()))


@pytest.fixture
def device(monkeypatch):
    """Pinned memory and the CUDA event, stood in; the events made, in order."""
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: empty(*a, **kw))
    monkeypatch.setattr(_Event, "made", [])
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    return _Event.made


@pytest.fixture
def clock(monkeypatch):
    """The capture's clock, stood in: its readings are 1, 2, 4 and on, so a span between two of them names both."""
    readings = itertools.count()
    monkeypatch.setattr(C, "time", SimpleNamespace(perf_counter=lambda: 2 ** next(readings)))


def _span(first: int, last: int) -> int:
    """The time between two readings of the stood-in clock, each by its number from 0."""
    return 2 ** last - 2 ** first


def _capture(tmp_path, config: ObserverConfig | None = None, steps: list[dict] = STEPS, header: Any = None,
             groups: tuple[list[str], ...] = ([LAYER],)) -> tuple[Capture, _Runner]:
    model = _Model()
    runner = _Runner(model, steps, groups)
    capture = Capture(SimpleNamespace(runner=runner, model=model), header=header, log_path=tmp_path / "log.jsonl",
                      config=config or ObserverConfig(triton=False))
    capture.install()
    return capture, runner


def _step(runner: _Runner, step: int, steps: list[dict] = STEPS) -> tuple[Any, Any]:
    """One step as the worker takes it: `execute_model`, then `sample` on the batch it left."""
    out = runner.execute_model(_scheduled(step, steps))
    return out, runner.sample("hidden", runner.execute_model_state.input_batch, step)


def _held(capture: Capture) -> tuple[int | None, int | None]:
    """The step the capture is in and the step it holds to write, each by its number."""
    cur, pending = capture._cur, capture._pending
    return (None if cur is None else cur.step, None if pending is None else pending.step)


def _kinds(events: list) -> list[str]:
    return [type(e).__name__ for e in events]


def _notes(events: list) -> list[tuple[str, dict]]:
    return [(e.text, e.data) for e in events if isinstance(e, E.Note)]


def test_a_prefill_step_and_a_decode_step_are_logged_with_their_rows_roots_links_and_sampled_ids(tmp_path, device, clock):
    capture, runner = _capture(tmp_path)
    outs = [_step(runner, 0), _step(runner, 1)]
    assert [o[0] for o in outs] == ["forward 0", "forward 1"]
    assert [(o[1][0].sampled_token_ids.tolist(), o[1][1]) for o in outs] == [([[9], [4]], "the rest"), ([[2], [6]], "the rest")]
    stats = capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    starts = [e for e in events if isinstance(e, E.StepStart)]
    assert [(s.step, s.rows, s.input_token_ids, s.positions) for s in starts] == [
        (0, [("a", 0), ("a", 1), ("b", 0)], [3, 5, 7], [0, 1, 0]), (1, [("a", 2), ("b", 1)], [9, 4], [2, 1])]
    assert [(s.seq_lens, s.num_computed) for s in starts] == [({"a": 2, "b": 1}, {"a": 0, "b": 0}), ({"a": 3, "b": 2}, {"a": 2, "b": 1})]
    assert [(s.block_tables, s.block_size, s.slot_mapping) for s in starts] == [
        ({"a": [4], "b": [6]}, 2, [8, 9, 12]), ({"a": [4, 9], "b": [6]}, 2, [18, 13])]
    assert [(s.logits_rows, s.logits_indices) for s in starts] == [([("a", 1), ("b", 0)], [1, 2]), ([("a", 2), ("b", 1)], [0, 1])]
    assert [(s.num_reqs, s.num_tokens, s.num_tokens_padded, s.phase_hint, s.kv_groups) for s in starts] == [
        (2, 3, 4, "prefill", None), (2, 2, 3, "decode", None)]
    ids_view, positions_view = capture.storage.view_of(runner.input_buffers.input_ids), capture.storage.view_of(runner.input_buffers.positions)
    assert [(s.input_ids_view, s.positions_view) for s in starts] == [(ids_view, positions_view)] * 2
    prompt = [e for e in events if isinstance(e, E.Root) and e.role == "prompt_token"]
    assert [(r.name, r.values, r.view) for r in prompt] == [
        (name, [token], E.ViewDesc(ids_view.alloc, row, (1,), (1,), "int64", "cpu"))
        for row, (name, token) in enumerate([("prompt[a][0]", 3), ("prompt[a][1]", 5), ("prompt[b][0]", 7)])]
    assert [e for e in events if isinstance(e, E.StepLink)] == [E.StepLink(1, 0, "a", 0, "eq", 9), E.StepLink(1, 1, "b", 0, "eq", 4)]
    ends = [e for e in events if isinstance(e, E.SampleEnd)]
    assert [(e.step, e.sampled_token_ids, e.sampled_view.shape, e.logits_view.shape) for e in ends] == [
        (0, [9, 4], (2, 1), (2, 16)), (1, [2, 6], (2, 1), (2, 16))]
    assert capture.sample_ids == {0: [9, 4], 1: [2, 6]}
    assert capture.step_rows == {0: [("a", 0), ("a", 1), ("b", 0)], 1: [("a", 2), ("b", 1)]}
    assert stats["steps"] == 2 and stats["errors"] == [] and stats["roots"] == 2 and stats["events_written"] == len(events) - 1
    assert stats["recorder"]["dispatch"] == sum(isinstance(e, E.DispatchEnter) for e in events) > 0
    assert stats["snapshot_bytes"] == 0 and stats["snapshot_files"] == 0 and sorted(stats["timing"]) == ["flush_s", "snapshot_s", "sync_s"]
    buffered = [sum(isinstance(e, (E.DispatchEnter, E.DispatchExit, E.StorageAlloc, E.StorageFree)) for e in events[a:b])
                for a, b in zip([events.index(s) for s in starts], [events.index(e) for e in ends])]
    assert [(t["step"], t["num_tokens"], t["num_reqs"], t["phase"], t["snapshots"], t["events"]) for t in stats["step_times"]] == [
        (0, 3, 2, "prefill", 0, buffered[0]), (1, 2, 2, "decode", 0, buffered[1])]
    # a step reads the clock six times, at its start, around its forward, around its sample and at its end, and writing it two more
    assert [(t["forward_s"], t["sample_s"], t["wall_s"]) for t in stats["step_times"]] == [
        (_span(1, 2), _span(3, 4), _span(0, 5)), (_span(9, 10), _span(11, 12), _span(8, 13))]
    assert stats["timing"] == {"flush_s": _span(6, 7) + _span(14, 15), "snapshot_s": 0, "sync_s": 0}
    assert isinstance(events[-1], E.RunEnd) and events[-1].stats["steps"] == 2
    assert [(e.recorded, e.synchronized) for e in device] == [(2, 1), (2, 1)]


def test_the_log_opens_with_the_header_the_roots_and_the_runners_buffers_and_holds_each_step_in_its_order(tmp_path, device):
    header = E.RunHeader(E.SCHEMA, "2026-11-13T00:00:00Z", "profile", {}, {}, {})
    capture, runner = _capture(tmp_path, header=header)
    _step(runner, 0)
    # what a call that raised inside the step before left open is not the next step's
    capture.rec.open.append(99)
    capture.rec.module_stack.append("a module of the step before")
    _step(runner, 1)
    assert capture.rec.active is False and capture.rec.phase == "other"
    capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    first = events.index(next(e for e in events if isinstance(e, E.StepStart)))
    assert events[0] == header
    assert _kinds(events[1:first]) == ["StorageAlloc", "Root", "StorageAlloc", "Root", "StorageAlloc", "StorageAlloc", "StorageAlloc", "Note"]
    assert [(e.alloc, e.origin) for e in events[1:first] if isinstance(e, E.StorageAlloc)] == [
        (1, "param"), (2, "buffer"), (3, "external"), (4, "external"), (5, "external")]
    assert [(e.role, e.name, e.view.alloc, e.view.shape, e.values) for e in events[1:first] if isinstance(e, E.Root)] == [
        ("weight", "table", 1, (16, 2), None), ("buffer", "scale", 2, (2,), None)]
    assert capture.root_allocs == {1, 2} and capture.roots == [e for e in events[1:first] if isinstance(e, E.Root)]
    assert _notes(events[1:first]) == [("external buffers (not roots): runner input buffers and KV caches", {
        "buffers": {"input_buffers.input_ids": {"alloc": 3, "shape": [8], "dtype": "int64"},
                    "input_buffers.positions": {"alloc": 4, "shape": [8], "dtype": "int64"},
                    "kv_cache[attn]": {"alloc": 5, "shape": [16, 2], "dtype": "float32", "offset": 0, "strides": [2, 1]},
                    "runner.kv_caches[0]": {"alloc": 5, "shape": [16, 2], "dtype": "float32"}},
        "kv_num_blocks": 16, "block_size": 2,
        "kv_cache_groups": [{"layers": ["attn"], "block_size": 2, "spec": "FullAttentionSpec"}],
        "storage_weakref_frees": capture.storage.use_weakrefs})]
    # a step: its start, its prompt rows' roots or its decode rows' links, what was recorded in it, then its two ends
    steps: list[list] = [[]]
    for e in events[first:-1]:
        steps[-1].append(e)
        if isinstance(e, E.StepEnd):
            steps.append([])
    assert len(steps) == 3 and steps[2] == [] and isinstance(events[-1], E.RunEnd)
    for step, head in zip(steps, (["StepStart", "Root", "Root", "Root"], ["StepStart", "StepLink", "StepLink"])):
        assert _kinds(step[: len(head)]) == head and _kinds(step[-2:]) == ["SampleEnd", "StepEnd"]
        body = step[len(head):-2]
        assert set(_kinds(body)) == {"StorageAlloc", "DispatchEnter", "DispatchExit", "StorageFree"}
        enters = [e for e in body if isinstance(e, E.DispatchEnter)]
        assert {e.step for e in enters} == {step[0].step}
        phases = [e.phase for e in enters]
        runs = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
        assert runs == ["other", "forward", "other", "sample"] and enters[phases.index("forward") + 3].op == "aten.clone.default"
        assert [(e.op, e.module) for e in enters if e.phase == "forward"] == [
            ("aten.index.Tensor", None), ("aten.mul.Tensor", None), ("aten.add.Tensor", "attn")]
        assert {e.parent for e in enters} == {None} and {e.module for e in enters if e.phase != "forward"} == {None}
        assert [e.op for e in enters if e.phase == "sample"][:3] == ["aten.index.Tensor", "aten.permute.default", "aten.mm.default"]
        allocs = [e.alloc for e in step if isinstance(e, E.StorageAlloc)]
        assert step[-2].logits_view.alloc in allocs and step[-2].sampled_view.alloc in allocs


def test_a_tensor_the_model_holds_as_a_parameter_and_as_a_buffer_is_one_root(tmp_path):
    model = _Model()
    model.register_buffer("tied", model.table)
    capture = Capture(SimpleNamespace(runner=_Runner(model, STEPS), model=model), header=None, log_path=tmp_path / "log.jsonl",
                      config=ObserverConfig(triton=False))
    capture.install()
    assert [(r.role, r.name, r.view.alloc) for r in capture.roots] == [("weight", "table", 1), ("buffer", "scale", 2)]
    assert capture.root_allocs == {1, 2} and capture.close()["roots"] == 2


def test_a_step_is_written_when_the_next_one_starts_or_the_capture_closes(tmp_path, device, clock):
    capture, runner = _capture(tmp_path)
    at_install = capture.events_written
    runner.execute_model(_scheduled(0))
    assert _held(capture) == (0, None) and capture.rec.step == 0
    assert capture._cur.facts["req_ids"] == ["a", "b"] and capture.events_written == at_install
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert _held(capture) == (None, 0) and capture.step_index == 1
    assert capture.events_written == at_install and device[0].synchronized == 0 and capture.step_times == []
    first = capture._pending
    assert first.events
    runner.execute_model(_scheduled(1))
    assert _held(capture) == (1, None) and device[0].synchronized == 1 and first.events == []
    written = capture.events_written
    in_the_file = list(read_log(tmp_path / "log.jsonl"))
    assert len(in_the_file) == written and in_the_file[-1] == E.StepEnd(0, capture.step_times[0]["forward_s"], capture.step_times[0]["sample_s"])
    assert written > at_install and [t["step"] for t in capture.step_times] == [0] and capture.timing["flush_s"] == _span(6, 7)
    runner.sample("hidden", runner.execute_model_state.input_batch, 1)
    assert capture.events_written == written and _held(capture) == (None, 1) and capture._pending.events
    capture.close()
    assert _held(capture) == (None, None) and [t["step"] for t in capture.step_times] == [0, 1]
    assert _kinds(list(read_log(tmp_path / "log.jsonl")))[-3:] == ["SampleEnd", "StepEnd", "RunEnd"]


def test_a_dummy_run_an_empty_step_and_a_sample_outside_a_step_pass_through(tmp_path, device):
    capture, runner = _capture(tmp_path)
    at_install = capture.events_written
    empty = SimpleNamespace(step=0, num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    assert runner.execute_model(_scheduled(0), "an argument", dummy_run=True) == "passed through"
    assert runner.execute_model(_scheduled(0), "an argument", True) == "passed through"
    assert runner.execute_model(empty, "an argument", key="a value") == "passed through"
    assert runner.execute_model(SimpleNamespace(), "an argument") == "passed through"
    assert runner.sample("hidden", None, "grammar", "an argument", key="a value") == "passed through"
    assert [c[0] for c in runner.calls] == ["execute_model"] * 4 + ["sample"]
    assert [c[2:] for c in runner.calls[:3]] == [(("an argument",), {"dummy_run": True}), (("an argument", True), {}), (("an argument",), {"key": "a value"})]
    assert runner.calls[4][1:] == ("hidden", None, "grammar", ("an argument",), {"key": "a value"})
    assert _held(capture) == (None, None) and capture.step_index == 0 and capture.events_written == at_install
    assert capture.rec.active is False and device == [] and capture.errors == []


def test_what_follows_the_schedulers_output_reaches_the_runner_and_only_its_second_argument_names_a_dummy_run(tmp_path, device):
    capture, runner = _capture(tmp_path)
    scheduled = _scheduled(0)
    assert runner.execute_model(scheduled, "intermediate tensors", key="a value") == "forward 0" and _held(capture) == (0, None)
    assert runner.calls[-1] == ("execute_model", scheduled, ("intermediate tensors",), {"key": "a value"})
    batch = runner.execute_model_state.input_batch
    assert runner.sample("hidden", batch, 0, "an argument", key="a value")[1] == "the rest" and _held(capture) == (None, 0)
    assert runner.calls[-1] == ("sample", "hidden", batch, 0, ("an argument",), {"key": "a value"}) and capture.errors == []


def test_a_capture_with_no_module_hooks_has_the_whole_of_execute_model_as_the_forward(tmp_path, device):
    capture, runner = _capture(tmp_path, ObserverConfig(triton=False, module_hooks=False))
    assert capture.namer is None and not capture.model._forward_pre_hooks and not capture.model._forward_hooks
    _step(runner, 0)
    capture.close()
    enters = [e for e in read_log(tmp_path / "log.jsonl") if isinstance(e, E.DispatchEnter)]
    phases = [e.phase for e in enters]
    assert [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p] == ["forward", "sample"]
    assert {e.module for e in enters} == {None} and capture.errors == []


def test_a_forward_that_raised_leaves_the_sample_after_it_its_own_phase(tmp_path, device, monkeypatch):
    capture, runner = _capture(tmp_path)
    _step(runner, 0)

    def raises(hidden):
        raise ZeroDivisionError("the layer failed")

    with monkeypatch.context() as patched:
        patched.setattr(capture.model.attn, "forward", raises)
        with pytest.raises(ZeroDivisionError):
            runner.execute_model(_scheduled(1))
    assert _held(capture) == (1, None) and capture.rec.active is False and capture.rec.phase == "other"

    def a_forward_then_a_copy(output):
        runner.model(runner.input_buffers.input_ids[:1])
        return SimpleNamespace(sampled_token_ids=output.sampled_token_ids.clone())

    runner.sampled = a_forward_then_a_copy
    runner.sample("hidden", runner.execute_model_state.input_batch, 1)
    capture.close()
    enters = [e for e in read_log(tmp_path / "log.jsonl") if isinstance(e, E.DispatchEnter) and e.step == 1]
    phases = [e.phase for e in enters]
    assert [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p] == ["other", "forward", "sample"]
    assert (enters[-1].op, enters[-1].phase) == ("aten.clone.default", "sample")
    assert [(e["where"], e["step"]) for e in capture.errors] == [("finalize", 1)]


@pytest.mark.parametrize("mode, entered", [("none", 0), ("empty", 2)])
def test_a_capture_that_does_not_record_times_each_step_and_logs_none(tmp_path, device, clock, monkeypatch, mode, entered):
    modes = []

    class Empty(C.EmptyMode):
        def __enter__(self):
            modes.append(capture.rec.active)
            return super().__enter__()

    monkeypatch.setattr(C, "EmptyMode", Empty)
    capture, runner = _capture(tmp_path, ObserverConfig(mode=mode, triton=False))
    observed = capture.storage.stats["observed"]
    assert _step(runner, 0)[0] == "forward 0" and _held(capture) == (None, None) and capture.step_index == 1
    assert _step(runner, 1)[1][0].sampled_token_ids.tolist() == [[2], [6]]
    stats = capture.close()
    assert modes == [False] * (2 * entered) and device == [] and capture.namer is None
    assert [{k: t[k] for k in ("step", "num_reqs", "events", "snapshots")} for t in stats["step_times"]] == [
        {"step": 0, "num_reqs": 2, "events": 0, "snapshots": 0}, {"step": 1, "num_reqs": 2, "events": 0, "snapshots": 0}]
    assert all(sorted(t) == ["events", "forward_s", "num_reqs", "sample_s", "snapshots", "step", "wall_s"] for t in stats["step_times"])
    assert [(t["forward_s"], t["sample_s"], t["wall_s"]) for t in stats["step_times"]] == [
        (_span(1, 2), _span(3, 4), _span(0, 5)), (_span(7, 8), _span(9, 10), _span(6, 11))]
    assert stats["steps"] == 2 and stats["recorder"]["dispatch"] == 0 and stats["errors"] == []
    assert stats["storage"]["observed"] == observed and capture.sample_ids == {} and capture.step_rows == {}
    assert _kinds(list(read_log(tmp_path / "log.jsonl"))) == [
        "StorageAlloc", "Root", "StorageAlloc", "Root", "StorageAlloc", "StorageAlloc", "StorageAlloc", "Note", "RunEnd"]


def test_a_capture_that_writes_no_log_still_records_and_keeps_each_steps_ids(tmp_path, device):
    capture, runner = _capture(tmp_path, ObserverConfig(triton=False, write_log=False))
    _step(runner, 0)
    capture.rec.note("outside a step")
    _step(runner, 1)
    stats = capture.close()
    assert capture.writer is None and not (tmp_path / "log.jsonl").exists() and stats["events_written"] == 0
    assert capture.sample_ids == {0: [9, 4], 1: [2, 6]} and stats["recorder"]["dispatch"] > 0 and stats["recorder"]["notes"] == 1
    assert [t["events"] > 0 for t in stats["step_times"]] == [True, True] and stats["errors"] == []


def test_what_is_recorded_outside_a_step_is_written_at_once(tmp_path, device):
    capture, runner = _capture(tmp_path)
    _step(runner, 0)
    written = capture.events_written
    capture.rec.note("between two steps", why="a test")
    assert capture.events_written > written and capture._pending.step == 0
    capture.writer.flush()
    events = list(read_log(tmp_path / "log.jsonl"))
    assert len(events) == capture.events_written and events[-1] == E.Note("between two steps", {"why": "a test"})
    assert set(_kinds(events[written:-1])) <= {"StorageAlloc", "StorageFree"}
    capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    at = events.index(E.Note("between two steps", {"why": "a test"}))
    assert isinstance(events[at + 1], E.StepStart) and not any(isinstance(e, E.StepStart) for e in events[:at])


def test_closing_takes_the_wrappers_and_the_hooks_off_and_is_done_once(tmp_path, device, monkeypatch):
    model = _Model()
    runner = _Runner(model, STEPS)
    execute_model, sample, compute_logits = runner.execute_model, runner.sample, model.compute_logits
    told = []
    monkeypatch.setattr(C.triton_adapter, "install", lambda rec: told.append(("install", rec)))
    monkeypatch.setattr(C.triton_adapter, "is_installed", lambda: True)
    monkeypatch.setattr(C.triton_adapter, "set_recorder", lambda rec: told.append(("set_recorder", rec)))
    capture = Capture(SimpleNamespace(runner=runner, model=model), header=None, log_path=tmp_path / "log.jsonl",
                      config=ObserverConfig(snapshot_steps=frozenset({5})))
    capture.install()
    assert told == [("install", capture.rec)] and capture.rec.after_call == capture._after_call
    assert (runner.execute_model, runner.sample, model.compute_logits) != (execute_model, sample, compute_logits)
    assert len(model._forward_pre_hooks) == len(model._forward_hooks) == 1 and len(model.attn._forward_hooks) == 1
    _step(runner, 0)
    stats = capture.close()
    assert (runner.execute_model, runner.sample, model.compute_logits) == (execute_model, sample, compute_logits)
    assert not model._forward_pre_hooks and not model._forward_hooks and not model.attn._forward_pre_hooks and not model.attn._forward_hooks
    assert capture.namer is None and capture._root_handles == [] and capture.installed is False
    assert capture.rec.after_call is None and capture.rec.after_launch is None and told[1:] == [("set_recorder", None)]
    assert stats["steps"] == 1 and stats["step_times"][0]["step"] == 0
    capture.uninstall()
    assert told[1:] == [("set_recorder", None)]
    assert _step(runner, 1)[0] == "forward 1" and capture.step_index == 1 and capture.stats()["events_written"] == stats["events_written"] + 1


def test_a_step_whose_facts_cannot_be_read_is_logged_as_failed_with_what_was_recorded_in_it(tmp_path, device):
    capture, runner = _capture(tmp_path)

    def no_state(batch):
        runner.execute_model_state = None

    runner.prepared = no_state
    assert runner.execute_model(_scheduled(0)) == "forward 0"
    error = "AttributeError(\"'NoneType' object has no attribute 'input_batch'\")"
    assert capture.errors == [{"where": "after_forward", "step": 0, "error": error}]
    assert capture._cur.facts == {"sched_req_ids": [A, B], "error": error} and device == []
    runner.execute_model_state = SimpleNamespace(input_batch=_batch(STEPS[0], runner.input_buffers))
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    stats = capture.close()
    assert stats["errors"][1] == {"where": "finalize", "step": 0, "error": f"RuntimeError({('no step facts: ' + error)!r})"}
    assert len(stats["errors"]) == 2 and stats["steps"] == 1 and stats["step_times"] == []
    assert [(e.recorded, e.synchronized) for e in device] == [(1, 1)]
    events = list(read_log(tmp_path / "log.jsonl"))
    at = events.index(E.Note("step finalisation failed", {"step": 0, "error": stats["errors"][1]["error"]}))
    assert not any(isinstance(e, (E.StepStart, E.SampleEnd)) for e in events) and _kinds(events[-2:]) == ["StepEnd", "RunEnd"]
    assert events[-2].step == 0 and events[-2].forward_s > 0 and events[-2].sample_s > 0
    body = events[at + 1:-2]
    assert {e.phase for e in body if isinstance(e, E.DispatchEnter)} == {"other", "forward", "sample"}
    assert set(_kinds(body)) == {"StorageAlloc", "DispatchEnter", "DispatchExit", "StorageFree"}


def test_a_sampler_output_whose_ids_cannot_be_read_ends_the_step_with_no_sampled_ids(tmp_path, device):
    capture, runner = _capture(tmp_path)
    runner.sampled = lambda output: "no sampler output"
    assert _step(runner, 0)[1] == "no sampler output"
    assert capture.errors == [{"where": "after_sample", "step": 0, "error": "AttributeError(\"'str' object has no attribute 'sampled_token_ids'\")"}]
    assert capture._pending.step == 0 and capture.step_index == 1 and capture._pending.wall_s > 0
    runner.sampled = lambda output: (output, "the rest")
    _step(runner, 1)
    capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    ends = [e for e in events if isinstance(e, E.SampleEnd)]
    assert [(e.step, e.sampled_token_ids, e.sampled_view) for e in ends][0] == (0, [], None) and ends[1].sampled_token_ids == [2, 6]
    assert capture.sample_ids == {0: [], 1: [2, 6]} and not any(isinstance(e, E.StepLink) for e in events)
    assert [n for n in _notes(events) if n[0] == "decode row without a known source step"] == [
        ("decode row without a known source step", {"step": 1, "row": row, "request_id": r, "position": p, "known": None})
        for row, r, p in [(1, "b", 1), (0, "a", 2)]]


@pytest.mark.parametrize("shape", [(2,), (2, 1, 1), (2, 3)])
def test_sampled_ids_are_read_as_a_row_a_request_whatever_their_shape(tmp_path, device, shape):
    capture, runner = _capture(tmp_path)
    ids = torch.tensor([[9, 0, 0], [4, 0, 0]]) if shape == (2, 3) else torch.tensor([9, 4]).reshape(shape)
    runner.sampled = lambda output: SimpleNamespace(sampled_token_ids=ids)
    assert _step(runner, 0)[1].sampled_token_ids is ids
    assert capture._pending.sampled_shape == (2, 3 if shape == (2, 3) else 1) and capture.errors == []
    capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    (end,) = [e for e in events if isinstance(e, E.SampleEnd)]
    assert end.sampled_token_ids == [9, 4] and end.sampled_view.shape == shape and capture.sample_ids == {0: [9, 4]}
    assert capture._pending_token == {"a": (0, 9, 2), "b": (0, 4, 1)}
    # the ids were made outside the step, so their allocation was first seen once the step's events were buffered: it is
    # written before the step's start, which is where a reader meets the step's views
    start = next(i for i, e in enumerate(events) if isinstance(e, E.StepStart))
    assert isinstance(events[start - 1], E.StorageAlloc) and events[start - 1].alloc == end.sampled_view.alloc
    assert sum(isinstance(e, E.StorageAlloc) and e.alloc == end.sampled_view.alloc for e in events) == 1
    if capture.storage.use_weakrefs:
        # the runner's own sampler output, whose place `ids` took, died once the step's last call was buffered: its free
        # follows what was buffered
        last = events[events.index(end) - 1]
        assert isinstance(last, E.StorageFree) and isinstance(events[events.index(end) - 4], E.StorageAlloc)
        assert events[events.index(end) - 4].alloc == last.alloc and _kinds(events[events.index(end) - 3:][:2]) == ["DispatchEnter", "DispatchExit"]


def test_a_pinned_buffer_is_kept_by_name_and_made_again_only_to_grow_or_for_another_dtype(tmp_path, device):
    capture, _ = _capture(tmp_path)
    two = capture._pin("x", 2, torch.int64)
    four = capture._pin("x", 4, torch.int64)
    three = capture._pin("x", 3, torch.int64)
    assert (two.shape, four.shape, three.shape) == ((2,), (4,), (3,)) and four.data_ptr() != two.data_ptr() == two.data_ptr()
    assert three.data_ptr() == four.data_ptr() and capture._pin_cache["x"].numel() == 4
    other = capture._pin("x", 3, torch.int32)
    assert other.dtype == torch.int32 and capture._pin_cache["x"].numel() == 3 and other.data_ptr() != four.data_ptr()
    table = capture._pin("x", 6, torch.int32, (2, 3))
    assert table.shape == (2, 3) and capture._pin_cache["x"].numel() == 6
    assert capture._pin("y", 1, torch.int64, (1, 1)).shape == (1, 1) and sorted(capture._pin_cache) == ["x", "y"]


def test_logits_whose_view_cannot_be_read_are_an_error_of_the_step_and_logits_outside_a_step_are_not_read(tmp_path, device, monkeypatch):
    capture, runner = _capture(tmp_path)
    logits = capture.model.compute_logits(torch.zeros(1, 2))
    assert logits.shape == (1, 16) and capture._cur is None and capture.errors == []
    runner.execute_model(_scheduled(0))
    view_of = capture.storage.view_of

    def no_logits_view(t, origin="op"):
        if tuple(t.shape) == (2, 16):
            raise ValueError("no view of the logits")
        return view_of(t, origin)

    monkeypatch.setattr(capture.storage, "view_of", no_logits_view)
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert capture.errors == [{"where": "compute_logits", "step": 0, "error": "ValueError('no view of the logits')"}]
    assert capture._pending.logits_view is None


def test_a_value_that_is_no_tensor_is_not_a_steps_logits(tmp_path, device, monkeypatch):
    monkeypatch.setattr(_Model, "compute_logits", lambda self, hidden: "not a tensor")
    capture, runner = _capture(tmp_path)
    runner.execute_model(_scheduled(0))
    assert capture.model.compute_logits(torch.zeros(1, 2)) == "not a tensor" and capture._cur.logits_view is None and capture.errors == []


def _logged(tmp_path, capture: Capture) -> tuple[list, list[E.StepStart]]:
    capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    return events, [e for e in events if isinstance(e, E.StepStart)]


def test_positions_the_model_read_that_differ_from_the_batchs_are_noted_and_taken(tmp_path, device):
    capture, runner = _capture(tmp_path)
    runner.prepared = lambda batch: batch.positions.__setitem__(slice(0, 3), torch.tensor([5, 0, 7]))
    _step(runner, 0)
    events, (start,) = _logged(tmp_path, capture)
    assert start.positions == [5, 0, 7] and start.rows == [("a", 5), ("a", 0), ("b", 7)] and start.phase_hint == "mixed"
    assert start.logits_rows == [("a", 1), ("b", 0)] and start.seq_lens == {"a": 2, "b": 1}
    notes = _notes(events[events.index(start):])
    assert notes[2] == ("device positions differ from host-derived positions", {"step": 0, "device": [5, 0, 7], "host": [0, 1, 0]})
    assert [(text, data["row"], data["position"], data["known"]) for text, data in notes[:2]] == [
        ("decode row without a known source step", 2, 7, None), ("decode row without a known source step", 0, 5, None)]
    assert len(notes) == 3
    assert [(r.name, r.values) for r in events if isinstance(r, E.Root) and r.role == "prompt_token"] == [("prompt[a][0]", [5])]
    assert capture.step_rows == {0: [("a", 5), ("a", 0), ("b", 7)]} and capture.errors == []


def test_a_decode_row_is_linked_to_the_step_that_sampled_it_and_what_does_not_fit_is_noted(tmp_path, device):
    steps = [STEPS[0], {"ids": {A: [11], B: [4]}, "computed": {A: 2, B: 2}, "blocks": {A: [4, 9], B: [6, 7]}, "sampled": {A: 2, B: 6}}]
    capture, runner = _capture(tmp_path, steps=steps)
    _step(runner, 0, steps)
    _step(runner, 1, steps)
    events, starts = _logged(tmp_path, capture)
    assert [e for e in events if isinstance(e, E.StepLink)] == [E.StepLink(1, 0, "a", 0, "eq", 9)]
    assert _notes(events[events.index(starts[1]):]) == [
        ("decode row without a known source step", {"step": 1, "row": 1, "request_id": "b", "position": 2, "known": [0, 4, 1]}),
        ("StepLink value != input token", {"step": 1, "row": 0, "request_id": "a", "link_value": 9, "input": 11})]
    assert starts[1].input_token_ids == [11, 4] and starts[1].rows == [("a", 2), ("b", 2)] and starts[1].phase_hint == "decode"
    assert capture._pending_token == {"a": (1, 2, 3), "b": (1, 6, 3)}


def test_a_step_with_prompt_rows_and_decode_rows_is_mixed_and_has_a_root_or_a_link_a_row(tmp_path, device):
    capture, runner = _capture(tmp_path)
    runner.req_states.prompt_len.np[STATE_ROW[A]] = 3
    _step(runner, 0)
    _step(runner, 1)
    events, starts = _logged(tmp_path, capture)
    assert [s.phase_hint for s in starts] == ["prefill", "mixed"] and capture.errors == []
    after = events[events.index(starts[1]):]
    ids_view = starts[1].input_ids_view
    assert [(r.name, r.values, r.view) for r in after if isinstance(r, E.Root)] == [
        ("prompt[a][2]", [9], E.ViewDesc(ids_view.alloc, 0, (1,), (1,), "int64", "cpu"))]
    assert [e for e in after if isinstance(e, E.StepLink)] == [E.StepLink(1, 1, "b", 0, "eq", 4)] and _notes(after) == []


def test_a_step_with_no_blocks_or_no_slot_mapping_has_empty_ones_whatever_an_earlier_step_left(tmp_path, device):
    capture, runner = _capture(tmp_path)
    _step(runner, 0)

    def none(batch):
        runner.execute_model_state.slot_mappings_by_layer = {}
        runner.block_tables.num_blocks.np[:] = 0

    runner.prepared = none
    _step(runner, 1)
    events, starts = _logged(tmp_path, capture)
    assert (starts[0].block_tables, starts[0].slot_mapping) == ({"a": [4], "b": [6]}, [8, 9, 12])
    assert (starts[1].block_tables, starts[1].slot_mapping, starts[1].block_size) == ({"a": [], "b": []}, [], 2)
    assert sorted(capture._pin_cache) == ["block_tables", "input_ids", "logits_indices", "positions", "sampled", "slots"]
    assert capture.errors == [] and starts[1].kv_groups is None


def test_a_runner_with_no_kv_cache_group_reads_no_block_table(tmp_path, device):
    capture, runner = _capture(tmp_path)

    def no_group(batch):
        runner.block_tables = SimpleNamespace(num_kv_cache_groups=0, num_blocks=SimpleNamespace(np=np.zeros((0, 4), dtype=np.int64)),
                                              input_block_tables=[], block_tables=[])

    runner.prepared = no_group
    _step(runner, 0)
    events, (start,) = _logged(tmp_path, capture)
    assert start.block_tables == {"a": [], "b": []} and start.slot_mapping == [8, 9, 12] and capture.errors == []
    assert "block_tables" not in capture._pin_cache


def test_request_ids_past_the_batchs_count_are_not_the_steps(tmp_path, device):
    capture, runner = _capture(tmp_path)
    runner.prepared = lambda batch: batch.req_ids.append("a row of padding")
    runner.execute_model(_scheduled(0))
    batch = runner.execute_model_state.input_batch
    batch.req_ids.pop()
    runner.sample("hidden", batch, 0)
    events, (start,) = _logged(tmp_path, capture)
    assert start.rows == [("a", 0), ("a", 1), ("b", 0)] and start.num_reqs == 2 and capture.errors == []


def test_a_step_smaller_than_the_one_before_it_reads_only_its_own_part_of_each_pinned_buffer(tmp_path, device):
    steps = [dict(STEPS[0], **{"more blocks": [{A: [12, 13], B: [14]}]}),
             {"ids": {A: [9]}, "computed": {A: 2}, "blocks": {A: [4, 9]}, "more blocks": [{A: [12]}], "sampled": {A: 2}}]
    capture, runner = _capture(tmp_path, steps=steps, groups=([LAYER], ["sliding"]))
    second = {0: torch.tensor([24, 25, 28, 0], dtype=torch.int32), 1: torch.tensor([26, 0], dtype=torch.int32)}

    def slots_of_the_second_group(batch):
        runner.execute_model_state.slot_mappings_by_layer["sliding"] = second[batch.num_reqs % 2]

    runner.prepared = slots_of_the_second_group
    _step(runner, 0, steps)
    _step(runner, 1, steps)
    events, starts = _logged(tmp_path, capture)
    assert (starts[0].logits_indices, starts[0].slot_mapping, starts[0].block_tables) == ([1, 2], [8, 9, 12], {"a": [4], "b": [6]})
    assert starts[0].kv_groups[1] == {"layers": ["sliding"], "block_size": 2, "block_tables": {"a": [12, 13], "b": [14]}, "slot_mapping": [24, 25, 28]}
    assert (starts[1].rows, starts[1].input_token_ids, starts[1].positions) == ([("a", 2)], [9], [2])
    assert (starts[1].logits_indices, starts[1].logits_rows, starts[1].slot_mapping, starts[1].block_tables) == ([0], [("a", 2)], [18], {"a": [4, 9]})
    assert starts[1].kv_groups[1] == {"layers": ["sliding"], "block_size": 2, "block_tables": {"a": [12]}, "slot_mapping": [26]}
    assert [e.sampled_token_ids for e in events if isinstance(e, E.SampleEnd)] == [[9, 4], [2]] and capture.errors == []


def test_a_step_of_a_runner_with_no_kv_cache_group_has_no_block_table_whatever_an_earlier_step_left(tmp_path, device):
    capture, runner = _capture(tmp_path)
    _step(runner, 0)
    runner.block_tables.num_kv_cache_groups = 0
    _step(runner, 1)
    events, starts = _logged(tmp_path, capture)
    assert starts[0].block_tables == {"a": [4], "b": [6]} and starts[1].block_tables == {"a": [], "b": []}
    assert starts[1].slot_mapping == [18, 13] and starts[1].kv_groups is None and capture.errors == []


def test_logits_rows_are_the_rows_the_indices_name_where_there_is_not_one_a_request(tmp_path, device):
    capture, runner = _capture(tmp_path)
    runner.prepared = lambda batch: setattr(batch, "logits_indices", torch.tensor([2, 0, -1]))
    runner.sampled = lambda output: SimpleNamespace(sampled_token_ids=torch.tensor([[9], [4]]))
    _step(runner, 0)
    events, (start,) = _logged(tmp_path, capture)
    assert start.logits_indices == [2, 0, -1] and start.logits_rows == [("b", 0), ("a", 0), ("?", -1)]
    assert capture._pending_token == {"b": (0, 9, 1), "a": (0, 4, 1)} and capture.errors == []


def test_each_kv_cache_group_of_a_hybrid_runner_has_its_layers_block_tables_and_slot_mapping(tmp_path, device):
    steps = [dict(STEPS[0], **{"more blocks": [{A: [12, 13], B: []}, {A: [], B: []}]})]
    capture, runner = _capture(tmp_path, steps=steps, groups=([LAYER], ["sliding.0", "sliding.1"], ["unmapped"]))

    def slots_of_the_second_group(batch):
        mappings = runner.execute_model_state.slot_mappings_by_layer
        mappings["sliding.1"] = torch.tensor([24, 25, 0, 99], dtype=torch.int32)

    runner.prepared = slots_of_the_second_group
    _step(runner, 0, steps)
    assert capture._cur is None and capture._pending.facts["kv_groups"] == [
        {"layers": [LAYER], "block_size": 2, "nblocks": [1, 1], "slot_layer": LAYER},
        {"layers": ["sliding.0", "sliding.1"], "block_size": 2, "nblocks": [2, 0], "slot_layer": "sliding.1"},
        {"layers": ["unmapped"], "block_size": 2, "nblocks": [0, 0], "slot_layer": None}]
    assert capture._pending.facts["slot_mapping_layer"] == LAYER
    events, (start,) = _logged(tmp_path, capture)
    assert start.kv_groups == [
        {"layers": [LAYER], "block_size": 2, "block_tables": {"a": [4], "b": [6]}, "slot_mapping": [8, 9, 12]},
        {"layers": ["sliding.0", "sliding.1"], "block_size": 2, "block_tables": {"a": [12, 13], "b": []}, "slot_mapping": [24, 25, 0]},
        {"layers": ["unmapped"], "block_size": 2, "block_tables": {"a": [], "b": []}, "slot_mapping": []}]
    assert (start.block_tables, start.slot_mapping) == ({"a": [4], "b": [6]}, [8, 9, 12]) and capture.errors == []
    assert sorted(k for k in capture._pin_cache if "_g" in k) == ["block_tables_g0", "block_tables_g1", "slots_g0", "slots_g1"]
    note = next(data for text, data in _notes(events) if text.startswith("external buffers"))
    assert note["kv_cache_groups"] == [{"layers": layers, "block_size": 2, "spec": "FullAttentionSpec"}
                                       for layers in ([LAYER], ["sliding.0", "sliding.1"], ["unmapped"])]


def test_a_hybrid_runners_groups_that_cannot_be_read_are_logged_empty_each_with_the_error(tmp_path, device):
    steps = [dict(STEPS[0], **{"more blocks": [{A: [12], B: [14]}]})]
    capture, runner = _capture(tmp_path, steps=steps, groups=([LAYER], ["sliding"]))
    runner.kv_cache_config.kv_cache_groups[1].kv_cache_spec = SimpleNamespace()
    _step(runner, 0, steps)
    error = "AttributeError(\"'types.SimpleNamespace' object has no attribute 'block_size'\")"
    assert capture.errors == [{"where": "after_forward.kv_groups", "step": 0, "error": error}]
    events, (start,) = _logged(tmp_path, capture)
    assert start.kv_groups == [{"layers": [], "block_size": 2, "block_tables": {"a": [], "b": []}, "slot_mapping": [], "error": error}] * 2
    assert (start.block_tables, start.slot_mapping, start.rows) == ({"a": [4], "b": [6]}, [8, 9, 12], [("a", 0), ("a", 1), ("b", 0)])
    assert [(e.recorded, e.synchronized) for e in device] == [(2, 1)] and len(capture.errors) == 1
