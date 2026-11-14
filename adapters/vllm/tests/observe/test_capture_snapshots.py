"""`verity_vllm.observe.capture`'s snapshots on a CPU: in a listed step a `Capture` copies each call's tensors that are no
roots, writes them by their content under `values/<step>/` and lists them in `snapshots.json`; a tensor over the byte limit
is copied a KV block in use at a time where one of its first two dimensions is the cache's block count, and listed as
skipped otherwise.  The runner, pinned memory and the CUDA event are `test_capture_steps`'s stand-ins;
`torch.cuda.synchronize` is stood in by a count."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.observe import test_capture_steps as S
from verity_vllm.observe import events as E
from verity_vllm.observe.capture import Capture, ObserverConfig
from verity_vllm.observe.log import read_log
from verity_vllm.observe.pinned import IDENTITY_ALIAS_OPS

#: the stand-ins for pinned memory and the CUDA event, and for the capture's clock
device, clock = S.device, S.clock


@pytest.fixture
def syncs(monkeypatch, device):
    """The times the capture synchronised with the device."""
    count: list[int] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: count.append(1))
    return count


def _capture(tmp_path, steps: list[dict] = S.STEPS, listed: tuple[int, ...] = (0,), **config: Any) -> tuple[Capture, S._Runner]:
    config.setdefault("arena_bytes", 1 << 16)
    return S._capture(tmp_path, ObserverConfig(triton=False, snapshot_steps=frozenset(listed), snapshot_dir=tmp_path / "snap", **config), steps)


def _index(tmp_path) -> dict:
    """The index a capture wrote, which is JSON with no space in it but a string's."""
    text = (tmp_path / "snap" / "snapshots.json").read_text()
    index = json.loads(text)
    assert text == json.dumps(index, separators=(",", ":"))
    return index


def test_a_listed_step_copies_each_calls_tensors_but_roots_and_lists_them_by_the_calls_id(tmp_path, syncs, clock):
    capture, runner = _capture(tmp_path, listed=(8, 0))
    S._step(runner, 0)
    assert syncs == [1] and capture._snap_pending == [] and capture._arena.off == 0
    S._step(runner, 1)
    stats = capture.close()
    events = list(read_log(tmp_path / "log.jsonl"))
    index = _index(tmp_path)
    entries = index["entries"]
    assert sorted(index) == ["entries", "layout", "note", "schema", "snapshot_max_bytes", "snapshot_max_bytes_op", "steps", "total_bytes"]
    assert (index["steps"], index["snapshot_max_bytes"], index["snapshot_max_bytes_op"]) == ([0, 8], 32 << 20, None)
    enters = {e.id: e for e in events if isinstance(e, E.DispatchEnter)}
    exits = {e.id: e.outs for e in events if isinstance(e, E.DispatchExit)}
    assert {enters[x["event"]].step for x in entries} == {0} == {x["step"] for x in entries} and syncs == [1]
    assert not any(enters[x["event"]].op in IDENTITY_ALIAS_OPS for x in entries)
    assert capture.root_allocs == {1, 2} and not any(x["view"]["$v"][0] in capture.root_allocs for x in entries)
    # a call's tensors, in the order the call has them: its arguments, then its output
    listed = [(x["event"], x["path"]) for x in entries]
    assert listed == sorted(listed, key=lambda at: at[0]) and {p.split("[")[0] for _, p in listed} == {"args", "outs"}
    calls = [i for i, e in enters.items() if e.step == 0 and e.op not in IDENTITY_ALIAS_OPS]
    assert sorted({x["event"] for x in entries}) == [i for i in calls if any(x["event"] == i for x in entries)] and len(calls) > 10
    for i in calls:
        out = exits[i]
        want = [out.to_json()] if isinstance(out, E.ViewDesc) and out.alloc not in capture.root_allocs and out.numel else []
        assert [x["view"] for x in entries if x["event"] == i and x["path"] == "outs"] == want
        views = [a.to_json() for a in enters[i].args if isinstance(a, E.ViewDesc) and a.alloc not in capture.root_allocs and a.numel]
        assert [x["view"] for x in entries if x["event"] == i and x["path"].startswith("args[") and "][" not in x["path"]] == views
    # the forward's product: the rows of the table the step's ids name
    (mul,) = [i for i, e in enters.items() if e.step == 0 and e.op == "aten.mul.Tensor"]
    (product,) = [x for x in entries if x["event"] == mul and x["path"] == "outs"]
    data = capture.model.table.detach()[[3, 5, 7]].numpy().tobytes()
    sha = hashlib.sha256(data).hexdigest()
    assert product == {"event": mul, "path": "outs", "view": exits[mul].to_json(), "dtype": "float32", "shape": [3, 2], "strides": [2, 1],
                       "byte_offset": 0, "nbytes": 24, "layout": "spanned-storage-bytes", "sha256": sha,
                       "file": f"values/0/{sha[:16]}.bin", "step": 0}
    assert (tmp_path / "snap" / product["file"]).read_bytes() == data
    # the same bytes are one file: the product is the next call's argument
    (add,) = [i for i, e in enters.items() if e.step == 0 and e.op == "aten.add.Tensor"]
    assert [x["file"] for x in entries if x["event"] == add and x["path"] == "args[0]"] == [product["file"]]
    files = {x["file"] for x in entries}
    assert all(hashlib.sha256((tmp_path / "snap" / x["file"]).read_bytes()).hexdigest() == x["sha256"] for x in entries)
    assert all(x["file"] == f"values/0/{x['sha256'][:16]}.bin" and x["nbytes"] == (tmp_path / "snap" / x["file"]).stat().st_size for x in entries)
    assert sorted(f"values/0/{f.name}" for f in (tmp_path / "snap" / "values" / "0").iterdir()) == sorted(files) and len(files) < len(entries)
    assert not (tmp_path / "snap" / "values" / "1").exists()
    total = sum((tmp_path / "snap" / f).stat().st_size for f in files)
    assert index["total_bytes"] == total == stats["snapshot_bytes"] and stats["snapshot_files"] == len(files)
    assert [t["snapshots"] for t in stats["step_times"]] == [len(entries), 0] and stats["errors"] == []
    # the listed step's copies are waited for and written between the readings after its end; a step is written between two more
    assert stats["timing"] == {"flush_s": S._span(9, 10) + S._span(17, 18), "snapshot_s": S._span(7, 8), "sync_s": S._span(6, 7)}


def test_bytes_a_listed_step_copies_again_are_the_file_of_the_step_that_wrote_them(tmp_path, syncs):
    steps = [S.STEPS[0], S.STEPS[0]]
    capture, runner = _capture(tmp_path, steps, listed=(0, 1))
    S._step(runner, 0, steps)
    arena, files, written = capture._arena, dict(capture._snap_files), capture._snap_bytes
    S._step(runner, 1, steps)
    assert capture._arena is arena and syncs == [1, 1] and capture.close()["errors"] == []
    index = _index(tmp_path)
    by_step = [[x for x in index["entries"] if x["step"] == step] for step in (0, 1)]
    assert index["steps"] == [0, 1] and len(by_step[0]) + len(by_step[1]) == len(index["entries"])
    # bytes the first step wrote are its file; the rest are files of the second
    assert files == {x["sha256"]: x["file"] for x in by_step[0]} and sum(x["sha256"] in files for x in by_step[1]) > 10
    assert all(x["file"] == files.get(x["sha256"], f"values/1/{x['sha256'][:16]}.bin") for x in by_step[1])
    new = {x["file"]: x["nbytes"] for x in by_step[1] if x["sha256"] not in files}
    assert sorted(f"values/1/{f.name}" for f in (tmp_path / "snap" / "values" / "1").iterdir()) == sorted(new) and new
    assert len(capture._snap_files) == len(files) + len(new) and capture._snap_bytes == written + sum(new.values()) == index["total_bytes"]


def test_a_listed_step_begun_again_drops_what_the_one_before_left_to_copy(tmp_path, syncs):
    capture, runner = _capture(tmp_path)
    runner.execute_model(S._scheduled(0))
    left, used = [p["entry"]["path"] for p in capture._snap_pending], capture._arena.off
    assert len(left) > 5 and used > 256 and syncs == []
    # its sample never came, and the runner takes the step again
    runner.execute_model(S._scheduled(0))
    assert [p["entry"]["path"] for p in capture._snap_pending] == left and capture._arena.off == used and syncs == []
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert capture.close()["errors"] == [] and syncs == [1]
    entries = _index(tmp_path)["entries"]
    assert [x["path"] for x in entries[: len(left)]] == left and {x["step"] for x in entries} == {0}
    assert all(hashlib.sha256((tmp_path / "snap" / x["file"]).read_bytes()).hexdigest() == x["sha256"] for x in entries)


def test_a_capture_with_no_snapshot_directory_or_no_listed_step_writes_no_index(tmp_path, syncs):
    capture, runner = S._capture(tmp_path, ObserverConfig(triton=False, snapshot_dir=tmp_path / "snap"))
    S._step(runner, 0)
    capture.close()
    assert capture._arena is None and syncs == [] and not (tmp_path / "snap").exists() and capture.rec.after_call is None


def _in_step(tmp_path, steps: list[dict] = S.STEPS, **config: Any) -> tuple[Capture, S._Runner]:
    """A capture in its first step, a listed one, after the forward; what the forward's calls left to copy is dropped."""
    capture, runner = _capture(tmp_path, steps, **config)
    runner.execute_model(S._scheduled(0, steps))
    capture._snap_pending.clear()
    capture._snap_index.clear()
    capture._arena.reset()
    return capture, runner


def _taken(capture: Capture) -> list[tuple]:
    """What is waiting to be written: each entry's event, path, sub-view and bytes."""
    return [(p["entry"]["event"], p["entry"]["path"], p["entry"].get("sub"), p["host"].numpy().tobytes()) for p in capture._snap_pending]


def _before_the_forward(tmp_path, body: Any, steps: list[dict] = S.STEPS, **config: Any) -> tuple[Capture, list[tuple], list[dict]]:
    """Runs `body(capture, runner)` in a capture's first step, a listed one, where the runner has prepared the step's inputs
    and the forward has not begun: the capture, and what `body` left waiting to be written and what it listed.  `body` runs
    under the capture's dispatch mode, so it makes no tensor of its own."""
    capture, runner = _capture(tmp_path, steps, **config)
    left: list = []

    def prepared(batch: Any) -> None:
        capture._snap_pending.clear()
        capture._snap_index.clear()
        body(capture, runner)
        left.extend([_taken(capture), list(capture._snap_index)])

    runner.prepared = prepared
    runner.execute_model(S._scheduled(0, steps))
    return capture, left[0], left[1]


def test_an_empty_tensor_and_a_root_are_not_copied_and_one_over_the_limit_is_listed_as_skipped(tmp_path, syncs):
    capture, runner = _in_step(tmp_path, snapshot_max_bytes=32)
    over, copied = torch.arange(40, dtype=torch.uint8), capture._cur.n_snap
    capture._snap_tensor(7, "outs", torch.zeros(0, 4))
    capture._snap_tensor(7, "args[0]", capture.model.table)
    capture._snap_tensor(7, "args[1]", capture.model.table[2])
    assert capture._snap_pending == [] and capture._snap_index == []
    capture._snap_tensor(7, "args[2]", over)
    capture._snap_tensor(7, "args[3]", over[:32])
    view = capture.storage.view_of(over)
    assert capture._snap_index == [{"event": 7, "path": "args[2]", "view": view.to_json(), "dtype": "uint8", "shape": [40],
                                    "skipped": "spanned bytes 40 > snapshot_max_bytes 32"}]
    assert _taken(capture) == [(7, "args[3]", None, bytes(range(32)))] and capture._cur.n_snap == copied + 1


def test_the_limit_at_a_dispatcher_call_is_its_own_and_a_launch_keeps_the_whole_one(tmp_path, syncs):
    capture, runner = _in_step(tmp_path, snapshot_max_bytes=64, snapshot_max_bytes_op=32)
    big, small = torch.arange(40, dtype=torch.uint8), torch.arange(16, dtype=torch.uint8)
    capture._after_call(8, "aten.mul.Tensor", (big, [small, (big,)]), {"other": big, "alpha": 2}, (small, big))
    why = "spanned bytes 40 > snapshot_max_bytes 32 (snapshot_max_bytes_op: dispatcher-call event; Triton launch events keep 64)"
    assert [(x["event"], x["path"], x["skipped"]) for x in capture._snap_index] == [
        (8, "args[0]", why), (8, "args[1][1][0]", why), (8, "kwargs.other", why), (8, "outs[1]", why)]
    assert _taken(capture) == [(8, "args[1][0]", None, bytes(range(16))), (8, "outs[0]", None, bytes(range(16)))]
    capture._after_launch(9, "a kernel", {"x": big, "n": 3, "y": small, "names": [small]})
    assert _taken(capture)[2:] == [(9, "args.x", None, bytes(range(40))), (9, "args.y", None, bytes(range(16)))]
    assert len(capture._snap_index) == 4 and capture.errors == []
    capture._after_launch(10, "a kernel", {"x": torch.arange(65, dtype=torch.uint8)})
    assert capture._snap_index[4]["skipped"] == "spanned bytes 65 > snapshot_max_bytes 64" and len(capture._snap_pending) == 4


def test_a_block_of_a_calls_tensor_is_held_to_the_calls_limit(tmp_path, syncs):
    narrow, wide = torch.arange(64, dtype=torch.uint8).reshape(16, 4), torch.arange(192, dtype=torch.uint8).reshape(16, 12)

    def call(capture: Capture, runner: S._Runner) -> None:
        capture._after_call(1000, "aten.mul.Tensor", (narrow,), {}, wide)

    capture, taken, listed = _before_the_forward(tmp_path, call, snapshot_max_bytes=64, snapshot_max_bytes_op=8)
    assert taken == [(1000, "args[0]", {"dim": 0, "index": [4]}, bytes(range(16, 20))), (1000, "args[0]", {"dim": 0, "index": [6]}, bytes(range(24, 28)))]
    why = "spanned bytes 12 > snapshot_max_bytes 8 (snapshot_max_bytes_op: dispatcher-call event; Triton launch events keep 64)"
    assert [(x["event"], x["path"], x["shape"], x["skipped"]) for x in listed] == [(1000, "outs", [12], why)] * 2 and capture.errors == []


def test_a_tensor_over_the_limit_with_the_caches_block_count_is_copied_a_block_in_use_at_a_time(tmp_path, syncs):
    by_block = torch.arange(64, dtype=torch.uint8).reshape(16, 4)
    by_layer = torch.arange(128, dtype=torch.uint8).reshape(2, 16, 4)
    # the block count as a later dimension, and a block that is itself over the limit, are listed as skipped
    late, wide = torch.zeros(2, 2, 16, dtype=torch.uint8), torch.zeros(16, 33, dtype=torch.uint8)
    # nor is a block cut again where a dimension of its own is the block count
    square = torch.zeros(16, 16, 3, dtype=torch.uint8)

    def launches(capture: Capture, runner: S._Runner) -> None:
        for i, t in enumerate([by_block, by_layer, late, wide, square]):
            capture._after_launch(1000 + i, "a kernel", {"x": t})

    capture, taken, listed = _before_the_forward(tmp_path, launches, snapshot_max_bytes=32)
    assert capture._kv_blocks_in_step == [4, 6] and capture.errors == []
    assert taken[:2] == [(1000, "args.x", {"dim": 0, "index": [4]}, bytes(range(16, 20))), (1000, "args.x", {"dim": 0, "index": [6]}, bytes(range(24, 28)))]
    assert taken[2:] == [(1001, "args.x", {"dim": 1, "index": [layer, block]}, bytes(range(64 * layer + 4 * block, 64 * layer + 4 * block + 4)))
                         for layer in (0, 1) for block in (4, 6)]
    assert [(x["event"], x["shape"], x["skipped"], "sub" in x) for x in listed] == [
        (1002, [2, 2, 16], "spanned bytes 64 > snapshot_max_bytes 32", False),
        (1003, [33], "spanned bytes 33 > snapshot_max_bytes 32", False), (1003, [33], "spanned bytes 33 > snapshot_max_bytes 32", False),
        (1004, [16, 3], "spanned bytes 48 > snapshot_max_bytes 32", False), (1004, [16, 3], "spanned bytes 48 > snapshot_max_bytes 32", False)]
    # a block is listed by the bytes of the whole tensor's storage it spans
    runner = capture.runner
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert capture.close()["errors"] == []
    entries = [x for x in _index(tmp_path)["entries"] if x["event"] in (1000, 1001)]
    assert [(x["byte_offset"], x["nbytes"], x["shape"], x["sub"]["index"]) for x in entries] == [
        (16, 4, [4], [4]), (24, 4, [4], [6]), (16, 4, [4], [0, 4]), (24, 4, [4], [0, 6]), (80, 4, [4], [1, 4]), (88, 4, [4], [1, 6])]
    assert all((tmp_path / "snap" / x["file"]).read_bytes() == bytes(range(x["byte_offset"], x["byte_offset"] + 4)) for x in entries)


def test_with_no_block_in_use_or_no_block_count_a_tensor_over_the_limit_is_listed_as_skipped(tmp_path, syncs):
    over = torch.zeros(16, 4, dtype=torch.uint8)

    def none_in_use(capture: Capture, runner: S._Runner) -> None:
        runner.block_tables.num_blocks.np[:] = 0
        capture._after_launch(1000, "a kernel", {"x": over})

    capture, taken, listed = _before_the_forward(tmp_path / "none in use", none_in_use, snapshot_max_bytes=32)
    assert capture._kv_blocks_in_step == [] and taken == [] and capture.errors == []
    assert [(x["event"], x["skipped"]) for x in listed] == [(1000, "spanned bytes 64 > snapshot_max_bytes 32")]

    def no_count(capture: Capture, runner: S._Runner) -> None:
        capture._kv_num_blocks = None
        capture._after_launch(1000, "a kernel", {"x": over})

    capture, taken, listed = _before_the_forward(tmp_path / "no count", no_count, snapshot_max_bytes=32)
    assert capture._kv_blocks_in_step == [] and taken == [] and capture.errors == []
    assert [(x["event"], x["skipped"]) for x in listed] == [(1000, "spanned bytes 64 > snapshot_max_bytes 32")]


def test_a_tensor_with_the_caches_block_count_first_copied_after_the_forward_is_listed_as_skipped(tmp_path, syncs):
    """What the capture does today, and no requirement of it: `_after_forward` replaces the step's facts, the new ones name
    no scheduled request, and so the blocks in use, where the forward did not read them, are none after it.  The tensor is
    listed as skipped with its reason and none of it is copied; before the forward ends its blocks in use are copied."""
    capture, runner = _in_step(tmp_path, snapshot_max_bytes=32)
    assert capture._kv_num_blocks == 16 and capture._kv_blocks_in_step == []
    capture._after_launch(1000, "a kernel", {"x": torch.arange(64, dtype=torch.uint8).reshape(16, 4)})
    assert capture._kv_blocks_in_step == [] and _taken(capture) == [] and capture.errors == []
    assert [(x["event"], x["shape"], x["skipped"]) for x in capture._snap_index] == [(1000, [16, 4], "spanned bytes 64 > snapshot_max_bytes 32")]


def test_the_blocks_in_use_are_those_of_the_steps_requests_in_every_group_read_once_a_step(tmp_path, syncs):
    steps = [dict(S.STEPS[0], **{"more blocks": [{S.A: [13, 12, 4], S.B: []}]}), S.STEPS[1]]
    capture, runner = S._capture(tmp_path, ObserverConfig(triton=False, snapshot_steps=frozenset({0, 1}), snapshot_dir=tmp_path / "snap",
                                                          arena_bytes=1 << 16), steps, groups=([S.LAYER], ["sliding"]))
    assert capture._blocks_in_use() == [] and capture.errors == []
    read: list = []

    def first(batch: Any) -> None:
        blocks = capture._blocks_in_use()
        runner.block_tables.block_tables[0].gpu.numpy()[:] = 15
        read.extend([list(blocks), capture._blocks_in_use() is blocks])

    def second(batch: Any) -> None:
        runner.block_tables.num_kv_cache_groups = 0
        read.extend([list(capture._kv_blocks_in_step), capture._blocks_in_use()])

    runner.prepared = first
    scheduled = S._scheduled(0, steps)
    scheduled.num_scheduled_tokens = {S.A: 2, "a request the runner does not know": 1, S.B: 1}
    runner.execute_model(scheduled)
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert capture._kv_blocks_in_step == [4, 6, 12, 13]
    runner.prepared = second
    runner.execute_model(S._scheduled(1, steps))
    assert read == [[4, 6, 12, 13], True, [], [4, 6, 9]] and capture.errors == []


def test_blocks_that_cannot_be_read_are_none_and_an_error_of_the_capture(tmp_path, syncs):
    read: list = []

    def no_states(capture: Capture, runner: S._Runner) -> None:
        states, runner.req_states = runner.req_states, SimpleNamespace()
        read.append(capture._blocks_in_use())
        runner.req_states = states

    capture, taken, listed = _before_the_forward(tmp_path, no_states)
    assert read == [[]] and capture._kv_blocks_in_step == []
    assert capture.errors == [{"where": "blocks_in_use", "error": "AttributeError(\"'types.SimpleNamespace' object has no attribute 'req_id_to_index'\")"}]


def test_copies_that_fill_the_arena_are_written_before_it_is_used_again_and_one_larger_than_it_has_memory_of_its_own(tmp_path, syncs):
    capture, runner = _in_step(tmp_path, arena_bytes=300)
    tensors = [torch.full((n,), k, dtype=torch.uint8) for k, n in enumerate([400, 24, 24, 24, 400, 24])]
    waiting, before = [], len(syncs)
    for k, t in enumerate(tensors):
        capture._after_launch(1000 + k, "a kernel", {"x": t})
        waiting.append(([p["entry"]["event"] for p in capture._snap_pending], len(syncs) - before, capture._arena.off, capture._arena.overflow))
    # the first is larger than the arena and nothing waits, so nothing is written for it; each row is what waits, the times
    # the capture synchronised, and the arena's offset and its count of copies it had no room for
    assert waiting == [([1000], 0, 0, 1), ([1000, 1001], 0, 24, 1), ([1000, 1001, 1002], 0, 280, 1), ([1003], 1, 24, 1),
                       ([1004], 2, 0, 2), ([1004, 1005], 2, 24, 2)]
    assert [(x["event"], x["step"]) for x in capture._snap_index] == [(1000, 0), (1001, 0), (1002, 0), (1003, 0)]
    runner.sample("hidden", runner.execute_model_state.input_batch, 0)
    assert capture.close()["errors"] == []
    entries = [x for x in _index(tmp_path)["entries"] if x["event"] >= 1000]
    assert [(x["event"], (tmp_path / "snap" / x["file"]).read_bytes()) for x in entries] == [(1000 + k, t.numpy().tobytes()) for k, t in enumerate(tensors)]


def test_outside_a_listed_step_no_call_and_no_launch_is_copied(tmp_path, syncs, monkeypatch):
    capture, runner = _capture(tmp_path, listed=(1,))
    t = torch.arange(4, dtype=torch.uint8)
    monkeypatch.setattr(capture.storage, "view_of", lambda *a, **kw: pytest.fail("a tensor was looked at"))
    capture._after_call(1, "aten.mul.Tensor", (t,), {}, t)
    capture._after_launch(2, "a kernel", {"x": t})
    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda, "Event", S._Event)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: empty(*a, **kw))
    runner.execute_model(S._scheduled(0))
    monkeypatch.setattr(capture.storage, "view_of", lambda *a, **kw: pytest.fail("a tensor was looked at"))
    capture._after_call(3, "aten.mul.Tensor", (t,), {}, t)
    capture._after_launch(4, "a kernel", {"x": t})
    assert capture._arena is None and capture._snap_pending == [] and capture._snap_index == [] and capture.errors == []


def test_a_copy_that_fails_is_an_error_of_the_step_with_the_calls_id(tmp_path, syncs, monkeypatch):
    capture, runner = _in_step(tmp_path)
    t = torch.arange(4, dtype=torch.uint8)

    def no_view(*a, **kw):
        raise ValueError("no view of this tensor")

    monkeypatch.setattr(capture.storage, "view_of", no_view)
    capture._after_call(5, "aten.mul.Tensor", (t,), {}, t)
    capture._after_launch(6, "a kernel", {"x": t})
    assert capture.errors == [{"where": "snapshot", "step": 0, "id": 5, "error": "ValueError('no view of this tensor')"},
                              {"where": "snapshot-triton", "step": 0, "id": 6, "error": "ValueError('no view of this tensor')"}]
    assert capture._snap_pending == [] and capture._snap_index == []
