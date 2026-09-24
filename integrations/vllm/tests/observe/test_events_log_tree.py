"""Contract tests for the torch-free half: event (de)serialisation, log round trip, event trees."""

from __future__ import annotations

import io

import pytest

from verity_vllm.observe.events import (DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepLink, StepStart,
                                   StorageAlloc, TritonLaunch, ViewDesc, from_json, to_json, views_in)
from verity_vllm.observe.log import LogWriter, dumps, read_log, read_log_lines, split_steps
from verity_vllm.observe.tree import build_trees, op_histogram, step_events


def _view(a=1, off=0, shape=(4, 576), strides=(576, 1), dtype="bfloat16"):
    return ViewDesc(a, off, shape, strides, dtype, "cuda:0")


def _sample_events():
    hdr = RunHeader("verity-capture/log/v1", "2026-09-16T00:00:00Z", "pid", {"vllm": "d9105ea80"}, {"repo": "SmolLM2"},
                    {"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2}]})
    w = _view(a=7, shape=(960, 576))
    evs = [hdr,
           StorageAlloc(7, 0x1000, 960 * 576 * 2, "cuda:0", 0, "param"),
           Root(w, "weight", "model.layers.0.self_attn.qkv_proj.weight"),
           StorageAlloc(1, 0x9000, 4 * 576 * 2, "cuda:0", 0, "op"),
           StepStart(0, [("r0", 0), ("r0", 1), ("r0", 2)], [1, 2, 3], [0, 1, 2], {"r0": 3}, {"r0": 0}, {"r0": [0]}, 16,
                     [0, 1, 2], [("r0", 2)], [2], 1, 3, 3, _view(a=3, shape=(3,), strides=(1,), dtype="int32"), None, "prefill"),
           DispatchEnter(10, None, 0, "forward", "vllm.unified_attention_with_output.default", "sig", [_view()], {"k": None}, 7, "layers.0.self_attn"),
           DispatchEnter(11, 10, 0, "forward", "_C.reshape_and_cache_flash.default", "sig", [_view(), _view(a=2)], {}, 7),
           DispatchExit(11, None),
           TritonLaunch(12, 10, 0, "forward", "vllm.model_executor.layers.batch_invariant.matmul_kernel_persistent", "abc",
                        {"a_ptr": _view(), "BLOCK_K": 32}, {"BLOCK_K": 32}, [4, 1, 1], 8, 3, 7),
           DispatchExit(10, [_view(a=1, shape=(3, 9, 64), strides=(576, 64, 1))]),
           DispatchEnter(13, None, 0, "sample", "aten.argmax.default", "sig", [_view(a=5, shape=(1, 49152), strides=(49152, 1))], {"dim": -1}, 7),
           DispatchExit(13, _view(a=6, shape=(1,), strides=(1,), dtype="int64")),
           SampleEnd(0, _view(a=5, shape=(1, 49152), strides=(49152, 1)), [42]),
           StepEnd(0, 0.01, 0.001),
           StepStart(1, [("r0", 3)], [42], [3], {"r0": 4}, {"r0": 3}, {"r0": [0]}, 16, [3], [("r0", 3)], [0], 1, 1, 1, None, None, "decode"),
           StepLink(1, 0, "r0", 0, "eq", 42),
           StepEnd(1)]
    return evs


def test_json_roundtrip_every_event():
    for ev in _sample_events():
        assert from_json(to_json(ev)) == ev


def test_viewdesc_helpers():
    v = _view()
    assert v.numel == 4 * 576 and v.itemsize == 2 and v.bits == 16 and v.byte_offset == 0
    with pytest.raises(ValueError):
        ViewDesc(1, 0, (2, 3), (3,), "bfloat16", "cuda:0")
    with pytest.raises(ValueError):
        ViewDesc(1, 0, (2,), (1,), "weird", "cuda:0")
    assert list(views_in([v, {"x": [None, v]}, 3])) == [v, v]


def test_log_roundtrip_file(tmp_path):
    evs = _sample_events()
    p = tmp_path / "log.jsonl.gz"
    with LogWriter(p) as w:
        for ev in evs:
            w.write(ev)
        assert w.count == len(evs)
    back = list(read_log(p))
    assert back == evs
    assert list(read_log_lines(io.StringIO(dumps(evs)))) == evs


def test_split_steps_and_trees():
    evs = _sample_events()
    groups = list(split_steps(evs))
    assert len(groups) == 2
    pre, s0 = groups[0]
    assert [type(e).__name__ for e in pre] == ["RunHeader", "StorageAlloc", "Root", "StorageAlloc"]
    se = step_events(s0)
    assert se.start.step == 0 and se.sample.sampled_token_ids == [42] and se.end is not None
    assert [r.op for r in se.roots] == ["vllm.unified_attention_with_output.default", "aten.argmax.default"]
    attn = se.roots[0]
    assert [c.op for c in attn.children] == ["_C.reshape_and_cache_flash.default",
                                             "vllm.model_executor.layers.batch_invariant.matmul_kernel_persistent"]
    assert attn.children[1].is_triton and attn.ids() == {10, 11, 12}
    assert [v.alloc for v in attn.output_views()] == [1]
    assert op_histogram(se.roots)["_C.reshape_and_cache_flash.default"] == 1
    pre1, s1 = groups[1]
    se1 = step_events(s1)
    assert se1.links[0].value == 42 and se1.roots == []


def test_build_trees_rejects_dangling_parent():
    from verity_vllm.observe.tree import TreeError
    with pytest.raises(TreeError):
        build_trees([DispatchEnter(5, 99, 0, "forward", "aten.mm.default", "s", [], {}, 0)])
