"""`verity_vllm.observe.log`: what a writer's flush leaves in its file, the line an event is there, and the group `split_steps` ends a
log with."""
from __future__ import annotations

import pytest

from verity_vllm.observe.events import Event, Note, RunHeader, StepEnd, StepStart
from verity_vllm.observe.log import LogWriter, header_of, read_log, split_steps


def _start(step: int) -> StepStart:
    return StepStart(step, [("r0", step)], [7], [step], {"r0": step + 1}, {"r0": step}, {"r0": [0]}, 16, [step], [("r0", step)], [0], 1, 1, 1)


def test_a_flush_leaves_every_event_written_so_far_in_the_file(tmp_path):
    path = tmp_path / "logs" / "capture.jsonl"
    with LogWriter(path) as w:
        w.write(Note("first"))
        w.flush()
        assert list(read_log(path)) == [Note("first")] and w.count == 1
        w.write(_start(0))
        w.flush()
        assert list(read_log(path)) == [Note("first"), _start(0)] and w.count == 2
    assert list(read_log(path)) == [Note("first"), _start(0)]


def test_an_event_is_one_compact_ascii_line_of_the_file(tmp_path):
    path = tmp_path / "capture.jsonl"
    with LogWriter(path) as w:
        w.write(Note("caf\u00e9 \u2192"))
        w.write(StepEnd(3))
    assert path.read_bytes() == b'{"$e":"Note","text":"caf\\u00e9 \\u2192","data":{}}\n{"$e":"StepEnd","step":3,"forward_s":0.0,"sample_s":0.0}\n'
    assert list(read_log(path)) == [Note("caf\u00e9 \u2192"), StepEnd(3)]


def test_split_steps_ends_with_the_events_that_follow_the_last_step_end():
    pre, tail = [Note("before")], [Note("after"), Note("last")]
    step: list[Event] = [_start(0), Note("inside"), StepEnd(0)]
    assert list(split_steps([*pre, *step])) == [(pre, step)]
    assert list(split_steps([*pre, *step, *tail])) == [(pre, step), (tail, [])], "a trailing preamble, with an empty step"
    open_step: list[Event] = [_start(1), Note("inside")]
    assert list(split_steps([*pre, *step, *tail, *open_step])) == [(pre, step), (tail, open_step)], "a step that never ended is the last group's"
    assert list(split_steps(open_step)) == [([], open_step)] and list(split_steps(tail)) == [(tail, [])]
    assert list(split_steps([])) == []


def test_header_of_is_the_logs_first_event_and_refuses_a_log_that_does_not_begin_with_one():
    header = RunHeader("verity-capture/log/v1", "2026-09-16T00:00:00Z", "pid", {}, {"repo": "toy"}, {"requests": []})
    assert header_of([header, Note("after")]) is header and header_of(iter([header])) is header
    for events in ([], [Note("before"), header], [_start(0), StepEnd(0)]):
        with pytest.raises(ValueError) as e:
            header_of(events)
        assert str(e.value) == "log does not start with a RunHeader"
    rest = iter([Note("before"), header, Note("after")])
    with pytest.raises(ValueError):
        header_of(rest)
    assert list(rest) == [header, Note("after")], "it reads the first event and no more"
