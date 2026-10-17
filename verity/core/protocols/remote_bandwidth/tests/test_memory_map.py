"""memory-map/v1, the record of KV blocks a serving node wrote and freed: its lines round-trip, `Replay` turns its steps
into the audit's appends, releases and plan changes, a session driven by it is accepted, and a map that breaks its
rules is refused with what to fix."""

from __future__ import annotations

import hashlib
import json

import pytest

from test_appends import N, Run, salts, store
from verity.core.protocols.remote_bandwidth import audit, codec, memory_map
from verity.core.protocols.remote_bandwidth.memory_map import Header, MemoryMapError, Replay, Step, Written


def page(slot: int, step: int) -> bytes:
    """The payload a step writes at a slot: 64-byte pieces with their top byte cleared, below p."""
    return b"".join(hashlib.sha512(b"kv %d at step %d/%d" % (slot, step, u)).digest()[:63] + b"\0"
                    for u in range(codec.BLOCK_BYTES // 64))


def written(step: int, slots: list[int], first: int) -> tuple[Written, ...]:
    return tuple(Written(s, first + u, codec.digest(page(s, step))) for u, s in enumerate(slots))


def serving(head: Header) -> list[Step]:
    """Three slots written at step 0; at step 3 slot 101 freed, and slots 103 and 101 written; at step 5 slots 100 and
    103 freed and slot 104 written; at step 6 slot 104 freed."""
    n = head.blocks
    return [Step(0, (), written(0, [100, 101, 102], n)),
            Step(3, (101,), written(3, [103, 101], n + 3)),
            Step(5, (100, 103), written(5, [104], n + 5)),
            Step(6, (104,), ())]


def test_a_memory_map_round_trips_through_its_lines() -> None:
    head = Header(bytes(range(64)), N)
    steps = serving(head)
    text = memory_map.dumps(head, steps)
    lines = text.splitlines()
    assert json.loads(lines[0]) == {"block_bytes": 4096, "blocks": N, "registration": bytes(range(64)).hex(),
                                    "schema": "verity/memory-map/v1"}
    assert json.loads(lines[2]) == {"freed": [101], "step": 3,
                                    "written": [[103, N + 3, codec.digest(page(103, 3)).hex()],
                                                [101, N + 4, codec.digest(page(101, 3)).hex()]]}
    parsed_head, parsed = memory_map.parse(text.splitlines(keepends=True) + ["\n"])
    assert parsed_head == head and list(parsed) == steps
    assert memory_map.dumps(parsed_head, steps) == text


def test_replay_turns_each_step_into_appends_releases_and_plan_changes() -> None:
    """Indices are the store's next fresh ones; a step's frees come before its writes, so step 3 rewrites slot 101 at a
    new index; a block appended and released between two passes is in neither side of the plan's change."""
    head = Header(bytes(64), N)
    r = Replay(head)
    c0 = r.apply(serving(head)[0])
    assert (c0.first, c0.count, c0.released) == (N, 3, ())
    assert r.take() == (((N, 3),), ())
    c3 = r.apply(serving(head)[1])
    assert (c3.first, c3.count, c3.released, r.live) == (N + 3, 2, ((N + 1, 1),), {100: N, 102: N + 2, 103: N + 3,
                                                                                    101: N + 4})
    c5 = r.apply(serving(head)[2])
    assert (c5.first, c5.count, c5.released) == (N + 5, 1, ((N, 1), (N + 3, 1)))
    c6 = r.apply(serving(head)[3])
    assert (c6.first, c6.count, c6.released, c6.append_tree(bytes(64), [])) == (N + 6, 0, ((N + 5, 1),), None)
    assert r.take() == (((N + 4, 1),), ((N, 2),))
    assert r.take() == ((), ()) and r.extent == N + 6 and r.live == {102: N + 2, 101: N + 4}
    tree = c3.append_tree(bytes(64), salts(b"s", 2))
    assert tree.commitment.root == audit.append_tree(bytes(64), N + 3, c3.digests, salts(b"s", 2)).commitment.root


def test_a_session_driven_by_a_memory_map_is_accepted() -> None:
    """The map drives both sides: the prover appends each step's pages (their digests the map's, at its indices) and
    sends the append's root and the released runs; the verifier takes them (`Session.append`, `Session.release`), and
    each pass's plan travels as `Replay.take`'s change.  Every receipt is drawn and every window accepted."""
    st = store()
    head = Header(st.registration.digest, N)
    run = Run(st)
    replay = Replay(head, st.registration)
    steps = serving(head)
    assert run.run() is None
    for k, s in enumerate(steps):
        change = replay.apply(s)
        if change.count:
            a = st.append([page(w.slot, s.step) for w in s.written], salts(b"page salts %d" % s.step, change.count))
            assert (a.first, a.digests) == (change.first, change.digests)
            assert run.s.append(a.root, a.count) == change.first
        if change.released:
            assert run.s.release(change.released) is None
        assert run.run(change=replay.take()) is None
        if k == 1:
            run.boundary()
    run.boundary()
    assert run.finish() == [audit.Verdict(0, True), audit.Verdict(1, True)]
    rec = run.s.record()
    assert (rec.appended, rec.released) == (6, 6 - 2)
    assert [w.post_seed for w in rec.windows] == [(0, 3, 4), (3, 2)]


@pytest.mark.parametrize("steps, match", [
    ([Step(0, (), ()), Step(0, (), ())], "strictly increase"),
    ([Step(1, (7,), ())], "frees slot 7, which no step wrote"),
    ([Step(0, (), (Written(7, N, bytes(64)),)), Step(1, (7, 7), ())], "frees a slot twice"),
    ([Step(0, (), (Written(7, N, bytes(64)),)), Step(1, (), (Written(7, N + 1, bytes(64)),))], "free it first"),
    ([Step(0, (), (Written(7, N, bytes(64)), Written(7, N + 1, bytes(64))))], "writes a slot twice"),
    ([Step(0, (), (Written(7, N + 1, bytes(64)),))], "not the store's next fresh one, 8"),
], ids=["order", "not-live", "freed-twice", "live", "written-twice", "index"])
def test_replay_refuses_a_map_that_breaks_its_rules(steps, match) -> None:
    r = Replay(Header(bytes(64), N))
    with pytest.raises(MemoryMapError, match=match):
        for s in steps:
            r.apply(s)


def test_a_refused_step_changes_nothing() -> None:
    r = Replay(Header(bytes(64), N))
    r.apply(Step(0, (), (Written(7, N, bytes(64)),)))
    with pytest.raises(MemoryMapError, match="next fresh"):
        r.apply(Step(1, (7,), (Written(8, N + 5, bytes(64)),)))
    assert (r.live, r.extent, r.last) == ({7: N}, N + 1, 0)


@pytest.mark.parametrize("line, match", [
    ({"schema": "verity/memory-map/v0", "registration": "00" * 64, "blocks": N, "block_bytes": 4096}, "schema"),
    ({"schema": memory_map.SCHEMA, "registration": "00" * 64, "blocks": N, "block_bytes": 8192}, "4096 bytes"),
    ({"schema": memory_map.SCHEMA, "registration": "AA" * 64, "blocks": N, "block_bytes": 4096}, "lowercase hex"),
    ({"schema": memory_map.SCHEMA, "registration": "00" * 64, "blocks": 0, "block_bytes": 4096}, "at least one"),
    ({"schema": memory_map.SCHEMA, "registration": "00" * 64, "blocks": N}, "first line"),
], ids=["schema", "block-bytes", "uppercase", "empty-store", "keys"])
def test_a_malformed_header_is_refused(line, match) -> None:
    with pytest.raises(MemoryMapError, match=match):
        memory_map.parse([json.dumps(line)])


@pytest.mark.parametrize("line, match", [
    ({"step": 0, "freed": [], "written": [[1, N, "zz" * 64]]}, "lowercase hex"),
    ({"step": 0, "freed": [], "written": [[1, N]]}, r"\[slot, index, digest\]"),
    ({"step": 0, "freed": [-1], "written": []}, "freed slot"),
    ({"step": 0, "freed": [1 << 63], "written": []}, "freed slot"),
    ({"step": True, "freed": [], "written": []}, "a step is an integer"),
    ({"step": 0, "freed": [], "written": [], "pages": []}, "step's line"),
], ids=["hex", "short", "negative", "slot-limit", "bool", "keys"])
def test_a_malformed_step_is_refused(line, match) -> None:
    head = json.dumps(Header(bytes(64), N).to_json())
    _, steps = memory_map.parse([head, json.dumps(line)])
    with pytest.raises(MemoryMapError, match=match):
        list(steps)


def test_a_map_beside_another_registration_is_refused() -> None:
    st = store()
    with pytest.raises(MemoryMapError, match="another registration"):
        Replay(Header(bytes(64), N), st.registration)
    with pytest.raises(MemoryMapError, match="another registration"):
        Replay(Header(st.registration.digest, N + 1), st.registration)
    with pytest.raises(MemoryMapError, match="header line"):
        memory_map.parse([])
