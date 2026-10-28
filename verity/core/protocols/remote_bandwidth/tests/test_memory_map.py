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
    """Registered blocks 4 … 7 are a KV pool's pages, registered as filler; slots 100 and up are memory outside the
    store.  Step 0 writes slots 4, 5 and 100; step 3 frees slot 100 and writes slots 6 and 4 (4 again, over the block
    step 0 wrote there); step 5 frees slot 5 and writes slot 101; step 6 frees slot 101."""
    n = head.blocks
    return [Step(0, (), written(0, [4, 5, 100], n)),
            Step(3, (100,), written(3, [6, 4], n + 3)),
            Step(5, (5,), written(5, [101], n + 5)),
            Step(6, (101,), ())]


def test_a_memory_map_round_trips_through_its_lines() -> None:
    head = Header(bytes(range(64)), N)
    steps = serving(head)
    text = memory_map.dumps(head, steps)
    lines = text.splitlines()
    assert json.loads(lines[0]) == {"block_bytes": 4096, "blocks": N, "registration": bytes(range(64)).hex(),
                                    "schema": "verity/memory-map/v1"}
    assert json.loads(lines[2]) == {"freed": [100], "step": 3,
                                    "written": [[6, N + 3, codec.digest(page(6, 3)).hex()],
                                                [4, N + 4, codec.digest(page(4, 3)).hex()]]}
    parsed_head, parsed = memory_map.parse(text.splitlines(keepends=True) + ["\n"])
    assert parsed_head == head and list(parsed) == steps
    assert memory_map.dumps(parsed_head, steps) == text


def test_a_line_carries_withdrawn_only_if_its_step_withdraws_a_block() -> None:
    """`withdrawn` is optional, empty by default: a map without withdrawals is the lines it was before the key, and a
    line with an empty list reads as one without."""
    head = Header(bytes(64), N)
    s = Step(3, (), written(1, [4], N), (N + 1,))
    assert s.to_json() == {"freed": [], "step": 3, "withdrawn": [N + 1],
                           "written": [[4, N, codec.digest(page(4, 1)).hex()]]}
    assert "withdrawn" not in memory_map.dumps(head, serving(head))
    assert list(memory_map.parse(memory_map.dumps(head, [s]).splitlines())[1]) == [s]
    bare = json.dumps({"freed": [], "step": 3, "withdrawn": [], "written": []})
    assert list(memory_map.parse([json.dumps(head.to_json()), bare])[1]) == [Step(3, (), ())]


def test_a_withdrawn_index_takes_its_place_among_its_line_s_fresh_ones_and_joins_no_plan() -> None:
    """Step 1 completes blocks in slots 4, 5 and 100 (4 and 5 registered filler, freed at step 1, which reached them),
    sent as one append, N … N + 2.  The request holding slot 5 ends before the append's encode lands, in step 3's
    line, so that line writes slots 4 and 100 at N and N + 2 and withdraws N + 1: no write, no free.  The release
    covers N + 1 and names it withdrawn, no plan's change adds or removes it, and the step has no append tree of its
    own.  A line may withdraw blocks and write none."""
    head = Header(bytes(64), N)
    r = Replay(head)
    assert r.apply(Step(1, (4, 5), ())).released == ((4, 2),)
    assert r.take(2) == ((), ((4, 2),))
    c = r.apply(Step(3, (), (Written(4, N, codec.digest(page(4, 1))), Written(100, N + 2, codec.digest(page(100, 1)))),
                     (N + 1,)))
    assert (c.first, c.count, c.indices, c.released, c.withdrawn) == (N, 2, (N, N + 2), ((N + 1, 1),), ((N + 1, 1),))
    assert (r.extent, r.holds(4), r.holds(5), r.holds(100)) == (N + 3, N, None, N + 2)
    assert r.take(4) == (((N, 1), (N + 2, 1)), ())
    with pytest.raises(MemoryMapError, match="withdraws a block"):
        c.append_tree(bytes(64), salts(b"s", 2))
    c = r.apply(Step(5, (), (), (N + 3, N + 4)))
    assert (c.first, c.count, c.indices, c.released, c.withdrawn) == (N + 3, 0, (), ((N + 3, 2),), ((N + 3, 2),))
    assert r.take(6) == ((), ()) and r.extent == N + 5


def test_replay_turns_each_step_into_appends_releases_and_plan_changes() -> None:
    """Indices are the store's next fresh ones.  A write releases the block its slot held (registered filler, or a
    block written before), a free releases its slot's block and leaves the slot empty, and a block appended and
    released between two passes is in neither side of the plan's change."""
    head = Header(bytes(64), N)
    r = Replay(head)
    assert [r.holds(s) for s in (0, 7, N, 100)] == [0, 7, None, None]
    c0 = r.apply(serving(head)[0])
    assert (c0.first, c0.count, c0.released) == (N, 3, ((4, 2),))
    assert r.take(1) == (((N, 3),), ((4, 2),))
    c3 = r.apply(serving(head)[1])
    assert (c3.first, c3.count, c3.released) == (N + 3, 2, ((6, 1), (N, 1), (N + 2, 1)))
    assert [r.holds(s) for s in (4, 5, 6, 7, 100)] == [N + 4, N + 1, N + 3, 7, None]
    c5 = r.apply(serving(head)[2])
    assert (c5.first, c5.count, c5.released) == (N + 5, 1, ((N + 1, 1),))
    c6 = r.apply(serving(head)[3])
    assert (c6.first, c6.count, c6.released, c6.append_tree(bytes(64), [])) == (N + 6, 0, ((N + 5, 1),), None)
    assert r.take(7) == (((N + 3, 2),), ((6, 1), (N, 3)))
    assert r.take(8) == ((), ()) and r.extent == N + 6
    assert [r.holds(s) for s in (4, 5, 6, 7, 100, 101)] == [N + 4, None, N + 3, 7, None, None]
    tree = c3.append_tree(bytes(64), salts(b"s", 2))
    assert tree.commitment.root == audit.append_tree(bytes(64), N + 3, c3.digests, salts(b"s", 2)).commitment.root


def test_a_session_driven_by_a_memory_map_is_accepted() -> None:
    """The map drives both sides: the prover appends each step's pages (their digests the map's, at its indices) and
    sends the append's root and the released runs; the verifier takes them (`AuditRun.append`, `AuditRun.release`), and
    each pass's plan travels as `Replay.take`'s change for the step it names, the one after the step just applied.
    Every receipt is drawn and every window accepted, and the record keeps each pass's step."""
    st = store()
    head = Header(st.registration.digest, N)
    run = Run(st)
    replay = Replay(head, st.registration)
    steps = serving(head)
    assert run.run(change=replay.take(0), step=0) is None
    for k, s in enumerate(steps):
        change = replay.apply(s)
        if change.count:
            a = st.append([page(w.slot, s.step) for w in s.written], salts(b"page salts %d" % s.step, change.count))
            assert (a.first, a.digests) == (change.first, change.digests)
            assert run.commit(a.root, a.count) == change.first
        if change.released:
            assert run.s.release(change.released) is None
        assert run.run(change=replay.take(s.step + 1), step=s.step + 1) is None
        if k == 1:
            run.boundary()
    run.boundary()
    assert run.finish() == [audit.Verdict(0, True), audit.Verdict(1, True)]
    rec = run.s.record()
    assert [w.lengths for w in rec.windows] == [(8, 9, 8), (8, 7)]
    assert [w.post_seed for w in rec.windows] == [(0, 3, 3), (3, 2)]
    assert [w.steps for w in rec.windows] == [(0, 1, 4), (6, 7)]
    assert rec.profile()["windows"][1]["steps"] == [6, 7]
    assert (rec.appended, rec.released) == (6, 7)


def test_a_registered_block_freed_at_its_values_last_reader_leaves_every_later_plan() -> None:
    """A free of a slot below N releases its registered block: the writer records it at the block's values' last
    structural reader (a weight only the first steps read), so no later pass holds it and none credits it.
    The verifier takes the release, and a plan that still holds the block is refused (`plan`)."""
    st = store()
    replay, run = Replay(Header(st.registration.digest, N), st.registration), Run(st)
    assert run.run(change=replay.take(0), step=0) is None
    change = replay.apply(Step(0, (2,), ()))
    assert (change.count, change.released, replay.holds(2)) == (0, ((2, 1),), None)
    assert run.s.release(change.released) is None
    assert replay.take(1) == ((), ((2, 1),))
    assert run.run(change=((), ((2, 1),)), step=1) is None
    assert run.plan.runs == ((0, 2), (3, 5))
    assert run.s.receive(0, 2, bytes(64), change=(((2, 1),), ()), step=2).failure == "plan"


def test_a_pass_covers_the_blocks_resident_at_its_step() -> None:
    """A step's record takes effect after the step's reads: the pass at step 3 holds what steps 0 … 2 left, step 3's
    own frees and writes are the next pass's, and a pass whose step is not after every step applied, or after the
    last pass's, is refused, as is a record of a step before a pass already taken."""
    head = Header(bytes(64), N)
    r = Replay(head)
    steps = serving(head)
    r.apply(steps[0])
    assert r.take(3) == (((N, 3),), ((4, 2),))
    r.apply(steps[1])
    for bad in (3, 2, -1, True, audit.STEP_LIMIT):
        with pytest.raises(MemoryMapError, match="take it before applying"):
            r.take(bad)
    assert r.take(4) == (((N + 3, 2),), ((6, 1), (N, 1), (N + 2, 1)))
    r = Replay(head)
    assert r.take(2) == ((), ())
    with pytest.raises(MemoryMapError, match="apply each step's record before the pass after it"):
        r.apply(steps[0])
    assert r.apply(Step(2, (), ())).step == 2


def test_a_block_whose_write_is_recorded_two_steps_after_it_completed_is_resident_from_the_pass_after() -> None:
    """Deferred residency: step 1 completes a block in slot 4 (registered filler, freed at step 1, which reached it),
    and its append goes out at once; the device encodes it in place once the coin is back, and step 4 is the first
    whose attention reads it encoded, so its write is recorded at step 3, two steps after it was completed.  Until then
    it is in no record: the passes at steps 2 and 3 don't hold it, and the pass at step 4 adds it.  The verifier takes
    the append before the record writes the block, the record's write names the append's index and digest, and every
    pass is accepted with every receipt drawn."""
    st = store()
    head = Header(st.registration.digest, N)
    replay, run = Replay(head, st.registration), Run(st)
    assert run.run(change=replay.take(0), step=0) is None
    freed = replay.apply(Step(1, (4,), ()))
    assert (freed.count, freed.released, replay.holds(4)) == (0, ((4, 1),), None)
    assert run.s.release(freed.released) is None
    a = st.append([page(4, 1)], salts(b"deferred page", 1))
    assert run.commit(a.root, a.count) == N
    assert replay.take(2) == ((), ((4, 1),))
    assert run.run(change=((), ((4, 1),)), step=2) is None
    assert replay.take(3) == ((), ())
    assert run.run(change=((), ()), step=3) is None
    wrote = replay.apply(Step(3, (), written(1, [4], N)))
    assert (wrote.first, wrote.digests, wrote.released) == (a.first, a.digests, ())
    assert replay.holds(4) == N
    assert replay.take(4) == (((N, 1),), ())
    assert run.run(change=(((N, 1),), ()), step=4) is None
    run.boundary()
    assert run.run(change=replay.take(5), step=5) is None
    run.boundary()
    assert run.finish() == [audit.Verdict(0, True), audit.Verdict(1, True)]
    rec = run.s.record()
    assert [w.steps for w in rec.windows] == [(0, 2, 3, 4), (5,)]
    assert [w.lengths for w in rec.windows] == [(8, 7, 7, 8), (8,)]
    assert [w.post_seed for w in rec.windows] == [(0, 0, 0, 1), (1,)]
    assert (rec.appended, rec.released) == (1, 1)


@pytest.mark.parametrize("steps, match", [
    ([Step(0, (), ()), Step(0, (), ())], "strictly increase"),
    ([Step(1, (100,), ())], "frees slot 100, which holds no block"),
    ([Step(0, (7,), ()), Step(1, (7,), ())], "frees slot 7, which holds no block"),
    ([Step(0, (7, 7), ())], "frees a slot twice"),
    ([Step(0, (), (Written(7, N, bytes(64)), Written(7, N + 1, bytes(64))))], "writes a slot twice"),
    ([Step(0, (), (Written(7, N + 1, bytes(64)),))], "not the store's next fresh one, 8"),
    ([Step(0, (), (), (N + 1,))], r"withdraws \[9\]"),
    ([Step(0, (), (), (N + 1, N))], "in increasing order"),
    ([Step(0, (), (), (N, N))], "in increasing order"),
    ([Step(0, (), (Written(7, N, bytes(64)),), (N,))], "not the store's next fresh one, 9"),
    ([Step(0, (), (), (N,)), Step(1, (), (Written(7, N, bytes(64)),))], "not the store's next fresh one, 9"),
], ids=["order", "outside", "freed", "freed-twice", "written-twice", "index", "withdrawn-outside", "withdrawn-order",
        "withdrawn-twice", "written-withdrawn", "written-after-withdrawn"])
def test_replay_refuses_a_map_that_breaks_its_rules(steps, match) -> None:
    r = Replay(Header(bytes(64), N))
    with pytest.raises(MemoryMapError, match=match):
        for s in steps:
            r.apply(s)


def test_a_refused_step_changes_nothing() -> None:
    r = Replay(Header(bytes(64), N))
    r.apply(Step(0, (), (Written(100, N, bytes(64)),)))
    with pytest.raises(MemoryMapError, match="next fresh"):
        r.apply(Step(1, (100,), (Written(3, N + 5, bytes(64)),)))
    with pytest.raises(MemoryMapError, match="withdraws"):
        r.apply(Step(1, (100,), (Written(3, N + 2, bytes(64)),), (N + 1, N + 4)))
    assert (r.moved, r.extent, r.last, r.holds(3)) == ({100: N}, N + 1, 0, 3)


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
    ({"step": 0, "freed": [], "written": [], "withdrawn": 3}, "are lists"),
    ({"step": 0, "freed": [], "written": [], "withdrawn": [-1]}, "withdrawn index"),
    ({"step": 0, "freed": [], "written": [], "withdrawn": [True]}, "withdrawn index"),
], ids=["hex", "short", "negative", "slot-limit", "bool", "keys", "withdrawn-list", "withdrawn-negative",
        "withdrawn-bool"])
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
