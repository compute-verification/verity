"""memory-map/v1's refresh re-key (`rekeyed`): a line carries it only when a re-key landed, `Replay` gives each re-key
its index and its origin (never an earlier re-key), and a re-key that breaks the format's rules is refused."""

from __future__ import annotations

import json

import pytest
from test_appends import N
from test_memory_map import page, written

from verity.protocols.resource_accounting.memory.poub.verifier import codec, memory_map
from verity.protocols.resource_accounting.memory.poub.verifier.memory_map import (
    Header,
    MemoryMapError,
    Rekeyed,
    Replay,
    Step,
    Written,
)


def test_a_line_carries_rekeyed_only_if_a_rekey_landed_at_its_step() -> None:
    head = Header(bytes(64), N)
    s = Step(2, (), written(2, [100], N), (), (Rekeyed(4, N + 1), Rekeyed(5, N + 2)))
    assert s.to_json() == {"freed": [], "rekeyed": [[4, N + 1], [5, N + 2]], "step": 2,
                           "written": [[100, N, codec.digest(page(100, 2)).hex()]]}
    assert "rekeyed" not in memory_map.dumps(head, [Step(2, (), written(2, [100], N))])
    assert list(memory_map.parse(memory_map.dumps(head, [s]).splitlines())[1]) == [s]
    bare = json.dumps({"freed": [], "rekeyed": [], "step": 3, "written": []})
    assert list(memory_map.parse([json.dumps(head.to_json()), bare])[1]) == [Step(3, (), ())]


def test_a_rekey_moves_its_slot_to_a_fresh_index_whose_origin_is_where_the_payload_was_first_committed() -> None:
    """Step 0 writes slot 100 at N.  Step 1 writes slot 101 at N + 2 and re-keys registered slots 4 and 5 and slot 100
    at N + 1, N + 3 and N + 4: writes and re-keys share the step's fresh indices.  Step 4 re-keys slot 4 again; its
    origin is still registered block 4, never the re-key at N + 1."""
    head = Header(bytes(64), N)
    r = Replay(head)
    r.apply(Step(0, (), written(0, [100], N)))
    assert r.take(1) == (((N, 1),), ())
    c = r.apply(Step(1, (), (Written(101, N + 2, codec.digest(page(101, 1))),), (),
                     (Rekeyed(4, N + 1), Rekeyed(5, N + 3), Rekeyed(100, N + 4))))
    assert (c.first, c.count, c.indices, c.released) == (N + 1, 1, (N + 2,), ((4, 2), (N, 1)))
    assert c.rekeyed == ((N + 1, 4), (N + 3, 5), (N + 4, N))
    assert [r.holds(s) for s in (4, 5, 100, 101)] == [N + 1, N + 3, N + 4, N + 2]
    assert r.take(2) == (((N + 1, 4),), ((4, 2), (N, 1)))
    with pytest.raises(MemoryMapError, match="records a re-key"):
        c.append_tree(bytes(64), [])
    c = r.apply(Step(4, (), (), (), (Rekeyed(4, N + 5),)))
    assert (c.first, c.count, c.released, c.rekeyed) == (N + 5, 0, ((N + 1, 1),), ((N + 5, 4),))
    assert r.take(5) == (((N + 5, 1),), ((N + 1, 1),)) and r.extent == N + 6


def test_the_schedule_frees_a_rekeyed_block_and_writes_its_rekey_at_the_step_it_landed() -> None:
    head = Header(bytes(64), N)
    steps = [Step(0, (), written(0, [100], N)), Step(2, (), (), (), (Rekeyed(4, N + 1), Rekeyed(100, N + 2)))]
    sched = memory_map.schedule(head, steps, [(N, 1), (N + 1, 1), (N + 2, 1)], [1, 3])
    assert sched.registered == ((4, 1, 2),)
    assert sched.appends == (((0, 2),), ((2, None),), ((2, None),))
    assert sched.plan(N, 3) == ((0, 4), (5, N - 5), (N + 1, 2))


@pytest.mark.parametrize(("rekeyed", "freed", "match"), [
    ((Rekeyed(N + 7, N),), (), "holds no block"),
    ((Rekeyed(4, N),), (4,), "also frees"),
    ((Rekeyed(4, N), Rekeyed(4, N + 1)), (), "also frees"),
    ((Rekeyed(4, N + 1), Rekeyed(5, N)), (), "increasing order"),
    ((Rekeyed(4, N + 1),), (), "increasing order"),
])
def test_replay_refuses_a_rekey_that_breaks_the_format(rekeyed, freed, match) -> None:
    r = Replay(Header(bytes(64), N))
    with pytest.raises(MemoryMapError, match=match):
        r.apply(Step(0, freed, (), (), rekeyed))
    assert (r.extent, r.last, r.holds(4), r.origin) == (N, -1, 4, {})


def test_a_rekey_cannot_take_a_withdrawn_index_or_a_written_one() -> None:
    r = Replay(Header(bytes(64), N))
    with pytest.raises(MemoryMapError, match="none withdrawn"):
        r.apply(Step(0, (), (), (N,), (Rekeyed(4, N),)))
    with pytest.raises(MemoryMapError, match="not the store's next fresh one"):
        r.apply(Step(0, (), (Written(100, N, codec.digest(page(100, 0))),), (), (Rekeyed(4, N),)))


@pytest.mark.parametrize(("entry", "match"), [([4], r"\[slot, index\]"), ([4, -1], "store index"),
                                              ("4", r"\[slot, index\]")])
def test_a_malformed_rekey_is_refused(entry, match) -> None:
    line = json.dumps({"freed": [], "rekeyed": [entry], "step": 0, "written": []})
    with pytest.raises(MemoryMapError, match=match):
        memory_map.step(json.loads(line))
