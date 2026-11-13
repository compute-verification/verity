"""The tick an execution phase's head leaves at (`WardenParams.head_offset`, `WardenParams.head_wait`, `warden`'s
"The head's tick"), driven tick by tick on a `VirtualClock`. A schedule sets the two together or neither. With them
whoever takes an execution phase's hand hands its head back to the network certifier, which publishes it at a tick
of the schedule: the due one for a head back by then, never sooner; the first it is back by for a late one, within
W ticks of the due one; and at no tick for one later still, whose execution phase has no head. A schedule without
them has the digest and the run it had."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from verity.protocols.accounting.certifier.verifier.active import VirtualClock
from verity.protocols.accounting.certifier.verifier.tests.test_network_certifier_lengths import LEN, T
from verity.protocols.accounting.certifier.verifier.tests.test_network_certifier_warden import TAU
from verity.protocols.accounting.certifier.verifier.warden import OPTIONAL_FIELDS, Warden, WardenParams

#: the lengths tests' schedule under form 2, and the same with a head due 3 buckets after its hand's close and no
#: tick to wait, then one
TWO = dataclasses.replace(LEN, lengths=2)
SET = dataclasses.replace(TWO, head_offset=3, head_wait=0)
WAIT = dataclasses.replace(SET, head_wait=1)
#: a run's execution phases, and the tick each one's head is due at under SET: the first is handed at the close of
#: the second, the last two at the run's last close
PHASES = 3
DUE = (2 * T - 1 + 3, 3 * T - 1 + 3, 3 * T - 1 + 3)


async def drive(params: WardenParams, back: dict[int, int | None] | None = None, *, phases: int = PHASES,
                late: dict[int, float] | None = None):
    """A run of `phases` execution phases without a connection, each tick at its boundary (`late[t]` seconds past it
    for tick t). Whoever takes a hand is this driver: it hands execution phase w's head back just before tick
    `back[w]` runs, never where that is None, and at the hand itself, as a publisher in the network certifier's own
    process does, where `back` doesn't name w. Returns the device, the hands as (tick, execution phase, strides),
    what left as (tick, execution phase, the tick `leave` was given, head), and what each `head_back` returned."""
    back = back or {}
    clock = VirtualClock(-TAU / 2)
    hands, left, returned = [], [], {}

    def lengths(w, strides) -> None:
        hands.append((device.ticks, w, strides))
        if w not in back and params.head_offset is not None:
            returned[w] = device.head_back(w, f"head {w}")

    device = Warden(params, None, origin=0.0, windows=phases, clock=clock, sleep=clock.sleep, lengths=lengths,
                    leave=lambda w, tick, head: left.append((device.ticks, w, tick, head)))
    running = asyncio.create_task(device.run())
    running.add_done_callback(lambda _: device._changed.set())
    t = 0
    while not running.done():
        for w, at in back.items():
            if at == t:
                returned[w] = device.head_back(w, f"head {w}")
        clock.advance(t * TAU + (late or {}).get(t, 0))
        await device.until(lambda d, t=t: d.ticks > t or running.done())
        t += 1
    await running
    return device, hands, left, returned


def run(params: WardenParams, back=None, **kw):
    return asyncio.run(drive(params, back, **kw))


# --- the schedule ---------------------------------------------------------------------------------------------------


def test_the_head_offset_and_the_wait_are_parameters_no_other_schedule_hashes():
    """Neither field has a default, and a schedule that sets neither has the digest it had. A schedule sets the
    offset, in buckets, and the wait, in ticks, together: each enters the digest, the wait at 0 as at any other
    value. They are the lengths rows' head's, so a schedule without lengths rows has none, under either form it
    may; and a value that is no count of buckets or of ticks is refused, naming the field."""
    assert WardenParams().head_offset is None and WardenParams().head_wait is None
    assert dataclasses.replace(SET, head_offset=None, head_wait=None) == TWO
    assert dataclasses.replace(SET, head_offset=None, head_wait=None).digest == TWO.digest
    assert OPTIONAL_FIELDS[-2:] == ("head_offset", "head_wait")
    others = [dataclasses.replace(TWO, head_offset=o, head_wait=w) for o, w in ((3, 0), (3, 1), (4, 0), (1, 3), (3, T))]
    assert len({TWO.digest, *(p.digest for p in others)}) == len(others) + 1 and others[:2] == [SET, WAIT]
    assert WardenParams(**json.loads(json.dumps(dataclasses.asdict(WAIT)))) == WAIT
    assert dataclasses.replace(LEN, head_offset=3, head_wait=0).digest not in (LEN.digest, SET.digest)
    with pytest.raises(ValueError, match="a schedule with head_offset has lengths rows"):
        dataclasses.replace(LEN, lengths=None, head_offset=3, head_wait=0)
    for bad in (0, -1, True, 2.0, "3", 1 << 32):
        with pytest.raises(ValueError, match="head_offset is a positive u32"):
            dataclasses.replace(TWO, head_offset=bad, head_wait=0)
    for bad in (-1, False, 1.0, "0", 1 << 32):
        with pytest.raises(ValueError, match="head_wait is a u32"):
            dataclasses.replace(TWO, head_offset=3, head_wait=bad)


def test_a_schedule_that_sets_the_wait_and_no_offset_is_refused_by_name_and_so_is_the_offset_alone():
    """The wait counts ticks from the due one, so a schedule that sets it and no offset is refused, naming both
    fields. One that sets the offset and no wait is refused too: the wait has no default, 0 included."""
    for wait in (0, 1, 7):
        with pytest.raises(ValueError, match=f"head_wait \\({wait}\\) is set and head_offset is not: the ticks a head "
                                             f"may wait are counted from the tick it is due at"):
            dataclasses.replace(TWO, head_wait=wait)
    with pytest.raises(ValueError, match="head_wait \\(0\\) is set and head_offset is not"):
        WardenParams(head_wait=0)
    with pytest.raises(ValueError, match="head_offset \\(3\\) is set and head_wait is not: a head not back at its "
                                         "tick leaves within head_wait ticks of it or never"):
        dataclasses.replace(TWO, head_offset=3)


def test_a_heads_tick_is_the_offset_past_the_close_that_hands_its_execution_phase():
    """An execution phase's head is due `head_offset` buckets after the last bucket of the window whose close hands
    it: the next window's, and for the run's last execution phase the run's last, so a run's last two are due at
    one tick, past the run's last bucket. It is a function of the schedule, the execution phase and the run's
    length alone. A schedule without an offset has no such tick."""
    assert tuple(SET.head_tick(w, PHASES) for w in range(PHASES)) == DUE == (34, 50, 50)
    assert [WAIT.head_tick(w, 5) for w in range(5)] == [34, 50, 66, 82, 82] and SET.head_tick(0, 1) == T - 1 + 3
    assert dataclasses.replace(SET, head_offset=1).head_tick(0, 2) == 2 * T
    for w, phases in ((3, 3), (-1, 3), (0, 0), (True, 3), (0, 2.0)):
        with pytest.raises(ValueError, match="is not one of a run of"):
            SET.head_tick(w, phases)
    with pytest.raises(ValueError, match="the schedule has no head offset"):
        TWO.head_tick(0, PHASES)


# --- the device -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("params", [SET, WAIT, dataclasses.replace(SET, head_offset=1)])
def test_a_head_back_before_its_tick_waits_for_it_and_leaves_at_it_never_sooner(params):
    """Each execution phase's head is handed back at its hand, in the tick that hands it, or some ticks later and
    still before its own. It leaves at its due tick, and at no tick before: the hand's tick, the ticks between and
    the tick before the due one publish nothing. The run's last two heads are due past its last bucket, and the run
    ticks on, with no cell, until they have left. Whatever the wait, a head back in time leaves at the due tick."""
    due = [params.head_tick(w, PHASES) for w in range(PHASES)]
    for back in ({}, {0: due[0] - params.head_offset + 1, 1: due[1], 2: due[2]}):
        device, hands, left, returned = run(params, back)
        assert [(t, w) for t, w, _ in hands] == [(2 * T - 1, 0), (3 * T - 1, 1), (3 * T - 1, 2)]
        assert all(strides is not None and not any(strides) for _, _, strides in hands)
        assert left == [(due[w], w, due[w], f"head {w}") for w in range(PHASES)]
        assert device.heads == [(w, due[w]) for w in range(PHASES)] and returned == dict.fromkeys(range(PHASES), True)
        assert device.ticks == len(device.lateness) == due[-1] + 1 and max(device.lateness) == 0
        assert len(device.records) == 3 * PHASES


def test_a_head_back_one_tick_late_leaves_at_the_next_tick_within_the_wait():
    """At a wait of one tick, a head that is back after its due tick has run and before the next leaves at the
    next, and the run goes on to that tick for its last execution phases. The due tick itself publishes nothing of
    it, and a head of another execution phase that is back in time leaves at its own tick."""
    back = {0: DUE[0] + 1, 2: DUE[2] + 1}
    device, _, left, returned = run(WAIT, back)
    assert left == [(DUE[0] + 1, 0, DUE[0] + 1, "head 0"), (DUE[1], 1, DUE[1], "head 1"),
                    (DUE[2] + 1, 2, DUE[2] + 1, "head 2")]
    assert device.heads == [(0, DUE[0] + 1), (1, DUE[1]), (2, DUE[2] + 1)]
    assert returned == {0: True, 1: True, 2: True} and device.ticks == DUE[2] + 2


def test_a_head_back_two_ticks_late_at_a_wait_of_one_never_leaves():
    """At a wait of one tick, a head not back when the tick after its due one runs never leaves: that tick tells
    `leave` so, with no head, and the execution phase has none. Handed back after it, the head is dropped, and
    `head_back` says so. The other execution phases' heads leave as they did."""
    device, _, left, returned = run(WAIT, {1: DUE[1] + 2})
    assert left == [(DUE[0], 0, DUE[0], "head 0"), (DUE[1], 2, DUE[1], "head 2"), (DUE[1] + 1, 1, DUE[1] + 1, None)]
    assert device.heads == [(0, DUE[0]), (2, DUE[2]), (1, None)] and device.ticks == DUE[1] + 2
    assert returned == {0: True, 2: True}  # the run was over before the tick the driver held that head to
    device, _, left, returned = run(WAIT, {0: DUE[0] + 2})
    assert left[0] == (DUE[0] + 1, 0, DUE[0] + 1, None) and returned[0] is False
    assert device.heads == [(0, None), (1, DUE[1]), (2, DUE[2])] and not device._back and not device._due
    device, _, left, _ = run(WAIT, {0: None, 1: None, 2: None})
    assert left == [(DUE[0] + 1, 0, DUE[0] + 1, None), (DUE[1] + 1, 1, DUE[1] + 1, None),
                    (DUE[2] + 1, 2, DUE[2] + 1, None)]


def test_at_a_wait_of_zero_a_late_head_is_no_head():
    """At a wait of 0 a head not back when its due tick runs never leaves: the due tick tells `leave` so. One tick
    late is too late, where at a wait of one it left; and the run ends at the last due tick."""
    device, _, left, returned = run(SET, {0: DUE[0] + 1, 2: None})
    assert left == [(DUE[0], 0, DUE[0], None), (DUE[1], 1, DUE[1], "head 1"), (DUE[2], 2, DUE[2], None)]
    assert device.heads == [(0, None), (1, DUE[1]), (2, None)] and returned == {0: False, 1: True}
    assert device.ticks == DUE[2] + 1
    assert [tick for _, _, tick, head in run(WAIT, {0: DUE[0] + 1})[2] if head == "head 0"] == [DUE[0] + 1]


def test_an_execution_phase_that_ends_without_releases_has_its_word_held_to_the_same_tick():
    """Under form 2 an execution phase that is not handed ends in the same call without releases, and what whoever
    takes it makes of that, the word that no head comes, is due at the tick its head would have been. A tick that
    runs late faults its window, so here no execution phase of the run is handed, and each one's word leaves at its
    tick."""
    device, hands, left, _ = run(SET, late={T + 2: 0.03})
    assert [(t, w, strides) for t, w, strides in hands] == [(2 * T - 1, 0, None), (3 * T - 1, 1, None),
                                                            (3 * T - 1, 2, None)]
    assert left == [(DUE[w], w, DUE[w], f"head {w}") for w in range(PHASES)]


def test_a_head_is_handed_back_once_for_an_execution_phase_that_was_handed_under_a_schedule_with_an_offset():
    """`head_back` refuses, by name, a schedule without a head offset, an execution phase whose hand hasn't been
    made, no head at all, and a second head of an execution phase, before the first has left and after."""
    device, _, left, _ = run(TWO)
    assert not left and not device.heads and device.ticks == PHASES * T
    with pytest.raises(ValueError, match="the schedule has no head offset"):
        device.head_back(0, "head 0")
    with pytest.raises(ValueError, match="execution phase 1 was not handed, so no head of it is due"):
        run(SET, {1: 2 * T})
    with pytest.raises(ValueError, match="execution phase 0's head is handed back, not None"):
        Warden(SET, None, origin=0.0, windows=PHASES, clock=VirtualClock(0.0)).head_back(0, None)
    for at in (DUE[0] - 1, DUE[0] + 1):
        with pytest.raises(ValueError, match="execution phase 0's head was handed back before"):
            asyncio.run(_twice(at))


async def _twice(at: int) -> None:
    """A run under SET in which execution phase 0's head is handed back at its hand and again just before tick
    `at`."""
    clock = VirtualClock(-TAU / 2)
    device = Warden(SET, None, origin=0.0, windows=PHASES, clock=clock, sleep=clock.sleep,
                    lengths=lambda w, strides: device.head_back(w, f"head {w}"))
    running = asyncio.create_task(device.run())
    try:
        for t in range(DUE[-1] + 1):
            if t == at:
                device.head_back(0, "another")
            clock.advance(t * TAU)
            await device.until(lambda d, t=t: d.ticks > t)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


def test_the_device_names_the_execution_phases_whose_head_it_awaits():
    """`awaited` is the execution phases that were handed and whose head isn't back: an execution phase from its
    hand to the tick its head is handed back before, or to the last tick the head could have left at."""
    seen: dict[int, list[int]] = {}

    async def main() -> Warden:
        clock = VirtualClock(-TAU / 2)
        device = Warden(WAIT, None, origin=0.0, windows=PHASES, clock=clock, sleep=clock.sleep)
        running = asyncio.create_task(device.run())
        running.add_done_callback(lambda _: device._changed.set())
        t = 0
        while not running.done():
            if t == DUE[0]:
                device.head_back(0, "head 0")
            clock.advance(t * TAU)
            await device.until(lambda d, t=t: d.ticks > t or running.done())
            seen[t] = device.awaited
            t += 1
        return device

    device = asyncio.run(main())
    assert [seen[t] for t in (2 * T - 2, 2 * T - 1, DUE[0] - 1, DUE[0])] == [[], [0], [0], []]
    assert [seen[t] for t in (3 * T - 2, 3 * T - 1, DUE[1], DUE[1] + 1)] == [[], [1, 2], [1, 2], []]
    assert device.heads == [(0, DUE[0]), (1, None), (2, None)] and device.left[0] == (0, DUE[0], "head 0")


def test_a_schedule_without_the_two_fields_runs_as_it_did():
    """Without the fields no tick publishes a head and the run ends at its last bucket: the hands are the same as
    with them, and `leave` is never called."""
    plain, hands, left, _ = run(TWO)
    held, with_fields, _, _ = run(SET)
    assert plain.ticks == PHASES * T and held.ticks == DUE[-1] + 1 and not left and not plain.heads
    assert [(t, w, strides) for t, w, strides in hands] == [(t, w, strides) for t, w, strides in with_fields]
    assert [r.grid for r in plain.records] == [r.grid for r in held.records]
