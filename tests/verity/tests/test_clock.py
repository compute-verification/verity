"""The one clock (`verity.clock`): readings of one device, the bracket, and the tick source, on injected clocks."""

from __future__ import annotations

import asyncio

import pytest

from verity import clock as C


class Counter:
    """A device's clock its test moves: `advance` passes deadlines that `sleep` waits on, one coroutine at a time."""

    def __init__(self, ns: int = 0) -> None:
        self.ns, self.waits = ns, []

    def __call__(self) -> int:
        return self.ns

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.ns += max(0, round(seconds * C.NS))


def device(clock, key: str = "attester", lateness_ns: int = 0) -> C.Device:
    return C.Device(key, clock, 1, lateness_ns)


def test_elapsed_is_one_devices_difference_and_two_devices_readings_are_refused_by_name():
    a, b = C.Reading("m", 5), C.Reading("m", 12)
    assert C.elapsed(a, b) == 7 and C.elapsed(b, a) == -7
    with pytest.raises(ValueError, match="two devices, 'm' and 'n'"):
        C.elapsed(a, C.Reading("n", 12))


def test_a_tick_is_a_reading_counted_in_its_period():
    assert [C.tick(10, C.Reading("m", ns)) for ns in (0, 9, 10, 25)] == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="at least 1"):
        C.tick(0, C.Reading("m", 1))


def test_a_device_names_its_key_and_its_registered_figures():
    with pytest.raises(ValueError, match="names none"):
        C.Device("", lambda: 0, 1, 0)
    with pytest.raises(ValueError, match="'m'.*0 and 0"):
        C.Device("m", lambda: 0, 0, 0)


def test_stamp_reads_the_devices_clock_into_its_own_log_and_nowhere_else():
    """The only reading made outside a test is `stamp`'s: in the device's own log, with what the entry says beside it;
    another key's log is refused, and so is a clock that runs backwards, by name."""
    now = Counter(40)
    d, log = device(now), C.Log("attester")
    t = C.stamp(d, log, {"coin": 3})
    assert t == C.Reading("attester", 40) and log.entries == [C.Entry({"coin": 3}, t)]
    with pytest.raises(ValueError, match="only its own log, not the log under 'other'"):
        C.stamp(d, C.Log("other"))
    now.ns = 39
    with pytest.raises(C.ClockBackwards, match="read 39 ns after it read 40 ns"):
        C.stamp(d, log)
    assert len(log.entries) == 1


def test_an_entry_is_between_two_readings_of_the_logs_device_by_index():
    now = Counter(0)
    d, log = device(now), C.Log("attester")
    C.stamp(d, log)
    i = log.record({"root": "r"})
    now.ns = 30
    C.stamp(d, log)
    assert C.between(log, i, 0, 2)
    assert not C.between(log, i, 0, 1) and not C.between(log, 0, 0, 2) and not C.between(log, i, 0, 3)
    j = log.record({"answer": "a"})
    assert not C.between(log, i, 0, j)
    assert C.elapsed(log.entries[0].reading, log.entries[2].reading) == 30


def _run(gen, n: int) -> list[C.Tick]:
    async def go():
        out = []
        async for t in gen:
            out.append(t)
            if len(out) == n:
                await gen.aclose()
                return out
    return asyncio.run(go())


def test_ticks_run_on_their_time_and_a_wait_that_ends_early_waits_again():
    """Every tick runs once the clock reads origin + k periods and not before: here each first wait ends after half its
    length, and the tick waits the rest."""
    now, early = Counter(1_000), [False]

    async def short(seconds: float) -> None:
        if seconds > 0:
            now.waits.append(seconds)
            early[0] = not early[0]
            now.ns += round(seconds * C.NS / 2) if early[0] else round(seconds * C.NS)

    d, log = device(now), C.Log("attester")
    origin = C.stamp(d, log)
    out = _run(C.ticks(d, origin, 100, sleep=short, log=log), 4)
    assert [(t.k, t.late_ns, t.late) for t in out] == [(k, 0, False) for k in range(4)]
    assert now.ns == 1_300 and log.entries[1:] == [C.Entry({"tick": k}, t.reading) for k, t in enumerate(out)]
    assert [t.reading.ns for t in out] == [1_000, 1_100, 1_200, 1_300]


def test_every_tick_is_an_entry_of_its_log_and_its_reading_says_whether_it_is_past_the_devices_lateness():
    now = Counter(0)
    late = {2: 7, 3: 3}

    async def sleep(seconds: float) -> None:
        now.ns += max(0, round(seconds * C.NS))
        k = now.ns // 100
        if now.ns % 100 == 0 and k in late:
            now.ns += late.pop(k)

    d, log = device(now, lateness_ns=5), C.Log("attester")
    origin = C.stamp(d, log)
    out = _run(C.ticks(d, origin, 100, sleep=sleep, log=log), 4)
    assert [(t.late_ns, t.late) for t in out] == [(0, False), (0, False), (7, True), (3, False)]
    assert log.entries[1:] == [C.Entry({"tick": k}, C.Reading("attester", ns)) for k, ns in enumerate((0, 100, 207, 303))]


def test_ticks_start_only_from_the_devices_own_reading():
    with pytest.raises(ValueError, match="its own reading, not one of 'other'"):
        _run(C.ticks(device(Counter()), C.Reading("other", 0), 100, sleep=Counter().sleep, log=C.Log("attester")), 1)


def test_the_raw_clock_refuses_a_host_without_one(monkeypatch):
    def missing(_):
        raise OSError("no such clock")

    monkeypatch.setattr(C.time, "clock_gettime_ns", missing)
    with pytest.raises(C.ClockRefused, match="no raw monotonic clock"):
        C.raw_ns()
