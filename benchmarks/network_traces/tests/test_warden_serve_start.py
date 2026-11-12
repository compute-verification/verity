"""`serve`'s start, without a real clock: the network certifier's runner on a clock its own wait moves, over loopback
with no traffic. A run holds its key's device state from before its first tick until its publisher has closed, so a
second run under the key is refused by name before it waits, ticks or writes anything; a session id the key has
started is refused, and a new one starts where the last one's whole span ended on the key's life; the span is on
disk before the publisher signs the start, and the start is signed before the origin is read; the start's
certificate in `deployment.json` is what the check reads, and a deployment without it, or with another base, is
refused; a host without the raw clock starts nothing; and the run's log says how far the wall time parted from the
clock, for a drift the test picks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from verity.protocols.certifier.verifier.active import VirtualClock
from verity.protocols.certifier.verifier.certificate import DeviceState, StartRefused
from verity.protocols.certifier.verifier.pacer import ClockRefused

from test_warden_serve import PARAMS, SESSION

import warden_serve as ws  # by its own name: `serve` spawns the publisher's process, which imports it

OTHER = bytes(range(1, 33))
THIRD = bytes(range(2, 34))
SPAN_MS = PARAMS.window * PARAMS.bucket_ms


class StepClock(VirtualClock):
    """A clock that its own wait moves: a wait of `delay` ends with the clock `delay` on, so a run takes no time of the
    host's. `waits` counts them and `reads` its readings, and `first`, when given, runs once inside the first wait."""

    def __init__(self, now: float = 0.0, first=None) -> None:
        super().__init__(now)
        self.waits, self.reads, self.first = 0, 0, first

    def __call__(self) -> float:
        self.reads += 1
        return super().__call__()

    async def sleep(self, delay: float) -> None:
        self.waits += 1
        if self.first is not None:
            first, self.first = self.first, None
            await first()
        self.advance(self.now + max(delay, 0.0))
        await super().sleep(0)


def _serve(out: Path, key: Path, session: bytes, clock, *, wall=lambda: 0.0, windows: int = 1):
    return ws.serve(PARAMS, out, upstream=("127.0.0.1", 1), listen=("127.0.0.1", 0), key=key, session=session,
                    windows=windows, start_in=1.0, announce=lambda _: None, paced_ms=0, clock=clock, sleep=clock.sleep,
                    wall=wall)


def _start(out: Path) -> dict:
    return json.loads((out / "deployment.json").read_text())["start"]


def test_a_run_holds_its_keys_device_state_and_a_session_starts_where_the_last_ones_span_ended(tmp_path, monkeypatch):
    key = tmp_path / "key"
    ws.make_key(key, 4)
    public = ws.registered_key(key).public
    seen: dict = {}
    clocks: dict[bytes, StepClock] = {}
    signed: list[tuple] = []

    class Publishing(ws.PublisherProcess):
        """`PublisherProcess`, noting what is on disk when it is asked for a start and the clock's readings so far
        once the start is signed."""

        def __init__(self, out, key_dir, session, *args, **kwargs) -> None:
            on_disk = json.loads((key_dir / ws.DEVICE).read_text())
            super().__init__(out, key_dir, session, *args, **kwargs)
            signed.append((session, kwargs["base_ms"], on_disk["count_ms"], session.hex() in on_disk["sessions"],
                           clocks[session].reads))

    monkeypatch.setattr(ws, "PublisherProcess", Publishing)

    def run(out: Path, session: bytes, clock: StepClock, **how):
        clocks[session] = clock
        return _serve(out, key, session, clock, **how)

    async def second_run() -> None:
        """Inside the first run's first wait: its state is held, and it has ticked nothing yet."""
        clock = StepClock()
        with pytest.raises(StartRefused, match="held by another run of the network certifier"):
            await run(tmp_path / "second", OTHER, clock)
        seen["second"] = (clock.waits, sorted(p.name for p in (tmp_path / "second").iterdir()))

    first = StepClock(100.0, first=second_run)
    log = asyncio.run(run(tmp_path / "first", SESSION, first))
    assert seen["second"] == (0, [])
    assert log["ticks"] == PARAMS.window and log["lateness_ms"]["over_slack"] == 0
    assert _start(tmp_path / "first")["base_ms"] == 0
    with DeviceState(key / ws.DEVICE, public) as state:
        assert state.count_ms == SPAN_MS

    again = StepClock()
    with pytest.raises(StartRefused, match=f"session {SESSION.hex()} is already started"):
        asyncio.run(run(tmp_path / "again", SESSION, again))
    assert again.waits == 0 and sorted(p.name for p in (tmp_path / "again").iterdir()) == []
    with DeviceState(key / ws.DEVICE, public) as state:
        assert state.count_ms == SPAN_MS

    asyncio.run(run(tmp_path / "third", THIRD, StepClock(7.0), windows=2))
    assert _start(tmp_path / "third")["base_ms"] == SPAN_MS
    with DeviceState(key / ws.DEVICE, public) as state:
        assert state.count_ms == 3 * SPAN_MS
    # only the two runs that started asked for a signature; each one's whole span and its id were on disk by then,
    # and the clock had been read once, for the refusal of a host without it: the origin is read after the start
    assert signed == [(SESSION, 0, SPAN_MS, True, 1), (THIRD, SPAN_MS, 3 * SPAN_MS, True, 1)]

    for out, session, base in ((tmp_path / "first", SESSION, 0), (tmp_path / "third", THIRD, SPAN_MS)):
        dep = json.loads((out / "deployment.json").read_text())
        start = ws.start_from_json(dep["start"])
        assert start.verify(ws.cs.read_public(dep["key"]), session, ws.LINK, PARAMS.digest) and start.base_ms == base
        assert dep["origin_raw"] == {0: 101.0, SPAN_MS: 8.0}[base] and dep["origin_unix"] == 1.0
        (out / "client.json").write_text(json.dumps({"connections": []}))
        result = ws.check(out)
        assert result["summary"]["start"] == {"base_ms": base, "certified": True}
        assert not [p for p in result["problems"] if "start" in p]


def test_the_check_refuses_a_deployment_whose_start_is_missing_or_is_not_the_one_certified(tmp_path):
    key = tmp_path / "key"
    ws.make_key(key, 4)
    signer = ws.cs.signer(key)
    try:
        issuer = ws.Issuer(signer, SESSION)
        start = ws.start_json(ws.certify_start(issuer, 5, PARAMS))
        public = signer.public_key()
    finally:
        signer.close()

    def problems(start_doc, session: bytes = SESSION, params=PARAMS) -> list[str]:
        dep = ws.deployment_record(params, public, session, origin_unix=0.0, windows=1, listen=("127.0.0.1", 1),
                                   upstream=("127.0.0.1", 2), slack=0.025, start=start_doc, origin_raw=0.0)
        return ws.check_start(dep, public, session, params)[1]

    assert problems(start) == []
    assert ws.check_start({}, public, SESSION, PARAMS) == \
        ({"base_ms": None, "certified": False}, ["the session's start isn't certified (no start's certificate)"])
    assert problems(None) == ["the session's start isn't certified (no start's certificate)"]
    assert problems(start | {"base_ms": 6}) == ["the session's start isn't certified (opening)"]
    assert problems(start | {"schedule": "00" * 64}) == ["the session's start isn't certified (schedule)"]
    assert problems(start, session=OTHER) == ["the session's start isn't certified (signature)"]
    other = ws.replace(PARAMS, slots=PARAMS.slots + 1)
    assert problems(start, params=other) == ["the session's start isn't certified (schedule)"]
    record = start | {"certificate": start["certificate"] | {"link": ws.LINK}}
    assert problems(record) == ["the session's start isn't certified (start)"]


def test_a_key_made_before_its_device_state_and_a_host_without_the_raw_clock_start_nothing(tmp_path):
    key = tmp_path / "key"
    ws.cs.load_key(key, 4)
    clock = StepClock()
    with pytest.raises(StartRefused, match="missing"):
        asyncio.run(_serve(tmp_path / "run", key, SESSION, clock))
    assert clock.waits == 0 and sorted(p.name for p in (tmp_path / "run").iterdir()) == []
    ws.make_key(key, 4)  # a key that exists is given no state: it may have signed
    assert not (key / ws.DEVICE).exists()

    made = tmp_path / "made"
    ws.make_key(made, 4)

    def no_clock() -> float:
        raise ClockRefused("this host gives no raw monotonic clock")

    with pytest.raises(ClockRefused):
        asyncio.run(ws.serve(PARAMS, tmp_path / "none", upstream=("127.0.0.1", 1), listen=("127.0.0.1", 0), key=made,
                             session=SESSION, windows=1, announce=lambda _: None, paced_ms=0, clock=no_clock))
    with DeviceState(made / ws.DEVICE, ws.registered_key(made).public) as state:
        assert state.count_ms == 0
    assert sorted(p.name for p in (tmp_path / "none").iterdir()) == []


@pytest.mark.parametrize("drift", [0.0, 2e-5, -3e-4])
def test_the_runs_log_says_how_far_wall_time_parted_from_the_clock(drift):
    """`drift` is the wall's gain on the clock a second, the test's own: the log gives it back in ms an hour."""
    start, length = (50.0, 1_000.0), 7_200.0
    got = ws.clocks_parted(start, (start[0] + length, start[1] + length * (1 + drift)))
    assert got["clock_s"] == length and got["wall_s"] == pytest.approx(length * (1 + drift))
    assert got["parted_ms"] == pytest.approx(drift * length * 1000, abs=1e-6)
    assert got["parted_ms_an_hour"] == pytest.approx(drift * 3_600_000, abs=1e-6)
    assert ws.clocks_parted(start, start)["parted_ms_an_hour"] is None
