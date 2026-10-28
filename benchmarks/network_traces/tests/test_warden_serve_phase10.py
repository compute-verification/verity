"""Adversarial cases against the relay's `PublisherProcess`:
the spawned process that alone holds the warden's key, rebuilds each record from the rows the warden releases a tick
at a time, and keeps, certifies and commits it. Every case runs the warden on a `VirtualClock`, with no sleeps and no
wall clock: the files the process writes are byte for byte the ones the thread publisher writes on the same run
and key, a faulted window's status included; a process killed mid-window fails the run at that window's close with
its exit code, and the check refuses the window; a process a window behind moves no tick and keeps every record whole;
a truncated file never makes the check accept. And the close marker carries the record's grid's SHA-512: a row lost,
doubled or changed between the processes is refused by the process, and a released row that isn't the record's by
the warden's side, so neither is signed."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, VirtualClock
from verity.core.protocols.certifier.certificate import record_head
from verity.core.protocols.certifier.grid import Reason
from verity.core.protocols.certifier.warden import END, Shim, Warden, cells, read_cell

from test_warden_serve import PARAMS, SESSION, _response, _Writer, ws

REQUESTS = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
RESPONSES = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
WINDOWS = 2
FILES = ("requests.records", "responses.records", "fails.records", "cells.salts")
FIFO = "PHASE10_FIFO"


def fixed_commit(link, direction, record, params, session, salts=None, wiring_salt=None):
    """`Cells.of` under salts fixed by the link-window and the row (and the wiring), so two publishers' files can be
    compared."""
    def salt(name, n):
        return hashlib.shake_256(f"{link}/{record.window}/{name}".encode()).digest(n)

    salts = [salt(i, WR.SALT_BYTES) for i in range(WR.rows_of(params))]
    return WR.Cells.of(link, direction, record, params, session, salts, salt("wiring", WR.WIRING_SALT_BYTES))


def blocking_commit(link, direction, record, params, session, salts=None, wiring_salt=None):
    """`fixed_commit`, which waits on the FIFO named in `PHASE10_FIFO` before window 0's ingress commit: a publisher
    held a window behind until the test writes to it."""
    if (direction.value, record.window) == ("ingress", 0):
        with open(os.environ[FIFO], "rb") as fh:
            fh.read(1)
    return fixed_commit(link, direction, record, params, session)


class ThreadPublisher:
    """The thread publisher: `Publisher.keep` on one worker thread of the relay's process, its results at `close`."""

    def __init__(self, out: Path, key: Path, commit) -> None:
        self.signer = ws.cs.signer(key)
        self.inline = ws.Publisher(out, ws.Issuer(self.signer, SESSION), PARAMS, commit)
        self.executor, self.futures = ThreadPoolExecutor(1), []
        self.key = self.signer.public_key()

    def __call__(self, record, grid) -> None:
        self.futures.append(self.executor.submit(self.inline.keep, record, grid))

    def close(self) -> list:
        self.executor.shutdown(wait=True)
        for f in self.futures:
            f.result()
        return self.inline.close()


def _key(tmp_path: Path, *names: str) -> list[Path]:
    """One LMS key, copied to `names` before any signature, so each publisher starts at the same leaf."""
    first = tmp_path / "key"
    ws.cs.load_key(first, 4)
    out = []
    for name in names:
        shutil.copytree(first, tmp_path / name)
        out.append(tmp_path / name)
    return out


def _drive(out: Path, key: Path, *, process: bool, commit=fixed_commit, late: dict | None = None,
           kill_at: int | None = None, before_close=None, release_edit=None, transport=None) -> dict:
    """The warden for `WINDOWS` windows on a `VirtualClock` over `REQUESTS`, published by `PublisherProcess` (or by
    `ThreadPublisher`), written as `serve` and `client` write a run; `late[t]` puts tick t that late, `kill_at` kills
    the process (by its pid) before tick t, `before_close(warden)` runs once every tick is done, before the
    publisher's `close`, `release_edit(release)` stands in for the release the warden calls, and `transport(put)` for
    the put of every message to the process. Returns the warden, the records it closed, and what the run and `close`
    raised."""
    out.mkdir()
    tau, late = PARAMS.bucket_ms / 1000, late or {}
    if process:
        import warden_serve

        publisher = warden_serve.PublisherProcess(out, key, SESSION, PARAMS, commit)
        release = publisher.release if release_edit is None else release_edit(publisher.release)
        if transport is not None:
            publisher.rows.put = transport(publisher.rows.put)
    else:
        publisher, release = ThreadPublisher(out, key, commit), None
    closed = []

    def publish(record, grid):
        closed.append(record)
        publisher(record, grid)

    answers = iter(RESPONSES)

    async def connect():
        reader = asyncio.StreamReader()
        reader.feed_data(next(answers))
        reader.feed_eof()
        return reader, _Writer()

    def settled(w: Warden) -> bool:
        return all(c.server is not None and c.ingress_ended and c.egress_ended for c in w.active)

    async def main():
        clock = VirtualClock(-tau / 2)
        warden = Warden(PARAMS, connect, origin=0.0, windows=WINDOWS, clock=clock, sleep=clock.sleep,
                        publish=publish, release=release)
        running = asyncio.create_task(warden.run())
        clients, tasks = [], []
        for request in REQUESTS:
            reader = asyncio.StreamReader()
            reader.feed_data(cells(request, PARAMS.cell_bytes))
            reader.feed_eof()
            clients.append(_Writer(warden))
            tasks.append(asyncio.create_task(warden.handle(reader, clients[-1])))
        await warden.until(lambda w: w.ordinals == len(REQUESTS))
        for t in range(WINDOWS * PARAMS.window):
            if t == kill_at:
                os.kill(publisher.process.pid, signal.SIGKILL)
                publisher.process.join()
            await warden.until(settled)
            clock.advance(t * tau + late.get(t, 0.0))
            ticked = asyncio.ensure_future(warden.until(lambda w, t=t: w.ticks > t))
            await asyncio.wait({ticked, running}, return_when=asyncio.FIRST_COMPLETED)
            if not ticked.done():
                ticked.cancel()
                break
        failed = None
        try:
            await running
        except (RuntimeError, EOFError) as e:
            failed = e
        await asyncio.gather(*tasks, return_exceptions=True)
        return clients, warden, failed

    clients, warden, failed = asyncio.run(main())
    if before_close is not None:
        before_close(warden)
    close_error = None
    try:
        publisher.close()
    except (RuntimeError, EOFError) as e:
        close_error = e
    deployment = ws.deployment_record(PARAMS, publisher.key, SESSION, origin_unix=0.0, windows=WINDOWS,
                                      listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=warden.slack)
    (out / "deployment.json").write_text(json.dumps(deployment))
    log, B = [], PARAMS.cell_bytes
    for index, (request, client) in enumerate(zip(REQUESTS, clients)):
        shim = Shim(B)
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        shim.outcome = "end" if shim.arrivals and END in [k for _, k in shim.arrivals] else "cut"
        log.append(ws.connection_log(index, 0.0, 0.0, request, shim, 0.0))
    (out / "client.json").write_text(json.dumps({"connections": log}))
    return {"warden": warden, "closed": closed, "failed": failed, "close_error": close_error}


def _heads(out: Path) -> dict:
    certified, _ = ws.read_certified(out / "certified.jsonl")
    return {(c.link, c.direction.value, c.window): c.head for c in certified.values()}


def _closed_heads(closed) -> dict:
    """The head of each complete record the warden closed, as the check recomputes it: the issuer certifies no other."""
    out = {}
    for name, (link, direction) in ws.CERTIFIED.items():
        grid = PARAMS.fails_grid if name == ws.FAILS else PARAMS.grid
        for r in closed:
            if r.link == name and r.status.complete:
                out[(link, direction.value, r.window)] = record_head(link, direction, r.window, r.grid, r.status, grid,
                                                                     PARAMS.digest)
    return out


def _unsalted(out: Path) -> list[dict]:
    """`certified.jsonl` line by line without what the certifier's fresh `salt_c` decides (the salt, the salted digest
    and the signature over it), with the signature's LMS leaf q kept, and a link-window the issuer refused as it is."""
    out_ = []
    for line in (out / "certified.jsonl").read_text().splitlines():
        doc = json.loads(line)
        if "refused" in doc:
            out_.append(doc)
            continue
        cert = doc["certificate"]
        out_.append({"head": doc["head"], "q": cert["signature"][:8],
                     "certificate": {k: v for k, v in cert.items() if k not in ("digest", "signature")}})
    return out_


def _cells_without_cost(out: Path) -> list[dict]:
    lines = (out / "cells.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(x).items() if k != "cost"} for x in lines]


@pytest.mark.parametrize("late", [None, {20: 0.1}], ids=["clean", "window-0-faulted"])
def test_the_process_writes_byte_for_byte_what_the_thread_publisher_wrote(tmp_path, late):
    """On one run and one key, the process's records, row salts and signed registrations are the thread
    publisher's byte for byte (the cost lines aside, which are timings), and its certificates are too but for what the
    certifier's fresh `salt_c` decides, each on the same LMS leaf and verifying on the head of the record the warden
    closed (no two runs share a `salt_c`, thread or process). With tick 20 late, window 0 of every link is
    `monitor-unavailable` on both sides: the status crosses the process boundary in the close marker, and the issuer
    certifies none of window 0's records."""
    thread_key, process_key = _key(tmp_path, "thread-key", "process-key")
    thread = _drive(tmp_path / "thread", thread_key, process=False, late=late)
    process = _drive(tmp_path / "process", process_key, process=True, late=late)
    for run in (thread, process):
        assert run["failed"] is None and run["close_error"] is None
    assert thread["closed"] == process["closed"]
    for name in FILES:
        assert (tmp_path / "thread" / name).read_bytes() == (tmp_path / "process" / name).read_bytes(), name
    assert _cells_without_cost(tmp_path / "thread") == _cells_without_cost(tmp_path / "process")
    assert _unsalted(tmp_path / "thread") == _unsalted(tmp_path / "process")
    assert _heads(tmp_path / "process") == _closed_heads(process["closed"])
    result = ws.check(tmp_path / "process")
    if late is None:
        assert result["accepted"], result["problems"]
    else:
        assert not result["accepted"]
        assert {f"{name} window 0 isn't certified" for name in ws.CERTIFIED} <= set(result["problems"])
        assert sorted(w for *_, w in result["summary"]["signer_refused"]) == [0] * len(ws.CERTIFIED)
        assert {(x["record"], x["status"]) for x in result["windows"] if x["window"] == 0} == \
            {(name, Reason.MONITOR_UNAVAILABLE.value) for name in ws.CERTIFIED}
    assert result["summary"]["certified_windows"] == 3 * (WINDOWS - (late is not None))


def test_a_process_killed_mid_window_fails_the_run_and_the_check_refuses_the_window(tmp_path):
    """The process killed (SIGKILL, by its pid) ten ticks into window 1: no tick runs late, the warden's run raises at
    window 1's close, before publishing it, and `close` raises too, without waiting on the dead process. Nothing of
    window 1 is on disk, the check refuses it on every link, and every head it does find certified is that of the
    record the relay closed. Both raise the process's exit code (rather than a bare `EOFError`, the dead
    process's end of the pipe reading as a message to `_receive`'s `poll` and `recv` hitting its EOF)."""
    (key,) = _key(tmp_path, "process-key")
    run = _drive(tmp_path / "run", key, process=True, kill_at=PARAMS.window + 10)
    warden = run["warden"]
    assert max(warden.lateness) <= warden.slack
    for error in (run["failed"], run["close_error"]):
        assert type(error) is RuntimeError
        assert str(error) == f"the publisher process exited ({-signal.SIGKILL}) before its done"
    heads = _heads(tmp_path / "run")
    assert {w for *_, w in heads} <= {0}
    assert all(_closed_heads(run["closed"])[k] == h for k, h in heads.items())
    result = ws.check(tmp_path / "run")
    assert not result["accepted"]
    assert {f"{name} window 1 isn't certified" for name in ws.CERTIFIED} <= set(result["problems"])


def test_a_process_a_window_behind_moves_no_tick_and_keeps_every_record_whole(tmp_path, monkeypatch):
    """The process held at window 0's ingress commit until every tick of both windows has run: no tick is late, at
    most window 0's ingress record is certified by then, and once released the process keeps every record, whole:
    the check accepts all six link-windows and four registrations."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setenv(FIFO, str(fifo))
    (key,) = _key(tmp_path, "process-key")
    seen = {}

    def behind(warden):
        seen["lateness"] = max(warden.lateness)
        seen["heads"] = set(_heads(tmp_path / "run"))
        with open(fifo, "wb") as fh:
            fh.write(b"\x01")

    run = _drive(tmp_path / "run", key, process=True, commit=blocking_commit, before_close=behind)
    assert run["failed"] is None and run["close_error"] is None
    assert seen["lateness"] == 0 and seen["heads"] <= {(ws.LINK, "ingress", 0)}
    result = ws.check(tmp_path / "run")
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["certified_windows"], s["cell_rows"]["registered"]) == (3 * WINDOWS, 2 * WINDOWS)
    assert _heads(tmp_path / "run") == _closed_heads(run["closed"])


@pytest.mark.parametrize("cut", ["certified.jsonl", "responses.records", "cells.jsonl"])
def test_a_file_cut_short_as_a_killed_process_leaves_it_is_never_accepted(tmp_path, cut):
    """A process killed while writing leaves its last line or blob short: the check then raises or refuses, and never
    accepts."""
    (key,) = _key(tmp_path, "process-key")
    run = _drive(tmp_path / "run", key, process=True)
    assert run["failed"] is None and ws.check(tmp_path / "run")["accepted"]
    path = tmp_path / "run" / cut
    data = path.read_bytes()
    path.write_bytes(data[:-7])
    try:
        result = ws.check(tmp_path / "run")
    except Exception as e:
        assert type(e).__name__ in ("JSONDecodeError", "ValueError", "error"), repr(e)
        return
    assert not result["accepted"]


def _egress_rows(edit):
    """A `transport` that hands the process's egress row messages, numbered from 0, through `edit(i, item)`, which
    gives the items to put in its place."""
    def transport(put):
        count = [0]

        def put_(item, *args, **kwargs):
            if item is not None and item[0] == ws.EGRESS and item[1] is not None:
                count[0] += 1
                for got in edit(count[0] - 1, item):
                    put(got, *args, **kwargs)
            else:
                put(item, *args, **kwargs)
        return put_
    return transport


def _shifted():
    """Window 0's first egress row sent twice and its last not at all: the count is the window's, the rows aren't."""
    return lambda i, item: [item, item] if i == 0 else [] if i == PARAMS.window - 1 else [item]


def _altered():
    """The last byte of the first egress row with a frame flipped, as it crosses."""
    flipped = []

    def edit(i, item):
        link, n, data = item
        if n and not flipped:
            flipped.append(i)
            return [(link, n, data[:-1] + bytes([data[-1] ^ 1]))]
        return [item]
    return edit


def test_the_close_marker_carries_the_records_grid_digest(tmp_path):
    """Each close marker the process gets is (window, status, the SHA-512 of the grid of the record the warden closed),
    in the order the warden closed them, and the run is accepted."""
    sent = []

    def transport(put):
        def put_(item, *args, **kwargs):
            sent.append(item)
            put(item, *args, **kwargs)
        return put_

    (key,) = _key(tmp_path, "process-key")
    run = _drive(tmp_path / "run", key, process=True, transport=transport)
    assert run["failed"] is None and run["close_error"] is None
    markers = [(link, data[0], data[2]) for link, n, data in filter(None, sent) if n is None]
    assert markers == [(r.link, r.window, ws.grid_digest(r.grid)) for r in run["closed"]]
    assert len(markers) == 3 * WINDOWS and sent[-1] is None
    assert ws.check(tmp_path / "run")["accepted"]


@pytest.mark.parametrize("edit", [_shifted, _altered], ids=["shifted", "altered"])
def test_a_row_lost_doubled_or_changed_between_the_processes_is_refused(tmp_path, edit):
    """The process once trusted that the rows since the last marker were the sampling block's. With the marker's
    digest it refuses sampling block 0's egress record when a row was lost and another doubled on the way (the count
    unchanged) or a frame's byte changed: the run fails with the process's refusal, nothing of that record is signed,
    and the check refuses it. Sampling block 0's ingress record, kept before it, stays certified."""
    (key,) = _key(tmp_path, "process-key")
    run = _drive(tmp_path / "run", key, process=True, transport=_egress_rows(edit()))
    first = run["failed"] or run["close_error"]
    assert type(first) is RuntimeError
    assert f"{ws.EGRESS} window 0: the rows rebuilt aren't the grid the warden closed" in str(first)
    assert type(run["close_error"]) is RuntimeError
    heads = _heads(tmp_path / "run")
    assert (ws.LINK, "egress", 0) not in heads and (ws.LINK, "ingress", 0) in heads
    assert all(_closed_heads(run["closed"])[k] == h for k, h in heads.items())
    problems = ws.check(tmp_path / "run")["problems"]
    assert f"{ws.EGRESS} window 0 isn't certified" in problems
    assert f"{ws.INGRESS} window 0 isn't certified" not in problems


def test_a_release_that_isnt_the_records_grid_is_refused_at_its_close(tmp_path):
    """The warden's side holds the record it closes to the rows it released: window 0's first egress row released
    twice and its last not at all fails the run at that close, before any marker for it, and the check refuses the
    record."""
    def release_edit(release):
        count = [0]

        def release_(link, row):
            if link == ws.EGRESS:
                count[0] += 1
                if count[0] == PARAMS.window:
                    return
                if count[0] == 1:
                    release(link, row)
            release(link, row)
        return release_

    (key,) = _key(tmp_path, "process-key")
    run = _drive(tmp_path / "run", key, process=True, release_edit=release_edit)
    assert type(run["failed"]) is RuntimeError and run["close_error"] is None
    assert str(run["failed"]) == f"{ws.EGRESS} window 0: the record's grid isn't the rows released"
    heads = _heads(tmp_path / "run")
    assert (ws.LINK, "egress", 0) not in heads and (ws.LINK, "ingress", 0) in heads
    assert f"{ws.EGRESS} window 0 isn't certified" in ws.check(tmp_path / "run")["problems"]
