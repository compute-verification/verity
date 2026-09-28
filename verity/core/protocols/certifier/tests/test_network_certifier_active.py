"""The active certifier (`active`): its shaper is `run_shaper`'s rule on frames that arrive, its faults fail closed, and
over real localhost sockets, on an injected clock, an honest run is accepted raw and committed on every link-window
while a sender that times its writes to signal either leaves the wire byte-identical or is rejected by the audit."""

import asyncio
import json
import math
import random
import resource
from collections import defaultdict
from concurrent.futures import Executor, Future
from pathlib import Path

import pytest
from verity.core.protocols.certifier import active, online
from verity.core.protocols.certifier.active import (
    DATA,
    LAST,
    Bucketer,
    Events,
    Proxy,
    Shaper,
    VirtualClock,
)
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.calibration import choose_syncs
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import GridParams, Reason, Certifier, row_count
from verity.core.protocols.certifier.schedule import (
    ClockSync,
    LogicalFrame,
    SyncSet,
    expected_grids,
    handing_buckets,
    run_shaper,
)


def test_frames_round_trip_and_padding_is_zero():
    frame = active.encode_frame(LAST, 7, b"abc", 32)
    assert len(frame) == 32 and active.decode_frame(frame) == (LAST, 7, b"abc")
    assert active.decode_frame(bytes(32)) == (active.PAD, 0, b"")
    with pytest.raises(ValueError):
        active.encode_frame(DATA, 1, b"x" * 24, 32)


def test_events_are_cut_by_content_not_by_reads():
    """However the bytes are read, the events are the same: each ends at its `\\n\\n`."""
    stream = b"HTTP/1.1 200 OK\r\nA: b\r\n\r\nid: 0\ndata: x\n\n" + b"".join(b"id: %d\ndata: y\n\n" % k for k in range(1, 9))
    want = Events().feed(stream)
    assert len(want) == 9 and [active.event_position(e) for e in want] == list(range(9))
    rng = random.Random(1)
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, len(stream)), 12))
        events = Events()
        got = [e for a, b in zip([0, *cuts], [*cuts, len(stream)]) for e in events.feed(stream[a:b])]
        assert got == want and not events.buffer
    assert active.event_position(b"data: x\n\n") is None and active.event_position(b"id: -1\n\n") is None


def random_instance(rng: random.Random, T: int = 4, windows: int = 3):
    """Frames in canonical order with anchors, indices and ready buckets, and one sync per window."""
    frames, ready, k = [], [], 0
    for _ in range(rng.randint(1, 4)):
        anchor = rng.randrange(windows * T)
        for i in range(rng.randint(1, 4)):
            frames.append(LogicalFrame(bytes([k]), anchor, i))
            ready.append(anchor + rng.randrange(0, 6))
            k += 1
    order = sorted(range(len(frames)), key=lambda j: (ready[j], frames[j].anchor, j))
    frames, ready = [frames[j] for j in order], [ready[j] for j in order]
    syncs = [ClockSync(rng.randrange(3), rng.choice((1, 2, 1 / 2))) for _ in range(windows)]
    return frames, ready, syncs


def feed_shaper(frames, ready, syncs, params, sync_set, arrive_bucket=None, prover=None, windows=None):
    """Drive a `Shaper` for the windows of `syncs` (or `windows` of them, declared by `prover`), each frame arriving
    before the tick of its arrival bucket (its ready bucket by default), in a shuffled order. Returns its records and
    its rows."""
    arrive_bucket = arrive_bucket or ready
    windows = len(syncs) if windows is None else windows
    shaper, records, rows = Shaper(Certifier("responses", params), syncs, sync_set, prover, windows), [], []
    rng = random.Random(len(frames))
    order = list(range(len(frames)))
    rng.shuffle(order)
    sent = set()
    for t in range(windows * params.buckets):
        for k in order:
            if k not in sent and arrive_bucket[k] <= t:
                shaper.arrive(k, frames[k], ready[k])
                sent.add(k)
        row, record = shaper.tick()
        rows.append(row)
        if record is not None:
            records.append(record)
    return shaper, records, rows


def check_late_frames(records, frames, hands, syncs, params, ready=None):
    """Egress records against the shaper's hand times: each window a frame is handed in after its h(k) is faulted
    `coverage-never-established`, and each complete one is the Program's grid on the frames handed by its close. Given
    the frames' ready buckets, each complete one is also `run_shaper`'s record, and with no frame late the records are
    `run_shaper`'s. Returns the windows a frame was handed in late."""
    logical = handing_buckets(frames, syncs, params.buckets)
    late = {a // params.buckets for a, h in zip(hands, logical) if a > h}
    honest = None if ready is None else run_shaper(frames, ready, syncs, Certifier("responses", params))
    for record in records:
        w = record.window
        if w in late:
            assert record.status.reason is Reason.COVERAGE_NEVER_ESTABLISHED
        if record.status.complete:
            by_close = frames[: sum(a < (w + 1) * params.buckets for a in hands)]
            assert record.grid == expected_grids(by_close, syncs, params)[w]
            assert honest is None or record == honest[w]
    if not late and honest is not None:
        assert records == honest
    return late


def test_the_shaper_is_run_shapers_rule():
    """On random instances, frames arriving in any order at their ready buckets: the shaper's hand times are
    `run_shaper`'s `max(h(k), ready(k), last)`. A window a frame is handed in late (after its h(k)) is faulted
    `coverage-never-established`; every other is complete, and is `run_shaper`'s record and the Program's grid on the
    frames handed by its close. Without a late frame the records are `run_shaper`'s."""
    P = GridParams(buckets=4, allocation=2, frame_bytes=1)
    sync_set = SyncSet((0, 1, 2), (1, 2, 1 / 2))
    rng = random.Random(7)
    seen = set()
    for _ in range(300):
        frames, ready, syncs = random_instance(rng)
        shaper, records, _ = feed_shaper(frames, ready, syncs, P, sync_set)
        hands, last = [], 0
        for h, r in zip(handing_buckets(frames, syncs, P.buckets), ready, strict=True):
            last = max(h, r, last)
            hands.append(last)
        assert shaper.hands == [h for h in hands if h < len(syncs) * P.buckets]
        late = check_late_frames(records, frames, shaper.hands, syncs, P, ready)
        assert all(r.status.reason in (None, Reason.COVERAGE_NEVER_ESTABLISHED) for r in records)
        seen.add(bool(late))
    assert seen == {False, True}


@pytest.mark.parametrize("name", ["last", "cover-2-1", "cover-2-0-fallback", "sticky-2-1-2"])
def test_a_shaper_with_an_online_prover_declares_what_it_declares_offline(name):
    """Frames arriving in any order at their ready buckets: the shaper's prover declares each window's sync at its
    first tick, the values `online.declare` gives on the whole run, and the records are `run_shaper`'s under them but
    in a window a frame is handed in late, which fails closed (`check_late_frames`)."""
    P = GridParams(buckets=4, allocation=2, frame_bytes=1)
    sync_set = SyncSet((0, 1, 2), (1, 2, 1 / 2))
    envelope = ClockSync(2, 1 / 2)
    rng = random.Random(8)
    for _ in range(200):
        frames, ready, _ = random_instance(rng, windows=4)
        ready = [max(r, f.anchor + 1) for f, r in zip(frames, ready, strict=True)]
        order = sorted(range(len(frames)), key=lambda j: (ready[j], frames[j].anchor, j))
        frames, ready = [frames[j] for j in order], [ready[j] for j in order]
        prover = online.Prover(sync_set, P.buckets, online.rule(name), envelope)
        shaper, records, _ = feed_shaper(frames, ready, [], P, sync_set, prover=prover, windows=4)
        want, _, _ = online.declare([(frames, ready)], sync_set, P.buckets, online.rule(name), 4, envelope)
        assert shaper.syncs == want
        check_late_frames(records, frames, shaper.hands, want, P, ready)


def test_out_of_set_advice_duplicates_and_overflow_fail_closed():
    """An out-of-set sync faults its own window only, and the grid still follows it; a position seen twice is
    malformed; more than Q frames waiting is an overflow."""
    P = GridParams(buckets=4, allocation=2, frame_bytes=1)
    sync_set = SyncSet((0, 1), (1,))
    frames = [LogicalFrame(bytes([k]), anchor, 0) for k, anchor in enumerate((0, 4, 8))]
    syncs = [ClockSync(0, 1), ClockSync(3, 1), ClockSync(0, 1)]
    _, records, _ = feed_shaper(frames, [0, 4, 8], syncs, P, sync_set)
    assert [r.status.reason for r in records] == [None, Reason.COVERAGE_NEVER_ESTABLISHED, None]
    assert [r.grid for r in records][::2] == expected_grids(frames, syncs, P)[::2]
    shaper = Shaper(Certifier("responses", P), syncs[:1], sync_set)
    shaper.arrive(0, frames[0], 1)
    shaper.arrive(0, frames[0], 1)
    assert shaper.certifier._fault is Reason.MALFORMED
    shaper = Shaper(Certifier("responses", GridParams(4, 2, 1, queue_capacity=1)), syncs[:1], sync_set)
    shaper.arrive(1, frames[1], 1)
    shaper.arrive(2, frames[2], 1)
    assert shaper.certifier._fault is Reason.OVERFLOW


def test_a_fault_that_drops_a_carried_queue_faults_the_next_window():
    """Five streams' first frames are due at bucket 2 at r = 1, so the Program's grid carries three into window 1. A
    fault in window 0 drops them: window 1 is faulted too, with the same reason, not a mismatch, and window 2, which
    nothing reaches, is complete. With one stream, whose frames are spaced, nothing is carried and window 1 is
    complete."""
    P = GridParams(buckets=4, allocation=1, frame_bytes=1)
    sync_set = SyncSet((0,), (1,))
    syncs = [ClockSync(0, 1)] * 3
    for frames, statuses in (([LogicalFrame(bytes([k]), 2, 0) for k in range(5)], [Reason.MONITOR_UNAVAILABLE] * 2),
                             ([LogicalFrame(bytes([k]), 2, k) for k in range(4)], [Reason.MONITOR_UNAVAILABLE])):
        shaper, records = Shaper(Certifier("responses", P), syncs, sync_set), []
        for t in range(len(syncs) * P.buckets):
            if t == 2:
                for k, frame in enumerate(frames):
                    shaper.arrive(k, frame, 2)
            if t == 3:
                shaper.certifier.fault(Reason.MONITOR_UNAVAILABLE)
            if (record := shaper.tick()[1]) is not None:
                records.append(record)
        statuses += [None] * (len(syncs) - len(statuses))
        assert [r.status.reason for r in records] == statuses
        expected = expected_grids(frames, syncs, P)
        assert all(r.grid == expected[r.window] for r in records if r.status.complete)


def test_the_bucketer_enqueues_at_the_next_boundary():
    """Frames arriving at bucket time s go out at ⌈s⌉, at most r_in per bucket, FIFO."""
    P = GridParams(buckets=4, allocation=1, frame_bytes=1)
    b = Bucketer(Certifier("requests", P))
    for frame, bucket in ((b"a", 0), (b"b", 1), (b"c", 1), (b"d", 3)):
        b.arrive(frame, bucket)
    rows = [b.tick()[0] for _ in range(4)]
    assert rows == [(b"a",), (b"b",), (b"c",), (b"d",)]


# --- real sockets, injected clock -------------------------------------------------------------------------------------

B = 512
EGRESS = GridParams(buckets=6, allocation=3, frame_bytes=B)
INGRESS = GridParams(buckets=6, allocation=2, frame_bytes=B)
SYNC_SET = SyncSet((0, 1, 2, 3, 4), (1, 2, 1 / 2))
WINDOWS = 3
HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
TAU = EGRESS.bucket_ms / 1000
#: (session, stream, send bucket, latencies d_k of its frames in buckets after its request is released)
SESSIONS = [("a", 1, 0, (1, 1, 2, 3)), ("b", 2, 0, (2, 2, 3)), ("c", 3, 4, (1, 2)), ("d", 4, 7, (3, 3, 4, 4, 5)),
            ("e", 5, 12, (1, 1)), ("f", 6, 13, (2, 4, 4))]


def event(session: str, index: int, n: int, position: int) -> bytes:
    choice = {"index": 0, "text": f" t{index}", "finish_reason": "length" if index == n - 1 else None}
    data = json.dumps({"id": session, "object": "text_completion", "choices": [choice]}, separators=(",", ":"))
    return (HEAD if index == 0 else b"") + f"id: {position}\ndata: {data}\n\n".encode()


def request(session: str, n: int) -> bytes:
    body = json.dumps({"model": "replay", "prompt": "x", "max_tokens": n, "stream": True, "user": session}).encode()
    return (b"POST /v1/completions HTTP/1.1\r\nHost: certifier\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body)) + body


def program(sessions=SESSIONS):
    """The Program's egress frames, their ready buckets and the honest prover's syncs. Each request goes out in the
    middle of its send bucket a, so the ingress certifier releases it at a + 1, its anchor."""
    rows = [(a + 1 + d, a + 1, name, k, stream, len(ds)) for name, stream, a, ds in sessions for k, d in enumerate(ds)]
    rows.sort()
    frames = [LogicalFrame(active.encode_frame(DATA, stream, event(name, k, n, p), B), anchor, k)
              for p, (_, anchor, name, k, stream, n) in enumerate(rows)]
    ready = [r[0] for r in rows]
    syncs, _, misses = choose_syncs([(frames, ready)], SYNC_SET, EGRESS.buckets)
    assert misses == 0
    return frames, ready, syncs + [ClockSync(0, 2)] * (WINDOWS - len(syncs)), rows


FRAMES, READY, SYNCS, ROWS = program()
STREAM = [f.payload[1:5] for f in FRAMES]


class Server:
    """An OpenAI-compatible completions server that streams what the test tells it to, when it is told."""

    def __init__(self) -> None:
        self.writers: dict[str, asyncio.StreamWriter] = {}
        self.changed = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        self.writers[json.loads(await reader.readexactly(length))["user"]] = writer
        self.changed.set()

    async def until(self, n: int) -> None:
        while len(self.writers) < n:
            self.changed.clear()
            await self.changed.wait()


def in_stream_order(buckets: dict[int, int]) -> dict[int, int]:
    """A sender's write buckets, made non-decreasing within each stream: its bytes go out in order."""
    out, last = {}, {}
    for p in sorted(buckets):
        out[p] = last[STREAM[p]] = max(buckets[p], last.get(STREAM[p], 0))
    return out


def honest() -> dict[int, int]:
    """Each frame written in the bucket before its ready bucket, as the trace recorded it."""
    return {p: r - 1 for p, r in enumerate(READY)}


async def replay(write_buckets: dict[int, int], prover=None, executor=None):
    """One run of the Program through a proxy on real sockets, its clock injected: the peer sends each request in the
    middle of its send bucket, and the server writes frame p in the middle of bucket `write_buckets[p]`. The syncs are
    the honest prover's, or an online `prover`'s on `executor`. Returns the proxy and the wire's bytes as the peer read
    them."""
    rows = ROWS
    advice = {"syncs": SYNCS} if prover is None else {"syncs": [], "prover": prover, "windows": WINDOWS,
                                                      "executor": executor}
    clock = VirtualClock(-TAU / 2)
    server = Server()
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, advice.pop("syncs"), origin=0.0,
                  clock=clock, sleep=clock.sleep, **advice)
    accepted: asyncio.Future = asyncio.get_running_loop().create_future()
    link = await asyncio.start_server(lambda r, w: accepted.set_result(asyncio.create_task(proxy.run(r, w))),
                                      "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection(*link.sockets[0].getsockname()[:2])
    wire = bytearray()

    async def read_wire() -> None:
        while data := await reader.read(1 << 16):
            wire.extend(data)

    peer = asyncio.create_task(read_wire())
    running = await accepted
    sent = written = 0
    horizon = WINDOWS * EGRESS.buckets
    for t in range(horizon):
        clock.advance(t * TAU)
        await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
        if t == horizon - 1:
            break  # the run ends at its last tick: what is written after it is ready past the horizon
        await server.until(proxy.requests_released)
        clock.advance(t * TAU + TAU / 2)
        for name, stream, a, ds in SESSIONS:
            if a == t:
                body = request(name, len(ds))
                chunks = [body[i : i + B - active.HEADER.size] for i in range(0, len(body), B - active.HEADER.size)]
                writer.write(b"".join(active.encode_frame(LAST if j == len(chunks) - 1 else DATA, stream, c, B)
                                      for j, c in enumerate(chunks)))
                sent += len(chunks)
        await proxy.until(lambda p, n=sent: p.link_frames_read >= n)
        for p, (_, _, name, k, _, n) in enumerate(rows):
            if write_buckets[p] == t:
                server.writers[name].write(event(name, k, n, p))
                written += 1
        await proxy.until(lambda p, n=written: len(p.frames) >= n)
    await running
    await peer
    for w in (writer, *server.writers.values()):
        w.close()
    for s in (srv, link):
        s.close()
    return proxy, bytes(wire)


def audit(proxy, frames, syncs):
    """The audit of the proxy's records, raw then committed: every egress and ingress link-window's verdict."""
    out = []
    for committed in (False, True):
        records = [commit(r, EGRESS if r.link == "responses" else INGRESS) if committed else r for r in proxy.records]
        a = Audit()
        a.check_link("responses", records, frames, syncs, SYNC_SET, EGRESS)
        a.check_ingress_link("requests", records, WINDOWS)
        out.append([(v.link, v.window, v.violation) for v in a.verdicts])
    return out


def expected_wire(frames, syncs) -> bytes:
    out = bytearray()
    for grid in expected_grids(frames, syncs, EGRESS):
        for row in grid:
            n = row_count(row)
            out += b"".join(row[:n]) + bytes((EGRESS.allocation - n) * B)
    return bytes(out)


@pytest.fixture(scope="module")
def honest_run():
    return asyncio.run(replay(honest()))


def test_an_honest_run_over_sockets_is_accepted_raw_and_committed(honest_run):
    """Every link-window is accepted, raw and committed; the records and hand times are `run_shaper`'s; the anchors
    are the ingress releases; and the peer reads exactly the Program's grids, r frames of B bytes per bucket."""
    proxy, wire = honest_run
    raw, committed = audit(proxy, FRAMES, SYNCS)
    assert raw == committed == [("responses", w, None) for w in range(WINDOWS)] + \
        [("requests", w, None) for w in range(WINDOWS)]
    assert [r for r in proxy.records if r.link == "responses"] == \
        run_shaper(FRAMES, READY, SYNCS, Certifier("responses", EGRESS))
    horizon = WINDOWS * EGRESS.buckets
    assert [(p, a, rd) for p, _, _, a, rd in sorted(proxy.frames)] == \
        [(p, f.anchor, r) for p, (f, r) in enumerate(zip(FRAMES, READY, strict=True)) if r < horizon]
    assert proxy.shaper.hands == [h for h in handing_buckets(FRAMES, SYNCS, EGRESS.buckets) if h < horizon]
    assert wire == expected_wire(FRAMES, SYNCS)
    assert len(wire) == horizon * EGRESS.allocation * B
    assert max(proxy.lateness["egress"]) == 0


def test_a_sender_that_signals_by_its_write_times_leaves_the_wire_identical(honest_run):
    """A sender inside X writes each frame in any bucket from its request's release up to the one before its logical
    hand time: a session bursts its frames at once to send a 1 and spreads them to the last moment to send a 0, or
    the sender picks each bucket at random. Every bit string gives the honest run's wire, byte for byte, and its
    records."""
    proxy, wire = honest_run
    hands = handing_buckets(FRAMES, SYNCS, EGRESS.buckets)
    assert all(h >= f.anchor + 1 for h, f in zip(hands, FRAMES, strict=True))
    sessions = sorted(set(STREAM))
    rng = random.Random(3)
    burst = {p: f.anchor for p, f in enumerate(FRAMES)}
    spread = {p: h - 1 for p, h in enumerate(hands)}
    strategies = [burst, spread, {p: rng.randrange(f.anchor, hands[p]) for p, f in enumerate(FRAMES)}]
    for _ in range(3):
        bits = {s: rng.randrange(2) for s in sessions}
        strategies.append({p: (burst if bits[STREAM[p]] else spread)[p] for p in burst})
    for buckets in strategies:
        other, other_wire = asyncio.run(replay(in_stream_order(buckets)))
        assert other_wire == wire
        assert other.records == proxy.records


def test_a_frame_late_past_every_sync_is_rejected(honest_run):
    """A sender holds session b's first frame, and so the rest of its stream, past the hand time the most lenient
    sync gives it. The wire changes. Each window a frame is handed in late fails closed (`coverage-never-established`),
    and a window that closed complete before the held frame was handed is the Program's on the frames handed by its
    close but a mismatch against the Program's frames, raw and committed, so the run is rejected; the ingress links are
    unaffected."""
    _, wire = honest_run
    lenient = [ClockSync(max(SYNC_SET.deltas), min(SYNC_SET.rhos))] * WINDOWS
    p = 3
    assert ROWS[p][2:4] == ("b", 0)
    hold = handing_buckets(FRAMES, lenient, EGRESS.buckets)[p]
    late, late_wire = asyncio.run(replay(in_stream_order(honest() | {p: hold})))
    raw, committed = audit(late, FRAMES, SYNCS)
    assert raw == committed
    egress = [r for r in late.records if r.link == "responses"]
    assert check_late_frames(egress, FRAMES, late.shaper.hands, SYNCS, EGRESS)
    expected = expected_grids(FRAMES, SYNCS, EGRESS)
    verdicts = ["status:coverage-never-established" if not r.status.complete else
                "mismatch" if r.grid != expected[r.window] else None for r in egress]
    assert "mismatch" in verdicts and None not in verdicts
    assert raw == [("responses", w, v) for w, v in enumerate(verdicts)] + [("requests", w, None) for w in range(WINDOWS)]
    assert late_wire != wire


ENVELOPE = ClockSync(max(SYNC_SET.deltas), min(SYNC_SET.rhos))


def read_prover(name: str, lag: int) -> online.Prover:
    """A prover that has read the honest run's every frame at its ready bucket."""
    prover = online.Prover(SYNC_SET, EGRESS.buckets, online.rule(name), ENVELOPE, lag=lag)
    for k, (f, r) in enumerate(zip(FRAMES, READY, strict=True)):
        prover.arrive(0, k, f, r)
    return prover


class Held(Executor):
    """Runs each choice at once, except those of the windows in `held`, which never finish: a choice slower than the
    proxy's run."""

    def __init__(self, held) -> None:
        self.held, self.windows = set(held), []

    def submit(self, fn, /, *args, **kwargs) -> Future:
        self.windows.append(w := len(args[0].syncs))
        return Future() if w in self.held else active.InlineExecutor().submit(fn, *args, **kwargs)


def test_a_choice_not_ready_at_its_window_declares_the_envelope():
    """Window 1's choice never finishes. Its tick doesn't wait: no tick is late, window 1 declares the envelope and
    logs it unready, and window 2's choice, started at its tick less the lag, reads that declaration. The audit
    accepts every link-window, raw and committed, since the envelope is in Σ_sync. The proxy refuses a prover that
    would choose at its tick (lag 0)."""
    lag = 2
    held = Held({1})
    proxy, _ = asyncio.run(replay(honest(), online.Prover(SYNC_SET, EGRESS.buckets, online.rule("last"), ENVELOPE,
                                                          lag=lag), held))
    assert held.windows == [0, 1, 2] and proxy.choices.log["unready"] == [1]
    assert max(proxy.lateness["egress"]) == max(proxy.lateness["ingress"]) == 0
    ready = read_prover("last", lag)
    assert ready.declare() == ENVELOPE and ready.choose() != ENVELOPE
    ready.declare(ENVELOPE)
    ready.declare()
    assert proxy.shaper.syncs == ready.syncs
    assert [r for r in proxy.records if r.link == "responses"] == \
        run_shaper(FRAMES, READY, ready.syncs, Certifier("responses", EGRESS))
    raw, committed = audit(proxy, FRAMES, ready.syncs)
    assert raw == committed == [(link, w, None) for link in ("responses", "requests") for w in range(WINDOWS)]
    with pytest.raises(ValueError, match="off its tick"):
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, [], origin=0.0, windows=WINDOWS,
              prover=online.Prover(SYNC_SET, EGRESS.buckets, online.rule("last"), ENVELOPE), executor=held)


@pytest.mark.parametrize("name", ["last", "cover-2-1", "sticky-2-1-2"])
def test_with_every_choice_ready_the_proxy_declares_what_online_declare_does(name):
    """On the injected clock every choice is ready by its tick (`InlineExecutor`): under each lag the proxy declares
    `online.declare`'s syncs with the same lag on the frames as read, and its records are `run_shaper`'s under them but
    in a window a frame is handed in late, which fails closed. Some lag declares other than the choice at the tick
    would."""
    rule = online.rule(name)
    at_tick, _, _ = online.declare([(FRAMES, READY)], SYNC_SET, EGRESS.buckets, rule, WINDOWS, ENVELOPE)
    differs = False
    for lag in range(1, EGRESS.buckets):
        prover = online.Prover(SYNC_SET, EGRESS.buckets, rule, ENVELOPE, lag=lag)
        proxy, _ = asyncio.run(replay(honest(), prover, active.InlineExecutor()))
        want, _, _ = online.declare([(FRAMES, READY)], SYNC_SET, EGRESS.buckets, rule, WINDOWS, ENVELOPE, lag=lag)
        assert proxy.shaper.syncs == want and proxy.choices.log["unready"] == []
        assert len(proxy.choices.log["choose_ms"]) == WINDOWS
        egress = [r for r in proxy.records if r.link == "responses"]
        check_late_frames(egress, FRAMES, proxy.shaper.hands, want, EGRESS, READY)
        differs |= want != at_tick
    assert differs


def test_the_worker_process_chooses_what_the_prover_does():
    """A snapshot pickles into the spawned worker (`active.worker()`), and the choice made there is the prover's."""
    prover = read_prover("cover-2-1", 2)
    prover.declare()
    prover.declare()
    pool = active.worker()
    try:
        sync, _ = pool.submit(active._choose, prover.snapshot()).result()
    finally:
        pool.shutdown()
    assert sync == prover.choose()


def test_reserve_fds_grows_the_fd_table_to_the_open_files_limit():
    """The proxy grows its fd table before its clock starts (`active.reserve_fds`), so no socket of a burst waits for the
    kernel to grow it under a second thread."""
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    want = 1 << 14 if soft == resource.RLIM_INFINITY else min(soft, 1 << 14)
    active.reserve_fds(1 << 14)
    status = Path("/proc/self/status").read_text().splitlines()
    size = next(int(line.split()[1]) for line in status if line.startswith("FDSize:"))
    assert size >= want


def test_a_late_tick_fails_closed_and_does_not_slip():
    """The clock jumps past bucket 8's boundary by more than the slack: window 1 is padding from there on, its status
    is monitor-unavailable on both links, it still has its T rows, and the other windows are unaffected."""

    async def run():
        clock = VirtualClock(0.0)
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=clock,
                      sleep=clock.sleep)
        writer = NullWriter()
        running = asyncio.create_task(proxy.run(EmptyReader(), writer))
        for t in range(WINDOWS * EGRESS.buckets):
            clock.advance(t * TAU + (2 * proxy.slack if t == 8 else 0))
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
        await running
        return proxy, writer

    proxy, writer = asyncio.run(run())
    statuses = [(r.link, r.window, r.status.reason) for r in proxy.records]
    assert sorted(statuses) == sorted([(link, w, Reason.MONITOR_UNAVAILABLE if w == 1 else None)
                                       for link in ("responses", "requests") for w in range(WINDOWS)])
    assert all(len(r.grid) == EGRESS.buckets for r in proxy.records)
    assert len(writer.data) == WINDOWS * EGRESS.buckets * EGRESS.allocation * B


def test_an_ingress_tick_read_early_fails_closed_and_the_run_goes_on():
    """The ingress ticker wakes 0.3τ before bucket 2's boundary and releases a's request, read in bucket 1 and so
    anchored at 2; the server answers at once, so its event is ready at bucket 2, no later than its anchor. The online
    prover refuses window 1's declaration, whose snapshot at bucket T − lag reads that frame (the proxy used to raise
    its `ValueError` and stop): egress window 1 is monitor-unavailable, padding with its T rows, its sync is the
    choice made on what the snapshot read, and windows 0 and 2 and every ingress window are accepted."""
    early = 2

    async def run():
        clock = VirtualClock(-TAU / 2)
        server = Server()
        srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)

        async def sleep(delay: float) -> None:
            ingress = asyncio.current_task().get_coro().__name__ == "_ingress"
            await clock.sleep(delay - (0.3 * TAU if ingress and abs(clock() + delay - early * TAU) < 1e-9 else 0))

        prover = online.Prover(SYNC_SET, EGRESS.buckets, online.rule("last"), ClockSync(1, 1), lag=1)
        proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, [], origin=0.0, clock=clock,
                      sleep=sleep, prover=prover, windows=WINDOWS, executor=active.InlineExecutor())
        link, writer = LinkReader(), NullWriter()
        running = asyncio.create_task(proxy.run(link, writer))

        async def until(predicate) -> None:
            waiting = asyncio.ensure_future(proxy.until(predicate))
            await asyncio.wait([waiting, running], return_when=asyncio.FIRST_COMPLETED)
            if not waiting.done():
                waiting.cancel()
                running.result()

        for t in range(WINDOWS * EGRESS.buckets):
            if t == early:
                clock.advance(early * TAU - 0.3 * TAU)
                await until(lambda p: p.requests_released == 1)
                await server.until(1)
                server.writers["a"].write(event("a", 0, 1, 0))
                await until(lambda p: len(p.frames) == 1)
            clock.advance(t * TAU)
            await until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
            if t == early - 1:
                clock.advance(t * TAU + TAU / 2)
                link.chunks.put_nowait(active.encode_frame(LAST, 1, request("a", 1), B))
                await until(lambda p: p.link_frames_read == 1)
        await running
        server.writers["a"].close()
        srv.close()
        return proxy, writer

    proxy, writer = asyncio.run(run())
    assert proxy.frames == [(0, 1, 0, early, early)]
    assert proxy.lateness["ingress"][early] < 0 and max(proxy.lateness["egress"]) == 0
    assert [(r.link, r.window, r.status.reason) for r in proxy.records if r.status.reason is not None] == \
        [("responses", 1, Reason.MONITOR_UNAVAILABLE)]
    assert sorted((r.link, r.window) for r in proxy.records) == \
        sorted((link, w) for link in ("responses", "requests") for w in range(WINDOWS))
    assert all(len(r.grid) == EGRESS.buckets for r in proxy.records)
    assert len(proxy.shaper.syncs) == WINDOWS and proxy.shaper.syncs == proxy.choices.prover.syncs
    assert proxy.shaper.hands == [early + 1]
    assert len(writer.data) == WINDOWS * EGRESS.buckets * EGRESS.allocation * B


def test_an_overrun_link_fails_closed():
    """When the link still holds more than a bucket unwritten at a tick, the window's status is delivery-refused."""

    async def run():
        clock = VirtualClock(0.0)
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=clock,
                      sleep=clock.sleep)
        writer = NullWriter(backlog={7: EGRESS.allocation * B + 1})
        running = asyncio.create_task(proxy.run(EmptyReader(), writer))
        for t in range(WINDOWS * EGRESS.buckets):
            writer.t = t
            clock.advance(t * TAU)
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
        await running
        return proxy

    proxy = asyncio.run(run())
    assert [r.status.reason for r in proxy.records if r.link == "responses"] == [None, Reason.DELIVERY_REFUSED, None]


def test_a_server_that_refuses_a_stream_fails_ingress_closed():
    """A request released to a server that refuses the connection makes its ingress window delivery-refused."""

    async def run():
        closed = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        upstream = closed.sockets[0].getsockname()[:2]
        closed.close()
        await closed.wait_closed()
        clock = VirtualClock(0.0)
        proxy = Proxy(upstream, EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=clock, sleep=clock.sleep)
        frame = active.encode_frame(LAST, 1, request("a", 1), B)
        running = asyncio.create_task(proxy.run(EmptyReader([frame]), NullWriter()))
        for t in range(WINDOWS * EGRESS.buckets):
            clock.advance(t * TAU)
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
            if proxy.streams:
                await asyncio.wait([st.task for st in proxy.streams.values()])
        await running
        return proxy

    proxy = asyncio.run(run())
    assert [(r.link, r.window, r.status.reason) for r in proxy.records if r.status.reason is not None] == \
        [("requests", 0, Reason.DELIVERY_REFUSED)]


#: x's frames interleave with s's, released a bucket after x's; y's are alone in window 2
LATE_SESSIONS = [("x", 1, 0, (1, 3, 4, 5)), ("s", 2, 1, (2, 2, 3)), ("y", 3, 13, (1, 2))]
PAUSE = TAU / 10


class ReceiptServer:
    """A server that numbers its events in the order it writes them, as the trace replay's does: it writes frame k of
    a request it received in bucket R in the middle of bucket R + d_k − 1, and closes the stream after the last."""

    def __init__(self, clock: VirtualClock, sessions) -> None:
        self.clock, self.latencies = clock, {name: ds for name, _, _, ds in sessions}
        self.received: dict[str, int] = {}
        self.due: dict[int, list] = defaultdict(list)
        self.position = 0
        self.changed = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        name = json.loads(await reader.readexactly(length))["user"]
        received = self.received[name] = math.floor(self.clock() / TAU + 1e-9)
        ds = self.latencies[name]
        for k, d in enumerate(ds):
            self.due[received + d - 1].append((received, name, k, len(ds), writer))
        self.changed.set()

    async def until(self, n: int) -> None:
        while len(self.received) < n:
            self.changed.clear()
            await self.changed.wait()

    def write_due(self, t: int) -> int:
        due = sorted(self.due.pop(t, []), key=lambda item: item[:3])
        for _, name, k, n, writer in due:
            writer.write(event(name, k, n, self.position))
            self.position += 1
            if k == n - 1:
                writer.close()
        return len(due)


async def replay_receipts(sessions, pause_at: int | None = None):
    """`replay` against a `ReceiptServer`. With `pause_at`, the proxy pauses after that bucket's ticks: the clock jumps
    to `PAUSE` past the next boundary before the requests released there are written upstream, so no tick is later
    than the slack but those requests reach the server a bucket late. Returns the proxy, the Program's frames and
    syncs, and where the server received each request."""
    frames, _, syncs, _ = program(sessions)
    clock = VirtualClock(-TAU / 2)
    server = ReceiptServer(clock, sessions)
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, syncs, origin=0.0, clock=clock,
                  sleep=clock.sleep)
    accepted: asyncio.Future = asyncio.get_running_loop().create_future()
    link = await asyncio.start_server(lambda r, w: accepted.set_result(asyncio.create_task(proxy.run(r, w))),
                                      "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection(*link.sockets[0].getsockname()[:2])

    async def read_wire() -> None:
        while await reader.read(1 << 16):
            pass

    peer = asyncio.create_task(read_wire())
    running = await accepted
    sent = written = 0
    horizon = WINDOWS * EGRESS.buckets
    for t in range(horizon):
        clock.advance(t * TAU)
        await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
        if t == horizon - 1:
            break
        if t == pause_at:
            assert not server.due.get(t) and all(a != t for _, _, a, _ in sessions)
            clock.advance((t + 1) * TAU + PAUSE)
            await server.until(proxy.requests_released)
            continue
        await server.until(proxy.requests_released)
        clock.advance(t * TAU + TAU / 2)
        for name, stream, a, ds in sessions:
            if a == t:
                body = request(name, len(ds))
                writer.write(active.encode_frame(LAST, stream, body, B))
                sent += 1
        await proxy.until(lambda p, n=sent: p.link_frames_read >= n)
        written += server.write_due(t)
        await proxy.until(lambda p, n=written: len(p.frames) >= n)
    await running
    await peer
    writer.close()
    for s in (srv, link):
        s.close()
    return proxy, frames, syncs, server.received


def test_a_request_forwarded_a_bucket_late_fails_closed():
    """The proxy pauses between two ticks, for less than the slack past the next boundary, after releasing s's request
    and before writing it upstream (live4's configuration 9). The server receives s a bucket late, so its events are
    ready a bucket late and the server's numbering of them and of x's interleaved ones isn't the Program's. The
    windows that would carry them are delivery-refused, ingress and egress, raw and committed; none is a mismatch, no
    tick is monitor-unavailable, and window 2 is accepted. On time, the same run is accepted."""
    proxy, frames, syncs, received = asyncio.run(replay_receipts(LATE_SESSIONS))
    assert received == {"x": 1, "s": 2, "y": 14}
    raw, committed = audit(proxy, frames, syncs)
    assert raw == committed == [(link, w, None) for link in ("responses", "requests") for w in range(WINDOWS)]

    proxy, frames, syncs, received = asyncio.run(replay_receipts(LATE_SESSIONS, pause_at=2))
    assert received == {"x": 1, "s": 3, "y": 14}
    assert max(max(proxy.lateness["egress"]), max(proxy.lateness["ingress"])) < proxy.slack
    raw, committed = audit(proxy, frames, syncs)
    refused = f"status:{Reason.DELIVERY_REFUSED.value}"
    assert raw == committed
    assert raw == [("responses", 0, refused), ("responses", 1, refused), ("responses", 2, None),
                   ("requests", 0, refused), ("requests", 1, None), ("requests", 2, None)]


class EmptyReader:
    """A link whose peer sends the given chunks, then nothing."""

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.chunks = list(chunks or [])

    async def read(self, n: int) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        await asyncio.Event().wait()
        return b""


class LinkReader:
    """A link whose peer sends what the test puts in `chunks`, one read each."""

    def __init__(self) -> None:
        self.chunks: asyncio.Queue[bytes] = asyncio.Queue()

    async def read(self, n: int) -> bytes:
        return await self.chunks.get()


class NullWriter:
    """A link that keeps what is written, and reports a backlog at the buckets given."""

    def __init__(self, backlog: dict[int, int] | None = None) -> None:
        self.data, self.backlog, self.t = bytearray(), backlog or {}, 0
        self.transport = self

    def get_write_buffer_size(self) -> int:
        return self.backlog.get(self.t, 0)

    def writelines(self, chunks) -> None:
        for c in chunks:
            self.data += c

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass
