"""The active certifier (`active`): its shaper is `run_shaper`'s rule on frames that arrive, its faults fail closed, and
over real localhost sockets, on an injected clock, an honest run is accepted raw and committed on every link-window
while a sender that times its writes to signal either leaves the wire byte-identical or is rejected by the audit."""

import asyncio
import dataclasses
import itertools
import json
import math
import random
import re
import resource
from collections import defaultdict
from concurrent.futures import Executor, Future
from fractions import Fraction
from pathlib import Path

import pytest
import signers
from verity.ml.crypto import aead, lms
from verity.core.protocols.certifier import active, online, served
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsKey, LmsSigner, parameters_digest
from verity.core.protocols.certifier.active import (
    DATA,
    Bucketer,
    Events,
    Proxy,
    RemoteCertifier,
    Revealed,
    Shaper,
    VirtualClock,
)
from verity.core.protocols.certifier.audit import Audit, Verdict
from verity.core.protocols.certifier.calibration import choose_syncs
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import GridParams, Reason, Record, Status, Certifier, occupancy, row_count
from verity.core.protocols.certifier.pacer import Report
from verity.core.protocols.certifier.schedule import (
    ClockSync,
    LogicalFrame,
    SyncSet,
    expected_grids,
    handing_buckets,
    run_shaper,
)
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration


def test_frames_round_trip_and_padding_is_zero():
    frame = active.encode_frame(DATA, 7, b"abc", 32)
    assert len(frame) == 32 and active.decode_frame(frame) == (DATA, 7, b"abc")
    assert active.decode_frame(bytes(32)) == (active.PAD, 0, b"")
    with pytest.raises(ValueError):
        active.encode_frame(DATA, 1, b"x" * 24, 32)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 63, 64, 65, 128, 200])
def test_a_token_is_events_of_one_length_each_its_template(n):
    """Every event is `EVENT_BYTES` long whatever it carries, its frame is the egress template's (the same bytes
    before and after the text), and a token longer than `TOKEN_BYTES` is cut over events that give it back."""
    token = bytes(random.Random(n).randrange(256) for _ in range(n))
    evs = active.events(token)
    assert len(evs) == max(1, -(-n // active.TOKEN_BYTES))
    assert b"".join(active.event_data(e) for e in evs) == token
    p, s, prefix, suffix = active.event_placement(B)
    for e in evs:
        frame = active.encode_frame(DATA, active.CANONICAL_STREAM, e, B)
        assert len(e) == active.EVENT_BYTES and frame[:p] == prefix and frame[s : s + len(suffix)] == suffix
        assert frame[s + len(suffix) :] == bytes(B - s - len(suffix))
    with pytest.raises(ValueError):
        active.event(bytes(active.TOKEN_BYTES + 1))
    assert active.event_data(evs[0][:-1]) is None and active.event_data(evs[0].replace(b"verity", b"Verity")) is None


def test_an_id_event_is_one_template_its_ordinal_and_token_id_at_fixed_places():
    """An id event is `ID_EVENT_BYTES` long whatever it carries, the bytes around its two fields are the same in
    every one, it is JSON with the ordinal and the id as numbers, and `id_event_data` gives back exactly the bytes
    `id_event` writes: no other spelling of the same two numbers, and nothing that isn't one."""
    a = len(active.ID_EVENT_HEAD)
    b = a + active.ID_DIGITS + len(active.ID_EVENT_MID)
    for request, token_id in ((1, 0), (7, 128009), (2**32 - 1, 2**32 - 1)):
        e = active.id_event(request, token_id)
        assert len(e) == active.ID_EVENT_BYTES and e.startswith(active.ID_EVENT_HEAD)
        assert e[a + active.ID_DIGITS : b] == active.ID_EVENT_MID and e.endswith(active.ID_EVENT_TAIL)
        choice = json.loads(e.removeprefix(b"data: "))["choices"][0]
        assert (choice["request"], choice["token_id"]) == (request, token_id)
        assert active.id_event_data(e) == (request, token_id)
    for bad in ((-1, 0), (0, 2**32), (True, 0), (0, 1.0)):
        with pytest.raises(ValueError):
            active.id_event(*bad)
    e = active.id_event(7, 42)
    for other in (e.replace(b"        42", b"0000000042"), e.replace(b"        42", b"42        "), e[:-1],
                  e.replace(b"verity", b"Verity"), active.event(b"a token"), b"x"):
        assert active.id_event_data(other) is None
    assert active.id_event_data(e.replace(b"         7", b"4294967296"[:10])) is None


@pytest.mark.parametrize("frame_bytes", [8, 16, 256])
def test_a_compact_record_is_eight_bytes_its_finish_in_the_token_fields_flags(frame_bytes):
    """A compact record is the ordinal and the token field, u32be each: the id while more follow, the id with
    `COMPACT_LAST` at max_tokens, and the id with `COMPACT_LAST` and `COMPACT_STOP` at a stop token, so a stop carries
    the EOS id the model sampled. Its cell is the record zero-filled to B, the compact template's (nothing before or
    after the record but zeros), never padding, and `compact_record_data` gives back exactly what `compact_record`
    wrote and nothing else: so under the K-draw tie only the token field's 32 bits of an unchecked cell are the
    server's. A field with `COMPACT_STOP` and not `COMPACT_LAST` is no record's; all ones is a stop on id 2^30 − 1."""
    p, s, prefix, suffix = active.compact_placement(frame_bytes)
    assert (p, s, prefix, suffix) == (0, active.COMPACT_RECORD.size, b"", b"") and active.COMPACT_RECORD.size == 8
    assert (active.COMPACT_LAST, active.COMPACT_STOP) == (1 << 31, 1 << 30)
    cases = [((1, 0, None), b"\0\0\0\x01\0\0\0\0"),
             ((7, 128006, None), (7).to_bytes(4, "big") + (128006).to_bytes(4, "big")),
             ((7, 128006, "length"), (7).to_bytes(4, "big") + (128006 | 1 << 31).to_bytes(4, "big")),
             ((2**32 - 1, 128009, "stop"), b"\xff" * 4 + (128009 | 3 << 30).to_bytes(4, "big")),
             ((3, 2**30 - 1, "length"), b"\0\0\0\x03\xbf\xff\xff\xff"),
             ((3, 2**30 - 1, "stop"), b"\0\0\0\x03" + b"\xff" * 4), ((3, 0, "stop"), b"\0\0\0\x03\xc0\0\0\0")]
    for (request, token_id, finish), raw in cases:
        record = active.compact_record(request, token_id, finish)
        cell = active.compact_cell(record, frame_bytes)
        assert record == raw and len(cell) == frame_bytes and cell[s:] == bytes(frame_bytes - s) != cell
        assert active.compact_record_data(cell) == active.compact_record_data(record) == (request, token_id, finish)
        assert active.compact_unit(5, record) == b"\0\0\0\x05" + record
    for bad in ((0, 1, None), (1, 2**30, None), (1, 2**31, None), (1, 2**30, "stop"), (1, -1, None), (True, 1, None),
                (1, None, "stop"), (1, None, None), (1, 7, "eos")):
        with pytest.raises(ValueError):
            active.compact_record(*bad)
    record = active.compact_record(7, 42)
    for other in (bytes(8), (0).to_bytes(4, "big") + record[4:], record[:4] + b"\x7f\xff\xff\xff",
                  record[:4] + b"\x40\0\0\x2a", record[:7], active.compact_cell(record, frame_bytes) + b"\x01"):
        assert active.compact_record_data(other) is None
    with pytest.raises(ValueError):
        active.compact_cell(record, 7)


def test_compact_units_cut_a_connection_by_its_bytes_alone():
    """A compact server connection is its HTTP head through `\\r\\n\\r\\n`, then 12-byte (position, record) units:
    fed in one read, a byte a read or at random cuts, it gives the same units, and keeps a partial one uncut."""
    records = [active.compact_record(3, t) for t in (5, 6)] + [active.compact_record(3, 128009, "stop")]
    data = HEAD + b"".join(active.compact_unit(p, r) for p, r in zip((4, 9, 10), records)) + b"\0\0"
    rng = random.Random(1)
    for cuts in ([len(data)], [1] * len(data), [rng.randrange(1, 20) for _ in range(len(data))]):
        units, at, got = active.CompactUnits(), 0, []
        for n in cuts:
            got += units.feed(data[at : at + n])
            at += n
        assert got == list(zip((4, 9, 10), records)) and units.head and bytes(units.buffer) == b"\0\0"
    assert active.CompactUnits().feed(active.compact_unit(0, records[0])) == []


@pytest.mark.parametrize("frame_bytes", [512, 256])
@pytest.mark.parametrize("length", [1, 2, 300, 371, 372, 1024, 2048])
def test_an_id_requests_fields_are_at_places_its_prompts_length_fixes(length, frame_bytes):
    """Two id requests of one prompt length differ only inside their fields' places, no field straddles a frame (in
    a frame of 512 bytes, every field but the prompt's is in the first), `id_request_frames` is the frames
    `request_frames` puts it in, and `id_request_fields` gives back its fields and refuses any other spelling of
    them."""
    B = frame_bytes
    rng = random.Random(length)
    x = {"model": "llama-3.1-8b-instruct", "user": "varied.00007",
         "prompt": [rng.randrange(1 << 17) for _ in range(length)], "max_tokens": 64, "temperature": 0.0,
         "top_p": 1.0, "seed": 7}
    y = {"model": "m", "user": "u" * 62, "prompt": [2**32 - 1] * length, "max_tokens": 1, "temperature": 1.5,
         "top_p": 0.95, "seed": 2**32 - 1}
    rx, ry = (active.id_request(**f, frame_bytes=B) for f in (x, y))
    assert len(rx) == len(ry)
    assert active.request_frame_count(len(rx), B) == active.id_request_frames(length, B) == \
        len(active.request_frames(rx, B)) // B
    data = active.LENGTH.pack(len(rx)) + rx
    room = B - active.HEADER.size
    prompt = data.index(b'"prompt":') + len(b'"prompt":')
    fields = [(m.start(), m.end()) for m in re.finditer(rb"[\[,] *\d+", data[prompt:])]
    assert len(fields) == length and all((prompt + s) // room == (prompt + t - 1) // room for s, t in fields)
    assert all(t - s == active.ID_FIELD_BYTES for s, t in fields)
    differ = [k for k, (u, v) in enumerate(zip(rx, ry)) if u != v]
    head = rx.index(b"\r\n\r\n") + 4
    places = [rx.index(k, head) + len(k) for k in (b'"max_tokens":', b'"temperature":', b'"top_p":', b'"seed":',
                                                   b'"model":', b'"user":')]
    widths = [active.ID_DIGITS, 8, 8, active.ID_DIGITS, active.ID_NAME_BYTES, active.ID_NAME_BYTES]
    ends = [active.LENGTH.size + p + w for p, w in zip(places, widths)]
    assert all((e - w) // room == (e - 1) // room for e, w in zip(ends, widths))
    assert B < 512 or max(ends) <= room and prompt + fields[0][1] <= room
    allowed = {k for p, w in zip(places, widths) for k in range(p, p + w)}
    allowed |= {prompt - active.LENGTH.size + s + 1 + j for s, _ in fields for j in range(active.ID_DIGITS)}
    assert set(differ) <= allowed and active.id_request_fields(rx, B) == x and active.id_request_fields(ry, B) == y
    body = json.loads(rx.partition(b"\r\n\r\n")[2])
    assert body == x | {"stop": [], "stream": True}
    compact = rx.partition(b"\r\n\r\n")[0].rsplit(b":", 1)[0] + b":%10d\r\n\r\n"
    raw = json.dumps(body, separators=(",", ":")).encode()
    assert active.id_request_fields(compact % len(raw) + raw, B) is None
    assert active.id_request_fields(rx.replace(b'"seed":         7', b'"seed":0000000007'), B) is None
    assert active.id_request_fields(rx[:-1], B) is None and active.id_request_fields(b"\x00\xff", B) is None


def test_an_id_request_refuses_what_has_no_fixed_place():
    """An empty prompt, a name past `ID_NAME_BYTES` as JSON, a number past its field or not a number, and a frame
    with no room for the request's head are refused."""
    ok = {"model": "m", "user": "u", "prompt": [1], "max_tokens": 1}
    for bad in ({"prompt": []}, {"prompt": "ab"}, {"user": "u" * 63}, {"model": 7}, {"max_tokens": 2**32},
                {"max_tokens": True}, {"seed": -1}, {"temperature": 10.0}, {"temperature": 1e-7},
                {"top_p": float("nan")}, {"prompt": [1, 2**32]}):
        with pytest.raises(ValueError):
            active.id_request(**ok | bad, frame_bytes=B)
    for small in (64, 112):
        with pytest.raises(ValueError):
            active.id_request(**ok, frame_bytes=small)


@pytest.mark.parametrize("length", [0, 1, 499, 500, 503, 1536])  # B = 512: a frame holds 503 bytes after its header
def test_requests_end_by_their_length_on_one_header(length):
    """A request's frames all carry one header, its last is zero-filled, and `Requests` gives back each request's
    bytes, numbering them by their order on the link."""
    reqs = [bytes([k + 1]) * length for k in range(3)]
    frames = [active.request_frames(r, B) for r in reqs]
    assert [len(f) // B for f in frames] == [active.request_frame_count(length, B)] * 3
    p, s, prefix, _ = active.request_placement(B)
    wire, split, got = b"".join(frames), active.Requests(), {}
    for f in (wire[i : i + B] for i in range(0, len(wire), B)):
        assert f[:p] == prefix and s == B
        k, data, last = split.feed(active.decode_frame(f)[2])
        got.setdefault(k, []).append((data, last))
    assert {k: b"".join(d for d, _ in v) for k, v in got.items()} == {k + 1: r for k, r in enumerate(reqs)}
    assert all([last for _, last in v] == [False] * (len(v) - 1) + [True] for v in got.values())


def test_an_event_body_is_what_follows_its_id_line():
    """The proxy frames what follows the `id:` line: the position and an HTTP head before it aren't recorded."""
    body = active.event(b"tok")
    assert active.event_body(HEAD + b"id: 7\n" + body) == body == active.event_body(b"id:3\n" + body)
    assert active.event_position(HEAD + b"id: 7\n" + body) == 7


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


def test_the_bucketer_enqueues_its_own_frame_first_in_each_window():
    """The certifier's own frame (the salt's) is enqueued at the first tick of each window `first` gives one for, at
    the head of the queue: ahead of that bucket's arrivals and of the queue carried from the window before, so it
    leaves in slot 0 of that bucket, and the rest is FIFO, at most r_in per bucket. `own` names its slot in the row
    that releases it, by its place: a link frame of the same bytes is never taken for it, and one a fault drops (a
    fault before the window's first tick, or one an arrival at that tick causes) is never released."""
    P = GridParams(buckets=3, allocation=1, frame_bytes=1)
    b = Bucketer(Certifier("requests", P), first=lambda w: None if w == 2 else bytes([100 + w]))
    for frame, bucket in ((b"a", 0), (b"b", 2), (b"c", 2), (bytes([101]), 3)):
        b.arrive(frame, bucket)
    out = [(b.tick()[0], dict(b.own)) for _ in range(9)]
    assert [row for row, _ in out] == [(b"d",), (b"a",), (b"b",), (b"e",), (b"c",), (b"e",), (None,), (None,), (None,)]
    assert [own for _, own in out] == [{0: 0}, {}, {}, {0: 1}, {}, {}, {}, {}, {}]

    g = Bucketer(Certifier("requests", P), first=lambda w: bytes([100 + w]) if w else None)
    for frame, bucket in ((b"x", 2), (b"y", 2), (b"z", 2), (b"??", 6), (bytes([102]), 10)):
        g.arrive(frame, bucket)
    out = []
    for t in range(12):
        if t == 3:  # before window 1's first tick, with y and z carried
            g.certifier.fault(Reason.MONITOR_UNAVAILABLE)
        out.append((g.tick()[0], dict(g.own)))
    assert [row for row, _ in out] == [(None,)] * 2 + [(b"x",)] + [(None,)] * 6 + [(b"g",), (b"f",), (None,)]
    assert [own for _, own in out] == [{}] * 9 + [{0: 3}, {}, {}]
    assert g.certifier.window == 4


class _SaltBehind(Certifier):
    """The ingress certifier under the rule before: its own frame queued behind any carried queue (FIFO)."""

    def enqueue(self, frame: bytes, *, first: bool = False) -> None:
        super().enqueue(frame)


def test_a_carried_queue_that_fills_a_windows_first_bucket_leaves_the_derived_request_ends_exact():
    """Request ends derived from the signed counts (request k ends at the first bucket where the counts, less one own
    frame at each window's bucket 0, add up to its and the earlier requests' frames) are where the proxy anchors each
    request: the bucket that releases its last frame. At r_in = 2, a request of 3 frames and one of 1 arrive in window
    0's last bucket, which releases two of them, so two are carried and fill window 1's first bucket. The salt frame
    at the head of that bucket's queue leaves there, and every derived end is exact. Queued behind the carried frames
    (the rule before), it leaves a bucket later: the counts, and so every head, are the same, but the 1-frame
    request's last frame leaves in bucket 4 while the counts derive bucket 5, a bucket late."""
    P = GridParams(buckets=4, allocation=2, frame_bytes=B)
    requests = [(b"a" * 100, 0), (b"b" * 1100, 3), (b"c" * 100, 3)]
    sizes = [active.request_frame_count(len(r), B) for r, _ in requests]
    assert sizes == [1, 3, 1]

    def run(certifier: Certifier) -> tuple[list[int], list[int], list[int]]:
        b = Bucketer(certifier, first=lambda w: active.salt_frame(w, active.PLACEHOLDER, B))
        for request, bucket in requests:
            for frame in itertools.batched(active.request_frames(request, B), B):
                b.arrive(bytes(frame), bucket)
        cut, anchors, records = active.Requests(), {}, []
        for t in range(2 * P.buckets):
            row, record = b.tick()
            for frame in row[: row_count(row)]:
                kind, _, payload = active.decode_frame(frame)
                if kind == DATA and (end := cut.feed(payload))[2]:
                    anchors[end[0]] = t
            records += [record] if record is not None else []
        counts = [row_count(row) for r in records for row in r.grid]
        own = [int(t % P.buckets == 0) for t in range(len(counts))]
        released = list(itertools.accumulate(n - o for n, o in zip(counts, own)))
        derived = [next(t for t, m in enumerate(released) if m >= total) for total in itertools.accumulate(sizes)]
        return counts, [anchors[k + 1] for k in range(len(requests))], derived

    counts, anchors, derived = run(Certifier("requests", P))
    assert counts == [2, 0, 0, 2, 2, 1, 0, 0]
    assert anchors == derived == [0, 4, 5]
    behind_counts, behind_anchors, behind_derived = run(_SaltBehind("requests", P))
    assert behind_counts == counts and behind_derived == derived
    assert behind_anchors == [0, 4, 4]


def test_the_bucketers_admission_admits_whole_requests_while_the_window_ends_idle():
    """Requests of two frames, each answered by two, under δ = 0, ρ = 1 with one frame a bucket on both ends: a and b
    complete a request at bucket 1, answered in buckets 1 and 2; c and d would complete one at bucket 3, answered past
    the window's end, so they wait for window 1, and e and f for window 2. Over random arrivals, with and without the
    certifier's own frame (the salt's) at each window's head, every window's released frames are that frame and whole
    requests in arrival order, and every window derives (`served.derive`). A served link takes an own frame exactly
    when its profile counts one, and its window holds that frame, one request and its response."""
    P = GridParams(buckets=4, allocation=1, frame_bytes=1)
    admission = served.Admission(served.ServedProfile(2, 2, 1), 1, lambda w: ClockSync(0, 1))
    b = Bucketer(Certifier("requests", P), served=admission)
    for frame, bucket in ((b"a", 0), (b"b", 0), (b"c", 1), (b"d", 2), (b"e", 5), (b"f", 6)):
        b.arrive(frame, bucket)
    rows = [b.tick()[0] for _ in range(12)]
    assert rows == [(b"a",), (b"b",), (None,), (None,), (b"c",), (b"d",), (None,), (None,), (b"e",), (b"f",), (None,),
                    (None,)]
    with pytest.raises(ValueError, match="own frame exactly when"):
        Bucketer(Certifier("requests", P), first=lambda w: b"s", served=admission)
    with pytest.raises(ValueError, match="own frame exactly when"):
        Bucketer(Certifier("requests", P), served=served.Admission(served.ServedProfile(2, 2, 1, 1), 1,
                                                                   lambda w: ClockSync(0, 1)))
    with pytest.raises(ValueError, match="holds that frame, one request and its response"):
        Bucketer(Certifier("requests", P), served=served.Admission(served.ServedProfile(4, 2, 1), 1,
                                                                   lambda w: ClockSync(0, 1)))

    rng = random.Random(54)
    for _ in range(200):
        P = GridParams(buckets=rng.randint(3, 8), allocation=rng.randint(1, 3), frame_bytes=1)
        n, r = rng.choice((1, 2, 4)), rng.randint(1, 3)
        sync = ClockSync(rng.randint(0, 2), rng.choice((Fraction(1), Fraction(2), Fraction(1, 2))))
        own = rng.randint(0, 1)
        admission = served.Admission(served.ServedProfile(rng.randint(1, 3), n, 1, own), r, lambda w: sync)
        try:
            b = Bucketer(Certifier("requests", P), (lambda w: b"s") if own else None, served=admission)
        except ValueError:
            continue
        sent = [bytes([k]) for k in range(n * rng.randint(0, 6))]
        for frame, bucket in zip(sent, sorted(rng.randrange(3 * P.buckets) for _ in sent)):
            b.arrive(frame, bucket)
        records = [r for r in (b.tick()[1] for _ in range(20 * P.buckets)) if r is not None]
        assert [f for r in records for row in r.grid for f in row if f not in (None, b"s")] == sent
        for rec in records:
            assert (rec.grid[0][0] == b"s") is bool(own)
            counts = [row_count(row) for row in rec.grid]
            assert served.derive(counts, sync, admission.profile, r) is not None


def test_a_shaper_and_a_bucketer_back_from_maintenance_publish_the_full_runs_later_windows():
    """On random instances where nothing reaches window 1 and window 0 carries nothing into it: a shaper started at
    window 2, at the position of the first frame anchored there and fed only the frames from it on, hands them when
    the full run does and publishes its records of windows 2 and 3, as a bucketer started at window 2 does for the
    arrivals from there on."""
    P = GridParams(buckets=4, allocation=2, frame_bytes=1)
    sync_set = SyncSet((0, 1, 2), (1, 2, 1 / 2))
    rng = random.Random(9)
    tested = 0
    for _ in range(2000):
        frames, ready, syncs = random_instance(rng, windows=4)
        split = sum(f.anchor < 2 * P.buckets for f in frames)
        if any(f.anchor >= 2 * P.buckets for f in frames[:split]) or any(
                f.anchor < 2 * P.buckets for f in frames[split:]):
            continue
        full, records, _ = feed_shaper(frames, ready, syncs, P, sync_set)
        if split == len(frames) or row_count_total(records[1]) or any(h >= P.buckets for h in full.hands[:split]):
            continue
        shaper = Shaper(Certifier("responses", P), syncs, sync_set, start=2, position=split)
        later = []
        for t in range(2 * P.buckets, 4 * P.buckets):
            for k in range(split, len(frames)):
                if ready[k] == t:
                    shaper.arrive(k, frames[k], ready[k])
            if (record := shaper.tick()[1]) is not None:
                later.append(record)
        assert later == records[2:] and shaper.hands == full.hands[split:]
        tested += 1
    assert tested >= 20

    arrivals = [(b"a", 0), (b"b", 1), (b"c", 1), (b"d", 9), (b"e", 9), (b"f", 13)]
    full = Bucketer(Certifier("requests", GridParams(buckets=4, allocation=1, frame_bytes=1)))
    back = Bucketer(Certifier("requests", GridParams(buckets=4, allocation=1, frame_bytes=1)), start=2)
    for frame, bucket in arrivals:
        full.arrive(frame, bucket)
        if bucket >= 8:
            back.arrive(frame, bucket)
    records = [r for r in (full.tick()[1] for _ in range(16)) if r is not None]
    assert [r for r in (back.tick()[1] for _ in range(8)) if r is not None] == records[2:]
    assert [r.window for r in records] == [0, 1, 2, 3] and not row_count_total(records[1])


def row_count_total(record) -> int:
    return sum(row_count(row) for row in record.grid)


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


def body(session: str, index: int) -> bytes:
    """Frame `index` of `session`'s response: the canonical event of its token."""
    return active.event(f"{session} t{index}".encode())


def event(session: str, index: int, n: int, position: int) -> bytes:
    return (HEAD if index == 0 else b"") + b"id: %d\n" % position + body(session, index)


def request(session: str, n: int) -> bytes:
    body = json.dumps({"model": "replay", "prompt": "x", "max_tokens": n, "stream": True, "user": session}).encode()
    return (b"POST /v1/completions HTTP/1.1\r\nHost: certifier\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body)) + body


def program(sessions=SESSIONS):
    """The Program's egress frames, their ready buckets and the honest prover's syncs. Each request goes out in the
    middle of its send bucket a, so the ingress certifier releases it at a + 1, its anchor."""
    rows = [(a + 1 + d, a + 1, name, k, stream, len(ds)) for name, stream, a, ds in sessions for k, d in enumerate(ds)]
    rows.sort()
    frames = [LogicalFrame(active.encode_frame(DATA, active.CANONICAL_STREAM, body(name, k), B), anchor, k)
              for _, anchor, name, k, _, _ in rows]
    ready = [r[0] for r in rows]
    syncs, _, misses = choose_syncs([(frames, ready)], SYNC_SET, EGRESS.buckets)
    assert misses == 0
    return frames, ready, syncs + [ClockSync(0, 2)] * (WINDOWS - len(syncs)), rows


FRAMES, READY, SYNCS, ROWS = program()
STREAM = [row[4] for row in ROWS]


class Server:
    """An OpenAI-compatible completions server that streams what the test tells it to, when it is told. A connection
    that starts with a salt frame is the salt stream: `salts` keeps its frames, each its header and the length the
    header gives, and `salted` is set at its end. One that starts with an `ORDINAL` header puts its request's
    ordinal in `ordinals`, by its `user`."""

    def __init__(self) -> None:
        self.writers: dict[str, asyncio.StreamWriter] = {}
        self.ordinals: dict[str, int] = {}
        self.changed = asyncio.Event()
        self.salts: list[bytes] = []
        self.salted = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        first = await reader.readexactly(1)
        if first[0] == active.SALT:
            head = first + await reader.readexactly(active.HEADER.size - 1)
            while True:
                self.salts.append(head + await reader.readexactly(active.HEADER.unpack(head)[2]))
                if not (head := await reader.read(active.HEADER.size)):
                    break
                head += await reader.readexactly(active.HEADER.size - len(head))
            self.salted.set()
            return
        ordinal = None
        if first[0] == active.ORDINAL:
            ordinal = active.HEADER.unpack(first + await reader.readexactly(active.HEADER.size - 1))[1]
            first = await reader.readexactly(1)
        head = first + await reader.readuntil(b"\r\n\r\n")
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        user = json.loads(await reader.readexactly(length))["user"]
        self.writers[user] = writer
        if ordinal is not None:
            self.ordinals[user] = ordinal
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


def seal_frames(sealer: active.Sealer, data: bytes, start: int) -> bytes:
    """`data`'s B-byte frames as the pair's other certifier seals them: the (start + i)-th unit it sends is slot
    start + i of its grid, counted from its window 0."""
    return b"".join(sealer.seal(*divmod(start + i // B, sealer.params.allocation), data[i : i + B])
                    for i in range(0, len(data), B))


async def replay(write_buckets: dict[int, int], prover=None, executor=None, salt=None, server=None, probe=None,
                 sessions=SESSIONS, start: int = 0, pair: active.Pair | None = None,
                 sealer: active.Sealer | None = None, egress: GridParams = EGRESS, write=event, **salting):
    """One run of the Program through a proxy on real sockets, its clock injected: the peer sends each request in the
    middle of its send bucket, and the server writes frame p in the middle of bucket `write_buckets[p]`. The syncs are
    the honest prover's, or an online `prover`'s on `executor`. With `salt`, the proxy's salt source (and `salting`,
    its `salt_bits` and `pouw_window`), each PoUW window's salt is delivered before the run goes on. `probe(proxy, t)`
    runs once tick t is done, and the run waits on it if it is a coroutine. With `start`, the proxy is back from maintenance at that window, and only what comes from
    there on is sent and written. With `pair`, the link is a sealed pair link, and the peer stands in for the pair's
    other certifier: `sealer` seals the k-th frame it sends as slot k of its grid (`seal_frames`). The proxy's egress
    grid is `egress`, and the server writes frame p as `write(session, k, n, p)`. Returns the proxy and the wire's
    bytes as the peer read them."""
    _, _, syncs, rows = (FRAMES, READY, SYNCS, ROWS) if sessions is SESSIONS else program(sessions)
    first = start * EGRESS.buckets
    advice = {"syncs": syncs} if prover is None else {"syncs": [], "prover": prover, "windows": WINDOWS,
                                                      "executor": executor}
    clock = VirtualClock(-TAU / 2)
    server = server or Server()
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    proxy = Proxy(srv.sockets[0].getsockname()[:2], egress, INGRESS, SYNC_SET, advice.pop("syncs"), origin=0.0,
                  ingress_link="requests", clock=clock, sleep=clock.sleep, salt=salt, start=start,
                  position=sum(anchor < first for _, anchor, *_ in rows), pair=pair, **advice, **salting)
    k = salting.get("pouw_window", 1)
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
    for t in range(first, horizon):
        clock.advance(t * TAU)
        await proxy.until(lambda p, n=t - first: len(p.lateness["egress"]) > n and len(p.lateness["ingress"]) > n)
        if salt is not None:
            await proxy.until(lambda p, t=t: p.salts_delivered > t // (k * INGRESS.buckets) - start)
        if probe is not None:
            probed = probe(proxy, t)
            if asyncio.iscoroutine(probed):
                await probed
        if t == horizon - 1:
            break  # the run ends at its last tick: what is written after it is ready past the horizon
        await server.until(proxy.requests_released)
        clock.advance(t * TAU + TAU / 2)
        for name, stream, a, ds in sessions:
            if a == t:
                request_wire = active.request_frames(request(name, len(ds)), B)
                if sealer is not None:
                    request_wire = seal_frames(sealer, request_wire, sent)
                writer.write(request_wire)
                sent += len(request_wire) // (B if sealer is None else B + aead.TAG_BYTES)
        await proxy.until(lambda p, n=sent: p.link_frames_read >= n)
        for p, (_, _, name, k, _, n) in enumerate(rows):
            if write_buckets[p] == t:
                server.writers[name].write(write(name, k, n, p))
                written += 1
        await proxy.until(lambda p, n=written: len(p.frames) >= n)
    await running
    await peer
    if salt is not None:
        await server.salted.wait()
    for w in (writer, *server.writers.values()):
        w.close()
    for s in (srv, link):
        s.close()
    return proxy, bytes(wire)


def audit(proxy, frames, syncs, egress=EGRESS):
    """The audit of the proxy's records, raw then committed: every egress and ingress link-window's verdict."""
    out = []
    for committed in (False, True):
        records = [commit(r, egress if r.link == "responses" else INGRESS) if committed else r for r in proxy.records]
        a = Audit()
        a.check_link("responses", records, frames, syncs, SYNC_SET, egress)
        a.check_ingress_link("requests", records, WINDOWS)
        out.append([(v.link, v.window, v.violation) for v in a.verdicts])
    return out


def expected_wire(frames, syncs, egress=EGRESS) -> bytes:
    out = bytearray()
    for grid in expected_grids(frames, syncs, egress):
        for row in grid:
            n = row_count(row)
            out += b"".join(row[:n]) + bytes((egress.allocation - n) * egress.frame_bytes)
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


def test_a_proxy_with_ordinals_names_each_request_to_its_server_and_changes_nothing_on_the_link(honest_run):
    """Under `ordinals` each stream's server connection opens with an `ORDINAL` header whose stream is the request's
    ingress ordinal, its stream on the link; the records and the wire are the run's without it."""
    server = Server()
    proxy, wire = asyncio.run(replay(honest(), server=server, ordinals=True))
    assert server.ordinals == {name: stream for name, stream, _, _ in SESSIONS}
    assert proxy.records == honest_run[0].records and wire == honest_run[1]


ORDINAL_OF = {name: stream for name, stream, _, _ in SESSIONS}


def compact_of(session: str, index: int, n: int) -> bytes:
    """Output `index` of `session`'s n as a compact record: session c's response ends at a stop token, the others at
    max_tokens."""
    stream, finish = ORDINAL_OF[session], None if index < n - 1 else "stop" if session == "c" else "length"
    return active.compact_record(stream, 1000 * stream + index, finish)


def compact_write(session: str, index: int, n: int, position: int) -> bytes:
    return (HEAD if index == 0 else b"") + active.compact_unit(position, compact_of(session, index, n))


@pytest.mark.parametrize("cell", [8, 16])
def test_compact_records_over_sockets_are_b_byte_cells_and_the_run_is_accepted(cell, honest_run):
    """Under `compact` the server writes each output as a (position, record) unit after its HTTP head, and the proxy
    frames each record as its cell of B = 8 or 16 bytes: every link-window is accepted raw and committed against the
    Program's cells, the records and hand times are `run_shaper`'s and the event-framed run's, the ingress records are
    unchanged, and the wire is B/512 of the event-framed run's, the same frames a bucket."""
    egress = dataclasses.replace(EGRESS, frame_bytes=cell)
    frames = [LogicalFrame(active.compact_cell(compact_of(name, k, n), cell), anchor, k)
              for _, anchor, name, k, _, n in ROWS]
    proxy, wire = asyncio.run(replay(honest(), egress=egress, write=compact_write, ordinals=True, compact=True))
    raw, committed = audit(proxy, frames, SYNCS, egress)
    assert raw == committed == [("responses", w, None) for w in range(WINDOWS)] + \
        [("requests", w, None) for w in range(WINDOWS)]
    out = [r for r in proxy.records if r.link == "responses"]
    assert out == run_shaper(frames, READY, SYNCS, Certifier("responses", egress))
    honest_proxy, honest_wire = honest_run
    assert [occupancy(r.grid) for r in out] == [occupancy(r.grid) for r in honest_proxy.records if r.link == "responses"]
    assert [r for r in proxy.records if r.link == "requests"] == \
        [r for r in honest_proxy.records if r.link == "requests"]
    assert proxy.shaper.hands == honest_proxy.shaper.hands and proxy.frames == honest_proxy.frames
    assert wire == expected_wire(frames, SYNCS, egress) and len(wire) * B == len(honest_wire) * cell
    cells = [active.compact_record_data(f) for r in out for row in r.grid for f in row[:row_count(row)]]
    assert cells == [(ORDINAL_OF[name], 1000 * ORDINAL_OF[name] + k,
                      None if k < n - 1 else "stop" if name == "c" else "length")
                     for _, _, name, k, _, n in ROWS[:len(cells)]]
    assert {c[2] for c in cells} == {None, "length", "stop"}


def test_a_compact_record_of_another_request_or_unframed_bytes_is_malformed():
    """The proxy takes a compact record only on its own request's connection, as one (`compact_record_data`): one
    naming another request, one that isn't a record, or one before the stream's request is released faults the
    window `malformed`, and so do bytes left uncut at the stream's end. It refuses `compact` without `ordinals` or
    with frames shorter than a record."""
    egress = dataclasses.replace(EGRESS, frame_bytes=8)
    with pytest.raises(ValueError):
        Proxy(("127.0.0.1", 1), egress, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests", compact=True)
    with pytest.raises(ValueError):
        Proxy(("127.0.0.1", 1), dataclasses.replace(EGRESS, frame_bytes=7), INGRESS, SYNC_SET, SYNCS, origin=0.0,
              ingress_link="requests", ordinals=True, compact=True)

    def compact_proxy(upstream=("127.0.0.1", 1)) -> Proxy:
        return Proxy(upstream, egress, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests", ordinals=True,
                     compact=True)

    for record, anchor in ((active.compact_record(2, 5), 0), (active.compact_record(1, 5)[:4] + b"\x7f\xff\xff\xff", 0),
                           (active.compact_record(1, 5), None)):
        proxy = compact_proxy()
        proxy._egress_record(1, active._Stream(anchor=anchor), 0, record, 1)
        assert proxy.shaper.certifier._fault == Reason.MALFORMED and not proxy.frames
    proxy = compact_proxy()
    proxy._egress_record(1, active._Stream(anchor=0), 0, active.compact_record(1, 5, "length"), 1)
    assert proxy.shaper.certifier._fault is None and proxy.frames == [(0, 1, 0, 0, 1)]

    async def ends_short(tail: bytes) -> Proxy:
        async def handle(reader, writer) -> None:
            await reader.readexactly(active.HEADER.size)
            writer.write(HEAD + active.compact_unit(0, active.compact_record(1, 5)) + tail)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        proxy = compact_proxy(server.sockets[0].getsockname()[:2])
        await proxy._upstream(1, active._Stream(anchor=0))
        server.close()
        return proxy

    for tail, fault in ((b"", None), (b"\0" * 5, Reason.MALFORMED)):
        proxy = asyncio.run(ends_short(tail))
        assert len(proxy.frames) == 1 and proxy.shaper.certifier._fault == fault


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


def test_a_paced_proxys_pacer_writes_its_records_buckets_and_one_it_got_late_faults_its_window(honest_run):
    """With `paced`, the pacer writes every egress bucket, in order, and reports each. On the injected clock a
    bucket's time (origin 0, on the host's monotonic clock) has long passed when its tick hands it, so each reaches the
    pacer late and every egress window fails closed (`monitor-unavailable`), each record published once its last
    bucket's report is in. The wire is the records' grids, and the ingress records are the unpaced run's."""
    async def reported(proxy, t):
        await proxy.until(lambda p: len(p.releases) > t)

    paced, wire = asyncio.run(replay(honest(), probe=reported, paced=0.05))
    assert [r.t for r in paced.releases] == list(range(WINDOWS * EGRESS.buckets))
    assert all(r.written and 0 < r.arrived <= r.start for r in paced.releases)
    egress = [r for r in paced.records if r.link == "responses"]
    assert [(r.window, r.status.reason) for r in egress] == [(w, Reason.MONITOR_UNAVAILABLE) for w in range(WINDOWS)]
    assert wire == b"".join(b"".join(row[: row_count(row)]) + bytes((EGRESS.allocation - row_count(row)) * B)
                            for r in egress for row in r.grid)
    assert [r for r in paced.records if r.link == "requests"] == \
        [r for r in honest_run[0].records if r.link == "requests"]


def test_a_pacers_report_faults_the_open_window_or_the_record_held_for_a_closed_one():
    """A report faults its bucket's window: the open window's certifier, or the record held for a window that closed
    before its last bucket's writes were reported, published then with its first fault kept. A bucket that reached the
    pacer in time and whose release started within the slack faults nothing; one that reached it late, or whose release
    started past the slack, is `monitor-unavailable` at its release's report, and one the link refused
    `delivery-refused` at its writes'."""
    clock, T = VirtualClock(), EGRESS.buckets
    proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  clock=clock, sleep=clock.sleep, paced=0.05)

    def both(t: int, arrived: float, start: float, written: bool) -> None:
        proxy._report(Report(t, arrived, start, None, True))
        proxy._report(Report(t, arrived, start, start + 0.001, written))

    proxy._unwritten[0] = 2
    both(0, -0.04, 0.0, True)
    assert proxy.shaper.certifier._fault is None
    proxy._report(Report(1, 0.001, 0.001, None, True))
    assert proxy.shaper.certifier._fault is Reason.MONITOR_UNAVAILABLE
    record = Record("responses", 1, ((None,) * EGRESS.allocation,) * T, Status())
    held = {1: record, 2: dataclasses.replace(record, window=2, status=Status(Reason.OVERFLOW)),
            3: dataclasses.replace(record, window=3)}
    proxy._held |= held
    proxy._unwritten.update({1: 2, 2: 1, 3: 1})
    both(T, -0.04, 0.0, True)
    assert proxy.records == []
    proxy._report(Report(2 * T - 1, -0.04, 0.03, None, True))
    assert proxy.records == []
    proxy._report(Report(2 * T - 1, -0.04, 0.03, 0.031, True))
    both(3 * T - 1, -0.04, 0.0, False)
    both(4 * T - 1, -0.04, 0.0, False)
    assert proxy.records == [dataclasses.replace(record, status=Status(Reason.MONITOR_UNAVAILABLE)), held[2],
                             dataclasses.replace(held[3], status=Status(Reason.DELIVERY_REFUSED))]
    assert not proxy._held and len(proxy.releases) == 5
    with pytest.raises(ValueError, match="past the slack"):
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests", paced=0.025)


SECRET, SESSION = bytes(range(aead.SECRET_BYTES)), b"session 1"


def pair_halves(secret: bytes = SECRET, session: bytes = SESSION) -> tuple[active.Pair, active.Sealer]:
    """This certifier's half of a sealed pair link, sealing its egress as direction "a>b" and opening the other
    certifier's "b>a", and the other certifier's sealer, whose egress grid is this one's ingress grid."""
    return (active.Pair.of(secret, session, "a>b", "b>a", EGRESS, INGRESS),
            active.Sealer(aead.keys(secret, session, "b>a"), INGRESS))


def test_a_sealed_pair_link_carries_only_ciphertext_and_every_record_is_the_unsealed_runs(honest_run):
    """With the link sealed between two of the auditor's certifiers (`Proxy(pair=…)`), each direction under its own
    keys, the proxy publishes the unsealed run's records, egress and ingress, so every verdict, raw and committed, is
    the unsealed run's: N2 and the timing bound read the same grids. The wire is r sealed units of B + 64 bytes a
    bucket, padding sealed too, so it holds no frame's bytes and no run of zeros, and opened in order under the
    egress direction's keys it is the unsealed run's wire, byte for byte. The same secret seals the run to the same
    wire, and another secret to another wire with the same records. Without a pair (the default) the proxy is
    unsealed."""
    plain, plain_wire = honest_run
    pair, sealer = pair_halves()
    proxy, wire = asyncio.run(replay(honest(), pair=pair, sealer=sealer))
    assert proxy.records == plain.records
    assert audit(proxy, FRAMES, SYNCS) == audit(plain, FRAMES, SYNCS)
    assert (proxy.requests_released, proxy.link_frames_read) == (plain.requests_released, plain.link_frames_read)
    unit = B + aead.TAG_BYTES
    assert len(wire) == WINDOWS * EGRESS.buckets * EGRESS.allocation * unit
    opener = active.Opener(aead.keys(SECRET, SESSION, "a>b"), EGRESS)
    assert b"".join(opener.open(wire[i : i + unit]) for i in range(0, len(wire), unit)) == plain_wire
    assert not opener.refused
    bodies = [body(name, k) for _, _, name, k, _, _ in ROWS]
    assert bytes(64) in plain_wire and any(b in plain_wire for b in bodies)
    assert bytes(64) not in wire and not any(b in wire for b in bodies)

    pair, sealer = pair_halves()
    _, again = asyncio.run(replay(honest(), pair=pair, sealer=sealer))
    pair, sealer = pair_halves(bytes(aead.SECRET_BYTES))
    other, other_wire = asyncio.run(replay(honest(), pair=pair, sealer=sealer))
    assert again == wire and other_wire != wire and len(other_wire) == len(wire)
    assert other.records == plain.records
    assert plain.pair is None and plain_wire == expected_wire(FRAMES, SYNCS)


def sealed_ingress(second: bytes | None) -> Proxy:
    """A proxy on a sealed pair link, its upstream a `Server`, whose peer sends two sealed units: a's request in the
    middle of bucket 0, and `second` (the other certifier's next unit, b's request, when None) in the middle of
    bucket 7, in ingress window 1."""

    async def run() -> Proxy:
        clock = VirtualClock(-TAU / 2)
        server = Server()
        srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
        pair, sealer = pair_halves()
        first = seal_frames(sealer, active.request_frames(request("a", 1), B), 0)
        units = {0: first, 7: seal_frames(sealer, active.request_frames(request("b", 1), B), 1) if second is None
                 else second}
        proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0,
                      ingress_link="requests", clock=clock, sleep=clock.sleep, pair=pair)
        link = LinkReader()
        running = asyncio.create_task(proxy.run(link, NullWriter()))
        sent = 0
        for t in range(WINDOWS * EGRESS.buckets):
            clock.advance(t * TAU)
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
            await server.until(proxy.requests_released)
            if t in units:
                clock.advance(t * TAU + TAU / 2)
                link.chunks.put_nowait(units[t])
                sent += 1
                await proxy.until(lambda p, n=sent: p.link_frames_read >= n)
        await running
        for w in server.writers.values():
            w.close()
        srv.close()
        return proxy

    return asyncio.run(run())


def test_a_sealed_unit_tampered_replayed_or_sealed_at_another_nonce_fails_the_link_closed():
    """The peer's second unit is b's request sealed as its next slot, or that unit with a ciphertext or a tag bit
    flipped, the first unit replayed, b's request sealed at another slot, in another session, under another secret,
    or under this certifier's own egress direction (a reflection). Honest, every ingress window is accepted and both
    requests are released; otherwise the unit doesn't open, nothing of it is released, and its window and every one
    after it are `malformed`, while window 0, before it, is complete."""
    def sealed(keys: aead.Keys, t: int, slot: int, session: str = "b") -> bytes:
        return active.Sealer(keys, INGRESS).seal(t, slot, active.request_frames(request(session, 1), B))

    def flipped(i: int) -> bytes:
        return honest_unit[:i] + bytes([honest_unit[i] ^ 1]) + honest_unit[i + 1 :]

    peer = aead.keys(SECRET, SESSION, "b>a")
    honest_unit = sealed(peer, 0, 1)
    bad = {"ciphertext": flipped(5), "tag": flipped(len(honest_unit) - 1), "replayed": sealed(peer, 0, 0, "a"),
           "slot": sealed(peer, 1, 0),
           "session": sealed(aead.keys(SECRET, b"session 2", "b>a"), 0, 1),
           "secret": sealed(aead.keys(bytes(64), SESSION, "b>a"), 0, 1),
           "reflected": sealed(aead.keys(SECRET, SESSION, "a>b"), 0, 1)}
    proxy = sealed_ingress(None)
    assert [r.status.reason for r in proxy.records if r.link == "requests"] == [None] * WINDOWS
    assert proxy.requests_released == 2 and not proxy.pair.opener.refused
    assert sealed_ingress(honest_unit).records == proxy.records
    for name, unit in bad.items():
        refused = sealed_ingress(unit)
        assert [r.status.reason for r in refused.records if r.link == "requests"] == \
            [None, Reason.MALFORMED, Reason.MALFORMED], name
        assert refused.requests_released == 1 and refused.link_frames_read == 2 and refused.pair.opener.refused, name


def test_a_pair_seals_its_own_egress_grid_and_opens_frames_of_its_ingress_size():
    """The proxy refuses a pair whose sealer is of another grid than its egress, or whose opener's frames aren't its
    ingress frames; a pair's two directions are two names."""
    for egress, peer in ((INGRESS, INGRESS), (EGRESS, GridParams(6, 2, B // 2))):
        with pytest.raises(ValueError, match="pair link seals"):
            Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  pair=active.Pair.of(SECRET, SESSION, "a>b", "b>a", egress, peer))
    with pytest.raises(ValueError, match="two directions"):
        active.Pair.of(SECRET, SESSION, "a>b", "a>b", EGRESS, INGRESS)


class Recorder:
    """A server that keeps what each stream's connection sends it after its `ORDINAL` header, by that ordinal, and
    answers nothing."""

    def __init__(self) -> None:
        self.got: dict[int, bytearray] = {}
        self.writers: list[asyncio.StreamWriter] = []
        self.changed = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.append(writer)
        _, stream, _ = active.HEADER.unpack(await reader.readexactly(active.HEADER.size))
        got = self.got[stream] = bytearray()
        self.changed.set()
        while data := await reader.read(1 << 16):
            got += data
            self.changed.set()

    async def until(self, predicate) -> None:
        while not predicate(self):
            self.changed.clear()
            await self.changed.wait()


def chunked_ingress(directory: Path, requests: list[bytes], cut) -> tuple[Proxy, list, dict[int, bytes]]:
    """A proxy with ordinals whose peer sends `requests`' link frames in the middle of bucket 1, in the reads `cut`
    makes of them (one read a chunk, all at that instant), each record certified as it closes by a fresh signer on
    fixed coins. Returns the proxy, its certified records and what its server got on each ordinal."""

    async def run():
        clock = VirtualClock(-TAU / 2)
        server = Recorder()
        srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
        directory.mkdir()
        signer = signers.signer(directory)
        issuer, certified = Issuer(signer, signers.SESSION, coins=lambda n: b"\x07" * n), []

        def publish(record, params):
            egress = record.link == "responses"
            certified.append(issuer.certify(record, Direction.EGRESS if egress else Direction.INGRESS, params,
                                            parameters_digest(params, SYNC_SET if egress else None)))

        proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0,
                      ingress_link="requests", clock=clock, sleep=clock.sleep, publish=publish, ordinals=True)
        link = LinkReader()
        running = asyncio.create_task(proxy.run(link, NullWriter()))
        wire = b"".join(active.request_frames(r, B) for r in requests)

        def delivered(_: Recorder) -> bool:
            return all(st.delivered is not None for st in proxy.streams.values() if st.anchor is not None)

        for t in range(WINDOWS * EGRESS.buckets):
            clock.advance(t * TAU)
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
            await server.until(delivered)
            if t == 1:
                clock.advance(t * TAU + TAU / 2)
                for chunk in cut(wire):
                    link.chunks.put_nowait(chunk)
                await proxy.until(lambda p: link.chunks.empty() and p.link_frames_read >= len(wire) // B)
                assert proxy.link_frames_read == len(wire) // B
        await running
        for w in server.writers:
            w.close()
        srv.close()
        signer.close()
        return proxy, certified, {k: bytes(v) for k, v in server.got.items()}

    return asyncio.run(run())


def _cuts(wire: bytes, sizes) -> list[bytes]:
    out, at = [], 0
    for n in sizes:
        if at >= len(wire):
            break
        out.append(wire[at : at + n])
        at += n
    return out + ([wire[at:]] if at < len(wire) else [])


def test_a_requests_frames_and_signed_records_are_its_token_counts_whatever_its_chunking(tmp_path):
    """Three id requests of 100, 7 and 40 prompt tokens, sent at one instant in one read, a byte a read, reads that
    end a byte past or short of each frame, TCP segments of 1448 bytes or random cuts: each request takes
    `id_request_frames` of its token count, its server gets it whole on its ordinal, and every certified record, both
    directions, is byte for byte the one-read run's, its signature included. Requests of other fields (ids, model,
    user, max_tokens, sampling, seed) and the same token counts give the same counts in every signed ingress head, only
    the frames' root differing, so each request's end, the first bucket where the signed counts reach its frames and
    those before it, is the bucket the proxy released its last frame in."""
    rng = random.Random(5)

    def fields(length: int, k: int, other: bool = False) -> dict:
        if other:
            return {"model": "m", "user": "u" * 62, "prompt": [2**32 - 1] * length, "max_tokens": 1,
                    "temperature": 1.5, "top_p": 0.95, "seed": 2**32 - 1}
        return {"model": "llama-3.1-8b-instruct", "user": f"varied.{k:05d}",
                "prompt": [rng.randrange(1 << 17) for _ in range(length)], "max_tokens": 64, "temperature": 0.0,
                "top_p": 1.0, "seed": k}

    lengths = (100, 7, 40)
    requests = [active.id_request(**fields(n, k), frame_bytes=B) for k, n in enumerate(lengths)]
    others = [active.id_request(**fields(n, k, True), frame_bytes=B) for k, n in enumerate(lengths)]
    frames = [active.id_request_frames(n, B) for n in lengths]
    assert frames == [3, 1, 2] and [len(r) for r in requests] == [len(r) for r in others]
    n = sum(len(active.request_frames(r, B)) for r in requests)
    cut = random.Random(7)
    chunkings = {"one read": [n], "a byte a read": [1] * n, "a byte past each frame": [B + 1] * n,
                 "a byte short of each frame": [B - 1] * n, "TCP segments": [1448] * n,
                 "random": [cut.randint(1, 2 * B) for _ in range(n)]}
    key = LmsKey(signers.SMALL, lms.public_key(signers.key()))
    runs = {name: chunked_ingress(tmp_path / name.replace(" ", "-"), requests, lambda w, s=sizes: _cuts(w, s))
            for name, sizes in chunkings.items()}
    proxy, certified, got = runs["one read"]
    assert None not in certified and all(c.verify(key, signers.SESSION) for c in certified)
    heads = [c.head for c in certified if c.direction is Direction.INGRESS]
    assert [h.status.reason for h in heads] == [None] * WINDOWS
    counts = [n for h in sorted(heads, key=lambda h: h.window) for n in h.counts]
    assert sum(counts) == sum(frames) == proxy.link_frames_read and proxy.requests_released == len(requests)
    assert got == {k + 1: r for k, r in enumerate(requests)}
    for name, (other, other_certified, other_got) in runs.items():
        assert other_certified == certified, name
        assert (other.link_frames_read, other.requests_released, other_got) == \
            (proxy.link_frames_read, proxy.requests_released, got), name

    released, ends = list(itertools.accumulate(counts)), []
    for total in itertools.accumulate(frames):
        ends.append(next(b for b, m in enumerate(released) if m >= total))
    assert ends == [proxy.streams[k + 1].anchor for k in range(len(requests))] == [3, 3, 4]
    changed, changed_certified, _ = chunked_ingress(tmp_path / "others", others, lambda w: _cuts(w, [B + 1] * n))
    changed_heads = [c.head for c in changed_certified if c.direction is Direction.INGRESS]
    assert [h.counts for h in changed_heads] == [h.counts for h in heads]
    assert [h.root != o.root for h, o in zip(heads, changed_heads)] == [sum(h.counts) > 0 for h in heads]
    assert [changed.streams[k + 1].anchor for k in range(len(others))] == ends


def registration(window: int, provenance: dict | None = None) -> Registration:
    """A stand-in for the proof system's registration of PoUW window `window` (`Stream.register`'s): a record whose
    provenance names the window, with its receipt and digest."""
    record = {"window": {"kind": "served", "window": window} if provenance is None else provenance}
    return Registration(record, R.receipt(record, "test", received=str(window)), R.digest(record), window)


def salted_run(seed: int, **salting):
    """The honest run with the salt on, each salt sampled from `random.Random(seed)`. Returns the proxy, the wire, the
    salts sampled by PoUW window, the server, and at each tick, for each of PoUW windows 0 … WINDOWS − 1, whether `reveal`
    gave its salt on its registration (True) or refused (None), so each window's True is the one tick it is revealed."""
    rng, drawn, revealable = random.Random(seed), {}, []

    def source(window: int, n: int) -> bytes:
        drawn[window] = rng.randbytes(n)
        return drawn[window]

    def probe(proxy, t):
        ok = []
        for p in range(WINDOWS):
            try:
                ok.append(proxy.reveal(p, registration(p)) == drawn[p])
            except RuntimeError:
                ok.append(None)
        revealable.append(ok)

    server = Server()
    proxy, wire = asyncio.run(replay(honest(), salt=source, server=server, probe=probe, **salting))
    return proxy, wire, drawn, server, revealable


def test_the_salt_reaches_only_the_blocks_side_and_no_record_depends_on_it(honest_run):
    """With the salt on, each window's salt reaches the server on a connection of its own, and nothing else carries
    it: the wire the peer reads and the egress records are the unsalted run's, byte for byte. Each window's ingress
    record holds one salt frame at its first bucket, carrying the placeholder, and is otherwise the unsalted run's, so
    every verdict is too, and two runs on different salts publish the same records. `reveal` refuses a window until its
    ingress record is published, gives the salt on its registration at that tick, and refuses it after; `check_salts`
    accepts the revealed
    salts and rejects a missing one, one of another width, a record with a second salt frame and the unsalted run."""
    plain, plain_wire = honest_run
    proxy, wire, drawn, server, revealable = salted_run(7)
    other, other_wire, other_drawn, _, _ = salted_run(8)
    assert sorted(drawn) == list(range(WINDOWS)) and {len(s) for s in drawn.values()} == {active.SALT_BYTES}
    assert other_drawn != drawn and other.records == proxy.records and other_wire == wire
    assert proxy.salts_released == proxy.salts_delivered == WINDOWS
    assert server.salts == [active.salt_frame(w, drawn[w], active.SALT_FRAME_BYTES) for w in range(WINDOWS)]
    assert len(server.writers) == len(SESSIONS)

    assert wire == plain_wire
    assert [r for r in proxy.records if r.link == "responses"] == [r for r in plain.records if r.link == "responses"]
    ingress = [r for r in proxy.records if r.link == "requests"]
    for r, p in zip(ingress, [r for r in plain.records if r.link == "requests"], strict=True):
        assert p.grid[0] == (None,) * INGRESS.allocation and r.grid[1:] == p.grid[1:] and r.status == p.status
        assert r.grid[0] == (active.salt_frame(r.window, active.PLACEHOLDER, B), *(None,) * (INGRESS.allocation - 1))
    held = [s for r in proxy.records for row in r.grid for s in row if s is not None]
    assert not any(salt in s for salt in drawn.values() for s in held)
    assert not any(salt in wire for salt in drawn.values())
    assert audit(proxy, FRAMES, SYNCS) == audit(plain, FRAMES, SYNCS)

    T = INGRESS.buckets
    assert revealable == [[True if t == (w + 1) * T - 1 else None for w in range(WINDOWS)] for t in range(WINDOWS * T)]
    with pytest.raises(RuntimeError):
        proxy.reveal(WINDOWS, registration(WINDOWS))
    with pytest.raises(ValueError):
        plain.reveal(0, registration(0))

    a = Audit()
    assert active.check_salts(a, "requests", proxy.records, WINDOWS, drawn) == \
        [Verdict("requests", w, None) for w in range(WINDOWS)] and a.accepted
    short = drawn | {1: drawn[1][:32]}
    missing = {w: s for w, s in drawn.items() if w != 2}
    twice = dataclasses.replace(ingress[1], grid=((ingress[1].grid[0][0],) * 2, *ingress[1].grid[1:]))
    for salts, bad, records in ((short, {1}, proxy.records), (missing, {2}, proxy.records),
                                (drawn, {1}, [ingress[0], twice, ingress[2]]),
                                (drawn, set(range(WINDOWS)), plain.records)):
        verdicts = active.check_salts(Audit(), "requests", records, WINDOWS, salts)
        assert [v.violation for v in verdicts] == ["salt" if w in bad else None for w in range(WINDOWS)]
    with pytest.raises(ValueError):
        active.check_salts(Audit(), "requests", [commit(r, INGRESS) for r in ingress], WINDOWS, drawn)
    with pytest.raises(ValueError):
        Proxy(("127.0.0.1", 9), EGRESS, GridParams(6, 2, active.SALT_FRAME_BYTES - 1), SYNC_SET, SYNCS, origin=0.0,
              ingress_link="requests", salt=active.fresh_salt)


def test_one_salt_per_pouw_window_of_its_width(honest_run):
    """With PoUW windows of two certifier windows and 256-bit salts, the proxy draws one 256-bit salt per PoUW window,
    at its first bucket: PoUW window 0's at certifier window 0, 1's at window 2. Window 1's ingress record is the
    unsalted run's, and the server receives each salt once. `reveal` gives PoUW window 0's salt at the tick windows 0 and
    1 are published (t_d), and only then, and never PoUW window 1's, whose window 3 the run doesn't reach. `check_salts` reads the salt
    frames at windows 0 and 2, of salts of that width."""
    plain, _ = honest_run
    proxy, _, drawn, server, revealable = salted_run(9, pouw_window=2, salt_bits=256)
    assert sorted(drawn) == [0, 1] and {len(s) for s in drawn.values()} == {32}
    assert server.salts == [active.salt_frame(p, drawn[p], active.HEADER.size + 8 + 32) for p in (0, 1)]
    ingress = {r.window: r for r in proxy.records if r.link == "requests"}
    unsalted = {r.window: r for r in plain.records if r.link == "requests"}
    assert ingress[1] == unsalted[1]
    for w, p in ((0, 0), (2, 1)):
        assert ingress[w].grid[0][0] == active.salt_frame(p, active.PLACEHOLDER, B)
        assert ingress[w].grid[1:] == unsalted[w].grid[1:]
    T = INGRESS.buckets
    assert revealable == [[True if p == 0 and t == 2 * T - 1 else None for p in range(WINDOWS)]
                          for t in range(WINDOWS * T)]
    records = list(ingress.values())
    assert active.check_salts(Audit(), "requests", records, WINDOWS, drawn, pouw_window=2, salt_bits=256) == \
        [Verdict("requests", 0, None), Verdict("requests", 2, None)]
    assert [v.violation for v in active.check_salts(Audit(), "requests", records, WINDOWS, drawn, pouw_window=2)] == \
        ["salt", "salt"]
    assert [v.violation for v in active.check_salts(Audit(), "requests", records, WINDOWS, drawn, salt_bits=256)] == \
        [None, "salt", "salt"]
    for bad in ({"salt_bits": 0}, {"salt_bits": 100}, {"pouw_window": 0}, {"pouw_window": 1.0}):
        with pytest.raises(ValueError):
            Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  salt=active.fresh_salt, **bad)


def test_reveal_takes_the_windows_registration_and_fails_closed_without_it():
    """`reveal(p, registration)` gives PoUW window p's salt only on the registration the proof system returned for p.
    It refuses none, another window's, a receipt or digest that isn't the record's, a receipt of another format, a
    provenance the reader (`registered`) doesn't read as p, and anything that isn't a `Registration`; a window whose
    last ingress record is unpublished; and one whose salt a proxy back from maintenance never drew. A reader the
    protocol gives reads its own provenance."""
    drawn: list[int] = []

    def proxy(windows: int, **kwargs) -> Proxy:
        def source(w: int, n: int) -> bytes:
            drawn.append(w)
            return bytes([w + 1]) * n

        p = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  clock=lambda: 0.0, salt=source, **kwargs)
        for _ in range(windows * INGRESS.buckets):
            p.bucketer.tick()
        return p

    one, reg = proxy(1), registration(0)
    assert one.reveal(0, reg) == bytes([1]) * active.SALT_BYTES
    other = {"window": {"kind": "served", "window": 1}}
    pouw = registration(0, {"kind": "served", "context": "c", "public": {}})
    refused = [None, registration(1), pouw, dataclasses.replace(reg, record=other),
               dataclasses.replace(reg, receipt={**reg.receipt, "registration": R.digest(other)}),
               dataclasses.replace(reg, digest=R.digest(other)), dataclasses.replace(reg, receipt=None),
               dataclasses.replace(reg, receipt={**reg.receipt, "format": "verity/registration-receipt/v0"}),
               dataclasses.replace(reg, record=None), registration(0, {"kind": "served", "window": 0, "row": 3}),
               registration(0, {"kind": "served", "window": True}), {"record": reg.record, "receipt": reg.receipt}]
    for r in refused:
        with pytest.raises(RuntimeError, match="no registration"):
            one.reveal(0, r)
    with pytest.raises(RuntimeError, match="published"):
        one.reveal(1, registration(1))
    reading = proxy(1, registered=lambda r: 0 if r.record["window"].get("context") == "c" else None)
    assert reading.reveal(0, pouw) == bytes([1]) * active.SALT_BYTES
    with pytest.raises(RuntimeError, match="no registration"):
        reading.reveal(0, reg)
    drawn.clear()
    back = proxy(2, pouw_window=2, start=1)
    assert drawn == [1]
    with pytest.raises(RuntimeError, match="wasn't drawn"):
        back.reveal(0, reg)
    with pytest.raises(RuntimeError, match="published"):
        back.reveal(1, registration(1))


def prover_signer(seed: bytes = b"prover"):
    """The prover's LMS key as the certifier registers it (`LmsKey`), and a signer of `weights_message`s at one-time
    keys 0, 1, ... in turn."""
    k, leaves = signers.key(seed), itertools.count()
    public = LmsKey(signers.SMALL, lms.public_key(k))
    return public, lambda message: lms.sign_at(k, next(leaves), message).encode()


def test_the_weights_are_recorded_only_before_the_salt_and_given_on_reveals_conditions():
    """`commit_weights(p, root, signature)` records PoUW window p's 64-byte weights' root once, while p's first ingress
    tick, the salt's sample, is still ahead, and only the registered prover's: the signature verifies under `prover_key`
    over `weights_message`, which binds the ingress link, p, the root and the prover's context. It refuses a root after
    that tick, a second root, a root of another length or type, a window the proxy doesn't serve, a proxy without the
    salt or without a prover key, a signature by anyone else, and one over another window, root or context; a proxy back
    from maintenance past p's first tick refuses p. `committed(p, registration)` gives the root on `reveal`'s conditions,
    as often as asked, and refuses a window whose weights weren't recorded."""
    key, sign = prover_signer()
    _, forge = prover_signer(b"someone else")

    def proxy(**kwargs) -> Proxy:
        return Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                     clock=lambda: 0.0, salt=lambda w, n: bytes([w + 1]) * n, prover_key=key, prover_context=b"epoch 7",
                     **kwargs)

    def signed(w: int, root: bytes, link: str = "requests", context: bytes = b"epoch 7") -> bytes:
        return sign(active.weights_message(link, w, root, context))

    T, roots = INGRESS.buckets, [bytes([w]) * active.WEIGHTS_ROOT_BYTES for w in range(WINDOWS)]
    p = proxy()
    for bad in (forge(active.weights_message("requests", 0, roots[0], b"epoch 7")), signed(1, roots[0]),
                signed(0, roots[1]), signed(0, roots[0], context=b"epoch 8"), signed(0, roots[0], link="responses"),
                b"", None):
        with pytest.raises(RuntimeError, match="isn't signed by the registered prover"):
            p.commit_weights(0, roots[0], bad)
    p.commit_weights(0, roots[0], signed(0, roots[0]))
    for bad in (roots[0][:32], roots[0].hex(), None):
        with pytest.raises(ValueError, match="64 bytes"):
            p.commit_weights(1, bad, b"")
    for w in (-1, WINDOWS, True):
        with pytest.raises(ValueError, match="serves"):
            p.commit_weights(w, roots[0], b"")
    with pytest.raises(ValueError, match="salt is off"):
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0,
              ingress_link="requests").commit_weights(0, roots[0], b"")
    keyless = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                    salt=active.fresh_salt)
    with pytest.raises(RuntimeError, match="no prover key"):
        keyless.commit_weights(0, roots[0], signed(0, roots[0]))
    p.bucketer.tick()
    with pytest.raises(RuntimeError, match="drawn"):
        p.commit_weights(0, roots[1], signed(0, roots[1]))
    for _ in range(T - 1):
        p.bucketer.tick()
    p.commit_weights(1, roots[1], signed(1, roots[1]))
    with pytest.raises(RuntimeError, match="already recorded"):
        p.commit_weights(1, roots[2], signed(1, roots[2]))
    with pytest.raises(RuntimeError, match="published"):
        p.committed(1, registration(1))
    for _ in range((WINDOWS - 1) * T):
        p.bucketer.tick()
    with pytest.raises(RuntimeError, match="drawn"):
        p.commit_weights(2, roots[2], signed(2, roots[2]))
    assert [p.committed(w, registration(w)) for w in (0, 1, 0)] == [roots[0], roots[1], roots[0]]
    assert p.reveal(0, registration(0)) == bytes([1]) * active.SALT_BYTES
    assert p.committed(0, registration(0)) == roots[0]
    with pytest.raises(RuntimeError, match="no weights"):
        p.committed(2, registration(2))
    with pytest.raises(RuntimeError, match="no registration"):
        p.committed(0, registration(1))

    back = proxy(start=1)
    with pytest.raises(RuntimeError, match="passed"):
        back.commit_weights(0, roots[0], signed(0, roots[0]))
    back.commit_weights(1, roots[1], signed(1, roots[1]))


def test_a_served_proxy_counts_the_salt_frame_exactly_when_the_salt_is_on():
    """A served link's profile counts the certifier's own frame (`own` 1) exactly when the salt is on, at a PoUW window
    of one certifier window, so every window's first ingress frame is the salt's: the proxy refuses a profile without
    it under the salt, one with it and no salt, and a longer PoUW window."""
    def proxy(own: int, **kwargs) -> Proxy:
        return Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                     clock=lambda: 0.0, served=served.ServedProfile(2, 1, WINDOWS, own), **kwargs)

    p = proxy(1, salt=lambda w, n: bytes([w + 1]) * n)
    for w in range(WINDOWS):
        for t in range(INGRESS.buckets):
            p.bucketer.tick()
            assert p.bucketer.own == ({0: w} if t == 0 else {})
    proxy(0)
    for own, kwargs in ((0, {"salt": active.fresh_salt}), (1, {}), (1, {"salt": active.fresh_salt, "pouw_window": 3})):
        with pytest.raises(ValueError, match="counts the salt frame"):
            proxy(own, **kwargs)


def test_reveal_gives_each_salt_once_and_only_until_its_deadline():
    """`reveal` gives a PoUW window's salt once: a second registration naming the window, or the same one again, gets
    none. It refuses once the certifier's clock is more than `reveal_deadline` past the window's end (one PoUW window by
    default), so a registration made after the window's work gets no salt; `committed` keeps that deadline. A deadline
    that isn't a positive number of seconds is refused."""
    T, now = INGRESS.buckets, [0.0]
    window_s = T * INGRESS.bucket_ms / 1000

    def proxy(**kwargs) -> Proxy:
        p = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  clock=lambda: now[0], salt=lambda w, n: bytes([w + 1]) * n, **kwargs)
        for _ in range(WINDOWS * T):
            p.bucketer.tick()
        return p

    p = proxy()
    assert p.reveal_deadline == window_s
    now[0] = 1 * window_s + window_s
    assert p.reveal(0, registration(0)) == bytes([1]) * active.SALT_BYTES
    with pytest.raises(RuntimeError, match="revealed already"):
        p.reveal(0, registration(0))
    now[0] = 2 * window_s + window_s + 1e-3
    with pytest.raises(RuntimeError, match="ended more than"):
        p.reveal(1, registration(1))
    with pytest.raises(RuntimeError, match="ended more than"):
        p.committed(1, registration(1))
    assert p.reveal(2, registration(2)) == bytes([3]) * active.SALT_BYTES
    late = proxy(reveal_deadline=10 * window_s)
    assert [late.reveal(w, registration(w)) for w in range(WINDOWS)] == \
        [bytes([w + 1]) * active.SALT_BYTES for w in range(WINDOWS)]
    for bad in (0, -1.0, "1", float("nan")):
        with pytest.raises(ValueError, match="deadline"):
            Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                  salt=active.fresh_salt, reveal_deadline=bad)


def lms_key(seed: bytes) -> LmsKey:
    return LmsKey(signers.SMALL, lms.public_key(signers.key(seed)))


#: a proxy's run id in the tests that fix it (`Proxy(run_nonce=…)`), and another run's
RUN, OTHER_RUN = bytes([9]) * active.RUN_ID_BYTES, bytes([10]) * active.RUN_ID_BYTES


def test_a_signed_reveal_takes_only_the_challengers_request_and_answers_under_the_certifiers_key(tmp_path):
    """With a challenger key and a signer, `reveal(p, registration, request)` and `committed` take only a request the
    registered challenger signed over `reveal_request_message`, which binds the certifier's key id and run id, the
    ingress link, p, the registration's digest and the kind. They refuse none (the two-argument call), one signed by
    anyone else, and one for another certifier, run, window, registration, kind or link, and a refusal spends nothing.
    A registration made up whole, receipt and all, which a keyless proxy takes, is refused unless the challenger signs
    it. They answer a `Revealed` that names the certifier's key id and run and verifies under the certifier's key over
    `reveal_response_message`, and under no other key, run, window, value, kind, registration or link. The once rule
    and the published condition are `reveal`'s. A signer that refuses spends the reveal and gives nothing. A proxy
    takes both keys or neither; a keyless one takes no request and serves no endpoint. Each proxy draws its own
    32-byte run id unless given a source. A deployed signature (`lms.DEPLOYED`), in hex, fits a request line with room
    for the registration."""
    challenger_key, sign = prover_signer(b"challenger")
    _, forge = prover_signer(b"someone else")
    prover_key, prove = prover_signer()
    certifier = signers.signer(tmp_path)
    key, root = certifier.public_key(), bytes([7]) * active.WEIGHTS_ROOT_BYTES
    stranger = lms_key(b"someone else")

    def proxy(**keys) -> Proxy:
        p = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=lambda: 0.0,
                  ingress_link="requests", salt=lambda w, n: bytes([w + 1]) * n, prover_key=prover_key,
                  **({"run_nonce": lambda n: RUN} | keys))
        p.commit_weights(0, root, prove(active.weights_message("requests", 0, root)))
        for _ in range(2 * INGRESS.buckets):
            p.bucketer.tick()
        return p

    def asks(w: int, reg: Registration, what: bytes = active.SALT_KIND, link: str = "requests", by=sign,
             key_id: bytes = key.key_id, run_id: bytes = RUN) -> bytes:
        return by(active.reveal_request_message(key_id, run_id, link, w, reg.digest, what))

    p, reg = proxy(challenger_key=challenger_key, signer=certifier), registration(0)
    assert p.run_id == RUN
    record = {"window": {"kind": "served", "window": 1}, "made": "up"}
    made_up = Registration(record, R.receipt(record, "anyone"), R.digest(record), 1)
    for w, r, request in ((0, reg, None), (0, reg, b""), (0, reg, asks(0, reg, by=forge)), (0, reg, asks(1, reg)),
                          (0, reg, asks(0, registration(1))), (0, reg, asks(0, reg, active.WEIGHTS_KIND)),
                          (0, reg, asks(0, reg, link="responses")), (0, reg, asks(0, reg, run_id=OTHER_RUN)),
                          (0, reg, asks(0, reg, key_id=stranger.key_id)), (0, None, asks(0, reg)),
                          (1, made_up, asks(1, made_up, by=forge))):
        with pytest.raises(RuntimeError, match="isn't signed by the registered challenger for this certifier's run"):
            p.reveal(w, r, request)
    with pytest.raises(RuntimeError, match="isn't signed by the registered challenger"):
        p.committed(0, reg, asks(0, reg))
    with pytest.raises(RuntimeError, match="published"):
        p.reveal(2, registration(2), asks(2, registration(2)))

    salt = p.reveal(0, reg, asks(0, reg))
    assert isinstance(salt, Revealed) and (salt.window, salt.value) == (0, bytes([1]) * active.SALT_BYTES)
    assert (salt.key_id, salt.run_id) == (key.key_id, RUN)
    assert salt.verify(key, RUN, "requests", reg.digest, active.SALT_KIND)
    for k, answer, run, link, digest, what in (
            (stranger, salt, RUN, "requests", reg.digest, active.SALT_KIND),
            (stranger, dataclasses.replace(salt, key_id=stranger.key_id), RUN, "requests", reg.digest,
             active.SALT_KIND),
            (key, salt, OTHER_RUN, "requests", reg.digest, active.SALT_KIND),
            (key, dataclasses.replace(salt, run_id=OTHER_RUN), OTHER_RUN, "requests", reg.digest, active.SALT_KIND),
            (key, dataclasses.replace(salt, run_id=b"short"), b"short", "requests", reg.digest, active.SALT_KIND),
            (key, dataclasses.replace(salt, window=1), RUN, "requests", reg.digest, active.SALT_KIND),
            (key, dataclasses.replace(salt, value=bytes(active.SALT_BYTES)), RUN, "requests", reg.digest,
             active.SALT_KIND),
            (key, salt, RUN, "requests", reg.digest, active.WEIGHTS_KIND),
            (key, salt, RUN, "requests", registration(1).digest, active.SALT_KIND),
            (key, salt, RUN, "responses", reg.digest, active.SALT_KIND)):
        assert not answer.verify(k, run, link, digest, what)
    with pytest.raises(RuntimeError, match="revealed already"):
        p.reveal(0, reg, asks(0, reg))
    weights = p.committed(0, reg, asks(0, reg, active.WEIGHTS_KIND))
    assert weights.value == root and weights.verify(key, RUN, "requests", reg.digest, active.WEIGHTS_KIND)
    assert not weights.verify(key, RUN, "requests", reg.digest, active.SALT_KIND)
    with pytest.raises(RuntimeError, match="no weights"):
        p.committed(1, registration(1), asks(1, registration(1), active.WEIGHTS_KIND))
    assert p.reveal(1, made_up, asks(1, made_up)).value == bytes([2]) * active.SALT_BYTES
    with pytest.raises(ValueError, match="a reveal is of"):
        active.reveal_request_message(key.key_id, RUN, "requests", 0, reg.digest, b"other")
    with pytest.raises(ValueError, match="run id is 32 bytes"):
        active.reveal_request_message(key.key_id, RUN[1:], "requests", 0, reg.digest, active.SALT_KIND)
    assert 2 * lms.DEPLOYED.signature_bytes() < active.REVEAL_LINE_BYTES // 8

    drawn = [Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests").run_id
             for _ in range(2)]
    assert [len(r) for r in drawn] == [active.RUN_ID_BYTES] * 2 and drawn[0] != drawn[1]
    for bad in (lambda n: RUN[1:], lambda n: RUN.hex()):
        with pytest.raises(ValueError, match="run id is 32 bytes"):
            proxy(run_nonce=bad)

    held = proxy(challenger_key=challenger_key, signer=LmsSigner(certifier.key, certifier.state))
    with pytest.raises(RuntimeError, match="signer refused"):
        held.reveal(0, reg, asks(0, reg))
    with pytest.raises(RuntimeError, match="revealed already"):
        held.reveal(0, reg, asks(0, reg))

    for keys in ({"challenger_key": challenger_key}, {"signer": certifier}):
        with pytest.raises(ValueError, match="neither"):
            proxy(**keys)
    keyless = proxy()
    with pytest.raises(RuntimeError, match="takes no signed request"):
        keyless.reveal(0, reg, asks(0, reg))
    assert keyless.reveal(1, made_up) == bytes([2]) * active.SALT_BYTES
    with pytest.raises(ValueError, match="challenger key and a signer"):
        asyncio.run(active.serve_reveals(keyless, "127.0.0.1", 0))


def test_reveals_over_the_endpoint_beside_a_run_are_signed_both_ways_and_given_once(tmp_path):
    """The endpoint (`serve_reveals`) beside a salted proxy's run, on its loop, over real localhost sockets on the
    injected clock, with the challenger's `RemoteCertifier` in a thread: at each PoUW window's t_d the client gets the
    window's salt and the weights' root the prover recorded, each verified under the certifier's key, as plain bytes;
    one tick before, the certifier refuses (unpublished) and the client raises its reason. A second reveal of the
    window is refused, the once rule over the network; a request by anyone but the registered challenger is refused
    and spends nothing; a window with no weights recorded is refused; a client holding another certifier's key, or
    another run's id, asks for that certifier's run and is refused; and a line that isn't a request, or is longer than
    `REVEAL_LINE_BYTES`, is refused. The run's records are the salted run's, and the audit accepts them and the salts
    the client got."""
    T, horizon = INGRESS.buckets, WINDOWS * EGRESS.buckets
    rng, drawn = random.Random(7), {}

    def source(window: int, n: int) -> bytes:
        drawn[window] = rng.randbytes(n)
        return drawn[window]

    for name in ("certifier", "challenger", "forger"):
        (tmp_path / name).mkdir()
    certifier = signers.signer(tmp_path / "certifier")
    challenger = signers.signer(tmp_path / "challenger", b"challenger")
    forger = signers.signer(tmp_path / "forger", b"someone else")
    prover_key, prove = prover_signer()
    roots = {w: bytes([w]) * active.WEIGHTS_ROOT_BYTES for w in range(1, WINDOWS)}
    got: dict[int, list] = defaultdict(list)
    endpoint: dict = {}

    async def ask(client: RemoteCertifier, call: str, w: int) -> tuple[str, bytes | str]:
        try:
            return "gave", await asyncio.to_thread(getattr(client, call), w, registration(w))
        except RuntimeError as e:
            return "refused", str(e)

    async def raw(line: bytes) -> dict:
        reader, writer = await asyncio.open_connection(*endpoint["at"])
        writer.write(line)
        writer.write_eof()
        answer = json.loads(await reader.readline())
        writer.close()
        return answer

    async def probe(proxy: Proxy, t: int) -> None:
        if t == 0:
            endpoint["server"] = await active.serve_reveals(proxy, "127.0.0.1", 0)
            at = endpoint["at"] = endpoint["server"].sockets[0].getsockname()[:2]
            key, run = certifier.public_key(), proxy.run_id
            endpoint["challenger"] = RemoteCertifier(*at, key, challenger, "requests", run)
            endpoint["forger"] = RemoteCertifier(*at, key, forger, "requests", run)
            endpoint["misled"] = RemoteCertifier(*at, lms_key(b"someone else"), challenger, "requests", run)
            endpoint["stale"] = RemoteCertifier(*at, key, challenger, "requests", OTHER_RUN)
            endpoint["raw"] = [await raw(line) for line in (b"not json\n", b'{"window": 0, "kind": "salt"}\n', b"[1]\n",
                                                             b"x" * (active.REVEAL_LINE_BYTES + 1))]
        for w in range(WINDOWS):
            published = (w + 1) * T - 1
            if t == published - 1:
                got[w].append(await ask(endpoint["challenger"], "reveal", w))
            if t == published:
                for client, call in (("forger", "reveal"), ("challenger", "committed"), ("misled", "committed"),
                                     ("stale", "reveal"), ("challenger", "reveal"), ("challenger", "reveal")):
                    got[w].append(await ask(endpoint[client], call, w))
            if w and t == w * T - 1:
                proxy.commit_weights(w, roots[w], prove(active.weights_message("requests", w, roots[w])))
        if t == horizon - 1:
            endpoint["server"].close()
            await endpoint["server"].wait_closed()

    proxy, _ = asyncio.run(replay(honest(), salt=source, probe=probe, prover_key=prover_key,
                                  challenger_key=challenger.public_key(), signer=certifier))
    for w in range(WINDOWS):
        early, forged, weights, misled, stale, salt, again = got[w]
        assert early == ("refused", f"the certifier refused window {w}'s salt: window {w}'s ingress records aren't all "
                                    "published, so its salt isn't revealed")
        unsigned = [("refused", f"the certifier refused window {w}'s {kind}: the request for window {w}'s {kind} isn't "
                                "signed by the registered challenger for this certifier's run") for kind in ("salt",
                                                                                                         "weights")]
        assert forged == stale == unsigned[0] and misled == unsigned[1]
        nothing = ("refused", "the certifier refused window 0's weights: no weights were recorded before PoUW window 0's "
                              "salt")
        assert weights == (("gave", roots[w]) if w else nothing)
        assert salt == ("gave", drawn[w])
        assert again == ("refused", f"the certifier refused window {w}'s salt: window {w}'s salt is revealed already")
    assert [list(a) for a in endpoint["raw"]] == [["refused"]] * 4
    assert endpoint["raw"][0] == endpoint["raw"][1] == endpoint["raw"][2] != endpoint["raw"][3]
    assert "one line of at most" in endpoint["raw"][3]["refused"]

    assert proxy.records == salted_run(7)[0].records
    assert audit(proxy, FRAMES, SYNCS)[0] == [("responses", w, None) for w in range(WINDOWS)] + \
        [("requests", w, None) for w in range(WINDOWS)]
    assert [v.violation for v in active.check_salts(Audit(), "requests", proxy.records, WINDOWS,
                                                    {w: got[w][5][1] for w in range(WINDOWS)})] == [None] * WINDOWS


def test_two_runs_on_one_link_name_neither_take_nor_give_each_others_reveals(tmp_path):
    """Two proxies with the same ingress link name, certifier key (one signer, as a certifier restarted on its host
    holds it) and PoUW window, in runs `RUN` and `OTHER_RUN` (their p's collide): a request the challenger
    signed for one run is refused by the other and spends nothing there, and each answer names its own run and
    verifies for it and not for the other, so a consumer keying on (key id, run id, p) never takes one for the other.
    A `RemoteCertifier` refuses another run's answer that an endpoint replays whole, or relabels with its run, and
    takes the same answer for its own run."""
    (tmp_path / "challenger").mkdir()
    certifier = signers.signer(tmp_path)
    challenger = signers.signer(tmp_path / "challenger", b"challenger")
    key, reg = certifier.public_key(), registration(0)

    def proxy(run: bytes, fill: int) -> Proxy:
        p = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=lambda: 0.0,
                  ingress_link="requests", salt=lambda w, n: bytes([fill]) * n,
                  challenger_key=challenger.public_key(), signer=certifier, run_nonce=lambda n: run)
        for _ in range(2 * INGRESS.buckets):
            p.bucketer.tick()
        return p

    def asks(run: bytes) -> bytes:
        return challenger.sign(active.reveal_request_message(key.key_id, run, "requests", 0, reg.digest,
                                                             active.SALT_KIND))

    runs = {RUN: proxy(RUN, 1), OTHER_RUN: proxy(OTHER_RUN, 2)}
    other = {RUN: OTHER_RUN, OTHER_RUN: RUN}
    for run, p in runs.items():
        with pytest.raises(RuntimeError, match="isn't signed by the registered challenger for this certifier's run"):
            p.reveal(0, reg, asks(other[run]))
    answers = {run: p.reveal(0, reg, asks(run)) for run, p in runs.items()}
    assert {(a.key_id, a.run_id, a.window): a.value for a in answers.values()} == {
        (key.key_id, RUN, 0): bytes([1]) * active.SALT_BYTES,
        (key.key_id, OTHER_RUN, 0): bytes([2]) * active.SALT_BYTES}
    for run, answer in answers.items():
        assert answer.verify(key, run, "requests", reg.digest, active.SALT_KIND)
        assert not answer.verify(key, other[run], "requests", reg.digest, active.SALT_KIND)
        assert not dataclasses.replace(answer, run_id=other[run]).verify(key, other[run], "requests", reg.digest,
                                                                         active.SALT_KIND)

    async def replaying(answer: Revealed, client_runs: tuple[bytes, ...]) -> list[bytes | str]:
        line = json.dumps({"window": answer.window} | {f: getattr(answer, f).hex() for f in (
            "value", "signature", "key_id", "run_id")}).encode() + b"\n"

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.readline()
            writer.write(line)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=active.REVEAL_LINE_BYTES)
        at, out = server.sockets[0].getsockname()[:2], []
        for run in client_runs:
            try:
                out.append(await asyncio.to_thread(RemoteCertifier(*at, key, challenger, "requests", run).reveal, 0,
                                                   reg))
            except RuntimeError as e:
                out.append(str(e))
        server.close()
        await server.wait_closed()
        return out

    assert asyncio.run(replaying(answers[RUN], (OTHER_RUN, RUN))) == [
        f"the answer for window 0's salt is another run's, not run {OTHER_RUN.hex()[:16]}…'s",
        bytes([1]) * active.SALT_BYTES]
    assert asyncio.run(replaying(dataclasses.replace(answers[RUN], run_id=OTHER_RUN), (OTHER_RUN,))) == [
        "the answer for window 0's salt isn't signed by the certifier's key"]
    with pytest.raises(ValueError, match="run id is 32 bytes"):
        RemoteCertifier("127.0.0.1", 9, key, challenger, "requests", RUN[1:])


def test_a_proxy_and_its_client_name_one_certifier_run_which_its_answers_bind(tmp_path):
    """`certifier_run`, (key id, run id, link), is one identity on both sides: a proxy's (its signer's key id, `run_id`
    and the ingress link the reveal messages bind, not the egress one) equals a `RemoteCertifier` client's of it, and
    a `Revealed` answer's key id and run id, with the link it verifies for, are it. A proxy without a signer has
    none."""
    (tmp_path / "challenger").mkdir()
    certifier = signers.signer(tmp_path)
    challenger = signers.signer(tmp_path / "challenger", b"challenger")
    key, reg = certifier.public_key(), registration(0)
    p = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, clock=lambda: 0.0,
              salt=lambda w, n: bytes([w + 1]) * n, egress_link="out-3", ingress_link="in-7",
              challenger_key=challenger.public_key(), signer=certifier)
    for _ in range(2 * INGRESS.buckets):
        p.bucketer.tick()
    client = RemoteCertifier("127.0.0.1", 9, key, challenger, "in-7", p.run_id)
    assert p.certifier_run == client.certifier_run == (key.key_id, p.run_id, "in-7")
    assert client.key_id == key.key_id
    key_id, run_id, link = client.certifier_run
    answer = p.reveal(0, reg, challenger.sign(active.reveal_request_message(key_id, run_id, link, 0, reg.digest,
                                                                            active.SALT_KIND)))
    assert (answer.key_id, answer.run_id, link) == p.certifier_run
    assert answer.verify(key, run_id, link, reg.digest, active.SALT_KIND)
    assert not answer.verify(key, run_id, "out-3", reg.digest, active.SALT_KIND)
    with pytest.raises(ValueError, match="no key id"):
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="in-7").certifier_run


ENVELOPE = ClockSync(max(SYNC_SET.deltas), min(SYNC_SET.rhos))


def read_prover(name: str, lag: int) -> online.Prover:
    """A prover that has read the honest run's every frame at its ready bucket."""
    prover = online.Prover(SYNC_SET, EGRESS.buckets, online.rule(name), ENVELOPE, lag=lag)
    for k, (f, r) in enumerate(zip(FRAMES, READY, strict=True)):
        prover.arrive(0, k, f, r)
    return prover


def test_a_proxy_back_from_maintenance_publishes_the_full_runs_later_windows():
    """A Program with nothing in window 1: a proxy back from maintenance at window 2, at the position of window 2's
    first frame, publishes exactly the full run's window-2 records, egress and ingress, hands each frame when the full
    run does and writes the full run's window 2 on the wire. It publishes nothing for the windows before, and a proxy
    can't start past its windows or with an online prover."""
    gap = [SESSIONS[k] for k in (0, 1, 4, 5)]
    frames, ready, syncs, rows = program(gap)
    writes = {p: r - 1 for p, r in enumerate(ready)}
    full, full_wire = asyncio.run(replay(writes, sessions=gap))
    back, wire = asyncio.run(replay(writes, sessions=gap, start=2))
    key = lambda r: (r.link, r.window)  # noqa: E731
    assert sorted(back.records, key=key) == sorted((r for r in full.records if r.window == 2), key=key)
    assert all(r.status.complete for r in back.records) and len(back.records) == 2
    split = sum(anchor < 2 * EGRESS.buckets for _, anchor, *_ in rows)
    assert split == 7 and back.shaper.hands == full.shaper.hands[split:]
    window_bytes = EGRESS.buckets * EGRESS.allocation * B
    assert wire == full_wire[2 * window_bytes :] and len(wire) == window_bytes
    with pytest.raises(ValueError, match="starts at one of windows"):
        Proxy(("127.0.0.1", 1), EGRESS, INGRESS, SYNC_SET, syncs, origin=0.0, ingress_link="requests",
              start=WINDOWS + 1)
    with pytest.raises(ValueError, match="online prover"):
        Proxy(("127.0.0.1", 1), EGRESS, INGRESS, SYNC_SET, [], origin=0.0, ingress_link="requests", start=1,
              windows=WINDOWS, prover=online.Prover(SYNC_SET, EGRESS.buckets, online.rule("last"), ClockSync(4, 1 / 2)))


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
        Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, [], origin=0.0, ingress_link="requests", windows=WINDOWS,
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
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                      clock=clock, sleep=clock.sleep)
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


def test_an_egress_tick_slow_to_write_its_bucket_leaves_the_ingress_tick_at_its_boundary_on_time():
    """Both grids tick at every boundary, and the link takes twice the slack to take each egress bucket, as a loaded
    host's loop can between a boundary and a tick's turn. The ingress tick due there runs first (`TICK_ORDER`) and
    reads its link on time, so every window of both links is accepted; run after the egress tick, as it was, it read
    its link twice the slack late and failed every ingress window closed."""

    async def run():
        clock = VirtualClock(0.0)
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                      clock=clock, sleep=clock.sleep)
        running = asyncio.create_task(proxy.run(EmptyReader(), SlowLink(clock, 2 * proxy.slack)))
        for t in range(WINDOWS * EGRESS.buckets):
            clock.advance(t * TAU)
            await proxy.until(lambda p, t=t: len(p.lateness["egress"]) > t and len(p.lateness["ingress"]) > t)
        await running
        return proxy

    proxy = asyncio.run(run())
    assert max(proxy.lateness["ingress"]) == max(proxy.lateness["egress"]) == 0
    assert sorted((r.link, r.window, r.status.reason) for r in proxy.records) == \
        sorted((link, w, None) for link in ("responses", "requests") for w in range(WINDOWS))


class SlackClock(VirtualClock):
    """A `VirtualClock` whose waits end as late as Linux lets an epoll wait of a niced process end: by 0.5 % of its
    timeout, up to 100 ms (`select_estimate_accuracy`)."""

    def __init__(self, now: float = 0.0) -> None:
        super().__init__(now)
        self.waits: list[float] = []

    async def sleep(self, delay: float) -> None:
        if delay > 0:
            self.waits.append(delay)
        await super().sleep(delay + min(delay / 200, 0.1) if delay > 0 else delay)


def test_a_first_tick_10_s_after_the_proxy_is_up_is_on_time_in_a_niced_process():
    """The proxy is up 10 s before its origin, as `certifier_serve` starts it. One wait for the first tick would end
    50 ms late, past the slack, and fault window 0 on both links (`monitor-unavailable`), as it did on the first
    rehearsals against vLLM at nice 19; waits of at most `MAX_WAIT_S` (`precise_sleep`, the proxy's own sleep) keep
    every tick within the slack."""

    async def run():
        clock = SlackClock(0.0)
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=10.0, ingress_link="requests",
                      clock=clock, sleep=lambda delay: active.wait_until(clock() + delay, clock, clock.sleep))
        running = asyncio.create_task(proxy.run(EmptyReader(), NullWriter()))
        while not running.done():
            await clock.sleep(0)
            clock.advance(clock.now + 1e-3)
        await running
        return proxy, clock

    proxy, clock = asyncio.run(run())
    assert max(proxy.lateness["egress"] + proxy.lateness["ingress"]) < proxy.slack
    assert sorted((r.link, r.window, r.status.reason) for r in proxy.records) == \
        sorted((link, w, None) for link in ("responses", "requests") for w in range(WINDOWS))
    assert max(clock.waits) <= active.MAX_WAIT_S
    assert Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=10.0,
                 ingress_link="requests").sleep is active.precise_sleep


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
        proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, [], origin=0.0,
                      ingress_link="requests", clock=clock, sleep=sleep, prover=prover, windows=WINDOWS, executor=active.InlineExecutor())
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
                link.chunks.put_nowait(active.request_frames(request("a", 1), B))
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
        proxy = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests",
                      clock=clock, sleep=clock.sleep)
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
        proxy = Proxy(upstream, EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests", clock=clock,
                      sleep=clock.sleep)
        frame = active.request_frames(request("a", 1), B)
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
    proxy = Proxy(srv.sockets[0].getsockname()[:2], EGRESS, INGRESS, SYNC_SET, syncs, origin=0.0,
                  ingress_link="requests", clock=clock, sleep=clock.sleep)
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
        for name, _, a, ds in sessions:
            if a == t:
                wire = active.request_frames(request(name, len(ds)), B)
                writer.write(wire)
                sent += len(wire) // B
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


class SlowLink(NullWriter):
    """A link that takes `seconds` of the clock to take each write."""

    def __init__(self, clock: VirtualClock, seconds: float) -> None:
        super().__init__()
        self.clock, self.seconds = clock, seconds

    def writelines(self, chunks) -> None:
        super().writelines(chunks)
        self.clock.advance(self.clock.now + self.seconds)
