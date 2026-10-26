"""The lab's Program (`lab.Lab`) and N3 over the certifiers' records (`lab.check`): the records of an honest run hold, on
the certifier's canonical framing, and each way of breaking them is refused by name."""
import pytest

from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    EVENT_BYTES,
    HEADER,
    LENGTH,
    data_frames,
    encode_frame,
    event,
    event_body,
    request_frames,
)
from verity.core.protocols.certifier.grid import Record, Status
from workload import framing as F
from workload import lab as L
from workload import values as V
from workload.block import Context
from workload.run import statement

B = 240  # a canonical event's frame is at least 235 bytes
CONFIG = {"spec": {"d": 4, "layers": 3, "requests": 2}, "blocks": 3, "devices": 2, "frame_bytes": B, "seed": 1}
BLOCK = "b1"


def frames_of(data: bytes) -> list[bytes]:
    return [data[i:i + B] for i in range(0, len(data), B)]


def links(lab: L.Lab, payloads: dict[int, bytes], block: str) -> tuple[list[bytes], list[bytes]]:
    """The frames an honest run's certifier releases on the block's egress (its events in position order, each what
    follows its `id:` line) and ingress (its requests in session order)."""
    egress = [encode_frame(DATA, CANONICAL_STREAM, event_body(lab.event(block, p, lab.payload(block, p, payloads))), lab.frame_bytes)
              for p in range(len(lab.events[block]))]
    ingress = [f for s in lab.sessions[block]
               for f in frames_of(request_frames(lab.request(s, payloads), lab.frame_bytes))]
    return egress, ingress


def records(egress: list[bytes], ingress: list[bytes]) -> list[Record]:
    """The frames one a bucket in windows of two buckets, the windows out of order."""
    out = []
    for link, frames in ((L.EGRESS, egress), (L.INGRESS, ingress)):
        rows = [(f,) for f in frames] + [(None,)] * (len(frames) % 2)
        out += [Record(link, w // 2, tuple(rows[w:w + 2]), Status()) for w in range(0, len(rows), 2)][::-1]
    return out


def honest_run(config: dict):
    """The Program, an honest run's statement and witness (every value evaluated in one process), each block's honest
    frames and each message's payload."""
    ctx = Context(config)
    lab = L.Lab.of(ctx, config["frame_bytes"])
    values = dict(ctx.inputs) | ctx.constants
    ctx.model.evaluate(values)
    results = {}
    for u in ctx.hw.unit_names():
        c = V.commit(u, lab.held[u], values, ctx.model.value_bits)
        results[u] = {"values": {"root": c.root.hex(), "count": len(c.gates),
                                 "rows": [r.hex() for r in c.registration.rows],
                                 "salts": [s.hex() for s in c.registration.salts]}}
    public, witness = statement(ctx, results)
    payloads = {s: F.encode(m, values) for s, m in lab.messages.items()}
    return ctx, lab, public, witness, {b: links(lab, payloads, b) for b in lab.blocks}, payloads


@pytest.fixture(scope="module")
def run():
    return honest_run(CONFIG)


def test_the_sessions_split_each_blocks_gates_and_messages(run):
    ctx, lab, *_ = run
    for b in lab.blocks:
        sessions = lab.sessions[b]
        computed = sorted(g for g in lab.held[b] if ctx.model.kind[g] == "computed")
        assert sorted(g for s in sessions for g in s.gates) == computed
        assert sorted(m.stream for s in sessions for m in s.inbound) == \
            sorted(m.stream for m in lab.messages.values() if m.dst == b)
        assert sorted(m.stream for s in sessions for m in s.outbound) == \
            sorted(m.stream for m in lab.messages.values() if m.src == b)
        assert [s.read_tick for s in sessions] == sorted({s.read_tick for s in sessions})
        for s in sessions:
            assert all(m.read_tick == s.read_tick for m in s.inbound)
            assert all(s.read_tick < m.read_tick for m in s.outbound)
            assert [lab.events[b][p][0] for p in lab.positions(b, s.index)] == [s.index] * len(lab.positions(b, s.index))


def test_every_frame_of_a_link_is_one_template(run):
    """Every egress frame is a canonical event of one length after one header, and every ingress frame `data_frames`'
    header: nothing of a frame but its text depends on the run."""
    *_, frames, _ = run
    for egress, ingress in frames.values():
        assert {f[:HEADER.size] for f in egress} == {HEADER.pack(DATA, CANONICAL_STREAM, EVENT_BYTES)}
        assert {f[:HEADER.size] for f in ingress} == {HEADER.pack(DATA, CANONICAL_STREAM, B - HEADER.size)}
        assert {len(f) for f in egress + ingress} == {B}


def test_an_honest_runs_records_hold(run):
    _, lab, public, witness, frames, _ = run
    report = L.check(public, witness, {b: records(*frames[b]) for b in lab.blocks})
    assert report.ok, report.violations[:3]
    s = report.stats
    assert s["explained"] == s["events"] == sum(len(lab.events[b]) for b in lab.blocks) > 0
    assert s["requests"] == s["sessions"] and s["frames"] == sum(len(e) + len(i) for e, i in frames.values())


@pytest.mark.slow
def test_a_served_programs_long_messages_and_requests_hold():
    """The served Program's messages are cut over many events of `TOKEN_BYTES` and its requests over several frames,
    and an honest run's records of them hold: the MLP's messages and requests each fit one."""
    pytest.importorskip("torch")
    pytest.importorskip("verity_vllm")
    config = {"workload": "served", "spec": {"layers": 4, "prompt": 2, "steps": 1}, "blocks": 3, "devices": 2,
              "frame_bytes": B, "seed": 1}
    _, lab, public, witness, frames, _ = honest_run(config)
    assert max(lab.chunks(m) for m in lab.messages.values()) > 4
    assert max(lab.request_frames(s) for b in lab.blocks for s in lab.sessions[b]) > 4
    report = L.check(public, witness, {b: records(*frames[b]) for b in lab.blocks})
    assert report.ok, report.violations[:3]
    assert report.stats["explained"] == report.stats["events"]


def flip(f: bytes, at: int) -> bytes:
    return f[:at] + bytes([f[at] ^ 1]) + f[at + 1:]


def other_event(run, p: int) -> bytes:
    """The frame of the event at position p of `BLOCK` with its chunk's first byte flipped."""
    _, lab, *_, payloads = run
    _, m, _ = lab.events[BLOCK][p]
    chunk = lab.token(BLOCK, p, payloads[m.stream])
    return encode_frame(DATA, CANONICAL_STREAM, event(bytes([chunk[0] ^ 1]) + chunk[1:]), B)


EGRESS = f"{BLOCK}:{L.EGRESS}"
INGRESS = f"{BLOCK}:{L.INGRESS}"
HEAD = HEADER.size + LENGTH.size + L.REQUEST.size


@pytest.mark.parametrize("edit, refused", [
    (lambda r, e, i: ([other_event(r, 0)] + e[1:], i), {("content", EGRESS)}),
    (lambda r, e, i: (e + e[:1], i), {("unexplained", EGRESS)}),
    (lambda r, e, i: (e[:-1], i), {("missing", EGRESS)}),
    (lambda r, e, i: ([e[0][:-1] + b"\x01"] + e[1:], i), {("fill", EGRESS)}),
    (lambda r, e, i: ([e[0][:-1]] + e[1:], i), {("malformed", EGRESS)}),
    (lambda r, e, i: ([encode_frame(DATA, 1, e[0][HEADER.size:HEADER.size + EVENT_BYTES], B)] + e[1:], i),
     {("malformed", EGRESS)}),
    (lambda r, e, i: ([flip(e[0], HEADER.size)] + e[1:], i), {("malformed", EGRESS)}),
    (lambda r, e, i: (e, [flip(i[0], HEAD)] + i[1:]), {("content", INGRESS)}),
    (lambda r, e, i: (e, i[:-1] + [i[-1][:-1] + b"\x01"]), {("fill", INGRESS)}),
    (lambda r, e, i: (e, i + frames_of(data_frames(LENGTH.pack(1) + b"x", B))), {("unexplained", INGRESS)}),
    (lambda r, e, i: (e, i[:-1]), {("missing", INGRESS)}),
    (lambda r, e, i: (e, [encode_frame(DATA, 1, i[0][HEADER.size:], B)] + i[1:]),
     {("malformed", INGRESS), ("content", INGRESS), ("missing", INGRESS)}),
], ids=["other-event", "past-the-program", "dropped-event", "fill", "short-frame", "other-stream", "off-template",
        "flipped-request", "request-fill", "stray-request", "dropped-request", "request-other-stream"])
def test_a_broken_link_is_refused_by_name(run, edit, refused):
    _, lab, public, witness, frames, _ = run
    broken = dict(frames, **{BLOCK: edit(run, *frames[BLOCK])})
    report = L.check(public, witness, {b: records(*broken[b]) for b in lab.blocks})
    assert {(v.code, v.where) for v in report.violations} == refused
