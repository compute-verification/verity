"""The lab's Program (`lab.Lab`) and N3 over the certifiers' records (`lab.check`): the records of an honest run hold, and
each way of breaking them is refused by name."""
import pytest

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.core.protocols.certifier.grid import Record, Status
from workload import framing as F
from workload import lab as L
from workload import values as V
from workload.block import Context
from workload.run import statement

B = 54  # the least frame an event fits in: a chunk of 3 bytes, so the MLP's messages take several
CONFIG = {"spec": {"d": 4, "layers": 3, "requests": 2}, "blocks": 3, "devices": 2, "frame_bytes": B, "seed": 1}
BLOCK = "b1"


def request_frames(stream: int, request: bytes, room: int = 5) -> list[bytes]:
    """The request in frames of `room` bytes, fewer than a frame holds, so that each takes several."""
    chunks = [request[i:i + room] for i in range(0, len(request), room)]
    return [encode_frame(LAST if j == len(chunks) - 1 else DATA, stream, c, B) for j, c in enumerate(chunks)]


def links(lab: L.Lab, payloads: dict[int, bytes], block: str) -> tuple[list[bytes], list[bytes]]:
    """The frames an honest run's certifier releases on the block's egress (its events in position order) and ingress
    (its requests in session order)."""
    egress = [encode_frame(DATA, j + 1, lab.event(block, p, payloads[m.stream]), B)
              for p, (j, m, _) in enumerate(lab.events[block])]
    ingress = [f for s in lab.sessions[block] for f in request_frames(s.stream, lab.request(s, payloads))]
    return egress, ingress


def records(egress: list[bytes], ingress: list[bytes]) -> list[Record]:
    """The frames one a bucket in windows of two buckets, the windows out of order."""
    out = []
    for link, frames in ((L.EGRESS, egress), (L.INGRESS, ingress)):
        rows = [(f,) for f in frames] + [(None,)] * (len(frames) % 2)
        out += [Record(link, w // 2, tuple(rows[w:w + 2]), Status()) for w in range(0, len(rows), 2)][::-1]
    return out


@pytest.fixture(scope="module")
def run():
    """The Program, an honest run's statement and witness (every value evaluated in one process), and each block's
    honest frames."""
    ctx = Context(CONFIG)
    lab = L.Lab.of(ctx, B)
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
    return ctx, lab, public, witness, {b: links(lab, payloads, b) for b in lab.blocks}


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
    assert max(lab.chunks(m) for m in lab.messages.values()) > 1
    longest = max((lab.event(b, p, bytes(4 * lab.chunk)) for b in lab.blocks for p in range(len(lab.events[b]))), key=len)
    assert HEADER.size + len(longest) <= B


def test_an_honest_runs_records_hold(run):
    _, lab, public, witness, frames = run
    report = L.check(public, witness, {b: records(*frames[b]) for b in lab.blocks})
    assert report.ok, report.violations[:3]
    s = report.stats
    assert s["explained"] == s["events"] == sum(len(lab.events[b]) for b in lab.blocks) > 0
    assert s["requests"] == s["sessions"] and s["frames"] == sum(len(e) + len(i) for e, i in frames.values())


def flip(f: bytes, at: int) -> bytes:
    return f[:at] + bytes([f[at] ^ 1]) + f[at + 1:]


def data_end(f: bytes) -> int:
    """The index in frame f of its event's last data byte, before the closing blank line."""
    return HEADER.size + HEADER.unpack_from(f)[2] - 3


EGRESS = f"{BLOCK}:{L.EGRESS}"
INGRESS = f"{BLOCK}:{L.INGRESS}"


@pytest.mark.parametrize("edit, refused", [
    (lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i), {("content", EGRESS)}),
    (lambda e, i: (e + e[:1], i), {("extra", EGRESS)}),
    (lambda e, i: (e[1:], i), {("missing", EGRESS)}),
    (lambda e, i: ([e[0][:-1] + b"\x01"] + e[1:], i), {("fill", EGRESS), ("missing", EGRESS)}),
    (lambda e, i: ([e[0][:-1]] + e[1:], i), {("malformed", EGRESS), ("missing", EGRESS)}),
    (lambda e, i: ([encode_frame(DATA, 1, b"id: 999\n\n", B)] + e, i), {("unexplained", EGRESS)}),
    (lambda e, i: (e, [flip(i[0], HEADER.size)] + i[1:]), {("content", INGRESS)}),
    (lambda e, i: (e, i + [encode_frame(LAST, 99, b"x", B)]), {("unexplained", INGRESS)}),
    (lambda e, i: (e, i + i[-1:]), {("extra", INGRESS)}),
    (lambda e, i: (e, i[:-1]), {("missing", INGRESS)}),
], ids=["flipped-event", "repeated-event", "dropped-event", "fill", "short-frame", "past-the-program",
        "flipped-request", "stray-stream", "after-last", "dropped-request"])
def test_a_broken_link_is_refused_by_name(run, edit, refused):
    _, lab, public, witness, frames = run
    broken = dict(frames, **{BLOCK: edit(*frames[BLOCK])})
    report = L.check(public, witness, {b: records(*broken[b]) for b in lab.blocks})
    assert {(v.code, v.where) for v in report.violations} == refused
