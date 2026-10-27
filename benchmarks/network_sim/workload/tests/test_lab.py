"""The lab's Program (`lab.Lab`): each block's sessions split its gates and messages, and every frame of a link is the
certifier's canonical framing."""
import pytest

from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    EVENT_BYTES,
    HEADER,
    encode_frame,
    event_body,
    request_frames,
)
from workload import framing as F
from workload import lab as L
from workload.block import Context

B = 240  # a canonical event's frame is at least 235 bytes
CONFIG = {"spec": {"d": 4, "layers": 3, "requests": 2}, "blocks": 3, "devices": 2, "frame_bytes": B, "seed": 1}


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


def honest_run(config: dict):
    """The Program (every value evaluated in one process) and each block's honest frames."""
    ctx = Context(config)
    lab = L.Lab.of(ctx, config["frame_bytes"])
    values = dict(ctx.inputs) | ctx.constants
    ctx.model.evaluate(values)
    payloads = {s: F.encode(m, values) for s, m in lab.messages.items()}
    return ctx, lab, {b: links(lab, payloads, b) for b in lab.blocks}


@pytest.fixture(scope="module")
def run():
    return honest_run(CONFIG)


def test_the_sessions_split_each_blocks_gates_and_messages(run):
    ctx, lab, _ = run
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
    *_, frames = run
    for egress, ingress in frames.values():
        assert {f[:HEADER.size] for f in egress} == {HEADER.pack(DATA, CANONICAL_STREAM, EVENT_BYTES)}
        assert {f[:HEADER.size] for f in ingress} == {HEADER.pack(DATA, CANONICAL_STREAM, B - HEADER.size)}
        assert {len(f) for f in egress + ingress} == {B}


@pytest.mark.slow
def test_a_served_programs_messages_and_requests_span_many_frames():
    """The served Program's messages are cut over many events of `TOKEN_BYTES` and its requests over several frames,
    each frame still the link's one template: the MLP's messages and requests each fit one."""
    pytest.importorskip("torch")
    pytest.importorskip("verity_vllm")
    config = {"workload": "served", "spec": {"layers": 4, "prompt": 2, "steps": 1}, "blocks": 3, "devices": 2,
              "frame_bytes": B, "seed": 1}
    _, lab, frames = honest_run(config)
    assert max(lab.chunks(m) for m in lab.messages.values()) > 4
    assert max(lab.request_frames(s) for b in lab.blocks for s in lab.sessions[b]) > 4
    for b, (egress, ingress) in frames.items():
        assert len(egress) == len(lab.events[b]) and {len(f) for f in egress + ingress} == {B}
