"""The hardware's routes, and φ: positions in messages, the canonical cut, HELLOs, and the occupancy profile."""
import pytest

from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, HEADER, decode_frame
from workload import framing as F
from workload import model as M
from workload import placement as P
from workload.hardware import OUTSIDE, Hardware, mock
from workload.placement import Edge

SPEC = M.Spec(d=2, layers=3, requests=2)


@pytest.fixture(scope="module")
def model():
    return M.Model(SPEC)


def phi(model, blocks=3, devices=1, frame_bytes=16, bucket_ticks=1):
    hw = mock(blocks, devices, bucket_ticks=bucket_ticks)
    p = P.place(model, hw, P.pipeline(model, hw))
    return F.phi(hw, P.crossings(model, p, hw), frame_bytes)


def test_the_mock_blocks_are_covered_and_routes_cross_both_uplinks():
    hw = mock(2, 3)
    assert all(hw.covered(u) for u in hw.units)
    assert [hw.links[e].name for e in hw.boundary("b0")] == ["b0.uplink"]
    src, dst = hw.leaf("b0.gpu2"), hw.leaf("b1.gpu1")
    assert [hw.key(e, to) for e, to in hw.watched(src, dst)] == ["b0.uplink>b0.cert", "b1.uplink>b1.gpu0"]
    assert [hw.key(e, to) for e, to in hw.watched(hw.client, src)] == ["b0.uplink>b0.gpu0"]
    assert Hardware.from_json(hw.to_json()) == hw


def test_a_declaration_reads_as_blocks_of_gpus_behind_their_uplinks(datacenter):
    dc = datacenter()
    hw = Hardware.from_datacenter(dc)
    assert hw.leaves[hw.client] == "ext0" and hw.bucket_ticks == 10
    assert {u: sorted(hw.leaves[x] for x in xs) for u, xs in hw.units.items()} == {
        b.name: sorted(d.name for d in b.devices) for b in dc.blocks}
    assert [hw.leaves[x] for x in hw.devices("b1")] == [f"b1.gpu{k}" for k in range(8)]
    assert {hw.leaves[x] for x in hw.cert} == {"b0.cert", "b1.cert"}
    assert all(hw.covered(u) for u in hw.units)
    assert hw.cross_latency == 2 + 2 + 1  # ext0 to the leaf, to b0's certifier, to its NIC
    uplink = next(k for k in hw.links if k.name == "b0.nic0-b0.cert")
    assert uplink.beta == 400e9 * 1e-6
    src, dst = hw.leaf("b0.gpu0"), hw.leaf("b1.gpu0")
    assert [hw.key(e, to) for e, to in hw.watched(src, dst)] == ["b0.nic0-b0.cert>b0.cert", "b1.nic0-b1.cert>b1.nic0"]
    assert Hardware.from_json(hw.to_json()) == hw


def test_a_declaration_with_no_outside_device_has_no_client(datacenter):
    dc = datacenter()
    dc.fabric = [d for d in dc.fabric if d.kind != "outside"]
    dc.links = [k for k in dc.links if "ext0" not in (k.a, k.b)]
    with pytest.raises(ValueError, match="outside"):
        Hardware.from_datacenter(dc)
    hw = Hardware.from_datacenter(dc, client="leaf0")
    assert hw.leaves[hw.client] == "leaf0"


def test_an_uncertified_boundary_is_not_covered():
    hw = mock(2)
    j = hw.to_json()
    j["cert"] = [hw.leaf("b0.cert")]
    assert not Hardware.from_json(j).covered("b1")


def test_values_round_trip_through_a_message_at_their_positions():
    edges = tuple(Edge(5, "b0", "b1", v, w, 2, 4, 3) for v, w in ((10, 1), (11, 3), (12, 8), (13, 1)))
    m = F.Message(1, "b0", "b1", 5, edges)
    value = {10: 1, 11: 0b101, 12: 0xA7, 13: 1}
    assert m.positions() == {10: 0, 11: 1, 12: 4, 13: 12}
    payload = F.encode(m, value)
    assert payload == (1 | 0b101 << 1 | 0xA7 << 4 | 1 << 12).to_bytes(2, "little")
    assert F.decode(m, payload) == value


@pytest.mark.parametrize("n, count", [(0, 1), (6, 1), (7, 1), (8, 2), (14, 2), (15, 3)])
def test_the_canonical_cut_fills_frames_of_one_header_and_zero_fills_the_last(n, count):
    """Every frame is a `DATA` frame on the canonical stream of B − 9 bytes, so no header says where a message ends."""
    payload = bytes(range(1, n + 1))
    frames = F.cut(payload, 16)
    assert len(frames) == count and all(len(f) == 16 and f[:HEADER.size] == F.head(16) for f in frames)
    assert {decode_frame(f)[:2] for f in frames} == {(DATA, CANONICAL_STREAM)}
    data = b"".join(decode_frame(f)[2] for f in frames)
    assert data[:n] == payload and not any(data[n:])


def test_a_hello_names_its_connection_and_needs_room():
    hw = mock(2)
    f = F.hello(hw, "b1", OUTSIDE, 13)
    assert len(f) == 13 and f[:HEADER.size] == F.head(13) and F.read_hello(hw, f) == ("b1", OUTSIDE)
    assert F.read_hello(hw, F.cut(b"\0\0\0\0", 13)[0]) == ("b0", "b0")
    assert F.read_hello(hw, F.cut(b"\0\x09\0\0", 13)[0]) is None
    wide = F.hello(hw, "b1", OUTSIDE, 14)
    assert F.read_hello(hw, wide) == ("b1", OUTSIDE)
    assert F.read_hello(hw, wide[:-1] + b"\1") is None and F.read_hello(hw, wide[:13]) is None
    with pytest.raises(ValueError):
        F.phi(hw, [], HEADER.size + F.HELLO.size - 1)


def test_messages_group_a_read_tick_and_a_pair_of_units(model):
    ph = phi(model)
    assert [(m.read_tick, m.src, m.dst) for m in ph.messages] == [
        (1, OUTSIDE, "b0"), (2, "b0", "b1"), (2, OUTSIDE, "b0"), (3, "b0", "b1"), (3, "b1", "b2"), (4, "b1", "b2"),
        (4, "b2", OUTSIDE), (5, "b2", OUTSIDE)]
    assert [m.stream for m in ph.messages] == list(range(1, 9))
    assert {m.bits for m in ph.messages} == {SPEC.d * SPEC.act_bits}
    assert {(src, dst): m.stream for (src, dst), m in ph.openers().items()} == {
        (OUTSIDE, "b0"): 1, ("b0", "b1"): 2, ("b1", "b2"): 5, ("b2", OUTSIDE): 7}


def test_the_occupancy_profile_counts_each_frame_at_its_readers_bucket(model):
    assert phi(model).occupancy() == {
        "b0.uplink>b0.gpu0": [0, 2, 1], "b0.uplink>b0.cert": [0, 0, 2, 1],
        "b1.uplink>b1.gpu0": [0, 0, 2, 1], "b1.uplink>b1.cert": [0, 0, 0, 2, 1],
        "b2.uplink>b2.gpu0": [0, 0, 0, 2, 1], "b2.uplink>b2.cert": [0, 0, 0, 0, 2, 1]}
    coarse = phi(model, bucket_ticks=2).occupancy()
    assert coarse["b0.uplink>b0.gpu0"] == [2, 1] and coarse["b2.uplink>b2.cert"] == [0, 0, 3]


def test_a_wide_message_takes_as_many_frames_as_its_bytes_need():
    m = F.Message(1, "b0", "b1", 5, (Edge(5, "b0", "b1", 7, 81, 2, 4, 3),))
    assert [F.frame_count(m, b) for b in (13, 14, 20, 64)] == [3, 3, 1, 1]
    assert len(F.cut(F.encode(m, {7: (1 << 81) - 1}), 13)) == 3
