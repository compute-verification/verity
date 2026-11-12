"""capture-flows, a packet capture summed by flow and direction and held against the network certifier's record. On the
fixture capture (`fixtures/capture_flows.pcap`, made by `fixtures/capture_flows.py` beside it): the sums of each flow
and direction, a retransmitted segment on the wire twice and in the stream once, the server's flows beside the
certifier's listed under names with no address printed or written, and the foreign flow refused, or counted and
skipped when told. On captures built from the fixture's packets: a stream counted across the sequence numbers' wrap
and past 2^32 bytes, a hole and a missing SYN or FIN as bounds and never agreement, the pairing's counts on each side,
every capture format read to the same sums, and each thing the reader doesn't read refused by its name, a packet kept
too short with the least length that reads it and with no length for every capture. Then a run of the network
certifier on a `VirtualClock`, published as `serve` publishes one, with a capture of exactly the bytes its
server's side saw: the two counts agree in each direction, the watermark stream's among them and the server's answer
on it in no sum; a run of two execution phases is counted across both phases' records; a run in the id egress form,
at the id front's cell, agrees as one in the bytes form does, and its capture kept at 54 bytes a packet holds no
content and is read as the one kept at 130 is; a request's place in the release order is its `END` cell's in the
ingress record, where a bucket's are not in their ordinals' order; the answer on a watermark connection in no pair
makes its direction a bound; and a connection that failed is a bound."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_warden_serve import PARAMS, SESSION, _response, _Writer, ws
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import VirtualClock
from verity.protocols.certifier.verifier.io_tie import TRAILER
from verity.protocols.certifier.verifier.release import WATERMARK_END, watermark_head, with_release
from verity.protocols.certifier.verifier.warden import Warden, WardenParams, audit, cells, read_cell

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURE = FIXTURES / "capture_flows.pcap"


def _load():
    sys.path.insert(0, str(HERE))
    mods = []
    for spec in (importlib.util.spec_from_file_location("network_traces_capture_flows", HERE / "capture_flows.py"),
                 importlib.util.spec_from_file_location("network_traces_capture_flows_fixture",
                                                        FIXTURES / "capture_flows.py")):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods


cf, fx = _load()
SERVER, CERTIFIER = fx.SERVER, (fx.CERTIFIER, 40001)
ADDRESSES = (fx.SERVER[0], fx.CERTIFIER, fx.BESIDE, *fx.FOREIGN, fx.SERVER6[0], fx.CERTIFIER6)


def _given(server: str = SERVER[0], certifier: str = CERTIFIER[0]) -> tuple[list, list]:
    return [cf.endpoint(server)], [cf.endpoint(certifier)]


def _tcp(packets: int, wire: int, payload: int, stream: int, holes: int = 0, syn: bool = True, fin: bool = True,
         rst: bool = False) -> dict:
    return {"packets": packets, "wire_bytes": wire, "payload_bytes": payload, "stream_bytes": stream, "holes": holes,
            "syn": syn, "fin": fin, "rst": rst}


def _sums(packets: int, wire: int, payload: int) -> dict:
    return {"packets": packets, "wire_bytes": wire, "payload_bytes": payload}


def _capture(tmp_path: Path, packets: list, name: str = "capture.pcap", frame=None, **how) -> Path:
    """A capture file of the IP packets `packets`, each in an Ethernet frame or `frame`'s."""
    path = tmp_path / name
    path.write_bytes(fx.pcap([(frame or fx.ethernet)(p) for p in packets], **how))
    return path


def _record(*counts: tuple[int, int | None], **more) -> dict:
    """A record's counts as `record_counts` gives them: exact connections of (bytes to the server, bytes from it)."""
    return {"phase_records": {"certified": 3, "of": 3},
            "connections": [{"connection": n + 1, "exact": True, "why": None, "to_server": t, "from_server": f} | more
                            for n, (t, f) in enumerate(counts)]}


#: the fixture's two connections between the certifier and the server, as the reader sums them
FIRST = {"server": "server:8000", "certifier": "certifier:40001",
         "to_server": _tcp(8, 832, 400, 300), "from_server": _tcp(8, 1132, 700, 700)}
SECOND = {"server": "server:8000", "certifier": "certifier:40002",
          "to_server": _tcp(6, 494, 170, 120), "from_server": _tcp(5, 334, 64, 64)}


# --- the fixture capture ----------------------------------------------------------------------------------------------

def test_the_fixture_capture_is_what_its_generator_writes_at_the_readers_snapshot_length():
    assert fx.KEPT == cf.SNAPSHOT
    assert CAPTURE.read_bytes() == fx.capture(), \
        "rewrite it: python benchmarks/network_traces/tests/fixtures/capture_flows.py"


def test_a_capture_is_summed_by_flow_and_direction():
    """Each flow's two directions have their own packets, bytes on the wire (every frame at its length on the wire,
    though the fixture keeps 130 bytes of each) and bytes of payload, whatever order the flows' packets come in."""
    result = cf.read([CAPTURE], *_given(), skip_foreign=True)
    assert result["capture"]["files"] == [{"packets": 45, "link_type": 1, "snapshot": 130}]
    assert result["between"] == [FIRST, SECOND]
    assert result["others"] == [
        {"of": "server:50000", "peer": "other-1:443", "protocol": "tcp",
         "sent": _tcp(5, 360, 90, 90), "received": _tcp(5, 670, 400, 400)},
        {"of": "server:53535", "peer": "other-1:53", "protocol": "udp",
         "sent": _sums(1, 82, 40), "received": _sums(1, 162, 120)},
        {"of": "server", "peer": "other-1", "protocol": "icmp", "sent": _sums(1, 98, 64), "received": _sums(1, 98, 64)}]
    assert "compared" not in result and "record" not in result


def test_a_retransmitted_segment_is_on_the_wire_twice_and_in_the_stream_once(tmp_path):
    """The first connection's request sends its last 100 bytes twice, and the second's sends its first 50 again
    inside a segment of 120: the payload counts each sending, the stream each sequence number once. A response's
    segments out of order are one stream with no hole. Without the second sending the wire and the payload lose
    it and the stream is the same."""
    first, second = cf.read([CAPTURE], *_given(), skip_foreign=True)["between"]
    assert (first["to_server"]["payload_bytes"], first["to_server"]["stream_bytes"]) == (300 + 100, 300)
    assert (second["to_server"]["payload_bytes"], second["to_server"]["stream_bytes"]) == (120 + 50, 120)
    assert (first["from_server"]["payload_bytes"], first["from_server"]["stream_bytes"]) == (700, 700)
    assert first["from_server"]["holes"] == 0
    packets = fx.first()
    again = [p for p in packets if packets.count(p) == 2]
    assert [len(p) for p in again] == [20 + 20 + 100] * 2
    packets.remove(again[0])
    assert cf.read([_capture(tmp_path, packets)], *_given())["between"] == \
        [FIRST | {"to_server": _tcp(7, 832 - 154, 300, 300)}]


def test_a_flow_beside_the_certifiers_is_listed_under_a_name_and_no_address_is_printed_or_written(tmp_path, capsys):
    """The server's flows with a third machine are listed, each with its bytes each way, the machine named `other-1`
    in every one; the table and the result file hold none of the capture's addresses."""
    out = tmp_path / "result.json"
    assert cf.main(["--capture", str(CAPTURE), "--server", SERVER[0], "--certifier", CERTIFIER[0], "--skip-foreign",
                    "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result == cf.read([CAPTURE], *_given(), skip_foreign=True)
    assert [(f["of"], f["peer"], f["protocol"]) for f in result["others"]] == \
        [("server:50000", "other-1:443", "tcp"), ("server:53535", "other-1:53", "udp"), ("server", "other-1", "icmp")]
    printed = capsys.readouterr().out
    assert printed == cf.table(result) + "\n"
    lines = printed.splitlines()
    assert lines[0] == "capture: 45 packets in 1 file(s); skipped 4 packets of 1 foreign flow(s) and 0 frame(s) " \
                       "that aren't IP"
    assert [line.split() for line in lines[3:5]] == \
        [["certifier", "to", "server", "2", "14", "1,326", "570", "420", "0"],
         ["server", "to", "certifier", "2", "13", "1,466", "764", "764", "0"]]
    assert [line.split() for line in lines[7:]] == \
        [["server:50000", "with", "other-1:443,", "tcp", "5", "360", "90", "5", "670", "400"],
         ["server:53535", "with", "other-1:53,", "udp", "1", "82", "40", "1", "162", "120"],
         ["server", "with", "other-1,", "icmp", "1", "98", "64", "1", "98", "64"]]
    for text in (printed, out.read_text()):
        assert not any(a in text for a in ADDRESSES)


def test_a_file_holding_a_foreign_flow_is_refused_unless_told_to_skip_it_and_then_it_is_counted(tmp_path, capsys):
    with pytest.raises(cf.Refused, match=r"holds 1 flow\(s\) \(4 packets\) between addresses that weren't given and 0 "
                                         r"frame\(s\) that aren't IP: .* --skip-foreign"):
        cf.read([CAPTURE], *_given())
    skipped = cf.read([CAPTURE], *_given(), skip_foreign=True)["capture"]["skipped"]
    assert skipped == {"foreign": {"flows": 1, "packets": 4, "wire_bytes": 3 * 54 + 64},
                       "not_ip": {"packets": 0, "wire_bytes": 0}}
    out = tmp_path / "result.json"
    assert cf.main(["--capture", str(CAPTURE), "--server", SERVER[0], "--certifier", CERTIFIER[0],
                    "--out", str(out)]) == 2
    said = capsys.readouterr()
    assert said.err.startswith("capture-flows: refused: the capture holds 1 flow(s)") and said.out == ""
    assert not out.exists() and not any(a in said.err for a in ADDRESSES)


def test_only_flows_with_a_given_address_are_read_and_a_port_tells_two_sides_of_one_address_apart(tmp_path):
    """With the server's address alone the certifier is one more machine the server talks to; with the certifier's
    alone the server's other flows are foreign. Where the server and the certifier share an address, as on one
    machine's loopback, the server's port says which side of a flow is which, and without it the flow is refused."""
    alone = cf.read_flows([CAPTURE], [cf.endpoint(SERVER[0])], skip_foreign=True)
    assert alone["compared"] == [] and alone["skipped"]["foreign"]["flows"] == 1
    assert [(f["of"], f["peer"]) for f in alone["others"]] == \
        [("server:8000", "other-1:40001"), ("server:8000", "other-1:40002"), ("server:50000", "other-2:443"),
         ("server:53535", "other-2:53"), ("server", "other-2")]
    assert (alone["others"][0]["received"], alone["others"][0]["sent"]) == (FIRST["to_server"], FIRST["from_server"])
    certifiers = cf.read_flows([CAPTURE], [], [cf.endpoint(CERTIFIER[0])], skip_foreign=True)
    assert certifiers["skipped"]["foreign"] == {"flows": 4, "packets": 18, "wire_bytes": 226 + 1030 + 244 + 196}
    assert [(f["of"], f["peer"]) for f in certifiers["others"]] == \
        [("certifier:40001", "other-1:8000"), ("certifier:40002", "other-1:8000")]
    one = SERVER[0]
    shared = _capture(tmp_path, fx.exchange((one, 40001), SERVER, fx.data(300), fx.data(700), mss=250))
    got = cf.read_flows([shared], [cf.endpoint(f"{one}:{SERVER[1]}")], [cf.endpoint(one)])
    assert [(f["server"], f["certifier"]) for f in got["compared"]] == [("server:8000", "certifier:40001")]
    assert (got["compared"][0]["to_server"]["stream_bytes"], got["compared"][0]["from_server"]["stream_bytes"]) == \
        (300, 700)
    with pytest.raises(cf.Refused, match="can't be told apart: give the server's port"):
        cf.read_flows([shared], [cf.endpoint(one)], [cf.endpoint(one)])


def test_an_address_is_read_with_or_without_its_port_and_a_bad_one_is_refused_without_repeating_it(capsys):
    assert cf.endpoint("192.0.2.10") == (bytes([192, 0, 2, 10]), None)
    assert cf.endpoint("192.0.2.10:8000") == (bytes([192, 0, 2, 10]), 8000)
    v6 = bytes.fromhex("20010db8000000000000000000000010")
    assert cf.endpoint("2001:db8::10") == (v6, None) and cf.endpoint("[2001:db8::10]:8000") == (v6, 8000)
    for bad in ("192.0.2.300", "192.0.2.10:http", "a-machine-name"):
        with pytest.raises(argparse.ArgumentTypeError, match="ADDR or ADDR:PORT") as e:
            cf.endpoint(bad)
        assert bad not in str(e.value)
    with pytest.raises(SystemExit):
        cf.main(["--capture", str(CAPTURE), "--server", "a-machine-name", "--out", "unwritten.json"])
    assert "a-machine-name" not in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cf.main(["--capture", str(CAPTURE), "--server", SERVER[0], "--run", "a-run", "--out", "unwritten.json"])
    assert "give --certifier" in capsys.readouterr().err


# --- sequence numbers -------------------------------------------------------------------------------------------------

def test_a_stream_is_counted_across_the_sequence_numbers_wrap(tmp_path):
    """Each side's sequence numbers start just under 2^32 and pass it mid-stream: the stream is whole, with no hole,
    also when its first segment, from before the wrap, is sent again after bytes past it, and when the segments on
    each side of the wrap arrive in the other order."""
    packets = fx.exchange(CERTIFIER, SERVER, fx.data(1000), fx.data(900), isn=(2 ** 32 - 151, 2 ** 32 - 1), mss=100)
    plain = cf.read([_capture(tmp_path, packets)], *_given())["between"]
    assert plain == [{"server": "server:8000", "certifier": "certifier:40001",
                      "to_server": _tcp(2 + 10 + 9 + 1, 22 * 54 + 1000, 1000, 1000),
                      "from_server": _tcp(1 + 10 + 9 + 2, 22 * 54 + 900, 900, 900)}]
    c = fx.Connection(CERTIFIER, SERVER, (2 ** 32 - 151, 7)).open()
    first = c.next[0]
    c.send(0, fx.data(100))
    late = c.send(0, fx.data(100), at=first + 200, acked=False)  # past the wrap, before the bytes under it
    early = c.send(0, fx.data(100), at=first + 100)
    c.send(0, fx.data(100), at=first)  # the first segment again
    c.close()
    assert (int.from_bytes(early[24:28], "big"), int.from_bytes(late[24:28], "big")) == (2 ** 32 - 50, 50)
    got = cf.read([_capture(tmp_path, c.packets)], *_given())["between"][0]["to_server"]
    assert (got["payload_bytes"], got["stream_bytes"], got["holes"], got["fin"]) == (400, 300, 0, True)


def test_a_stream_longer_than_the_sequence_numbers_is_counted_whole(tmp_path):
    """66,100 segments of 65,000 bytes, the first 130 bytes of each kept: more than 2^32 bytes on one connection, whose
    sequence numbers so pass every value once. The stream is their sum, with no hole and its FIN where the bytes
    end."""
    n, size = 66_100, 65_000
    assert n * size > 2 ** 32
    (src, sport), (dst, dport) = CERTIFIER, SERVER
    frames = [fx.ethernet(fx.ip(src, dst, fx.TCP, fx.tcp(sport, dport, 9, fx.SYN)))]
    frames += [(fx.ethernet(fx.ip(src, dst, fx.TCP, fx.tcp(sport, dport, 10 + i * size, fx.ACK), above=20 + size)),
                54 + size) for i in range(n)]
    frames.append(fx.ethernet(fx.ip(src, dst, fx.TCP, fx.tcp(sport, dport, 10 + n * size, fx.FIN | fx.ACK))))
    path = tmp_path / "long.pcap"
    path.write_bytes(fx.pcap(frames, snapshot=cf.SNAPSHOT))
    got = cf.read([path], *_given())["between"]
    assert [f["to_server"] for f in got] == [_tcp(n + 2, (n + 2) * 54 + n * size, n * size, n * size)]
    assert got[0]["from_server"] == _tcp(0, 0, 0, 0, syn=False, fin=False)


def test_a_second_connection_on_the_same_ports_is_another_flow(tmp_path):
    again = fx.exchange(CERTIFIER, SERVER, fx.data(300), fx.data(700), mss=250) + \
        fx.exchange(CERTIFIER, SERVER, fx.data(40), fx.data(90), isn=(123456, 654321))
    got = cf.read([_capture(tmp_path, again)], *_given())["between"]
    assert [(f["server"], f["certifier"], f["to_server"]["stream_bytes"], f["from_server"]["stream_bytes"])
            for f in got] == [("server:8000", "certifier:40001", 300, 700), ("server:8000", "certifier:40001", 40, 90)]
    assert all(f[way]["syn"] and f[way]["fin"] and not f[way]["holes"] for f in got
               for way in ("to_server", "from_server"))


def test_a_keep_alives_byte_and_a_resets_sequence_number_are_no_bytes_of_the_stream(tmp_path):
    """A keep-alive carries one byte at the sequence number before the next, a byte the receiver has: it is payload
    on the wire and no new byte of the stream, also before the stream's first byte. A reset is flagged, and its
    sequence number, which needn't be the stream's, makes no hole."""
    c = fx.Connection(CERTIFIER, SERVER).open()
    c.packet(0, c.next[0] - 1, fx.ACK, b"\0")
    c.send(0, fx.data(100))
    c.packet(0, c.next[0] - 1, fx.ACK, b"\0")
    c.packet(1, 0x7FFF0000, fx.RST)
    got = cf.read([_capture(tmp_path, c.packets)], *_given())["between"][0]
    assert got["to_server"] == _tcp(2 + 3, 5 * 54 + 102, 102, 100, fin=False)
    assert got["from_server"] == _tcp(1 + 1 + 1, 3 * 54, 0, 0, fin=False, rst=True)


# --- what the capture missed, and what the two counts say -------------------------------------------------------------

def test_the_fixtures_two_connections_agree_with_a_record_of_their_counts_and_differ_from_any_other():
    flows = cf.read([CAPTURE], *_given(), skip_foreign=True)["between"]
    got = cf.compare(flows, _record((300, 700), (120, 64)))
    assert {way: (got[way]["record_bytes"], got[way]["difference"], got[way]["says"])
            for way in ("to_server", "from_server")} == \
        {"to_server": (420, [0, 0], "agrees"), "from_server": (764, [0, 0], "agrees")}
    assert got["to_server"]["capture"] == {"packets": 14, "wire_bytes": 1326, "payload_bytes": 570,
                                           "stream_bytes": 420, "holes": 0, "flows": 2, "without_syn": 0,
                                           "without_fin": 0, "reset": 0}
    assert got["pairing"] == {"connections": 2, "flows": 2, "pairs": 2, "pairs_inside_bounds": 0,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0, "says": "agrees"}
    assert got["bounded"] == got["unpaired"] == [] and got["watermark_answer_bytes"] is None
    # the record's bytes are compared with the stream's, not the payload's: the retransmissions are in neither count
    more = cf.compare(flows, _record((300, 700), (120, 65)))
    assert (more["to_server"]["says"], more["from_server"]["says"]) == ("agrees", "differs by -1 bytes")
    assert more["from_server"]["difference"] == [-1, -1]
    assert more["pairing"] | {"says": None} == {"connections": 2, "flows": 2, "pairs": 1, "pairs_inside_bounds": 0,
                                                "unpaired_connections": 1, "unpaired_flows": 1, "partial_flows": 0,
                                                "says": None}
    assert more["pairing"]["says"] == "1 connection(s) and 1 flow(s) have no pair of equal counts"
    assert more["unpaired"] == [{"connection": 2, "why": None, "to_server": 120, "from_server": 65}]
    fewer = cf.compare(flows, _record((300, 700)))
    assert (fewer["to_server"]["says"], fewer["from_server"]["says"]) == \
        ("differs by +120 bytes", "differs by +64 bytes")
    assert (fewer["pairing"]["pairs"], fewer["pairing"]["unpaired_connections"],
            fewer["pairing"]["unpaired_flows"]) == (1, 0, 1)


def test_two_connections_of_equal_counts_pair_as_a_multiset_and_the_sums_can_agree_where_the_pairs_dont(tmp_path):
    """Equal connections pair with equal flows, whichever is which. Two flows whose bytes sum to the record's and
    are split otherwise agree in each direction's sum and pair with nothing."""
    twice = fx.interleaved(fx.exchange(CERTIFIER, SERVER, fx.data(80), fx.data(500)),
                           fx.exchange((CERTIFIER[0], 40002), SERVER, fx.data(80), fx.data(500), isn=(9, 99)))
    flows = cf.read([_capture(tmp_path, twice)], *_given())["between"]
    got = cf.compare(flows, _record((80, 500), (80, 500)))
    assert (got["to_server"]["says"], got["from_server"]["says"], got["pairing"]["says"]) == ("agrees",) * 3
    assert got["pairing"]["pairs"] == 2
    split = cf.compare(flows, _record((60, 600), (100, 400)))
    assert (split["to_server"]["says"], split["from_server"]["says"]) == ("agrees", "agrees")
    assert split["pairing"] == {"connections": 2, "flows": 2, "pairs": 0, "pairs_inside_bounds": 0,
                                "unpaired_connections": 2, "unpaired_flows": 2, "partial_flows": 0,
                                "says": "2 connection(s) and 2 flow(s) have no pair of equal counts"}
    result = cf.read([tmp_path / "capture.pcap"], *_given()) | {"record": _record((60, 600), (100, 400)),
                                                                "compared": split}
    assert "pairing, a connection's two counts with a flow's: 0 pair(s) of equal counts, 0 inside a connection's " \
           "bounds; unpaired, 2 of 2 connection(s) and 2 of 2 flow(s), 0 of them held in part: the multisets differ" \
        in cf.table(result).splitlines()


def test_a_hole_is_counted_in_bytes_and_its_direction_is_a_bound_never_agreement(tmp_path):
    """The capture misses the middle 300 bytes of the first connection's response, a segment sent once. The direction
    has a hole of 300 bytes, and with the record's 764 it says the range the difference lies in, which holds 0;
    with a record one byte more than the capture can hold, or one less than it holds, it differs by at least that."""
    packets = fx.first()
    lost = [p for p in packets if len(p) == 340 and int.from_bytes(p[24:28], "big") == 5001 + 300]
    assert len(lost) == 1
    packets.remove(lost[0])
    result = cf.read([_capture(tmp_path, fx.interleaved(packets, fx.second()))], *_given())
    flows = result["between"]
    assert flows[0]["from_server"] == _tcp(7, 1132 - 354, 400, 400, holes=300) and flows[1] == SECOND
    got = cf.compare(flows, _record((300, 700), (120, 64)))
    assert got["to_server"]["says"] == "agrees"
    assert (got["from_server"]["capture"]["stream_bytes"], got["from_server"]["capture"]["holes"]) == (464, 300)
    assert (got["from_server"]["difference"], got["from_server"]["says"]) == \
        ([-300, 0], "bound: difference -300 to 0 bytes")
    assert got["pairing"] | {"says": None} == {"connections": 2, "flows": 2, "pairs": 1, "pairs_inside_bounds": 0,
                                               "unpaired_connections": 1, "unpaired_flows": 1, "partial_flows": 1,
                                               "says": None}
    table = cf.table(result | {"record": _record((300, 700), (120, 64)), "compared": got}).splitlines()
    assert table[3].split()[:9] == ["flows", "packets", "wire", "bytes", "payload", "bytes", "stream", "bytes", "holes"]
    assert table[5].split() == ["server", "to", "certifier", "2", "12", "1,112", "464", "464", "300", "764", "bound:",
                                "difference", "-300", "to", "0", "bytes"]
    assert cf.compare(flows, _record((300, 701), (120, 64)))["from_server"]["says"] == \
        "differs: the capture has at least 1 bytes less"
    assert cf.compare(flows, _record((300, 399), (120, 64)))["from_server"]["says"] == \
        "differs: the capture has at least 1 bytes more"


def test_a_flow_without_its_syn_or_its_fin_is_a_floor(tmp_path):
    """A capture that starts after a connection opened, or stops before it closed, holds some of its bytes and
    doesn't say how many it missed: the direction's count is a floor, so the difference has no upper bound."""
    packets = fx.second()
    fin = [p for p in packets if p[33] & fx.FIN and p[12:16] == bytes([192, 0, 2, 10])]
    packets.remove(fin[0])
    result = cf.read([_capture(tmp_path, fx.interleaved(fx.first(), packets))], *_given())
    assert result["between"][1]["from_server"] == _tcp(4, 334 - 54, 64, 64, fin=False)
    got = cf.compare(result["between"], _record((300, 700), (120, 64)))
    assert (got["from_server"]["difference"], got["from_server"]["says"]) == \
        ([0, None], "bound: difference at least 0 bytes")
    assert got["from_server"]["capture"]["without_fin"] == 1 and got["pairing"]["partial_flows"] == 1
    assert "    the capture holds 1 flow(s) without their FIN" in \
        cf.table(result | {"record": _record((300, 700), (120, 64)), "compared": got}).splitlines()
    assert cf.compare(result["between"], _record((300, 700), (120, 60)))["from_server"]["says"] == \
        "differs: the capture has at least 4 bytes more"
    late = fx.first()[3:]  # from the request's first segment on: no SYN either way
    flow = cf.read([_capture(tmp_path, late)], *_given())["between"][0]
    assert flow["to_server"] == _tcp(6, 832 - 2 * 54, 400, 300, syn=False)
    assert flow["from_server"] == _tcp(7, 1132 - 54, 700, 700, syn=False)
    assert cf.compare([flow], _record((300, 700)))["to_server"]["says"] == "bound: difference at least 0 bytes"


def test_a_difference_is_worded_as_exact_only_where_both_counts_are():
    assert cf.says(0, 0) == "agrees"
    assert (cf.says(5, 5), cf.says(-1234, -1234)) == ("differs by +5 bytes", "differs by -1,234 bytes")
    assert cf.says(-300, 0) == "bound: difference -300 to 0 bytes"
    assert cf.says(-1, 1000) == "bound: difference -1 to +1,000 bytes"
    assert (cf.says(None, 12), cf.says(0, None)) == \
        ("bound: difference at most +12 bytes", "bound: difference at least 0 bytes")
    assert cf.says(None, None) == "bound: difference not bounded"
    assert (cf.says(3, None), cf.says(3, 9)) == ("differs: the capture has at least 3 bytes more",) * 2
    assert (cf.says(None, -4), cf.says(-9, -4)) == ("differs: the capture has at least 4 bytes less",) * 2


# --- the capture's formats, and what is refused -----------------------------------------------------------------------

def _between(tmp_path: Path, packets: list, frame=None, given=None, **how) -> list:
    return cf.read([_capture(tmp_path, packets, frame=frame, **how)], *(given or _given()))["between"]


def _wider(flows: list, more: int) -> list:
    """The flows with `more` bytes more on the wire in every packet."""
    return [f | {way: f[way] | {"wire_bytes": f[way]["wire_bytes"] + more * f[way]["packets"]}
                 for way in ("to_server", "from_server")} for f in flows]


def test_every_capture_format_is_read_to_the_same_sums(tmp_path):
    """Either byte order, microseconds or nanoseconds, cut to each packet's first 130 bytes or whole, under VLAN tags,
    a Linux cooked capture and raw IP: the same flows, each packet's bytes on the wire those of its own link header."""
    packets = fx.interleaved(fx.first(), fx.second())
    assert _between(tmp_path, packets) == [FIRST, SECOND]
    for how in ({"order": ">"}, {"nanos": True}, {"order": ">", "nanos": True}, {"snapshot": cf.SNAPSHOT}):
        assert _between(tmp_path, packets, **how) == [FIRST, SECOND], how
    assert _between(tmp_path, packets, frame=lambda p: fx.ethernet(p, (7,))) == _wider([FIRST, SECOND], 4)
    assert _between(tmp_path, packets, frame=lambda p: fx.ethernet(p, (7, 9)), snapshot=cf.SNAPSHOT) == \
        _wider([FIRST, SECOND], 8)
    assert _between(tmp_path, packets, frame=fx.cooked, link=fx.LINUX_SLL) == _wider([FIRST, SECOND], 2)
    assert _between(tmp_path, packets, frame=lambda p: fx.cooked(p, 2), link=fx.LINUX_SLL2) == \
        _wider([FIRST, SECOND], 6)
    assert _between(tmp_path, packets, frame=bytes, link=fx.RAW) == _wider([FIRST, SECOND], -14)


def test_ipv6_is_read_as_ipv4_is_through_its_extension_headers(tmp_path):
    packets = fx.exchange((fx.CERTIFIER6, 40001), fx.SERVER6, fx.data(300), fx.data(700), mss=250)
    given = _given(fx.SERVER6[0], fx.CERTIFIER6)
    want = {"server": "server:8000", "certifier": "certifier:40001",
            "to_server": _tcp(2 + 2 + 3 + 1, 8 * 74 + 300, 300, 300),
            "from_server": _tcp(1 + 2 + 3 + 2, 8 * 74 + 700, 700, 700)}
    assert _between(tmp_path, packets, given=given) == [want]
    options = bytes([fx.TCP, 0, 1, 4, 0, 0, 0, 0])  # destination options: the next header, 8 bytes, a padding option
    under = [p[:4] + (int.from_bytes(p[4:6], "big") + 8).to_bytes(2, "big") + bytes([60]) + p[7:40] + options + p[40:]
             for p in packets]
    assert _between(tmp_path, under, given=given) == _wider([want], 8)
    with pytest.raises(cf.Refused, match=r"holds 1 flow\(s\) \(16 packets\) between addresses that weren't given"):
        _between(tmp_path, packets)


def test_several_files_are_one_capture(tmp_path):
    """A capture point that writes each direction of the fibre to its own file: the two files read together are the
    one capture, and either alone holds flows with one direction."""
    packets = fx.interleaved(fx.first(), fx.second())
    server = bytes([192, 0, 2, 10])
    ways = [_capture(tmp_path, [p for p in packets if (p[12:16] == server) == from_server], name=f"{from_server}.pcap")
            for from_server in (False, True)]
    result = cf.read(ways, *_given())
    assert result["between"] == [FIRST, SECOND]
    assert [f["packets"] for f in result["capture"]["files"]] == [14, 13]
    alone = cf.read(ways[:1], *_given())["between"]
    assert [f["to_server"] for f in alone] == [FIRST["to_server"], SECOND["to_server"]]
    assert [f["from_server"] for f in alone] == [_tcp(0, 0, 0, 0, syn=False, fin=False)] * 2


def _packet(protocol: int, body: bytes = b"", **how) -> bytes:
    return fx.ip(CERTIFIER[0], SERVER[0], protocol, body, **how)


def _with_ip_options(packet: bytes, n: int = 8) -> bytes:
    """One of the fixture's IPv4 packets with `n` bytes of IP options, each the end of the list."""
    return (bytes([0x40 | (5 + n // 4)]) + packet[1:2] + (int.from_bytes(packet[2:4], "big") + n).to_bytes(2, "big")
            + packet[4:20] + bytes(n) + packet[20:])


@pytest.mark.parametrize("name, frames, how, said", [
    ("an IP fragment", [fx.ethernet(_packet(fx.TCP, fx.tcp(40001, 8000, 1, fx.ACK, fx.data(64)), fragment=0x2000))],
     {}, r"capture\.pcap's packet 0 is an IP fragment of a flow with a given address: .* reassembles nothing"),
    ("a later fragment", [fx.ethernet(_packet(fx.TCP, fx.data(64), fragment=8))], {}, "is an IP fragment"),
    ("GRE", [fx.ethernet(_packet(47, fx.data(40)))], {}, "packet 0 is GRE, an encapsulation this doesn't read"),
    ("IP in IP", [fx.ethernet(_packet(4, fx.first()[0]))], {}, "is IP in IP, an encapsulation"),
    ("ESP", [fx.ethernet(_packet(50, fx.data(40)))], {}, "is ESP, an encapsulation"),
    ("VXLAN", [fx.ethernet(_packet(fx.UDP, fx.udp(50000, 4789, bytes(8) + fx.ethernet(fx.first()[0]))))], {},
     "is VXLAN by its UDP port, an encapsulation"),
    ("MPLS", [fx.ethernet(bytes(4) + fx.first()[0], kind=0x8847)], {}, "is MPLS, an encapsulation"),
    ("cut inside its headers", [fx.ethernet(p) for p in fx.first()], {"snapshot": 40},
     r"capture\.pcap's packet 0 is kept too short for its headers: this reads its first 48 bytes \(14 of its frame "
     r"header, 20 of its IP header and TCP's first 14\) and 40 are kept\. Capture again keeping at least that many of "
     r"each packet; a packet with longer headers needs more\. No one length suits every capture, since a length past "
     r"a packet's headers keeps the first bytes of its content: see \"Taking the capture\" \(a link\) and \"On the "
     r"server's loopback\" in benchmarks/network_traces/README\.md$"),
    ("IPv6 kept at 54 bytes a packet",
     [fx.ethernet(fx.ip(fx.CERTIFIER6, fx.SERVER6[0], fx.TCP, fx.tcp(40001, 8000, 1, fx.SYN)))], {"snapshot": 54},
     r"packet 0 is kept too short for its headers: this reads its first 68 bytes \(14 of its frame header, 40 of its "
     r"IP header and TCP's first 14\) and 54 are kept\. Capture again keeping at least that many of each packet"),
    ("eight bytes of IP options kept at 54 bytes a packet", [fx.ethernet(_with_ip_options(fx.first()[0]))],
     {"snapshot": 54},
     r"packet 0 is kept too short for its headers: this reads its first 56 bytes \(14 of its frame header, 28 of its "
     r"IP header and TCP's first 14\) and 54 are kept\. Capture again keeping at least that many of each packet"),
    ("cut inside its IP header", [fx.ethernet(p) for p in fx.first()], {"snapshot": 30},
     r"capture\.pcap's packet 0 is kept too short for its headers: the 30 bytes kept of it end before its IPv4 header "
     r"is whole, so this can't tell how many it reads\. Capture again keeping more of each packet\. No one length "
     r"suits every capture, since a length past a packet's headers keeps the first bytes of its content: see "
     r"\"Taking the capture\" \(a link\) and \"On the server's loopback\" in benchmarks/network_traces/README\.md$"),
    ("another link type", [fx.ethernet(p) for p in fx.first()], {"link": 105},
     r"capture\.pcap has link type 105: this reads Ethernet \(1\), raw IP \(101\) and Linux cooked \(113, 276\)"),
])
def test_what_the_reader_doesnt_read_is_refused_by_its_name(tmp_path, name, frames, how, said):
    path = tmp_path / "capture.pcap"
    path.write_bytes(fx.pcap(frames, **how))
    for skip in (False, True):
        with pytest.raises(cf.Refused, match=said):
            cf.read([path], *_given(), skip_foreign=skip)


_SYN = fx.tcp(40001, 8000, 1, fx.SYN)
_SYN6 = fx.ip(fx.CERTIFIER6, fx.SERVER6[0], fx.TCP, _SYN)
_UNDER = fx.ip(fx.CERTIFIER6, fx.SERVER6[0], fx.TCP, _SYN, extensions=bytes([fx.TCP, 0, 1, 4, 0, 0, 0, 0]))
#: one packet of each kind of header the reader walks: what it is, its link type, its frame, each header the reader
#: holds whole before it can tell a length (the length that holds it, its name), the length the reader reads of the
#: packet, and the headers that length is the sum of
_WALKED = [
    ("IPv4 in an Ethernet frame", fx.ETHERNET, fx.ethernet(_packet(fx.TCP, _SYN)), ((14, "Ethernet"), (34, "IPv4")),
     48, "14 of its frame header, 20 of its IP header and TCP's first 14"),
    ("under two VLAN tags", fx.ETHERNET, fx.ethernet(_packet(fx.TCP, _SYN), (7, 9)), ((22, "Ethernet"), (42, "IPv4")),
     56, "22 of its frame header, 20 of its IP header and TCP's first 14"),
    ("with 40 bytes of IP options", fx.ETHERNET, fx.ethernet(_with_ip_options(_packet(fx.TCP, _SYN), 40)),
     ((14, "Ethernet"), (34, "IPv4")), 88, "14 of its frame header, 60 of its IP header and TCP's first 14"),
    ("in a Linux cooked frame", fx.LINUX_SLL, fx.cooked(_packet(fx.TCP, _SYN)), ((16, "Linux cooked"), (36, "IPv4")),
     50, "16 of its frame header, 20 of its IP header and TCP's first 14"),
    ("in a Linux cooked v2 frame", fx.LINUX_SLL2, fx.cooked(_packet(fx.TCP, _SYN), 2),
     ((20, "Linux cooked"), (40, "IPv4")), 54, "20 of its frame header, 20 of its IP header and TCP's first 14"),
    ("raw IP", fx.RAW, _packet(fx.TCP, _SYN), ((20, "IPv4"),), 34, "20 of its IP header and TCP's first 14"),
    ("IPv6", fx.ETHERNET, fx.ethernet(_SYN6), ((14, "Ethernet"), (54, "IPv6")), 68,
     "14 of its frame header, 40 of its IP header and TCP's first 14"),
    ("IPv6 under an extension header", fx.ETHERNET, fx.ethernet(_UNDER),
     ((14, "Ethernet"), (54, "IPv6"), (62, "IPv6 extension")), 76,
     "14 of its frame header, 48 of its IP header and TCP's first 14"),
    ("UDP", fx.ETHERNET, fx.ethernet(fx.ip(SERVER[0], fx.BESIDE, fx.UDP, fx.udp(53535, 53, fx.data(40)))),
     ((14, "Ethernet"), (34, "IPv4")), 38, "14 of its frame header, 20 of its IP header and UDP's first 4"),
]


@pytest.mark.parametrize("link, frame, cuts, need, headers", [case[1:] for case in _WALKED],
                         ids=[case[0] for case in _WALKED])
def test_a_packet_kept_too_short_is_refused_with_the_least_length_that_reads_it(tmp_path, link, frame, cuts, need,
                                                                                headers):
    """Kept at each length under the one the reader reads of it, a packet is refused: with that length and the headers
    it is the sum of, once the bytes kept hold every header that says it; before that, with the first header they
    don't hold whole. Kept at that length it is read. Neither names a length for every capture. A raw IP packet with
    no byte kept has no version to be read by, and is a frame that isn't IP."""
    path = tmp_path / "capture.pcap"
    servers, certifiers = (SERVER[0], fx.SERVER6[0]), (CERTIFIER[0], fx.CERTIFIER6)
    given = [cf.endpoint(a) for a in servers], [cf.endpoint(a) for a in certifiers]
    for kept in range(1 if link == fx.RAW else 0, need):
        path.write_bytes(fx.pcap([frame], link=link, snapshot=kept))
        header = next((header for under, header in cuts if kept < under), None)
        met = (f"the {kept} bytes kept of it end before its {header} header is whole, so this can't tell how many it "
               f"reads. Capture again keeping more of each packet" if header else
               f"this reads its first {need} bytes ({headers}) and {kept} are kept. Capture again keeping at least "
               f"that many of each packet; a packet with longer headers needs more")
        with pytest.raises(cf.Refused) as refused:
            cf.read([path], *given)
        assert str(refused.value) == f"capture.pcap's packet 0 is kept too short for its headers: {met}. {cf.LENGTHS}"
    path.write_bytes(fx.pcap([frame], link=link, snapshot=need))
    assert cf.read([path], *given)["capture"]["files"] == [{"packets": 1, "link_type": link, "snapshot": need}]


def test_the_refusal_of_a_packet_kept_too_short_names_headings_the_readme_has():
    """The refusal names no length for every capture: it names the README's two commands by their headings. Each
    heading is there with its command under it, and the command states the length the heading's paragraph is about:
    the link's the reader's `SNAPSHOT`, the loopback's 54."""
    where, headings = cf.LENGTHS.rsplit(" ", 1)[1], cf.LENGTHS.split('"')[1::2]
    assert HERE.parents[1] / where == HERE / "README.md"
    assert headings == ["Taking the capture", "On the server's loopback"]
    readme, lengths = (HERE / "README.md").read_text(), []
    for heading in headings:
        assert f"\n**{heading}**" in readme, \
            f"the README has no heading {heading!r}: name the one it has in capture_flows.py's LENGTHS"
        command = readme.split(f"\n**{heading}**", 1)[1].split("\n    tcpdump ", 1)[1].splitlines()[0]
        lengths.append(int(command.split(" -s ")[1].split()[0]))
    assert lengths == [cf.SNAPSHOT, 54]


def test_a_file_that_isnt_a_whole_pcap_capture_is_refused_with_what_to_do(tmp_path):
    whole = fx.pcap([fx.ethernet(p) for p in fx.first()])
    for name, content, said in (
            ("capture.pcapng", b"\x0a\x0d\x0d\x0a" + bytes(28),
             r"capture\.pcapng is pcapng, which this doesn't read: `tcpdump -r capture\.pcapng -w OUT\.pcap` rewrites "
             r"it as pcap"),
            ("notes.txt", b"not a capture, though long enough for a head", "notes.txt isn't a capture file as tcpdump"),
            ("short.pcap", whole[:10], "short.pcap isn't a capture file"),
            ("stopped.pcap", whole[:-5], r"stopped\.pcap ends inside its packet 15: a capture stopped while writing"),
            ("headless.pcap", whole[:24 + 16 + 54 + 7], "headless.pcap ends inside its packet 1")):
        (tmp_path / name).write_bytes(content)
        with pytest.raises(cf.Refused, match=said):
            cf.read([tmp_path / name], *_given())


def test_a_foreign_fragment_and_a_frame_that_isnt_ip_are_skipped_only_when_told_and_counted(tmp_path):
    """A fragment between two addresses that weren't given is a foreign packet like any other. A frame that isn't IP
    has no address to be given: it is refused with the foreign flows and counted on its own."""
    fragment = fx.ip(*fx.FOREIGN, fx.UDP, fx.data(64), fragment=0x2000)
    arp = fx.ethernet(bytes(28), kind=0x0806)
    path = tmp_path / "capture.pcap"
    path.write_bytes(fx.pcap([fx.ethernet(p) for p in fx.first()] + [fx.ethernet(fragment), arp, arp]))
    with pytest.raises(cf.Refused, match=r"holds 1 flow\(s\) \(1 packets\) between addresses that weren't given and "
                                         r"2 frame\(s\) that aren't IP"):
        cf.read([path], *_given())
    result = cf.read([path], *_given(), skip_foreign=True)
    assert result["capture"]["skipped"] == {"foreign": {"flows": 1, "packets": 1, "wire_bytes": 98},
                                            "not_ip": {"packets": 2, "wire_bytes": 84}}
    assert result["between"] == [FIRST]
    assert cf.table(result).splitlines()[0] == \
        "capture: 19 packets in 1 file(s); skipped 1 packets of 1 foreign flow(s) and 2 frame(s) that aren't IP"


# --- against a run of the network certifier ---------------------------------------------------------------------------

class _Uncommitted(ws.Publisher):
    """`Publisher` with no commitment to a record's rows (`attest`): each record kept and certified, as `keep` does."""

    def attest(self, record, link, direction, withheld=False) -> None:
        pass


def _served(out: Path, key: Path, requests: list[bytes], responses: list[bytes], *, held: tuple[int, ...] = (),
            watermark: bool = True, params: WardenParams = PARAMS, order=None, phases: int = 1,
            arrive: tuple[int, ...] = ()) -> tuple[Warden, list[_Writer], _Writer]:
    """`phases` execution phases of the network certifier at `params` on a `VirtualClock`, one unless given, published
    as `serve` publishes a run: its records and certificates through `Publisher`, `deployment.json`, and
    `warden.json`'s connections and, with `watermark`, its watermark. Every request arrives before the first tick,
    or with `arrive` request k before tick `arrive[k]`, none earlier than the one before it. The server answers
    connection k (in connect order, from 0) `responses[k]` and ends, or for k in `held` sends it and never ends. With
    `order`, the network certifier's connections are sorted by it once they are admitted, so it cuts each bucket's
    cells, and releases its `END`s, in that order and not in their ordinals'; its records are then kept and certified
    with no commitment to their rows, which refuses cells that aren't where the schedule lays them. Returns the
    network certifier, the server's side of each connection in connect order, and its side of the watermark stream."""
    tau = params.bucket_ms / 1000
    starts = arrive or (0,) * len(requests)
    assert len(starts) == len(requests) and list(starts) == sorted(starts)
    signer = ws.cs.signer(key, 4)
    publisher = (ws.Publisher if order is None else _Uncommitted)(out, ws.Issuer(signer, SESSION), params, WR.Cells.of)
    servers: list[_Writer] = []
    mark = _Writer()

    async def connect():
        reader = asyncio.StreamReader()
        reader.feed_data(responses[len(servers)])
        if len(servers) not in held:
            reader.feed_eof()
        servers.append(_Writer())
        return reader, servers[-1]

    async def open_watermark():
        return asyncio.StreamReader(), mark

    def settled(w: Warden) -> bool:
        return all(c.server is not None and c.ingress_ended
                   and (c.egress_ended or (c.ordinal - 1 in held and c.bytes_out)) for c in w.active)

    async def main() -> Warden:
        clock = VirtualClock(-tau / 2)
        relay = Warden(params, connect, origin=0.0, windows=phases, clock=clock, sleep=clock.sleep, publish=publisher,
                       withhold=publisher.withhold, watermark=open_watermark if watermark else None)
        running = asyncio.create_task(relay.run())
        tasks = []
        for t in range(phases * params.window):
            while len(tasks) < len(requests) and starts[len(tasks)] <= t:
                reader = asyncio.StreamReader()
                reader.feed_data(cells(requests[len(tasks)], params.cell_bytes))
                reader.feed_eof()
                tasks.append(asyncio.create_task(relay.handle(reader, _Writer(relay))))
            await relay.until(lambda w: w.ordinals + w.refused == len(tasks))
            await relay.until(settled)
            clock.advance(t * tau)
            await relay.until(lambda w, t=t: w.ticks > t)
            if t == 0 and order is not None:
                assert len(relay.active) == len(requests)
                relay.active.sort(key=order)
        await running
        await asyncio.gather(*tasks)
        return relay

    relay = asyncio.run(main())
    publisher.close()
    (out / "deployment.json").write_text(json.dumps(ws.deployment_record(
        params, signer.public_key(), SESSION, origin_unix=0.0, windows=phases, listen=(CERTIFIER[0], 1),
        upstream=SERVER, slack=relay.slack, run_id=relay.run_id)))
    log: dict = {"connections": relay.log}
    if watermark:
        log["watermark"] = {"last": relay.watermarked and list(relay.watermarked), "lost": relay.watermark_lost,
                            "error": relay.watermark_error}
    (out / "warden.json").write_text(json.dumps(log))
    return relay, servers, mark


def _sent(writer: _Writer) -> bytes:
    return b"".join(data for _, data in writer.writes)


def _seen(tmp_path: Path, streams: list[tuple[bytes, bytes]], name: str = "seen.pcap", **how) -> Path:
    """A capture of one connection from the certifier to the server for each of `streams`, (the bytes the certifier
    sent, the bytes the server sent), their packets interleaved, some of their sequence numbers passing the wrap."""
    return _capture(tmp_path, fx.interleaved(*(
        fx.exchange((CERTIFIER[0], 40001 + k), SERVER, sent, answered, isn=(2 ** 32 - 97 * k, 1000 * k), mss=400)
        for k, (sent, answered) in enumerate(streams))), name=name, **how)


ANSWER = b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n"


def _three(tmp_path: Path):
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    relay, servers, mark = _served(out, key, requests, responses)
    return out, relay, requests, responses, servers, mark


def test_a_capture_of_the_servers_side_agrees_with_the_certifiers_record_in_each_direction(tmp_path, capsys):
    """Three requests through the network certifier, and the watermark stream. The record's count of each connection
    is what the server's side saw: the request with its release headers one way, the response's bytes the other, and
    the watermark stream's head, lines and end. A capture of those bytes agrees in each direction and pairs every
    connection, the watermark's by its bytes to the server; the server's answer on it is given and is in no sum."""
    out, relay, requests, responses, servers, mark = _three(tmp_path)
    release = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in relay.log}
    assert [_sent(s) for s in servers] == [with_release(r, release[k + 1][0], relay.run_id, release[k + 1][1])
                                           for k, r in enumerate(requests)]
    assert _sent(mark).startswith(watermark_head(relay.run_id)) and _sent(mark).endswith(WATERMARK_END)
    record = cf.record_counts(out)
    assert record["phase_records"] == {"certified": 3, "of": 3}
    assert record["connections"] == [
        *({"connection": k + 1, "release": release[k + 1][0], "exact": True, "why": None, "to_server": len(_sent(s)),
           "from_server": len(r)} for k, (s, r) in enumerate(zip(servers, responses))),
        {"connection": "watermark", "release": None, "exact": True, "why": None, "to_server": len(_sent(mark)),
         "from_server": None}]
    seen = _seen(tmp_path, [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), ANSWER)])
    result = cf.read([seen], *_given(), run=out)
    got = result["compared"]
    assert result["record"] == record and len(result["between"]) == 4
    to_server, from_server = sum(len(_sent(s)) for s in servers) + len(_sent(mark)), sum(map(len, responses))
    assert (got["to_server"]["capture"]["stream_bytes"], got["to_server"]["record_bytes"],
            got["to_server"]["difference"], got["to_server"]["says"]) == (to_server, to_server, [0, 0], "agrees")
    assert (got["from_server"]["capture"]["stream_bytes"], got["from_server"]["record_bytes"],
            got["from_server"]["difference"], got["from_server"]["says"]) == \
        (from_server + len(ANSWER), from_server, [0, 0], "agrees")
    assert got["pairing"] == {"connections": 4, "flows": 4, "pairs": 4, "pairs_inside_bounds": 0,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0, "says": "agrees"}
    assert got["watermark_answer_bytes"] == len(ANSWER) and got["bounded"] == got["unpaired"] == []
    # the command: the same result in its file, and the table, with no address in either
    path = tmp_path / "result.json"
    assert cf.main(["--capture", str(seen), "--server", SERVER[0], "--certifier", CERTIFIER[0], "--run", str(out),
                    "--out", str(path)]) == 0
    assert json.loads(path.read_text()) == result
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "record: 4 connection(s) to the server; 3 of its execution phases' 3 records certified"
    assert [line.split()[-2:] for line in lines[4:6]] == [[f"{to_server:,}", "agrees"], [f"{from_server:,}", "agrees"]]
    assert lines[6] == "pairing, a connection's two counts with a flow's: 4 pair(s) of equal counts, 0 inside a " \
                       "connection's bounds; unpaired, 0 of 4 connection(s) and 0 of 4 flow(s), 0 of them held in " \
                       "part: the multisets agree"
    assert lines[7] == f"the server's answer on the watermark's connection: {len(ANSWER)} bytes, in no record and " \
                       f"no sum above"
    assert lines[9] == "no other flow has a given address on one side"
    assert not any(a in text for a in ADDRESSES for text in ("\n".join(lines), path.read_text()))


def test_a_run_of_two_execution_phases_is_counted_across_both_phases_records(tmp_path):
    """Two execution phases of the network certifier, the watermark stream through both, and four requests: one
    served whole in the first phase, one whose response starts in the first and ends in the second, one whose request
    does, and one served whole in the second. Each phase's ingress record and egress record hold some of the bytes,
    and the record's count in each direction is their sum over both: each connection's is what its server's side
    saw, whichever phase's records hold its cells. All six records, three a phase, are counted as certified, and
    five once one's certificate is gone. A capture of exactly the bytes the server's side saw agrees in each
    direction and pairs every connection."""
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    T = PARAMS.window
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 20, 300, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([" c"]), _response([])]
    relay, servers, mark = _served(out, key, requests, responses, phases=2, arrive=(0, T - 10, T - 3, T + 30))
    # the phase of each connection's admission, of its request's `END` and of its last cell
    assert [(e["admitted"] // T, e["ingress_end"] // T, e["ended"] // T)
            for e in sorted(relay.log, key=lambda e: e["ordinal"])] == [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
    logged = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in relay.log}
    assert [_sent(s) for s in servers] == [with_release(r, logged[k + 1][0], relay.run_id, logged[k + 1][1])
                                           for k, r in enumerate(requests)]
    # the bytes each phase's record holds, in each direction: some in each, and the requests' and responses' in all
    ingress, egress = ([sum(len(read_cell(frame, PARAMS.cell_bytes)[2]) for row in r.grid for frame in row if frame)
                        for r in sorted(ws.read_records(out, PARAMS)[name], key=lambda r: r.window)]
                       for name in (ws.INGRESS, ws.EGRESS))
    assert len(ingress) == len(egress) == 2 and all(ingress) and all(egress)
    assert (sum(ingress), sum(egress)) == (sum(map(len, requests)), sum(map(len, responses)))
    record = cf.record_counts(out)
    phases = json.loads((out / "deployment.json").read_text())["windows"]
    assert phases == 2 and record["phase_records"] == {"certified": 6, "of": len(ws.CERTIFIED) * phases}
    assert record["connections"] == [
        *({"connection": k + 1, "release": logged[k + 1][0], "exact": True, "why": None, "to_server": len(_sent(s)),
           "from_server": len(r)} for k, (s, r) in enumerate(zip(servers, responses))),
        {"connection": "watermark", "release": None, "exact": True, "why": None, "to_server": len(_sent(mark)),
         "from_server": None}]
    seen = _seen(tmp_path, [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), b"")])
    result = cf.read([seen], *_given(), run=out)
    got = result["compared"]
    to_server, from_server = sum(len(_sent(s)) for s in servers) + len(_sent(mark)), sum(egress)
    assert (got["to_server"]["record_bytes"], got["to_server"]["capture"]["stream_bytes"],
            got["to_server"]["says"]) == (to_server, to_server, "agrees")
    assert (got["from_server"]["record_bytes"], got["from_server"]["capture"]["stream_bytes"],
            got["from_server"]["says"]) == (from_server, from_server, "agrees")
    assert got["pairing"] == {"connections": 5, "flows": 5, "pairs": 5, "pairs_inside_bounds": 0,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0, "says": "agrees"}
    assert cf.table(result).splitlines()[1] == \
        "record: 5 connection(s) to the server; 6 of its execution phases' 6 records certified"
    # one record's certificate gone: the other five are counted, of the six
    certificates = (out / "certified.jsonl").read_text().splitlines()
    assert len(certificates) == 6
    (out / "certified.jsonl").write_text("".join(line + "\n" for line in certificates[:-1]))
    assert cf.record_counts(out) == record | {"phase_records": {"certified": 5, "of": 6}}


def _three_ids(tmp_path: Path):
    """Three id requests through the device in the id egress form at 4,096-byte cells and one slot, each answered an
    event of exactly P bytes a token, its ids 0, 1, …, and then the trailer, and the watermark stream."""
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    params = dataclasses.replace(PARAMS, cell_bytes=4096, slots=1, egress="id")
    tokens, prompts = (1, 8, 3), ([7], list(range(1, 801)), [3, 4, 5])
    requests = [ws.id_request(model="m", user=f"user-{k}", prompt=prompt, max_tokens=n, frame_bytes=params.cell_bytes,
                              offset=0) for k, (prompt, n) in enumerate(zip(prompts, tokens))]
    responses = [ws.id_response(list(range(n)), params.payload) for n in tokens]
    relay, servers, mark = _served(out, key, requests, responses, params=params)
    return out, params, tokens, requests, responses, relay, servers, mark


def test_a_run_in_the_id_egress_form_at_the_id_fronts_cell_agrees_in_each_direction(tmp_path):
    """The id egress form at the id front's cell of 4,096 bytes (`ids.sh`'s): three id requests
    (`active.id_request`), one whose prompt takes three cells, each answered in the id form, an event of exactly P
    bytes a token and then the trailer, and the watermark stream. The device's own read accepts the run, so every
    egress cell is one the id form reads. The schedule the reader rebuilds from the deployment is the id form's,
    whose digest each record's head holds and the bytes form's is not, so all three records are counted as
    certified. A connection's count to the server is its request with its release headers, and from it P bytes a
    token and the trailer. A capture of exactly the bytes the server's side saw agrees in each direction and pairs
    every connection, the server's answer on the watermark's in no sum."""
    out, params, tokens, requests, responses, relay, servers, mark = _three_ids(tmp_path)
    P = params.payload
    assert 2 * P < len(requests[1]) <= 3 * P and [len(r) for r in responses] == [n * P + len(TRAILER) for n in tokens]
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["params"]["egress"] == "id"
    assert deployment["digest"] == params.digest.hex() != dataclasses.replace(params, egress="bytes").digest.hex()
    records = ws.read_records(out, params)
    device = audit(params, records[ws.INGRESS], records[ws.EGRESS], records[ws.FAILS])
    assert device["accepted"], device["windows"]
    assert [c["egress_cells"] for c in device["connections"]] == [n + 1 for n in tokens]
    release = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in relay.log}
    assert [_sent(s) for s in servers] == [with_release(r, release[k + 1][0], relay.run_id, release[k + 1][1])
                                           for k, r in enumerate(requests)]
    record = cf.record_counts(out)
    assert record["phase_records"] == {"certified": 3, "of": 3}
    assert record["connections"] == [
        *({"connection": k + 1, "release": release[k + 1][0], "exact": True, "why": None, "to_server": len(_sent(s)),
           "from_server": n * P + len(TRAILER)} for k, (s, n) in enumerate(zip(servers, tokens))),
        {"connection": "watermark", "release": None, "exact": True, "why": None, "to_server": len(_sent(mark)),
         "from_server": None}]
    seen = _seen(tmp_path, [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), ANSWER)])
    result = cf.read([seen], *_given(), run=out)
    got = result["compared"]
    to_server, from_server = sum(len(_sent(s)) for s in servers) + len(_sent(mark)), sum(map(len, responses))
    assert (got["to_server"]["capture"]["stream_bytes"], got["to_server"]["record_bytes"],
            got["to_server"]["difference"], got["to_server"]["says"]) == (to_server, to_server, [0, 0], "agrees")
    assert (got["from_server"]["capture"]["stream_bytes"], got["from_server"]["record_bytes"],
            got["from_server"]["difference"], got["from_server"]["says"]) == \
        (from_server + len(ANSWER), from_server, [0, 0], "agrees")
    assert got["pairing"] == {"connections": 4, "flows": 4, "pairs": 4, "pairs_inside_bounds": 0,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0, "says": "agrees"}
    assert got["watermark_answer_bytes"] == len(ANSWER) and got["bounded"] == got["unpaired"] == []
    assert cf.table(result).splitlines()[1] == \
        "record: 4 connection(s) to the server; 3 of its execution phases' 3 records certified"


def _timestamped(packet: bytes) -> bytes:
    """One of the fixture's IPv4 TCP packets with the twelve bytes of TCP options a Linux loopback's packets carry,
    two no-ops and a timestamp: in an Ethernet frame its headers are 66 bytes."""
    return (packet[:2] + (int.from_bytes(packet[2:4], "big") + 12).to_bytes(2, "big") + packet[4:32] + bytes([8 << 4])
            + packet[33:40] + b"\x01\x01\x08\x0a" + bytes(8) + packet[40:])


def test_a_capture_kept_at_54_bytes_a_packet_holds_no_content_and_is_read_as_one_kept_at_130_is(tmp_path):
    """The id-form run's capture with a Linux loopback's headers, 66 bytes a packet (Ethernet, IPv4, TCP with its
    timestamp), kept twice: at the reader's 130 bytes a packet and at 54, the least a TCP packet's headers are over
    IPv4 in an Ethernet frame. The file kept at 130 holds the first 64 bytes of every segment, a request's start and
    a token id among them; the file kept at 54 holds no byte past any packet's headers. The reader reads none of
    TCP's options and takes every length from the headers and the length on the wire, so what it reads of the two is
    equal field for field but for the snapshot length it reports: each direction agrees and every connection pairs."""
    out, _, _, _, responses, _, servers, mark = _three_ids(tmp_path)
    streams = [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), ANSWER)]
    files = {kept: _seen(tmp_path, streams, name=f"kept{kept}.pcap", frame=lambda p: fx.ethernet(_timestamped(p)),
                         snapshot=kept) for kept in (cf.SNAPSHOT, 54)}
    held = {kept: [data for _, _, data in cf.packets(path, {})] for kept, path in files.items()}
    assert {len(data) for data in held[54]} == {54} and min(map(len, held[cf.SNAPSHOT])) == 66
    start, token = _sent(servers[0])[:64], b'"token_id":%10d' % 4
    assert start.startswith(b"POST ") and any(data[66:] == start for data in held[cf.SNAPSHOT])
    assert any(token in data[66:] for data in held[cf.SNAPSHOT])
    assert not any(word in files[54].read_bytes() for word in (b"POST", b"token_id", b"data:"))
    results = {kept: cf.read([path], *_given(), run=out) for kept, path in files.items()}
    assert [f["snapshot"] for f in results[54]["capture"]["files"]] == [54]
    results[54]["capture"]["files"][0]["snapshot"] = cf.SNAPSHOT
    assert results[54] == results[cf.SNAPSHOT]
    got = results[54]["compared"]
    assert got["to_server"]["says"] == got["from_server"]["says"] == got["pairing"]["says"] == "agrees"
    assert got["to_server"]["difference"] == got["from_server"]["difference"] == [0, 0]
    assert (got["pairing"]["connections"], got["pairing"]["flows"], got["pairing"]["pairs"]) == (4, 4, 4)


def test_a_requests_place_in_the_release_order_is_its_end_cells_place_in_the_ingress_record(tmp_path):
    """Twelve requests at once through a network certifier of twelve lanes, six that end their ingress in one bucket
    and six in the next. A request's place N in the release order is a decimal header of the request as the server
    gets it, so past ten releases its length follows its place. The device keeps its connections in the order it
    admitted them, their ordinals'; its audit flags a bucket whose cells are in any other, and its commitment to a
    record's rows refuses one. Here they are sorted the other way once admitted, so each bucket's `END`s are in its
    row, and released, by falling ordinal. The reader's place of each connection is the one the device logged and its
    count the bytes that connection's server got, where the order by bucket and then ordinal has four of them a byte
    off; a capture of what the servers got agrees and pairs every connection."""
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    params = dataclasses.replace(PARAMS, lanes=12, queue=12)
    requests = [ws.http_request({"prompt": "x" * (10 + k if k % 2 == 0 else 100 + 5 * k)}, "h:1") for k in range(12)]
    responses = [_response([" w"] * (k + 1)) for k in range(12)]
    relay, servers, mark = _served(out, key, requests, responses, params=params, order=lambda c: -c.ordinal)
    logged = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in relay.log}
    by_bucket: dict[int, list[int]] = {}
    for o in sorted(logged, key=lambda o: logged[o][0]):
        by_bucket.setdefault(logged[o][1], []).append(o)
    assert list(by_bucket.values()) == [[11, 9, 7, 5, 3, 1], [12, 10, 8, 6, 4, 2]]
    assert [_sent(s) for s in servers] == [with_release(r, logged[k + 1][0], relay.run_id, logged[k + 1][1])
                                           for k, r in enumerate(requests)]
    record = cf.record_counts(out)
    assert record["phase_records"] == {"certified": 3, "of": 3}
    assert {c["connection"]: c["release"] for c in record["connections"][:-1]} == {o: n for o, (n, _) in logged.items()}
    assert [c["to_server"] for c in record["connections"]] == [*(len(_sent(s)) for s in servers), len(_sent(mark))]
    assert all(c["exact"] for c in record["connections"])
    by_ordinal = {o: n for n, (_, o) in enumerate(sorted((a, o) for o, (_, a) in logged.items()))}
    assert [o for o in sorted(logged) if len(with_release(requests[o - 1], by_ordinal[o], relay.run_id, logged[o][1]))
            != len(_sent(servers[o - 1]))] == [2, 4, 10, 12]
    seen = _seen(tmp_path, [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), b"")])
    got = cf.read([seen], *_given(), run=out)["compared"]
    assert (got["to_server"]["says"], got["from_server"]["says"]) == ("agrees", "agrees")
    assert got["pairing"] == {"connections": 13, "flows": 13, "pairs": 13, "pairs_inside_bounds": 0,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0, "says": "agrees"}


def test_the_servers_answer_on_a_watermark_connection_in_no_pair_makes_its_direction_a_bound(tmp_path):
    """The capture holds one byte fewer to the server on the watermark's connection than the record has, so that
    connection pairs with no flow, and the server answered on it. The answer is in no record and can't be told from
    the server's other bytes, so it stays in the sum from the server: that direction says the range, from 0 to the
    bytes the capture counts from the server on the largest flow in no pair, never that it differs by the answer's
    bytes. Where the capture holds no flow in no pair, no answer is in the sum and the direction is exact; a flow in
    no pair that lacks its FIN from the server counts as it does whole, and its direction has no most."""
    out, _, _, responses, servers, mark = _three(tmp_path)
    streams = [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark)[:-1], ANSWER)]
    result = cf.read([_seen(tmp_path, streams)], *_given(), run=out)
    got = result["compared"]
    assert (got["to_server"]["says"], got["from_server"]["difference"], got["from_server"]["says"]) == \
        ("differs by -1 bytes", [0, len(ANSWER)], f"bound: difference 0 to +{len(ANSWER)} bytes")
    assert [c["connection"] for c in got["unpaired"]] == ["watermark"]
    assert (got["watermark_answer_bytes"], got["watermark_answer_in_sum"]) == (None, len(ANSWER))
    assert "the watermark's connection is in no pair: the server's answer on it is in no record, and the sum from " \
        f"the server holds at most {len(ANSWER)} bytes of it" in cf.table(result).splitlines()
    # on the fixture's two flows: the second as the flow of a watermark connection whose record has a byte more
    record = _record((300, 700), (121, None))
    held = cf.compare([FIRST, SECOND], record)
    assert (held["to_server"]["says"], held["from_server"]["says"], held["watermark_answer_in_sum"]) == \
        ("differs by -1 bytes", "bound: difference 0 to +64 bytes", 64)
    absent = cf.compare([FIRST], record)
    assert (absent["to_server"]["says"], absent["from_server"]["says"], absent["watermark_answer_in_sum"]) == \
        ("differs by -121 bytes", "agrees", 0)
    cut = cf.compare([FIRST, SECOND | {"from_server": _tcp(5, 334, 64, 64, fin=False)}], record)
    assert (cut["from_server"]["says"], cut["watermark_answer_in_sum"]) == ("bound: difference at least 0 bytes", 64)
    paired = cf.compare([FIRST, SECOND], _record((300, 700), (120, None)))
    assert (paired["from_server"]["says"], paired["watermark_answer_bytes"], paired["watermark_answer_in_sum"]) == \
        ("agrees", 64, 0)


def test_a_byte_the_record_doesnt_hold_and_a_flow_it_doesnt_hold_are_differences(tmp_path):
    """One byte more from the server on a connection, one byte fewer to it, a connection the capture doesn't hold
    and a flow the record doesn't: each is the difference in bytes in its direction, and unpaired on its side."""
    out, _, _, responses, servers, mark = _three(tmp_path)
    streams = [*((_sent(s), r) for s, r in zip(servers, responses)), (_sent(mark), b"")]

    def compared(changed: list) -> dict:
        return cf.read([_seen(tmp_path, changed)], *_given(), run=out)["compared"]

    more = compared([(streams[0][0], streams[0][1] + b"!"), *streams[1:]])
    assert (more["to_server"]["says"], more["from_server"]["says"]) == ("agrees", "differs by +1 bytes")
    assert (more["pairing"]["pairs"], more["pairing"]["unpaired_connections"], more["pairing"]["unpaired_flows"]) == \
        (3, 1, 1)
    assert [c["connection"] for c in more["unpaired"]] == [1]
    less = compared([streams[0], (streams[1][0][:-1], streams[1][1]), *streams[2:]])
    assert (less["to_server"]["says"], less["from_server"]["says"]) == ("differs by -1 bytes", "agrees")
    without = compared([streams[0], streams[2], streams[3]])
    assert (without["to_server"]["says"], without["from_server"]["says"]) == \
        (f"differs by -{len(streams[1][0]):,} bytes", f"differs by -{len(streams[1][1]):,} bytes")
    assert (without["pairing"]["unpaired_connections"], without["pairing"]["unpaired_flows"]) == (1, 0)
    extra = compared([*streams, (b"GET /metrics HTTP/1.1\r\n\r\n", b"HTTP/1.1 404 Not Found\r\n\r\n")])
    assert (extra["to_server"]["says"], extra["from_server"]["says"]) == \
        ("differs by +25 bytes", "differs by +26 bytes")
    assert (extra["pairing"]["unpaired_connections"], extra["pairing"]["unpaired_flows"]) == (0, 1)
    # the watermark's bytes are the record's only where `warden.json` says the run had one
    (out / "warden.json").write_text(json.dumps({"connections": []}))
    plain = compared(streams)
    assert (plain["to_server"]["says"], plain["from_server"]["says"]) == \
        (f"differs by +{len(_sent(mark)):,} bytes", "agrees")
    assert plain["watermark_answer_bytes"] is None and plain["pairing"]["unpaired_flows"] == 1


def test_a_connection_that_failed_is_a_bound_and_an_uncertified_record_is_said(tmp_path):
    """The server sends 100 bytes of a response and never ends it, so the network certifier fails the connection
    (late) with one cell of them in its record and stops reading. The record's count of that connection is a floor
    from the server and a ceiling to it: the capture's 100 bytes are inside it, the flow pairs inside the bounds, and
    each direction says the range, never that it agrees. A record with no certificates is counted as one."""
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 20)]
    responses = [_response([" a", " b"]), fx.data(100)]
    _, servers, _ = _served(out, key, requests, responses, held=(1,), watermark=False)
    payload = PARAMS.payload
    assert 0 < payload < 100
    record = cf.record_counts(out)
    assert record["connections"] == [
        {"connection": 1, "release": 0, "exact": True, "why": None, "to_server": len(_sent(servers[0])),
         "from_server": len(responses[0])},
        {"connection": 2, "release": 1, "exact": False, "why": "failed (late)", "to_server": len(_sent(servers[1])),
         "from_server": payload}]
    seen = _seen(tmp_path, [(_sent(s), r) for s, r in zip(servers, responses)])
    result = cf.read([seen], *_given(), run=out)
    got = result["compared"]
    released = len(_sent(servers[1]))
    assert (got["to_server"]["difference"], got["to_server"]["says"]) == \
        ([0, released], f"bound: difference 0 to +{released:,} bytes")
    assert (got["from_server"]["difference"], got["from_server"]["says"]) == \
        ([None, 100 - payload], f"bound: difference at most +{100 - payload} bytes")
    assert got["pairing"] == {"connections": 2, "flows": 2, "pairs": 1, "pairs_inside_bounds": 1,
                              "unpaired_connections": 0, "unpaired_flows": 0, "partial_flows": 0,
                              "says": "1 connection(s) and 1 flow(s) have no pair of equal counts"}
    assert got["bounded"] == [{"connection": 2, "why": "failed (late)", "to_server": released, "from_server": payload}]
    assert f"    connection 2 is a bound, failed (late): at most {released:,} bytes to the server, at least " \
           f"{payload} from it" in cf.table(result).splitlines()
    (out / "certified.jsonl").unlink()
    again = cf.read([seen], *_given(), run=out)
    assert again["record"]["phase_records"] == {"certified": 0, "of": 3} and again["compared"] == got
    assert cf.table(again).splitlines()[1] == ("record: 2 connection(s) to the server; 0 of its execution phases' 3 "
                                                "records certified")
    # fewer bytes than the record's floor is outside the bound
    short = _seen(tmp_path, [(_sent(servers[0]), responses[0]), (_sent(servers[1]), responses[1][:payload - 1])])
    assert cf.read([short], *_given(), run=out)["compared"]["from_server"]["says"] == \
        "differs: the capture has at least 1 bytes less"
