"""The consistency check on synthetic network certificates: φ's canonical framing, the certified requests against the
workload's run inputs, and the egress bytes by class, where nothing is left unexplained but a frame that isn't a
canonical event."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    EVENT_BYTES,
    EVENT_TEXT,
    encode_frame,
    event_body,
)
from verity.core.protocols.certifier.grid import COMPLETE, Record

HERE = Path(__file__).resolve().parents[1]
B = 4096


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_consistency", HERE / "consistency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cons = _load()
ar = cons.ar


def _record(link: str, frames: list[bytes], allocation: int = 4) -> Record:
    rows = [frames[i : i + allocation] for i in range(0, len(frames), allocation)] or [[]]
    return Record(link, 0, [row + [None] * (allocation - len(row)) for row in rows], COMPLETE)


def _request(session: str, tamper: bool = False) -> list[bytes]:
    body = ar.live_body(session, "served")
    if tamper:
        body = body | {"max_tokens": body["max_tokens"] + 1}
    raw = ar.live_request_bytes(session, body, None, B)
    framed = ar.request_frames(raw, B)
    return [framed[i : i + B] for i in range(0, len(framed), B)]


def _response(first_position: int) -> tuple[list[bytes], list[bytes]]:
    events = [ar.event_bytes(t, first_position + k, k == 0) for k, t in enumerate((b" to", b" be"))]
    return [encode_frame(DATA, CANONICAL_STREAM, event_body(e), B) for e in events], events


def test_requests_equal_the_workloads_run_inputs_and_the_rest_is_declared_or_unexplained():
    frames = _request("b4.0.0") + _request("b4.0.1", tamper=True)
    payloads, phi = cons.deframe([_record("requests", frames)], B)
    assert phi["non_canonical"] == [] and phi["frames"] == len(frames)
    sessions, tally = cons.ingress(payloads, "served")
    assert sessions == {1: "b4.0.0", 2: "b4.0.1"} and tally["fill_not_zero"] == []
    assert tally["run_inputs_checked"] == 2 and tally["run_inputs_mismatched"] == ["b4.0.1"]
    streams, _ = cons.requests(payloads)
    assert sum(tally["bytes"].values()) == sum(4 + len(r) for r in streams.values())
    assert tally["bytes"]["crossing"] > 0 and tally["bytes"]["unexplained"] >= len(b'"b4.0.0"') + len(b'"b4.0.1"')


def test_a_frame_with_nonzero_fill_or_off_the_canonical_stream_is_not_canonical():
    bad = bytearray(encode_frame(DATA, CANONICAL_STREAM, b"x", B))
    bad[-1] = 1
    off = encode_frame(DATA, 1, b"y", B)
    _, phi = cons.deframe([_record("responses", [bytes(bad), off, encode_frame(DATA, CANONICAL_STREAM, b"z", B)])], B)
    assert phi["non_canonical"] == [[0, 0, 0], [0, 0, 1]] and phi["frames"] == 1


def test_a_requests_fill_that_isnt_zero_is_named():
    frames = _request("b4.0.0")
    last = bytearray(frames[-1])
    last[-1] = 1
    payloads, _ = cons.deframe([_record("requests", frames[:-1] + [bytes(last)])], B)
    assert cons.ingress(payloads, "served")[1]["fill_not_zero"] == [1]


def test_egress_bytes_are_crossing_or_declared():
    frames, events = _response(first_position=10)
    payloads, _ = cons.deframe([_record("responses", frames)], B)
    log = [(10 + k, "b4.0.0", k, 0, e) for k, e in enumerate(events)]
    tally = cons.egress(payloads, log)
    assert tally["sessions_differing_from_server_log"] == [] and tally["events_compared_with_server_log"] == 2
    assert cons.egress(payloads, [])["events_compared_with_server_log"] == 0
    assert tally["bytes"] == {"crossing": 2 * EVENT_TEXT, "declared": 2 * (EVENT_BYTES - EVENT_TEXT)}
    junk, _ = cons.deframe([_record("responses", [encode_frame(DATA, CANONICAL_STREAM, b"data: x\n\n", B)])], B)
    assert cons.egress(junk, [])["bytes"] == {"unexplained": len(b"data: x\n\n")}


def test_an_event_that_differs_from_the_servers_log_is_named():
    frames, events = _response(first_position=0)
    payloads, _ = cons.deframe([_record("responses", frames)], B)
    log = [(k, "b4.0.0", k, 0, ar.event_bytes(b" by", k, k == 0)) for k in range(len(events))]
    assert cons.egress(payloads, log)["sessions_differing_from_server_log"] == ["b4.0.0"]


def test_check_run_reads_a_runs_files(tmp_path):
    frames, _events = _response(first_position=0)
    ar.write_blobs(tmp_path / "requests.records", ar.pack_record(_record("requests", _request("b4.0.0"))))
    ar.write_blobs(tmp_path / "responses.records", ar.pack_record(_record("responses", frames)))
    out = cons.check_run(tmp_path, B, "served")
    assert out["ingress"]["run_inputs_mismatched"] == [] and out["egress"]["events"] == 2
