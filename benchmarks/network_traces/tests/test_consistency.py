"""The consistency check on synthetic network certificates: φ's canonical framing, the certified requests against the
workload's run inputs, and the egress bytes by class, with a completion id and a `created` second that nothing
explains."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from verity.core.protocols.certifier.active import DATA, encode_frame
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


def _request(stream: int, session: str, tamper: bool = False) -> list[bytes]:
    body = ar.live_body(session, "served")
    if tamper:
        body = body | {"max_tokens": body["max_tokens"] + 1}
    raw = ar.live_request_bytes(session, body, None, B)
    framed = ar.request_frames(stream, raw, B)
    return [framed[i : i + B] for i in range(0, len(framed), B)]


def _chunk(text: str, finish: str | None, cid: str = "cmpl-0123456789abcdef", created: int = 1791259035) -> bytes:
    chunk = {"id": cid, "object": "text_completion", "created": created, "model": "served",
             "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish, "stop_reason": None,
                          "prompt_token_ids": None, "token_ids": None}]}
    return b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"


def _response(stream: int, first_position: int) -> tuple[list[bytes], list[bytes]]:
    bodies = [_chunk(" to", None), _chunk(" be", "length"), b"data: [DONE]\n\n"]
    events = [(ar.HTTP_HEAD if k == 0 else b"") + b"id: %d\n" % (first_position + k) + e for k, e in enumerate(bodies)]
    return [encode_frame(DATA, stream, e, B) for e in events], events


def test_requests_equal_the_workloads_run_inputs_and_the_rest_is_declared_or_unexplained():
    frames = _request(1, "b4.0.0") + _request(2, "b4.0.1", tamper=True)
    streams, phi = cons.deframe([_record("requests", frames)], B)
    assert phi["non_canonical"] == [] and phi["frames"]["LAST"] == 2
    sessions, tally = cons.ingress(streams, "served")
    assert sessions == {1: "b4.0.0", 2: "b4.0.1"}
    assert tally["run_inputs_checked"] == 2 and tally["run_inputs_mismatched"] == ["b4.0.1"]
    assert sum(tally["bytes"].values()) == sum(map(len, streams.values()))
    assert tally["bytes"]["crossing"] > 0 and tally["bytes"]["unexplained"] >= len(b'"b4.0.0"') + len(b'"b4.0.1"')


def test_a_frame_with_nonzero_fill_is_not_canonical():
    bad = bytearray(encode_frame(DATA, 1, b"x", B))
    bad[-1] = 1
    _, phi = cons.deframe([_record("responses", [bytes(bad), encode_frame(DATA, 1, b"y", B)])], B)
    assert phi["non_canonical"] == [[0, 0, 0]] and phi["frames"] == {"DATA": 1}


def test_egress_bytes_by_class_and_the_unexplained_fields():
    frames, events = _response(5, first_position=10)
    streams, _ = cons.deframe([_record("responses", frames)], B)
    log = [(10 + k, "b4.0.0", k, 0, e) for k, e in enumerate(events)]
    tally = cons.egress(streams, {5: "b4.0.0"}, log)
    assert tally["streams_differing_from_server_log"] == [] and tally["events_reserialized_exactly"] == 2
    assert tally["streams_compared_with_server_log"] == 1
    assert cons.egress(streams, {5: "b4.0.0"}, [])["streams_compared_with_server_log"] == 0
    assert tally["completion_id_hex_digits"] == {16: 1} and tally["distinct_created_per_request"] == {1: 1}
    by = tally["bytes"]
    assert by["crossing"] == len(b'" to"nullnull" be""length"null')
    assert by["unexplained"] == 2 * len(b'"cmpl-0123456789abcdef"1791259035')
    assert by["advice"] == len(b"id: 10id: 11id: 12")
    assert sum(by.values()) == sum(map(len, events))


def test_an_event_that_differs_from_the_servers_log_is_named():
    frames, events = _response(5, first_position=0)
    streams, _ = cons.deframe([_record("responses", frames)], B)
    log = [(k, "b4.0.0", k, 0, e.replace(b" be", b" by")) for k, e in enumerate(events)]
    assert cons.egress(streams, {5: "b4.0.0"}, log)["streams_differing_from_server_log"] == ["b4.0.0"]


def test_check_run_reads_a_runs_files(tmp_path):
    frames, events = _response(1, first_position=0)
    ar.write_blobs(tmp_path / "requests.records", ar.pack_record(_record("requests", _request(1, "b4.0.0"))))
    ar.write_blobs(tmp_path / "responses.records", ar.pack_record(_record("responses", frames)))
    out = cons.check_run(tmp_path, B, "served")
    assert out["ingress"]["run_inputs_mismatched"] == [] and out["egress"]["requests"] == 1
