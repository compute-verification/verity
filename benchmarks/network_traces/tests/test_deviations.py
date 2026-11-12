"""deviations, the server that deviates behind the relay. Each class's deviation of an id-form response, whatever
chunks it arrives in: the planted bits are where the receiver reads them, and nothing else changes. Then each class
through the relay on a `VirtualClock` (`served_cheats.relay_run`, the id egress form): the honest responses are
accepted by the relay's check and by the I/O tie; changed ids and a dropped event pass the relay's check and are
refused by the tie's content, and the receiver has every planted bit; a withheld response fails late and bytes out of
the id form fail (format), each refused by the relay's audit in its own words, the second with nothing read."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dv = _load("network_traces_deviations", "benchmarks/network_traces/deviations.py")
sc = _load("network_traces_served_cheats_relay", "benchmarks/network_traces/served_cheats.py")
ws, IT = dv.ws, dv.IT
PARAMS = dv.WardenParams(cell_bytes=256, slots=2, bucket_ms=100, delay=3, lanes=8, queue=8, window=60,
                         ingress_cells=40, ingress_buckets=30, egress_cells=64, egress="id")
P = PARAMS.payload
IDS = [1000, 1001, 2002, 2003, 3004, 3005, 4006]


def _bits(bits: list[int]):
    left = iter(bits)
    return lambda: next(left)


def _through(dev, response: bytes, chunk: int) -> bytes:
    out = b"".join(dev.feed(response[i:i + chunk]) for i in range(0, len(response), chunk))
    return out + dev.end()


@pytest.mark.parametrize("chunk", [1, 7, P, P + 1, 10 * P])
def test_the_tokens_class_puts_a_planted_bit_in_each_id_and_changes_nothing_else(chunk):
    bits = [1, 1, 0, 1, 0, 0, 1]
    dev = dv.Deviation("tokens", _bits(bits), P)
    ids, rest = dv.events_of(_through(dev, ws.id_response(IDS, P), chunk), P)
    assert [x & 1 for x in ids] == bits == dev.planted
    assert [x >> 1 for x in ids] == [x >> 1 for x in IDS] and rest == IT.TRAILER


@pytest.mark.parametrize("chunk", [1, P, P + 3, 10 * P])
@pytest.mark.parametrize("n", [0, 1, 4, 7])
@pytest.mark.parametrize("bit", [0, 1])
def test_the_length_class_drops_at_most_the_last_event_to_set_the_counts_parity(chunk, n, bit):
    dev = dv.Deviation("length", _bits([bit]), P)
    ids, rest = dv.events_of(_through(dev, ws.id_response(IDS[:n], P), chunk), P)
    assert ids == IDS[:n if n % 2 == bit or n == 0 else n - 1] and rest == IT.TRAILER
    assert n == 0 or len(ids) % 2 == bit


def test_the_other_classes_hold_withhold_or_append_and_honest_changes_nothing():
    response = ws.id_response(IDS, P)
    honest = dv.Deviation("honest", _bits([]), P)
    assert _through(honest, response, 5) == response and honest.planted == [] and not honest.delay
    held, on_time = dv.Deviation("bucket", _bits([1]), P, 0.5), dv.Deviation("bucket", _bits([0]), P, 0.5)
    assert (held.delay, on_time.delay) == (0.5, 0.0) and _through(held, response, 9) == response
    withheld, served = dv.Deviation("fail", _bits([1]), P), dv.Deviation("fail", _bits([0]), P)
    assert withheld.withhold and _through(withheld, response, 9) == b""
    assert not served.withhold and _through(served, response, 9) == response
    extra = dv.Deviation("extra", _bits([1, 0, 1, 1, 0, 0, 1, 0]), P)
    out = _through(extra, response, 11)
    assert out[:len(response)] == response and out[len(response):] == b"10110010".ljust(P, b" ")
    with pytest.raises(ValueError, match="a class is one of"):
        dv.Deviation("late", _bits([]), P)


def test_the_score_is_zero_when_every_reading_is_the_same_and_whole_when_the_reading_is_the_bit():
    planted = [[0], [1], [1], [0]]
    assert dv.score("fail", planted, [[0], [1], [1], [0]]) == {"planted": 4, "read": 4, "right": 4, "information": 4.0}
    same = dv.score("bucket", planted, [[(3, 0)]] * 4)
    assert same["information"] == 0.0 and same["right"] == 2
    assert dv.score("bucket", planted, [[(3, 0)], [(3, 1)], [(3, 1)], [(3, 0)]])["right"] == 4
    unread = dv.score("extra", [[1, 0]] * 2, [[], []])
    assert (unread["read"], unread["right"], unread["information"]) == (0, 0, 0.0)


def test_the_fronts_address_comes_from_the_named_variable_and_no_message_holds_it():
    assert dv.front_of("FRONT", {"FRONT": "front.example:8000"}) == ("front.example", 8000)
    for environ in ({}, {"FRONT": "front.example"}, {"FRONT": "front.example:port"}):
        with pytest.raises(SystemExit) as e:
            dv.front_of("FRONT", environ)
        assert "FRONT" in str(e.value) and "front.example" not in str(e.value)


def _requests(users: str = "user") -> tuple[list[dict], list[bytes], list[list[int]]]:
    """Eight id requests named ``<users>-<k>``: their fields, their bytes, and the stand-in's own ids of each."""
    fields = [{"model": "fake", "user": f"{users}-{k}", "prompt": [7, 8, 9 + k], "max_tokens": 4 + k, "seed": k}
              for k in range(sc.REQUESTS)]
    requests = [dv.id_request(**f, frame_bytes=PARAMS.cell_bytes, offset=0) for f in fields]
    return fields, requests, [ws.fake_ids(json.loads(request.partition(dv.BLANK)[2])) for request in requests]


FIELDS, REQUESTS, HONEST = _requests()


def _try(tmp: Path, cls: str, bits: list[int], chunk: int = 300, users: str = "user") -> dict:
    """Class `cls` through the relay on a virtual clock: eight id requests named ``<users>-<k>``, each answered by the
    stand-in's own ids as the class deviates them, the k-th connection planting from `bits` in turn; its row."""
    fields, requests, honest = _requests(users)
    bit, served, lines = _bits(bits), [], []
    for k in range(len(requests)):
        dev = dv.Deviation(cls, bit, P)
        out = _through(dev, ws.id_response(honest[k], P), chunk)
        served.append((out, not dev.withhold))
        lines.append({"release": k, "class": cls, "planted": "".join(map(str, dev.planted))})
    log = sc.relay_run(tmp, tmp.parent / f"key-{cls}", {}, PARAMS, requests=requests, served=served)
    assert [c["release"] for c in sorted(log, key=lambda c: c["ordinal"])] == list(range(sc.REQUESTS))
    (tmp / "traffic.json").write_text(json.dumps([{"at": 0.0, "ids": f} for f in fields]))
    (tmp / dv.PLANTED).write_text("".join(json.dumps(x) + "\n" for x in lines))
    (tmp / "check.json").write_text(json.dumps(ws.check(tmp)))
    return dv.row(cls, tmp, dv.stand_in_ids(tmp))


def test_honest_responses_pass_the_relays_check_and_the_tie(tmp_path):
    row = _try(tmp_path / "honest", "honest", [])
    assert (row["relay"], row["tie"], row["ok"], row["planted"]) == ("accepted", "accepted", True, 0)


def test_planted_ids_pass_the_relays_check_and_the_ties_content_refuses_them(tmp_path):
    row = _try(tmp_path / "tokens", "tokens", [1, 0] * 60)
    assert (row["relay"], row["tie"], row["ok"]) == ("accepted", "refused", True)
    assert row["words"].startswith("IoTieWindow.content at window 0, bucket ") and "isn't the Commit's id" in row["words"]
    assert row["planted"] == row["right"] == row["read"] > sc.REQUESTS and row["information"] > 0


def test_a_dropped_event_passes_the_relays_check_and_the_tie_finds_the_output_no_cell_carries(tmp_path):
    row = _try(tmp_path / "length", "length", [(len(ids) + 1) % 2 for ids in HONEST])
    assert (row["relay"], row["tie"], row["ok"]) == ("accepted", "refused", True)
    assert row["words"].startswith("IoTieWindow.content at window -1: connection ")
    assert row["planted"] == row["right"] == sc.REQUESTS and row["charged"].startswith(f"{sc.REQUESTS * 7} bits")


def test_a_withheld_response_fails_late_and_the_relays_audit_refuses_the_run(tmp_path):
    row = _try(tmp_path / "fail", "fail", [0, 1, 1, 0, 0, 0, 1, 0])
    assert (row["relay"], row["ok"], row["refusals"]) == ("refused", True, 3)
    assert row["words"].startswith("audit: connection 2 failed closed (late) at egress cell 0")
    assert row["planted"] == row["right"] == sc.REQUESTS and row["information"] > 0


def test_bytes_out_of_the_id_form_fail_their_connection_and_none_is_read(tmp_path):
    row = _try(tmp_path / "extra", "extra", [1, 0, 0, 1] * 16)
    assert (row["relay"], row["ok"], row["refusals"]) == ("refused", True, sc.REQUESTS)
    assert "failed closed (format) at egress cell" in row["words"]
    assert (row["planted"], row["read"], row["right"], row["information"]) == (8 * sc.REQUESTS, 0, 0, 0.0)


def test_the_tie_is_not_run_without_the_commits_ids_and_the_row_says_so(tmp_path):
    _try(tmp_path / "tokens", "tokens", [1, 0] * 60)
    row = dv.row("tokens", tmp_path / "tokens", None)
    assert (row["tie"], row["ok"]) == (dv.NOT_RUN, None) and "--ids" in row["words"]
    assert "not shown either way" in dv.table([row])


def test_a_trys_requests_take_its_name_and_one_record_over_two_tries_holds_each_once(tmp_path):
    """Two tries against one server leave one record of its sampled ids by `user`. `--users` names a try's requests,
    which changes nothing else of its traffic; named apart, each try's tie reads its own rows of the one record."""
    named = ws.id_traffic(3, model="m", seed=5, users="tokens-ab")
    assert [x["ids"]["user"] for x in named] == ["tokens-ab-0", "tokens-ab-1", "tokens-ab-2"]
    plain = ws.id_traffic(3, model="m", seed=5)
    assert [x["ids"] | {"user": f"user-{i}"} for i, x in enumerate(named)] == [x["ids"] for x in plain]
    args = ws.parser().parse_args(["rehearse", "--fake", "--ids", "--out", str(tmp_path), "--requests", "2", "--users",
                                   "honest-ab"])
    assert [x["ids"]["user"] for x in ws.traffic_of(args, "m")] == ["honest-ab-0", "honest-ab-1"]

    tries = {"honest": tmp_path / "honest", "tokens": tmp_path / "tokens"}
    _try(tries["honest"], "honest", [], users="honest-ab")
    _try(tries["tokens"], "tokens", [1, 0] * 60, users="tokens-ab")
    record = tmp_path / "requests.jsonl"
    record.write_text("".join(json.dumps({"user": f["user"], "output_ids": ids}) + "\n"
                              for users in ("honest-ab", "tokens-ab") for f, ids in zip(*_requests(users)[::2])))
    rows = {cls: dv.row(cls, run, dv.committed_ids(run, record)) for cls, run in tries.items()}
    assert (rows["honest"]["tie"], rows["tokens"]["tie"]) == ("accepted", "refused")
    assert all(r["ok"] for r in rows.values())
    assert _requests("honest-ab")[2] != _requests("tokens-ab")[2]
