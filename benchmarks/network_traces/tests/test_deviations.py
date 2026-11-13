"""deviations, the server that deviates behind the relay. Each class's deviation of an id-form response, whatever
chunks it arrives in: the planted bits are where the receiver reads them, and nothing else changes. Then each class
through the relay on a `VirtualClock` (`_relay_run`, the id egress form): the honest responses are
accepted by the relay's check and by the I/O tie; changed ids and a dropped event pass the relay's check and are
refused by the tie's content, and the receiver has every planted bit; a withheld response fails late and bytes out of
the id form fail (format), each refused by the relay's audit in its own words, the second with nothing read; a held
response that is on time is accepted and not seen. A refusal counts only where it is the deviation's. What a try
keeps holds no salt, key or address, and the table is made from it alone."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from verity.protocols.certifier.verifier import active
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import DATA, VirtualClock
from verity.protocols.certifier.verifier.certificate import Issuer
from verity.protocols.certifier.verifier.warden import END, FAIL, REFUSED, Shim, Warden, WardenParams, cells, read_cell

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dv = _load("network_traces_deviations", "benchmarks/network_traces/deviations.py")
ws, IT = dv.ws, dv.IT
#: the connections of a try on the virtual clock: a lane and a place in the queue each, so that they arrive together
N = 8
PARAMS = WardenParams(cell_bytes=256, slots=2, bucket_ms=100, delay=3, lanes=N, queue=N, window=60,
                      ingress_cells=40, ingress_buckets=30, egress_cells=64, egress="id")
P = PARAMS.payload
IDS = [1000, 1001, 2002, 2003, 3004, 3005, 4006]
SESSION = bytes(range(32))


class _Side:
    """One end of an in-memory connection as the relay holds it: a reader fed what the peer sends, and a transport
    that keeps each write with the tick it was made in."""

    def __init__(self, warden: Warden | None = None) -> None:
        self.warden, self.writes, self.closed, self.transport = warden, [], False, self
        self.reader, self.fed, self.eof = asyncio.StreamReader(), 0, False

    def feed(self, data: bytes, end: bool) -> None:
        if data:
            self.reader.feed_data(data)
        self.fed += len(data)
        if end:
            self.reader.feed_eof()
            self.eof = True

    def write(self, data: bytes) -> None:
        self.writes.append((self.warden.ticks if self.warden else None, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


class _Cut(_Side):
    """A stream whose server closes it once it has taken `left` writes."""

    def __init__(self, left: int) -> None:
        super().__init__()
        self.left = left

    def write(self, data: bytes) -> None:
        super().write(data)
        self.left -= 1
        self.closed = self.left <= 0


def _watermark_log(relay) -> dict:
    """What `serve` keeps of a relay's watermark in its log: the last line, the lost releases and its error."""
    return {"watermark": {"last": relay.watermarked and list(relay.watermarked), "lost": relay.watermark_lost,
                          "error": relay.watermark_error}}


def _relay_run(out: Path, key: Path, params: WardenParams, requests: Sequence[bytes],
               served: Sequence[tuple[bytes, bool]], watermark: bool = False, refuse: Sequence[int] = (),
               cut: int | None = None) -> list[dict]:
    """`requests` through the relay for one window on a `VirtualClock`, the k-th server to be connected sending
    `served[k]`'s bytes and, when its flag says so, ending: a server that doesn't end withholds the rest. Published in
    `out` as `warden_serve`'s `serve` and `client` publish a run (`test_warden_serve._run`, whose servers all end),
    with the relay's log (`warden.json`), each bucket ticked once the relay has read what its servers sent. Under
    `watermark` the relay also sends the watermark stream, whose bytes go to ``watermark.bin`` and whose end to the
    log as `serve` writes it; the servers of `refuse` take no connection, and the watermark's server closes the stream
    once it has taken `cut` writes. Returns the relay's log, an entry a connection."""
    out.mkdir(parents=True, exist_ok=True)
    tau, B = params.bucket_ms / 1000, params.cell_bytes
    signer = ws.cs.signer(key, 4)
    public = signer.public_key()
    publisher = ws.Publisher(out, Issuer(signer, SESSION), params, WR.Cells.of)
    servers: list[_Side] = []
    streams: list[_Side] = []

    async def connect():
        side, k = _Side(), len(servers)
        servers.append(side)
        if k in refuse:
            raise ConnectionRefusedError("the server took no connection")
        side.feed(*served[k])
        return side.reader, side

    async def mark():
        streams.append(_Side() if cut is None else _Cut(cut))
        return streams[-1].reader, streams[-1]

    def settled(w: Warden) -> bool:
        return all(c.ingress_ended and (c.upstream is not None if c.server is None else
                                        c.bytes_out == c.server[1].fed and c.egress_ended == c.server[1].eof)
                   for c in w.active)

    async def main() -> tuple[list[_Side], Warden]:
        clock = VirtualClock(-tau / 2)
        warden = Warden(params, connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep, publish=publisher,
                        withhold=publisher.withhold, watermark=mark if watermark else None)
        running = asyncio.create_task(warden.run())
        clients, tasks = [], []
        for request in requests:
            side = _Side(warden)
            side.feed(cells(request, B), True)
            clients.append(side)
            tasks.append(asyncio.create_task(warden.handle(side.reader, side)))
        await warden.until(lambda w: w.ordinals + w.refused == len(requests))
        for t in range(params.window):
            await warden.until(settled)
            clock.advance(t * tau)
            await warden.until(lambda w, t=t: w.ticks > t)
        await running
        await asyncio.gather(*tasks)
        return clients, warden

    try:
        clients, warden = asyncio.run(main())
    finally:
        publisher.close()
        signer.close()
    deployment = ws.deployment_record(params, public, SESSION, origin_unix=0.0, windows=1, listen=("127.0.0.1", 1),
                                      upstream=("127.0.0.1", 2), slack=warden.slack, run_id=warden.run_id)
    (out / "deployment.json").write_text(json.dumps(deployment))
    log = []
    for index, (request, client) in enumerate(zip(requests, clients)):
        shim = Shim(B)
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        kinds = [k for _, k in shim.arrivals]
        shim.outcome = ("refused" if REFUSED in kinds else "end" if END in kinds else "fail" if FAIL in kinds
                        else "cut")
        log.append(ws.connection_log(index, 0.0, 0.0, request, shim, 0.0,
                                     params.payload if request.startswith(active.ID_REQUEST_HEAD) else None))
    (out / "client.json").write_text(json.dumps({"connections": log}))
    marks = _watermark_log(warden) if watermark else {}
    (out / dv.RELAY_LOG).write_text(json.dumps({"connections": warden.log} | marks))
    if watermark:
        (out / "watermark.bin").write_bytes(b"".join(data for side in streams for _, data in side.writes))
    return warden.log


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
              for k in range(N)]
    requests = [dv.id_request(**f, frame_bytes=PARAMS.cell_bytes, offset=0) for f in fields]
    return fields, requests, [ws.fake_ids(json.loads(request.partition(dv.BLANK)[2])) for request in requests]


FIELDS, REQUESTS, HONEST = _requests()


def _attempt(run: Path, cls: str, bits: list[int], chunk: int = 300, users: str = "user", late: bool = False,
             down: tuple[int, ...] = (), params=PARAMS, answers: dict[int, bytes] | None = None,
             watermark: bool = False, refuse: tuple[int, ...] = (), cut: int | None = None) -> float:
    """Class `cls` through the relay on a virtual clock, its run directory `run` as a try leaves it: eight id requests
    named ``<users>-<k>``, each answered by the stand-in's own ids as the class deviates them, the k-th connection
    planting from `bits` in turn. Under `late` a held response comes past its deadline, so never; the servers of
    `down` send nothing, whatever the class, and connection k's server answers `answers[k]` where given. Under
    `watermark` the relay sends the watermark too, the servers of `refuse` take no connection, and the watermark's
    server closes the stream once it has taken `cut` writes."""
    fields, requests, honest = _requests(users)
    bit, served, lines = _bits(bits), [], []
    for k in range(len(requests)):
        dev = dv.Deviation(cls, bit, P, 0.5)
        out = _through(dev, (answers or {}).get(k, ws.id_response(honest[k], P)), chunk)
        gone = dev.withhold or (late and dev.delay > 0) or k in down
        served.append((b"" if gone else out, not gone))
        lines.append({"release": k, "class": cls, "planted": "".join(map(str, dev.planted)), "whole": dev.whole})
    log = _relay_run(run, run.parent / f"key-{run.name}", params, requests, served, watermark, refuse, cut)
    assert [c["release"] for c in sorted(log, key=lambda c: c["ordinal"])] == list(range(N))
    (run / "traffic.json").write_text(json.dumps([{"at": 0.0, "ids": f} for f in fields]))
    (run / dv.PLANTED).write_text("".join(json.dumps(x) + "\n" for x in lines))
    (run / "check.json").write_text(json.dumps(ws.check(run)))
    return 0.0


def _kept(tmp: Path, cls: str, bits: list[int], **how) -> dict:
    """What class `cls`'s try keeps (`_attempt` in ``tmp/CLASS``), written beside its run directory as `try` does."""
    _attempt(tmp / cls, cls, bits, **how)
    kept = dv.keep(cls, tmp / cls, stand_in=True, seconds=0.0, hold_ms=500.0)
    (tmp / f"{cls}.json").write_text(json.dumps(kept))
    return kept


def _try(tmp: Path, cls: str, bits: list[int], **how) -> dict:
    """Class `cls`'s row against the stand-in's own ids."""
    kept = _kept(tmp, cls, bits, **how)
    return dv.row(kept, dv.stand_in_ids(kept))


def test_honest_responses_pass_the_relays_check_and_the_tie(tmp_path):
    row = _try(tmp_path, "honest", [])
    assert (row["relay"], row["tie"], row["ok"], row["planted"]) == ("accepted", "accepted", True, 0)


def test_planted_ids_pass_the_relays_check_and_the_ties_content_refuses_them(tmp_path):
    row = _try(tmp_path, "tokens", [1, 0] * 60)
    assert (row["relay"], row["tie"], row["ok"]) == ("accepted", "refused", True)
    assert row["words"].startswith("IoTieWindow.content at window 0, bucket ") and "isn't the Commit's id" in row["words"]
    assert row["planted"] == row["right"] == row["read"] > N and row["information"] > 0


def test_a_dropped_event_passes_the_relays_check_and_the_tie_finds_the_output_no_cell_carries(tmp_path):
    row = _try(tmp_path, "length", [(len(ids) + 1) % 2 for ids in HONEST])
    assert (row["relay"], row["tie"], row["ok"]) == ("accepted", "refused", True)
    assert row["words"].startswith("IoTieWindow.content at window -1: connection ")
    assert row["planted"] == row["right"] == N and row["charged"].startswith(f"{N * 7} bits")


def test_a_withheld_response_fails_late_and_the_relays_audit_refuses_the_run(tmp_path):
    row = _try(tmp_path, "fail", [0, 1, 1, 0, 0, 0, 1, 0])
    assert (row["relay"], row["ok"], row["refusals"]) == ("refused", True, 3)
    assert row["words"].startswith("audit: connection 2 failed closed (late) at egress cell 0")
    assert row["planted"] == row["right"] == N and row["information"] > 0


def test_bytes_out_of_the_id_form_fail_their_connection_and_none_is_read(tmp_path):
    row = _try(tmp_path, "extra", [1, 0, 0, 1] * 16)
    assert (row["relay"], row["ok"], row["refusals"]) == ("refused", True, N)
    assert "failed closed (format) at egress cell" in row["words"]
    assert (row["planted"], row["read"], row["right"], row["information"]) == (8 * N, 0, 0, 0.0)


def test_under_a_response_cap_bytes_after_a_full_response_fail_at_its_limit(tmp_path):
    """A schedule with a response cap holds a response to its request's `max_tokens`: the cell after that many ids
    fails (limit) before its form is read, and the cell after fewer fails (format). Either is the extra bytes' cell."""
    capped = replace(PARAMS, max_tokens_cap=32)
    kept = _kept(tmp_path, "extra", [1, 0, 0, 1] * 16, params=capped)
    reasons = {c["failed"] for c in kept["connections"]}
    full = [c for c in kept["connections"] if len(c["ids"]) == c["max_tokens"]]
    assert full and {c["failed"] for c in full} == {"limit"} and reasons <= {"limit", "format"}
    assert all(c["failed_at"] == len(c["ids"]) for c in kept["connections"])
    row = dv.row(kept, None)
    assert (row["relay"], row["ok"], row["refusals"], row["read"]) == ("refused", True, N, 0)
    kept["connections"][0]["failed_at"] -= 1
    assert dv.row(kept, None)["ok"] is False


def test_a_server_that_answers_out_of_the_id_form_or_stops_short_is_no_extra_bytes_refused(tmp_path):
    """The relay fails a connection whose server answers out of the id form, or ends a response without its trailer,
    at the very cell the extra bytes' would fail at. Neither counts as the deviation refused: the front's response
    wasn't whole. Nor does a response that was whole and isn't the Commit's."""
    _, _, honest = _requests("user")
    bits = [1, 0, 0, 1] * 16
    short = ws.id_response(honest[2], P)[:-len(IT.TRAILER)]
    for name, answer in (("form", b"HTTP/1.1 502 Bad Gateway\r\n\r\n"), ("short", short)):
        kept = _kept(tmp_path / name, "extra", bits, answers={2: answer})
        two = next(c for c in kept["connections"] if c["index"] == 2)
        assert (two["failed"], two["whole"], two["failed_at"]) == ("format", False, len(two["ids"]))
        row = dv.row(kept, None)
        assert (row["relay"], row["ok"], row["refusals"]) == ("refused", False, N)
    kept = _kept(tmp_path / "whole", "extra", bits)
    assert dv.row(kept, dv.stand_in_ids(kept))["ok"] is True
    other = {o: [x + 2 for x in ids] for o, ids in dv.stand_in_ids(kept).items()}
    assert dv.row(kept, other)["ok"] is False


def test_a_hold_that_is_on_time_is_accepted_and_not_seen_and_one_past_the_deadline_fails_late(tmp_path):
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    row = _try(tmp_path / "short", "bucket", bits)
    assert (row["relay"], row["ok"], row["refusals"]) == ("accepted", True, 0)
    assert row["words"] == ("accepted, and the hold is not seen at the client's bucket: 4 of 8 connections held 500 "
                            "ms, and the first cell of 8 of 8 is 3 buckets after its admission")
    assert (row["planted"], row["information"]) == (N, 0.0) and "refused" not in row["words"]
    row = _try(tmp_path / "long", "bucket", bits, late=True)
    assert (row["relay"], row["ok"], row["refusals"]) == ("refused", True, 4)
    assert row["words"].startswith("audit: connection 2 failed closed (late) at egress cell 0")
    assert row["right"] == N and row["information"] > 0


def test_a_refusal_counts_only_where_it_is_the_deviations(tmp_path):
    """A try the relay's check refuses for something the class didn't do isn't a refusal of the class: a server that
    is down under the tokens class, a connection that failed and wasn't withheld or held, a Commit id that differs
    where no bit was planted."""
    down = _try(tmp_path / "down", "tokens", [1, 0] * 60, down=(3,))
    assert (down["relay"], down["tie"], down["ok"]) == ("refused", "refused", False)
    assert down["words"].startswith("the relay's check refused a try that fails no connection: ")
    for cls, bits in (("fail", [0, 1, 1, 0, 0, 0, 1, 0]), ("bucket", [0] * 8)):
        other = _try(tmp_path / f"other-{cls}", cls, bits, down=(0,))
        assert (other["relay"], other["ok"]) == ("refused", False)
    kept = _kept(tmp_path, "tokens", [1, 0] * 60)
    commit = dv.stand_in_ids(kept)
    assert dv.row(kept, commit)["ok"] is True
    first = min(commit)
    moved = commit | {first: [commit[first][0] ^ 2, *commit[first][1:]]}
    assert dv.row(kept, moved)["ok"] is False
    assert dv.row(kept, commit | {first: commit[first][:-1]})["ok"] is False
    unplanted = _kept(tmp_path / "unplanted", "tokens", [x & 1 for ids in HONEST for x in ids])
    assert dv.row(unplanted, dv.stand_in_ids(unplanted))["ok"] is None


def test_what_a_try_keeps_holds_no_salt_or_address_and_the_table_is_made_from_it_alone(tmp_path):
    kept = _kept(tmp_path, "tokens", [1, 0] * 60)
    run, text = tmp_path / "tokens", (tmp_path / "tokens.json").read_text()
    assert set(kept) == {"class", "run_id", "digest", "params", "windows", "hold_ms", "seconds", "check",
                         "connections", "egress", "watermark", "files"}
    assert kept["watermark"] == {"last": None, "lost": [], "error": None}
    assert "upstream" not in kept["check"] and "127.0.0.1" not in text and "salt" not in text.replace(".salts", "")
    digest = dv.hashlib.sha256((run / "deployment.json").read_bytes()).hexdigest()
    assert kept["files"]["deployment.json"] == {"bytes": (run / "deployment.json").stat().st_size, "sha256": digest}
    assert set(kept["files"]) == {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()}
    conn = kept["connections"][0]
    assert (conn["user"], conn["release"], conn["held"], conn["failed"]) == ("user-0", 0, True, None)
    assert [x >> 1 for x in conn["ids"]] == [x >> 1 for x in HONEST[0]]
    assert conn["planted"] == ("10" * 60)[:len(HONEST[0])] == "".join(str(x & 1) for x in conn["ids"])
    record = ws.read_records(run, PARAMS)[ws.EGRESS][0]
    cells = {(t, i): frame for t, row in enumerate(record.grid) for i, frame in enumerate(row)
             if frame is not None and frame != IT.pad_cell(PARAMS.cell_bytes)}
    assert {(e[0], e[1]): dv.cell_of(e, PARAMS.cell_bytes) for e in kept["egress"][0]["cells"]} == cells
    before = dv.row(kept, dv.stand_in_ids(kept))
    shutil.rmtree(run)
    assert dv.main(["table", "--out", str(tmp_path), "--stand-in"]) == 0
    assert json.loads((tmp_path / "rows.json").read_text()) == [json.loads(json.dumps(before))]
    assert (tmp_path / "table.txt").read_text() == dv.table([before]) + "\n"
    failed = _kept(tmp_path, "fail", [0, 1, 1, 0, 0, 0, 1, 0])
    assert [e for e in failed["egress"][0]["cells"] if len(e) == 3] and dv.row(failed, None)["ok"] is True
    with pytest.raises(SystemExit, match="keeps no try"):
        dv.main(["table", "--out", str(tmp_path / "none")])


def test_the_servers_own_record_names_a_request_by_its_run_and_release(tmp_path):
    """The certified tick names a request ``verity-<run>-<N>``, and the server's recorder writes its sampled ids under
    that `id`: the tie reads those rows as they are, among the rows of what the server served before."""
    kept = _kept(tmp_path, "tokens", [1, 0] * 60) | {"run_id": "ab" * 16}
    record = tmp_path / "requests.jsonl"
    rows = [{"id": f"verity-{'ab' * 16}-{k}", "engine_id": f"cmpl-{k}", "output_ids": ids, "finish_reason": "length"}
            for k, ids in enumerate(HONEST)]
    earlier = [{"id": f"verity-{'cd' * 16}-{k}", "output_ids": [5, 6, 7]} for k in range(3)]
    record.write_text("".join(json.dumps(r) + "\n" for r in earlier + rows))
    commit = dv.committed_ids(kept, dv.read_commit(record))
    assert commit == dv.stand_in_ids(kept) and dv.row(kept, commit)["ok"] is True
    assert dv.committed_ids(kept | {"run_id": "cd" * 16}, dv.read_commit(record))[1] == [5, 6, 7]
    assert set(map(tuple, dv.committed_ids(kept | {"run_id": None}, dv.read_commit(record)).values())) == {()}


def test_honest_is_tried_first_and_a_refused_control_stops_the_tries(tmp_path):
    assert dv.CLASSES == ("honest", "tokens", "length", "fail", "extra", "bucket")
    flags = ["try", "--stand-in", "--out", str(tmp_path / "out"), "--scratch", str(tmp_path / "scratch")]
    args = dv.parser().parse_args(flags)
    bits = {"tokens": [1, 0] * 60, "fail": [0, 1, 1, 0, 0, 0, 1, 0]}
    tried, stopped = dv.tries(["honest", "tokens", "fail"], args,
                              lambda cls, run, _: _attempt(run, cls, bits.get(cls, [])))
    assert (tried, stopped) == (["honest", "tokens", "fail"], None)
    assert [r["ok"] for r in dv.rows_of(args.out, tried, None, True)] == [True, True, True]
    assert sorted(p.name for p in args.out.iterdir()) == ["fail.json", "honest.json", "tokens.json"]
    assert (args.scratch / "honest" / "check.json").exists()

    args = dv.parser().parse_args([*flags[:3], str(tmp_path / "refused")])
    seen = []

    def down(cls, run, _):
        seen.append(run)
        return _attempt(run, cls, [], down=(2,))

    assert dv.tries(["honest", "tokens", "fail"], args, down) == (["honest"], dv.HONEST_STOP)
    assert [p.name for p in args.out.iterdir()] == ["honest.json"] and not seen[0].exists()
    assert dv.tries(["honest"], args, down) == (["honest"], None)


def test_a_try_with_no_check_has_no_row_and_the_classes_after_it_are_tried(tmp_path, capsys):
    flags = ["try", "--stand-in", "--out", str(tmp_path / "out")]
    args = dv.parser().parse_args(flags)

    def attempt(cls, run, _):
        if cls == "tokens":
            raise dv.NoCheck("tokens: the rehearsal wrote no check")
        return _attempt(run, cls, [0, 1, 1, 0, 0, 0, 1, 0])

    assert dv.tries(["honest", "tokens", "fail"], args, attempt) == (["honest", "fail"], None)
    assert "tokens: the rehearsal wrote no check\ntokens: no row" in capsys.readouterr().out
    assert sorted(p.name for p in args.out.iterdir()) == ["fail.json", "honest.json"]

    def none(cls, run, _):
        raise dv.NoCheck(f"{cls}: the deviating process ended before it listened")

    args = dv.parser().parse_args([*flags[:3], str(tmp_path / "none")])
    assert dv.tries(["honest", "tokens"], args, none) == ([], dv.HONEST_STOP)


WATERMARK = ["--", "--watermark", "--window", "60", "--lanes", "64"]


def test_a_server_connection_that_fails_loses_its_release_and_the_tries_stop_at_that_try(tmp_path, capsys, monkeypatch):
    """The relay's connect to connection 2's server fails, so its release is never written: the watermark names it
    lost, on the line of its bucket, and the network certifier's log keeps it. No class is tried after that try, the
    command says which try, and its exit is the lost release's, whatever the rows; the table made again from what
    the tries kept says and exits the same."""
    flags = ["try", "--stand-in", "--out", str(tmp_path / "out"), "--scratch", str(tmp_path / "scratch"), *WATERMARK]
    args = dv.parser().parse_args(flags)
    seen = []

    def attempt(cls, run, _):
        seen.append(cls)
        return _attempt(run, cls, [1, 0] * 60, watermark=True, refuse=(2,) if cls == "tokens" else ())

    tried, stopped = dv.tries(["honest", "tokens", "fail"], args, attempt)
    assert (tried, seen) == (["honest", "tokens"], ["honest", "tokens"])
    words = ("tokens: STOPPED at a lost release: its network certifier's log names 1 lost on the watermark (2), so "
             "the try did not reach its server whole")
    assert stopped == words + "; not tried: fail"
    run = args.scratch / "tokens"
    assert dv.watermark_log(run)["lost"] == [2]
    assert dv.watermark_log(args.scratch / "honest") == {"last": [PARAMS.window - 1, N], "lost": [], "error": None}
    log = json.loads((run / dv.RELAY_LOG).read_text())["connections"]
    lost = next(c for c in log if c["release"] == 2)
    assert lost["upstream"].startswith("ConnectionRefusedError") and lost["failed"] == "upstream"
    t = lost["ingress_end"]
    line = dv.REL.watermark_chunk(t, sum(c["ingress_end"] <= t for c in log), [2])
    assert line in (run / "watermark.bin").read_bytes() and line.endswith(b" 2\n\r\n")
    honest = (args.scratch / "honest" / "watermark.bin").read_bytes()
    assert all(len(x.split()) == 2 for x in honest.split(b"\r\n") if b" " in x and not x.startswith(b"POST"))
    kept = json.loads((args.out / "tokens.json").read_text())
    assert kept["watermark"] == {"last": [PARAMS.window - 1, N], "lost": [2], "error": None}
    assert json.loads((args.out / "honest.json").read_text())["watermark"]["lost"] == []
    assert sorted(p.name for p in args.out.iterdir()) == ["honest.json", "tokens.json"]

    assert dv.main(["table", "--out", str(args.out), "--stand-in"]) == dv.WATERMARK_EXIT == 5
    out = capsys.readouterr().out
    assert out.rstrip().endswith(words) and "    lost:    1 release(s) named lost on the watermark" in out
    rows = json.loads((args.out / "rows.json").read_text())
    assert [(r["lost"], r["watermark_error"]) for r in rows] == [([], None), ([2], None)]
    assert "stopped with an error" not in out

    monkeypatch.setattr(dv, "tries", lambda classes, args, real=dv.tries: real(classes, args, attempt))
    monkeypatch.setattr(dv.signal, "signal", lambda *_: None)
    seen.clear()
    assert dv.main([*flags[:3], str(tmp_path / "again"), *WATERMARK]) == dv.WATERMARK_EXIT
    assert seen == ["honest", "tokens"]
    assert capsys.readouterr().out.rstrip().endswith(words + "; not tried: length, fail, extra, bucket")


def test_a_try_with_no_check_is_read_for_a_lost_release_too_and_a_missing_log_stops_the_tries(tmp_path, capsys):
    """A try that wrote no check has no row, and its network certifier's log is still read. With no log the tries stop
    only where a watermark line could have been sent: the network certifier ran, with the watermark."""
    seen = []

    def args(name, *rehearse):
        return dv.parser().parse_args(["try", "--stand-in", "--out", str(tmp_path / name / "out"), "--scratch",
                                       str(tmp_path / name / "scratch"), *rehearse])

    def attempt(how):
        def run_one(cls, run, _):
            seen.append(cls)
            if cls != "tokens":
                return _attempt(run, cls, [0, 1, 1, 0, 0, 0, 1, 0], watermark=True)
            if how == "lost":
                _attempt(run, cls, [1, 0] * 60, watermark=True, refuse=(5,))
                (run / "check.json").unlink()
            elif how == "no log":
                run.mkdir(parents=True)
                (run / "deployment.json").write_text("{}")
            raise dv.NoCheck("tokens: the rehearsal wrote no check")
        return run_one

    def tries(name, how, *rehearse):
        seen.clear()
        return *dv.tries(["honest", "tokens", "fail"], args(name, *rehearse), attempt(how)), list(seen)

    tried, stopped, ran = tries("lost", "lost", *WATERMARK)
    assert (tried, ran) == (["honest"], ["honest", "tokens"])
    assert stopped.startswith("tokens: STOPPED at a lost release")
    assert "tokens: no row" in capsys.readouterr().out and not (tmp_path / "lost" / "out" / "tokens.json").exists()

    tried, stopped, ran = tries("no-log", "no log", *WATERMARK)
    assert (tried, ran) == (["honest"], ["honest", "tokens"])
    assert stopped == (f"tokens: STOPPED: its network certifier ran with the watermark and wrote no log "
                       f"({dv.RELAY_LOG}), so whether a line named a lost release isn't known; not tried: fail")

    # the network certifier never ran: nothing was sent, and the classes after it are tried, as without the watermark
    assert tries("never-ran", "never ran", *WATERMARK) == (["honest", "fail"], None, ["honest", "tokens", "fail"])
    assert tries("plain", "no log") == (["honest", "fail"], None, ["honest", "tokens", "fail"])
    assert dv.watermark_log(tmp_path / "none") is None and dv.watermark_stop("fail", tmp_path / "none", True) is None


def test_a_watermark_its_server_closed_stops_the_tries_at_that_try_with_the_errors_kind(tmp_path, capsys):
    """The watermark's server closes the stream after its head and three lines. No release is lost, and the relay's
    log says the stream stopped: its server did not get the watermark whole, as a server that broke it would not. No
    class is tried after that try, the line names the error's kind, and the exit is the watermark's, from the tries
    and from the table made again."""
    args = dv.parser().parse_args(["try", "--stand-in", "--out", str(tmp_path / "out"), "--scratch",
                                   str(tmp_path / "scratch"), *WATERMARK])
    seen = []

    def attempt(cls, run, _):
        seen.append(cls)
        return _attempt(run, cls, [1, 0] * 60, watermark=True, cut=4 if cls == "tokens" else None)

    tried, stopped = dv.tries(["honest", "tokens", "fail"], args, attempt)
    words = ("tokens: STOPPED at a watermark error (closed): its network certifier's log says the stream stopped, so "
             "its server did not get the watermark whole")
    assert (tried, seen, stopped) == (["honest", "tokens"], ["honest", "tokens"], words + "; not tried: fail")
    mark = dv.watermark_log(args.scratch / "tokens")
    assert (mark["lost"], mark["error"]) == ([], "closed")
    assert json.loads((args.out / "tokens.json").read_text())["watermark"] == mark
    sent = (args.scratch / "tokens" / "watermark.bin").read_bytes()
    assert sent.partition(b"\r\n\r\n")[2] == b"".join(dv.REL.watermark_chunk(t, n) for t, n in ((0, 0), (1, N), (2, N)))
    assert dv.watermark_log(args.scratch / "honest")["error"] is None
    assert (args.scratch / "honest" / "watermark.bin").read_bytes().endswith(dv.REL.WATERMARK_END)

    assert dv.main(["table", "--out", str(args.out), "--stand-in"]) == dv.WATERMARK_EXIT
    out = capsys.readouterr().out
    assert out.rstrip().endswith(words)
    assert ("    watermark: stopped with an error (closed), so its server did not get it whole: no class was tried "
            "after this one") in out
    rows = json.loads((args.out / "rows.json").read_text())
    assert [(r["lost"], r["watermark_error"]) for r in rows] == [([], None), ([], "closed")]


def test_a_watermarks_error_is_kept_by_its_kind_and_named_beside_a_lost_release(tmp_path):
    """The network certifier's log gives the error that stopped its watermark as text, which can hold the address it
    connected to: what is kept, and what the stop says, is the error's first word. A try with a lost release and an
    error is stopped for both."""
    run = tmp_path / "run"
    run.mkdir()
    mark = {"last": [3, 1], "lost": [], "error": "ConnectionResetError: [Errno 104] reset by 192.0.2.7:8000"}
    (run / dv.RELAY_LOG).write_text(json.dumps({"connections": [], "watermark": mark}))
    assert dv.watermark_log(run) == {"last": [3, 1], "lost": [], "error": "ConnectionResetError"}
    stop = dv.watermark_stop("honest", run, True)
    assert stop.startswith("honest: STOPPED at a watermark error (ConnectionResetError): ") and "192.0.2.7" not in stop
    mark |= {"error": "closed", "lost": [4]}
    (run / dv.RELAY_LOG).write_text(json.dumps({"connections": [], "watermark": mark}))
    assert dv.watermark_log(run) == {"last": [3, 1], "lost": [4], "error": "closed"}
    stop = dv.watermark_stop("honest", run, False)
    assert stop.startswith("honest: STOPPED at a lost release: ") and "; and at a watermark error (closed): " in stop
    assert dv.stop_words("honest", [], None) is None


def test_a_tail_shorter_than_a_connection_is_refused_before_any_try(tmp_path):
    served = ["--", "--slots", "1", "--bucket-ms", "250", "--delay", "60", "--window", "600"]
    args = dv.parser().parse_args(["try", "--stand-in", "--out", str(tmp_path), *served])
    assert dv.lane_seconds(args) == (60 + 16 + 2) * 0.25
    assert dv.lane_seconds(dv.parser().parse_args(["try", "--stand-in", "--out", str(tmp_path)])) == (30 + 8 + 2) * 0.1
    with pytest.raises(SystemExit, match="--tail 8: .* give --tail at least 20"):
        dv.main(["try", "--stand-in", "--out", str(tmp_path / "out"), *served])
    assert not (tmp_path / "out").exists()


def test_the_tie_is_not_run_without_the_commits_ids_and_the_row_says_so(tmp_path):
    kept = _kept(tmp_path, "tokens", [1, 0] * 60)
    row = dv.row(kept, None)
    assert (row["tie"], row["ok"]) == (dv.NOT_RUN, None) and "--ids" in row["words"]
    assert row["charged"] == "nothing, and the tie that refuses it wasn't run"
    table = ["table", "--out", str(tmp_path), "--class", "tokens"]
    assert dv.main(table) == 2 and dv.main([*table, "--stand-in"]) == 0
    rows = json.loads((tmp_path / "rows.json").read_text())
    assert [(r["class"], r["tie"], r["ok"]) for r in rows] == [("tokens", "refused", True)]
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

    tries = {"honest": _kept(tmp_path, "honest", [], users="honest-ab"),
             "tokens": _kept(tmp_path, "tokens", [1, 0] * 60, users="tokens-ab")}
    record = tmp_path / "requests.jsonl"
    record.write_text("".join(json.dumps({"user": f["user"], "output_ids": ids}) + "\n"
                              for users in ("honest-ab", "tokens-ab") for f, ids in zip(*_requests(users)[::2])))
    rows = {cls: dv.row(kept, dv.committed_ids(kept, dv.read_commit(record))) for cls, kept in tries.items()}
    assert (rows["honest"]["tie"], rows["tokens"]["tie"]) == ("accepted", "refused")
    assert all(r["ok"] for r in rows.values())
    assert _requests("honest-ab")[2] != _requests("tokens-ab")[2]
