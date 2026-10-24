"""warden-serve without the wall clock: its varied traffic, its HTTP bytes against the fake endpoint over loopback, and
the offline check of a run whose warden is driven tick by tick on a `VirtualClock`, its records certified by an LMS
key: the check accepts the run, and refuses a record that isn't what was certified, a cell that reached its client in
another bucket, and a response that isn't the record's."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from verity.core.protocols.certifier.active import DATA, VirtualClock
from verity.core.protocols.certifier.warden import END, Shim, Warden, WardenParams, cells, read_cell, reason_frame

HERE = Path(__file__).resolve().parents[1]
PARAMS = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=4, window=60, ingress_cells=40,
                      ingress_buckets=30, egress_cells=128)
SESSION = bytes(range(32))


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_warden_serve", HERE / "warden_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load()


async def _no_sleep(_: float) -> None:
    pass


def test_varied_traffic_is_fixed_by_its_seed_and_varies():
    a = ws.varied_traffic(200, model="m", rate=0.5, seed=3)
    assert a == ws.varied_traffic(200, model="m", rate=0.5, seed=3) != ws.varied_traffic(200, model="m", seed=4)
    assert all(x["at"] <= y["at"] for x, y in zip(a, a[1:]))
    bodies = [x["body"] for x in a]
    assert all(16 <= b["max_tokens"] <= 256 and b["stream"] and b["model"] == "m" for b in bodies)
    assert {b["temperature"] for b in bodies} == set(ws.TEMPERATURES)
    assert 0 < sum("seed" in b for b in bodies) < len(bodies)
    words = [len(b["prompt"].split()) for b in bodies]
    assert min(words) < 30 and max(words) > 500


def test_the_fake_endpoint_answers_a_request_as_vllm_streams_it():
    """Over loopback: the request's bytes reach the endpoint, which answers a chunked event stream and closes; the
    response parses to the endpoint's tokens, its finish reason and usage, and a cut response isn't complete."""
    body = ws.varied_traffic(1, model="m", seed=1)[0]["body"]

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(ws.http_request(body, f"127.0.0.1:{port}"))
        data = await reader.read()
        writer.close()
        server.close()
        return data

    data = asyncio.run(main())
    got = ws.parse_response(data)
    n = ws.fake_tokens(body)
    assert got | {"prompt_tokens": None} == {"status": 200, "complete": True, "events": n + 1, "text_bytes":
                                             len("".join(f" w{k}" for k in range(n))),
                                             "finish_reason": "length" if n == body["max_tokens"] else "stop",
                                             "prompt_tokens": None, "completion_tokens": n}
    assert not ws.parse_response(data[:-5])["complete"]
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n") == (b"abcde", True)
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nd") == (b"abcd", False)


class _Writer:
    def __init__(self, warden: Warden | None = None) -> None:
        self.warden, self.writes, self.closed, self.transport = warden, [], False, self

    def write(self, data: bytes) -> None:
        self.writes.append((self.warden.ticks if self.warden else None, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def _response(texts: list[str]) -> bytes:
    events = [{"choices": [{"index": 0, "text": t, "finish_reason": "stop" if i == len(texts) - 1 else None}]}
              for i, t in enumerate(texts)]
    events.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": len(texts)}})
    return (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            + b"".join(ws._sse(e) for e in events) + b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")


def _run(out: Path, key: Path, requests: list[bytes], responses: list[bytes]) -> None:
    """A run of the warden for one window on a `VirtualClock`, published as `serve` and `client` publish one: its
    records and certificates through `Publisher`, `deployment.json`, and `client.json` with every cell's arrival 1 ms
    into the bucket it was sent in."""
    tau = PARAMS.bucket_ms / 1000
    signer = ws.cs.signer(key, 4)
    publisher = ws.Publisher(out, ws.Issuer(signer, SESSION), PARAMS)
    answers = iter(responses)

    async def connect():
        reader = asyncio.StreamReader()
        reader.feed_data(next(answers))
        reader.feed_eof()
        return reader, _Writer()

    def settled(w: Warden) -> bool:
        return all(c.server is not None and c.ingress_ended and c.egress_ended for c in w.active)

    async def main() -> list[_Writer]:
        clock = VirtualClock(-tau / 2)
        warden = Warden(PARAMS, connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep, publish=publisher)
        running = asyncio.create_task(warden.run())
        clients, tasks = [], []
        for request in requests:
            reader = asyncio.StreamReader()
            reader.feed_data(cells(request, PARAMS.cell_bytes))
            reader.feed_eof()
            clients.append(_Writer(warden))
            tasks.append(asyncio.create_task(warden.handle(reader, clients[-1])))
        await warden.until(lambda w: w.ordinals == len(requests))
        for t in range(PARAMS.window):
            await warden.until(settled)
            clock.advance(t * tau)
            await warden.until(lambda w, t=t: w.ticks > t)
        await running
        await asyncio.gather(*tasks)
        return clients

    clients = asyncio.run(main())
    publisher.close()
    deployment = ws.deployment_record(PARAMS, signer, SESSION, origin_unix=0.0, windows=1, listen=("127.0.0.1", 1),
                                      upstream=("127.0.0.1", 2), slack=0.025)
    (out / "deployment.json").write_text(json.dumps(deployment))
    log = []
    for index, (request, client) in enumerate(zip(requests, clients)):
        shim = Shim(PARAMS.cell_bytes)
        B = PARAMS.cell_bytes
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        shim.outcome = "end" if shim.arrivals and END in [k for _, k in shim.arrivals] else "cut"
        log.append(ws.connection_log(index, 0.0, 0.0, request, shim, 0.0))
    (out / "client.json").write_text(json.dumps({"connections": log}))


def test_the_check_accepts_a_run_and_refuses_what_wasnt_certified_or_scheduled(tmp_path):
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    _run(out, key, requests, responses)
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["requests"], s["ended"], s["certified_windows"], s["completion_tokens"]) == (3, 3, 3, 42)
    assert s["cells_in_their_bucket"][0] == s["cells_in_their_bucket"][1] > 0
    assert s["wire_bits"]["egress"] == s["cells"]["egress"] * PARAMS.cell_bytes * 8
    assert s["data_bytes"] == {"ingress": sum(map(len, requests)), "egress": sum(map(len, responses))}
    assert "ACCEPTED: 3/3 ended" in ws.table(result)
    rows = PARAMS.window * PARAMS.lanes * PARAMS.slots
    assert {k: s["cell_rows"][k] for k in ("registered", "of", "rows")} == {"registered": 2, "of": 2, "rows": [rows]}
    assert s["cell_rows"]["commit_ms"]["n"] == s["cell_rows"]["sign_ms"]["n"] == 2

    def refused(edit) -> list[str]:
        saved = {p: p.read_bytes() for p in out.iterdir()}
        edit()
        try:
            return ws.check(out)["problems"]
        finally:
            for p, data in saved.items():
                p.write_bytes(data)

    def flip_record() -> None:
        records = ws.read_records(out, PARAMS)
        record = records["responses"][0]
        row = next(b for b, r in enumerate(record.grid) if any(f is not None and f[0] == DATA for f in r))
        frames = list(record.grid[row])
        k = next(i for i, f in enumerate(frames) if f is not None and f[0] == DATA)
        frames[k] = frames[k][:-1] + bytes([frames[k][-1] ^ 1])
        grid = record.grid[:row] + (tuple(frames),) + record.grid[row + 1:]
        (out / "responses.records").unlink()
        ws.ar.write_blobs(out / "responses.records", ws.ar.pack_record(ws.replace(record, grid=grid)))

    problems = refused(flip_record)
    assert "responses window 0 isn't certified" in problems and "responses window 0's cells aren't registered" in problems

    def edit_cells(change) -> None:
        lines = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
        salts = ws.ar.read_blobs(out / "cells.salts")
        change(lines, salts)
        (out / "cells.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (out / "cells.salts").unlink()
        for blob in salts:
            ws.ar.write_blobs(out / "cells.salts", blob)

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = bytes([salts[1][0] ^ 1]) + salts[1][1:]

    for change in (other_root, other_salt):
        assert "responses window 0's cells aren't registered" in refused(lambda: edit_cells(change))

    def forge_reason() -> None:
        (record,) = ws.read_records(out, PARAMS)["fails"]
        grid = ((reason_frame(0, "late", PARAMS.cell_bytes),) + record.grid[0][1:],) + record.grid[1:]
        (out / "fails.records").unlink()
        ws.ar.write_blobs(out / "fails.records", ws.ar.pack_record(ws.replace(record, grid=grid)))

    assert "fails window 0 isn't certified" in refused(forge_reason)

    def edit_client(change) -> None:
        doc = json.loads((out / "client.json").read_text())
        change(doc["connections"][1])
        (out / "client.json").write_text(json.dumps(doc))

    def late_cell(conn: dict) -> None:
        conn["arrivals_ms"][-1] += PARAMS.bucket_ms

    problems = refused(lambda: edit_client(late_cell))
    assert any(p.startswith("connection 1:") and "'in_bucket'" in p for p in problems)
    problems = refused(lambda: edit_client(lambda c: c.update(data_sha256="0" * 64)))
    assert any(p.startswith("connection 1:") and "'response_certified': False" in p for p in problems)
    assert ws.check(out)["accepted"]
