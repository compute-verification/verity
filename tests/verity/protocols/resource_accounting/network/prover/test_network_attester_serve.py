"""network-certifier-serve without the wall clock: its varied traffic, its HTTP bytes against the fake endpoint over loopback, and
the offline check of a run whose network attester is driven tick by tick on a `VirtualClock`, its records attested by
an LMS key: the check accepts the run, whose server got each request with its release order and bucket under the run
id `deployment.json` names, and refuses a deployment whose start isn't the one attested, a record that isn't what
was attested, a cell that reached its client in another bucket, and a response that isn't the record's; a window in
which the network attester refused a connection is
attested and signed nowhere; and a window's slow commit, which faults the next window when it runs in the network
attester's process, moves no tick in `serve`'s publisher process. The registrations' file a check of a run with
connection rows writes, as the stop verdicts' reader (`stop_verdicts`) reads it: each head it holds under the key
and the session the check was handed, none of a run under another key, and every class at its whole schedule when
a registration's signature is changed."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import socket
import sys
import types
from pathlib import Path

import pytest

from verity.protocols.resource_accounting.network.verifier.link_frame import DATA
from verity.protocols.resource_accounting.network.verifier.grid import Reason
from verity.protocols.resource_accounting.network.verifier.release import WATERMARK_END, watermark_chunk, watermark_head, with_release
from verity.protocols.resource_accounting.network.verifier.network_attester import (END, audit, cells, reason_frame)
from verity.protocols.resource_accounting.network.verifier import load_check as LK, relay_check as RC, record_files as RF
from verity.protocols.resource_accounting.network.attester import runner_parts as rp, network_attester_serve as ws
from tests.verity.protocols.resource_accounting.network.attester.relay_runs import (PARAMS, SESSION, SLOW_COMMIT_S, _response, _run, _session_argv, _Writer, refusing_commit, slow_commit)
from verity.protocols.resource_accounting.network.verifier import link_frame
from verity.protocols.resource_accounting.network.verifier import connection_rows as CR, connection_rows_check as CRC
from verity.protocols.resource_accounting.network.attester import connection_rows_run as crr
from tests.verity.protocols.resource_accounting.network.attester.test_connection_rows_run import (ROWS, command, device_key,
    edit_rows, handed, refused, served)
from tests.verity.protocols.resource_accounting.network.prover import verdict_records as TSV

#: the reader of the registrations' file, and its tests' records
SV = TSV.SV

ROOT = Path(__file__).resolve().parents[6]


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


def test_the_network_attester_always_paces_and_serve_and_rehearse_refuse_a_release_within_the_slack(tmp_path):
    """`serve` and `rehearse`, the deployed network_attester and the gate's rehearsal of it, always pace: `--paced-ms 0` (once the
    loop's write) and a release within the slack or past the next boundary are refused before anything runs."""
    serve = ["serve", "--upstream", "127.0.0.1:9", "--listen", "127.0.0.1:0", "--key", str(tmp_path / "key"),
             "--session", "00" * 32, "--windows", "1", "--out", str(tmp_path / "run")]
    for paced in ("0", "25", "100"):
        for argv in (serve, ["rehearse", "--fake", "--out", str(tmp_path / "run")]):
            with pytest.raises(SystemExit, match="the network attester always paces"):
                ws.main([*argv, "--paced-ms", paced])
    assert not (tmp_path / "run").exists()


def test_a_group_of_id_requests_admitted_at_a_windows_start_ends_within_its_group_windows(tmp_path):
    """The bound `group_windows` rests on, on the network attester itself at the scan's schedule: a group of `lanes` id
    requests admitted together at a window's first bucket, each answered with `tokens` id events and the trailer (the
    loop's two, and the gate's most, 64), ends every connection with its last cell D + ⌊tokens/r⌋ buckets after its
    admission, so that one admitted two buckets late (`prompt_traffic`'s half a bucket, and the client's lag) still
    ends within the group's G windows; the check accepts the run."""
    from verity.protocols.resource_accounting.network.prover import scan_zk as sz
    p = sz.PARAMS
    for tokens in (2, 64):
        G = ws.group_windows(p, tokens)
        requests = [ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=tokens,
                                  frame_bytes=p.cell_bytes, offset=0) for j in range(p.lanes)]
        out = tmp_path / f"t{tokens}"
        out.mkdir()
        responses = [ws.id_response(list(range(tokens)), p.payload)] * p.lanes
        network_attester = _run(out, tmp_path / f"key{tokens}", requests, responses, windows=G, params=p, id_payload=p.payload)
        assert len(network_attester.log) == p.lanes and all(c["failed"] is None and not c["cut"] for c in network_attester.log)
        assert {c["ended"] - c["admitted"] for c in network_attester.log} == {p.delay + tokens // p.slots}
        assert all(c["admitted"] + 2 + p.delay + tokens // p.slots < G * p.window for c in network_attester.log)
        result = RC.check(out)
        assert result["accepted"], result["problems"]
        assert (result["summary"]["ended"], result["summary"]["windows"]) == (p.lanes, G)
    assert (ws.group_windows(p, 2), ws.group_windows(p, 64)) == (1, 2)


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

    async def models() -> str:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await asyncio.to_thread(rp.served_model, f"http://127.0.0.1:{port}")
        finally:
            server.close()

    assert asyncio.run(models()) == "fake"
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n") == (b"abcde", True)
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nd") == (b"abcd", False)


def test_the_fake_endpoint_reads_the_watermark_stream_to_its_end_and_answers_nothing():
    """`rehearse --fake --watermark`: the network attester's watermark stream (`release`'s `POST /verity/watermark`)
    isn't a completion request, so the fake endpoint takes all of it, answers nothing and closes; `serve --watermark`
    and `rehearse --watermark` both take the flag."""
    stream = watermark_head(bytes(32)) + b"".join(watermark_chunk(t, n) for t, n in ((0, 0), (1, 2), (2, 2))) + \
        watermark_chunk(3, 4, [2]) + WATERMARK_END

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(stream)
        writer.write_eof()
        data = await reader.read()
        writer.close()
        server.close()
        return data

    assert asyncio.run(main()) == b""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    assert [ws.parser().parse_args(serve + extra).watermark for extra in ([], ["--watermark"])] == [False, True]
    assert ws.parser().parse_args(["rehearse", "--fake", "--out", "o", "--watermark"]).watermark


def test_serve_releases_the_id_form_unless_told_and_rehearse_names_its_traffics_form(tmp_path):
    """`serve` is the deployed network attester, so its egress form is the egress record's id form unless `--egress`
    names another; `rehearse` names its traffic's, id with `--ids` and bytes for its text, and passes it to its
    `serve`, whose deployment's digest binds it. A text response through the id form fails its connection at its first
    cell (format), and the check names it."""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    rehearse = ["rehearse", "--fake", "--out", "o"]
    forms = [ws._params(ws.parser().parse_args(argv)).egress for argv in
             (serve, serve + ["--egress", "bytes"], rehearse, rehearse + ["--ids"], rehearse + ["--egress", "id"])]
    assert forms == ["id", "bytes", "bytes", "id", "id"]
    for argv, form in ((rehearse, "bytes"), (rehearse + ["--ids"], "id")):
        flags = ws._flags(ws.parser().parse_args(argv))
        assert flags[flags.index("--egress") + 1] == form
        assert ws._params(ws.parser().parse_args(serve + flags)) == ws._params(ws.parser().parse_args(argv))
    with pytest.raises(SystemExit):
        ws.parser().parse_args(serve + ["--egress", "text"])

    out = tmp_path / "run"
    out.mkdir()
    gated = dataclasses.replace(PARAMS, egress="id")
    requests = [ws.http_request({"prompt": "x"}, "h:1")]
    network_attester = _run(out, tmp_path / "key", requests, [_response([" a"])], params=gated)
    assert [(c["failed"], c["failed_at"]) for c in network_attester.log] == [("format", 0)]
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["params"]["egress"] == "id" and deployment["digest"] == gated.digest.hex() != PARAMS.digest.hex()
    problems = RC.check(out)["problems"]
    assert any("connection 1 failed closed (format) at egress cell 0" in p for p in problems), problems


def test_a_response_cap_is_a_schedule_flag_rehearse_passes_to_serve():
    """`--max-tokens-cap` sets the schedule's response cap, which `rehearse` passes to its `serve` and the digest
    binds; without it the schedule has none and no flag carries it."""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    rehearse = ["rehearse", "--fake", "--out", "o", "--ids"]
    capped = ws._params(ws.parser().parse_args(rehearse + ["--max-tokens-cap", "7"]))
    assert capped.max_tokens_cap == 7 and capped.digest != ws._params(ws.parser().parse_args(rehearse)).digest
    flags = ws._flags(ws.parser().parse_args(rehearse + ["--max-tokens-cap", "7"]))
    assert flags[flags.index("--max-tokens-cap") + 1] == "7"
    assert ws._params(ws.parser().parse_args(serve + flags)) == capped
    assert "--max-tokens-cap" not in ws._flags(ws.parser().parse_args(rehearse))
    assert ws._params(ws.parser().parse_args(serve)).max_tokens_cap is None


def test_the_fake_endpoint_answers_an_id_request_id_form_as_the_scans_egress_reads_it():
    """`rehearse --ids`: an `id_traffic` request, sent network-certifier-form (`id_form.id_request`), is answered with
    its `fake_ids` as id events of P bytes and the trailer, every cell an id-form `DATA` cell, then `END`'s trailer;
    `parse_id_response` reads its count, and a cut or altered response isn't complete."""
    from verity.protocols.resource_accounting.network.prover import scan_zk as sz
    traffic = ws.id_traffic(20, model="m", seed=2, max_tokens=(1, sz.PARAMS.egress_cells - 1))
    assert traffic == ws.id_traffic(20, model="m", seed=2, max_tokens=(1, sz.PARAMS.egress_cells - 1))
    assert all(1 <= x["ids"]["max_tokens"] <= 64 and all(i < 1 << 17 for i in x["ids"]["prompt"]) for x in traffic)
    fields, P, B = traffic[0]["ids"], sz.PARAMS.payload, sz.PARAMS.cell_bytes
    request = ws.id_request(**fields, frame_bytes=B, offset=0)

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep, id_payload=P),
                                            "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(request)
        data = await reader.read()
        writer.close()
        server.close()
        return data

    data = asyncio.run(main())
    ids = ws.fake_ids(json.loads(request.partition(b"\r\n\r\n")[2]))
    assert data == ws.id_response(ids, P) and 1 <= len(ids) <= fields["max_tokens"]
    assert data[len(ids) * P:] == sz.IT.TRAILER
    assert [sz.IT.data_cell(x, 1, B)[sz.IT.HEADER.size:sz.IT.HEADER.size + P] for x in ids] == [
        data[i:i + P] for i in range(0, len(ids) * P, P)]
    got = ws.parse_id_response(data, P)
    assert (got["ids"], got["complete"], got["completion_tokens"]) == (len(ids), True, len(ids))
    assert not ws.parse_id_response(data[:-1], P)["complete"] and not ws.parse_id_response(data[:-len(sz.IT.TRAILER)],
                                                                                            P)["complete"]
    assert not ws.parse_id_response(data.replace(b"data:", b"datum", 1), P)["complete"]


def test_the_check_accepts_a_run_and_refuses_what_wasnt_certified_or_scheduled(tmp_path):
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    servers: list[_Writer] = []
    network_attester = _run(out, key, requests, responses, servers=servers)
    release = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in network_attester.log}
    assert sorted(n for n, _ in release.values()) == [0, 1, 2]
    assert [b"".join(d for _, d in s.writes) for s in servers] == \
        [with_release(r, release[k + 1][0], network_attester.run_id, release[k + 1][1]) for k, r in enumerate(requests)]
    assert json.loads((out / "deployment.json").read_text())["run_id"] == network_attester.run_id.hex()
    result = RC.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["requests"], s["ended"], s["certified_windows"], s["completion_tokens"]) == (3, 3, 3, 42)
    assert s["cells_in_their_bucket"][0] == s["cells_in_their_bucket"][1] > 0
    assert s["wire_bits"]["egress"] == s["cells"]["egress"] * PARAMS.cell_bytes * 8
    assert s["data_bytes"] == {"ingress": sum(map(len, requests)), "egress": sum(map(len, responses))}
    assert "ACCEPTED: 3/3 ended" in ws.table(result) and "start at 0 ms of the key's life" in ws.table(result)
    assert s["start"] == {"base_ms": 0, "certified": True}
    assert s["cell_rows"] == \
        {"registered": 2, "of": 2, "rows": [PARAMS.window], "row_bytes": [PARAMS.lanes * PARAMS.slots * PARAMS.cell_bytes]}
    assert s["cell_timing"]["commit_ms"]["n"] == s["cell_timing"]["sign_ms"]["n"] == 2
    assert all(sorted(json.loads(x)["cost"]) == ["row_bytes", "rows"]
               for x in (out / "cells.jsonl").read_text().splitlines())

    def refused(edit) -> list[str]:
        saved = {p: p.read_bytes() for p in out.iterdir()}
        edit()
        try:
            return RC.check(out)["problems"]
        finally:
            for p, data in saved.items():
                p.write_bytes(data)

    def flip_record() -> None:
        records = RC.read_records(out, PARAMS)
        record = records["responses"][0]
        row = next(b for b, r in enumerate(record.grid) if any(f is not None and f[0] == DATA for f in r))
        frames = list(record.grid[row])
        k = next(i for i, f in enumerate(frames) if f is not None and f[0] == DATA)
        frames[k] = frames[k][:-1] + bytes([frames[k][-1] ^ 1])
        grid = record.grid[:row] + (tuple(frames),) + record.grid[row + 1:]
        (out / "responses.records").unlink()
        ws.bf.write_blobs(out / "responses.records", RF.pack_record(ws.replace(record, grid=grid)))

    def edit_start(change) -> None:
        doc = json.loads((out / "deployment.json").read_text())
        change(doc)
        (out / "deployment.json").write_text(json.dumps(doc))

    assert refused(lambda: edit_start(lambda d: d.pop("start"))) == \
        ["the session's start isn't certified (no start's certificate)"]
    assert refused(lambda: edit_start(lambda d: d["start"].update(base_ms=PARAMS.window * PARAMS.bucket_ms))) == \
        ["the session's start isn't certified (opening)"]

    problems = refused(flip_record)
    assert "responses window 0 isn't certified" in problems and "responses window 0's cells aren't registered" in problems

    def edit_cells(change) -> None:
        lines = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
        salts = ws.bf.read_blobs(out / "cells.salts")
        change(lines, salts)
        (out / "cells.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (out / "cells.salts").unlink()
        for blob in salts:
            ws.bf.write_blobs(out / "cells.salts", blob)

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = bytes([salts[1][0] ^ 1]) + salts[1][1:]

    def other_wiring_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])

    for change in (other_root, other_salt, other_wiring_salt):
        assert "responses window 0's cells aren't registered" in refused(lambda: edit_cells(change))
    for k in RC.TIMINGS:
        problems = refused(lambda k=k: edit_cells(lambda lines, salts: lines[1]["cost"].update({k: 1.0})))
        assert (f"responses window 0's registration carries the device's {k}: its timings go in {ws.CELLS_TIMING}"
                in problems and "responses window 0's cells aren't registered" in problems)

    def address(k: str) -> None:
        dep = json.loads((out / "deployment.json").read_text())
        (out / "deployment.json").write_text(json.dumps(dep | {k: ["127.0.0.1", 1]}))

    for k in RC.ADDRESSES:
        assert (f"the deployment holds {k}, an address: it goes in {ws.ENDPOINTS}, which is never published"
                in refused(lambda k=k: address(k)))

    def forge_reason() -> None:
        (record,) = RC.read_records(out, PARAMS)["fails"]
        grid = ((reason_frame(0, "late", PARAMS.cell_bytes),) + record.grid[0][1:],) + record.grid[1:]
        (out / "fails.records").unlink()
        ws.bf.write_blobs(out / "fails.records", RF.pack_record(ws.replace(record, grid=grid)))

    assert "fails window 0 isn't certified" in refused(forge_reason)

    def edit_client(change) -> None:
        doc = json.loads((out / "client.json").read_text())
        change(doc["connections"][1])
        (out / "client.json").write_text(json.dumps(doc))

    def late_cell(conn: dict) -> None:
        conn["arrivals_ms"][-1] += PARAMS.bucket_ms

    problems = refused(lambda: edit_client(late_cell))
    assert any(p.startswith("connection 2:") and "'in_bucket'" in p for p in problems)
    problems = refused(lambda: edit_client(lambda c: c.update(data_sha256="0" * 64)))
    assert any(p.startswith("connection 2:") and "'response_certified': False" in p for p in problems)
    assert RC.check(out)["accepted"]


def test_the_client_connects_where_the_endpoints_file_says_and_names_its_upstream_as_the_host(tmp_path):
    """`client` takes the listener and the endpoint from `ENDPOINTS` beside the deployment, which holds neither: its
    request reaches that listener as cells, with the endpoint as its host."""
    body = {"prompt": "x", "stream": True}
    got = []

    async def main() -> dict:
        async def take(reader, writer) -> None:
            got.append(await reader.readexactly(len(want)))
            writer.close()

        server = await asyncio.start_server(take, "127.0.0.1", 0)
        ends = ws.endpoints_record(server.sockets[0].getsockname()[:2], ("127.0.0.1", 2))
        (tmp_path / ws.ENDPOINTS).write_text(json.dumps(ends))
        (tmp_path / "deployment.json").write_text(json.dumps({"params": dataclasses.asdict(PARAMS), "origin_unix": 0.0,
                                                              "origin_raw": 0.0}))
        dep = json.loads((tmp_path / "deployment.json").read_text())
        async with server:
            return await ws.client(dep, ws.read_endpoints(tmp_path / "deployment.json"), [{"at": 0.0, "body": body}],
                                   tmp_path / "out", clock=lambda: 0.0)

    want = cells(ws.http_request(body, "127.0.0.1:2"), PARAMS.cell_bytes)
    doc = asyncio.run(main())
    assert got == [want] and "error" not in doc["connections"][0]


@pytest.mark.parametrize("process", [False, True])
def test_a_window_with_a_refused_connection_is_certified_nowhere(tmp_path, process):
    """Five connections at once on two lanes and a queue of four: the fifth is refused, traffic past the window's
    Program slots, so window 0's three records are kept and its two rows committed but none is attested or signed,
    each line naming the execution phase under `refused`, withheld, inline and in `serve`'s publisher process; window 1
    is attested whole, and the check refuses the run on window 0 alone."""
    out = tmp_path / "run"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 20, 30, 40, 50)]
    responses = [_response([" a"]) for _ in range(4)]
    network_attester = _run(out, tmp_path / "key", requests, responses, windows=2, process=process)
    assert (network_attester.refused, network_attester.refused_windows, len(network_attester.log)) == (1, {0}, 4)
    certified = [json.loads(x) for x in (out / "certified.jsonl").read_text().splitlines()]
    cells_ = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
    assert [(x["refused"][2], x["withheld"]) for x in certified if "refused" in x] == [(0, True)] * 3
    assert [RF.certified_from_json(x).window for x in certified if "refused" not in x] == [1] * 3
    assert [("refused" in x, x.get("withheld")) for x in cells_] == [(True, True)] * 2 + [(False, None)] * 2
    result = RC.check(out)
    s = result["summary"]
    assert (s["refused"], s["ended"], s["certified_windows"], s["cell_rows"]["registered"]) == (1, 4, 3, 2)
    assert {p for p in result["problems"] if "window" in p} == \
        {f"{name} window 0 isn't certified" for name in RC.CERTIFIED} | {
            f"{name} window 0's cells aren't registered" for name in ("requests", "responses")}


def test_a_phase_with_no_certificate_or_no_registration_is_a_problem_and_one_fewer_in_its_count(tmp_path):
    """The gate's end holds three of the check's counts only when it is accepted, on the ground that each is then
    the schedule's (`auditor.ACCEPTED_COUNTS`, `auditor.ACCEPTED_CELL_ROWS`): how many records were attested, by
    how many attestations, and how many of the cells' registrations the check accepted. That stands on the check
    refusing a run with one missing. A run whose every execution phase is attested and registered is accepted,
    each count the schedule's. The same run less one record's attestation, for each record and execution phase, or
    less one cells' registration, is refused on that execution phase by name, with its own count one fewer and the
    others as they were."""
    out, windows = tmp_path / "run", 2
    out.mkdir()
    _run(out, tmp_path / "key", [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200)],
         [_response([" a", " b"]), _response([" word"] * 40)], windows=windows)

    def counts(result: dict) -> tuple[int, int, int]:
        s = result["summary"]
        return s["certified_windows"], s["certificates"], s["cell_rows"]["registered"]

    whole = ws.check(out)
    assert whole["accepted"], whole["problems"]
    assert whole["summary"]["windows"] == windows and whole["summary"]["cell_rows"]["of"] == 2 * windows
    assert counts(whole) == (3 * windows, 3 * windows, 2 * windows)
    names = {(link, direction.value): name for name, (link, direction) in ws.CERTIFIED.items()}

    certified = (out / "certified.jsonl").read_text().splitlines()
    assert len(certified) == 3 * windows
    for i, line in enumerate(certified):
        record = RF.certified_from_json(json.loads(line))
        (out / "certified.jsonl").write_text("".join(x + "\n" for x in certified[:i] + certified[i + 1:]))
        got = ws.check(out)
        assert not got["accepted"]
        assert f"{names[(record.link, record.direction.value)]} window {record.window} isn't certified" in got["problems"]
        assert counts(got) == (3 * windows - 1, 3 * windows - 1, 2 * windows)
    (out / "certified.jsonl").write_text("".join(x + "\n" for x in certified))

    cells, salts = (out / "cells.jsonl").read_text().splitlines(), ws.bf.read_blobs(out / "cells.salts")
    assert len(cells) == len(salts) == 2 * windows
    for i, line in enumerate(cells):
        r = json.loads(line)["registration"]
        (out / "cells.jsonl").write_text("".join(x + "\n" for x in cells[:i] + cells[i + 1:]))
        (out / "cells.salts").unlink()
        for blob in salts[:i] + salts[i + 1:]:
            ws.bf.write_blobs(out / "cells.salts", blob)
        got = ws.check(out)
        assert not got["accepted"]
        assert f"{names[(r['link'], r['direction'])]} window {r['window']}'s cells aren't registered" in got["problems"]
        assert counts(got) == (3 * windows, 3 * windows, 2 * windows - 1)


def _held_listener() -> socket.socket:
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    return held


def _session(tmp_path: Path, windows: int):
    """`_session_argv` parsed, with its schedule, and a maker of `n` id requests of the class's most tokens with their
    id-form answers."""
    args = ws.parser().parse_args(_session_argv(tmp_path / "run", windows))
    p = ws._params(args)
    tokens = p.egress_cells - 1

    def load(n: int) -> tuple[list[bytes], list[bytes]]:
        return ([ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=tokens,
                               frame_bytes=p.cell_bytes, offset=0) for j in range(n)],
                [ws.id_response(list(range(tokens)), p.payload)] * n)
    return args, p, load


def test_a_session_serves_its_length_to_clients_not_its_own_and_its_check_accepts_the_records_alone(
        tmp_path, monkeypatch):
    """`session` as the gate's `relay-session` runs it: one `serve` at the gate's listener for the request's
    execution phases and no other process, so no traffic and no client of its own. In `serve`'s place here, the
    relay on a virtual clock takes a load from another party's clients, eight connections arriving across the
    session's three execution phases, the last at the last bucket that still ends inside the run, and their log
    stays with them. The run's own check, on the records alone, accepts: every execution phase attested and
    registered and the audit, counting no connection. The clients' check is theirs, from their own log against the
    run's records: every connection ended, its request and its answer the audited connection's, each cell of the
    egress schedule."""
    args, p, load = _session(tmp_path, 3)
    last = args.windows * p.window - p.span - 1
    arrive = [0, 0, 7, p.window - 1, p.window, p.window + 8, 2 * p.window, last]
    requests, responses = load(len(arrive))
    served, own = [], {}

    class Serve:
        """`serve` as a process: the relay for its `--windows` on a virtual clock, its clients' log kept apart."""

        def __init__(self, argv, **kw):
            a = ws.parser().parse_args(argv[2:])
            assert a.command == "serve" and ws._params(a) == p
            served.append((a, _run(a.out, a.key, requests, responses, windows=a.windows, params=p,
                                   id_payload=p.payload, arrive=dict(enumerate(arrive)))))
            own.update(json.loads((a.out / "client.json").read_text()))
            (a.out / "client.json").unlink()
            self.stdout = io.StringIO(json.dumps({"listening": list(a.listen)}) + "\n")

        def wait(self) -> int:
            return 0

        poll = wait

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=Serve, PIPE=None))
    assert ws.rehearse(args) == 0
    ((a, relay),) = served
    out = args.out
    assert (a.listen, a.upstream, a.windows) == (("127.0.0.1", 40002), ("127.0.0.1", 40001), 3)
    assert {e["admitted"] // p.window for e in relay.log} == {0, 1, 2} and relay.log[-1]["admitted"] == last
    assert not {"traffic.json", "client.json", "client.log", "groups.json"} & {f.name for f in out.iterdir()}
    result = json.loads((out / "check.json").read_text())
    assert result["accepted"] and result["problems"] == [] and result["connections"] == []
    assert (result["model"], result["exit"]) == (None, {"serve": 0, "client": None})
    s = result["summary"]
    assert (s["requests"], s["ended"], s["cells_in_their_bucket"], s["windows"], s["certified_windows"]) == \
        (0, 0, [0, 0], 3, 9)
    assert (s["cell_rows"]["registered"], s["cell_rows"]["of"]) == (6, 6)
    assert s["data_bytes"] == {"ingress": sum(map(len, requests)), "egress": sum(map(len, responses))}

    records = RC.read_records(out, p)
    audited = {c["ordinal"]: c for c in
               audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])["connections"]}
    schedule = RC.egress_cells(records[RC.EGRESS], p)
    assert len(own["connections"]) == len(audited) == len(arrive)
    for conn in own["connections"]:
        o = conn["ordinal"]
        assert conn["outcome"] == "end" and audited[o]["ended"] is not None, o
        assert (audited[o]["ingress_sha256"], audited[o]["egress_sha256"]) == \
            (conn["request_sha256"], conn["data_sha256"]), o
        assert "".join(RC.KIND_CHARS[k] for _, k in schedule[o]) == conn["kinds"], o
    (out / "client.json").write_text(json.dumps(own))
    theirs = RC.check(out)
    assert theirs["accepted"] and (theirs["summary"]["requests"], theirs["summary"]["ended"]) == (8, 8)
    assert theirs["summary"]["cells_in_their_bucket"][0] == theirs["summary"]["cells_in_their_bucket"][1] > 0


def test_a_session_given_its_callers_listener_hands_it_to_serve_which_serves_on_it_and_leaves_it_listening(
        tmp_path, monkeypatch):
    """The gate's unit holds a session's listener and gives the runner its descriptor (`--listen-fd`). `session`
    passes that descriptor on to `serve`'s process and names it there, and without one passes none. The relay's
    server on a caller's socket binds nothing: a connection made to the caller's listener before the server starts
    is the one its handler gets, and once the server has closed the caller's socket still listens, so a later
    connection arrives there with nothing accepting it."""
    class Started(Exception):
        pass

    def popen(argv, **kw):
        raise Started(argv, kw)

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=popen, PIPE=None))
    for fd in (None, 9):
        with pytest.raises(Started) as started:
            ws.rehearse(ws.parser().parse_args(_session_argv(tmp_path / f"run-{fd}", 1, listen_fd=fd)))
        serve, kw = started.value.args
        a = ws.parser().parse_args(serve[2:])
        assert (a.command, a.listen, a.listen_fd) == ("serve", ("127.0.0.1", 40002), fd)
        assert kw["pass_fds"] == (() if fd is None else (fd,))

    held = _held_listener()
    early = socket.create_connection(held.getsockname())
    early.sendall(b"early")

    async def relay() -> list[bytes]:
        got, done = [], asyncio.Event()

        async def handle(reader, writer):
            got.append(await reader.readexactly(5))
            writer.write(b"served")
            await writer.drain()
            writer.close()
            done.set()

        server = await ws.listening(handle, held.getsockname(), socket.socket(fileno=os.dup(held.fileno())))
        assert [s.getsockname() for s in server.sockets] == [held.getsockname()]
        await done.wait()
        server.close()
        await server.wait_closed()
        return got

    assert asyncio.run(relay()) == [b"early"] and early.recv(6) == b"served"
    with socket.create_connection(held.getsockname()) as late:
        late.sendall(b"late")
        conn, _ = held.accept()
        assert conn.recv(4) == b"late"
        conn.close()
    early.close()
    held.close()


def test_serve_through_its_command_line_serves_on_the_socket_its_descriptor_names_and_binds_none(
        tmp_path, monkeypatch, capsys):
    """`serve --listen-fd FD` through `main` and `serve` themselves, on the command `session` issues for a listener
    its caller holds: `main` makes the socket FD names, and `serve` gives it to the relay's server. The relay, its
    publisher, the stall probe and the process's preparation are stood in for, so nothing ticks: the relay here ends
    once its handler has answered one connection. That connection is one made to the caller's listener before
    `serve` started, the deployment and the line `serve` prints name the caller's port, and the caller's socket
    still listens once `serve` has returned. With either link cut, the flag to the socket or the socket to the
    server, `serve` binds `--listen`, which is the caller's own address and is taken."""
    args, _, _ = _session(tmp_path, 1)
    held = _held_listener()
    args.listen, args.listen_fd = held.getsockname(), os.dup(held.fileno())

    class Started(Exception):
        pass

    def popen(argv, **kw):
        raise Started(argv, kw)

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=popen, PIPE=None))
    with pytest.raises(Started) as started:
        ws.rehearse(args)
    serve, kw = started.value.args
    assert kw["pass_fds"] == (args.listen_fd,)
    key = tmp_path / "key"
    ws.make_key(key, 4)
    serve[serve.index("--key") + 1] = str(key)

    public, got = ws.registered_key(key), []

    class Publisher:
        """`PublisherProcess` with no process: the registered key, no start's attestation, and nothing published."""

        usage = feeder = start = None
        process = types.SimpleNamespace(pid=None)

        def __init__(self, *a, **kw) -> None:
            self.key = public

        def __call__(self, *a) -> None:
            pass

        withhold = release = lengths = __call__

        def close(self) -> list:
            return []

    class Relay:
        """The relay's stand-in: its run ends once its handler has answered one connection."""

        run_id, lateness, log, ordinals, refused, refused_windows, ticks, releases = bytes(16), [], [], 0, 0, set(), 0, []
        turned_away, turned_away_reset = 0, 0

        def __init__(self, *a, **kw) -> None:
            self.answered = asyncio.Event()

        async def handle(self, reader, writer) -> None:
            got.append(await reader.readexactly(5))
            writer.write(b"served")
            await writer.drain()
            writer.close()
            self.answered.set()

        async def run(self) -> None:
            await self.answered.wait()

    monkeypatch.setattr(ws, "PublisherProcess", Publisher)
    monkeypatch.setattr(ws, "NetworkAttester", Relay)
    monkeypatch.setattr(ws, "prepare_process", lambda params: {})
    monkeypatch.setattr(ws, "stall_probe", types.SimpleNamespace(start=lambda *a: None, stop=lambda *a: None))
    monkeypatch.setattr(ws, "_quiet_runtime", lambda: None)
    for name in ("PYTHONMALLOC", ws.C_ALLOCATOR):
        monkeypatch.delenv(name, raising=False)
    early = socket.create_connection(held.getsockname())
    early.sendall(b"early")
    capsys.readouterr()
    assert ws.main(serve[2:]) == 0
    assert got == [b"early"] and early.recv(6) == b"served"
    a = ws.parser().parse_args(serve[2:])
    said = json.loads(capsys.readouterr().out.splitlines()[0])
    assert said["listening"] == list(held.getsockname())
    assert json.loads((a.out / ws.ENDPOINTS).read_text())["listen"] == list(held.getsockname())
    assert not set(RC.ADDRESSES) & set(json.loads((a.out / "deployment.json").read_text()))
    with socket.create_connection(held.getsockname()) as late:
        late.sendall(b"late")
        conn, _ = held.accept()
        assert conn.recv(4) == b"late"
        conn.close()
    early.close()
    held.close()


def test_a_connection_still_open_when_a_session_ends_is_cut_and_the_records_say_which(tmp_path):
    """A session stops at its last execution phase's end and cuts what is open, and the records say what each was. One
    admitted at bucket s with s + D inside the run and no room to end (the first bucket past windows · T − span − 1)
    has data and no `END` in the record: the audit refuses it in its admission's execution phase, and the check the
    run, every execution phase still attested. Ones admitted later are cut before s + D: their requests are whole in
    the ingress record, their egress is padding, the audit takes them and the check accepts, counting them as not
    ended. One still in the queue then (it arrived behind four that hold the lanes) got no cell and is in no
    record. The clients' own logs say `cut` for each."""
    args, p, load = _session(tmp_path, 2)
    end = args.windows * p.window
    assert (p.lanes, p.queue, p.delay, p.span, end) == (4, 4, 27, 28, 64)

    def run(name: str, arrive: list[int]):
        out = tmp_path / name
        out.mkdir()
        relay = _run(out, tmp_path / f"key-{name}", *load(len(arrive)), windows=args.windows, params=p,
                     id_payload=p.payload, arrive=dict(enumerate(arrive)))
        own = json.loads((out / "client.json").read_text())["connections"]
        (out / "client.json").unlink()
        records = RC.read_records(out, p)
        return relay, own, RC.check(out), audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])

    relay, own, result, audit_ = run("ends", [end - p.span - 1])
    assert result["accepted"] and relay.log[0]["ended"] == end - 1 and not relay.log[0]["cut"]
    assert [c["outcome"] for c in own] == ["end"] and audit_["connections"][0]["ended"] == end - 1

    relay, own, result, audit_ = run("late", [end - p.span])
    assert relay.log[0]["cut"] and relay.log[0]["ended"] is None and [c["outcome"] for c in own] == ["cut"]
    assert own[0]["kinds"].count(RC.KIND_CHARS[DATA]) == p.slots and RC.KIND_CHARS[END] not in own[0]["kinds"]
    assert not result["accepted"] and result["problems"] == [
        "audit: connection 1: its egress isn't padding to s + D, data in every slot, one END or FAIL, padding"]
    assert [(w["window"], w["accepted"]) for w in audit_["windows"]] == [(0, True), (1, False)]
    assert result["summary"]["certified_windows"] == 6 and all(w["certified"] for w in result["windows"])

    early = end - p.delay + 3
    relay, own, result, audit_ = run("early", [early] * p.lanes + [early + 1])
    assert result["accepted"] and result["problems"] == [] and audit_["accepted"]
    assert [(e["admitted"], e["ended"], e["cut"], e["ingress_cells"], e["egress_cells"]) for e in relay.log] == \
        [(early, None, True, 1, 0)] * p.lanes + [(None, None, True, 0, 0)]
    assert [(c["ordinal"], c["admitted"], c["ended"], c["egress_bytes"]) for c in audit_["connections"]] == \
        [(o, early, None, 0) for o in range(1, p.lanes + 1)]
    assert all(c["ingress_bytes"] > 0 for c in audit_["connections"])
    assert [c["outcome"] for c in own] == ["cut"] * (p.lanes + 1)
    assert [set(c["kinds"]) for c in own] == [{RC.KIND_CHARS[link_frame.PAD]}] * p.lanes + [set()]
    assert result["summary"]["data_bytes"] == {"ingress": sum(map(len, load(p.lanes)[0])), "egress": 0}
    assert result["summary"]["certified_windows"] == 6


def test_one_connection_failed_closed_refuses_a_sessions_whole_run_whichever_client_it_was(tmp_path):
    """The verifier's verdict is the whole session's. Of four connections, three end and one's answer is out of the id
    form, which the relay fails closed: every execution phase is still attested, the audit refuses the one the
    failure is in and accepts the other, and the check refuses the run, whoever's the three that ended were."""
    args, p, load = _session(tmp_path, 2)
    requests, responses = load(4)
    out = tmp_path / "run"
    out.mkdir()
    relay = _run(out, tmp_path / "key", requests, responses[:3] + [_response([" a"])], windows=args.windows,
                 params=p, id_payload=p.payload, arrive=dict(enumerate([0, 0, 1, 2])))
    (out / "client.json").unlink()
    records = RC.read_records(out, p)
    audit_ = audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])
    result = RC.check(out)
    assert sorted(e["failed"] or "" for e in relay.log) == ["", "", "", "format"]
    assert [(c["failed"], c["ended"] is not None) for c in audit_["connections"]] == \
        [(None, True)] * 3 + [("format", True)]
    assert sorted(w["accepted"] for w in audit_["windows"]) == [False, True]
    assert not result["accepted"] and all(w["certified"] for w in result["windows"])
    assert result["problems"] == ["audit: connection 4 failed closed (format) at egress cell 0"]


def test_the_summarys_failed_list_holds_the_checks_fixed_words_and_none_of_a_clients(tmp_path):
    """The summary's `failed` says how the connections that neither ended nor were refused failed, in `FAILED`'s
    words alone: the gate's end carries the list, and a word in it is a word a key with no name reads. The connection
    this run's relay failed closed is in it by its client's outcome, `cut`. A client's connection that raised has its
    exception's words on its own line of `client.json`; the check's problems keep them, and so does the connection's
    row, and the summary says `error`."""
    args, p, load = _session(tmp_path, 2)
    requests, responses = load(4)
    out = tmp_path / "run"
    out.mkdir()
    _run(out, tmp_path / "key", requests, responses[:3] + [_response([" a"])], windows=args.windows, params=p,
         id_payload=p.payload, arrive=dict(enumerate([0, 0, 1, 2])))
    assert ws.check(out)["summary"]["failed"] == ["cut"]
    client = json.loads((out / "client.json").read_text())
    said = "ConnectionResetError: the endpoint hung up on the first request"
    index = client["connections"][0]["index"]
    client["connections"][0] = {"index": index, "error": said}
    (out / "client.json").write_text(json.dumps(client))
    result = ws.check(out)
    assert result["summary"]["failed"] == ["cut", "error"] and said not in json.dumps(result["summary"])
    assert f"client connection {index}: {said}" in result["problems"]
    assert {"index": index, "outcome": said} in result["connections"]
    assert RC.failed_words(["cut", said, "cut", None, 3]) == ["cut", "error"] and RC.failed_words(said) == ["error"]
    assert RC.failed_words([]) == [] and set(RC.failed_words(RC.FAILED)) == set(RC.FAILED)


def test_the_check_names_each_execution_phase_the_audit_refused_and_why_uncut(tmp_path, monkeypatch):
    """`check.json`'s `audit_windows` is the audit's verdict of each execution phase, whole. Of a session's two, the
    second, where a connection fails closed, is refused with that problem and the first accepted, while `problems`
    gives the problem's text without its execution phase; a run the audit accepts has each one accepted, with no
    problem. And where the audit has more problems than the 50 `problems` keeps, `audit_windows` has them all."""
    args, p, load = _session(tmp_path, 2)
    requests, responses = load(4)

    def run(name: str, answers: list[bytes]) -> dict:
        out = tmp_path / name
        out.mkdir()
        _run(out, tmp_path / f"key-{name}", requests, answers, windows=args.windows, params=p, id_payload=p.payload,
             arrive=dict(enumerate([0, 0, 1, p.window])))
        (out / "client.json").unlink()
        return RC.check(out)

    result = run("ends", responses)
    assert result["accepted"] and result["audit_windows"] == [
        {"window": 0, "accepted": True, "problems": []}, {"window": 1, "accepted": True, "problems": []}]

    result = run("fails", responses[:3] + [_response([" a"])])
    why = "connection 4 failed closed (format) at egress cell 0"
    assert not result["accepted"] and result["problems"] == [f"audit: {why}"]
    assert result["audit_windows"] == [
        {"window": 0, "accepted": True, "problems": []}, {"window": 1, "accepted": False, "problems": [why]}]

    def many(*records) -> dict:
        verdict = audit(*records)
        verdict["windows"][1]["problems"] += [f"problem {i}" for i in range(60)]
        return verdict

    monkeypatch.setattr(RC, "audit", many)
    result = RC.check(tmp_path / "fails")
    assert len(result["problems"]) == 50 and result["problems"][:2] == [f"audit: {why}", "audit: problem 0"]
    assert result["audit_windows"][1]["problems"] == [why] + [f"problem {i}" for i in range(60)]


def test_a_windows_commit_and_signatures_move_no_tick_past_its_slack_in_the_publisher_process(tmp_path):
    """A window's commit that takes `SLOW_COMMIT_S` of its process's time (`slow_commit`), run in the network attester's process
    as the inline `Publisher` runs it, puts the ticks after the window's end past their slack: the next window faults
    (`monitor-unavailable`), the issuer certifies none of its records, and the check refuses it. In `serve`'s
    `PublisherProcess` the same commit runs in the spawned process, which signs and commits from the rows released to
    it: every tick is on time, and the check accepts every execution phase, attested and registered. A commit that fails
    there fails `close`, and the network attester's process doesn't wait on it."""
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    runs = {name: tmp_path / name for name in ("inline", "process", "refusing")}
    for out in runs.values():
        out.mkdir()

    inline = _run(runs["inline"], tmp_path / "inline-key", requests, responses, windows=2, commit=slow_commit)
    late = [t for t, x in enumerate(inline.lateness) if x > inline.slack]
    assert late[0] == PARAMS.window and max(inline.lateness) > SLOW_COMMIT_S
    result = RC.check(runs["inline"])
    assert {f"{name} window 1 isn't certified" for name in RC.CERTIFIED} <= set(result["problems"])
    assert {(x["record"], x["status"]) for x in result["windows"] if x["window"] == 1} == \
        {(name, Reason.MONITOR_UNAVAILABLE.value) for name in RC.CERTIFIED}
    assert sorted(w for *_, w in result["summary"]["signer_refused"]) == [1] * len(RC.CERTIFIED)

    process = _run(runs["process"], tmp_path / "process-key", requests, responses, windows=2, commit=slow_commit,
                   process=True)
    assert len(process.lateness) == 2 * PARAMS.window and max(process.lateness) == 0
    result = RC.check(runs["process"])
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["certified_windows"], s["cell_rows"]["registered"], s["ended"], s["signer_refused"]) == (6, 4, 3, [])

    with pytest.raises(RuntimeError, match="a refusing commit"):
        _run(runs["refusing"], tmp_path / "refusing-key", requests, responses, commit=refusing_commit, process=True)


def test_the_load_is_signed_by_the_serving_publisher_in_the_runs_session_before_its_windows(tmp_path):
    """The gate's load end to end at the tiny shape: the store relayed over four streams to the prover's sink on
    loopback (`load_relay.py sink`, a process), then a run whose `PublisherProcess`, the network attester's one signer,
    signs the relay's heads first: `load.json` is under the deployment's key and session, on the key's first leaves, and
    `check` reads every value's registration at the relay's roots; another session's, values swapped, heads that
    aren't the relay's and an unsigned load are refused."""
    import hashlib
    import subprocess

    store, out = tmp_path / "store", tmp_path / "run"
    store.mkdir()
    out.mkdir()
    lr, program = ws.lr, hashlib.sha512(b"the reading program").hexdigest()
    lr.write_store(store, LK.TINY)
    sink = subprocess.Popen([sys.executable, str(ROOT / "verity/protocols/resource_accounting/network/attester/load_relay.py"), "sink",
                             "--listen", "127.0.0.1:0",
                             "--weights", str(tmp_path / "weights.bin"), "--witness", str(tmp_path / "witness.bin")],
                            stdout=subprocess.PIPE, text=True)
    port = json.loads(sink.stdout.readline())["listening"][1]
    lr.relay_load(store, LK.TINY, [("127.0.0.1", port)] * 4, program, out)
    assert json.loads(sink.communicate()[0].splitlines()[-1])["accepted"] and sink.returncode == 0
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 30)]
    _run(out, tmp_path / "key", requests, [_response([" a", " b"]), _response([])], process=True,
         load_heads=out / "load-heads.json")
    result = RC.check(out)
    assert result["accepted"], result["problems"]
    n = len(LK.manifest(LK.TINY))
    assert result["summary"]["load"] == {"values": n, "registered": n, "relayed": True}
    assert ws.table(result).startswith(f"ACCEPTED: load {n}/{n} values registered; 2/2 ended")
    doc, dep = (json.loads((out / f).read_text()) for f in ("load.json", "deployment.json"))
    assert (doc["key"], doc["session"]) == (dep["key"], dep["session"]) and doc["session"] == SESSION.hex()
    leaf = lambda c: int(c["signature"][:8], 16)  # noqa: E731
    windows = [leaf(json.loads(x)["certificate"]) for x in (out / "certified.jsonl").read_text().splitlines()]
    assert sorted(leaf(r["certificate"]) for r in doc["values"]) == list(range(n)) and min(windows) >= n

    def problems(name: str, edit) -> list[str]:
        saved = (out / name).read_text()
        edit(out / name)
        try:
            return RC.check(out)["problems"]
        finally:
            (out / name).write_text(saved)

    def rewrite(change):
        return lambda p: p.write_text(json.dumps(change(json.loads(p.read_text()))))

    assert "load: not signed under the deployment's key and session" in \
        problems("load.json", rewrite(lambda d: d | {"session": "00" * 32}))
    swapped = problems("load.json", rewrite(lambda d: d | {"values": [d["values"][1], d["values"][0], *d["values"][2:]]}))
    assert sum(p.startswith("load tok: ") or p.startswith("load pos: ") for p in swapped) == 2
    assert "load: its registrations aren't the relay's heads (load-heads.json)" in \
        problems("load-heads.json", rewrite(lambda d: d | {"roots": {**d["roots"], "tok": "00" * 64}}))
    assert problems("load.json", lambda p: p.unlink()) == ["load: the relay's heads aren't signed (no load.json)"]


def test_reserve_fds_grows_the_descriptor_table_and_leaves_every_descriptor_as_it_was():
    """`serve`'s process holds a table for every connection before its origin (`prepare_process`), so that no tick
    waits on the kernel growing it (`reserve_fds`); the table's size is `FDSize` in /proc/self/status."""
    def table() -> int:
        status = Path("/proc/self/status").read_text().splitlines()
        return int(next(x.split()[1] for x in status if x.startswith("FDSize:")))

    held = sorted(os.listdir("/proc/self/fd"))
    n = ws.reserve_fds(table() + 1)
    assert table() >= n and sorted(os.listdir("/proc/self/fd")) == held


def test_a_usage_record_holds_its_cpus_as_one_string_of_ranges():
    """`cpu_ranges` writes CPU indices in the kernel's list form, and a process's usage record holds its affinity so:
    a string, never a list of integers."""
    assert [rp.cpu_ranges(c) for c in ([], [5], [3, 0, 1, 2, 8, 10, 11], range(120))] == ["", "5", "0-3,8,10-11", "0-119"]
    usage = rp._usage()
    assert isinstance(usage["cpus"], str) and usage["cpus"] == rp.cpu_ranges(os.sched_getaffinity(0))


def answer(tmp, file, registrations: list[dict]) -> SV.Stop:
    """The reader's answer (`stop_verdicts.of_runs`) on a run that holds the registrations' file as the check left
    it at `file`, or none where it left none, with the four verdicts of every class of each of `registrations`'
    heads, their verifiers' controls and the serve's registrations."""
    heads = [SV.head_of(doc) for doc in registrations]
    files = {"out/the-verdicts.jsonl": TSV.jsonl([r for h in heads for cls in SV.classes(h) for r in TSV.four(h, cls)]),
             "out/the-controls.jsonl": TSV.jsonl(TSV.the_controls()),
             SV.SERVES_FILE: TSV.jsonl(TSV.with_streams(TSV.SERVED))}
    if file.exists():
        files[SV.REGISTRATIONS_FILE] = file.read_text()
    return SV.of_runs(TSV.Store(tmp, {"a run": files}), ["a run"])


def test_the_check_writes_what_it_was_checked_under_then_each_registration_as_the_publisher_kept_it(
        tmp_path, monkeypatch):
    """Handed the SHA-512 of the network attester's public key and the session, and given a file, a check that
    accepts the run writes it. The first line, under its own format's word, names the hash and the session it was
    handed, and the reader's `checked_under` reads them from it; the hash is over the LMS public key ``I ‖ T[1]``,
    the bytes the deployment record writes in hex. Each line after it is one registration the check held to both,
    every handed execution
    phase's, in order: byte for byte the registration as the publisher kept it in `rows.jsonl`, and the head the
    reader's `head_of` reads from it is the head `check_registration` verified under the key, the session and the
    schedule. The reader, handed that file with the four verdicts of each head's classes, holds all three
    execution phases and prints what the file names. `check --key-sha512 HEX --session HEX --registrations FILE`
    writes the same file; a check that is given no file writes none, handed the two or not."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    under = handed(out)
    public = bytes.fromhex(json.loads((out / "deployment.json").read_text())["key"]["public"])
    assert under[0] == hashlib.sha512(public).digest() and len(public) == 16 + 64
    assert ws.check(out)["accepted"] and ws.check(out, None, under)["accepted"] and not file.parent.exists()
    assert ws.check(out, file, under)["accepted"]
    written, kept = file.read_text(), (out / crr.ROWS).read_text().splitlines()
    first, *lines = written.splitlines()
    names = (under[0].hex(), under[1].hex())
    assert first == json.dumps({"format": "verity-network-certifier/checked-registrations/v1",
                                "checked_under": {"key_sha512": names[0], "session": names[1]}})
    assert SV.checked_under(first) == names and CRC.CHECKED == SV.CHECKED == json.loads(first)["format"]
    assert len(lines) == len(kept) == 3 and written == "".join(line + "\n" for line in (first, *lines))
    key, session = device_key(out)
    for w, (line, publishers) in enumerate(zip(lines, kept)):
        assert publishers.startswith(f'{{"registration": {line}, "cost": ')
        verified, _ = CR.check_registration(json.loads(line), key, session, ROWS, ws.LINK, w)
        assert SV.head_of(json.loads(line)) == verified and verified.window == w
    held = answer(tmp_path / "read", file, [json.loads(line) for line in lines])
    # the reader is handed no schedule, so it holds no class below one, but it reads each head the file holds
    assert held.held == 0 and f"{TSV.CC.NOT_HANDED} for 9 of the 9 class(es)" in held.why[0], held.why
    assert TSV.names_under("a run", under=names) in held.why, held.why
    by_command = tmp_path / "out" / "by-command.jsonl"
    assert ws.main(command(out, under, by_command)) == 0 and by_command.read_text() == written
    assert ws.main(command(out)) == ws.main(command(out, under)) == 0
    assert sorted(p.name for p in file.parent.iterdir()) == ["by-command.jsonl", "registrations.jsonl"]


def test_a_run_under_another_key_gets_no_file_under_the_keys_hash_and_one_under_its_own(tmp_path, monkeypatch):
    """A run directory whose deployment record names another key than the network attester's, a rehearsal's own:
    its attestations and its registrations are signed under that key and verify under it. Handed the hash of the
    network attester's key, the check names the problem, the key and not the session, as its only one, and leaves
    no file; so does its command, which exits 1, with a file asked or not. Handed the hash of the run's own key, the
    same directory is accepted and its file is written, naming that hash: the reader holds all three execution
    phases on it and prints the hash, which it is not told to refuse."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "a rehearsal's key")
    own, session = handed(out)
    ws.ck.load_key(tmp_path / "key", 5)
    true = CRC.key_sha512(RF.read_public(json.loads((tmp_path / "key" / "public.json").read_text())))
    assert true != own and ws.check(out, file, (own, session))["accepted"]
    earlier = file.read_text()
    result = ws.check(out, file, (true, session))
    assert result["problems"] == ["the deployment record's key's SHA-512 is not the one the check was handed: "
                                  f"{own.hex()}, not {true.hex()}"]
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"]) == (3, 3) and not result["accepted"] and not file.exists()
    assert ws.main(command(out, (true, session), file)) == 1 and not file.exists()
    assert json.loads((out / "check.json").read_text())["problems"] == result["problems"]
    # the command names the problem with no file asked too, and a check handed nothing holds the run to no key
    assert ws.main(command(out, (true, session))) == 1 and ws.main(command(out)) == 0
    # under its own key's hash: the file, which says so
    assert ws.main(command(out, (own, session), file)) == 0 and file.read_text() == earlier
    first, *lines = earlier.splitlines()
    assert SV.checked_under(first) == (own.hex(), session.hex()) and len(lines) == 3
    held = answer(tmp_path / "read", file, [json.loads(line) for line in lines])
    assert held.held == 0 and f"{TSV.CC.NOT_HANDED} for 9 of the 9 class(es)" in held.why[0]
    assert TSV.names_under("a run", under=(own.hex(), session.hex())) in held.why


def test_a_registration_with_a_wrong_signature_gives_no_file_and_the_reader_holds_nothing(tmp_path, monkeypatch):
    """A registration whose signature is changed still names its own head's digest, so the reader's `head_of` reads
    the head it read before, and a reader handed that registration would hold its execution phase: the reader
    checks no signature. The verifier refuses it under the key, so it writes no file and removes the one an earlier
    check wrote, with the lines of the two execution phases it still holds; the reader, handed what the check left,
    charges every class its whole schedule and says no registration was read. With the run's files put back the
    check writes the file it wrote before."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    under = handed(out)
    assert ws.check(out, file, under)["accepted"]
    written = file.read_text()
    signed = [json.loads(line) for line in written.splitlines()[1:]]

    def other_signature(lines, salts) -> None:
        cert = lines[1]["registration"]["certificate"]
        cert["signature"] = ("1" if cert["signature"][0] == "0" else "0") + cert["signature"][1:]
        forged.append(json.loads(json.dumps(lines[1]["registration"])))

    forged: list[dict] = []
    problems = refused(out, lambda: edit_rows(out, other_signature), file, under)
    assert forged[0] != signed[1] and SV.head_of(forged[0]) == SV.head_of(signed[1])
    assert ("execution phase 1's connection rows: the registration's certificate doesn't verify under the network "
            "attester's key in the session") in problems
    assert not any("execution phase 0" in p or "execution phase 2" in p for p in problems)
    assert not file.exists()
    left = answer(tmp_path / "left", file, signed)
    assert left.held == 0 and "verdict record(s) were read and no registration" in left.why[0]
    # the reader alone reads a head nobody signed, under a first line anybody wrote, and would hold its class under
    # the schedule it names: only the check's file keeps it from one
    file.write_text(TSV.checked(*forged))
    assert f"{TSV.CC.NOT_HANDED} for 3 of the 3 class(es)" in answer(tmp_path / "forged", file, forged).why[0]
    assert ws.check(out, file, under)["accepted"] and file.read_text() == written
