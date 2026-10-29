"""The demo's stand-in (`demo_stub.py`): a warden's id request of the demo's class, as the warden forwards it, is
answered id-form with exactly the class's generated ids below its vocabulary, then the trailer; a request off the
class, or one that isn't a warden's id request, is answered 400; its prompts are the class's and the gate takes them."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier import active as A, io_tie as IT
from verity.core.protocols.certifier.release import with_release

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_challenger"))
import auditor as G  # noqa: E402
import demo_stub as S  # noqa: E402
import warden_serve as ws  # noqa: E402

P = S.ID_PAYLOAD
NAME = S.Shape().name
PROMPT = list(b"hello wo")


class _Writer:
    def __init__(self) -> None:
        self.data, self.closed = bytearray(), False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def respond(request: bytes, shape: S.Shape = S.Shape()) -> bytes:
    """The stand-in's whole answer to `request` on one connection, which it closes."""
    async def main() -> bytes:
        reader, writer = asyncio.StreamReader(), _Writer()
        reader.feed_data(request)
        reader.feed_eof()
        await S.Endpoint(shape)(reader, writer)
        assert writer.closed
        return bytes(writer.data)
    return asyncio.run(main())


def ids_request(**kw) -> bytes:
    fields = {"model": NAME, "user": "user-0", "prompt": PROMPT, "max_tokens": 24, "temperature": 0.0, "top_p": 1.0,
              "seed": 12345} | kw
    return A.id_request(**fields, frame_bytes=P + A.HEADER.size, offset=0)


def refused(data: bytes) -> str:
    head, _, body = data.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 400 Bad Request\r\n")
    return json.loads(body)["error"]


def test_the_class_is_eight_byte_ids_in_and_twenty_four_out_and_its_model_says_so():
    assert (S.PROMPT_IDS, S.TOKENS, S.VOCAB, S.ID_PAYLOAD) == (8, 24, 256, 503)
    assert NAME == "demo-stub/p8-g24-v256"
    assert P == G.SCAN["--cell-bytes"] - A.HEADER.size
    for bad in ({"prompt_ids": 0}, {"tokens": 0}, {"vocab": 1}, {"vocab": 257}, {"payload": IT.ID_EVENT_BYTES - 1}):
        with pytest.raises(ValueError):
            S.Shape(**bad)


def test_a_wardens_id_request_of_the_class_is_answered_id_form_its_ids_below_the_vocabulary_then_the_trailer():
    """With or without the release headers the warden puts on the server's copy, the answer is the prompt's 24 ids
    (`outputs`, fixed by the prompt alone), each an id event of P bytes, then the trailer: whole id-form, as the warden's
    rehearsal reads it (`warden_serve.parse_id_response`), and the scan's cells' payloads cut from it."""
    ids = S.outputs(PROMPT)
    want = b"".join(IT.id_event(t, P) for t in ids) + IT.TRAILER
    assert len(ids) == 24 and all(0 <= t < 256 for t in ids) and len(set(ids)) > 1
    assert S.answer(PROMPT) == want and len(want) == 24 * P + len(IT.TRAILER)
    assert want == ws.id_response(ids, P)
    assert ws.parse_id_response(want, P)["complete"] and ws.parse_id_response(want, P)["ids"] == 24
    for request in (ids_request(), with_release(ids_request(), 7, b"\x5a" * A.RUN_ID_BYTES),
                    with_release(ids_request(), 0)):
        assert respond(request) == want
    assert respond(ids_request(seed=1, user="user-9")) == want
    assert S.outputs(list(b"hello wp")) != ids
    cells = [IT.data_cell(t, 1, P + A.HEADER.size)[A.HEADER.size:] for t in ids]
    assert b"".join(cells) == want[:24 * P]


@pytest.mark.parametrize("kw, why", [
    ({"prompt": PROMPT[:7]}, "a prompt of exactly 8 ids below 256"),
    ({"prompt": PROMPT + [1]}, "a prompt of exactly 8 ids below 256"),
    ({"prompt": PROMPT[:7] + [256]}, "a prompt of exactly 8 ids below 256"),
    ({"max_tokens": 23}, "max_tokens exactly 24"),
    ({"max_tokens": 25}, "max_tokens exactly 24"),
    ({"temperature": 0.7}, "greedy"),
    ({"top_p": 0.9}, "greedy"),
    ({"model": "fake"}, "the model demo-stub/p8-g24-v256"),
])
def test_an_id_request_off_the_class_is_refused_over_http(kw, why):
    assert why in refused(respond(ids_request(**kw)))


def test_what_isnt_a_wardens_id_request_is_refused_and_the_models_list_is_the_one_model():
    """An id request built for smaller cells (a class request is one scan cell, so a larger cell's is the same bytes), a
    forwarded copy whose release header isn't where the warden puts it, a vLLM-shaped body and another path are none of
    the warden's id requests."""
    fields = {"model": NAME, "user": "user-0", "prompt": PROMPT, "max_tokens": 24, "seed": 12345}
    assert len(ids_request()) < P and A.id_request(**fields, frame_bytes=4096, offset=0) == ids_request()
    moved = ids_request().replace(b"\r\n\r\n", b"\r\nx-verity-release: 3\r\n\r\n", 1)
    for other in (A.id_request(**fields, frame_bytes=256, offset=0), moved,
                  ws.http_request({"model": NAME, "prompt": PROMPT, "max_tokens": 24}, "h"),
                  ids_request().replace(b"POST /v1/completions", b"POST /v1/chat/completions", 1)):
        assert other != ids_request() and S.warden_request(other) is None
        assert "a warden's id request" in refused(respond(other))
    models = respond(b"GET /v1/models HTTP/1.1\r\nhost: s\r\n\r\n")
    assert models.startswith(b"HTTP/1.1 200 OK\r\n")
    assert [m["id"] for m in json.loads(models.partition(b"\r\n\r\n")[2])["data"]] == [NAME]


def test_its_prompts_are_the_class_and_the_gate_takes_them_as_its_scan_rehearsals_prompts():
    """`prompts` (what `demo_stub.py prompts` prints) are distinct prompts of exactly 8 ids below 256, fixed by their
    index, and the gate's `warden-rehearse --scan --prompts --tokens 24` takes them off its stdin as they are."""
    got = S.prompts(4)
    assert got == S.prompts(4) and got[:2] == S.prompts(2) and len({tuple(p) for p in got}) == 4
    assert all(len(p) == 8 and all(0 <= t < 256 for t in p) for p in got)
    assert all(len(p) == 3 and all(t < 5 for t in p) for p in S.prompts(2, S.Shape(3, vocab=5)))
    req = {"requests": 8, "tokens": 24, "height": 10, "minutes": 4, "prompts": True, "scan": True}
    data = json.dumps(got, separators=(",", ":")).encode()
    assert G.warden_prompts(req, data) == data and req["prompts"] == {"count": 4, "ids": 32}


def test_serve_prints_where_it_listens_and_answers_over_loopback(capsys):
    """`serve` on a free loopback port, reached as the warden reaches its upstream: one request a connection."""
    async def main() -> tuple[bytes, bytes]:
        server = await asyncio.start_server(S.Endpoint(), "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        got = []
        for request in (with_release(ids_request(), 2, b"\x01" * A.RUN_ID_BYTES), ids_request(max_tokens=2)):
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(request)
            await writer.drain()
            got.append(await reader.read())
            writer.close()
        server.close()
        await server.wait_closed()
        return tuple(got)
    served, off = asyncio.run(main())
    assert served == S.answer(PROMPT) and "max_tokens exactly 24" in refused(off)
    assert S.main(["prompts", "--count", "2"]) == 0
    assert json.loads(capsys.readouterr().out) == S.prompts(2)


#: (the requests arriving, then the engine steps run before the next arrive; None: until the engine is idle): two
#: arriving together, one while they decode, two out of their ingress order, a Commit window cut at --window-steps
#: with requests running (carried into the next) and one closed at idle, then one on an idle engine
PLAN = [([0, 1], 3), ([2], 30), ([4, 3], 5), ([5], None), ([6], None)]
RUN = b"\x5a" * A.RUN_ID_BYTES


def _served_with_records(out: Path, window_steps: int) -> tuple[dict[int, bytes], S.Endpoint]:
    """`PLAN`'s class requests, user-j with prompt j and the warden's release N = j, through ``serve --out``'s
    scheduler and HTTP side, stepped here (`Endpoint.step`) on a clock of one second a reading once the connections'
    handlers have added them (the scheduler's arrivals, which each handler signals on `Endpoint.wake`), in the order
    their connections opened; each one's answer."""
    import demo_endpoint as DE
    a = argparse.Namespace(out=out, host="127.0.0.1", port=0, vocab=S.VOCAB, window_steps=window_steps, run_id="t")
    ticks = itertools.count()
    sched = DE.scheduler(S.endpoint_args(a), S.Toy(), clock=lambda: float(next(ticks)))
    server = S.Endpoint(S.Shape(), sched)
    prompts = S.prompts(7)

    async def main() -> dict[int, bytes]:
        tasks, writers = [], {}
        for arriving, steps in PLAN:
            for j in arriving:
                reader, writers[j] = asyncio.StreamReader(), _Writer()
                reader.feed_data(with_release(ids_request(user=f"user-{j}", prompt=prompts[j], seed=j), j, RUN))
                reader.feed_eof()
                tasks.append(asyncio.create_task(server(reader, writers[j])))
            while sched.arrivals < len(tasks):
                await server.wake.wait()
                server.wake.clear()
            ran = 0
            while sched.busy() and (steps is None or ran < steps):
                server.step()
                ran += 1
            if not sched.busy():
                sched.idle()
        await asyncio.gather(*tasks)
        assert all(w.closed for w in writers.values())
        return {j: bytes(w.data) for j, w in writers.items()}

    got = asyncio.run(main())
    sched.trace.close()
    (out / "serve.json").write_text(json.dumps(DE.serve_record(S.endpoint_args(a), sched, server, ("127.0.0.1", 0))))
    return got, server


@pytest.mark.parametrize("window_steps", [16, 64])
def test_with_out_its_records_are_the_endpoints_that_the_endpoint_step_binds_and_rebuilds(tmp_path, window_steps):
    """``serve --out``: each class request is answered the same bytes as without it, while the scheduler keeps
    serve-openai's records; requests.jsonl holds each one's user, the warden's release headers, its ids and clock. Then
    what chain_record's warden endpoint step runs on them: `verity-vllm advice` at the warden's bucket (its own
    `main`, a fixed compiler digest in place of the tree's code identity), and network-e2e's `served_advice` with the
    certified I/O (each user's prompt, max_tokens and the ids it was answered) in ingress order and the toy's EOS id:
    every Commit window's advice, those joined across a carry too, is bound to the certified requests, each once, and
    rebuilt from them. At --window-steps 16 the plan's windows are cut with requests running, at 64 only at idle."""
    sys.path.insert(0, str(HERE.parent / "network_e2e"))
    import chain
    from verity_vllm.pipeline import advice as ADV

    got, server = _served_with_records(tmp_path, window_steps)
    prompts = S.prompts(7)
    assert got == {j: S.answer(prompts[j]) for j in range(7)} and (server.served, server.refused) == (7, 0)
    records = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    by_user = {r["user"]: r for r in records}
    assert sorted(by_user) == sorted(f"user-{j}" for j in range(7))
    for j in range(7):
        r = by_user[f"user-{j}"]
        assert (r["prompt_ids"], r["streamed_ids"], r["release"]) == (prompts[j], S.outputs(prompts[j]), [j, RUN.hex()])
        assert r["finish"] == {"reason": "length", "stop_reason": None}
        assert r["times"]["arrived"] < r["times"]["first_token"] < r["times"]["finished"]
    serve = json.loads((tmp_path / "serve.json").read_text())
    assert serve["engine"]["engine_kwargs"]["max_model_len"] == 32 and serve["eos_token_ids"] == [S.VOCAB]

    window, T = {"origin_unix": 0.0, "windows": 4, "window_s": 100.0}, 10
    assert ADV.main(ADV.Options(run=str(tmp_path), tau=window["window_s"] / T, compiler_digest="ab" * 64)) == 0
    io = [{"user": f"user-{j}", "prompt": prompts[j], "max_tokens": 24, "output": S.outputs(prompts[j])}
          for j in range(7)]
    _, bits = chain.served_advice(tmp_path, tmp_path / "advice", window, io, T, [S.VOCAB])
    windows = bits["commit_windows"]
    assert (bits["bound"], bits["rebuilt"], bits["bound_why"], bits["why"]) == (True, True, None, None), bits
    assert len(windows) >= 2 and all(w["opened"] and w["commit_is_the_windows"] for w in windows)
    assert bool(bits["chains"]) == (window_steps == 16) and sum(w["requests"] for w in windows) == 7


@pytest.mark.parametrize("window_steps", [16, 64])
def test_on_vllms_records_art_the_endpoint_step_binds_a_requests_jsonl_rebuilt_from_its_users_map(tmp_path,
                                                                                                  monkeypatch,
                                                                                                  window_steps):
    """The served run's records artifact: advice_trace.jsonl, windows.jsonl, serve.json and request_users.jsonl,
    requests.jsonl left on the served node. served_session's chain rebuilds requests.jsonl from the map and the
    certified I/O, and the endpoint step's advice on it is bound to the certified requests and rebuilt from them, across
    a carry too; a map naming a user the I/O doesn't certify is refused."""
    sys.path.insert(0, str(HERE.parent / "network_e2e"))
    import chain
    import served_session as SS
    from verity_vllm.pipeline import advice as ADV

    (tmp_path / "ep").mkdir()
    _served_with_records(tmp_path / "ep", window_steps)
    art, prompts = tmp_path / "art", S.prompts(7)
    art.mkdir()
    for f in ("advice_trace.jsonl", "windows.jsonl", "serve.json"):
        (art / f).write_bytes((tmp_path / "ep" / f).read_bytes())
    own = [json.loads(x) for x in (tmp_path / "ep" / "requests.jsonl").read_text().splitlines()]
    (art / SS.USERS).write_text("".join(json.dumps({"engine_id": r["engine_id"], "user": r["user"]}) + "\n" for r in own))
    io = [{"user": f"user-{j}", "prompt": prompts[j], "max_tokens": 24, "output": S.outputs(prompts[j])}
          for j in range(7)]
    monkeypatch.setattr(SS, "certified_io", lambda fetch: {"accepted": True, "violations": [], "records": io})
    keep = tmp_path / "keep"
    ep = Path(SS.endpoint_records(str(art), keep / "rebuilt", keep / "fetch.tar.gz", [S.VOCAB]))
    assert ep == keep / "rebuilt" and ep.stat().st_mode & 0o777 == 0o700 and not (art / "requests.jsonl").exists()
    assert ADV.io_record(ep / "requests.jsonl") == ADV.io_record(tmp_path / "ep" / "requests.jsonl")

    window, T = {"origin_unix": 0.0, "windows": 4, "window_s": 100.0}, 10
    assert ADV.main(ADV.Options(run=str(ep), tau=window["window_s"] / T, compiler_digest="ab" * 64)) == 0
    _, bits = chain.served_advice(ep, ep / "advice", window, io, T, [S.VOCAB])
    assert (bits["bound"], bits["rebuilt"], bits["bound_why"], bits["why"]) == (True, True, None, None), bits
    assert bool(bits["chains"]) == (window_steps == 16) and sum(w["requests"] for w in bits["commit_windows"]) == 7

    (art / SS.USERS).write_text(json.dumps({"engine_id": "cmpl-x-0", "user": "user-9"}) + "\n")
    with pytest.raises(SS.Refused, match="user 'user-9' is no certified request"):
        SS.endpoint_records(str(art), keep / "again", keep / "fetch.tar.gz", [S.VOCAB])
