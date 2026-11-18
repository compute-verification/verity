"""`scheme_plugin`'s engine side on stand-ins for vLLM's engine core and configuration: the schemes attached by name with
the engine's context, the exchange answered from the configuration, a refused input aborted, and what a scheme holds
released at shutdown."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity_vllm import messages
from verity_vllm.engine import admission, kv_blocks
from verity_vllm.pipeline import scheme_plugin as SP

FEED = "/tmp/feed"
ENV = {"VERITY_PROTOCOLS": "stand-in", "VERITY_PROTOCOLS_CONFIG": '{"feed": "%s", "stand-in": {"k": 1}}' % FEED}


def vllm_config(tp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="m", enforce_eager=True, dtype="torch.bfloat16", quantization=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False, num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(async_scheduling=False))


class Held:
    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


@pytest.fixture
def stand_in(monkeypatch):
    got: list[tuple[Any, dict]] = []
    held = Held()

    def attach(exchange, ctx):
        got.append((exchange, ctx))
        exchange.request("stand-in", H.FixRanges(64))
        return held
    monkeypatch.setitem(S.SCHEMES, "stand-in", SimpleNamespace(name="stand-in", settings={"k": 0}, check_settings=lambda s: None, attach=attach,
                                                               requests=lambda ctx: ()))
    yield got, held
    messages.close()


def test_the_settings_name_the_schemes_and_their_shared_feed():
    assert SP.config({}) == ([], {})
    assert SP.config(ENV) == (["stand-in"], {"feed": FEED, "stand-in": {"k": 1}})
    with pytest.raises(ValueError, match="with feed, the directory the server's front and its engine share"):
        SP.config({"VERITY_PROTOCOLS": "stand-in"})


def test_the_engine_attaches_each_scheme_with_its_context_and_answers_what_vllms_configuration_declares(stand_in):
    got, held = stand_in
    ex, cfg = messages.Exchange(), vllm_config()
    assert SP.attach(ex, cfg, *SP.config(ENV)) == [held]
    assert got == [(ex, {"settings": {"k": 1}, "build": ex.key, "process": H.ENGINE, "feed": FEED})]
    assert cfg.cache_config.num_gpu_blocks_override == 64
    assert ex.declared.properties == {"model": "m", "execution": None, "eager": True, "world": 1, "dtype": "bfloat16",
                                      "quantization": None, "prefix_cache": False, "kernels": "profile",
                                      "async_steps": False}
    assert (ex.declared.schemes, ex.declared.settings) == (("stand-in",), {"stand-in": {"k": 1}})
    with pytest.raises(NotImplementedError, match="vllm/scheme-on-in-vllm-server: schemes inside vLLM's server on one GPU"):
        SP.attach(messages.Exchange(), vllm_config(tp=2), *SP.config(ENV))


def test_register_attaches_before_the_model_loads_and_releases_at_shutdown(stand_in, monkeypatch):
    got, held = stand_in
    seams: list = []
    order: list[str] = []

    class Runner:
        vllm_config = vllm_config()

        def load_model(self):
            order.append(f"load after {len(got)} attach")

    def kv_attach():
        orig = Runner.load_model

        def load_model(runner):
            orig(runner)
            order.append("registered values")
        Runner.load_model = load_model
        return lambda: setattr(Runner, "load_model", orig)

    class Core:
        def shutdown(self):
            order.append(f"shutdown after {held.released} release")

    monkeypatch.setattr(kv_blocks, "attach", kv_attach)
    monkeypatch.setattr(admission, "register", lambda attach, refuse=None: seams.append(refuse))
    monkeypatch.setattr(SP, "_PLUGIN", [])
    SP.register(ENV, core=Core, runners=(Runner,))
    try:
        SP.register(ENV, core=Core, runners=(Runner,))
        assert len(SP._PLUGIN) == 1 and seams == [SP.aborts]
        Runner().load_model()
        Core().shutdown()
        assert order == ["load after 1 attach", "registered values", "shutdown after 1 release"]
        assert got[0][0] is messages.current() and Runner.vllm_config.cache_config.num_gpu_blocks_override == 64
    finally:
        SP._PLUGIN[0][0].uninstall()
        SP._PLUGIN[0][1]()


def test_a_refused_input_is_aborted_in_the_scheduler_and_its_reason_logged():
    finished, logged = [], []
    hook = SimpleNamespace(held={"cmpl-a-0-1": "a"}, live={"cmpl-b-0-1": "b", "cmpl-a-1-1": "a"},
                           sched=SimpleNamespace(finish_requests=lambda ids, status: finished.append((ids, status))))
    SP.aborts(hook, logged.append, status="ABORTED")("a", "its release was lost")
    assert finished == [(["cmpl-a-0-1", "cmpl-a-1-1"], "ABORTED")]
    assert logged == ["verity: input a refused (2 request(s) aborted): its release was lost"]


GREEDY = {"type": "http.request", "body": b'{"temperature": 0}'}
FRONT_ENV = {"VERITY_PROTOCOLS": "stand-in", "VERITY_PROTOCOLS_CONFIG": '{"feed": "%s"}' % FEED}


@pytest.fixture
def front_scheme(monkeypatch):
    seen: dict[str, Any] = {"ctx": [], "events": [], "bodies": []}

    def serve(first_line, headers, body):
        seen["bodies"].append((first_line, headers.get("x-run"), list(body)))
        return 200, "fed"

    def attach(exchange, ctx):
        seen["ctx"].append(ctx)
        exchange.request("stand-in", H.ServeRoute("/feed", serve))

        def on(e):
            seen["events"].append(e)
            if isinstance(e, H.InputArrived) and "x-bad" in e.headers:
                exchange.request("stand-in", H.RefuseInput(e.input, "bad"))
        exchange.subscribe("stand-in", "steps", on)
    monkeypatch.setitem(S.SCHEMES, "stand-in", SimpleNamespace(name="stand-in", settings={"k": 0}, check_settings=lambda s: None, attach=attach,
                                                               requests=lambda ctx: ()))
    yield seen
    messages.close()


def http(path, headers=()):
    return {"type": "http", "method": "POST", "path": path, "root_path": "", "http_version": "1.1",
            "headers": [(k.encode(), v.encode()) for k, v in headers]}


def drive(front, scope, messages_in, during=None):
    sent: list[dict] = []

    async def main():
        queue_in = list(messages_in)

        async def receive():
            if queue_in:
                return queue_in.pop(0)
            await asyncio.Future()

        async def send(m):
            sent.append(m)
        task = asyncio.ensure_future(front(scope, receive, send))
        if during is not None:
            await during()
        await task
    asyncio.run(main())
    return sent


def answer(sent):
    return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:]).decode()


def test_the_front_answers_a_schemes_route_with_the_body_as_it_streams_in(front_scheme):
    app_calls: list = []
    front = SP.Front(lambda *a: app_calls.append(a), FRONT_ENV)
    sent = drive(front, http("/feed", [("x-run", "ab")]), [{"type": "http.request", "body": b"0 1\n", "more_body": True},
                                                          {"type": "http.request", "body": b"1 2\n"}])
    assert answer(sent) == (200, "fed") and not app_calls
    assert front_scheme["bodies"] == [("POST /feed HTTP/1.1", "ab", [b"0 1\n", b"1 2\n"])]
    assert front_scheme["ctx"] == [{"settings": {"k": 0}, "build": front.ex.key, "process": H.FRONT, "feed": FEED}]


def test_an_input_a_scheme_refuses_on_arrival_is_answered_400_and_never_reaches_the_server(front_scheme):
    front = SP.Front(lambda *a: pytest.fail("the server saw a refused input"), FRONT_ENV)
    sent = drive(front, http("/v1/completions", [("x-bad", "1"), ("x-request-id", "r1")]), [GREEDY])
    assert answer(sent) == (400, "bad")
    arrived, done = front_scheme["events"]
    assert (arrived.input, arrived.length, arrived.first_line, done) == (
        "r1", len(GREEDY["body"]), "POST /v1/completions HTTP/1.1", H.OutputsComplete("r1", 0))


@pytest.mark.parametrize("path,body,why", [
    ("/v1/responses", b"", "/v1/responses is not served under a scheme"),
    ("/v1/completions", b'{"temperature": 0.7}', "needs an integer seed"),
])
def test_an_unserved_route_and_a_sampled_request_without_its_seed_are_refused_after_arriving(front_scheme, path, body, why):
    front = SP.Front(lambda *a: pytest.fail("the server saw a refused input"), FRONT_ENV)
    sent = drive(front, http(path, [("x-request-id", "r1")]), [{"type": "http.request", "body": body}])
    assert answer(sent)[0] == 400 and why in answer(sent)[1]
    assert [type(e) for e in front_scheme["events"]] == [H.InputArrived, H.OutputsComplete]


def test_a_websocket_is_closed_under_a_scheme(front_scheme):
    front = SP.Front(lambda *a: pytest.fail("the server saw a websocket"), FRONT_ENV)
    sent = drive(front, {"type": "websocket", "path": "/v1/realtime", "headers": []}, [{"type": "websocket.connect"}])
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_an_input_refused_inside_the_server_gets_its_disconnect_and_a_400(front_scheme):
    got: dict[str, Any] = {}

    async def app(scope, receive, send):
        got["id"] = dict(scope["headers"])[b"x-request-id"].decode()
        got["started"].set()
        got["body"] = await receive()
        got["heard"] = await receive()

    front = SP.Front(app, FRONT_ENV)

    async def during():
        await got["started"].wait()
        front.refuse(got["id"], "its release was lost")

    sent = drive_with_event(front, got, during)
    assert got["body"] == GREEDY and got["heard"] == {"type": "http.disconnect"} and got["id"].startswith("verity-")
    assert answer(sent) == (400, "its release was lost")
    assert front_scheme["events"][-1] == H.OutputsComplete(got["id"], 0)


def drive_with_event(front, got, during):
    sent: list[dict] = []

    async def main():
        got["started"] = asyncio.Event()

        bodies = [GREEDY]

        async def receive():
            if bodies:
                return bodies.pop()
            await asyncio.Future()

        async def send(m):
            sent.append(m)
        task = asyncio.ensure_future(front(http("/v1/completions"), receive, send))
        await during()
        await task
    asyncio.run(main())
    return sent


def test_without_the_setting_the_front_passes_everything():
    calls: list = []

    async def app(scope, receive, send):
        calls.append(scope["path"])
    front = SP.Front(app, {})
    drive(front, http("/anything"), [])
    assert calls == ["/anything"] and front.ex is None


HEAD = {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
SIZED = {"type": "http.response.start", "status": 200, "headers": [(b"Content-Length", b"4")]}
NULL = {"type": "http.response.body", "body": b"null"}


class Waiting:
    """vLLM's server with a request it waits on. It reads the body and sends ``head`` at once, if it has one: a stream's
    200 head, which vLLM sends before any token, or a whole answer's with its length. From there it waits on the
    request's disconnect alone, as vLLM's listener beside each handler does, and once it has it sends ``late``."""

    def __init__(self, head: dict | None, late: list[dict]) -> None:
        self.head, self.late, self.waits, self.ended = head, late, 0, 0

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        more = True
        while more:
            more = (await receive()).get("more_body", False)
        if self.head is not None:
            await send(self.head)
        self.waits += 1
        while (await receive())["type"] != "http.disconnect":
            pass
        for m in self.late:
            await send(m)
        self.ended += 1


async def turns(k: int = 16) -> None:
    loop = asyncio.get_running_loop()
    for _ in range(k):
        turn = loop.create_future()
        loop.call_soon(turn.set_result, None)
        await turn


@pytest.mark.parametrize("head", [None, HEAD, SIZED], ids=["no response begun", "a stream's head gone", "a whole answer's head gone"])
def test_a_lost_releases_request_in_the_server_is_given_its_disconnect_and_answered(tmp_path, head):
    """Release 0's request is in vLLM's server, its client's connection open, when a watermark line names it lost, and
    the front ends it: the server gets the request's disconnect. Where the server had begun no response, what it sends
    after goes nowhere and the client gets 400 and the network attester's sentence. Where a stream's 200 head had gone,
    the client gets the stream's end, with no later chunk. Where a whole answer's head had gone with its length, which
    takes an open gate, its body goes on, and the front adds nothing."""
    from verity.protocols.resource_accounting.network.prover import admission as NA
    from verity.protocols.resource_accounting.network.verifier import release as REL

    name = NA.scheme_name()
    late = ([SIZED, NULL] if head is None else [NULL] if head is SIZED else
            [{"type": "http.response.body", "body": b"data: late\n\n", "more_body": True}])
    app = Waiting(head, late)
    front = SP.Front(app, {"VERITY_PROTOCOLS": name, "VERITY_PROTOCOLS_CONFIG": json.dumps(
        {"feed": str(tmp_path), name: {"bucket_us": 100_000, "sat_us": 30_000, "free_us": 40_000}})})
    numbered = http("/v1/completions", [(NA.RELEASE_HEADER, "0"), (NA.RUN_HEADER, "ab"), ("x-request-id", "c0")])
    mark = http(NA.WATERMARK_PATH, [(NA.RUN_HEADER, "ab"), ("transfer-encoding", "chunked")])
    sent: list[dict] = []
    try:
        async def run() -> tuple[bool, list[dict], bool]:
            body, never = [GREEDY], asyncio.Event()

            async def receive() -> dict:
                if not body:
                    await never.wait()
                return body.pop(0)

            async def send(m: dict) -> None:
                sent.append(m)

            call = asyncio.ensure_future(front(numbered, receive, send))
            await turns()
            held = app.waits == 1 and not app.ended and not call.done()
            lines = [{"type": "http.request", "body": REL.watermark_chunk(0, 2, [0]).split(b"\r\n")[1], "more_body": True},
                     {"type": "http.request", "body": b"", "more_body": False}]
            answer: list[dict] = []

            async def put(m: dict) -> None:
                answer.append(m)

            async def lines_in() -> dict:
                return lines.pop(0)
            await front(mark, lines_in, put)
            await turns()
            done = call.done()
            call.cancel()
            await asyncio.wait({call})
            return held, answer, done

        held, answer, done = asyncio.run(run())
        assert held and answer[0]["status"] == 200 and done and app.ended == 1
        if head is None:
            assert [m.get("status") for m in sent] == [400, None]
            assert sent[1]["body"] == NA.LOST.format(n=0, run="ab").encode()
        else:
            assert sent == [head, NULL if head is SIZED else {"type": "http.response.body", "body": b""}]
        assert SP.FeedReader(NA.feed_file(str(tmp_path)))() == [
            {"key": "c0", "run": "ab", "n": 0, "bucket": None}, {"run": "ab", "t": 0, "n": 2, "lost": [0]},
            {"run": "ab", "end": True}, *([{"run": "ab", "n": 0, "passed": True}] if head is None else [])]
        assert front.inside == {}
    finally:
        messages.close()
