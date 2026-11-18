"""`scheme_plugin`'s engine side on stand-ins for vLLM's engine core and configuration: the schemes attached by name with
the engine's context, the exchange answered from the configuration, a refused input aborted, and what a scheme holds
released at shutdown."""

from __future__ import annotations

import asyncio
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
    monkeypatch.setitem(S.SCHEMES, "stand-in", SimpleNamespace(attach=attach))
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
    assert got == [(ex, {"process": H.ENGINE, "feed": FEED, "settings": {"k": 1}})]
    assert cfg.cache_config.num_gpu_blocks_override == 64
    assert ex.declared.properties == {"model": "m", "execution": None, "eager": True, "world": 1, "dtype": "bfloat16",
                                      "quantization": None, "prefix_cache": False, "kernels": "profile",
                                      "async_steps": False}
    assert (ex.declared.schemes, ex.declared.settings) == (("stand-in",), {"stand-in": {"k": 1}})
    with pytest.raises(NotImplementedError, match="vllm/scheme-on-in-vllm-server: schemes inside vLLM's server on one GPU"):
        SP.attach(messages.Exchange(), vllm_config(tp=2), *SP.config(ENV))


def test_register_attaches_before_the_kv_cache_and_releases_at_shutdown(stand_in, monkeypatch):
    got, held = stand_in
    seams: list = []
    monkeypatch.setattr(kv_blocks, "attach", lambda: (lambda: None))
    monkeypatch.setattr(admission, "register", lambda attach, refuse=None: seams.append(refuse))
    order: list[str] = []

    class Core:
        def _initialize_kv_caches(self, vllm_config):
            order.append(f"kv after {len(got)} attach")

        def shutdown(self):
            order.append(f"shutdown after {held.released} release")

    monkeypatch.setattr(SP, "_PLUGIN", [])
    SP.register(ENV, core=Core)
    try:
        SP.register(ENV, core=Core)
        assert len(SP._PLUGIN) == 1 and seams == [SP.aborts]
        core = Core()
        core._initialize_kv_caches(vllm_config())
        core.shutdown()
        assert order == ["kv after 1 attach", "shutdown after 1 release"] and got[0][0] is messages.current()
    finally:
        SP._PLUGIN[0][0].uninstall()


def test_a_refused_input_is_aborted_in_the_scheduler_and_its_reason_logged():
    finished, logged = [], []
    hook = SimpleNamespace(held={"cmpl-a-0-1": "a"}, live={"cmpl-b-0-1": "b", "cmpl-a-1-1": "a"},
                           sched=SimpleNamespace(finish_requests=lambda ids, status: finished.append((ids, status))))
    SP.aborts(hook, logged.append, status="ABORTED")("a", "its release was lost")
    assert finished == [(["cmpl-a-0-1", "cmpl-a-1-1"], "ABORTED")]
    assert logged == ["verity: input a refused (2 request(s) aborted): its release was lost"]


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
    monkeypatch.setitem(S.SCHEMES, "stand-in", SimpleNamespace(attach=attach))
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
    assert front_scheme["ctx"] == [{"process": H.FRONT, "feed": FEED, "settings": {}}]


def test_an_input_a_scheme_refuses_on_arrival_is_answered_400_and_never_reaches_the_server(front_scheme):
    front = SP.Front(lambda *a: pytest.fail("the server saw a refused input"), FRONT_ENV)
    sent = drive(front, http("/v1/completions", [("x-bad", "1"), ("x-request-id", "r1")]), [])
    assert answer(sent) == (400, "bad")
    arrived, done = front_scheme["events"]
    assert (arrived.input, arrived.first_line, done) == ("r1", "POST /v1/completions HTTP/1.1", H.OutputsComplete("r1", 0))


def test_an_input_refused_inside_the_server_gets_its_disconnect_and_a_400(front_scheme):
    got: dict[str, Any] = {}

    async def app(scope, receive, send):
        got["id"] = dict(scope["headers"])[b"x-request-id"].decode()
        got["started"].set()
        got["heard"] = await receive()

    front = SP.Front(app, FRONT_ENV)

    async def during():
        await got["started"].wait()
        front.refuse(got["id"], "its release was lost")

    sent = drive_with_event(front, got, during)
    assert got["heard"] == {"type": "http.disconnect"} and got["id"].startswith("verity-")
    assert answer(sent) == (400, "its release was lost")
    assert front_scheme["events"][-1] == H.OutputsComplete(got["id"], 0)


def drive_with_event(front, got, during):
    sent: list[dict] = []

    async def main():
        got["started"] = asyncio.Event()

        async def receive():
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
