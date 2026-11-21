"""Schemes turned on by name inside vLLM's own OpenAI server (``vllm serve``), each the scheme its name registers in the
one table (`verity.protocols.run.schemes.get`), as ``VERITY_PROTOCOLS`` lists them, with ``VERITY_PROTOCOLS_CONFIG``.

`Front`, the server's ``--middleware``, is the front's side, in the API server's process. `register`, a vLLM general
plugin, is the engine's side, in the engine core's process: the adapter's memory hook
(`engine.kv_blocks`) on, and each engine core's schemes attached on the process's exchange (``process`` `hooks.ENGINE`)
just before its model runner loads the model, so that they hear the model's `hooks.RegisteredValues` and its linear
site attaches, the exchange answered with what vLLM's
configuration declares, the admission hook (`engine.admission.register`, keyed by the client's X-Request-Id) on its
scheduler, and what each scheme holds released when the engine core shuts down, then each scheme's
report written under the run output (`write_reports`). A scheme's
`hooks.RefuseInput` there aborts the input, its reason the client's ``stop_reason`` (`aborts`).

No committer runs here, so there are no execution phases (no `hooks.PhaseOpened`, `PhaseClosed` or `PhaseCommitted`):
a scheme's epoch is the server's life, from its attach to its release.

vLLM loads its general plugins in each of its processes, so `register` imports vLLM only once ``VERITY_PROTOCOLS`` is
set, and the engine's side acts only in a process that builds an engine core.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity import hooks as H
from verity_vllm import messages, messages as M, settings

__all__ = ["GOAL", "OTHER_PATHS", "SAMPLED_PATHS", "FeedReader", "Front", "attach", "config", "declared", "register",
           "route_of", "unseeded", "write_reports"]

#: the goal whose unbuilt parts raise `NotImplementedError` here
GOAL = "vllm/scheme-on-in-vllm-server"

_PLUGIN: list[Any] = []
#: the OpenAI server's routes whose request body carries sampling parameters, which `Front` reads (`unseeded`)
SAMPLED_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
#: the routes `Front` passes unread besides `SAMPLED_PATHS` and the schemes' own (`hooks.ServeRoute`): none generates
OTHER_PATHS = frozenset({"/health", "/ping", "/version", "/v1/models", "/metrics", "/metrics/", "/tokenize", "/detokenize"})


def config(environ: Mapping[str, str] | None = None) -> tuple[list[str], dict[str, Any]]:
    """``VERITY_PROTOCOLS``'s names and ``VERITY_PROTOCOLS_CONFIG``'s object (``feed`` required with a name; the
    engine's ``out``, the run output its schemes' reports go under, and ``private``, the directory each scheme's own
    goes under (`messages.Exchange.attach`); ``gate`` infra's challenger gate command where a scheme's coins come
    through it; and each scheme's settings under its name); no names, nothing."""
    names = [n for n in settings.get("VERITY_PROTOCOLS", environ) if n]
    if not names:
        return [], {}
    cfg = settings.get("VERITY_PROTOCOLS_CONFIG", environ)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("feed"), str) or not cfg["feed"]:
        raise ValueError("VERITY_PROTOCOLS_CONFIG: a JSON object with feed, the directory the server's front and its engine "
                         f"share, and each scheme's configuration under its name, not {cfg!r:.80}")
    return names, cfg


def challenger_gate(command: str | None) -> Callable[..., Mapping[str, Any]] | None:
    """The auditor device's challenger gate over infra's gate command ``command`` (``VERITY_PROTOCOLS_CONFIG``'s
    ``gate``), each scheme's ``ctx["gate"]`` in the engine; None without one."""
    if not command:
        return None
    from verity_vllm.pipeline.commit_coins import ChallengerGate
    return ChallengerGate(command).call


def declared(vllm_config: Any, attached: Mapping[str, Mapping[str, Any]] | None = None) -> H.Declared:
    """What vLLM's configuration declares, the properties `engine.build.build_engine` declares of its own engine, with
    the schemes ``attached`` (`messages.Exchange.attached`) and their settings."""
    from verity_vllm.engine import served

    attached = attached or {}
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    return H.Declared("", {
        "model": mc.model, "execution": None, "eager": bool(mc.enforce_eager),
        "world": pc.tensor_parallel_size * pc.pipeline_parallel_size,
        "dtype": str(mc.dtype).removeprefix("torch."), "quantization": mc.quantization,
        "prefix_cache": bool(vllm_config.cache_config.enable_prefix_caching),
        "kernels": served.kernels({"quantization": mc.quantization}),
        "async_steps": bool(getattr(vllm_config.scheduler_config, "async_scheduling", False))},
        tuple(attached), {n: dict(v) for n, v in attached.items()})


def attach(exchange: messages.Exchange, vllm_config: Any, names: list[str], cfg: Mapping[str, Any]) -> list[Any]:
    """Each scheme ``names`` on ``exchange`` in the engine (`messages.attach_listed`), then the exchange answered (`messages.Exchange.answer`); what
    each attach returned, in order. An engine on more than one GPU, or whose model runner is not in the engine core's
    process (where its scheduler and steps are), is refused by `GOAL`."""
    from verity.protocols.run import schemes

    pc = vllm_config.parallel_config
    shape = (pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.data_parallel_size, pc.distributed_executor_backend)
    if shape != (1, 1, 1, "uni"):
        raise NotImplementedError(f"{GOAL}: schemes inside vLLM's server on one GPU with its uniprocess executor, not "
                                  f"tensor_parallel_size {shape[0]} × pipeline_parallel_size {shape[1]} × "
                                  f"data_parallel_size {shape[2]} under distributed_executor_backend {shape[3]!r}")
    gate = challenger_gate(cfg.get("gate"))
    got = M.attach_listed(exchange, [(schemes.get(n), cfg.get(n)) for n in names], gate=gate, private=cfg["private"],
                          process=H.ENGINE, feed=cfg["feed"])
    if fixed := sorted({r.ranges for _, r in exchange.requests if isinstance(r, H.FixRanges)}):
        if len(fixed) > 1:
            raise RuntimeError(f"schemes fix the KV pool at {fixed} blocks: one size")
        vllm_config.cache_config.num_gpu_blocks_override = fixed[0]
    exchange.answer(declared(vllm_config, exchange.attached))
    return got


def write_reports(exchange: messages.Exchange, out: str | Path) -> None:
    """At the workload's stop, each scheme attached on ``exchange`` released, then its report written
    under ``out/schemes/<its name>/``: ``configuration.json`` its settings and ``record``, ``summary.json`` its
    ``summary``, and ``commitment.json``, its closed ``commitment``, each where the scheme's report has it
    (`verity.protocols.accounting.scheme`). Every scheme is released
    and written before the first release's failure is raised. What a scheme keeps private it keeps under its own
    ``ctx["private"]``."""
    failed: list[BaseException] = []
    for got in exchange.reports.values():
        if callable(getattr(got, "release", None)):
            try:
                got.release()
            except Exception as e:  # noqa: BLE001 -- every scheme is released and written before it is raised
                failed.append(e)
    for name, got in exchange.reports.items():
        d = Path(out) / "schemes" / name
        d.mkdir(parents=True, exist_ok=True)
        record = got.record() if callable(getattr(got, "record", None)) else {}
        M.write(d / "configuration.json", "ServingConfiguration",
                json.dumps({"scheme": name, "settings": exchange.attached.get(name), "record": record}, indent=1,
                           default=str).encode())
        if callable(getattr(got, "summary", None)):
            M.write(d / "summary.json", "RunTotals", json.dumps(got.summary(), indent=1, default=str).encode())
        if callable(getattr(got, "commitment", None)):
            M.write(d / "commitment.json", "Run", json.dumps(got.commitment(), indent=1, default=str).encode())
    if failed:
        raise failed[0]


def aborts(hook: Any, core: Any, log: Callable[[str], Any] = lambda s: print(s, file=sys.stderr, flush=True),
           vllm: Any = None) -> Callable[[str, str], None]:
    """The engine's answer to a `hooks.RefuseInput`: input ``input``'s requests in ``hook``'s scheduler finished as
    aborted, and each sent to its client finished with ``finish_reason`` abort and ``why`` as its ``stop_reason``, which
    vLLM's OpenAI server puts in the response's choice; ``why`` logged too. ``vllm`` holds vLLM's ``RequestStatus``,
    ``FinishReason``, ``EngineCoreOutput`` and ``EngineCoreOutputs`` unless given."""
    def refuse(input: str, why: str) -> None:
        v = vllm or _vllm()
        rids = [rid for table in (hook.held, hook.live) for rid, i in table.items() if i == input]
        log(f"verity: input {input} refused ({len(rids)} request(s) aborted): {why}")
        by_client: dict[int, list[str]] = {}
        for r in hook.sched.finish_requests(rids, v.RequestStatus.FINISHED_ABORTED) if rids else ():
            by_client.setdefault(r.client_index, []).append(r.request_id)
        for client, ids in by_client.items():
            outs = [v.EngineCoreOutput(i, [], finish_reason=v.FinishReason.ABORT, stop_reason=why) for i in ids]
            core.output_queue.put_nowait((client, v.EngineCoreOutputs(finished_requests=ids, outputs=outs)))
    return refuse


def _vllm() -> Any:
    from types import SimpleNamespace

    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.request import RequestStatus
    return SimpleNamespace(RequestStatus=RequestStatus, FinishReason=FinishReason, EngineCoreOutput=EngineCoreOutput,
                           EngineCoreOutputs=EngineCoreOutputs)


def register(environ: Mapping[str, str] | None = None, core: type | None = None,
             runners: tuple[type, ...] | None = None) -> None:
    """The engine's side (the module docstring), as a vLLM general plugin with ``VERITY_PROTOCOLS`` set; ``core`` is
    vLLM's ``EngineCore`` and ``runners`` its model runners (`engine.kv_blocks.RUNNERS`) unless given. vLLM runs it at
    the top of ``EngineCore.__init__``, too late to wrap that, so the schemes attach in the runner's ``load_model``,
    outside `engine.kv_blocks`' wrap of it. Idempotent."""
    names, cfg = config(environ)
    if not names or _PLUGIN:
        return
    if not all(isinstance(cfg.get(k), str) and cfg[k] for k in ("out", "private")):
        raise ValueError("VERITY_PROTOCOLS_CONFIG: the engine's schemes need out, the run output their reports go under, "
                         "and private, the directory outside it each scheme keeps its own under")
    from verity_vllm.engine import admission, kv_blocks
    from verity_vllm.engine.hooks import Hooks

    if core is None:
        from vllm.v1.engine.core import EngineCore as core
    if runners is None:
        import importlib
        import importlib.util
        runners = tuple(importlib.import_module(m).GPUModelRunner for m in kv_blocks.RUNNERS
                        if importlib.util.find_spec(m) is not None)
    hooks = Hooks(tag="scheme-plugin")
    built: list[messages.Exchange] = []

    def load(orig: Callable[..., Any]) -> Callable[..., Any]:
        def load_model(runner: Any, *a: Any, **kw: Any) -> Any:
            built.append(ex := messages.current())
            attach(ex, runner.vllm_config, names, cfg)
            return orig(runner, *a, **kw)
        return load_model

    def stop(orig: Callable[..., Any]) -> Callable[..., Any]:
        def shutdown(self: Any, *a: Any, **kw: Any) -> Any:
            failed: list[BaseException] = []
            while built:
                try:
                    write_reports(built.pop(0), cfg["out"])
                except Exception as e:  # noqa: BLE001 -- every scheme is released, and vLLM shut down, before it is raised
                    failed.append(e)
            r = orig(self, *a, **kw)
            if failed:
                raise failed[0]
            return r
        return shutdown

    detach = kv_blocks.attach()
    for runner in runners:
        hooks.wrap(runner, "load_model", load)
    hooks.wrap(core, "shutdown", stop)
    admission.register(lambda exchange: None, refuse=aborts)
    _PLUGIN.append((hooks, detach))


def route_of(scope: Mapping[str, Any]) -> str:
    """An ASGI request's route: its ``path`` less the server's ``root_path`` where the path starts with it."""
    path, root = scope.get("path") or "", (scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        return path[len(root):] or "/"
    return path


def user_of(body: bytes) -> str | None:
    """The ``user`` a request ``body`` (a JSON object) names; None where it names none, or isn't one."""
    try:
        req = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None
    user = req.get("user") if isinstance(req, dict) else None
    return user if isinstance(user, str) else None


def unseeded(body: bytes) -> str | None:
    """Why `Front` refuses a completions or chat request ``body``, None where it passes. A stochastic request must carry
    an integer ``seed``: vLLM gives one without a seed a seed it samples from ``np.random``, which nobody can recompute.
    Only a numeric ``temperature`` of 0 is greedy; without one the server's default applies (the model's generation
    config's, else 1.0), so it needs a seed too. A body that isn't a JSON object passes, for the server to answer."""
    try:
        req = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(req, dict):
        return None
    temp, seed = req.get("temperature"), req.get("seed")
    if type(temp) in (int, float) and temp == 0:
        return None
    if type(seed) is int:
        return None
    at = "the server's default temperature" if temp is None else f"temperature {json.dumps(temp)[:40]}"
    return (f"under a scheme a request sampled at {at} needs an integer seed, not {json.dumps(seed)[:40]}: "
            "send one, or temperature 0")


class FeedReader:
    """The records appended to a feed file (JSON lines) since the last call; a partial last line waits for its end."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path, self.fd, self.tail = os.fspath(path), None, b""

    def __call__(self) -> list[dict[str, Any]]:
        if self.fd is None:
            try:
                self.fd = os.open(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return []
        data = self.tail
        while chunk := os.read(self.fd, 1 << 16):
            data += chunk
        *lines, self.tail = data.split(b"\n")
        return [json.loads(x) for x in lines if x.strip()]


async def _buffered(receive: Callable[[], Any]) -> tuple[bytes, Callable[[], Any]]:
    """The request's whole body, and a ``receive`` that replays the messages it read before the rest."""
    msgs: list[dict[str, Any]] = []
    while True:
        m = await receive()
        msgs.append(m)
        if m.get("type") != "http.request" or not m.get("more_body", False):
            break
    body = b"".join(m.get("body", b"") for m in msgs if m.get("type") == "http.request")

    async def replay() -> Any:
        return msgs.pop(0) if msgs else await receive()
    return body, replay


async def _answer(send: Callable[[dict[str, Any]], Any], status: int, text: str) -> None:
    body = text.encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", b"%d" % len(body))]})
    await send({"type": "http.response.body", "body": body})


class Front:
    """vLLM's OpenAI server's ``--middleware`` with ``VERITY_PROTOCOLS`` set, the front's side: each scheme attached on
    this process's exchange (``process`` `hooks.FRONT`), a `hooks.ServeRoute` answered by its scheme's route (on a thread
    of its own, given the body as it streams in), and every other request handed to the schemes as an `hooks.InputArrived`
    (its X-Request-Id, set where the client sent none, its first line, headers, body's length and the user its body names) before the server sees
    it. Only `SAMPLED_PATHS` and `OTHER_PATHS` are served, and a sampled request only with its seed (`unseeded`): any
    other is answered 400, and a websocket closed. A scheme's `hooks.RefuseInput` answers it 400 with the reason too;
    an input refused before the server sees it is complete at the front at step 0 (`hooks.OutputsComplete`). Filed while the server has it, the server is given its
    disconnect, and the request is answered 400 where no response began, its stream ended where one did. An input the
    server answered with no 200 never reached the engine, and is complete at the front (`hooks.OutputsComplete`).
    Without the variable it passes everything."""

    def __init__(self, app: Any, environ: Mapping[str, str] | None = None) -> None:
        from verity.protocols.run import schemes

        self.app, self.ex = app, None
        names, cfg = config(environ)
        if names:
            self.ex = messages.current()
            self.ex.refuse = self.refuse
            for n in names:
                self.ex.attach(schemes.get(n), cfg.get(n), process=H.FRONT, feed=cfg["feed"])
        self.inside: dict[str, Any] = {}           # input -> done once a scheme refuses it, while the server has it

    def refuse(self, input: str, why: str) -> None:
        gone = self.inside.get(input)
        if gone is not None and not gone.done():
            gone.get_loop().call_soon_threadsafe(lambda: gone.done() or gone.set_result(why))

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Any], send: Callable[[dict[str, Any]], Any]) -> None:
        if self.ex is None or scope.get("type") not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await receive()
            return await send({"type": "websocket.close", "code": 1008})
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or ()}
        path = route_of(scope)
        first_line = f"{scope.get('method', 'GET')} {path} HTTP/{scope.get('http_version', '1.1')}"
        if (served := self.ex.route(path)) is not None:
            return await self._route(served[1], first_line, headers, receive, send)
        input = headers.get("x-request-id") or f"verity-{uuid.uuid4().hex}"
        if "x-request-id" not in headers:
            scope = {**scope, "headers": [*scope.get("headers", ()), (b"x-request-id", input.encode())]}
            headers["x-request-id"] = input
        body = b""
        if path in SAMPLED_PATHS:
            body, receive = await _buffered(receive)
        self.ex.emit(H.InputArrived(input, len(body), 0, headers, first_line, user_of(body)))
        why = [w for _, w in self.ex.refused(input)]
        if path not in SAMPLED_PATHS | OTHER_PATHS:
            why.append(f"{path} is not served under a scheme: only {sorted(SAMPLED_PATHS | OTHER_PATHS)} and the schemes' "
                       "own routes")
        elif path in SAMPLED_PATHS and (seedless := unseeded(body)):
            why.append(seedless)
        if why:
            self.ex.emit(H.OutputsComplete(input, 0))
            return await _answer(send, 400, "; ".join(why))
        gone = self.inside[input] = asyncio.get_running_loop().create_future()
        began = started = sized = ended = False

        async def heard() -> Any:
            if gone.done():
                return {"type": "http.disconnect"}
            message = asyncio.ensure_future(receive())
            try:
                await asyncio.wait({message, gone}, return_when=asyncio.FIRST_COMPLETED)
                if message.done():
                    return message.result()
            finally:
                message.cancel()
            return {"type": "http.disconnect"}

        async def watched(message: dict[str, Any]) -> None:
            nonlocal began, started, sized, ended
            if gone.done() and not sized:
                return
            kind = message.get("type")
            if kind == "http.response.start":
                started, began = True, began or message.get("status") == 200
                sized = any(k.lower() == b"content-length" for k, _ in message.get("headers") or ())
            elif kind == "http.response.body" and not message.get("more_body", False):
                ended = True
            await send(message)

        try:
            await self.app(scope, heard, watched)
        finally:
            del self.inside[input]
            if not began:
                self.ex.emit(H.OutputsComplete(input, 0))
        if gone.done() and not started:
            await _answer(send, 400, gone.result())
        elif gone.done() and not sized and not ended:
            await send({"type": "http.response.body", "body": b""})

    async def _route(self, serve: H.Route, first_line: str, headers: Mapping[str, str], receive: Callable[[], Any],
                     send: Callable[[dict[str, Any]], Any]) -> None:
        pieces: queue.Queue[bytes | None] = queue.Queue()
        answer = asyncio.get_running_loop().run_in_executor(None, lambda: serve(first_line, headers, iter(pieces.get, None)))
        while not answer.done():
            message = asyncio.ensure_future(receive())
            await asyncio.wait({message, answer}, return_when=asyncio.FIRST_COMPLETED)
            if not message.done():
                message.cancel()
                break
            got = message.result()
            if got.get("type") == "http.disconnect":
                break
            if body := got.get("body", b""):
                pieces.put(body)
            if not got.get("more_body", False):
                break
        pieces.put(None)
        status, text = await answer
        await _answer(send, status, text)
