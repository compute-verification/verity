"""``--admit tick`` under vLLM's own OpenAI server, without serve-openai (`serve_admit`'s certified tick): `TickFeed`, the
server's ASGI ``--middleware``, appends the network certifier's releases and its watermark to a feed file, and `register`, a
vLLM general plugin, puts a `serve_admit.TickGate` that reads the feed (`FeedReader`) in front of every V1 scheduler the
engine core process builds, and makes the engine core's input loop wait on it. Both read ``VERITY_TICK`` (`tick_config`).
The trace corpus serves so (`benchmarks/network_e2e/traces/corpus.py`, ``--admit tick``).

vLLM loads its general plugins in each of its processes, the API server's included, so this module imports only the
stdlib, `verity_vllm.settings`, `verity_vllm.engine.hooks` and `serve_admit`, and `register` imports vLLM only once
``VERITY_TICK`` is set. `serve_admit` reads a watermark line with the network certifier's own reader
(`verity.protocols.certifier.verifier.release`, which imports only the stdlib), and imports it when the first line
comes, so only the API server's process does, and it needs ``verity``, this package's dependency, on its path.
"""
from __future__ import annotations

import json
import os
import queue
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline.serve_admit import RELEASE_HEADER, RUN_HEADER, WATERMARK_PATH, Tick, TickGate, Watermark, install, release_of

__all__ = ["OTHER_PATHS", "SAMPLED_PATHS", "TICK_ENV", "FeedReader", "TickFeed", "register", "route_of", "tick_config",
           "unseeded"]

#: `register`'s and `TickFeed`'s configuration, a JSON object (`tick_config`)
TICK_ENV = "VERITY_TICK"
#: the OpenAI server's routes whose request body carries sampling parameters, which `TickFeed` reads (`unseeded`)
SAMPLED_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
#: the routes `TickFeed` passes unread besides `SAMPLED_PATHS` and the watermark's: none reaches the engine
OTHER_PATHS = frozenset({"/health", "/ping", "/version", "/v1/models", "/metrics", "/metrics/", "/tokenize", "/detokenize"})


def tick_config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """``VERITY_TICK``'s JSON object (in ``environ``, by default the process environment), None where it isn't set:
    ``bucket_us``, ``sat_us`` and ``free_us`` (the `serve_admit.Tick`), ``hold_ms`` (5000 by default), ``feed`` (the file
    `TickFeed` appends to and `register`'s gate reads) and ``log`` (a directory for each gate's ``tick.PID.jsonl``,
    optional)."""
    cfg = settings.get(TICK_ENV, environ)
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or not cfg.get("feed"):
        raise ValueError(f"{TICK_ENV}: a JSON object with bucket_us, sat_us, free_us and feed, not {json.dumps(cfg)[:80]!r}")
    return {"tick": Tick(cfg.get("bucket_us"), cfg.get("sat_us"), cfg.get("free_us")),
            "hold_s": float(cfg.get("hold_ms", 5000)) / 1000, "feed": str(cfg["feed"]), "log": cfg.get("log")}


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


def unseeded(body: bytes) -> str | None:
    """Why `TickFeed` refuses a completions or chat request ``body``, None where it passes. A stochastic request must carry
    an integer ``seed``: vLLM gives one without a seed a seed it draws from ``np.random``, which nobody can recompute.
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
    return (f"under the certified tick a request sampled at {at} needs an integer seed, not {json.dumps(seed)[:40]}: "
            "send one, or temperature 0")


def route_of(scope: Mapping[str, Any]) -> str:
    """An ASGI request's route: its ``path`` less the server's ``root_path`` where the path starts with it."""
    path, root = scope.get("path") or "", (scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        return path[len(root):] or "/"
    return path


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


class TickFeed:
    """vLLM's OpenAI server's side of ``--admit tick`` (an ASGI ``--middleware``): with ``VERITY_TICK`` set, it appends to
    the feed file the records `serve_admit.TickGate.take` reads: a numbered request's release before the server sees the
    request (its key the request's X-Request-Id, ``verity-<run>-<N>`` set where the client sent none), that it won't reach
    the engine (no 200 response began), and the watermark's lines from its ``POST /verity/watermark``, which it answers
    itself. A malformed or repeated release and a malformed watermark are answered 400.

    It serves an allow-list, matched on the route less the server's ``root_path`` (`route_of`): ``/v1/completions`` and
    ``/v1/chat/completions`` (`SAMPLED_PATHS`), the watermark's, and ``/health``, ``/ping``, ``/version``, ``/v1/models``,
    ``/metrics``, ``/tokenize`` and ``/detokenize`` (`OTHER_PATHS`), which don't generate. Any other route
    (``/v1/responses``, ``/v1/audio/...``, ``/pooling``, ``/invocations``, ...) is answered 400, and a websocket closed,
    before the engine sees it. So is a completions or chat request sampled without a seed (`unseeded`): serve-openai fills
    such a request's seed with the public default its registration fixes (`engine.served_seed`); this path has no
    registered default, so it refuses instead. A refused numbered request is fed as passed. Without the variable it
    passes everything."""

    def __init__(self, app: Any) -> None:
        self.app = app
        cfg = tick_config()
        self.fd = None if cfg is None else os.open(cfg["feed"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        self.seen: dict[str, set[int]] = {}

    def write(self, rec: dict[str, Any]) -> None:
        os.write(self.fd, (json.dumps(rec, separators=(",", ":")) + "\n").encode())

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Any], send: Callable[[dict[str, Any]], Any]) -> None:
        if self.fd is None or scope.get("type") not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await receive()
            return await send({"type": "websocket.close", "code": 1008})
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or ()}
        route = route_of(scope)
        if route == WATERMARK_PATH:
            return await self._watermark(headers.get(RUN_HEADER, "").strip(), receive, send)
        try:
            rel = release_of(headers)
            if rel is not None and rel[1] in self.seen.setdefault(rel[0], set()):
                raise ValueError(f"{RELEASE_HEADER}: {rel[1]} came twice in run {rel[0]!r}")
        except ValueError as e:
            return await _answer(send, 400, str(e))
        why = None
        if route in SAMPLED_PATHS:
            body, receive = await _buffered(receive)
            why = unseeded(body)
        elif route not in OTHER_PATHS:
            why = (f"under the certified tick this server serves only {', '.join(sorted(SAMPLED_PATHS | OTHER_PATHS))}, "
                   f"not {route[:80]!r}")
        if why is not None:
            if rel is not None:
                self.seen[rel[0]].add(rel[1])
                self.write({"run": rel[0], "n": rel[1], "passed": True})
            return await _answer(send, 400, why)
        if rel is None:
            return await self.app(scope, receive, send)
        run, n, bucket = rel
        self.seen[run].add(n)
        key = headers.get("x-request-id")
        if not key:
            key = f"verity-{run}-{n}"
            scope = {**scope, "headers": [*scope.get("headers", ()), (b"x-request-id", key.encode())]}
        self.write({"key": key, "run": run, "n": n, "bucket": bucket})
        began = False

        async def watched(message: dict[str, Any]) -> None:
            nonlocal began
            if message.get("type") == "http.response.start" and message.get("status") == 200:
                began = True
            await send(message)

        try:
            await self.app(scope, receive, watched)
        finally:
            if not began:
                self.write({"run": run, "n": n, "passed": True})

    async def _watermark(self, run: str, receive: Callable[[], Any], send: Callable[[dict[str, Any]], Any]) -> None:
        w = Watermark(run, self.write)
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                w.close("the connection closed before the watermark's end")
                return
            if not w.feed(message.get("body", b"")):
                break
            if not message.get("more_body", False):
                w.close()
                break
        await _answer(send, 200 if w.error is None else 400, w.error or "")


def _log_to(directory: str | None) -> Callable[[dict[str, Any]], None] | None:
    if not directory:
        return None
    f = open(Path(directory) / f"tick.{os.getpid()}.jsonl", "a", encoding="utf-8")

    def log(rec: dict[str, Any]) -> None:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
    return log


def _gated_init(init: Callable[..., None], cfg: Mapping[str, Any]) -> Callable[..., None]:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        gate = TickGate(cfg["tick"], hold_s=cfg["hold_s"], pull=FeedReader(cfg["feed"]), log=_log_to(cfg["log"]))
        self._verity_tick_hooks = install(self, gate)
    return __init__


def _gated_input(orig: Callable[[Any], None]) -> Callable[[Any], None]:
    """`EngineCoreProc._process_input_queue` under a gate: it returns once the gate is ready and the engine has work,
    handling client requests meanwhile, and waits on the input queue at most ``poll_s`` at a time while the gate waits."""
    def _process_input_queue(self: Any) -> None:
        gate = getattr(self.scheduler, "_verity_tick", None)
        if gate is None:
            return orig(self)
        while True:
            while not self.input_queue.empty():
                self._handle_client_request(*self.input_queue.get_nowait())
            if not self.is_running():
                gate.open("the engine core is shutting down")
                return
            ok = gate.ready(self.scheduler)
            if ok and self.has_work():
                return
            if ok:
                self._notify_idle_state_callbacks()
                if self.input_queue.empty():
                    with self.aborts_queue.mutex:
                        self.aborts_queue.queue.clear()
            try:
                req = self.input_queue.get(timeout=None if ok else gate.poll_s)
            except queue.Empty:
                continue
            self._handle_client_request(*req)
    return _process_input_queue


def _gated_has_work(orig: Callable[[Any], bool]) -> Callable[[Any], bool]:
    def has_work(self: Any) -> bool:
        gate = getattr(self.scheduler, "_verity_tick", None)
        return orig(self) or (gate is not None and gate.pending())
    return has_work


_PLUGIN: list[Hooks] = []


def register() -> None:
    """``--admit tick`` under vLLM's own OpenAI server, as a vLLM general plugin (entry point group
    ``vllm.general_plugins``, run in each of its processes): with ``VERITY_TICK`` set (`tick_config`), every V1 scheduler
    built in this process gets a `serve_admit.TickGate` that reads the feed (`serve_admit.install`, keyed by the client's
    X-Request-Id), and the engine core process's input loop waits on it (`EngineCoreProc._process_input_queue` and
    ``has_work``). Idempotent."""
    cfg = tick_config()
    if cfg is None or _PLUGIN:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc
    hooks = Hooks(tag="admit-tick")
    hooks.wrap(Scheduler, "__init__", lambda init: _gated_init(init, cfg))
    hooks.wrap(EngineCoreProc, "_process_input_queue", _gated_input)
    hooks.wrap(EngineCoreProc, "has_work", _gated_has_work)
    _PLUGIN.append(hooks)
