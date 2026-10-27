"""The API server's side of a request's arrival: an ASGI middleware for vLLM's OpenAI server
(``--middleware vllm_accept_stamp.AcceptStamp``) that appends one line per HTTP request to ``accept.PID.jsonl`` under
``VERITY_TRACE_DIR``, where `vllm_trace_plugin` writes the engine core's records:

- ``id``, the request's X-Request-Id (the client's, as the tracer's records carry it), ``path`` and ``peer``;
- ``accept``: unix seconds when uvicorn accepted the TCP connection the request came on (its protocol's
  ``connection_made``), and ``conn_request``, the request's place on that connection (1 for its first; a later one's
  ``accept`` is the connection's, not its own);
- ``headers``: when the app got the request (uvicorn had parsed its headers), and ``body``: when the app had read the
  whole body, or null if it never did.

Importing the module wraps ``connection_made`` of uvicorn's two HTTP protocols (httptools' and h11's) to record each
connection's accept time by its peer address; vLLM imports it while it builds the app, before it serves. An error in the
middleware is written to ``accept_errors.PID.txt`` and never reaches the request."""
from __future__ import annotations

import importlib
import json
import os
import time
import traceback
from pathlib import Path

TRACE_DIR = "VERITY_TRACE_DIR"
PROTOCOLS = (("uvicorn.protocols.http.httptools_impl", "HttpToolsProtocol"),
             ("uvicorn.protocols.http.h11_impl", "H11Protocol"))
MAX_ERRORS = 20
#: peer address -> [accept time, requests seen on the connection]
ACCEPTS: dict[tuple, list] = {}


def stamp_accepts(cls) -> None:
    """Wrap `cls.connection_made` (once) so that it records the connection's accept time under its peer address."""
    made = cls.connection_made
    if getattr(made, "_accept_stamp", False):
        return

    def connection_made(self, transport, *args, **kwargs):
        t = time.time()
        try:
            peer = transport.get_extra_info("peername")
            if peer:
                ACCEPTS[tuple(peer[:2])] = [t, 0]
        except Exception:  # noqa: BLE001 -- the connection must be served
            pass
        return made(self, transport, *args, **kwargs)

    connection_made._accept_stamp = True
    cls.connection_made = connection_made


def install() -> None:
    for module, name in PROTOCOLS:
        try:
            cls = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
        stamp_accepts(cls)


class AcceptStamp:
    """The middleware: the module docstring's record of each HTTP request, written when its response is done."""

    def __init__(self, app):
        self.app = app
        d = os.environ.get(TRACE_DIR)
        self.out = Path(d) if d else None
        self.file = None
        self.errors = 0

    def error(self, where: str) -> None:
        self.errors += 1
        if self.errors <= MAX_ERRORS and self.out is not None:
            with open(self.out / f"accept_errors.{os.getpid()}.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {where} (error {self.errors})\n{traceback.format_exc()}\n")

    def write(self, rec: dict) -> None:
        if self.file is None:
            self.out.mkdir(parents=True, exist_ok=True)
            self.file = open(self.out / f"accept.{os.getpid()}.jsonl", "a", encoding="utf-8")
        self.file.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.file.flush()

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or self.out is None:
            return await self.app(scope, receive, send)
        rec = {"id": None, "path": scope.get("path"), "peer": None, "accept": None, "conn_request": None,
               "headers": time.time(), "body": None}
        try:
            peer = tuple(scope.get("client") or ())
            rec["peer"] = list(peer)
            conn = ACCEPTS.get(peer)
            if conn is not None:
                conn[1] += 1
                rec["accept"], rec["conn_request"] = conn
            rec["id"] = next((v.decode("latin-1") for k, v in scope.get("headers") or () if k.lower() == b"x-request-id"),
                             None)
        except Exception:  # noqa: BLE001
            self.error("scope")

        async def stamped():
            msg = await receive()
            if rec["body"] is None and msg.get("type") == "http.request" and not msg.get("more_body"):
                rec["body"] = time.time()
            return msg

        try:
            return await self.app(scope, stamped, send)
        finally:
            try:
                self.write(rec)
            except Exception:  # noqa: BLE001
                self.error("write")


install()
