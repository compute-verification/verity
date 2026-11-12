"""vLLM's benchmark client's shim to the network certifier (`serve_tax`'s ``tick-certifier`` arm).

vLLM's client (``vllm bench serve``) speaks HTTP: a streamed ``POST /v1/completions`` whose prompt is text or token ids.
The network certifier (`verity.protocols.certifier.verifier.warden`) speaks fixed-size cells, one request a connection,
and in its id egress form takes a canonical token-id request (`active.id_request`) and answers one id event a ``DATA``
cell (`io_tie.id_event`), then an ``END`` that holds `io_tie.TRAILER`. `HttpShim` listens between them on the client's
side of the certifier, the client's thin shim (`warden.cells`, `warden.Shim`) with an HTTP face:

- a completion's body becomes the id request of its fields (`id_fields`): its model, its prompt as token ids (a text
  prompt encoded by the served model's tokenizer, as vLLM's server encodes one), ``max_tokens``, ``temperature``,
  ``top_p``, ``seed`` and, as its ``user``, the request's X-Request-Id. The id request has no other field, so a body
  that asks for anything else that changes its answer is refused (400) rather than sent as a different request.
  ``ignore_eos`` is the one such field, and it is taken, unsent, only as true: the arm's server decodes every request
  to its ``max_tokens`` (`servetax_plugin`, ``to_length``), which is what that field asks of stock vLLM, so a body
  that doesn't ask for it would get an answer it didn't ask for;
- the request goes to the certifier in cells on a connection of its own, and each ``DATA`` cell that comes back is
  written at once as one streamed chunk with the cell's token id and no text, so the client's first-token and
  inter-token times are the times the certifier's cells arrive; the ``END`` is the usage chunk and ``[DONE]``;
- a connection the certifier refuses is a 503 and one it fails or cuts before its first id a 502; one that fails or is
  cut after an id is closed without the stream's last chunk, so the client counts it failed and never as a short answer.

Each connection's record is one line of ``--log``, in the keys of `warden_serve.connection_log` (every cell's kind and
arrival after the certifier's origin, the request's and the response's digests), which is what the certifier's offline
check reads as its ``client.json`` (`client_doc`), with the request's id, its token counts and the HTTP status beside
them. A request the shim did not send has a line too, with why (``unsent``): one it refused, and one it never read
whole, its client having closed first. No prompt and no token id is kept.

    python cell_shim.py --deployment WARDEN_OUT/deployment.json --tokenizer CHECKPOINT --log shim.jsonl
        [--listen 127.0.0.1:0]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from verity.protocols.certifier.verifier import io_tie as IT  # noqa: E402 (the tree's, put on sys.path above)
from verity.protocols.certifier.verifier.active import id_request  # noqa: E402
from verity.protocols.certifier.verifier.warden import (DATA, END, FAIL, KIND_NAMES, PAD, REFUSED, cells,  # noqa: E402
                                                        read_cell)

#: the longest request body taken
BODY_BYTES = 1 << 24
#: what an id request holds for the sampling fields a greedy body doesn't name: neither changes a greedy answer
SENT = {"top_p": 1.0, "seed": 0}
#: the fields taken and not sent: how the answer is framed (the shim streams, and counts the usage itself), and
#: ``ignore_eos``, which is true (the module docstring)
FRAMING = ("stream", "stream_options", "ignore_eos")
#: the fields taken only at the value that changes nothing
NEUTRAL = {"repetition_penalty": 1.0, "logprobs": None, "n": 1, "echo": False, "stop": []}
STATUS = {200: b"OK", 400: b"Bad Request", 404: b"Not Found", 502: b"Bad Gateway", 503: b"Service Unavailable"}
#: why a connection got no answer, by its outcome
WHY = {"refused": (503, "the network certifier refused the connection: every lane and its queue were taken"),
       "fail": (502, "the network certifier failed the connection"),
       "cut": (502, "the network certifier closed the connection before its end"),
       "malformed": (502, "the network certifier sent a cell that is none of its own")}


class Refused(Exception):
    """A request the shim does not send, with the HTTP status it answers."""

    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status


def id_fields(body: Any, request_id: str | None, encode: Callable[[str], list[int]]) -> dict[str, Any]:
    """`active.id_request`'s keywords (but its frame) for a completion's body, or `Refused` (400) for a body the id
    request can't carry as it is: one that isn't a streamed greedy completion of one prompt to its ``max_tokens``
    (``ignore_eos``), or that names a field that changes its answer and that the id request has no place for. A greedy
    body is one whose ``temperature`` is 0: a body that names none leaves it to the server's default, which the id
    request, always stating one, can't."""
    if not isinstance(body, dict):
        raise Refused(400, "the body is a JSON object")
    if body.get("stream") is not True:
        raise Refused(400, "the network certifier's id form answers a stream: stream is true")
    if body.get("temperature") != 0 or isinstance(body["temperature"], bool):
        raise Refused(400, "the shim sends greedy requests alone: temperature is 0")
    if body.get("ignore_eos") is not True:
        raise Refused(400, "the server behind the network certifier decodes each request to its max_tokens: "
                           "ignore_eos is true")
    for name, same in NEUTRAL.items():
        if body.get(name, same) not in (same, None):
            raise Refused(400, f"an id request has no {name}: it is taken only as {same!r}")
    unknown = sorted(set(body) - {"model", "prompt", "max_tokens", "temperature", "user", *SENT, *FRAMING, *NEUTRAL})
    if unknown:
        raise Refused(400, f"an id request has no {', '.join(unknown)}: its fields are model, prompt, max_tokens, "
                           f"temperature, {', '.join(SENT)} and user")
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        prompt = encode(prompt)
    if not (isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt)):
        raise Refused(400, "prompt is one text or one list of token ids")
    sent = {k: default if body.get(k) is None else body[k] for k, default in SENT.items()}
    return {"model": body.get("model"), "user": body.get("user") or request_id or "client", "prompt": prompt,
            "max_tokens": body.get("max_tokens"), "temperature": 0.0, **sent}


def event_id(payload: bytes) -> int | None:
    """The token id of a ``DATA`` cell's payload, or None for bytes that aren't its one id event (`io_tie.id_event` at
    the payload's length)."""
    digits = payload[len(IT.ID_EVENT_HEAD):len(IT.ID_EVENT_HEAD) + IT.ID_DIGITS]
    if not digits.lstrip(b" ").isdigit() or int(digits) >= 1 << 32:
        return None
    return int(digits) if IT.id_event(int(digits), len(payload)) == payload else None


def _sse(doc: dict) -> bytes:
    return b"data: " + json.dumps(doc, separators=(",", ":")).encode() + b"\n\n"


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n" % len(data) + data + b"\r\n"


def _head(status: int, headers: list[tuple[bytes, bytes]]) -> bytes:
    lines = [b"HTTP/1.1 %d %s" % (status, STATUS[status]), *(k + b": " + v for k, v in headers), b"connection: close"]
    return b"\r\n".join(lines) + b"\r\n\r\n"


def _error(status: int, why: str) -> bytes:
    body = json.dumps({"error": {"message": why, "code": status}}).encode()
    return _head(status, [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))]) + body


class HttpShim:
    """The listener (the module docstring). ``connect`` opens a connection to the network certifier, ``encode`` is the
    served model's tokenizer on a text, ``log`` takes each connection's record, and ``clock`` reads the time the
    records' arrivals are in, ``origin`` the certifier's on that clock."""

    def __init__(self, connect: Callable[[], Awaitable[tuple[Any, Any]]], cell_bytes: int,
                 encode: Callable[[str], list[int]], log: Callable[[dict], None] = lambda record: None,
                 clock: Callable[[], float] = time.time, origin: float = 0.0) -> None:
        self.connect, self.cell_bytes, self.encode, self.log = connect, cell_bytes, encode, log
        self.clock, self.origin = clock, origin
        self.connections = 0

    def _ms(self, t: float) -> float:
        return round((t - self.origin) * 1000, 3)

    def unread(self, why: str, status: int | None = None) -> None:
        """The record of a connection no request was read from, so none was sent: why, and the status answered, None
        where its client had closed."""
        self.log({"index": None, "id": None, "status": status, "unsent": why})

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        """One HTTP connection: one request, answered, then closed."""
        try:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError as e:
                self.unread(f"the client closed {len(e.partial)} bytes into its request's head")
                return
            except asyncio.LimitOverrunError as e:
                writer.write(_error(400, "a request's head is longer than this listener reads"))
                self.unread(f"a request's head had no end in its first {e.consumed} bytes", 400)
                return
            line, *rest = head[:-4].split(b"\r\n")
            headers = {k.strip().lower(): v.strip() for k, _, v in (x.partition(b":") for x in rest)}
            method, _, target = line.partition(b" ")
            path = target.partition(b" ")[0].partition(b"?")[0]
            if (method, path) != (b"POST", b"/v1/completions"):
                writer.write(_error(404, "this shim takes POST /v1/completions alone"))
                return
            size = headers.get(b"content-length", b"")
            if not size.isdigit() or int(size) > BODY_BYTES:
                writer.write(_error(400, f"a completion's body has a content-length of at most {BODY_BYTES}"))
                return
            try:
                raw = await reader.readexactly(int(size))
            except asyncio.IncompleteReadError as e:
                self.unread(f"the client closed {len(e.partial)} bytes into its request's body of {int(size)}")
                return
            await self.completion(raw, headers.get(b"x-request-id", b"").decode("latin-1") or None, writer)
        finally:
            writer.close()  # what is written and not yet sent goes out before the connection closes

    async def completion(self, raw: bytes, request_id: str | None, writer: Any) -> None:
        index, self.connections = self.connections, self.connections + 1
        try:
            try:
                body = json.loads(raw)
            except ValueError:
                raise Refused(400, "the body is not JSON") from None
            fields = id_fields(body, request_id, self.encode)
            try:
                request = id_request(**fields, frame_bytes=self.cell_bytes, offset=0)
            except (ValueError, TypeError) as e:
                raise Refused(400, f"not an id request: {e}") from None
        except Refused as e:
            writer.write(_error(e.status, str(e)))
            self.log({"index": index, "id": request_id, "status": e.status, "unsent": str(e)})
            return
        sent = self.clock()
        record = {"index": index, "id": request_id, "at": round(sent - self.origin, 6), "sent_ms": self._ms(sent),
                  "request_bytes": len(request), "request_sha256": hashlib.sha256(request).hexdigest(),
                  "prompt_tokens": len(fields["prompt"]), "max_tokens": fields["max_tokens"]}
        kinds, arrivals, digest, data_bytes, ids, ordinal = [], [], hashlib.sha256(), 0, 0, None
        outcome = status = None  # the connection's outcome as `warden.Shim` reads one, and the HTTP status answered
        name = "cmpl-" + (request_id or str(index))
        stream = _head(200, [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
                             (b"transfer-encoding", b"chunked")])

        def chunk(**doc: Any) -> bytes:
            return _chunk(_sse({"id": name, "object": "text_completion", "model": fields["model"], **doc}))

        def answer(of: str) -> int:
            """The rest of the HTTP answer once the connection's outcome is `of`: the stream's end, an error where no
            id was sent, and nothing after one, which leaves the stream without its last chunk."""
            if of == "end":
                usage = {"prompt_tokens": len(fields["prompt"]), "completion_tokens": ids,
                         "total_tokens": len(fields["prompt"]) + ids}
                writer.write((b"" if ids else stream) + chunk(choices=[], usage=usage)
                             + _chunk(b"data: [DONE]\n\n") + _chunk(b""))
            elif not ids:
                writer.write(_error(*WHY[of]))
            return 200 if ids or of == "end" else WHY[of][0]

        up = None
        try:
            reader, up = await self.connect()
            up.write(cells(request, self.cell_bytes))
            while True:
                try:
                    kind, ordinal, payload = read_cell(await reader.readexactly(self.cell_bytes), self.cell_bytes)
                except asyncio.IncompleteReadError as e:
                    if e.partial or outcome is None:
                        outcome = "cut"
                    break
                except ValueError:
                    outcome = "malformed"
                    break
                kinds.append(KIND_NAMES[kind][0])
                arrivals.append(self._ms(self.clock()))
                if outcome is not None:  # after its end a connection holds padding alone, until it closes
                    if kind != PAD:
                        outcome = "cut"
                elif kind in (FAIL, REFUSED):
                    outcome = KIND_NAMES[kind]
                    status = answer(outcome)
                elif kind != PAD:
                    token = event_id(payload) if kind == DATA else None
                    if (kind == DATA and token is None) or (kind == END and payload != IT.TRAILER):
                        outcome = "malformed"
                        break
                    digest.update(payload)
                    data_bytes += len(payload)
                    if kind == DATA:
                        writer.write((b"" if ids else stream) + chunk(choices=[
                            {"index": 0, "text": "", "token_ids": [token], "finish_reason": None}]))
                        ids += 1
                    else:
                        outcome = "end"
                        status = answer(outcome)
                    await writer.drain()
        except OSError:
            outcome = "cut"
        finally:
            if up is not None:
                up.close()
        if status is None:
            status = answer(outcome)
        self.log(record | {
            "ordinal": ordinal, "outcome": outcome, "kinds": "".join(kinds), "arrivals_ms": arrivals,
            "data_bytes": data_bytes, "data_sha256": digest.hexdigest(), "status": status,
            "response": {"status": None, "ids": ids if outcome == "end" else None, "complete": outcome == "end",
                         "events": ids, "text_bytes": 0, "finish_reason": None, "prompt_tokens": None,
                         "completion_tokens": ids}})


def client_doc(lines: list[dict]) -> dict:
    """The shim's records as the ``client.json`` the certifier's offline check reads (`warden_serve.check`): one
    connection each request it sent, in the order sent."""
    return {"connections": [x for x in lines if "unsent" not in x]}


def tokenizer(checkpoint: str) -> Callable[[str], list[int]]:
    """The served model's tokenizer on a text prompt, as vLLM's completions endpoint encodes one: with its special
    tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint)
    return lambda text: list(tok(text, add_special_tokens=True)["input_ids"])


def address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host, int(port)


async def serve(a: argparse.Namespace) -> None:
    dep = json.loads(Path(a.deployment).read_text())
    certifier, cell_bytes = tuple(dep["listen"]), dep["params"]["cell_bytes"]
    with open(a.log, "a", buffering=1) as fh:
        shim = HttpShim(lambda: asyncio.open_connection(*certifier), cell_bytes, tokenizer(a.tokenizer),
                        lambda record: fh.write(json.dumps(record) + "\n"), origin=dep["origin_unix"])
        server = await asyncio.start_server(shim.handle, *a.listen, limit=1 << 20, backlog=4096)
        print(json.dumps({"listening": list(server.sockets[0].getsockname()[:2])}), flush=True)
        async with server:
            await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deployment", required=True, help="the network certifier's deployment.json (warden_serve.py serve)")
    ap.add_argument("--tokenizer", required=True, help="the served model's checkpoint, for text prompts")
    ap.add_argument("--log", required=True, help="each connection's record, a line of JSON")
    ap.add_argument("--listen", type=address, default=("127.0.0.1", 0), help="HOST:PORT (port 0: any)")
    asyncio.run(serve(ap.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
