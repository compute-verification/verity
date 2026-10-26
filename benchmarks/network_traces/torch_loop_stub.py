"""torch-loop-stub: an OpenAI streaming endpoint shaped like the torch loop (`integrations/torch_loop`), the upstream of the
network certifier's rehearsal (`certifier_serve.py rehearse --upstream URL --traffic varied --compact 256`).

    python torch_loop_stub.py [--host 127.0.0.1] [--port 0] [--steps 2] [--h 128] [--seed 0] [--id-payload 503]

It prints ``{"listening": URL}`` once it listens, then serves until it is stopped, each connection on its own:

- ``GET /v1/models``: one model, ``torch-loop/mlp-w8a8-h{H}``;
- ``POST /tokenize``: one token a character, its code mod `loop.VOCAB`, so every id is a row of the loop's embedding;
- ``POST /v1/completions``, streamed (``transfer-encoding: chunked``, one ``data:`` event a chunk, then ``data: [DONE]``):
  the request's last prompt id through the loop's model (`loop.Model(h, seed)`: embedded, then each decode step the
  W8A8 MLP over the step before, as `loop.TorchLoop.run` serves a request) for min(``max_tokens``, ``--steps``) steps,
  one chunk a step. A step's output is the MLP's H bf16 words, 2H bytes (256 at H = 128, little-endian): the chunk's
  ``text`` is their hex, and its one token id the step's (`step_id`). The first chunk carries ``prompt_token_ids``, the
  last ``finish_reason`` "length". A request is greedy (temperature 0, top_p 1) and either varied, asking for its
  token ids (``return_token_ids``) with no stop strings, as the certifier's shim forwards a varied request
  (`certifier_serve.varied_body`), or fixed, exactly its ``max_tokens`` (``ignore_eos`` and no ``min_tokens``, at
  most ``--steps``), as a fixed plan's (`certifier_serve.request_body`), whose shim frames each chunk's text, 512 hex
  characters at H = 128, as 8 canonical events of 64 bytes (`active.TOKEN_BYTES`); any other is answered 400.
- a warden's id request (`id_request_of`: `active.id_request` exactly, in the scan's B = P + 9-byte cells from their
  first byte, the release headers aside, as the gate's `warden-rehearse --scan --prompts` sends it), greedy, its last
  prompt id below `loop.VOCAB`: answered id-form, the bytes the warden puts in its egress cells as they come, with no
  HTTP head: each step's `io_tie.id_event` of P = ``--id-payload`` bytes (`scan_id`, below 2^17, the scan's IDB),
  then `io_tie.TRAILER` (`id_response`), as vLLM's ``--protocol ids`` answers the warden's.

``--tamper KIND`` makes the server a covert sender on one id request's answer, the ``--tamper-at``-th id request it
serves (from 0, in arrival order), every other answered as before (`TAMPERS`, `tampered`): ``byte``, one output byte
changed before its id is taken; ``extra``, one id event more before the trailer; ``late``, the first event at once and
the rest ``--late-s`` seconds later, past the warden's release of the next cell, so the warden fails the connection at
that slot.

`test_certifier_serve.py`'s ``_Varied`` is the same endpoint with varied responses and EOS.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

from verity.core.protocols.certifier import active as A, release as REL
from verity.ml.protocols.certifier import io_tie as IT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations" / "torch_loop"))
from verity_torch_loop import loop as L  # noqa: E402

#: a compact record's token field holds an id below 2^30 (`active.COMPACT_STOP`)
ID_BITS = 30
#: an id-form event's id is below 2^17, the scan's IDB (`scan_zk.CLASS`)
SCAN_ID_BITS = 17
#: the scan's cell payload, P = B − 9 at its B = 512-byte cells: ``--id-payload``'s default
ID_PAYLOAD = 503
#: ``--tamper``'s kinds (module docstring)
TAMPERS = ("byte", "extra", "late")
#: ``late``'s hold, past the scan schedule's D = 21 buckets of 100 ms from admission to a connection's first egress cell
LATE_S = 3.0


def step_outputs(model: L.Model, token: int, steps: int) -> list[bytes]:
    """The loop's request of ``token`` for ``steps`` decode steps: each step's output, the second linear's H bf16
    words (`loop.bits`) as 2H bytes."""
    x, out = model.embed[token][None, :], []
    for _ in range(steps):
        for layer, lin in enumerate(model.layers):
            xq, sx = L.quantize(x)
            y = L.linear(xq, sx, lin.wq, lin.sw)
            x = torch.relu(y.to(torch.float32)) if layer == 0 else y.to(torch.float32)
        out.append(L.bits(y).reshape(model.h).astype("<u2").tobytes())
    return out


def step_id(output: bytes) -> int:
    """A step's token id: its output's SHA-512, the first four bytes as a u32be, below 2^`ID_BITS`."""
    return int.from_bytes(hashlib.sha512(output).digest()[:4], "big") & ((1 << ID_BITS) - 1)


def scan_id(output: bytes) -> int:
    """A step's id in id form: `step_id`'s low `SCAN_ID_BITS` bits."""
    return step_id(output) & ((1 << SCAN_ID_BITS) - 1)


def id_request_of(request: bytes, payload: int) -> dict | None:
    """An id request's body as the warden forwards it (`request`: its head and body as read), or None: without the
    release headers the warden puts on the server's copy (`release.with_release`), exactly `active.id_request` of its
    own fields in B = `payload` + 9-byte cells, from their first byte (``offset`` 0)."""
    head, _, raw = request.partition(b"\r\n\r\n")
    line, *fields = head.split(b"\r\n")
    kept = b"".join(f + b"\r\n" for f in fields
                    if f.partition(b":")[0].strip().lower() not in (REL.RELEASE_HEADER, REL.RUN_HEADER))
    try:
        body = json.loads(raw)
        got = {k: body[k] for k in ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")}
        canonical = A.id_request(**got, frame_bytes=payload + A.HEADER.size, offset=0)
    except (ValueError, KeyError, TypeError):
        return None
    return body if canonical == line + b"\r\n" + kept + b"\r\n" + raw else None


def id_refusal(body: dict) -> str | None:
    """Why the endpoint answers an id request 400 (None: it serves it): greedy, its last prompt id below `loop.VOCAB`,
    at least one token."""
    if body["temperature"] != 0 or body["top_p"] != 1:
        return "greedy (temperature 0, top_p 1)"
    if not 0 <= body["prompt"][-1] < L.VOCAB or body["max_tokens"] < 1:
        return f"the last prompt id below {L.VOCAB} and max_tokens >= 1"
    return None


def id_response(outputs: list[bytes], payload: int) -> bytes:
    """An id request's answer: each step's `scan_id` as an `io_tie.id_event` of `payload` bytes, then
    `io_tie.TRAILER`."""
    return b"".join(IT.id_event(scan_id(o), payload) for o in outputs) + IT.TRAILER


def tampered(outputs: list[bytes], kind: str | None) -> list[bytes]:
    """The outputs a ``byte`` or ``extra`` answer takes its ids from: the first output's first byte flipped, or the
    last output once more; any other kind's are the outputs."""
    if kind == "byte":
        return [bytes([outputs[0][0] ^ 1]) + outputs[0][1:], *outputs[1:]]
    if kind == "extra":
        return [*outputs, outputs[-1]]
    return list(outputs)


def refusal(body: dict, steps: int | None = None) -> str | None:
    """Why the endpoint answers 400 (None: it serves the body): a varied body (module docstring), or a fixed one
    (`certifier_serve.request_body`: exactly ``max_tokens`` tokens, ``ignore_eos`` and no ``min_tokens``, at most the
    endpoint's ``steps``). A body with ``min_tokens`` is refused: it masks the EOS logit, which the served Program's
    sampling doesn't model."""
    if "min_tokens" in body:
        return "no min_tokens (it masks the EOS logit, which the served Program's sampling doesn't model)"
    fixed = body.get("ignore_eos") is True
    if not fixed:
        if body.get("return_token_ids") is not True:
            return "return_token_ids must be true"
        if "ignore_eos" in body:
            return "ignore_eos true (a fixed body) or absent (a varied one)"
        if body.get("stop") != [] or type(body.get("seed")) is not int:
            return "stop [] and an int seed"
    if body.get("temperature") != 0 or body.get("top_p") != 1:
        return "greedy (temperature 0, top_p 1)"
    prompt = body.get("prompt")
    if not (isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt)):
        return "the prompt is a list of token ids"
    if not 0 <= prompt[-1] < L.VOCAB or type(body.get("max_tokens")) is not int or body["max_tokens"] < 1:
        return f"the last prompt id below {L.VOCAB} and max_tokens >= 1"
    if fixed and steps is not None and body["max_tokens"] > steps:
        return f"a fixed body's max_tokens at most the loop's {steps} steps"
    return None


class Endpoint:
    def __init__(self, model: L.Model, steps: int, id_payload: int = ID_PAYLOAD, tamper: str | None = None,
                 tamper_at: int = 0, late_s: float = LATE_S) -> None:
        IT.id_event(0, id_payload)
        if tamper is not None and tamper not in TAMPERS:
            raise ValueError(f"--tamper: one of {TAMPERS}, not {tamper!r}")
        self.model, self.steps, self.name = model, steps, f"torch-loop/mlp-w8a8-h{model.h}"
        self.id_payload, self.tamper, self.tamper_at, self.late_s = id_payload, tamper, tamper_at, late_s
        self.id_requests = 0

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self.handle(reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"(?im)^content-length:\s*(\d+)", head)
        raw = await reader.readexactly(int(n.group(1))) if n else b""
        body = json.loads(raw) if n else None
        if head.startswith(b"POST /v1/completions ") and (ids := id_request_of(head + raw, self.id_payload)):
            if (why := id_refusal(ids)) is not None:
                return await self.json(writer, 400, {"error": why})
            outputs = step_outputs(self.model, ids["prompt"][-1], min(ids["max_tokens"], self.steps))
            k, self.id_requests = self.id_requests, self.id_requests + 1
            kind = self.tamper if k == self.tamper_at else None
            data = id_response(tampered(outputs, kind), self.id_payload)
            if kind == "late":
                writer.write(data[:self.id_payload])
                await writer.drain()
                await asyncio.sleep(self.late_s)
                data = data[self.id_payload:]
            writer.write(data)
            return await writer.drain()
        if not head.startswith(b"POST /v1/completions "):
            doc = ({"data": [{"id": self.name}]} if head.startswith(b"GET /v1/models ")
                   else {"tokens": [ord(c) % L.VOCAB for c in body["prompt"]], "model": body["model"]})
            return await self.json(writer, 200, doc)
        if (why := refusal(body, self.steps)) is not None:
            return await self.json(writer, 400, {"error": why})
        outputs = step_outputs(self.model, body["prompt"][-1], min(body["max_tokens"], self.steps))
        choices = [{"index": 0, "text": o.hex(), "token_ids": [step_id(o)], "finish_reason": None} for o in outputs]
        choices[0]["prompt_token_ids"] = body["prompt"]
        choices[-1]["finish_reason"] = "length"
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        for choice in choices:
            p = b"data: %s\n\n" % json.dumps({"object": "text_completion", "model": self.name,
                                              "choices": [choice]}).encode()
            writer.write(b"%x\r\n%s\r\n" % (len(p), p))
            await writer.drain()
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
        await writer.drain()

    @staticmethod
    async def json(writer: asyncio.StreamWriter, status: int, doc: dict) -> None:
        data = json.dumps(doc).encode()
        reason = b"OK" if status == 200 else b"Bad Request"
        writer.write(b"HTTP/1.1 %d %s\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                     % (status, reason, len(data), data))
        await writer.drain()


async def serve(host: str, port: int, endpoint: Endpoint) -> None:
    server = await asyncio.start_server(endpoint, host, port)
    h, p = server.sockets[0].getsockname()[:2]
    print(json.dumps({"listening": f"http://{h}:{p}", "model": endpoint.name, "steps": endpoint.steps}), flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2, help="the loop's decode steps, a response's most tokens")
    ap.add_argument("--h", type=int, default=L.H, help="the MLP's width: a step's output is 2H bytes")
    ap.add_argument("--seed", type=int, default=0, help="the model's weights' seed (`loop.Model`)")
    ap.add_argument("--id-payload", type=int, default=ID_PAYLOAD,
                    help="P: a warden's id requests' cells' payload, B - 9, each id-form event's bytes")
    ap.add_argument("--tamper", choices=TAMPERS, help="one id request's answer tampered (module docstring)")
    ap.add_argument("--tamper-at", type=int, default=0, help="the id request tampered, from 0 in arrival order")
    ap.add_argument("--late-s", type=float, default=LATE_S, help="late's hold, in seconds")
    a = ap.parse_args(argv)
    asyncio.run(serve(a.host, a.port, Endpoint(L.Model(h=a.h, seed=a.seed), a.steps, a.id_payload, a.tamper,
                                               a.tamper_at, a.late_s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
