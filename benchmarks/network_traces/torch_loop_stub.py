"""torch-loop-stub: an OpenAI streaming endpoint shaped like the torch loop (`integrations/torch_loop`), the upstream of the
network certifier's rehearsal (`certifier_serve.py rehearse --upstream URL --traffic varied --compact 256`).

    python torch_loop_stub.py [--host 127.0.0.1] [--port 0] [--steps 2] [--h 128] [--seed 0]

It prints ``{"listening": URL}`` once it listens, then serves until it is stopped, each connection on its own:

- ``GET /v1/models``: one model, ``torch-loop/mlp-w8a8-h{H}``;
- ``POST /tokenize``: one token a character, its code mod `loop.VOCAB`, so every id is a row of the loop's embedding;
- ``POST /v1/completions``, streamed (``transfer-encoding: chunked``, one ``data:`` event a chunk, then ``data: [DONE]``):
  the request's last prompt id through the loop's model (`loop.Model(h, seed)`: embedded, then each decode step the
  W8A8 MLP over the step before, as `loop.TorchLoop.run` serves a request) for min(``max_tokens``, ``--steps``) steps,
  one chunk a step. A step's output is the MLP's H bf16 words, 2H bytes (256 at H = 128, little-endian): the chunk's
  ``text`` is their hex, and its one token id the step's (`step_id`). The first chunk carries ``prompt_token_ids``, the
  last ``finish_reason`` "length". A request asks for its token ids (``return_token_ids``), greedily (temperature 0,
  top_p 1), with no stop strings, as the certifier's shim forwards a varied request (`certifier_serve.varied_body`);
  any other is answered 400.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations" / "torch_loop"))
from verity_torch_loop import loop as L  # noqa: E402

#: a compact record's token field holds an id below 2^30 (`active.COMPACT_STOP`)
ID_BITS = 30


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


def refusal(body: dict) -> str | None:
    if body.get("return_token_ids") is not True:
        return "return_token_ids must be true"
    if {"ignore_eos", "min_tokens"} & set(body):
        return "no ignore_eos or min_tokens"
    if body.get("temperature") != 0 or body.get("top_p") != 1 or body.get("stop") != [] or type(body.get("seed")) is not int:
        return "greedy (temperature 0, top_p 1), stop [], an int seed"
    prompt = body.get("prompt")
    if not (isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt)):
        return "the prompt is a list of token ids"
    if not 0 <= prompt[-1] < L.VOCAB or type(body.get("max_tokens")) is not int or body["max_tokens"] < 1:
        return f"the last prompt id below {L.VOCAB} and max_tokens >= 1"
    return None


class Endpoint:
    def __init__(self, model: L.Model, steps: int) -> None:
        self.model, self.steps, self.name = model, steps, f"torch-loop/mlp-w8a8-h{model.h}"

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self.handle(reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: (\d+)", head, re.IGNORECASE)
        body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
        if not head.startswith(b"POST /v1/completions "):
            doc = ({"data": [{"id": self.name}]} if head.startswith(b"GET /v1/models ")
                   else {"tokens": [ord(c) % L.VOCAB for c in body["prompt"]], "model": body["model"]})
            return await self.json(writer, 200, doc)
        if (why := refusal(body)) is not None:
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
    a = ap.parse_args(argv)
    asyncio.run(serve(a.host, a.port, Endpoint(L.Model(h=a.h, seed=a.seed), a.steps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
