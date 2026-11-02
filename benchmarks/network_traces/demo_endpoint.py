"""demo-endpoint: an OpenAI endpoint in front of a decoder that isn't vLLM (the H200 demo's serving loop), for
the network certifier's rehearsal (`certifier_serve.py rehearse --upstream URL --traffic varied`), writing the records a
`verity-vllm serve-openai` endpoint writes, so network-e2e's endpoint binding and I/O tie read it unchanged.

    python demo_endpoint.py --out DIR [--engine hash|seeded:SHAPE|MODULE:FACTORY] [--seed 0] [--host 127.0.0.1]
        [--port 0] [--eos 0] [--max-num-seqs 8] [--max-num-batched-tokens 2048] [--window-steps 16]

It prints ``{"listening": URL, ...}`` once it listens, then serves until SIGTERM or SIGINT, when it closes the open
Commit window and writes ``summary.json``.

* **HTTP** (one request a connection): ``GET /v1/models`` lists the engine's model; ``POST /tokenize`` makes one token a
  UTF-8 byte (the demo's byte vocabulary, no BOS); ``POST /v1/completions``, streamed (``transfer-encoding: chunked``,
  one ``data:`` event a token, then ``data: [DONE]``), each chunk carrying its ``token_ids`` and the first its
  ``prompt_token_ids``, as vLLM's server does under ``return_token_ids``. A request is the certifier's varied body
  (`certifier_serve.varied_body`): its prompt as token ids, ``max_tokens``, greedy (temperature 0, top_p 1), no stop
  strings, its ``user`` the id the certified I/O names it by. Its prompt and ``max_tokens`` must fit the engine's
  context, as vLLM refuses a request past ``max_model_len``. Any other body is answered 400.
* **The scheduler** (`Scheduler`) is vLLM V1's synchronous one as network-e2e's derivation models it
  (`encodings/derived.py`'s walk): first come first served, the running requests in admission order each scheduled
  the rest of its known tokens within the step's token budget, then the waiting ones in arrival order while a seat and
  budget are left, the last of them cut short (chunked prefill); no prefix cache and a KV pool no run fills, so no hit
  and no preemption. A row that reaches its request's known tokens samples one token (greedy, the engine's); a request
  finishes at its first EOS id (``--eos``) or at ``max_tokens``. Each step's rows are written in the runner's batch
  order, the scheduling order sorted stably by scheduled tokens (Model Runner V2's).
* **The engine** (`Engine`: each sampling row's known tokens to its next token) is ``hash`` (a token from a SHA-512 of
  the sequence, for the tests), ``seeded:SHAPE`` (`SeededDecoder`: the seeded weights of `e2e_h200.served_weights`'s
  SHAPE, ``tiny`` or ``phase1``, through this module's own W8A8 forward on CPU, not the H200 loop's kernels), or
  ``MODULE:FACTORY``, a factory the serving loop provides, called with ``seed`` and returning an `Engine`.
* **The records**, in serve-openai's formats under ``--out``: ``serve.json`` (the engine, its context as
  ``engine.engine_kwargs.max_model_len``, the address, the run id), ``advice_trace.jsonl`` (`verity_vllm.engine.step_trace`'s
  schema: the scheduler's configuration, each request's arrival, each step's rows, emitted tokens and finishes, each
  Commit window's open and close, with unix clock readings), ``requests.jsonl`` (one line a finished request: its
  engine id, ``user``, the warden's release headers on it when it has them, prompt and streamed ids, finish, and the
  clock at its arrival, first token and finish) and ``windows.jsonl`` (one line a closed Commit window: its run id
  and run root). A Commit window opens at the first step after the engine was idle and closes when it is idle again or
  after ``--window-steps`` steps, so a request may run in two windows (the carry `window_chain.join` binds). Its run root
  is a hiding commitment (`verity.ml.commitments.advice` rows under fresh salts, kept in ``served_rows/w<k>.json``) of
  the tokens it served, one row a sampled token: (step, request, position, token). It stands in for the serving loop's
  own Commit (its per-slot trees), which network accounting doesn't read.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import re
import secrets
import signal
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from verity.ml.commitments import advice as AC

SCHEMA = "verity-vllm/step-trace/v1"  # verity_vllm.engine.step_trace.SCHEMA
ROW_BITS = 128
BINDING = b"verity/demo-endpoint/served-rows/v1"
SAMPLING = ("temperature", "top_p", "seed", "max_tokens", "stop")
#: a KV pool no run of a context of at most 32 positions and `max_num_seqs` requests fills (derived's no-eviction bound)
NUM_GPU_BLOCKS = 4096
BLOCK_SIZE = 16


class Engine(Protocol):
    name: str
    vocab: int
    context: int

    def next_tokens(self, seqs: Sequence[Sequence[int]]) -> list[int]: ...


class HashEngine:
    """A token from SHA-512 of the seed and the sequence: deterministic and cheap, the tests' engine."""

    def __init__(self, seed: int = 0, vocab: int = 256, context: int = 32) -> None:
        self.seed, self.vocab, self.context, self.name = seed, vocab, context, f"demo-endpoint/hash-v{vocab}-c{context}"

    def next_tokens(self, seqs: Sequence[Sequence[int]]) -> list[int]:
        return [hashlib.sha512(b"%d:" % self.seed + bytes(s)).digest()[0] % self.vocab for s in seqs]


def _bf16(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


class SeededDecoder:
    """The seeded weights (`e2e_h200.served_weights`, interface 1) of `shape`, through a W8A8 pre-norm decoder in
    numpy: embeddings and positions added, per layer RMSNorm, the query, key and value projections (``kv_heads`` KV heads),
    causal softmax attention, the output projection and the residual, RMSNorm, the ReLU MLP and the residual, then the
    final norm and the head, greedy. A W8A8 linear quantizes its input row to s8 codes at max|x|/127 and scales the s8 dot by
    that and its output channel's scale. The whole sequence is recomputed each step (the context is at most 32)."""

    def __init__(self, shape: str, seed: int = 0) -> None:
        from e2e_h200 import served_weights as W

        self.shape = s = W.ALIASES.get(shape) or W.SHAPES[shape]
        self.vocab, self.context, self.name = s.vocab, s.context, f"demo-endpoint/seeded-{s.name}"
        key = b"%d" % seed
        g = {t.name: W.generate(s, key, t.name) for t in W.tensors(s)}
        self.tok, self.pos = _bf16(g["tok"]), _bf16(g["pos"])
        self.layers = [{n: (g[f"l{i}.{n}"].astype(np.float32), g[f"l{i}.{n}.scale"][0]) if n.startswith("w")
                        else _bf16(g[f"l{i}.{n}"][0]) for n in ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "wup",
                                                                "wdown")} for i in range(s.layers)]
        self.final, self.head = _bf16(g["final_norm"][0]), (g["head"].astype(np.float32), g["head.scale"][0])

    @staticmethod
    def _linear(x: np.ndarray, w: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        sx = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-30) / 127
        xq = np.clip(np.rint(x / sx), -127, 127)
        return (xq @ w[0].T) * sx * w[1]

    @staticmethod
    def _norm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5) * g

    def logits(self, seq: Sequence[int]) -> np.ndarray:
        s, n = self.shape, len(seq)
        x = (self.tok[:, list(seq)] + self.pos[:, :n]).T.astype(np.float32)
        mask = np.triu(np.full((n, n), -np.inf, np.float32), 1)
        for L in self.layers:
            h = self._norm(x, L["attn_norm"])
            q = self._linear(h, L["wq"]).reshape(n, s.heads, s.head_dim)
            k = self._linear(h, L["wk"]).reshape(n, s.kv_heads, s.head_dim)
            v = self._linear(h, L["wv"]).reshape(n, s.kv_heads, s.head_dim)
            g = s.heads // s.kv_heads
            k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
            a = np.einsum("qhd,khd->hqk", q, k) + mask
            a = np.exp(a - a.max(axis=-1, keepdims=True))
            a /= a.sum(axis=-1, keepdims=True)
            x = x + self._linear(np.einsum("hqk,khd->qhd", a, v).reshape(n, -1), L["wo"])
            h = self._norm(x, L["mlp_norm"])
            x = x + self._linear(np.maximum(self._linear(h, L["wup"]), 0), L["wdown"])
        return self._linear(self._norm(x[-1:], self.final), self.head)[0]

    def next_tokens(self, seqs: Sequence[Sequence[int]]) -> list[int]:
        return [int(np.argmax(self.logits(s))) for s in seqs]


def engine_of(spec: str, seed: int) -> Engine:
    if spec == "hash":
        return HashEngine(seed)
    if spec.startswith("seeded:"):
        return SeededDecoder(spec.split(":", 1)[1], seed)
    module, _, factory = spec.partition(":")
    return getattr(importlib.import_module(module), factory)(seed=seed)


# --- the scheduler ----------------------------------------------------------------------------------------------------

@dataclass
class Request:
    rid: str
    http_id: str
    user: str | None
    prompt: list[int]
    max_tokens: int
    sampling: dict
    arrival: int
    out: list[int] = field(default_factory=list)
    computed: int = 0
    admitted: int = -1
    finish: dict | None = None
    windows: set = field(default_factory=set)
    queue: asyncio.Queue | None = None
    release: list | None = None                 # the warden's release headers on it (N, run hex), when it has them
    times: dict = field(default_factory=dict)   # the clock's readings at its arrive record, first token and finish

    @property
    def known(self) -> int:
        return len(self.prompt) + len(self.out)


class Scheduler:
    """vLLM V1's synchronous FCFS scheduler as `derived`'s walk models it (the module docstring), with the step trace,
    the Commit windows and the requests' records written as serve-openai writes them."""

    def __init__(self, engine: Engine, out: Path, *, eos: Sequence[int], max_num_seqs: int, budget: int,
                 window_steps: int, run_id: str, clock: Callable[[], float] = time.time) -> None:
        self.engine, self.out, self.eos, self.seats, self.budget = engine, out, set(eos), max_num_seqs, budget
        self.window_steps, self.run_id, self.clock = window_steps, run_id, clock
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.arrivals = self.admissions = self.scheduled = 0
        self.window, self.window_open, self.window_steps_done = -1, False, 0
        self.window_rows: list[tuple[int, int, int, int]] = []
        self.window_reqs: dict[str, Request] = {}
        self.finished: list[Request] = []
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace = (out / "advice_trace.jsonl").open("a")
        self.emit({"kind": "header", "schema": SCHEMA, "config": self.config()})

    def config(self) -> dict:
        return {"scheduler": "demo_endpoint.Scheduler", "max_num_batched_tokens": self.budget, "max_num_seqs": self.seats,
                "max_model_len": self.engine.context, "enable_chunked_prefill": True,
                "long_prefill_token_threshold": 0, "max_num_partial_prefills": 1, "policy": "fcfs",
                "async_scheduling": False, "block_size": BLOCK_SIZE, "enable_prefix_caching": False,
                "num_gpu_blocks": NUM_GPU_BLOCKS}

    def emit(self, rec: dict) -> None:
        if rec["kind"] in ("arrive", "step"):
            rec["t"] = self.clock()
        self.trace.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.trace.flush()

    def refusal(self, body: dict) -> str | None:
        prompt, m = body.get("prompt"), body.get("max_tokens")
        if not (isinstance(prompt, list) and prompt and all(type(t) is int and 0 <= t < self.engine.vocab
                                                            for t in prompt)):
            return f"the prompt is a list of token ids below {self.engine.vocab}"
        if type(m) is not int or m < 1 or len(prompt) + m > self.engine.context:
            return (f"max_tokens >= 1 and the prompt's {len(prompt)} tokens plus max_tokens within the context "
                    f"({self.engine.context})")
        if body.get("temperature") not in (0, 0.0) or body.get("top_p") not in (None, 1, 1.0) or body.get("stop"):
            return "greedy (temperature 0, top_p 1) with no stop strings"
        if {"ignore_eos", "min_tokens"} & set(body):
            return "no ignore_eos or min_tokens"
        return None

    def add(self, body: dict, queue: asyncio.Queue | None = None, release: list | None = None) -> Request:
        q = Request(f"cmpl-{secrets.token_hex(8)}", f"cmpl-{secrets.token_hex(8)}", body.get("user"), list(body["prompt"]),
                    body["max_tokens"], {k: body.get(k) for k in SAMPLING}, self.arrivals, queue=queue, release=release)
        self.arrivals += 1
        self.waiting.append(q)
        rec = {"kind": "arrive", "request": q.rid, "before_step": self.scheduled, "prompt_len": len(q.prompt),
               "eos_token_id": min(self.eos) if self.eos else None, "sampling": q.sampling}
        self.emit(rec)
        q.times["arrived"] = rec["t"]
        return q

    def busy(self) -> bool:
        return bool(self.waiting or self.running)

    def idle(self) -> None:
        if self.window_open and self.window_steps_done:
            self.close()

    def step(self) -> list[tuple[Request, int]]:
        """One engine step: its rows, the engine's tokens for the rows that sample, the finishes, its trace record.
        Returns each sampled (request, token)."""
        if self.window_open and self.window_steps_done >= self.window_steps:
            self.close()
        if not self.window_open:
            self.open()
        left, sched, new = self.budget, [], []
        for q in self.running:
            want = min(q.known - q.computed, self.engine.context - 1 - q.computed)
            n = 1 if q.known - q.computed == 1 else min(want, left)
            if n <= 0 or left <= 0:
                break
            sched.append((q, q.computed, n))
            left -= n
        while self.waiting and left > 0 and len(self.running) < self.seats:
            q = self.waiting.popleft()
            n = min(len(q.prompt), left)
            q.admitted, self.admissions = self.admissions, self.admissions + 1
            self.running.append(q)
            sched.append((q, 0, n))
            new.append(q)
            left -= n
        rows = sorted(sched, key=lambda r: r[2])
        sampling = [q for q, start, n in rows if start + n >= q.known]
        tokens = dict(zip((q.rid for q in sampling), self.engine.next_tokens([q.prompt + q.out for q in sampling])))
        emitted, finished, got = {}, {}, []
        step = self.scheduled
        for q, start, n in rows:
            q.computed = start + n
            q.windows.add(self.window)
            self.window_reqs[q.rid] = q
            if q.rid in tokens:
                tok = tokens[q.rid]
                self.window_rows.append((self.window_steps_done, q.arrival, q.known, tok))
                q.out.append(tok)
                emitted[q.rid] = 1
                got.append((q, tok))
                if tok in self.eos or len(q.out) >= q.max_tokens:
                    q.finish = {"reason": "stop" if tok in self.eos else "length", "stop_reason": None}
                    finished[q.rid] = q.finish
        self.running = [q for q in self.running if q.finish is None]
        self.finished += [q for q, _ in got if q.finish is not None]
        self.scheduled += 1
        self.window_steps_done += 1
        rec = {"kind": "step", "step": step, "T": sum(n for _, _, n in rows), "order": "runner",
               "rows": [[q.rid, start, n] for q, start, n in rows], "new": [q.rid for q in new], "resumed": [],
               "cache_hit": {}, "preempted": [], "emitted": emitted, "finished": finished}
        self.emit(rec)
        for q, _ in got:
            q.times.setdefault("first_token", rec["t"])
            if q.finish is not None:
                q.times["finished"] = rec["t"]
        return got

    def open(self) -> None:
        self.window += 1
        self.window_open, self.window_steps_done, self.window_rows, self.window_reqs = True, 0, [], {}
        self.emit({"kind": "window", "window": self.window, "event": "open", "before_step": self.scheduled})

    def close(self) -> dict:
        w, rows = self.window, self.window_rows
        data = [struct.pack(">IIII", *r) for r in rows] or [bytes(ROW_BITS // 8)]
        run_id = f"{self.run_id}/w{w}"
        reg = AC.register(data, ROW_BITS, hashlib.sha512(BINDING + b"\0" + run_id.encode()).digest())
        d = self.out / "served_rows"
        d.mkdir(exist_ok=True)
        (d / f"w{w}.json").write_text(json.dumps({"window": w, "rows": [list(r) for r in rows],
                                                  "salts": [s.hex() for s in reg.salts]}))
        rec = {"window": w, "run_id": run_id, "run_root": reg.root.hex(), "steps": self.window_steps_done,
               "rows": len(rows), "scheme": "demo-endpoint/served-rows/v1 (verity.ml.commitments.advice, hm96-sha512)",
               "requests": [{"engine_id": q.rid, "user": q.user} for q in self.window_reqs.values()], "passed": True}
        with open(self.out / "windows.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.window_open = False
        self.emit({"kind": "window", "window": w, "event": "close", "before_step": self.scheduled})
        self.write_finished()
        return rec

    def write_finished(self) -> None:
        with open(self.out / "requests.jsonl", "a") as fh:
            for q in self.finished:
                fh.write(json.dumps({"http_id": q.http_id, "engine_id": q.rid, "user": q.user, "arrival": q.arrival,
                                     "release": q.release, "admitted": q.admitted, "prompt_ids": q.prompt,
                                     "streamed_ids": q.out, "ingress": {"brought": q.sampling}, "finish": q.finish,
                                     "windows": sorted(q.windows), "times": q.times}) + "\n")
        self.finished = []


# --- HTTP -------------------------------------------------------------------------------------------------------------

def sse(doc: dict | str) -> bytes:
    p = b"data: %s\n\n" % (doc.encode() if isinstance(doc, str) else json.dumps(doc).encode())
    return b"%x\r\n%s\r\n" % (len(p), p)


class Server:
    protocol = "demo_endpoint (vLLM's completion chunks, token_ids)"

    def __init__(self, sched: Scheduler) -> None:
        self.sched, self.wake, self.served, self.refused = sched, asyncio.Event(), 0, 0

    async def run_engine(self) -> None:
        while True:
            if not self.sched.busy():
                self.sched.idle()
                self.wake.clear()
                await self.wake.wait()
                continue
            for q, tok in self.sched.step():
                q.queue.put_nowait((tok, q.finish))
            await asyncio.sleep(0)

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self.handle(reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: *(\d+)", head, re.IGNORECASE)
        raw = await reader.readexactly(int(n.group(1))) if n else b""
        body = json.loads(raw) if raw else {}
        name = self.sched.engine.name
        if head.startswith(b"GET /v1/models "):
            return await self.json(writer, 200, {"object": "list", "data": [{"id": name, "object": "model",
                                                                           "max_model_len": self.sched.engine.context}]})
        if head.startswith(b"POST /tokenize "):
            tokens = list(str(body.get("prompt", "")).encode())
            return await self.json(writer, 200, {"tokens": tokens, "count": len(tokens),
                                                 "max_model_len": self.sched.engine.context})
        if not head.startswith(b"POST /v1/completions "):
            return await self.json(writer, 404, {"error": "GET /v1/models, POST /tokenize, POST /v1/completions"})
        if (why := self.sched.refusal(body)) is not None:
            self.refused += 1
            return await self.json(writer, 400, {"error": {"message": why, "type": "BadRequestError"}})
        queue: asyncio.Queue = asyncio.Queue()
        q = self.sched.add(body, queue)
        self.wake.set()
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        first, created = True, int(time.time())
        while True:
            tok, finish = await queue.get()
            choice = {"index": 0, "text": bytes([tok]).decode("latin-1"), "token_ids": [tok], "logprobs": None,
                      "finish_reason": finish and finish["reason"], "stop_reason": None}
            if first:
                choice["prompt_token_ids"], first = q.prompt, False
            writer.write(sse({"id": q.http_id, "object": "text_completion", "created": created, "model": name,
                              "choices": [choice]}))
            await writer.drain()
            if finish is not None:
                break
        writer.write(sse("[DONE]") + b"0\r\n\r\n")
        await writer.drain()
        self.served += 1

    @staticmethod
    async def json(writer: asyncio.StreamWriter, status: int, doc: dict) -> None:
        data = json.dumps(doc).encode()
        reason = {200: b"OK", 400: b"Bad Request", 404: b"Not Found"}[status]
        writer.write(b"HTTP/1.1 %d %s\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                     % (status, reason, len(data), data))
        await writer.drain()


def scheduler(a: argparse.Namespace, engine: Engine, clock: Callable[[], float] = time.time) -> Scheduler:
    """The scheduler `parser`'s arguments ``a`` give ``engine``."""
    return Scheduler(engine, a.out, eos=[int(x) for x in a.eos.split(",")] if a.eos else [],
                     max_num_seqs=a.max_num_seqs, budget=a.max_num_batched_tokens, window_steps=a.window_steps,
                     run_id=a.run_id, clock=clock)


def serve_record(a: argparse.Namespace, sched: Scheduler, server: Server, address: Sequence) -> dict:
    """``serve.json``: the engine, its context (``engine_kwargs.max_model_len``), the EOS ids, the address, the run id."""
    engine = sched.engine
    return {"address": list(address), "served_model_name": engine.name, "engine": {
        "kind": "demo_endpoint", "spec": a.engine, "seed": a.seed, "engine_kwargs": {
            "max_model_len": engine.context, "max_num_seqs": a.max_num_seqs,
            "max_num_batched_tokens": a.max_num_batched_tokens, "enable_prefix_caching": False}},
        "eos_token_ids": sorted(sched.eos), "window_steps": a.window_steps, "idle_close_steps": 1, "admit": "arrival",
        "step_rows": "runner", "run_id": a.run_id, "protocol": server.protocol}


async def serve(a: argparse.Namespace, engine: Engine | None = None,
                server_of: Callable[[Scheduler], Server] = Server) -> dict:
    """Serve until SIGTERM or SIGINT, under `parser`'s arguments ``a``: ``engine`` (default ``a.engine``'s) through the
    scheduler, the HTTP side ``server_of``'s."""
    engine = engine or engine_of(a.engine, a.seed)
    sched = scheduler(a, engine)
    server = server_of(sched)
    listener = await asyncio.start_server(server, a.host, a.port)
    host, port = listener.sockets[0].getsockname()[:2]
    (a.out / "serve.json").write_text(json.dumps(serve_record(a, sched, server, (host, port)), indent=1))
    print(json.dumps({"listening": f"http://{host}:{port}", "model": engine.name, "context": engine.context}), flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(server.run_engine())
    async with listener:
        await stop.wait()
    task.cancel()
    if sched.window_open:
        sched.close()
    summary = {"served": server.served, "refused": server.refused, "requests": sched.arrivals,
               "steps": sched.scheduled, "windows": sched.window + 1, "unfinished": len(sched.running) + len(sched.waiting)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    sched.trace.close()
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="demo_endpoint.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", default="hash", help="hash, seeded:SHAPE (tiny, phase1) or MODULE:FACTORY")
    ap.add_argument("--seed", type=int, default=0, help="the seeded weights' seed or the hash engine's")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--eos", default="0", help="EOS token ids, comma-separated (the byte vocabulary has none of its own)")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--window-steps", type=int, default=16, help="engine steps a Commit window holds at most")
    ap.add_argument("--run-id", default=f"demo-endpoint-{secrets.token_hex(4)}")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    print(json.dumps({"stopped": asyncio.run(serve(a))}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
