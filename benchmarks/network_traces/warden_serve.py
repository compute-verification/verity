"""warden-serve: the active warden (`verity.core.protocols.certifier.warden`) on one host, a relay that knows bytes
only, in front of any TCP server, and the offline check of what it published.

**Topology.** Each client connection reaches the warden's listener. The warden opens one connection to the server
(`--upstream HOST:PORT`) for it and forwards each direction's bytes unchanged in content, on its schedule, in B-byte
cells; the client speaks cells through the thin, workload-blind shim (`warden.cells` sending, `warden.Shim` reading):

    client ──cells──▶ warden ──bytes──▶ server ──bytes──▶ warden ──cells──▶ client

Nothing here is the warden's but bytes. Tonight's clients send HTTP/1.1 requests for vLLM's streamed `/v1/completions`
with `Connection: close`, so the server's end of stream ends egress (the warden never half-closes toward a server).
The server gets each request whole at its release, with its place in the warden's release order and the warden's run
id (`release`), which `deployment.json` names (`run_id`).

**What it can't enforce here.**
- Run directly, the warden runs on the server's host (`trusted-host` stands for the auditor's host). The gate's
  `warden-rehearse` (`benchmarks/live_challenger/auditor.py`) runs it on the auditor's host instead, the server reached
  through an ssh tunnel to the peer's loopback. Either way the server's host can reach anyone by any other route,
  which nothing here sees.
- TLS is not terminated: both legs are plain TCP.
- The rehearsal's client is on the same host, so its link is loopback, and its arrival times are read on the warden's
  host clock (`origin_unix`).
- The keys are LMS at H = `cs.HEIGHT` (2^10 one-time keys) unless `--height` says otherwise, where the deployed key's
  is H = 20 (the auditor gate's default, `benchmarks/live_challenger/auditor.py`).

**Traffic** (`traffic`, `varied_traffic`): varied, as `certifier_serve`'s: Poisson arrivals at `--rate` a second,
prompts of 16–1024 tokens (log-uniform; words at 0.75 a token, so approximately), `max_tokens` of 16 … `--max-tokens`
(log-uniform) under real EOS, temperature 0, 0.7 or 1, and a seed on about half the requests. `--temperature` and
`--top-p` put the same on every request instead (the gate's `warden-rehearse` always gives both): an endpoint applies
its own generation config to a request that omits them. `rehearse --ids` sends the scan's traffic instead (`id_traffic`,
at the same sampling when given): warden-form id requests (`active.id_request`, offset 0), each answered by the fake
endpoint as an id-form response (`id_response`: one `io_tie.id_event` cell an output id, then `io_tie.TRAILER`, at most
`egress_cells` − 1 ids), what `scan_zk.py prove --warden-run` proves; the gate's `warden-rehearse --scan` sends it to
the peer's endpoint, at scan_zk's schedule. `--prompts FILE` (with `--ids`) gives the id requests' prompts instead
(`read_prompts`, `prompt_traffic`: request j's the (j mod n)-th, each with exactly `--max-tokens`), and
`--window-per-group` sends them in groups of the n, group g's half a bucket into link-window g·G (`group_windows`), so
that group g's cells are exactly windows g·G … (g+1)·G − 1 (`groups.json`), and the run is exactly those windows: a
loop's period of requests a group, as the gate's `warden-rehearse --scan --prompts --window-per-group` runs it.

**Subcommands**, from the repository root or with `benchmarks/network_traces` on the path:

    python warden_serve.py keys --dir KEYDIR [--height 10]
    python warden_serve.py traffic --out TRAFFIC.json --model NAME [--requests 100] [--rate 0.5] [--seed 0]
        [--temperature T] [--top-p P]
    python warden_serve.py fake --listen HOST:PORT [--token-ms 20] [--id-payload P]
    python warden_serve.py serve --upstream HOST:PORT --listen HOST:PORT --out DIR --key KEYDIR --session HEX
        --windows N [--start-in 5] [--load-heads FILE] [--paced-ms 50] [schedule]
    python warden_serve.py client --deployment DIR/deployment.json --traffic TRAFFIC.json --out DIR
    python warden_serve.py check --run DIR
    python warden_serve.py rehearse (--upstream HOST:PORT | --fake) --out DIR [--requests 100] [--temperature T]
        [--top-p P] [--ids [--prompts FILE [--window-per-group]]] [--paced-ms 50] [--load-store DIR --load-sinks HOST:PORT,...
        --load-program HEX [--load-shape h200|tiny] [--load-workers N] [--load-ready-s 900]] [schedule]

- `serve` writes `deployment.json` (the warden's public key, the session, the schedule's parameters and their digest,
  the origin, the listener), runs the warden for `--windows` windows, each bucket released by its pacer `--paced-ms`
  after its boundary (`warden`'s "The pacer"; 0: written on the warden's loop), and writes `warden.json` (each
  connection's outcome, the tick lateness and each tick's (`ticks_ms`), the refusals and their windows, and each
  bucket's pacer report, `paced`). Each window's records go to `requests.records`,
  `responses.records` and `fails.records` (each failed connection's reason, witness), and each is certified
  (`certificate.Issuer`, with the warden's parameters digest) into `certified.jsonl`, except a window in which the
  warden refused a connection, whose records and rows it certifies and signs none of (`warden`'s "Admission"). Each window of the wire's two records is also committed as
  rows, a row a bucket, with its wiring (`warden_rows`), its head signed: `cells.jsonl` holds the registration and the
  cost (rows, row bytes, commit and sign ms), `cells.salts` the rows' salts and the wiring's (witness, the prover's,
  who holds the records the rows and the wiring are read off). All of that is done by a spawned process
  (`PublisherProcess`), which alone holds the key and gets each tick's rows as the warden releases them: a window's
  commit can take seconds of CPU, and in the warden's own process, even on another thread, it holds the ticker's GIL and
  puts ticks past their slack. Before the origin `serve` runs on the C allocator (it re-executes itself with
  `PYTHONMALLOC=malloc`) and makes its process ready for the run's growth (`prepare_process`: its descriptor table, its
  heap, its pages locked where permitted), and from it to the end a process of its own (`stall_probe`, unless
  `--no-stall-probe`) samples every thread of both processes each millisecond into `stalls.json`, every episode in
  uninterruptible sleep with the kernel function it slept in; `warden.json` keeps what was prepared (`process`) and
  each thread's episodes and page faults (`stalls`).
- `client` sends each request of the traffic at its time after the origin on a connection of its own, reads the cells
  to the warden's close, and writes `client.json`: per connection its ordinal, outcome, every cell's kind and arrival,
  the SHA-256 of what it sent and of the bytes it got, and the HTTP response read from them. Its descriptor table holds
  every connection before it sends (`reserve_fds`): its process has numpy's threads too, and a table grown mid-run
  put cells that left the warden on time a bucket late (168 ms, every connection open as the client passed 256).
- `check` (`check`): every window of the three records has a certificate that verifies and whose head is the
  record's, every window of the wire's two has a registration that verifies whose root is its record's rows under
  their salts (`check_cells`), the audit (`warden.audit`) accepts the records, and every connection's client got
  exactly its egress record's cells, each in its bucket, its request is the ingress record's and its response the
  egress record's; `check.json`.
- `rehearse` makes a key in a scratch directory, writes the traffic, starts the fake endpoint when `--fake`, and runs
  `serve` and `client` as processes, then `check`.

**The load** (`--load-store`, the gate's `load-relay`): the serving session's weights' load, relayed and committed by
this device and signed by its one signer. `rehearse` first relays the store to the H200's sink
(`load_relay.relay_load`: `--load-sinks`, one stream an address, each value committed under the verifier's domain for
`--load-program`, its salts sent on to the sink's witness and kept nowhere here), writing `load-heads.json` and
`load-relay.json`; a refused load serves nothing. It then waits up to `--load-ready-s` for the endpoint, which the
prover starts on the weights its sink wrote, and runs `serve --load-heads`: the publisher process signs every head in
the run's session before its first window (`sign_load`), so one LMS key and one session carry the load and the windows,
and writes `load.json`, the client's kept registrations of the load. `check` reads it under the deployment's key and
session (`check_load`) at the relay's roots.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import dataclasses
import fcntl
import hashlib
import http.client
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import random
import re
import resource
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import active_replay as ar
import certifier_serve as cs
import load_relay as lr
import stall_probe
from verity.core.protocols.certifier import io_tie as IT, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, PAD, id_request
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction, Issuer, LmsKey, record_head
from verity.core.protocols.certifier.grid import GridParams, Record, Status, row_count
from verity.core.protocols.certifier.warden import END, FAIL, REFUSED, Shim, Warden, WardenParams, audit, cells

HERE = Path(__file__).resolve().parent
INGRESS, EGRESS, FAILS, LINK = "requests", "responses", "fails", "warden"
#: each record's certified link name and direction: the wire's two under `LINK`, the reasons under their own name
CERTIFIED = {INGRESS: (LINK, Direction.INGRESS), EGRESS: (LINK, Direction.EGRESS),
             FAILS: (f"{LINK}-{FAILS}", Direction.EGRESS)}
KIND_CHARS = {PAD: "p", DATA: "d", END: "e", FAIL: "f", REFUSED: "r"}
#: the heap `prepare_process` grows and touches before the origin, a lane's and at least: heavy's opening burst through
#: the warden (1000 connections at 300 a second, 1 KiB cells) grew its heap by about 35 KB a connection
RESERVE_PER_LANE, RESERVE_MIN = 256 << 10, 64 << 20
#: the descriptors the warden holds beside its connections' (its listener, loop, pipes and the probe's)
FD_MARGIN = 256
#: in the environment of a `serve` that re-executed itself on the C allocator, so that it does so once
C_ALLOCATOR = "WARDEN_SERVE_C_ALLOCATOR"
#: what a finite RLIMIT_MEMLOCK must leave beyond the process's mapping for `_memlock` to lock it: the warden's heap
#: grows tens of MB in a run
LOCK_HEADROOM = 4 << 30
M_TRIM_THRESHOLD, M_MMAP_THRESHOLD, MCL_CURRENT, MCL_FUTURE, CAP_IPC_LOCK = -1, -3, 1, 2, 14
#: the schedule's defaults here, where they differ from `WardenParams`'s: r·P = 494 bytes a bucket each way stays
#: under what vLLM streams a request even with its tokens merged into fewer events, and D = 3 s from admission covers
#: the request's ingress and a queued prefill
SCHEDULE = asdict(WardenParams()) | {"slots": 2, "delay": 30, "lanes": 8, "ingress_cells": 96, "ingress_buckets": 100}
QUESTIONS = ("Explain how a bloom filter works.", "Write a haiku about the sea.", "List three uses of copper.",
             "What is the capital of Australia? Answer in one word.", "Summarize the notes above in one sentence.",
             "Translate 'good morning' into French, Spanish and German.",
             "Give a short proof that the square root of 2 is irrational.",
             "Name the planets of the solar system in order.")
FILLER = ("Notes: the meeting moved to Thursday, the printer on the third floor is out of toner, the parking lot is "
          "closed for repaving next week, and the quarterly report is due on Friday.").split()
TEMPERATURES = (0.0, 0.7, 1.0)
#: a prompt id's bits, the scan's IDB (`scan_zk.CLASS`): the scan reads each prompt id and output id in IDB bits
PROMPT_ID_BITS = 17
#: the public file of a `--window-per-group` rehearsal: each group's requests and link-windows (`groups`)
GROUPS = "groups.json"


# Traffic and the bytes on it.

def varied_traffic(n: int, *, model: str, rate: float = 0.5, seed: int = 0, prompt_tokens: tuple[int, int] = (16, 1024),
                   max_tokens: tuple[int, int] = (16, 256), start_s: float = 1.0, temperature: float | None = None,
                   top_p: float | None = None) -> list[dict]:
    """n requests, each {"at": seconds after the origin, "body": a streamed `/v1/completions` body}. `temperature` and
    `top_p`, when given, are every request's (else a temperature drawn from `TEMPERATURES` and no top_p); the arrivals,
    prompts, lengths and seeds are the seed's either way."""
    rng = random.Random(seed)
    at, out = start_s, []
    for _ in range(n):
        question = rng.choice(QUESTIONS)
        words = round(math.exp(rng.uniform(*map(math.log, prompt_tokens))) * 0.75) - len(question.split())
        filler = " ".join(FILLER[j % len(FILLER)] for j in range(max(0, words)))
        most = round(math.exp(rng.uniform(*map(math.log, max_tokens))))
        drawn = rng.choice(TEMPERATURES)
        body = {"model": model, "prompt": f"{filler}\n\n{question}" if filler else question,
                "max_tokens": most, "stream": True, "temperature": drawn if temperature is None else temperature,
                "stream_options": {"include_usage": True}} | ({} if top_p is None else {"top_p": top_p})
        if rng.random() < 0.5:
            body["seed"] = rng.randrange(1 << 31)
        out.append({"at": round(at, 3), "body": body})
        at += rng.expovariate(rate)
    return out


def id_traffic(n: int, *, model: str, rate: float = 0.5, seed: int = 0, prompt_tokens: tuple[int, int] = (16, 1024),
               max_tokens: tuple[int, int] = (1, 64), start_s: float = 0.0, temperature: float | None = None,
               top_p: float | None = None) -> list[dict]:
    """n warden-form id requests (`active.id_request`'s fields), each {"at": seconds after the origin, "ids": its
    fields}: Poisson arrivals at `rate`, prompts of log-uniform length of ids below 2^17, `max_tokens` log-uniform, and
    the sampling drawn as `varied_traffic` draws it (top_p 1), or `temperature` and `top_p`, when given, every
    request's; the arrivals, prompts, lengths and seeds are the seed's either way. The scan (`ingress_scan`) reads
    exactly these."""
    rng = random.Random(seed)
    at, out = start_s, []
    for i in range(n):
        length = round(math.exp(rng.uniform(*map(math.log, prompt_tokens))))
        prompt = [rng.randrange(128_256) for _ in range(length)]
        most = round(math.exp(rng.uniform(*map(math.log, max_tokens))))
        drawn = rng.choice(TEMPERATURES)
        fields = {"model": model, "user": f"user-{i}", "prompt": prompt, "max_tokens": most,
                  "temperature": drawn if temperature is None else temperature,
                  "top_p": 1.0 if top_p is None else top_p, "seed": rng.randrange(1 << 31)}
        out.append({"at": round(at, 3), "ids": fields})
        at += rng.expovariate(rate)
    return out


def group_windows(params: WardenParams, tokens: int) -> int:
    """The link-windows a group of at most `lanes` id requests takes when it arrives half a bucket into a window's
    first bucket (`prompt_traffic`'s ``window_per_group``), each answered by at most `tokens` id events and the trailer:
    admitted by the bucket after its arrival's (and one more for the client's lag), its last egress cell in bucket
    s + D + ⌊tokens/r⌋ (the warden's schedule), so a group's cells are all in its own windows."""
    return -(-(params.delay + tokens // params.slots + 3) // params.window)


def read_prompts(data: bytes, params: WardenParams) -> list[list[int]]:
    """``--prompts``: a JSON array of prompts, each a non-empty array of token ids below 2^`PROMPT_ID_BITS` (the
    scan's IDB), each prompt's id request (`active.id_request`, offset 0) within the schedule's ingress bound."""
    try:
        prompts = json.loads(data)
    except ValueError as e:
        raise SystemExit(f"--prompts: not JSON: {e}") from None
    if not (isinstance(prompts, list) and prompts and all(isinstance(p, list) and p for p in prompts)
            and all(type(t) is int and 0 <= t < 1 << PROMPT_ID_BITS for p in prompts for t in p)):
        raise SystemExit(f"--prompts: a JSON array of prompts, each a non-empty array of token ids below "
                         f"2^{PROMPT_ID_BITS}")
    for p in prompts:
        cells_in = -(-len(id_request(model="m" * 62, user="u" * 62, prompt=p, max_tokens=1,
                                     frame_bytes=params.cell_bytes, offset=0)) // params.payload)
        if cells_in > params.ingress_cells:
            raise SystemExit(f"--prompts: a prompt of {len(p)} ids takes {cells_in} ingress cells, past the "
                             f"schedule's {params.ingress_cells}")
    return prompts


def prompt_traffic(prompts: list[list[int]], n: int, *, model: str, params: WardenParams, max_tokens: int,
                   rate: float = 0.5, seed: int = 0, start_s: float = 0.0, temperature: float | None = None,
                   top_p: float | None = None, window_per_group: bool = False) -> list[dict]:
    """n warden-form id requests (`id_traffic`'s items) of the given prompts, request j's ``prompts[j mod len]`` with
    exactly `max_tokens`: at `id_traffic`'s Poisson arrivals, or under `window_per_group` in groups of ``len(prompts)``
    (at most `lanes`), group g's requests together half a bucket into window g·G (G = `group_windows`), so group g's
    cells are exactly its windows' (``groups``). The seeds are the seed's either way."""
    if window_per_group and len(prompts) > params.lanes:
        raise SystemExit(f"--window-per-group: a group of {len(prompts)} prompts, past the schedule's "
                         f"{params.lanes} lanes")
    rng = random.Random(seed)
    at, out = start_s, []
    G, window_s = group_windows(params, max_tokens), params.window * params.bucket_ms / 1000
    for j in range(n):
        drawn = rng.choice(TEMPERATURES)
        fields = {"model": model, "user": f"user-{j}", "prompt": list(prompts[j % len(prompts)]),
                  "max_tokens": max_tokens, "temperature": drawn if temperature is None else temperature,
                  "top_p": 1.0 if top_p is None else top_p, "seed": rng.randrange(1 << 31)}
        when = (j // len(prompts)) * G * window_s + params.bucket_ms / 2000 if window_per_group else at
        out.append({"at": round(when, 3), "ids": fields})
        at += rng.expovariate(rate)
    return out


def groups(n: int, per: int, params: WardenParams, max_tokens: int) -> list[dict]:
    """`prompt_traffic`'s groups under ``window_per_group``: each group's requests (indices) and its link-windows."""
    G = group_windows(params, max_tokens)
    return [{"group": g, "requests": list(range(g * per, min(n, (g + 1) * per))), "windows": [g * G, (g + 1) * G]}
            for g in range(-(-n // per))]


def fake_ids(body: dict) -> list[int]:
    """The fake endpoint's output ids for an id request: `fake_tokens` of them, each below 2^17, fixed by the request."""
    h = int.from_bytes(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()[:4], "big")
    return [(h + 7919 * k) % 128_256 for k in range(fake_tokens(body))]


def id_response(ids: list[int], payload: int) -> bytes:
    """An id-form response as the scan's egress reads it: one `io_tie.id_event` of P = `payload` bytes an id, each one
    `DATA` cell, then `io_tie.TRAILER`, which the warden's `END` holds."""
    return b"".join(IT.id_event(x, payload) for x in ids) + IT.TRAILER


def parse_id_response(data: bytes, payload: int) -> dict:
    """An id-form response's reading (`id_response`), in `parse_response`'s keys: ``ids`` its count of id events,
    ``complete`` when the bytes are whole id events and then `io_tie.TRAILER`."""
    k, rest = divmod(len(data) - len(IT.TRAILER), payload)
    ids = None
    if k >= 0 and rest == 0:
        at = len(IT.ID_EVENT_HEAD)
        digits = [data[i * payload + at:i * payload + at + IT.ID_DIGITS] for i in range(k)]
        if all(d.lstrip(b" ").isdigit() for d in digits):
            got = [int(d) for d in digits]
            try:
                ids = got if data == id_response(got, payload) else None
            except ValueError:
                ids = None
    n = 0 if ids is None else len(ids)
    return {"status": None, "ids": None if ids is None else n, "complete": ids is not None, "events": n,
            "text_bytes": 0, "finish_reason": None, "prompt_tokens": None, "completion_tokens": n}


def http_request(body: dict, host: str, path: str = "/v1/completions") -> bytes:
    data = json.dumps(body, separators=(",", ":")).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
            "Connection: close\r\n\r\n").encode() + data


def dechunk(data: bytes) -> tuple[bytes, bool]:
    """A chunked body's bytes, and whether its last chunk arrived."""
    out, i = bytearray(), 0
    while (j := data.find(b"\r\n", i)) >= 0:
        try:
            n = int(data[i:j].split(b";")[0], 16)
        except ValueError:
            break
        if n == 0:
            return bytes(out), True
        if len(data) < j + 4 + n:
            out += data[j + 2:j + 2 + n]
            break
        out += data[j + 2:j + 2 + n]
        i = j + 4 + n
    return bytes(out), False


def parse_response(data: bytes) -> dict:
    """An HTTP/1.1 response's status and, of a streamed completion (server-sent events), its events, text bytes,
    finish reason and usage; `complete` once it has ended with `data: [DONE]` (and a chunked body's last chunk)."""
    head, sep, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(m.group(1)) if (m := re.match(r"HTTP/1\.[01] (\d{3})", lines[0])) else None
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    body, ended = (dechunk(rest) if "chunked" in headers.get("transfer-encoding", "").lower() else (rest, bool(sep)))
    events, text, finish, usage, done = 0, [], None, None, False
    for line in body.split(b"\n"):
        if line.strip() == b"data: [DONE]":
            done = True
        elif line.startswith(b"data: "):
            try:
                doc = json.loads(line[6:])
            except ValueError:
                continue
            events += 1
            for choice in doc.get("choices") or []:
                text.append(choice.get("text") or "")
                finish = choice.get("finish_reason") or finish
            usage = doc.get("usage") or usage
    return {"status": status, "complete": ended and done, "events": events,
            "text_bytes": len("".join(text).encode()), "finish_reason": finish,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens")}


def _sse(doc: dict) -> bytes:
    data = b"data: " + json.dumps(doc, separators=(",", ":")).encode() + b"\n\n"
    return b"%x\r\n%s\r\n" % (len(data), data)


def fake_tokens(body: dict) -> int:
    """The fake endpoint's response length: `max_tokens`, or fewer (its EOS), fixed by the request."""
    m = int(body.get("max_tokens") or 16)
    h = int.from_bytes(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()[:4], "big")
    return m if h % 3 == 0 else 1 + h % m


async def fake_endpoint(reader: asyncio.StreamReader, writer, *, sleep=asyncio.sleep, first_s: float = 0.1,
                        token_s: float = 0.02, id_payload: int | None = None) -> None:
    """A stand-in for vLLM's streamed `/v1/completions` for a rehearsal: it reads one request, answers its first token
    `first_s` after, then one event every `token_s`, its usage, `[DONE]`, and closes. `GET /v1/models` lists its one
    model, `fake`, so that a rehearsal that names no model (the gate's `warden-rehearse`) runs against it too. With
    `id_payload` P it answers every request `first_s` after with its `fake_ids` as one id-form response
    (`id_response`), whole, and no HTTP head: the bytes the scan's egress reads."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        if head.startswith(b"GET /v1/models "):
            doc = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n"
                         b"content-length: %d\r\n\r\n%s" % (len(doc), doc))
            await writer.drain()
            return
        m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
        body = json.loads(await reader.readexactly(int(m.group(1)))) if m else {}
        if id_payload is not None:
            await sleep(first_s)
            writer.write(id_response(fake_ids(body), id_payload))
            await writer.drain()
            return
        n, model = fake_tokens(body), body.get("model")
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\nconnection: close\r\n"
                     b"transfer-encoding: chunked\r\n\r\n")
        await sleep(first_s)
        for k in range(n):
            finish = None if k < n - 1 else "length" if n == body.get("max_tokens") else "stop"
            writer.write(_sse({"id": "cmpl-fake", "object": "text_completion", "created": 0, "model": model,
                               "choices": [{"index": 0, "text": f" w{k}", "logprobs": None, "finish_reason": finish,
                                            "stop_reason": None}]}))
            await writer.drain()
            await sleep(token_s)
        if (body.get("stream_options") or {}).get("include_usage"):
            prompt = len(str(body.get("prompt", "")).split())
            writer.write(_sse({"id": "cmpl-fake", "object": "text_completion", "created": 0, "model": model,
                               "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": n,
                                                        "total_tokens": prompt + n}}))
        writer.write(b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
        pass
    finally:
        writer.close()


async def probe(upstream: tuple[str, int], traffic: list[dict], in_flight: int = 4, clock=time.monotonic) -> dict:
    """Each request of `traffic` straight to the server, `in_flight` at a time, no warden: per request the seconds to
    its first byte and to the server's close, its bytes, and the bytes a second between them (what r·P a bucket and
    D must stay under and cover)."""
    host, gate = "%s:%d" % upstream, asyncio.Semaphore(in_flight)

    async def one(item: dict) -> dict:
        async with gate:
            start = clock()
            reader, writer = await asyncio.open_connection(*upstream)
            writer.write(http_request(item["body"], host))
            data, first = bytearray(), None
            while chunk := await reader.read(1 << 16):
                first = first or clock()
                data += chunk
            end = clock()
            writer.close()
        rate = len(data) / (end - first) if first and end > first else None
        return {"first_s": first and round(first - start, 3), "closed_s": round(end - start, 3), "bytes": len(data),
                "bytes_per_s": rate and round(rate), **parse_response(bytes(data))}

    rows = await asyncio.gather(*(one(item) for item in traffic))
    return {"upstream": list(upstream), "in_flight": in_flight, "requests": rows,
            "first_s": _quantiles([r["first_s"] for r in rows if r["first_s"] is not None]),
            "bytes_per_s": _quantiles([r["bytes_per_s"] for r in rows if r["bytes_per_s"]]),
            "statuses": sorted({str(r["status"]) for r in rows})}


# The warden's run.

class Publisher:
    """The warden's `publish`, inline: each record packed to its file (`requests.records`, `responses.records`,
    `fails.records`), and certified by `issuer` under its link name and direction (`CERTIFIED`) with the schedule's
    parameters digest, one line of `certified.jsonl` each, or the link-window under `refused`; the wire's two also
    committed as rows by `commit` (`attest`). `serve` runs it in `PublisherProcess`: on the ticker's thread, or on any
    thread of its process, a window's commit holds the ticker's GIL for seconds. `close` gives the issuer's refusals."""

    def __init__(self, out: Path, issuer: Issuer, params: WardenParams, commit=WR.Cells.of) -> None:
        self.out, self.issuer, self.params, self.commit = out, issuer, params, commit

    def __call__(self, record: Record, grid: GridParams) -> None:
        self.keep(record, grid)

    def withhold(self, record: Record, grid: GridParams) -> None:
        """The warden's `withhold`: a record of a window in which it refused a connection, kept and committed as
        `keep` does but neither certified nor signed, its link-window under `refused` with ``"withheld": true``."""
        self.keep(record, grid, withheld=True)

    def keep(self, record: Record, grid: GridParams, withheld: bool = False) -> None:
        ar.write_blobs(self.out / f"{record.link}.records", ar.pack_record(record))
        link, direction = CERTIFIED[record.link]
        certified = None if withheld else \
            self.issuer.certify(replace(record, link=link), direction, grid, self.params.digest)
        line = (_refused(link, direction, record.window, withheld) if certified is None
                else cs.certified_json(certified))
        with open(self.out / "certified.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        if record.link != FAILS:
            self.attest(record, link, direction, withheld)

    def attest(self, record: Record, link: str, direction: Direction, withheld: bool = False) -> None:
        """The link-window's cells as rows and its wiring (`warden_rows`), their head signed (never `withheld`'s): one
        line of `cells.jsonl` (the registration, or the link-window under `refused`, and the cost), the salts to
        `cells.salts` (witness)."""
        t0 = time.perf_counter()
        cells_ = self.commit(link, direction, record, self.params, self.issuer.session)
        head = cells_.head
        t1 = time.perf_counter()
        cert = None if withheld else self.issuer.sign(link, direction, record.window, head.digest())
        t2 = time.perf_counter()
        cost = {"rows": len(cells_.rows), "row_bytes": len(cells_.rows[0]), "commit_ms": round((t1 - t0) * 1000, 3),
                "sign_ms": round((t2 - t1) * 1000, 3)}
        line = (_refused(link, direction, record.window, withheld) if cert is None
                else {"registration": WR.registration(head, cert, self.params)})
        ar.write_blobs(self.out / "cells.salts", cells_.witness)
        with open(self.out / "cells.jsonl", "a") as fh:
            fh.write(json.dumps(line | {"cost": cost}) + "\n")

    def close(self) -> list:
        return list(self.issuer.refused)


def _refused(link: str, direction: Direction, window: int, withheld: bool) -> dict:
    return {"refused": [link, direction.value, window]} | ({"withheld": True} if withheld else {})


def _hash_row(h, row) -> None:
    """Row `row` into `h`, a grid's SHA-512: each slot a 0 byte, or a 1 byte and its frame."""
    for frame in row:
        h.update(b"\x00" if frame is None else b"\x01" + frame)


def grid_digest(grid) -> bytes:
    """The SHA-512 of a record's grid, row by row (`_hash_row`): the close marker's, which the publisher process
    holds its rebuilt grid to."""
    h = hashlib.sha512()
    for row in grid:
        _hash_row(h, row)
    return h.digest()


def sign_load(issuer: Issuer, key: LmsKey, heads: Path, out: Path) -> dict:
    """The load's heads (`heads`, the relay's ``load-heads.json``) signed by the warden's `issuer` in its session, before
    it serves: ``load.json`` in `out` (`load_relay.load_doc`), the load's registrations the block record pins, under
    the key and session that then certify the run's windows. Raises `load_relay.Refused` when the signer refuses."""
    shape, heads_ = lr.heads_of(json.loads(heads.read_text()))
    doc = lr.load_doc(shape, heads_, lr.sign_heads(issuer, heads_), cs.public_json(key), issuer.session)
    (out / "load.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _publisher_main(rows, results, out: Path, key: Path, session: bytes, params: WardenParams, commit,
                    load_heads: Path | None = None) -> None:
    """`PublisherProcess`'s process: the warden's key and its leaf state are this process's alone. It signs the load's
    heads first when there are any (`sign_load`), sends the public key, rebuilds each record from its released rows,
    keeps it as `Publisher` does at its close once the rebuilt grid's digest is the close marker's, and at the end
    sends the issuer's refusals and its own usage."""
    try:
        own = cs.signer(key)
        issuer = Issuer(own, session)
        if load_heads is not None:
            sign_load(issuer, own.public_key(), load_heads, out)
        results.send(("key", own.public_key()))
        publisher = Publisher(out, issuer, params, commit or WR.Cells.of)
        grids = {INGRESS: params.grid, EGRESS: params.grid, FAILS: params.fails_grid}
        filling: dict[str, list] = {link: [] for link in grids}
        B = params.cell_bytes
        while (item := rows.get()) is not None:
            link, n, data = item
            if link is None:
                continue
            grid = grids[link]
            if n is not None:
                filling[link].append(tuple(data[i * B:(i + 1) * B] for i in range(n)) + (None,) * (grid.allocation - n))
                continue
            (window, reason, digest, withheld), got = data, tuple(filling[link])
            if len(got) != grid.buckets:
                raise ValueError(f"{link} window {window} closed after {len(got)} rows, not {grid.buckets}")
            if grid_digest(got) != digest:
                raise ValueError(f"{link} window {window}: the rows rebuilt aren't the grid the warden closed")
            publisher.keep(Record(link, window, got, Status(reason)), grid, withheld=withheld)
            filling[link] = []
        results.send(("done", {"refused": publisher.close(), "usage": ar._usage()}))
        own.close()
    except BaseException:
        results.send(("error", traceback.format_exc()))
        raise


class PublisherProcess:
    """`serve`'s `publish` and `release`: `Publisher` in one spawned process, which alone holds the warden's key.
    `release` hands it each grid's row as the warden ticks (its frames as one bytes object), and `__call__` the
    record's close with its grid's SHA-512 (`grid_digest`), hashed a row at a time as each is released and held to the
    record's grid at its close: the ticker only queues them, and the queue's own thread pickles each tick's few
    kilobytes, never a whole window. The process rebuilds each record from its rows, refuses one whose grid isn't the
    marker's, and keeps, certifies and commits it, so neither a commit nor a signature takes the ticker's GIL. `key`
    is the warden's public key; `close` waits for every record and gives the issuer's refusals; `usage` is then the
    process's own. A process that exits early, killed or failed, raises at the next close or at `close`. `commit`
    stands in for `Cells.of` (a test's). With `load_heads` the process signs the load's heads before anything else
    (`sign_load`), so construction returns once ``load.json`` is written. The queue's thread is started at
    construction, with a row of the grid's size that the process drops, so that the first tick starts no thread and
    the thread's buffers have grown to a row's; `feeder` is its thread id."""

    def __init__(self, out: Path, key: Path, session: bytes, params: WardenParams, commit=None,
                 load_heads: Path | None = None) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.rows = ctx.Queue()
        self.results, results = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_publisher_main, name="warden-publisher", daemon=True,
                                   args=(self.rows, results, out, key, session, params, commit, load_heads))
        self.process.start()
        results.close()
        self.usage: dict | None = None
        self._released: dict[str, list] = {}
        self._digests: dict[str, object] = {}
        self.rows.put((None, None, bytes(params.grid.allocation * params.cell_bytes)))
        self.feeder: int | None = self.rows._thread.native_id
        self.key: LmsKey = self._receive("key")

    def _receive(self, kind: str):
        multiprocessing.connection.wait([self.results, self.process.sentinel])
        try:
            message = self.results.recv() if self.results.poll() else None
        except EOFError:
            message = None
        if message is None:
            self.process.join()
            raise RuntimeError(f"the publisher process exited ({self.process.exitcode}) before its {kind}")
        got, value = message
        if got == "error":
            raise RuntimeError(f"the publisher process failed: {value}")
        if got != kind:
            raise RuntimeError(f"the publisher process sent its {got}, not its {kind}")
        return value

    def _end(self) -> dict:
        """The process's last message once every record is kept, else its failure, raised; the rows still queued are
        then dropped, so that this process's exit doesn't wait on a reader that is gone."""
        try:
            return self._receive("done")
        except BaseException:
            self.rows.cancel_join_thread()
            raise

    def release(self, link: str, row) -> None:
        n = row_count(row)
        self._released.setdefault(link, []).append(row)
        _hash_row(self._digests.setdefault(link, hashlib.sha512()), row)
        self.rows.put((link, n, b"".join(row[:n])))

    def __call__(self, record: Record, grid: GridParams, withheld: bool = False) -> None:
        if not self.process.is_alive():
            self._end()
        released, digest = self._released.pop(record.link, []), self._digests.pop(record.link, hashlib.sha512())
        if record.grid != tuple(released):
            raise RuntimeError(f"{record.link} window {record.window}: the record's grid isn't the rows released")
        self.rows.put((record.link, None, (record.window, record.status.reason, digest.digest(), withheld)))

    def withhold(self, record: Record, grid: GridParams) -> None:
        """The warden's `withhold`: the record's close, kept as `Publisher.withhold` keeps it."""
        self(record, grid, withheld=True)

    def close(self) -> list:
        self.rows.put(None)
        done = self._end()
        self.rows.close()
        self.rows.join_thread()
        self.process.join()
        self.usage = done["usage"]
        return done["refused"]


def _quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    pick = (lambda q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)) if xs else (lambda q: None)
    return {"n": len(xs), "p50": pick(0.5), "p99": pick(0.99), "max": pick(1.0)}


def deployment_record(params: WardenParams, key: LmsKey, session: bytes, *, origin_unix: float, windows: int,
                      listen: tuple[str, int], upstream: tuple[str, int], slack: float,
                      run_id: bytes | None = None) -> dict:
    """`deployment.json`, the auditor's record of a run: the warden's public key and session, the schedule's
    parameters and their digest, the origin and where it listened, and the run id its server's requests carry."""
    return {"key": cs.public_json(key), "session": session.hex(), "link": LINK,
            "params": asdict(params), "digest": params.digest.hex(), "slack_ms": slack * 1000,
            "origin_unix": origin_unix, "windows": windows, "listen": list(listen), "upstream": list(upstream),
            "run_id": None if run_id is None else run_id.hex()}


def _memlock(libc) -> str:
    """mlockall(MCL_CURRENT | MCL_FUTURE) where the kernel permits it and the run can still grow under it: the process
    has CAP_IPC_LOCK, or RLIMIT_MEMLOCK (its soft limit raised to its hard one) is unlimited or holds the whole of what
    the process maps (VmSize, which MCL_CURRENT counts) and `LOCK_HEADROOM` more, since MCL_FUTURE then refuses any
    mapping past it; else what kept it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        soft = hard
    status = dict(x.split(":", 1) for x in Path("/proc/self/status").read_text().splitlines() if ":" in x)
    caps, mapped = int(status["CapEff"], 16), int(status["VmSize"].split()[0]) << 10
    if not caps >> CAP_IPC_LOCK & 1 and soft != resource.RLIM_INFINITY and soft < mapped + LOCK_HEADROOM:
        return (f"not locked: RLIMIT_MEMLOCK is {soft} bytes, under the {mapped} the process maps and "
                f"{LOCK_HEADROOM} to grow, and the process lacks CAP_IPC_LOCK")
    if libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
        return f"not locked: mlockall: {os.strerror(ctypes.get_errno())}"
    return "locked, current and future"


def reserve_fds(n: int) -> int:
    """Grow the process's descriptor table to hold `n` (at most RLIMIT_NOFILE's soft limit) now; returns its size. The
    kernel grows the table as descriptors pass 64, 128, 256 …, and in a process of more than one thread each growth
    waits for an RCU grace period (`expand_fdtable`'s synchronize_rcu), the thread asleep in D: tens of ms on a busy
    host. The table never shrinks."""
    n = min(n, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        os.close(fcntl.fcntl(fd, fcntl.F_DUPFD, n - 1))
    finally:
        os.close(fd)
    return n


def prepare_process(params: WardenParams) -> dict:
    """The warden's process made ready before its origin, so that no tick waits in the kernel (state D) on what the
    run's growth would otherwise ask of it then:
    - descriptors: its table holds every connection's two and every queued one's (`reserve_fds`): the opening burst of
      the trace corpus's heavy trace grew it past 256, 512 and 1024 descriptors, and the ticker waited a grace period at
      each (`stall_probe` found every D episode of the ticker in `__wait_rcu_gp` there, 14-28 ms on CPU);
    - memory: a page fault for fresh memory charges the memory cgroup, and a cgroup at its limit reclaims in the
      faulting thread. glibc's heap is kept whole (no trim) and serves every allocation under 32 MiB (no mmap), a
      lane's `RESERVE_PER_LANE` of it is touched and freed, so the run's allocations reuse resident pages, and every
      page is locked where that is permitted (`_memlock`). `serve`'s command runs on the C allocator
      (`PYTHONMALLOC=malloc`), so the interpreter's small objects come from that heap too and not from arenas it maps
      as it grows.
    Returns what `warden.json` keeps of it; what it couldn't do is also printed, as a warning."""
    reserve = max(RESERVE_MIN, params.lanes * RESERVE_PER_LANE)
    libc = ctypes.CDLL(None, use_errno=True)
    report = {"allocator": "pymalloc" if sys.flags.ignore_environment else os.environ.get("PYTHONMALLOC", "pymalloc"),
              "fds": reserve_fds(2 * params.lanes + params.queue + FD_MARGIN), "heap": None, "reserve_mb": 0}
    if hasattr(libc, "mallopt") and libc.mallopt(M_MMAP_THRESHOLD, 32 << 20) == 1 and \
            libc.mallopt(M_TRIM_THRESHOLD, (1 << 31) - 1) == 1:
        report["heap"] = "kept whole, no mmap under 32 MiB"
        libc.malloc.restype, libc.malloc.argtypes, libc.free.argtypes = ctypes.c_void_p, [ctypes.c_size_t], \
            [ctypes.c_void_p]
        chunks = [p for p in (libc.malloc(1 << 20) for _ in range(reserve >> 20)) if p]
        for p in chunks:
            ctypes.memset(p, 0, 1 << 20)
        for p in chunks:
            libc.free(p)
        report["reserve_mb"] = len(chunks)
    else:
        report["heap"] = "not glibc's: it may give memory back and map it again during the run"
    report["mlock"] = _memlock(libc)
    for key, value in report.items():
        if (key == "allocator" and value != "malloc") or str(value).startswith("not"):
            print(f"warden-serve: {key}: {value}", file=sys.stderr, flush=True)
    return report


async def serve(params: WardenParams, out: Path, *, upstream: tuple[str, int], listen: tuple[str, int], key: Path,
                session: bytes, windows: int, start_in: float = 5.0, slack: float = 0.025,
                announce=cs._announce, probe: bool = False, process: dict | None = None,
                load_heads: Path | None = None, paced_ms: float = cs.PACED_MS) -> dict:
    """One run of the warden on the host's clock (the module docstring's `serve`), its records published by a
    `PublisherProcess` that holds the key in `key` and first signs the load's heads `load_heads` when given, its
    threads sampled by `stall_probe` when `probe`, each bucket released by its pacer `paced_ms` after its boundary
    (`Warden(paced=…)`; 0: written on its loop). Returns what `warden.json` holds, with `process`, what
    `prepare_process` did. The loop's errors are kept, not logged: the ticker writes no file."""
    out.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    errors: list[str] = []
    loop.set_exception_handler(lambda _, context: errors.append(f"{context.get('message')}: "
                                                               f"{context.get('exception')!r}"))
    publish = PublisherProcess(out, key, session, params, load_heads=load_heads)
    origin = time.monotonic() + start_in
    warden = Warden(params, lambda: asyncio.open_connection(*upstream), origin=origin, windows=windows,
                    publish=publish, withhold=publish.withhold, release=publish.release, slack=slack,
                    paced=paced_ms / 1000 if paced_ms else None)
    server = await asyncio.start_server(warden.handle, *listen)
    sampler = stall_probe.start(out / "stalls.json", origin, {"warden": os.getpid(), "publisher": publish.process.pid},
                                {threading.get_native_id(): "ticker", publish.feeder: "feeder"}) if probe else None
    deployment = deployment_record(params, publish.key, session, origin_unix=time.time() + origin - time.monotonic(),
                                   windows=windows, listen=(listen[0], server.sockets[0].getsockname()[1]),
                                   upstream=upstream, slack=slack, run_id=warden.run_id)
    (out / "deployment.json").write_text(json.dumps(deployment, indent=1))
    announce({"listening": deployment["listen"], "deployment": str(out / "deployment.json")})
    try:
        await warden.run()
    finally:
        server.close()
        refused = await loop.run_in_executor(None, publish.close)
        stalls = sampler and stall_probe.stop(sampler, out / "stalls.json")
    lateness = [x * 1000 for x in warden.lateness]
    log = {"connections": warden.log, "ordinals": warden.ordinals, "refused": warden.refused,
           "refused_windows": sorted(warden.refused_windows), "ticks": warden.ticks,
           "lateness_ms": _quantiles(lateness) | {"over_slack": sum(x > slack * 1000 for x in lateness)},
           "issuer_refused": [[link, d.value, w] for link, d, w in refused], "usage": ar._usage(),
           "publisher_usage": publish.usage, "process": process, "stalls": stalls, "loop_errors": errors,
           "ticks_ms": [round(x, 3) for x in lateness]}
    if warden.paced is not None:
        log["paced"] = {"ms": paced_ms, "columns": ["bucket", "arrived_ms", "start_ms", "end_ms", "written"],
                        "releases": [[r.t, round(r.arrived * 1000, 3), round(r.start * 1000, 3),
                                      round(r.end * 1000, 3), r.written] for r in warden.releases]}
    (out / "warden.json").write_text(json.dumps(log, indent=1))
    announce({"served": True, "out": str(out)})
    return log


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def connection_log(index: int, at: float, sent: float, request: bytes, shim: Shim, origin: float,
                   id_payload: int | None = None) -> dict:
    """One connection's entry in `client.json`; times in ms after the origin. An id request's response is read as
    id-form (`parse_id_response`, at P = `id_payload`)."""
    data = bytes(shim.data)
    return {"index": index, "at": at, "sent_ms": round(sent * 1000, 3), "request_bytes": len(request),
            "request_sha256": _sha(request), "ordinal": shim.ordinal, "outcome": shim.outcome,
            "kinds": "".join(KIND_CHARS[k] for _, k in shim.arrivals),
            "arrivals_ms": [round((t - origin) * 1000, 3) for t, _ in shim.arrivals], "data_bytes": len(shim.data),
            "data_sha256": _sha(data),
            "response": parse_response(data) if id_payload is None else parse_id_response(data, id_payload)}


async def client(deployment: dict, traffic: list[dict], out: Path, clock=time.time) -> dict:
    """Each request of `traffic` on a connection of its own at its time after the deployment's origin, through the
    shim; returns what `client.json` holds. An item with ``ids`` (`id_traffic`) is sent as its warden-form id request
    (`active.id_request`, offset 0) and its response read as id-form."""
    params = WardenParams(**deployment["params"])
    origin, listen = deployment["origin_unix"], tuple(deployment["listen"])
    host = "%s:%d" % tuple(deployment["upstream"])

    async def one(index: int, item: dict) -> dict:
        ids = "ids" in item
        request = (id_request(**item["ids"], frame_bytes=params.cell_bytes, offset=0) if ids
                   else http_request(item["body"], host))
        await asyncio.sleep(max(0.0, origin + item["at"] - clock()))
        sent = clock()
        reader, writer = await asyncio.open_connection(*listen)
        try:
            writer.write(cells(request, params.cell_bytes))
            shim = await Shim(params.cell_bytes, clock).read(reader)
        finally:
            writer.close()
        return connection_log(index, item["at"], sent - origin, request, shim, origin,
                              params.payload if ids else None)

    got = await asyncio.gather(*(one(i, item) for i, item in enumerate(traffic)), return_exceptions=True)
    doc = {"connections": [g if isinstance(g, dict) else {"index": i, "error": f"{type(g).__name__}: {g}"}
                           for i, g in enumerate(got)]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "client.json").write_text(json.dumps(doc))
    return doc


# The offline check.

def read_records(run: Path, params: WardenParams) -> dict[str, list[Record]]:
    return {name: [ar.unpack_record(b, params.cell_bytes) for b in ar.read_blobs(run / f"{name}.records")]
            for name in CERTIFIED}


def read_certified(path: Path) -> tuple[dict[tuple[str, Direction, int], CertifiedRecord], list]:
    """`certified.jsonl`: the certified records by (link, direction, window), and the link-windows the signer
    refused (not those the warden withheld, `Publisher.withhold`)."""
    got, refused = {}, []
    for line in path.read_text().splitlines() if path.exists() else []:
        doc = json.loads(line)
        if "refused" in doc:
            if not doc.get("withheld"):
                refused.append(doc["refused"])
        else:
            record = cs.certified_from_json(doc)
            got[(record.link, record.direction, record.window)] = record
    return got, refused


def read_cells(run: Path) -> list[tuple[dict, list[bytes], bytes]]:
    """`cells.jsonl` and `cells.salts`, line by line: each link-window's registration (or refusal) and cost, its rows'
    salts and its wiring's (`warden_rows.read_witness`)."""
    path = run / "cells.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    salts = ar.read_blobs(run / "cells.salts")
    if len(salts) != len(lines):
        raise ValueError(f"cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    return [(doc, *WR.read_witness(s)) for doc, s in zip(lines, salts)]


def check_cells(run: Path, dep: dict, params: WardenParams, records: dict[str, list[Record]]) -> tuple[dict, list]:
    """Every link-window of the wire's two records has a registration that `warden_rows.check_registration` accepts
    under the warden's key and session, whose head is the record's rows and wiring under their salts; the per-window
    cost."""
    key, session = cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    got = {}
    for doc, salts, wiring_salt in read_cells(run):
        if "registration" in doc:
            r = doc["registration"]
            got[(r["link"], r["direction"], r["window"])] = (doc, salts, wiring_salt)
    problems, costs, ok_n = [], [], 0
    for name in (INGRESS, EGRESS):
        link, direction = CERTIFIED[name]
        by_window = {r.window: r for r in records[name]}
        for w in range(dep["windows"]):
            doc, salts, wiring_salt = got.get((link, direction.value, w), (None, None, None))
            ok = False
            if doc is not None and w in by_window:
                costs.append(doc["cost"])
                try:
                    head, _ = WR.check_registration(doc["registration"], key, session, params, link, direction, w)
                    ok = WR.Cells.of(link, direction, by_window[w], params, session, salts, wiring_salt).head == head
                except ValueError as e:
                    problems.append(f"{name} window {w}'s cells: {e}")
            ok_n += ok
            if not ok:
                problems.append(f"{name} window {w}'s cells aren't registered")
    summary = {"registered": ok_n, "of": 2 * dep["windows"], "rows": sorted({c["rows"] for c in costs}),
               "row_bytes": sorted({c["row_bytes"] for c in costs}),
               "commit_ms": _quantiles([c["commit_ms"] for c in costs]),
               "sign_ms": _quantiles([c["sign_ms"] for c in costs])}
    return summary, problems


def egress_cells(records: list[Record], params: WardenParams) -> dict[int, list[tuple[int, int]]]:
    """Each connection's egress cells as (bucket, kind), in the records' order."""
    out: dict[int, list[tuple[int, int]]] = {}
    for record in sorted(records, key=lambda r: r.window):
        for b, row in enumerate(record.grid):
            for f in row:
                if f is not None:
                    out.setdefault(int.from_bytes(f[1:5], "big"), []).append((record.window * params.window + b, f[0]))
    return out


def check_load(run: Path, dep: dict, key: LmsKey, session: bytes) -> tuple[dict | None, list[str]]:
    """A run's load read as the verifier reads it, None when it carried none: ``load.json`` signed under the
    deployment's own key and session, every value's registration accepted (`load_relay.check_load`), each at the root
    the relay committed (``load-heads.json``)."""
    doc_path, heads_path = run / "load.json", run / "load-heads.json"
    if not doc_path.exists() and not heads_path.exists():
        return None, []
    try:
        doc = json.loads(doc_path.read_text())
    except (OSError, ValueError):
        return {"values": 0, "registered": 0, "relayed": False}, ["load: the relay's heads aren't signed (no load.json)"]
    problems = [] if (doc.get("key"), doc.get("session")) == (dep["key"], dep["session"]) else \
        ["load: not signed under the deployment's key and session"]
    summary, found = lr.check_load(doc, key, session)
    try:
        heads = json.loads(heads_path.read_text())
        relayed = (heads.get("program"), heads.get("link"), list(heads.get("roots", {}).items())) == \
            (doc.get("program"), doc.get("link"), [(r.get("value"), r.get("root")) for r in doc.get("values", [])])
    except (OSError, ValueError, AttributeError):
        relayed = False
    if not relayed:
        problems.append("load: its registrations aren't the relay's heads (load-heads.json)")
    return summary | {"relayed": relayed}, problems + found


def check(run: Path) -> dict:
    """The offline check of a served run (the module docstring's `check`)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    certified, refused = read_certified(run / "certified.jsonl")
    records = read_records(run, params)
    problems = [] if params.digest.hex() == dep["digest"] else ["the deployment's digest isn't its parameters'"]
    load, load_problems = check_load(run, dep, key, session)
    problems += load_problems
    windows = []
    for name, (link, direction) in CERTIFIED.items():
        got = {r.window: r for r in records[name]}
        grid = params.fails_grid if name == FAILS else params.grid
        for w in range(dep["windows"]):
            r, c = got.get(w), certified.get((link, direction, w))
            ok = (r is not None and c is not None and c.verify(key, session)
                  and c.head == record_head(link, direction, w, r.grid, r.status, grid, params.digest))
            windows.append({"record": name, "direction": direction.value, "window": w, "certified": ok,
                            "status": None if r is None else r.status.reason and r.status.reason.value,
                            "cells": None if r is None else sum(f is not None for row in r.grid for f in row)})
            if not ok:
                problems.append(f"{name} window {w} isn't certified")
            elif r.status.reason is not None:
                problems.append(f"{name} window {w}: {r.status.reason.value}")
    cells_summary, cells_problems = check_cells(run, dep, params, records)
    problems += cells_problems
    verdict = audit(params, records[INGRESS], records[EGRESS], records[FAILS])
    if not verdict["accepted"]:
        problems += [f"audit: {p}" for w in verdict["windows"] for p in w["problems"]]
    audited = {c["ordinal"]: c for c in verdict["connections"]}
    schedule = egress_cells(records[EGRESS], params)
    client_doc = json.loads((run / "client.json").read_text()) if (run / "client.json").exists() else {}
    tau, rows, lags = params.bucket_ms, [], []
    for conn in client_doc.get("connections", []):
        if "error" in conn or conn["outcome"] == "refused":
            rows.append({"index": conn["index"], "outcome": conn.get("outcome") or conn["error"]})
            if "error" in conn:
                problems.append(f"client connection {conn['index']}: {conn['error']}")
            continue
        o, a = conn["ordinal"], audited.get(conn["ordinal"], {})
        sched = schedule.get(o, [])
        lag = [ms - t * tau for (t, _), ms in zip(sched, conn["arrivals_ms"])]
        lags += lag
        row = {"index": conn["index"], "ordinal": o, "outcome": conn["outcome"], "cells": len(sched),
               "same_cells": "".join(KIND_CHARS[k] for _, k in sched) == conn["kinds"],
               "in_bucket": sum(0 <= x < tau for x in lag), "request_certified":
                   a.get("ingress_sha256") == conn["request_sha256"],
               "response_certified": a.get("egress_sha256") == conn["data_sha256"], **conn["response"]}
        rows.append(row)
        if not (row["same_cells"] and row["in_bucket"] == row["cells"] and row["request_certified"]
                and row["response_certified"]):
            keys = ("same_cells", "in_bucket", "cells", "request_certified", "response_certified")
            problems.append(f"connection {o}: { {k: row[k] for k in keys} }")
        if conn["outcome"] != "end" or not row["complete"] or (row["status"] != 200 and row.get("ids") is None):
            problems.append(f"connection {o}: {conn['outcome']}, HTTP {row['status']}, complete {row['complete']}")
    warden_log = json.loads((run / "warden.json").read_text()) if (run / "warden.json").exists() else {}
    served = [r for r in rows if "ordinal" in r]
    B = params.cell_bytes
    summary = {
        "requests": len(rows), "ended": sum(r["outcome"] == "end" for r in rows),
        "refused": sum(r["outcome"] == "refused" for r in rows),
        "failed": sorted({r["outcome"] for r in rows} - {"end", "refused"}),
        "windows": dep["windows"], "certified_windows": sum(w["certified"] for w in windows),
        "certificates": len(certified), "signer_refused": refused,
        "withheld_windows": warden_log.get("refused_windows", []), "cell_rows": cells_summary,
        "cells": verdict["cells"], "wire_bits": {k: n * B * 8 for k, n in verdict["cells"].items()},
        "data_bytes": {"ingress": sum(c["ingress_bytes"] for c in verdict["connections"]),
                       "egress": sum(c["egress_bytes"] for c in verdict["connections"])},
        "cells_in_their_bucket": [sum(r["in_bucket"] for r in served), sum(r["cells"] for r in served)],
        "arrival_lag_ms": _quantiles(lags), "tick_lateness_ms": warden_log.get("lateness_ms"),
        "ticker_stalls": None if not warden_log.get("stalls") else
        warden_log["stalls"]["by_label"].get("ticker", {"episodes": 0, "max_ms": 0.0, "wchan": {}}),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in served),
        "prompt_tokens": [min((r.get("prompt_tokens") or 0 for r in served), default=None),
                          max((r.get("prompt_tokens") or 0 for r in served), default=None)],
        "finish_reasons": {f: sum(r.get("finish_reason") == f for r in served)
                           for f in sorted({str(r.get("finish_reason")) for r in served})},
        **({} if load is None else {"load": load}),
    }
    return {"accepted": not problems, "problems": problems[:50], "summary": summary, "windows": windows,
            "connections": rows}


def table(result: dict) -> str:
    s = result["summary"]
    load = "" if "load" not in s else (f"load {s['load']['registered']}/{s['load']['values']} values registered"
                                       f"{'' if s['load'].get('relayed') else ', not the relay heads'}; ")
    return (f"{'ACCEPTED' if result['accepted'] else 'REFUSED'}: {load}{s['ended']}/{s['requests']} ended, "
            f"{s['refused']} refused, failed {s['failed']}; {s['certified_windows']}/{len(CERTIFIED) * s['windows']} "
            f"link-windows "
            f"certified; cells in their bucket {s['cells_in_their_bucket'][0]}/{s['cells_in_their_bucket'][1]}; "
            f"cells {s['cells']}; tick lateness {s['tick_lateness_ms']}; ticker D episodes "
            f"{s.get('ticker_stalls') or 'not probed'}; cell rows {s['cell_rows']['registered']}/"
            f"{s['cell_rows']['of']} link-windows registered, {s['cell_rows']['rows']} rows a window, commit ms "
            f"{s['cell_rows']['commit_ms']}, sign ms {s['cell_rows']['sign_ms']}")


# The rehearsal.

def traffic_of(args, model: str) -> list[dict]:
    if getattr(args, "prompts", None) is not None:
        params = _params(args)
        return prompt_traffic(read_prompts(args.prompts.read_bytes(), params), args.requests, model=model, params=params,
                              max_tokens=min(args.max_tokens, args.egress_cells - 1), rate=args.rate, seed=args.seed,
                              temperature=args.temperature, top_p=args.top_p, window_per_group=args.window_per_group)
    if getattr(args, "ids", False):
        return id_traffic(args.requests, model=model, rate=args.rate, seed=args.seed, prompt_tokens=(16, args.prompt_max),
                          max_tokens=(1, min(args.max_tokens, args.egress_cells - 1)), temperature=args.temperature,
                          top_p=args.top_p)
    return varied_traffic(args.requests, model=model, rate=args.rate, seed=args.seed,
                          prompt_tokens=(16, args.prompt_max), max_tokens=(16, args.max_tokens),
                          temperature=args.temperature, top_p=args.top_p)


def _params(args) -> WardenParams:
    return WardenParams(**{f.name: getattr(args, f.name) for f in dataclasses.fields(WardenParams)})


def _flags(args) -> list[str]:
    return [x for f in dataclasses.fields(WardenParams) for x in (f"--{f.name.replace('_', '-')}",
                                                                 str(getattr(args, f.name)))]


def await_model(url: str, seconds: float, poll: float = 5.0) -> str:
    """The endpoint's one model (`cs.served_model`), asked again each `poll` seconds while it doesn't answer, for at most
    `seconds`: after the load the prover starts its server on the weights its sink wrote."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return cs.served_model(url)
        except (OSError, ValueError, http.client.HTTPException) as e:
            if time.monotonic() >= deadline:
                raise SystemExit(f"{url} didn't answer within {seconds:g} s of the load: {e}") from None
            time.sleep(poll)


def _addresses(text: str) -> list[tuple[str, int]]:
    return [cs._address(a) for a in text.split(",")]


def rehearse(args) -> int:
    if args.prompts is not None and not args.ids:
        raise SystemExit("--prompts: id requests' prompts, with --ids")
    if args.window_per_group and args.prompts is None:
        raise SystemExit("--window-per-group: groups of the --prompts")
    load = args.load_store is not None
    if load and not (args.load_sinks and args.load_program and lr.HEX64.fullmatch(args.load_program)):
        raise SystemExit("--load-store: with --load-sinks and --load-program, the reading program's digest in hex")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = _params(args)
    me = [sys.executable, str(HERE / "warden_serve.py")]
    scratch = Path(tempfile.mkdtemp(prefix="warden-serve-"))
    procs: list[subprocess.Popen] = []
    try:
        cs.load_key(scratch / "warden", args.height)
        if load:
            try:
                lr.relay_load(args.load_store, lr.SHAPES[args.load_shape], args.load_sinks, args.load_program, out,
                              workers=args.load_workers, progress=lr.progress_lines())
            except lr.Refused as e:
                (out / "check.json").write_text(json.dumps({"accepted": False, "problems": [f"load: {e}"],
                                                            "summary": {}}, indent=1))
                print(f"REFUSED: the load: {e}", flush=True)
                return 1
        if args.fake:
            ids = ["--id-payload", str(params.payload)] if args.ids else []
            fake = subprocess.Popen(me + ["fake", "--listen", "127.0.0.1:0", "--token-ms", str(args.token_ms),
                                          "--first-ms", str(args.first_ms), *ids], stdout=subprocess.PIPE, text=True)
            procs.append(fake)
            upstream = tuple(json.loads(fake.stdout.readline())["listening"])
            model = "fake"
        else:
            upstream, url = args.upstream, "http://%s:%d" % args.upstream
            model = args.model or (await_model(url, args.load_ready_s) if load else cs.served_model(url))
        traffic = traffic_of(args, model)
        (out / "traffic.json").write_text(json.dumps(traffic))
        window_s = params.window * params.bucket_ms / 1000
        windows = math.ceil((traffic[-1]["at"] + args.tail) / window_s)
        if args.window_per_group:
            grouped = groups(len(traffic), len(read_prompts(args.prompts.read_bytes(), params)), params,
                             traffic[0]["ids"]["max_tokens"])
            (out / GROUPS).write_text(json.dumps({"groups": grouped}, indent=1))
            windows = grouped[-1]["windows"][1]
        with open(out / "serve.log", "w") as log:
            serving = subprocess.Popen(me + ["serve", "--upstream", "%s:%d" % upstream, "--listen", "127.0.0.1:0",
                                             "--out", str(out), "--key", str(scratch / "warden"), "--session",
                                             secrets.token_hex(32), "--windows", str(windows), "--start-in",
                                             str(args.start_in), "--slack", str(args.slack), "--paced-ms",
                                             str(args.paced_ms), *_flags(args),
                                             *(["--load-heads", str(out / "load-heads.json")] if load else [])],
                                       stdout=subprocess.PIPE, stderr=log, text=True)
            procs.append(serving)
            if "listening" not in json.loads(serving.stdout.readline() or "{}"):
                raise SystemExit("serve exited before listening (serve.log)")
            with open(out / "client.log", "w") as clog:
                client_exit = subprocess.run(me + ["client", "--deployment", str(out / "deployment.json"),
                                                   "--traffic", str(out / "traffic.json"), "--out", str(out)],
                                             stdout=clog, stderr=subprocess.STDOUT).returncode
            serve_exit = serving.wait()
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        shutil.rmtree(scratch)
    result = check(out) | {"upstream": list(upstream), "model": model, "exit": {"serve": serve_exit,
                                                                              "client": client_exit}}
    (out / "check.json").write_text(json.dumps(result, indent=1))
    print(table(result), flush=True)
    return 0 if result["accepted"] else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="warden_serve.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    k = sub.add_parser("keys", help="make or load an LMS key and write its public.json")
    t = sub.add_parser("traffic", help="write varied traffic")
    f = sub.add_parser("fake", help="a stand-in for vLLM's streamed /v1/completions")
    s = sub.add_parser("serve", help="the warden in front of the server, for one run")
    c = sub.add_parser("client", help="the traffic, through the shim")
    x = sub.add_parser("check", help="the offline check of a run")
    r = sub.add_parser("rehearse", help="traffic, then serve, client and check")
    pr = sub.add_parser("probe", help="the traffic's first requests straight to the server, no warden")
    for q in (k,):
        q.add_argument("--dir", required=True, type=Path)
    for q in (k, r):
        q.add_argument("--height", type=int, default=cs.HEIGHT, help="the LMS key's tree height")
    for q in (t, s, c, r, pr):
        q.add_argument("--out", required=True, type=Path)
    for q in (t, r, pr):
        q.add_argument("--model", help="the endpoint's model name (default: the one /v1/models lists)")
        q.add_argument("--requests", type=int, default=100)
        q.add_argument("--rate", type=float, default=0.5, help="Poisson arrivals a second")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--max-tokens", type=int, default=256, help="the most max_tokens a request draws")
        q.add_argument("--prompt-max", type=int, default=1024, help="the most prompt tokens a request draws")
        q.add_argument("--temperature", type=float, help="every request's temperature (default: drawn, 0, 0.7 or 1)")
        q.add_argument("--top-p", type=float, help="every request's top_p (default: none sent, the endpoint's own)")
    t.set_defaults(model="model")
    for q in (f, s):
        q.add_argument("--listen", required=True, type=cs._address, help="HOST:PORT (port 0: any)")
    for q in (f, r):
        q.add_argument("--token-ms", type=float, default=20.0, help="the fake endpoint's milliseconds a token")
        q.add_argument("--first-ms", type=float, default=100.0,
                       help="the fake endpoint's milliseconds to its first token, or to its whole id-form response "
                            "(past the warden's delay D, every connection fails late: a void run)")
    for q in (s, r):
        q.add_argument("--start-in", type=float, default=5.0, help="the origin, seconds after the warden is up")
        q.add_argument("--slack", type=float, default=0.025, help="seconds a tick may run late before it faults")
        q.add_argument("--paced-ms", type=float, default=cs.PACED_MS,
                       help="when the warden's pacer releases each bucket, ms after its boundary (past the slack, "
                            "before the next boundary); 0: the warden writes each bucket on its loop")
        for field in dataclasses.fields(WardenParams):
            q.add_argument(f"--{field.name.replace('_', '-')}", type=int, default=SCHEDULE[field.name])
    s.add_argument("--upstream", required=True, type=cs._address, help="the server's HOST:PORT")
    s.add_argument("--key", required=True, type=Path, help="the warden's key directory (keys)")
    s.add_argument("--session", required=True, help="the certifier session, 32 bytes in hex")
    s.add_argument("--windows", required=True, type=int)
    s.add_argument("--stall-probe", action=argparse.BooleanOptionalAction, default=True,
                   help="sample the warden's threads every millisecond for D episodes (stall_probe.py; stalls.json)")
    s.add_argument("--load-heads", type=Path,
                   help="the load's heads (load_relay.py's load-heads.json), signed first in the session: load.json")
    r.add_argument("--load-store", type=Path,
                   help="relay the load first: the store's directory (load_relay.py write), its stream sent to "
                        "--load-sinks and committed as it crosses, its heads signed by the warden before it serves")
    r.add_argument("--load-shape", choices=sorted(lr.SHAPES), default="h200")
    r.add_argument("--load-sinks", type=_addresses, help="the H200 sink's HOST:PORT, one a stream, comma-separated")
    r.add_argument("--load-program", help="the reading program's digest, 64 bytes in lowercase hex")
    r.add_argument("--load-workers", type=int, default=os.cpu_count() or 1, help="processes committing the load")
    r.add_argument("--load-ready-s", type=float, default=900.0,
                   help="seconds the endpoint has after the load to answer /v1/models")
    up = r.add_mutually_exclusive_group(required=True)
    up.add_argument("--upstream", type=cs._address, help="the server's HOST:PORT (an OpenAI endpoint)")
    up.add_argument("--fake", action="store_true", help="start the fake endpoint and use it")
    pr.add_argument("--upstream", required=True, type=cs._address, help="the server's HOST:PORT (an OpenAI endpoint)")
    pr.add_argument("--in-flight", type=int, default=4)
    r.add_argument("--tail", type=float, default=60.0, help="seconds the run lasts past the last request's time")
    r.add_argument("--ids", action="store_true",
                   help="warden-form id requests (id_traffic), answered id-form (the fake's --id-payload): the scan's "
                        "traffic (scan_zk.py --warden-run)")
    r.add_argument("--prompts", type=Path,
                   help="with --ids: the requests' prompts, a JSON array of token-id arrays, request j's the (j mod n)-th "
                        "of the n, each with exactly --max-tokens (at most --egress-cells - 1)")
    r.add_argument("--window-per-group", action="store_true",
                   help="with --prompts: the requests in groups of the n prompts (at most --lanes), group g's half a "
                        "bucket into window g·G, G = group_windows, the run exactly its groups' windows (groups.json)")
    f.add_argument("--id-payload", type=int, help="answer every request id-form (id_response) at P bytes an event")
    c.add_argument("--deployment", required=True, type=Path)
    c.add_argument("--traffic", required=True, type=Path)
    x.add_argument("--run", required=True, type=Path)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.command == "keys":
        cs.load_key(args.dir, args.height)
        print((args.dir / "public.json").read_text())
        return 0
    if args.command == "traffic":
        args.out.write_text(json.dumps(traffic_of(args, args.model)))
        return 0
    if args.command == "probe":
        model = args.model or cs.served_model("http://%s:%d" % args.upstream)
        result = asyncio.run(probe(args.upstream, traffic_of(args, model), args.in_flight))
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("first_s", "bytes_per_s", "statuses")}))
        return 0
    if args.command == "fake":
        async def fake() -> None:
            server = await asyncio.start_server(lambda rd, wr: fake_endpoint(rd, wr, first_s=args.first_ms / 1000,
                                                                             token_s=args.token_ms / 1000,
                                                                             id_payload=args.id_payload), *args.listen)
            cs._announce({"listening": [args.listen[0], server.sockets[0].getsockname()[1]]})
            await server.serve_forever()
        asyncio.run(fake())
        return 0
    if args.command == "serve":
        out = args.out.resolve()
        if out == args.key.resolve() or out in args.key.resolve().parents:
            raise SystemExit(f"the key {args.key} is inside --out {args.out}, which is published")
        session = bytes.fromhex(args.session)
        if len(session) != 32:
            raise SystemExit("--session is 32 bytes in hex")
        own = argv is None or list(argv) == sys.argv[1:]
        if own and os.environ.get("PYTHONMALLOC") != "malloc" and C_ALLOCATOR not in os.environ:
            os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]],
                      os.environ | {"PYTHONMALLOC": "malloc", C_ALLOCATOR: "1"})
        params = _params(args)
        ar._quiet_runtime()
        process = prepare_process(params)
        for name in ("PYTHONMALLOC", C_ALLOCATOR):
            os.environ.pop(name, None)
        asyncio.run(serve(params, args.out, upstream=args.upstream, listen=args.listen, key=args.key,
                          session=session, windows=args.windows, start_in=args.start_in, slack=args.slack,
                          probe=args.stall_probe, process=process, load_heads=args.load_heads,
                          paced_ms=args.paced_ms))
        return 0
    if args.command == "client":
        ar._quiet_runtime()
        traffic = json.loads(args.traffic.read_text())
        reserve_fds(len(traffic) + FD_MARGIN)
        asyncio.run(client(json.loads(args.deployment.read_text()), traffic, args.out))
        return 0
    if args.command == "check":
        result = check(args.run)
        (args.run / "check.json").write_text(json.dumps(result, indent=1))
        print(table(result))
        return 0 if result["accepted"] else 1
    return rehearse(args)


if __name__ == "__main__":
    raise SystemExit(main())
