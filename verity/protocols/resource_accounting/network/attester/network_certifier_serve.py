"""network-certifier-serve: the active network certifier (`verity.protocols.resource_accounting.network.verifier.network_certifier`) on one host, a relay in front of any TCP
server, and the offline check of what it published.

**Topology.** Each client connection reaches the network certifier's listener. The network certifier opens one
connection to the server (`--upstream HOST:PORT`) for it and forwards each direction's bytes unchanged in content, on
its schedule, in B-byte cells; the client speaks cells through the thin, workload-blind shim (`network_certifier.cells` sending,
`network_certifier.Shim` reading):

    client ──cells──▶ network certifier ──bytes──▶ server ──bytes──▶ network certifier ──cells──▶ client

**The egress form** (`--egress`, `network_certifier`'s "The egress form"). `serve` releases the id form, the egress record's, by
default: only `io_tie`'s id-form cells leave, and a server's other bytes fail their connection (format). `rehearse`
names its traffic's: `id` with `--ids`, whose answers are id-form responses, else `bytes`, a relay of vLLM's stock
framing and the fake's text whose parameters no period check takes. Beside that gate nothing here is the network
certifier's but bytes. Tonight's clients send HTTP/1.1 requests for vLLM's streamed
`/v1/completions` with `Connection: close`, so the server's end of stream ends egress (the network certifier never
half-closes toward a server). The server gets each request whole at its release, with its place in the network
certifier's release order, its release bucket and the network certifier's run id (`release`), which `deployment.json`
names (`run_id`). With `--watermark` the network certifier also sends the server the watermark stream (`release`'s
`POST /verity/watermark`, one line `t n` a tick, naming the releases its server connections lost) on a connection of
its own, which it never reads; `warden.json` keeps its last line, the lost releases and why it stopped, if it did
(`watermark`).

**What it can't enforce here.**
- Run directly, the network certifier runs on the server's host (`trusted-device` stands for the auditor's device). The
  gate's `warden-rehearse` (`verity/protocols/compose/verifier/live_challenger/auditor.py`) runs it on the auditor's device instead, the server
  reached through an ssh tunnel to the peer's loopback. Either way the server's host can reach anyone by any other
  route, which nothing here sees.
- TLS is not terminated: both legs are plain TCP.
- The rehearsal's client is on the same host, so its link is loopback, and its arrival times are read on the network
  certifier's own clock, its host's raw clock (`origin_raw`).
- The keys are LMS at H = `ck.HEIGHT` (2^10 one-time keys) unless `--height` says otherwise, where the deployed key's
  is H = 20 (the auditor gate's default, `verity/protocols/compose/verifier/live_challenger/auditor.py`).

**Traffic** (`traffic`, `varied_traffic`): varied: Poisson arrivals at `--rate` a second,
prompts of 16–1024 tokens (log-uniform; words at 0.75 a token, so approximately), `max_tokens` of 16 … `--max-tokens`
(log-uniform) under real EOS, temperature 0, 0.7 or 1, and a seed on about half the requests. `--temperature` and
`--top-p` put the same on every request instead (the gate's `warden-rehearse` always gives both): an endpoint applies
its own generation config to a request that omits them. `rehearse --ids` sends the scan's traffic instead (`id_traffic`,
at the same sampling when given): network-certifier-form id requests (`id_form.id_request`, offset 0), each answered by
the fake endpoint as an id-form response (`id_response`: one `io_tie.id_event` cell an output id, then `io_tie.TRAILER`,
at most `egress_cells` − 1 ids), what `scan_zk.py prove --warden-run` proves; the gate's `warden-rehearse --scan` sends
it to the peer's endpoint, at scan_zk's schedule. `--prompts FILE` (with `--ids`) gives the id requests' prompts instead
(`read_prompts`, `prompt_traffic`: request j's the (j mod n)-th, each with exactly `--max-tokens`), and
`--window-per-group` sends them in groups of the n, group g's half a bucket into execution phase g·G (`group_windows`),
so that group g's cells are exactly windows g·G … (g+1)·G − 1 (`groups.json`), and the run is exactly those windows: a
loop's period of requests a group, as the gate's `warden-rehearse --scan --prompts --window-per-group` runs it. Under
`--lengths 2` the run is longer where it must be to admit its last group (`closing_windows`).

**Subcommands**, in this directory with `verity` installed (`uv sync`):

    python network_certifier_serve.py keys --dir KEYDIR [--height 10]
    python network_certifier_serve.py traffic --out TRAFFIC.json --model NAME [--requests 100] [--rate 0.5] [--seed 0]
        [--temperature T] [--top-p P]
    python network_certifier_serve.py fake --listen HOST:PORT [--token-ms 20] [--id-payload P]
    python network_certifier_serve.py serve --upstream HOST:PORT --listen HOST:PORT --out DIR --key KEYDIR --session HEX
        --windows N [--start-in 5] [--load-heads FILE] [--paced-ms 50] [--watermark] [schedule] [--egress id|bytes]
        [--max-tokens-cap N]
    python network_certifier_serve.py client --deployment DIR/deployment.json --traffic TRAFFIC.json --out DIR
    python network_certifier_serve.py check --run DIR [--key-sha512 HEX --session HEX [--registrations FILE]]
    python network_certifier_serve.py rehearse (--upstream HOST:PORT | --fake) --out DIR [--requests 100] [--temperature T]
        [--top-p P] [--ids [--users NAME] [--prompts FILE [--window-per-group]]] [--paced-ms 50] [--watermark] [--load-store DIR --load-sinks HOST:PORT,...
        --load-program HEX [--load-shape h200|tiny] [--load-workers N] [--load-ready-s 900]] [schedule]
        [--egress id|bytes] [--max-tokens-cap N]
    python network_certifier_serve.py session (--upstream HOST:PORT | --fake) --listen HOST:PORT [--listen-fd FD] --windows N
        --out DIR [--height 10] [--start-in 5] [--paced-ms 50] [--watermark] [schedule] [--egress id|bytes]
        [--max-tokens-cap N]

- `keys` makes the key directory: the LMS key, its leaf state and, with a key made there, its device state
  (`make_key`; `certificate.DeviceState`): the key's life so far, in milliseconds on the network certifier's clock,
  and the sessions started under it.
- `serve` runs on the host's raw clock and no other (the device's "The clock"), and refuses to start where the host
  has none. It holds the key's device state from before its first tick until its publisher has closed, so a second
  `serve` under the key is refused meanwhile, as is a session id the key has started. The device state gives the
  session its base, the key's life before the run, and its ordinal, the number of the key's sessions before it, and
  moves the life on by the run's whole span, `--windows` execution phases of T ticks, before the base is signed: a run
  that ends early has used its whole span. The publisher signs the start's certificate at that base and ordinal
  (`certificate.CertifiedStart`) before the origin is set.
  `serve` then writes `deployment.json` (the network certifier's public key, the session, the schedule's parameters
  and their digest, the start's certificate, the origin) and beside it `endpoints.json` (the listener and the
  endpoint, which a client reads and nothing publishes). The origin is there on the network certifier's clock
  (`origin_raw`) and as wall time: `origin_unix` is one reading of the host's wall time at the origin, and wall
  time minus it is not the network certifier's bucket, since the host steers its wall time and not the raw clock.
  `serve` runs the network certifier for `--windows` windows, each bucket released by its
  pacer `--paced-ms` after its boundary (`network_certifier`'s "The pacer"; 0: written on the network certifier's loop), and writes
  `warden.json` (each connection's outcome, the tick lateness and each tick's (`ticks_ms`), the refusals and their
  windows, each bucket's pacer report, `paced`, and how far the host's wall time parted from the network
  certifier's clock between the origin's setting and the last tick, `clocks`). Each window's records go to
  `requests.records`, `responses.records` and `fails.records` (each failed connection's reason, witness), and each
  is certified
  (`certificate.Issuer`, with the network certifier's parameters digest) into `certified.jsonl`, except a window in
  which the network certifier refused a connection, whose records and rows it certifies and signs none of (`network_certifier`'s
  "Admission"). Each window of the wire's two records is also committed as rows, a row a bucket, with its wiring
  (`network_certifier_rows`), its head signed: `cells.jsonl` holds the registration and the cost (rows and row bytes),
  `cells.salts` the rows' salts and the wiring's (witness, the prover's, who holds the records the rows and the
  wiring are read off), `cells-timing.jsonl` the commit's and signature's milliseconds (witness). A line of
  `cells.jsonl` is the run's private side whole: the auditor's gate publishes it as it is written here, key for key,
  less the registration's wiring and the cost, which are private, and by a list of its own, so a key added to the
  line fails the gate until the list rules it. All of that is done by a spawned process (`PublisherProcess`), which
  alone holds the key and
  gets each tick's rows as the network certifier releases them: a window's commit can take seconds of CPU, and in the
  network certifier's own process, even on another thread, it holds the ticker's GIL and puts ticks past their slack.
  Before the origin `serve` runs on the C allocator (it re-executes itself with `PYTHONMALLOC=malloc`) and makes its
  process ready for the run's growth (`prepare_process`: its descriptor table, its heap, its pages locked where
  permitted), and from it to the end a process of its own (`stall_probe`, unless `--no-stall-probe`) samples every
  thread of both processes each millisecond into `stalls.json`, every episode in uninterruptible sleep with the kernel
  function it slept in; `warden.json` keeps what was prepared (`process`) and each thread's episodes and page faults
  (`stalls`).
- `client` sends each request of the traffic at its time after the origin on a connection of its own, reads the cells
  to the network certifier's close, and writes `client.json`: per connection its ordinal, outcome, every cell's kind and
  arrival, the SHA-256 of what it sent and of the bytes it got, and the HTTP response read from them. Its descriptor
  table holds every connection before it sends (`reserve_fds`): its process has numpy's threads too, and a table grown
  mid-run put cells that left the network certifier on time a bucket late (168 ms, every connection open as the client
  passed 256).
- `check` (`check`): the session's start is certified under the deployment's key and session for its schedule
  (`check_start`), every window of the three records has a certificate that verifies and whose head is the
  record's, every window of the wire's two has a registration that verifies whose root is its record's rows under
  their salts (`check_cells`), the audit (`network_certifier.audit`) accepts the records, and every connection's client got
  exactly its egress record's cells, each in its bucket, its request is the ingress record's and its response the
  egress record's; `check.json`. With `--key-sha512 HEX --session HEX`, which the auditor hands it, the SHA-512 of
  the network certifier's public key and its session, a deployment record that names another key or session is a
  problem, the first of the problems `check.json` lists. With `--registrations FILE` besides, a check that found no
  problem writes there what it was checked under and then each registration of connection rows it held, one a
  line (`connection_rows_run`'s "The registrations' file"); `--registrations` without the two is refused.
- `rehearse` makes a key and its device state in a scratch directory, writes the traffic, starts the fake endpoint
  when `--fake`, and runs `serve` and `client` as processes, then `check`.

**The load** (`--load-store`, the gate's `load-relay`): the serving session's weights' load, relayed and committed by
this device and signed by its one signer. `rehearse` first relays the store to the H200's sink
(`load_relay.relay_load`: `--load-sinks`, one stream an address, each value committed under the verifier's domain for
`--load-program`, its salts sent on to the sink's witness and kept nowhere here), writing `load-heads.json` and
`load-relay.json`; a refused load serves nothing. It then waits up to `--load-ready-s` for the endpoint, which the
prover starts on the weights its sink wrote, and runs `serve --load-heads`: the publisher process signs every head in
the run's session before its first window (`sign_load`), so one LMS key and one session carry the load and the windows,
and writes `load.json`, the client's kept registrations of the load. `check` reads it under the deployment's key and
session (`check_load`) at the relay's roots.

**A session** (`session`, the gate's `relay-session`): the relay kept up for clients that are not its own. It is
`rehearse` without its traffic and its client, and runs through the same code: a key in a scratch directory, `serve`
as a process at `--listen` (the gate's: a port of the loopback its ssh forwards the peer's client port to) for exactly
`--windows` execution phases, then `check`. With `--listen-fd FD` the relay serves on the listening socket it inherits
as that descriptor, which its caller bound at `--listen` and holds, and binds none: the gate's unit holds it from before
its ssh starts until the ssh has exited, so its forward never ends at a port another process could take. A connection
made before the relay serves waits at that socket, and the relay takes it as it starts; one made after the relay stops
waits there, unread, until the caller closes the socket. It makes no request, so it asks the endpoint nothing before a
client does, and writes no `traffic.json`, `client.log` or `client.json`.
- *Where a client connects, and what tells it the relay is ready.* `serve` writes `endpoints.json` (the listener) and
  then `deployment.json` once it listens: the schedule, the session, the key, the run id, `windows` and `origin_unix`,
  bucket 0's start, `--start-in` seconds later. A client opens one connection a request to the listener (through the gate, to the
  peer's client port, which the gate's `status` names with the deployment; the listener itself the gate gives no one)
  and speaks cells (`network_certifier.cells`, `network_certifier.Shim`). One that connects before the origin waits in the queue.
- *The end.* The relay ticks windows · T buckets from the origin and stops; nothing ends it earlier. A connection
  still open then is cut (the relay's `run`): one in the queue gets no cell and is in no record; one admitted at
  bucket s and cut before s + D has its ingress cells in the record and no egress cell but padding, which the audit
  takes; one cut from s + D on, with no `END` or `FAIL`, is refused by the audit in the execution phase of its
  admission. A connection admitted by bucket windows · T − `NetworkCertifierParams.span` − 1 ends inside the run.
- *A full queue.* A connection that arrives while the queue holds the schedule's `queue` gets one `REFUSED` cell, and
  the relay certifies no record of the execution phase it arrived in, so `check` does not accept the run: a client
  has no more connections open at once than the schedule's lanes and queue hold.
- *What the check holds.* With no `client.json`, `check` holds what it holds of the records alone: every
  execution phase's certificates and registrations and the audit. That each cell reached its client in its bucket, and
  that a client's bytes are the records', is the client's to check, from its own log against the run: the summary's
  `requests`, `ended` and `cells_in_their_bucket` count no connection.
- *Where a refused run says why.* `check.json`, with the run's witness: `problems` names a record the relay didn't
  certify or register by record and window, and `audit_windows` each execution phase the audit refused with its
  problems (a connection failed closed, by ordinal and reason; one cut with no `END`).
- *What it establishes.* The records cover the connections that crossed the relay. A server reached by another
  route is in none of them ("What it can't enforce here").

**The connection rows** (`connection_rows_run`). `serve` and `rehearse` take `--lengths 1|2 [--content P,O[,ROWS] ...]`,
the schedule's lengths rows and its content rows' classes (`P,O,ROWS` under `--lengths 1`, `P,O` under `--lengths 2`,
where a class's tables hold a row a release filed in it), and with `--lengths 2`, `--content` and `--max-tokens-cap`
`--request-words`, a fourth table a class of what the network certifier reads of each released request (its
`max_tokens` and count of prompt ids), which `rehearse` passes to its `serve`; without them a
run is what it was. With them the network certifier hands each execution phase's releases once the next one's records
are published, and the publisher process lays that execution phase's tables, commits them and signs their one head:
`rows.jsonl` (the registration and the cost, the run's private side as `cells.jsonl` is) and `rows.salts` (witness),
and with `--content` `filed.jsonl` (witness: the releases filed in each class, for the prover's serve). `check` holds
every execution phase that has connection rows to a registration that verifies and whose head is the tables laid from
the records. Under `--lengths 2` every execution phase of the run ends, its last at the run's last close: in its
head, or in the network certifier's signed word that it has none, a line of `rows.jsonl` too, and `check` holds each
to the one that is due. The network certifier then admits nothing in the run's last `NetworkCertifierParams.span` buckets, so
`rehearse` runs long enough for its last request to be admitted before them (`closing_windows`). An arrival in them
is turned away (`network_certifier`'s "The lengths rows"): no refusal, so no window's records are withheld for it. Its client
reads the one `REFUSED` cell a refused one does, `warden.json` counts it on its own (`turned_away`, `counts`), and
`check`'s summary has that count beside the clients' refused outcomes, which hold it. With `--head-offset N
--head-wait W` beside `--lengths` (`network_certifier`'s "The head's tick") the publisher process writes no line: it hands each
one back, and the network certifier's tick appends it to `rows.jsonl` at the tick the schedule gives it, N buckets
after the close that handed its execution phase or up to W ticks later. A line not back by then is never written, and
the tick writes its note that the line missed. The run ticks on past its last window until its last line has left or
missed. A line holds no tick: the tick it left at is the network certifier's own note, in `warden.json` alone
(`heads`). No check and no verifier reads it. Nothing holds the head's time yet: that is the auditor's side, which is
not built. `check` counts the execution phases whose line missed, which have no head.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
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
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path


from verity.protocols.resource_accounting.network.verifier import blob_files as bf
from verity.protocols.resource_accounting.network.attester import certifier_key as ck
from verity.protocols.resource_accounting.network.verifier import record_files as RF
from verity.protocols.resource_accounting.network.attester import runner_parts as rp
from verity.protocols.resource_accounting.network.attester import load_relay as lr
from verity.protocols.resource_accounting.network.verifier import load_check as LK
from verity.protocols.resource_accounting.network.attester import runner_cli as cli
from verity.protocols.resource_accounting.network.attester import stall_probe
from verity.protocols.resource_accounting.network.attester.runner_parts import _quiet_runtime, _usage
from verity.protocols.resource_accounting.network.verifier.record_files import pack_record
from verity.protocols.resource_accounting.network.attester.runner_parts import served_model
from verity.protocols.resource_accounting.network.verifier.record_files import certified_json
from verity.commitments.crypto import lms
from verity.protocols.resource_accounting.network.verifier import connection_rows_check as CRC, io_tie as IT, network_certifier_rows as WR
from verity.protocols.resource_accounting.network.verifier.id_form import id_request
from verity.protocols.resource_accounting.network.verifier.certificate import (CertifiedStart, DeviceState, Direction, Issuer, LmsKey,
    SignerRefused, StartRefused)
from verity.protocols.resource_accounting.network.verifier.grid import GridParams, BlockGrid, Status, row_count
from verity.protocols.resource_accounting.network.verifier.pacer import ClockRefused, raw_clock
from verity.protocols.resource_accounting.network.verifier.release import WATERMARK_PATH
from verity.protocols.resource_accounting.network.verifier.network_certifier import (EGRESS_FORMS, INTEGER_FIELDS, Shim, NetworkCertifier,
    NetworkCertifierParams, cells)
from verity.protocols.resource_accounting.network.verifier.relay_check import (CELLS_TIMING, CERTIFIED, EGRESS, ENDPOINTS,
    FAILS, INGRESS, KIND_CHARS, LINK, _quantiles, check)

HERE = Path(__file__).resolve().parent


#: in a key directory, beside the key and its leaf state: the key's device state (`certificate.DeviceState`)
DEVICE = "device.json"
#: the heap `prepare_process` grows and touches before the origin, a lane's and at least: heavy's opening burst through
#: the network certifier (1000 connections at 300 a second, 1 KiB cells) grew its heap by about 35 KB a connection
RESERVE_PER_LANE, RESERVE_MIN = 256 << 10, 64 << 20
#: the descriptors the network certifier holds beside its connections' (its listener, loop, pipes and the probe's)
FD_MARGIN = 256
#: in the environment of a `serve` that re-executed itself on the C allocator, so that it does so once
C_ALLOCATOR = "WARDEN_SERVE_C_ALLOCATOR"
#: what a finite RLIMIT_MEMLOCK must leave beyond the process's mapping for `_memlock` to lock it: the network
#: certifier's heap grows tens of MB in a run
LOCK_HEADROOM = 4 << 30
M_TRIM_THRESHOLD, M_MMAP_THRESHOLD, MCL_CURRENT, MCL_FUTURE, CAP_IPC_LOCK = -1, -3, 1, 2, 14
#: the schedule's defaults here, where they differ from `NetworkCertifierParams`'s: r·P = 494 bytes a bucket each way stays
#: under what vLLM streams a request even with its tokens merged into fewer events, and D = 3 s from admission covers
#: the request's ingress and a queued prefill
SCHEDULE = asdict(NetworkCertifierParams()) | {"slots": 2, "delay": 30, "lanes": 8, "ingress_cells": 96, "ingress_buckets": 100}
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
#: the public file of a `--window-per-group` rehearsal: each group's requests and execution phases (`groups`)
GROUPS = "groups.json"


# Traffic and the bytes on it.

def varied_traffic(n: int, *, model: str, rate: float = 0.5, seed: int = 0, prompt_tokens: tuple[int, int] = (16, 1024),
                   max_tokens: tuple[int, int] = (16, 256), start_s: float = 1.0, temperature: float | None = None,
                   top_p: float | None = None) -> list[dict]:
    """n requests, each {"at": seconds after the origin, "body": a streamed `/v1/completions` body}. `temperature` and
    `top_p`, when given, are every request's (else a temperature sampled from `TEMPERATURES` and no top_p); the arrivals,
    prompts, lengths and seeds are the seed's either way."""
    rng = random.Random(seed)
    at, out = start_s, []
    for _ in range(n):
        question = rng.choice(QUESTIONS)
        words = round(math.exp(rng.uniform(*map(math.log, prompt_tokens))) * 0.75) - len(question.split())
        filler = " ".join(FILLER[j % len(FILLER)] for j in range(max(0, words)))
        most = round(math.exp(rng.uniform(*map(math.log, max_tokens))))
        sampled = rng.choice(TEMPERATURES)
        body = {"model": model, "prompt": f"{filler}\n\n{question}" if filler else question,
                "max_tokens": most, "stream": True, "temperature": sampled if temperature is None else temperature,
                "stream_options": {"include_usage": True}} | ({} if top_p is None else {"top_p": top_p})
        if rng.random() < 0.5:
            body["seed"] = rng.randrange(1 << 31)
        out.append({"at": round(at, 3), "body": body})
        at += rng.expovariate(rate)
    return out


def id_traffic(n: int, *, model: str, rate: float = 0.5, seed: int = 0, prompt_tokens: tuple[int, int] = (16, 1024),
               max_tokens: tuple[int, int] = (1, 64), start_s: float = 0.0, temperature: float | None = None,
               top_p: float | None = None, users: str = "user") -> list[dict]:
    """n network-certifier-form id requests (`id_form.id_request`'s fields), each {"at": seconds from the origin, "ids":
    its fields}: Poisson arrivals at `rate`, prompts of log-uniform length of ids below 2^17, `max_tokens` log-uniform,
    and the sampling parameters picked as `varied_traffic` picks them (top_p 1), or `temperature` and `top_p`, when
    given, every request's; the arrivals, prompts, lengths and seeds are the seed's either way. Request i's `user` is
    ``<users>-<i>``: a server's record names a request by it, so two runs against one server whose records are read
    together take two `users`. The scan (`ingress_scan`) reads exactly these."""
    rng = random.Random(seed)
    at, out = start_s, []
    for i in range(n):
        length = round(math.exp(rng.uniform(*map(math.log, prompt_tokens))))
        prompt = [rng.randrange(128_256) for _ in range(length)]
        most = round(math.exp(rng.uniform(*map(math.log, max_tokens))))
        sampled = rng.choice(TEMPERATURES)
        fields = {"model": model, "user": f"{users}-{i}", "prompt": prompt, "max_tokens": most,
                  "temperature": sampled if temperature is None else temperature,
                  "top_p": 1.0 if top_p is None else top_p, "seed": rng.randrange(1 << 31)}
        out.append({"at": round(at, 3), "ids": fields})
        at += rng.expovariate(rate)
    return out


def group_windows(params: NetworkCertifierParams, tokens: int) -> int:
    """The execution phases a group of at most `lanes` id requests takes when it arrives half a bucket into a window's
    first bucket (`prompt_traffic`'s ``window_per_group``), each answered by at most `tokens` id events and the trailer:
    admitted by the bucket after its arrival's (and one more for the client's lag), its last egress cell in bucket
    s + D + ⌊tokens/r⌋ (the network certifier's schedule), so a group's cells are all in its own windows."""
    return -(-(params.delay + tokens // params.slots + 3) // params.window)


def closing_windows(params: NetworkCertifierParams, at: float) -> int:
    """The fewest execution phases of a run under the lengths rows' form 2 in which a request sent `at` seconds after
    the origin is admitted when a lane is free: by the bucket after its arrival's (and one more for the client's
    lag), which is `NetworkCertifierParams.span` buckets before the run's end at the latest (`network_certifier`'s "The lengths rows")."""
    return -(-(math.floor(at * 1000 / params.bucket_ms) + params.span + 3) // params.window)


def read_prompts(data: bytes, params: NetworkCertifierParams) -> list[list[int]]:
    """``--prompts``: a JSON array of prompts, each a non-empty array of token ids below 2^`PROMPT_ID_BITS` (the
    scan's IDB), each prompt's id request (`id_form.id_request`, offset 0) within the schedule's ingress bound."""
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


def prompt_traffic(prompts: list[list[int]], n: int, *, model: str, params: NetworkCertifierParams, max_tokens: int,
                   rate: float = 0.5, seed: int = 0, start_s: float = 0.0, temperature: float | None = None,
                   top_p: float | None = None, window_per_group: bool = False, users: str = "user") -> list[dict]:
    """n network-certifier-form id requests (`id_traffic`'s items) of the given prompts, request j's
    ``prompts[j mod len]`` with exactly `max_tokens`: at `id_traffic`'s Poisson arrivals, or under `window_per_group` in
    groups of ``len(prompts)`` (at most `lanes`), group g's requests together half a bucket into window g·G (G =
    `group_windows`), so group g's cells are exactly its windows' (``groups``). The seeds are the seed's either way."""
    if window_per_group and len(prompts) > params.lanes:
        raise SystemExit(f"--window-per-group: a group of {len(prompts)} prompts, past the schedule's "
                         f"{params.lanes} lanes")
    rng = random.Random(seed)
    at, out = start_s, []
    G, window_s = group_windows(params, max_tokens), params.window * params.bucket_ms / 1000
    for j in range(n):
        sampled = rng.choice(TEMPERATURES)
        fields = {"model": model, "user": f"{users}-{j}", "prompt": list(prompts[j % len(prompts)]),
                  "max_tokens": max_tokens, "temperature": sampled if temperature is None else temperature,
                  "top_p": 1.0 if top_p is None else top_p, "seed": rng.randrange(1 << 31)}
        when = (j // len(prompts)) * G * window_s + params.bucket_ms / 2000 if window_per_group else at
        out.append({"at": round(when, 3), "ids": fields})
        at += rng.expovariate(rate)
    return out


def groups(n: int, per: int, params: NetworkCertifierParams, max_tokens: int) -> list[dict]:
    """`prompt_traffic`'s groups under ``window_per_group``: each group's requests (indices) and its execution
    phases."""
    G = group_windows(params, max_tokens)
    return [{"group": g, "requests": list(range(g * per, min(n, (g + 1) * per))), "windows": [g * G, (g + 1) * G]}
            for g in range(-(-n // per))]


def fake_ids(body: dict) -> list[int]:
    """The fake endpoint's output ids for an id request: `fake_tokens` of them, each below 2^17, fixed by the request."""
    h = int.from_bytes(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()[:4], "big")
    return [(h + 7919 * k) % 128_256 for k in range(fake_tokens(body))]


def id_response(ids: list[int], payload: int) -> bytes:
    """An id-form response as the scan's egress reads it: one `io_tie.id_event` of P = `payload` bytes an id, each one
    `DATA` cell, then `io_tie.TRAILER`, which the network certifier's `END` holds."""
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
    (`id_response`), whole, and no HTTP head: the bytes the scan's egress reads. The watermark stream
    (`--watermark`) it reads to its end and answers nothing, since the network certifier reads nothing back."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        if head.startswith(b"POST " + WATERMARK_PATH + b" "):
            while await reader.read(1 << 16):
                pass
            return
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
    """Each request of `traffic` straight to the server, `in_flight` at a time, no network certifier: per request the
    seconds to its first byte and to the server's close, its bytes, and the bytes a second between them (what r·P a
    bucket and D must stay under and cover)."""
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


# The network certifier's run.

class Publisher:
    """The network certifier's `publish`, inline: each record packed to its file (`requests.records`,
    `responses.records`, `fails.records`), and certified by `issuer` under its link name and direction (`CERTIFIED`)
    with the schedule's parameters digest, one line of `certified.jsonl` each, or the execution phase under `refused`;
    the wire's two also committed as rows by `commit` (`attest`). `serve` runs it in `PublisherProcess`: on the ticker's
    thread, or on any thread of its process, a window's commit holds the ticker's GIL for seconds. `close` gives the
    issuer's refusals."""

    def __init__(self, out: Path, issuer: Issuer, params: NetworkCertifierParams, commit=WR.Cells.of) -> None:
        self.out, self.issuer, self.params, self.commit = out, issuer, params, commit

    def __call__(self, record: BlockGrid, grid: GridParams) -> None:
        self.keep(record, grid)

    def withhold(self, record: BlockGrid, grid: GridParams) -> None:
        """The network certifier's `withhold`: a record of a window in which it refused a connection, kept and committed as
        `keep` does but neither certified nor signed, its execution phase under `refused` with ``"withheld": true``."""
        self.keep(record, grid, withheld=True)

    def keep(self, record: BlockGrid, grid: GridParams, withheld: bool = False) -> None:
        bf.write_blobs(self.out / f"{record.link}.records", pack_record(record))
        link, direction = CERTIFIED[record.link]
        certified = None if withheld else \
            self.issuer.certify(replace(record, link=link), direction, grid, self.params.digest)
        line = (_refused(link, direction, record.window, withheld) if certified is None
                else certified_json(certified))
        with open(self.out / "certified.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        if record.link != FAILS:
            self.attest(record, link, direction, withheld)

    def attest(self, record: BlockGrid, link: str, direction: Direction, withheld: bool = False) -> None:
        """The execution phase's cells as rows and its wiring (`network_certifier_rows`), their head signed (never `withheld`'s): one
        line of `cells.jsonl` (the registration, or the execution phase under `refused`, and the cost: its rows and a
        row's bytes), the salts to `cells.salts` and the commit's and signature's milliseconds to `CELLS_TIMING`
        (witness both)."""
        t0 = time.perf_counter()
        cells_ = self.commit(link, direction, record, self.params, self.issuer.session)
        head = cells_.head
        t1 = time.perf_counter()
        cert = None if withheld else self.issuer.sign(link, direction, record.window, head.digest())
        t2 = time.perf_counter()
        cost = {"rows": len(cells_.rows), "row_bytes": len(cells_.rows[0])}
        line = (_refused(link, direction, record.window, withheld) if cert is None
                else {"registration": WR.registration(head, cert, self.params)})
        bf.write_blobs(self.out / "cells.salts", cells_.witness)
        with open(self.out / "cells.jsonl", "a") as fh:
            fh.write(json.dumps(line | {"cost": cost}) + "\n")
        with open(self.out / CELLS_TIMING, "a") as fh:
            fh.write(json.dumps({"link": link, "direction": direction.value, "window": record.window,
                                 "commit_ms": round((t1 - t0) * 1000, 3), "sign_ms": round((t2 - t1) * 1000, 3)}) + "\n")

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
    """The load's heads (`heads`, the relay's ``load-heads.json``) signed by the network certifier's `issuer` in its
    session, before it serves: ``load.json`` in `out` (`load_relay.load_doc`), the load's registrations the block record
    pins, under the key and session that then certify the run's windows. Raises `load_relay.Refused` when the signer
    refuses."""
    shape, heads_ = lr.heads_of(json.loads(heads.read_text()))
    doc = lr.load_doc(shape, heads_, lr.sign_heads(issuer, heads_), RF.public_json(key), issuer.session)
    (out / "load.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def make_key(directory: Path, height: int = ck.HEIGHT) -> lms.PrivateKey:
    """The network certifier's key directory, made or loaded (`certifier_key.load_key`), and with a key made here
    its device state (`DEVICE`, `certificate.DeviceState`): created once, with the key, at none of its life used. A
    key that was already there keeps its device state as it is, and one without it is given none: nothing says how
    much of its life it has used, so `serve` refuses to start under it."""
    made = not (directory / "seed").exists()
    key = ck.load_key(directory, height)
    if made:
        DeviceState.create(directory / DEVICE, lms.public_key(key))
    return key


def registered_key(key: Path) -> LmsKey:
    """The registered key of the key directory `key`, read from its ``public.json``. A directory without one holds
    no key the network certifier starts under (`StartRefused`)."""
    try:
        return RF.read_public(json.loads((key / "public.json").read_text()))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise StartRefused(f"{key}: no registered key ({type(e).__name__}); `keys --dir` makes one") from None


def certify_start(issuer: Issuer, base_ms: int, ordinal: int, params: NetworkCertifierParams) -> CertifiedStart:
    """The start's certificate of `issuer`'s session (`Issuer.certify_start`, under `LINK`): the session starts at
    `base_ms` of its key's life, is the key's session number `ordinal`, and runs under `params`'s schedule. A signer
    that refuses raises `SignerRefused`, and the network certifier does not start."""
    start = issuer.certify_start(LINK, base_ms, ordinal, params.digest)
    if start is None:
        raise SignerRefused(f"the signer refused the start of session {issuer.session.hex()}")
    return start


def start_json(start: CertifiedStart) -> dict:
    """A start's certificate as `deployment.json` carries it: all of it is public."""
    return {"base_ms": start.base_ms, "ordinal": start.ordinal, "schedule": start.schedule.hex(),
            "certificate": RF.certificate_json(start.certificate)}


def _publisher_main(rows, results, out: Path, key: Path, session: bytes, params: NetworkCertifierParams, commit,
                    load_heads: Path | None, base_ms: int, ordinal: int, lines=None) -> None:
    """`PublisherProcess`'s process: the network certifier's key and its leaf state are this process's alone. It signs
    the load's heads first when there are any (`sign_load`), then the start's certificate of its session at `base_ms`
    and `ordinal` (`certify_start`), sends that and the public key, rebuilds each record from its released rows, keeps
    it as `Publisher` does at its close once the rebuilt grid's digest is the close marker's, and at the end sends the
    issuer's refusals and its own usage. In a schedule with a head offset it sends each execution phase's line of
    connection rows on `lines` and writes none (`runner_parts.Keeper`), and None once it sends no more."""
    try:
        own = ck.signer(key)
        issuer = Issuer(own, session)
        if load_heads is not None:
            sign_load(issuer, own.public_key(), load_heads, out)
        results.send(("start", start_json(certify_start(issuer, base_ms, ordinal, params))))
        results.send(("key", own.public_key()))
        publisher = Publisher(out, issuer, params, commit or WR.Cells.of)
        keeper = rp.Keeper(out, issuer, params, LINK, back=lines and (lambda window, line: lines.send((window, line))))
        grids = {INGRESS: params.grid, EGRESS: params.grid, FAILS: params.fails_grid}
        filling: dict[str, list] = {link: [] for link in grids}
        B = params.cell_bytes
        while (item := rows.get()) is not None:
            link, n, data = item
            if link is None:
                continue
            if link == rp.HAND:
                keeper(n, data)
                continue
            grid = grids[link]
            if n is not None:
                filling[link].append(tuple(data[i * B:(i + 1) * B] for i in range(n)) + (None,) * (grid.allocation - n))
                continue
            (window, reason, digest, withheld), got = data, tuple(filling[link])
            if len(got) != grid.buckets:
                raise ValueError(f"{link} window {window} closed after {len(got)} rows, not {grid.buckets}")
            if grid_digest(got) != digest:
                raise ValueError(f"{link} window {window}: the rows rebuilt aren't the grid the network certifier closed")
            publisher.keep(BlockGrid(link, window, got, Status(reason)), grid, withheld=withheld)
            filling[link] = []
        results.send(("done", {"refused": publisher.close(), "usage": _usage()}))
        own.close()
    except BaseException:
        results.send(("error", traceback.format_exc()))
        raise
    finally:
        if lines is not None:
            lines.send(None)


class PublisherProcess:
    """`serve`'s `publish` and `release`: `Publisher` in one spawned process, which alone holds the network certifier's
    key. `release` hands it each grid's row as the network certifier ticks (its frames as one bytes object), and
    `__call__` the record's close with its grid's SHA-512 (`grid_digest`), hashed a row at a time as each is released
    and held to the record's grid at its close: the ticker only queues them, and the queue's own thread pickles each
    tick's few kilobytes, never a whole window. The process rebuilds each record from its rows, refuses one whose grid
    isn't the marker's, and keeps, certifies and commits it, so neither a commit nor a signature takes the ticker's GIL.
    `key` is the network certifier's public key; `close` waits for every record and gives the issuer's refusals; `usage`
    is then the process's own. A process that exits early, killed or failed, raises at the next close or at `close`.
    `commit` stands in for `Cells.of` (a test's). With `load_heads` the process signs the load's heads before anything
    else (`sign_load`), so construction returns once ``load.json`` is written. It then signs the start's certificate
    of its session at `base_ms` and `ordinal`, which the key's device state gave the session (`certify_start`):
    `start` is that certificate as `deployment.json` carries it, there before construction returns, so before any
    origin is set, and a signer that refuses it fails the construction. The queue's thread is started at
    construction, with a row of the grid's size that the process drops, so that the first tick starts no thread and
    the thread's buffers have grown to a row's; `feeder` is its thread id. In a schedule with a head offset the
    process writes no line of connection rows: it sends each one back, and `heads` names who takes them."""

    def __init__(self, out: Path, key: Path, session: bytes, params: NetworkCertifierParams, commit=None,
                 load_heads: Path | None = None, *, base_ms: int, ordinal: int) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.rows = ctx.Queue()
        self.results, results = ctx.Pipe(duplex=False)
        self.lines, lines = ctx.Pipe(duplex=False) if params.head_offset is not None else (None, None)
        self.process = ctx.Process(target=_publisher_main, name="network-certifier-publisher", daemon=True,
                                   args=(self.rows, results, out, key, session, params, commit, load_heads, base_ms,
                                         ordinal, lines))
        self.process.start()
        results.close()
        if lines is not None:
            lines.close()
        self._reader: threading.Thread | None = None
        self.usage: dict | None = None
        self._released: dict[str, list] = {}
        self._digests: dict[str, object] = {}
        self.rows.put((None, None, bytes(params.grid.allocation * params.cell_bytes)))
        self.feeder: int | None = self.rows._thread.native_id
        self.start: dict = self._receive("start")
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

    def __call__(self, record: BlockGrid, grid: GridParams, withheld: bool = False) -> None:
        if not self.process.is_alive():
            self._end()
        released, digest = self._released.pop(record.link, []), self._digests.pop(record.link, hashlib.sha512())
        if record.grid != tuple(released):
            raise RuntimeError(f"{record.link} window {record.window}: the record's grid isn't the rows released")
        self.rows.put((record.link, None, (record.window, record.status.reason, digest.digest(), withheld)))

    def withhold(self, record: BlockGrid, grid: GridParams) -> None:
        """The network certifier's `withhold`: the record's close, kept as `Publisher.withhold` keeps it."""
        self(record, grid, withheld=True)

    def close(self) -> list:
        self.rows.put(None)
        done = self._end()
        self.rows.close()
        self.rows.join_thread()
        self.process.join()
        if self._reader is not None:
            self._reader.join()
        self.usage = done["usage"]
        return done["refused"]

    def heads(self, back, ended=None) -> None:
        """In a schedule with a head offset: from now `back(execution phase, line)` takes each line of connection
        rows the process sends back, on a thread of this object's that only waits on them, until the process says
        it sends no more or is gone, when the thread calls `ended()` if given. `serve` has the network certifier's
        loop hand each line to `NetworkCertifier.head_back`. A process that is gone without that word raises on the thread,
        as it does at `close`."""
        if self.lines is None or self._reader is not None:
            raise ValueError("the publisher process sends lines back in a schedule with a head offset, to one taker")

        def read() -> None:
            try:
                while (item := self.lines.recv()) is not None:
                    back(*item)
            finally:
                if ended is not None:
                    ended()

        self._reader = threading.Thread(target=read, name="lines-back", daemon=True)
        self._reader.start()

    def lengths(self, window: int, strides) -> None:
        """The network certifier's `lengths`: an execution phase's hand, queued a slot stride an item
        (`runner_parts.hand`); the process lays its tables, commits them and signs their head
        (`runner_parts.Keeper`), or, for one that ends without releases (`strides` None), signs the word
        that it has no head."""
        for item in rp.hand(window, strides):
            self.rows.put(item)


def deployment_record(params: NetworkCertifierParams, key: LmsKey, session: bytes, *, origin_unix: float, windows: int,
                      slack: float, run_id: bytes | None = None, start: dict | None = None,
                      origin_raw: float | None = None) -> dict:
    """`deployment.json`, the auditor's record of a run: the network certifier's public key and session, the schedule's
    parameters and their digest, the start's certificate (`start_json`: where the session starts on its key's life),
    the origin, and the run id its server's requests carry. Its addresses are `endpoints_record`'s. The origin is
    given twice: `origin_raw` is its reading on the network certifier's clock, which a process on its host reads the
    grid by. `origin_unix` is one reading of the host's wall time, taken when the origin was set: wall time minus it
    is not the network certifier's bucket, since the two clocks part at the rate the host steers its wall time
    (`clocks_parted`)."""
    return {"key": RF.public_json(key), "session": session.hex(), "link": LINK,
            "params": asdict(params), "digest": params.digest.hex(), "slack_ms": slack * 1000, "start": start,
            "origin_unix": origin_unix, "origin_raw": origin_raw, "windows": windows,
            "run_id": None if run_id is None else run_id.hex()}


def endpoints_record(listen: tuple[str, int], upstream: tuple[str, int]) -> dict:
    """`ENDPOINTS`: where the network certifier listened and the endpoint it relays to, which a client reads."""
    return {"listen": list(listen), "upstream": list(upstream)}


def read_endpoints(deployment: Path) -> dict:
    """The run's `ENDPOINTS`, beside its `deployment` file."""
    return json.loads((deployment.parent / ENDPOINTS).read_text())


def clocks_parted(start: tuple[float, float], end: tuple[float, float]) -> dict:
    """How far the host's wall time parted from the network certifier's clock between two readings of both (each a
    pair: the clock's, then the wall's): the seconds each counted, and the wall's gain on the clock in milliseconds
    and in milliseconds an hour of the clock. It is what one run saw, a step of the wall time included, and no bound
    on either clock."""
    clock, wall = end[0] - start[0], end[1] - start[1]
    return {"clock_s": clock, "wall_s": wall, "parted_ms": (wall - clock) * 1000,
            "parted_ms_an_hour": (wall - clock) / clock * 3_600_000 if clock > 0 else None}


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


def prepare_process(params: NetworkCertifierParams) -> dict:
    """The network certifier's process made ready before its origin, so that no tick waits in the kernel (state D) on
    what the run's growth would otherwise ask of it then:
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
            print(f"network-certifier-serve: {key}: {value}", file=sys.stderr, flush=True)
    return report


async def listening(handle, listen: tuple[str, int], sock: socket.socket | None = None) -> asyncio.Server:
    """The relay's server: on the listening socket `sock` when its caller gives one, which the caller keeps open, so
    the port still listens, with nothing accepting, once this server closes; else on a socket it binds at `listen`."""
    if sock is None:
        return await asyncio.start_server(handle, *listen)
    return await asyncio.start_server(handle, sock=sock)


async def serve(params: NetworkCertifierParams, out: Path, *, upstream: tuple[str, int], listen: tuple[str, int], key: Path,
                session: bytes, windows: int, start_in: float = 5.0, slack: float = 0.025,
                announce=cli._announce, probe: bool = False, process: dict | None = None,
                load_heads: Path | None = None, paced_ms: float = cli.PACED_MS, watermark: bool = False,
                clock=raw_clock, sleep=None, wall=time.time, sock: socket.socket | None = None) -> dict:
    """One run of the network certifier on its host's raw clock (the module docstring's `serve`), on the listening
    socket `sock` when given (its caller's, bound at `listen`: "A session") and else one it binds at `listen`, its
    records published by a `PublisherProcess` that holds the key in `key` and first signs the load's heads
    `load_heads` when given, its threads sampled by `stall_probe` when `probe`, each bucket released by its pacer
    `paced_ms` after its boundary (`NetworkCertifier(paced=…)`, which a network certifier on the host's clock must have), and the watermark
    stream sent to `upstream` when `watermark` (`NetworkCertifier(watermark=…)`). Returns what `warden.json` holds, with
    `process`, what `prepare_process` did.
    The loop's errors are kept, not logged: the ticker writes no file, but in a schedule with a head offset, where
    the tick a line of connection rows leaves at appends it to `rows.jsonl` (`runner_parts.Lines`): the
    publisher's process hands each line back, and nothing off the tick publishes one. `warden.json` then has
    `heads`, each execution phase with the tick its line left at, or null for one that was never written. With
    `paced_ms` 0 there is no pacer, which the network certifier refuses on a clock of its host's: only a run on a
    caller's own clock has none.

    It starts in this order, and a step that refuses starts nothing after it. The clock is read: a host without the
    raw clock runs no network certifier (`pacer.ClockRefused`). The key's device state (`DEVICE` in `key`) is opened
    and held until the publisher has closed, so no other run under the key ticks or signs meanwhile, and gives the
    session its base: the key's life, moved on by the run's whole span on disk before the base is used
    (`certificate.StartRefused` for a state that is held, missing or another key's, and for a session already
    started). The publisher signs the start's certificate at that base. Only then is the origin set, a reading of
    the clock, with the host's wall time read beside it; both are read again when the last tick is done, and
    the run's log says how far they parted (`clocks`, `clocks_parted`). `clock`, `sleep` and `wall` are the network
    certifier's clock, its wait and the host's wall time, for a caller that drives a run itself."""
    out.mkdir(parents=True, exist_ok=True)
    clock()
    registered = registered_key(key)
    with DeviceState(key / DEVICE, registered.public) as device:
        base_ms, ordinal = device.start(session, windows * params.window * params.bucket_ms)
        loop = asyncio.get_running_loop()
        errors: list[str] = []
        loop.set_exception_handler(lambda _, context: errors.append(f"{context.get('message')}: "
                                                                   f"{context.get('exception')!r}"))
        publish = PublisherProcess(out, key, session, params, load_heads=load_heads, base_ms=base_ms, ordinal=ordinal)
        if publish.key != registered:
            publish.close()
            raise StartRefused(f"{key}: the signer's key isn't the registered key")
        lines = rp.Lines(out, LINK) if params.head_offset is not None else None
        started = (clock(), wall())
        origin = started[0] + start_in
        network_certifier = NetworkCertifier(params, lambda: asyncio.open_connection(*upstream), origin=origin, windows=windows,
                        clock=clock, sleep=sleep, publish=publish, withhold=publish.withhold, release=publish.release,
                        slack=slack, paced=paced_ms / 1000 if paced_ms else None, leave=lines,
                        watermark=(lambda: asyncio.open_connection(*upstream)) if watermark else None)
        if lines is not None:
            publish.heads(lambda window, line: loop.call_soon_threadsafe(network_certifier.head_back, window, line))
        server = await listening(network_certifier.handle, listen, sock)
        sampler = stall_probe.start(out / "stalls.json", origin,
                                    {"warden": os.getpid(), "publisher": publish.process.pid},
                                    {threading.get_native_id(): "ticker", publish.feeder: "feeder"}) if probe else None
        deployment = deployment_record(params, publish.key, session, origin_unix=started[1] + start_in,
                                       windows=windows, slack=slack, run_id=network_certifier.run_id, start=publish.start,
                                       origin_raw=origin)
        endpoints = endpoints_record((listen[0], server.sockets[0].getsockname()[1]), upstream)
        (out / ENDPOINTS).write_text(json.dumps(endpoints, indent=1))
        (out / "deployment.json").write_text(json.dumps(deployment, indent=1))
        network_certifier.hand_lengths = publish.lengths
        announce({"listening": endpoints["listen"], "deployment": str(out / "deployment.json")})
        try:
            await network_certifier.run()
            ended = (clock(), wall())
        finally:
            server.close()
            refused = await loop.run_in_executor(None, publish.close)
            if lines is not None:
                lines.close()
            stalls = sampler and stall_probe.stop(sampler, out / "stalls.json")
    lateness = [x * 1000 for x in network_certifier.lateness]
    log = {"connections": network_certifier.log, **counts(network_certifier),
           "lateness_ms": _quantiles(lateness) | {"over_slack": sum(x > slack * 1000 for x in lateness)},
           "issuer_refused": [[link, d.value, w] for link, d, w in refused], "usage": _usage(),
           "publisher_usage": publish.usage, "process": process, "stalls": stalls, "loop_errors": errors,
           "clocks": clocks_parted(started, ended), "ticks_ms": [round(x, 3) for x in lateness],
           "paced": {"ms": paced_ms, "columns": ["bucket", "arrived_ms", "start_ms", "end_ms", "written"],
                     "releases": [[r.t, round(r.arrived * 1000, 3), round(r.start * 1000, 3),
                                   round(r.end * 1000, 3), r.written] for r in network_certifier.releases]}}
    if watermark:
        log["watermark"] = {"last": network_certifier.watermarked and list(network_certifier.watermarked), "lost": network_certifier.watermark_lost,
                            "error": network_certifier.watermark_error}
    if lines is not None:
        log["heads"] = [list(h) for h in network_certifier.heads]
    (out / "warden.json").write_text(json.dumps(log, indent=1))
    announce({"served": True, "out": str(out)})
    return log


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def counts(device: NetworkCertifier) -> dict:
    """What `warden.json` counts of a run: the connections that took an ordinal, the refusals and their windows, the
    arrivals turned away, which are neither, those of them whose client reset its link after the cell, and the
    ticks."""
    return {"ordinals": device.ordinals, "refused": device.refused, "refused_windows": sorted(device.refused_windows),
            "turned_away": device.turned_away, "turned_away_reset": device.turned_away_reset, "ticks": device.ticks}


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


async def client(deployment: dict, endpoints: dict, traffic: list[dict], out: Path, clock=raw_clock) -> dict:
    """Each request of `traffic` on a connection of its own at its time after the deployment's origin, through the
    shim, to the listener `endpoints` names; returns what `client.json` holds. An item with ``ids`` (`id_traffic`) is
    sent as its network-certifier-form id request (`id_form.id_request`, offset 0) and its response read as id-form. It
    runs on the network certifier's host and reads that host's raw clock, the one the origin is a reading of
    (`origin_raw`), so that a cell's arrival is laid against the grid on the grid's own clock however long the run."""
    params = NetworkCertifierParams(**deployment["params"])
    origin, listen = deployment["origin_raw"], tuple(endpoints["listen"])
    host = "%s:%d" % tuple(endpoints["upstream"])

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


def table(result: dict) -> str:
    s = result["summary"]
    load = "" if "load" not in s else (f"load {s['load']['registered']}/{s['load']['values']} values registered"
                                       f"{'' if s['load'].get('relayed') else ', not the relay heads'}; ")
    return (f"{'ACCEPTED' if result['accepted'] else 'REFUSED'}: {load}{s['ended']}/{s['requests']} ended, "
            f"{s['refused']} refused or turned away ({s['turned_away']} turned away), failed {s['failed']}; "
            f"{s['certified_windows']}/{len(CERTIFIED) * s['windows']} "
            f"execution phases "
            f"certified; cells in their bucket {s['cells_in_their_bucket'][0]}/{s['cells_in_their_bucket'][1]}; "
            f"cells {s['cells']}; start at {s['start']['base_ms']} ms of the key's life; tick lateness "
            f"{s['tick_lateness_ms']}; wall time against the clock {s.get('clocks') or 'not read'}; ticker D episodes "
            f"{s.get('ticker_stalls') or 'not probed'}; cell rows {s['cell_rows']['registered']}/"
            f"{s['cell_rows']['of']} execution phases registered, {s['cell_rows']['rows']} rows a window")


# The rehearsal.

def traffic_of(args, model: str) -> list[dict]:
    if getattr(args, "prompts", None) is not None:
        params = _params(args)
        return prompt_traffic(read_prompts(args.prompts.read_bytes(), params), args.requests, model=model, params=params,
                              max_tokens=min(args.max_tokens, args.egress_cells - 1), rate=args.rate, seed=args.seed,
                              temperature=args.temperature, top_p=args.top_p, window_per_group=args.window_per_group,
                              users=args.users)
    if getattr(args, "ids", False):
        return id_traffic(args.requests, model=model, rate=args.rate, seed=args.seed, prompt_tokens=(16, args.prompt_max),
                          max_tokens=(1, min(args.max_tokens, args.egress_cells - 1)), temperature=args.temperature,
                          top_p=args.top_p, users=args.users)
    return varied_traffic(args.requests, model=model, rate=args.rate, seed=args.seed,
                          prompt_tokens=(16, args.prompt_max), max_tokens=(16, args.max_tokens),
                          temperature=args.temperature, top_p=args.top_p)


def schedule_flags(q: argparse.ArgumentParser, egress: str | None) -> None:
    """`NetworkCertifierParams` as `q`'s flags: each integer field at `SCHEDULE`'s, and `--egress` at `egress` (None: the
    traffic's, id with `--ids` and else bytes)."""
    for name in INTEGER_FIELDS:
        q.add_argument(f"--{name.replace('_', '-')}", type=int, default=SCHEDULE[name])
    q.add_argument("--egress", choices=EGRESS_FORMS, default=egress,
                   help="the egress form (network certifier's \"The egress form\"): id, the egress record's, which the period "
                        "check takes, or bytes, any server's bytes, a rehearsal's relay of another form (default: "
                        + (egress or "id with --ids, else bytes") + ")")
    q.add_argument("--max-tokens-cap", type=int, default=None,
                   help="the response cap (network certifier's \"The response cap\"), in the id form: a request whose max_tokens "
                        "exceeds it, or doesn't parse, is refused (cap) and every response is held to its request's "
                        "(default: none)")


def _params(args) -> NetworkCertifierParams:
    egress = args.egress or ("id" if getattr(args, "ids", False) else "bytes")
    return NetworkCertifierParams(**{name: getattr(args, name) for name in INTEGER_FIELDS}, egress=egress,
                        max_tokens_cap=getattr(args, "max_tokens_cap", None))


def _check_paced(args) -> None:
    if not args.slack < args.paced_ms / 1000 < args.bucket_ms / 1000:
        raise SystemExit(f"--paced-ms {args.paced_ms:g}: the network certifier always paces, each bucket released past the slack "
                         f"({args.slack * 1000:g} ms) and before the next boundary ({args.bucket_ms} ms)")


def _flags(args) -> list[str]:
    return [x for k, v in asdict(_params(args)).items() if v is not None for x in (f"--{k.replace('_', '-')}", str(v))]


def await_model(url: str, seconds: float, poll: float = 5.0) -> str:
    """The endpoint's one model (`served_model`), asked again each `poll` seconds while it doesn't answer, for at most
    `seconds`: after the load the prover starts its server on the weights its sink wrote."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return served_model(url)
        except (OSError, ValueError, http.client.HTTPException) as e:
            if time.monotonic() >= deadline:
                raise SystemExit(f"{url} didn't answer within {seconds:g} s of the load: {e}") from None
            time.sleep(poll)


def _addresses(text: str) -> list[tuple[str, int]]:
    return [cli._address(a) for a in text.split(",")]


def traffic_windows(args, params: NetworkCertifierParams, model: str, out: Path) -> int:
    """A rehearsal's traffic, written to `out` (`traffic.json`, and a `--window-per-group` run's `groups.json`), and
    the execution phases its run lasts: to its last request's time and `--tail`, its groups' windows, and under
    `--lengths 2` as many as admit its last request (`closing_windows`)."""
    traffic = traffic_of(args, model)
    (out / "traffic.json").write_text(json.dumps(traffic))
    window_s = params.window * params.bucket_ms / 1000
    windows = math.ceil((traffic[-1]["at"] + args.tail) / window_s)
    if args.window_per_group:
        grouped = groups(len(traffic), len(read_prompts(args.prompts.read_bytes(), params)), params,
                         traffic[0]["ids"]["max_tokens"])
        (out / GROUPS).write_text(json.dumps({"groups": grouped}, indent=1))
        windows = grouped[-1]["windows"][1]
    if params.closes:
        windows = max(windows, closing_windows(params, traffic[-1]["at"]))
    return windows


def own_client(me: list[str], out: Path) -> int:
    """A rehearsal's own client, `client` as a process over the run's traffic to its end: its exit."""
    with open(out / "client.log", "w") as clog:
        return subprocess.run(me + ["client", "--deployment", str(out / "deployment.json"),
                                    "--traffic", str(out / "traffic.json"), "--out", str(out)],
                              stdout=clog, stderr=subprocess.STDOUT).returncode


def rehearse(args) -> int:
    """`rehearse`, and `session` (the module docstring's "A session"): the same run with no traffic and no client,
    the relay at `--listen` for `--windows` execution phases."""
    kept = args.command == "session"
    held = args.listen_fd if kept else None
    _check_paced(args)
    if args.prompts is not None and not args.ids:
        raise SystemExit("--prompts: id requests' prompts, with --ids")
    if args.window_per_group and args.prompts is None:
        raise SystemExit("--window-per-group: groups of the --prompts")
    load = args.load_store is not None
    if load and not (args.load_sinks and args.load_program and lr.HEX64.fullmatch(args.load_program)):
        raise SystemExit("--load-store: with --load-sinks and --load-program, the reading program's digest in hex")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = rp.schedule(_params(args), args)
    me = [sys.executable, str(HERE / "network_certifier_serve.py")]
    scratch = Path(tempfile.mkdtemp(prefix="network-certifier-serve-"))
    procs: list[subprocess.Popen] = []
    try:
        make_key(scratch / "network_certifier", args.height)
        if load:
            try:
                lr.relay_load(args.load_store, LK.SHAPES[args.load_shape], args.load_sinks, args.load_program, out,
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
            model = None if kept else \
                args.model or (await_model(url, args.load_ready_s) if load else served_model(url))
        windows = args.windows if kept else traffic_windows(args, params, model, out)
        with open(out / "serve.log", "w") as log:
            serving = subprocess.Popen(me + ["serve", "--upstream", "%s:%d" % upstream, "--listen",
                                             "%s:%d" % args.listen if kept else "127.0.0.1:0",
                                             "--out", str(out), "--key", str(scratch / "network_certifier"), "--session",
                                             secrets.token_hex(32), "--windows", str(windows), "--start-in",
                                             str(args.start_in), "--slack", str(args.slack), "--paced-ms",
                                             str(args.paced_ms), *_flags(args), *rp.flags_of(params),
                                             *(["--watermark"] if args.watermark else []),
                                             *(["--load-heads", str(out / "load-heads.json")] if load else []),
                                             *(["--listen-fd", str(held)] if held is not None else [])],
                                       stdout=subprocess.PIPE, stderr=log, text=True,
                                       pass_fds=() if held is None else (held,))
            procs.append(serving)
            if "listening" not in json.loads(serving.stdout.readline() or "{}"):
                raise SystemExit("serve exited before listening (serve.log)")
            client_exit = None if kept else own_client(me, out)
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


def _sha512(text: str) -> bytes:
    """A SHA-512 from its hex, as a flag's value; ValueError for another length."""
    if len(raw := bytes.fromhex(text)) != 64:
        raise ValueError("a SHA-512 is 64 bytes")
    return raw


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="network_certifier_serve.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    k = sub.add_parser("keys", help="make or load an LMS key, with a key made its device state, and write its public.json")
    t = sub.add_parser("traffic", help="write varied traffic")
    f = sub.add_parser("fake", help="a stand-in for vLLM's streamed /v1/completions")
    s = sub.add_parser("serve", help="the network certifier in front of the server, for one run")
    c = sub.add_parser("client", help="the traffic, through the shim")
    x = sub.add_parser("check", help="the offline check of a run")
    r = sub.add_parser("rehearse", help="traffic, then serve, client and check")
    se = sub.add_parser("session", help="serve at --listen for --windows execution phases with no client, then check")
    pr = sub.add_parser("probe", help="the traffic's first requests straight to the server, no network certifier")
    for q in (k,):
        q.add_argument("--dir", required=True, type=Path)
    for q in (k, r, se):
        q.add_argument("--height", type=int, default=ck.HEIGHT, help="the LMS key's tree height")
    for q in (t, s, c, r, se, pr):
        q.add_argument("--out", required=True, type=Path)
    for q in (t, r, pr):
        q.add_argument("--model", help="the endpoint's model name (default: the one /v1/models lists)")
        q.add_argument("--requests", type=int, default=100)
        q.add_argument("--rate", type=float, default=0.5, help="Poisson arrivals a second")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--max-tokens", type=int, default=256, help="the most max_tokens a request samples")
        q.add_argument("--prompt-max", type=int, default=1024, help="the most prompt tokens a request samples")
        q.add_argument("--temperature", type=float, help="every request's temperature (default: sampled, 0, 0.7 or 1)")
        q.add_argument("--top-p", type=float, help="every request's top_p (default: none sent, the endpoint's own)")
        q.add_argument("--users", default="user", help="the id requests' `user` names: request i's is USERS-i")
    t.set_defaults(model="model")
    for q in (f, s, se):
        q.add_argument("--listen", required=True, type=cli._address, help="HOST:PORT (port 0: any)")
    for q in (s, se):
        q.add_argument("--listen-fd", type=int, metavar="FD",
                       help="serve on the listening socket inherited as descriptor FD, which the caller bound at "
                            "--listen and holds, and bind none")
    for q in (f, r, se):
        q.add_argument("--token-ms", type=float, default=20.0, help="the fake endpoint's milliseconds a token")
        q.add_argument("--first-ms", type=float, default=100.0,
                       help="the fake endpoint's milliseconds to its first token, or to its whole id-form response "
                            "(past the network certifier's delay D, every connection fails late: a void run)")
    for q in (s, r, se):
        q.add_argument("--start-in", type=float, default=5.0, help="the origin, seconds after the network certifier is up")
        q.add_argument("--slack", type=float, default=0.025, help="seconds a tick may run late before it faults")
        q.add_argument("--paced-ms", type=float, default=cli.PACED_MS,
                       help="when the network certifier's pacer releases each bucket, ms after its boundary (past the slack, "
                            "before the next boundary): the network certifier always paces")
        q.add_argument("--watermark", action="store_true",
                       help="also send the server the watermark stream, a chunked POST /verity/watermark on its own "
                            "connection, one line 't n l...' a tick: bucket t has closed, its releases are N < n, and "
                            "each l is one of its releases that the release's server connection lost")
    schedule_flags(s, "id")
    schedule_flags(r, None)
    schedule_flags(se, "id")
    for q in (s, r):
        rp.flags(q)
    s.add_argument("--upstream", required=True, type=cli._address, help="the server's HOST:PORT")
    s.add_argument("--key", required=True, type=Path, help="the network certifier's key directory (keys)")
    s.add_argument("--session", required=True, help="the certifier session, 32 bytes in hex")
    for q in (s, se):
        q.add_argument("--windows", required=True, type=int, help="the run's length, in execution phases")
    se.set_defaults(ids=True, prompts=None, window_per_group=False, load_store=None)
    s.add_argument("--stall-probe", action=argparse.BooleanOptionalAction, default=True,
                   help="sample the network certifier's threads every millisecond for D episodes (stall_probe.py; stalls.json)")
    s.add_argument("--load-heads", type=Path,
                   help="the load's heads (load_relay.py's load-heads.json), signed first in the session: load.json")
    r.add_argument("--load-store", type=Path,
                   help="relay the load first: the store's directory (load_relay.py write), its stream sent to "
                        "--load-sinks and committed as it crosses, its heads signed by the network certifier before it serves")
    r.add_argument("--load-shape", choices=sorted(LK.SHAPES), default="h200")
    r.add_argument("--load-sinks", type=_addresses, help="the H200 sink's HOST:PORT, one a stream, comma-separated")
    r.add_argument("--load-program", help="the reading program's digest, 64 bytes in lowercase hex")
    r.add_argument("--load-workers", type=int, default=os.cpu_count() or 1, help="processes committing the load")
    r.add_argument("--load-ready-s", type=float, default=900.0,
                   help="seconds the endpoint has after the load to answer /v1/models")
    for q in (r, se):
        up = q.add_mutually_exclusive_group(required=True)
        up.add_argument("--upstream", type=cli._address, help="the server's HOST:PORT (an OpenAI endpoint)")
        up.add_argument("--fake", action="store_true", help="start the fake endpoint and use it")
    pr.add_argument("--upstream", required=True, type=cli._address, help="the server's HOST:PORT (an OpenAI endpoint)")
    pr.add_argument("--in-flight", type=int, default=4)
    r.add_argument("--tail", type=float, default=60.0,
                   help="seconds the run lasts past the last request's time; under --lengths 2 the network certifier "
                        "admits nothing in the run's last span buckets, and the run lasts at least long enough to "
                        "admit the last request before them (closing_windows)")
    r.add_argument("--ids", action="store_true",
                   help="network-certifier-form id requests (id_traffic), answered id-form (the fake's --id-payload): "
                        "the scan's traffic (scan_zk.py --warden-run)")
    r.add_argument("--prompts", type=Path,
                   help="with --ids: the requests' prompts, a JSON array of token-id arrays, request j's the (j mod n)-th "
                        "of the n, each with exactly --max-tokens (at most --egress-cells - 1)")
    r.add_argument("--window-per-group", action="store_true",
                   help="with --prompts: the requests in groups of the n prompts (at most --lanes), group g's half a "
                        "bucket into window g·G, G = group_windows, the run exactly its groups' windows (groups.json), "
                        "or under --lengths 2 as many more as admit the last group (closing_windows)")
    f.add_argument("--id-payload", type=int, help="answer every request id-form (id_response) at P bytes an event")
    c.add_argument("--deployment", required=True, type=Path)
    c.add_argument("--traffic", required=True, type=Path)
    x.add_argument("--run", required=True, type=Path)
    x.add_argument("--key-sha512", type=_sha512,
                   help="the SHA-512 of the network certifier's public key, its LMS public key I ‖ T[1], in hex, as "
                        "the auditor holds it: a deployment record that names a key of another hash is a problem")
    x.add_argument("--session", type=bytes.fromhex,
                   help="the network certifier's session, in hex, as the auditor holds it: a deployment record that "
                        "names another is a problem; handed with --key-sha512")
    x.add_argument("--registrations", type=Path,
                   help="the registrations' file (connection_rows_run's \"The registrations' file\"): what the "
                        "check was handed, --key-sha512 and --session, then each registration of connection rows "
                        "it held, one a line, written only where the check found no problem; refused without the "
                        "two; a job names $OUT/registrations.jsonl")
    return ap


def main(argv=None) -> int:
    ap = parser()
    args = ap.parse_args(argv)

    if args.command == "keys":
        make_key(args.dir, args.height)
        print((args.dir / "public.json").read_text())
        return 0
    if args.command == "traffic":
        args.out.write_text(json.dumps(traffic_of(args, args.model)))
        return 0
    if args.command == "probe":
        model = args.model or served_model("http://%s:%d" % args.upstream)
        result = asyncio.run(probe(args.upstream, traffic_of(args, model), args.in_flight))
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("first_s", "bytes_per_s", "statuses")}))
        return 0
    if args.command == "fake":
        async def fake() -> None:
            server = await asyncio.start_server(lambda rd, wr: fake_endpoint(rd, wr, first_s=args.first_ms / 1000,
                                                                             token_s=args.token_ms / 1000,
                                                                             id_payload=args.id_payload), *args.listen)
            cli._announce({"listening": [args.listen[0], server.sockets[0].getsockname()[1]]})
            await server.serve_forever()
        asyncio.run(fake())
        return 0
    if args.command == "serve":
        _check_paced(args)
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
        params = rp.schedule(_params(args), args)
        _quiet_runtime()
        process = prepare_process(params)
        for name in ("PYTHONMALLOC", C_ALLOCATOR):
            os.environ.pop(name, None)
        try:
            asyncio.run(serve(params, args.out, upstream=args.upstream, listen=args.listen, key=args.key,
                              session=session, windows=args.windows, start_in=args.start_in, slack=args.slack,
                              probe=args.stall_probe, process=process, load_heads=args.load_heads,
                              paced_ms=args.paced_ms, watermark=args.watermark,
                              sock=None if args.listen_fd is None else socket.socket(fileno=args.listen_fd)))
        except (ClockRefused, StartRefused) as e:
            raise SystemExit(f"the network certifier did not start: {e}") from None
        return 0
    if args.command == "client":
        _quiet_runtime()
        traffic = json.loads(args.traffic.read_text())
        reserve_fds(len(traffic) + FD_MARGIN)
        asyncio.run(client(json.loads(args.deployment.read_text()), read_endpoints(args.deployment), traffic, args.out))
        return 0
    if args.command == "check":
        under = None if None in (args.key_sha512, args.session) else (args.key_sha512, args.session)
        try:
            # before the refusal of a file that has nothing to be checked under, one already there is removed
            CRC.open_registrations(args.registrations, under)
        except ValueError as e:
            ap.error(f"--registrations needs --key-sha512 and --session: {e}")
        if under is None and (args.key_sha512, args.session) != (None, None):
            ap.error("--key-sha512 and --session are handed together")
        result = check(args.run, args.registrations, under)
        (args.run / "check.json").write_text(json.dumps(result, indent=1))
        print(table(result))
        return 0 if result["accepted"] else 1
    return rehearse(args)


if __name__ == "__main__":
    raise SystemExit(main())
