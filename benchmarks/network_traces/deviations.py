"""deviations: a server that deviates behind the network certifier's relay, six classes, each tried, read back and scored.

A process between the network certifier (`warden_serve`'s relay, its id egress form) and the server's id front plays
the cheating server. It plants a random string and tries one way to send it out, a class a try:

- ``honest``: nothing changes; the control, tried first: when it is refused no deviation is tried.
- ``tokens``: each output id's low bit is the string's next bit, a bit a token. The cells stay id-form cells, which the
  relay releases, so only the I/O tie refuses them: it holds each data cell's id to the Commit's sampled id.
- ``length``: a response loses its last id event when its count's parity isn't the connection's bit, a bit a
  connection. The I/O tie refuses it too: the Commit then holds an output that no cell carries.
- ``fail``: the server withholds a response when the connection's bit is 1, so the relay fails the connection (late)
  at its first egress cell. The relay's audit refuses a run in which a connection failed, and the leak bound charges
  each connection's fail cell.
- ``extra``: after a response's trailer come bytes out of the id form that carry eight of the string's bits. The
  relay fails the connection at that cell (format; under a response cap limit, where the response already holds its
  request's ``max_tokens``), and none of its bytes leaves.
- ``bucket``: a response is held ``--hold-ms`` when the connection's bit is 1. The relay releases egress cell i in a
  bucket that the admission fixes, so a hold short of a cell's deadline doesn't show, and one past it fails the
  connection (late), which the relay's audit refuses. It plants a bit a connection, so its try has
  ``--bucket-requests`` connections.

**The receiver** (`reading`) reads only what the client got: each connection's cells' kinds as `client.json` has
them, and the bytes of its `DATA` and `END` cells in the egress record where their SHA-256 is the one the client
logged. Its reading of a class is fixed before the try: the ids' low bits, the count's parity, the
bucket of the first cell that isn't padding and whether the connection failed, whether it failed, the bytes after the
trailer. `score` sets the reading beside the planted bits: how many it has right, and the mutual information between
the two over the try.

**The checks.** The relay's is `warden_serve.check`, whose refusal's words are its audit's (`warden.audit`). The I/O
tie's is its content part (`io_tie.cells_match`, `IoTieContent`'s reference) on each bucket of the egress record
(`tie`), with two stand-ins for what a try doesn't have: the map is the certified placement, each data cell its
stream's next output, and the Commit is the ids given (``--ids``, the server's record of each request's sampled ids),
or under ``--stand-in`` the stand-in endpoint's own (`warden_serve.fake_ids` of each request). Without either the tie
isn't run, and the row says so. ``--ids`` is JSON lines, a request each with its `output_ids`: under its `user`, or
under its `id` ``verity-<run>-<N>``, the name the certified tick gives a request that brings none (`tick_plugin`), the
certifier's run id and the request's release number, so the server's own record of its requests is read as it is.

**A refusal counts only where it is the deviation's** (`row`). The tie's content refuses exactly the ids the planted
bits changed, or exactly the connections whose last event was dropped, in a try the relay's check accepts. The relay
failed exactly the connections withheld (late), every connection at the cell after its last id (format or limit) of
a response the front sent whole, or held connections alone (late). A refusal of anything else, or a connection that
failed undeviated, is a row that didn't do what its check says: a server that answers out of the id form, or stops,
is never a deviation refused.

**What the bound charges** (`leak_bounds.terms`): a connection's schedule and its fail cell, each ⌈log2(cells + 1)⌉
bits, and the jitter, 0 at a jitter bound of 1; a refused class is charged nothing.

**What a try keeps** (`keep`). ``--out`` gets a file a class, ``CLASS.json``: the schedule, the certifier's run id,
the check's verdict in its words and its counts; of each connection its request's name, its release number, its cells'
kinds, the ids it was sent, whether and why it failed, the bits planted on it and whether the front's response to
it was whole; the egress record's cells, each as
(bucket, slot, kind, stream, id); and the size and SHA-256 of every file of the try's run directory. The table is made
from these files alone. The run directory itself (the relay's records and their salts, the traffic, the logs) is under
``--scratch``, by default a temporary directory removed when the command ends. No key is in either: `rehearse` keeps
its own in a directory it removes.

    python deviations.py try (--stand-in | --front-env NAME) --out DIR [--class all|CLASS] [--ids FILE] [--seed S]
        [--tag T] [--requests 8] [--bucket-requests 64] [--rate 2] [--max-tokens 16] [--tail 8] [--hold-ms 500]
        [--cell-bytes B] [--scratch DIR] [-- REHEARSE-FLAG ...]
    python deviations.py relay --class CLASS --front-env NAME --payload P [--listen 127.0.0.1:0] [--log FILE]
        [--hold-ms 500] [--seed S]
    python deviations.py table --out DIR [--ids FILE] [--stand-in]

- `try` runs each class as one try, with a network certifier process of its own: the deviating process (`relay`) in
  front of the front, then `warden_serve.py rehearse --ids` through it (its key, its traffic, `serve`, `client` and
  `check`; the flags after ``--`` are its own, the schedule's among them), then what it keeps and the row. It prints
  the table, a class a row: what is changed, the check that refuses it, the refusal's own words, the bits planted,
  read right and charged, and the try's seconds end to end; `rows.json` holds the rows and `table.txt` the table. It
  exits 0 when every class did what its row's check says: honest accepted, and each other class refused by that check
  or not shown to the client. It exits 1 when a class didn't, when honest was refused and the rest weren't tried, or
  when a try wrote no check (it has no row, and the classes after it are still tried); and 2 when none of those and a
  row is shown neither way: the tie it aims at wasn't run (no ``--ids`` yet), or its deviation changed no response.
  ``--tail`` covers a connection admitted at the last request's time, the delay and its cells.
  `deviations.sh` is `try` as a job, behind one ssh forward to a front that listens on its own node's loopback.
- A try's request i is named ``<class>-<tag>-<i>`` (its `user`), the tag ``--tag`` or a random one a command: the
  server's record names a request by its `user`, so one record over every try, and over what the server served before
  them, holds each of a try's requests once, and ``--ids`` reads the right row.
- The front's address is the value of the environment variable ``--front-env`` names, ``HOST:PORT``. It has no
  default, only `relay` reads it, and nothing a try writes holds it: the network certifier's upstream is the
  deviating process on the loopback. ``--stand-in`` starts `warden_serve.py fake --id-payload` as the front instead.
- `relay` is the deviating process alone. It relays an id request (``POST /v1/completions``) to the front and deviates
  its response as the class says, as the front's bytes arrive; every other connection it relays byte for byte. Once it
  listens it writes ``{"listening": [HOST, PORT]}`` on stdout. ``--log`` gets one JSON line an id request when its
  connection ends: its release number, the bits it planted, the bytes it read and sent. It holds no id.
- `table` makes the table again from what the tries in a directory kept, with `try`'s exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import leak_bounds as LB  # noqa: E402
import negatives as N  # noqa: E402
import warden_serve as ws  # noqa: E402
from verity.protocols.certifier.verifier import io_tie as IT  # noqa: E402
from verity.protocols.certifier.verifier import release as REL  # noqa: E402
from verity.protocols.certifier.verifier.active import DATA, PAD, id_request  # noqa: E402
from verity.protocols.certifier.verifier.warden import END, FAIL, WardenParams, read_cell  # noqa: E402

#: the classes in the order they are tried: the control first, and the one with the longest try last
CLASSES = ("honest", "tokens", "length", "fail", "extra", "bucket")
#: a class's row: what the deviating process changes, and the check that refuses it
CASES = {
    "honest": ("nothing (the control)", "none: the relay's check accepts, and the I/O tie where it is run"),
    "tokens": ("each output id's low bit is a planted bit",
               "the I/O tie's content: a data cell's id against the Commit's sampled id"),
    "length": ("a response loses its last id event when its count's parity isn't the planted bit",
               "the I/O tie's content: the Commit holds an output no cell carries"),
    "fail": ("a response is withheld when the planted bit is 1",
             "the relay's audit: no connection failed"),
    "extra": ("bytes out of the id form after the trailer carry eight planted bits",
              "the relay's egress form: a cell out of it fails its connection (format, or limit under a response "
              "cap), and the audit refuses the run"),
    "bucket": ("a response is held when the planted bit is 1",
               "the relay's schedule: a cell's bucket is the admission's, so a hold doesn't show, and one past a "
               "cell's deadline fails the connection (late), which the relay's audit refuses"),
}
#: the planted bits an `extra` connection's bytes carry
EXTRA_BITS = 8
COMPLETIONS = b"POST /v1/completions "
BLANK = b"\r\n\r\n"
READ_BYTES, HEAD_BYTES = 1 << 16, 1 << 16
#: the deviating process's log in a try's directory, and the environment variable `try --stand-in` names the stand-in
#: front by, in the deviating process's environment alone
PLANTED, STAND_IN_ENV = "planted.jsonl", "DEVIATIONS_STAND_IN_FRONT"
#: how long the deviating process keeps a connection it withholds, should the network certifier never close it
LINGER_S = 600.0
NOT_RUN = "not-run"
ADDRESS = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b(?::\d+)?")


# --- the deviations ----------------------------------------------------------------------------------------------------

def event_id(event: bytes, payload: int) -> int | None:
    """The id of `event` when it is the canonical id event of `payload` bytes (`io_tie.id_event`), else None."""
    at = len(IT.ID_EVENT_HEAD)
    digits = event[at:at + IT.ID_DIGITS]
    if len(event) != payload or not digits.lstrip(b" ").isdigit():
        return None
    x = int(digits)
    return x if x < 1 << 32 and event == IT.id_event(x, payload) else None


def events_of(data: bytes, payload: int) -> tuple[list[int], bytes]:
    """The ids of the id events `data` starts with, and the rest: a response's trailer and whatever follows it."""
    ids = []
    while (x := event_id(data[:payload], payload)) is not None:
        ids.append(x)
        data = data[payload:]
    return ids, data


class Deviation:
    """One connection's deviation of class `cls` (the module docstring's): `feed` takes the front's bytes as they
    arrive and gives the bytes to send on now, `end` those to send once the front has ended. `delay` is the seconds
    every byte is held, `withhold` whether nothing is sent and the connection is left open, `planted` the bits it
    planted, in the order the receiver reads them, and `whole` whether what the front has sent ends with a response's
    trailer. `bit` gives the planted string's next bit."""

    def __init__(self, cls: str, bit: Callable[[], int], payload: int, hold_s: float = 0.0) -> None:
        if cls not in CLASSES:
            raise ValueError(f"a class is one of {', '.join(CLASSES)}, not {cls!r}")
        self.cls, self.payload, self._bit = cls, payload, bit
        self.planted: list[int] = []
        self.buffer, self.held, self.events, self.tail = bytearray(), b"", 0, b""
        self.delay, self.withhold, self.extra = 0.0, False, b""
        if cls in ("length", "bucket", "fail"):
            self.bit = self._plant()
            self.delay = hold_s if cls == "bucket" and self.bit else 0.0
            self.withhold = cls == "fail" and self.bit == 1
        elif cls == "extra":
            bits = "".join(str(self._plant()) for _ in range(EXTRA_BITS))
            self.extra = bits.encode().ljust(payload, b" ")

    def _plant(self) -> int:
        self.planted.append(self._bit())
        return self.planted[-1]

    @property
    def whole(self) -> bool:
        return self.tail == IT.TRAILER

    def feed(self, data: bytes) -> bytes:
        self.tail = (self.tail + data)[-len(IT.TRAILER):]
        if self.withhold:
            return b""
        if self.cls not in ("tokens", "length"):
            return data
        self.buffer += data
        out, P = bytearray(), self.payload
        while (x := event_id(bytes(self.buffer[:P]), P)) is not None:
            del self.buffer[:P]
            self.events += 1
            if self.cls == "tokens":
                out += IT.id_event(x & ~1 | self._plant(), P)
            else:
                out, self.held = out + self.held, IT.id_event(x, P)
        return bytes(out)

    def end(self) -> bytes:
        if self.withhold:
            return b""
        out = bytearray()
        if self.cls == "length" and self.events % 2 == self.bit:
            out += self.held
        return bytes(out + self.buffer + self.extra)


def bits_of(seed: int | None = None) -> Callable[[], int]:
    """The planted string, a bit a call: `seed`'s (None: drawn from the system's randomness)."""
    rng = random.Random(secrets.randbits(64) if seed is None else seed)
    return lambda: rng.getrandbits(1)


def front_of(name: str, environ: Mapping[str, str] = os.environ) -> tuple[str, int]:
    """The front's address, the value ``HOST:PORT`` of the environment variable `name`. No message holds the value."""
    host, sep, port = environ.get(name, "").rpartition(":")
    if not (sep and host and port.isdigit()):
        raise SystemExit(f"the environment variable {name} isn't set to the front's HOST:PORT")
    return host, int(port)


async def deviate(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, front: tuple[str, int], cls: str,
                  payload: int, bit: Callable[[], int], hold_s: float, log: Callable[[dict], None],
                  linger_s: float = LINGER_S) -> None:
    """One connection of the network certifier's, relayed to `front` (the module docstring's `relay`). What it logs of
    a front it can't reach names the error's kind alone."""
    loop = asyncio.get_running_loop()
    head = b""
    while BLANK not in head and len(head) < HEAD_BYTES and (more := await reader.read(READ_BYTES)):
        head += more
    try:
        up_reader, up_writer = await asyncio.open_connection(*front)
    except OSError as e:
        log({"error": f"the front took no connection ({type(e).__name__}, errno {e.errno})"})
        writer.close()
        return
    dev = Deviation(cls, bit, payload, hold_s) if head.startswith(COMPLETIONS) else None
    release = REL.release_of(head.partition(BLANK)[0] + BLANK) if dev is not None else None
    delay = dev.delay if dev is not None else 0.0

    read = wrote = 0

    def send(data: bytes) -> None:
        nonlocal wrote
        if data and not writer.is_closing():
            wrote += len(data)
            loop.call_later(delay, writer.write, data) if delay else writer.write(data)

    async def up() -> None:
        up_writer.write(head)
        with contextlib.suppress(ConnectionError):
            while data := await reader.read(READ_BYTES):
                up_writer.write(data)
                await up_writer.drain()

    async def down() -> None:
        nonlocal read
        with contextlib.suppress(ConnectionError):
            while data := await up_reader.read(READ_BYTES):
                read += len(data)
                send(dev.feed(data) if dev is not None else data)
        send(dev.end() if dev is not None else b"")
        if dev is None or not dev.withhold:
            loop.call_later(delay, writer.close) if delay else writer.close()

    closed = asyncio.create_task(up())
    answered = asyncio.create_task(down())
    try:
        # the network certifier never half-closes toward a server: its end of stream is the connection's end. Only a
        # withheld connection is bounded: the watermark's lasts the run
        if dev is not None and dev.withhold:
            for task in (await asyncio.wait({closed}, timeout=linger_s))[0]:
                task.result()
        else:
            await closed
    finally:
        answered.cancel()
        closed.cancel()
        await asyncio.gather(answered, closed, return_exceptions=True)
        up_writer.close()
        writer.close()
        if dev is not None:
            log({"release": None if release is None else release[0], "class": cls,
                 "planted": "".join(map(str, dev.planted)), "bytes_read": read, "bytes_sent": wrote,
                 "hold_ms": round(delay * 1000, 3), "withheld": dev.withhold, "whole": dev.whole})


async def serve(listen: tuple[str, int], *, front: tuple[str, int], cls: str, payload: int, bit: Callable[[], int],
                hold_s: float, log: Callable[[dict], None], announce=cs._announce) -> None:
    """The deviating process: `deviate` on every connection to `listen`, until SIGTERM or SIGINT."""
    server = await asyncio.start_server(
        lambda r, w: deviate(r, w, front=front, cls=cls, payload=payload, bit=bit, hold_s=hold_s, log=log), *listen)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    announce({"listening": [listen[0], server.sockets[0].getsockname()[1]]})
    await stop.wait()
    server.close()


# --- what a try keeps -------------------------------------------------------------------------------------------------

def digests(run: Path) -> dict[str, dict]:
    """Every file of a run directory by its path in it: its size and its SHA-256."""
    return {str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(run.rglob("*")) if p.is_file()}


def keep(cls: str, run: Path, *, stand_in: bool = False, seconds: float | None = None, hold_ms: float = 0.0) -> dict:
    """What the table needs of class `cls`'s try in the run directory `run` (the module docstring's "What a try
    keeps"), and nothing else of it. A connection's `ids` and `rest` are the id events and what follows them in its
    `DATA` and `END` cells of the egress record, where their SHA-256 is the one its client logged of what it got
    (`held`; else none). An egress cell is [bucket, slot, kind, stream, id] where `io_tie.parse_cell` reads it, else
    [bucket, slot, its bytes in hex without their zero fill]; stream 0's padding isn't listed. Under `stand_in` a
    connection also has the stand-in endpoint's own ids of its request (`warden_serve.fake_ids`)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    B, blank = params.cell_bytes, IT.pad_cell(params.cell_bytes)
    check = json.loads((run / "check.json").read_text())
    traffic = json.loads((run / "traffic.json").read_text())
    log = {c["ordinal"]: c for c in json.loads((run / "warden.json").read_text())["connections"]}
    lines = [json.loads(x) for x in (run / PLANTED).read_text().splitlines()] if (run / PLANTED).exists() else []
    bits = {d["release"]: d["planted"] for d in lines if d.get("release") is not None}
    whole = {d["release"]: bool(d.get("whole")) for d in lines if d.get("release") is not None}
    data: dict[int, bytearray] = defaultdict(bytearray)
    egress = []
    for record in sorted(ws.read_records(run, params)[ws.EGRESS], key=lambda r: r.window):
        cells = []
        for t, slots in enumerate(record.grid):
            for i, frame in enumerate(slots):
                if frame is None or frame == blank:
                    continue
                kind, ordinal, payload = read_cell(frame, B)
                if kind in (DATA, END):
                    data[ordinal] += payload
                p = IT.parse_cell(frame)
                cells.append([t, i, frame.rstrip(b"\0").hex()] if p is None else [t, i, p.kind, p.stream, p.id])
        egress.append({"window": record.window, "buckets": len(record.grid),
                       "width": len(record.grid[0]) if record.grid else 0, "cells": cells})
    connections = []
    for conn in json.loads((run / "client.json").read_text())["connections"]:
        o = conn.get("ordinal")
        if o is None:
            continue
        got = bytes(data.get(o, b""))
        held = hashlib.sha256(got).hexdigest() == conn["data_sha256"]
        ids, rest = events_of(got if held else b"", params.payload)
        fields, entry = traffic[conn["index"]]["ids"], log.get(o, {})
        kept = {"index": conn["index"], "ordinal": o, "user": fields["user"], "max_tokens": fields["max_tokens"],
                "release": entry.get("release"), "admitted": entry.get("admitted"), "failed": entry.get("failed"),
                "failed_at": entry.get("failed_at"), "kinds": conn["kinds"], "held": held, "ids": ids,
                "rest": rest.hex(), "planted": bits.get(entry.get("release"), ""),
                "whole": whole.get(entry.get("release"), False)}
        if stand_in:
            request = id_request(**fields, frame_bytes=B, offset=0)
            kept["stand_in"] = ws.fake_ids(json.loads(request.partition(BLANK)[2]))
        connections.append(kept)
    return {"class": cls, "run_id": dep.get("run_id"), "digest": dep.get("digest"), "params": dep["params"],
            "windows": dep.get("windows"), "hold_ms": hold_ms,
            "seconds": None if seconds is None else round(seconds, 1),
            "check": {k: v for k, v in check.items() if k != "upstream"} | {"words": ws.table(check)},
            "connections": connections, "egress": egress, "files": digests(run)}


def cell_of(entry: Sequence, cell_bytes: int) -> bytes:
    """The egress cell a kept entry (`keep`) stands for."""
    if len(entry) == 3:
        return bytes.fromhex(entry[2]).ljust(cell_bytes, b"\0")
    _, _, kind, stream, x = entry
    if kind == DATA:
        return IT.data_cell(x, stream, cell_bytes)
    return IT.end_cell(stream, cell_bytes) if kind == END else IT.pad_cell(cell_bytes, stream)


# --- the receiver ------------------------------------------------------------------------------------------------------

def reading(cls: str, conn: dict, params: WardenParams) -> list:
    """The receiver's reading of one connection (`keep`'s entry) for class `cls`, a symbol a planted bit, in the
    order the deviating process plants them."""
    ids, rest, kinds = conn["ids"], bytes.fromhex(conn["rest"]), conn["kinds"]
    if cls == "tokens":
        return [x & 1 for x in ids]
    if cls == "length":
        return [len(ids) % 2]
    failed = int(ws.KIND_CHARS[FAIL] in kinds)
    if cls == "bucket":
        return [(first_bucket(conn, params), failed)]
    if cls == "fail":
        return [failed]
    after = rest[len(IT.TRAILER):] if rest.startswith(IT.TRAILER) else rest
    return [chr(c) for c in after[:EXTRA_BITS]] if cls == "extra" else []


def first_bucket(conn: dict, params: WardenParams) -> int:
    """The bucket of the first cell a connection's client got that isn't padding, counted from its first cell."""
    kinds = conn["kinds"]
    return next((i for i, k in enumerate(kinds) if k != ws.KIND_CHARS[PAD]), len(kinds)) // params.slots


def decoded(cls: str, symbols: Sequence) -> list[int | None]:
    """The bits the receiver takes its readings of a try for: a reading that is a bit as it is, an `extra` byte the
    bit it spells, and a `bucket` reading 1 where it is past the try's least (a later bucket, or a failure)."""
    if cls == "bucket":
        least = min((y for y in symbols if y is not None), default=0)
        return [None if y is None else int(y > least) for y in symbols]
    return [int(y) if y in (0, 1, "0", "1") else None for y in symbols]


def score(cls: str, planted: Sequence[Sequence[int]], read: Sequence[Sequence]) -> dict:
    """The receiver's readings beside the planted bits, a connection an entry of each: the bits planted, those with a
    reading, those the receiver has right (`decoded`), and the mutual information between a planted bit and its
    reading over the try's bits, times their count: 0 when every reading is the same."""
    pairs = [(x, ys[i] if i < len(ys) else None) for xs, ys in zip(planted, read, strict=True)
             for i, x in enumerate(xs)]
    n = len(pairs)
    bits = decoded(cls, [y for _, y in pairs])
    joint, xs, ys = Counter(pairs), Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    info = sum(c / n * math.log2(c * n / (xs[x] * ys[y])) for (x, y), c in joint.items()) if n else 0.0
    return {"planted": n, "read": sum(y is not None for _, y in pairs),
            "right": sum(b == x for (x, _), b in zip(pairs, bits)), "information": round(max(0.0, n * info), 3)}


# --- the I/O tie on the relay's cells ----------------------------------------------------------------------------------

def read_commit(path: Path) -> dict[str, list[int]]:
    """Each request's sampled ids in the server's record `path`, a JSON line a request: its `output_ids` (serve-openai's
    `streamed_ids`) by its `user` where the line has one, else by its `id` (serve-openai's `http_id`)."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            name, ids = r.get("user") or r.get("id") or r.get("http_id"), r.get("output_ids", r.get("streamed_ids"))
            if name is not None and ids is not None:
                out[str(name)] = [int(x) for x in ids]
    return out


def committed_ids(kept: dict, sampled: Mapping[str, Sequence[int]]) -> dict[int, list[int]]:
    """The Commit's sampled ids of each request of a try, by its connection's ordinal: `sampled`'s (`read_commit`)
    under the request's `user`, else under ``verity-<run>-<N>``, the try's run id and the request's release number;
    a request the record doesn't hold has none."""
    out = {}
    for c in kept["connections"]:
        ids = sampled.get(c["user"])
        if ids is None and kept.get("run_id") and c["release"] is not None:
            ids = sampled.get(f"verity-{kept['run_id']}-{c['release']}")
        out[c["ordinal"]] = list(ids or ())
    return out


def stand_in_ids(kept: dict) -> dict[int, list[int]]:
    """The stand-in endpoint's own ids of each request of a try, by its connection's ordinal (`keep`'s `stand_in`)."""
    if not all("stand_in" in c for c in kept["connections"]):
        raise SystemExit(f"--stand-in: the {kept['class']} try wasn't run against the stand-in front")
    return {c["ordinal"]: c["stand_in"] for c in kept["connections"]}


def tie(kept: dict, committed: Mapping[int, Sequence[int]] | None) -> dict:
    """The I/O tie's content part on a try's egress record (the module docstring's stand-ins), `committed` each
    connection's sampled ids by its ordinal: `verdict` refused when a bucket's cells aren't its records' with the
    Commit's ids (`io_tie.cells_match`) or the Commit holds an output no cell carries (window -1, as
    `negatives.tie_run` names it), `failing` those windows, `codes` and the first's `words`; `NOT_RUN` without the
    ids. What it refused, by kind: `changed` the (connection, output) whose cell's id isn't the Commit's, `past` those
    past the Commit's outputs, `short` the connections with fewer data cells than the Commit's outputs, and `form`
    the cells that are no id-form cell."""
    if committed is None:
        return {"verdict": NOT_RUN, "codes": [], "failing": {}, "words": "the Commit's ids weren't given (--ids)"}
    params = WardenParams(**kept["params"])
    B = params.cell_bytes
    placed: dict[int, int] = defaultdict(int)
    failing: dict[int, list[str]] = {}
    words, changed, past, form = [], [], [], 0
    for record in kept["egress"]:
        rows = [[IT.pad_cell(B)] * record["width"] for _ in range(record["buckets"])]
        for entry in record["cells"]:
            rows[entry[0]][entry[1]] = cell_of(entry, B)
        for t, cells in enumerate(rows):
            port, records, why = [], [], None
            for cell in cells:
                p = IT.parse_cell(cell)
                if p is None:
                    kind, q, _ = read_cell(cell, B)
                    records.append(IT.UNHELD)
                    form += 1
                    why = why or f"connection {q}'s cell is no id-form cell (kind {kind:#x})"
                elif not p.data:
                    records.append(IT.MapEntry(p.stream, IT.END_ENTRY if p.kind == END else IT.EMPTY))
                else:
                    ids, k = committed.get(p.stream, ()), placed[p.stream]
                    placed[p.stream] += 1
                    if k < len(ids):
                        records.append(IT.MapEntry(p.stream, len(port)))
                        port.append(IT.port_row([ids[k]], 1))
                        if ids[k] != p.id:
                            changed.append((p.stream, k))
                            why = why or f"connection {p.stream}'s output {k} isn't the Commit's id"
                    else:
                        records.append(IT.MapEntry(p.stream, IT.EMPTY))
                        past.append((p.stream, k))
                        why = why or f"connection {p.stream}'s output {k} is past the Commit's {len(ids)}"
            shape = IT.Shape(B, max(2, record["buckets"]), len(cells), P=1, S=max(1, len(port)), DS=1)
            if not IT.cells_match(shape, cells, records, port or [IT.port_row([], 1)]):
                failing.setdefault(record["window"], ["content"])
                words.append(f"{N.TIE}.content at window {record['window']}, bucket {t}: {why}")
    short = [q for q, ids in sorted(committed.items()) if placed[q] < len(ids)]
    for q in short:
        failing.setdefault(-1, ["content"])
        words.append(f"{N.TIE}.content at window -1: connection {q} has {placed[q]} data cells, the Commit "
                     f"{len(committed[q])} outputs")
    return {"verdict": "refused" if failing else "accepted", "codes": [f"{N.TIE}.content"] if failing else [],
            "failing": failing, "words": words[0] if words else "", "refusals": len(words), "changed": sorted(changed),
            "past": sorted(past), "short": short, "form": form}


# --- a try and its row -------------------------------------------------------------------------------------------------

def charged(cls: str, connections: int, params: WardenParams) -> str:
    """What the leak bound charges the class's way out over the try's connections (`leak_bounds.terms`)."""
    t = LB.terms(connections, 1, cells=params.egress_cells, steps=params.window)
    if cls == "length":
        return f"{t.schedules} bits: the schedules, {connections} connections of at most {params.egress_cells} cells"
    if cls in ("fail", "extra"):
        return f"{t.cells} bits: the fail cells, {connections} connections of at most {params.egress_cells} cells"
    if cls == "bucket":
        return (f"{t.jitter} bits: the jitter at a jitter bound of 1; a hold that fails its connection is charged as "
                f"its fail cell, {t.cells} bits over the {connections} connections")
    return "nothing: no way out" if cls == "honest" else "nothing: refused, not charged"


def deviated(cls: str, kept: dict, committed: Mapping[int, Sequence[int]]) -> tuple[list, bool]:
    """What class `cls`'s planted bits changed of the Commit's outputs in a try: for ``tokens`` each (connection,
    output) whose id's low bit isn't its planted bit, for ``length`` each connection whose last event was dropped. And
    whether each connection was sent the Commit's ids so changed, and nothing else."""
    out, exact = [], True
    for c in kept["connections"]:
        ids, bits = list(committed.get(c["ordinal"], ())), [int(b) for b in c["planted"]]
        if cls == "tokens":
            out += [(c["ordinal"], k) for k, (x, b) in enumerate(zip(ids, bits)) if x & 1 != b]
            sent = [x & ~1 | b for x, b in zip(ids, bits)] if len(ids) == len(bits) else None
        else:
            dropped = bool(ids and bits and len(ids) % 2 != bits[0])
            out += [c["ordinal"]] * dropped
            sent = ids[:-1] if dropped else ids
        exact = exact and c["ids"] == sent
    return sorted(out), exact


def row(kept: dict, committed: Mapping[int, Sequence[int]] | None) -> dict:
    """A try's row from what it kept (`keep`): what is changed, the check that refuses it, each check's verdict and
    the refusal's own words, the receiver's score, what the bound charges, and `ok`: the class did what its check
    says, by the deviation alone (the module docstring's "A refusal counts only where it is the deviation's"; None:
    the tie it aims at wasn't run, or the deviation changed no response)."""
    cls, check, conns = kept["class"], kept["check"], kept["connections"]
    params = WardenParams(**kept["params"])
    result = score(cls, [[int(b) for b in c["planted"]] for c in conns], [reading(cls, c, params) for c in conns])
    tied = tie(kept, committed)
    accepted, relay_words = check["accepted"], check["words"].split(";")[0]
    audit = [p for p in check["problems"] if p.startswith("audit:")]
    failed = {c["ordinal"]: c["failed"] for c in conns if c["failed"]}
    one = {c["ordinal"] for c in conns if c["planted"] == "1"}
    shown = result["information"] > 0
    refusals = len(failed)
    if cls == "honest":
        ok = accepted and tied["verdict"] != "refused"
        words = relay_words if ok else tied["words"] if accepted else "; ".join([relay_words, *check["problems"][:2]])
    elif cls in ("tokens", "length"):
        refusals = tied.get("refusals", 0)
        if not accepted:
            ok = False
            words = "the relay's check refused a try that fails no connection: " + "; ".join(check["problems"][:2])
        elif tied["verdict"] == NOT_RUN:
            ok, words = None, tied["words"]
        else:
            want, exact = deviated(cls, kept, committed)
            mine, other = ("changed", ("past", "short")) if cls == "tokens" else ("short", ("changed", "past"))
            alone = exact and tied[mine] == want and not tied["form"] and not any(tied[k] for k in other)
            ok = False if not alone else True if want else None
            words = tied["words"] or "the tie's content accepted every bucket"
    elif cls == "bucket" and accepted:
        ok, n = not shown, len(conns)
        on_time = sum(first_bucket(c, params) == params.delay for c in conns)
        words = (f"accepted, and the hold is {'not ' if ok else ''}seen at the client's bucket: {len(one)} of {n} "
                 f"connections held {kept['hold_ms']:g} ms, and the first cell of {on_time} of {n} is {params.delay} "
                 f"buckets after its admission")
    else:
        want = {"fail": one, "extra": {c["ordinal"] for c in conns}, "bucket": one}[cls]
        reasons = {"format", "limit"} if cls == "extra" else {"late"}
        mine = bool(failed) and set(failed.values()) <= reasons and (set(failed) <= want if cls == "bucket" else
                                                                     set(failed) == want)
        if cls == "extra":   # the cell after a whole response's last id, and before it the Commit's ids where given
            mine = mine and not shown and all(
                c["whole"] and c["planted"] and c["failed_at"] == len(c["ids"]) and
                (committed is None or c["ids"] == list(committed.get(c["ordinal"], ()))) for c in conns)
        ok = not accepted and mine
        if accepted and not want and not failed:
            ok = None
        words = "; ".join(audit[:2]) or "; ".join([relay_words, *check["problems"][:2]])
    cost = charged(cls, len(conns), params)
    if cls == "tokens" and tied["verdict"] != "refused":
        cost = "nothing, and the tie that refuses it " + ("wasn't run" if tied["verdict"] == NOT_RUN else "accepted")
    return {"class": cls, "changed": CASES[cls][0], "check": CASES[cls][1],
            "relay": "accepted" if accepted else "refused", "tie": tied["verdict"], "words": words,
            "refusals": refusals, **result, "charged": cost, "connections": len(conns), "seconds": kept["seconds"],
            "run_id": kept["run_id"], "ok": ok}


def lane_seconds(args) -> float:
    """The seconds a connection admitted at a try's last request's time may last, at the rehearsal's schedule: its
    delay, its response's cells and a bucket each side."""
    flags = ["rehearse", "--fake", "--ids", "--out", ".", "--cell-bytes", str(args.cell_bytes), *args.rehearse]
    try:
        params = ws._params(ws.parser().parse_args(flags))
    except ValueError as e:
        raise SystemExit(f"the rehearsal's flags: {e}") from None
    return (params.delay + args.max_tokens // params.slots + 2) * params.bucket_ms / 1000


def last_lines(path: Path, n: int = 5) -> str:
    """A log's last lines, each address in them cut out."""
    lines = path.read_text(errors="replace").splitlines()[-n:] if path.exists() else []
    return "\n".join("    " + ADDRESS.sub("<address>", x)[:300] for x in lines) or "    (none)"


class NoCheck(Exception):
    """A try that didn't run to its check."""


def _stop(procs: Sequence[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        proc.wait()


def one_try(cls: str, run: Path, args, clock=time.monotonic) -> float:
    """One try of class `cls` in the run directory `run` (the module docstring's `try`): the stand-in front when
    asked, the deviating process before the front, and `warden_serve.py rehearse` through it; the seconds it took end
    to end. `NoCheck` when it wrote no check."""
    run.mkdir(parents=True, exist_ok=True)
    payload = WardenParams(cell_bytes=args.cell_bytes).payload
    requests = args.bucket_requests if cls == "bucket" else args.requests
    start, procs, env, name = clock(), [], dict(os.environ), args.front_env
    me, relay = [sys.executable, str(HERE / "deviations.py")], [sys.executable, str(HERE / "warden_serve.py")]
    try:
        if args.stand_in:
            fake = subprocess.Popen(relay + ["fake", "--listen", "127.0.0.1:0", "--id-payload", str(payload),
                                             "--first-ms", str(args.first_ms)], stdout=subprocess.PIPE, text=True)
            procs.append(fake)
            name, env[STAND_IN_ENV] = STAND_IN_ENV, "%s:%d" % tuple(json.loads(fake.stdout.readline())["listening"])
        seed = [] if args.seed is None else ["--seed", str(args.seed)]
        with open(run / "relay.log", "w") as log:
            deviating = subprocess.Popen(me + ["relay", "--class", cls, "--front-env", name, "--payload", str(payload),
                                               "--log", str(run / PLANTED), "--hold-ms", str(args.hold_ms), *seed],
                                         stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        procs.append(deviating)
        line = deviating.stdout.readline()
        if not line:
            raise NoCheck(f"{cls}: the deviating process ended before it listened (exit {deviating.wait()}):\n"
                          + last_lines(run / "relay.log"))
        upstream = "%s:%d" % tuple(json.loads(line)["listening"])
        with open(run / "rehearse.log", "w") as log:
            subprocess.run(relay + ["rehearse", "--upstream", upstream, "--ids", "--users", f"{cls}-{args.tag}",
                                    "--out", str(run), "--requests", str(requests), "--rate", str(args.rate),
                                    "--max-tokens", str(args.max_tokens), "--tail", str(args.tail), "--cell-bytes",
                                    str(args.cell_bytes), *args.rehearse],
                           stdout=log, stderr=subprocess.STDOUT)
    finally:
        _stop(procs)
    if not (run / "check.json").exists():
        raise NoCheck(f"{cls}: the rehearsal wrote no check; its log's last lines:\n"
                      + last_lines(run / "rehearse.log"))
    return clock() - start


def tries(classes: Sequence[str], args, attempt: Callable[[str, Path, Any], float] = one_try) -> tuple[list[str], bool]:
    """Each class of `classes` as one try (`attempt`) in the scratch directory, what it keeps written to ``--out``
    (`keep`): the classes tried to their check, and whether the tries stopped at an honest control that was refused
    or had no check, the rest untried. A deviation's try with no check (`NoCheck`) has no row, and the classes after
    it are still tried."""
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="deviations-"))
    sampled = None if args.ids is None else read_commit(args.ids)
    args.out.mkdir(parents=True, exist_ok=True)
    tried = []
    try:
        for cls in classes:
            try:
                seconds = attempt(cls, scratch / cls, args)
            except NoCheck as e:
                print(f"{e}\n{cls}: no row", flush=True)
                if cls == "honest" and len(classes) > 1:
                    return tried, True
                continue
            kept = keep(cls, scratch / cls, stand_in=args.stand_in, seconds=seconds, hold_ms=args.hold_ms)
            (args.out / f"{cls}.json").write_text(json.dumps(kept))
            tried.append(cls)
            print(f"{cls}: tried in {seconds:.1f} s, the certifier's run {str(kept['run_id'])[:12]}", flush=True)
            if cls == "honest" and len(classes) > 1 and not row(kept, committed_of(kept, sampled, args.stand_in))["ok"]:
                return tried, True
    finally:
        if args.scratch is None:
            shutil.rmtree(scratch, ignore_errors=True)
    return tried, False


def committed_of(kept: dict, sampled: Mapping[str, Sequence[int]] | None, stand_in: bool) -> dict | None:
    """A try's Commit: the server's record's ids when given, else the stand-in's own under `stand_in`, else none."""
    return committed_ids(kept, sampled) if sampled is not None else stand_in_ids(kept) if stand_in else None


def table(rows: Sequence[dict]) -> str:
    out = []
    for r in rows:
        verdict = {True: "as its check says", False: "NOT as its check says", None: "not shown either way"}[r["ok"]]
        took = "" if r["seconds"] is None else f"; the try took {r['seconds']} s"
        out += [f"[{r['class']}] changed: {r['changed']}",
                f"    check:   {r['check']}",
                f"    verdict: the relay's check {r['relay']}, the I/O tie {r['tie']}: {verdict}",
                f"    words:   {r['words'] or 'none'}" + (f" ({r['refusals']} in all)" if r["refusals"] > 1 else ""),
                f"    bits:    {r['planted']} planted over {r['connections']} connections, {r['right']} read right, "
                f"information {r['information']}; charged {r['charged']}{took}"]
    return "\n".join(out)


def rows_of(out: Path, classes: Sequence[str], ids: Path | None, stand_in: bool) -> list[dict]:
    """The rows of the tries `classes` kept in `out`."""
    sampled = None if ids is None else read_commit(ids)
    rows = []
    for cls in classes:
        kept = json.loads((out / f"{cls}.json").read_text())
        rows.append(row(kept, committed_of(kept, sampled, stand_in)))
    return rows


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deviations.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("try", help="each class as one try through a network certifier of its own, and the table")
    r = sub.add_parser("relay", help="the deviating process alone")
    x = sub.add_parser("table", help="the table of the tries a directory keeps")
    for q in (t, x):
        q.add_argument("--out", required=True, type=Path, help="what each try keeps, a file a class, and the table")
        q.add_argument("--class", dest="cls", choices=("all", *CLASSES), default="all")
        q.add_argument("--ids", type=Path, help="the server's record of each request's sampled ids, JSON lines: its "
                       "output_ids by its user, or by its id verity-RUN-N (the Commit's rows the I/O tie reads)")
    x.add_argument("--stand-in", action="store_true", help="the tries ran against the stand-in front")
    front = t.add_mutually_exclusive_group(required=True)
    front.add_argument("--stand-in", action="store_true", help="start the stand-in id endpoint as the front")
    for q in (front, r):
        q.add_argument("--front-env", metavar="NAME", required=q is r,
                       help="the environment variable that holds the front's HOST:PORT")
    for q in (t, r):
        q.add_argument("--hold-ms", type=float, default=500.0, help="how long the bucket class holds a response")
        q.add_argument("--seed", type=int, help="the planted string's seed (default: the system's randomness)")
    t.add_argument("--tag", help="a try's request i is named CLASS-TAG-i in the server's record (default: a random "
                                 "tag a command)")
    t.add_argument("--requests", type=int, default=8)
    t.add_argument("--bucket-requests", type=int, default=64,
                   help="the bucket class's requests: it plants a bit a connection")
    t.add_argument("--rate", type=float, default=2.0, help="Poisson arrivals a second")
    t.add_argument("--max-tokens", type=int, default=16, help="the most max_tokens a request samples")
    t.add_argument("--tail", type=float, default=8.0, help="seconds a try's run lasts past its last request's time")
    t.add_argument("--cell-bytes", type=int, default=WardenParams().cell_bytes, help="the network certifier's B")
    t.add_argument("--first-ms", type=float, default=100.0, help="the stand-in's milliseconds to its response")
    t.add_argument("--scratch", type=Path, help="where each try's run directory goes, and stays (default: a temporary "
                                                "directory, removed when the command ends)")
    t.add_argument("rehearse", nargs="*", metavar="REHEARSE-FLAG", default=["--window", "60", "--lanes", "64"],
                   help="after --: warden_serve.py rehearse's own flags, the schedule's among them (default: "
                        "--window 60 --lanes 64)")
    r.add_argument("--class", dest="cls", choices=CLASSES, required=True)
    r.add_argument("--payload", type=int, required=True, help="an id event's bytes, the network certifier's B - 9")
    r.add_argument("--listen", type=cs._address, default=("127.0.0.1", 0), help="HOST:PORT (port 0: any)")
    r.add_argument("--log", type=Path, help="one JSON line an id request, when its connection ends")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "relay":
        front = front_of(args.front_env)

        def log(doc: dict) -> None:
            if args.log is not None:
                with open(args.log, "a") as f:
                    f.write(json.dumps(doc) + "\n")

        asyncio.run(serve(args.listen, front=front, cls=args.cls, payload=args.payload, bit=bits_of(args.seed),
                          hold_s=args.hold_ms / 1000, log=log))
        return 0
    classes = asked = list(CLASSES if args.cls == "all" else (args.cls,))
    stopped = False
    if args.command == "try":
        if not args.stand_in:
            front_of(args.front_env)
        if args.tail < (need := lane_seconds(args)):
            raise SystemExit(f"--tail {args.tail:g}: a connection admitted at the last request's time lasts up to "
                             f"{need:g} s at this schedule (its delay, its cells and a bucket each side); give --tail "
                             f"at least {math.ceil(need)}")
        args.tag = args.tag or secrets.token_hex(4)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a cancelled command stops the processes it started
        classes, stopped = tries(classes, args)
    else:
        classes = [cls for cls in classes if (args.out / f"{cls}.json").exists()]
        if not classes:
            raise SystemExit(f"{args.out} keeps no try (CLASS.json)")
    rows = rows_of(args.out, classes, args.ids, args.stand_in)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    (args.out / "table.txt").write_text(table(rows) + "\n")
    print(table(rows))
    if args.command == "try":
        print(f"{len(rows)} tries in {sum(r['seconds'] for r in rows):.1f} s")
    if stopped:
        print("the honest control was refused or had no check: no deviation was tried")
    untried = args.command == "try" and len(classes) < len(asked)
    return 1 if untried or any(r["ok"] is False for r in rows) else 2 if any(r["ok"] is None for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
