"""deviations: a server that deviates behind the network certifier's relay, six classes, each tried, read back and scored.

A process between the network certifier (`warden_serve`'s relay, its id egress form) and the server's id front plays
the cheating server. It plants a random string and tries one way to send it out, a class a try:

- ``honest``: nothing changes; the control.
- ``tokens``: each output id's low bit is the string's next bit, a bit a token. The cells stay id-form cells, which the
  relay releases, so only the I/O tie refuses them: it holds each data cell's id to the Commit's sampled id.
- ``length``: a response loses its last id event when its count's parity isn't the connection's bit, a bit a
  connection. The I/O tie refuses it too: the Commit then holds an output that no cell carries.
- ``bucket``: a response is held ``--hold-ms`` when the connection's bit is 1. The relay releases egress cell i in a
  bucket that the admission fixes, so a hold short of a cell's deadline doesn't show, and one past it fails the
  connection (late), which the relay's audit refuses.
- ``fail``: the server withholds a response when the connection's bit is 1, so the relay fails the connection (late)
  at its first egress cell. The relay's audit refuses a run in which a connection failed, and the leak bound charges
  each connection's fail cell.
- ``extra``: after a response's trailer come bytes out of the id form that carry eight of the string's bits. The
  relay fails the connection (format) at that cell, and none of its bytes leaves.

**The receiver** (`received`, `reading`) reads only what the client got: each connection's cells' kinds as
`client.json` has them, and the bytes of its `DATA` and `END` cells in the egress record where their SHA-256 is the
one the client logged. Its reading of a class is fixed before the try: the ids' low bits, the count's parity, the
bucket of the first cell that isn't padding and whether the connection failed, whether it failed, the bytes after the
trailer. `score` sets the reading beside the planted bits: how many it has right, and the mutual information between
the two over the try.

**The checks.** The relay's is `warden_serve.check`, whose refusal's words are its audit's (`warden.audit`). The I/O
tie's is its content part (`io_tie.cells_match`, `IoTieContent`'s reference) on each bucket of the egress record
(`tie`), with two stand-ins for what a try doesn't have: the map is the certified placement, each data cell its
stream's next output, and the Commit is the ids given (``--ids``, the server's record of each request's sampled ids
by its `user`), or under ``--stand-in`` the stand-in endpoint's own (`warden_serve.fake_ids` of each request). Without
either the tie isn't run, and the row says so.

**What the bound charges** (`leak_bounds.terms`): a connection's schedule and its fail cell, each ⌈log2(cells + 1)⌉
bits, and the jitter, 0 at a jitter bound of 1; a refused class is charged nothing.

    python deviations.py try (--stand-in | --front-env NAME) --out DIR [--class all|CLASS] [--ids FILE] [--seed S]
        [--tag T] [--requests 8] [--rate 2] [--max-tokens 16] [--tail 8] [--hold-ms 500] [--cell-bytes B]
        [-- REHEARSE-FLAG ...]
    python deviations.py relay --class CLASS --front-env NAME --payload P [--listen 127.0.0.1:0] [--log FILE]
        [--hold-ms 500] [--seed S]
    python deviations.py table --out DIR [--ids FILE] [--stand-in]

- `try` runs each class as one try, with a network certifier process of its own: the deviating process (`relay`) in
  front of the front, then `warden_serve.py rehearse --ids` through it (its key, its traffic, `serve`, `client` and
  `check`; the flags after ``--`` are its own, the schedule's among them), then the receiver and the row. It prints the
  table, a class a row: what is changed, the check that refuses it, the refusal's own words, the bits planted, read
  right and charged, and the try's seconds end to end; `rows.json` holds the rows. It exits 0 when every class did
  what its row's check says: honest accepted, and each other class refused by that check or not shown to the client.
  It exits 1 when a class didn't, and 2 when none didn't and a row is shown neither way: the tie it aims at wasn't
  run (no ``--ids`` yet), or its deviation changed no response.
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
- `table` reads the tries already in a directory again and prints the table, with `try`'s exit code.
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
import secrets
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

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

CLASSES = ("honest", "tokens", "length", "bucket", "fail", "extra")
#: a class's row: what the deviating process changes, and the check that refuses it
CASES = {
    "honest": ("nothing (the control)", "none: the relay's check accepts, and the I/O tie where it is run"),
    "tokens": ("each output id's low bit is a planted bit",
               "the I/O tie's content: a data cell's id against the Commit's sampled id"),
    "length": ("a response loses its last id event when its count's parity isn't the planted bit",
               "the I/O tie's content: the Commit holds an output no cell carries"),
    "bucket": ("a response is held when the planted bit is 1",
               "the relay's schedule: a cell's bucket is the admission's, so a hold doesn't show, and one past a "
               "cell's deadline fails the connection (late), which the relay's audit refuses"),
    "fail": ("a response is withheld when the planted bit is 1",
             "the relay's audit: no connection failed"),
    "extra": ("bytes out of the id form after the trailer carry eight planted bits",
              "the relay's egress form: a cell out of it fails its connection (format), and the audit refuses the "
              "run"),
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
    every byte is held, `withhold` whether nothing is sent and the connection is left open, and `planted` the bits it
    planted, in the order the receiver reads them. `bit` draws the planted string's next bit."""

    def __init__(self, cls: str, bit: Callable[[], int], payload: int, hold_s: float = 0.0) -> None:
        if cls not in CLASSES:
            raise ValueError(f"a class is one of {', '.join(CLASSES)}, not {cls!r}")
        self.cls, self.payload, self._bit = cls, payload, bit
        self.planted: list[int] = []
        self.buffer, self.held, self.events = bytearray(), b"", 0
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

    def feed(self, data: bytes) -> bytes:
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
        # the network certifier never half-closes toward a server: its end of stream is the connection's end
        await asyncio.wait_for(closed, linger_s)
    except (TimeoutError, asyncio.TimeoutError):
        pass
    finally:
        answered.cancel()
        closed.cancel()
        await asyncio.gather(answered, closed, return_exceptions=True)
        up_writer.close()
        writer.close()
        if dev is not None:
            log({"release": None if release is None else release[0], "class": cls,
                 "planted": "".join(map(str, dev.planted)), "bytes_read": read, "bytes_sent": wrote,
                 "hold_ms": round(delay * 1000, 3), "withheld": dev.withhold})


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


# --- the receiver ------------------------------------------------------------------------------------------------------

def received(run: Path) -> list[dict]:
    """What each connection's client got in the try `run`, in `client.json`'s order: its ordinal, its cells' kinds, and
    the bytes of its `DATA` and `END` cells, read from the egress record where their SHA-256 is the one the client
    logged of what it got (`held`; else no bytes)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    data: dict[int, bytearray] = defaultdict(bytearray)
    for record in sorted(ws.read_records(run, params)[ws.EGRESS], key=lambda r: r.window):
        for row in record.grid:
            for frame in row:
                if frame is not None:
                    kind, ordinal, payload = read_cell(frame, params.cell_bytes)
                    if kind in (DATA, END):
                        data[ordinal] += payload
    out = []
    for conn in json.loads((run / "client.json").read_text())["connections"]:
        if conn.get("ordinal") is None:
            continue
        got = bytes(data.get(conn["ordinal"], b""))
        held = hashlib.sha256(got).hexdigest() == conn["data_sha256"]
        out.append({"index": conn["index"], "ordinal": conn["ordinal"], "kinds": conn["kinds"],
                    "data": got if held else b"", "held": held})
    return out


def reading(cls: str, conn: dict, params: WardenParams) -> list:
    """The receiver's reading of one connection (`received`'s entry) for class `cls`, a symbol a planted bit, in the
    order the deviating process plants them."""
    ids, rest = events_of(conn["data"], params.payload)
    kinds = conn["kinds"]
    if cls == "tokens":
        return [x & 1 for x in ids]
    if cls == "length":
        return [len(ids) % 2]
    failed = int(ws.KIND_CHARS[FAIL] in kinds)
    if cls == "bucket":
        first = next((i for i, k in enumerate(kinds) if k != ws.KIND_CHARS[PAD]), len(kinds))
        return [(first // params.slots, failed)]
    if cls == "fail":
        return [failed]
    after = rest[len(IT.TRAILER):] if rest.startswith(IT.TRAILER) else rest
    return [chr(c) for c in after[:EXTRA_BITS]] if cls == "extra" else []


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


def planted(run: Path) -> dict[int, list[int]]:
    """The bits the deviating process planted on each connection, by its ordinal: its log's lines (`PLANTED`), each
    under its release number, which the network certifier's log gives the ordinal of."""
    ordinal = {c["release"]: c["ordinal"] for c in json.loads((run / "warden.json").read_text())["connections"]
               if c.get("release") is not None}
    lines = [json.loads(x) for x in (run / PLANTED).read_text().splitlines()] if (run / PLANTED).exists() else []
    return {ordinal[d["release"]]: [int(c) for c in d["planted"]] for d in lines if d.get("release") in ordinal}


# --- the I/O tie on the relay's cells ----------------------------------------------------------------------------------

def stand_in_ids(run: Path) -> dict[int, list[int]]:
    """The stand-in endpoint's own ids of each request of the try, by its connection's ordinal: `warden_serve.fake_ids`
    of the body it read."""
    dep = json.loads((run / "deployment.json").read_text())
    traffic = json.loads((run / "traffic.json").read_text())
    out = {}
    for conn in json.loads((run / "client.json").read_text())["connections"]:
        if conn.get("ordinal") is not None:
            request = id_request(**traffic[conn["index"]]["ids"], frame_bytes=dep["params"]["cell_bytes"], offset=0)
            out[conn["ordinal"]] = ws.fake_ids(json.loads(request.partition(BLANK)[2]))
    return out


def committed_ids(run: Path, ids: Path) -> dict[int, list[int]]:
    """The Commit's sampled ids of each request of the try, by its connection's ordinal: `ids`' rows
    (`negatives.read_ids`) under each request's `user`; a request the record doesn't hold has none."""
    sampled = N.read_ids(ids)
    traffic = json.loads((run / "traffic.json").read_text())
    return {conn["ordinal"]: sampled.get(traffic[conn["index"]]["ids"]["user"], [])
            for conn in json.loads((run / "client.json").read_text())["connections"] if conn.get("ordinal") is not None}


def tie(run: Path, committed: Mapping[int, Sequence[int]] | None) -> dict:
    """The I/O tie's content part on the try's egress record (the module docstring's stand-ins), `committed` each
    connection's sampled ids by its ordinal: `verdict` refused when a bucket's cells aren't its records' with the
    Commit's ids (`io_tie.cells_match`) or the Commit holds an output no cell carries (window -1, as
    `negatives.tie_run` names it), `failing` those windows, `codes` and the first's `words`; `NOT_RUN` without the
    ids."""
    if committed is None:
        return {"verdict": NOT_RUN, "codes": [], "failing": {}, "words": "the Commit's ids weren't given (--ids)"}
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    B = params.cell_bytes
    placed: dict[int, int] = defaultdict(int)
    failing: dict[int, list[str]] = {}
    words = []
    for record in sorted(ws.read_records(run, params)[ws.EGRESS], key=lambda r: r.window):
        for t, row in enumerate(record.grid):
            cells = [IT.pad_cell(B) if f is None else f for f in row]
            port, records, why = [], [], None
            for cell in cells:
                p = IT.parse_cell(cell)
                if p is None:
                    kind, q, _ = read_cell(cell, B)
                    records.append(IT.UNHELD)
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
                            why = why or f"connection {p.stream}'s output {k} isn't the Commit's id"
                    else:
                        records.append(IT.MapEntry(p.stream, IT.EMPTY))
                        why = why or f"connection {p.stream}'s output {k} is past the Commit's {len(ids)}"
            shape = IT.Shape(B, max(2, len(record.grid)), len(cells), P=1, S=max(1, len(port)), DS=1)
            if not IT.cells_match(shape, cells, records, port or [IT.port_row([], 1)]):
                failing.setdefault(record.window, ["content"])
                words.append(f"{N.TIE}.content at window {record.window}, bucket {t}: {why}")
    for q, ids in sorted(committed.items()):
        if placed[q] < len(ids):
            failing.setdefault(-1, ["content"])
            words.append(f"{N.TIE}.content at window -1: connection {q} has {placed[q]} data cells, the Commit "
                         f"{len(ids)} outputs")
    return {"verdict": "refused" if failing else "accepted", "codes": [f"{N.TIE}.content"] if failing else [],
            "failing": failing, "words": words[0] if words else "", "refusals": len(words)}


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


def row(cls: str, run: Path, committed: Mapping[int, Sequence[int]] | None, seconds: float | None = None) -> dict:
    """Class `cls`'s row from its try's directory: what is changed, the check that refuses it, each check's verdict
    and the refusal's own words, the receiver's score, what the bound charges, and `ok`: the class did what its check
    says (None: the tie it aims at wasn't run, or the deviation changed no response)."""
    check = json.loads((run / "check.json").read_text())
    params = WardenParams(**json.loads((run / "deployment.json").read_text())["params"])
    conns = received(run)
    bits = planted(run)
    result = score(cls, [bits.get(c["ordinal"], []) for c in conns], [reading(cls, c, params) for c in conns])
    tied = tie(run, committed)
    audit = [p for p in check["problems"] if p.startswith("audit:")]
    relay = "accepted" if check["accepted"] else "refused"
    shown = result["information"] > 0
    if cls == "honest":
        ok = check["accepted"] and tied["verdict"] != "refused"
        words = ws.table(check).split(";")[0]
    elif cls in ("tokens", "length"):
        changed = tied["verdict"] == "refused" or shown
        ok = None if tied["verdict"] == NOT_RUN or not changed else tied["verdict"] == "refused"
        words = tied["words"]
    elif cls == "bucket":
        ok = bool(audit) if not check["accepted"] else not shown
        words = "; ".join(audit[:2]) if audit else "none: " + ws.table(check).split(";")[0]
    else:
        reason = "(format)" if cls == "extra" else "failed closed"
        ok = (None if cls == "fail" and not any(bits.values()) else
              any(reason in p for p in audit) and not (cls == "extra" and shown))
        words = "; ".join(audit[:2])
    cost = charged(cls, len(conns), params)
    if cls == "tokens" and tied["verdict"] != "refused":
        cost = "nothing, and the tie that refuses it " + ("wasn't run" if tied["verdict"] == NOT_RUN else "accepted")
    return {"class": cls, "changed": CASES[cls][0], "check": CASES[cls][1], "relay": relay, "tie": tied["verdict"],
            "words": words, "refusals": len(audit) if cls not in ("tokens", "length") else tied.get("refusals", 0),
            **result, "charged": cost, "connections": len(conns),
            "seconds": None if seconds is None else round(seconds, 1), "ok": ok}


def _stop(procs: Sequence[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        proc.wait()


def one_try(cls: str, run: Path, args, clock=time.monotonic) -> float:
    """One try of class `cls` in `run` (the module docstring's `try`): the stand-in front when asked, the deviating
    process before the front, and `warden_serve.py rehearse` through it; the seconds it took end to end."""
    run.mkdir(parents=True, exist_ok=True)
    payload = WardenParams(cell_bytes=args.cell_bytes).payload
    start, procs, env, name = clock(), [], dict(os.environ), args.front_env
    me, relay = [sys.executable, str(HERE / "deviations.py")], [sys.executable, str(HERE / "warden_serve.py")]
    try:
        if args.stand_in:
            fake = subprocess.Popen(relay + ["fake", "--listen", "127.0.0.1:0", "--id-payload", str(payload),
                                             "--first-ms", str(args.first_ms)], stdout=subprocess.PIPE, text=True)
            procs.append(fake)
            name, env[STAND_IN_ENV] = STAND_IN_ENV, "%s:%d" % tuple(json.loads(fake.stdout.readline())["listening"])
        seed = [] if args.seed is None else ["--seed", str(args.seed)]
        deviating = subprocess.Popen(me + ["relay", "--class", cls, "--front-env", name, "--payload", str(payload),
                                           "--log", str(run / PLANTED), "--hold-ms", str(args.hold_ms), *seed],
                                     stdout=subprocess.PIPE, text=True, env=env)
        procs.append(deviating)
        line = deviating.stdout.readline()
        if not line:
            raise SystemExit(f"the deviating process ended before it listened (exit {deviating.wait()})")
        upstream = "%s:%d" % tuple(json.loads(line)["listening"])
        with open(run / "rehearse.log", "w") as log:
            subprocess.run(relay + ["rehearse", "--upstream", upstream, "--ids", "--users", f"{cls}-{args.tag}",
                                    "--out", str(run), "--requests", str(args.requests), "--rate", str(args.rate),
                                    "--max-tokens", str(args.max_tokens), "--tail", str(args.tail), "--cell-bytes",
                                    str(args.cell_bytes), *args.rehearse],
                           stdout=log, stderr=subprocess.STDOUT)
    finally:
        _stop(procs)
    if not (run / "check.json").exists():
        raise SystemExit(f"{cls}: the rehearsal wrote no check ({run / 'rehearse.log'})")
    return clock() - start


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


def rows_of(out: Path, classes: Sequence[str], ids: Path | None, stand_in: bool,
            seconds: Mapping[str, float] = {}) -> list[dict]:
    rows = []
    for cls in classes:
        run = out / cls
        committed = committed_ids(run, ids) if ids is not None else stand_in_ids(run) if stand_in else None
        rows.append(row(cls, run, committed, seconds.get(cls)))
    return rows


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deviations.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("try", help="each class as one try through a network certifier of its own, and the table")
    r = sub.add_parser("relay", help="the deviating process alone")
    x = sub.add_parser("table", help="the table of the tries already in a directory")
    for q in (t, x):
        q.add_argument("--out", required=True, type=Path)
        q.add_argument("--class", dest="cls", choices=("all", *CLASSES), default="all")
        q.add_argument("--ids", type=Path, help="the server's record of each request's sampled ids by its user "
                       "(requests.jsonl: user, output_ids), the Commit's rows the I/O tie reads")
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
    t.add_argument("--rate", type=float, default=2.0, help="Poisson arrivals a second")
    t.add_argument("--max-tokens", type=int, default=16, help="the most max_tokens a request samples")
    t.add_argument("--tail", type=float, default=8.0, help="seconds a try's run lasts past its last request's time")
    t.add_argument("--cell-bytes", type=int, default=WardenParams().cell_bytes, help="the network certifier's B")
    t.add_argument("--first-ms", type=float, default=100.0, help="the stand-in's milliseconds to its response")
    t.add_argument("rehearse", nargs="*", metavar="REHEARSE-FLAG", default=["--window", "60"],
                   help="after --: warden_serve.py rehearse's own flags, the schedule's among them (default: "
                        "--window 60)")
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
    classes = CLASSES if args.cls == "all" else (args.cls,)
    seconds = {}
    if args.command == "try":
        if not args.stand_in:
            front_of(args.front_env)
        args.tag = args.tag or secrets.token_hex(4)
        for cls in classes:
            seconds[cls] = one_try(cls, args.out / cls, args)
            print(f"{cls}: tried in {seconds[cls]:.1f} s", flush=True)
    rows = rows_of(args.out, classes, args.ids, args.stand_in, seconds)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(table(rows))
    if seconds:
        print(f"{len(seconds)} tries in {sum(seconds.values()):.1f} s")
    return 1 if any(r["ok"] is False for r in rows) else 2 if any(r["ok"] is None for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
