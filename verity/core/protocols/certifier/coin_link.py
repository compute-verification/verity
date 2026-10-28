"""The network certifier's coin link (D7 of `Security/Definitions/NetworkCertifier/Guarantees.lean`, "The coins"): the one
path a window's live coins take from the challenger to the server it audits, counted and re-clocked on the certifier's
own grids.

The challenger (`sampled_proofs.remote`, served by `benchmarks/live_challenger/serve.py`) and this link run on one auditor
host (`trusted-host`), the device that runs the server's network certifier. The link takes each connection from the
prover's side (the prover's client, on the server), opens one to the challenger for it, and carries `remote`'s frames
both ways, one connection at a time, in the challenger's order:

- **Cells.** A frame (u32be len(head) ‖ u32be 0 ‖ head) is re-cut into whole cells of B = `cell_bytes` bytes: its head is
  padded with spaces, which JSON reads as nothing, to the least length that makes the frame whole cells (`recut`). So a
  frame's cells are a function of its head alone, never of how its bytes arrived.
- **Grids.** The challenger's frames go to the prover on the `coins` grid, the coin link's ingress, at r_coin =
  `coin_cells` cells a bucket of τ = `bucket_ms`; the prover's frames go to the challenger on `coin-requests`, egress, at
  `request_cells`. Each is a `grid.Certifier`: constant-rate, left-packed and FIFO. A cell leaves the link only at its
  bucket's tick, and so do a direction's end and a connection's close, so what either side sees of the other's timing is
  the grid's occupancy, T·log2(r + 1) bits a link-window (the K charge's unit). Nothing the prover does moves a tick or
  slows the link: a prover that doesn't take what a tick releases fails the link closed, so there is no flow control from
  inside. The ticks read one clock, this host's monotonic one, which the challenger's release log reads too.
- **Content (RP5).** Each challenger frame answers the prover's oldest unanswered frame (`remote.serve` answers in order).
  It is exempt, a `COIN`, only when that frame asked for coin (label, index) after `after` and the answer is an opening
  of that coin that opens against the session's committed coin root (`live_reason` against `commitment`, the one the
  challenger's signed record states), it is that family's next coin to be exempt, so a family's exempt coins are a
  prefix of its coin stream, each once, and its head is that opening's canonical encoding (`canonical`), byte for byte.
  Every other challenger frame is counted, under the name of what it is (`COUNTED`), and every prover frame is egress,
  counted.
- **Fail closed.** A frame `remote` would refuse whole (a body, a head past `remote.MAX_HEAD`, a stream that ends inside
  one), a queue past `queue_cells`, a prover that leaves more than `backlog` bytes untaken, cells still queued when their
  connection ends, and a tick later than `slack` each fault a grid (`grid.Reason`), which cuts the connection at that
  tick: nothing leaves the link uncertified, and the fault is the link-window's status.

`CoinLink.summary` is what the link publishes for its window: its parameters, each link-window's status, the coins'
frames and bytes, every counted frame's name and bytes, and the charge. Each bucket's occupancy and every frame's content
stay witness. `protocol/coin-link.md` has the premises. The module imports only the standard library and `verity`.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any

from verity.core.protocols.certifier.grid import Certifier, GridParams, Reason, BlockGrid, row_count
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, live_reason
from verity.core.protocols.sampled_proofs.remote import HEADER, MAX_HEAD
from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512

PARAMETERS_TAG = "verity-network-coin-link/parameters/v1"
#: the link's two grids: the challenger's frames to the prover (ingress), the prover's to the challenger (egress)
COINS, REQUESTS = "coins", "coin-requests"
#: an exempt challenger frame: a verified, expected, once-only coin opening
COIN = "coin"
#: why a challenger frame is counted: what it is (the session's commitment, the close's answer, a refusal, a frame of
#: another type, one that isn't a JSON object with a type), or why an opening isn't a coin (no open asked for it, it
#: doesn't open against the committed root for what was asked, its bytes aren't its canonical encoding, or its family's
#: coins before it weren't all exempt: an earlier coin again, or a later one first)
COUNTED = ("commitment", "closed", "refused", "other", "malformed", "unrequested", "unverified", "non-canonical",
           "replayed", "out-of-order")
#: the most a side of the link may leave untaken before the link fails closed
BACKLOG = 1 << 20
#: an open's `after` the classifier reads: lowercase hex of whole bytes
_HEX = re.compile(r"(?:[0-9a-f]{2})*")


@dataclass(frozen=True)
class CoinLinkParams:
    """The coin link's public parameters, fixed before the window (`digest` binds them). The defaults are D7's: one
    4 KiB cell a bucket of 100 ms each way, link-windows of T = 600 buckets."""

    cell_bytes: int = 4096  # B
    coin_cells: int = 1  # r_coin
    request_cells: int = 1  # the egress grid's r
    bucket_ms: int = 100  # τ_in
    window: int = 600  # T
    queue_cells: int = 1024  # Q, each grid's

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} is a positive integer, not {value!r}")
        if self.cell_bytes <= HEADER.size:
            raise ValueError(f"a cell holds a frame's {HEADER.size}-byte header and more")

    @property
    def coins_grid(self) -> GridParams:
        return GridParams(self.window, self.coin_cells, self.cell_bytes, self.bucket_ms, self.queue_cells)

    @property
    def requests_grid(self) -> GridParams:
        return GridParams(self.window, self.request_cells, self.cell_bytes, self.bucket_ms, self.queue_cells)

    @cached_property
    def digest(self) -> bytes:
        return tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(asdict(self) | {"counted": list(COUNTED)}))


def recut(head: bytes, cell_bytes: int) -> list[bytes]:
    """The frame of `head` as whole cells of `cell_bytes`: the head padded with spaces to the least length that makes the
    frame a multiple of `cell_bytes`; ValueError when that is past `remote.MAX_HEAD`, which the reader would refuse."""
    size = len(head) + (-(HEADER.size + len(head)) % cell_bytes)
    if size > MAX_HEAD:
        raise ValueError(f"a head of {len(head)} bytes is past {MAX_HEAD} once padded to whole cells")
    data = HEADER.pack(size, 0) + head + b" " * (size - len(head))
    return [data[i:i + cell_bytes] for i in range(0, len(data), cell_bytes)]


async def read_frame(reader: asyncio.StreamReader) -> bytes | None:
    """The head of the next frame on `reader`; None at the end of the stream between frames; ValueError for what `remote`
    refuses whole: a body, a head past `MAX_HEAD`, or a stream that ends inside a frame."""
    try:
        header = await reader.readexactly(HEADER.size)
    except asyncio.IncompleteReadError as e:
        if e.partial:
            raise ValueError("the stream ended inside a frame's header") from None
        return None
    nh, nb = HEADER.unpack(header)
    if nh > MAX_HEAD or nb:
        raise ValueError(f"a frame is a head of at most {MAX_HEAD} bytes and no body, not {nh} + {nb}")
    try:
        return await reader.readexactly(nh)
    except asyncio.IncompleteReadError:
        raise ValueError("the stream ended inside a frame's head") from None


def canonical(head: dict) -> bytes:
    """`head` as `remote`'s wire writes it."""
    return json.dumps(head, sort_keys=True, separators=(",", ":")).encode()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("a head repeats a key")
    return dict(pairs)


def _object(raw: bytes) -> dict | str:
    """`raw` as a JSON object with a string `type`, or why it isn't one."""
    try:
        head = json.loads(raw, object_pairs_hook=_pairs)
    except (ValueError, RecursionError) as e:
        return f"not JSON: {type(e).__name__}"
    return head if type(head) is dict and type(head.get("type")) is str else "not an object with a string type"


class Classifier:
    """RP5: which challenger frames are coins. `ask` records each frame the prover sends, in order, and `answer` names the
    challenger frame that answers the oldest unanswered one: `COIN`, or one of `COUNTED`. `begin` starts a connection, a
    conversation of its own with the challenger; each family's next exempt coin carries over, so a coin is exempt once."""

    def __init__(self, commitment: LiveCommitment) -> None:
        self.commitment = commitment
        self.next = {label: 0 for label, _ in commitment.families}
        self._asked: deque[tuple[str, int, bytes] | None] = deque()

    def begin(self) -> None:
        self._asked.clear()

    def ask(self, raw: bytes) -> None:
        head, asked = _object(raw), None
        if type(head) is dict and head["type"] == "open" and sorted(head) == ["after", "index", "label", "type"]:
            label, index, after = head["label"], head["index"], head["after"]
            if type(label) is str and type(index) is int and type(after) is str and _HEX.fullmatch(after):
                asked = (label, index, bytes.fromhex(after))
        self._asked.append(asked)

    def answer(self, raw: bytes) -> str:
        asked = self._asked.popleft() if self._asked else None
        head = _object(raw)
        if type(head) is not dict:
            return "malformed"
        if head["type"] in ("commitment", "closed", "refused"):
            return head["type"]
        if head["type"] != "opening":
            return "other"
        if asked is None:
            return "unrequested"
        try:
            opening = LiveOpening.from_json(head.get("opening"))
        except (CoinError, ValueError, TypeError):
            return "malformed"
        label, index, after = asked
        try:
            why = live_reason(self.commitment, opening, label, index, after)
        except CoinError as e:
            why = str(e)
        if why is not None:
            return "unverified"
        if raw != canonical({"type": "opening", "opening": opening.to_json()}):
            return "non-canonical"
        n = self.next[label]
        if index != n:
            return "replayed" if index < n else "out-of-order"
        self.next[label] = n + 1
        return COIN


@dataclass(eq=False)
class _Connection:
    ordinal: int
    reader: asyncio.StreamReader
    writer: Any
    done: asyncio.Future
    challenger: tuple[Any, Any] | None = None
    unreachable: str | None = None  # why the challenger's connection didn't open
    prover_ended: bool = False  # the prover's end of stream read, every frame before it queued
    challenger_ended: bool = False
    up_ended: bool = False  # the prover's end, sent on to the challenger
    down_ended: bool = False  # the challenger's end, sent on to the prover
    requests: int = 0
    answers: int = 0
    errors: dict[str, str] = field(default_factory=dict)  # the OSError that ended a side's stream, by side
    tasks: list[asyncio.Task] = field(default_factory=list)


def _status(record: BlockGrid) -> str | None:
    return None if record.status.complete else record.status.reason.value


class CoinLink:
    """The coin link (the module docstring): `handle(reader, writer)` serves one connection from the prover's side, after
    every one before it; `run()` ticks both grids from `origin`, the clock's time of bucket 0, until `stop()` or
    `buckets`, then on to the end of that link-window, and cuts whatever is still open. `connect()` opens a connection to
    the challenger, a (reader, writer) pair.

    - `windows` has each link-window's two statuses as it closes, and `publish(record, params)`, when given, receives its
      two records, coins then requests;
    - `log` has one entry a connection once it ends;
    - `until(predicate)` waits until the link's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, params: CoinLinkParams, commitment: LiveCommitment,
                 connect: Callable[[], Awaitable[tuple[Any, Any]]], *, origin: float,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 slack: float = 0.025, buckets: int | None = None, backlog: int = BACKLOG,
                 publish: Callable[[BlockGrid, GridParams], None] | None = None) -> None:
        self.params, self.connect = params, connect
        self.origin, self.clock, self.sleep, self.slack, self.buckets = origin, clock, sleep, slack, buckets
        self.backlog, self.publish = backlog, publish
        self.coins = Certifier(COINS, params.coins_grid)
        self.requests = Certifier(REQUESTS, params.requests_grid)
        self.classifier = Classifier(commitment)
        self.windows: list[dict] = []
        self.log: list[dict] = []
        self.frames: Counter[str] = Counter()
        self.frame_bytes: Counter[str] = Counter()
        self.request_frames = self.request_bytes = 0
        self.cells = {COINS: 0, REQUESTS: 0}
        self.late, self.max_late = 0, 0.0
        self.ticks = self.connections = 0
        self.current: _Connection | None = None
        self.stopping = self.closed = False
        self._serial = asyncio.Lock()
        self._changed = asyncio.Event()

    def stop(self) -> None:
        self.stopping = True
        self._changed.set()

    async def until(self, predicate: Callable[[CoinLink], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        async with self._serial:
            if self.closed:
                writer.close()
                return
            self.connections += 1
            c = _Connection(self.connections, reader, writer, asyncio.get_running_loop().create_future())
            self.classifier.begin()
            self.current = c
            try:
                c.challenger = await self.connect()
            except OSError as e:
                c.unreachable = f"{type(e).__name__}: {e}"
            else:
                if c.done.done():
                    c.challenger[1].close()
                else:
                    c.tasks = [asyncio.create_task(self._prover(c)), asyncio.create_task(self._challenger(c))]
            self._changed.set()
            await c.done

    async def run(self) -> None:
        tau, T = self.params.bucket_ms / 1000, self.params.window
        t = 0
        try:
            while not ((self.stopping or (self.buckets is not None and t >= self.buckets)) and t % T == 0):
                deadline = self.origin + t * tau
                await self.sleep(deadline - self.clock())
                late = self.clock() - deadline
                self.max_late = max(self.max_late, late)
                if late > self.slack:
                    self.late += 1
                    for grid in (self.coins, self.requests):
                        grid.fault(Reason.MONITOR_UNAVAILABLE)
                self._tick(t)
                t = self.ticks = t + 1
                self._changed.set()
        finally:
            self.closed = True
            if self.current is not None:
                self._end(self.current, cut=True)
            self._changed.set()

    async def _prover(self, c: _Connection) -> None:
        """The prover's frames, each asked of the classifier and re-cut onto the requests grid, until its end of stream."""
        try:
            while (head := await read_frame(c.reader)) is not None:
                cells = recut(head, self.params.cell_bytes)
                self.classifier.ask(head)
                c.requests += 1
                self.request_frames += 1
                self.request_bytes += HEADER.size + len(head)
                for cell in cells:
                    self.requests.enqueue(cell)
                self._changed.set()
        except ValueError:
            self.requests.fault(Reason.MALFORMED)
        except OSError as e:
            c.errors["prover"] = type(e).__name__
        finally:
            c.prover_ended = True
            self._changed.set()

    async def _challenger(self, c: _Connection) -> None:
        """The challenger's frames, each named by the classifier and re-cut onto the coins grid, until its end of stream."""
        try:
            while (head := await read_frame(c.challenger[0])) is not None:
                cells = recut(head, self.params.cell_bytes)
                name = self.classifier.answer(head)
                c.answers += 1
                self.frames[name] += 1
                self.frame_bytes[name] += HEADER.size + len(head)
                for cell in cells:
                    self.coins.enqueue(cell)
                self._changed.set()
        except ValueError:
            self.coins.fault(Reason.MALFORMED)
        except OSError as e:
            c.errors["challenger"] = type(e).__name__
        finally:
            c.challenger_ended = True
            self._changed.set()

    def _tick(self, t: int) -> None:
        c = self.current
        coins, requests = self.coins.tick(), self.requests.tick()
        n_in, n_out = row_count(coins), row_count(requests)
        self.cells[COINS] += n_in
        self.cells[REQUESTS] += n_out
        if c is not None and c.unreachable is not None:
            self._end(c)
        elif c is not None and c.challenger is not None:
            if n_out:
                self._send(c.challenger[1], b"".join(requests[:n_out]), self.requests)
            if n_in:
                self._send(c.writer, b"".join(coins[:n_in]), self.coins)
            if c.prover_ended and not self.requests.queued and not c.up_ended:
                c.up_ended = True
                self._eof(c.challenger[1])
            if c.challenger_ended and not self.coins.queued and not c.down_ended:
                c.down_ended = True
                self._eof(c.writer)
            if self.coins.faulted or self.requests.faulted:
                self._end(c, cut=True)
            elif c.up_ended and c.down_ended:
                self._end(c)
        if (t + 1) % self.params.window == 0:
            records = (self.coins.close_window(), self.requests.close_window())
            if self.publish is not None:
                self.publish(records[0], self.params.coins_grid)
                self.publish(records[1], self.params.requests_grid)
            self.windows.append({"window": records[0].window, COINS: _status(records[0]),
                                 REQUESTS: _status(records[1])})

    def _send(self, writer: Any, data: bytes, grid: Certifier) -> None:
        if writer.is_closing() or writer.transport.get_write_buffer_size() > self.backlog:
            grid.fault(Reason.DELIVERY_REFUSED)
            return
        writer.write(data)

    @staticmethod
    def _eof(writer: Any) -> None:
        if writer.is_closing():
            return
        if writer.can_write_eof():
            writer.write_eof()
        else:
            writer.close()

    def _end(self, c: _Connection, cut: bool = False) -> None:
        for task in c.tasks:
            task.cancel()
        dropped = {COINS: self.coins.queued, REQUESTS: self.requests.queued}
        for grid in (self.coins, self.requests):
            if grid.queued:
                grid.fault(Reason.DROPPED)
        if c.challenger is not None:
            c.challenger[1].close()
        c.writer.close()
        if not c.done.done():
            c.done.set_result(None)
        self.current = None
        self.log.append({"ordinal": c.ordinal, "requests": c.requests, "answers": c.answers, "cut": cut,
                         "unreachable": c.unreachable, "errors": dict(c.errors), "dropped": dropped})
        self._changed.set()

    def summary(self) -> dict:
        """What the link publishes for its window: its parameters and their digest, the buckets it ticked, each
        link-window's statuses, whether every one is complete with no cell dropped, the coins' frames and bytes, every
        counted challenger frame's count and bytes by name, the prover's frames and bytes, the cells each grid released,
        each connection's entry, the late ticks, and the charge in bits: each grid's timing, T·log2(r + 1) a link-window,
        and its counted content, 8 bits a counted byte (the coins' content is 0)."""
        p = self.params
        counted = {name: {"frames": self.frames[name], "bytes": self.frame_bytes[name]}
                   for name in COUNTED if self.frames[name]}
        complete = (bool(self.windows) and all(w[COINS] is None and w[REQUESTS] is None for w in self.windows)
                    and not any(any(e["dropped"].values()) for e in self.log))
        return {"parameters": asdict(p), "digest": p.digest.hex(), "buckets": self.ticks, "windows": self.windows,
                "complete": complete, "coins": {"frames": self.frames[COIN], "bytes": self.frame_bytes[COIN]},
                "counted": counted, "requests": {"frames": self.request_frames, "bytes": self.request_bytes},
                "cells": dict(self.cells), "connections": self.log,
                "late": {"ticks": self.late, "max_s": round(self.max_late, 6), "slack_s": self.slack},
                "charge_bits": {COINS: {"timing": self.ticks * math.log2(p.coin_cells + 1),
                                        "content": 8 * sum(v["bytes"] for v in counted.values())},
                                REQUESTS: {"timing": self.ticks * math.log2(p.request_cells + 1),
                                           "content": 8 * self.request_bytes}}}
