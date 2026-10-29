"""The remote session's wire: framed messages between the verifier's process and the prover's, over one TCP stream.

~~~text
frame     u32be len(head) ‖ u32be len(body) ‖ head ‖ body: head a JSON object with its "type", body raw bytes
~~~

An `openings` frame's body is the remote audit's openings frame (`verity/core/protocols/poub/wire.py`),
which decoding never raises past: a body it cannot read leaves its window no openings, which rejects as `malformed`.
Every byte each side sends and receives is counted (`Conn.sent`, `Conn.received`, by message type).

The prover's side (`connect`) reads in a process of its own (`Conn.read_apart`, `READER`): it takes each frame off the
stream and stamps it as its last byte arrives, on the host's monotonic clock, then hands it to `recv` through a pipe.
A boundary's stamp, which the prover's boundary rule compares with a pass's start, is then the frame's arrival
whatever holds the prover's GIL; only how soon the session acts on it still waits on that.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import Counter

from verity.core.protocols.poub import audit, coins
from verity.core.protocols.poub import codec as rcodec

HEADER = struct.Struct(">II")
MAX_HEAD = 1 << 20
MAX_BODY = 1 << 31
STAMP = struct.Struct(">Q")

READER = f"""
import os, select, struct, sys, time
sock, out = int(sys.argv[1]), int(sys.argv[2])
watch = select.poll()
watch.register(sock, select.POLLIN)
watch.register(out, 0)
def exact(n):
    got = bytearray()
    while len(got) < n:
        if any(fd == out for fd, _ in watch.poll()):
            sys.exit(0)
        chunk = os.read(sock, min(n - len(got), 1 << 22))
        if not chunk:
            sys.exit(0)
        got += chunk
    return got
def put(rec):
    rec = memoryview(rec)
    while rec:
        rec = rec[os.write(out, rec):]
while True:
    head = exact(8)
    nh, nb = struct.unpack(">II", head)
    if nh > {MAX_HEAD} or nb > {MAX_BODY}:
        put(struct.pack(">Q", time.monotonic_ns()) + head)
        sys.exit(0)
    frame = head + exact(nh) + (exact(nb) if nb else b"")
    put(struct.pack(">Q", time.monotonic_ns()) + frame)
"""
"""`Conn.read_apart`'s reader: argv the stream's descriptor and the pipe's write end. Each record is the frame's arrival
(u64be monotonic ns) and the frame as it came; a frame over the limits goes on as its stamp and header alone, which
`recv` refuses. It exits when the stream closes or the pipe's read end does (the prover's process gone)."""


class Closed(ConnectionError):
    """The other side closed the stream."""


class Conn:
    """One side of the stream: `send` and `recv` frames, each counted. `recv` returns the frame and the monotonic
    nanosecond at which its last byte was read."""

    def __init__(self, sock: socket.socket, clock=time.monotonic_ns) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock, self.clock = sock, clock
        self.sent: Counter = Counter()
        self.received: Counter = Counter()
        self._lock = threading.Lock()
        self.reader: subprocess.Popen | None = None
        self._pipe: int | None = None

    def read_apart(self) -> None:
        """Read in a process of the reader's own (`READER`), whose stamps are on the host's monotonic clock: `recv`
        then reads its records from a pipe, and nothing in this process reads the stream again."""
        if self.clock is not time.monotonic_ns:
            raise ValueError("a reader apart stamps on time.monotonic_ns, not this side's clock")
        r, w = os.pipe()
        fd = self.sock.fileno()
        self.reader = subprocess.Popen([sys.executable, "-I", "-c", READER, str(fd), str(w)], pass_fds=(fd, w),
                                       stdin=subprocess.DEVNULL)
        os.close(w)
        self._pipe = r

    def send(self, head: dict, body: bytes = b"") -> None:
        h = json.dumps(head, separators=(",", ":")).encode()
        frame = HEADER.pack(len(h), len(body)) + h + body
        with self._lock:
            self.sock.sendall(frame)
            self.sent[head["type"]] += len(frame)

    def _exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            k = min(n - len(out), 1 << 22)
            chunk = self.sock.recv(k) if self._pipe is None else os.read(self._pipe, k)
            if not chunk:
                raise Closed("the stream closed")
            out += chunk
        return bytes(out)

    def recv(self) -> tuple[dict, bytes, int]:
        stamp = None if self._pipe is None else STAMP.unpack(self._exact(STAMP.size))[0]
        nh, nb = HEADER.unpack(self._exact(HEADER.size))
        if nh > MAX_HEAD or nb > MAX_BODY:
            raise ValueError(f"a frame of {nh} + {nb} bytes is over the limit")
        head = json.loads(self._exact(nh))
        body = self._exact(nb) if nb else b""
        if stamp is None:
            stamp = self.clock()
        if type(head) is not dict or type(head.get("type")) is not str:
            raise ValueError("a frame's head is an object with a type")
        self.received[head["type"]] += HEADER.size + nh + nb
        return head, body, stamp

    def totals(self) -> dict:
        return {"sent": dict(self.sent), "received": dict(self.received), "sent_bytes": sum(self.sent.values()),
                "received_bytes": sum(self.received.values())}


def listen(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


def connect(host: str, port: int, attempts: int = 600, apart: bool = True) -> Conn:
    """Connect, retrying while the verifier is not yet listening (each retry waits on the refused connect itself); the
    prover's side, reading in a process of its own unless not `apart`."""
    for k in range(attempts):
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.settimeout(None)
            conn = Conn(sock)
            if apart:
                conn.read_apart()
            return conn
        except (ConnectionRefusedError, TimeoutError):
            if k + 1 == attempts:
                raise
            threading.Event().wait(0.1)
    raise AssertionError


def codec_named(name: str) -> rcodec.Codec:
    for c in (rcodec.DEFAULT, rcodec.FALLBACK, rcodec.Codec(rcodec.P512, 2, False)):
        if c.name == name:
            return c
    field, k, *masked = name.split("/")
    return rcodec.Codec(rcodec.FIELDS[field], int(k[1:]), masked == ["masked"])


def registration_json(r: audit.Registration) -> dict:
    return {"node": r.node.hex(), "blocks": r.blocks, "payload_root": r.payload_root.hex(),
            "codec_seed": r.codec_seed.hex(), "codec": r.codec.name, "receipt": r.receipt, "bind": r.bind,
            "layout": r.layout, "rows": r.rows, "digest": r.digest.hex()}


def registration_of(d: dict) -> audit.Registration:
    r = audit.Registration(bytes.fromhex(d["node"]), d["blocks"], bytes.fromhex(d["payload_root"]),
                           bytes.fromhex(d["codec_seed"]), codec_named(d["codec"]), d["receipt"], d["bind"],
                           d["layout"], d["rows"])
    if r.digest.hex() != d["digest"]:
        raise ValueError("the registration's digest is not its fields'")
    return r


def commitment_of(d: dict) -> coins.LiveCommitment:
    return coins.LiveCommitment.from_json(d)


def coin_json(c: coins.LiveOpening) -> dict:
    return c.to_json()


def coin_of(d: dict) -> coins.LiveOpening:
    return coins.LiveOpening.from_json(d)
