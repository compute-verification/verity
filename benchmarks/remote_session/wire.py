"""The remote session's wire: framed messages between the verifier's process and the prover's, over one TCP stream.

~~~text
frame     u32be len(head) ‖ u32be len(body) ‖ head ‖ body: head a JSON object with its "type", body raw bytes
opening   u32be pass ‖ u64be position ‖ block (4096) ‖ u16be len(leaf) ‖ leaf ‖ leaf salt (192) ‖ payload salt (192)
          ‖ u8 receipt path's length ‖ its digests (64 each) ‖ u8 payload path's ‖ its digests: no tile (bind off)
~~~

Every byte each side sends and receives is counted (`Conn.sent`, `Conn.received`, by message type). Decoding an
opening the prover sent never raises past `decode_openings`, which returns None for a body it cannot read, so a
malformed body reaches `Session.verify` as no openings and rejects as `malformed`.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections import Counter

from verity.core.protocols.remote_bandwidth import audit, coins, receipts
from verity.core.protocols.remote_bandwidth import codec as rcodec

HEADER = struct.Struct(">II")
MAX_HEAD = 1 << 20
MAX_BODY = 1 << 31


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

    def send(self, head: dict, body: bytes = b"") -> None:
        h = json.dumps(head, separators=(",", ":")).encode()
        frame = HEADER.pack(len(h), len(body)) + h + body
        with self._lock:
            self.sock.sendall(frame)
            self.sent[head["type"]] += len(frame)

    def _exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.sock.recv(min(n - len(out), 1 << 22))
            if not chunk:
                raise Closed("the stream closed")
            out += chunk
        return bytes(out)

    def recv(self) -> tuple[dict, bytes, int]:
        nh, nb = HEADER.unpack(self._exact(HEADER.size))
        if nh > MAX_HEAD or nb > MAX_BODY:
            raise ValueError(f"a frame of {nh} + {nb} bytes is over the limit")
        head = json.loads(self._exact(nh))
        body = self._exact(nb) if nb else b""
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


def connect(host: str, port: int, attempts: int = 600) -> Conn:
    """Connect, retrying while the verifier is not yet listening (each retry waits on the refused connect itself)."""
    for k in range(attempts):
        try:
            return Conn(socket.create_connection((host, port)))
        except ConnectionRefusedError:
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
            "digest": r.digest.hex()}


def registration_of(d: dict) -> audit.Registration:
    r = audit.Registration(bytes.fromhex(d["node"]), d["blocks"], bytes.fromhex(d["payload_root"]),
                           bytes.fromhex(d["codec_seed"]), codec_named(d["codec"]), d["receipt"], d["bind"])
    if r.digest.hex() != d["digest"]:
        raise ValueError("the registration's digest is not its fields'")
    return r


def commitment_of(d: dict) -> coins.CoinCommitment:
    return coins.CoinCommitment(bytes.fromhex(d["registration"]), d["session"], d["count"], bytes.fromhex(d["root"]))


def coin_json(c: coins.CoinOpening) -> dict:
    return {"index": c.index, "coin": c.coin.hex(), "salt": c.salt.hex(), "path": [x.hex() for x in c.path]}


def coin_of(d: dict) -> coins.CoinOpening:
    return coins.CoinOpening(d["index"], bytes.fromhex(d["coin"]), bytes.fromhex(d["salt"]),
                             tuple(bytes.fromhex(x) for x in d["path"]))


_OPEN_HEAD = struct.Struct(">IQ")
DIGEST = receipts.DIGEST_BYTES
SALT = receipts.SALT_BYTES
BLOCK = rcodec.BLOCK_BYTES


def encode_opening(o: audit.Opening) -> bytes:
    if o.tile:
        raise ValueError("the wire carries no tile evidence: the registration does not bind z")
    return b"".join((_OPEN_HEAD.pack(o.pass_index, o.position), o.block, struct.pack(">H", len(o.leaf)), o.leaf,
                     o.leaf_salt, o.payload_salt, bytes((len(o.receipt_path),)), *o.receipt_path,
                     bytes((len(o.payload_path),)), *o.payload_path))


def encode_openings(openings) -> bytes:
    return struct.pack(">I", len(openings)) + b"".join(encode_opening(o) for o in openings)


def decode_openings(body: bytes) -> list[audit.Opening] | None:
    try:
        (n,), at, out = struct.unpack_from(">I", body), 4, []
        for _ in range(n):
            i, p = _OPEN_HEAD.unpack_from(body, at)
            at += _OPEN_HEAD.size
            block = body[at:at + BLOCK]
            at += BLOCK
            (nl,) = struct.unpack_from(">H", body, at)
            at += 2
            leaf = body[at:at + nl]
            at += nl
            leaf_salt, payload_salt = body[at:at + SALT], body[at + SALT:at + 2 * SALT]
            at += 2 * SALT
            paths = []
            for _ in range(2):
                d = body[at]
                at += 1
                paths.append(tuple(body[at + DIGEST * k:at + DIGEST * (k + 1)] for k in range(d)))
                at += DIGEST * d
            if at > len(body):
                return None
            out.append(audit.Opening(i, p, block, leaf, leaf_salt, paths[0], payload_salt, paths[1], b""))
        return out if at == len(body) else None
    except (struct.error, IndexError):
        return None
