"""The handshake that opens every channel between two parties, before any protocol message. Protocols never see it; the
deployment layer runs it on each channel and keeps its frames in the party's transcript, as `structure`.

~~~text
each side   Hello {wire, party, peer, protocol, protocol_version, configuration, nonce, key}
each side   Signature, its body sign(signed(own name, the two hellos' frames in the order of their parties' names))
~~~

A side checks the other's hello against its own and refuses by name (`REFUSALS`): another wire version
(`handshake-version`), a party other than the peer it expects or a hello not addressed to it (`handshake-party`), another
protocol or protocol version (`handshake-protocol`), another configuration hash (`handshake-configuration`), a key other
than the one the configuration gives the peer (`handshake-key`), its own nonce sent back (`handshake-nonce`), and a
signature that doesn't verify (`handshake-signature`). Each side's signature covers both nonces and its own name, so
neither a recorded exchange nor the other side's signature passes for it. Signing and verifying are the caller's (a party
key, LMS in deployment, `Key`, `Verify`): this unit imports none. Soundness doesn't rest on the handshake; it binds a
channel to the configuration and to the parties' keys, as hygiene.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Protocol

from . import NAME, WIRE_VERSION, Body, Hex, Int, Message, Phase, Refused, Registry, Schema, Str

HELLO = Schema("Hello", 1, "structure", {"wire": Int(1, (1 << 16) - 1), "party": Str(64, NAME.pattern),
                                         "peer": Str(64, NAME.pattern), "protocol": Str(128),
                                         "protocol_version": Int(1, (1 << 31) - 1), "configuration": Hex(64),
                                         "nonce": Hex(32), "key": Hex(max=1 << 10)})
SIGNATURE = Schema("Signature", 1, "structure", body=Body(max=1 << 17))
REGISTRY = Registry([HELLO, SIGNATURE])
TAG = b"verity/wire/handshake/v1\0"
NONCE_BYTES = 32
REFUSALS = ("handshake-version", "handshake-party", "handshake-protocol", "handshake-configuration", "handshake-key",
            "handshake-nonce", "handshake-signature")


class Key(Protocol):
    """A party's key: its public half, and `sign(message)`, which a stateful hash-based scheme refuses past its last leaf."""

    public_key: bytes

    def sign(self, message: bytes) -> bytes: ...


type Verify = Callable[[bytes, bytes, bytes], bool]
"""`verify(public_key, message, signature)`."""


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(4, "big") + b


def signed(signer: str, first: bytes, second: bytes) -> bytes:
    """What `signer` signs: SHA-512(TAG ‖ signer ‖ first hello ‖ second hello), each length-prefixed (u32be), the hellos'
    frames in the order of their parties' names."""
    return hashlib.sha512(TAG + _field(signer.encode()) + _field(first) + _field(second)).digest()


class Handshake:
    """One side of the handshake on one channel: `start` its hello, `on` each frame the other side sends, until `done`.
    `peer_key` is the key the configuration gives the peer; None takes the key the peer's hello names (its signature
    still binds the exchange to it)."""

    def __init__(self, *, party: str, peer: str, protocol: str, version: int, configuration: bytes, nonce: bytes,
                 key: Key, verify: Verify, peer_key: bytes | None = None) -> None:
        if party == peer:
            raise ValueError(f"a channel joins two parties, not {party} to itself")
        if len(configuration) != 64 or len(nonce) != NONCE_BYTES:
            raise ValueError(f"a configuration hash is 64 bytes and a nonce {NONCE_BYTES}")
        self.party, self.peer, self.key, self.verify, self.peer_key = party, peer, key, verify, peer_key
        self.hello = Message("Hello", {"wire": WIRE_VERSION, "party": party, "peer": peer, "protocol": protocol,
                                       "protocol_version": version, "configuration": configuration.hex(),
                                       "nonce": nonce.hex(), "key": bytes(key.public_key).hex()})
        self.phase = Phase()
        self.frames: list[bytes] = []
        self.peer_hello: dict[str, Any] | None = None
        self.done = False

    def start(self) -> list[bytes]:
        if self.frames:
            raise RuntimeError("a handshake starts once")
        out = REGISTRY.encode(self.hello)
        self.frames.append(out)
        self.phase.expect("Hello")
        return [out]

    def _hellos(self) -> tuple[bytes, bytes]:
        mine, theirs = self.frames[0], self.frames[1]
        return (mine, theirs) if self.party < self.peer else (theirs, mine)

    def on(self, data: bytes) -> list[bytes]:
        if self.done or not self.frames:
            raise Refused("out-of-phase", "a frame outside the handshake's exchange")
        msg = self.phase.check(REGISTRY.decode(data))
        if msg.type == "Hello":
            self._check(msg.fields)
            self.frames.append(bytes(data))
            self.peer_hello = dict(msg.fields)
            sig = REGISTRY.encode(Message("Signature", body=bytes(self.key.sign(signed(self.party, *self._hellos())))))
            self.frames.append(sig)
            self.phase.expect("Signature")
            return [sig]
        key = bytes.fromhex(self.peer_hello["key"])
        try:
            ok = bool(self.verify(key, signed(self.peer, *self._hellos()), msg.body))
        except Exception as e:
            raise Refused("handshake-signature", f"{self.peer}'s signature over the exchange doesn't parse: "
                                                 f"{type(e).__name__}: {e}") from e
        if not ok:
            raise Refused("handshake-signature", f"{self.peer}'s signature over the exchange doesn't verify")
        self.frames.append(bytes(data))
        self.phase.expect()
        self.done = True
        return []

    def _check(self, h: Any) -> None:
        mine = self.hello.fields
        if h["wire"] != WIRE_VERSION:
            raise Refused("handshake-version", f"wire version {h['wire']}, not {WIRE_VERSION}")
        if h["party"] != self.peer or h["peer"] != self.party:
            raise Refused("handshake-party", f"a hello from {h['party']} to {h['peer']}, where {self.party} expects "
                                             f"{self.peer}")
        if (h["protocol"], h["protocol_version"]) != (mine["protocol"], mine["protocol_version"]):
            raise Refused("handshake-protocol", f"{h['protocol']} v{h['protocol_version']}, not "
                                                f"{mine['protocol']} v{mine['protocol_version']}")
        if h["configuration"] != mine["configuration"]:
            raise Refused("handshake-configuration", f"{self.peer} runs configuration {h['configuration'][:16]}…, not "
                                                     f"{mine['configuration'][:16]}…")
        if self.peer_key is not None and h["key"] != self.peer_key.hex():
            raise Refused("handshake-key", f"{self.peer}'s key is not the one the configuration gives it")
        if h["nonce"] == mine["nonce"]:
            raise Refused("handshake-nonce", "the peer's nonce is this side's own")


__all__ = ["HELLO", "NONCE_BYTES", "REFUSALS", "REGISTRY", "SIGNATURE", "TAG", "Handshake", "Key", "Verify", "signed"]
