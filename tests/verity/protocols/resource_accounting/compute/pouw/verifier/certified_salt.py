"""A network attester as PoUW reads it, for PoUW's tests: what `salt.revealed` and `salt.committed` take
(`reveal(p, registration)` and `committed(p, registration)`, 64 bytes each, or a `RuntimeError`), and what a certified
`served_commit.Challenger` takes beside them: the run its stream names (`certifier_run`: the key's id, a 32-byte run id
and the ingress link) and the key and run id its `Deployment` pins (`certifier_key`, `run_id`).

It holds the salts it sampled and the records the prover made with it before them, by PoUW execution phase, and gives
them back on any registration. It signs, times and serves nothing, so it holds none of the certifier's own rules (when a
salt is published, its reveal deadline, one reveal per execution phase, a record the prover signed): those are the
certifier device's to state and its tests'."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

#: the certifier run's id, fixed so a prover can name the run in the stream it records
RUN_ID = bytes(range(32))
KEY_ID = hashlib.sha512(b"the network certifier's key").digest()[:16]


@dataclass(frozen=True)
class Key:
    """The certifier's public key as a `Deployment` pins it: its id, and a `verify` no PoUW verifier calls."""
    key_id: bytes

    def verify(self, message: bytes, signature: bytes) -> bool:
        return False


def salts(windows: int, seed: bytes = b"") -> dict[int, bytes]:
    """The 64-byte salts a certifier samples for PoUW execution phases 0 … `windows` − 1, under `seed`."""
    return {p: hashlib.shake_256(b"certifier salt %d %s" % (p, seed)).digest(64) for p in range(windows)}


@dataclass
class Certifier:
    """A certifier run that sampled `salts` and holds `records` by PoUW execution phase."""
    salts: dict[int, bytes]
    records: dict[int, bytes] = field(default_factory=dict)
    key_id: bytes = KEY_ID
    run_id: bytes = RUN_ID
    link: str = "requests"

    @property
    def certifier_run(self) -> tuple[bytes, bytes, str]:
        return self.key_id, self.run_id, self.link

    @property
    def certifier_key(self) -> Key:
        return Key(self.key_id)

    def reveal(self, window: int, registration: Any) -> bytes:
        if window not in self.salts:
            raise RuntimeError(f"no salt was sampled for PoUW window {window}")
        return self.salts[window]

    def committed(self, window: int, registration: Any) -> bytes:
        if window not in self.records:
            raise RuntimeError(f"nothing was recorded for PoUW window {window} before its salt")
        return self.records[window]
