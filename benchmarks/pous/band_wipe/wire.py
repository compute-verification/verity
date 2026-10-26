"""What the band-fill wipe's prover and memory challenger share: the scheme, the binding a session signs, the cheat's
pattern, and the control channel.

The control channel is JSON lines over one TCP connection, the prover listening on its own 127.0.0.1 (the memory
challenger reaches it through an SSH tunnel):

~~~text
prover -> {"op": "hello", "segments", "params", "scheme", "device", ...}
loop:  challenger -> {"op": "fill", "wipe": j, "salt": hex(24 bytes), "cheat": bool}
       prover     -> {"op": "filled", "port", "fill", "gate", "cheat"}   or {"op": "refused", "reason"}
       (the memory challenger's sessions, on the responder's port)
       challenger -> {"op": "done"}
       prover     -> {"op": "report", ...}
challenger -> {"op": "bye"}
~~~
"""
from __future__ import annotations

import hashlib
import json
import socket
from typing import Any

SCHEME = "band-chain/d12/v1"
"""The band at in-degree 12, `band.SECURE_FIRST`'s geometry: 512 blocks of 64 KiB per 32 MiB segment."""
BINDING_TAG = b"verity-pous/band-wipe/binding/v1\0"
DROP_EVERY = 19
"""The cheat keeps foreign bytes in block `i` when `i % 19 == 18`: `1/19` of `C`, the guarantee's `f`.  Blocks 19 apart
leave every dropped block's 12 parents kept, so one chain of 13 `Π₂` calls rebuilds it."""


def binding(scheme: str, manifest: dict[str, Any], salt: bytes) -> bytes:
    """The 64 bytes a session signs as what it audits: `SHA-512(BINDING_TAG ‖ scheme ‖ 0 ‖ manifest ‖ 0 ‖ salt)`, the
    manifest as sorted JSON.  With `W = 0`, the scheme, its geometry and the salt fix every block of `C`, so this names `C`
    without the setup root of all its blocks (which only a full encode of `C` would give)."""
    return hashlib.sha512(BINDING_TAG + scheme.encode() + b"\0" + json.dumps(manifest, sort_keys=True).encode() + b"\0"
                          + salt).digest()


def dropped(i: int) -> bool:
    return i % DROP_EVERY == DROP_EVERY - 1


def reference_blocks(params: Any, salt: bytes, s: int, count: int) -> list[bytes]:
    """Blocks `0 .. count - 1` of segment `s` of `C` at `W = 0`, by `verity.core.protocols.pous`'s reference encoder
    (`dense.encode_segment_prog` over the band's parents, Feistel-SHAKE256 `Π₂`)."""
    from verity.core.protocols.pous import band
    from verity.core.protocols.pous.dense import encode_segment_prog
    from verity.core.protocols.pous.labels import salt_value, tag
    from verity.core.protocols.pous.oracle import run

    prims = band.reference(params)
    c, _ = run(encode_segment_prog(params, prims.h, tag(salt_value(salt), s), [0] * params.blocks_per_segment, count=count),
               prims)
    return [x.to_bytes(params.block_bytes, "little") for x in c]


def segment_digests(job: tuple[Any, bytes, int, list[int]]) -> tuple[int, dict[int, bytes]]:
    """`(params, salt, s, vs)`: `memory_challenger.block_digest` of blocks `vs` of segment `s` (a process pool's job)."""
    from verity.core.protocols.pous.memory_challenger import block_digest

    params, salt, s, vs = job
    blocks = reference_blocks(params, salt, s, max(vs) + 1)
    return s, {v: block_digest(blocks[v]) for v in vs}


class Channel:
    """One JSON object a line, each way."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.reader = sock.makefile("rb")

    def send(self, obj: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def recv(self) -> dict[str, Any]:
        line = self.reader.readline()
        if not line:
            raise ConnectionError("the control channel closed")
        return json.loads(line)

    def close(self) -> None:
        self.reader.close()
        self.sock.close()
