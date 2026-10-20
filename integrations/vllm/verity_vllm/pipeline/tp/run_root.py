"""The TP run root: one SHA-256 over the world and every rank's run root, the identity a TP Commit's ranks commit under."""
from __future__ import annotations

import hashlib
import struct

TP_ROOT_TAG = b"verity/tp-run-root/v1"


def tp_run_root(world: int, roots: dict[int, bytes]) -> bytes:
    if sorted(roots) != list(range(world)):
        raise ValueError(f"roots for ranks {sorted(roots)} != 0..{world - 1} (fail-closed: missing rank)")
    h = hashlib.sha256(TP_ROOT_TAG + struct.pack(">I", world))
    for r in range(world):
        if len(roots[r]) != 32:
            raise ValueError(f"rank {r} root is not 32 bytes")
        h.update(struct.pack(">I", r) + roots[r])
    return h.digest()
