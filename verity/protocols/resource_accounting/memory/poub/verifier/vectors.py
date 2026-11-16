"""PoUB's pinned vectors (`vectors.json` beside this file) and the payload blocks they are over, for code that checks a
device against them: a GPU receipter fills its store with `payload`'s blocks and holds its words to `PATH`'s.

`tests/poub_vectors.py` writes the file and states everything else it pins; `tests/test_poub_vectors.py` holds this
package to it."""

from __future__ import annotations

import hashlib
from pathlib import Path

from verity.protocols.resource_accounting.memory.poub.verifier import codec

PATH = Path(__file__).with_name("vectors.json")
PAYLOAD_DOMAIN = b"verity/remote-bandwidth/test-payload/v0\0"


def payload(j: int) -> bytes:
    """Payload block `j`: all zeros at 0, else 64-byte pieces, piece u the first 64 bytes of
    `SHA512(PAYLOAD_DOMAIN ‖ u64be j ‖ u32be u)` with its top byte cleared, so every word is below p."""
    if j == 0:
        return bytes(codec.BLOCK_BYTES)
    pieces = (hashlib.sha512(PAYLOAD_DOMAIN + j.to_bytes(8, "big") + u.to_bytes(4, "big")).digest()[:63] + b"\0"
              for u in range(codec.BLOCK_BYTES // 64))
    return b"".join(pieces)
