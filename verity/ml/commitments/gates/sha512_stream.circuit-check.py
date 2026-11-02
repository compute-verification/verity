"""circuit-check's bindings of `verity.ml.commitments.gates.sha512_stream`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _sha512_stream_roots() -> list[Any]:
    """The streamed SHA-512 (`verity.ml.commitments.gates.sha512_stream`): a row at an odd phase (37 pending bytes and 48
    words: one block, 5 bytes left, the last carry word's pad byte), a row at phase 0 that ends on a block (64 words, no
    carry in or out), a row of no whole block (pure wiring), and the last row at one block with no carry (8 words of a
    16-byte stream), after an odd carry (37 bytes, then 48 words: a block of the row, then the end's), and with an end of
    two blocks (1 pending byte and 56 words: 113 bytes left before the padding). Each compression is `lms`'s
    `Sha512Compress_v1`, checked in its own module's bindings; a deployed row (512 words, eight blocks) repeats these
    layouts."""
    from verity.ml.commitments.gates.sha512_stream import last, row

    return [row(37, 48), row(0, 64), row(2, 3), last(0, 8, 16), last(37, 48, 133), last(1, 56, 113)]
