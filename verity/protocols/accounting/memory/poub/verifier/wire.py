"""The openings frame, `lean/Poub/Wire.lean`'s: the bytes a prover sends a window's openings in, decoded
strictly (`decode_openings`: None for a body that ends early, runs past its openings, or has a bound flag above 2), and a
window's openings from its frames in the order they came (`openings_of`, `Wire.openingsOf`): none at all if any frame
doesn't decode, so a malformed frame leaves the window no openings and it rejects as `malformed`. Every opening carries
no tile.

~~~text
body      u32be n ‖ opening × n
opening   u32be pass ‖ u64be position ‖ chunk (4096) ‖ u16be len(leaf) ‖ leaf ‖ leaf salt (192) ‖ payload salt (192)
          ‖ u8 d ‖ d × 64 (receipt path) ‖ u8 d ‖ d × 64 (payload path) ‖ u8 bound ‖ bound's tail
bound     0, nothing: a registered chunk; 1, d_j (64): a chunk appended after the codec seed; 2, d_j (64) ‖ u64be
          origin_position: a chunk of a re-key append (`Params.refresh`), its payload path its origin's leaf's
~~~
"""
from __future__ import annotations

import struct
from collections.abc import Iterable

from verity.protocols.accounting.memory.poub.verifier import audit, receipts
from verity.protocols.accounting.memory.poub.verifier import codec as rcodec

_OPEN_HEAD = struct.Struct(">IQ")
DIGEST = receipts.DIGEST_BYTES
SALT = receipts.SALT_BYTES
BLOCK = rcodec.BLOCK_BYTES


def encode_opening(o: audit.Opening) -> bytes:
    if o.tile:
        raise ValueError("the wire carries no tile evidence: the registration does not bind z")
    if len(o.digest) not in (0, DIGEST):
        raise ValueError(f"an opening's d_j is {DIGEST} bytes, or none for a registered block")
    if o.origin_position is None:
        bound = bytes((len(o.digest) // DIGEST,)) + o.digest
    elif len(o.digest) == DIGEST:
        bound = b"\2" + o.digest + struct.pack(">Q", o.origin_position)
    else:
        raise ValueError("a re-key's block carries its d_j beside its origin's position")
    return b"".join((_OPEN_HEAD.pack(o.pass_index, o.position), o.block, struct.pack(">H", len(o.leaf)), o.leaf,
                     o.leaf_salt, o.payload_salt, bytes((len(o.receipt_path),)), *o.receipt_path,
                     bytes((len(o.payload_path),)), *o.payload_path, bound))


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
            bound = body[at]
            if bound > 2:
                return None
            digest = body[at + 1:at + 1 + DIGEST * min(bound, 1)]
            at += 1 + DIGEST * min(bound, 1)
            origin = None
            if bound == 2:
                (origin,) = struct.unpack_from(">Q", body, at)
                at += 8
            if at > len(body):
                return None
            out.append(audit.Opening(i, p, block, leaf, leaf_salt, paths[0], payload_salt, paths[1], b"", digest,
                                     origin))
        return out if at == len(body) else None
    except (struct.error, IndexError):
        return None


def openings_of(frames: Iterable[bytes]) -> list[audit.Opening]:
    out = []
    for body in frames:
        got = decode_openings(body)
        if got is None:
            return []
        out.extend(got)
    return out


__all__ = ["encode_openings"]
