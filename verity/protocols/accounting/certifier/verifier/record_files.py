"""A run's files as a verifier reads them: a record packed, and a certificate, a certified record and the network
certifier's public key as JSON."""
from __future__ import annotations

import json
import struct
import zlib

from verity.commitments.crypto import lms
from verity.protocols.accounting.certifier.verifier.certificate import (Certificate, CertifiedRecord, Direction, LmsKey,
    RecordHead)
from verity.protocols.accounting.certifier.verifier.grid import BlockGrid, COMPLETE, Reason, Status, row_count
from verity.protocols.accounting.certifier.verifier.link_frame import HEADER, decode_frame, encode_frame


def pack_record(record: BlockGrid) -> bytes:
    allocation = len(record.grid[0]) if record.grid else 0
    head = json.dumps({"link": record.link, "window": record.window, "allocation": allocation,
                       "reason": record.status.reason.value if record.status.reason else None}).encode()
    parts = [struct.pack(">II", len(head), len(record.grid)), head]
    for row in record.grid:
        n = row_count(row)
        if any(slot is not None for slot in row[n:]):
            raise ValueError("a row that isn't left-packed")
        parts.append(struct.pack(">I", n))
        for slot in row[:n]:
            same = False
            if len(slot) >= HEADER.size:
                kind, stream, payload = decode_frame(slot)  # the payload is cut from the slot, so it fits its length
                same = encode_frame(kind, stream, payload, len(slot)) == slot
            if same:
                parts += [struct.pack(">BBII", 0, kind, stream, len(payload)), payload]
            else:
                parts += [struct.pack(">BI", 1, len(slot)), slot]
    return zlib.compress(b"".join(parts), 1)


def unpack_record(blob: bytes, frame_bytes: int, intern: dict[bytes, bytes] | None = None) -> BlockGrid:
    data = zlib.decompress(blob)
    head_len, rows = struct.unpack_from(">II", data)
    i = 8 + head_len
    head = json.loads(data[8:i])
    grid = []
    for _ in range(rows):
        (n,), i = struct.unpack_from(">I", data, i), i + 4
        slots = []
        for _ in range(n):
            if data[i] == 0:
                _, kind, stream, length = struct.unpack_from(">BBII", data, i)
                i += 10
                slot = encode_frame(kind, stream, data[i : i + length], frame_bytes)
            else:
                (length,) = struct.unpack_from(">I", data, i + 1)
                i += 5
                slot = data[i : i + length]
            i += length
            slots.append(intern.get(slot, slot) if intern is not None else slot)
        grid.append(tuple(slots) + (None,) * (head["allocation"] - n))
    status = Status(Reason(head["reason"])) if head["reason"] else COMPLETE
    return BlockGrid(head["link"], head["window"], tuple(grid), status)


_BYTES = {"parameters", "root", "key_id", "digest", "signature"}


def _enc(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Direction):
        return value.value
    if isinstance(value, Status):
        return None if value.reason is None else value.reason.value
    return list(value) if isinstance(value, tuple) else value


def _dec(field: str, value):
    if field in _BYTES:
        return bytes.fromhex(value)
    if field == "direction":
        return Direction(value)
    if field == "status":
        return COMPLETE if value is None else Status(Reason(value))
    return tuple(value) if field == "counts" else value


def certificate_json(c: Certificate) -> dict:
    return {f: _enc(getattr(c, f)) for f in Certificate.__slots__}


def certificate_from_json(doc: dict) -> Certificate:
    return Certificate(**{f: _dec(f, doc[f]) for f in Certificate.__slots__})


def certified_json(record: CertifiedRecord) -> dict:
    return {"salt_c": record.salt.hex(), "head": {f: _enc(getattr(record.head, f)) for f in RecordHead.__slots__},
            "certificate": certificate_json(record.certificate)}


def certified_from_json(doc: dict) -> CertifiedRecord:
    return CertifiedRecord(RecordHead(**{f: _dec(f, v) for f, v in doc["head"].items()}), bytes.fromhex(doc["salt_c"]),
                           certificate_from_json(doc["certificate"]))


def public_json(key: LmsKey) -> dict:
    return {"params": [key.params.N, key.params.W, key.params.H], "public": key.public.hex(), "key_id": key.key_id.hex()}


def read_public(doc: dict) -> LmsKey:
    return LmsKey(lms.Params(*doc["params"]), bytes.fromhex(doc["public"]))
