"""The TP run root, the identity a TP Commit's ranks commit under: one SHA-512 over the world and every rank's 64-byte run root
(`tp_run_root`, tag ``verity/tp-run-root/v2``), each rank committed under rows-sha512 (`commit.step_rows`).

    tp_run_root = SHA-512("verity/tp-run-root/v2" || u32 world || (u32 rank_r || root_r) for r in 0..world-1)

A 32-byte rank root is a vllm-v1 (SHA-256) root and is refused by name.  `tp_run_root_sha256` is the SHA-256 form over 32-byte
vllm-v1 rank roots (``verity/tp-run-root/v1``), only to reread the records committed before v2 (a TP Commit refuses
``--commit-scheme vllm-v1``); `combine` takes the form a record's ``root_tag`` names."""
from __future__ import annotations

import hashlib
import struct

TP_ROOT_TAG = b"verity/tp-run-root/v2"
TP_ROOT_TAG_V1 = b"verity/tp-run-root/v1"
RANK_ROOT_BYTES = 64


def _ranks(world: int, roots: dict[int, bytes]) -> None:
    if sorted(roots) != list(range(world)):
        raise ValueError(f"roots for ranks {sorted(roots)} != 0..{world - 1} (fail-closed: missing rank)")


def tp_run_root(world: int, roots: dict[int, bytes]) -> bytes:
    _ranks(world, roots)
    h = hashlib.sha512(TP_ROOT_TAG + struct.pack(">I", world))
    for r in range(world):
        if len(roots[r]) == 32:
            raise ValueError(f"rank {r} root is 32 bytes, a vllm-v1 (SHA-256) root: {TP_ROOT_TAG.decode()} takes rows-sha512's "
                             f"{RANK_ROOT_BYTES}-byte rank roots")
        if len(roots[r]) != RANK_ROOT_BYTES:
            raise ValueError(f"rank {r} root is {len(roots[r])} bytes, not {RANK_ROOT_BYTES}")
        h.update(struct.pack(">I", r) + roots[r])
    return h.digest()


def tp_run_root_sha256(world: int, roots: dict[int, bytes]) -> bytes:
    _ranks(world, roots)
    h = hashlib.sha256(TP_ROOT_TAG_V1 + struct.pack(">I", world))
    for r in range(world):
        if len(roots[r]) != 32:
            raise ValueError(f"rank {r} root is not 32 bytes")
        h.update(struct.pack(">I", r) + roots[r])
    return h.digest()


FORMS = {TP_ROOT_TAG.decode(): tp_run_root, TP_ROOT_TAG_V1.decode(): tp_run_root_sha256}


def combine(root_tag: str | bytes | None, world: int, roots: dict[int, bytes]) -> bytes:
    """The TP run root of `roots` under the form `root_tag` names (a record's ``versions.json`` ``root_tag``)."""
    tag = root_tag.decode() if isinstance(root_tag, bytes) else root_tag
    if tag not in FORMS:
        raise ValueError(f"root tag {tag!r} is not a TP run root form ({', '.join(FORMS)})")
    return FORMS[tag](world, roots)
