"""Halevi–Micali commitments on SHA-512 (`hm96-sha512/v1`, core PR #93's `verity.commitments.hm96.SHA512`).

A commitment to a byte string m is the hm96 tree leaf over the inner digest `x = SHA-512(prefix ‖ m)`, with `prefix` one
block naming what is committed, so a gadget starts from a constant midstate:

    x = SHA-512(prefix ‖ m),  c = SHA-512(salt_prefix ‖ y),  b = x ⊕ M·y,  HM(m; y) = SHA-512(leaf_prefix(key) ‖ b ‖ c)

with the 192-byte salt y from the operating system and the pinned key. Statistically hiding, binding under SHA-512
collision resistance (hm96 PROTOCOL.md §3, §4). `commit_circuit` is the same opening inside V[B]. This module mirrors the
SHA-512 instance until PR #93 is on main; `tests/test_recursion.py` pins it to PR #93's vectors.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

from . import gadgets
from .program import Program, Reg

N_BITS, SALT_BYTES = 512, 192
KEY_BITS = N_BITS + 8 * SALT_BYTES - 1
KEY_BYTES = (KEY_BITS + 7) // 8
SALT_TAG = b"verity/hm96-sha512/salt/v1\0"
LEAF_TAG = b"verity/hm96-sha512/leaf/v1\0"
KEY_LABEL = b"verity/hm96-sha512/key/v1\0"
BLOCK = 128


def _h(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


@lru_cache(maxsize=None)
def default_key() -> bytes:
    raw = b"".join(_h(KEY_LABEL + i.to_bytes(4, "big")) for i in range(-(-KEY_BYTES // 64)))[:KEY_BYTES]
    top = (1 << (KEY_BITS - 8 * (KEY_BYTES - 1))) - 1
    return raw[:-1] + bytes([raw[-1] & top])


def salt_prefix() -> bytes:
    return SALT_TAG.ljust(BLOCK, b"\0")


def leaf_prefix(key: bytes | None = None) -> bytes:
    return (LEAF_TAG + _h(key or default_key())).ljust(BLOCK, b"\0")


def inner_prefix(what: str) -> bytes:
    return f"verity/private-circuit/{what}/v1\0".encode().ljust(BLOCK, b"\0")


@lru_cache(maxsize=None)
def matrix(key: bytes | None = None) -> np.ndarray:
    """M (512 x 1536) over GF(2): M[i][j] = key bit i + j."""
    kb = np.unpackbits(np.frombuffer(key or default_key(), dtype=np.uint8), bitorder="little")
    i, j = np.meshgrid(np.arange(N_BITS), np.arange(8 * SALT_BYTES), indexing="ij")
    return kb[i + j].astype(np.uint8)


def mask(salt: bytes) -> bytes:
    y = np.unpackbits(np.frombuffer(salt, dtype=np.uint8), bitorder="little").astype(np.int32)
    return np.packbits((matrix().astype(np.int32) @ y) & 1, bitorder="little").tobytes()


def commit_string(inner_digest: bytes, salt: bytes) -> bytes:
    assert len(salt) == SALT_BYTES and len(inner_digest) == 64
    return bytes(a ^ b for a, b in zip(inner_digest, mask(salt))) + _h(salt_prefix() + salt)


def tree_leaf(proven: bytes) -> bytes:
    return _h(leaf_prefix() + proven)


def commit(what: str, m: bytes, salt: bytes) -> bytes:
    return tree_leaf(commit_string(_h(inner_prefix(what) + m), salt))


def fresh_salt() -> bytes:
    return os.urandom(SALT_BYTES)


def _midstate(block: bytes) -> np.ndarray:
    iv = np.frombuffer(gadgets.iv_bytes("sha512"), dtype=np.uint8)
    return gadgets.sha_native("sha512", iv, np.frombuffer(block, dtype=np.uint8)[None, :])


def commit_circuit(prog: Program, what: str, m: Reg, salt: Reg) -> Reg:
    """HM(m; salt) for byte registers m (..., L) and salt (..., 192): the 64-byte leaf register (..., 64)."""
    L = m.shape[-1]
    lead = m.shape[:-1]
    pad = np.frombuffer(gadgets.md_pad("sha512", BLOCK + L), dtype=np.uint8)
    blocks = prog.concat([m, prog.const("byte", np.broadcast_to(pad, lead + pad.shape).copy())], axis=len(lead))
    blocks = blocks.reshape(*lead, -1, BLOCK)
    x = prog.sha("sha512", prog.const("byte", _midstate(inner_prefix(what))), blocks)
    spad = np.frombuffer(gadgets.md_pad("sha512", BLOCK + SALT_BYTES), dtype=np.uint8)
    sblocks = prog.concat([salt, prog.const("byte", np.broadcast_to(spad, lead + spad.shape).copy())], axis=len(lead))
    c = prog.sha("sha512", prog.const("byte", _midstate(salt_prefix())), sblocks.reshape(*lead, -1, BLOCK))
    b = prog.bytes_(prog.bits(x) ^ prog.linmap(prog.bits(salt), matrix()))
    lpad = np.frombuffer(gadgets.md_pad("sha512", 2 * BLOCK), dtype=np.uint8)
    lblocks = prog.concat([b, c, prog.const("byte", np.broadcast_to(lpad, lead + lpad.shape).copy())], axis=len(lead))
    return prog.sha("sha512", prog.const("byte", _midstate(leaf_prefix())), lblocks.reshape(*lead, -1, BLOCK))
