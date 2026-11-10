"""A KV block's key, the SHA-512 chain that names a cached block across windows, and its print, the 128-bit value a window's
memories hold in its place, as Boolean Definitions.

    BlockKey_v1{B, TOK}(delta: F, parent: Array<8, Word>, tokens: Array<B, Array<TOK, Value<1>>>) -> (Array<8, Word>, F)
    KeyPrint_v1(delta: F, key: Array<8, Word>) -> F
        Word = Array<64, Value<1>>, F = Array<128, Value<1>>; B, TOK >= 1, B·TOK a multiple of 8

**The key.** A block's key is SHA-512(parent ‖ tokens) (`block_key`): the parent is the key of the block before it in its
request, `ROOT` for a request's first, and its B tokens, TOK bits each, are packed most significant bit first into B·TOK/8
bytes. A key, like every SHA-512 digest here, is eight words as `merkle` lays them out (bytes big-endian, each word's bits
LSB first), and a token's bits are LSB first. Two blocks with one key hold the same tokens after the same parents back to
`ROOT` unless SHA-512 collides (``cr/sha-512``): a key names a block's whole prefix in 64 bytes, whichever window computed
it. At the deployed binding, B = 64 tokens of TOK = 18 bits (vocabularies to 2^18), the message is 208 bytes, two
compressions (`sha512.compress_gadget`, traced with the padding's constant bytes folded).

**The print.** Within a window the memories hold a key's print in its place (`key_print`): Σ_{i<4} δ^i c_i over GF(2^128)
(`gf128`), c_i the key's 128-bit chunks (words 2i and 2i + 1, the first the low half), δ the window's live coin, drawn after
its keys are committed; three products, by Horner's rule. Two distinct keys share a print with probability at most
3/2^128, so a window's K keys have distinct prints except with probability at most 3·K(K − 1)/2^129. BlockKey_v1 returns a
computed key with its print; KeyPrint_v1 prints a key carried in from an earlier window.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE, ZERO, const_bits
from verity.circuits.boolean.gf128 import mul, mul_forms
from verity.circuits.types import Array, Tuple

from verity.commitments.gates.sha512 import IV, compress_gadget, pad_tail

__all__ = ["BlockKey", "KeyPrint", "ROOT", "block_key", "chain", "key_print", "pack_tokens"]

BIT = BL.BIT
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
F = Array(128, BIT)
#: the parent of a request's first block
ROOT = hashlib.sha512(b"verity block key v1").digest()


def _check(S) -> None:
    for k in ("B", "TOK"):
        x = getattr(S, k)
        if not isinstance(x, int) or isinstance(x, bool) or x < 1:
            raise ValueError(f"BlockKey's {k} is {x!r}, not a positive int")
    if S.B * S.TOK % 8:
        raise ValueError(f"BlockKey's {S.B} tokens of {S.TOK} bits are not whole bytes")


# --- the references ----------------------------------------------------------------------------------------------------

def pack_tokens(tokens: Sequence[int], TOK: int) -> bytes:
    """The tokens, TOK bits each, most significant bit first."""
    x = 0
    for t in tokens:
        if not 0 <= t < 1 << TOK:
            raise ValueError(f"token {t} is not {TOK} bits")
        x = x << TOK | t
    if len(tokens) * TOK % 8:
        raise ValueError(f"{len(tokens)} tokens of {TOK} bits are not whole bytes")
    return x.to_bytes(len(tokens) * TOK // 8, "big")


def block_key(parent: bytes, tokens: Sequence[int], TOK: int) -> bytes:
    if len(parent) != 64:
        raise ValueError(f"a parent key is 64 bytes, not {len(parent)}")
    return hashlib.sha512(parent + pack_tokens(tokens, TOK)).digest()


def chain(tokens: Sequence[int], B: int, TOK: int, parent: bytes = ROOT) -> list[bytes]:
    """The keys of the full blocks of a request's tokens, each block's parent the one before it."""
    out = []
    for i in range(0, len(tokens) - B + 1, B):
        parent = block_key(parent, tokens[i:i + B], TOK)
        out.append(parent)
    return out


def key_print(key: bytes, delta: int) -> int:
    w = struct.unpack(">8Q", key)
    cs = [w[2 * i] | w[2 * i + 1] << 64 for i in range(4)]
    e = cs[3]
    for c in reversed(cs[:3]):
        e = mul(e, delta) ^ c
    return e


# --- the circuits ------------------------------------------------------------------------------------------------------

def key_forms(C, S, parent: Sequence[list], tokens: Sequence[list]) -> list[list]:
    """SHA-512(parent ‖ tokens) over words and tokens of forms: the message as one bit stream, most significant bit first,
    padded, then as words."""
    stream = [x for w in parent for x in reversed(w)] + [x for t in tokens for x in reversed(t)]
    stream += [ONE if byte >> (7 - i) & 1 else ZERO for byte in pad_tail(len(stream) // 8) for i in range(8)]
    msg = [stream[64 * k:64 * (k + 1)][::-1] for k in range(len(stream) // 64)]
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    return st


def print_forms(C, delta: list, key: Sequence[list]) -> list:
    cs = [list(key[2 * i]) + list(key[2 * i + 1]) for i in range(4)]
    e = cs[3]
    for c in reversed(cs[:3]):
        e = [x ^ y for x, y in zip(mul_forms(C, e, delta), c)]
    return e


def _sig_key(S):
    _check(S)
    return (("delta", F), ("parent", DIGEST), ("tokens", Array(S.B, Array(S.TOK, BIT)))), Tuple(DIGEST, F)


def _key(C, S, delta, *e):
    key = key_forms(C, S, e[:8], e[8:])
    return [*key, print_forms(C, delta, key)]


BlockKey = TR.part(
    "BlockKey", ("B", "TOK"), _sig_key, _key,
    "A KV block's key SHA-512(parent ‖ tokens), its B tokens of TOK bits packed most significant bit first, and the key's "
    "print under δ, Σ δ^i c_i over GF(2^128) on its four 128-bit chunks")

KeyPrint = TR.definition(
    "KeyPrint", 1, lambda C, delta, *key: print_forms(C, delta, key), (("delta", F), ("key", DIGEST)), F,
    doc="A block key's print under δ: Σ_{i<4} δ^i c_i over GF(2^128), c_i its words 2i and 2i + 1 (the first the low half), "
        "three products")
