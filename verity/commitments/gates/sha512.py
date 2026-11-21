"""SHA-512's compression function as a GF(2) circuit over `forms` (formerly Flock's `sha512_circuit`, which keeps its row
circuits and imports this), with its plain-integer reference. Its constants and gates are `verity.circuits.boolean.sha512`'s.

* `compress_ref`: one compression on word values (FIPS 180-4 §6.4.2), with the feed-forward.
* `compress_gadget`: the same over words of circuit bits, 58,120 ANDs in the ripple-carry model (80 rounds of 7 additions, Ch
  and Maj; 64 schedule words of 3 additions; 8 final additions), fewer where an operand is a constant (a constant message
  block's schedule is free). On a `forms.Circuit` it also commits copies (`commit`) that keep every row's linear forms short;
  traced (`trace.TracingCircuit`) a copy is the form itself, so a Boolean Definition built on it has exactly the ANDs.
* `Sha512CompressFolded`: `compress_gadget` as a Definition of its arguments' pattern (`trace.folded`), which a Definition
  traced with calls calls in place of each compression's gates.

Words are 64-bit values as little-endian bit lists (bit i of a word is its value's bit i); a digest's bytes are its eight words
big-endian.
"""
from __future__ import annotations

import struct

from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import Circuit, const_bits
from verity.circuits.boolean.sha512 import (CARRY_EVERY, IV, K, M64, add, big_sigma0, big_sigma1, ch, commit, maj,
                                            small_sigma0, small_sigma1)

__all__ = ["CARRY_EVERY", "IV", "K", "M64", "Sha512CompressFolded", "commit", "compress_gadget", "compress_ref", "midstate",
           "pad_tail", "words_be"]


# --- reference (plain integers) ------------------------------------------------------------------------------------------------

def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & M64


def compress_ref(cv: list[int], msg: list[int]) -> list[int]:
    """One SHA-512 compression on word values (FIPS 180-4 §6.4.2), with the feed-forward."""
    w = list(msg)
    for t in range(16, 80):
        s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ (w[t - 15] >> 7)
        s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ (w[t - 2] >> 6)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & M64)
    a, b, c, d, e, f, g, h = cv
    for t in range(80):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + ((e & f) ^ (~e & g)) + K[t] + w[t]) & M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(cv, [a, b, c, d, e, f, g, h])]


def words_be(block: bytes) -> list[int]:
    return list(struct.unpack(">%dQ" % (len(block) // 8), block))


def midstate(prefix: bytes) -> list[int]:
    """The chaining value after the 128-byte blocks of ``prefix`` (a constant of a scheme whose prefix is whole blocks)."""
    assert len(prefix) % 128 == 0
    cv = list(IV)
    for i in range(0, len(prefix), 128):
        cv = compress_ref(cv, words_be(prefix[i:i + 128]))
    return cv


def pad_tail(total_len: int) -> bytes:
    """SHA-512 padding after a message of ``total_len`` bytes (0x80, zeros, the 128-bit length)."""
    z = (-(total_len + 17)) % 128
    return b"\x80" + bytes(z) + (8 * total_len).to_bytes(16, "big")


# --- circuits ------------------------------------------------------------------------------------------------------------------

def compress_gadget(C: Circuit, cv: list[list], msg: list[list]) -> list[list]:
    """:func:`compress_ref` over words of circuit bits; the schedule words and each round's new ``a`` and ``e`` are committed.
    Traced with calls (`trace.CallTracingCircuit`), it is a call of `Sha512CompressFolded` instead."""
    fold = getattr(C, "fold_call", None)
    if fold is not None:
        out = fold(Sha512CompressFolded, [f for x in (*cv, *msg) for f in x])
        return [out[64 * i:64 * i + 64] for i in range(8)]
    return _compress(C, cv, msg)


def _compress(C: Circuit, cv: list[list], msg: list[list]) -> list[list]:
    w = list(msg)
    for t in range(16, 80):
        w.append(commit(C, add(C, w[t - 16], small_sigma0(w[t - 15]), w[t - 7], small_sigma1(w[t - 2]))))
    a, b, c, d, e, f, g, h = cv
    for t in range(80):
        t1 = add(C, h, big_sigma1(e), ch(C, e, f, g), const_bits(K[t], 64), w[t])
        t2 = add(C, big_sigma0(a), maj(C, a, b, c))
        h, g, f, e, d, c, b, a = g, f, e, commit(C, add(C, d, t1)), c, b, a, commit(C, add(C, t1, t2))
    return [add(C, x, y) for x, y in zip(cv, [a, b, c, d, e, f, g, h])]


def _compress_bits(C: Circuit, bits: list) -> list:
    words = [bits[64 * i:64 * i + 64] for i in range(24)]
    return [f for w in _compress(C, words[:8], words[8:]) for f in w]


Sha512CompressFolded = TR.folded(
    "Sha512CompressFolded", 1, _compress_bits, 1536, 512,
    doc="One SHA-512 compression (`compress_gadget`) at the argument pattern P (`trace.pattern_of`): the chaining value's 512 "
        "bits then the block's 1024, each a constant or the XOR of parameters, to the 512 bits of the next chaining value. The "
        "hashing Definitions traced with calls call it once per compression, so a pattern's circuit is written once.")
