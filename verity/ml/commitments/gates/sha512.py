"""SHA-512's compression function as a GF(2) circuit over `forms` (formerly C-Flock's `sha512_circuit`, which keeps its row
circuits and imports this), with its plain-integer reference.

* `compress_ref`: one compression on word values (FIPS 180-4 §6.4.2), with the feed-forward.
* `compress_gadget`: the same over words of circuit bits, 58,120 ANDs in the ripple-carry model (80 rounds of 7 additions, Ch
  and Maj; 64 schedule words of 3 additions; 8 final additions), fewer where an operand is a constant (a constant message
  block's schedule is free). On a `forms.Circuit` it also commits copies (`commit`) that keep every row's linear forms short;
  traced (`trace.TracingCircuit`) a copy is the form itself, so a Boolean Definition built on it has exactly the ANDs.

Words are 64-bit values as little-endian bit lists (bit i of a word is its value's bit i); a digest's bytes are its eight words
big-endian.
"""
from __future__ import annotations

import struct

from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, const_bits, maj

__all__ = ["CARRY_EVERY", "IV", "K", "M64", "commit", "compress_gadget", "compress_ref", "midstate", "pad_tail", "words_be"]

K = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
    0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
    0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec, 0x6c44198c4a475817]
IV = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
      0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]
M64 = (1 << 64) - 1


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

def _rot(x: list, n: int) -> list:
    return x[n:] + x[:n]


def _shr(x: list, n: int) -> list:
    return x[n:] + [ZERO] * n


def _xor3(a: list, b: list, c: list) -> list:
    return [p ^ q ^ r for p, q, r in zip(a, b, c)]


#: A ripple carry is c ^ AND(a ^ c, b ^ c), a form one term longer per bit; committing it every CARRY_EVERY bits keeps every
#: row's linear forms short (the rows' nonzeros, which a witness generator pays per compression) for a few thousand rows.
CARRY_EVERY = 5


def _add2(C: Circuit, a: list, b: list) -> list:
    out, c = [], ZERO
    for i in range(64):
        out.append(a[i] ^ b[i] ^ c)
        if i < 63:
            c = maj(C, a[i], b[i], c)
            if i % CARRY_EVERY == CARRY_EVERY - 1:
                c = commit(C, [c])[0]
    return out


def _add(C: Circuit, *xs: list) -> list:
    acc = xs[0]
    for x in xs[1:]:
        acc = _add2(C, acc, x)
    return acc


def commit(C: Circuit, word: list) -> list:
    """The word as committed bits (one row each, ``A = form, B = 1``): a sum's XOR form would otherwise compound round over
    round."""
    return [w if (len(w.s) == 1 and w.c == 0) or w.is_const() else C._new("and", (w, ONE)) for w in word]


def compress_gadget(C: Circuit, cv: list[list], msg: list[list]) -> list[list]:
    """:func:`compress_ref` over words of circuit bits; the schedule words and each round's new ``a`` and ``e`` are committed."""
    w = list(msg)
    for t in range(16, 80):
        s0 = _xor3(_rot(w[t - 15], 1), _rot(w[t - 15], 8), _shr(w[t - 15], 7))
        s1 = _xor3(_rot(w[t - 2], 19), _rot(w[t - 2], 61), _shr(w[t - 2], 6))
        w.append(commit(C, _add(C, w[t - 16], s0, w[t - 7], s1)))
    a, b, c, d, e, f, g, h = cv
    for t in range(80):
        s1 = _xor3(_rot(e, 14), _rot(e, 18), _rot(e, 41))
        ch = [gg ^ C.AND(ee, ff ^ gg) for ee, ff, gg in zip(e, f, g)]
        t1 = _add(C, h, s1, ch, const_bits(K[t], 64), w[t])
        s0 = _xor3(_rot(a, 28), _rot(a, 34), _rot(a, 39))
        mj = [maj(C, x, y, z) for x, y, z in zip(a, b, c)]
        t2 = _add(C, s0, mj)
        h, g, f, e, d, c, b, a = g, f, e, commit(C, _add(C, d, t1)), c, b, a, commit(C, _add(C, t1, t2))
    return [_add(C, x, y) for x, y in zip(cv, [a, b, c, d, e, f, g, h])]
