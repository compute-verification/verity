"""The aggregate's candidate maps f: Z/2^16 -> Z/2^16, as numpy references over all 2^16 inputs, and the AES S-box."""
from __future__ import annotations

import numpy as np

N = 1 << 16
M16 = N - 1
Z = np.arange(N, dtype=np.uint32)

#: f_mix's shifts and odd multipliers: x ^= x >> 8; x *= C1; x ^= x >> 7; x *= C2; x ^= x >> 9 (all mod 2^16)
MIX = (8, 0x88B5, 7, 0xDB2D, 9)


def mix(z, p=MIX):
    s1, c1, s2, c2, s3 = p
    x = np.asarray(z, dtype=np.uint32) & M16
    x ^= x >> s1
    x = (x * c1) & M16
    x ^= x >> s2
    x = (x * c2) & M16
    x ^= x >> s3
    return x


MIX3 = (8, 0x88B5, 7, 0xDB2D, 8, 0x9E37, 7)


def mix3(z, p=MIX3):
    """Three rounds: x ^= x >> s; x *= c, then a last xorshift."""
    x = np.asarray(z, dtype=np.uint32) & M16
    for s, c in zip(p[0:6:2], p[1:6:2]):
        x ^= x >> s
        x = (x * c) & M16
    x ^= x >> p[6]
    return x


MIX4 = (8, 0x88B5, 7, 0xDB2D, 8, 0x9E37, 7, 0xA3D5, 8)


def mix4(z, p=MIX4):
    """Four rounds: x ^= x >> s; x *= c, then a last xorshift."""
    x = np.asarray(z, dtype=np.uint32) & M16
    for s, c in zip(p[0:8:2], p[1:8:2]):
        x ^= x >> s
        x = (x * c) & M16
    x ^= x >> p[8]
    return x


def m32a(z):
    """murmur3's fmix32 on z zero-extended to 32 bits (its first xorshift is void there), low 16 bits: two IMADs."""
    x = np.asarray(z, dtype=np.uint64) & M16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & M16).astype(np.uint32)


def m32b(z):
    """Two 32-bit multiplies around one xorshift, high 16 bits: IMAD, SHF, LOP3, IMAD, SHF."""
    x = np.asarray(z, dtype=np.uint64) & M16
    x = (x * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    return (x >> 16).astype(np.uint32)


def mix1(z):
    """One multiply: x ^= x >> 8; x *= C1; x ^= x >> 8."""
    x = np.asarray(z, dtype=np.uint32) & M16
    x ^= x >> 8
    x = (x * MIX[1]) & M16
    x ^= x >> 8
    return x


def _aes_sbox() -> np.ndarray:
    def gmul(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            a = ((a << 1) ^ 0x11B) if a & 0x80 else a << 1
            b >>= 1
        return r & 0xFF

    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if gmul(a, b) == 1:
                inv[a] = b
                break
    s = []
    for a in range(256):
        x = inv[a]
        y = x
        for r in range(1, 5):
            y ^= ((x << r) | (x >> (8 - r))) & 0xFF
        s.append(y ^ 0x63)
    return np.array(s, dtype=np.uint32)


SBOX = _aes_sbox()
assert SBOX[0] == 0x63 and SBOX[1] == 0x7C and SBOX[0x53] == 0xED


def feistel(z, rounds=3, keys=(0x00, 0x5A, 0xC3, 0x96)):
    """`rounds` Feistel rounds on z's bytes with the AES S-box: (hi, lo) -> (lo, hi ^ S(lo ^ k_r))."""
    x = np.asarray(z, dtype=np.uint32) & M16
    hi, lo = x >> 8, x & 0xFF
    for r in range(rounds):
        hi, lo = lo, hi ^ SBOX[lo ^ keys[r]]
    return (hi << 8) | lo


def ident(z):
    return np.asarray(z, dtype=np.uint32) & M16


def mod8(z):
    return np.asarray(z, dtype=np.uint32) & 0xFF


def square(z):
    x = np.asarray(z, dtype=np.uint64) & M16
    return ((x * x) & M16).astype(np.uint32)


def tfun(z):
    """Klimov–Shamir's invertible T-function x + (x^2 OR 5) mod 2^16."""
    x = np.asarray(z, dtype=np.uint64) & M16
    return ((x + ((x * x) | 5)) & M16).astype(np.uint32)


_RAND = np.random.default_rng(20261028).permutation(N).astype(np.uint32)


def rand(z):
    return _RAND[np.asarray(z, dtype=np.uint32) & M16]


#: name -> (map, what it stands for)
CANDIDATES = {
    "ident": (ident, "z mod 2^16: linear"),
    "mod8": (mod8, "z mod 2^8 (b = 8), summed mod 2^16"),
    "square": (square, "z^2 mod 2^16: a quadratic polynomial"),
    "tfun": (tfun, "x + (x^2 | 5): an invertible T-function"),
    "mix1": (mix1, "xorshift-multiply, one multiply"),
    "mix": (mix, "xorshift-multiply, two multiplies (f_mix)"),
    "mix3": (mix3, "xorshift-multiply, three multiplies"),
    "mix4": (mix4, "xorshift-multiply, four multiplies"),
    "m32a": (m32a, "murmur3 fmix32 on 32-bit lanes, low half (two IMADs)"),
    "m32b": (m32b, "two 32-bit IMADs around one xorshift, high half"),
    "feistel3": (lambda z: feistel(z, 3), "3-round Feistel on bytes, AES S-box"),
    "feistel4": (lambda z: feistel(z, 4), "4-round Feistel on bytes, AES S-box"),
    "rand": (rand, "a uniformly random permutation (the ideal, a 128 KiB table)"),
}
