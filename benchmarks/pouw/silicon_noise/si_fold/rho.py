"""Reference models of the accumulate-path mixers (rho) SI-FOLD prices, bit for bit.

Each rho is one round of a standard primitive with the accumulator words absorbed into it, or one of two mixers of our
own kept as controls (`nv_arx`, `affine`). `selftest()` checks every standard round against its primitive's published
output (hashlib for SHA-512 and SHA3-256, RFC 8439 for ChaCha20, FIPS-197 for AES-128, the Ascon and Xoodoo reference
permutations through their known-answer values here), so the RTL in `rtl.py` is checked against these.
"""

from __future__ import annotations

import hashlib
import struct

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


def rotr(x: int, r: int, w: int) -> int:
    r %= w
    m = (1 << w) - 1
    return ((x >> r) | (x << (w - r))) & m


def rotl(x: int, r: int, w: int) -> int:
    return rotr(x, w - (r % w), w)


# SHA-512 (FIPS 180-4)
SHA512_K = [
    0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC, 0x3956C25BF348B538,
    0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118, 0xD807AA98A3030242, 0x12835B0145706FBE,
    0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2, 0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235,
    0xC19BF174CF692694, 0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5, 0x983E5152EE66DFAB,
    0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4, 0xC6E00BF33DA88FC2, 0xD5A79147930AA725,
    0x06CA6351E003826F, 0x142929670A0E6E70, 0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED,
    0x53380D139D95B3DF, 0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
    0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30, 0xD192E819D6EF5218,
    0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8, 0x19A4C116B8D2D0C8, 0x1E376C085141AB53,
    0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8, 0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373,
    0x682E6FF3D6B2B8A3, 0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
    0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B, 0xCA273ECEEA26619C,
    0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178, 0x06F067AA72176FBA, 0x0A637DC5A2C898A6,
    0x113F9804BEF90DAE, 0x1B710B35131C471B, 0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC,
    0x431D67C49C100D4C, 0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817,
]
SHA512_IV = [
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179,
]


def sha512_round(st: list[int], w: int, k: int) -> list[int]:
    a, b, c, d, e, f, g, h = st
    s1 = rotr(e, 14, 64) ^ rotr(e, 18, 64) ^ rotr(e, 41, 64)
    ch = (e & f) ^ (~e & g)
    t1 = (h + s1 + ch + k + w) & M64
    s0 = rotr(a, 28, 64) ^ rotr(a, 34, 64) ^ rotr(a, 39, 64)
    maj = (a & b) ^ (a & c) ^ (b & c)
    t2 = (s0 + maj) & M64
    return [(t1 + t2) & M64, a, b, c, (d + t1) & M64, e, f, g]


def sha512_compress(st: list[int], block: bytes) -> list[int]:
    w = list(struct.unpack(">16Q", block))
    for t in range(16, 80):
        s0 = rotr(w[t - 15], 1, 64) ^ rotr(w[t - 15], 8, 64) ^ (w[t - 15] >> 7)
        s1 = rotr(w[t - 2], 19, 64) ^ rotr(w[t - 2], 61, 64) ^ (w[t - 2] >> 6)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & M64)
    s = list(st)
    for t in range(80):
        s = sha512_round(s, w[t], SHA512_K[t])
    return [(x + y) & M64 for x, y in zip(st, s)]


# Keccak-p[b, n_r] (FIPS 202), lane width w = b / 25
KECCAK_RC64 = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
    0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
KECCAK_ROT = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
]  # KECCAK_ROT[x][y]


def keccak_round(lanes: list[int], rc: int, w: int) -> list[int]:
    """One Keccak-p round on 25 lanes of w bits, lanes[x + 5 y]; rc already truncated to w bits."""
    m = (1 << w) - 1
    a = [[lanes[x + 5 * y] for y in range(5)] for x in range(5)]
    c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
    d = [c[(x - 1) % 5] ^ rotl(c[(x + 1) % 5], 1, w) for x in range(5)]
    a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
    b = [[0] * 5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            b[y][(2 * x + 3 * y) % 5] = rotl(a[x][y], KECCAK_ROT[x][y], w)
    out = [0] * 25
    for x in range(5):
        for y in range(5):
            out[x + 5 * y] = (b[x][y] ^ ((~b[(x + 1) % 5][y] & m) & b[(x + 2) % 5][y])) & m
    out[0] ^= rc & m
    return out


def keccak_f1600(lanes: list[int]) -> list[int]:
    for r in range(24):
        lanes = keccak_round(lanes, KECCAK_RC64[r], 64)
    return lanes


def sha3_256(msg: bytes) -> bytes:
    rate = 136
    p = bytearray(msg) + b"\x06" + b"\x00" * ((-len(msg) - 2) % rate) + b"\x80"
    if len(p) % rate:
        raise AssertionError
    lanes = [0] * 25
    for off in range(0, len(p), rate):
        blk = p[off : off + rate]
        for i in range(rate // 8):
            lanes[i] ^= int.from_bytes(blk[8 * i : 8 * i + 8], "little")
        lanes = keccak_f1600(lanes)
    return b"".join(lanes[i].to_bytes(8, "little") for i in range(4))


# ChaCha (RFC 8439)
def chacha_qr(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    a = (a + b) & M32
    d = rotl(d ^ a, 16, 32)
    c = (c + d) & M32
    b = rotl(b ^ c, 12, 32)
    a = (a + b) & M32
    d = rotl(d ^ a, 8, 32)
    c = (c + d) & M32
    b = rotl(b ^ c, 7, 32)
    return a, b, c, d


def chacha_column_round(s: list[int]) -> list[int]:
    s = list(s)
    for i in range(4):
        s[i], s[4 + i], s[8 + i], s[12 + i] = chacha_qr(s[i], s[4 + i], s[8 + i], s[12 + i])
    return s


def chacha_diagonal_round(s: list[int]) -> list[int]:
    s = list(s)
    for a, b, c, d in ((0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
        s[a], s[b], s[c], s[d] = chacha_qr(s[a], s[b], s[c], s[d])
    return s


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    st = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
    st += list(struct.unpack("<8I", key)) + [counter] + list(struct.unpack("<3I", nonce))
    s = list(st)
    for _ in range(10):
        s = chacha_diagonal_round(chacha_column_round(s))
    return struct.pack("<16I", *[(x + y) & M32 for x, y in zip(s, st)])


# AES (FIPS-197)
def _aes_sbox() -> list[int]:
    sbox = [0] * 256
    p = q = 1
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ rotl(q, 1, 8) ^ rotl(q, 2, 8) ^ rotl(q, 3, 8) ^ rotl(q, 4, 8)
        sbox[p] = x ^ 0x63
        if p == 1:
            break
    sbox[0] = 0x63
    return sbox


AES_SBOX = _aes_sbox()


def _xt(b: int) -> int:
    return ((b << 1) ^ (0x1B if b & 0x80 else 0)) & 0xFF


def aes_round(state: bytes, rk: bytes, last: bool = False) -> bytes:
    """SubBytes, ShiftRows, MixColumns (unless last), AddRoundKey; state column-major as FIPS-197."""
    s = [AES_SBOX[b] for b in state]
    s = [s[(i + 4 * (i % 4)) % 16] for i in range(16)]
    if not last:
        out = []
        for c in range(4):
            a0, a1, a2, a3 = s[4 * c : 4 * c + 4]
            out += [
                _xt(a0) ^ _xt(a1) ^ a1 ^ a2 ^ a3,
                a0 ^ _xt(a1) ^ _xt(a2) ^ a2 ^ a3,
                a0 ^ a1 ^ _xt(a2) ^ _xt(a3) ^ a3,
                _xt(a0) ^ a0 ^ a1 ^ a2 ^ _xt(a3),
            ]
        s = out
    return bytes(x ^ k for x, k in zip(s, rk))


def aes128_encrypt(key: bytes, pt: bytes) -> bytes:
    rcon = [1, 2, 4, 8, 16, 32, 64, 128, 27, 54]
    w = [list(key[4 * i : 4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = [AES_SBOX[x] for x in t[1:] + t[:1]]
            t[0] ^= rcon[i // 4 - 1]
        w.append([x ^ y for x, y in zip(w[i - 4], t)])
    rks = [bytes(sum(w[4 * r : 4 * r + 4], [])) for r in range(11)]
    s = bytes(x ^ k for x, k in zip(pt, rks[0]))
    for r in range(1, 11):
        s = aes_round(s, rks[r], last=(r == 10))
    return s


# Ascon permutation (NIST SP 800-232), five 64-bit lanes
def ascon_round(x: list[int], c: int) -> list[int]:
    x0, x1, x2, x3, x4 = x
    x2 ^= c
    x0 ^= x4
    x4 ^= x3
    x2 ^= x1
    t0, t1, t2, t3, t4 = (~x0 & M64) & x1, (~x1 & M64) & x2, (~x2 & M64) & x3, (~x3 & M64) & x4, (~x4 & M64) & x0
    x0 ^= t1
    x1 ^= t2
    x2 ^= t3
    x3 ^= t4
    x4 ^= t0
    x1 ^= x0
    x0 ^= x4
    x3 ^= x2
    x2 = ~x2 & M64
    x0 ^= rotr(x0, 19, 64) ^ rotr(x0, 28, 64)
    x1 ^= rotr(x1, 61, 64) ^ rotr(x1, 39, 64)
    x2 ^= rotr(x2, 1, 64) ^ rotr(x2, 6, 64)
    x3 ^= rotr(x3, 10, 64) ^ rotr(x3, 17, 64)
    x4 ^= rotr(x4, 7, 64) ^ rotr(x4, 41, 64)
    return [x0, x1, x2, x3, x4]


def ascon_p12(x: list[int]) -> list[int]:
    for r in range(12):
        x = ascon_round(x, ((0xF - r) << 4) | r)
    return x


# Xoodoo (Daemen et al. 2018), 3 planes of 4 32-bit lanes, a[y][x]
XOODOO_RC = [0x58, 0x38, 0x3C0, 0xD0, 0x120, 0x14, 0x60, 0x2C, 0x380, 0xF0, 0x1A0, 0x12]


def xoodoo_round(a: list[list[int]], rc: int) -> list[list[int]]:
    a = [list(p) for p in a]
    p = [a[0][x] ^ a[1][x] ^ a[2][x] for x in range(4)]
    e = [rotl(p[(x - 1) % 4], 5, 32) ^ rotl(p[(x - 1) % 4], 14, 32) for x in range(4)]
    a = [[a[y][x] ^ e[x] for x in range(4)] for y in range(3)]
    a[1] = [a[1][(x - 1) % 4] for x in range(4)]
    a[2] = [rotl(v, 11, 32) for v in a[2]]
    a[0][0] ^= rc
    b = [[(~a[(y + 1) % 3][x] & M32) & a[(y + 2) % 3][x] for x in range(4)] for y in range(3)]
    a = [[a[y][x] ^ b[y][x] for x in range(4)] for y in range(3)]
    a[1] = [rotl(v, 1, 32) for v in a[1]]
    a[2] = [rotl(a[2][(x - 2) % 4], 8, 32) for x in range(4)]
    return a


# Speck64 and Simon64 (NSA 2013; ISO/IEC 29167-22 and -21), 32-bit words, one round key a round
def speck64_round(x: int, y: int, k: int) -> tuple[int, int]:
    x = ((rotr(x, 8, 32) + y) & M32) ^ k
    y = rotl(y, 3, 32) ^ x
    return x, y


def simon64_round(x: int, y: int, k: int) -> tuple[int, int]:
    f = (rotl(x, 1, 32) & rotl(x, 8, 32)) ^ rotl(x, 2, 32)
    return y ^ f ^ k, x


# Controls of our own (rank 5): a per-lane ARX mixer, and an affine mixer that a linear denoiser would pass through
def nv_arx(t: int, w: int) -> int:
    return rotl((t + w) & M32, 7, 32) ^ t


AFFINE_A = 0x9E3779B1


def affine(t: int, w: int) -> int:
    return (t * AFFINE_A + w) & M32


def selftest() -> list[tuple[str, bool]]:
    out = []
    for msg in (b"", b"abc", bytes(range(200))):
        p = bytearray(msg) + b"\x80" + b"\x00" * ((112 - len(msg) - 1) % 128) + (8 * len(msg)).to_bytes(16, "big")
        st = list(SHA512_IV)
        for off in range(0, len(p), 128):
            st = sha512_compress(st, bytes(p[off : off + 128]))
        out.append((f"sha512 rounds = hashlib, {len(msg)} B", b"".join(x.to_bytes(8, "big") for x in st) == hashlib.sha512(msg).digest()))
    for msg in (b"", b"abc", bytes(i & 0xFF for i in range(300))):
        out.append((f"keccak-f[1600] rounds = hashlib sha3_256, {len(msg)} B", sha3_256(msg) == hashlib.sha3_256(msg).digest()))
    key = bytes(range(32))
    nonce = bytes.fromhex("000000090000004a00000000")
    want = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
    )
    out.append(("chacha20 block = RFC 8439 2.3.2", chacha20_block(key, 1, nonce) == want))
    out.append((
        "aes-128 = FIPS-197 C.1",
        aes128_encrypt(bytes(range(16)), bytes.fromhex("00112233445566778899aabbccddeeff")).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a",
    ))
    # Ascon-Hash256's initial state (SP 800-232 / ascon-c): p12 of (IV, 0, 0, 0, 0)
    asc = ascon_p12([0x0000080100CC0002, 0, 0, 0, 0])
    out.append(("ascon p12 = Ascon-Hash256 initial state", asc == [0x9B1E5494E934D681, 0x4BC3A01E333751D2, 0xAE65396C6B34B81A, 0x3C7FD4A4D56A4DB3, 0x1A5C464906C5976D]))
    # Xoodoo[12] of the all-zero state, first lane (XKCP Xoodoo test)
    a = [[0] * 4 for _ in range(3)]
    for r in XOODOO_RC:
        a = xoodoo_round(a, r)
    out.append(("xoodoo[12](0) lane 0 = XKCP value", a[0][0] == 0x89D5D88D))
    # Speck64/128 and Simon64/128 test vectors (Beaulieu et al. 2013, appendix C)
    out.append(("speck64/128 = paper vector", _speck64_128() == (0x8C6FA548, 0x454E028B)))
    out.append(("simon64/128 = paper vector", _simon64_128() == (0x44C8FC20, 0xB9DFA07A)))
    return out


def _speck64_128() -> tuple[int, int]:
    l = [0x0B0A0908, 0x13121110, 0x1B1A1918]
    k = 0x03020100
    x, y = 0x3B726574, 0x7475432D
    for i in range(27):
        x, y = speck64_round(x, y, k)
        nl, k = speck64_round(l[i % 3], k, i)
        l[i % 3] = nl
    return x, y


def _simon64_128() -> tuple[int, int]:
    z3 = 0b11011011101011000110010111100000010010001010011100110100001111
    k = [0x03020100, 0x0B0A0908, 0x13121110, 0x1B1A1918]
    for i in range(4, 44):
        t = rotr(k[i - 1], 3, 32) ^ k[i - 3]
        t ^= rotr(t, 1, 32)
        k.append((~k[i - 4] & M32) ^ t ^ ((z3 >> ((61 - (i - 4)) % 62)) & 1) ^ 3)
    x, y = 0x656B696C, 0x20646E75
    for i in range(44):
        x, y = simon64_round(x, y, k[i])
    return x, y


if __name__ == "__main__":
    for name, ok in selftest():
        print(("ok  " if ok else "FAIL"), name)
