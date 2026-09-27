"""The Boolean gadgets V[B]'s tables are made of, their AND counts, and a bit-sliced evaluator.

- SHA-256 and SHA-512 compression (FIPS 180-4) with one-AND-per-bit adders, Ch and Maj;
- GF(2^128) multiplication (PROTOCOL.md §2.1): Karatsuba down to single bits (3^7 ANDs), then the XOR-only reduction.

`gate_check` runs sampled instances of an evaluated operation through its gadget circuit, in bit-sliced form, and compares
with the native values (`program.evaluate(..., gate_sample=k)`): the gate-level check that the program's native semantics
are the circuit's.
"""
from __future__ import annotations

import hashlib
import struct
from functools import lru_cache

import numpy as np

from . import gf

SHA_STATE = {"sha256": 32, "sha512": 64}
SHA_BLOCK = {"sha256": 64, "sha512": 128}


class Circ:
    """Gates over wires; wires 0 and 1 are the constants 0 and 1."""

    def __init__(self):
        self.n = 2
        self.gates: list[tuple[int, int, int, int]] = []  # (op 0=xor 1=and, out, a, b)
        self.inputs: list[int] = []
        self.outputs: list[int] = []

    def inp(self, k: int) -> list[int]:
        w = list(range(self.n, self.n + k))
        self.n += k
        self.inputs += w
        return w

    def xor(self, a: int, b: int) -> int:
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return 0
        o = self.n
        self.n += 1
        self.gates.append((0, o, a, b))
        return o

    def and_(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        o = self.n
        self.n += 1
        self.gates.append((1, o, a, b))
        return o

    def not_(self, a: int) -> int:
        return self.xor(a, 1)

    @property
    def ands(self) -> int:
        return sum(1 for g in self.gates if g[0] == 1)

    # word helpers (LSB first)
    def wxor(self, *ws):
        out = ws[0]
        for w in ws[1:]:
            out = [self.xor(a, b) for a, b in zip(out, w)]
        return out

    def add(self, a, b):
        c, out = 0, []
        for i, (x, y) in enumerate(zip(a, b)):
            out.append(self.xor(self.xor(x, y), c))
            if i + 1 < len(a):
                c = self.xor(c, self.and_(self.xor(x, c), self.xor(y, c)))
        return out

    def const_word(self, v: int, bits: int):
        return [1 if (v >> i) & 1 else 0 for i in range(bits)]


def _rotr(w, r):
    return w[r:] + w[:r]


def _shr(w, r):
    return w[r:] + [0] * r


_K256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
_K512 = [
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
IV256 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
IV512 = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1,
         0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]
_SHA_PARAMS = {
    "sha256": dict(bits=32, rounds=64, K=_K256, S0=(2, 13, 22), S1=(6, 11, 25), s0=(7, 18, 3), s1=(17, 19, 10)),
    "sha512": dict(bits=64, rounds=80, K=_K512, S0=(28, 34, 39), S1=(14, 18, 41), s0=(1, 8, 7), s1=(19, 61, 6)),
}


def iv_bytes(algo: str) -> bytes:
    return b"".join(w.to_bytes(4 if algo == "sha256" else 8, "big") for w in (IV256 if algo == "sha256" else IV512))


@lru_cache(maxsize=None)
def sha_circuit(algo: str) -> Circ:
    """Inputs: state bytes (big-endian words) then block bytes, each byte LSB first; outputs: the new state bytes."""
    P = _SHA_PARAMS[algo]
    w, H, Bk = P["bits"], SHA_STATE[algo], SHA_BLOCK[algo]
    c = Circ()
    st_bits, blk_bits = c.inp(8 * H), c.inp(8 * Bk)

    def words(bits, count):
        out = []
        for i in range(count):
            byts = [bits[8 * (i * (w // 8) + j):8 * (i * (w // 8) + j) + 8] for j in range(w // 8)]
            out.append([b for byte in reversed(byts) for b in byte])  # big-endian words, LSB first
        return out

    V = words(st_bits, 8)
    W = words(blk_bits, 16)
    S0, S1, s0, s1 = P["S0"], P["S1"], P["s0"], P["s1"]
    for t in range(16, P["rounds"]):
        x = W[t - 15]
        y = W[t - 2]
        sig0 = c.wxor(_rotr(x, s0[0]), _rotr(x, s0[1]), _shr(x, s0[2]))
        sig1 = c.wxor(_rotr(y, s1[0]), _rotr(y, s1[1]), _shr(y, s1[2]))
        W.append(c.add(c.add(sig1, W[t - 7]), c.add(sig0, W[t - 16])))
    a, b, cc, d, e, f, g, h = V
    for t in range(P["rounds"]):
        Sig1 = c.wxor(_rotr(e, S1[0]), _rotr(e, S1[1]), _rotr(e, S1[2]))
        ch = [c.xor(gg, c.and_(ee, c.xor(ff, gg))) for ee, ff, gg in zip(e, f, g)]
        T1 = c.add(c.add(h, Sig1), c.add(ch, c.add(c.const_word(P["K"][t], w), W[t])))
        Sig0 = c.wxor(_rotr(a, S0[0]), _rotr(a, S0[1]), _rotr(a, S0[2]))
        maj = [c.xor(x, c.and_(c.xor(x, y), c.xor(x, z))) for x, y, z in zip(a, b, cc)]
        T2 = c.add(Sig0, maj)
        h, g, f, e, d, cc, b, a = g, f, e, c.add(d, T1), cc, b, a, c.add(T1, T2)
    new = [c.add(x, y) for x, y in zip(V, [a, b, cc, d, e, f, g, h])]
    for word in new:
        for j in reversed(range(w // 8)):
            c.outputs += word[8 * j:8 * j + 8]
    return c


def _karatsuba(c: Circ, a: list[int], b: list[int]) -> list[int]:
    n = len(a)
    if n == 1:
        return [c.and_(a[0], b[0])]
    h = n // 2
    lo = _karatsuba(c, a[:h], b[:h])
    hi = _karatsuba(c, a[h:], b[h:])
    mid = _karatsuba(c, [c.xor(x, y) for x, y in zip(a[:h], a[h:])], [c.xor(x, y) for x, y in zip(b[:h], b[h:])])
    out = [0] * (2 * n - 1)
    for i, x in enumerate(lo):
        out[i] = c.xor(out[i], x)
        out[i + h] = c.xor(out[i + h], x)
    for i, x in enumerate(hi):
        out[i + 2 * h] = c.xor(out[i + 2 * h], x)
        out[i + h] = c.xor(out[i + h], x)
    for i, x in enumerate(mid):
        out[i + h] = c.xor(out[i + h], x)
    return out


@lru_cache(maxsize=None)
def gf_mul_circuit() -> Circ:
    """Inputs a (128 bits), b (128 bits), bit i = coefficient of x^i; output the 128 bits of a·b mod x^128+x^7+x^2+x+1."""
    c = Circ()
    a, b = c.inp(128), c.inp(128)
    p = _karatsuba(c, a, b)
    p += [0] * (256 - len(p))
    for i in range(255, 127, -1):
        for t in (0, 1, 2, 7):
            p[i - 128 + t] = c.xor(p[i - 128 + t], p[i])
        p[i] = 0
    c.outputs = p[:128]
    return c


GF_MUL_ANDS = 3 ** 7


@lru_cache(maxsize=None)
def sha_ands(algo: str) -> int:
    return sha_circuit(algo).ands


# ---------------------------------------------------------------------------------------------------------------------
# bit-sliced evaluation
# ---------------------------------------------------------------------------------------------------------------------

def run_sliced(c: Circ, inputs: np.ndarray) -> np.ndarray:
    """inputs (len(c.inputs), N) of 0/1 -> outputs (len(c.outputs), N), evaluating every gate on packed words."""
    N = inputs.shape[1]
    packed = np.packbits(np.ascontiguousarray(inputs, dtype=np.uint8), axis=1, bitorder="little")
    words = (packed.shape[1] + 7) // 8
    packed = np.ascontiguousarray(np.pad(packed, ((0, 0), (0, words * 8 - packed.shape[1])))).view(np.uint64)
    w = np.zeros((c.n, words), dtype=np.uint64)
    w[1] = ~np.uint64(0)
    w[c.inputs] = packed
    for op, o, a, b in c.gates:
        w[o] = (w[a] & w[b]) if op else (w[a] ^ w[b])
    out = np.ascontiguousarray(w[c.outputs]).view(np.uint8)
    return np.unpackbits(out, axis=1, bitorder="little")[:, :N]


# ---------------------------------------------------------------------------------------------------------------------
# native SHA compressions (vectorized)
# ---------------------------------------------------------------------------------------------------------------------

def _compress_words(algo: str, V: np.ndarray, W16: np.ndarray) -> np.ndarray:
    P = _SHA_PARAMS[algo]
    bits = P["bits"]
    dt = np.uint32 if bits == 32 else np.uint64
    B = dt(bits)
    rot = lambda x, r: (x >> dt(r)) | (x << dt(bits - r))
    W = [W16[..., i] for i in range(16)]
    s0, s1, S0, S1 = P["s0"], P["s1"], P["S0"], P["S1"]
    for t in range(16, P["rounds"]):
        x, y = W[t - 15], W[t - 2]
        W.append(W[t - 16] + (rot(x, s0[0]) ^ rot(x, s0[1]) ^ (x >> dt(s0[2]))) + W[t - 7]
                 + (rot(y, s1[0]) ^ rot(y, s1[1]) ^ (y >> dt(s1[2]))))
    a, b, c, d, e, f, g, h = [V[..., i] for i in range(8)]
    for t in range(P["rounds"]):
        T1 = h + (rot(e, S1[0]) ^ rot(e, S1[1]) ^ rot(e, S1[2])) + ((e & f) ^ (~e & g)) + dt(P["K"][t]) + W[t]
        T2 = (rot(a, S0[0]) ^ rot(a, S0[1]) ^ rot(a, S0[2])) + ((a & b) ^ (a & c) ^ (b & c))
        h, g, f, e, d, c, b, a = g, f, e, d + T1, c, b, a, T1 + T2
    return np.stack([V[..., i] + x for i, x in enumerate((a, b, c, d, e, f, g, h))], axis=-1)


def sha_native(algo: str, state: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """state (..., H) bytes, blocks (..., nb, B) bytes -> state (..., H) after nb compressions."""
    wb = 4 if algo == "sha256" else 8
    dt = ">u4" if wb == 4 else ">u8"
    lead = np.broadcast_shapes(state.shape[:-1], blocks.shape[:-2])
    st = np.broadcast_to(state, lead + state.shape[-1:])
    bl = np.broadcast_to(blocks, lead + blocks.shape[-2:])
    V = np.ascontiguousarray(st).view(dt).astype(np.uint32 if wb == 4 else np.uint64)
    with np.errstate(over="ignore"):
        for j in range(bl.shape[-2]):
            W16 = np.ascontiguousarray(bl[..., j, :]).view(dt).astype(V.dtype)
            V = _compress_words(algo, V, W16)
    return V.astype(dt).view(np.uint8).reshape(lead + (8 * wb,))


def md_pad(algo: str, n_bytes: int) -> bytes:
    """FIPS 180-4 padding after a message of n_bytes (the prefix's bytes included)."""
    Bk = SHA_BLOCK[algo]
    lenb = 8 if algo == "sha256" else 16
    k = (-(n_bytes + 1 + lenb)) % Bk
    return b"\x80" + b"\0" * k + (8 * n_bytes).to_bytes(lenb, "big")


# ---------------------------------------------------------------------------------------------------------------------
# the gate-level check of an evaluated program
# ---------------------------------------------------------------------------------------------------------------------

def gate_check(op: str, ins: list, out: np.ndarray, k: int, rng) -> int:
    """Run k random instances of one evaluated fmul / sha op through its gadget circuit; raise on any disagreement.
    For a compression chain, a sampled instance is one (chain, block): the chain is replayed natively up to the block."""
    if op == "fmul":
        a, b = (np.broadcast_to(x, out.shape).reshape(-1, 2) for x in ins)
        o = out.reshape(-1, 2)
        sel = rng.choice(len(o), min(k, len(o)), replace=False)
        bits = lambda x: np.unpackbits(np.ascontiguousarray(x[sel]).view(np.uint8), axis=1, bitorder="little").T
        if not np.array_equal(run_sliced(gf_mul_circuit(), np.concatenate([bits(a), bits(b)])), bits(o)):
            raise AssertionError("fmul: the gadget circuit disagrees with the native product")
        return len(sel)
    H = SHA_STATE[op]
    lead = out.shape[:-1]
    st = np.broadcast_to(ins[0], lead + (H,)).reshape(-1, H)
    bl = np.broadcast_to(ins[1], lead + ins[1].shape[-2:]).reshape(-1, *ins[1].shape[-2:])
    n, nb = bl.shape[:2]
    picks = rng.choice(n * nb, min(k, n * nb), replace=False)
    inst, blk = picks // nb, picks % nb
    before = np.stack([sha_native(op, st[i], bl[i, :j]) if j else st[i] for i, j in zip(inst, blk)])
    after = np.stack([sha_native(op, st[i], bl[i, :j + 1]) for i, j in zip(inst, blk)])
    inp = np.concatenate([np.unpackbits(before, axis=1, bitorder="little").T,
                          np.unpackbits(np.ascontiguousarray(bl[inst, blk]), axis=1, bitorder="little").T])
    got = np.packbits(run_sliced(sha_circuit(op), inp).T, axis=1, bitorder="little")
    if not np.array_equal(got, after):
        raise AssertionError(f"{op}: the gadget circuit disagrees with the native compression")
    final = sha_native(op, st[inst], bl[inst])
    if not np.array_equal(final, out.reshape(-1, H)[inst]):
        raise AssertionError(f"{op}: the native chain disagrees with the op's output")
    return len(picks)
