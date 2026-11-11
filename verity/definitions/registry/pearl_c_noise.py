"""Pearl-C -h3's noise derivation as Definitions: the labelled hashes that seed a call's noise, and a noise line from its hash.

-h3 derives every noise line of a call from the epoch's salt, the fixed operand's root and each row's digest
(`verity.protocols.pouw.prover.schemes.pearl_c`: `seed_b`, `row_seed`, `sample_line`; in Lean `Pouw.PearlC.NoiseSha512`):

    s512(ℓ, m)  = the first 32 bytes of SHA-512(pearl-c/h3s/ℓ\\0, zero-padded to 128 bytes ‖ m)
    seed_B      = s512("seed-B", salt ‖ root_B)                                          PearlCSeedB
    leaf        = sha512/row-seg/v1 of the row's FP32 words, little-endian               PearlCRowDigest: its compressions
                                                                                         PearlCRowBlock, PearlCShaBlock
    seed_A      = s512("seed-A", salt ‖ leaf ‖ root_B ‖ u64le(unit · 2^32 + row))        PearlCRowSeed: its two compressions
                                                                                         PearlCRowSeedHead, PearlCRowSeedTail
    key         = s512("noise-line", seed)                                               PearlCLineKey
    raw         = s512("line", key ‖ side ‖ factor ‖ u32le(line))                        PearlCLineRaw
    x_p         = (raw_p mod 128) + 1, negated when raw_p >= 128                         PearlCLineInt
    root        = ⌊√(1024 · Σ_p x_p²)⌋                                                    PearlCLineNormRoot
    scale       = bf16(f32(norm32) / f32(bf16(root)))                                    PearlCLineScale
    code_p      = e4m3(f32(bf16(f32(bf16(x_p)) · f32(scale))))                           PearlCLineCode, PearlCLineCodes
    line        = the codes of raw                                                       PearlCNoiseLine
    E_A,i       = the line (side 0, factor 0, line 0, norm 256) of row i's seed_A         PearlCRowLine

``leaf`` is the row's 64-byte `sha512/row-seg/v1` digest (`pearl_c.row_digest`, the digest its hm96 leaf in A's tree hides;
on a bit row `verity.commitments.gates.rowleaf`'s `Sha512RowSeg_v1`, which has no word form and whose segment is one
circuit of up to nine compressions), and ``norm32`` the BF16 word of 32 times the line's norm:
`NORM_E` for an E line (norm 256), `NORM_F` for an F line (norm 16). bf16 is `F2fpBf16_v1`, the product `F32Mul_v2`, the
quotient `F32Div_v2` and e4m3 `F32ToE4m3Sat_v1`, so a line is total on every encoding. `pearl_c.sample_line` rejects a
non-finite step; at the two norms none is (a root is 181 to 23,170, a product at most 128 times the scale), and there the
tests hold a line to it.

A byte string is an ``Array<n, Value<8>>`` in byte order, an index a ``Value<32>``. Each Definition has a word form (``_v1``)
and a Boolean version (``_v2``) whose word view it is; on bits a byte is 8 bits, least significant first.

On bits each hash is one compression of its message from its label's midstate, a constant, written with
`verity.commitments.gates.sha512`'s `compress_gadget`. A word gate returns one value, so a compression's word form is one
gate a byte of one primitive, byte ``at`` of its value (``PearlCSeedBByte_v1`` and its like), each fed its index as a
constant; no Boolean version computes one byte alone. That primitive's evaluator reads no table by index and keeps no memo,
so it declares none (`PrimitiveDefinition.tables`): a labelled hash is `hashlib`'s SHA-512 of the label's block and the
message, and a compression from a given state, which `hashlib` does not give, is `_compress`, whose round constants are
computed as FIPS 180-4 defines them. One written with `compress_ref` would reach SHA-512's round constants as a table.

A Boolean Definition is a proof unit only up to 65,536 ANDs, the size circuit-check lowers one whole at, and a compression is
about 56,000. So a Definition of more than one compression is a composition of units under that size, each lowered and pinned
on its own, and its Boolean version the σ of the composition (`sigma_lift`): `PearlCRowSeed` is its two compressions,
`PearlCRowSeedTail` of `PearlCRowSeedHead`, and `PearlCNoiseLine` its hash and its codes, `PearlCLineCodes` of
`PearlCLineRaw`. `UNITS` names each such Definition's units. A row's digest is a compression a Call: `PearlCRowBlock` on each
32 words of a segment and its padding and `PearlCShaBlock` on each block of the segments' digests, from each prefix's
midstate; and a row's line, `PearlCRowLine`, is the digest, the seed, the key and the line, so it has no gate that is not in one of those.
"""
from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache
from typing import Callable

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, compress_columns, const_bits, multiply
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind, composite, primitive
from verity.definitions.ops.boolean.scalar import f32_to_e4m3_sat
from verity.definitions.ops.boolean.sqrt import isqrt
from verity.definitions.ops.fp32 import F32DivV2, F32MulV2
from verity.definitions.ops.prims import Bf16ToF32, F2fpBf16, F32ToE4m3Sat, S32ToF32Rn, const
from verity.circuits.refs import array_of
from verity.definitions.silicon import fp
from verity.circuits.types import Array, Value
from verity.commitments import rowleaf
from verity.commitments.gates.sha512 import IV, compress_gadget, midstate, pad_tail
from verity.definitions.boolean import sigma_lift

__all__ = ["BLOCK", "BYTES32", "BYTES64", "NORM_E", "NORM_F", "ROLES", "ROW_WORDS", "PearlCLineCode", "PearlCLineCodeV2", "PearlCLineCodes",
           "PearlCLineCodesV2", "PearlCLineInt", "PearlCLineIntV2", "PearlCLineKey", "PearlCLineKeyByte", "PearlCLineKeyV2",
           "PearlCLineNormRoot", "PearlCLineNormRootV2", "PearlCLineRaw", "PearlCLineRawByte", "PearlCLineRawV2",
           "PearlCLineScale", "PearlCLineScaleV2", "PearlCNoiseLine", "PearlCNoiseLineV2", "PearlCRowBlock", "PearlCRowBlockByte",
           "PearlCRowBlockV2", "PearlCRowDigest", "PearlCRowDigestV2", "PearlCRowLine", "PearlCRowLineV2", "PearlCRowSeed",
           "PearlCRowSeedHead",
           "PearlCRowSeedHeadByte", "PearlCRowSeedHeadV2", "PearlCRowSeedTail", "PearlCRowSeedTailByte", "PearlCRowSeedTailV2",
           "PearlCRowSeedV2", "PearlCSeedB", "PearlCSeedBByte", "PearlCSeedBV2", "PearlCShaBlock", "PearlCShaBlockByte",
           "PearlCShaBlockV2", "UNITS", "label", "s512"]

U8, BF16, U32 = Value(8), Value(16), Value(32)
#: a seed, a line key, a line's hash or its 32 codes; and a SHA-512 digest (a root, a row's leaf)
BYTES32, BYTES64 = Array(32, U8), Array(64, U8)
#: one SHA-512 block as its bytes, and as the 32 words of a row it holds
BLOCK, ROW_WORDS = Array(128, U8), Array(32, U32)
#: the roles -h3's hashes are labelled by
ROLES = ("seed-A", "seed-B", "noise-line", "line")
#: the BF16 words of 32 times a line's norm: an E line's 256 and an F line's 16
NORM_E, NORM_F = 0x4600, 0x4400
#: `F2fpBf16_v1`'s word for a NaN
_NAN_BF16 = 0x7FFF
_M64 = (1 << 64) - 1
#: where the tests hold each Definition to its reference
_TESTS = "verity/definitions/tests/test_pearl_c_noise.py"


# ---- the hashes ------------------------------------------------------------------------------------------------------------

def label(role: str) -> bytes:
    """A role's label block: ``pearl-c/h3s/<role>`` and a zero byte, zero-padded to one SHA-512 block."""
    if role not in ROLES:
        raise ValueError(f"-h3 labels {', '.join(ROLES)}, not {role!r}")
    return f"pearl-c/h3s/{role}\0".encode().ljust(128, b"\0")


@lru_cache(maxsize=None)
def _midstate(role: str) -> tuple[int, ...]:
    return tuple(midstate(label(role)))


def s512(role: str, message: bytes) -> bytes:
    """-h3's labelled hash: the first 32 bytes of SHA-512(``label(role)`` ‖ ``message``)."""
    return hashlib.sha512(label(role) + message).digest()[:32]


def _round_constants() -> list[int]:
    """SHA-512's 80 round constants as FIPS 180-4 §4.2.3 defines them: the first 64 bits of the fractional part of the cube
    root of each of the first 80 primes, so ⌊∛(p · 2^192)⌋ mod 2^64, by Newton's iteration from above (p < 2^9)."""
    out, p = [], 1
    while len(out) < 80:
        p += 1
        if all(p % q for q in range(2, math.isqrt(p) + 1)):
            n, x = p << 192, 1 << 67
            while (y := (2 * x + n // (x * x)) // 3) < x:
                x = y
            out.append(x & _M64)
    return out


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & _M64


def _compress(state: bytes, block: bytes) -> bytes:
    """One SHA-512 compression (FIPS 180-4 §6.4.2) of a 128-byte ``block`` from the chaining value ``state``, each as its
    bytes: `sha512.compress_ref`, which the tests hold it to, with the round constants computed rather than read. A byte
    primitive's evaluator is this, so that it reads no table (`sha512.K` is one) and keeps no memo."""
    k, cv, w = _round_constants(), struct.unpack(">8Q", state), list(struct.unpack(">16Q", block))
    for t in range(16, 80):
        s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ (w[t - 15] >> 7)
        s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ (w[t - 2] >> 6)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & _M64)
    a, b, c, d, e, f, g, h = cv
    for t in range(80):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + ((e & f) ^ (~e & g)) + k[t] + w[t]) & _M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & _M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & _M64, c, b, a, (t1 + t2) & _M64
    return struct.pack(">8Q", *((x + y) & _M64 for x, y in zip(cv, (a, b, c, d, e, f, g, h))))


def _words(data: list[list]) -> list[list]:
    """Bytes of circuit bits as 64-bit words, eight bytes each: a word is big-endian, so its low bits are its last byte's."""
    return [[f for b in reversed(data[i:i + 8]) for f in b] for i in range(0, len(data), 8)]


def _word_bytes(words: list[list]) -> list[list]:
    """Words of circuit bits as their bytes, each word big-endian: a chaining value's or a digest's bytes."""
    return [w[8 * j:8 * j + 8] for w in words for j in range(7, -1, -1)]


def _s512_bits(C, role: str, message: list[list]) -> list[list]:
    """`s512` over bytes of circuit bits, from the label's midstate: its 32 bytes."""
    data = list(message) + [const_bits(b, 8) for b in pad_tail(128 + len(message))]
    cv = [const_bits(v, 64) for v in _midstate(role)]
    for i in range(0, len(data), 128):
        cv = compress_gadget(C, cv, _words(data[i:i + 128]))
    return _word_bytes(cv[:4])


def _le32(x: int) -> bytes:
    return (x & 0xFFFFFFFF).to_bytes(4, "little")


def _bytes(word: list) -> list[list]:
    """A word of circuit bits as its bytes, little-endian."""
    return [word[i:i + 8] for i in range(0, len(word), 8)]


def _compression(name: str, params: tuple, n: int, value: Callable, value_bits: Callable, rule: str, reference: str,
                 on_bits: str) -> tuple[PrimitiveDefinition, CompositeDefinition, CompositeDefinition]:
    """A value of ``n`` bytes that SHA-512 compressions make of ``params``: (its byte primitive, its word form, its Boolean
    version). ``value`` gives the bytes from the parameters' words, ``value_bits`` the same bytes from a circuit and the
    parameters' elements on bits."""
    held = f"protocol-defined; equal to `{reference}` on its vectors ({_TESTS})"
    at_type, ret = Value((n - 1).bit_length()), Array(n, U8)
    byte = PrimitiveDefinition(
        f"{name}Byte", 1, (*params, ("at", at_type)), U8, lambda *x: value([int(v) for v in x[:-1]])[int(x[-1])],
        doc=f"Byte ``at`` of {rule}: {name}_v1's gates.", conformance="protocol-defined")

    def body(B, S, *args):
        at = array_of([B.call(const(at_type.w, i)) for i in range(n)])
        return B.batch(byte, *args, at, axes=(None,) * len(args) + (0,))

    word = CompositeDefinition(name, 1, (), lambda S: (params, ret), body, conformance=held,
                               doc=f"{rule}, its {n} bytes: `{reference}`.")
    bits = definition(name, 2, lambda C, *el: value_bits(C, el), tuple((p, BL.bit_type(t)) for p, t in params),
                      BL.bit_type(ret), word=bind(word), conformance=held, doc=f"{rule} on bits ({on_bits}): {name}_v1")
    return byte, word, bits


def _hash(name: str, role: str, params: tuple, message: Callable, message_bits: Callable, rule: str, reference: str
          ) -> tuple[PrimitiveDefinition, CompositeDefinition, CompositeDefinition]:
    """A hash of -h3 over ``params`` whose message fits one block, so one compression: (its byte primitive, its word form,
    its Boolean version). ``message`` gives the hashed bytes from the parameters' words, ``message_bits`` the same bytes from
    the parameters' elements on bits."""
    return _compression(name, params, 32, lambda x: s512(role, message(x)),
                        lambda C, el: _s512_bits(C, role, message_bits(el)), rule, reference,
                        "one SHA-512 compression from the label's midstate")


PearlCSeedBByte, PearlCSeedB, PearlCSeedBV2 = _hash(
    "PearlCSeedB", "seed-B", (("salt", BYTES32), ("root", BYTES64)), bytes, list,
    'seed_B = s512("seed-B", salt ‖ root), the seed of a fixed operand\'s noise from the epoch\'s salt and the operand\'s root',
    "pearl_c.seed_b")

PearlCLineKeyByte, PearlCLineKey, PearlCLineKeyV2 = _hash(
    "PearlCLineKey", "noise-line", (("seed", BYTES32),), bytes, list,
    'key = s512("noise-line", seed), the key of a seed\'s lines',
    "pearl_c.line_key")

PearlCLineRawByte, PearlCLineRaw, PearlCLineRawV2 = _hash(
    "PearlCLineRaw", "line", (("key", BYTES32), ("side", U8), ("factor", U8), ("line", U32)),
    lambda x: bytes(x[:34]) + _le32(x[34]),
    lambda el: [*el[:34], *_bytes(el[34])],
    'raw = s512("line", key ‖ side ‖ factor ‖ u32le(line)), the 32 bytes a noise line is made from',
    "pearl_c.sample_line")


# ---- the row seed, a compression a unit ------------------------------------------------------------------------------------
#
# A row seed's message is 168 bytes, two blocks after the label's, so two compressions: 112,691 ANDs in one circuit, where a
# Boolean Definition is lowered whole only up to 65,536. Each compression is a Definition of its own, and the row seed their
# composition: the head takes the first block (the salt, the row's digest and the root's first half) from the label's
# midstate to the chaining value, the tail takes that and the second block (the root's second half, the row's place and the
# padding of a 296-byte message) to the seed.

#: the length SHA-512 pads a row seed's message to: the label's block and the 168 bytes
_ROW_SEED_BYTES = 128 + 168


def _state(cv: list[int]) -> bytes:
    """A chaining value as its 64 bytes, each word big-endian."""
    return b"".join(w.to_bytes(8, "big") for w in cv)


def _row_seed_tail(x: list[int]) -> bytes:
    """`PearlCRowSeedTail_v1`'s 32 bytes from its parameters' words: the state's 64 bytes, the root's last 32, the unit and
    the row."""
    block = bytes(x[64:96]) + _le32(x[97]) + _le32(x[96]) + pad_tail(_ROW_SEED_BYTES)
    return _compress(bytes(x[:64]), block)[:32]


def _row_seed_tail_bits(C, el: list) -> list[list]:
    block = [*el[64:96], *_bytes(el[97]), *_bytes(el[96]), *(const_bits(b, 8) for b in pad_tail(_ROW_SEED_BYTES))]
    return _word_bytes(compress_gadget(C, _words(el[:64]), _words(block))[:4])


PearlCRowSeedHeadByte, PearlCRowSeedHead, PearlCRowSeedHeadV2 = _compression(
    "PearlCRowSeedHead", (("salt", BYTES32), ("leaf", BYTES64), ("root_head", BYTES32)), 64,
    lambda x: _compress(_compress(_state(IV), label("seed-A")), bytes(x)),
    lambda C, el: _word_bytes(compress_gadget(C, [const_bits(v, 64) for v in _midstate("seed-A")], _words(list(el)))),
    "the chaining value of a row seed's hash after its message's first block, salt ‖ leaf ‖ the root's first 32 bytes",
    "pearl_c.row_seed", "the first of its two SHA-512 compressions, from the label's midstate")

PearlCRowSeedTailByte, PearlCRowSeedTail, PearlCRowSeedTailV2 = _compression(
    "PearlCRowSeedTail", (("state", BYTES64), ("root_tail", BYTES32), ("unit", U32), ("row", U32)), 32,
    _row_seed_tail, _row_seed_tail_bits,
    "seed_A from the chaining value `PearlCRowSeedHead_v1` leaves and its message's second block, the root's last 32 bytes "
    "‖ u64le(unit · 2^32 + row) and the padding of a 296-byte message",
    "pearl_c.row_seed", "the second of its two SHA-512 compressions")

_SEED = f"protocol-defined; equal to `pearl_c.row_seed` on its vectors ({_TESTS})"


@composite("PearlCRowSeed", 1, [], lambda S: ((("salt", BYTES32), ("leaf", BYTES64), ("root", BYTES64), ("unit", U32),
                                                ("row", U32)), BYTES32), conformance=_SEED)
def PearlCRowSeed(B, S, salt, leaf, root, unit, row):
    """seed_A = s512("seed-A", salt ‖ leaf ‖ root ‖ u64le(unit · 2^32 + row)), the seed of a row's E line from the salt, the
    row's 64-byte digest, the fixed operand's root and the row's place, its 32 bytes (`pearl_c.row_seed`): its hash's two
    compressions, `PearlCRowSeedTail_v1` of `PearlCRowSeedHead_v1`."""
    return B.call(PearlCRowSeedTail, B.call(PearlCRowSeedHead, salt, leaf, root[:32]), root[32:], unit, row)


PearlCRowSeedV2 = sigma_lift(PearlCRowSeed, 2)


# ---- a row's digest, a compression a unit ----------------------------------------------------------------------------------
#
# sha512/row-seg/v1 hashes each 1 KiB segment of a row from its prefix's midstate, then the segments' digests from the outer
# prefix's (`verity.commitments.rowleaf`). A prefix is one block, and a row of FP32 words in whole blocks is 32 words a
# block (a segment's padding starts on a word and ends a block, so it is words too), so every compression after a midstate
# is one of two Definitions: `PearlCRowBlock` on 32 words, the row's or the padding's, and `PearlCShaBlock` on a block of
# bytes, the outer hash's digests and padding.

PearlCRowBlockByte, PearlCRowBlock, PearlCRowBlockV2 = _compression(
    "PearlCRowBlock", (("state", BYTES64), ("x", ROW_WORDS)), 64,
    lambda x: _compress(bytes(x[:64]), b"".join(_le32(w) for w in x[64:])),
    lambda C, el: _word_bytes(compress_gadget(C, _words(el[:64]), _words([b for w in el[64:] for b in _bytes(w)]))),
    "the chaining value of SHA-512 after one block, 32 words of a row little-endian, from the chaining value ``state``",
    "sha512.compress_ref", "one SHA-512 compression")

PearlCShaBlockByte, PearlCShaBlock, PearlCShaBlockV2 = _compression(
    "PearlCShaBlock", (("state", BYTES64), ("block", BLOCK)), 64,
    lambda x: _compress(bytes(x[:64]), bytes(x[64:])),
    lambda C, el: _word_bytes(compress_gadget(C, _words(el[:64]), _words(list(el[64:])))),
    "the chaining value of SHA-512 after one block of 128 bytes, from the chaining value ``state``",
    "sha512.compress_ref", "one SHA-512 compression")


def _refuse_row(b: dict) -> str | None:
    return None if b.get("K", 1) >= 1 else "a row has at least one word"


def _row_digest(B, S, x):
    consts: dict = {}

    def lit(w: int, v: int):
        if (w, v) not in consts:
            consts[(w, v)] = B.call(const(w, v))
        return consts[(w, v)]

    def state_of(prefix: bytes):
        return array_of([lit(8, b) for b in _state(midstate(prefix))])

    segments = rowleaf.sha512_row_seg_segments(32 * S.K)
    digests, at = [], 0
    for j, m in enumerate(segments):
        pad = pad_tail(128 + m // 8)
        words = [x[t] for t in range(at, at + m // 32)]
        words += [lit(32, int.from_bytes(pad[i:i + 4], "little")) for i in range(0, len(pad), 4)]
        state = state_of(rowleaf.sha512_row_seg_segment_prefix(j, len(segments), m))
        for lo in range(0, len(words), 32):
            state = B.call(PearlCRowBlock, state, array_of(words[lo:lo + 32]))
        digests.append(state)
        at += m // 32
    state = state_of(rowleaf.sha512_row_seg_prefix(32 * S.K))
    message = [d[i] for d in digests for i in range(64)] + [lit(8, b) for b in pad_tail(128 + 64 * len(digests))]
    for lo in range(0, len(message), 128):
        state = B.call(PearlCShaBlock, state, array_of(message[lo:lo + 128]))
    return state


PearlCRowDigest = CompositeDefinition(
    "PearlCRowDigest", 1, ("K",), lambda S: ((("x", Array(S.K, U32)),), BYTES64), _row_digest, refuse=_refuse_row,
    conformance=f"protocol-defined; equal to `pearl_c.row_digest` on its vectors ({_TESTS})",
    doc="A row's digest, its 64 bytes: sha512/row-seg/v1 of the row's K FP32 words little-endian (`pearl_c.row_digest`, "
        "`rowleaf.sha512_row_seg_digest`), a compression a Call: each segment from its prefix's midstate through "
        "`PearlCRowBlock_v1` on each 32 words of it and of its padding, then the segments' digests and their padding "
        "through `PearlCShaBlock_v1` from the outer prefix's midstate.")
PearlCRowDigestV2 = sigma_lift(PearlCRowDigest, 2)


# ---- a line from its hash --------------------------------------------------------------------------------------------------

_LINE = f"protocol-defined; equal to `pearl_c.sample_line` on its vectors ({_TESTS})"


@primitive("PearlCLineInt", 1, [("b", U8)], U32, conformance=_LINE)
def PearlCLineInt(b: int) -> int:
    """A line's integer from its byte, as an s32 word: (b mod 128) + 1, negated when b >= 128; so 1 to 128 in magnitude."""
    x = (int(b) & 0x7F) + 1
    return (-x if int(b) >> 7 else x) & 0xFFFFFFFF


@primitive("PearlCLineNormRoot", 1, [("raw", BYTES32)], U32, conformance=_LINE)
def PearlCLineNormRoot(*raw: int) -> int:
    """⌊√(1024 · Σ x²)⌋ over a line's 32 integers (`PearlCLineInt_v1` of each byte): 32 times the line's norm before scaling,
    rounded down; 181 to 23,170."""
    return math.isqrt(1024 * sum(((int(b) & 0x7F) + 1) ** 2 for b in raw))


def _int_bits(C, b: list) -> list:
    """`PearlCLineInt_v1` on bits: with v = b mod 128 and s = b's top bit, v + 1 when s is clear and NOT v (−(v + 1)) when
    set, which is (v XOR s) + NOT s on 8 bits with s as the operand's top bit; the 24 bits above are s."""
    s = b[7]
    return add(C, [x ^ s for x in b[:7]] + [s], [], cin=NOT(s), width=8) + [s] * 24


def _root_bits(C, *raw: list) -> list:
    """`PearlCLineNormRoot_v1` on bits: with v = b mod 128, Σ (v + 1)² = Σ v² + 2 Σ v + 32, each v² its diagonal bits and
    one AND a pair of bits, all in one carry-save sum of 20 bits; then the digit-by-digit root of the sum times 1024."""
    cols: list[list] = [[] for _ in range(14)]
    cols[5].append(ONE)
    for b in raw:
        v = b[:7]
        for i in range(7):
            cols[2 * i].append(v[i])
            cols[i + 1].append(v[i])
            for j in range(i + 1, 7):
                cols[i + j + 1].append(C.AND(v[i], v[j]))
    total = compress_columns(C, cols)[:20]
    return isqrt(C, [ZERO] * 10 + total) + [ZERO] * 17


PearlCLineIntV2 = definition("PearlCLineInt", 2, _int_bits, (("b", BL.bit_type(U8)),), BL.bit_type(U32), word=PearlCLineInt,
                             conformance=_LINE, doc="A line's integer from its byte, on bits: PearlCLineInt_v1")
PearlCLineNormRootV2 = definition("PearlCLineNormRoot", 2, _root_bits, (("raw", BL.bit_type(BYTES32)),), BL.bit_type(U32),
                                  word=PearlCLineNormRoot, conformance=_LINE,
                                  doc="⌊√(1024 · Σ x²)⌋ over a line's 32 integers, on bits: PearlCLineNormRoot_v1")


@composite("PearlCLineScale", 1, [], lambda S: ((("raw", BYTES32), ("norm32", BF16)), BF16), conformance=_LINE)
def PearlCLineScale(B, S, raw, norm32):
    """A line's scale, a BF16 word: bf16(f32(norm32) / f32(bf16(root))) for root = `PearlCLineNormRoot_v1` of the line's
    bytes and ``norm32`` the BF16 word of 32 times the line's norm, by `S32ToF32Rn_v1`, `F2fpBf16_v1` and `F32Div_v2`
    (`pearl_kw.bf16_div`)."""
    root = B.call(F2fpBf16, B.call(S32ToF32Rn, B.call(PearlCLineNormRoot, raw)))
    return B.call(F2fpBf16, B.call(F32DivV2, B.call(Bf16ToF32, norm32), B.call(Bf16ToF32, root)))


@composite("PearlCLineCode", 1, [], lambda S: ((("b", U8), ("scale", BF16)), U8), conformance=_LINE)
def PearlCLineCode(B, S, b, scale):
    """One code of a line: e4m3(f32(bf16(f32(bf16(x)) · f32(scale)))) for x = `PearlCLineInt_v1` of its byte, by
    `S32ToF32Rn_v1`, `F2fpBf16_v1`, `F32Mul_v2` (`pearl_kw.bf16_mul`) and `F32ToE4m3Sat_v1`."""
    x = B.call(F2fpBf16, B.call(S32ToF32Rn, B.call(PearlCLineInt, b)))
    y = B.call(F2fpBf16, B.call(F32MulV2, B.call(Bf16ToF32, x), B.call(Bf16ToF32, scale)))
    return B.call(F32ToE4m3Sat, B.call(Bf16ToF32, y))


def _bf16_bits(C, x: list) -> list:
    return fp.f32_to_bf16(C, x, _NAN_BF16)


def _scale_bits(C, *el: list) -> list:
    root = _bf16_bits(C, fp.s32_to_f32(C, _root_bits(C, *el[:32])))
    return _bf16_bits(C, fp.f32_div(C, fp.bf16_to_f32(el[32]), fp.bf16_to_f32(root), nan_word=fp.F32_NAN_PTX))


def _code_bits(C, b: list, scale: list) -> list:
    """`PearlCLineCode_v1` on bits, its product rounded once. The integer x = ±(v + 1) is exact as a BF16 word, and its FP32
    product with a BF16 word is exact short of overflow (16 significant bits, a multiple of FP32's least subnormal), so the
    product's BF16 word is the one nearest (v + 1) · sig · 2^(e − 134), for the scale's significand sig and exponent e: one
    `round_fp`, which never shifts right since x is at least 1. An infinite scale overflows there to its infinity, and a NaN
    is `F2fpBf16_v1`'s."""
    S = fp.unpack_fp(C, scale, 8, 7)
    s = b[7] ^ S.s
    product = multiply(C, add(C, b[:7], [], cin=ONE, width=8), S.sig)
    y = fp.round_fp(C, s, add(C, S.e_eff, const_bits(8, 4), width=10), 0, product, s, 8, 7, underflow=False)
    y = [OR(C, S.nan, f) for f in y[:7]] + y[7:15] + [C.AND(NOT(S.nan), y[15])]
    return f32_to_e4m3_sat(C, fp.bf16_to_f32(y))


def _codes_bits(C, *el: list) -> list[list]:
    scale = _scale_bits(C, *el)
    return [_code_bits(C, b, scale) for b in el[:32]]


PearlCLineScaleV2 = definition(
    "PearlCLineScale", 2, _scale_bits, (("raw", BL.bit_type(BYTES32)), ("norm32", BL.bit_type(BF16))), BL.bit_type(BF16),
    word=bind(PearlCLineScale), conformance=_LINE,
    doc="A line's scale on bits, its steps' builders in one circuit: PearlCLineScale_v1")
PearlCLineCodeV2 = definition(
    "PearlCLineCode", 2, _code_bits, (("b", BL.bit_type(U8)), ("scale", BL.bit_type(BF16))), BL.bit_type(U8),
    word=bind(PearlCLineCode), conformance=_LINE,
    doc="One code of a line on bits, its product of 8-bit significands rounded once to BF16: PearlCLineCode_v1")


@composite("PearlCLineCodes", 1, [], lambda S: ((("raw", BYTES32), ("norm32", BF16)), BYTES32), conformance=_LINE)
def PearlCLineCodes(B, S, raw, norm32):
    """A line's 32 E4M3 codes from its hash: `PearlCLineCode_v1` of each byte at the line's `PearlCLineScale_v1`
    (`pearl_c.sample_line` after its hash)."""
    return B.batch(PearlCLineCode, raw, B.call(PearlCLineScale, raw, norm32), axes=(0, None))


PearlCLineCodesV2 = definition(
    "PearlCLineCodes", 2, _codes_bits, (("raw", BL.bit_type(BYTES32)), ("norm32", BL.bit_type(BF16))), BL.bit_type(BYTES32),
    word=bind(PearlCLineCodes), conformance=_LINE,
    doc="A line's 32 codes on bits, in one circuit, so what a code reads of the scale alone is computed once: "
        "PearlCLineCodes_v1")


@composite("PearlCNoiseLine", 1, [], lambda S: ((("key", BYTES32), ("side", U8), ("factor", U8), ("line", U32),
                                                  ("norm32", BF16)), BYTES32), conformance=_LINE)
def PearlCNoiseLine(B, S, key, side, factor, line, norm32):
    """A noise line's 32 E4M3 codes: `PearlCLineCodes_v1` of `PearlCLineRaw_v1`; `pearl_c.sample_line(seed, side, factor,
    line, norm)` for key = `PearlCLineKey_v1` of the seed and ``norm32`` the BF16 word of 32 · norm."""
    return B.call(PearlCLineCodes, B.call(PearlCLineRaw, key, side, factor, line), norm32)


PearlCNoiseLineV2 = sigma_lift(PearlCNoiseLine, 2)


@composite("PearlCRowLine", 1, ["K"], lambda S: ((("salt", BYTES32), ("x", Array(S.K, U32)), ("root", BYTES64), ("unit", U32),
                                                   ("row", U32)), BYTES32), refuse=_refuse_row,
           conformance=f"protocol-defined; equal to `pearl_c.sample_line` of `pearl_c.row_seed` on its vectors ({_TESTS})")
def PearlCRowLine(B, S, salt, x, root, unit, row):
    """-h3's noise line E_A of row ``row`` of unit ``unit``, its 32 E4M3 codes, from the salt, the row's K FP32 words and the
    fixed operand's root: `PearlCNoiseLine_v1` (side 0, factor 0, line 0, norm 256) under `PearlCLineKey_v1` of the row's
    `PearlCRowSeed_v1` over its `PearlCRowDigest_v1`; `pearl_c.sample_line(row_seed(salt, row_digest(x), root, unit, row),
    0, 0, 0)`."""
    seed = B.call(PearlCRowSeed, salt, B.call(bind(PearlCRowDigest, K=S.K), x), root, unit, row)
    zero = B.call(const(8, 0))
    return B.call(PearlCNoiseLine, B.call(PearlCLineKey, seed), zero, zero, B.call(const(32, 0)), B.call(const(16, NORM_E)))


PearlCRowLineV2 = sigma_lift(PearlCRowLine, 2)

#: the Boolean Definitions over the size one is lowered whole at (65,536 ANDs), each with the units its body Calls and no
#: gate besides: every AND of it is in one of them, and each is under that size with its lowering pinned
UNITS = {"PearlCRowSeed_v2": ("PearlCRowSeedHead_v2", "PearlCRowSeedTail_v2"),
         "PearlCNoiseLine_v2": ("PearlCLineRaw_v2", "PearlCLineCodes_v2")}
