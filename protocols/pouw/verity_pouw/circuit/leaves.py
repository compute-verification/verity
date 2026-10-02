"""The tile checks' hiding leaves in gates (`pc4`, `pc8`): each row a unit reads or writes, and each tile's digest, under
`hm96-sha512/row/v1` (`verity.commitments.rowleaf`, `verity.commitments.hm96`), so a unit publishes a commit string b ‖ c
where the scheme published a row.

* `Hm96Sha512{ROLE,WORDS}`: one row's b ‖ c from the row and its 192-byte salt.  x = SHA-512(`sha512_row_prefix`(ROLE, 16,
  WORDS) ‖ the row's WORDS 16-bit words, little-endian) from the prefix's constant midstate; c = SHA-512(salt prefix ‖ y)
  from its midstate (two compressions); b = x ⊕ M·y (`Hm96Mask`).  The verifier computes the tree leaf, H(leaf_prefix(key)
  ‖ b ‖ c), itself.  The key is `hm96.SHA512`'s pinned key, a constant of the gates.
* `Hm96Mask`: M·y for that key, 16 × 48 blocks of `words.PouwHm96Hankel` and their XORs, no AND.
* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.

The rows (`SCHEMAS`), each a row of 16-bit words with every narrower element packed low bits first (flock-fp's packed
rows), so the row's bytes are the tensor's memory; a registration domain binds the schema id and the word count:

    row (schema id)               words                                              n words            role
    bf16 (pouw/row-bf16/v1)       k BF16 words                                       k                  X (A_i) or W (B_j)
    fp32 (pouw/row-fp32/v1)       k FP32 words, each as two, low half first          2k                 X or W
    nvfp4-p (pouw/row-nvfp4-p/v1) k E2M1 codes four per word (code 4w + q at bit     k/4 + k/32 + 64    X (A′_i ‖ P_A,i) or
                                  4q), k/16 UE4M3 scale bytes two per word (low                         W (B̃_j ‖ P_B,j)
                                  byte first), then P's 64 BF16 words
    fp8-p (pouw/row-fp8-p/v1)     k E4M3 codes two per word (low byte first), then   k/2 + 64           X or W
                                  P's 64 BF16 words
    tile (pouw/tile-h0/v1)        the 32-byte tile digest (`PearlTileDigest`), two   16                 X
                                  bytes per word, low byte first

nvfp4-p's codes and scales are `rowleaf.nvfp4_row_bytes`' bytes.  `words_*` and `commit` are the reference;
``tests/vectors/pouw_rows.json`` pins them.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import ROLE_W, ROLE_X, sha512_row_digest, sha512_row_prefix
from verity.evaluation import evaluate
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import array_of
from verity.ir.types import Array

from ..schemes import pearl_c as C
from . import hashes as H
from . import words as W

U32, M32 = W.U32, W.M32
INST = hm96.SHA512
KEY = INST.default_key
SALT, BC = Array(INST.salt_bytes // 4, U32), Array(INST.proven_bytes // 4, U32)
P_WORDS = 64
#: each row's schema id, which a registration domain binds with the row's word count (`ROW_WORDS`, at depth k)
SCHEMAS = {"bf16": "pouw/row-bf16/v1", "fp32": "pouw/row-fp32/v1", "nvfp4-p": "pouw/row-nvfp4-p/v1",
           "fp8-p": "pouw/row-fp8-p/v1", "tile": "pouw/tile-h0/v1"}
ROW_WORDS = {"bf16": lambda k: k, "fp32": lambda k: 2 * k, "nvfp4-p": lambda k: k // 4 + k // 32 + P_WORDS,
             "fp8-p": lambda k: k // 2 + P_WORDS, "tile": lambda k: 16}


def halves(data: bytes) -> list[int]:
    """Bytes (a multiple of 4) as big-endian halves."""
    return [int.from_bytes(data[i:i + 4], "big") for i in range(0, len(data), 4)]


_KEY = tuple(halves(KEY))


@lru_cache(maxsize=64)
def _midstate(prefix: bytes) -> tuple[int, ...]:
    """SHA-512's state after the one-block ``prefix``, as 16 halves: `Sha512Compress` evaluated on constants."""
    iv = [h for v in H.SHA512_IV for h in (v >> 32, v & M32)]
    return tuple(evaluate(H.Sha512Compress, iv, halves(prefix)))


def _sha512_after(e: H.Emit, state: Sequence, msg: Sequence, nbytes: int) -> list:
    """The SHA-512 digest (16 halves) of a message whose first block is behind ``state`` and whose other bytes are the
    big-endian halves ``msg``, ``nbytes`` the whole message's length."""
    msg = list(msg) + [0x80000000]
    while len(msg) % 32 != 28:
        msg.append(0)
    msg += [0, 0, (8 * nbytes) >> 32, (8 * nbytes) & M32]
    st = list(state)
    for blk in range(0, len(msg), 32):
        st = list(e.B.call(H.Sha512Compress, e.array(st), e.array(msg[blk:blk + 32])))
    return st


def _mask_body(B, S_, y):
    e = H.Emit(B)
    out = []
    for o in range(16):
        acc = None
        for s in range(SALT.n):
            t = e.call(W.PouwHm96Hankel, y[s], _KEY[o + s], _KEY[o + s + 1])
            acc = t if acc is None else e.call(W.PouwXor32, acc, t)
        out.append(acc)
    return e.array(out)


Hm96Mask = bind(CompositeDefinition(
    "PouwHm96Mask", 1, (), lambda S_: ((("salt", SALT),), Array(16, U32)), _mask_body,
    doc="hm96-sha512/v1's M·y for its pinned key: output half o is the XOR over salt halves s of the Hankel block of key "
        "halves o + s and o + s + 1 (XORs alone).", conformance="protocol-defined"))


def _hm96_body(B, S_, row, salt):
    e, n = H.Emit(B), S_.WORDS // 2
    x = _sha512_after(e, _midstate(sha512_row_prefix(S_.ROLE, 16, S_.WORDS)), [row[i] for i in range(n)], 128 + 2 * S_.WORDS)
    c = _sha512_after(e, _midstate(INST.salt_prefix), [salt[i] for i in range(SALT.n)], 128 + INST.salt_bytes)
    m = B.call(Hm96Mask, salt)
    return e.array([e.call(W.PouwXor32, x[i], m[i]) for i in range(16)] + c)


Hm96Sha512Def = CompositeDefinition(
    "PouwHm96Sha512", 1, ("ROLE", "WORDS"), lambda S_: ((("row", Array(S_.WORDS // 2, U32)), ("salt", SALT)), BC),
    _hm96_body, doc="A row's hm96-sha512/row/v1 commit string b ‖ c (big-endian halves): the row is WORDS 16-bit words as "
                    "big-endian halves of its little-endian bytes, x its sha512/row/v1 digest under role ROLE, and the salt "
                    "192 bytes.", conformance="protocol-defined")


@lru_cache(maxsize=64)
def hm96_leaf(role: int, words: int) -> SpecializedDefinition:
    if role not in (ROLE_X, ROLE_W) or words <= 0 or words % 2:
        raise ValueError(f"an hm96 row here is role X or W and an even number of 16-bit words, not ({role}, {words})")
    return bind(Hm96Sha512Def, ROLE=role, WORDS=words)


def _tile_digest_body(B, S_, c, u):
    e = H.Emit(B)
    digests: list = []
    for r in range(S_.TM):
        for h in range(0, S_.TN, C.CHUNK):
            cols = range(h, min(h + C.CHUNK, S_.TN))
            ws = [c[r][j] for j in cols] + [u[r][j] for j in cols]
            digests += H.turboshake128(e, list(zip(ws[0::2], ws[1::2])), 32, C.D_MESSAGE)
    out = H.turboshake128(e, list(zip(digests[0::2], digests[1::2])), 32, C.D_LEAF)
    return e.array([e.call(W.PouwBswap32, h) for h in out])


PearlTileDigestDef = CompositeDefinition(
    "PouwPearlTileDigest", 1, ("TM", "TN"),
    lambda S_: ((("c", Array(S_.TM, Array(S_.TN, W.F32))), ("u", Array(S_.TM, Array(S_.TN, W.F32)))), Array(8, U32)),
    _tile_digest_body, doc="pearl_c.leaf (h0) of a TM x TN tile's C̃ and U: its 32 bytes as big-endian halves.",
    conformance="protocol-defined")


@lru_cache(maxsize=16)
def tile_digest(tm: int, tn: int) -> SpecializedDefinition:
    return bind(PearlTileDigestDef, TM=tm, TN=tn)


# ---- a unit's rows in gates (`e` a body's emitter: `pc4._Emit` or `hashes.Emit`) ---------------------------------------

def bf16_row(e, x, k: int) -> list:
    return [e.call(W.PouwPack16x2, x[2 * t], x[2 * t + 1]) for t in range(k // 2)]


def fp32_row(e, x, k: int) -> list:
    return [e.call(W.PouwBswap32, x[t]) for t in range(k)]


def nvfp4_p_row(e, codes, scales, p, k: int) -> list:
    return ([e.call(W.PouwPack4x8, *(codes[8 * t + i] for i in range(8))) for t in range(k // 8)]
            + [e.call(W.PouwPack8x4, *(scales[4 * t + i] for i in range(4))) for t in range(k // 64)] + bf16_row(e, p, P_WORDS))


def fp8_p_row(e, codes, p, k: int) -> list:
    return [e.call(W.PouwPack8x4, *(codes[4 * t + i] for i in range(4))) for t in range(k // 4)] + bf16_row(e, p, P_WORDS)


def leaf(e, role: int, row: list, salt):
    """The commit string of the row whose big-endian halves are ``row``."""
    return e.B.call(hm96_leaf(role, 2 * len(row)), array_of(row), salt)


def tile_leaf(e, tm: int, tn: int, c, u, salt):
    """The commit string of the tile digest of C̃ ``c`` and U ``u`` (each TM x TN FP32 words)."""
    d = e.B.call(tile_digest(tm, tn), c, u)
    return leaf(e, ROLE_X, [d[i] for i in range(8)], salt)


# ---- the reference -----------------------------------------------------------------------------------------------------

def words_bf16(x: Sequence[int]) -> list[int]:
    return [v & 0xFFFF for v in x]


def words_fp32(x: Sequence[int]) -> list[int]:
    return [h for v in x for h in (v & 0xFFFF, (v >> 16) & 0xFFFF)]


def words_nvfp4_p(codes: Sequence[int], scales: Sequence[int], p: Sequence[int]) -> list[int]:
    if len(codes) % 64 or len(scales) * 16 != len(codes) or len(p) != P_WORDS:
        raise ValueError("an nvfp4-p row is k codes (64 | k), k/16 scale bytes and 64 BF16 words")
    return ([sum((codes[4 * w + q] & 0xF) << (4 * q) for q in range(4)) for w in range(len(codes) // 4)]
            + [(scales[2 * w] & 0xFF) | (scales[2 * w + 1] & 0xFF) << 8 for w in range(len(scales) // 2)] + words_bf16(p))


def words_fp8_p(codes: Sequence[int], p: Sequence[int]) -> list[int]:
    if len(codes) % 4 or len(p) != P_WORDS:
        raise ValueError("an fp8-p row is k codes (4 | k) and 64 BF16 words")
    return [(codes[2 * w] & 0xFF) | (codes[2 * w + 1] & 0xFF) << 8 for w in range(len(codes) // 2)] + words_bf16(p)


def words_tile(digest: bytes) -> list[int]:
    if len(digest) != 32:
        raise ValueError("a tile digest is 32 bytes")
    return [int.from_bytes(digest[2 * i:2 * i + 2], "little") for i in range(16)]


def commit(role: int, words: Sequence[int], salt: bytes) -> bytes:
    """b ‖ c of the row of 16-bit ``words`` under ``salt``: what `Hm96Sha512` computes."""
    return INST.commit_string(KEY, sha512_row_digest(words, 16, role), salt)


def tree_leaf(bc: bytes) -> bytes:
    """The verifier's native step: the frame leaf's value, H(leaf_prefix(key) ‖ b ‖ c)."""
    return INST.tree_leaf(KEY, bc)


__all__ = ["BC", "Hm96Mask", "Hm96Sha512Def", "INST", "KEY", "PearlTileDigestDef", "P_WORDS", "ROW_WORDS", "SALT", "SCHEMAS",
           "bf16_row", "commit", "fp32_row", "fp8_p_row", "halves", "hm96_leaf", "leaf", "nvfp4_p_row", "tile_digest",
           "tile_leaf", "tree_leaf", "words_bf16", "words_fp32", "words_fp8_p", "words_nvfp4_p", "words_tile"]
