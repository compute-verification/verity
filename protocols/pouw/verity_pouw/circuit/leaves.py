"""The tile checks' hiding leaves in gates (`pc4`, `pc8`): each row a unit reads or writes, and each tile's digest, as
C-Flock commits a row (`verity_flock.circuit.Port`; `verity.commitments.rowleaf`, `verity.commitments.hm96`), so a unit
publishes a commit string b ‖ c where the scheme published a row.

* `Hm96Sha512{ROW,WORDS}`: one row's b ‖ c from the row and its 192-byte salt.  x = SHA-512(prefix ‖ the row's WORDS 16-bit
  words, little-endian) from the prefix's constant midstate, the prefix `sha512_row_prefix`(role X, 16, WORDS) for ROW 1
  (`sha512/row/v1`) or `sha512_row_v2_prefix`(16 WORDS) for ROW 2 (`sha512/row/v2`); c = SHA-512(salt prefix ‖ y) from its
  midstate (two compressions); b = x ⊕ M·y (`Hm96Mask`).  The verifier computes the tree leaf, H(leaf_prefix(key) ‖ b ‖ c),
  itself.  The key is `hm96.SHA512`'s pinned key, a constant of the gates.
* `Hm96Mask`: M·y for that key, 16 × 48 blocks of `words.PouwHm96Hankel` and their XORs, no AND.
* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.

The rows (`SCHEMAS`), each a row of 16-bit words with every narrower element packed low bits first (flock-fp's packed
rows), so the row's bytes are the tensor's memory.  Each is the row C-Flock commits from the same words (`ROW_SCHEMA`):
an `hm96-sha512/row/v1` row (`sha512/row/v1`, role X and 16-bit words, as every C-Flock v1 port is, of whole SHA-512
blocks) or an `hm96-sha512/row/v2` bit row (`sha512/row/v2`, 16 bits a word, any length).  Neither digest names the role
or the schema, so the registration domain of the tree a row sits in binds the schema id, the word count and the role:

    row (schema id)               words                                              n words            role       C-Flock row
    bf16 (pouw/row-bf16/v1)       k BF16 words                                       k                  X (A_i) or v1 (64 | k)
                                                                                                        W (B_j)
    fp32 (pouw/row-fp32/v1)       k FP32 words, each as two, low half first          2k                 X or W     v1 (32 | k)
    nvfp4-p (pouw/row-nvfp4-p/v1) k E2M1 codes four per word (code 4w + q at bit     k/4 + k/32 + 64    X (A′_i ‖  v2
                                  4q), k/16 UE4M3 scale bytes two per word (low                         P_A,i) or W
                                  byte first), then P's 64 BF16 words                                   (B̃_j ‖ P_B,j)
    fp8-p (pouw/row-fp8-p/v1)     k E4M3 codes two per word (low byte first), then   k/2 + 64           X or W     v2
                                  P's 64 BF16 words
    tile (pouw/tile-h0/v1)        the 32-byte tile digest (`PearlTileDigest`), two   16                 X          v2
                                  bytes per word, low byte first

nvfp4-p's codes and scales are `rowleaf.nvfp4_row_bytes`' bytes, and every row's bytes are `rowleaf.bit_row`'s of its
elements.  `words_*` and `commit` are the reference; ``tests/vectors/pouw_rows.json`` pins them, and C-Flock's tests
recompute every vector through its ports (``backends/flock/tests/test_pouw_rows.py``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.commitments import hm96
from verity.commitments.frame_v3 import BitRowLeaf, RowLeaf
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, ROLE_X, row_bytes
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
#: each row's frame-leaf schema as C-Flock commits it: rows of 16-bit elements v1, packed rows and the tile digest v2
ROW_SCHEMA = {"bf16": SCHEMA_HM96_SHA512_ROW, "fp32": SCHEMA_HM96_SHA512_ROW, "nvfp4-p": SCHEMA_HM96_SHA512_ROW_V2,
              "fp8-p": SCHEMA_HM96_SHA512_ROW_V2, "tile": SCHEMA_HM96_SHA512_ROW_V2}
#: `Hm96Sha512`'s ROW for each frame-leaf schema
ROW_VERSION = {SCHEMA_HM96_SHA512_ROW: 1, SCHEMA_HM96_SHA512_ROW_V2: 2}


def row_leaf(row: str, n_words: int) -> RowLeaf | BitRowLeaf:
    """The leaf kind of a ``row`` row of ``n_words`` 16-bit words, as C-Flock's port of ``n_words`` u16 leaves holds it: a
    `sha512/row/v1` row of role X and whole SHA-512 blocks, or a `sha512/row/v2` bit row of 16 ``n_words`` bits."""
    if n_words <= 0:
        raise ValueError(f"a {row} row is a positive number of words, not {n_words}")
    if ROW_SCHEMA[row] == SCHEMA_HM96_SHA512_ROW_V2:
        return BitRowLeaf(16 * n_words)
    if 2 * n_words % 128:
        raise ValueError(f"a {row} row ({SCHEMA_HM96_SHA512_ROW}) is whole SHA-512 blocks, not {2 * n_words} bytes")
    return RowLeaf("sha512", ROLE_X, 16, n_words)


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


def _row_prefix(version: int, words: int) -> bytes:
    return (RowLeaf("sha512", ROLE_X, 16, words) if version == 1 else BitRowLeaf(16 * words)).layout().prefix


def _hm96_body(B, S_, row, salt):
    e, n = H.Emit(B), S_.WORDS // 2
    x = _sha512_after(e, _midstate(_row_prefix(S_.ROW, S_.WORDS)), [row[i] for i in range(n)], 128 + 2 * S_.WORDS)
    c = _sha512_after(e, _midstate(INST.salt_prefix), [salt[i] for i in range(SALT.n)], 128 + INST.salt_bytes)
    m = B.call(Hm96Mask, salt)
    return e.array([e.call(W.PouwXor32, x[i], m[i]) for i in range(16)] + c)


Hm96Sha512Def = CompositeDefinition(
    "PouwHm96Sha512", 1, ("ROW", "WORDS"), lambda S_: ((("row", Array(S_.WORDS // 2, U32)), ("salt", SALT)), BC),
    _hm96_body, doc="A row's hm96-sha512 commit string b ‖ c (big-endian halves) as C-Flock commits it: the row is WORDS "
                    "16-bit words as big-endian halves of its little-endian bytes, x its sha512/row/v1 digest (role X, "
                    "16-bit words) for ROW 1 or its sha512/row/v2 digest (16 WORDS bits) for ROW 2, and the salt 192 bytes.",
    conformance="protocol-defined")


@lru_cache(maxsize=64)
def hm96_leaf(version: int, words: int) -> SpecializedDefinition:
    if version not in (1, 2) or words <= 0 or words % 2:
        raise ValueError(f"an hm96 row here is sha512/row/v1 or v2 and an even number of 16-bit words, not "
                         f"({version}, {words})")
    return bind(Hm96Sha512Def, ROW=version, WORDS=words)


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


def leaf(e, schema: str, row: list, salt):
    """The commit string of the ``schema`` row (a key of `SCHEMAS`) whose big-endian halves are ``row``."""
    row_leaf(schema, 2 * len(row))
    return e.B.call(hm96_leaf(ROW_VERSION[ROW_SCHEMA[schema]], 2 * len(row)), array_of(row), salt)


def tile_leaf(e, tm: int, tn: int, c, u, salt):
    """The commit string of the tile digest of C̃ ``c`` and U ``u`` (each TM x TN FP32 words)."""
    d = e.B.call(tile_digest(tm, tn), c, u)
    return leaf(e, "tile", [d[i] for i in range(8)], salt)


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


def row_digest(schema: str, words: Sequence[int]) -> bytes:
    """x, the ``schema`` row's digest of its 16-bit ``words`` (`row_leaf`'s layout)."""
    return row_leaf(schema, len(words)).layout().digest(row_bytes(words, 16))


def commit(schema: str, words: Sequence[int], salt: bytes) -> bytes:
    """b ‖ c of the ``schema`` row of 16-bit ``words`` under ``salt``: what C-Flock commits and `Hm96Sha512` computes."""
    return INST.commit_string(KEY, row_digest(schema, words), salt)


def tree_leaf(bc: bytes) -> bytes:
    """The verifier's native step: the frame leaf's value, H(leaf_prefix(key) ‖ b ‖ c)."""
    return INST.tree_leaf(KEY, bc)


__all__ = ["BC", "Hm96Mask", "Hm96Sha512Def", "INST", "KEY", "PearlTileDigestDef", "P_WORDS", "ROW_SCHEMA", "ROW_VERSION",
           "ROW_WORDS", "SALT", "SCHEMAS", "bf16_row", "commit", "fp32_row", "fp8_p_row", "halves", "hm96_leaf", "leaf",
           "nvfp4_p_row", "row_digest", "row_leaf", "tile_digest", "tile_leaf", "tree_leaf", "words_bf16", "words_fp32",
           "words_fp8_p", "words_nvfp4_p", "words_tile"]
