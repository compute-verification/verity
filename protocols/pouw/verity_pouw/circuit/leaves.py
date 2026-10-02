"""The tile checks' committed rows (`pc4`, `pc8`): each row a unit reads or writes is a C-Flock-committed port, which C-Flock
commits as an hm96-sha512 row and hashes in the proof (`verity.commitments.hm96`), so the Programs hold no leaf.  What stays
in gates is the work's own digest:

* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.  A hiding tile unit returns
  it in place of its checked words.

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

nvfp4-p's codes and scales are `rowleaf.nvfp4_row_bytes`' bytes.  Which unit ports make up each row is `pc4.PORTS` and
`pc8.PORTS`.  `words_*` are each row's words from its elements, and `commit` and `tree_leaf` the hm96-sha512/row/v1 commit
string and tree leaf of a row; ``tests/vectors/pouw_rows.json`` pins them, against which C-Flock's ports are mapped.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import sha512_row_digest
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.types import Array

from ..schemes import pearl_c as C
from . import hashes as H
from . import words as W

U32 = W.U32
INST = hm96.SHA512
KEY = INST.default_key
DIGEST = Array(8, U32)
P_WORDS = 64
#: each row's schema id, which a registration domain binds with the row's word count (`ROW_WORDS`, at depth k)
SCHEMAS = {"bf16": "pouw/row-bf16/v1", "fp32": "pouw/row-fp32/v1", "nvfp4-p": "pouw/row-nvfp4-p/v1",
           "fp8-p": "pouw/row-fp8-p/v1", "tile": "pouw/tile-h0/v1"}
ROW_WORDS = {"bf16": lambda k: k, "fp32": lambda k: 2 * k, "nvfp4-p": lambda k: k // 4 + k // 32 + P_WORDS,
             "fp8-p": lambda k: k // 2 + P_WORDS, "tile": lambda k: 16}


def _tile_digest_body(B, S_, c, u):
    e = H.Emit(B)
    digests: list = []
    for r in range(S_.TM):
        for h in range(0, S_.TN, C.CHUNK):
            cols = range(h, min(h + C.CHUNK, S_.TN))
            ws = [c[r][j] for j in cols] + [u[r][j] for j in cols]
            digests += H.turboshake128(e, list(zip(ws[0::2], ws[1::2])), 32, C.D_MESSAGE)
    return e.array(H.turboshake128(e, list(zip(digests[0::2], digests[1::2])), 32, C.D_LEAF))


PearlTileDigestDef = CompositeDefinition(
    "PouwPearlTileDigest", 1, ("TM", "TN"),
    lambda S_: ((("c", Array(S_.TM, Array(S_.TN, W.F32))), ("u", Array(S_.TM, Array(S_.TN, W.F32)))), DIGEST),
    _tile_digest_body, doc="pearl_c.leaf (h0) of a TM x TN tile's C̃ and U: its 32 bytes as eight little-endian words.",
    conformance="protocol-defined")


@lru_cache(maxsize=16)
def tile_digest(tm: int, tn: int) -> SpecializedDefinition:
    return bind(PearlTileDigestDef, TM=tm, TN=tn)


def digest_bytes(words: Sequence[int]) -> bytes:
    """`PearlTileDigest`'s output as the digest's 32 bytes."""
    return b"".join((v & W.M32).to_bytes(4, "little") for v in words)


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
    """b ‖ c of the row of 16-bit ``words`` under ``salt`` (hm96-sha512/row/v1 over its `sha512/row/v1` digest)."""
    return INST.commit_string(KEY, sha512_row_digest(words, 16, role), salt)


def tree_leaf(bc: bytes) -> bytes:
    """The verifier's native step: the frame leaf's value, H(leaf_prefix(key) ‖ b ‖ c)."""
    return INST.tree_leaf(KEY, bc)


__all__ = ["DIGEST", "INST", "KEY", "PearlTileDigestDef", "P_WORDS", "ROW_WORDS", "SCHEMAS", "commit", "digest_bytes",
           "tile_digest", "tree_leaf", "words_bf16", "words_fp32", "words_fp8_p", "words_nvfp4_p", "words_tile"]
