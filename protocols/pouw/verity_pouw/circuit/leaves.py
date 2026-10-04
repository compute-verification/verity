"""The tile checks' committed rows (`pc4`, `pc8`): each row a unit reads or writes is a C-Flock-committed port, which C-Flock
commits as an hm96-sha512 row and hashes in the proof (`verity.commitments.hm96`), so the Programs hold no leaf.  What stays
in gates is the work's own digest:

* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.  A hiding tile unit returns
  it in place of its checked words.

The rows (`SCHEMAS`), each a row of 16-bit words with every narrower element packed low bits first (flock-fp's packed
rows), so the row's bytes are the tensor's memory.  Each is the row C-Flock's port of its words commits (`ROW_SCHEMA`): an
`hm96-sha512/row/v1` row (`sha512/row/v1`, role X and 16-bit words, as every C-Flock v1 port is, of whole SHA-512 blocks)
or an `hm96-sha512/row/v2` bit row (`sha512/row/v2`, 16 bits a word, any length).  Neither digest names the role or the
schema, so the registration domain of the tree a row sits in binds the schema id, the word count and the role:

    row (schema id)               words                                              n words            role
    bf16 (pouw/row-bf16/v1)       k BF16 words                                       k                  X (A_i) or W (B_j)
    fp32 (pouw/row-fp32/v1)       k FP32 words, each as two, low half first          2k                 X or W
    nvfp4-p (pouw/row-nvfp4-p/v1) k E2M1 codes four per word (code 4w + q at bit     k/4 + k/32 + 64    X (A′_i ‖ P_A,i) or
                                  4q), k/16 UE4M3 scale bytes two per word (low                         W (B̃_j ‖ P_B,j)
                                  byte first), then P's 64 BF16 words
    fp8-p (pouw/row-fp8-p/v1)     k E4M3 codes two per word (low byte first), then   k/2 + 64           X or W
                                  P's 64 BF16 words
    ok (pouw/row-ok/v1)           the row's admission bit, bit 0 of one word whose   1                  X (A_i) or W (B_j)
                                  bits 1-15 are zero
    tile (pouw/tile-h0/v1)        the 32-byte tile digest (`PearlTileDigest`), two   16                 X
                                  bytes per word, low byte first

bf16 and fp32 are v1 rows (whole blocks: 64 | k and 32 | k), nvfp4-p, fp8-p, ok and tile v2 rows.

A row unit cut along k (`rowk`, ``Shape.row_k``) passes its pieces' values in six more rows, all v2, each in the session
of one row's units (k is the segment's length where the row is a segment's):

    row (schema id)                       words                                                      n words
    pc8-head (pouw/row-pc8-head/v1)       α (two words), E′'s 32 E4M3 codes two per word, θ (four    23
                                          words), then a word whose bit 0 is the finite bit, bit 1
                                          the floor bit and bits 2-15 zero
    pc8-carry (pouw/row-pc8-carry/v1)     the dead count so far (two words), then the 32 chains'     66
                                          totals so far (FP32 words, two each)
    fp8-codes (pouw/row-fp8-codes/v1)     a segment's k E4M3 codes two per word                      k/2
    pc4-head (pouw/row-pc4-head/v1)       α (two words), then a word whose low byte is a_E and       3
                                          high byte zero
    pc4-carry (pouw/row-pc4-carry/v1)     the 32 chains' accumulators so far (FP32 words, two each)  64
    nvfp4-codes (pouw/row-nvfp4-codes/v1) a segment's k E2M1 codes four per word, then its k/16      k/4 + k/32
                                          UE4M3 scale bytes two per word

nvfp4-p's codes and scales are `rowleaf.nvfp4_row_bytes`' bytes, and every row's bytes are `rowleaf.bit_row`'s of its
elements.  Which unit ports make up each row is `pc4.PORTS` and `pc8.PORTS`, and `rowk.PORTS4` and `rowk.PORTS8` for the
cut.  `words_*` are each row's words from its elements, and `commit` and
`tree_leaf` a row's commit string and tree leaf; ``tests/vectors/pouw_rows.json`` pins them, and C-Flock's tests recompute
every vector through its ports (``backends/flock/tests/test_pouw_rows.py``).  The anchored inputs are no row: they are
public input regions (`anchors.region_words`).

An admission value has one ok leaf only while the word's bits 1-15 are zero, and no row check sees them: C-Flock's and the
Lean verifier's hash a hidden row's bits as given.  So the row unit that writes the row returns the word (`OK`), not the bit,
as a select between the constants 1 and 0, every bit fixed in gates (`pc4.ok_word`), and `ok_of` refuses any other word.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.commitments import hm96
from verity.commitments.frame_v3 import BitRowLeaf, RowLeaf
from verity.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, row_bytes
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.types import Array, Value

from ..schemes import pearl_c as C
from . import hashes as H
from . import words as W

U32 = W.U32
INST = hm96.SHA512
KEY = INST.default_key
DIGEST = Array(8, U32)
#: an ok row's one word, as its row unit returns it: the admission bit at bit 0, bits 1-15 zero
OK = Value(16)
P_WORDS = 64
#: each row's schema id, which a registration domain binds with the row's word count (`ROW_WORDS`, at depth k) and role
SCHEMAS = {"bf16": "pouw/row-bf16/v1", "fp32": "pouw/row-fp32/v1", "nvfp4-p": "pouw/row-nvfp4-p/v1",
           "fp8-p": "pouw/row-fp8-p/v1", "ok": "pouw/row-ok/v1", "tile": "pouw/tile-h0/v1",
           "pc8-head": "pouw/row-pc8-head/v1", "pc8-carry": "pouw/row-pc8-carry/v1", "fp8-codes": "pouw/row-fp8-codes/v1",
           "pc4-head": "pouw/row-pc4-head/v1", "pc4-carry": "pouw/row-pc4-carry/v1",
           "nvfp4-codes": "pouw/row-nvfp4-codes/v1"}
ROW_WORDS = {"bf16": lambda k: k, "fp32": lambda k: 2 * k, "nvfp4-p": lambda k: k // 4 + k // 32 + P_WORDS,
             "fp8-p": lambda k: k // 2 + P_WORDS, "ok": lambda k: 1, "tile": lambda k: 16,
             "pc8-head": lambda k: 23, "pc8-carry": lambda k: 66, "fp8-codes": lambda k: k // 2, "pc4-head": lambda k: 3,
             "pc4-carry": lambda k: 64, "nvfp4-codes": lambda k: k // 4 + k // 32}
#: each row's frame-leaf schema as C-Flock commits it: rows of 16-bit elements v1, packed rows, the admission bit, the
#: tile digest and the cut's rows v2
ROW_SCHEMA = {"bf16": SCHEMA_HM96_SHA512_ROW, "fp32": SCHEMA_HM96_SHA512_ROW, "nvfp4-p": SCHEMA_HM96_SHA512_ROW_V2,
              "fp8-p": SCHEMA_HM96_SHA512_ROW_V2, "ok": SCHEMA_HM96_SHA512_ROW_V2, "tile": SCHEMA_HM96_SHA512_ROW_V2,
              **dict.fromkeys(("pc8-head", "pc8-carry", "fp8-codes", "pc4-head", "pc4-carry", "nvfp4-codes"),
                              SCHEMA_HM96_SHA512_ROW_V2)}


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


def words_ok(ok: int) -> list[int]:
    if ok not in (0, 1):
        raise ValueError("an ok row is one admission bit")
    return [ok]


def ok_of(words: Sequence[int]) -> int:
    """The admission bit of an ok row's words, refusing any row but one word with bits 1-15 zero."""
    if len(words) != 1 or words[0] not in (0, 1):
        raise ValueError("an ok row is one word: the admission bit at bit 0, bits 1-15 zero")
    return words[0]


def words_tile(digest: bytes) -> list[int]:
    if len(digest) != 32:
        raise ValueError("a tile digest is 32 bytes")
    return [int.from_bytes(digest[2 * i:2 * i + 2], "little") for i in range(16)]


def _halves(v: int, n: int) -> list[int]:
    return [(v >> (16 * h)) & 0xFFFF for h in range(n)]


def words_pc8_head(alpha: int, ep: Sequence[int], th: int, finite: int, floor: int) -> list[int]:
    if len(ep) != C.R or finite not in (0, 1) or floor not in (0, 1):
        raise ValueError("a pc8-head row is α, E′'s 32 codes, θ and the finite and floor bits")
    return _halves(alpha, 2) + [(ep[2 * w] & 0xFF) | (ep[2 * w + 1] & 0xFF) << 8 for w in range(C.R // 2)] + \
        _halves(th, 4) + [finite | floor << 1]


def words_pc8_carry(dead: int, tot: Sequence[int]) -> list[int]:
    if len(tot) != C.R:
        raise ValueError("a pc8-carry row is the dead count and 32 FP32 totals")
    return _halves(dead, 2) + words_fp32(tot)


def words_fp8_codes(codes: Sequence[int]) -> list[int]:
    if len(codes) % 2:
        raise ValueError("an fp8-codes row is an even number of codes")
    return [(codes[2 * w] & 0xFF) | (codes[2 * w + 1] & 0xFF) << 8 for w in range(len(codes) // 2)]


def words_pc4_head(alpha: int, ae: int) -> list[int]:
    return _halves(alpha, 2) + [ae & 0xFF]


def words_pc4_carry(tot: Sequence[int]) -> list[int]:
    if len(tot) != C.R:
        raise ValueError("a pc4-carry row is 32 FP32 accumulators")
    return words_fp32(tot)


def words_nvfp4_codes(codes: Sequence[int], scales: Sequence[int]) -> list[int]:
    if len(codes) % 32 or len(scales) * 16 != len(codes):
        raise ValueError("an nvfp4-codes row is k codes (32 | k) and k/16 scale bytes")
    return ([sum((codes[4 * w + q] & 0xF) << (4 * q) for q in range(4)) for w in range(len(codes) // 4)]
            + [(scales[2 * w] & 0xFF) | (scales[2 * w + 1] & 0xFF) << 8 for w in range(len(scales) // 2)])


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


def row_digest(row: str, words: Sequence[int]) -> bytes:
    """x, the ``row`` row's digest of its 16-bit ``words`` (`row_leaf`'s layout); an ok row's words pass `ok_of`."""
    if row == "ok":
        ok_of(words)
    return row_leaf(row, len(words)).layout().digest(row_bytes(words, 16))


def commit(row: str, words: Sequence[int], salt: bytes) -> bytes:
    """b ‖ c of the ``row`` row of 16-bit ``words`` under ``salt``: hm96-sha512 over its `row_digest`, as C-Flock commits
    the row's port."""
    return INST.commit_string(KEY, row_digest(row, words), salt)


def tree_leaf(bc: bytes) -> bytes:
    """The verifier's native step: the frame leaf's value, H(leaf_prefix(key) ‖ b ‖ c)."""
    return INST.tree_leaf(KEY, bc)


__all__ = ["DIGEST", "INST", "KEY", "OK", "PearlTileDigestDef", "P_WORDS", "ROW_SCHEMA", "ROW_WORDS", "SCHEMAS", "commit",
           "digest_bytes", "ok_of", "row_digest", "row_leaf", "tile_digest", "tree_leaf", "words_bf16", "words_fp32",
           "words_fp8_codes", "words_fp8_p", "words_nvfp4_codes", "words_nvfp4_p", "words_ok", "words_pc4_carry",
           "words_pc4_head", "words_pc8_carry", "words_pc8_head", "words_tile"]
