"""The tile checks' committed rows (`pc4`, `pc8`): each row a unit reads or writes is a C-Flock-committed port, which C-Flock
commits as an hm96-sha512 row and hashes in the proof (`verity.ml.commitments.hm96`), so the Programs hold no leaf.  What stays
in gates is the work's own digest:

* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.  A hiding tile unit returns
  it in place of its checked words.
* Under -h3 the leaf is ``sha512/row-seg/v1`` over the messages themselves, so the hiding tile (`pc8.Pc8TileHiddenH3`)
  returns no digest: its value is the messages' words (`tile_messages`), which the unit that hashes them reads.

The rows (`SCHEMAS`), each a row of 16-bit words with every narrower element packed low bits first (flock-fp's packed
rows), so the row's bytes are the tensor's memory.  Each is committed under its row's and role's frame-leaf schema
(`ROW_SCHEMA`): an `hm96-sha512/row-seg/v1` row (`sha512/row-seg/v1`: the bit row, 16 bits a word, any length, in 1 KiB
segments that hash in parallel) or an `hm96-sha512/row/v2` row (`sha512/row/v2`: the bit row, 16 bits a word, any length,
serial).  No digest names the row's role or schema, so the registration domain of the tree a row sits in binds the schema
id, the word count and the role, and each tree holds exactly one (schema id, word count, role):

    row (schema id)               words                                              n words            role
    bf16 (pouw/row-bf16/v1)       k BF16 words                                       k                  X (A_i) or W (B_j)
    fp32 (pouw/row-fp32/v1)       k FP32 words, each as two, low half first          2k                 X or W
    nvfp4-p (pouw/row-nvfp4-p/v1) k E2M1 codes four per word (code 4w + q at bit     k/4 + k/32 + 64    X (A′_i ‖ P_A,i) or
                                  4q), k/16 UE4M3 scale bytes two per word (low                         W (B̃_j ‖ P_B,j)
                                  byte first), then P's 64 BF16 words
    fp8-p (pouw/row-fp8-p/v1)     k E4M3 codes two per word (low byte first), then   k/2 + 64           X or W
                                  P's 64 BF16 words
    fp8-pi (pouw/row-fp8-pi/v1)   an fp8-p row, then its index field: the row's      k/2 + 72           X (A′_i ‖ P_A,i ‖ I)
                                  place in its call as a u32 (two words, low first)                     or W (B̃_j ‖ P_B,j ‖ J)
                                  and six zero words (a window's side row,
                                  `window.side_row`)
    ok (pouw/row-ok/v1)           the row's admission bit, bit 0 of one word whose   1                  X (A_i) or W (B_j)
                                  bits 1-15 are zero
    tile (pouw/tile-h0/v1)        the 32-byte tile digest (`PearlTileDigest`), two   16                 X
                                  bytes per word, low byte first
    sd (pouw/row-sd/v1)           a row of A's k/32 salt-dead counts (`cap`), each   k/16               X
                                  32-bit word as two, low half first
    cap-word (pouw/cap-word/v1)   a word's C̃ and U as two words each, low half      4                  X
                                  first (`cap`)
    cap-flags (pouw/cap-flags/v1) a word's k/32 flag bits sixteen per word, low bit  ⌈k/512⌉ + 5        X
                                  first, the last word's high bits zero, then its
                                  flag count and its later +0 promotions as two
                                  words each, then its good bit's ok word (`cap`)
    cap-rows (pouw/cap-rows/v1)   a strip's admission bits sixteen per word, low     64                 X
                                  bit first, 64 bits (TM <= 64, the rest zero),
                                  then the cap's bound as two words, then 58 zero
                                  words (`cap`)
    cap-chunks                    a tile row's two chunk digests, two bytes per      64                 X
      (pouw/cap-chunks/v1)        word, low byte first: `PearlTileDigest`'s
                                  TurboSHAKE128 (D = 0x21) of each 32 columns' C̃
                                  and U (TN <= 64; the second zero at TN <= 32),
                                  then 32 zero words
    cap-sums (pouw/cap-sums/v1)   a tile row's flag count, later +0 promotions and   7                  X
                                  forming elements as two words each, then its
                                  words' good bit's ok word (`cap`)

    row        X (per call)    W (registered)
    bf16       row-seg/v1      row/v2
    fp32       row-seg/v1      row/v2
    nvfp4-p    row-seg/v1      row/v2
    fp8-p      row-seg/v1      row/v2
    fp8-pi     row-seg/v1      row/v2
    sd         row-seg/v1      (none)
    ok         row/v2          row/v2
    tile       row/v2          (none)
    cap-word   row/v2          (none)
    cap-flags  row/v2          (none)
    cap-sums   row/v2          (none)
    cap-rows   row/v2          (none)
    cap-chunks row/v2          (none)

Every per-call row (role X) but ok, the tile digest, the cap's rows and the row_k cut's rows (below) is a row-seg row,
so decode's serial hash chain is a segment's, not the row's (Daniel's ruling, card 637f70c6).  The cap's per-word and
per-row-sum rows are at most one SHA-512 block at any k up to 2^14, so a segment's chain is the row's and v2 hashes them
in fewer blocks.  The cap's strip row and chunk digests are padded with zero words to 64, one whole block, because
C-Flock's composer refuses a unit none of whose rows hashes a block before its last one (its SHA-512 range would hold no
slot), and without a whole block the strip (ok rows in, cap-rows out), a row digest (cap-word rows in, cap-chunks out),
the tile (cap-chunks rows in, the tile digest out) and the tally (cap-sums rows and cap-rows in) would read and write
only one-block v2 rows; a row tally has one through sd.  They are v2 rather than row-seg rows because C-Flock's widened
proved scope takes a v2 row whose bytes are whole blocks or end in its first block, and a row-seg row only of whole
blocks: padded, each is two compressions, as its short row-seg row was, where a row-seg row of a block is three.  Every
other row is a v2 row, the registered ones included, so each (row, role) has one leaf schema in the registered tree and
in every statement that reads or writes it, and no row is `sha512/row/v1` (Daniel's ruling, Slack 1790973124.259289).  A
v2 row costs what a v1 row of the same words would, a 128-byte prefix block and the row; the role byte that v2 drops is
bound by the registration domain.

A row unit cut along k (`rowk`, ``Shape.row_k``) passes its pieces' values in six more rows, v2 in either role, each in the
session of one row's units (k is the segment's length where the row is a segment's):

    row (schema id)                       words                                                      n words
    pc8-head (pouw/row-pc8-head/v1)       α (two words), E′'s 32 E4M3 codes two per word, θ (four    23
                                          words), then a word whose bit 0 is the finite bit, bit 1
                                          the floor bit and bits 2-15 zero
    pc8-carry (pouw/row-pc8-carry/v1)     the dead count so far (two words), the 32 chains' totals   128
                                          so far (FP32 words, two each), then 62 zero words
    fp8-codes (pouw/row-fp8-codes/v1)     a segment's k E4M3 codes two per word                      k/2
    pc4-head (pouw/row-pc4-head/v1)       α (two words), then a word whose low byte is a_E and       3
                                          high byte zero
    pc4-carry (pouw/row-pc4-carry/v1)     the 32 chains' accumulators so far (FP32 words, two each)  64
    nvfp4-codes (pouw/row-nvfp4-codes/v1) a segment's k E2M1 codes four per word, then its k/16      k/4 + k/32
                                          UE4M3 scale bytes two per word

At the deployed row_k (2,048) each of the six is short (one block with SHA-512's padding: at most 55 words) or whole
blocks (a multiple of 64 words), the v2 rows C-Flock's widened scope takes; pc8-carry's zero words are for that.

nvfp4-p's codes and scales are `rowleaf.nvfp4_row_bytes`' bytes, and every row's bytes are `rowleaf.bit_row`'s of its
elements.

Which unit ports make up each row is `pc4.PORTS`, `pc8.PORTS` and the cap's `cap.PORTS`, `rowk.PORTS4` and
`rowk.PORTS8` for the cut, and `pc8.SIDE_PORTS` for the audit window's tile (`pc8.Pc8TileSide`), which reads fp8-pi rows:
its window's side rows, whose index field no unit writes.  `statement_rows` gives a unit's C-Flock rows from these:
META ``row_schemas``, a [schema, bits] per row (the input rows, then the output rows), and ``out_row_ports``, each
output row's run of output ports.  Every output row is a row of exactly its words, as every input row is: the ok word
and the tile digest each one port's v2 row, not padded to whole blocks, and a row a unit writes as several outputs (A′_i
‖ P_A,i or B̃_j ‖ P_B,j: the codes, the scales and P) one run's row-seg or v2 row, the row its readers commit.
`words_*` are each row's words from its elements, and `commit` and `tree_leaf` a row's commit string and tree leaf;
``tests/vectors/pouw_rows.json`` pins them, and C-Flock's tests recompute every row's vector through its ports, commit
each output row as its reader does, and compose each unit's statement rows (``verity/ml/flock/tests/test_pouw_rows.py``).
The anchored inputs are no row: they are public input regions (`anchors.region_words`).

An admission value has one ok leaf only while the word's bits 1-15 are zero, and no row check sees them: C-Flock's and the
Lean verifier's hash a hidden row's bits as given.  So the row unit that writes the row returns the word (`OK`), not the bit,
as a select between the constants 1 and 0, every bit fixed in gates (`pc4.ok_word`), and `ok_of` refuses any other word.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import (ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2, row_bytes,
                                        sha512_row_seg_word_digest, sha512_row_v2_digest)
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value

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
#: an fp8-pi row's index field, as the window's tile unit reads it: eight words, the u32 index low word first, then zeros
INDEX_WORDS = 8
INDEX = Array(INDEX_WORDS, Value(16))
#: a cap-chunks row's chunk digests: a tile row of at most 64 columns (pc8's tile)
CAP_CHUNKS = 2
#: a cap-rows or cap-chunks row's words, its elements then zeros: one whole SHA-512 block
CAP_ROW_WORDS = 64
#: each row's schema id, which a registration domain binds with the row's word count (`ROW_WORDS`, at depth k) and role
SCHEMAS = {"bf16": "pouw/row-bf16/v1", "fp32": "pouw/row-fp32/v1", "nvfp4-p": "pouw/row-nvfp4-p/v1",
           "fp8-p": "pouw/row-fp8-p/v1", "fp8-pi": "pouw/row-fp8-pi/v1", "ok": "pouw/row-ok/v1", "tile": "pouw/tile-h0/v1", "sd": "pouw/row-sd/v1",
           "cap-word": "pouw/cap-word/v1", "cap-flags": "pouw/cap-flags/v1", "cap-rows": "pouw/cap-rows/v1",
           "cap-chunks": "pouw/cap-chunks/v1", "cap-sums": "pouw/cap-sums/v1",
           "pc8-head": "pouw/row-pc8-head/v1", "pc8-carry": "pouw/row-pc8-carry/v1", "fp8-codes": "pouw/row-fp8-codes/v1",
           "pc4-head": "pouw/row-pc4-head/v1", "pc4-carry": "pouw/row-pc4-carry/v1",
           "nvfp4-codes": "pouw/row-nvfp4-codes/v1"}
ROW_WORDS = {"bf16": lambda k: k, "fp32": lambda k: 2 * k, "nvfp4-p": lambda k: k // 4 + k // 32 + P_WORDS,
             "fp8-p": lambda k: k // 2 + P_WORDS, "fp8-pi": lambda k: k // 2 + P_WORDS + INDEX_WORDS, "ok": lambda k: 1, "tile": lambda k: 16, "sd": lambda k: k // 16,
             "cap-word": lambda k: 4, "cap-flags": lambda k: -(-k // 512) + 5, "cap-rows": lambda k: CAP_ROW_WORDS,
             "cap-chunks": lambda k: CAP_ROW_WORDS, "cap-sums": lambda k: 7,
             "pc8-head": lambda k: 23, "pc8-carry": lambda k: 128, "fp8-codes": lambda k: k // 2, "pc4-head": lambda k: 3,
             "pc4-carry": lambda k: 64, "nvfp4-codes": lambda k: k // 4 + k // 32}
_V2, _SEG = SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG
#: each row's frame-leaf schema in a tree of each role it sits in: the per-call rows (role X) segmented, and the registered
#: rows (role W), the admission bit, the tile digest, the cap's rows and the cut's rows v2
ROW_SCHEMA = {("bf16", ROLE_X): _SEG, ("bf16", ROLE_W): _V2, ("fp32", ROLE_X): _SEG, ("fp32", ROLE_W): _V2,
              ("nvfp4-p", ROLE_X): _SEG, ("nvfp4-p", ROLE_W): _V2, ("fp8-p", ROLE_X): _SEG, ("fp8-p", ROLE_W): _V2,
              ("fp8-pi", ROLE_X): _SEG, ("fp8-pi", ROLE_W): _V2,
              ("ok", ROLE_X): _V2, ("ok", ROLE_W): _V2, ("tile", ROLE_X): _V2, ("sd", ROLE_X): _SEG,
              ("cap-word", ROLE_X): _V2, ("cap-flags", ROLE_X): _V2, ("cap-sums", ROLE_X): _V2, ("cap-rows", ROLE_X): _V2,
              ("cap-chunks", ROLE_X): _V2,
              **{(row, role): _V2 for row in ("pc8-head", "pc8-carry", "fp8-codes", "pc4-head", "pc4-carry", "nvfp4-codes")
                 for role in (ROLE_X, ROLE_W)}}

def chunk_digests(e: H.Emit, c, u, tn: int) -> list:
    """`PearlTileDigest`'s digests of one tile row (``c``, ``u``: its TN words' C̃ and U): TurboSHAKE128 (D = 0x21) of each
    32-column chunk, eight half-lanes each."""
    out: list = []
    for h in range(0, tn, C.CHUNK):
        cols = range(h, min(h + C.CHUNK, tn))
        ws = [c[j] for j in cols] + [u[j] for j in cols]
        out += H.turboshake128(e, list(zip(ws[0::2], ws[1::2])), 32, C.D_MESSAGE)
    return out


def leaf_digest(e: H.Emit, digests: Sequence) -> list:
    """`PearlTileDigest`'s h0 from its rows' chunk digests in order: TurboSHAKE128 (D = 0x22) of them."""
    return H.turboshake128(e, list(zip(digests[0::2], digests[1::2])), 32, C.D_LEAF)


def _tile_digest_body(B, S_, c, u):
    e = H.Emit(B)
    digests: list = []
    for r in range(S_.TM):
        digests += chunk_digests(e, c[r], u[r], S_.TN)
    return e.array(leaf_digest(e, digests))


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


def messages_type(tm: int, tn: int) -> Array:
    """-h3's tile value (`tile_messages`): the 2·TM·TN words of a TM x TN tile's messages."""
    return Array(2 * tm * tn, W.F32)


def tile_messages(c, u, tm: int, tn: int):
    """A tile's C̃ and U (``c``, ``u``: `Array<TM, Array<TN, F32>>`) as its messages' words (`messages_type`), wiring only:
    `pearl_c.messages`' order, row-major, per row each 32-column chunk's C̃ words, then its U words.  Each word's bits least
    significant first (a Boolean element's, `boolean.bit_type`) are then ``b"".join(messages)``'s bits, bit i bit i % 8 of
    byte i // 8: the bit row `Sha512RowSeg_v1{N_BITS = 64·TM·TN}` reads (`rowleaf.bit_row`)."""
    return coll_concat(messages_type(tm, tn), [x[i][h:h + C.CHUNK] for i in range(tm) for h in range(0, tn, C.CHUNK)
                                               for x in (c, u)])


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


def words_index(index: int) -> list[int]:
    if not (isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 1 << 32):
        raise ValueError(f"a row's index is a u32, not {index!r}")
    return [index & 0xFFFF, index >> 16] + [0] * (INDEX_WORDS - 2)


def words_fp8_pi(codes: Sequence[int], p: Sequence[int], index: int) -> list[int]:
    return words_fp8_p(codes, p) + words_index(index)


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


def words_sd(counts: Sequence[int]) -> list[int]:
    return words_fp32(counts)


def words_cap_word(c: int, u: int) -> list[int]:
    return words_fp32([c, u])


def words_cap_flags(flags: Sequence[int], atoms: int, zeros: int, good: int) -> list[int]:
    if not flags or any(f not in (0, 1) for f in flags):
        raise ValueError("a cap-flags row is k/32 flag bits, two 32-bit counts and an ok word")
    bits = list(flags) + [0] * (-len(flags) % 16)
    return ([sum(bits[w + q] << q for q in range(16)) for w in range(0, len(bits), 16)] + words_fp32([atoms, zeros])
            + words_ok(good))


def words_cap_rows(oks: Sequence[int], bound: int) -> list[int]:
    if not 0 < len(oks) <= 64 or any(o not in (0, 1) for o in oks):
        raise ValueError("a cap-rows row is a strip's 1 to 64 admission bits, then a 32-bit bound")
    bits = list(oks) + [0] * (64 - len(oks))
    out = [sum(bits[w + q] << q for q in range(16)) for w in range(0, 64, 16)] + words_fp32([bound])
    return out + [0] * (CAP_ROW_WORDS - len(out))


def words_cap_chunks(digests: Sequence[bytes]) -> list[int]:
    if not 0 < len(digests) <= CAP_CHUNKS or any(len(d) != 32 for d in digests):
        raise ValueError(f"a cap-chunks row is a tile row's 1 to {CAP_CHUNKS} 32-byte chunk digests")
    raw = b"".join(digests) + bytes(2 * CAP_ROW_WORDS - 32 * len(digests))
    return [int.from_bytes(raw[2 * i:2 * i + 2], "little") for i in range(CAP_ROW_WORDS)]


def words_cap_sums(atoms: int, zeros: int, elems: int, good: int) -> list[int]:
    return words_fp32([atoms, zeros, elems]) + words_ok(good)


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
    words = _halves(dead, 2) + words_fp32(tot)
    return words + [0] * (ROW_WORDS["pc8-carry"](0) - len(words))


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


def row_digest(row: str, role: int, words: Sequence[int]) -> bytes:
    """x, the digest of a ``row`` row of 16-bit ``words`` in a tree of ``role``, under its schema (`ROW_SCHEMA`): core's
    `sha512/row-seg/v1` or `sha512/row/v2` digest of the words as a bit row, 16 bits a word; an ok row's words, and a
    cap-flags and cap-sums row's last word, pass `ok_of`."""
    schema = ROW_SCHEMA.get((row, role))
    if schema is None:
        raise ValueError(f"no {row} row sits in a tree of role {role}")
    if row in ("ok", "cap-flags", "cap-sums"):
        ok_of(words[-1:] if row != "ok" else words)
    if not words:
        raise ValueError(f"a {row} row is a positive number of words, not 0")
    if schema == _SEG:
        return sha512_row_seg_word_digest(words, 16)
    return sha512_row_v2_digest(row_bytes(words, 16), 16 * len(words))


def commit(row: str, role: int, words: Sequence[int], salt: bytes) -> bytes:
    """b ‖ c of a ``row`` row of 16-bit ``words`` in a tree of ``role`` under ``salt``: hm96-sha512 over its `row_digest`,
    as C-Flock commits the row's port."""
    return INST.commit_string(KEY, row_digest(row, role, words), salt)


def tree_leaf(bc: bytes) -> bytes:
    """The verifier's native step: the frame leaf's value, H(leaf_prefix(key) ‖ b ‖ c)."""
    return INST.tree_leaf(KEY, bc)


def statement_rows(unit: dict, k: int, reads: dict[str, int] | None = None) -> dict:
    """A unit's committed rows as its C-Flock statement's META names them, from its ports (``unit``: `pc4.PORTS[role]` or
    `pc8.PORTS[role]`) at depth ``k``: ``row_schemas``, each row's [schema, bits] (`ROW_SCHEMA`, ``16 ROW_WORDS``), the input
    rows in their order (``reads[p]`` rows of a row whose ports a tile reads one row of each of, row r of each port its r-th)
    then the output rows, each a row of exactly its words, and ``out_row_ports``, each output row's run of output ports,
    which META names when an output row is a row-seg row or a run of several (else every output row is one port's).  The
    output rows' ports, then the public outputs (``unit["public"]``, `cap.PORTS`: the tally's verdict, which the verifier
    reads), are the unit's outputs, once each and in order, so no row here holds a public output.  An advice port
    (``unit["advice"]``, `pc8.PORTS`) is a parameter in no row and no public input region (else ValueError), so no row here
    holds a bit of it."""
    reads = reads or {}
    advice, public = set(unit.get("advice", ())), list(unit.get("public", ()))
    bound = {p for _row, _role, ps in unit["rows"] for p in ps} | set(unit.get("regions", {}))
    if not all(isinstance(p, str) for p in advice) or advice & bound:
        raise ValueError(f"advice ports {sorted(map(str, advice))} are parameters in no committed row and no public "
                         f"input region: not {sorted(map(str, advice & bound)) or 'outputs'}")
    ins, outs, runs = [], [], []
    for row, role, ps in unit["rows"]:
        entry = [ROW_SCHEMA[row, role], 16 * ROW_WORDS[row](k)]
        if all(isinstance(p, str) for p in ps):
            n = {reads.get(p, 1) for p in ps}
            if len(n) != 1:
                raise ValueError(f"a {row} row's ports {ps} are not read the same number of rows")
            ins += [entry] * n.pop()
        elif all(isinstance(p, int) for p in ps):
            outs.append(entry)
            runs.append(list(ps))
        else:
            raise ValueError(f"a {row} row is input ports or output ports, not {ps}")
    if [p for run in runs for p in run] + public != list(range(sum(map(len, runs)) + len(public))):
        raise ValueError(f"the output rows' ports {runs} and public outputs {public} are not the unit's outputs, once each "
                         f"and in order")
    counts = [len(run) for run in runs]
    named = any(s == _SEG for s, _bits in outs) or max(counts, default=1) > 1
    return {"row_schemas": ins + outs, **({"out_row_ports": counts} if named else {})}


__all__ = ["CAP_CHUNKS", "CAP_ROW_WORDS", "DIGEST", "INDEX", "INDEX_WORDS", "INST", "KEY", "OK", "P_WORDS",
           "PearlTileDigestDef", "ROW_SCHEMA", "ROW_WORDS", "SCHEMAS", "chunk_digests", "commit", "digest_bytes", "leaf_digest",
           "messages_type", "ok_of", "row_digest", "statement_rows", "tile_digest", "tile_messages", "tree_leaf", "words_bf16",
           "words_cap_chunks", "words_cap_flags", "words_cap_rows", "words_cap_sums", "words_cap_word", "words_fp32",
           "words_fp8_codes", "words_fp8_p", "words_fp8_pi", "words_index", "words_nvfp4_codes", "words_nvfp4_p", "words_ok", "words_pc4_carry", "words_pc4_head",
           "words_pc8_carry", "words_pc8_head", "words_sd", "words_tile"]
