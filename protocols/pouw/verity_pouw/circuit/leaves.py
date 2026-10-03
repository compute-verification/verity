"""The tile checks' committed rows (`pc4`, `pc8`): each row a unit reads or writes is a C-Flock-committed port, which C-Flock
commits as an hm96-sha512 row and hashes in the proof (`verity.commitments.hm96`), so the Programs hold no leaf.  What stays
in gates is the work's own digest:

* `PearlTileDigest{TM,TN}`: `pearl_c.leaf` (hashing h0) of a tile's C̃ and U: TurboSHAKE128 (D = 0x21) of each row's
  32-column chunks, then TurboSHAKE128 (D = 0x22) of those digests, on `hashes.KeccakP1600R12`.  A hiding tile unit returns
  it in place of its checked words.

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
    ok (pouw/row-ok/v1)           the row's admission bit, bit 0 of one word whose   1                  X (A_i) or W (B_j)
                                  bits 1-15 are zero
    tile (pouw/tile-h0/v1)        the 32-byte tile digest (`PearlTileDigest`), two   16                 X
                                  bytes per word, low byte first

    row        X (per call)    W (registered)
    bf16       row-seg/v1      row/v2
    fp32       row-seg/v1      row/v2
    nvfp4-p    row-seg/v1      row/v2
    fp8-p      row-seg/v1      row/v2
    ok         row/v2          row/v2
    tile       row/v2          (none)

Every per-call row (role X) but ok and the tile digest is a row-seg row, so decode's serial hash chain is a segment's, not
the row's (Daniel's ruling, card 637f70c6).  Every other row is a v2 row, the registered ones included, so each (row, role)
has one leaf schema in the registered tree and in every statement that reads or writes it, and no row is `sha512/row/v1`
(Daniel's ruling, Slack 1790973124.259289).  A v2 row costs what a v1 row of the same words would, a 128-byte prefix block
and the row; the role byte that v2 drops is bound by the registration domain.  nvfp4-p's codes and scales are
`rowleaf.nvfp4_row_bytes`' bytes, and every row's bytes are `rowleaf.bit_row`'s of its elements.

Which unit ports make up each row is `pc4.PORTS` and `pc8.PORTS`, and `statement_rows` gives a unit's C-Flock rows from
them: META ``row_schemas``, a [schema, bits] per row (the input rows, then the output rows), and ``out_row_ports``, each
output row's run of output ports.  Every output row is a row of exactly its words, as every input row is: the ok word and
the tile digest each one port's v2 row, not padded to whole blocks, and a row a unit writes as several outputs (A′_i ‖
P_A,i or B̃_j ‖ P_B,j: the codes, the scales and P) one run's row-seg or v2 row, the row its readers commit.  `words_*`
are each row's words from its elements, and `commit` and `tree_leaf` a row's commit string and tree leaf;
``tests/vectors/pouw_rows.json`` pins them, and C-Flock's tests recompute every row's vector through its ports, commit
each output row as its reader does, and compose each unit's statement rows (``backends/flock/tests/test_pouw_rows.py``).
The anchored inputs are no row: they are public input regions (`anchors.region_words`).

An admission value has one ok leaf only while the word's bits 1-15 are zero, and no row check sees them: C-Flock's and the
Lean verifier's hash a hidden row's bits as given.  So the row unit that writes the row returns the word (`OK`), not the bit,
as a select between the constants 1 and 0, every bit fixed in gates (`pc4.ok_word`), and `ok_of` refuses any other word.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import (ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2, row_bytes,
                                        sha512_row_seg_word_digest, sha512_row_v2_digest)
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
           "fp8-p": "pouw/row-fp8-p/v1", "ok": "pouw/row-ok/v1", "tile": "pouw/tile-h0/v1"}
ROW_WORDS = {"bf16": lambda k: k, "fp32": lambda k: 2 * k, "nvfp4-p": lambda k: k // 4 + k // 32 + P_WORDS,
             "fp8-p": lambda k: k // 2 + P_WORDS, "ok": lambda k: 1, "tile": lambda k: 16}
_V2, _SEG = SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG
#: each row's frame-leaf schema in a tree of each role it sits in: the per-call rows (role X) segmented, and the registered
#: rows (role W), the admission bit and the tile digest v2
ROW_SCHEMA = {("bf16", ROLE_X): _SEG, ("bf16", ROLE_W): _V2, ("fp32", ROLE_X): _SEG, ("fp32", ROLE_W): _V2,
              ("nvfp4-p", ROLE_X): _SEG, ("nvfp4-p", ROLE_W): _V2, ("fp8-p", ROLE_X): _SEG, ("fp8-p", ROLE_W): _V2,
              ("ok", ROLE_X): _V2, ("ok", ROLE_W): _V2, ("tile", ROLE_X): _V2}


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


def row_digest(row: str, role: int, words: Sequence[int]) -> bytes:
    """x, the digest of a ``row`` row of 16-bit ``words`` in a tree of ``role``, under its schema (`ROW_SCHEMA`): core's
    `sha512/row-seg/v1` or `sha512/row/v2` digest of the words as a bit row, 16 bits a word; an ok row's words pass
    `ok_of`."""
    schema = ROW_SCHEMA.get((row, role))
    if schema is None:
        raise ValueError(f"no {row} row sits in a tree of role {role}")
    if row == "ok":
        ok_of(words)
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
    output rows' ports are the unit's outputs, once each and in order.  An advice port (``unit["advice"]``, `pc8.PORTS`)
    is a parameter in no row and no public input region (else ValueError), so no row here holds a bit of it."""
    reads = reads or {}
    advice = set(unit.get("advice", ()))
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
    if [p for run in runs for p in run] != list(range(sum(map(len, runs)))):
        raise ValueError(f"the output rows' ports {runs} are not the unit's outputs, once each and in order")
    counts = [len(run) for run in runs]
    named = any(s == _SEG for s, _bits in outs) or max(counts, default=1) > 1
    return {"row_schemas": ins + outs, **({"out_row_ports": counts} if named else {})}


__all__ = ["DIGEST", "INST", "KEY", "OK", "PearlTileDigestDef", "P_WORDS", "ROW_SCHEMA", "ROW_WORDS", "SCHEMAS", "commit",
           "digest_bytes", "ok_of", "row_digest", "statement_rows", "tile_digest", "tree_leaf", "words_bf16", "words_fp32",
           "words_fp8_p", "words_nvfp4_p", "words_ok", "words_tile"]
