"""The embedding's row read at a hidden position: `Embedding_v1{V, H}`'s function on bits with its table a registered value,
so a unit reads the token's row alone (its row digest, its hm96 leaf and one frame-v3 read), never the table, and never opens
the token's position in the clear.

    EmbeddingRead_v2{V, H}
        (tok: Value<32>, root: Digest, domain: Digest, row: Array<H, Array<16, Value<1>>>, salt: Array<24, Word>,
         path: Array<D, Digest>) -> Tuple<Array<H, Value<16>>, Value<1>>                                             (h, ok)
        Digest = Array<8, Word>, Word = Array<64, Value<1>>; D = merkle_depth(V); 2 <= V <= 2^31, H >= 1

``tok`` and ``h`` are words at the widths of the token's writers and the row's readers; their bits are wiring inside
(`WordBits_v1{32}`, `BitsWord_v1{16}`), in the order of the bits the rest of the read takes, so a proof block holding the
read has the rows `Flock.ReadNet.readTmpl` builds.

**The table** is registered one token a row (`one_stage.registered` with ``words = H`` 16-bit leaves under
``hm96-sha512/row-seg/v1``): row r is token r's H bf16 words as a bit row (`rowleaf.bit_row`, each word little-endian), its
``sha512/row-seg/v1`` digest hidden under its 192-byte salt (`gates.hm96`), and the leaves a dense frame-v3-sha512 tree of V
rows, so a row's rank and position are its token.

**What v2 computes.** With in = (tok < V, tok read as an unsigned 32-bit integer, so a negative i32 is at least 2^31):

    h    = row if in, else 0x7FFF in every word        `Embedding_v1`: h[j] = table[tok][j], 0x7FFF outside [0, V)
    leaf = Hm96Sha512Leaf_v1(Sha512RowSeg_v1{16 H}(row), salt)[2]
    ok   = not in, or FrameV3Read_v1{D, hm96-sha512/row-seg/v1}(root, domain, leaf, tok mod 2^D, path)

`EmbeddingRead_v4{V, H}` (`opening`) takes the same arguments and returns ok as a 16-bit verdict word, ok = in and
FrameV3Read_v1(...), and h = row if ok, else 0x7FFF in every word: a key outside [0, V) and a row that doesn't open there
both give ok = 0 and the fill, so a prover may make any key read the fill row, and never a row of its choice.

So when ok = 1 and tok is in [0, V), h is the row registered at position tok under ``root`` and ``domain``: the leaf binds
the row's digest under the salt (hm96's binding), tok as its rank and position, the schema and the domain id, and the path
climbs to ``root``. A unit that holds the read takes ``root`` and ``domain`` as the registered value's root and domain id,
each a registered read of that root (a supplied value, `one_stage.registered`), never a free input, and publishes ``ok``.
``tok``, ``row``, ``salt`` and ``path`` are values and ``h`` an output value, so neither the token nor its row is opened
outside the proof. A Definition has no assertion, so the read returns its verdict as ``ok``, as
`block_check.BlockOpeningCheck` returns its verdict word.

**The parts:** `Sha512RowSeg_v1{16 H}` (`gates.rowleaf`), `Hm96Sha512Leaf_v1` (`gates.hm96`),
`FrameV3Read_v1{D, hm96-sha512/row-seg/v1}` (`gates.frame_v3`) and

    EmbeddingReadSelect_v1{V, H}   (tok: Array<32, Value<1>>, opens: Value<1>, row: Array<H, Array<16, Value<1>>>) -> (h, ok)
                                   in by at most an AND a bit of tok (23 at V = 128,256), then an AND a bit
                                   of h and one for ok

**Cost** at V = 128,256 (D = 17): the row digest is nine compressions an 8,192-bit segment and s / 2 + 1 for the outer hash
of s segments (39 at H = 2,048, 77 at H = 4,096), the hm96 leaf four, the read's leaf two and its 17 nodes three each:
5,392,976 ANDs of 32,322,856 gates at H = 2,048 and 7,574,672 of 45,294,886 at H = 4,096. ``READS`` in the tests pins the
gates.
"""
from __future__ import annotations

from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE, ZERO, const_bits
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.refs import Coll, coll_concat
from verity.circuits.types import Array, Tuple, Value

from ..merkle import merkle_depth
from ..rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, bit_row
from .frame_v3 import opens
from .frame_v3 import read as frame_read
from .hm96 import SALT, Hm96Sha512Leaf
from .hm96 import reference as hm96_leaf
from .rowleaf import DIGEST, digest, digest_bits
from .rowleaf import reference as row_digest

__all__ = ["EmbeddingRead", "EmbeddingReadGuard", "EmbeddingReadSelect", "EmbeddingReadWord", "FILL", "SCHEMA", "arguments",
           "opening", "read", "reference", "reference_opening"]

BIT = BL.BIT
BF16 = Array(16, BIT)
TOKEN = Array(32, BIT)
WORD = Value(16)
SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG
#: the word a token outside the vocabulary gives (`Embedding_v2`'s gathers)
FILL = 0x7FFF
MAX_V = 1 << 31
SALT_BYTES = SALT.leaves // 8


def _check(V: int, H: int) -> int:
    """D, the tree's depth; refuses a malformed binding."""
    if isinstance(V, bool) or not isinstance(V, int) or not 2 <= V <= MAX_V:
        raise ValueError(f"an embedding read's V is an int in 2 ... 2^31, not {V!r}")
    if isinstance(H, bool) or not isinstance(H, int) or H < 1:
        raise ValueError(f"an embedding read's H is a positive int, not {H!r}")
    return merkle_depth(V)


# --- reference ---------------------------------------------------------------------------------------------------------------

def reference(V: int, tok: int, root: bytes, domain_id: bytes, row: Sequence[int], salt: bytes,
              path: Sequence[bytes]) -> tuple[list[int], int]:
    """`EmbeddingRead_v2{V, H = len(row)}`'s reference: (h, ok) for a 32-bit token (signed or unsigned) and bf16 words."""
    _check(V, len(row))
    t = tok & 0xFFFFFFFF
    if t >= V:
        return [FILL] * len(row), 1
    leaf = hm96_leaf(row_digest(*bit_row((row, 16))), salt)[2]
    return list(row), opens(SCHEMA, root, domain_id, leaf, t, path)


def reference_opening(V: int, tok: int, root: bytes, domain_id: bytes, row: Sequence[int], salt: bytes,
                      path: Sequence[bytes]) -> tuple[list[int], int]:
    """`EmbeddingRead_v4{V, H = len(row)}`'s reference: (h, ok), ``ok`` the verdict word, 1 iff the token is below V and
    the row opens there, else 0; h the row if ok, else the fill."""
    h, ok = reference(V, tok, root, domain_id, row, salt, path)
    ok = int((tok & 0xFFFFFFFF) < V and ok == 1)
    return (h if ok else [FILL] * len(row)), ok


def arguments(V: int, tok: int, root: bytes, domain_id: bytes, row: Sequence[int], salt: bytes,
              path: Sequence[bytes]) -> tuple[list[int], ...]:
    """`EmbeddingRead_v2{V, H = len(row)}`'s six arguments, each a flat list of its bits, each word's least significant first."""
    D = _check(V, len(row))
    if len(path) != D:
        raise ValueError(f"a read of {V} rows takes a path of {D} siblings, not {len(path)}")
    if len(salt) != SALT_BYTES:
        raise ValueError(f"a salt is {SALT_BYTES} bytes, not {len(salt)}")
    if any(not 0 <= int(w) < 1 << 16 for w in row):
        raise ValueError("a row's words are 16 bits")
    return ([(tok >> j) & 1 for j in range(32)], digest_bits(root), digest_bits(domain_id),
            [(int(w) >> j) & 1 for w in row for j in range(16)],
            [b for k in range(0, SALT_BYTES, 64) for b in digest_bits(salt[k:k + 64])], [b for s in path for b in digest_bits(s)])


# --- the selection -----------------------------------------------------------------------------------------------------------

def _below(C, bits: Sequence, bound: int):
    """1 iff the unsigned ``bits`` (LSB first) are below the constant ``bound``: an AND a bit past bound's lowest set bit."""
    lt = ZERO
    for k, b in enumerate(bits):
        lt = C.AND(b, lt.inv()).inv() if bound >> k & 1 else C.AND(b.inv(), lt)
    return lt


def _select(C, S, tok, opened, *row) -> list:
    inside = _below(C, tok, S.V)
    fill = const_bits(FILL, 16)
    h = [[f ^ C.AND(inside, r ^ f) for r, f in zip(word, fill)] for word in row]
    return [*h, [ONE ^ C.AND(inside, opened[0] ^ ONE)]]


def _sig_select(S):
    _check(S.V, S.H)
    return (("tok", TOKEN), ("opens", BIT), ("row", Array(S.H, BF16))), Tuple(Array(S.H, BF16), BIT)


EmbeddingReadSelect = _part(
    "EmbeddingReadSelect", ("V", "H"), _sig_select, _select,
    "(h, ok): h = row if tok < V (unsigned), else 0x7FFF in every word; ok = opens, or 1 when tok >= V")


def _guard(C, S, tok, opened) -> list:
    inside = _below(C, tok, S.V)
    return [[inside], [C.AND(inside, opened[0])] + [ZERO] * 15]


def _sig_guard(S):
    _check(S.V, 1)
    return (("tok", TOKEN), ("opens", BIT)), Tuple(BIT, Array(16, BIT))


EmbeddingReadGuard = _part(
    "EmbeddingReadGuard", ("V",), _sig_guard, _guard,
    "(in, ok): in = tok < V (unsigned); ok = in AND opens, the bit of a verdict word, its other 15 bits zero")


def _word(C, S, inside, word) -> list:
    return [f ^ C.AND(inside[0], r ^ f) for r, f in zip(word, const_bits(FILL, 16))]


EmbeddingReadWord = _part(
    "EmbeddingReadWord", (), lambda S: ((("in", BIT), ("word", BF16)), BF16), _word,
    "one word of the row if in, else 0x7FFF: a copy of the selection a word, so a batch over the row steps whole copies")


# --- the read ----------------------------------------------------------------------------------------------------------------

def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _signature(S):
    D = _check(S.V, S.H)
    return ((("tok", Value(32)), ("root", DIGEST), ("domain", DIGEST), ("row", Array(S.H, BF16)), ("salt", SALT),
             ("path", Array(D, DIGEST))), Tuple(Array(S.H, WORD), BIT))


def _body(B, S, tok, root, domain, row, salt, path):
    D = merkle_depth(S.V)
    bits = B.call(bind(WordBits, W=32), tok)
    x = B.call(digest(16 * S.H), Coll(Array(16 * S.H, BIT), row.refs))
    leaf = Coll(DIGEST, TR._refs(_refs(B.call(Hm96Sha512Leaf, x, salt))[2 * DIGEST.leaves:]))
    pos = Coll(Array(D, BIT), TR._refs(_refs(bits)[:D]))
    opened = B.call(frame_read(D, SCHEMA), root, domain, leaf, pos, path)
    h, ok = B.call(bind(EmbeddingReadSelect, V=S.V, H=S.H), bits, opened, row)
    return coll_concat(Tuple(Array(S.H, WORD), BIT), [B.batch(bind(BitsWord, W=16), h, axes=(0,)), ok])


EmbeddingRead = CompositeDefinition(
    "EmbeddingRead", 2, ("V", "H"), _signature, _body, conformance="exact-model-tested",
    doc="Embedding_v1{V, H} with its table registered one token a row under hm96-sha512/row-seg/v1: (h, ok), h the witness row "
        "if tok < V (unsigned) else 0x7FFF in every word, ok = 1 iff tok >= V or the row under salt opens at position tok to "
        "root in the dense frame-v3-sha512 tree of domain id domain (Sha512RowSeg_v1, Hm96Sha512Leaf_v1, FrameV3Read_v1)")


def read(V: int, H: int):
    return bind(EmbeddingRead, V=V, H=H)


def _signature_v4(S):
    params, _ = _signature(S)
    return params, Tuple(Array(S.H, WORD), WORD)


def _body_v4(B, S, tok, root, domain, row, salt, path):
    D = merkle_depth(S.V)
    bits = B.call(bind(WordBits, W=32), tok)
    x = B.call(digest(16 * S.H), Coll(Array(16 * S.H, BIT), row.refs))
    leaf = Coll(DIGEST, TR._refs(_refs(B.call(Hm96Sha512Leaf, x, salt))[2 * DIGEST.leaves:]))
    pos = Coll(Array(D, BIT), TR._refs(_refs(bits)[:D]))
    opened = B.call(frame_read(D, SCHEMA), root, domain, leaf, pos, path)
    _inside, ok = B.call(bind(EmbeddingReadGuard, V=S.V), bits, opened)
    h = B.batch(bind(EmbeddingReadWord), Coll(BIT, TR._refs(_refs(ok)[:1])), row, axes=(None, 0))
    return coll_concat(Tuple(Array(S.H, WORD), WORD),
                       [B.batch(bind(BitsWord, W=16), h, axes=(0,)), B.call(bind(BitsWord, W=16), ok)])


EmbeddingRead_v4 = CompositeDefinition(
    "EmbeddingRead", 4, ("V", "H"), _signature_v4, _body_v4, conformance="exact-model-tested",
    doc="a table's row read at a hidden key, its table registered one key a row under hm96-sha512/row-seg/v1: (h, ok), ok the "
        "verdict word, 1 iff tok < V (unsigned) and the row under salt opens at position tok to root in the dense "
        "frame-v3-sha512 tree of domain id domain, else 0; h the table's row at tok where it opens so (ok = 1), else the "
        "constant 0x7FFF in every word, so a prover may make any key read the fill row, and no key reads a row of the "
        "prover's choice (Sha512RowSeg_v1, Hm96Sha512Leaf_v1, FrameV3Read_v1, EmbeddingReadGuard_v1, EmbeddingReadWord_v1, "
        "BitsWord_v1{16})")


def opening(V: int, H: int):
    """`EmbeddingRead_v4{V, H}`: a read whose verdict is a word, 0 for a key outside [0, V) or a row that doesn't open there,
    and whose row is then the fill."""
    return bind(EmbeddingRead_v4, V=V, H=H)
