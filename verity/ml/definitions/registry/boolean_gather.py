"""The vocabulary gather and the embedding as Boolean Definitions: the next version of each word Definition, the same function on
bits (`verity.ml.circuits.boolean.trace`).

    GatherBf16x{V}_v2       (token: Array<32, Value<1>>, column: Array<V, Array<16, Value<1>>>) -> Array<16, Value<1>>
                            word view: GatherBf16x{V}_v1, column[token] for 0 <= token < V (signed), else 0x7FFF
    Embedding_v2{V, H}      (tok: Array<32, Value<1>>, table: Array<V, Array<H, Array<16, Value<1>>>>) -> Array<H, Array<16, Value<1>>>
                            word view: Embedding_v1{V, H}
    GatherBf16x{N}u{k}_v1   (idx: Array<k, Value<1>>, column: Array<N, Array<16, Value<1>>>) -> Array<16, Value<1>>
                            column[idx] for idx < N (unsigned), else 0x7FFF: a level of a gather's tree, no word Definition
    GatherI32x{V}_v2        (token: Array<32, Value<1>>, column: Array<V, Array<32, Value<1>>>) -> Array<32, Value<1>>
                            word view: GatherI32x{V}_v1 (`dsv4_decode_indices.gather_i32`), column[token], else -1
    GatherI32x{N}u{k}_v1    the i32 levels, -1 past N
    GatherF32x{V}_v2        (token: Array<32, Value<1>>, column: Array<V, Array<32, Value<1>>>) -> Array<32, Value<1>>
                            word view: GatherF32x{V}_v1 (`ref_prims.gather_f32`), column[token], else 0x7FC00000
    GatherF32x{N}u{k}_v1    the f32 levels, 0x7FC00000 past N
    GatherF32Chunked_v1{V, W}  (token: Array<32, Value<1>>, column: Array<V, Array<32, Value<1>>>) -> Array<32, Value<1>>
                            word view: GatherF32x{V}_v1, its column cut into V-chunks of at most W words, a gather each
    GatherChunkSplit_v1{V, W}  (token: Array<32, Value<1>>) -> Array<32 + k, Value<1>>
                            the token's offset in its V-chunk, then the V-chunk's index (k bits): no word Definition

A gather of at most `CHUNK` (1024) words is traced from `verity.ml.circuits.boolean.gather.gather`: the multiplexer tree over the
token's 32 bits, a word past the end standing for the constant 0x7FFF, so a token outside [0, V) (a negative one is at least
2^31 unsigned) selects 0x7FFF with no separate range check. A larger one is a tree of Calls, each Definition in it about 50k
gates at most: the column's whole chunks of 1024 words by the token's low 10 bits (`GatherBf16x1024u10_v1`, one batch node),
its partial last chunk of r words likewise (`GatherBf16x{r}u10_v1`, 0x7FFF past r), then the m chunk words by the token's high
22 bits (`GatherBf16x{m}u22_v1`). With token = hi * 1024 + lo, a token is past the end exactly when hi >= m, or hi = m - 1 and
lo >= r, and each case selects 0x7FFF. `GatherBf16x{N}u{k}_v1` over more than 1024 words is the same tree on k bits.

No gate of a gather reads the token alone (`gather`), so `Embedding_v2`'s H gathers, one per column of the table by one
token, share no gate, and since its body reads only its parameters and returns every gather's word, it is cut gather by gather
(`verity.ml.circuits.cut.separable`): each gather one unit of 16 output bits under `Q_word_v1{X=16,W=32}`.

The i32 and f32 gathers are the same tree over 32-bit words with -1 (0xFFFFFFFF) and the quiet NaN 0x7FC00000 for the word past
the end, each made on demand as its word primitive is (`verity.ml.definitions.boolean._families`). `GatherF32x{V}_v2` is one
Definition at every V: the vocabulary's and `boolean_fp8_moe`'s expert gathers (`boolean_fp8_moe.gather_f32`) alike.

`GatherF32Chunked_v1{V, W}` is that gather with its column cut so that each part is one proof block: c = ceil(V / W)
V-chunks, as even as whole words allow, the wider first (`chunk_widths`). `GatherChunkSplit_v1{V, W}` writes the token as
start_h + lo, h the number of V-chunk starts past the first that are at most the token (unsigned; one constant subtraction
each, whose borrow is the comparison, and h's bits are XORs of those comparisons). Each V-chunk is `GatherF32x{W_j}_v2` over
its row by lo (one batch node per width), and `GatherF32x{c}u{k}_v1` takes V-chunk h's word by the high part. A token
outside [0, V), a negative one included, has h = c - 1 and an lo that the last V-chunk's gather reads as an i32 outside its
width (V <= 2^31), so it selects 0x7FC00000: the chunked gather is `GatherF32x{V}_v1` at every token, and a larger
vocabulary takes more V-chunks, not another Definition.

`W_MAX` is the most words whose `GatherF32x{W}_v2`, with each of its three rows a ``hm96-sha512/row/v2`` row opened in the
block, fits one C-Flock block of 2^27 bits (`verity_flock.circuit.compose`, `benchmarks/unit_block`). In slots of 2^18 the
block holds 512: the gather's unit takes a range of 2^22 (16), the rows' four hm96 slots a range of 2^20 (4), the mask
aligns the packed sha512x3 range up by one, and the column row's 4W bytes take ceil((4W + 17) / 128) - 1 compressions, three
to a sha512x3 slot. So 491 slots, 1,473 compressions, hold 4W + 17 <= 1,474 * 128 bytes: W <= 47,163. The split's and the
select's rows are shorter than one SHA-512 block, and `compose` refuses a statement with no compression before a row's last
block, so neither is a block alone.
"""
from __future__ import annotations

from functools import reduce
from operator import xor
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean.forms import NOT, const_bits, mux, sub
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.boolean.trace import definition, part

from . import b1
from . import prims as P

__all__ = ["CHUNK", "CHUNK_BITS", "Embedding", "F32_FILL", "FILL", "GatherChunkSplit", "GatherF32Chunked", "I32_FILL", "W_MAX",
           "chunk_widths", "gather_bf16", "gather_f32", "gather_i32", "gather_index"]

BIT = BL.BIT
WORD = Array(16, BIT)
TOKEN = Array(32, BIT)
#: the word a token outside the vocabulary selects (`prims.gather_bf16`'s extension)
FILL = 0x7FFF
CHUNK_BITS = 10
CHUNK = 1 << CHUNK_BITS
I32 = Array(32, BIT)
#: the word an index outside an i32 gather's column selects (`dsv4_decode_indices.FILL`)
I32_FILL = 0xFFFFFFFF
#: the word an index outside an f32 gather's column selects (`ref_prims.gather_f32`'s extension, a quiet NaN)
F32_FILL = 0x7FC00000
#: the most words a V-chunk of `GatherF32Chunked_v1` holds as one C-Flock block of 2^27 bits (the module's doc)
W_MAX = 47163

_GATHERS: dict[Any, Any] = {}
_INDEXED: dict[tuple[int, int, str], Any] = {}


def _select(C, idx, *column):
    return gather(C, idx, list(column), FILL)


def _select_i32(C, idx, *column):
    return gather(C, idx, list(column), I32_FILL)


def _select_f32(C, idx, *column):
    return gather(C, idx, list(column), F32_FILL)


#: kind -> (its word, the word past the end, the traced selection)
_KINDS: dict[str, tuple[Any, int, Any]] = {"Bf16": (WORD, FILL, _select), "I32": (I32, I32_FILL, _select_i32),
                                           "F32": (I32, F32_FILL, _select_f32)}


def _tree(B, idx: Coll, column: Coll, n: int, k: int, kind: str = "Bf16") -> Coll:
    """``column[idx]`` (the kind's fill past n) by a k-bit index, n > CHUNK: the chunks by the low bits, then their words by the
    rest."""
    word = _KINDS[kind][0]
    full, r = divmod(n, CHUNK)
    lo = idx[0:CHUNK_BITS]
    words = []
    if full:
        chunks = Coll(Array(full, Array(CHUNK, word)), column.refs.slice(0, full * CHUNK * word.leaves))
        words.append(B.batch(gather_index(CHUNK, CHUNK_BITS, kind), lo, chunks, axes=(None, 0)).refs)
    if r:
        words.append(B.call(gather_index(r, CHUNK_BITS, kind), lo, column[full * CHUNK:n]).refs)
    m = full + (1 if r else 0)
    return B.call(gather_index(m, k - CHUNK_BITS, kind), idx[CHUNK_BITS:k], Coll(Array(m, word), concat(words)))


def _gather_definition(name: str, version: int, n: int, k: int, idx_name: str, word: Any, doc: str,
                       kind: str = "Bf16") -> CompositeDefinition:
    w, _, select = _KINDS[kind]
    params = ((idx_name, Array(k, BIT)), ("column", Array(n, w)))
    if n <= CHUNK:
        return definition(name, version, select, params, w, word=word, doc=doc)
    d = CompositeDefinition(name, version, (), lambda S: (params, w), lambda B, S, idx, col: _tree(B, idx, col, n, k, kind),
                            doc=doc, conformance="exact-model-tested")
    d.word = word  # type: ignore[attr-defined]
    return d


def gather_index(n: int, k: int, kind: str = "Bf16") -> CompositeDefinition:
    """`Gather{kind}x{n}u{k}_v1`: ``column[idx]`` for the unsigned k-bit idx < n, else the kind's fill (0x7FFF, -1)."""
    if not 1 <= n <= 1 << k:
        raise ValueError(f"a {k}-bit index selects from 1 to {1 << k} words, not {n}")
    d = _INDEXED.get((n, k, kind))
    if d is None:
        d = _INDEXED[(n, k, kind)] = _gather_definition(
            f"Gather{kind}x{n}u{k}", 1, n, k, "idx", None,
            f"column[idx] for the unsigned {k}-bit idx < {n}, else 0x{_KINDS[kind][1]:X}, on bits: a level of a gather's "
            "multiplexer tree", kind)
    return d


def gather_bf16(vocab: int) -> CompositeDefinition:
    """`GatherBf16x{V}_v2`: `GatherBf16x{V}_v1` on bits, column[token] for 0 <= token < V, else 0x7FFF."""
    if vocab < 1:
        raise ValueError(f"a gather selects from at least one word, not {vocab}")
    d = _GATHERS.get(vocab)
    if d is None:
        d = _GATHERS[vocab] = _gather_definition(
            f"GatherBf16x{vocab}", 2, vocab, 32, "token", P.gather_bf16(vocab),
            f"GatherBf16x{vocab}_v1 on bits: column[token] for 0 <= token < {vocab} (i32), else 0x7FFF; the multiplexer tree "
            "over the token's bits")
    return d


def gather_i32(vocab: int) -> CompositeDefinition:
    """`GatherI32x{V}_v2`: `GatherI32x{V}_v1` on bits, column[token] for 0 <= token < V, else -1."""
    from . import dsv4_decode_indices as DI

    if vocab < 1:
        raise ValueError(f"a gather selects from at least one word, not {vocab}")
    d = _GATHERS.get(("I32", vocab))
    if d is None:
        d = _GATHERS[("I32", vocab)] = _gather_definition(
            f"GatherI32x{vocab}", 2, vocab, 32, "token", DI.gather_i32(vocab),
            f"GatherI32x{vocab}_v1 on bits: column[token] for 0 <= token < {vocab} (i32), else -1; the multiplexer tree over the "
            "token's bits", "I32")
    return d


def gather_f32(vocab: int) -> CompositeDefinition:
    """`GatherF32x{V}_v2`: `GatherF32x{V}_v1` on bits, column[token] for 0 <= token < V, else 0x7FC00000."""
    from . import ref_prims as RP

    if vocab < 1:
        raise ValueError(f"a gather selects from at least one word, not {vocab}")
    d = _GATHERS.get(("F32", vocab))
    if d is None:
        d = _GATHERS[("F32", vocab)] = _gather_definition(
            f"GatherF32x{vocab}", 2, vocab, 32, "token", RP.gather_f32(vocab),
            f"GatherF32x{vocab}_v1 on bits: column[token] for 0 <= token < {vocab} (i32), else {F32_FILL:#x}; the multiplexer tree "
            "over the token's bits", "F32")
    return d


def chunk_widths(V: int, W: int) -> list[int]:
    """`GatherF32Chunked_v1{V, W}`'s V-chunk widths: ceil(V / W) of them, as even as whole words allow, the wider first."""
    if not (1 <= V <= 1 << 31 and W >= 1):
        raise ValueError(f"V-chunks of at most W = {W} words of a column of V = {V} (1 <= V <= 2^31, W >= 1)")
    c = -(-V // W)
    q, r = divmod(V, c)
    return [q + 1] * r + [q] * (c - r)


def _high_bits(c: int) -> int:
    return max(1, (c - 1).bit_length())


def _split_type(S):
    c = len(chunk_widths(S.V, S.W))
    if c < 2:
        raise ValueError(f"a column of V = {S.V} words is one V-chunk of at most W = {S.W}: nothing to split")
    return (("token", TOKEN),), Array(32 + _high_bits(c), BIT)


def _split(C, S, token):
    """The token's offset lo in its V-chunk h, then h's bits: lo = token - start_h, h the number of V-chunk starts past the
    first at most the token (unsigned), so h = c - 1 past the column and for a negative token."""
    widths = chunk_widths(S.V, S.W)
    lo, ge = list(token), []
    for j in range(1, len(widths)):
        d, borrow = sub(C, token, const_bits(sum(widths[:j]), 32), 32)
        ge.append(NOT(borrow))
        lo = mux(C, ge[-1], d, lo)
    return lo + [reduce(xor, (ge[j - 1] for j in range(1, len(widths)) if (j ^ (j - 1)) >> b & 1))
                 for b in range(_high_bits(len(widths)))]


GatherChunkSplit = part(
    "GatherChunkSplit", ("V", "W"), _split_type, _split,
    "A token as its V-chunk's start plus its offset (`chunk_widths`): the offset's 32 bits, then the V-chunk's index, the "
    "number of V-chunk starts past the first at most the token (unsigned); a level of `GatherF32Chunked_v1`")


def _chunked(B, S, token, column):
    """`GatherF32x{V}_v1` on bits by V-chunks: `GatherChunkSplit_v1` splits the token, each V-chunk's `GatherF32x{W_j}_v2`
    gathers by the offset, and `GatherF32x{c}u{k}_v1` takes the V-chunk the token is in by its index."""
    widths = chunk_widths(S.V, S.W)
    if len(widths) == 1:
        return B.call(gather_f32(S.V), token, column)
    k = _high_bits(len(widths))
    split = B.call(bind(GatherChunkSplit, V=S.V, W=S.W), token)
    words, at = [], 0
    for w in sorted(set(widths), reverse=True):
        n = widths.count(w)
        rows = Coll(Array(n, Array(w, I32)), column.refs.slice(at * I32.leaves, (at + n * w) * I32.leaves))
        words.append(B.batch(gather_f32(w), split[0:32], rows, axes=(None, 0)).refs)
        at += n * w
    return B.call(gather_index(len(widths), k, "F32"), split[32:32 + k], Coll(Array(len(widths), I32), concat(words)))


def _chunked_type(S):
    chunk_widths(S.V, S.W)
    return (("token", TOKEN), ("column", Array(S.V, I32))), I32


def _f32_word(b: dict):
    from . import ref_prims as RP

    return RP.gather_f32(b["V"])


GatherF32Chunked = CompositeDefinition("GatherF32Chunked", 1, ("V", "W"), _chunked_type, _chunked, doc=_chunked.__doc__ or "",
                                       conformance="exact-model-tested")
GatherF32Chunked.word = _f32_word  # type: ignore[attr-defined]


def _embedding(B, S, tok, table):
    """`Embedding_v1` on bits: ``h[j] = table[tok][j]``, one `GatherBf16x{V}_v2` per hidden coordinate (column j of the table)."""
    L = WORD.leaves
    cols = Coll(Array(S.H, Array(S.V, WORD)), concat([strided_view(table, L * j, L, L * S.H, 1, L * S.V) for j in range(S.H)]))
    return B.batch(gather_bf16(S.V), tok, cols, axes=(None, 0))


Embedding = CompositeDefinition(
    "Embedding", 2, ("V", "H"), lambda S: ((("tok", TOKEN), ("table", Array(S.V, Array(S.H, WORD)))), Array(S.H, WORD)), _embedding,
    doc=_embedding.__doc__ or "", conformance="exact-model-tested")
Embedding.word = lambda b: bind(b1.Embedding, V=b["V"], H=b["H"])  # type: ignore[attr-defined]
