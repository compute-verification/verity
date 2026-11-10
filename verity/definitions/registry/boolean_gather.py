"""The vocabulary gather and the embedding as Boolean Definitions: the next version of each word Definition, the same function on
bits (`verity.circuits.boolean.trace`).

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
    GatherChunkOffsets_v1{V, W}  (token: Array<32, Value<1>>) -> Array<c - 1, Array<32, Value<1>>>
                            token - start_j mod 2^32 for each V-chunk j past the first: no word Definition
    GatherChunkXor_v1{C}    (words: Array<C, Array<32, Value<1>>>) -> Array<32, Value<1>>
                            the C words' XOR, and 0x7FC00000's once more for an even C: no word Definition
    BitAtx{W}_v2            (word: Array<W, Value<1>>, i: Array<32, Value<1>>) -> Value<1>
                            word view: BitAtx{W}_v1 (`sampling.bit_at`), bit i of the word for 0 <= i < W (signed), else 0

A gather of at most `CHUNK` (1024) words is traced from `verity.circuits.boolean.gather.gather`: the multiplexer tree over the
token's 32 bits, a word past the end standing for the constant 0x7FFF, so a token outside [0, V) (a negative one is at least
2^31 unsigned) selects 0x7FFF with no separate range check. A larger one is a tree of Calls, each Definition in it about 50k
gates at most: the column's whole chunks of 1024 words by the token's low 10 bits (`GatherBf16x1024u10_v1`, one batch node),
its partial last chunk of r words likewise (`GatherBf16x{r}u10_v1`, 0x7FFF past r), then the m chunk words by the token's high
22 bits (`GatherBf16x{m}u22_v1`). With token = hi * 1024 + lo, a token is past the end exactly when hi >= m, or hi = m - 1 and
lo >= r, and each case selects 0x7FFF. `GatherBf16x{N}u{k}_v1` over more than 1024 words is the same tree on k bits.

No gate of a gather reads the token alone (`gather`), so `Embedding_v2`'s H gathers, one per column of the table by one
token, share no gate, and since its body reads only its parameters and returns every gather's word, it is cut gather by gather
(`verity.circuits.cut.separable`): each gather one unit of 16 output bits under `Q_word_v1{X=16,W=32}`.

The i32 and f32 gathers are the same tree over 32-bit words with -1 (0xFFFFFFFF) and the quiet NaN 0x7FC00000 for the word past
the end, each made on demand as its word primitive is (`verity.definitions.boolean._families`). `GatherF32x{V}_v2` is one
Definition at every V: the vocabulary's and `boolean_fp8_moe`'s expert gathers (`boolean_fp8_moe.gather_f32`) alike.

`GatherF32Chunked_v1{V, W}` is that gather with its column cut so that each part is one proof block: c = ceil(V / W)
V-chunks, as even as whole words allow, the wider first (`chunk_widths`). V-chunk j, from start_j, is `GatherF32x{W_j}_v2`
over its row by token - start_j mod 2^32 (one batch node per width), which `GatherChunkOffsets_v1{V, W}` computes for j >= 1
with one constant addition each and no carry out, and `GatherChunkXor_v1{c}` sums their words. As u32, token - start_j is in
[0, W_j) exactly when the token is in V-chunk j (start_j + W_j <= V <= 2^31, so nothing wraps), and otherwise it is an i32
outside the V-chunk, which then selects 0x7FC00000. So a token in V-chunk h leaves its word and c - 1 fills in the sum, a
token outside [0, V), a negative one included, leaves c fills, and the fill once more for an even c makes the sum the word
or the fill: the chunked gather is `GatherF32x{V}_v1` at every token, and a larger vocabulary takes more V-chunks, not
another Definition. The offsets and the sum read only the token and the V-chunks' words, so in the block that holds the
V-chunks they add no fresh input, and only whole 32-bit words pass between blocks.

`W_MAX` is the most words whose `GatherF32x{W}_v2`, with each of its three rows a ``hm96-sha512/row/v2`` row opened in the
block, fits one Flock block of 2^27 bits (`verity_flock.circuit.compose`, `benchmarks/unit_block`). In slots of 2^18 the
block holds 512: the gather's unit takes a range of 2^22 (16), the rows' four hm96 slots a range of 2^20 (4), the mask
aligns the packed sha512x3 range up by one, and the column row's 4W bytes take ceil((4W + 17) / 128) - 1 compressions, three
to a sha512x3 slot. So 491 slots, 1,473 compressions, hold 4W + 17 <= 1,474 * 128 bytes: W <= 47,163.

`BitAtx{W}_v2` is the gather of the word's W bits as one-bit words by i, 0 past the end, traced whole and made on demand as its
word primitive is: an i outside [0, W) (a negative one is at least 2^31 unsigned) selects 0 with no separate range check.
"""
from __future__ import annotations

from functools import reduce
from operator import xor
from typing import Any

from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll, concat, strided_view
from verity.circuits.types import Array
from verity.circuits.boolean.forms import add, const, const_bits
from verity.circuits.boolean.gather import gather
from verity.circuits.boolean.trace import definition, part

from . import b1
from . import prims as P

__all__ = ["CHUNK", "CHUNK_BITS", "Embedding", "F32_FILL", "FILL", "GatherChunkOffsets", "GatherChunkXor", "GatherF32Chunked",
           "I32_FILL", "W_MAX", "bit_at", "chunk_widths", "gather_bf16", "gather_f32", "gather_i32", "gather_index"]

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
#: the most words a V-chunk of `GatherF32Chunked_v1` holds as one Flock block of 2^27 bits (the module's doc)
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


def _offsets_type(S):
    c = len(chunk_widths(S.V, S.W))
    if c < 2:
        raise ValueError(f"a column of V = {S.V} words is one V-chunk of at most W = {S.W}: no offsets")
    return (("token", TOKEN),), Array(c - 1, TOKEN)


def _offsets(C, S, token):
    """token - start_j mod 2^32 for each V-chunk j past the first: the token plus -start_j, with no carry out."""
    widths = chunk_widths(S.V, S.W)
    return [add(C, token, const_bits(-sum(widths[:j]) % (1 << 32), 32), width=32) for j in range(1, len(widths))]


GatherChunkOffsets = part(
    "GatherChunkOffsets", ("V", "W"), _offsets_type, _offsets,
    "token - start_j mod 2^32 for each V-chunk j past the first (`chunk_widths`), the index each V-chunk's gather reads; a "
    "level of `GatherF32Chunked_v1`")


def _xor(C, S, *words):
    return [reduce(xor, bits, const(F32_FILL >> b & 1 if S.C % 2 == 0 else 0)) for b, bits in enumerate(zip(*words))]


GatherChunkXor = part(
    "GatherChunkXor", ("C",), lambda S: ((("words", Array(S.C, I32)),), I32), _xor,
    "The C words' XOR, and 0x7FC00000's once more for an even C: C - 1 fills and a word sum to the word, C fills to the fill; "
    "a level of `GatherF32Chunked_v1`")


def _chunked(B, S, token, column):
    """`GatherF32x{V}_v1` on bits by V-chunks: each V-chunk's `GatherF32x{W_j}_v2` gathers by token - start_j
    (`GatherChunkOffsets_v1`), so only the V-chunk holding the token returns other than 0x7FC00000, and `GatherChunkXor_v1`
    sums the words."""
    widths = chunk_widths(S.V, S.W)
    if len(widths) == 1:
        return B.call(gather_f32(S.V), token, column)
    offsets = B.call(bind(GatherChunkOffsets, V=S.V, W=S.W), token)
    index = [token.refs] + [offsets.refs.slice((j - 1) * TOKEN.leaves, j * TOKEN.leaves) for j in range(1, len(widths))]
    words, j, at = [], 0, 0
    for w in sorted(set(widths), reverse=True):
        n = widths.count(w)
        rows = Coll(Array(n, Array(w, I32)), column.refs.slice(at * I32.leaves, (at + n * w) * I32.leaves))
        words.append(B.batch(gather_f32(w), Coll(Array(n, TOKEN), concat(index[j:j + n])), rows, axes=(0, 0)).refs)
        j, at = j + n, at + n * w
    return B.call(bind(GatherChunkXor, C=len(widths)), Coll(Array(len(widths), I32), concat(words)))


def _chunked_type(S):
    chunk_widths(S.V, S.W)
    return (("token", TOKEN), ("column", Array(S.V, I32))), I32


def _f32_word(b: dict):
    from . import ref_prims as RP

    return RP.gather_f32(b["V"])


GatherF32Chunked = CompositeDefinition("GatherF32Chunked", 1, ("V", "W"), _chunked_type, _chunked, doc=_chunked.__doc__ or "",
                                       conformance="exact-model-tested")
GatherF32Chunked.word = _f32_word  # type: ignore[attr-defined]


def _bit_at(C, word, i):
    return gather(C, i, [[b] for b in word], 0)


def bit_at(width: int) -> CompositeDefinition:
    """`BitAtx{W}_v2`: `BitAtx{W}_v1` on bits, bit i of the word for 0 <= i < W (i32), else 0; the multiplexer tree over i's
    bits."""
    from . import sampling

    if width < 1:
        raise ValueError(f"a word has at least one bit, not {width}")
    d = _GATHERS.get(("Bit", width))
    if d is None:
        d = _GATHERS[("Bit", width)] = definition(
            f"BitAtx{width}", 2, _bit_at, (("word", Array(width, BIT)), ("i", TOKEN)), BIT, word=sampling.bit_at(width),
            doc=f"BitAtx{width}_v1 on bits: bit i of the word for 0 <= i < {width} (i32), else 0; the multiplexer tree over "
                "i's bits")
    return d


def _embedding(B, S, tok, table):
    """`Embedding_v1` on bits: ``h[j] = table[tok][j]``, one `GatherBf16x{V}_v2` per hidden coordinate (column j of the table)."""
    L = WORD.leaves
    cols = Coll(Array(S.H, Array(S.V, WORD)), concat([strided_view(table, L * j, L, L * S.H, 1, L * S.V) for j in range(S.H)]))
    return B.batch(gather_bf16(S.V), tok, cols, axes=(None, 0))


Embedding = CompositeDefinition(
    "Embedding", 2, ("V", "H"), lambda S: ((("tok", TOKEN), ("table", Array(S.V, Array(S.H, WORD)))), Array(S.H, WORD)), _embedding,
    doc=_embedding.__doc__ or "", conformance="exact-model-tested")
Embedding.word = lambda b: bind(b1.Embedding, V=b["V"], H=b["H"])  # type: ignore[attr-defined]
