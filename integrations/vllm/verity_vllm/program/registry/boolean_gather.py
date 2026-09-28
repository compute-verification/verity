"""The vocabulary gather and the embedding as Boolean Definitions: the next version of each word Definition, the same function on
bits (`verity.ml.circuits.boolean.trace`).

    GatherBf16x{V}_v2       (token: Array<32, Value<1>>, column: Array<V, Array<16, Value<1>>>) -> Array<16, Value<1>>
                            word view: GatherBf16x{V}_v1, column[token] for 0 <= token < V (signed), else 0x7FFF
    Embedding_v2{V, H}      (tok: Array<32, Value<1>>, table: Array<V, Array<H, Array<16, Value<1>>>>) -> Array<H, Array<16, Value<1>>>
                            word view: Embedding_v1{V, H}
    GatherBf16x{N}u{k}_v1   (idx: Array<k, Value<1>>, column: Array<N, Array<16, Value<1>>>) -> Array<16, Value<1>>
                            column[idx] for idx < N (unsigned), else 0x7FFF: a level of a gather's tree, no word Definition

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
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.boolean.trace import definition

from . import b1
from . import prims as P

__all__ = ["CHUNK", "CHUNK_BITS", "Embedding", "FILL", "gather_bf16", "gather_index"]

BIT = BL.BIT
WORD = Array(16, BIT)
TOKEN = Array(32, BIT)
#: the word a token outside the vocabulary selects (`prims.gather_bf16`'s extension)
FILL = 0x7FFF
CHUNK_BITS = 10
CHUNK = 1 << CHUNK_BITS

_GATHERS: dict[int, Any] = {}
_INDEXED: dict[tuple[int, int], Any] = {}


def _select(C, idx, *column):
    return gather(C, idx, list(column), FILL)


def _tree(B, idx: Coll, column: Coll, n: int, k: int) -> Coll:
    """``column[idx]`` (0x7FFF past n) by a k-bit index, n > CHUNK: the chunks by the low bits, then their words by the rest."""
    full, r = divmod(n, CHUNK)
    lo = idx[0:CHUNK_BITS]
    words = []
    if full:
        chunks = Coll(Array(full, Array(CHUNK, WORD)), column.refs.slice(0, full * CHUNK * WORD.leaves))
        words.append(B.batch(gather_index(CHUNK, CHUNK_BITS), lo, chunks, axes=(None, 0)).refs)
    if r:
        words.append(B.call(gather_index(r, CHUNK_BITS), lo, column[full * CHUNK:n]).refs)
    m = full + (1 if r else 0)
    return B.call(gather_index(m, k - CHUNK_BITS), idx[CHUNK_BITS:k], Coll(Array(m, WORD), concat(words)))


def _gather_definition(name: str, version: int, n: int, k: int, idx_name: str, word: Any, doc: str) -> CompositeDefinition:
    params = ((idx_name, Array(k, BIT)), ("column", Array(n, WORD)))
    if n <= CHUNK:
        return definition(name, version, _select, params, WORD, word=word, doc=doc)
    d = CompositeDefinition(name, version, (), lambda S: (params, WORD), lambda B, S, idx, col: _tree(B, idx, col, n, k), doc=doc,
                            conformance="exact-model-tested")
    d.word = word  # type: ignore[attr-defined]
    return d


def gather_index(n: int, k: int) -> CompositeDefinition:
    """`GatherBf16x{n}u{k}_v1`: ``column[idx]`` for the unsigned k-bit idx < n, else 0x7FFF."""
    if not 1 <= n <= 1 << k:
        raise ValueError(f"a {k}-bit index selects from 1 to {1 << k} words, not {n}")
    d = _INDEXED.get((n, k))
    if d is None:
        d = _INDEXED[(n, k)] = _gather_definition(
            f"GatherBf16x{n}u{k}", 1, n, k, "idx", None,
            f"column[idx] for the unsigned {k}-bit idx < {n}, else 0x7FFF, on bits: a level of a gather's multiplexer tree")
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


def _embedding(B, S, tok, table):
    """`Embedding_v1` on bits: ``h[j] = table[tok][j]``, one `GatherBf16x{V}_v2` per hidden coordinate (column j of the table)."""
    L = WORD.leaves
    cols = Coll(Array(S.H, Array(S.V, WORD)), concat([strided_view(table, L * j, L, L * S.H, 1, L * S.V) for j in range(S.H)]))
    return B.batch(gather_bf16(S.V), tok, cols, axes=(None, 0))


Embedding = CompositeDefinition(
    "Embedding", 2, ("V", "H"), lambda S: ((("tok", TOKEN), ("table", Array(S.V, Array(S.H, WORD)))), Array(S.H, WORD)), _embedding,
    doc=_embedding.__doc__ or "", conformance="exact-model-tested")
Embedding.word = lambda b: bind(b1.Embedding, V=b["V"], H=b["H"])  # type: ignore[attr-defined]
