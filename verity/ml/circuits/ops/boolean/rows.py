"""Row units on bits: the next version of each `ops.rows` word Definition, and the embedding of a token at a public position,
each one proof unit a row that writes the whole row (the e2e demo's decoder, CMP-92).

    Bf16AddRow_v2{N}                  (a, b: Array<N, bf16 bits>) -> Array<N, bf16 bits>          word view: Bf16AddRow_v1{N}
    Argmax_v2{V}                      (logits: Array<V, bf16 bits>) -> Array<16, Value<1>>        word view: Argmax_v1{V}
    EmbeddingPosition_v1{V, H}        (tok: Array<16, Value<1>>, root: Digest, domain: Digest, row: Array<H, bf16 bits>,
                                       salt: Array<24, Word>, path: Array<D, Digest>, pos: Array<H, bf16 bits>)
                                      -> Tuple<Array<H, bf16 bits>, Value<1>>                                    (x, ok)
    EmbeddingPositionSelect_v1{V, H}  (tok: Array<16, Value<1>>, opens: Value<1>, row: Array<H, bf16 bits>) -> (h, ok)
        bf16 bits = Array<16, Value<1>>, Digest = Array<8, Word>, Word = Array<64, Value<1>>; D = merkle_depth(V);
        2 <= V <= 2^16, H >= 1

`Bf16AddRow_v2` is `Bf16AddF2fp_v2` (`elementwise`) per element. `Argmax_v2` is traced (`elementwise.specialized`): each
logit's order key and NaN bit (`scalar.bf16_key`) once, the running best held as its key and NaN bit, and each step's select
with the logit and the constant index as its else branch, so a step costs an AND a key bit, an AND a live index bit and the
comparison, and the forms stay flat.

**The embedding of token tok at public position p**, one unit a position: ``x[j] = Bf16AddF2fp_v1(E[tok][j], P[p][j])``,
the table E registered one token a row as `verity.ml.commitments.gates.embedding` reads it (``hm96-sha512/row-seg/v1``
leaves of H bf16 words in a dense frame-v3-sha512 tree of V rows, `one_stage.registered`), and P's row p the unit's ``pos``,
a registered read at the public step, as ``root`` and ``domain`` are reads of E's root and domain id. The token is a value
(a request's byte, or the previous step's `Argmax_v2` index, whose 16 bits it takes as they are), and so are its row, salt
and path: neither the token nor its row is opened outside the proof. With in = (tok < V, unsigned):

    h  = row if in, else 0x7FFF in every word                       `EmbeddingRead_v1{V, H}`'s h, for a 16-bit token
    ok = not in, or FrameV3Read_v1{D, hm96-sha512/row-seg/v1}(root, domain, Hm96Sha512Leaf_v1(Sha512RowSeg_v1{16 H}(row),
         salt)[2], tok mod 2^D, path)                               `EmbeddingRead_v1{V, H}`'s ok
    x  = Bf16AddF2fp_v2(h[j], pos[j]) for each j                    `Bf16AddRow_v2{H}`

So ok = 1 with tok in [0, V) means x is the row registered at tok plus pos, added as sm_90 adds them (`ops.rows`); a token
outside the vocabulary gives 0x7FFF in every word (a NaN plus anything) with ok = 1. A Boolean Definition has no assertion,
so the unit publishes ok, as `EmbeddingRead_v1` does.

**The read, not a select over the table.** A select over the registered table makes the whole table the unit's input, V
rows of C-Flock slots and a multiplexer tree of (V - 1) 16 H ANDs (522,240 at V = 256, H = 128; 66,846,720 at H = 16,384),
where the read costs one row's digest, its hm96 leaf and D frame-v3 levels whatever V is: `EmbeddingRead_v1`'s read, the
in-circuit check a window's hidden-position opening declares.

**One unit a row, in the size classes.** The unit's outputs are 16 H + 1 bits: 2,049 at H = 128 (class 4,096). At H = 16,384
they are 262,145, one past the largest class (`one_stage.partition.UNIT_CLASS_MAX`, 262,144), so a position there is two
units over two registered half-tables of 8,192 columns, `EmbeddingPosition_v1{V, 8192}` each with its half of the position
row, 131,073 bits each (class 262,144).
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, const_bits, mux
from verity.ml.circuits.defs import CompositeDefinition, bind, composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments.gates import embedding as ER
from verity.ml.commitments.gates.frame_v3 import read as frame_read
from verity.ml.commitments.gates.hm96 import SALT, Hm96Sha512Leaf
from verity.ml.commitments.gates.rowleaf import DIGEST, digest
from verity.ml.commitments.merkle import merkle_depth

from .. import rows as W
from . import elementwise as EW
from .scalar import bf16_key, gt_u

__all__ = ["Argmax", "Bf16AddRow", "EmbeddingPosition", "EmbeddingPositionSelect", "FILL", "arguments", "position", "reference"]

BIT = BL.BIT
BF16 = Array(16, BIT)
TOKEN = Array(16, BIT)
FILL = ER.FILL


# ---- the residual add and the argmax -------------------------------------------------------------------------------------

@composite("Bf16AddRow", 2, ["N"], lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested", refuse=W.refuse_n)
def Bf16AddRow(B, S, a, b):
    """`Bf16AddRow_v1{N}`'s body on bits: `Bf16AddF2fp_v2` per element."""
    return B.batch(EW.Bf16AddF2fp, a, b, axes=(0, 0))


def _argmax(S):
    width = (S.V - 1).bit_length()

    def build(C, *logits):
        keys = [bf16_key(C, x) for x in logits]
        nan, key = keys[0]
        at = const_bits(0, width)
        for k in range(1, S.V):
            nk, kk = keys[k]
            keep = NOT(C.AND(NOT(OR(C, nan, nk)), gt_u(C, kk, key)))
            at = mux(C, keep, at, const_bits(k, width))
            if k < S.V - 1:
                nan, key = mux(C, keep, [nan], [nk])[0], mux(C, keep, key, kk)
        return at + [ZERO] * (16 - width)

    return build


Argmax = EW.specialized(
    "Argmax", 2, ("V",), _argmax, lambda S: ((("logits", Array(S.V, BF16)),), TOKEN),
    word=lambda b: bind(W.Argmax, V=b["V"]),
    doc="Argmax_v1{V} on bits: the scan from (logits[0], 0) taking (logits[k], k) iff Bf16GtStrict(logits[k], best), on the "
        "logits' order keys, the index as 16 bits")
Argmax.refuse = W.refuse_v
Bf16AddRow.word = lambda b: bind(W.Bf16AddRow, N=b["N"])  # type: ignore[attr-defined]


# ---- the embedding at a public position ----------------------------------------------------------------------------------

def _check(V: int, H: int) -> int:
    """D, the tree's depth; refuses a malformed binding."""
    if isinstance(V, bool) or not isinstance(V, int) or not 2 <= V <= W.MAX_V:
        raise ValueError(f"an embedding of a 16-bit token has a V in 2 ... 2^16, not {V!r}")
    if isinstance(H, bool) or not isinstance(H, int) or H < 1:
        raise ValueError(f"an embedding's H is a positive int, not {H!r}")
    return merkle_depth(V)


def _refuse(b: dict) -> str | None:
    try:
        _check(b.get("V", 2), b.get("H", 1))
    except ValueError as e:
        return str(e)
    return None


def _below(C, bits: Sequence, bound: int):
    """1 iff the unsigned ``bits`` (LSB first) are below the constant ``bound``: an AND a bit past bound's lowest set bit."""
    if bound >> len(bits):
        return ONE
    lt = ZERO
    for k, b in enumerate(bits):
        lt = C.AND(b, lt.inv()).inv() if bound >> k & 1 else C.AND(b.inv(), lt)
    return lt


def _select(S):
    fill = const_bits(FILL, 16)

    def build(C, tok, opened, *row):
        inside = _below(C, tok, S.V)
        h = [[f ^ C.AND(inside, r ^ f) for r, f in zip(word, fill)] for word in row]
        return [*h, [ONE ^ C.AND(inside, opened[0] ^ ONE)]]

    return build


def _sig_select(S):
    _check(S.V, S.H)
    return (("tok", TOKEN), ("opens", BIT), ("row", Array(S.H, BF16))), Tuple(Array(S.H, BF16), BIT)


EmbeddingPositionSelect = EW.specialized(
    "EmbeddingPositionSelect", 1, ("V", "H"), _select, _sig_select,
    doc="(h, ok): h = row if the 16-bit tok < V (unsigned), else 0x7FFF in every word; ok = opens, or 1 when tok >= V")
EmbeddingPositionSelect.refuse = _refuse


def _signature(S):
    D = _check(S.V, S.H)
    return ((("tok", TOKEN), ("root", DIGEST), ("domain", DIGEST), ("row", Array(S.H, BF16)), ("salt", SALT),
             ("path", Array(D, DIGEST)), ("pos", Array(S.H, BF16))), Tuple(Array(S.H, BF16), BIT))


def _body(B, S, tok, root, domain, row, salt, path, pos):
    D = merkle_depth(S.V)
    leaf = B.call(Hm96Sha512Leaf, B.call(digest(16 * S.H), row.reshape(Array(16 * S.H, BIT))), salt)[2]
    opened = B.call(frame_read(D, ER.SCHEMA), root, domain, leaf, tok[0:D], path)
    h = B.call(bind(EmbeddingPositionSelect, V=S.V, H=S.H), tok, opened, row)
    return tuple_of(B.call(bind(Bf16AddRow, N=S.H), h[0], pos), h[1])


EmbeddingPosition = CompositeDefinition(
    "EmbeddingPosition", 1, ("V", "H"), _signature, _body, conformance="exact-model-tested", refuse=_refuse,
    doc="the embedding of a 16-bit token at a public position, its table registered one token a row under "
        "hm96-sha512/row-seg/v1: (x, ok), x[j] = Bf16AddF2fp(h[j], pos[j]) with h the witness row if tok < V (unsigned) else "
        "0x7FFF in every word, ok = 1 iff tok >= V or the row under salt opens at position tok to root in the dense "
        "frame-v3-sha512 tree of domain id domain (Sha512RowSeg_v1, Hm96Sha512Leaf_v1, FrameV3Read_v1, Bf16AddF2fp_v2)")


def position(V: int, H: int):
    return bind(EmbeddingPosition, V=V, H=H)


def reference(V: int, tok: int, root: bytes, domain_id: bytes, row: Sequence[int], salt: bytes, path: Sequence[bytes],
              pos: Sequence[int]) -> tuple[list[int], int]:
    """`EmbeddingPosition_v1{V, H = len(row)}`'s reference: (x, ok) for a 16-bit token and bf16 words."""
    _check(V, len(row))
    if not 0 <= tok < 1 << 16:
        raise ValueError(f"a token is 16 bits, not {tok}")
    h, ok = ER.reference(V, tok, root, domain_id, row, salt, path)
    return W.add_row(h, pos), ok


def arguments(V: int, tok: int, root: bytes, domain_id: bytes, row: Sequence[int], salt: bytes, path: Sequence[bytes],
              pos: Sequence[int]) -> tuple[list[int], ...]:
    """`EmbeddingPosition_v1{V, H = len(row)}`'s seven arguments, each a flat list of bits in leaf order."""
    _check(V, len(row))
    if not 0 <= tok < 1 << 16:
        raise ValueError(f"a token is 16 bits, not {tok}")
    if len(pos) != len(row) or any(not 0 <= int(w) < 1 << 16 for w in pos):
        raise ValueError(f"a position row is {len(row)} 16-bit words")
    _, *rest = ER.arguments(V, tok, root, domain_id, row, salt, path)
    return ([(tok >> j) & 1 for j in range(16)], *rest, [(int(w) >> j) & 1 for w in pos for j in range(16)])
