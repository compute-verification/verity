"""A producer unit bound to its segment of a shared row, as one Boolean Definition: the unit reads the row that several producers
write together, takes its segment's index as a public position, and returns a check that is 0 exactly when that segment
holds its output (form (B) of circuits' shared-row commit, CIR-113).

    SharedRowCheck_v1{G, W}    (y: Array<W, Array<16, Value<1>>>, row: Array<G, Array<16, Value<1>>>,
                               j: Array<16, Value<1>>) -> Array<16, Value<1>>
    SharedRowWindow_v1{FN, G}  (FN's parameters, row: Array<G, Array<16, Value<1>>>, j: Array<16, Value<1>> of kind position)
                               -> Array<16, Value<1>>: `SharedRowCheck_v1{G, W}` of FN's output, the row and j, W the bf16
                               words FN returns

A shared row is G bf16 words, G/W segments of W words: segment j is words [W·j, W·j + W), bits [16·W·j, 16·W·(j + 1)), each
the output of the one producer unit the partition puts there, committed once in place of each producer's own row. The check is

    c = OR over w < W of (row[W·j + w] XOR y[w]),   every bit of c set when W·j + W > G (j unsigned),

so c is 0 exactly when segment j lies inside the row and is y. The unit has no output row: c is pinned to forced-zero cells,
and the row is the committed one its readers read, so a proof of the unit binds its output to segment j of that row. c is one
word whatever W is (1 for an s8 coordinate or the embedding's, 32 for a padded softmax row, 64 for PV's head row), so the
unit's outputs are 16 bits. G is a multiple of W, so the segments tile the row.

j is one unsigned 16-bit word of kind position (`ports.In`), as `softmax_row.QkSoftmaxRowPad_v1`'s n is: a public input, the
verifier's own derivation from the unit's place among the row's producers, never a value the prover chooses. The ports FN
declares carry over to the parameters the wrapper passes it.

The segment is `boolean.gather`'s multiplexer tree over j's low ceil(log2(G/W)) bits, one whole segment at each leaf (a missing
partner, which only an out-of-range j selects, meets 0), and W·j + W > G is the OR of j's other bits, or the borrow of
j - G/W when G/W isn't a power of two. Each row word XOR y word is free and the OR over W words is (W - 1)·16 ANDs, so the
check is (G/W - 1 + p)·16·W + (W - 1)·16 + 16 ANDs and the range bit's, about 16·G: at a 1 KiB row (G = 512), 8,198 for
W = 1, 8,203 for W = 32 and 8,204 for W = 64; at G = 128, W = 1, 2,056 (`MoeSelectBf16x128_v2` is 2,072).

FN, the producer, is a static: a complete Boolean Definition whose return is one bf16 word of bits or an Array of them, such as
`gemm.ScaledMmS8CoordinateRows_v2` (whose word view, `ScaledMmS8CoordinateRows_v1`, the served step's units bind),
`gemm.PvRow_v2`, `softmax_row.QkSoftmaxRowPad_v1` or `rowwise.EmbeddingCoordinate_v1`. A word Definition is refused at `bind`:
its output is a word, not bits.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.defs import CompositeDefinition, FunctionDefinition, SpecializedDefinition, bind, composite
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array

from .elementwise import specialized

__all__ = ["J_LIMIT", "SharedRowCheck", "SharedRowWindow", "WINDOW", "segments", "shared_row_check_reference", "window_check"]

BF16 = Array(16, BL.BIT)
#: the segment index's type: one unsigned word
U16 = Array(16, BL.BIT)
#: the most segments a 16-bit j indexes
J_LIMIT = 1 << 16
#: the segment index j: a run input of kind position, public as `QkSoftmaxRowPad_v1`'s key count is. No coin port is beside
#: it, so the Definition it types is never a seed (`tests/test_seed_ports.py`)
WINDOW = In(kind="position")
_RESERVED = ("row", "j")


def window_check(C, y, row, j):
    """``c``: the OR over the segment's words of ``row[W·j + w] ^ y[w]``, all ones when ``W·j + W > G`` (``y`` W words, ``row``
    G words, G a multiple of W, ``j`` 16 bits, each a list of forms LSB first)."""
    W, n = len(y), len(row) // len(y)
    b = (n - 1).bit_length()
    segs = [[x for word in row[W * s:W * (s + 1)] for x in word] for s in range(n)]
    got = gather(C, j[:b], segs, 0)
    diff = [g ^ v for g, v in zip(got, (x for word in y for x in word))]
    c = diff[:16]
    for w in range(1, W):
        c = [OR(C, a, d) for a, d in zip(c, diff[16 * w:16 * (w + 1)])]
    out = or_reduce(C, j[b:]) if n == 1 << b else NOT(sub(C, j, const_bits(n, len(j)), len(j))[1])
    return [OR(C, x, out) for x in c]


def segments(G: Any, W: Any) -> int:
    """G/W, the segments of a row of G words; ValueError for a G and W that don't tile a row a 16-bit j indexes."""
    for name, v in (("G", G), ("W", W)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} is a count of bf16 words, at least 1, not {v!r}")
    if G % W:
        raise ValueError(f"a row of G = {G} words isn't whole segments of W = {W}: G is a multiple of W, so the segments tile "
                         f"the row")
    if G // W > J_LIMIT:
        raise ValueError(f"G / W = {G // W} segments is past {J_LIMIT}, the most a 16-bit j indexes")
    return G // W


def _check_signature(S: Any):
    segments(S.G, S.W)
    return (("y", Array(S.W, BF16)), ("row", Array(S.G, BF16)), ("j", U16)), BF16


SharedRowCheck = specialized(
    "SharedRowCheck", 1, ("G", "W"), lambda S: lambda C, *e: window_check(C, e[:S.W], e[S.W:-1], e[-1]), _check_signature,
    doc="a shared row's segment check on bits: c = OR over w < W of (row[W·j + w] XOR y[w]), every bit set when W·j + W > G "
        "(j unsigned), so c = 0 exactly when segment j lies inside the row and is y; a row of G bf16 words, G a multiple of W")


def _producer(FN: Any) -> tuple[FunctionDefinition, int]:
    """FN as the function the wrapper calls, and W, the bf16 words it returns; ValueError for an FN the wrapper doesn't take."""
    f = bind(FN) if isinstance(FN, CompositeDefinition) else FN
    if not isinstance(f, FunctionDefinition) or isinstance(f, SpecializedDefinition) and not f.complete:
        raise ValueError(f"FN is a complete Definition, the producer the wrapper binds to its segment, not {FN!r}")
    params, ret = f.signature()
    if ret == BF16:
        W = 1
    elif isinstance(ret, Array) and ret.elem == BF16 and ret.n >= 1:
        W = ret.n
    else:
        raise ValueError(f"FN = {f.id} returns {ret!r}, not one bf16 word of bits or an Array of them: bind its Boolean form "
                         f"(a word Definition's `_v2`, whose word view it is)")
    clash = sorted({n for n, _ in params} & set(_RESERVED))
    if clash:
        raise ValueError(f"FN = {f.id} has parameters named {clash}, which are the wrapper's own (the shared row and j)")
    return f, W


def _refuse(b: dict) -> str | None:
    try:
        W = _producer(b["FN"])[1] if "FN" in b else 1
        if "G" in b:
            segments(b["G"], W)
    except ValueError as e:
        return str(e)
    return None


def _ports(S: Any) -> Ports:
    f, _ = _producer(S.FN)
    decl = f.defn.ports if isinstance(f, SpecializedDefinition) else getattr(f, "ports", None)
    if callable(decl):
        decl = decl(f.S())
    return Ports(params={**(decl.params if decl is not None else {}), "j": WINDOW})


@composite("SharedRowWindow", 1, ["FN", "G"],
           lambda S: ((*_producer(S.FN)[0].signature()[0], ("row", Array(S.G, BF16)), ("j", U16)), BF16),
           conformance="protocol-defined", refuse=_refuse, ports=_ports)
def SharedRowWindow(B, S, *ports):
    """FN's output bound to segment j of the shared row: `SharedRowCheck_v1{G, W}` of FN's W words, the row and j, 0 exactly
    when W·j + W <= G and the row's segment j is FN's output."""
    f, W = _producer(S.FN)
    *xs, row, j = ports
    y = B.call(f, *xs)
    return B.call(bind(SharedRowCheck, G=S.G, W=W), Coll(Array(W, BF16), y.refs), row, j)


def shared_row_check_reference(y, row, j) -> np.ndarray:
    """What `SharedRowCheck_v1{G, W}` computes, for a batch: ``y`` ``(N, W)``, ``row`` ``(N, G)`` bf16 words and ``j`` ``(N,)``
    unsigned 16-bit words, each instance's c a uint16."""
    y, row, j = (np.asarray(a, dtype=np.uint16) for a in (y, row, j))
    N, W = y.shape
    if row.ndim != 2 or row.shape[0] != N or j.shape != (N,):
        raise ValueError(f"y {y.shape}, row {row.shape} and j {j.shape} are not (N, W), (N, G) and (N,)")
    n = segments(row.shape[1], W)
    inside = j.astype(np.int64) < n
    at = np.where(inside, j, 0).astype(np.int64)[:, None] * W + np.arange(W)[None, :]
    c = np.bitwise_or.reduce(np.take_along_axis(row, at, axis=1) ^ y, axis=1)
    return np.where(inside, c, 0xFFFF).astype(np.uint16)
