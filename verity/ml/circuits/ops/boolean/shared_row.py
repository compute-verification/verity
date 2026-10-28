"""A proof block's output written as a segment of a shared row, and an input read as one, as Boolean Definitions (form (B) of
the shared-row commit). A shared row is one committed row that several proof blocks' outputs tile; the writer
proves its segment holds its output, and a reader takes its input from a segment of the row it reads.

    SharedRowCheck_v1{W, ROW, OFF, G}       (y: Array<W, U16>, row: Array<ROW, U16>, j: U16) -> Array<W, U16>
    SharedRowWindow_v1{FN, ROW, OFF, G}     (FN's parameters, row: Array<ROW, U16>, j: U16 of kind position) -> Array<W, U16>:
                                            `SharedRowCheck_v1{W, ROW, OFF, G}` of FN's output, the row and j
    SharedRowSelect_v1{W, ROW, OFF, G}      (row: Array<ROW, U16>, j: U16) -> Array<W, U16>
    SharedRowRead_v1{FN, PORT, ROW, OFF, G} (FN's parameters with PORT's an Array<ROW, U16>, and at G > 1 PORT_j: U16 of kind
                                            position) -> FN's return: FN with its parameter PORT read from segment PORT_j
                                            of the row (segment 0 at G = 1)

(U16 is ``Array<16, Value<1>>``.) The statics are the writers table's (FAMILIES' `circuits/shared-row-bindings/v2`) in words:
ROW the row's words (its bytes / 2), OFF the word its first segment starts at (``row_offset`` / 2), G the segments of W words
that one family's proof blocks write (its G), W the words a proof block writes (``window_bytes`` / 2). Segment j is words
[OFF + W·j, OFF + W·j + W), so a row may hold several families' segments, each at its own OFF. A row that is one proof
block's whole output (a quant chunk's codes and scale) is that block's own output row, and needs no writer.

A segment's words are its value's bits, 16 a word, least significant first: an f32 is two words, its low half first, as the
tensor's bytes are in memory. FN's return, or PORT's type, is any type of bits whose count is a multiple of 16 (W words).

**The writer.** `SharedRowCheck_v1` is the difference d, bit for bit,

    d[w] = (row[OFF + W·j + w] XOR y[w]) OR (j >= G)   for w < W (j unsigned),

so d is segment j XOR y when j < G and every bit set when j >= G: d is 0 exactly when j < G and segment j is y.
`SharedRowWindow_v1`'s proof block has no output row: d is its pinned output (C-Flock's META ``pinned_outputs``, each bit a
copy of a forced-zero row), and the row is the committed one its readers read, so a proof binds FN's output to segment j of
that row. Each bit of d is one linear form over the circuit's wires, as a pinned bit's row must be. j is one unsigned
16-bit word of kind position (`ports.In`), as `softmax_row.QkSoftmaxRowPad_v1`'s n is: a public input, the verifier's own
derivation from the proof block's index i (j = i mod G), never a value the prover chooses. A j >= G names no segment of the
family, even where the row goes on (another family's segment, at another OFF), so it is refused there too. The ports FN
declares carry over.

The segment is `boolean.gather`'s multiplexer over j's low ceil(log2 G) bits, one whole segment at each leaf (a missing
partner, which only a j >= G selects, meets 0); j >= G is the OR of j's other bits, or the borrow of j - G when G isn't a
power of two. The XORs are free and setting d's bits past G is 16·W ANDs, so the check is (G - 1)·16·W + 16·W ANDs and the
range bit's: at a 1 KiB row tiled by one family (ROW = 512, G·W = 512), 8,198 for W = 1, 8,203 for W = 32 and 8,204 for
W = 64; at G = 4 of W = 128 (the norm's and the residual's rows), 8,205.

**The reader.** `SharedRowRead_v1` gives FN's parameter PORT segment PORT_j of the row in its place. At G = 1 it is wiring:
PORT's W words are the row's words [OFF, OFF + W), FN reads the rest of the row not at all, and it costs no gate. That is the
read of a family whose every proof block reads the same segment: the RMS norm's chunk sums, the attention norm's at OFF = 0
and the MLP norm's at OFF = 256 of their layer's pooled 1 KiB row, or a quant chunk's amax word in its pooled amax row. At
G > 1 it is `SharedRowSelect_v1`, the same multiplexer with each bit ANDed with j < G, so (G - 1)·16·W + 16·W ANDs and the
range bit's, and a j >= G reads 0: a chunk of a shared row whose index each proof block takes from its own, such as the norm's
or the residual's 128 words of a 1 KiB residual row (G = 4, 8,205 ANDs). PORT_j is a public input of kind position, as j is.
The row is an ordinary input row, opened as a registered read of the writers' row.

FN, the producer or the reader, is a static: a complete Boolean Definition, such as `gemm.ScaledMmS8CoordinateRows_v2` (whose
word view, `ScaledMmS8CoordinateRows_v1`, the served step binds), `gemm.PvRow_v2`, `softmax_row.QkSoftmaxRowPad_v1`,
`rowwise.EmbeddingCoordinate_v1`, `rmsnorm.RmsSumSqChunks_v1`, `rmsnorm.RmsNormOutSums_v1`, `rows.Bf16AddRow_v2`,
`rows.Argmax_v2`, `quant.RowAbsMax_v2` or `quant.QuantChunkS8_v2`. A word Definition is refused at `bind`: its values are
words, not bits. A proof block that reads and writes shared rows is a `SharedRowWindow_v1` of its `SharedRowRead_v1`s, one a
port read: the norm's is ``SharedRowWindow_v1{FN=SharedRowRead_v1{FN=SharedRowRead_v1{FN=RmsNormOutSums_v1{...},
PORT='s', ROW=512, OFF=0 or 256, G=1}, PORT='x', ROW=512, OFF=0, G=4}, ROW=512, OFF=0, G=4}``, with ports
(x, w, s, x_j, row, j).
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

__all__ = ["J_LIMIT", "SharedRowCheck", "SharedRowRead", "SharedRowSelect", "SharedRowWindow", "U16", "WINDOW", "segments",
           "shared_row_check_reference", "shared_row_read_reference", "window_check", "window_select"]

U16 = Array(16, BL.BIT)
#: the most segments a 16-bit j indexes
J_LIMIT = 1 << 16
#: the segment index j: a run input of kind position, public as `QkSoftmaxRowPad_v1`'s key count is. No coin port is beside
#: it, so the Definition it types is never a seed (`tests/test_seed_ports.py`)
WINDOW = In(kind="position")
_RESERVED = ("row", "j")


def _segment(C, row, j, G: int, OFF: int, W: int):
    """Segment j's 16·W bits by `gather` over j's low bits (garbage when ``j >= G``), and the bit ``j >= G``."""
    b = (G - 1).bit_length()
    segs = [[x for word in row[OFF + W * s:OFF + W * (s + 1)] for x in word] for s in range(G)]
    past = or_reduce(C, j[b:]) if G == 1 << b else NOT(sub(C, j, const_bits(G, len(j)), len(j))[1])
    return gather(C, j[:b], segs, 0), past


def window_check(C, y, row, j, G: int, OFF: int):
    """``d``: W words, ``row[OFF + W·j + w] ^ y[w]``, every bit set when ``j >= G`` (``y`` W words, ``row`` the row's words,
    ``j`` 16 bits, each a list of forms LSB first)."""
    W = len(y)
    got, past = _segment(C, row, j, G, OFF, W)
    diff = [g ^ v for g, v in zip(got, (x for word in y for x in word))]
    return [[OR(C, x, past) for x in diff[16 * w:16 * (w + 1)]] for w in range(W)]


def window_select(C, row, j, G: int, OFF: int, W: int):
    """Segment j's W words, ``row[OFF + W·j:OFF + W·j + W]``, every bit 0 when ``j >= G``."""
    got, past = _segment(C, row, j, G, OFF, W)
    keep = NOT(past)
    return [[C.AND(x, keep) for x in got[16 * w:16 * (w + 1)]] for w in range(W)]


def _count(name: str, v: Any, least: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < least:
        raise ValueError(f"{name} is a count of words, at least {least}, not {v!r}")
    return v


def segments(ROW: Any, OFF: Any, G: Any, W: Any) -> int:
    """G, checked: ValueError for statics whose G segments of W words from word OFF don't fit a row of ROW words, or more
    segments than a 16-bit j indexes."""
    ROW, OFF, G, W = _count("ROW", ROW, 1), _count("OFF", OFF, 0), _count("G", G, 1), _count("W", W, 1)
    if G > J_LIMIT:
        raise ValueError(f"G = {G} segments is past {J_LIMIT}, the most a 16-bit j indexes")
    if OFF + G * W > ROW:
        raise ValueError(f"G = {G} segments of W = {W} words from word OFF = {OFF} end at word {OFF + G * W}, past the row's "
                         f"ROW = {ROW}")
    return G


def _check_signature(S: Any):
    segments(S.ROW, S.OFF, S.G, S.W)
    return (("y", Array(S.W, U16)), ("row", Array(S.ROW, U16)), ("j", U16)), Array(S.W, U16)


SharedRowCheck = specialized(
    "SharedRowCheck", 1, ("W", "ROW", "OFF", "G"),
    lambda S: lambda C, *e: window_check(C, e[:S.W], e[S.W:-1], e[-1], S.G, S.OFF), _check_signature,
    doc="a shared row's segment check on bits: d[w] = row[OFF + W·j + w] XOR y[w] for w < W, every bit set when j >= G "
        "(j unsigned), so d = 0 exactly when j < G and segment j is y; a row of ROW words, G segments of W words from word OFF")


def _select_signature(S: Any):
    segments(S.ROW, S.OFF, S.G, S.W)
    return (("row", Array(S.ROW, U16)), ("j", U16)), Array(S.W, U16)


SharedRowSelect = specialized(
    "SharedRowSelect", 1, ("W", "ROW", "OFF", "G"),
    lambda S: lambda C, *e: window_select(C, e[:-1], e[-1], S.G, S.OFF, S.W), _select_signature,
    doc="a shared row's segment read on bits: words [OFF + W·j, OFF + W·j + W) of a row of ROW words when j < G (j unsigned), "
        "every bit 0 when j >= G; G segments of W words from word OFF")


def _words(t: Any, what: str) -> int:
    """W, the 16-bit words of a type of bits; ValueError for any other type."""
    if not t.width_set() <= {1} or t.leaves % 16 or not t.leaves:
        raise ValueError(f"{what} is {t!r}, not bits in whole 16-bit words: bind its Boolean form (a word Definition's `_v2`, "
                         f"whose word view it is)")
    return t.leaves // 16


def _function(FN: Any) -> FunctionDefinition:
    f = bind(FN) if isinstance(FN, CompositeDefinition) else FN
    if not isinstance(f, FunctionDefinition) or isinstance(f, SpecializedDefinition) and not f.complete:
        raise ValueError(f"FN is a complete Definition, not {FN!r}")
    return f


def _declared(f: FunctionDefinition) -> dict:
    decl = f.defn.ports if isinstance(f, SpecializedDefinition) else getattr(f, "ports", None)
    if callable(decl):
        decl = decl(f.S())
    return dict(decl.params) if decl is not None else {}


def _producer(FN: Any) -> tuple[FunctionDefinition, int]:
    """FN as the function the writer calls, and W, the words it returns; ValueError for an FN the writer doesn't take."""
    f = _function(FN)
    params, ret = f.signature()
    W = _words(ret, f"FN = {f.id}'s return")
    clash = sorted({n for n, _ in params} & set(_RESERVED))
    if clash:
        raise ValueError(f"FN = {f.id} has parameters named {clash}, which are the wrapper's own (the shared row and j)")
    return f, W


def _refuse_window(b: dict) -> str | None:
    try:
        W = _producer(b["FN"])[1] if "FN" in b else 1
        if {"ROW", "OFF", "G"} <= set(b):
            segments(b["ROW"], b["OFF"], b["G"], W)
    except ValueError as e:
        return str(e)
    return None


def _window_ports(S: Any) -> Ports:
    return Ports(params={**_declared(_producer(S.FN)[0]), "j": WINDOW})


def _window_signature(S: Any):
    f, W = _producer(S.FN)
    return (*f.signature()[0], ("row", Array(S.ROW, U16)), ("j", U16)), Array(W, U16)


@composite("SharedRowWindow", 1, ["FN", "ROW", "OFF", "G"], _window_signature, conformance="protocol-defined",
           refuse=_refuse_window, ports=_window_ports)
def SharedRowWindow(B, S, *ports):
    """FN's output bound to segment j of the shared row: `SharedRowCheck_v1{W, ROW, OFF, G}` of FN's W words, the row and j,
    0 exactly when j < G and the row's segment j is FN's output."""
    f, W = _producer(S.FN)
    *xs, row, j = ports
    y = B.call(f, *xs)
    return B.call(bind(SharedRowCheck, W=W, ROW=S.ROW, OFF=S.OFF, G=S.G), Coll(Array(W, U16), y.refs), row, j)


def _reader(FN: Any, PORT: Any) -> tuple[FunctionDefinition, int, int]:
    """FN as the function the reader calls, PORT's index among its parameters and PORT's words; ValueError otherwise."""
    f = _function(FN)
    names = [n for n, _ in f.signature()[0]]
    if not isinstance(PORT, str) or PORT not in names:
        raise ValueError(f"PORT is one of FN = {f.id}'s parameters {names}, not {PORT!r}")
    if PORT + "_j" in names:
        raise ValueError(f"FN = {f.id} has a parameter {PORT}_j, which is the reader's own (PORT's segment index)")
    i = names.index(PORT)
    return f, i, _words(f.signature()[0][i][1], f"FN = {f.id}'s parameter {PORT}")


def _read_signature(S: Any):
    f, i, W = _reader(S.FN, S.PORT)
    segments(S.ROW, S.OFF, S.G, W)
    params, ret = f.signature()
    params = tuple((n, Array(S.ROW, U16)) if k == i else (n, t) for k, (n, t) in enumerate(params))
    return (params + ((S.PORT + "_j", U16),) if S.G > 1 else params), ret


def _refuse_read(b: dict) -> str | None:
    try:
        if "FN" in b and "PORT" in b:
            W = _reader(b["FN"], b["PORT"])[2]
            if {"ROW", "OFF", "G"} <= set(b):
                segments(b["ROW"], b["OFF"], b["G"], W)
    except ValueError as e:
        return str(e)
    return None


def _read_ports(S: Any) -> Ports:
    return Ports(params={**_declared(_function(S.FN)), **({S.PORT + "_j": WINDOW} if S.G > 1 else {})})


@composite("SharedRowRead", 1, ["FN", "PORT", "ROW", "OFF", "G"], _read_signature, conformance="protocol-defined",
           refuse=_refuse_read, ports=_read_ports)
def SharedRowRead(B, S, *ports):
    """FN with its parameter PORT read from segment PORT_j of the row in its place: at G = 1, words [OFF, OFF + W), wiring and
    no gate; at G > 1, `SharedRowSelect_v1{W, ROW, OFF, G}` of the row and PORT_j."""
    f, i, W = _reader(S.FN, S.PORT)
    t = f.signature()[0][i][1]
    args = list(ports[:len(f.signature()[0])])
    row = args[i]
    if S.G == 1:
        args[i] = Coll(t, row.refs.slice(16 * S.OFF, 16 * (S.OFF + W)))
    else:
        args[i] = Coll(t, B.call(bind(SharedRowSelect, W=W, ROW=S.ROW, OFF=S.OFF, G=S.G), row, ports[-1]).refs)
    return B.call(f, *args)


def shared_row_check_reference(y, row, j, G: int, OFF: int = 0) -> np.ndarray:
    """What `SharedRowCheck_v1{W, ROW, OFF, G}` computes, for a batch: ``y`` ``(N, W)``, ``row`` ``(N, ROW)`` 16-bit words and
    ``j`` ``(N,)`` unsigned 16-bit words, each instance's d ``(N, W)``."""
    y, row, j = (np.asarray(a, dtype=np.uint16) for a in (y, row, j))
    N, W = y.shape
    if row.ndim != 2 or row.shape[0] != N or j.shape != (N,):
        raise ValueError(f"y {y.shape}, row {row.shape} and j {j.shape} are not (N, W), (N, ROW) and (N,)")
    segments(row.shape[1], OFF, G, W)
    inside = j.astype(np.int64) < G
    at = OFF + np.where(inside, j, 0).astype(np.int64)[:, None] * W + np.arange(W)[None, :]
    return np.where(inside[:, None], np.take_along_axis(row, at, axis=1) ^ y, 0xFFFF).astype(np.uint16)


def shared_row_read_reference(row, j, W: int, G: int = 1, OFF: int = 0) -> np.ndarray:
    """What `SharedRowSelect_v1{W, ROW, OFF, G}` computes, the W words `SharedRowRead_v1` gives FN's PORT, for a batch: ``row``
    ``(N, ROW)`` 16-bit words and ``j`` ``(N,)`` unsigned 16-bit words, each instance's words ``(N, W)``, 0 where ``j >= G``."""
    row, j = np.asarray(row, dtype=np.uint16), np.asarray(j, dtype=np.uint16)
    if row.ndim != 2 or j.shape != (row.shape[0],):
        raise ValueError(f"row {row.shape} and j {j.shape} are not (N, ROW) and (N,)")
    segments(row.shape[1], OFF, G, W)
    inside = j.astype(np.int64) < G
    at = OFF + np.where(inside, j, 0).astype(np.int64)[:, None] * W + np.arange(W)[None, :]
    return np.where(inside[:, None], np.take_along_axis(row, at, axis=1), 0).astype(np.uint16)
