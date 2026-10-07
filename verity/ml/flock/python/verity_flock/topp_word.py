"""``TopPMaskWordx{V}_v1``, the top-p keep word, as a circuit: bit for bit ``sampling.topp_keep``.

The word runs vLLM's split top-p pipeline (``topp_split``: the stats kernel, five step / combine rounds of F = 8 pivots, the
mask kernel) for every served split count S in {1, 2, 4, 8, 16, 32} and keeps the pipeline the ``splits`` operand names (word 0
at any other value). At V = 128256 that is about 10^11 ANDs, more than one ``gf2`` circuit can hold, so the circuit is a
composition: :func:`keep_word` is written once over vectors of words (:class:`Vec`) with word-level pieces (:data:`OPS`, each a
``gf2`` circuit built once per constant-operand variant by :func:`build`), and an :class:`Engine` either counts every variant's
instances (``T = 0``: the export's composed record) or evaluates every instance on T test vectors (the check against
``topp_keep``). Both run the same builder, so the circuit counted is the circuit checked.

The circuit computes the reference's function, not each of its operations. Where they differ, the difference cannot reach the
keep bits:

* butterfly reductions build only the lanes their result reads (lane 0; lanes 0 and S/2 for the fma-contracted ``Z``);
* counts (finite lanes, ``tcnt``, ``nmin``, ``numdup``, ``numkeep``, the tie prefix sums) are integers: the reference's f32
  sums of 0/1 terms are exact below 2^24, and ``num_finite < 1`` is "no lane > -inf";
* max / min reductions pair their terms in any order: they propagate NaN, and a result differs at most in the sign of a zero,
  which no later operation distinguishes (it only enters ``exp``, a compare, or an add to a nonzero value);
* NaN payloads are free: the keep bits come from compares, which see only whether a value is a NaN;
* a padded lane of the step kernel (logit -inf) is above no pivot: its probability is +0 or NaN and every pivot is >= +0 or
  NaN (``lo >= 1e-30`` or NaN, times an ``exp``), so it adds the constants 0 and +inf to the block sums and minima.

Where the two warp halves disagree on stopping, every lane follows its own half, as ``sampling.topp_keep`` states it: a lane of a
stopped half is kept and left out of the tie count.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import fp, gf2
from . import tail_pieces as TP
from .gf2 import (
    NOT,
    ONE,
    OR,
    ZERO,
    Circuit,
    add,
    and_reduce,
    const_bits,
    mux,
    or_reduce,
    sub,
)

# the pipeline, its word gates and their constants are written once, in the registry (`topp_keep_builder`, `topp_word_gates`);
# F, R, CW, WIDTH and SPLIT_COUNTS are re-exported
from verity_vllm.program.registry.topp_keep_builder import F, R, SPLIT_COUNTS, keep_word  # noqa: E402, F401
from verity_vllm.program.registry.topp_keep_builder import Engine as _Engine  # noqa: E402
from verity_vllm.program.registry.topp_word_gates import CW, EPS9, HALF, LOG2E, M126, QNAN, WIDTH  # noqa: E402, F401


def _c(v: int):
    return const_bits(v & 0xFFFFFFFF, 32)


# --- the pieces ------------------------------------------------------------------------------------------------------------

def tl_exp(C, x):
    """``topp_split.tl_exp``: MUFU.EX2(x * log2e), with ptxas' non-ftz path (t < -126: EX2(t / 2) squared)."""
    t = fp.f32_mul(C, x, _c(LOG2E), ir_nan=True)
    small = TP.f32_gt(C, _c(M126), t)[0]
    e = TP.mufu_ex2(C, mux(C, small, fp.f32_mul(C, t, _c(HALF), ir_nan=True), t))
    return mux(C, small, fp.f32_mul(C, e, e, ir_nan=True), e)


def tl_div(C, a, b, r):
    """``div.full.f32``'s a / b from ``r = DivFullRcp(b)``: ``F32Mul(DivFullScaleA(a, b), r)``."""
    return fp.f32_mul(C, TP.div_full_scale_a(C, a, b), r, ir_nan=True)


def f32_ge(C, a, b):
    return [OR(C, TP.f32_gt(C, a, b)[0], TP.f32_eq(C, a, b)[0])]


def f32_near(C, a, b):
    """``|a - b| < 1e-9`` in f32."""
    d = TP.f32_sub(C, a, b)
    return TP.f32_gt(C, _c(EPS9), d[:31] + [ZERO])


def py_max(C, a, b):
    """Python's ``max(a, b)`` on f32 values: ``b > a ? b : a``."""
    return mux(C, TP.f32_gt(C, b, a)[0], b, a)


def py_min(C, a, b):
    """Python's ``min(a, b)``: ``b < a ? b : a``."""
    return mux(C, TP.f32_gt(C, a, b)[0], b, a)


def nan_max(C, a, b):
    """numpy's ``maximum``: NaN when either is a NaN."""
    nan = OR(C, TP._is_nan(C, a), TP._is_nan(C, b))
    return mux(C, nan, _c(QNAN), py_max(C, a, b))


def nan_min(C, a, b):
    nan = OR(C, TP._is_nan(C, a), TP._is_nan(C, b))
    return mux(C, nan, _c(QNAN), py_min(C, a, b))


def int_to_f32(C, a):
    """A count as the f32 it equals (exact below 2^24)."""
    return TP._i32_to_f32(C, list(a) + [ZERO] * (32 - len(a)))


def numkeep(C, nd, q):
    """``combine``'s ``numkeep = min(max(numdup - qi, 1), numdup)``, qi the u32 truncation of q (0 unless q is finite and > 0),
    saturated at 2^CW - 1 (numdup < 2^CW, so the result is the same)."""
    s, e, m = q[31], list(q[23:31]), list(q[:23])
    pos = C.AND(NOT(s), NOT(and_reduce(C, e)))
    ge1 = NOT(TP._ult_const(C, e, 127))
    big = NOT(TP._ult_const(C, e, 127 + CW))
    ip = gf2.shr(C, m + [ONE], sub(C, const_bits(150, 8), e, 8)[0][:5])[:CW]
    qi = [C.AND(pos, OR(C, big, C.AND(ge1, b))) for b in ip]
    lt = sub(C, qi, nd, CW)[1]
    r = mux(C, lt, sub(C, nd, qi, CW)[0], const_bits(1, CW))
    nz = or_reduce(C, nd)
    return [C.AND(nz, b) for b in r]


def word_eq(C, a, b):
    return [and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])]


#: name -> (piece(C, *operands) -> bits, operand kinds, result kind)
OPS = {
    "sub": (TP.f32_sub, "ff", "f"),
    "add": (lambda C, a, b: fp.f32_add(C, a, b, ir_nan=True), "ff", "f"),
    "mul": (lambda C, a, b: fp.f32_mul(C, a, b, ir_nan=True), "ff", "f"),
    "fma": (fp.f32_fma, "fff", "f"),
    "exp": (tl_exp, "f", "f"),
    "rcp": (TP.div_full_rcp, "f", "f"),
    "div": (tl_div, "fff", "f"),
    "log": (TP.nv_logf, "f", "f"),
    "gt": (TP.f32_gt, "ff", "b"),
    "ge": (f32_ge, "ff", "b"),
    "eq": (TP.f32_eq, "ff", "b"),
    "near": (f32_near, "ff", "b"),
    "max": (py_max, "ff", "f"),
    "min": (py_min, "ff", "f"),
    "nanmax": (nan_max, "ff", "f"),
    "nanmin": (nan_min, "ff", "f"),
    "sel": (lambda C, c, a, b: mux(C, c[0], a, b), "bff", "f"),
    "seli": (lambda C, c, a, b: mux(C, c[0], a, b), "bii", "i"),
    "selb": (lambda C, c, a, b: mux(C, c[0], a, b), "bbb", "b"),
    "gate": (lambda C, c, a: [C.AND(c[0], x) for x in a], "bf", "f"),
    "gatei": (lambda C, c, a: [C.AND(c[0], x) for x in a], "bi", "i"),
    "and": (lambda C, a, b: [C.AND(a[0], b[0])], "bb", "b"),
    "or": (lambda C, a, b: [OR(C, a[0], b[0])], "bb", "b"),
    "xor": (lambda C, a, b: [a[0] ^ b[0]], "bb", "b"),
    "xorf": (lambda C, a, b: [x ^ y for x, y in zip(a, b)], "ff", "f"),
    "xori": (lambda C, a, b: [x ^ y for x, y in zip(a, b)], "ii", "i"),
    "not": (lambda C, a: [NOT(a[0])], "b", "b"),
    "iadd": (lambda C, a, b: add(C, a, b, width=CW), "ii", "i"),
    "ile": (lambda C, a, b: [NOT(sub(C, b, a, CW)[1])], "ii", "b"),
    "ilt": (lambda C, a, b: [sub(C, a, b, CW)[1]], "ii", "b"),
    "i2f": (int_to_f32, "i", "f"),
    "b2i": (lambda C, a: [a[0]] + [ZERO] * (CW - 1), "b", "i"),
    "numkeep": (numkeep, "if", "i"),
    "weq": (word_eq, "ff", "b"),
}
#: display names of the pieces (the export's subcircuit names)
NAMES = {
    "sub": "f32 subtract", "add": "f32 add", "mul": "f32 multiply", "fma": "f32 fused multiply-add", "exp": "tl.exp (MUFU.EX2)",
    "rcp": "div.full reciprocal (DivFullRcp)", "div": "div.full (scale, times the reciprocal)", "log": "tl.log (__nv_logf)",
    "gt": "f32 compare (>)", "ge": "f32 compare (>=)", "eq": "f32 compare (==)", "near": "|a - b| < 1e-9",
    "max": "max (Python)", "min": "min (Python)", "nanmax": "max (numpy)", "nanmin": "min (numpy)", "sel": "select (f32)",
    "seli": "select (count)", "selb": "select (bit)", "gate": "gate (f32)", "gatei": "gate (count)", "and": "AND", "or": "OR",
    "xor": "XOR", "xorf": "XOR (f32)", "xori": "XOR (count)", "not": "NOT", "iadd": "count add", "ile": "count compare (<=)",
    "ilt": "count compare (<)", "i2f": "count to f32", "b2i": "bit to count", "numkeep": "numkeep (truncate, clamp)",
    "weq": "word compare (==)",
}


def build(name: str, key: tuple) -> Circuit:
    """Piece ``name`` with operand j the constant ``key[j]`` (None: the input ``in{j}``); its result is the output ``out``."""
    fn, ins, _out = OPS[name]
    C = Circuit(cse=True)
    ops = [const_bits(c, WIDTH[k]) if c is not None else C.input(f"in{j}", WIDTH[k]) for j, (c, k) in enumerate(zip(key, ins))]
    C.output("out", fn(C, *ops))
    return C


# --- the engine: the registry's builder, the pieces' circuits ---------------------------------------------------------------

class Engine(_Engine):
    """`topp_keep_builder.Engine` with each operation computed by its piece's ``gf2`` circuit (:func:`build`, once per constant-operand
    variant): counting its instances (``T = 0``) or evaluating them on T test vectors."""

    def __init__(self, T: int = 0):
        super().__init__(T)
        self._circ: dict = {}

    def _circuit(self, name, key):
        if (name, key) not in self._circ:
            self._circ[(name, key)] = build(name, key)
        return self._circ[(name, key)]

    def _run(self, name, key, ins):
        """The piece's results on ``ins`` (per free operand a (T', m) array)."""
        C = self._circuit(name, key)
        Tp, m = ins[0].shape
        N = Tp * m
        free = [j for j, c in enumerate(key) if c is None]
        kinds = OPS[name][1]
        iv = {f"in{j}": _pack(x.reshape(-1), WIDTH[kinds[j]]) for j, x in zip(free, ins)}
        outs, _ = C.evaluate(iv, N)
        return _unpack(outs["out"], N).reshape(Tp, m)


def _pack(x, w: int) -> list[int]:
    return [int.from_bytes(np.packbits(((x >> np.uint64(j)) & np.uint64(1)).astype(np.uint8), bitorder="little").tobytes(), "little")
            for j in range(w)]


def _unpack(bits, N: int):
    nb = (N + 7) // 8
    out = np.zeros(N, np.uint64)
    for j, b in enumerate(bits):
        a = np.unpackbits(np.frombuffer(b.to_bytes(nb, "little"), np.uint8), bitorder="little")[:N]
        out |= a.astype(np.uint64) << np.uint64(j)
    return out


def evaluate(x_words, p_words, splits_words, parts_of=None):
    """The circuit's keep bits (T, V) on T rows (uint32 logit words (T, V), p and splits words (T,))."""
    x_words = np.asarray(x_words, dtype=np.uint64)
    T, V = x_words.shape
    E = Engine(T)
    out = keep_word(E, E.free(V, 32, x_words), E.free(1, 32, np.asarray(p_words, np.uint64).reshape(T, 1)),
                    E.free(1, 32, np.asarray(splits_words, np.uint64).reshape(T, 1)), parts_of)
    return out.d.astype(bool), E


def count(V: int, p: int | None = None, splits: int | None = None) -> Counter:
    """The circuit's piece instances (stage path, piece, constant operands) -> count, with p / splits free or constant."""
    E = Engine(0)
    pv = E.free(1) if p is None else E.const([p])
    sv = E.free(1) if splits is None else E.const([splits])
    keep_word(E, E.free(V), pv, sv)
    return E.counts
