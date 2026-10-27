"""``nn.Embedding``'s backward in training, as Boolean Definitions: the gates of torch 2.13/2.14's CUDA
``embedding_dense_backward`` (Embedding.cu, EmbeddingBackwardKernel.cu; the two versions' kernels are the same) on bf16
gradients, for one vocabulary entry v and one feature, every NaN of an f32 operation the GPU's 0x7FFFFFFF and every rounding to
bf16 F2FP's, none flushed. smollm2-135m has no ``pad_token_id``, so ``padding_idx`` is -1 and no row is skipped. A token id is
the int64 id's low 32 bits; the forward's gather asserts every id is below V, so two ids are equal exactly when those bits are.

    EmbeddingBackwardSortStep_v1           (carry: Tuple<Value<4>, F32, F32>, t: U32, g: BF16, v: U32) -> Tuple<carry, Value<1>>
    EmbeddingBackwardSortRun_v1{N}         (carry, v: U32, t: Array<N, U32>, g: Array<N, BF16>) -> carry
    EmbeddingBackwardSortEnd_v1            (carry) -> BF16
    EmbeddingBackwardTorch_v1{M}           (v: U32, t: Array<M, U32>, g: Array<M, BF16>) -> BF16           M > 3072
    EmbeddingBackwardFeatureChunk_v1{N}    (acc: BF16, v: U32, t: Array<N, U32>, g: Array<N, BF16>) -> BF16  N <= 32
    EmbeddingBackwardFeature_v1{M}         (v: U32, t: Array<M, U32>, g: Array<M, BF16>) -> BF16           M <= 3072
    EmbeddingBackwardEntry_v1{M, H}        (v: U32, t: Array<M, U32>, g: Array<H, Array<M, BF16>>) -> Array<H, BF16>
    EmbeddingBackwardTied_v1{M, V, H}      (t: Array<M, U32>, g: Array<M, Array<H, BF16>>, w: Array<V, Array<H, BF16>>)
                                           -> Array<V, Array<H, BF16>>

**Above 3072 indices** (training's B·S = 4096), ``embedding_dense_backward_cuda`` stable-sorts the ids (``radix_sort_pairs`` of
(id, position)), so each entry's segment lists its tokens in position order. Under ``use_deterministic_algorithms(True)`` the
fused atomic kernel is off, and the two passes are ``compute_grad_weight`` and ``sum_and_scatter``. The first splits each segment
into partials of NROWS_PER_THREAD = 10 tokens and sums each from +0 in order, ``weight += g * scale`` with scale 1.0 (an FFMA by 1
or an FMUL and an FADD, either RN(weight + g)). The second sums an entry's partials from +0 in order and stores the f32 total as
bf16; an entry no token has stays +0. One pass over the tokens in position order computes that for an entry v with wiring that
doesn't depend on the ids (`EmbeddingBackwardSortStep`): it keeps the count of v's tokens so far mod 10, the open partial p and
the total; at a token of v whose count is 0 mod 10 it closes p into the total and opens p = +0 + g (`_plus_zero`), and at any
other token of v it adds g to p, so each token takes one FADD. p and the total start at +0, so the first close adds +0 + +0.
`EmbeddingBackwardSortEnd` closes the last partial and rounds. `EmbeddingBackwardTorch{M}` chains `EmbeddingBackwardSortRun`s of
at most RUN = 8192 tokens, a proof block each, passing the carry.

**At most 3072 indices** (the 2-layer step at M = 256), it runs ``embedding_backward_feature_kernel``: a block of 32 warps per
32 features walks the tokens in chunks of 32 consecutive positions. In each chunk the first warp holding v's id sums the chunk's
tokens of v from its own f32 gradient (not from +0) in position order, rounds the sum to bf16, and adds it into the bf16 grad
weight, ``grad_weight += bf16(sum)``: c10's BFloat16 ``+`` is an FADD of the two widened words and an F2FP. A chunk without v
leaves it alone. `EmbeddingBackwardFeatureChunk{N}` is one chunk; `EmbeddingBackwardFeature{M}` starts at +0 and chains the
M / 32 chunks, the last of M mod 32 tokens.

**The step's blocks** wire these with no gate of their own. `EmbeddingBackwardEntry{M, H}` is entry v's row, its kernel
(`kernel(M)`) on each of the H features' columns. `EmbeddingBackwardTied{M, V, H}` is the tied weight's gradient: the
embedding's output gradient g, committed token-major, transposed by `Transpose_v1` tiles of at most 128 x 128 (`transpose`),
each entry's row with its id a constant of the circuit, then c10's bf16 ``+`` of the LM head's dW w and that row,
`Bf16AddF2fp_v2` on each word (autograd adds the two uses' gradients, the LM head's first).
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.refs import Coll, Explicit, concat, strided_view
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple

from .torch_attention import SM120, SM120_SHAPES, SM120_STEP
from .torch_train import Transpose

__all__ = ["EmbeddingBackwardEntry", "EmbeddingBackwardFeature", "EmbeddingBackwardFeatureChunk", "EmbeddingBackwardSortEnd",
           "EmbeddingBackwardSortRun", "EmbeddingBackwardSortStep", "EmbeddingBackwardTied", "EmbeddingBackwardTorch", "kernel",
           "plus_zero_bf16", "transpose"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
U32 = Array(32, BIT)
COUNT = Array(4, BIT)
CARRY = Tuple(COUNT, F32, F32)
_NAN = fp.F32_NAN_PTX
#: NROWS_PER_THREAD: the tokens of one partial sum
NROWS = 10
#: the most indices the feature kernel takes, and the tokens of one of its chunks
FEATURE_MAX, CHUNK = 3072, 32
#: the tokens of one `EmbeddingBackwardSortRun` in `EmbeddingBackwardTorch`, a proof block
RUN = 8192
M_MAX = 1 << 24
#: a `Transpose_v1` tile's most rows and columns: a row of 128 bf16 words is one 256 B commitment row
TILE = 128
_SORT = f"at M = 4,096 in {SM120_SHAPES}, all 576 features of 65 entries (the model: of all 3,859 the tokens hold)"
_IN_SORT = f"exact-model-tested; inside EmbeddingBackwardTorch_v1, {_SORT}: {SM120}"
_FEATURE = f"at M = 256 in {SM120_STEP}, all 576 features of the 253 entries its tokens hold and of 32 they don't"


def _add(C, a, b):
    return fp.f32_add(C, a, b, nan_word=_NAN)


def _bf16(C, x):
    return fp.f32_to_bf16(C, x, W.BF16_NAN_F2FP)


def _eq(C, a, b):
    return and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])


def plus_zero_bf16(C, g):
    """+0 + f32(g) for a bf16 word g, as an FADD makes it, in 39 ANDs: a NaN is 0x7FFFFFFF, -0 is +0, and every other word is
    itself widened (``test_plus_zero_is_the_fadd_on_every_bf16_word``)."""
    e, m = or_reduce(C, g[7:15]), or_reduce(C, g[:7])
    nan = C.AND(and_reduce(C, g[7:15]), m)
    rest = OR(C, e, m)
    sign = C.AND(g[15], C.AND(rest, NOT(nan)))
    return [nan] * 16 + [OR(C, b, nan) for b in g[:15]] + [sign]


def _sort_step(C, c, p, tot, t, g, v):
    hit = _eq(C, t, v)
    start = C.AND(hit, and_reduce(C, [NOT(b) for b in c]))
    gf = fp.bf16_to_f32(g)
    r = _add(C, mux(C, start, tot, p), mux(C, start, p, gf))
    nine = and_reduce(C, [c[0], NOT(c[1]), NOT(c[2]), c[3]])
    inc = add(C, c, const_bits(1, 4), width=4)
    nc = mux(C, hit, [b ^ C.AND(nine, b) for b in inc], c)
    np_ = mux(C, start, plus_zero_bf16(C, g), mux(C, hit, r, p))
    return [nc, np_, mux(C, start, r, tot), [hit]]


def _sort_end(C, c, p, tot):
    return _bf16(C, _add(C, tot, p))


def _feature_chunk(C, S, acc, v, *w):
    t, g = w[:S.N], w[S.N:]
    have, s = ZERO, fp.f32_const(0)
    for ti, gi in zip(t, g):
        hit = _eq(C, ti, v)
        gf = fp.bf16_to_f32(gi)
        s = mux(C, hit, mux(C, have, _add(C, s, gf), gf), s)
        have = OR(C, have, hit)
    return mux(C, have, _bf16(C, _add(C, fp.bf16_to_f32(acc), fp.bf16_to_f32(_bf16(C, s)))), acc)


def _refusal(k: str, v: Any, lo: int, hi: int) -> str | None:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        return f"{k} = {v!r} is outside {lo}..{hi}, the range of the kernel this is"
    return None


def _refuse(k: str, lo: int, hi: int):
    def refuse(b: dict) -> str | None:
        return _refusal(k, b[k], lo, hi) if b.get(k) is not None else None
    return refuse


def _statics(S: Any, k: str, lo: int, hi: int) -> Any:
    if r := _refusal(k, getattr(S, k), lo, hi):
        raise ValueError(r)
    return S


def _tokens(n: int) -> tuple:
    return ("t", Array(n, U32)), ("g", Array(n, BF16))


def _zeros(B, t) -> Coll:
    z = B.call(BL.constant(0))
    return Coll(t, concat([z.refs] * t.leaves))


EmbeddingBackwardSortStep = definition(
    "EmbeddingBackwardSortStep", 1, _sort_step, (("carry", CARRY), ("t", U32), ("g", BF16), ("v", U32)), Tuple(CARRY, BIT),
    doc="one token of embedding_dense_backward's sort path for entry v: with c the count of v's tokens so far mod 10, at t = v "
        "and c = 0 the total += p and p = +0 + g, at t = v otherwise p += g; c counts; the second output is t = v",
    conformance=_IN_SORT)
EmbeddingBackwardSortEnd = definition(
    "EmbeddingBackwardSortEnd", 1, _sort_end, (("carry", CARRY),), BF16,
    doc="sum_and_scatter's store for entry v: bf16(total + p), the last partial closed", conformance=_IN_SORT)


def _sort_run(B, S, carry, v, t, g):
    return B.scan(EmbeddingBackwardSortStep, carry, xs=(t, g), shared=(v,))[0]


EmbeddingBackwardSortRun = CompositeDefinition(
    "EmbeddingBackwardSortRun", 1, ("N",),
    lambda S: ((("carry", CARRY), ("v", U32)) + _tokens(_statics(S, "N", 1, M_MAX).N), CARRY), _sort_run,
    doc="N consecutive tokens of embedding_dense_backward's sort path for entry v: EmbeddingBackwardSortStep's scan from carry",
    conformance=_IN_SORT, refuse=_refuse("N", 1, M_MAX))


def _sort(B, S, v, t, g):
    carry = _zeros(B, CARRY)
    for a in range(0, S.M, RUN):
        n = min(RUN, S.M - a)
        carry = B.call(bind(EmbeddingBackwardSortRun, N=n), carry, v, t[a:a + n], g[a:a + n])
    return B.call(EmbeddingBackwardSortEnd, carry)


EmbeddingBackwardTorch = CompositeDefinition(
    "EmbeddingBackwardTorch", 1, ("M",), lambda S: ((("v", U32),) + _tokens(_statics(S, "M", FEATURE_MAX + 1, M_MAX).M), BF16),
    _sort,
    doc="embedding_dense_backward_cuda's sort path (M > 3072, deterministic) for entry v and one feature over M tokens: "
        "partials of 10 of v's tokens summed from +0 in position order, the partials summed from +0, rounded to bf16; "
        "EmbeddingBackwardSortRun on runs of 8192 tokens from +0, then EmbeddingBackwardSortEnd",
    conformance=f"exact-model-tested; {_SORT}: {SM120}", refuse=_refuse("M", FEATURE_MAX + 1, M_MAX))
EmbeddingBackwardFeatureChunk = specialized(
    "EmbeddingBackwardFeatureChunk", 1, ("N",), lambda S: lambda C, *w: _feature_chunk(C, S, *w),
    lambda S: ((("acc", BF16), ("v", U32)) + _tokens(_statics(S, "N", 1, CHUNK).N), BF16),
    doc="one chunk of N <= 32 tokens of embedding_backward_feature_kernel for entry v: v's tokens summed in f32 from the "
        "first's gradient in position order, then acc = bf16(acc + bf16(sum)); acc when no token is v",
    refuse=_refuse("N", 1, CHUNK), conformance=f"exact-model-tested; inside EmbeddingBackwardFeature_v1, {_FEATURE}: {SM120}")


def _feature(B, S, v, t, g):
    acc = _zeros(B, BF16)
    for a in range(0, S.M, CHUNK):
        n = min(CHUNK, S.M - a)
        acc = B.call(bind(EmbeddingBackwardFeatureChunk, N=n), acc, v, t[a:a + n], g[a:a + n])
    return acc


EmbeddingBackwardFeature = CompositeDefinition(
    "EmbeddingBackwardFeature", 1, ("M",), lambda S: ((("v", U32),) + _tokens(_statics(S, "M", 1, FEATURE_MAX).M), BF16),
    _feature,
    doc="embedding_dense_backward_cuda's feature kernel (M <= 3072) for entry v and one feature over M tokens: "
        "EmbeddingBackwardFeatureChunk on each 32 consecutive tokens, from +0",
    conformance=f"exact-model-tested; {_FEATURE}: {SM120}", refuse=_refuse("M", 1, FEATURE_MAX))

# --- the step's blocks ------------------------------------------------------------------------------------------------------


def kernel(M: int) -> Any:
    """embedding_dense_backward_cuda's kernel at M ids, bound: the feature kernel up to 3072, else the sort path."""
    return bind(EmbeddingBackwardFeature if M <= FEATURE_MAX else EmbeddingBackwardTorch, M=M)


def _refuse_all(*keys: str):
    def refuse(b: dict) -> str | None:
        return next((r for k in keys if (r := _refuse(k, 1, M_MAX)(b))), None)
    return refuse


def _entry(B, S, v, t, g):
    return B.batch(kernel(S.M), v, t, g, axes=(None, None, 0))


EmbeddingBackwardEntry = CompositeDefinition(
    "EmbeddingBackwardEntry", 1, ("M", "H"),
    lambda S: ((("v", U32), ("t", Array(_statics(S, "M", 1, M_MAX).M, U32)),
                ("g", Array(_statics(S, "H", 1, M_MAX).H, Array(S.M, BF16)))), Array(S.H, BF16)), _entry,
    doc="entry v's row of embedding_dense_backward_cuda over M tokens: kernel(M) on each of the H features' gradient columns",
    conformance=f"exact-model-tested; at M = 256, 13 features of the 253 entries the tokens of {SM120_STEP} hold, and at "
                f"M = 4,096, 8 features of 4 entries in {SM120_SHAPES}: {SM120}", refuse=_refuse_all("M", "H"))


def transpose(B, x: Coll, R: int, C: int) -> Coll:
    """A token-major R x C matrix of bf16 words as its C x R transpose, by `Transpose_v1` tiles of at most TILE x TILE, one
    batch a tile shape."""
    L = BF16.leaves
    rows = [(r, min(TILE, R - r)) for r in range(0, R, TILE)]
    cols = [(c, min(TILE, C - c)) for c in range(0, C, TILE)]
    out = {}
    for rn, cn in sorted({(rn, cn) for _, rn in rows for _, cn in cols}):
        at = [(r, c) for r, n in rows for c, k in cols if (n, k) == (rn, cn)]
        tiles = Coll(Array(len(at), Array(rn, Array(cn, BF16))),
                     concat([strided_view(x, L * (r * C + c), L * cn, L * C, 1, L * rn * cn) for r, c in at]))
        y = B.batch(bind(Transpose, R=rn, C=cn), tiles, axes=(0,))
        out.update({rc: y[i] for i, rc in enumerate(at)})
    return Coll(Array(C, Array(R, BF16)), concat([out[(r, j - j % TILE)][j % TILE].refs for j in range(C) for r, _ in rows]))


def _ids(B, n: int) -> Coll:
    """The entries' ids 0 .. n - 1 as U32 words of the circuit's constant bits."""
    bit = (B.call(BL.constant(0)).refs[0], B.call(BL.constant(1)).refs[0])
    return Coll(Array(n, U32), Explicit(tuple(bit[(v >> i) & 1] for v in range(n) for i in range(32))))


def _tied(B, S, t, g, w):
    e = B.batch(bind(EmbeddingBackwardEntry, M=S.M, H=S.H), _ids(B, S.V), t, transpose(B, g, S.M, S.H),
                axes=(0, None, None))
    n = S.V * S.H
    return B.batch(EW.Bf16AddF2fp, w.reshape(Array(n, BF16)), e.reshape(Array(n, BF16)), axes=(0, 0)).reshape(
        Array(S.V, Array(S.H, BF16)))


EmbeddingBackwardTied = CompositeDefinition(
    "EmbeddingBackwardTied", 1, ("M", "V", "H"),
    lambda S: ((("t", Array(_statics(S, "M", 1, M_MAX).M, U32)), ("g", Array(S.M, Array(_statics(S, "H", 1, M_MAX).H, BF16))),
                ("w", Array(_statics(S, "V", 1, M_MAX).V, Array(S.H, BF16)))), Array(S.V, Array(S.H, BF16))), _tied,
    doc="the tied embedding's gradient over M tokens: g (M x H, token-major) through transpose, EmbeddingBackwardEntry_v1 on "
        "each of the V entries, then Bf16AddF2fp_v2 of the LM head's dW w and each entry's row",
    conformance="exact-model-tested; against the model (tests/test_torch_embedding.py), whose tied add is the GPU's on every "
                f"word of {SM120_STEP}: {SM120}",
    refuse=_refuse_all("M", "V", "H"))
