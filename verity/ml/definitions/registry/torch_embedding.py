"""``nn.Embedding`` in training as Boolean Definitions: its forward, and its backward, the gates of torch 2.13/2.14's CUDA
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
    EmbeddingTorch_v1{M, V, H}             (t: Array<M, U32>, w: Array<V, Array<H, BF16>>) -> Array<M, Array<H, BF16>>

Version 2 of the tied gradient, its function v1's, with KB = position_bits(M) + id_bits(V) and NS = KB:

    EmbeddingBackwardKey_v1{V}             (t: U32) -> Array<id_bits(V), Value<1>>
    EmbeddingBackwardSwap_v1{KB, H}        (a, b: Array<KB, Value<1>>, x, y: Array<H, BF16>) -> Tuple<a', b', x', y'>
    EmbeddingBackwardSort_v1{M, V, H}      (t: Array<M, U32>, g: Array<M, Array<H, BF16>>)
                                           -> Tuple<Array<M, Array<KB, Value<1>>>, Array<M, Array<H, BF16>>>
    EmbeddingBackwardSegmentStep_v1{V, NP, FEATURE}, EmbeddingBackwardSegmentRun_v1{N, V, NP, FEATURE}
                                           a sorted token's flags (seg, start), mark and shift (z, e)
    EmbeddingBackwardSortedStep_v1{FEATURE}, EmbeddingBackwardSortedRun_v1{N, FEATURE}
                                           (carry, f: Array<N, Array<2, Value<1>>>, g: Array<N, BF16>) -> Tuple<carry, Array<N, BF16>>
    EmbeddingBackwardMove_v1{NS, H, J}     (pa, sa, xa, pb, sb, xb) -> Tuple<Array<1, Value<1>>, Array<NS, Value<1>>, Array<H, BF16>>
    EmbeddingBackwardSlot_v1{H}            (p: Array<1, Value<1>>, x: Array<H, BF16>) -> Array<H, BF16>
    EmbeddingBackwardRoute_v1{M, V, H}     (p: Array<M, Array<1, Value<1>>>, s: Array<M, Array<NS, Value<1>>>, x: Array<M, Array<H, BF16>>)
                                           -> Array<V, Array<H, BF16>>
    EmbeddingBackwardTied_v2{M, V, H}      v1's signature

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
embedding's output gradient g, committed token-major, transposed by `Transpose_v1` tiles of at most 128 x 128
(`torch_attention.transpose`), each entry's row with its id a constant of the circuit, then c10's bf16 ``+`` of the LM head's
dW w and that row, `Bf16AddF2fp_v2` on each word (autograd adds the two uses' gradients, the LM head's first).
`EmbeddingTorch{M, V, H}` is the forward, ``index_select`` of the weight's rows by the M ids: `boolean_gather.Embedding_v2{V, H}`
on each id, the weight shared. torch device-asserts on an id outside [0, V), where the gather selects 0x7FFF.

**Version 2 of the tied gradient** runs v1's per-entry scan once over sorted tokens instead of once per entry. Each kernel's
order for entry v reads only v's tokens in position order, and on the feature kernel their chunks of 32 positions, so sorting
the tokens by (id, position) once keeps it. `EmbeddingBackwardSort` sorts the rows of g by that key with `sort_net(M)`, Batcher's
odd-even merge network (139,263 comparators at M = 4096), each comparator an `EmbeddingBackwardSwap` that carries the row; an
id outside [0, V) sorts as V (`EmbeddingBackwardKey`) and lands in no slot, as no v1 entry reads it. `EmbeddingBackwardSegmentRun`
walks the sorted ids once: seg marks an id's first token, start a partial's first (every 10th of the id's tokens) or, on the
feature kernel, a chunk's first, and mark the id's last token when the id is below V. `EmbeddingBackwardSortedRun`, on each
feature's column of the sorted rows, is v1's step with "a token of v" read as "the same id as the token before": the sort path
closes the partial into the total at start and resets both at seg, the feature kernel adds the chunk's bf16 sum into acc at
start; at an id's last token its out is v1's word for that entry. `EmbeddingBackwardRoute` takes each marked token's row to
slot id: down by z (the unmarked tokens before it, lowest bit first) so the marked rows are consecutive, then up by e (its id
less the marked tokens before it, highest bit first), the two shifting networks under which no two rows meet; a slot no row
reaches is +0. The tied add is v1's.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.scalar import gt_u
from verity.ml.circuits.refs import Affine, Coll, Explicit, Strided, broadcast, concat, strided_view, tuple_of
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple

from . import boolean_gather as BGA
from .torch_attention import GPU_SHAPES, GPU_STEP, ON_GPU, transpose, transposes

__all__ = ["EmbeddingBackwardEntry", "EmbeddingBackwardFeature", "EmbeddingBackwardFeatureChunk", "EmbeddingBackwardKey",
           "EmbeddingBackwardMove", "EmbeddingBackwardRoute", "EmbeddingBackwardSegmentRun", "EmbeddingBackwardSegmentStep",
           "EmbeddingBackwardSlot", "EmbeddingBackwardSort", "EmbeddingBackwardSortEnd", "EmbeddingBackwardSortRun",
           "EmbeddingBackwardSortStep", "EmbeddingBackwardSortedRun", "EmbeddingBackwardSortedStep", "EmbeddingBackwardSwap",
           "EmbeddingBackwardTied", "EmbeddingBackwardTiedV2", "EmbeddingBackwardTorch", "EmbeddingTorch", "id_bits", "kernel",
           "plus_zero_bf16", "position_bits", "sort_net"]

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
_SORT = f"at M = 4,096 in {GPU_SHAPES}, all 576 features of 65 entries (the model: of all 3,859 the tokens hold)"
_IN_SORT = f"exact-model-tested; inside EmbeddingBackwardTorch_v1, {_SORT}: {ON_GPU}"
_FEATURE = f"at M = 256 in {GPU_STEP}, all 576 features of the 253 entries its tokens hold and of 32 they don't"


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
    conformance=f"exact-model-tested; {_SORT}: {ON_GPU}", refuse=_refuse("M", FEATURE_MAX + 1, M_MAX))
EmbeddingBackwardFeatureChunk = specialized(
    "EmbeddingBackwardFeatureChunk", 1, ("N",), lambda S: lambda C, *w: _feature_chunk(C, S, *w),
    lambda S: ((("acc", BF16), ("v", U32)) + _tokens(_statics(S, "N", 1, CHUNK).N), BF16),
    doc="one chunk of N <= 32 tokens of embedding_backward_feature_kernel for entry v: v's tokens summed in f32 from the "
        "first's gradient in position order, then acc = bf16(acc + bf16(sum)); acc when no token is v",
    refuse=_refuse("N", 1, CHUNK), conformance=f"exact-model-tested; inside EmbeddingBackwardFeature_v1, {_FEATURE}: {ON_GPU}")


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
    conformance=f"exact-model-tested; {_FEATURE}: {ON_GPU}", refuse=_refuse("M", 1, FEATURE_MAX))

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
    conformance=f"exact-model-tested; at M = 256, 13 features of the 253 entries the tokens of {GPU_STEP} hold, and at "
                f"M = 4,096, 8 features of 4 entries in {GPU_SHAPES}: {ON_GPU}", refuse=_refuse_all("M", "H"))


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
    conformance="exact-model-tested; against numerics.torch_blocks' model (tests/test_torch_embedding.py), whose tied add is "
                f"the GPU's on every word of {GPU_STEP}: {ON_GPU}",
    refuse=_refuse_all("M", "V", "H"))


def _forward(B, S, t, w):
    return B.batch(bind(BGA.Embedding, V=S.V, H=S.H), t, w, axes=(0, None))


EmbeddingTorch = CompositeDefinition(
    "EmbeddingTorch", 1, ("M", "V", "H"),
    lambda S: ((("t", Array(_statics(S, "M", 1, M_MAX).M, U32)),
                ("w", Array(_statics(S, "V", 1, M_MAX).V, Array(_statics(S, "H", 1, M_MAX).H, BF16)))),
               Array(S.M, Array(S.H, BF16))), _forward,
    doc="the embedding's forward over M tokens, index_select of w's rows: boolean_gather.Embedding_v2 on each id, w shared; an "
        "id outside [0, V), where torch device-asserts, selects 0x7FFF",
    conformance=f"exact-model-tested; every word of its output at M = 256 in {GPU_STEP}: {ON_GPU}",
    refuse=_refuse_all("M", "V", "H"))

# --- version 2 of the tied gradient: the ids sorted once, then routed ------------------------------------------------------------

FLAGS = Array(2, BIT)
#: a token's one-bit flag as its own element, so an array of them is not one wide word
P1 = Array(1, BIT)
_CHUNK_BITS = CHUNK.bit_length() - 1
_ON_GPU_V2 = (f"at M = 256 in {GPU_STEP}, 13 features of the 253 entries its tokens hold, and at M = 4,096 in {GPU_SHAPES}, "
              f"8 features of the 37 entries kept, each with its ids renumbered by rank: {ON_GPU}")
_V2 = f"inside EmbeddingBackwardTied_v2, {_ON_GPU_V2}"


def _refuse_ranges(*ranges: tuple[str, int, int]):
    def refuse(b: dict) -> str | None:
        return next((r for k, lo, hi in ranges if (r := _refuse(k, lo, hi)(b))), None)
    return refuse


def position_bits(M: int) -> int:
    """The bits of a position among M tokens, at least 1."""
    return max(1, (M - 1).bit_length())


def id_bits(V: int) -> int:
    """The bits of a sort key's id: an id below V, or V itself for every id outside [0, V)."""
    return V.bit_length()


def sort_net(n: int) -> list[list[tuple[int, int]]]:
    """Batcher's odd-even merge sort on n items as its stages of disjoint comparators (a, b), a < b, each putting the lesser key
    at a: the network on the next power of two without the comparators that reach an index of n or more, the places of items
    greater than every other, which no comparator moves."""
    N = 1 << (n - 1).bit_length() if n > 1 else 1
    stages, p = [], 1
    while p < N:
        k = p
        while k:
            s = [(i + j, i + j + k) for j in range(k % p, N - k, 2 * k) for i in range(min(k, N - j - k))
                 if (i + j) // (2 * p) == (i + j + k) // (2 * p) and i + j + k < n]
            if s:
                stages.append(s)
            k //= 2
        p *= 2
    return stages


def _gather(parts: list) -> Coll:
    """Colls of one type as an Array, each run of equal-length Affine slices of one value at a constant step one Strided."""
    out, i = [], 0
    while i < len(parts):
        r, j, step = parts[i].refs, i + 1, 0
        if isinstance(r, Affine) and j < len(parts):
            def at(q, k: int) -> bool:
                return isinstance(q, Affine) and (q.space, q.idx, q.count) == (r.space, r.idx, r.count) and q.base == r.base + k
            q = parts[j].refs
            step = q.base - r.base if isinstance(q, Affine) and at(q, q.base - r.base) else 0
            while step > 0 and j < len(parts) and at(parts[j].refs, (j - i) * step):
                j += 1
        n = j - i
        out.append(r if n == 1 else Affine(r.space, r.idx, r.base, r.count * n) if step == r.count
                   else Strided(r.space, r.idx, r.base, r.count, step, 1, r.count * n))
        i = j
    return Coll(Array(len(parts), parts[0].type), concat(out))


def _fields(parts: list[Coll], n: int, off: int, width: int, t) -> Coll:
    """Field (off, width) of each member of the Affine Arrays ``parts`` (their members' total ``n``) as one Array of type t."""
    refs = []
    for c in parts:
        m, L = c.type.n, c.type.elem.leaves
        refs.append(strided_view(c, off, width, L, 1, width * m))
    return Coll(Array(n, t), concat(refs))


def _key(S: Any):
    NV = id_bits(S.V)
    return lambda C, t: mux(C, gt_u(C, const_bits(S.V, 32), t), t[:NV], const_bits(S.V, NV))


EmbeddingBackwardKey = specialized(
    "EmbeddingBackwardKey", 1, ("V",), _key,
    lambda S: ((("t", U32),), Array(id_bits(_statics(S, "V", 1, M_MAX).V), BIT)),
    doc="a token's id as the sort reads it: t when t < V, else V, so every id outside [0, V) sorts after the vocabulary's",
    refuse=_refuse("V", 1, M_MAX), conformance=f"exact-model-tested; {_V2}")


def _swap(S: Any):
    def build(C, a, b, *w):
        x, y = w[:S.H], w[S.H:]
        s = gt_u(C, a, b)
        d = [[C.AND(s, u ^ v) for u, v in zip(p, q)] for p, q in [(a, b), *zip(x, y)]]
        lo = [[u ^ e for u, e in zip(p, f)] for p, f in zip([a, *x], d)]
        hi = [[v ^ e for v, e in zip(q, f)] for q, f in zip([b, *y], d)]
        return [lo[0], hi[0], *lo[1:], *hi[1:]]
    return build


def _swap_signature(S: Any):
    k, r = Array(_statics(S, "KB", 1, 64).KB, BIT), Array(_statics(S, "H", 1, M_MAX).H, BF16)
    return (("a", k), ("b", k), ("x", r), ("y", r)), Tuple(k, k, r, r)


EmbeddingBackwardSwap = specialized(
    "EmbeddingBackwardSwap", 1, ("KB", "H"), _swap, _swap_signature,
    doc="one comparator of the sort: the keys a and b and their rows x and y, swapped when a > b (unsigned), so the lesser key "
        "and its row come first", refuse=_refuse_ranges(("KB", 1, 64), ("H", 1, M_MAX)), conformance=f"exact-model-tested; {_V2}")


def _sort_once(B, S, t, g):
    M, H, NP = S.M, S.H, position_bits(S.M)
    KB = NP + id_bits(S.V)
    ids = B.batch(bind(EmbeddingBackwardKey, V=S.V), t, axes=(0,))
    bit = (B.call(BL.constant(0)).refs[0], B.call(BL.constant(1)).refs[0])
    keys = [Coll(Array(KB, BIT), concat([Explicit(tuple(bit[(k >> i) & 1] for i in range(NP))), ids[k].refs])) for k in range(M)]
    rows = [g[k] for k in range(M)]
    swap = bind(EmbeddingBackwardSwap, KB=KB, H=H)
    for stage in sort_net(M):
        o = B.batch(swap, *(_gather([c[p[e]] for p in stage]) for c, e in ((keys, 0), (keys, 1), (rows, 0), (rows, 1))),
                    axes=(0,) * 4)
        for m, (a, b) in enumerate(stage):
            y = o[m]
            keys[a], keys[b], rows[a], rows[b] = y[0], y[1], y[2], y[3]
    return tuple_of(_gather(keys), _gather(rows))


def _sort_signature(S: Any):
    M, V, H = _statics(S, "M", 1, M_MAX).M, _statics(S, "V", 1, M_MAX).V, _statics(S, "H", 1, M_MAX).H
    return ((("t", Array(M, U32)), ("g", Array(M, Array(H, BF16)))),
            Tuple(Array(M, Array(position_bits(M) + id_bits(V), BIT)), Array(M, Array(H, BF16))))


EmbeddingBackwardSort = CompositeDefinition(
    "EmbeddingBackwardSort", 1, ("M", "V", "H"), _sort_signature, _sort_once,
    doc="the M tokens' rows of g sorted by (id, position), each with its key: the position's bits, then "
        "EmbeddingBackwardKey_v1's; EmbeddingBackwardSwap_v1 on each comparator of sort_net(M)'s stages, a batch a stage",
    conformance=f"exact-model-tested; {_V2}", refuse=_refuse_all("M", "V", "H"))


def _segment_types(V: int, NP: int, feature: int):
    NV = id_bits(V)
    carry = Tuple(BIT, Array(NV, BIT), Array(NP if feature else 4, BIT), Array(NP, BIT), Array(NV, BIT))
    return carry, Tuple(FLAGS, P1, Array(NP + NV, BIT))


def _segment_step(S: Any):
    NV, NP = id_bits(S.V), S.NP

    def build(C, started, prev, st, z, r, pos, i, nx, end):
        same = C.AND(started[0], _eq(C, i, prev))
        if S.FEATURE:
            start, st = OR(C, NOT(same), or_reduce(C, [a ^ b for a, b in zip(pos[_CHUNK_BITS:], st[_CHUNK_BITS:])])), pos
        else:
            c = [C.AND(same, b) for b in st]
            start = and_reduce(C, [NOT(b) for b in c])
            nine = and_reduce(C, [c[0], NOT(c[1]), NOT(c[2]), c[3]])
            st = [b ^ C.AND(nine, b) for b in add(C, c, const_bits(1, 4), width=4)]
        mark = C.AND(OR(C, end[0], NOT(_eq(C, i, nx))), NOT(_eq(C, i, const_bits(S.V, NV))))
        return [[ONE], i, st, add(C, z, [NOT(mark)], width=NP), add(C, r, [mark], width=NV), [NOT(same), start], [mark],
                z + sub(C, i, r, NV)[0]]
    return build


_SEGMENT = (("V", 1, M_MAX), ("NP", 1, 24), ("FEATURE", 0, 1))


def _segment_step_signature(S: Any):
    for k, lo, hi in _SEGMENT:
        _statics(S, k, lo, hi)
    carry, out = _segment_types(S.V, S.NP, S.FEATURE)
    return ((("carry", carry), ("pos", Array(S.NP, BIT)), ("id", Array(id_bits(S.V), BIT)), ("nx", Array(id_bits(S.V), BIT)),
             ("end", P1)), Tuple(carry, out))


EmbeddingBackwardSegmentStep = specialized(
    "EmbeddingBackwardSegmentStep", 1, ("V", "NP", "FEATURE"), _segment_step, _segment_step_signature,
    doc="one sorted token's flags, from its position pos, its id, the next token's id nx and end (the last token): seg (the "
        "first of its id), start (the first of a partial of 10 of its id's tokens, or with FEATURE of its id's tokens in a "
        "chunk of 32 positions), mark (the last of its id and the id below V), z (the unmarked tokens before it) and e (its "
        "id less the marked tokens before it); the carry holds whether a token came before, its id, the count mod 10 or "
        "position, z and the marked count",
    refuse=_refuse_ranges(*_SEGMENT), conformance=f"exact-model-tested; {_V2}")


def _segment_run_signature(S: Any):
    N = _statics(S, "N", 1, M_MAX).N
    (p, carry), *xs = _segment_step_signature(S)[0]
    return ((p, carry),) + tuple((n, Array(N, t)) for n, t in xs), Tuple(carry, Array(N, _segment_types(S.V, S.NP, S.FEATURE)[1]))


EmbeddingBackwardSegmentRun = CompositeDefinition(
    "EmbeddingBackwardSegmentRun", 1, ("N", "V", "NP", "FEATURE"), _segment_run_signature,
    lambda B, S, carry, *xs: B.scan(bind(EmbeddingBackwardSegmentStep, V=S.V, NP=S.NP, FEATURE=S.FEATURE), carry, xs=xs),
    doc="N consecutive sorted tokens' flags: EmbeddingBackwardSegmentStep_v1's scan from carry",
    refuse=_refuse_ranges(("N", 1, M_MAX), *_SEGMENT), conformance=f"exact-model-tested; {_V2}")


def _sorted_carry(feature: int):
    return Tuple(BF16, F32, BF16) if feature else Tuple(F32, F32)


def _sorted_step(S: Any):
    def build(C, *w):
        (seg, start), g = w[-2], w[-1]
        gf, keep = fp.bf16_to_f32(g), NOT(seg)
        if S.FEATURE:
            acc, s, o = w[:3]
            acc = [C.AND(keep, b) for b in mux(C, start, o, acc)]
            s = mux(C, start, gf, _add(C, s, gf))
            o = _bf16(C, _add(C, fp.bf16_to_f32(acc), fp.bf16_to_f32(_bf16(C, s))))
            return [acc, s, o, o]
        tot, p = w[:2]
        r = _add(C, mux(C, start, tot, p), mux(C, start, p, gf))
        tot = [C.AND(keep, b) for b in mux(C, start, r, tot)]
        p = mux(C, start, plus_zero_bf16(C, g), r)
        return [tot, p, _bf16(C, _add(C, tot, p))]
    return build


def _sorted_step_signature(S: Any):
    carry = _sorted_carry(_statics(S, "FEATURE", 0, 1).FEATURE)
    return (("carry", carry), ("f", FLAGS), ("g", BF16)), Tuple(carry, BF16)


EmbeddingBackwardSortedStep = specialized(
    "EmbeddingBackwardSortedStep", 1, ("FEATURE",), _sorted_step, _sorted_step_signature,
    doc="one sorted token of one feature, its flags f = (seg, start) and gradient g. The sort path: at start the total += p and "
        "p = +0 + g, else p += g, the total +0 at seg, out bf16(total + p). With FEATURE, the feature kernel: at start the "
        "chunk sum s = g and acc = the last out (+0 at seg), else s += g, out bf16(acc + bf16(s)). A token that is its id's "
        "last has out its entry's word",
    refuse=_refuse("FEATURE", 0, 1), conformance=f"exact-model-tested; {_V2}")


def _sorted_run_signature(S: Any):
    N, carry = _statics(S, "N", 1, M_MAX).N, _sorted_carry(_statics(S, "FEATURE", 0, 1).FEATURE)
    return (("carry", carry), ("f", Array(N, FLAGS)), ("g", Array(N, BF16))), Tuple(carry, Array(N, BF16))


EmbeddingBackwardSortedRun = CompositeDefinition(
    "EmbeddingBackwardSortedRun", 1, ("N", "FEATURE"), _sorted_run_signature,
    lambda B, S, carry, f, g: B.scan(bind(EmbeddingBackwardSortedStep, FEATURE=S.FEATURE), carry, xs=(f, g)),
    doc="N consecutive sorted tokens of one feature: EmbeddingBackwardSortedStep_v1's scan from carry, each token's out",
    refuse=_refuse_ranges(("N", 1, M_MAX), ("FEATURE", 0, 1)), conformance=f"exact-model-tested; {_V2}")


def _move(S: Any):
    def build(C, pa, sa, *w):
        xa, pb, sb, xb = w[:S.H], w[S.H], w[S.H + 1], w[S.H + 2:]
        go = C.AND(pb[0], sb[S.J])
        return [[OR(C, go, C.AND(pa[0], NOT(sa[S.J])))], mux(C, go, sb, sa), *(mux(C, go, u, v) for u, v in zip(xb, xa))]
    return build


def _move_signature(S: Any):
    s, r = Array(_statics(S, "NS", 1, 64).NS, BIT), Array(_statics(S, "H", 1, M_MAX).H, BF16)
    _statics(S, "J", 0, S.NS - 1)
    return (("pa", P1), ("sa", s), ("xa", r), ("pb", P1), ("sb", s), ("xb", r)), Tuple(P1, s, r)


EmbeddingBackwardMove = specialized(
    "EmbeddingBackwardMove", 1, ("NS", "H", "J"), _move, _move_signature,
    doc="one place of a routing stage: item a there and item b, each (present p, shift s, row x); b when b is present and moves "
        "(s's bit J), else a, present when it stays (not moving)",
    refuse=_refuse_ranges(("NS", 1, 64), ("H", 1, M_MAX), ("J", 0, 63)), conformance=f"exact-model-tested; {_V2}")
EmbeddingBackwardSlot = specialized(
    "EmbeddingBackwardSlot", 1, ("H",), lambda S: lambda C, p, *x: [[C.AND(p[0], b) for b in w] for w in x],
    lambda S: ((("p", P1), ("x", Array(_statics(S, "H", 1, M_MAX).H, BF16))), Array(S.H, BF16)),
    doc="a slot's row: x when an item is present, else +0", refuse=_refuse("H", 1, M_MAX),
    conformance=f"exact-model-tested; {_V2}")


def _route(B, S, p, s, x):
    M, V, H, NP, NV = S.M, S.V, S.H, position_bits(S.M), id_bits(S.V)
    zero = B.call(BL.constant(0)).refs[0]

    def absent(ns: int):
        return tuple(Coll(t, broadcast(*zero, t.leaves)) for t in (P1, Array(ns, BIT), Array(H, BF16)))

    def stage(items, src, ns: int, j: int) -> list:
        o = B.batch(bind(EmbeddingBackwardMove, NS=ns, H=H, J=j), *(_gather([it[f] for it in items]) for f in range(3)),
                    *(_gather([it[f] for it in src]) for f in range(3)), axes=(0,) * 6)
        return [tuple(o[m]) for m in range(len(items))]

    items, gone = [(p[k], s[k], x[k]) for k in range(M)], absent(NP + NV)
    for j in range(NP):
        items = stage(items, [items[k + (1 << j)] if k + (1 << j) < M else gone for k in range(M)], NP + NV, j)
    n, gone = min(M, V), absent(NV)
    items = [(a, Coll(Array(NV, BIT), b.refs.slice(NP, NP + NV)), c) for a, b, c in items[:n]] + [gone] * (V - n)
    held = [k < n for k in range(V)]
    for j in reversed(range(NV)):
        d = 1 << j
        live = [k for k in range(V) if held[k] or (k >= d and held[k - d])]
        out = stage([items[k] for k in live], [items[k - d] if k >= d and held[k - d] else gone for k in live], NV, j)
        held = [False] * V
        for k, it in zip(live, out):
            items[k], held[k] = it, True
    live = [k for k in range(V) if held[k]]
    rows = B.batch(bind(EmbeddingBackwardSlot, H=H), _gather([items[k][0] for k in live]), _gather([items[k][2] for k in live]),
                   axes=(0, 0))
    at = dict(zip(live, range(len(live))))
    return _gather([rows[at[k]] if k in at else gone[2] for k in range(V)])


def _route_signature(S: Any):
    M, V, H = _statics(S, "M", 1, M_MAX).M, _statics(S, "V", 1, M_MAX).V, _statics(S, "H", 1, M_MAX).H
    return ((("p", Array(M, P1)), ("s", Array(M, Array(position_bits(M) + id_bits(V), BIT))), ("x", Array(M, Array(H, BF16)))),
            Array(V, Array(H, BF16)))


EmbeddingBackwardRoute = CompositeDefinition(
    "EmbeddingBackwardRoute", 1, ("M", "V", "H"), _route_signature, _route,
    doc="the present items (p) of M, each with its row x, moved down by z (s's low position_bits(M) bits, lowest bit first) "
        "and then up by e (its high id_bits(V) bits, highest first) to V slots, EmbeddingBackwardMove_v1 a stage, then "
        "EmbeddingBackwardSlot_v1: +0 where no item lands. With z each item's count of absent items before it and e its slot "
        "less its place among the present items, the slots ascending, no two items meet and item k lands at k - z + e",
    conformance=f"exact-model-tested; {_V2}", refuse=_refuse_all("M", "V", "H"))


def _tied_sorted(B, S, t, g, w):
    M, V, H = S.M, S.V, S.H
    NP, NV, F = position_bits(M), id_bits(V), int(M <= FEATURE_MAX)
    KB, runs = NP + NV, [(a, min(RUN, M - a)) for a in range(0, M, RUN)]
    srt = B.call(bind(EmbeddingBackwardSort, M=M, V=V, H=H), t, g)
    keys, rows = srt[0], srt[1]
    bit = (B.call(BL.constant(0)).refs[0], B.call(BL.constant(1)).refs[0])
    ids = Coll(Array(M, Array(NV, BIT)), strided_view(keys, NP, NV, KB, 1, NV * M))
    xs = (Coll(Array(M, Array(NP, BIT)), strided_view(keys, 0, NP, KB, 1, NP * M)), ids,
          Coll(ids.type, concat([strided_view(keys, KB + NP, NV, KB, 1, NV * (M - 1)), ids[M - 1].refs]) if M > 1 else ids.refs),
          Coll(Array(M, P1), concat([broadcast(*bit[0], M - 1), Explicit((bit[1],))]) if M > 1 else Explicit((bit[1],))))
    carry, segs = _zeros(B, _segment_types(V, NP, F)[0]), []
    for a, n in runs:
        o = B.call(bind(EmbeddingBackwardSegmentRun, N=n, V=V, NP=NP, FEATURE=F), carry, *(c[a:a + n] for c in xs))
        carry = o[0]
        segs.append(o[1])
    cols = transpose(B, rows, M, H)
    cw = _sorted_carry(F).leaves
    carry, outs = _zeros(B, Array(H, _sorted_carry(F))), []
    for k, (a, n) in enumerate(runs):
        f = _fields([segs[k]], n, 0, 2, FLAGS)
        g_run = cols if len(runs) == 1 else Coll(Array(H, Array(n, BF16)), concat([cols[h][a:a + n].refs for h in range(H)]))
        o = B.batch(bind(EmbeddingBackwardSortedRun, N=n, FEATURE=F), carry, f, g_run, axes=(0, None, 0))
        carry = _fields([o], H, 0, cw, _sorted_carry(F))
        outs.append(transposes(B, o, H, n, (cw // BF16.leaves,), (cw + n * BF16.leaves) // BF16.leaves)[0])
    out = outs[0] if len(outs) == 1 else Coll(Array(M, Array(H, BF16)), concat([c.refs for c in outs]))
    e = B.call(bind(EmbeddingBackwardRoute, M=M, V=V, H=H), _fields(segs, M, 2, 1, P1),
               _fields(segs, M, 3, KB, Array(KB, BIT)), out)
    n = V * H
    return B.batch(EW.Bf16AddF2fp, w.reshape(Array(n, BF16)), e.reshape(Array(n, BF16)), axes=(0, 0)).reshape(
        Array(V, Array(H, BF16)))


EmbeddingBackwardTiedV2 = CompositeDefinition(
    "EmbeddingBackwardTied", 2, ("M", "V", "H"), EmbeddingBackwardTied.signature, _tied_sorted,
    doc="EmbeddingBackwardTied_v1's function with the ids sorted once: EmbeddingBackwardSort_v1, "
        "EmbeddingBackwardSegmentRun_v1 over the sorted ids, the sorted rows' transpose, EmbeddingBackwardSortedRun_v1 on "
        "each feature's column (the sort path above 3072 tokens, else the feature kernel), its outs transposed back, "
        "EmbeddingBackwardRoute_v1 of each id's last token to the id's slot, then Bf16AddF2fp_v2 of the LM head's dW w and "
        "each slot",
    conformance=f"exact-model-tested; against numerics.torch_blocks' model (tests/test_torch_embedding.py), and {_ON_GPU_V2}",
    refuse=_refuse_all("M", "V", "H"))
