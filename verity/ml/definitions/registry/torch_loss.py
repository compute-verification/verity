"""HF `ForCausalLMLoss`'s cross-entropy in training, as Boolean Definitions: each is the gates of the torch 2.13/2.14 CUDA kernels
of ``F.cross_entropy(logits.float(), labels, ignore_index=-100)`` and its backward (the two versions' kernels are the same), every
NaN of an f32 operation the GPU's 0x7FFFFFFF and every rounding to bf16 F2FP's, none flushed. The logits are the LM head's bf16
words; ``.float()`` widens them exactly, so each Definition reads them as bf16.

    WarpReduceMax_v1                   (v: Array<32, F32>) -> F32
    WarpReduceSum_v1                   (v: Array<32, F32>) -> F32
    LogSoftmaxLaneMax_v1{KL}           (x: Array<KL, Array<4, BF16>>) -> F32
    LogSoftmaxWarpMax_v1{V}            (x: Array<C, BF16>) -> F32                                       C = V / 32
    LogSoftmaxLaneSumExp_v1{KL}        (x: Array<KL, Array<4, BF16>>, m: F32) -> F32
    LogSoftmaxWarpSumExp_v1{V}         (x: Array<C, BF16>, m: Array<32, F32>) -> F32
    LogSoftmaxAt_v1                    (x: BF16, mx: F32, ls: F32) -> F32
    LogSoftmaxRowEnd_v1                (m: Array<32, F32>, s: Array<32, F32>, xt: BF16) -> Tuple<F32, F32, F32>
    NllLossTorch_v1{M, V, MEAN}        (lp: Array<M, F32>, t: Array<M, U32>) -> Tuple<F32, F32>
    NllLossTorchBackward_v1{MEAN}      (go: F32, tw: F32) -> F32          MEAN = 0: (go: F32) -> F32
    LogSoftmaxNllSlots_v1{V}           (t: U32, g: F32) -> Tuple<Array<32, U32>, F32>
    LogSoftmaxNllBackwardElem_v1       (x: BF16, hit: Value<1>, mx: F32, ls: F32, s: F32, g: F32) -> BF16
    LogSoftmaxNllBackwardWarp_v1{V}    (x: Array<C, BF16>, p: U32, mx: F32, ls: F32, s: F32, g: F32) -> Array<C, BF16>

**The forward.** ``log_softmax`` of an f32 row of V = 4096·KL logits, KL >= 3 (smollm2's V = 49,152 is KL = 12), is
`host_softmax`'s `cunn_SoftMaxForward<ILP = 4>` with a block of 1024 threads: V is past the persistent kernels' 1024, the
register kernel needs fewer than 10 elements a thread and the shared-memory one fewer than 12,256 (48 KiB). The row is 16 B
aligned, so ``shift`` is 0, and V % 4096 == 0, so no thread takes a tail. Thread t reads the four logits from 4(t + 1024k) in
each of its KL loads, so warp w (threads 32w .. 32w + 31) reads KL runs of 128 consecutive logits, from 4096k + 128w. Warp w's
*chunk* is those C = 128·KL logits in column order; at chunk position p = 128k + 4l + j sits lane l's element j of load k.
- The max: each thread folds ``fmaxf`` (`F32Fmaxf_v1`: +0 over -0, a NaN operand yields the other) from -FLT_MAX over its
  elements in load order (`LogSoftmaxLaneMax`). ``blockReduceWarp`` is ``cuda_utils::BlockReduce``: ``WarpReduce``
  (``__shfl_down`` at 16 .. 1, ``combine(own, other)``) gives lane 0 its warp's value, lane 0 of each warp writes it to shared
  memory, and warp 0 runs ``WarpReduce`` once more over the 32. Lane 0's tree is ``s_l = op(s_l, s_{l+h})`` for h = 16 .. 1
  (`WarpReduceMax`, ``a < b ? b : a``; `WarpReduceSum`, ``a + b``). A thread's max is never a NaN, so neither is the row's.
- The sum: each thread folds ``sum + expf(x - max)`` from +0 in the same order (`LogSoftmaxLaneSumExp`), and the same two
  trees add. ``expf`` is libdevice's `__nv_expf` (`NvExpf_v2`'s gates), whose MUFU.EX2 is `MufuEx2Ftz_v2`: KATTN-REF's sweep
  (run r20261027-024703-41d4) found sm_120's ``ex2.approx.ftz.f32`` equal to it on all 2^32 inputs.
- `LogSoftmaxWarpMax{V}` and `LogSoftmaxWarpSumExp{V}` are one warp's part, a proof block each; the second takes the 32 warps'
  maxes and reduces them itself, so the row's max is never a value of its own. `LogSoftmaxRowEnd` reduces both, takes
  ``logsum = logf(sum)`` (libdevice's `__nv_logf`, `NvLogf_v3`; its argument is at least 1 or not finite, so none of its
  operations meets a subnormal and FTZ changes nothing) and the target's ``(x_t - max) - logsum``, two FADDs, the epilogue's
  ``input - max_input - logsum`` (`LogSoftmaxAt`). The target's logit x_t is `GatherBf16x{V}_v2` of the row.

**The loss.** ``nll_loss`` over the M rows is ``nll_loss_forward_reduce_cuda_kernel_2d`` on one block of
T = clamp(2^round(log2(M // 16)), 32, 1024) threads (`nll_loss_threads`). Thread t takes rows t, t + T, ..., and for each row
with a valid label subtracts its log-probability, ``sh -= lp * 1`` (an FMUL by 1, or an FFMA, either RN(sh - lp)), and adds
1 to ``acc``. A stride tree (T/2 .. 1, ``s_t += s_{t+stride}``) sums both, and the outputs are ``sh / acc`` (MEAN, div.rn) or
``sh``, and ``total_weight = acc``. Every ``acc`` is an integer below 2^24, where f32 addition is exact, so ``acc`` is computed as
the count of valid labels, converted once. A label is the int64 label's low 32 bits, valid when below V unsigned. torch's test
is ``t != -100`` and it asserts on any other label outside [0, V), so the two agree on every run that completes. HF's sum
reduction (``num_items_in_batch``) divides the sum afterwards (`F32Div`, or `F32Mul` by RN32(1/N) when N is a CPU scalar).

**The backward.** ``nll_loss_backward_reduce_cuda_kernel_2d`` writes ``g = -(MEAN ? go / tw : go)`` (`NllLossTorchBackward`)
into each valid row's target entry of a zeroed f32 gradient. ``log_softmax``'s backward (`cunn_SoftMaxBackward`, block 512) sums
that row from +0 to ``s`` and writes ``gI_j = gO_j - expf(out_j) * s``, one FFMA, ``fma(-e, s, gO)``, as nvcc contracts it.
Only one gO_j is nonzero, so every order of the sum gives s = +0 + g for a valid row (-0 becomes +0, a NaN 0x7FFFFFFF) and +0
for an ignored one, and only the target entry depends on the contraction. ``out_j`` is recomputed from the chunk, its max and
its logsum, the forward epilogue's two FADDs, so the f32 log-probabilities are never committed. ``.float()``'s backward rounds
gI to bf16 (F2FP). `LogSoftmaxNllSlots` makes, from the label and g, the target's position in each warp's chunk (0xFFFFFFFF
in the 31 others, and in all 32 for an ignored row) and s. `LogSoftmaxNllBackwardWarp{V}` is a warp's chunk, one proof block:
`DsaTopKOneHot_v1{N = C}` of its position, then `LogSoftmaxNllBackwardElem` on each logit.
"""
from __future__ import annotations

import math
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, mux
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.scalar import gt_u
from verity.ml.circuits.ops.boolean.softmax import f32_gt
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple

from . import boolean_moe as BMOE
from .boolean_dsa_topk import DsaTopKOneHot, popcount
from .gumbel_boolean import NvLogf
from .torch_attention import nv_expf

__all__ = ["LogSoftmaxAt", "LogSoftmaxLaneMax", "LogSoftmaxLaneSumExp", "LogSoftmaxNllBackwardElem", "LogSoftmaxNllBackwardWarp",
           "LogSoftmaxNllSlots", "LogSoftmaxRowEnd", "LogSoftmaxWarpMax", "LogSoftmaxWarpSumExp", "NllLossTorch",
           "NllLossTorchBackward", "WarpReduceMax", "WarpReduceSum", "chunk_columns", "nll_threads"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
U32 = Array(32, BIT)
_NAN = fp.F32_NAN_PTX
#: ``std::numeric_limits<float>::lowest()``, the max's start
LOWEST = 0xFF7FFFFF
#: a chunk position no logit has: the target isn't in the chunk
NO_SLOT = 0xFFFFFFFF
LANES, ILP, THREADS = 32, 4, 1024
#: the logits one load of the block covers
SPAN = ILP * THREADS
#: KL loads of 4096 logits: the least the block kernel runs at (below, the shared-memory kernel), and a chunk position's bound
KL_MIN, KL_MAX = 3, 1 << 10
M_MIN, M_MAX = 16, 1 << 24
_PENDING = "not-yet-established"


def _add(C, a, b):
    return fp.f32_add(C, a, b, nan_word=_NAN)


def _bf16(C, x):
    return fp.f32_to_bf16(C, x, W.BF16_NAN_F2FP)


def _max_combine(C, a, b):
    """`Max`'s ``a < b ? b : a``, a the thread's own: NaN compares false, and +0 and -0 equal, so both keep a."""
    return mux(C, f32_gt(C, b, a)[0], b, a)


def _warp_tree(C, op, v):
    """``WarpReduce``'s value at lane 0: ``s_l = op(s_l, s_{l+h})`` for h = 16 .. 1."""
    v = list(v)
    while len(v) > 1:
        h = len(v) // 2
        v = [op(C, v[l], v[l + h]) for l in range(h)]
    return v[0]


def _lane_max(C, *xs):
    m = const_bits(LOWEST, 32)
    for x in xs:
        m = BMOE._fmaxf(C, m, fp.bf16_to_f32(x))
    return m


def _lane_sum_exp(C, *w):
    xs, m = w[:-1], fp.f32_neg(w[-1])
    s = fp.f32_const(0)
    for x in xs:
        s = _add(C, s, nv_expf(C, _add(C, fp.bf16_to_f32(x), m)))
    return s


def _at(C, x, mx, ls):
    return _add(C, _add(C, fp.bf16_to_f32(x), fp.f32_neg(mx)), fp.f32_neg(ls))


def nll_threads(M: int) -> int:
    """``nll_loss_threads(M)``: clamp(2^round(log2(M // 16)), 32, 1024), ``std::round`` taking halves away from zero."""
    return min(max(1 << math.floor(math.log2(M // 16) + 0.5), 32), 1024)


def _nll(C, S, *w):
    lp, t = w[:S.M], w[S.M:]
    T = nll_threads(S.M)
    lim = const_bits(S.V, 32)
    valid = [gt_u(C, lim, x) for x in t]
    sh = [fp.f32_const(0)] * T
    for i in range(S.M):
        sh[i % T] = mux(C, valid[i], _add(C, sh[i % T], fp.f32_neg(lp[i])), sh[i % T])
    stride = T // 2
    while stride:
        sh = [_add(C, sh[i], sh[i + stride]) for i in range(stride)] + sh[2 * stride:]
        stride //= 2
    n = popcount(C, valid)
    tw = fp.s32_to_f32(C, list(n) + [ZERO] * (32 - len(n)))
    out = fp.f32_div(C, sh[0], tw, nan_word=_NAN) if S.MEAN else sh[0]
    return [out, tw]


def _nll_backward(C, S, go, *tw):
    return fp.f32_neg(fp.f32_div(C, go, tw[0], nan_word=_NAN) if S.MEAN else go)


def _slots(C, S, t, g):
    """The target's position 128k + (t mod 128) in warp (t >> 7) mod 32's chunk, k = t >> 12, and s = +0 + g; NO_SLOT and +0 for
    an ignored row."""
    valid = gt_u(C, const_bits(S.V, 32), t)
    pos = list(t[:7]) + list(t[12:]) + [ZERO] * 5
    w = t[7:12]
    out = []
    for k in range(LANES):
        here = C.AND(valid, and_reduce(C, [b if (k >> i) & 1 else NOT(b) for i, b in enumerate(w)]))
        out.append(mux(C, here, pos, const_bits(NO_SLOT, 32)))
    s = [C.AND(valid, b) for b in _add(C, fp.f32_const(0), g)]
    return out + [s]


def _backward_elem(C, x, hit, mx, ls, s, g):
    e = nv_expf(C, _at(C, x, mx, ls))
    go = [C.AND(hit[0], b) for b in g]
    return _bf16(C, fp.f32_fma(C, fp.f32_neg(e), s, go, nan_word=_NAN))


def chunk_columns(V: int, w: int) -> list[int]:
    """The columns of warp w's chunk, in chunk order: 4096k + 128w + c for k < V / 4096 and c < 128."""
    return [SPAN * k + 128 * w + c for k in range(V // SPAN) for c in range(128)]


def _refusal(k: str, v: Any, lo: int, hi: int, step: int = 1) -> str | None:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi or v % step:
        return (f"{k} = {v!r} is outside {lo}..{hi}{f' in steps of {step}' if step > 1 else ''}, "
                "the range of the kernel this is")
    return None


_RANGES = {"KL": (KL_MIN, KL_MAX, 1), "V": (KL_MIN * SPAN, KL_MAX * SPAN, SPAN), "M": (M_MIN, M_MAX, 1), "MEAN": (0, 1, 1)}


def _refuse(*keys):
    def refuse(b: dict) -> str | None:
        return next((r for k in keys if b.get(k) is not None and (r := _refusal(k, b[k], *_RANGES[k]))), None)
    return refuse


def _statics(S: Any, *keys) -> Any:
    for k in keys:
        if r := _refusal(k, getattr(S, k), *_RANGES[k]):
            raise ValueError(r)
    return S


def _chunk(S) -> Array:
    return Array(_statics(S, "V").V // LANES, BF16)


def _lanes(S) -> Array:
    return Array(_statics(S, "KL").KL, Array(ILP, BF16))


def _lane_view(x: Coll, V: int) -> Coll:
    return x.reshape(Array(V // SPAN, Array(LANES, Array(ILP, BF16))))


WarpReduceMax = definition(
    "WarpReduceMax", 1, lambda C, *v: _warp_tree(C, _max_combine, v), (("v", Array(LANES, F32)),), F32,
    doc="cuda_utils::WarpReduce with Max at lane 0: s_l = (s_l < s_{l+h} ? s_{l+h} : s_l) for h = 16 .. 1",
    conformance=_PENDING)
WarpReduceSum = definition(
    "WarpReduceSum", 1, lambda C, *v: _warp_tree(C, _add, v), (("v", Array(LANES, F32)),), F32,
    doc="cuda_utils::WarpReduce with Add at lane 0: s_l = s_l + s_{l+h} for h = 16 .. 1", conformance=_PENDING)
LogSoftmaxLaneMax = specialized(
    "LogSoftmaxLaneMax", 1, ("KL",), lambda S: _lane_max, lambda S: ((("x", _lanes(S)),), F32),
    doc="one thread's max in cunn_SoftMaxForward<4>: fmaxf from -FLT_MAX over its KL loads of 4 bf16 logits, in order",
    refuse=_refuse("KL"), conformance=_PENDING)
LogSoftmaxLaneSumExp = specialized(
    "LogSoftmaxLaneSumExp", 1, ("KL",), lambda S: _lane_sum_exp, lambda S: ((("x", _lanes(S)), ("m", F32)), F32),
    doc="one thread's sum in cunn_SoftMaxForward<4>: sum + expf(x - m) from +0 over its KL loads of 4 bf16 logits, in order",
    refuse=_refuse("KL"), conformance=_PENDING)


def _warp_max(B, S, x):
    lanes = B.batch(bind(LogSoftmaxLaneMax, KL=S.V // SPAN), _lane_view(x, S.V), axes=(1,))
    return B.call(WarpReduceMax, lanes)


def _warp_sum_exp(B, S, x, m):
    mx = B.call(WarpReduceMax, m)
    lanes = B.batch(bind(LogSoftmaxLaneSumExp, KL=S.V // SPAN), _lane_view(x, S.V), mx, axes=(1, None))
    return B.call(WarpReduceSum, lanes)


LogSoftmaxWarpMax = CompositeDefinition(
    "LogSoftmaxWarpMax", 1, ("V",), lambda S: ((("x", _chunk(S)),), F32), _warp_max,
    doc="warp w's max in cunn_SoftMaxForward<4> on a row of V logits: LogSoftmaxLaneMax on each lane's elements of its "
        "chunk (columns 4096k + 128w + c), then WarpReduceMax",
    conformance=_PENDING, refuse=_refuse("V"))
LogSoftmaxWarpSumExp = CompositeDefinition(
    "LogSoftmaxWarpSumExp", 1, ("V",), lambda S: ((("x", _chunk(S)), ("m", Array(LANES, F32))), F32), _warp_sum_exp,
    doc="warp w's sum in cunn_SoftMaxForward<4>: the row's max (WarpReduceMax of the 32 warps'), LogSoftmaxLaneSumExp on "
        "each lane's elements of its chunk, then WarpReduceSum",
    conformance=_PENDING, refuse=_refuse("V"))
LogSoftmaxAt = definition(
    "LogSoftmaxAt", 1, _at, (("x", BF16), ("mx", F32), ("ls", F32)), F32,
    doc="LogSoftMaxForwardEpilogue on one bf16 logit: (f32(x) - max) - logsum, two FADDs", conformance=_PENDING)


def _row_end(B, S, m, s, xt):
    mx = B.call(WarpReduceMax, m)
    ls = B.call(NvLogf, B.call(WarpReduceSum, s))
    lp = B.call(LogSoftmaxAt, xt, mx, ls)
    return Coll(Tuple(F32, F32, F32), concat([mx.refs, ls.refs, lp.refs]))


LogSoftmaxRowEnd = CompositeDefinition(
    "LogSoftmaxRowEnd", 1, (), lambda S: ((("m", Array(LANES, F32)), ("s", Array(LANES, F32)), ("xt", BF16)),
                                          Tuple(F32, F32, F32)), _row_end,
    doc="the end of cunn_SoftMaxForward<4> on a row: (max, logsum, the target's log-probability) from the 32 warps' maxes and "
        "sums: WarpReduceMax, WarpReduceSum, NvLogf_v3, LogSoftmaxAt", conformance=_PENDING)


def _nll_sig(S):
    S = _statics(S, "M", "V", "MEAN")
    return (("lp", Array(S.M, F32)), ("t", Array(S.M, U32))), Tuple(F32, F32)


def _nll_backward_sig(S):
    S = _statics(S, "MEAN")
    return (("go", F32),) + ((("tw", F32),) if S.MEAN else ()), F32


NllLossTorch = specialized(
    "NllLossTorch", 1, ("M", "V", "MEAN"), lambda S: lambda C, *w: _nll(C, S, *w), _nll_sig,
    doc="nll_loss_forward_reduce_cuda_kernel_2d on M rows: T = nll_loss_threads(M) threads, sh -= lp over each thread's rows "
        "with a label below V, the stride tree, (MEAN ? sh / tw : sh, tw), tw the count of valid labels",
    refuse=_refuse("M", "V", "MEAN"), conformance=_PENDING)
NllLossTorchBackward = specialized(
    "NllLossTorchBackward", 1, ("MEAN",), lambda S: lambda C, *w: _nll_backward(C, S, *w), _nll_backward_sig,
    doc="nll_loss_backward_reduce_cuda_kernel_2d's gradient: -(MEAN ? go / tw : go)",
    refuse=_refuse("MEAN"), conformance=_PENDING)
LogSoftmaxNllSlots = specialized(
    "LogSoftmaxNllSlots", 1, ("V",), lambda S: lambda C, t, g: _slots(C, S, t, g),
    lambda S: ((("t", U32), ("g", F32)), Tuple(Array(LANES, U32), F32)),
    doc="a row's label t and gradient g as each warp's chunk reads them: t's position in its warp's chunk, 0xFFFFFFFF in the "
        "others (all 32 when t >= V), and the row's gradient sum, +0 + g (+0 when t >= V)",
    refuse=_refuse("V"), conformance=_PENDING)
LogSoftmaxNllBackwardElem = definition(
    "LogSoftmaxNllBackwardElem", 1, _backward_elem,
    (("x", BF16), ("hit", BIT), ("mx", F32), ("ls", F32), ("s", F32), ("g", F32)), BF16,
    doc="LogSoftMaxBackwardEpilogue on one bf16 logit, then .float()'s backward: bf16(fma(-expf((f32(x) - mx) - ls), s, "
        "hit ? g : +0))", conformance=_PENDING)


def _backward_warp(B, S, x, p, mx, ls, s, g):
    hits = B.call(bind(DsaTopKOneHot, N=S.V // LANES), p)
    return B.batch(LogSoftmaxNllBackwardElem, x, hits, mx, ls, s, g, axes=(0, 0, None, None, None, None))


LogSoftmaxNllBackwardWarp = CompositeDefinition(
    "LogSoftmaxNllBackwardWarp", 1, ("V",),
    lambda S: ((("x", _chunk(S)), ("p", U32), ("mx", F32), ("ls", F32), ("s", F32), ("g", F32)), _chunk(S)), _backward_warp,
    doc="cross-entropy's backward on one warp's chunk of a row of V bf16 logits: DsaTopKOneHot_v1{N = V / 32} of the target's "
        "position p, then LogSoftmaxNllBackwardElem on each logit",
    conformance=_PENDING, refuse=_refuse("V"))
