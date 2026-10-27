"""HF `LlamaAttention`'s eager attention in training, as Boolean Definitions: each is the gates of the torch 2.13/2.14 CUDA
kernels that compute it (the two versions' kernels are the same at f32 and bf16), every NaN of an f32 operation the GPU's
0x7FFFFFFF and every rounding to bf16 F2FP's (`__float2bfloat16`, NaN 0x7FFF), none flushed.

    SoftmaxTorch_v1{N}             (x: Array<N, F32>) -> Array<N, F32>                                     1 <= N <= 2048
    SoftmaxTorchBackward_v1{N}     (g: Array<N, F32>, y: Array<N, F32>) -> Array<N, F32>                   1 <= N <= 1024
    EagerSoftmax_v1{N, D}          (s: Array<N, BF16>, n: Array<16, Value<1>> of kind position)
                                   -> Tuple<Array<N, F32>, Array<N, BF16>>
    EagerSoftmaxBackward_v1{N, D}  (dp: Array<N, BF16>, y: Array<N, F32>) -> Array<N, BF16>
    RopeHf_v1{D}                   (x: Array<D, BF16>, cs: Array<D, BF16>) -> Array<D, BF16>
    RopeHfBackward_v1{D}           (g: Array<D, BF16>, cs: Array<D, BF16>) -> Array<D, BF16>
    RepeatKvBackward_v1{NREP, N}   (g: Array<NREP, Array<N, BF16>>) -> Array<N, BF16>                      2 <= NREP <= 16
    EagerAttentionTorch_v1{N, NH, KVH, D}
                                   (q: Array<N, Array<NH·D, BF16>>, k, v: Array<N, Array<KVH·D, BF16>>, cs: Array<N, Array<D, BF16>>,
                                   n: Array<N, U16> of kind position)
                                   -> Tuple<o: Array<N, Array<NH·D, BF16>>, qe: Array<N, Array<NH·D, BF16>>,
                                            ke: Array<N, Array<KVH·D, BF16>>, y: Array<N, Array<NH, Array<N, F32>>>,
                                            p: Array<N, Array<NH, Array<N, BF16>>>>
    EagerAttentionTorchBackward_v1{N, NH, KVH, D}
                                   (do: Array<N, Array<NH·D, BF16>>, qe, ke, v, y, p, cs)
                                   -> Tuple<dq: Array<N, Array<NH·D, BF16>>, dk, dv: Array<N, Array<KVH·D, BF16>>>

`SoftmaxTorch` is `softmax_warp_forward<float, float, float, L>` (PersistentSoftmax.cuh), which `host_softmax` runs on an f32 row
of N <= 2048. With P = 2^L the least power of two >= N, a warp of WS = min(P, 32) lanes holds the row: lane l reads
x_{l + it*WS} for it < P/WS, -inf past N. Each lane's max starts at its first element and folds ``m > e ? m : e``; the
butterfly (``__shfl_xor`` at WS/2 .. 1) folds ``a < b ? b : a`` with a the lane's own. Neither is symmetric in +0/-0 or a NaN,
so every lane keeps its own max, and each element's ``e = expf(x - m)`` uses its lane's. Each lane sums its e's from +0 in
order (the padding's included), and the butterfly adds; IEEE addition is commutative, so every lane ends with the one sum the
tree makes. The output is ``sum == 0 ? 0x7FC00000 : e / sum`` (div.rn). ``expf`` is libdevice's `__nv_expf`
(`boolean_moe.nv_expf_in`, MUFU.EX2, `nv_expf_out`). Its MUFU.EX2 is `MufuEx2Ftz_v2`, which sm_120's MUFU.EX2 equals on all
2^32 inputs.

`SoftmaxTorchBackward` is `_softmax_backward_data`'s ``tmp = grad * output`` (its own FMUL kernel), then
`softmax_warp_backward<float, float, float, L>`, which `host_softmax_backward` runs on an f32 row of N <= 1024. The lanes
are the forward's, but padded with +0: each lane sums its tmp's from +0, the butterfly adds, and
``gI = tmp - output * sum`` is one FFMA, ``fma(-y, sum, tmp)``, as nvcc contracts it.

`EagerSoftmax` is `eager_attention_forward`'s row between its two bmm's at head dimension D:
- ``a = bf16(f32(s) * RN32(D^-0.5))``, the `* scaling` of the score (a bf16 tensor times a Python float);
- ``+ attention_mask``, a bf16 add (f32 add, then F2FP) of +0 for keys j < n and of finfo(bf16).min = 0xFF7F for the rest,
  the causal mask HF's `eager_mask` writes for a query that sees n keys;
- ``softmax(dim=-1, dtype=float32)``, which is ``.toType(f32)`` (exact) and then `SoftmaxTorch_v1{N}`. Autograd saves this f32
  row, the first output;
- ``.to(bf16)``, the second output, which the PV bmm reads. Dropout at p = 0 is the identity.

`EagerSoftmaxBackward` is that row's backward. ``.to(bf16)``'s backward widens dp to f32. Then come
`SoftmaxTorchBackward_v1{N}` on the saved f32 row, ``.toType(f32)``'s backward (F2FP), the mask add's (the identity) and
``* scaling``'s, ``bf16(f32(g) * RN32(D^-0.5))``. Its output is the score's gradient, which the dQ and dK bmm's read.

`RopeHf` is HF's ``q * cos + rotate_half(q) * sin`` in bf16 on one head. The row ``cs = [cos[0:R] | sin[0:R]]`` is the
public bf16 table, R = D/2, since HF's cos and sin repeat their first half. Each product is a bf16 multiply and the sum a
bf16 add: ``out_j = bf16(f32(bf16(x_j c_j)) + f32(bf16(r_j s_j)))``, where ``r_j = -x_{j+R}`` for j < R and ``x_{j-R}``
otherwise (`rotate_half`'s ``cat(-x2, x1)``).

`RopeHfBackward` is its backward. Two bf16 products are formed, ``A_j = bf16(g_j c_j)`` and ``B_j = bf16(g_j s_j)``. The input's
gradient sums three terms: A (MulBackward), x1's SliceBackward (``B_{j+R}`` for j < R, +0 above) and x2's (+0 below R,
``-B_{j-R}`` above). Autograd's InputBuffer adds them as bf16 adds in arrival order. Since one term is an exact +0, every
order gives ``z(bf16(f32(A_j) + f32(B')))``, where z maps -0 to +0: adding +0 changes a word only by z, and a zero sum that
includes +0 is +0.

`RepeatKvBackward` is the backward of `repeat_kv`'s expand: ``sum_to``, a bf16 ``sum(dim=2)`` over the NREP query heads that
read one KV head (Reduce.cuh, ``sum_functor<BFloat16, float>``, vt0 = 4). The reduced dimension isn't the fastest, so each
thread folds one output's NREP inputs alone (no input split). ``thread_reduce_impl`` keeps four f32 accumulators from +0 and
adds input i into accumulator i mod 4, in order. It then combines them as ``((a0 + a1) + a2) + a3`` and rounds once to bf16.

**The step's blocks** are one sequence's attention, wiring these and the bmm's with no gate of their own. Values are committed
token-major: q, k and v are the projections' outputs, row i holding token i's heads, and cs's row i is position i's
[cos | sin]. Query head h reads KV head h / R, R = NH / KVH. `EagerAttentionTorch` applies `RopeHf` to each head of q and k,
then for each query row i takes the scores of its NH heads, one bmm (`BatchedGemm_v2` over the NH heads, k16 steps
`cublas_bf16.TRAIN_DOT`, `HopperBF16WgmmaDot16_v3`) against each head's key rows. `EagerSoftmax` follows on each (i, h),
where row i sees n[i] keys (i + 1 under the causal mask). The PV bmm then takes P's row against Vᵀ, which `Transpose_v1` tiles of at most 128 x 128 make
(`transposes`). o is o_proj's input, and qe, ke, y and p are what autograd saves. `EagerAttentionTorchBackward` runs from
o_proj's dX (do) to dq, dk and dv in projection space:
- dP is dO's row against V's rows, and dS is `EagerSoftmaxBackward` of each (i, h);
- dV, dQ and dK are each row's bmm: Pᵀ's row against dOᵀ, dS's row against Kᵀ, and dSᵀ's row against Qᵀ;
- `RepeatKvBackward` sums the dK and dV of each KV head's R query heads (none at R = 1), and `RopeHfBackward` takes dq and dk
  back through RoPE.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, const_bits, mux, or_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.mufu import mufu_ex2
from verity.ml.circuits.ops.boolean.scalar import gt_u
from verity.ml.circuits.ops.boolean.softmax import f32_gt
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.refs import Coll, concat, strided_view, tuple_of
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple

from . import boolean_moe as BMOE
from . import cublas_bf16 as CB
from .torch_train import Transpose

__all__ = ["EagerAttentionTorch", "EagerAttentionTorchBackward", "EagerSoftmax", "EagerSoftmaxBackward", "RepeatKvBackward",
           "RopeHf", "RopeHfBackward", "SoftmaxTorch", "SoftmaxTorchBackward", "geometry", "scale_bits", "transpose",
           "transposes"]

F32 = Array(32, BL.BIT)
BF16 = Array(16, BL.BIT)
U16 = Array(16, BL.BIT)
_NAN = fp.F32_NAN_PTX
NEG_INF = 0xFF800000
#: ``std::numeric_limits<float>::quiet_NaN()``, the forward's output on a zero sum
QNAN = 0x7FC00000
#: ``torch.finfo(torch.bfloat16).min``, the eager mask's word for a key the query doesn't see
BF16_MIN = 0xFF7F
MAX_FORWARD, MAX_BACKWARD, MAX_NREP = 2048, 1024, 16
VT0 = 4
#: a `Transpose_v1` tile's most rows and columns: a row of 128 bf16 words is one 256 B commitment row
TILE = 128
#: the bmm's k16 step: N and D are multiples of it
K16 = 16


def _add(C, a, b):
    return fp.f32_add(C, a, b, nan_word=_NAN)


def _mul(C, a, b):
    return fp.f32_mul(C, a, b, nan_word=_NAN)


def _bf16(C, x):
    return fp.f32_to_bf16(C, x, W.BF16_NAN_F2FP)


def _bf16_mul(C, a, b):
    return _bf16(C, _mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)))


def _bf16_add(C, a, b):
    return _bf16(C, _add(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)))


def nv_expf(C, x):
    """libdevice's ``expf`` (`NvExpf_v2`'s gates), its MUFU.EX2 `MufuEx2Ftz_v2`."""
    f, L = BMOE.nv_expf_in(C, x)
    return BMOE.nv_expf_out(C, mufu_ex2(C, f), L)


def geometry(N: int) -> tuple[int, int]:
    """(WS, IT): the persistent kernels' lanes and elements a lane for a row of N."""
    p = 1 << (N - 1).bit_length()
    ws = min(p, 32)
    return ws, p // ws


def scale_bits(D: int) -> int:
    """``RN32(D ** -0.5)``, HF's `scaling` as the mul kernel's f32 operand."""
    import numpy as np

    return int(np.array([D ** -0.5], np.float32).view(np.uint32)[0])


def _lanes(xs, pad):
    ws, it = geometry(len(xs))
    return [[xs[l + k * ws] if l + k * ws < len(xs) else const_bits(pad, 32) for k in range(it)] for l in range(ws)]


def _tree_sum(C, s):
    """The butterfly's sum, as lane 0 makes it: each step adds lane l's word and lane l + h's, h = WS/2 .. 1."""
    while len(s) > 1:
        h = len(s) // 2
        s = [_add(C, s[l], s[l + h]) for l in range(h)]
    return s[0]


def softmax_warp_forward(C, xs):
    el = _lanes(xs, NEG_INF)
    m = []
    for e in el:
        v = e[0]
        for x in e[1:]:
            v = mux(C, f32_gt(C, v, x)[0], v, x)
        m.append(v)
    off = len(m) // 2
    while off:
        m = [mux(C, f32_gt(C, m[l ^ off], m[l])[0], m[l ^ off], m[l]) for l in range(len(m))]
        off //= 2
    ex = [[nv_expf(C, _add(C, x, fp.f32_neg(m[l]))) for x in e] for l, e in enumerate(el)]
    sums = []
    for e in ex:
        s = fp.f32_const(0)
        for v in e:
            s = _add(C, s, v)
        sums.append(s)
    total = _tree_sum(C, sums)
    zero = NOT(or_reduce(C, total[:31]))
    ws = len(el)
    return [mux(C, zero, const_bits(QNAN, 32), fp.f32_div(C, ex[j % ws][j // ws], total, nan_word=_NAN))
            for j in range(len(xs))]


def softmax_warp_backward(C, gs, ys):
    tmp = [_mul(C, g, y) for g, y in zip(gs, ys, strict=True)]
    sums = []
    for e in _lanes(tmp, 0):
        s = fp.f32_const(0)
        for v in e:
            s = _add(C, s, v)
        sums.append(s)
    total = _tree_sum(C, sums)
    return [fp.f32_fma(C, fp.f32_neg(y), total, t, nan_word=_NAN) for y, t in zip(ys, tmp)]


def eager_softmax(C, D, ss, n):
    sc = fp.f32_const(scale_bits(D))
    xs = []
    for j, s in enumerate(ss):
        a = _bf16(C, _mul(C, fp.bf16_to_f32(s), sc))
        mask = mux(C, gt_u(C, n, const_bits(j, len(n))), const_bits(0, 16), const_bits(BF16_MIN, 16))
        xs.append(fp.bf16_to_f32(_bf16_add(C, a, mask)))
    y = softmax_warp_forward(C, xs)
    return y + [_bf16(C, v) for v in y]


def eager_softmax_backward(C, D, dps, ys):
    sc = fp.f32_const(scale_bits(D))
    gi = softmax_warp_backward(C, [fp.bf16_to_f32(d) for d in dps], ys)
    return [_bf16(C, _mul(C, fp.bf16_to_f32(_bf16(C, v)), sc)) for v in gi]


def _neg16(w):
    return list(w[:15]) + [NOT(w[15])]


def rope_hf(C, xs, cs):
    R = len(xs) // 2
    cos, sin = cs[:R], cs[R:]
    out = []
    for j, x in enumerate(xs):
        r = _neg16(xs[j + R]) if j < R else xs[j - R]
        out.append(_bf16_add(C, _bf16_mul(C, x, cos[j % R]), _bf16_mul(C, r, sin[j % R])))
    return out


def _z(C, w):
    """-0 as +0."""
    return list(w[:15]) + [C.AND(w[15], or_reduce(C, w[:15]))]


def rope_hf_backward(C, gs, cs):
    R = len(gs) // 2
    cos, sin = cs[:R], cs[R:]
    A = [_bf16_mul(C, g, cos[j % R]) for j, g in enumerate(gs)]
    Bv = [_bf16_mul(C, g, sin[j % R]) for j, g in enumerate(gs)]
    return [_z(C, _bf16_add(C, A[j], Bv[j + R] if j < R else _neg16(Bv[j - R]))) for j in range(len(gs))]


def thread_reduce_sum(C, xs):
    """``thread_reduce_impl``'s fold of one output's inputs (f32 words) with sum_functor: VT0 accumulators from +0, input i into
    accumulator i mod VT0 in order, then ``((a0 + a1) + a2) + ...``."""
    acc = [fp.f32_const(0)] * VT0
    for i, x in enumerate(xs):
        acc[i % VT0] = _add(C, acc[i % VT0], x)
    out = acc[0]
    for a in acc[1:]:
        out = _add(C, out, a)
    return out


def repeat_kv_backward(C, NREP, N, gs):
    rows = [gs[r * N:(r + 1) * N] for r in range(NREP)]
    return [_bf16(C, thread_reduce_sum(C, [fp.bf16_to_f32(rows[r][i]) for r in range(NREP)])) for i in range(N)]


#: each static's range: (least, most, even)
_RANGES = {"N": (1, MAX_FORWARD, False), "D": (2, 1 << 16, True), "NREP": (2, MAX_NREP, False)}


def _refusal(k: str, v: Any, most: int | None = None) -> str | None:
    lo, hi, even = _RANGES[k]
    hi = hi if most is None else most
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi or (even and v % 2):
        return f"{k} = {v!r} is outside {lo}..{hi}{' and even' if even else ''}, the range of the kernel this is"
    return None


def _refuse(**most):
    def refuse(b: dict) -> str | None:
        return next((r for k in most if b.get(k) is not None and (r := _refusal(k, b[k], most[k]))), None)
    return refuse


def _statics(S: Any, **most) -> Any:
    for k in most:
        if r := _refusal(k, getattr(S, k), most[k]):
            raise ValueError(r)
    return S


def _softmax_sig(S):
    S = _statics(S, N=MAX_FORWARD)
    return (("x", Array(S.N, F32)),), Array(S.N, F32)


def _softmax_backward_sig(S):
    S = _statics(S, N=MAX_BACKWARD)
    return (("g", Array(S.N, F32)), ("y", Array(S.N, F32))), Array(S.N, F32)


def _eager_sig(S):
    S = _statics(S, N=MAX_FORWARD, D=None)
    return (("s", Array(S.N, BF16)), ("n", U16)), Tuple(Array(S.N, F32), Array(S.N, BF16))


def _eager_backward_sig(S):
    S = _statics(S, N=MAX_BACKWARD, D=None)
    return (("dp", Array(S.N, BF16)), ("y", Array(S.N, F32))), Array(S.N, BF16)


def _rope_sig(S):
    S = _statics(S, D=None)
    return (("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)


def _rope_backward_sig(S):
    S = _statics(S, D=None)
    return (("g", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)


def _repeat_kv_sig(S):
    S = _statics(S, NREP=None, N=1 << 20)
    return (("g", Array(S.NREP, Array(S.N, BF16))),), Array(S.N, BF16)


_POSITION = Ports(params={"n": In(kind="position")})
#: the words a GPU computed in a 2-layer training step at S = 256 and at the training run's shapes (B 8, S 512);
#: `tests/test_torch_training_gpu.py` keeps a cut of each
GPU_STEP = "a 2-layer training step at S = 256"
GPU_SHAPES = "the training run's shapes"
ON_GPU = "a cc 12.0 GPU (torch 2.14.0+cu130), 0 mismatches; tests/test_torch_training_gpu.py"
_HEADS = f"exact-model-tested; every q and k head (D = 64) of both layers of {GPU_STEP}: {ON_GPU}"
_ROWS = f"every row (N = 256) of both layers' 9 heads of {GPU_STEP} and rows of 512 and 1,024 keys of {GPU_SHAPES}"

SoftmaxTorch = specialized(
    "SoftmaxTorch", 1, ("N",), lambda S: lambda C, *xs: softmax_warp_forward(C, xs), _softmax_sig,
    doc="torch's softmax_warp_forward on an f32 row of N <= 2048: per-lane max then the shfl_xor butterfly, expf(x - m), "
        "per-lane sums from +0 then the butterfly, sum == 0 ? qNaN : e / sum",
    refuse=_refuse(N=MAX_FORWARD), conformance=f"exact-model-tested; as EagerSoftmax_v1's saved row, on {_ROWS}: {ON_GPU}")
SoftmaxTorchBackward = specialized(
    "SoftmaxTorchBackward", 1, ("N",), lambda S: lambda C, *w: softmax_warp_backward(C, w[:S.N], w[S.N:]),
    _softmax_backward_sig,
    doc="torch's _softmax_backward_data on an f32 row of N <= 1024: tmp = g * y, per-lane sums from +0 then the shfl_xor "
        "butterfly, fma(-y, sum, tmp)",
    refuse=_refuse(N=MAX_BACKWARD), conformance=f"exact-model-tested; inside EagerSoftmaxBackward_v1, on {_ROWS}: {ON_GPU}")
EagerSoftmax = specialized(
    "EagerSoftmax", 1, ("N", "D"), lambda S: lambda C, *w: eager_softmax(C, S.D, w[:-1], w[-1]), _eager_sig,
    doc="HF eager attention's row: bf16(s * D^-0.5), the bf16 causal mask's add (+0 below n, 0xFF7F from n), "
        "SoftmaxTorch_v1 in f32 (the saved row) and its bf16 copy",
    refuse=_refuse(N=MAX_FORWARD, D=None), ports=_POSITION, conformance=f"exact-model-tested; both outputs on {_ROWS}: {ON_GPU}")
EagerSoftmaxBackward = specialized(
    "EagerSoftmaxBackward", 1, ("N", "D"), lambda S: lambda C, *w: eager_softmax_backward(C, S.D, w[:S.N], w[S.N:]),
    _eager_backward_sig,
    doc="HF eager attention's row backward: SoftmaxTorchBackward_v1 of f32(dp) and the saved row, F2FP, then "
        "bf16(f32(.) * D^-0.5)",
    refuse=_refuse(N=MAX_BACKWARD, D=None), conformance=f"exact-model-tested; the score's gradient on {_ROWS}: {ON_GPU}")
RopeHf = specialized(
    "RopeHf", 1, ("D",), lambda S: lambda C, *w: rope_hf(C, w[:S.D], w[S.D:]), _rope_sig,
    doc="HF apply_rotary_pos_emb on one bf16 head, cs = [cos[0:R] | sin[0:R]]: bf16(bf16(x c) + bf16(rotate_half(x) s))",
    refuse=_refuse(D=None), conformance=_HEADS)
RopeHfBackward = specialized(
    "RopeHfBackward", 1, ("D",), lambda S: lambda C, *w: rope_hf_backward(C, w[:S.D], w[S.D:]), _rope_backward_sig,
    doc="HF apply_rotary_pos_emb's backward on one bf16 head: z(bf16(bf16(g c) + bf16(g s) rotated back)), -0 as +0",
    refuse=_refuse(D=None), conformance=_HEADS)
RepeatKvBackward = specialized(
    "RepeatKvBackward", 1, ("NREP", "N"), lambda S: lambda C, *w: repeat_kv_backward(C, S.NREP, S.N, w), _repeat_kv_sig,
    doc="repeat_kv's backward: sum_to over the NREP query heads, Reduce.cuh's thread fold (four f32 accumulators from +0, "
        "input i into i mod 4, combined in order), one F2FP",
    refuse=_refuse(NREP=None, N=1 << 20),
    conformance=f"exact-model-tested; the key and value gradients (NREP = 3, N = 64) of both layers of {GPU_STEP}: {ON_GPU}")

# --- the step's blocks ------------------------------------------------------------------------------------------------------


def transposes(B, x: Coll, R: int, C: int, bases, stride: int) -> list[Coll]:
    """The R x C matrices of bf16 words in x at word offsets ``bases``, each row ``stride`` words past the last, as their C x R
    transposes, by `Transpose_v1` tiles of at most TILE x TILE: one batch a tile shape over all the matrices' tiles."""
    L = BF16.leaves
    rows = [(r, min(TILE, R - r)) for r in range(0, R, TILE)]
    cols = [(c, min(TILE, C - c)) for c in range(0, C, TILE)]
    out = {}
    for rn, cn in sorted({(rn, cn) for _, rn in rows for _, cn in cols}):
        at = [(m, r, c) for m in range(len(bases)) for r, n in rows for c, k in cols if (n, k) == (rn, cn)]
        tiles = Coll(Array(len(at), Array(rn, Array(cn, BF16))), concat(
            [strided_view(x, L * (bases[m] + r * stride + c), L * cn, L * stride, 1, L * rn * cn) for m, r, c in at]))
        y = B.batch(bind(Transpose, R=rn, C=cn), tiles, axes=(0,))
        out.update({k: y[i] for i, k in enumerate(at)})
    return [Coll(Array(C, Array(R, BF16)), concat([out[(m, r, j - j % TILE)][j % TILE].refs for j in range(C) for r, _ in rows]))
            for m in range(len(bases))]


def transpose(B, x: Coll, R: int, C: int) -> Coll:
    """A token-major R x C matrix of bf16 words as its C x R transpose (`transposes`)."""
    return transposes(B, x, R, C, (0,), C)[0]


def _bmm(op: str, heads: int, N: int, D: int) -> Any:
    """torch.bmm ``op``'s row over the heads at N tokens (`cublas_bf16.TRAIN_BMM`): `BatchedGemm_v2{heads, K, N}` on the k16
    step `cublas_bf16.TRAIN_DOT`, the very binding `cublas_bf16.attention_bmm` makes at the (N, D) its capture covers."""
    k, n = CB.TRAIN_BMM[op][2](N, D)
    return bind(CB.BatchedGemm, B=heads, K=k, N=n, DOT=CB.TRAIN_DOT)


def _heads(x: Coll, N: int, NH: int, KVH: int, D: int) -> Coll:
    """Each query head's N rows of D words in x (N rows of KVH heads): head h reads KV head h // (NH / KVH)."""
    L, R = BF16.leaves, NH // KVH
    return Coll(Array(NH, Array(N, Array(D, BF16))),
                concat([strided_view(x, L * (h // R) * D, L * D, L * KVH * D, 1, L * N * D) for h in range(NH)]))


def _stack(ms: list[Coll], picks) -> Coll:
    """The matrices ms[g] for g in picks, as one Array."""
    picks = list(picks)
    return Coll(Array(len(picks), ms[0].type), concat([ms[g].refs for g in picks]))


def _rows_of(ms: list[Coll]) -> Coll:
    """For each row index j, row j of each matrix of ms."""
    t = ms[0].type
    return Coll(Array(t.n, Array(len(ms), t.elem)), concat([m[j].refs for j in range(t.n) for m in ms]))


def _each_head(B, d, x: Coll, cs: Coll, N: int, H: int, D: int) -> Coll:
    """d at D on each of the H heads of each of x's N rows, with that row's cs."""
    rows = Coll(Array(N * H, Array(D, BF16)), concat([cs[i].refs for i in range(N) for _ in range(H)]))
    return B.batch(bind(d, D=D), x.reshape(Array(N * H, Array(D, BF16))), rows, axes=(0, 0)).reshape(
        Array(N, Array(H * D, BF16)))


def _sum_heads(B, g: Coll, N: int, KVH: int, R: int, D: int) -> Coll:
    """repeat_kv's backward on each row's KVH·R heads of g: `RepeatKvBackward` over each KV head's R, g itself at R = 1."""
    if R == 1:
        return g.reshape(Array(N, Array(KVH * D, BF16)))
    return B.batch(bind(RepeatKvBackward, NREP=R, N=D), g.reshape(Array(N * KVH, Array(R, Array(D, BF16)))),
                   axes=(0,)).reshape(Array(N, Array(KVH * D, BF16)))


def _attention(B, S, q, k, v, cs, n):
    N, NH, KVH, D = S.N, S.NH, S.KVH, S.D
    R = NH // KVH
    qe = _each_head(B, RopeHf, q, cs, N, NH, D)
    ke = _each_head(B, RopeHf, k, cs, N, KVH, D)
    s = B.batch(_bmm("qk", NH, N, D), qe.reshape(Array(N, Array(NH, Array(D, BF16)))), _heads(ke, N, NH, KVH, D), axes=(0, None))
    ns = Coll(Array(N * NH, U16), concat([n[i].refs for i in range(N) for _ in range(NH)]))
    sm = B.batch(bind(EagerSoftmax, N=N, D=D), s.reshape(Array(N * NH, Array(N, BF16))), ns, axes=(0, 0))
    F, H = F32.leaves * N, BF16.leaves * N
    y = Coll(Array(N, Array(NH, Array(N, F32))), strided_view(sm, 0, F, F + H, 1, N * NH * F))
    p = Coll(Array(N, Array(NH, Array(N, BF16))), strided_view(sm, F, H, F + H, 1, N * NH * H))
    vt = transposes(B, v, N, D, [g * D for g in range(KVH)], KVH * D)
    o = B.batch(_bmm("pv", NH, N, D), p, _stack(vt, [h // R for h in range(NH)]), axes=(0, None))
    return tuple_of(o.reshape(Array(N, Array(NH * D, BF16))), qe, ke, y, p)


def _attention_backward(B, S, do, qe, ke, v, y, p, cs):
    N, NH, KVH, D = S.N, S.NH, S.KVH, S.D
    R = NH // KVH
    dp = B.batch(_bmm("dp", NH, N, D), do.reshape(Array(N, Array(NH, Array(D, BF16)))), _heads(v, N, NH, KVH, D), axes=(0, None))
    ds = B.batch(bind(EagerSoftmaxBackward, N=N, D=D), dp.reshape(Array(N * NH, Array(N, BF16))),
                 y.reshape(Array(N * NH, Array(N, F32))), axes=(0, 0))
    by_head = [h * N for h in range(NH)]
    qh = [h * D for h in range(NH)]
    kvh = [g * D for g in range(KVH)]
    dv = B.batch(_bmm("dv", NH, N, D), _rows_of(transposes(B, p, N, N, by_head, NH * N)),
                 _stack(transposes(B, do, N, D, qh, NH * D), range(NH)), axes=(0, None))
    dq = B.batch(_bmm("dq", NH, N, D), ds.reshape(Array(N, Array(NH, Array(N, BF16)))),
                 _stack(transposes(B, ke, N, D, kvh, KVH * D), [h // R for h in range(NH)]), axes=(0, None))
    dk = B.batch(_bmm("dk", NH, N, D), _rows_of(transposes(B, ds, N, N, by_head, NH * N)),
                 _stack(transposes(B, qe, N, D, qh, NH * D), range(NH)), axes=(0, None))
    return tuple_of(_each_head(B, RopeHfBackward, dq, cs, N, NH, D),
                    _each_head(B, RopeHfBackward, _sum_heads(B, dk, N, KVH, R, D), cs, N, KVH, D),
                    _sum_heads(B, dv, N, KVH, R, D))


_BLOCK_KEYS = ("N", "NH", "KVH", "D")


def _block_refusal(b: dict, most: int) -> str | None:
    for k, lo, hi in (("N", K16, most), ("NH", 1, 1 << 10), ("KVH", 1, 1 << 10), ("D", K16, 1 << 16)):
        v, k16 = b.get(k), k in ("N", "D")
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi or (k16 and v % K16)):
            return (f"{k} = {v!r} is outside {lo}..{hi}{' or not a multiple of 16 (the bmm k16 step)' if k16 else ''}, the "
                    "range of the kernels this is")
    NH, KVH = b.get("NH"), b.get("KVH")
    if isinstance(NH, int) and isinstance(KVH, int) and (NH % KVH or NH // KVH > MAX_NREP):
        return f"NH = {NH} is not 1 to {MAX_NREP} times KVH = {KVH}, repeat_kv's query heads a KV head"
    return None


def _block_refuse(most: int):
    return lambda b: _block_refusal(b, most)


def _block(S: Any, most: int) -> tuple[int, int, int, int]:
    if r := _block_refusal({k: getattr(S, k) for k in _BLOCK_KEYS}, most):
        raise ValueError(r)
    return S.N, S.NH, S.KVH, S.D


def _block_types(S: Any, most: int):
    N, NH, KVH, D = _block(S, most)
    return (Array(N, Array(NH * D, BF16)), Array(N, Array(KVH * D, BF16)), Array(N, Array(D, BF16)),
            Array(N, Array(NH, Array(N, F32))), Array(N, Array(NH, Array(N, BF16))))


def _attention_sig(S):
    q, kv, cs, y, p = _block_types(S, MAX_FORWARD)
    return (("q", q), ("k", kv), ("v", kv), ("cs", cs), ("n", Array(S.N, U16))), Tuple(q, q, kv, y, p)


def _attention_backward_sig(S):
    q, kv, cs, y, p = _block_types(S, MAX_BACKWARD)
    return (("do", q), ("qe", q), ("ke", kv), ("v", kv), ("y", y), ("p", p), ("cs", cs)), Tuple(q, kv, kv)


_BLOCK = (f"exact-model-tested; against numerics.torch_blocks' model, composed of its Definitions' models, at N = 16 "
          f"(tests/test_torch_attention.py), "
          f"which is the GPU's on every output and intermediate of both layers' attention, forward and backward, of {GPU_STEP} "
          f"(10,452,992 words): a cc 12.0 GPU (torch 2.14.0+cu130), 0 mismatches; its bmm's are "
          f"cublas_bf16.attention_bmm's wherever that binds (S = 256 and 512 at D = 64, TRAIN_CAPTURE)")

EagerAttentionTorch = CompositeDefinition(
    "EagerAttentionTorch", 1, _BLOCK_KEYS, _attention_sig, _attention,
    doc="HF's eager attention forward on one sequence of N tokens, NH query heads on KVH KV heads of D: RopeHf_v1 on each "
        "head of q and k, each query row's bmm against its heads' keys (BatchedGemm_v2), EagerSoftmax_v1 on each (row, head) "
        "with row i seeing n[i] keys, then P's row against Vᵀ (Transpose_v1 tiles); o_proj's input and what autograd saves",
    conformance=_BLOCK, refuse=_block_refuse(MAX_FORWARD), ports=_POSITION)
EagerAttentionTorchBackward = CompositeDefinition(
    "EagerAttentionTorchBackward", 1, _BLOCK_KEYS, _attention_backward_sig, _attention_backward,
    doc="EagerAttentionTorch_v1's backward from o_proj's dX: dP (dO's row against V), EagerSoftmaxBackward_v1, the dV, dQ "
        "and dK bmm's on rows of Pᵀ, dS and dSᵀ (Transpose_v1 tiles), RepeatKvBackward_v1 over each KV head's query heads, "
        "RopeHfBackward_v1 on dq and dk",
    conformance=_BLOCK, refuse=_block_refuse(MAX_BACKWARD))
