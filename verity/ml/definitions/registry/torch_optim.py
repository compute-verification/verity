"""torch 2.14's foreach optimizers on f32 state as Boolean Definitions: one proof block a run of N elements of one tensor.

    AdamWForeach_v1{N, LERP_FMA, BF16}  (p, g, m, v: Array<N, F32>, k: Array<7, F32>)
                                        -> (Array<N, F32>, Array<N, F32>, Array<N, F32>[, Array<N, BF16>])
    SgdForeach_v1{N, ALPHA_FMA}         (p, g: Array<N, F32>, nlr: F32) -> Array<N, F32>

(F32 is ``Array<32, Value<1>>``, BF16 ``Array<16, Value<1>>``.)

`AdamWForeach` is one element's step of ``torch.optim.AdamW(foreach=True)`` (`_multi_tensor_adam`, not capturable, not fused,
not amsgrad, not maximize), each line one foreach kernel's f32 arithmetic (opmath float, no fast math, nothing flushed):

1. ``p1 = RN(p * a)``                                    `_foreach_mul_(params, 1 - lr * weight_decay)`
2. ``m1 = lerp(m, g, w)``                               `_foreach_lerp_(exp_avgs, grads, 1 - beta1)`, Lerp.h:
   ``|w| < 0.5 ? m + w * d : g - d * RN(1 - w)``, ``d = RN(g - m)``
3. ``v1 = RN(v * b2)``                                   `_foreach_mul_(exp_avg_sqs, beta2)`
4. ``v2 = c2 == 1 ? fma(g, g, v1) : fma(c2, RN(g * g), v1)``   `_foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)`
5. ``dd = RN(RN(sqrt.rn(v2) / s2) + e)``                 `_foreach_sqrt`, `_foreach_div_(…, bias_correction2_sqrt)`,
                                                          `_foreach_add_(…, eps)`
6. ``p2 = fma(st, RN(m1 / dd), p1)``                     `_foreach_addcdiv_(params, exp_avgs, denom, step_size)`

and returns (p2, m1, v2), with BF16 also ``bf16_rn(p2)`` (``param.copy_(master)``, cvt.rn.bf16.f32, NaN 0x7FFF). Every f32 NaN
result is the GPU's 0x7FFFFFFF. Steps 4 and 6 are `DeviceAddCmulCdiv.cuh`'s ``pointwise_op_impl``, whose two ``std::fma`` are
explicit: at ``alpha == 1`` addcmul is ``fma(t1, t2, input)`` (a runtime branch on c2, a private value here) and addcdiv
``input + t1 / t2``, the same word as ``fma(1, t1 / t2, input)``. Step 1 always multiplies: torch skips it at weight decay 0,
where `AdamWScalars` sets a = 1.0 and ``RN(p * 1) = p`` for every word but a NaN, which step 6 turns to 0x7FFFFFFF either way.

LERP_FMA is the one contraction the source leaves to nvcc: 1 reads step 2's branches as ``fma(w, d, m)`` and
``fma(-d, RN(1 - w), g)``, 0 as ``RN(m + RN(w * d))`` and ``RN(g - RN(d * RN(1 - w)))``. The sm_120 SASS of
``multi_tensor_apply_kernel<…LerpFunctor…>`` in the wheel's libtorch_cuda.so settles it; until then a Program names the one
its check read.

k is the step's seven f32 scalars, the kernels' ``Scalar::to<float>()`` of the host's Python floats, which
`AdamWScalars` computes from the private hyperparameters: ``(a, w, b2, c2, s2, e, st) = (f32(1 - lr * lambda), f32(1 - beta1),
f32(beta2), f32(1 - beta2), f32((1 - beta2 ** t) ** 0.5), f32(eps), f32((lr / (1 - beta1 ** t)) * -1))``.

`SgdForeach` is ``torch.optim.SGD(foreach=True)`` with momentum 0 and weight decay 0 (`_multi_tensor_sgd`):
``_foreach_add_(params, grads, alpha=-lr)``, ForeachFunctors' ``BinaryOpListAlphaFunctor``: ``p + alpha * g`` in f32,
alpha ``nlr = f32(-lr)``. ALPHA_FMA is that expression's contraction, the second the SASS check settles: 1 is
``fma(nlr, g, p)``, 0 ``RN(p + RN(nlr * g))``.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, and_reduce, const_bits, mux, sub
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.sqrt import f32_sqrt_rn

__all__ = ["AdamWForeach", "SCALARS", "SgdForeach", "adamw_element", "lerp", "sgd_element"]

F32 = Array(32, BL.BIT)
BF16 = Array(16, BL.BIT)
NAN = fp.F32_NAN_PTX
BF16_NAN = 0x7FFF
ONE = 0x3F800000
HALF = 0x3F000000
#: the order of AdamWForeach's k
SCALARS = ("a", "w", "b2", "c2", "s2", "e", "st")


def _eq(C, x, value: int):
    """``x == value`` for a word and a constant."""
    return and_reduce(C, [b if (value >> i) & 1 else NOT(b) for i, b in enumerate(x)])


def lerp(C, m, g, w, fused: bool):
    """Lerp.h's ``lerp(m, g, w)`` in f32: ``|w| < 0.5 ? m + w (g - m) : g - (g - m)(1 - w)``, contracted when ``fused``."""
    d = fp.f32_sub(C, g, m, nan_word=NAN)
    _, small = sub(C, list(w[:31]), const_bits(HALF, 31), 31)
    om = fp.f32_sub(C, fp.f32_const(ONE), w, nan_word=NAN)
    if fused:
        # one FFMA either way, its operands chosen by the branch: (w, d, m) or (-d, 1 - w, g)
        x = mux(C, small, list(w), fp.f32_neg(d))
        y = mux(C, small, d, om)
        z = mux(C, small, list(m), list(g))
        return fp.f32_fma(C, x, y, z, nan_word=NAN)
    lo = fp.f32_add(C, m, fp.f32_mul(C, w, d, nan_word=NAN), nan_word=NAN)
    hi = fp.f32_sub(C, g, fp.f32_mul(C, d, om, nan_word=NAN), nan_word=NAN)
    return mux(C, small, lo, hi)


def adamw_element(C, p, g, m, v, k, fused: bool):
    """One element's (p2, m1, v2) of `AdamWForeach`, k the seven scalars in `SCALARS` order."""
    a, w, b2, c2, s2, e, st = k
    p1 = fp.f32_mul(C, p, a, nan_word=NAN)
    m1 = lerp(C, m, g, w, fused)
    v1 = fp.f32_mul(C, v, b2, nan_word=NAN)
    one = _eq(C, c2, ONE)
    gg = fp.f32_mul(C, g, g, nan_word=NAN)
    v2 = fp.f32_fma(C, mux(C, one, list(g), list(c2)), mux(C, one, list(g), gg), v1, nan_word=NAN)
    r = fp.f32_div(C, f32_sqrt_rn(C, v2), s2, nan_word=NAN)
    dd = fp.f32_add(C, r, e, nan_word=NAN)
    q = fp.f32_div(C, m1, dd, nan_word=NAN)
    p2 = fp.f32_fma(C, st, q, p1, nan_word=NAN)
    return p2, m1, v2


def sgd_element(C, p, g, nlr, fused: bool):
    """``p + nlr * g`` in f32, one FFMA when ``fused``."""
    if fused:
        return fp.f32_fma(C, nlr, g, p, nan_word=NAN)
    return fp.f32_add(C, p, fp.f32_mul(C, nlr, g, nan_word=NAN), nan_word=NAN)


def _flag(S: Any, k: str) -> bool:
    v = getattr(S, k)
    if v not in (0, 1) or isinstance(v, bool):
        raise ValueError(f"{k} is 0 or 1, not {v!r}")
    return bool(v)


def _n(S: Any) -> int:
    v = S.N
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"N is the number of elements, at least 1, not {v!r}")
    return v


def _adamw_build(S: Any):
    n, fused, bf16 = _n(S), _flag(S, "LERP_FMA"), _flag(S, "BF16")

    def build(C, *e):
        p, g, m, v, k = e[:n], e[n:2 * n], e[2 * n:3 * n], e[3 * n:4 * n], e[4 * n:]
        outs = [adamw_element(C, *row, k, fused) for row in zip(p, g, m, v, strict=True)]
        res = [o[0] for o in outs] + [o[1] for o in outs] + [o[2] for o in outs]
        if bf16:
            res += [fp.f32_to_bf16(C, o[0], BF16_NAN) for o in outs]
        return res
    return build


def _adamw_signature(S: Any):
    n, _, bf16 = _n(S), _flag(S, "LERP_FMA"), _flag(S, "BF16")
    row = Array(n, F32)
    outs = (row, row, row) + ((Array(n, BF16),) if bf16 else ())
    return (("p", row), ("g", row), ("m", row), ("v", row), ("k", Array(len(SCALARS), F32))), Tuple(*outs)


def _sgd_build(S: Any):
    n, fused = _n(S), _flag(S, "ALPHA_FMA")
    return lambda C, *e: [sgd_element(C, p, g, e[2 * n], fused) for p, g in zip(e[:n], e[n:2 * n], strict=True)]


def _sgd_signature(S: Any):
    _flag(S, "ALPHA_FMA")
    row = Array(_n(S), F32)
    return (("p", row), ("g", row), ("nlr", F32)), row


AdamWForeach = specialized(
    "AdamWForeach", 1, ("N", "LERP_FMA", "BF16"), _adamw_build, _adamw_signature,
    doc="torch 2.14 AdamW(foreach=True)'s step of N f32 elements on bits: p1 = RN(p a); m1 = lerp(m, g, w) (Lerp.h, LERP_FMA "
        "its contraction); v2 = fma(c2, RN(g g), RN(v b2)), fma(g, g, ..) at c2 = 1; p2 = fma(st, RN(m1 / RN(RN(sqrt.rn(v2) / s2) "
        "+ e)), p1); -> (p2, m1, v2[, bf16_rn(p2)]); k = (a, w, b2, c2, s2, e, st); every f32 NaN 0x7FFFFFFF, bf16 NaN 0x7FFF")
SgdForeach = specialized(
    "SgdForeach", 1, ("N", "ALPHA_FMA"), _sgd_build, _sgd_signature,
    doc="torch 2.14 SGD(foreach=True)'s step of N f32 elements on bits, momentum and weight decay 0: p + nlr g (fma(nlr, g, p) "
        "at ALPHA_FMA = 1, else RN(p + RN(nlr g))); every NaN 0x7FFFFFFF")
