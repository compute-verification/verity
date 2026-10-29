"""torch 2.14's foreach optimizers on f32 state as Boolean Definitions: one proof block a run of N elements of one tensor, and
the step's f32 scalars from the private hyperparameters, once a step.

    AdamWForeach_v1{N, LERP_FMA, BF16}  (p, g, m, v: Array<N, F32>, k: Array<7, F32>)
                                        -> (Array<N, F32>, Array<N, F32>, Array<N, F32>[, Array<N, BF16>])
    SgdForeach_v1{N, ALPHA_FMA}         (p, g: Array<N, F32>, nlr: F32) -> Array<N, F32>
    AdamWScalars_v1                     (lr, beta1, beta2, eps, wd: F64, step: F32) -> (Array<7, F32>, F32, BIT)
      TorchAdamBiasCorrection_v1        (beta1, beta2: F64, step: F32) -> (F64, F64, F32, BIT)
      TorchPowHalf_v1                   (x: F64) -> (F64, BIT)
      TorchAdamScalarsOut_v1            (lr, beta1, beta2, eps, wd, bc1, s2: F64, ok1, ok2: BIT) -> (Array<7, F32>, BIT)
    TorchSgdScalar_v1                   (lr: F64) -> (F32, BIT)

(F32 is ``Array<32, Value<1>>``, F64 ``Array<64, Value<1>>``, BF16 ``Array<16, Value<1>>``, BIT ``Value<1>``.)

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
``multi_tensor_apply_kernel<…LerpFunctor…>`` in torch 2.14's libtorch_cuda.so has both FFMAs, LERP_FMA = 1.

k is the step's seven f32 scalars, the kernels' ``Scalar::to<float>()`` of the host's Python floats, which
`AdamWScalars` computes from the private hyperparameters: ``(a, w, b2, c2, s2, e, st) = (f32(1 - lr * lambda), f32(1 - beta1),
f32(beta2), f32(1 - beta2), f32((1 - beta2 ** t) ** 0.5), f32(eps), f32((lr / (1 - beta1 ** t)) * -1))``.

`SgdForeach` is ``torch.optim.SGD(foreach=True)`` with momentum 0 and weight decay 0 (`_multi_tensor_sgd`):
``_foreach_add_(params, grads, alpha=-lr)``, ForeachFunctors' ``BinaryOpListAlphaFunctor``: ``p + alpha * g`` in f32,
alpha ``nlr = f32(-lr)``. ALPHA_FMA is that expression's contraction: 1 is ``fma(nlr, g, p)``, 0
``RN(p + RN(nlr * g))``; the same SASS check finds the FFMA, ALPHA_FMA = 1.

The scalars are `_multi_tensor_adam`'s host code (not capturable) in CPython on x86-64 Linux, `host_f64`'s arithmetic, once a
step for every tensor:

1. ``t = RN(step + 1)``, the CPU f32 step tensor's ``_foreach_add_``, and ``float(t)`` (exact);
2. ``bc1 = 1 - beta1 ** t`` and ``bc2 = 1 - beta2 ** t``, each ``**`` glibc 2.39's ``pow`` (`host_f64.pow_main`); a beta of
   zero or subnormal, or a power below e^-512, gives 1, as the exact power does;
3. ``s2 = bc2 ** 0.5``, glibc's ``pow`` (CPython returns 1.0 at bc2 = 1);
4. k's seven words, each ``Scalar::to<float>()``: f32 round to nearest, and refused (``c10::checked_convert``) for a finite
   value past FLT_MAX. ``a`` is 1.0 at weight decay ±0, where torch skips the multiply.

``ok`` is the conjunction of what would make torch or CPython raise instead: AdamW's ``0 <= lr``, ``0 <= eps``,
``0 <= beta < 1`` and ``0 <= weight_decay`` (a NaN fails each), ``lr / bc1``'s ZeroDivisionError and the refused casts, and
the step state itself: t a finite integer of at least 1, as every run from step 0 has. Every NaN word is canonical
(0x7FF8000000000000, 0x7FC00000) where x86 keeps a payload; `AdamWForeach`'s outputs read no NaN scalar's payload (each one
reaches an f32 operation, whose NaN result is 0x7FFFFFFF, or a compare a NaN fails either way). `TorchSgdScalar` is SGD's
``f32(-lr)``, ``ok`` its ``not lr < 0.0`` (a NaN passes) and the cast.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, OR, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.ops.boolean.sqrt import f32_sqrt_rn
from verity.ml.definitions.registry import host_f64 as H

__all__ = ["AdamWForeach", "AdamWScalars", "SCALARS", "SgdForeach", "TorchAdamBiasCorrection", "TorchAdamScalarsOut",
           "TorchPowHalf", "TorchSgdScalar", "adam_bias_corrections", "adam_scalars_out", "adamw_element", "lerp", "pow_half",
           "sgd_element", "sgd_scalar"]

F32 = Array(32, BL.BIT)
F64 = Array(64, BL.BIT)
BF16 = Array(16, BL.BIT)
NAN = fp.F32_NAN_PTX
BF16_NAN = 0x7FFF
ONE = 0x3F800000
HALF = 0x3F000000
HALF64 = 0x3FE0000000000000
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


def _nonneg(C, x):
    """Python's ``0.0 <= x`` for an f64 word: not a NaN, and ±0 or positive."""
    X = H.unpack(C, x)
    return C.AND(NOT(X.nan), OR(C, NOT(X.s), X.zero))


def _beta_ok(C, b):
    """AdamW's ``0.0 <= beta < 1.0``."""
    return C.AND(H.u_less(C, b[:63], H.F64_ONE), OR(C, NOT(b[63]), H.unpack(C, b).zero))


def _whole(C, t):
    """The f32 word t is a finite integer of at least 1: positive, exponent field 127 up, not all ones, and the mantissa bits
    below the binary point (those `fp.shl` by ``e - 127`` keeps) zero."""
    e = t[23:31]
    amt, below = sub(C, e, const_bits(127, 8), 8)
    frac = fp.shl(C, list(t[:23]), amt)
    return and_reduce(C, [NOT(t[31]), NOT(below), NOT(and_reduce(C, e)), NOT(or_reduce(C, frac))])


def _one_minus_pow(C, beta, y):
    """``1 - beta ** y`` for 0 <= beta < 1 and y at least 1: 1 at beta zero or subnormal and past glibc's main path (``big``,
    where the power is below e^-512), else ``1 - pow_main(beta, y)``."""
    p, big = H.pow_main(C, beta, y)
    one = OR(C, NOT(H.unpack(C, beta).e_nz), big)
    return mux(C, one, H.c64(H.F64_ONE), H.f64_sub(C, H.c64(H.F64_ONE), p))


def adam_bias_corrections(C, b1, b2, step):
    """``(bc1, bc2, t, ok)``: the incremented step t = RN(step + 1) and ``1 - beta ** float(t)`` for each beta; ``ok`` is both
    betas in [0, 1) and t a finite integer of at least 1."""
    t = fp.f32_add(C, step, fp.f32_const(ONE))
    y = H.f32_to_f64(C, t)
    ok = and_reduce(C, [_beta_ok(C, b1), _beta_ok(C, b2), _whole(C, t)])
    return _one_minus_pow(C, b1, y), _one_minus_pow(C, b2, y), t, ok


def pow_half(C, x):
    """``(x ** 0.5, ok)`` as CPython computes it, 1.0 at x = 1 and else glibc's pow, whose main path covers every positive
    normal x (``|0.5 log x| < 355``); ``ok`` is x a positive normal."""
    X = H.unpack(C, x)
    p, _ = H.pow_main(C, x, H.c64(HALF64))
    ok = and_reduce(C, [NOT(X.s), X.e_nz, NOT(X.e_max)])
    return mux(C, _eq(C, x, H.F64_ONE), H.c64(H.F64_ONE), p), ok


def adam_scalars_out(C, lr, b1, b2, eps, wd, bc1, s2, ok1, ok2):
    """``(k, ok)``: k's seven f32 words in `SCALARS` order from the Python floats, and ``ok1 & ok2`` with the hyperparameter
    checks, ``bc1 != 0`` and no refused cast (module docstring)."""
    one = H.c64(H.F64_ONE)
    wd0 = H.unpack(C, wd).zero
    vals = [H.f64_sub(C, one, H.f64_mul(C, lr, wd)), H.f64_sub(C, one, b1), b2, H.f64_sub(C, one, b2), s2, eps,
            H.f64_neg(H.f64_div(C, lr, bc1))]
    words, refused = zip(*(H.f64_to_f32(C, v) for v in vals))
    k = [mux(C, wd0, fp.f32_const(ONE), words[0])] + list(words[1:])
    refused = [C.AND(NOT(wd0), refused[0])] + list(refused[1:])
    ok = and_reduce(C, [ok1, ok2, _nonneg(C, lr), _nonneg(C, eps), _nonneg(C, wd), NOT(H.unpack(C, bc1).zero)]
                    + [NOT(r) for r in refused])
    return k, ok


def sgd_scalar(C, lr):
    """``(f32(-lr), ok)``: ``ok`` is SGD's ``not lr < 0.0`` and the cast not refused."""
    X = H.unpack(C, lr)
    nlr, refused = H.f64_to_f32(C, H.f64_neg(lr))
    return nlr, C.AND(NOT(C.AND(X.s, NOT(OR(C, X.zero, X.nan)))), NOT(refused))


K = Array(len(SCALARS), F32)

TorchAdamBiasCorrection = definition(
    "TorchAdamBiasCorrection", 1, lambda C, b1, b2, step: (lambda r: [r[0], r[1], r[2], [r[3]]])(adam_bias_corrections(C, b1, b2, step)),
    (("beta1", F64), ("beta2", F64), ("step", F32)), Tuple(F64, F64, F32, BL.BIT),
    doc="_multi_tensor_adam's step increment and bias corrections on bits: t = RN(step + 1) (f32), (1 - beta1 ** t, "
        "1 - beta2 ** t) with CPython's ** (glibc 2.39 pow, FMA variant), ok = 0 <= betas < 1 and t a finite integer >= 1")
TorchPowHalf = definition(
    "TorchPowHalf", 1, lambda C, x: (lambda r: [r[0], [r[1]]])(pow_half(C, x)), (("x", F64),), Tuple(F64, BL.BIT),
    doc="CPython's x ** 0.5 on bits (1.0 at x = 1, else glibc 2.39 pow, FMA variant), ok = x a positive normal")
TorchAdamScalarsOut = definition(
    "TorchAdamScalarsOut", 1,
    lambda C, lr, b1, b2, eps, wd, bc1, s2, ok1, ok2: (lambda r: [*r[0], [r[1]]])(
        adam_scalars_out(C, lr, b1, b2, eps, wd, bc1, s2, ok1[0], ok2[0])),
    (("lr", F64), ("beta1", F64), ("beta2", F64), ("eps", F64), ("wd", F64), ("bc1", F64), ("s2", F64), ("ok1", BL.BIT),
     ("ok2", BL.BIT)), Tuple(K, BL.BIT),
    doc="AdamW's seven f32 scalars on bits: Scalar::to<float> of (1 - lr wd or 1.0 at wd = 0, 1 - beta1, beta2, 1 - beta2, s2, "
        "eps, (lr / bc1) * -1), ok = ok1 & ok2 & 0 <= lr, eps, wd & bc1 != 0 & no cast refused")
TorchSgdScalar = definition(
    "TorchSgdScalar", 1, lambda C, lr: (lambda r: [r[0], [r[1]]])(sgd_scalar(C, lr)), (("lr", F64),), Tuple(F32, BL.BIT),
    doc="SGD's alpha Scalar::to<float>(-lr) on bits, ok = not lr < 0 and the cast not refused")


def _scalars_signature(S: Any):
    return ((("lr", F64), ("beta1", F64), ("beta2", F64), ("eps", F64), ("wd", F64), ("step", F32)),
            Tuple(K, F32, BL.BIT))


@composite("AdamWScalars", 1, [], _scalars_signature, conformance="exact-model-tested")
def AdamWScalars(B, S, lr, beta1, beta2, eps, wd, step):
    """`_multi_tensor_adam`'s host scalars once a step: `TorchAdamBiasCorrection_v1`, `TorchPowHalf_v1` of bc2, then
    `TorchAdamScalarsOut_v1` -> (k, the incremented step, ok)."""
    bc = B.call(TorchAdamBiasCorrection, beta1, beta2, step)
    s2 = B.call(TorchPowHalf, bc[1])
    out = B.call(TorchAdamScalarsOut, lr, beta1, beta2, eps, wd, bc[0], s2[0], bc[3], s2[1])
    return tuple_of(out[0], bc[2], out[1])
