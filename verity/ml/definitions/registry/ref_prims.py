"""Reference-mode kinds (lane NUMERIC, GUIDANCE §43.1/§43.2 (B)): registered functions whose semantics is the Verity
evaluator's own fully specified arithmetic, for programs derived from ordinary `nn.Module` code.

    Gate -> exactly one Value<w>.  Every evaluator is total on encodings.  Nothing in `prims.py`/`b1.py` changes.

**What a reference kind is.**  A composite whose body is explicit dataflow over IEEE primitives (`F32Add/Sub/Mul/Fma/
Div/Sqrt`, the aten conversions `Bf16ToF32_v1` / `F32ToBf16Rn_v1`) plus the three correctly-rounded transcendental
primitives below.  Every reduction is **sequential, ascending index, from +0** (stated per kind).  The body *is* the
definition; torch CPU is only the conformance oracle over a tested domain (`tests/program/test_ref_prims.py`,
records under `docs/data/ref-prims/`): torch's reductions are vectorised/blocked (`RowwiseMoments`, oneDNN/MKL GEMM,
`vec::reduce_all`) and its transcendentals are not correctly rounded (MKL VML for `torch.exp/erf/tanh`, ATen vectorised
polynomials/Sleef inside fused kernels, libm on strided tensors — three last-bit-different implementations in one build),
so bit agreement with torch is *measured*, never assumed, and a reference program's correspondence is to the reference
execution only — never to a CUDA/vLLM run (brief).

**Rounding points.**  `DT ∈ {"bf16", "f32"}` is the source model's tensor dtype (a static of every kind).  Internal
arithmetic is f32; a `DT` rounding (`F32ToBf16Rn_v1`, RNE, NaN -> 0x7FC0 — torch's `.to(bfloat16)`) is applied exactly
where the torch op materialises a `DT` tensor.  With `DT="f32"` the widen/narrow steps are identities (no gates).

**New primitives** (all `Name_v1`, total, NaN outputs canonical `0x7FC00000`):
  `F32Sub`      RN(a - b)                       `F32Sqrt`    RN(sqrt x) (IEEE; negative -> NaN, -0 -> -0)
  `F32MaxRef`   NaN-propagating max             `F32GtStrict` a > b on f32 values (NaN false) -> bit
  `F32ExpRn`, `F32ErfRn`, `F32TanhRn`  RN32 of the value computed by the stated `decimal` algorithm at 100 digits
      (`_DEC_PREC`); this coincides with correct rounding unless the exact value lies within 10^-90 relative of a
      rounding boundary — no binary32 argument is known to (Lefèvre/Muller worst cases for exp are ~2^-60), and the
      evaluator, not the mathematical function, is the registered definition.
  `GatherF32x{V}` the f32 twin of `prims.gather_bf16` (out-of-range token -> 0x7FC00000, documented extension).
  `I64ToI32`    range-checked int64 -> int32 (DERIVE R4; out of range -> 0x80000000 sentinel, torch wraps).

**Composites** (statics in capitals; see each docstring for the exact order):
  `DotRef{K}`, `LinearRef{K,N,DT,BIAS}`, `RMSNormRef{N,EPS,DT}`, `LayerNormRef{N,EPS,DT}`, `GeluRefElem{APPROX}` /
  `GeluRef{N,DT,APPROX}`, `SiluRefElem` / `SiluRef{N,DT}`, `SoftmaxRef{N,DT}`, `RoPERef{D,DT}` / `RoPEHeadsRef{NHEADS,D,DT}`,
  `RoPEInterleavedRef{D,DT}` (GPT-J pairing), `EagerAttentionHeadRef{T,D,DT,SCALE_OP}` / `EagerAttentionRef{T,NH,KVH,D,DT,SCALE_OP}`,
  `EmbeddingRef{V,H,DT}`, `AddRef{N,DT}`, `MulRef{N,DT}`, `ReluRef{N,DT}`, `ArgmaxStepF32` / `ArgmaxRef{V,DT}`.
  Profile-mode additions at DERIVE's request, NOT in the reference digest: `RoPEHeadGptJ{D}` / `RoPEGptJ{NHEADS,D}` (vLLM
  `rotary_embedding(is_neox=False)`, proof-supported through the `RopeOut`/`RopeOutAdd` twins; superseded by `boolean_rope`'s v2).

**Loader note.**  Decoding a descriptor that names a reference kind requires `import verity.ml.definitions.registry.ref_prims` first
(as for `hopper`/`fp8`).  `GatherF32x{V}_v1` is an on-demand family like `GatherBf16x{V}`; the codec's lazy-family table
(`codec._LAZY_PRIMITIVE_FAMILIES`, a frozen file) is *extended by rebinding* from this module at import — additive, no
edit to `codec.py` (stated per the lane brief).

**Proof support:** none tonight (no Rust twin for any kind here; `zk/sp1/common/src/gateset.rs` untouched).

**Vocabulary pin:** `REF_VOCAB_VERSION` and `ref_vocab_digest()` (SHA-256 over the version string, this module's source,
every reference kind's `(id, body/evaluator source hash)` and the shared `prims.py` primitives' `(id, evaluator source
hash)`), for the importer's provenance record (brief boundary 5: "pin the primitive vocabulary and numerical dependencies").
"""

from __future__ import annotations

import decimal
import hashlib
import inspect
import os
from fractions import Fraction
from functools import lru_cache

import numpy as np

from verity.ml.circuits.codec import register_lazy_family
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind, composite, primitive
from verity.ml.circuits.refs import Coll as _Coll, Explicit, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, BIT, F32, I32, F32_NAN, bits_of, f32_is_nan, f32_of

REF_VOCAB_VERSION = "verity-ref-vocab/1"

DTYPES: dict[str, Value] = {"bf16": BF16, "f32": F32}
_DEC_PREC = 100
_CTX = decimal.Context(prec=_DEC_PREC, rounding=decimal.ROUND_HALF_EVEN, Emax=10 ** 9, Emin=-(10 ** 9),
                       traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
_PI_100 = decimal.Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651")

np.seterr(all="ignore")


# ---------------------------------------------------------------------------------------------------------
# exact helpers on encodings
# ---------------------------------------------------------------------------------------------------------


def _canon_nan(u: int) -> int:
    u &= 0xFFFFFFFF
    return F32_NAN if f32_is_nan(u) else u


def _frac_of_f32(u: int) -> tuple[int, Fraction | None]:
    """(class, exact value): class 0 finite, 1 +inf, -1 -inf, 2 NaN (`prims._decode`)."""
    return P._decode(u)


def _dec_of_frac(fr: Fraction) -> decimal.Decimal:
    return _CTX.divide(decimal.Decimal(fr.numerator), decimal.Decimal(fr.denominator))


def _f32_of_dec(d: decimal.Decimal, neg_zero: bool = False) -> int:
    """RN-even of an exact Decimal to an f32 encoding (through `prims._round_f32`: overflow -> inf, subnormals, zero)."""
    return P._round_f32(Fraction(d), neg_zero=neg_zero)


def _dec_exp(d: decimal.Decimal) -> decimal.Decimal:
    return _CTX.exp(d)


def _dec_erf(x: decimal.Decimal) -> decimal.Decimal:
    """erf(x) = (2/sqrt(pi)) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1)), summed until the term is below 10^-(prec+5)
    relative to 1 (the series alternates; |x| < 4.2 here, so the largest term is < 10^6 and >= 90 digits survive)."""
    x2 = _CTX.multiply(x, x)
    term = x  # x^(2n+1) / n!
    total = x
    n = 0
    tiny = decimal.Decimal(10) ** -(_DEC_PREC + 5)
    while True:
        n += 1
        term = _CTX.divide(_CTX.multiply(term, x2), decimal.Decimal(n))
        contrib = _CTX.divide(term, decimal.Decimal(2 * n + 1))
        total = _CTX.add(total, contrib) if n % 2 == 0 else _CTX.subtract(total, contrib)
        if abs(contrib) < tiny:
            break
    two_over_sqrt_pi = _CTX.divide(decimal.Decimal(2), _CTX.sqrt(_PI_100))
    return _CTX.multiply(total, two_over_sqrt_pi)


@lru_cache(maxsize=1 << 22)
def f32_exp_rn_bits(u: int) -> int:
    """`F32ExpRn_v1`: RN32(exp(x)). NaN -> 0x7FC00000; +inf -> +inf; -inf -> +0; x > 89 -> +inf early exit (the last
    finite result is exp(0x42B17217 = 88.722831) = 0x7F7FFF84; 0x42B17218 rounds to +inf through the general path);
    x < -105 -> +0 (exp(-104) < 2^-150 rounds to zero); otherwise the 100-digit `decimal` exp rounded RN-even."""
    c, v = _frac_of_f32(u)
    if c == 2:
        return F32_NAN
    if c == 1:
        return 0x7F800000
    if c == -1:
        return 0
    if v > 89:
        return 0x7F800000
    if v < -105:
        return 0
    return _f32_of_dec(_dec_exp(_dec_of_frac(v)))


@lru_cache(maxsize=1 << 22)
def f32_erf_rn_bits(u: int) -> int:
    """`F32ErfRn_v1`: RN32(erf(x)). NaN -> 0x7FC00000; ±0 -> ±0; ±inf and |x| >= 4.2 -> ±1.0 (1 - erf(4.2) < 2^-25, so
    RN32 is exactly 1); otherwise the 100-digit Taylor series rounded."""
    c, v = _frac_of_f32(u)
    if c == 2:
        return F32_NAN
    if c == 1:
        return 0x3F800000
    if c == -1:
        return 0xBF800000
    if v == 0:
        return u & 0x80000000
    if v >= Fraction(21, 5):
        return 0x3F800000
    if v <= -Fraction(21, 5):
        return 0xBF800000
    return _f32_of_dec(_dec_erf(_dec_of_frac(v)))


@lru_cache(maxsize=1 << 22)
def f32_tanh_rn_bits(u: int) -> int:
    """`F32TanhRn_v1`: RN32(tanh(x)) with tanh(x) = (e^{2x} - 1) / (e^{2x} + 1). NaN -> 0x7FC00000; ±0 -> ±0; ±inf and
    |x| >= 9.2 -> ±1.0 (1 - tanh(9.2) < 2^-25); otherwise the 100-digit `decimal` evaluation rounded."""
    c, v = _frac_of_f32(u)
    if c == 2:
        return F32_NAN
    if c == 1:
        return 0x3F800000
    if c == -1:
        return 0xBF800000
    if v == 0:
        return u & 0x80000000
    if v >= Fraction(46, 5):
        return 0x3F800000
    if v <= -Fraction(46, 5):
        return 0xBF800000
    e2 = _dec_exp(_CTX.multiply(decimal.Decimal(2), _dec_of_frac(v)))
    t = _CTX.divide(_CTX.subtract(e2, decimal.Decimal(1)), _CTX.add(e2, decimal.Decimal(1)))
    return _f32_of_dec(t)


# ---------------------------------------------------------------------------------------------------------
# new scalar primitives
# ---------------------------------------------------------------------------------------------------------

_REF_CONF = ("reference-defined; the evaluator is the definition; conformance vs torch CPU over the tested domain only: "
             "docs/data/ref-prims/ (tests/program/test_ref_prims.py)")


@primitive("F32Sub", 1, [("a", F32), ("b", F32)], F32, conformance=_REF_CONF)
def F32Sub(a: int, b: int) -> int:
    """FSUB: RN(a - b), IEEE RN-even, no ftz; NaN -> 0x7FC00000 (canonical, host-independent)."""
    return _canon_nan(bits_of(np.float32(f32_of(a)) - np.float32(f32_of(b))))


@primitive("F32Sqrt", 1, [("x", F32)], F32, conformance=_REF_CONF)
def F32Sqrt(x: int) -> int:
    """IEEE `sqrt.rn.f32`: RN(sqrt(x)); negative (other than -0) and NaN -> 0x7FC00000; -0 -> -0; +inf -> +inf."""
    x &= 0xFFFFFFFF
    if x == 0x80000000:
        return x
    return _canon_nan(bits_of(np.sqrt(np.float32(f32_of(x)))))


@primitive("F32MaxRef", 1, [("a", F32), ("b", F32)], F32, conformance=_REF_CONF)
def F32MaxRef(a: int, b: int) -> int:
    """NaN-propagating maximum (torch `max`/`amax` reduction): NaN if either operand is NaN (-> 0x7FC00000); otherwise
    `a` if a >= b else `b` (equal values, i.e. ±0, return `a`)."""
    if f32_is_nan(a) or f32_is_nan(b):
        return F32_NAN
    return (a & 0xFFFFFFFF) if f32_of(a) >= f32_of(b) else (b & 0xFFFFFFFF)


@primitive("F32GtStrict", 1, [("a", F32), ("b", F32)], BIT, conformance=_REF_CONF)
def F32GtStrict(a: int, b: int) -> int:
    """`a > b` on f32 values (NaN compares false) — the f32 twin of `Bf16GtStrict_v1` for greedy argmax."""
    return 1 if f32_of(a) > f32_of(b) else 0


@primitive("F32ExpRn", 1, [("x", F32)], F32, conformance=_REF_CONF)
def F32ExpRn(x: int) -> int:
    """Correctly-rounded exp: see `f32_exp_rn_bits`."""
    return f32_exp_rn_bits(x & 0xFFFFFFFF)


@primitive("F32ErfRn", 1, [("x", F32)], F32, conformance=_REF_CONF)
def F32ErfRn(x: int) -> int:
    """Correctly-rounded erf: see `f32_erf_rn_bits`."""
    return f32_erf_rn_bits(x & 0xFFFFFFFF)


@primitive("F32TanhRn", 1, [("x", F32)], F32, conformance=_REF_CONF)
def F32TanhRn(x: int) -> int:
    """Correctly-rounded tanh: see `f32_tanh_rn_bits`."""
    return f32_tanh_rn_bits(x & 0xFFFFFFFF)


_GATHERS_F32: dict[int, PrimitiveDefinition] = {}


def gather_f32(vocab: int) -> PrimitiveDefinition:
    """`GatherF32x{V}`: (token: i32, column: Array<V, f32>) -> f32 = column[token]; out-of-range or negative token ->
    0x7FC00000 (a total, documented extension; torch raises IndexError)."""
    p = _GATHERS_F32.get(vocab)
    if p is None:
        def ev(tok: int, *col: int, _V=vocab) -> int:
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return col[t] if 0 <= t < _V else F32_NAN

        def ev_lazy(get, _V=vocab) -> int:
            tok = get(0)
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return get(1 + t) if 0 <= t < _V else F32_NAN

        p = PrimitiveDefinition(f"GatherF32x{vocab}", 1, (("token", I32), ("column", Array(vocab, F32))), F32, ev,
                                doc=gather_f32.__doc__ or "", conformance=_REF_CONF)
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _GATHERS_F32[vocab] = p
    return p


register_lazy_family(r"^GatherF32x(\d+)_v1$", lambda m: gather_f32(int(m[1])))   # decodable where the program was not built

REF_PRIMITIVES = [F32Sub, F32Sqrt, F32MaxRef, F32GtStrict, F32ExpRn, F32ErfRn, F32TanhRn]


# ---------------------------------------------------------------------------------------------------------
# dtype helpers for bodies
# ---------------------------------------------------------------------------------------------------------


def _dt(DT: str) -> Value:
    try:
        return DTYPES[DT]
    except KeyError:
        raise ValueError(f"DT must be one of {sorted(DTYPES)}, got {DT!r}") from None


def _flat(x: _Coll, n: int, t: Value) -> _Coll:
    return x.reshape(Array(n, t))


def _widen(B, x: _Coll, DT: str) -> _Coll:
    """Array<..., DT> (any nesting) -> the same shape over f32 (`Bf16ToF32_v1` per element; identity for f32)."""
    if DT == "f32":
        return x
    n = x.type.leaves
    shape_f32 = _retype(x.type, F32)
    return B.batch(P.Bf16ToF32, _flat(x, n, BF16), axes=(0,)).reshape(shape_f32)


def _narrow(B, y: _Coll, DT: str) -> _Coll:
    """Array<..., f32> -> Array<..., DT> (`F32ToBf16Rn_v1` per element, torch RNE / NaN 0x7FC0; identity for f32)."""
    if DT == "f32":
        return y
    n = y.type.leaves
    shape_dt = _retype(y.type, BF16)
    return B.batch(P.F32ToBf16Rn, _flat(y, n, F32), axes=(0,)).reshape(shape_dt)


def _retype(t, v: Value):
    if isinstance(t, Value):
        return v
    if isinstance(t, Array):
        return Array(t.n, _retype(t.elem, v))
    raise TypeError(t)


def _widen1(B, x: _Coll, DT: str) -> _Coll:
    return x if DT == "f32" else B.call(P.Bf16ToF32, x)


def _narrow1(B, y: _Coll, DT: str) -> _Coll:
    return y if DT == "f32" else B.call(P.F32ToBf16Rn, y)


def _seq_sum(B, terms: list[_Coll]) -> _Coll:
    """acc_0 = Const32[+0]; acc_{i+1} = F32Add(acc_i, terms[i]) — sequential, ascending index."""
    acc = B.call(b1.ZERO32)
    for t in terms:
        acc = B.call(P.F32Add, acc, t)
    return acc


# ---------------------------------------------------------------------------------------------------------
# dot product and linear
# ---------------------------------------------------------------------------------------------------------


@composite("DotRef", 1, ["K"], lambda S: ((("x", Array(S.K, F32)), ("w", Array(S.K, F32))), F32), conformance=_REF_CONF)
def DotRef(B, S, x, w):
    """IEEE sequential fused dot product: `acc_0 = +0; acc_{k+1} = F32Fma(x[k], w[k], acc_k)` for k = 0..K-1 ascending;
    returns acc_K (f32, no final rounding).  One rounding per step.  NOT the tensor-core `DotBf16_v1` (which aligns
    groups of eight products to 25 bits and truncates) and not torch's blocked GEMM order — the reference definition."""
    acc = B.call(b1.ZERO32)
    for k in range(S.K):
        acc = B.call(P.F32Fma, x[k], w[k], acc)
    return acc


def _linear_sig(S):
    t = _dt(S.DT)
    params = [("x", Array(S.K, t)), ("w", Array(S.N, Array(S.K, t)))]
    if S.BIAS:
        params.append(("b", Array(S.N, t)))
    return tuple(params), Array(S.N, t)


@composite("LinearRef", 1, ["K", "N", "DT", "BIAS"], _linear_sig, conformance=_REF_CONF)
def LinearRef(B, S, x, w, *bias):
    """`F.linear(x, W, b)` = `x[K] @ W[N,K]^T (+ b)` on one row, reference arithmetic: inputs widened to f32
    (`Bf16ToF32_v1`, exact), `acc_n = DotRef{K}(x, W[n,:])` (sequential fma from +0), then
      BIAS:  `y_n = DT(RN(acc_n + f32(b_n)))`  — one f32 add, then one DT rounding (oneDNN/MKL add the bias to the f32
             accumulator before the store; the `Gemm_v1` + `BiasAdd_v1` profile pair rounds to bf16 *before* the add);
      else:  `y_n = DT(acc_n)`.
    GPT-2 `Conv1D` (`x @ W[K,N] + b`) is the same function with W transposed by the importer."""
    K, N, DT = S.K, S.N, S.DT
    xf = _widen(B, x, DT)
    wf = _widen(B, w, DT)
    acc = B.batch(bind(DotRef, K=K), xf, wf, axes=(None, 0))
    if S.BIAS:
        bf = _widen(B, bias[0], DT)
        acc = B.batch(P.F32Add, acc, bf, axes=(0, 0))
    return _narrow(B, acc, DT)


# ---------------------------------------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------------------------------------


@composite("RMSNormRef", 1, ["N", "EPS", "DT"],
           lambda S: ((("x", Array(S.N, _dt(S.DT))), ("w", Array(S.N, _dt(S.DT)))), Array(S.N, _dt(S.DT))),
           conformance=_REF_CONF)
def RMSNormRef(B, S, x, w, eps=None):
    """HF `LlamaRMSNorm` / `Qwen2RMSNorm` (`h = x.to(f32); var = h.pow(2).mean(-1); h = h * rsqrt(var + eps);
    return w * h.to(dt)`), reference arithmetic:
      `sq_i = RN(x_i * x_i)`; `acc = RN(acc + sq_i)` sequential ascending from +0;  `mean = RN(acc / N)`;
      `rs = RN(1 / RN(sqrt(RN(mean + eps))))`  (two roundings: torch CPU x86 `rsqrt` = `1 / sqrt`, NOT `rsqrt.approx`);
      `h_i = DT(RN(x_i * rs))`;  `out_i = DT(RN(f32(w_i) * f32(h_i)))`.
    The weight multiply happens AFTER the DT cast (vLLM's `RMSNormTriton_v1`/`RMSNormFusedCuda_v2` multiply before it and
    use MUFU sqrt/rsqrt) — a different function, not a different implementation.  `EPS` is rounded to f32 (`eps` is a
    Python float added to an f32 tensor)."""
    N, DT = S.N, S.DT
    xf = _widen(B, x, DT)
    sq = B.batch(P.F32Mul, xf, xf, axes=(0, 0))
    acc = _seq_sum(B, [sq[i] for i in range(N)])
    mean = B.call(P.F32Div, acc, B.call(b1.f32c(float(N))))
    var_eps = B.call(P.F32Add, mean, B.call(b1.f32c(S.EPS)) if eps is None else eps)
    rs = B.call(P.F32Div, B.call(b1.f32c(1.0)), B.call(F32Sqrt, var_eps))
    h = B.batch(P.F32Mul, xf, rs, axes=(0, None))
    hf = _widen(B, _narrow(B, h, DT), DT)
    wf = _widen(B, w, DT)
    y = B.batch(P.F32Mul, wf, hf, axes=(0, 0))
    return _narrow(B, y, DT)


@composite("LayerNormRef", 1, ["N", "EPS", "DT"],
           lambda S: ((("x", Array(S.N, _dt(S.DT))), ("g", Array(S.N, _dt(S.DT))), ("b", Array(S.N, _dt(S.DT)))),
                      Array(S.N, _dt(S.DT))),
           conformance=_REF_CONF)
def LayerNormRef(B, S, x, g, b, eps=None):
    """`F.layer_norm(x, (N,), g, b, eps)` on one row, reference arithmetic (two-pass, biased variance, f32 internally):
      `mean = RN(Σ_i x_i / N)` with the sum sequential ascending from +0;  `d_i = RN(x_i - mean)`;
      `var = RN(Σ_i RN(d_i * d_i) / N)` (sequential);  `rstd = RN(1 / RN(sqrt(RN(var + eps))))`;
      `out_i = DT(RN(RN(RN(d_i * rstd) * f32(g_i)) + f32(b_i)))`.
    torch CPU computes the moments with `RowwiseMoments` (chunked Welford) and applies `x * (rstd*g) + (b - mean*rstd*g)`
    in a vectorised loop — the same real function, a different rounding sequence; agreement is measured, not assumed."""
    N, DT = S.N, S.DT
    xf = _widen(B, x, DT)
    n_c = B.call(b1.f32c(float(N)))
    mean = B.call(P.F32Div, _seq_sum(B, [xf[i] for i in range(N)]), n_c)
    d = B.batch(F32Sub, xf, mean, axes=(0, None))
    sq = B.batch(P.F32Mul, d, d, axes=(0, 0))
    var = B.call(P.F32Div, _seq_sum(B, [sq[i] for i in range(N)]), n_c)
    rstd = B.call(P.F32Div, B.call(b1.f32c(1.0)), B.call(F32Sqrt, B.call(P.F32Add, var, B.call(b1.f32c(S.EPS)) if eps is None else eps)))
    gf = _widen(B, g, DT)
    bf = _widen(B, b, DT)
    y = B.batch(P.F32Add, B.batch(P.F32Mul, B.batch(P.F32Mul, d, rstd, axes=(0, None)), gf, axes=(0, 0)), bf, axes=(0, 0))
    return _narrow(B, y, DT)


# ---------------------------------------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------------------------------------

GELU_APPROX = ("none", "tanh")


@composite("GeluRefElem", 1, ["APPROX"], lambda S: ((("x", F32),), F32), conformance=_REF_CONF)
def GeluRefElem(B, S, x):
    """One f32 GELU.  `APPROX="none"` (torch `F.gelu`, erf form):
        `y = RN(RN(x * 0.5) * RN(1 + erf(RN(x * RN32(1/sqrt 2)))))`  with `erf` = `F32ErfRn_v1`;
    `APPROX="tanh"` (torch `F.gelu(approximate="tanh")` and HF `NewGELUActivation` / GPT-2 `gelu_new`):
        `x3 = RN(RN(x * x) * x)`; `inner = RN(β * RN(x + RN(κ * x3)))`, `β = RN32(sqrt(2/π))`, `κ = RN32(0.044715)`;
        `y = RN(RN(x * 0.5) * RN(1 + tanh(inner)))`  with `tanh` = `F32TanhRn_v1`.
    Left-to-right evaluation of torch's `x * 0.5 * (1 + f(...))` (the `0.5` product is exact).  torch CPU's `f` is not
    correctly rounded (contiguous: ATen's vectorised erf polynomial, abs. error up to ~3.5e-7 near 0, and Sleef tanh ≤ 1 ulp;
    strided: libm), and for x < 0 the `1 + f` cancellation amplifies that, so f32 results differ from torch on a large
    fraction of words while staying within `0.5·|x|·2^-21 + 4 ulp` absolutely; bf16 results agree on all but a handful of
    the 65,280 finite words (measured, `docs/data/ref-prims/GeluRef_v1[...]`)."""
    if S.APPROX not in GELU_APPROX:
        raise ValueError(f"APPROX must be one of {GELU_APPROX}, got {S.APPROX!r}")
    half_x = B.call(P.F32Mul, x, B.call(b1.f32c(0.5)))
    one = B.call(b1.f32c(1.0))
    if S.APPROX == "none":
        a = B.call(P.F32Mul, x, B.call(b1.f32c(0.7071067811865476)))
        f = B.call(F32ErfRn, a)
    else:
        x2 = B.call(P.F32Mul, x, x)
        x3 = B.call(P.F32Mul, x2, x)
        kx3 = B.call(P.F32Mul, B.call(b1.f32c(0.044715)), x3)
        inner = B.call(P.F32Mul, B.call(b1.f32c(0.7978845608028654)), B.call(P.F32Add, x, kx3))
        f = B.call(F32TanhRn, inner)
    return B.call(P.F32Mul, half_x, B.call(P.F32Add, one, f))


@composite("GeluRef", 1, ["N", "DT", "APPROX"], lambda S: ((("x", Array(S.N, _dt(S.DT))),), Array(S.N, _dt(S.DT))),
           conformance=_REF_CONF)
def GeluRef(B, S, x):
    """`GeluRefElem{APPROX}` per element: widen to f32, compute, one DT rounding (torch's bf16 GELU kernel computes in f32
    and rounds once)."""
    y = B.batch(bind(GeluRefElem, APPROX=S.APPROX), _widen(B, x, S.DT), axes=(0,))
    return _narrow(B, y, S.DT)


@composite("SiluRefElem", 1, [], lambda S: ((("x", F32),), F32), conformance=_REF_CONF)
def SiluRefElem(B, S, x):
    """One f32 SiLU as torch `F.silu` computes it: `y = RN(x / RN(1 + exp(RN(0 - x))))` with `exp` = `F32ExpRn_v1`.
    (`0 - x` differs from negation only at x = +0, where exp(±0) = 1 either way.)  Two roundings after the exp — unlike
    `SiluMulBf16_v1`, whose `silu_f32` is RN32 of the exact real value (the CUDA-kernel model)."""
    e = B.call(F32ExpRn, B.call(F32Sub, B.call(b1.ZERO32), x))
    return B.call(P.F32Div, x, B.call(P.F32Add, B.call(b1.f32c(1.0)), e))


@composite("SiluRef", 1, ["N", "DT"], lambda S: ((("x", Array(S.N, _dt(S.DT))),), Array(S.N, _dt(S.DT))), conformance=_REF_CONF)
def SiluRef(B, S, x):
    """`SiluRefElem` per element with one DT rounding."""
    y = B.batch(SiluRefElem, _widen(B, x, S.DT), axes=(0,))
    return _narrow(B, y, S.DT)


@composite("SoftmaxRef", 1, ["N", "DT"], lambda S: ((("x", Array(S.N, _dt(S.DT))),), Array(S.N, _dt(S.DT))), conformance=_REF_CONF)
def SoftmaxRef(B, S, x):
    """`F.softmax(x, dim=-1)` (or `softmax(..., dtype=f32).to(dt)`) on one row, reference arithmetic:
      `m = max_i x_i` (`F32MaxRef_v1` fold, ascending, NaN-propagating);  `e_i = exp(RN(x_i - m))` (`F32ExpRn_v1`);
      `s = Σ_i e_i` sequential ascending from +0;  `r = RN(1 / s)`;  `out_i = DT(RN(e_i * r))`.
    torch multiplies by the reciprocal (`tmp_sum = 1 / tmp_sum; out = exp * tmp_sum`); it does not divide per element."""
    N, DT = S.N, S.DT
    xf = _widen(B, x, DT)
    m = xf[0]
    for i in range(1, N):
        m = B.call(F32MaxRef, m, xf[i])
    e = B.batch(F32ExpRn, B.batch(F32Sub, xf, m, axes=(0, None)), axes=(0,))
    s = _seq_sum(B, [e[i] for i in range(N)])
    r = B.call(P.F32Div, B.call(b1.f32c(1.0)), s)
    return _narrow(B, B.batch(P.F32Mul, e, r, axes=(0, None)), DT)


# ---------------------------------------------------------------------------------------------------------
# rotary embedding (HF rotate_half form)
# ---------------------------------------------------------------------------------------------------------


@composite("RoPERef", 1, ["D", "DT"],
           lambda S: ((("x", Array(S.D, _dt(S.DT))), ("cos", Array(S.D, _dt(S.DT))), ("sin", Array(S.D, _dt(S.DT)))),
                      Array(S.D, _dt(S.DT))),
           conformance=_REF_CONF)
def RoPERef(B, S, x, cos, sin):
    """HF `apply_rotary_pos_emb`: `x * cos + rotate_half(x) * sin`, `rotate_half(x) = cat(-x[R:], x[:R])`, R = D/2, every
    op a DT tensor op: `a_i = DT(RN(x_i * cos_i))`; `p_i = DT(RN(x_{i+R} * sin_i))` (i < R), `q_i = DT(RN(x_{i-R} * sin_i))`
    (i >= R); `out_i = DT(RN(f32(a_i) - f32(p_i)))` for i < R (the negation of `-x[R:]` commutes with RNE, so `a - p` is
    `a + (-p)` exactly), `out_i = DT(RN(f32(a_i) + f32(q_i)))` for i >= R.  THREE DT roundings per output (the vLLM CUDA
    `RoPEHead_v1` rounds once: `bf16(fma(x, c, -(y*s)))`).  `cos`/`sin` are the model's D-long rows (HF duplicates the
    half-table: `cos[i] == cos[i+R]`) — Inputs computed by torch `cos`/`sin` in f32 and cast to DT: a derived table, the
    same qualification class as vLLM's `cos_sin_cache` (the values are not reproducible from the formula bit-exactly)."""
    D, DT = S.D, S.DT
    R = D // 2
    xf, cf, sf = _widen(B, x, DT), _widen(B, cos, DT), _widen(B, sin, DT)
    a = _widen(B, _narrow(B, B.batch(P.F32Mul, xf, cf, axes=(0, 0)), DT), DT)
    p = _widen(B, _narrow(B, B.batch(P.F32Mul, xf[R:D], sf[0:R], axes=(0, 0)), DT), DT)  # x[i+R] * sin[i]
    q = _widen(B, _narrow(B, B.batch(P.F32Mul, xf[0:R], sf[R:D], axes=(0, 0)), DT), DT)  # x[i-R] * sin[i]
    lo = B.batch(F32Sub, a[0:R], p, axes=(0, 0))
    hi = B.batch(P.F32Add, a[R:D], q, axes=(0, 0))
    return _narrow(B, _Coll(Array(D, F32), concat([lo.refs, hi.refs])), DT)


@composite("RoPEHeadsRef", 1, ["NHEADS", "D", "DT"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, _dt(S.DT)))), ("cos", Array(S.D, _dt(S.DT))), ("sin", Array(S.D, _dt(S.DT)))),
                      Array(S.NHEADS, Array(S.D, _dt(S.DT)))),
           conformance=_REF_CONF)
def RoPEHeadsRef(B, S, x, cos, sin):
    """`RoPERef{D,DT}` over NHEADS heads with the shared cos/sin row of one position."""
    return B.batch(bind(RoPERef, D=S.D, DT=S.DT), x, cos, sin, axes=(0, None, None))


# ---------------------------------------------------------------------------------------------------------
# eager attention (HF `eager_attention_forward`)
# ---------------------------------------------------------------------------------------------------------

SCALE_OPS = ("mul", "div")


@composite("EagerAttentionHeadRef", 1, ["T", "D", "DT", "SCALE_OP"],
           lambda S: ((("q", Array(S.D, _dt(S.DT))), ("k", Array(S.T, Array(S.D, _dt(S.DT)))), ("v", Array(S.T, Array(S.D, _dt(S.DT))))),
                      Array(S.D, _dt(S.DT))),
           conformance=_REF_CONF)
def EagerAttentionHeadRef(B, S, q, k, v):
    """One query row (position T-1) of one head over keys 0..T-1, all visible, as HF eager attention computes it:
      `s_j = DT(DotRef{D}(q, k_j))`                        — `matmul(q, k^T)`: f32 sequential fma, one DT rounding;
      `t_j = DT(RN(f32(s_j) * RN32(D^-0.5)))`  (SCALE_OP="mul", HF `* scaling`) or
      `t_j = DT(RN(f32(s_j) / RN32(sqrt D)))`  (SCALE_OP="div", GPT-2 `/ sqrt(head_dim)`);
      `p = SoftmaxRef{T,DT}(t)`                            — `softmax(dtype=f32).to(dt)`: one DT rounding of the f32 result;
      `o_d = DT(DotRef{T}(f32(p), v[:, d]))`               — `matmul(p, v)`: f32 sequential fma over keys ascending.
    A causal T×T prefill is one call per query row i with T = i+1: the additive `finfo(dt).min` mask makes every masked
    column's exponential exactly 0, and adding `0.0` to a visible score changes at most the sign of a zero, which
    `exp(x - m)` does not see — so omitting masked columns is exact, not an approximation."""
    if S.SCALE_OP not in SCALE_OPS:
        raise ValueError(f"SCALE_OP must be one of {SCALE_OPS}, got {S.SCALE_OP!r}")
    T, D, DT = S.T, S.D, S.DT
    qf, kf, vf = _widen(B, q, DT), _widen(B, k, DT), _widen(B, v, DT)
    s = _widen(B, _narrow(B, B.batch(bind(DotRef, K=D), qf, kf, axes=(None, 0)), DT), DT)
    if S.SCALE_OP == "mul":
        t = B.batch(P.F32Mul, s, B.call(b1.f32c(float(D) ** -0.5)), axes=(0, None))
    else:
        t = B.batch(P.F32Div, s, B.call(b1.f32c(float(D) ** 0.5)), axes=(0, None))
    p = B.call(bind(SoftmaxRef, N=T, DT=DT), _narrow(B, t, DT))
    pf = _widen(B, p, DT)
    vt = _Coll(Array(D, Array(T, F32)), strided_view(vf, 0, T, 1, D, T * D))  # column d over the T keys
    o = B.batch(bind(DotRef, K=T), pf, vt, axes=(None, 0))
    return _narrow(B, o, DT)


@composite("EagerAttentionRef", 1, ["T", "NH", "KVH", "D", "DT", "SCALE_OP"],
           lambda S: ((("q", Array(S.NH, Array(S.D, _dt(S.DT)))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, _dt(S.DT))))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, _dt(S.DT)))))),
                      Array(S.NH, Array(S.D, _dt(S.DT)))),
           conformance=_REF_CONF)
def EagerAttentionRef(B, S, q, kc, vc):
    """GQA over `EagerAttentionHeadRef`: head h reads KV head h // (NH/KVH) (HF `repeat_kv`); caches `[T][KVH][D]` as in
    `Attention_v1` so the importer's KV-state translation is shared between the two modes."""
    T, NH, KVH, D, DT = S.T, S.NH, S.KVH, S.D, S.DT
    G = NH // KVH
    head = bind(EagerAttentionHeadRef, T=T, D=D, DT=DT, SCALE_OP=S.SCALE_OP)
    vt = _dt(DT)
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, vt)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, vt)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return _Coll(Array(NH, Array(D, vt)), concat([o.refs for o in outs]))


# ---------------------------------------------------------------------------------------------------------
# embedding, element-wise, argmax (DT-generic twins of the bf16 profile kinds)
# ---------------------------------------------------------------------------------------------------------


@composite("EmbeddingRef", 1, ["V", "H", "DT"],
           lambda S: ((("tok", I32), ("table", Array(S.V, Array(S.H, _dt(S.DT))))), Array(S.H, _dt(S.DT))),
           conformance=_REF_CONF)
def EmbeddingRef(B, S, tok, table):
    """`nn.Embedding`: `h[j] = table[tok][j]` — one wide-fan-in gather per hidden coordinate (`GatherBf16x{V}` for bf16,
    exactly `Embedding_v1`'s body; `GatherF32x{V}` for f32).  A gather has no arithmetic: exact for in-range tokens."""
    vt = _dt(S.DT)
    cols = _Coll(Array(S.H, Array(S.V, vt)), strided_view(table, 0, S.V, 1, S.H, S.V * S.H))
    g = P.gather_bf16(S.V) if S.DT == "bf16" else gather_f32(S.V)
    return B.batch(g, tok, cols, axes=(None, 0))


@composite("AddRef", 1, ["N", "DT"], lambda S: ((("a", Array(S.N, _dt(S.DT))), ("b", Array(S.N, _dt(S.DT)))), Array(S.N, _dt(S.DT))),
           conformance=_REF_CONF)
def AddRef(B, S, a, b):
    """`aten.add(a, b)` element-wise in DT: bf16 -> `Bf16Add_v1` (= `ResidualAdd_v1`'s body: f32 add, RNE, NaN 0x7FC0);
    f32 -> `F32Add_v1`."""
    return B.batch(P.Bf16Add if S.DT == "bf16" else P.F32Add, a, b, axes=(0, 0))


@composite("MulRef", 1, ["N", "DT"], lambda S: ((("a", Array(S.N, _dt(S.DT))), ("b", Array(S.N, _dt(S.DT)))), Array(S.N, _dt(S.DT))),
           conformance=_REF_CONF)
def MulRef(B, S, a, b):
    """`aten.mul(a, b)` element-wise in DT: `DT(RN(f32(a) * f32(b)))` (bf16 via widen/`F32Mul`/`F32ToBf16Rn`, torch NaN
    word; `Bf16MulBf16_v1` is the same function with the CUDA NaN word 0x7FFF); f32 -> `F32Mul_v1`."""
    if S.DT == "f32":
        return B.batch(P.F32Mul, a, b, axes=(0, 0))
    return _narrow(B, B.batch(P.F32Mul, _widen(B, a, "bf16"), _widen(B, b, "bf16"), axes=(0, 0)), "bf16")


@composite("ReluRef", 1, ["N", "DT"], lambda S: ((("x", Array(S.N, _dt(S.DT))),), Array(S.N, _dt(S.DT))), conformance=_REF_CONF)
def ReluRef(B, S, x):
    """`aten.relu` (DERIVE R1): `y_i = F32MaxRef(x_i, +0)` per element: x > 0 -> x; x < 0 -> +0.0; ±0 -> the same ±0 (torch
    2.4.1 CPU returns -0.0 for -0.0 on both its vectorised and scalar `clamp_min` paths — measured, `ReluRef_v1-*.json`);
    NaN -> NaN (0x7FC00000, DT NaN word after narrowing).  Widen/narrow are exact, so the kind is exact in either DT."""
    zero = B.call(b1.f32c(0.0))
    if S.DT == "f32":
        return B.batch(F32MaxRef, x, zero, axes=(0, None))
    return _narrow(B, B.batch(F32MaxRef, _widen(B, x, "bf16"), zero, axes=(0, None)), "bf16")


@composite("RoPEInterleavedRef", 1, ["D", "DT"],
           lambda S: ((("x", Array(S.D, _dt(S.DT))), ("cos", Array(S.D // 2, _dt(S.DT))), ("sin", Array(S.D // 2, _dt(S.DT)))),
                      Array(S.D, _dt(S.DT))),
           conformance=_REF_CONF)
def RoPEInterleavedRef(B, S, x, cos, sin):
    """HF GPT-J `apply_rotary_pos_emb` (DERIVE R3; interleaved pairing, `rotate_every_two`): with `cos`/`sin` the D/2-long
    table rows (HF `repeat_interleave(2)`s them; the kind indexes them directly), R = D/2, for i < R:
        `a_e = DT(RN(x_{2i} * cos_i))`,  `a_o = DT(RN(x_{2i+1} * cos_i))`,
        `p   = DT(RN(x_{2i+1} * sin_i))`, `q   = DT(RN(x_{2i} * sin_i))`,
        `out_{2i} = DT(RN(f32(a_e) - f32(p)))`,  `out_{2i+1} = DT(RN(f32(a_o) + f32(q)))`.
    THREE DT roundings per output, exactly `RoPERef_v1`'s arithmetic with the pairing (2i, 2i+1) instead of (i, i+R).
    `D` is the rotary dimension: GPT-J applies it to `x[:rotary_dim]` and passes the rest through (a view, no kind).  The
    profile-mode twin (vLLM `rotary_embedding(is_neox=False)`, one rounding) is `RoPEHeadGptJ_v1` below."""
    D, DT = S.D, S.DT
    R = D // 2
    xf, cf, sf = _widen(B, x, DT), _widen(B, cos, DT), _widen(B, sin, DT)
    xe = _Coll(Array(R, F32), strided_view(xf, 0, R, 0, 2, R))  # x[2i]
    xo = _Coll(Array(R, F32), strided_view(xf, 1, R, 0, 2, R))  # x[2i+1]
    a_e = _widen(B, _narrow(B, B.batch(P.F32Mul, xe, cf, axes=(0, 0)), DT), DT)
    a_o = _widen(B, _narrow(B, B.batch(P.F32Mul, xo, cf, axes=(0, 0)), DT), DT)
    p = _widen(B, _narrow(B, B.batch(P.F32Mul, xo, sf, axes=(0, 0)), DT), DT)
    q = _widen(B, _narrow(B, B.batch(P.F32Mul, xe, sf, axes=(0, 0)), DT), DT)
    ev = B.batch(F32Sub, a_e, p, axes=(0, 0))
    od = B.batch(P.F32Add, a_o, q, axes=(0, 0))
    inter = Explicit(tuple(r for i in range(R) for r in (ev.refs[i], od.refs[i])))
    return _narrow(B, _Coll(Array(D, F32), inter), DT)


# --- profile-mode addition requested by DERIVE (R3): vLLM's GPT-J rotary kernel, built only from exact-model-tested
#     primitives that have Rust twins (RRopeOut/RRopeOutAdd) — so it is proof-supported, unlike the reference kinds. ---

_GPTJ_CONF = ("superseded by RoPEHeadGptJ_v2 / RoPEGptJ_v2: a pairing permutation of RoPEHead_v1, the same RopeOut_v1/"
              "RopeOutAdd_v1 gates with (2i, 2i+1) in place of (i, i+D/2); vLLM d9105ea80's `rotary_embedding(is_neox=False)` "
              "(bf16, run r20261004-060203-ba52) differs in 513 words of 17 of 60 difftest cases, every one an odd output "
              "whose products overflow f32: the kernel rounds y*cos and fuses x*sin, where NeoX's instantiation rounds x*sin")


@composite("RoPEHeadGptJ", 1, ["D"], lambda S: ((("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
           conformance=_GPTJ_CONF)
def RoPEHeadGptJ(B, S, x, cs):
    """vLLM `rotary_embedding(..., is_neox=False)` on one head with the bf16 cache row `cs = [cos[0:R] | sin[0:R]]`, R = D/2:
    `out[2i] = bf16(x[2i]*cos[i] - x[2i+1]*sin[i])`, `out[2i+1] = bf16(x[2i+1]*cos[i] + x[2i]*sin[i])` — `RopeOut_v1` /
    `RopeOutAdd_v1` (f32 math, ONE bf16 rounding, the CUDA kernel's form), i.e. `RoPEHead_v1` with GPT-J pairing.
    `D` is vLLM's `rotary_dim` (the kernel leaves `x[rotary_dim:]` untouched: a view, no kind)."""
    R = S.D // 2
    cos, sin = cs[0:R], cs[R:S.D]
    xe = _Coll(Array(R, BF16), strided_view(x, 0, R, 0, 2, R))
    xo = _Coll(Array(R, BF16), strided_view(x, 1, R, 0, 2, R))
    lo = B.batch(P.RopeOut, xe, xo, cos, sin, axes=(0, 0, 0, 0))  # x[2i]*cos - x[2i+1]*sin
    hi = B.batch(P.RopeOutAdd, xe, xo, cos, sin, axes=(0, 0, 0, 0))  # x[2i+1]*cos + x[2i]*sin
    return _Coll(Array(S.D, BF16), Explicit(tuple(r for i in range(R) for r in (lo.refs[i], hi.refs[i]))))


@composite("RoPEGptJ", 1, ["NHEADS", "D"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("cs", Array(S.D, BF16))), Array(S.NHEADS, Array(S.D, BF16))),
           conformance=_GPTJ_CONF)
def RoPEGptJ(B, S, x, cs):
    """`RoPEHeadGptJ_v1{D}` over NHEADS heads sharing one cache row (the `RoPE_v1` shape)."""
    return B.batch(bind(RoPEHeadGptJ, D=S.D), x, cs, axes=(0, None))


I64 = Value(64)


@primitive("I64ToI32", 1, [("t", I64)], I32, conformance=_REF_CONF)
def I64ToI32(t: int) -> int:
    """Range-checked narrowing of a torch int64 index to I32 (DERIVE R4): v = the two's-complement value of the 64-bit word;
    `-2^31 <= v < 2^31` -> v as a 32-bit two's-complement word (identical to torch `.to(torch.int32)` there); otherwise
    0x80000000 (INT32_MIN — a negative token, which every Gather kind maps to its NaN word: total and visible).  torch's
    conversion WRAPS modulo 2^32 outside the range instead; that divergence is deliberate and recorded."""
    t &= (1 << 64) - 1
    v = t - (1 << 64) if t >> 63 else t
    return (v & 0xFFFFFFFF) if -(1 << 31) <= v < (1 << 31) else 0x80000000


@composite("ArgmaxStepF32", 1, [], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", F32)), Tuple(Tuple(F32, I32, I32), I32)),
           conformance=_REF_CONF)
def ArgmaxStepF32(B, S, carry, x):
    """`ArgmaxStep_v1` over f32: take (x, i) iff `x > best` strictly (first index wins ties; NaN never wins — torch.argmax
    returns a NaN's index, so the reference domain is NaN-free logits), then i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    gt = B.call(F32GtStrict, x, best)
    nb = B.call(P.SelectF32, gt, x, best)
    ni = B.call(P.SelectI32, gt, i, best_i)
    nxt = B.call(P.I32Add, i, B.call(b1.const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("ArgmaxRef", 1, ["V", "DT"], lambda S: ((("logits", Array(S.V, _dt(S.DT))),), I32), conformance=_REF_CONF)
def ArgmaxRef(B, S, logits):
    """`torch.argmax(logits)` (greedy token selection): lowest index among the maxima; bf16 -> `TokenSelect_v1`'s scan
    (`ArgmaxStep_v1`), f32 -> `ArgmaxStepF32`.  Exact on NaN-free logits."""
    step = b1.ArgmaxStep if S.DT == "bf16" else ArgmaxStepF32
    init = tuple_of(logits[0], B.call(b1.const(32, 0)), B.call(b1.const(32, 1)))
    out = B.scan(step, init, xs=(logits[1:S.V],))
    return out[0][1]


REF_PRIMITIVES.append(I64ToI32)
REF_COMPOSITES = [DotRef, LinearRef, RMSNormRef, LayerNormRef, GeluRefElem, GeluRef, SiluRefElem, SiluRef, SoftmaxRef,
                  RoPERef, RoPEHeadsRef, EagerAttentionHeadRef, EagerAttentionRef, EmbeddingRef, AddRef, MulRef, ReluRef,
                  RoPEInterleavedRef, ArgmaxStepF32, ArgmaxRef]
#: profile-mode kinds registered here at DERIVE's request (vLLM GPT-J rotary); NOT part of the reference vocabulary digest —
#: they belong to the profile vocabulary (`b1.py` conventions, bf16 cache row, one rounding) and are proof-supported.
PROFILE_ADDITIONS = [RoPEHeadGptJ, RoPEGptJ]

#: `prims.py` / `b1.py` primitives a reference body may call (shared with the profile-mode vocabulary; pinned in the digest).
SHARED_PRIMITIVES = [P.Bf16ToF32, P.F32ToBf16Rn, P.F32Add, P.F32Mul, P.F32Fma, P.F32Div, P.Bf16Add, P.SelectF32, P.SelectBf16,
                     P.SelectI32, P.I32Add, P.Bf16GtStrict]
SHARED_COMPOSITES = [b1.ArgmaxStep]
SHARED_FAMILIES = {"GatherBf16x{V}": P.gather_bf16, "GatherF32x{V}": gather_f32}


# ---------------------------------------------------------------------------------------------------------
# vocabulary pin
# ---------------------------------------------------------------------------------------------------------


def _src_hash(obj) -> str:
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        src = repr(obj)
    return hashlib.sha256(src.encode()).hexdigest()


def ref_vocab_manifest() -> dict:
    """The reference vocabulary as data: version, module source hash, every reference kind with its body/evaluator
    source hash, the shared primitives with their evaluator source hashes, the on-demand families."""
    here = os.path.abspath(__file__)
    with open(here, "rb") as fh:
        module_sha = hashlib.sha256(fh.read()).hexdigest()
    kinds = {}
    for p in REF_PRIMITIVES:
        kinds[p.id] = {"kind": "primitive", "evaluator_sha256": _src_hash(p.evaluate),
                       "params": [[n, repr(t)] for n, t in p.params], "ret": repr(p.ret)}
    for c in REF_COMPOSITES:
        kinds[c.id] = {"kind": "composite", "statics": list(c.statics), "body_sha256": _src_hash(c.body),
                       "signature_sha256": _src_hash(c.signature)}
    shared = {p.id: {"kind": "primitive", "evaluator_sha256": _src_hash(p.evaluate)} for p in SHARED_PRIMITIVES}
    for c in SHARED_COMPOSITES:
        shared[c.id] = {"kind": "composite", "body_sha256": _src_hash(c.body)}
    families = {name: _src_hash(fn) for name, fn in SHARED_FAMILIES.items()}
    return {"schema": "verity-ir/ref-vocab/v0", "version": REF_VOCAB_VERSION, "module": "verity.ml.definitions.registry.ref_prims",
            "module_sha256": module_sha, "decimal_precision": _DEC_PREC, "kinds": kinds, "shared": shared, "families": families}


def ref_vocab_digest() -> str:
    """SHA-256 over the canonical JSON of `ref_vocab_manifest()` — what DERIVE records in a derived program's provenance."""
    import json
    m = ref_vocab_manifest()
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["REF_VOCAB_VERSION", "DTYPES", "REF_PRIMITIVES", "REF_COMPOSITES", "SHARED_PRIMITIVES", "ref_vocab_manifest",
           "ref_vocab_digest", "gather_f32", "F32Sub", "F32Sqrt", "F32MaxRef", "F32GtStrict", "F32ExpRn", "F32ErfRn", "F32TanhRn",
           "DotRef", "LinearRef", "RMSNormRef", "LayerNormRef", "GeluRefElem", "GeluRef", "SiluRefElem", "SiluRef", "SoftmaxRef",
           "RoPERef", "RoPEHeadsRef", "EagerAttentionHeadRef", "EagerAttentionRef", "EmbeddingRef", "AddRef", "MulRef",
           "ArgmaxStepF32", "ArgmaxRef", "f32_exp_rn_bits", "f32_erf_rn_bits", "f32_tanh_rn_bits"]
