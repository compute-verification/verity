"""[X-06] Stochastic token selection of vLLM d9105ea80's batch-invariant sampler with the per-request SEED, the POSITION and the
TEMPERATURE as OPERANDS (bound inputs of the Program) — seeds are inputs, not structure.

The launched chain for a request with temperature T (`vllm/v1/worker/gpu/sample/sampler.py::Sampler.sample`, no top-k/top-p,
no penalties / logit bias / min_p):

    logits32 = f32(logits_bf16)                                  # `torch.empty_like(logits, dtype=float32).copy_(logits)` (exact widening)
    _temperature_kernel:  logits32[v] = logits32[v] / T           # Triton fp32 `/` = `div.full.f32`; SKIPPED (early return) when T in {0, 1}
    _gumbel_sample_kernel (APPLY_TEMPERATURE=False, IS_DRAFTING=False, USE_FP64=False, PER_TOKEN_COL=False, no logits cache):
        key   = tl.randint(seed, pos)                              # Philox4x32-10 word 0, key (seed_lo, seed_hi), counter (pos_lo, pos_hi, 0, 0)
        u[v]  = max(tl.rand(key, v), 2^-31')                       # per vocabulary index
        g[v]  = -logf(-log1pf(-u[v]))                              # libdevice __nv_logf / __nv_log1pf, FTZ
        y[v]  = logits32[v] + g[v]     if T != 0  else  logits32[v]   (temperature 0: greedy, no noise)
        token = first index of max_v y[v]                          # per 1024-lane block `tl.max(return_indices=True)` (tie -> left),
                                                                   # then `torch.argmax(local_max)` (tie -> first): the FIRST global maximum

Definitions (this module; the exact fp32 helpers, Philox and the libdevice transcriptions are the ones validated bit-exact on the
RTX 4090 by the static-seed composites below, 64/64 G4 cases):

  * `TemperatureScale_v1{V}(logits: Array<V, bf16>, temp: f32) -> Array<V, f32>` — the widening + `_temperature_kernel` with its
    T in {0, 1} early return (`SelectF32`), `div.full.f32` as the registry's `F32Mul(DivFullScaleA(x, T), DivFullRcp(T))` model.
  * `GumbelSelectF32_v1{V}(logits: Array<V, f32>, seed: i64, pos: i64, temp: f32) -> i32` — `_gumbel_sample_kernel` + the block argmax
    over ALREADY PROCESSED fp32 logits (tempered; masked lanes are -inf and never win under the strict `>`): the slot a later
    `TopPMask_v1{V}` composes into.  `temp` enters only through `temp != 0` (noise or not).
  * `GumbelTokenSelect_v1{V}(logits: Array<V, bf16>, seed: i64, pos: i64, temp: f32) -> i32` — the two chained: the served token of a
    request WITHOUT top-k/top-p.  At temp == 0 it is the registered greedy `TokenSelect_v1{V}` (first maximum).
  * `GumbelArgmaxBf16_v1{V, SEED, POS}(logits: Array<V, bf16>) -> i32` and `GumbelArgmaxTempBf16_v1{V, SEED, POS, T}` (steps
    `GumbelArgmaxStep_v1`, `GumbelArgmaxTempStep_v1{T}`) — the same selection at temperature 1 and at a T not in {0, 1}, with
    seed, position and T as STATICS: the forms checked bit-exact against vLLM's kernels on the RTX 4090, which the sampled-row
    profiles bind (`engine/profiles/gen_ov_sampling_patterns.py`).  They came from the quarantine lane `ov-sampling`
    (`SEMANTIC_ID` "ov-sampling/GumbelArgmaxBf16@v1") with their bodies verbatim, so their ids are the lane's.

NOT defined here (fail-closed in Bet A's `SAMPLERS_BOUND`): top-p / top-k masking (`_topp_sb_{stats,step,mask}_kernel`,
`_topk_topp_kernel`) — see the lane's failure record F-dense-02 for what an exact model needs.
"""
from __future__ import annotations

import math
import struct

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.ops.scalar import BF16, BIT, BitNot, BitOr, F32, F32Eq, I32, I32Add, SelectF32, SelectI32
from verity_vllm.program.registry import ref_prims as RP
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.fp32 import F32AddFtz, F32Mul
from verity.ml.circuits.ops.mufu import DivFullRcp, DivFullScaleA
from verity.ml.circuits.ops.prims import Bf16ToF32, const

SEMANTIC_ID = "sampling/GumbelTokenSelect@v1"

I64 = Value(64)
U32 = Value(32)
M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# Triton `python/triton/language/random.py` (v3.7.1) constants
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
TL_RAND_SCALE_BITS = 0x2FFFFFFF          # fp32 4.6566127342e-10 = (2^23-1)/2^23 * 2^-31: `uint_to_uniform_float`
TL_RAND_MIN_BITS = 0x2FFFFFFF            # vLLM `_TL_RAND_MIN` (the same fp32 word)

# libdevice polynomial coefficients (fp32 words, Horner order), from libdevice.10.bc `__nv_logf` / `__nv_log1pf`
LOGF_POLY = (0xBE055027, 0x3E1039F6, 0xBDF8CDCC, 0x3E0F2955, 0xBE2AD8B9, 0x3E4CED0B, 0xBE7FFF22, 0x3EAAAA78)
LOG1PF_POLY = (0xBD39BF78, 0x3DD80012, 0xBE0778E0, 0x3E146475, 0xBE2A68DD, 0x3E4CAF9E, 0xBE800042, 0x3EAAAAE6)
LN2_BITS = 0x3F317218
TWO_M23_BITS = 0x34000000
TWO_P23_BITS = 0x4B000000
FLT_MIN_BITS = 0x00800000
ONE_BITS = 0x3F800000
#: `__nv_logf` / `__nv_log1pf` outside their domain (a < 0, a < -1, -inf): their special case's `fma(a, inf, inf)` is -inf + inf,
#: whose NaN is pinned here rather than left to the host's default NaN (x86 and aarch64 differ in the sign)
LOG_INVALID_NAN = 0xFFC00000
#: every NaN result of the `_v2` logs (`nv_logf_gpu`, `nv_log1pf_gpu`): the GPU's canonical NaN, for an input outside the domain and a
#: NaN input alike, no payload or sign kept (the L40S / H100 probe art:e5dd6ce9..., every build); the `_v1` logs keep `LOG_INVALID_NAN`
#: and a NaN input's payload
LOG_NAN = 0x7FFFFFFF


def _neg_not_nan(a: int) -> bool:
    return a >= 0x80000000 and (a & 0x7FFFFFFF) <= 0x7F800000


# ---------------------------------------------------------------------------------------------------------
# fp32 helpers on encodings (pure Python; every operation is a single correctly rounded IEEE step)
# ---------------------------------------------------------------------------------------------------------


def _f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


def _u(x: float) -> int:
    """RN-even fp32 word of the double `x` (struct rounds to nearest-even; overflow -> inf)."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def _ftz(u: int) -> int:
    u &= M32
    return u & 0x80000000 if (u & 0x7F800000) == 0 and (u & 0x7FFFFF) else u


def _round_exact(s: float, e: float) -> int:
    """fp32 word of the exact value s + e (s double, |e| <= ulp(s)/2 from TwoSum), RN-even."""
    r = _u(s)
    if e == 0.0 or math.isinf(s) or math.isnan(s):
        return r
    rf = _f(r)
    if rf == s:
        return r
    lo, hi = (math.nextafter(rf, -math.inf), rf) if rf > s else (rf, math.nextafter(rf, math.inf))
    if (lo + hi) * 0.5 == s:
        return _u(hi) if e > 0 else _u(lo)
    return r


def fma_ftz(a: int, b: int, c: int) -> int:
    """`fma.rn.ftz.f32` on words."""
    fa, fb, fc = _f(_ftz(a)), _f(_ftz(b)), _f(_ftz(c))
    p = fa * fb
    if math.isinf(p) or math.isnan(p) or math.isinf(fc) or math.isnan(fc):
        return _ftz(_u(p + fc))
    s = p + fc
    bb = s - p
    e = (p - (s - bb)) + (fc - bb)
    return _ftz(_round_exact(s, e))


def mul_ftz(a: int, b: int) -> int:
    return _ftz(_u(_f(_ftz(a)) * _f(_ftz(b))))


def add_ftz(a: int, b: int) -> int:
    fa, fb = _f(_ftz(a)), _f(_ftz(b))
    s = fa + fb
    if math.isinf(s) or math.isnan(s):
        return _ftz(_u(s))
    bb = s - fa
    e = (fa - (s - bb)) + (fb - bb)
    return _ftz(_round_exact(s, e))


def add_rz_ftz(a: int, b: int) -> int:
    """`add.rz.ftz.f32`: a + b rounded TOWARD ZERO in fp32 (libdevice `__fadd_rz`) -- the v2 evaluator (F-stoch-r17-01 fix).

    The exact sum is s + e (TwoSum in double: exact for fp32 operands); r = RN32(s) is stepped ONE fp32 ulp toward zero when its
    magnitude exceeds the exact value (RN rounded away from zero, or RN landed on s while e pulls the exact value inside it); RZ never
    overflows to inf from a finite sum (it saturates at FLT_MAX); a zero sum keeps its sign.  The same rule the GPU-validated numpy
    transcription `verity_vllm/program/kernels/libdevice_sampling.py::add_rz_f32` applies (np.nextafter on float32 = one f32 ulp)."""
    fa, fb = _f(_ftz(a)), _f(_ftz(b))
    s = fa + fb
    if s == 0.0 or math.isnan(s):
        return _ftz(_u(s))
    if math.isinf(s):                            # only when an operand is infinite: fp32 operands cannot overflow a double sum
        return _ftz(_u(s))
    bb = s - fa
    e = (fa - (s - bb)) + (fb - bb)              # exact = s + e
    r = _u(s)
    rf = _f(r)
    if math.isinf(rf):                           # RN32 overflowed a finite sum: toward zero saturates at the largest finite word
        return (r & 0x80000000) | 0x7F7FFFFF
    if abs(rf) > abs(s) or (rf == s and e != 0.0 and (e < 0) == (s > 0)):
        if r & 0x7FFFFFFF:
            r = (r & 0x80000000) | ((r & 0x7FFFFFFF) - 1)      # one fp32 ulp toward zero
    return _ftz(r)


def _i32(u: int) -> int:
    u &= M32
    return u - (1 << 32) if u & 0x80000000 else u


def nv_logf(a: int) -> int:
    """libdevice `__nv_logf` on an fp32 word (FTZ)."""
    a &= M32
    small = _f(a) < _f(FLT_MIN_BITS)
    a2 = mul_ftz(a, TWO_P23_BITS) if small else a
    i0 = 0xC1B80000 if small else 0
    ai = _i32(a2)
    e = (ai - 1059760811) & -8388608
    e_i32 = _i32(e & M32)
    m = (ai - e_i32) & M32
    i2 = fma_ftz(_u(float(e_i32)), TWO_M23_BITS, i0)
    f = add_ftz(m, 0xBF800000)
    r = fma_ftz(LOGF_POLY[0], f, LOGF_POLY[1])
    for k in LOGF_POLY[2:]:
        r = fma_ftz(r, f, k)
    r = fma_ftz(r, f, 0xBF000000)
    r = mul_ftz(r, f)
    r = fma_ftz(r, f, f)
    r = fma_ftz(i2, LN2_BITS, r)
    if a2 >= 0x7F800000:
        r = LOG_INVALID_NAN if _neg_not_nan(a2) else fma_ftz(a2, 0x7F800000, 0x7F800000)
    if (a2 & 0x7FFFFFFF) == 0:
        r = 0xFF800000
    return r


def nv_log1pf(a: int) -> int:
    """libdevice `__nv_log1pf` on an fp32 word (FTZ): `__fadd_rz(a, 1.0f)` is a TRUE toward-zero add (`add_rz_ftz`; F-stoch-r17-01 fix),
    then the binade selector and the polynomial (`_log1pf_from_sum`).  The sum feeds only the selector, so the fix moves exactly one
    word of the whole fp32 domain (a = 0x3EFFFFFF, positive) and none in the Gumbel lane's domain a = -u, u in [2^-31', 1):
    `GumbelNoiseLane_v1` is the same function it was (exhaustive, test_sampling_rows)."""
    return _log1pf_from_sum(a & M32, add_rz_ftz(a & M32, ONE_BITS))


def _gpu_nan(u: int) -> int:
    return LOG_NAN if (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) else u


def nv_logf_gpu(a: int) -> int:
    """`nv_logf` with every NaN result `LOG_NAN`, the GPU's (`NvLogf_v2`); every other word is `nv_logf`'s."""
    return _gpu_nan(nv_logf(a))


def nv_log1pf_gpu(a: int) -> int:
    """`nv_log1pf` with every NaN result `LOG_NAN`, the GPU's; every other word is `nv_log1pf`'s."""
    return _gpu_nan(nv_log1pf(a))


def _log1pf_from_sum(a: int, s: int) -> int:
    """The body of libdevice `__nv_log1pf` after its first instruction: `s` = the `__fadd_rz(a, 1.0f)` word (v1 model or true RZ)."""
    si = _i32(s)
    e = (si - 1061158912) & -8388608
    e_i32 = _i32(e & M32)
    m1 = (_i32(a) - e_i32) & M32
    t = (1082130432 - e_i32) & M32
    r0 = fma_ftz(0x3E800000, t, 0xBF800000)
    u = add_ftz(m1, r0)
    ef = mul_ftz(_u(float(e_i32)), TWO_M23_BITS)
    r = fma_ftz(LOG1PF_POLY[0], u, LOG1PF_POLY[1])
    for k in LOG1PF_POLY[2:]:
        r = fma_ftz(r, u, k)
    r = fma_ftz(r, u, 0xBF000000)
    r = mul_ftz(r, u)
    r = fma_ftz(r, u, u)
    r = fma_ftz(ef, LN2_BITS, r)
    if a >= 0x7F800000:
        if _i32(a) > -1082130432:
            r = LOG_INVALID_NAN if _neg_not_nan(a) else fma_ftz(a, 0x7F800000, 0x7F800000)
        if (a & 0x7FFFFFFF) == 0:
            r = 0x80000000
    return r


# ---------------------------------------------------------------------------------------------------------
# Philox4x32-10 and Triton's rand
# ---------------------------------------------------------------------------------------------------------


def philox4x32_10_word0(c0: int, c1: int, c2: int, c3: int, k0: int, k1: int) -> int:
    """First output word of Triton's `philox_impl` (Random123 Philox4x32-10) on uint32 state."""
    for _ in range(10):
        pb = c2 * PHILOX_ROUND_B
        pa = c0 * PHILOX_ROUND_A
        c0, c1, c2, c3 = (((pb >> 32) & M32) ^ c1 ^ k0), pb & M32, (((pa >> 32) & M32) ^ c3 ^ k1), pa & M32
        k0 = (k0 + PHILOX_KEY_A) & M32
        k1 = (k1 + PHILOX_KEY_B) & M32
    return c0


def stream_key(seed: int, pos: int) -> int:
    """`tl.randint(seed, pos)` with an int64 seed and int64 offset: key = (seed_lo, seed_hi), counter = (pos_lo, pos_hi, 0, 0)."""
    seed &= M64
    pos &= M64
    return philox4x32_10_word0(pos & M32, (pos >> 32) & M32, 0, 0, seed & M32, (seed >> 32) & M32)


def _rand_lane(key: int, v: int) -> int:
    """fp32 word of max(tl.rand(key, v), 2^-31'): Philox word 0 -> `uint_to_uniform_float` -> the vLLM floor."""
    w = philox4x32_10_word0(v & M32, 0, 0, 0, key & M32, 0)
    xi = _i32(w)
    if xi < 0:
        xi = -xi - 1
    u = mul_ftz(_u(float(xi)), TL_RAND_SCALE_BITS)
    if _f(u) < _f(TL_RAND_MIN_BITS):
        u = TL_RAND_MIN_BITS
    return u


def noise_lane(key: int, v: int) -> int:
    """fp32 word of g(key)[v] = -logf(-log1pf(-max(rand(key, v), 2^-31'))) -- `GumbelNoiseLane_v1` (libdevice bodies; log1pf's
    `__fadd_rz` a true toward-zero add since F-stoch-r17-01, which changed no word of this function: see nv_log1pf)."""
    u = _rand_lane(key, v)
    l1 = nv_log1pf(u ^ 0x80000000)
    return nv_logf(l1 ^ 0x80000000) ^ 0x80000000


def noise_lane_gpu(key: int, v: int) -> int:
    """`noise_lane` through the GPU-NaN logs (`GumbelNoiseLane_v2`).  The same word on every lane: log1pf sees -u in (-1, 0) and logf a
    positive word, so neither reaches a NaN result."""
    u = _rand_lane(key, v)
    l1 = nv_log1pf_gpu(u ^ 0x80000000)
    return nv_logf_gpu(l1 ^ 0x80000000) ^ 0x80000000


# ---------------------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------------------

GumbelStreamKey = PrimitiveDefinition(
    "GumbelStreamKey", 1, (("seed", I64), ("pos", I64)), U32, lambda seed, pos: stream_key(seed, pos),
    doc="`tl.randint(seed, pos)`: Philox4x32-10 word 0 with key (seed_lo, seed_hi) and counter (pos_lo, pos_hi, 0, 0); "
        "the per-(request seed, position) stream key of vLLM's Gumbel sampler.", conformance="exact-model-tested")

GumbelNoiseLane = PrimitiveDefinition(
    "GumbelNoiseLane", 1, (("key", U32), ("v", I32)), F32, lambda key, v: noise_lane(key, v),
    doc="g[v] = -__nv_logf(-__nv_log1pf(-max(tl.rand(key, v), 2^-31'))) in fp32 (FTZ): the Gumbel noise the kernel adds to "
        "vocabulary index v; libdevice bodies transcribed from libdevice.10.bc (log1pf's __fadd_rz a true toward-zero add, "
        "F-stoch-r17-01: the word map of this primitive is unchanged by it, exhaustively).",
    conformance="exact-model-tested")

GumbelNoiseLaneV2 = PrimitiveDefinition(
    "GumbelNoiseLane", 2, (("key", U32), ("v", I32)), F32, lambda key, v: noise_lane_gpu(key, v),
    doc="`GumbelNoiseLane_v1` through `NvLogf_v2`'s logs (every NaN result of __nv_logf / __nv_log1pf the GPU's 0x7FFFFFFF): the same "
        "word on every lane, since the noise never reaches those results; for the re-baseline that moves the logs to `_v2`.",
    conformance="exact-model-tested")


# ---------------------------------------------------------------------------------------------------------
# composites — operands, not statics, for seed / pos / temperature
# ---------------------------------------------------------------------------------------------------------


@composite("TemperatureLane", 1, [], lambda S: ((("x", BF16), ("temp", F32), ("rcp", F32), ("skip", BIT)), F32),
           conformance="exact-model-tested")
def TemperatureLane(B, S, x, temp, rcp, skip):
    """One lane of `_temperature_kernel` on the fp32 widening of a bf16 logit: `div.full.f32(f32(x), T)` =
    F32Mul(DivFullScaleA(f32(x), T), DivFullRcp(T)); the widening alone when the kernel early-returns (T in {0, 1})."""
    xf = B.call(Bf16ToF32, x)
    q = B.call(F32Mul, B.call(DivFullScaleA, xf, temp), rcp)
    return B.call(SelectF32, skip, xf, q)


@composite("TemperatureScale", 1, ["V"], lambda S: ((("logits", Array(S.V, BF16)), ("temp", F32)), Array(S.V, F32)),
           conformance="exact-model-tested")
def TemperatureScale(B, S, logits, temp):
    """`Sampler.apply_sampling_params`: `logits32 = f32(logits)`; `_temperature_kernel`: `logits32 /= T` in place unless T in {0, 1}
    (the kernel's early return).  Triton fp32 `/` is `div.full.f32` (registry model, bit-exact on the B1 temp07 capture)."""
    rcp = B.call(DivFullRcp, temp)
    skip = B.call(BitOr, B.call(F32Eq, temp, B.call(const(32, 0))), B.call(F32Eq, temp, B.call(const(32, ONE_BITS))))
    return B.batch(TemperatureLane, logits, temp, rcp, skip, axes=(0, None, None, None))


@composite("GumbelSelectStep", 1, [], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", F32), ("key", U32), ("noisy", BIT)),
                                                  Tuple(Tuple(F32, I32, I32), I32)), conformance="exact-model-tested")
def GumbelSelectStep(B, S, carry, x, key, noisy):
    """One vocabulary lane: y = x + g(key)[i] (add.ftz.f32) when noisy (temp != 0) else x; take (y, i) iff y > best (strict: the
    first maximum wins; NaN never wins); i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    g = B.call(GumbelNoiseLane, key, i)
    y = B.call(SelectF32, noisy, B.call(F32AddFtz, x, g), x)
    gt = B.call(RP.F32GtStrict, y, best)
    nb = B.call(SelectF32, gt, y, best)
    ni = B.call(SelectI32, gt, i, best_i)
    nxt = B.call(I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("GumbelSelectF32", 1, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
           conformance="exact-model-tested")
def GumbelSelectF32(B, S, logits, seed, pos, temp):
    """`_gumbel_sample_kernel` (APPLY_TEMPERATURE=False, fp32) + `torch.argmax(local_max)` / `gather(local_argmax)` over already
    processed fp32 logits: the first index of max_v (logits[v] + g(seed, pos)[v]) — or of max_v logits[v] when temp == 0 (no noise).
    Masked lanes (-inf) never win.  seed / pos / temp are OPERANDS: the per-request `SamplingParams.seed` (int64), the absolute
    position of the sampled token, the request temperature."""
    key = B.call(GumbelStreamKey, seed, pos)
    noisy = B.call(BitNot, B.call(F32Eq, temp, B.call(const(32, 0))))
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(GumbelSelectStep, init, xs=(logits[1:S.V],), shared=(key, noisy))
    return step[0][1]


@composite("GumbelTokenSelect", 1, ["V"], lambda S: ((("logits", Array(S.V, BF16)), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
           conformance="exact-model-tested")
def GumbelTokenSelect(B, S, logits, seed, pos, temp):
    """The served token of a request WITHOUT top-k / top-p: `GumbelSelectF32(TemperatureScale(logits, temp), seed, pos, temp)` —
    exactly the kernel pair `_temperature_kernel` -> `_gumbel_sample_kernel` on the fp32 widening of the bf16 logits row.
    temp == 0 is the greedy `TokenSelect_v1{V}` (first maximum, no noise, no scaling)."""
    scaled = B.call(bind(TemperatureScale, V=S.V), logits, temp)
    return B.call(bind(GumbelSelectF32, V=S.V), scaled, seed, pos, temp)


# ---------------------------------------------------------------------------------------------------------
# [R15 stoch] top-p: `apply_top_k_top_p_triton` split pipeline (`_topp_sb_{stats,step,mask}_kernel`, batch <= 64, no top-k)
# ---------------------------------------------------------------------------------------------------------
#
# The pipeline masks a row IN PLACE to -inf outside the nucleus; its keep predicate per lane is a function of the whole fp32 row,
# p, and S (the split count = f(sampler batch size, SM count), `topp_geometry.splits_for`).  A primitive returns exactly one Value and
# a per-lane Array-input primitive would cost V^2 operand references, so the row-level result is ONE `Value<V>` keep word
# (`TopPMaskWordx{V}`, reference = `topp_split.topp_keep_row`, bit-exact on 63 captured + 368 random rows over 11 V and on the R16 GPU
# fixture at every S) and `TopPMask_v1{V}` applies it lane by lane (`SelectF32(BitAt(keep, i), x, -inf)`) so the masked row IS an
# Array Program value the Gumbel select composes into.
#
# [R16 stoch, coord M-0169 / global M-0189 (1)] S is an OPERAND (i32), not a static: the split count is the launch geometry
# S_t = SplitsFor_v1(|live(t)|, num_SMs) (`topp_geometry.splits_for_v1`), a WORKLOAD-level derived value -- the request Program reads it
# from its request-owned `splits` Input per sampling event, the fold carries the split kernels' constexpr S as a `Const32[S]` operand,
# and the Match checks the relation per engine step (GM-01 G6 'sampler geometry').  The keep word is TOTAL over the i32 operand:
# an S outside {1, 2, 4, 8, 16, 32} keeps no lane (`topp_keep`), a word no launched S produces; it never defaults to a split count.

_TOPP_WORDS: dict = {}
_BIT_ATS: dict = {}
NEG_INF_BITS = 0xFF800000
SPLIT_COUNTS = (1, 2, 4, 8, 16, 32)


def _splits_of(word: int) -> int | None:
    """The i32 `splits` operand as the split count; None when it is not a power of two <= 32 (no launch of the pipeline has it)."""
    S = int(word) & M32
    return S if S in SPLIT_COUNTS else None


def topp_keep(x_f32, p: float, splits: int):
    """The keep bits of `TopPMaskWordx{V}` for every i32 `splits`: `topp_split.topp_keep_row(x, p, S)` at a split count, and no lane
    at any other value.  The one statement of the keep word's semantics (the primitive, `sampling_rows.topp_mask_row`, C-Flock's
    native twin in `verity_flock.ir_sampling`).

    Defined on every row.  The split kernels hold the row's scalars per warp half, and on a row where the two halves disagree on
    stopping (the uniform-or-empty test, or a `nomask` search outcome) every lane follows its own half: a lane of a half that stops
    keeps its logit, and a lane of a half that goes on is masked by that half's pivot, dup and quota (`topp_split.mask_kernel`).
    That is each GPU thread's own early return, and the kernels' result wherever no collective follows the divergent return; the
    same reading defines the rest."""
    import numpy as np
    from verity_vllm.program.registry import topp_split as _TS
    S = _splits_of(splits)
    if S is None:
        return np.zeros(len(x_f32), dtype=bool)
    return np.asarray(_TS.topp_keep_row(x_f32, p, S), dtype=bool).reshape(-1)


def topp_mask_word(vocab: int) -> PrimitiveDefinition:
    """`TopPMaskWordx{V}`: (logits: Array<V, f32>, top_p: f32, splits: i32) -> Value<V>; bit v = the split top-p pipeline's `keep` for
    lane v under S = splits (all ones when the pipeline leaves the row untouched: p >= 1, (nearly) uniform / empty row, `nomask` search
    outcome; ones on its lanes when only one warp half stops, `topp_keep`).  `splits` is the launch's split count as an OPERAND (`SplitsFor_v1(|live|, num_SMs)`: 32 at one live request on any GPU
    with >= 64 SMs, 16 at 8 live on 142 SMs).  The word is defined for every i32 `splits`: any value outside {1, 2, 4, 8, 16, 32}
    (0, negative, not a power of two, above 32) keeps no lane -- word 0.  Every launched split count keeps at least one lane, so word 0
    is a value no served S produces (a wrong S is not mapped onto a plausible one), and the served S is held to the workload outside
    this word: the Match's sampler geometry (GM-01 G6: S == SplitsFor_v1(|live(t)|, num_SMs) per engine step) and the Commit's
    prescribed-input linkage (the committed `splits` == SplitsFor_v1(committed live count, num_SMs of record)), both of which yield only
    the six split counts."""
    key = int(vocab)
    prim = _TOPP_WORDS.get(key)
    if prim is None:
        import numpy as np

        def ev(*ops, _V=int(vocab)) -> int:
            x = np.array(ops[:_V], dtype=np.uint32).view(np.float32)
            keep = topp_keep(x, _f(ops[_V]), ops[_V + 1])
            return int.from_bytes(np.packbits(keep, bitorder="little").tobytes(), "little")

        prim = PrimitiveDefinition(f"TopPMaskWordx{vocab}", 1, (("logits", Array(int(vocab), F32)), ("top_p", F32), ("splits", I32)),
                                   Value(int(vocab)), ev, doc=topp_mask_word.__doc__ or "", conformance="exact-model-tested")
        _TOPP_WORDS[key] = prim
    return prim


def bit_at(width: int) -> PrimitiveDefinition:
    """`BitAtx{W}`: (word: Value<W>, i: i32) -> bit = (word >> i) & 1 (i out of range -> 0)."""
    prim = _BIT_ATS.get(int(width))
    if prim is None:
        def ev(word: int, i: int, _W=int(width)) -> int:
            i = i - (1 << 32) if i & 0x80000000 else i
            return (word >> i) & 1 if 0 <= i < _W else 0

        prim = PrimitiveDefinition(f"BitAtx{width}", 1, (("word", Value(int(width))), ("i", I32)), BIT, ev,
                                   doc=bit_at.__doc__ or "", conformance="exact-model-tested")
        _BIT_ATS[int(width)] = prim
    return prim


@composite("TopPMaskStep", 1, ["V"], lambda S: ((("i", I32), ("x", F32), ("keep", Value(S.V))), Tuple(I32, F32)),
           conformance="exact-model-tested")
def TopPMaskStep(B, S, i, x, keep):
    """One lane of `_topp_sb_mask_kernel`'s store: `tl.where(keep, x, -inf)`; i += 1."""
    y = B.call(SelectF32, B.call(bit_at(S.V), keep, i), x, B.call(const(32, NEG_INF_BITS)))
    return tuple_of(B.call(I32Add, i, B.call(const(32, 1))), y)


@composite("TopPMask", 1, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("top_p", F32), ("splits", I32)), Array(S.V, F32)),
           conformance="exact-model-tested")
def TopPMask(B, S, logits, top_p, splits):
    """`apply_top_k_top_p_triton` for a row of a batch <= 64 without top-k (`_apply_topp_split`, `splits` splits): the fp32 row with
    every lane outside the top-p nucleus set to -inf, exactly as the three Triton kernels compute it (registry/topp_split.py).  `splits`
    (i32) is the launch's split count S_t = SplitsFor_v1(|live(t)|, num_SMs), an OPERAND of the event (a derived value of the workload)."""
    keep = B.call(topp_mask_word(S.V), logits, top_p, splits)
    res = B.scan(bind(TopPMaskStep, V=S.V), B.call(const(32, 0)), xs=(logits,), shared=(keep,))
    return res[1]


@composite("GumbelTopPTokenSelect", 1, ["V"],
           lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelect(B, S, logits, top_p, seed, pos, temp, splits):
    """The served token of a request with temperature + top-p (no top-k, no penalties / logit bias / min_p), vLLM d9105ea80
    `Sampler.sample`: `_temperature_kernel` -> `apply_top_k_top_p_triton` (split pipeline, `splits` splits) -> `_gumbel_sample_kernel`
    + argmax, on the fp32 widening of the bf16 logits row.  temp / top_p are the request's SamplingParams (structural inputs of the
    row's declared policy, bound as operands); seed (int64) and pos are OPERANDS; `splits` (i32, the 6th operand) is the launch
    geometry S_t = SplitsFor_v1(|live(t)|, num_SMs) -- a derived value of the workload the request Program reads from its `splits`
    Input per sampling event and the fold carries as the split kernels' constexpr (`Const32[S]`)."""
    scaled = B.call(bind(TemperatureScale, V=S.V), logits, temp)
    masked = B.call(bind(TopPMask, V=S.V), scaled, top_p, splits)
    return B.call(bind(GumbelSelectF32, V=S.V), masked, seed, pos, temp)


# ---------------------------------------------------------------------------------------------------------
# the same two selects with `temp == 0` computed once and read by both halves (`TargetProfile.sampler_construction`, `select_for`).
# `GumbelTokenSelect_v1` / `GumbelTopPTokenSelect_v1` compute it in `TemperatureScale_v1` (the early return) and again in
# `GumbelSelectF32_v1` (noise or not): one value computed twice inside the sampler's one unit, 1 redundant gate per Call.
# ---------------------------------------------------------------------------------------------------------


@composite("TemperatureScaleGivenGreedy", 1, ["V"], lambda S: ((("logits", Array(S.V, BF16)), ("temp", F32), ("greedy", BIT)), Array(S.V, F32)),
           conformance="exact-model-tested")
def TemperatureScaleGivenGreedy(B, S, logits, temp, greedy):
    """`TemperatureScale_v1{V}` with `greedy` = (temp == 0) read, not computed: the early return is greedy | (temp == 1)."""
    rcp = B.call(DivFullRcp, temp)
    skip = B.call(BitOr, greedy, B.call(F32Eq, temp, B.call(const(32, ONE_BITS))))
    return B.batch(TemperatureLane, logits, temp, rcp, skip, axes=(0, None, None, None))


@composite("GumbelSelectF32GivenGreedy", 1, ["V"], lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("greedy", BIT)), I32),
           conformance="exact-model-tested")
def GumbelSelectF32GivenGreedy(B, S, logits, seed, pos, greedy):
    """`GumbelSelectF32_v1{V}` with `greedy` = (temp == 0) read, not computed: noise iff not greedy."""
    key = B.call(GumbelStreamKey, seed, pos)
    noisy = B.call(BitNot, greedy)
    zero = B.call(const(32, 0))
    y0 = B.call(SelectF32, noisy, B.call(F32AddFtz, logits[0], B.call(GumbelNoiseLane, key, zero)), logits[0])
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(GumbelSelectStep, init, xs=(logits[1:S.V],), shared=(key, noisy))
    return step[0][1]


@composite("GumbelTokenSelectSharedGreedy", 1, ["V"],
           lambda S: ((("logits", Array(S.V, BF16)), ("seed", I64), ("pos", I64), ("temp", F32)), I32), conformance="exact-model-tested")
def GumbelTokenSelectSharedGreedy(B, S, logits, seed, pos, temp):
    """`GumbelTokenSelect_v1{V}`'s function and signature, `temp == 0` computed once."""
    greedy = B.call(F32Eq, temp, B.call(const(32, 0)))
    scaled = B.call(bind(TemperatureScaleGivenGreedy, V=S.V), logits, temp, greedy)
    return B.call(bind(GumbelSelectF32GivenGreedy, V=S.V), scaled, seed, pos, greedy)


@composite("GumbelTopPTokenSelectSharedGreedy", 1, ["V"],
           lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32), ("splits", I32)), I32),
           conformance="exact-model-tested")
def GumbelTopPTokenSelectSharedGreedy(B, S, logits, top_p, seed, pos, temp, splits):
    """`GumbelTopPTokenSelect_v1{V}`'s function and signature, `temp == 0` computed once."""
    greedy = B.call(F32Eq, temp, B.call(const(32, 0)))
    scaled = B.call(bind(TemperatureScaleGivenGreedy, V=S.V), logits, temp, greedy)
    masked = B.call(bind(TopPMask, V=S.V), scaled, top_p, splits)
    return B.call(bind(GumbelSelectF32GivenGreedy, V=S.V), masked, seed, pos, greedy)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@composite("GumbelArgmaxStep", 1, [], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", BF16), ("key", U32)),
                                                  Tuple(Tuple(F32, I32, I32), I32)), conformance="model-untested")
def GumbelArgmaxStep(B, S, carry, x, key):
    """One vocabulary lane: y = f32(x) + g(key)[i] (add.ftz.f32); take (y, i) iff y > best (strict: first index wins,
    NaN never wins); i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    g = B.call(GumbelNoiseLane, key, i)
    y = B.call(F32AddFtz, B.call(Bf16ToF32, x), g)
    gt = B.call(RP.F32GtStrict, y, best)
    nb = B.call(SelectF32, gt, y, best)
    ni = B.call(SelectI32, gt, i, best_i)
    nxt = B.call(I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("GumbelArgmaxBf16", 1, ["V", "SEED", "POS"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def GumbelArgmaxBf16(B, S, logits):
    """Gumbel-max token selection at temperature 1: argmax_v(f32(logits[v]) + g(SEED, POS)[v]), lowest index on ties."""
    key = B.call(GumbelStreamKey, B.call(const(64, int(S.SEED) & M64)), B.call(const(64, int(S.POS) & M64)))
    zero = B.call(const(32, 0))
    y0 = B.call(F32AddFtz, B.call(Bf16ToF32, logits[0]), B.call(GumbelNoiseLane, key, zero))
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(GumbelArgmaxStep, init, xs=(logits[1:S.V],), shared=(key,))
    return step[0][1]


def _tempered(B, x, t_bits: int, rcp):
    """`_temperature_kernel`: `f32(x) / T` with Triton's fp32 `/` = `div.full.f32` (registry model, b1.RMSNormTriton):
    F32Mul(DivFullScaleA(f32(x), T), DivFullRcp(T)).  Verified bit-exact on 16 x 151936 lanes of the B1 temp07 capture
    (REPORT.md: IEEE division reproduces only ~90.7% of the lanes)."""
    xf = B.call(Bf16ToF32, x)
    return B.call(F32Mul, B.call(DivFullScaleA, xf, B.call(const(32, t_bits))), rcp)


@composite("GumbelArgmaxTempStep", 1, ["T"], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", BF16), ("key", U32), ("rcp", F32)),
                                                          Tuple(Tuple(F32, I32, I32), I32)), conformance="model-untested")
def GumbelArgmaxTempStep(B, S, carry, x, key, rcp):
    """One vocabulary lane at temperature T: y = div.full(f32(x), T) + g(key)[i]; take (y, i) iff y > best; i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    g = B.call(GumbelNoiseLane, key, i)
    y = B.call(F32AddFtz, _tempered(B, x, int(S.T) & M32, rcp), g)
    gt = B.call(RP.F32GtStrict, y, best)
    nb = B.call(SelectF32, gt, y, best)
    ni = B.call(SelectI32, gt, i, best_i)
    nxt = B.call(I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("GumbelArgmaxTempBf16", 1, ["V", "SEED", "POS", "T"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def GumbelArgmaxTempBf16(B, S, logits):
    """Gumbel-max token selection at temperature T (static, fp32 bits, T not in {0, 1}):
    argmax_v(div.full(f32(logits[v]), T) + g(SEED, POS)[v]), lowest index on ties.  vLLM: `_temperature_kernel` (in place,
    div.full.f32) then `_gumbel_sample_kernel` (APPLY_TEMPERATURE=False)."""
    t_bits = int(S.T) & M32
    key = B.call(GumbelStreamKey, B.call(const(64, int(S.SEED) & M64)), B.call(const(64, int(S.POS) & M64)))
    rcp = B.call(DivFullRcp, B.call(const(32, t_bits)))
    zero = B.call(const(32, 0))
    y0 = B.call(F32AddFtz, _tempered(B, logits[0], t_bits, rcp), B.call(GumbelNoiseLane, key, zero))
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(bind(GumbelArgmaxTempStep, T=t_bits), init, xs=(logits[1:S.V],), shared=(key, rcp))
    return step[0][1]


__all__ = ["SEMANTIC_ID", "GumbelStreamKey", "GumbelNoiseLane", "F32Eq", "BitOr", "BitNot", "TemperatureLane", "TemperatureScale",
           "GumbelSelectStep", "GumbelSelectF32", "GumbelTokenSelect", "stream_key", "noise_lane", "nv_logf", "nv_log1pf", "fma_ftz",
           "add_rz_ftz", "f32_bits", "topp_mask_word", "bit_at", "TopPMaskStep", "TopPMask", "GumbelTopPTokenSelect", "NEG_INF_BITS",
           "SPLIT_COUNTS", "LOG_NAN", "GumbelNoiseLaneV2", "nv_logf_gpu", "nv_log1pf_gpu", "noise_lane_gpu",
           # the static-seed forms
           "GumbelArgmaxStep", "GumbelArgmaxBf16", "GumbelArgmaxTempStep", "GumbelArgmaxTempBf16"]
