//! The `veritor.reduce-mufu@1` gates: exact FP32 scalar arithmetic and the
//! measured MUFU table lookups of the vLLM reduction kernels (lane
//! PROOF-REDUCE), bit for bit as `veritor/core/reduce_gates.py` pins them.
//!
//! Everything is integer arithmetic on the raw encodings, in the style of
//! `crate::tc`: no `f32` operation of the target is ever mapped to a host
//! float instruction, so the guest (rv32im, no FPU) and every host agree on
//! every word.
//!
//! The arithmetic gates mirror the registered primitives of
//! `verity_ir/registry/prims.py`:
//!
//! * `f32_add`, `f32_mul`, `f32_div` -- IEEE-754 binary32, round to nearest
//!   even, subnormals kept (`FADD`/`FMUL`/`div.rn.f32`).  On finite and
//!   single-NaN operands they equal numpy's x86 words exactly; the pinned
//!   NaN rule is SSE's (the first NaN operand quieted; an invalid operation
//!   -- `inf - inf`, `0 * inf`, `0 / 0`, `inf / inf` -- gives the indefinite
//!   QNaN `0xFFC00000`), which is what the registered primitives compute on
//!   the x86-64 hosts the conformance evidence was measured on.
//! * `f32_fma` -- `FFMA`: one rounding of the exact `a*b + c`, every NaN
//!   canonicalised to `0x7FC00000` (`prims.f32_fma_bits`, which defines the
//!   NaN word rather than inheriting a platform rule).
//! * `bf16_to_f32`, `f2fp_bf16`, `bf16_add_f2fp`, `rope_out`, `rope_out_add`,
//!   `bf16_gt`, the selects and the i32 gates -- direct transcriptions of the
//!   primitives of the same names.
//!
//! The MUFU gates take the hardware table word as an *argument*: a 64-bit
//! `(index << 32) | word` leaf opened from a committed table tree
//! (`zk/bench/reduce_tables.py` builds the trees over the measured tables of
//! `verity_vllm.program.numerics`; the statement pins the roots).  Each gate
//! recomputes the index the hardware would use from its FP32 operand and
//! returns `None` -- a rejection -- when the opened leaf is for any other
//! index, so a proof can only use the one true table row.  The evaluation
//! around the lookup (special cases, exponent reconstruction, `div.full.f32`
//! range scaling) is `vllm-poc/verity_vllm/program/numerics/cpp/rms_triton_model.cpp`
//! (`mufu_rcp`, `mufu_sqrt`, `mufu_rsq`, `div_full_parts`, `rsqrt_nonftz`),
//! measured exhaustively on sm_89 and confirmed on the A100 for every
//! captured word (docs/vllm-poc/rmsnorm-model.md).

/// The canonical quiet-NaN FP32 word of the registered FMA (`prims.F32_NAN`).
pub const F32_NAN: u64 = 0x7FC0_0000;
/// SSE's "indefinite" QNaN: the word an invalid `addss`/`mulss`/`divss` makes.
pub const F32_INDEFINITE: u64 = 0xFFC0_0000;
/// `prims.BF16_NAN_TORCH`: the aten / torch canonical BF16 NaN.
pub const BF16_NAN_TORCH: u64 = 0x7FC0;
/// `prims.BF16_NAN_F2FP`: the SASS `F2FP.BF16.F32` canonical BF16 NaN.
pub const BF16_NAN_F2FP: u64 = 0x7FFF;

#[inline]
fn exp_field(u: u64) -> u64 {
    (u >> 23) & 0xFF
}

#[inline]
fn is_nan(u: u64) -> bool {
    exp_field(u) == 0xFF && (u & 0x7F_FFFF) != 0
}

#[inline]
fn is_inf(u: u64) -> bool {
    exp_field(u) == 0xFF && (u & 0x7F_FFFF) == 0
}

#[inline]
fn quiet(u: u64) -> u64 {
    u | 0x40_0000
}

/// The finite value of `u` as `(negative, magnitude, exponent)` with value
/// `(-1)^negative * magnitude * 2^exponent`; `magnitude` is `0` iff the value
/// is a (signed) zero.  `u` must not have exponent field `0xFF`.
#[inline]
fn unpack_finite(u: u64) -> (bool, u64, i32) {
    let negative = (u >> 31) & 1 == 1;
    let e = exp_field(u);
    let m = u & 0x7F_FFFF;
    if e == 0 {
        (negative, m, -149)
    } else {
        (negative, m | 0x80_0000, e as i32 - 150)
    }
}

/// Round the exact value `(-1)^sign * magnitude * 2^exponent` to FP32,
/// nearest-even, subnormals kept, overflow to the signed infinity.  The
/// magnitude is exact: every discarded bit participates as round/sticky.
/// `magnitude == 0` must be handled by the caller (zero-sign rules differ
/// per operation).
fn round_pack(negative: bool, magnitude: u128, exponent: i32) -> u64 {
    debug_assert!(magnitude != 0);
    let sign = (negative as u64) << 31;
    let bits = 128 - magnitude.leading_zeros() as i32;
    // value = magnitude * 2^exponent, top bit at 2^(exponent + bits - 1)
    let mut result_exp = exponent + bits - 1; // unbiased exponent of the leading bit
    // keep 24 significand bits: shift so the leading bit lands at position 23
    let mut shift = bits - 24;
    if result_exp < -126 {
        // subnormal: the significand grid is fixed at 2^-149
        shift += -126 - result_exp;
        result_exp = -126;
    }
    let (mut kept, round, sticky) = if shift <= 0 {
        ((magnitude << (-shift) as u32) as u128, false, false)
    } else if shift as u32 >= 128 {
        (0u128, false, true)
    } else {
        let s = shift as u32;
        let kept = magnitude >> s;
        let round = (magnitude >> (s - 1)) & 1 == 1;
        let below = if s >= 2 { magnitude & ((1u128 << (s - 1)) - 1) } else { 0 };
        (kept, round, below != 0)
    };
    if round && (sticky || kept & 1 == 1) {
        kept += 1;
    }
    if kept >= 1 << 24 {
        kept >>= 1;
        result_exp += 1;
    }
    if result_exp > 127 {
        return sign | 0x7F80_0000;
    }
    let kept = kept as u64;
    if kept < 1 << 23 {
        // subnormal (result_exp == -126) or a zero after total cancellation upstream
        return sign | kept;
    }
    sign | (((result_exp + 127) as u64) << 23) | (kept & 0x7F_FFFF)
}

/// `FADD`, IEEE RN-even (`prims.f32_add_bits` on the x86 hosts): SSE NaN rule.
pub fn f32_add(a: u64, b: u64) -> u64 {
    if is_nan(a) {
        return quiet(a);
    }
    if is_nan(b) {
        return quiet(b);
    }
    match (is_inf(a), is_inf(b)) {
        (true, true) => {
            if (a >> 31) == (b >> 31) {
                return a;
            }
            return F32_INDEFINITE;
        }
        (true, false) => return a,
        (false, true) => return b,
        (false, false) => {}
    }
    let (na, ma, ea) = unpack_finite(a);
    let (nb, mb, eb) = unpack_finite(b);
    if ma == 0 && mb == 0 {
        // signed-zero addition: -0 + -0 = -0, else +0 (RN)
        return if na && nb { 0x8000_0000 } else { 0 };
    }
    // align exactly; a gap beyond 30 bits only matters as sticky
    let (mh, eh, nh, ml, el, nl) = if ea >= eb {
        (ma, ea, na, mb, eb, nb)
    } else {
        (mb, eb, nb, ma, ea, na)
    };
    let gap = (eh - el) as u32;
    let (hi, lo, e) = if gap <= 64 {
        ((mh as u128) << gap, ml as u128, el)
    } else {
        // the low operand is far below the ULP grid: keep one sticky bit
        (((mh as u128) << 30) << 1, (ml != 0) as u128, eh - 31)
    };
    let (negative, magnitude) = if nh == nl {
        (nh, hi + lo)
    } else if hi >= lo {
        (nh, hi - lo)
    } else {
        (nl, lo - hi)
    };
    if magnitude == 0 {
        return 0; // exact cancellation of nonzero operands is +0 under RN
    }
    round_pack(negative, magnitude, e)
}

/// `FMUL`, IEEE RN-even: SSE NaN rule; `0 * inf` is the indefinite QNaN.
pub fn f32_mul(a: u64, b: u64) -> u64 {
    if is_nan(a) {
        return quiet(a);
    }
    if is_nan(b) {
        return quiet(b);
    }
    let sign = ((a ^ b) >> 31) & 1;
    let (_, ma, ea) = unpack_finite(if is_inf(a) { 0 } else { a });
    let (_, mb, eb) = unpack_finite(if is_inf(b) { 0 } else { b });
    match (is_inf(a), is_inf(b)) {
        (true, true) => return (sign << 31) | 0x7F80_0000,
        (true, false) => {
            return if mb == 0 { F32_INDEFINITE } else { (sign << 31) | 0x7F80_0000 };
        }
        (false, true) => {
            return if ma == 0 { F32_INDEFINITE } else { (sign << 31) | 0x7F80_0000 };
        }
        (false, false) => {}
    }
    if ma == 0 || mb == 0 {
        return sign << 31;
    }
    round_pack(sign == 1, ma as u128 * mb as u128, ea + eb)
}

/// `div.rn.f32`, IEEE RN-even: SSE NaN rule; `0/0` and `inf/inf` are the
/// indefinite QNaN, `x/0` the signed infinity, `x/inf` the signed zero.
pub fn f32_div(a: u64, b: u64) -> u64 {
    if is_nan(a) {
        return quiet(a);
    }
    if is_nan(b) {
        return quiet(b);
    }
    let sign = ((a ^ b) >> 31) & 1;
    match (is_inf(a), is_inf(b)) {
        (true, true) => return F32_INDEFINITE,
        (true, false) => return (sign << 31) | 0x7F80_0000,
        (false, true) => return sign << 31,
        (false, false) => {}
    }
    let (_, ma, ea) = unpack_finite(a);
    let (_, mb, eb) = unpack_finite(b);
    if mb == 0 {
        return if ma == 0 { F32_INDEFINITE } else { (sign << 31) | 0x7F80_0000 };
    }
    if ma == 0 {
        return sign << 31;
    }
    // quotient with 64 extra bits (so the quotient always has more than the
    // 26 significand+guard bits rounding reads), remainder folded into sticky
    let numerator = (ma as u128) << 64;
    let quotient = numerator / mb as u128;
    let sticky = (numerator % mb as u128 != 0) as u128;
    round_pack(sign == 1, (quotient << 1) | sticky, ea - eb - 65)
}

/// `FFMA` (`prims.f32_fma_bits`): one rounding of the exact `a*b + c`; every
/// NaN result is the canonical `0x7FC00000`, and the exact-zero sign rule is
/// the IEEE RN one the primitive spells out.
pub fn f32_fma(a: u64, b: u64, c: u64) -> u64 {
    if is_nan(a) || is_nan(b) || is_nan(c) {
        return F32_NAN;
    }
    let product_sign = ((a ^ b) >> 31) & 1 == 1;
    if is_inf(a) || is_inf(b) {
        let zero_a = !is_inf(a) && (a & 0x7FFF_FFFF) == 0;
        let zero_b = !is_inf(b) && (b & 0x7FFF_FFFF) == 0;
        // also a finite subnormal/normal times inf is inf; only exact 0 * inf is invalid
        let finite_zero = (!is_inf(a) && unpack_finite(a).1 == 0) || (!is_inf(b) && unpack_finite(b).1 == 0);
        if zero_a || zero_b || finite_zero {
            return F32_NAN;
        }
        if is_inf(c) && ((c >> 31) & 1 == 1) != product_sign {
            return F32_NAN;
        }
        return if product_sign { 0xFF80_0000 } else { 0x7F80_0000 };
    }
    if is_inf(c) {
        return c;
    }
    let (na, ma, ea) = unpack_finite(a);
    let (nb, mb, eb) = unpack_finite(b);
    let (nc, mc, ec) = unpack_finite(c);
    let _ = (na, nb);
    let mp = ma as u128 * mb as u128;
    let ep = ea + eb;
    if mp == 0 && mc == 0 {
        // exact zero: -0 only when the product is a negative zero and c is -0
        let neg = product_sign && nc;
        return if neg { 0x8000_0000 } else { 0 };
    }
    if mp == 0 {
        return round_pack(nc, mc as u128, ec);
    }
    if mc == 0 {
        return round_pack(product_sign, mp, ep);
    }
    let (mh, eh, nh, ml, el, nl) = if ep >= ec {
        (mp, ep, product_sign, mc as u128, ec, nc)
    } else {
        (mc as u128, ec, nc, mp, ep, product_sign)
    };
    let gap = (eh - el) as u32;
    let (hi, lo, e) = if gap <= 72 {
        (mh << gap, ml, el)
    } else {
        ((mh << 54) << 1, (ml != 0) as u128, eh - 55)
    };
    let (negative, magnitude) = if nh == nl {
        (nh, hi + lo)
    } else if hi >= lo {
        (nh, hi - lo)
    } else {
        (nl, lo - hi)
    };
    if magnitude == 0 {
        return 0; // exact cancellation of nonzero addends is +0 (the primitive's rule)
    }
    round_pack(negative, magnitude, e)
}

/// `prims.Bf16ToF32`: exact widening.
pub fn bf16_to_f32(b: u64) -> u64 {
    (b & 0xFFFF) << 16
}

/// `prims.f32_to_bf16_rn_bits`: RNE to BF16 with the given canonical NaN.
fn f32_to_bf16_rn(u: u64, nan: u64) -> u64 {
    let u = u & 0xFFFF_FFFF;
    if is_nan(u) {
        return nan;
    }
    let lsb = (u >> 16) & 1;
    ((u + 0x7FFF + lsb) >> 16) & 0xFFFF
}

/// `prims.F2fpBf16`: SASS `F2FP.BF16.F32` -- RNE, NaN -> `0x7FFF`.
pub fn f2fp_bf16(u: u64) -> u64 {
    f32_to_bf16_rn(u, BF16_NAN_F2FP)
}

/// `prims.Bf16AddF2fp`: the fused-norm residual add `bf16(f32(a) + f32(b))`,
/// NaN -> `0x7FFF`.
pub fn bf16_add_f2fp(a: u64, b: u64) -> u64 {
    f32_to_bf16_rn(f32_add(bf16_to_f32(a), bf16_to_f32(b)), BF16_NAN_F2FP)
}

/// `prims.RopeOut`: `bf16_rn(RN32(x*c - y*s))` on BF16 words (NeoX rotation,
/// first half); products of two BF16 are exact in FP32, and the primitive's
/// FMA canonicalises NaN, so this is `f32_to_bf16_rn(fma(x, c, -(y*s)), 0x7FC0)`.
pub fn rope_out(x: u64, y: u64, c: u64, s: u64) -> u64 {
    let t = f32_mul(bf16_to_f32(y), bf16_to_f32(s));
    let o = f32_fma(bf16_to_f32(x), bf16_to_f32(c), t ^ 0x8000_0000);
    f32_to_bf16_rn(o, BF16_NAN_TORCH)
}

/// `prims.RopeOutAdd`: `bf16_rn(RN32(y*c + x*s))` (NeoX rotation, second half).
pub fn rope_out_add(x: u64, y: u64, c: u64, s: u64) -> u64 {
    let t = f32_mul(bf16_to_f32(x), bf16_to_f32(s));
    let o = f32_fma(bf16_to_f32(y), bf16_to_f32(c), t);
    f32_to_bf16_rn(o, BF16_NAN_TORCH)
}

/// `prims.Bf16GtStrict`: `a > b` on the BF16 values (NaN compares false).
pub fn bf16_gt(a: u64, b: u64) -> u64 {
    let fa = bf16_to_f32(a);
    let fb = bf16_to_f32(b);
    if is_nan(fa) || is_nan(fb) {
        return 0;
    }
    let (na, ma, _) = unpack_finite_or_inf(fa);
    let (nb, mb, _) = unpack_finite_or_inf(fb);
    if ma == 0 && mb == 0 {
        return 0; // +-0 == +-0
    }
    // compare as sign-magnitude on the ordered encoding: for non-negative
    // floats the raw encoding is monotone; for negative, reversed
    let key = |u: u64, neg: bool, mag: u64| -> i64 {
        if mag == 0 {
            0
        } else if neg {
            -((u & 0x7FFF_FFFF) as i64)
        } else {
            (u & 0x7FFF_FFFF) as i64
        }
    };
    (key(fa, na, ma) > key(fb, nb, mb)) as u64
}

#[inline]
fn unpack_finite_or_inf(u: u64) -> (bool, u64, i32) {
    if is_inf(u) {
        ((u >> 31) & 1 == 1, 1 << 24, 128)
    } else {
        unpack_finite(u)
    }
}

/// `prims.I32Add`: wrapping 32-bit addition.
pub fn i32_add(a: u64, b: u64) -> u64 {
    (a as u32).wrapping_add(b as u32) as u64
}

/// `prims.I32Eq`.
pub fn i32_eq(a: u64, b: u64) -> u64 {
    ((a as u32) == (b as u32)) as u64
}

/// `prims.I32Le`: signed comparison.
pub fn i32_le(a: u64, b: u64) -> u64 {
    ((a as u32 as i32) <= (b as u32 as i32)) as u64
}

/// `prims.SelectBf16` / `prims.SelectI32`: `c ? a : b`.
pub fn select(c: u64, a: u64, b: u64) -> u64 {
    if c & 1 == 1 {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// MUFU table lookups (rms_triton_model.cpp, measured on sm_89, verified on
// the A100).  A table argument is the 64-bit leaf `(index << 32) | word`.
// ---------------------------------------------------------------------------

#[inline]
fn table_word(t: u64, expected_index: u64) -> Option<u64> {
    if t >> 32 != expected_index {
        return None;
    }
    Some(t & 0xFFFF_FFFF)
}

/// The table index MUFU.RCP reads for the operand `x`: its mantissa field.
/// The RCP table has `2^23` rows.
pub fn mufu_rcp_index(x: u64) -> u64 {
    x & 0x7F_FFFF
}

/// `mufu_rcp` of `rms_triton_model.cpp` with the table row as an argument.
/// Special cases never read the table, but the gate still pins the row of
/// the operand's mantissa so the witness is a function of the statement.
fn mufu_rcp_word(x: u64, t: u64) -> Option<u64> {
    let word = table_word(t, mufu_rcp_index(x))?;
    let s = x & 0x8000_0000;
    let ax = x & 0x7FFF_FFFF;
    let e = (ax >> 23) & 0xFF;
    if e == 0xFF {
        return Some(if ax & 0x7F_FFFF != 0 { 0x7FFF_FFFF } else { s });
    }
    if e == 0 {
        return Some(s | 0x7F80_0000);
    }
    let re = ((word >> 23) & 0xFF) as i64 - (e as i64 - 127);
    if re <= 0 {
        return Some(s);
    }
    if re >= 0xFF {
        return Some(s | 0x7F80_0000);
    }
    Some(s | ((re as u64) << 23) | (word & 0x7F_FFFF))
}

/// The table index MUFU.SQRT / MUFU.RSQ read: `(exponent parity << 23) | mantissa`.
/// Both tables have `2^24` rows.
pub fn mufu_sqrt_index(x: u64) -> u64 {
    let ax = x & 0x7FFF_FFFF;
    let e = ((ax >> 23) & 0xFF) as i64;
    let parity = ((e - 127) & 1) as u64;
    (parity << 23) | (ax & 0x7F_FFFF)
}

/// `sqrt.approx.ftz.f32` (`mufu_sqrt` of the model): the Triton RMSNorm
/// epilogue's square root; `prims.MufuSqrtFtz` with the table row opened.
pub fn mufu_sqrt_tab(x: u64, t: u64) -> Option<u64> {
    let word = table_word(t, mufu_sqrt_index(x))?;
    let s = x & 0x8000_0000;
    let ax = x & 0x7FFF_FFFF;
    let e = ((ax >> 23) & 0xFF) as i64;
    if e == 0xFF {
        if ax & 0x7F_FFFF != 0 {
            return Some(0x7FFF_FFFF);
        }
        return Some(if s != 0 { 0x7FFF_FFFF } else { 0x7F80_0000 });
    }
    if e == 0 {
        return Some(s); // ftz: +-0 and subnormal -> +-0
    }
    if s != 0 {
        return Some(0x7FFF_FFFF);
    }
    let k = e - 127;
    let parity = k & 1;
    let re = ((word >> 23) & 0xFF) as i64 + (k - parity) / 2;
    // the C++ packs `(uint32_t)re << 23`: 32-bit wrapping, exactly mirrored
    Some((((re as u32) << 23) | (word as u32 & 0x7F_FFFF)) as u64)
}

/// `mufu_rsq` of the model: MUFU.RSQ, the CUDA fused-norm reciprocal square root core.
fn mufu_rsq_word(x: u64, t: u64) -> Option<u64> {
    let word = table_word(t, mufu_sqrt_index(x))?;
    let s = x & 0x8000_0000;
    let ax = x & 0x7FFF_FFFF;
    let e = ((ax >> 23) & 0xFF) as i64;
    if e == 0xFF {
        if ax & 0x7F_FFFF != 0 {
            return Some(0x7FFF_FFFF);
        }
        return Some(if s != 0 { 0x7FFF_FFFF } else { 0 });
    }
    if e == 0 {
        return Some(s | 0x7F80_0000);
    }
    if s != 0 {
        return Some(0x7FFF_FFFF);
    }
    let k = e - 127;
    let parity = k & 1;
    let re = ((word >> 23) & 0xFF) as i64 - (k - parity) / 2;
    Some((((re as u32) << 23) | (word as u32 & 0x7F_FFFF)) as u64)
}

const TWO_POW_126: u64 = 0x7E80_0000; // 8.507059173023462e37f
const MIN_NORMAL: u64 = 0x0080_0000; // 1.1754943508222875e-38f
const QUARTER: u64 = 0x3E80_0000; // 0.25f
const TWO_POW_24: u64 = 0x4B80_0000; // 16777216.0f
const TWO_POW_12: u64 = 0x4580_0000; // 4096.0f

/// `x > 2^126` on the FP32 value (false for NaN), as `div_full_parts` compares.
#[inline]
fn abs_gt(x: u64, threshold: u64) -> bool {
    let ax = x & 0x7FFF_FFFF;
    !is_nan(x) && ax > threshold
}

/// `!(x < 2^-126)` on the FP32 value (true for NaN).
#[inline]
fn abs_ge_or_nan(x: u64, threshold: u64) -> bool {
    let ax = x & 0x7FFF_FFFF;
    is_nan(x) || ax >= threshold
}

/// The `div.full.f32` denominator after range scaling (`div_full_parts`):
/// `|b| > 2^126` -> `b * 0.25f`; `|b| < 2^-126` -> `b * 2^24`; else `b`.
fn div_full_scale_b(b: u64) -> u64 {
    if abs_gt(b, TWO_POW_126) {
        f32_mul(b, QUARTER)
    } else if abs_ge_or_nan(b, MIN_NORMAL) {
        b
    } else {
        f32_mul(b, TWO_POW_24)
    }
}

/// The index of the RCP row `div_full_rcp(b, .)` reads: the mantissa of the
/// *scaled* denominator (the builder derives the witness position from this).
pub fn div_full_rcp_index(b: u64) -> u64 {
    mufu_rcp_index(div_full_scale_b(b))
}

/// `prims.DivFullRcp` with the table row opened: range-scale the denominator,
/// then MUFU.RCP.
pub fn div_full_rcp(b: u64, t: u64) -> Option<u64> {
    mufu_rcp_word(div_full_scale_b(b), t)
}

/// `prims.DivFullScaleA`: the numerator scaling of `div.full.f32` -- decided
/// by `|b|`, applied to `a`.
pub fn div_full_scale_a(a: u64, b: u64) -> u64 {
    if abs_gt(b, TWO_POW_126) {
        f32_mul(a, QUARTER)
    } else if abs_ge_or_nan(b, MIN_NORMAL) {
        a
    } else {
        f32_mul(a, TWO_POW_24)
    }
}

/// The index of the RSQ row `rsqrt_approx(x, .)` reads: parity/mantissa of
/// the scaled operand (`rsqrt_nonftz`: `|x| < 2^-126` -> `x * 2^24` first).
pub fn rsqrt_approx_index(x: u64) -> u64 {
    let scaled = if abs_ge_or_nan(x, MIN_NORMAL) {
        x
    } else {
        f32_mul(x, TWO_POW_24)
    };
    mufu_sqrt_index(scaled)
}

/// `prims.RsqrtApprox` (`rsqrt.approx.f32`, `rsqrt_nonftz` of the model) with
/// the table row opened: subnormal pre-scale by `2^24`, MUFU.RSQ, post-scale
/// by `2^12`.
pub fn rsqrt_approx(x: u64, t: u64) -> Option<u64> {
    if abs_ge_or_nan(x, MIN_NORMAL) {
        mufu_rsq_word(x, t)
    } else {
        let r = mufu_rsq_word(f32_mul(x, TWO_POW_24), t)?;
        Some(f32_mul(r, TWO_POW_12))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f32) -> u64 {
        x.to_bits() as u64
    }

    #[test]
    fn family_digest_matches_the_python_mirror() {
        // veritor.core.reduce_gates.make_reduce_mufu_gate_set().digest
        let set = crate::gateset::GateSet::resolve(crate::gateset::REDUCE_MUFU_ID, 32).unwrap();
        let digest: String = set.digest().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            digest,
            "a348a7bbe15090308af57cb7e5f1dc035263cc9092b853ba7a2dff7b9d462837"
        );
        // resolves only at the statement width 32
        assert!(crate::gateset::GateSet::resolve(crate::gateset::REDUCE_MUFU_ID, 64).is_none());
    }

    #[test]
    fn add_mul_matches_native_on_finite_words() {
        // native f32 on the test host is IEEE RN, so finite cases must agree
        let words: Vec<u64> = vec![
            0,
            0x8000_0000,
            f(1.0),
            f(-1.0),
            f(1.5),
            f(0.1),
            f(3.0e38),
            f(-3.0e38),
            f(1.0e-40),
            f(-1.0e-40),
            0x0000_0001,
            0x8000_0001,
            0x007F_FFFF,
            0x0080_0000,
            f(576.0),
            f(1e-8),
            f(2.0f32.powi(126)),
            f(65504.0),
        ];
        for &a in &words {
            for &b in &words {
                let fa = f32::from_bits(a as u32);
                let fb = f32::from_bits(b as u32);
                assert_eq!(
                    f32_add(a, b),
                    (fa + fb).to_bits() as u64,
                    "add {a:08x} {b:08x}"
                );
                assert_eq!(
                    f32_mul(a, b),
                    (fa * fb).to_bits() as u64,
                    "mul {a:08x} {b:08x}"
                );
                if b & 0x7FFF_FFFF != 0 {
                    assert_eq!(
                        f32_div(a, b),
                        (fa / fb).to_bits() as u64,
                        "div {a:08x} {b:08x}"
                    );
                }
            }
        }
    }

    #[test]
    fn fma_is_single_rounding() {
        // 1 + 2^-24 * 2^-24 rounds away the tail under two roundings but not here
        let a = f(2.0f32.powi(-24));
        let one = f(1.0);
        assert_eq!(f32_fma(a, a, one), f(1.0));
        // f32::mul_add on the host is a hardware FMA: finite cases must agree
        let words = [f(1.5), f(-2.25), f(1.0e-20), f(3.0e18), f(576.0), f(0.1)];
        for &x in &words {
            for &y in &words {
                for &z in &words {
                    let native = f32::from_bits(x as u32)
                        .mul_add(f32::from_bits(y as u32), f32::from_bits(z as u32));
                    assert_eq!(f32_fma(x, y, z), native.to_bits() as u64, "{x:08x} {y:08x} {z:08x}");
                }
            }
        }
        assert_eq!(f32_fma(f(f32::NAN), one, one), F32_NAN);
        assert_eq!(f32_fma(f(f32::INFINITY), 0, one), F32_NAN);
        assert_eq!(
            f32_fma(f(f32::INFINITY), one, f(f32::NEG_INFINITY)),
            F32_NAN
        );
        assert_eq!(f32_fma(f(-0.0), f(0.0), f(-0.0)), 0x8000_0000);
        assert_eq!(f32_fma(f(0.0), f(0.0), f(-0.0)), 0);
    }

    #[test]
    fn nan_rules_are_sse() {
        let snan_a = 0x7F80_0001u64;
        let qnan_b = 0xFFC0_0123u64;
        assert_eq!(f32_add(snan_a, f(1.0)), 0x7FC0_0001);
        assert_eq!(f32_add(f(1.0), qnan_b), qnan_b);
        assert_eq!(f32_add(snan_a, qnan_b), 0x7FC0_0001);
        assert_eq!(f32_add(f(f32::INFINITY), f(f32::NEG_INFINITY)), F32_INDEFINITE);
        assert_eq!(f32_mul(f(0.0), f(f32::INFINITY)), F32_INDEFINITE);
        assert_eq!(f32_div(f(0.0), f(-0.0)), F32_INDEFINITE);
        assert_eq!(f32_div(f(f32::INFINITY), f(f32::INFINITY)), F32_INDEFINITE);
        assert_eq!(f32_div(f(1.0), f(0.0)), f(f32::INFINITY));
        assert_eq!(f32_div(f(1.0), f(-0.0)), f(f32::NEG_INFINITY));
        assert_eq!(f32_div(f(1.0), f(f32::INFINITY)), 0);
    }

    #[test]
    fn table_gates_reject_the_wrong_row() {
        let x = f(2.0); // exponent 128, parity 1, mantissa 0
        let index = mufu_sqrt_index(x);
        assert_eq!(index, 1 << 23);
        let row = (index << 32) | 0x3FB5_04F3; // any word
        assert!(mufu_sqrt_tab(x, row).is_some());
        assert!(mufu_sqrt_tab(x, ((index + 1) << 32) | 0x3FB5_04F3).is_none());
    }

    #[test]
    fn mufu_specials_never_use_the_word() {
        // sqrt of a negative normal is the all-ones NaN whatever the word says
        let x = f(-2.0);
        let row = (mufu_sqrt_index(x) << 32) | 0xDEAD_BEEF;
        assert_eq!(mufu_sqrt_tab(x, row), Some(0x7FFF_FFFF));
        // ftz: sqrt of +-0 / subnormal is the signed zero
        assert_eq!(mufu_sqrt_tab(0x8000_0001, (1 << 55) | 0x1), None); // wrong row still rejects
        let sub = 0x0000_0001u64;
        let sub_row = mufu_sqrt_index(sub) << 32;
        assert_eq!(mufu_sqrt_tab(sub, sub_row), Some(0));
    }

    #[test]
    fn bf16_gt_matches_float_compare() {
        let words = [0u64, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x7F7F];
        for &a in &words {
            for &b in &words {
                let fa = f32::from_bits((a << 16) as u32);
                let fb = f32::from_bits((b << 16) as u32);
                assert_eq!(bf16_gt(a, b), (fa > fb) as u64, "{a:04x} {b:04x}");
            }
        }
    }

    /// Differential against the host's native IEEE floats over random words,
    /// subnormals and exponent boundaries included (NaN words excluded: the
    /// host quietly canonicalises differently per platform; the pinned NaN
    /// rules are asserted in `nan_rules_are_sse` and by the cross-language
    /// vectors).  Heavy: run with `--ignored` in a release build.
    #[test]
    #[ignore]
    fn differential_native_floats() {
        let mut state = 0x243F_6A88_85A3_08D3u64; // xorshift
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut word = |raw: u64| -> u64 {
            // bias toward small exponents and boundaries
            let u = (raw & 0xFFFF_FFFF) as u32;
            let pick = (raw >> 60) & 0xF;
            let v = match pick {
                0 => u & 0x807F_FFFF,                    // subnormal / zero
                1 => (u & 0x803F_FFFF) | 0x0080_0000,    // smallest normals
                2 => (u & 0x807F_FFFF) | 0x7E80_0000,    // near overflow
                3 => u & 0x8000_FFFF | 0x3F80_0000,      // near 1.0
                _ => u,
            };
            if (v & 0x7F80_0000) == 0x7F80_0000 && (v & 0x7F_FFFF) != 0 {
                (v & 0xFF80_0000) as u64 // squash NaN to inf
            } else {
                v as u64
            }
        };
        for _ in 0..5_000_000 {
            let a = word(next());
            let b = word(next());
            let c = word(next());
            let (fa, fb, fc) = (
                f32::from_bits(a as u32),
                f32::from_bits(b as u32),
                f32::from_bits(c as u32),
            );
            let add = (fa + fb).to_bits() as u64;
            if !is_nan(add) {
                assert_eq!(f32_add(a, b), add, "add {a:08x} {b:08x}");
            }
            let mul = (fa * fb).to_bits() as u64;
            if !is_nan(mul) {
                assert_eq!(f32_mul(a, b), mul, "mul {a:08x} {b:08x}");
            }
            let div = (fa / fb).to_bits() as u64;
            if !is_nan(div) {
                assert_eq!(f32_div(a, b), div, "div {a:08x} {b:08x}");
            }
            let fma = fa.mul_add(fb, fc).to_bits() as u64;
            if !is_nan(fma) {
                assert_eq!(f32_fma(a, b, c), fma, "fma {a:08x} {b:08x} {c:08x}");
            }
        }
    }

    #[test]
    fn div_full_scaling_thresholds() {
        // 2^126 exactly is NOT scaled (strict >)
        assert_eq!(div_full_scale_a(f(1.0), f(2.0f32.powi(126))), f(1.0));
        assert_eq!(div_full_scale_a(f(1.0), f(3.0e38)), f(0.25));
        assert_eq!(div_full_scale_a(f(1.0), f(1.0e-40)), f(16777216.0));
        // the minimum normal is not scaled
        assert_eq!(div_full_scale_a(f(1.0), 0x0080_0000), f(1.0));
        // NaN denominator: no scaling
        assert_eq!(div_full_scale_a(f(3.0), f(f32::NAN)), f(3.0));
    }
}
