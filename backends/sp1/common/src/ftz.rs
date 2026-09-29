//! The `veritor.ftz-f32@1` gates: the FTZ scalar relations of the fast-math FA2 attention
//! kernel, bit for bit as `verity_ir/registry/prims.py` registers them.  Written by E4
//! (EOD 2026-09-11) as `zk/sp1/programs/ftz/core/src/ftz.rs`; relocated verbatim into the
//! protocol guest's shared crate by E7 so the ONE approved guest carries the family
//! (`gateset::FTZ_F32_ID`; the digest `21f8f7d2...` is E4's `family.rs` / Python `ck_ftz`):
//!
//! | registered primitive | rule (prims.py) | here |
//! |---|---|---|
//! | `F32AddFtz_v1`    (:236) | `ftz(RN(ftz(a) + ftz(b)))`            | [`f32_add_ftz`] |
//! | `F32SubFtz_v1`    (:242) | `ftz(RN(ftz(a) - ftz(b)))`            | [`f32_sub_ftz`] |
//! | `F32MulFtz_v1`    (:248) | `ftz(RN(ftz(a) * ftz(b)))`            | [`f32_mul_ftz`] |
//! | `F32FmaFtz_v1`    (:254) | `ftz(RN(ftz(a) * ftz(b) + ftz(c)))`   | [`f32_fma_ftz`] |
//! | `F32FmaSubFtz_v1` (:260) | `ftz(RN(ftz(a) * ftz(b) - ftz(c)))`   | [`f32_fma_sub_ftz`] |
//! | `F32Max_v1`       (:266) | `ftz(a) > ftz(b) ? ftz(a) : ftz(b)`   | [`f32_max_ftz`] |
//! | `GuardNegInfZero_v1` (:272) | `m == -inf ? +0 : m`               | [`guard_neg_inf_zero`] |
//! | `MufuEx2Ftz_v1`   (:426) | `ex2.approx.ftz.f32`, measured table `T[j]` + reduction rule (`fa2_model.cpp` `mufu_ex2_bits`) | [`mufu_ex2_tab`] |
//! | `Fa2InvSum_v1`    (:440) | `(sum == ±0 \|\| NaN) ? 1.0 : rcp.approx.ftz.f32(sum)` (`mufu_rcp_bits`) | [`fa2_inv_sum_tab`] |
//!
//! Arithmetic between the flushes is the approved backend's soft-float
//! (`veritor_zk_common::reduce::{f32_add, f32_mul, f32_fma}`, the twin of
//! `veritor.core.reduce_gates`): IEEE-754 binary32 round-to-nearest-even, exact
//! product-sum for the FMA, subnormals kept *inside* the operation and flushed by
//! `ftz` after it -- exactly the order `prims.F32AddFtz` etc. spell.  Nothing here maps
//! an `f32` to a host float instruction, so the SP1 guest (rv32im, no FPU) and every
//! host agree on every word.
//!
//! **NaN rule (stated, not hidden).**  The registered `F32{Add,Sub,Mul}Ftz` evaluators
//! call numpy, whose NaN *payload* propagation is the host CPU's.  The approved guest
//! pins the x86-SSE rule for `f32_add`/`f32_mul` ("the platform the registry's
//! conformance evidence was measured on", `reduce.rs`): a NaN operand is returned
//! quieted, the first operand first; an invalid operation (`inf - inf`, `0 * inf`) is the
//! indefinite QNaN `0xFFC0_0000`.  This module inherits that pin.  Payloads are NOT
//! banned: a NaN operand with any payload is a legal input and its quieted word is the
//! registered output.  Where the AArch64 evaluator differs (two NaN operands where only
//! the second is signalling; the sign of a *generated* NaN), the vectors say so
//! (`ftz_vectors/README.md`).  The FMA family canonicalises every NaN to `0x7FC0_0000`
//! (`prims.f32_fma_bits`) and is platform independent.  The MUFU rules produce
//! `0x7FFF_FFFF` for NaN (the C model).
//!
//! **Table gates.**  `MufuEx2Ftz` and `Fa2InvSum` are defined by the measured 2^23-entry
//! sm_89 tables (`vllm-poc/fixtures/W11-40f0cebeb670-20260907T1800Z/mufu`).  As in
//! `veritor.reduce-mufu@1` (`mufu_sqrt_tab`, `div_full_rcp`), the table row is an
//! *argument*: the 64-bit word `(index << 32) | T[index]` opened from a committed table
//! tree.  The gate recomputes the index the hardware reads from its operand and returns
//! `None` -- a rejection -- when the opened row is any other index, so a proof can only
//! use the one true row; the row's authenticity is the table tree's root, pinned by the
//! statement (E6/E7: the roots of the ex2/rcp trees are not pinned by any record today).

use crate::reduce::{f32_add, f32_fma, f32_mul};

/// `prims.F32_NAN`: the canonical quiet NaN of the FMA family.
pub const F32_NAN: u64 = 0x7FC0_0000;
/// The MUFU units' canonical NaN word (`fa2_model.cpp`).
pub const MUFU_NAN: u64 = 0x7FFF_FFFF;
/// `1.0f`.
pub const F32_ONE: u64 = 0x3F80_0000;
const SIGN: u64 = 0x8000_0000;
const EXP: u64 = 0x7F80_0000;
const MANT: u64 = 0x7F_FFFF;
const NEG_INF: u64 = 0xFF80_0000;

#[inline]
fn is_nan(u: u64) -> bool {
    (u & EXP) == EXP && (u & MANT) != 0
}

#[inline]
fn quiet(u: u64) -> u64 {
    u | 0x40_0000
}

/// `prims.ftz_bits`: flush a subnormal encoding to its signed zero; every other word
/// (normals, zeros, infinities, NaNs of any payload) is returned unchanged.
#[inline]
pub fn ftz(u: u64) -> u64 {
    let u = u & 0xFFFF_FFFF;
    if (u & EXP) == 0 && (u & MANT) != 0 {
        u & SIGN
    } else {
        u
    }
}

/// `F32AddFtz_v1`: `ftz(RN(ftz(a) + ftz(b)))` (FADD.FTZ).
pub fn f32_add_ftz(a: u64, b: u64) -> u64 {
    ftz(f32_add(ftz(a), ftz(b)))
}

/// `FADD` with the second operand negated, as SSE `subss` computes it: a NaN operand is
/// returned quieted (first operand first, sign untouched); otherwise `a + (-b)`, which is
/// bit-identical to IEEE `a - b` (negation is exact; the signed-zero rules coincide:
/// `(+0) - (+0) = +0`, `(-0) - (+0) = -0`, `(+0) - (-0) = +0`, `(-0) - (-0) = +0`).
fn f32_sub(a: u64, b: u64) -> u64 {
    if is_nan(a) {
        return quiet(a);
    }
    if is_nan(b) {
        return quiet(b);
    }
    f32_add(a, b ^ SIGN)
}

/// `F32SubFtz_v1`: `ftz(RN(ftz(a) - ftz(b)))`.
pub fn f32_sub_ftz(a: u64, b: u64) -> u64 {
    ftz(f32_sub(ftz(a), ftz(b)))
}

/// `F32MulFtz_v1`: `ftz(RN(ftz(a) * ftz(b)))` (FMUL.FTZ).
pub fn f32_mul_ftz(a: u64, b: u64) -> u64 {
    ftz(f32_mul(ftz(a), ftz(b)))
}

/// `F32FmaFtz_v1`: `ftz(RN(ftz(a) * ftz(b) + ftz(c)))`, one rounding (FFMA.FTZ).
pub fn f32_fma_ftz(a: u64, b: u64, c: u64) -> u64 {
    ftz(f32_fma(ftz(a), ftz(b), ftz(c)))
}

/// `F32FmaSubFtz_v1`: `ftz(RN(ftz(a) * ftz(b) - ftz(c)))` -- the addend's sign is
/// flipped *after* its flush (`prims.F32FmaSubFtz`: `ftz_bits(c) ^ 0x80000000`), so a
/// subnormal `c` enters as the zero of the opposite sign.
pub fn f32_fma_sub_ftz(a: u64, b: u64, c: u64) -> u64 {
    ftz(f32_fma(ftz(a), ftz(b), ftz(c) ^ SIGN))
}

/// The total order key of a non-NaN encoding: zeros of both signs are `0`, the rest is
/// sign-magnitude on the ordered encoding (infinities included).
#[inline]
fn order_key(u: u64) -> i64 {
    let mag = u & 0x7FFF_FFFF;
    if mag == 0 {
        0
    } else if u & SIGN != 0 {
        -(mag as i64)
    } else {
        mag as i64
    }
}

/// `F32Max_v1` (FA2 `MaxOp`, `FMNMX` as the fast-math build uses it):
/// `f32_max_bits(ftz(a), ftz(b))` = `ftz(a) > ftz(b) ? ftz(a) : ftz(b)`.  The comparison
/// is IEEE `>`: a NaN on either side compares false and the (flushed) second operand is
/// returned; `+0 > -0` is false, so of two zeros the second is returned.  The output is
/// the *flushed* word of the chosen operand.
pub fn f32_max_ftz(a: u64, b: u64) -> u64 {
    let (a, b) = (ftz(a), ftz(b));
    if is_nan(a) || is_nan(b) {
        return b;
    }
    if order_key(a) > order_key(b) {
        a
    } else {
        b
    }
}

/// `GuardNegInfZero_v1` (FA2 `Check_inf`): `m == -inf ? +0 : m`.  No flush: a subnormal
/// `m` passes through unchanged.
pub fn guard_neg_inf_zero(m: u64) -> u64 {
    let m = m & 0xFFFF_FFFF;
    if m == NEG_INF {
        0
    } else {
        m
    }
}

// ---------------------------------------------------------------------------
// MUFU table gates: `fa2_model.cpp` `mufu_ex2_bits` / `mufu_rcp_bits` with the
// table row supplied as `(index << 32) | word` and checked against the index
// the operand reads (the `veritor.reduce-mufu@1` table-gate contract).
// ---------------------------------------------------------------------------

#[inline]
fn table_word(row: u64, expected_index: u64) -> Option<u64> {
    if row >> 32 != expected_index {
        return None;
    }
    Some(row & 0xFFFF_FFFF)
}

/// The fixed-point reduction of `mufu_ex2_bits` for a word that reads the table:
/// `(n, j)` with `ex2(x) = T[j] * 2^n`.  `None` for the words the C rule answers before
/// touching the table (`e == 0xFF`, `e == 0`, `unb >= 30`).
///
/// `K = floor(|x| 2^23)` exactly: `mant < 2^24`, so every right shift of 24 or more is
/// `0` and each `|x| < 2^-23` reads row 0 at `n = 0` (`ex2 = 1.0`, either sign).  The C
/// model clamps the shift (a count >= 64 is undefined there), `checked_shr` here: the
/// exhaustive sm_89 verification (all 2^32 inputs, 0 mismatches, attention-model.md §3.1)
/// ran the model on the device, where the shift clamps.
fn ex2_reduce(x: u64) -> Option<(i64, u64)> {
    let ax = x & 0x7FFF_FFFF;
    let e = (ax >> 23) as i64;
    if e == 0xFF || e == 0 {
        return None;
    }
    let unb = e - 127;
    if unb >= 30 {
        return None;
    }
    let mant: u64 = (ax & MANT) | 0x80_0000;
    let k: u64 = if unb >= 0 {
        mant << unb
    } else {
        mant.checked_shr((-unb) as u32).unwrap_or(0)
    };
    let fixed: i64 = if x & SIGN == 0 {
        k as i64
    } else if k & MANT == 0 {
        -(k as i64)
    } else {
        -(k as i64) - 1
    };
    let n = fixed >> 23; // arithmetic shift, as `long long >> 23`
    let j = (fixed as u64) & MANT;
    Some((n, j))
}

/// The EX2 table row the operand `x` reads: `j` of the reduction, or row `0` for the
/// words the rule answers without the table (a pinned, deterministic row so the witness
/// is a function of the statement, as `mufu_rcp_word` pins the mantissa row).
pub fn mufu_ex2_index(x: u64) -> u64 {
    ex2_reduce(x & 0xFFFF_FFFF).map_or(0, |(_, j)| j)
}

/// `MufuEx2Ftz_v1` (`ex2.approx.ftz.f32`, MUFU.EX2) with the table row opened:
/// `mufu_ex2_bits(x, T)` of `fa2_model.cpp`, `None` when `row` is not row
/// [`mufu_ex2_index`]`(x)`.
///
/// Specials in the C order: NaN -> `0x7FFF_FFFF`; `ex2(-inf) = +0`, `ex2(+inf) = +inf`;
/// `e == 0` (±0 and every subnormal: ftz) -> `1.0`; `unb >= 30` -> `+0` (negative) or
/// `+inf`; otherwise `T[j] * 2^n` with overflow to `+inf` and underflow (`re <= 0`) to
/// `+0` (the ftz of the result).
pub fn mufu_ex2_tab(x: u64, row: u64) -> Option<u64> {
    let x = x & 0xFFFF_FFFF;
    let word = table_word(row, mufu_ex2_index(x))?;
    let ax = x & 0x7FFF_FFFF;
    let e = (ax >> 23) as i64;
    if e == 0xFF {
        return Some(if ax & MANT != 0 {
            MUFU_NAN
        } else if x & SIGN != 0 {
            0
        } else {
            EXP
        });
    }
    if e == 0 {
        return Some(F32_ONE);
    }
    let unb = e - 127;
    if unb >= 30 {
        return Some(if x & SIGN != 0 { 0 } else { EXP });
    }
    let (n, _) = ex2_reduce(x).expect("reads the table");
    let re = ((word >> 23) & 0xFF) as i64 + n;
    if re >= 255 {
        return Some(EXP);
    }
    if re <= 0 {
        return Some(0);
    }
    Some(((re as u64) << 23) | (word & MANT))
}

/// The RCP table row `Fa2InvSum` pins for the operand `s`: its mantissa field (the
/// `mufu_rcp_index` of `reduce-mufu`; 2^23 rows).  Pinned also for the degenerate sums
/// (±0, NaN) the epilogue answers with `1.0` without reading the table.
pub fn fa2_inv_sum_index(s: u64) -> u64 {
    s & MANT
}

/// `Fa2InvSum_v1` (FA2 `normalize_softmax_lse`): `(sum == ±0 || sum != sum) ? 1.0 :
/// rcp.approx.ftz.f32(sum)`, the reciprocal by `mufu_rcp_bits` with the opened row;
/// `None` when `row` is not row [`fa2_inv_sum_index`]`(s)`.
///
/// `mufu_rcp_bits`: NaN -> `0x7FFF_FFFF` (unreachable here: the epilogue answers NaN
/// first); `±inf -> ±0`; `e == 0` (a subnormal: ftz) -> `±inf`; else `R[m] * 2^-(e-127)`
/// with `re <= 0 -> ±0`, `re >= 0xFF -> ±inf`.
pub fn fa2_inv_sum_tab(s: u64, row: u64) -> Option<u64> {
    let s = s & 0xFFFF_FFFF;
    let word = table_word(row, fa2_inv_sum_index(s))?;
    if (s & 0x7FFF_FFFF) == 0 || is_nan(s) {
        return Some(F32_ONE);
    }
    let sign = s & SIGN;
    let ax = s & 0x7FFF_FFFF;
    let e = (ax >> 23) as i64;
    if e == 0xFF {
        return Some(sign); // ±inf -> ±0 (the NaN arm is the epilogue's)
    }
    if e == 0 {
        return Some(sign | EXP);
    }
    let re = ((word >> 23) & 0xFF) as i64 - (e - 127);
    if re <= 0 {
        return Some(sign);
    }
    if re >= 0xFF {
        return Some(sign | EXP);
    }
    Some(sign | ((re as u64) << 23) | (word & MANT))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f32) -> u64 {
        x.to_bits() as u64
    }

    #[test]
    fn ftz_flushes_only_subnormals_and_keeps_the_sign() {
        assert_eq!(ftz(0x0000_0001), 0);
        assert_eq!(ftz(0x8000_0001), SIGN);
        assert_eq!(ftz(0x007F_FFFF), 0);
        assert_eq!(ftz(0x807F_FFFF), SIGN);
        assert_eq!(ftz(0x0080_0000), 0x0080_0000); // smallest normal stays
        assert_eq!(ftz(0), 0);
        assert_eq!(ftz(SIGN), SIGN);
        assert_eq!(ftz(EXP), EXP);
        assert_eq!(ftz(0x7FC0_1234), 0x7FC0_1234); // NaN payload untouched
    }

    #[test]
    fn mul_flushes_a_subnormal_product_and_operands() {
        // 2^-100 * 2^-30 = 2^-130 is subnormal: flushed to +0
        assert_eq!(f32_mul_ftz(f(2f32.powi(-100)), f(2f32.powi(-30))), 0);
        // a subnormal operand is a zero before the multiply: -tiny * 1 = -0
        assert_eq!(f32_mul_ftz(0x8000_0001, F32_ONE), SIGN);
        // and 0 * inf after the flush is the indefinite QNaN (SSE rule)
        assert_eq!(f32_mul_ftz(0x0000_0001, EXP), 0xFFC0_0000);
        assert_eq!(f32_mul_ftz(f(1.5), f(2.0)), f(3.0));
    }

    #[test]
    fn add_sub_ties_and_zeros() {
        assert_eq!(f32_add_ftz(f(1.0), f(-1.0)), 0); // exact cancellation is +0
        assert_eq!(f32_add_ftz(SIGN, SIGN), SIGN); // -0 + -0 = -0
        assert_eq!(f32_sub_ftz(SIGN, SIGN), 0); // -0 - -0 = +0
        assert_eq!(f32_sub_ftz(SIGN, 0), SIGN); // -0 - +0 = -0
        // tie to even: 1 + 2^-24 -> 1 ; 1 + 3*2^-24 -> 1 + 2^-22
        assert_eq!(f32_add_ftz(F32_ONE, f(2f32.powi(-24))), F32_ONE);
        assert_eq!(f32_add_ftz(F32_ONE, f(3.0 * 2f32.powi(-24))), F32_ONE + 2);
        // smallest-normal difference underflows to a subnormal -> flushed
        assert_eq!(f32_sub_ftz(0x0080_0001, 0x0080_0000), 0);
    }

    #[test]
    fn fma_variants() {
        // exact product-sum, one rounding: 2^12 * 2^12 - (2^24 - 1) = 1
        assert_eq!(f32_fma_sub_ftz(f(4096.0), f(4096.0), f(16777215.0)), F32_ONE);
        assert_eq!(f32_fma_ftz(f(4096.0), f(4096.0), f(-16777215.0)), F32_ONE);
        // the addend's flush happens before its negation: a subnormal c is -0 here
        assert_eq!(f32_fma_sub_ftz(0, F32_ONE, 0x0000_0001), 0); // (+0)*(1) + (-0) = +0
        assert_eq!(f32_fma_sub_ftz(SIGN, F32_ONE, 0x0000_0001), SIGN); // (-0)*(1) + (-0) = -0
        assert_eq!(f32_fma_ftz(f(1.0), f(1.0), 0x7F80_0001), F32_NAN);
    }

    #[test]
    fn max_and_guard() {
        assert_eq!(f32_max_ftz(f(1.0), f(2.0)), f(2.0));
        assert_eq!(f32_max_ftz(f(-1.0), f(-2.0)), f(-1.0));
        assert_eq!(f32_max_ftz(0x7FC0_0000, f(1.0)), f(1.0));
        assert_eq!(f32_max_ftz(f(1.0), 0x7FC0_0000), 0x7FC0_0000);
        assert_eq!(f32_max_ftz(0, SIGN), SIGN);
        assert_eq!(f32_max_ftz(SIGN, 0), 0);
        assert_eq!(f32_max_ftz(0x0000_0001, SIGN), SIGN); // flushed first operand is +0, not > -0
        assert_eq!(f32_max_ftz(0x0000_0001, 0x8000_0002), SIGN); // both flushed, returns flushed b
        assert_eq!(f32_max_ftz(NEG_INF, EXP), EXP);
        assert_eq!(guard_neg_inf_zero(NEG_INF), 0);
        assert_eq!(guard_neg_inf_zero(EXP), EXP);
        assert_eq!(guard_neg_inf_zero(0x0000_0001), 0x0000_0001);
    }

    #[test]
    fn ex2_reduction_matches_the_c_rule_on_landmarks() {
        // x = 1.0: e = 127, unb = 0, K = 0x800000, fixed = K, n = 1, j = 0 -> T[0] * 2
        assert_eq!(ex2_reduce(F32_ONE), Some((1, 0)));
        // x = -1.0: K & MANT == 0 -> fixed = -K -> n = -1, j = 0
        assert_eq!(ex2_reduce(f(-1.0)), Some((-1, 0)));
        // x = -0.5: K = 0x400000, fixed = -K - 1 -> n = -1, j = 2^23 - 1 - 0x400000
        assert_eq!(ex2_reduce(f(-0.5)), Some((-1, 0x3F_FFFF)));
        // x = +-2^-64 (a shift by 64) and every other |x| < 2^-23 (biased exponent 1..=103): K = 0 -> n = 0,
        // j = 0, so ex2 = T[0] = 1.0 for either sign
        assert_eq!(ex2_reduce(f(2f32.powi(-64))), Some((0, 0)));
        assert_eq!(ex2_reduce(f(2f32.powi(-63))), Some((0, 0)));
        for e in 1u64..=103 {
            for m in [0, 1, 0x40_0000, MANT] {
                for s in [0, SIGN] {
                    let x = s | e << 23 | m;
                    assert_eq!(ex2_reduce(x), Some((0, 0)), "{x:#010x}");
                    assert_eq!(mufu_ex2_tab(x, F32_ONE), Some(F32_ONE), "{x:#010x}");
                }
            }
        }
        assert_eq!(ex2_reduce(f(2f32.powi(-23))), Some((0, 1)));
        assert_eq!(ex2_reduce(f(-(2f32.powi(-23)))), Some((-1, MANT - 1)));
        assert_eq!(ex2_reduce(0), None);
        assert_eq!(ex2_reduce(EXP), None);
        assert_eq!(ex2_reduce(f(2f32.powi(30))), None);
        let one_row = 1u64 << 32 | F32_ONE; // row 1 supplied where row 0 is read
        assert_eq!(mufu_ex2_tab(0, one_row), None);
        assert_eq!(mufu_ex2_tab(0, F32_ONE), Some(F32_ONE)); // row 0 = 1.0 (ex2(0))
        assert_eq!(mufu_ex2_tab(F32_ONE, F32_ONE), Some(f(2.0)));
        assert_eq!(mufu_ex2_tab(f(-1.0), F32_ONE), Some(f(0.5)));
        assert_eq!(mufu_ex2_tab(NEG_INF, F32_ONE), Some(0));
        assert_eq!(mufu_ex2_tab(EXP, F32_ONE), Some(EXP));
        assert_eq!(mufu_ex2_tab(0x7F80_0001, F32_ONE), Some(MUFU_NAN));
        assert_eq!(mufu_ex2_tab(f(-200.0), F32_ONE), Some(0)); // underflow -> +0
        assert_eq!(mufu_ex2_tab(f(200.0), F32_ONE), Some(EXP)); // overflow -> +inf
    }

    #[test]
    fn inv_sum_rule() {
        let row0 = F32_ONE; // R[0] = rcp(1.0) = 1.0
        assert_eq!(fa2_inv_sum_tab(0, row0), Some(F32_ONE));
        assert_eq!(fa2_inv_sum_tab(SIGN, row0), Some(F32_ONE));
        assert_eq!(fa2_inv_sum_tab(0x7FC0_0000, (0x40_0000u64 << 32) | F32_ONE), Some(F32_ONE)); // NaN: row = its mantissa
        assert_eq!(fa2_inv_sum_tab(0x7FC0_0000, row0), None);
        assert_eq!(fa2_inv_sum_tab(0x7F80_0001, 1u64 << 32 | F32_ONE), Some(F32_ONE)); // NaN payload 1 -> row 1
        assert_eq!(fa2_inv_sum_tab(0x7F80_0001, row0), None);
        assert_eq!(fa2_inv_sum_tab(f(2.0), row0), Some(f(0.5)));
        assert_eq!(fa2_inv_sum_tab(f(-4.0), row0), Some(f(-0.25)));
        assert_eq!(fa2_inv_sum_tab(EXP, row0), Some(0));
        assert_eq!(fa2_inv_sum_tab(NEG_INF, row0), Some(SIGN));
        assert_eq!(fa2_inv_sum_tab(0x0000_0001, 1u64 << 32 | F32_ONE), Some(EXP)); // subnormal -> +inf
        assert_eq!(fa2_inv_sum_tab(f(2f32.powi(127)), row0), Some(0)); // rcp = 2^-127 is subnormal: ftz -> +0
        assert_eq!(fa2_inv_sum_tab(f(2f32.powi(-126)), row0), Some(f(2f32.powi(126))));
    }
}
