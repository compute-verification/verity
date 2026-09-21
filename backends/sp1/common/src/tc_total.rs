//! Total semantics of one Ampere BF16 tensor-core step
//! (`veritor.tensor-core.ampere_bf16_m16n8k16@3`), bit for bit as
//! `veritor/core/silicon_total.py` and `lean/VeritorSpec/TCTotal.lean`.
//!
//! `tc_dot16(acc, a, b)` is `D = acc + sum_k a[k] * b[k]` for one accumulator
//! coordinate of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, as
//! the silicon computes it: two groups of eight exact products, a 25-bit
//! truncating adder with exponent floor `-132` (Hawkeye's parameters for the
//! finite domain), and, on every other encoding, the rules measured on the
//! A100 and the RTX 4090 (`docs/verified-backend/tc-total-semantics.md`):
//!
//! * a NaN operand or accumulator (quiet or signalling, any payload, either
//!   sign) gives the canonical NaN `0x7FFF_FFFF`;
//! * `inf * 0` (`±0` only; a subnormal is not a zero) gives NaN; `inf * x` for
//!   a nonzero finite or infinite `x` is an infinity with the XOR of the signs;
//! * within a group, infinities of both signs give NaN and infinities of one
//!   sign give that infinity whatever the finite terms; a finite group result
//!   beyond the FP32 range saturates to an infinity of its sign, which the
//!   next group treats as an ordinary infinity;
//! * a zero result is always `+0`.
//!
//! The function is total and deterministic on every `u32` accumulator and
//! every pair of sixteen `u16` operands; there is no error path.  `core` only,
//! no allocation, no floating point, no panicking arithmetic.

/// Bit 23's exponent of the smallest normal FP32 value.
const FP32_MIN_NORMAL_EXPONENT: i32 = -126;
/// Bit 23's exponent of the largest finite FP32 value.
const FP32_MAX_EXPONENT: i32 = 127;
/// Significand bits of an FP32 term.
const FP32_SIGNIFICAND_WIDTH: i32 = 24;
/// Significand width of the Ampere adder.
const WIDTH: i32 = 25;
/// Exponent floor of the Ampere adder (the alignment exponent when every term is zero).
const ZERO_EXPONENT: i32 = -132;
/// Products per group, in k order.
const GROUPS: [usize; 2] = [8, 8];

/// The canonical NaN word the step produces.
pub const FP32_CANONICAL_NAN: u32 = 0x7FFF_FFFF;
/// `+inf` as an FP32 word.
pub const FP32_INF: u32 = 0x7F80_0000;

/// An unnormalised signed binary term `(-1)^negative * magnitude * 2^(exponent - 23)`.
/// `magnitude == 0` is the (unsigned) zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Term {
    negative: bool,
    exponent: i32,
    magnitude: u64,
}

const ZERO: Term = Term {
    negative: false,
    exponent: 0,
    magnitude: 0,
};

/// A value of the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Val {
    NaN,
    Inf { negative: bool },
    Fin(Term),
}

#[inline]
fn shr_u64(x: u64, shift: i32) -> u64 {
    if shift >= 64 {
        0
    } else {
        x >> shift
    }
}

/// Exact product of two finite BF16 words as a 25-bit-scaled term
/// (`silicon.bf16_product`); the caller has excluded exponent field `0xFF`.
fn bf16_product_finite(a: u16, b: u16) -> Term {
    let ea = ((a >> 7) & 0xFF) as i32;
    let eb = ((b >> 7) & 0xFF) as i32;
    let sa = ((a & 0x7F) as u64) | if ea != 0 { 0x80 } else { 0 };
    let sb = ((b & 0x7F) as u64) | if eb != 0 { 0x80 } else { 0 };
    let magnitude = (sa * sb) << 9;
    if magnitude == 0 {
        return ZERO;
    }
    let ea = if ea == 0 { 1 } else { ea };
    let eb = if eb == 0 { 1 } else { eb };
    Term {
        negative: ((a ^ b) >> 15) & 1 == 1,
        exponent: ea + eb - 254,
        magnitude,
    }
}

/// `silicon_total.bf16_product_total`.
fn bf16_product(a: u16, b: u16) -> Val {
    let ea = (a >> 7) & 0xFF;
    let eb = (b >> 7) & 0xFF;
    let a_special = ea == 0xFF;
    let b_special = eb == 0xFF;
    if !a_special && !b_special {
        return Val::Fin(bf16_product_finite(a, b));
    }
    if (a_special && a & 0x7F != 0) || (b_special && b & 0x7F != 0) {
        return Val::NaN;
    }
    // at least one infinity, no NaN
    if a & 0x7FFF == 0 || b & 0x7FFF == 0 {
        return Val::NaN;
    }
    Val::Inf {
        negative: ((a ^ b) >> 15) & 1 == 1,
    }
}

/// `silicon_total.fp32_value`: the accumulator word as a value.
fn fp32_value(bits: u32) -> Val {
    let exponent_field = ((bits >> 23) & 0xFF) as i32;
    let mantissa = (bits & 0x7F_FFFF) as u64;
    let negative = bits >> 31 == 1;
    if exponent_field == 0xFF {
        return if mantissa != 0 {
            Val::NaN
        } else {
            Val::Inf { negative }
        };
    }
    if exponent_field == 0 {
        if mantissa == 0 {
            return Val::Fin(ZERO);
        }
        return Val::Fin(Term {
            negative,
            exponent: FP32_MIN_NORMAL_EXPONENT,
            magnitude: mantissa,
        });
    }
    Val::Fin(Term {
        negative,
        exponent: exponent_field - 127,
        magnitude: (1 << 23) | mantissa,
    })
}

/// `silicon.group_sum` for the Ampere parameters: one grouped, truncating
/// accumulation of the nonzero terms among `terms` (a `None` slot is absent).
fn group_sum(terms: &[Term]) -> Term {
    let rescale = WIDTH - FP32_SIGNIFICAND_WIDTH; // 1
    let mut max_exponent = ZERO_EXPONENT;
    for term in terms {
        if term.magnitude != 0 && term.exponent > max_exponent {
            max_exponent = term.exponent;
        }
    }
    let mut total: i64 = 0;
    for term in terms {
        if term.magnitude == 0 {
            continue;
        }
        let scaled = term.magnitude << rescale;
        let aligned = shr_u64(scaled, max_exponent - term.exponent) as i64;
        total += if term.negative { -aligned } else { aligned };
    }
    if total == 0 {
        return ZERO;
    }
    let negative = total < 0;
    let mut magnitude = total.unsigned_abs();
    let bit_length = 64 - magnitude.leading_zeros() as i32;
    let mut exponent = max_exponent + bit_length - WIDTH;
    if bit_length > WIDTH {
        magnitude >>= bit_length - WIDTH;
    } else {
        magnitude <<= WIDTH - bit_length;
    }
    if exponent < FP32_MIN_NORMAL_EXPONENT {
        magnitude = shr_u64(magnitude, FP32_MIN_NORMAL_EXPONENT - exponent);
        exponent = FP32_MIN_NORMAL_EXPONENT;
    }
    magnitude >>= rescale;
    if magnitude == 0 {
        return ZERO;
    }
    Term {
        negative,
        exponent,
        magnitude,
    }
}

/// `silicon_total.group_step_total`: the incoming value and up to eight products to the outgoing value.
fn group_step(acc: Val, products: &[Val]) -> Val {
    let mut any_nan = false;
    let mut pos_inf = false;
    let mut neg_inf = false;
    let mut finite = [ZERO; 9];
    let mut n = 0;
    for v in core::iter::once(&acc).chain(products.iter()) {
        match *v {
            Val::NaN => any_nan = true,
            Val::Inf { negative } => {
                if negative {
                    neg_inf = true
                } else {
                    pos_inf = true
                }
            }
            Val::Fin(t) => {
                finite[n] = t;
                n += 1;
            }
        }
    }
    if any_nan || (pos_inf && neg_inf) {
        return Val::NaN;
    }
    if pos_inf {
        return Val::Inf { negative: false };
    }
    if neg_inf {
        return Val::Inf { negative: true };
    }
    let total = group_sum(&finite[..n]);
    if total.magnitude != 0 && total.exponent > FP32_MAX_EXPONENT {
        Val::Inf {
            negative: total.negative,
        }
    } else {
        Val::Fin(total)
    }
}

/// `silicon.pack_fp32` of a finite term that is within range (zero is `+0`).
fn pack_fp32(term: Term) -> u32 {
    if term.magnitude == 0 {
        return 0;
    }
    if term.exponent > FP32_MAX_EXPONENT {
        return ((term.negative as u32) << 31) | FP32_INF;
    }
    let mut exponent_field = term.exponent + 127;
    if term.magnitude & (1 << 23) == 0 {
        exponent_field -= 1;
    }
    ((term.negative as u32) << 31)
        | ((exponent_field as u32) << 23)
        | (term.magnitude as u32 & 0x7F_FFFF)
}

/// `silicon_total.pack_value`.
fn pack_value(value: Val) -> u32 {
    match value {
        Val::NaN => FP32_CANONICAL_NAN,
        Val::Inf { negative } => ((negative as u32) << 31) | FP32_INF,
        Val::Fin(t) => pack_fp32(t),
    }
}

/// `tc_dot16`: one `mma.sync.m16n8k16` BF16 step on accumulator word `acc`
/// and sixteen operand pairs, on every encoding.
pub fn tc_dot16(acc: u32, a: &[u16; 16], b: &[u16; 16]) -> u32 {
    let mut value = fp32_value(acc);
    let mut start = 0;
    for size in GROUPS {
        let mut products = [Val::NaN; 8];
        for i in 0..size {
            products[i] = bf16_product(a[start + i], b[start + i]);
        }
        value = group_step(value, &products[..size]);
        start += size;
    }
    pack_value(value)
}

/// `tc_dot16_0`: `tc_dot16` with a `+0` accumulator.
pub fn tc_dot16_0(a: &[u16; 16], b: &[u16; 16]) -> u32 {
    tc_dot16(0, a, b)
}

/// The gate-set evaluation of `veritor.tensor-core.ampere_bf16_m16n8k16@3`:
/// `None` only for an unknown op, the wrong number of arguments or an
/// argument outside its width; a value for every other argument list.
pub fn evaluate(op: &str, args: &[u64]) -> Option<u32> {
    fn operands(words: &[u64]) -> Option<([u16; 16], [u16; 16])> {
        if words.len() != 32 || words.iter().any(|&w| w >= 1 << 16) {
            return None;
        }
        let mut a = [0u16; 16];
        let mut b = [0u16; 16];
        for i in 0..16 {
            a[i] = words[i] as u16;
            b[i] = words[16 + i] as u16;
        }
        Some((a, b))
    }
    match op {
        "tc_dot16" => {
            let (&acc, rest) = args.split_first()?;
            if acc >= 1 << 32 {
                return None;
            }
            let (a, b) = operands(rest)?;
            Some(tc_dot16(acc as u32, &a, &b))
        }
        "tc_dot16_0" => {
            let (a, b) = operands(args)?;
            Some(tc_dot16_0(&a, &b))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: u16 = 0x3F80;
    const MAX: u16 = 0x7F7F;
    const INF: u16 = 0x7F80;
    const NEG_INF: u16 = 0xFF80;
    const QNAN: u16 = 0x7FC0;

    fn row(fill: u16, at: &[(usize, u16)]) -> [u16; 16] {
        let mut r = [fill; 16];
        for &(k, w) in at {
            r[k] = w;
        }
        r
    }

    #[test]
    fn finite_cases() {
        assert_eq!(tc_dot16_0(&[ONE; 16], &[ONE; 16]), 0x4180_0000); // 16.0
        assert_eq!(tc_dot16(0x3F80_0000, &[ONE; 16], &[ONE; 16]), 0x4188_0000); // 17.0
        assert_eq!(tc_dot16(0x8000_0000, &[0; 16], &[0; 16]), 0); // -0 acc, zero products: +0
        assert_eq!(tc_dot16_0(&[0x8000; 16], &[ONE; 16]), 0); // -0 products: +0
        assert_eq!(tc_dot16_0(&[MAX; 16], &[ONE; 16]), 0x7F80_0000); // 16 * max overflows
        assert_eq!(
            tc_dot16_0(
                &row(0, &[(0, MAX), (1, MAX)]),
                &row(ONE, &[(0, MAX), (1, 0xFF7F)])
            ),
            0
        );
    }

    #[test]
    fn special_cases_measured_on_a100_and_rtx4090() {
        assert_eq!(tc_dot16_0(&[INF; 16], &[INF; 16]), 0x7F80_0000);
        assert_eq!(tc_dot16_0(&[INF; 16], &[NEG_INF; 16]), 0xFF80_0000);
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, INF)]), &row(ONE, &[(0, 0)])),
            FP32_CANONICAL_NAN
        ); // inf * 0
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, INF)]), &row(ONE, &[(0, 0x0001)])),
            0x7F80_0000
        ); // inf * subnormal
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, INF), (1, NEG_INF)]), &[ONE; 16]),
            FP32_CANONICAL_NAN
        );
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, INF), (8, NEG_INF)]), &[ONE; 16]),
            FP32_CANONICAL_NAN
        );
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, QNAN)]), &[ONE; 16]),
            FP32_CANONICAL_NAN
        );
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, 0xFF81)]), &[ONE; 16]),
            FP32_CANONICAL_NAN
        ); // -sNaN
        assert_eq!(
            tc_dot16(0xFFC0_0001, &[0; 16], &[0; 16]),
            FP32_CANONICAL_NAN
        ); // NaN acc
        assert_eq!(
            tc_dot16(0x7F80_0000, &row(0, &[(15, NEG_INF)]), &[ONE; 16]),
            FP32_CANONICAL_NAN
        );
        // group-1 overflow saturates; the saturated infinity meets group 2's operand
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, MAX), (1, NEG_INF)]), &row(ONE, &[(0, MAX)])),
            0xFF80_0000
        );
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, MAX), (8, NEG_INF)]), &row(ONE, &[(0, MAX)])),
            FP32_CANONICAL_NAN
        );
        assert_eq!(
            tc_dot16_0(&row(0, &[(0, MAX), (8, QNAN)]), &row(ONE, &[(0, MAX)])),
            FP32_CANONICAL_NAN
        );
        assert_eq!(
            tc_dot16_0(
                &row(0, &[(0, MAX), (8, 0xFF7F)]),
                &row(ONE, &[(0, MAX), (8, MAX)])
            ),
            0x7F80_0000
        );
    }

    #[test]
    fn evaluate_shapes() {
        assert_eq!(evaluate("tc_dot16_0", &[0x3F80; 32]), Some(0x4180_0000));
        assert_eq!(evaluate("tc_dot16_0", &[0x3F80; 31]), None);
        let mut wide_acc = [0x3F80u64; 33];
        wide_acc[0] = 1 << 32;
        assert_eq!(evaluate("tc_dot16", &wide_acc), None);
        wide_acc[0] = 0x3F80_0000;
        assert_eq!(evaluate("tc_dot16", &wide_acc), Some(0x4188_0000));
        assert_eq!(evaluate("tc_dot16_0", &[0x1_0000; 32]), None);
        assert_eq!(evaluate("mul8", &[1, 2]), None);
    }
}
