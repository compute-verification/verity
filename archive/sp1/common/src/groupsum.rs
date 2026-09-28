//! Hawkeye's grouped, truncating accumulation for any operand format, bit for
//! bit as `verity.ml.circuits.silicon.models.GroupSum.step` computes it (`group_sum`,
//! `term.fp32_term`, `term.pack_fp32`): the model behind every non-block-scaled
//! Table 2 row.  [`crate::tc`] is the same function specialised to the Ampere
//! pipeline and BF16 products; this one takes the pipeline's groups, adder
//! width and exponent floor as data and the product as a closure, so the FP8
//! (`tc_fp8`) and Hopper BF16 (`tc_hopper_bf16`) rows share it.
//!
//! An adder narrower than the FP32 significand (`width < 24`: the 14-bit FP8
//! adder) truncates every term -- the incoming accumulator included -- to
//! `width` bits before the aligned sum and scales the normalised result back
//! up exactly (the Python's negative `rescale`).
//!
//! Widths: products are carried on the 25-bit grid (`term.PRODUCT_WIDTH`) and
//! the accumulator has 24 bits, so a term is below `2^27` after the rescale of
//! the widest adder here (26 bits) and a group of at most [`MAX_GROUP`]
//! products plus the accumulator sums below `2^33`: `i64` is exact, and every
//! shift by a distance of at least 64 is made explicit (it yields `0`, as
//! Python's big-integer shifts do).  No allocation.

/// An unnormalised signed binary term `(-1)^negative * magnitude * 2^(exponent - 23)`
/// (`term.Term`); `magnitude == 0` is the (unsigned) zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub negative: bool,
    pub exponent: i32,
    pub magnitude: u32,
}

/// `term.ZERO`.
pub const ZERO: Term = Term { negative: false, exponent: 0, magnitude: 0 };

/// The largest group any pipeline here takes (Hopper FP8: one group of 32).
pub const MAX_GROUP: usize = 32;

const FP32_MIN_NORMAL_EXPONENT: i32 = -126;
const FP32_MAX_EXPONENT: i32 = 127;
const FP32_SIGNIFICAND_WIDTH: i32 = 24;

/// The silicon parameters of one grouped-sum shape (`models.GroupSum`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSum {
    /// How many products each grouped sum takes, in k order (each at most [`MAX_GROUP`]).
    pub groups: &'static [usize],
    /// The internal significand width of the adder.
    pub width: u32,
    /// The exponent floor (the group maximum when every term is zero).
    pub zero_exponent: i32,
}

impl GroupSum {
    /// Operand pairs per instruction step.
    pub const fn k(&self) -> usize {
        let mut k = 0;
        let mut i = 0;
        while i < self.groups.len() {
            k += self.groups[i];
            i += 1;
        }
        k
    }
}

/// `term.fp32_term`: an FP32 word as a 24-bit term; `None` if non-finite.
#[inline]
pub fn fp32_term(bits: u32) -> Option<Term> {
    let field = (bits >> 23) & 0xFF;
    if field == 0xFF {
        return None;
    }
    let mantissa = bits & 0x7F_FFFF;
    let negative = bits >> 31 == 1;
    if field == 0 {
        if mantissa == 0 {
            return Some(ZERO);
        }
        return Some(Term { negative, exponent: FP32_MIN_NORMAL_EXPONENT, magnitude: mantissa });
    }
    Some(Term { negative, exponent: field as i32 - 127, magnitude: (1 << 23) | mantissa })
}

/// `term.pack_fp32`: a term as FP32 bits (zero is `+0`; an exponent above 127
/// is `+-inf`; a magnitude without bit 23 at exponent `-126` is a subnormal).
#[inline]
pub fn pack_fp32(term: Term) -> u32 {
    if term.magnitude == 0 {
        return 0;
    }
    let sign = (term.negative as u32) << 31;
    if term.exponent > FP32_MAX_EXPONENT {
        return sign | 0x7F80_0000;
    }
    let mut field = term.exponent + 127;
    if term.magnitude & (1 << 23) == 0 {
        field -= 1;
    }
    sign | ((field as u32) << 23) | (term.magnitude & 0x7F_FFFF)
}

/// The number of significant bits of `1 <= m < 2^32` (the kernels' group sums; `leading_zeros` is a long
/// software sequence on the zkVM's RV64IM).
#[inline(always)]
pub(crate) fn bit_length(m: u64) -> i32 {
    let (mut v, mut n) = (m, 0i32);
    if v >> 16 != 0 {
        v >>= 16;
        n += 16;
    }
    if v >> 8 != 0 {
        v >>= 8;
        n += 8;
    }
    if v >> 4 != 0 {
        v >>= 4;
        n += 4;
    }
    if v >> 2 != 0 {
        v >>= 2;
        n += 2;
    }
    if v >> 1 != 0 {
        v >>= 1;
        n += 1;
    }
    n + v as i32
}

/// [`bit_length`] of `1 <= m < 2^64`.
#[inline(always)]
pub(crate) fn bit_length64(m: u64) -> i32 {
    if m >> 32 != 0 {
        32 + bit_length(m >> 32)
    } else {
        bit_length(m)
    }
}

#[inline(always)]
fn shr64(m: u64, s: i32) -> u64 {
    if s >= 64 {
        0
    } else {
        m >> s
    }
}

/// `models.group_sum`: one grouped, truncating accumulation step of an adder of
/// `width` significand bits with exponent floor `zero_exponent`.
#[inline]
pub fn group_sum(terms: &[Term], width: u32, zero_exponent: i32) -> Term {
    let rescale = width as i32 - FP32_SIGNIFICAND_WIDTH;
    let mut max_exponent = zero_exponent;
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
        let m = term.magnitude as u64;
        let scaled = if rescale >= 0 { m << rescale } else { m >> -rescale };
        let aligned = shr64(scaled, max_exponent - term.exponent) as i64;
        if term.negative {
            total -= aligned;
        } else {
            total += aligned;
        }
    }
    if total == 0 {
        return ZERO;
    }
    let negative = total < 0;
    let mut magnitude = total.unsigned_abs();
    let bit_length = 64 - magnitude.leading_zeros() as i32;
    let width = width as i32;
    let mut exponent = max_exponent + bit_length - width;
    if bit_length > width {
        magnitude >>= bit_length - width;
    } else {
        magnitude <<= width - bit_length;
    }
    if exponent < FP32_MIN_NORMAL_EXPONENT {
        magnitude = shr64(magnitude, FP32_MIN_NORMAL_EXPONENT - exponent);
        exponent = FP32_MIN_NORMAL_EXPONENT;
    }
    let magnitude = if rescale >= 0 { magnitude >> rescale } else { magnitude << -rescale };
    if magnitude == 0 {
        return ZERO;
    }
    Term { negative, exponent, magnitude: magnitude as u32 }
}

/// `models.GroupSum.step`: `D = acc + sum_k a[k] * b[k]` exactly as `pipeline`'s
/// silicon computes it, on an FP32 accumulator word and `pipeline.k()` operand
/// words each side, with `product` the format's exact product (`None` for a
/// non-finite operand).  `None` for a non-finite accumulator or operand, or
/// operand slices of the wrong length.  A group whose result leaves the FP32
/// range saturates to an infinity of its sign; the remaining groups' operands
/// are then not examined (as in the Python).
#[inline]
pub fn step<W: Copy>(
    pipeline: &GroupSum,
    acc_bits: u32,
    a: &[W],
    b: &[W],
    product: impl Fn(W, W) -> Option<Term>,
) -> Option<u32> {
    if a.len() != pipeline.k() || b.len() != pipeline.k() {
        return None;
    }
    let mut acc = fp32_term(acc_bits)?;
    let mut terms = [ZERO; MAX_GROUP + 1];
    let mut start = 0;
    for &size in pipeline.groups {
        terms[0] = acc;
        for i in 0..size {
            terms[1 + i] = product(a[start + i], b[start + i])?;
        }
        acc = group_sum(&terms[..1 + size], pipeline.width, pipeline.zero_exponent);
        if acc.magnitude != 0 && acc.exponent > FP32_MAX_EXPONENT {
            return Some(pack_fp32(acc));
        }
        start += size;
    }
    Some(pack_fp32(acc))
}

/// `models.tc_dot_chain` from `c_0 = +0`: the final FP32 accumulator word of a
/// `GemmAccumulator<K>` over `pipeline` (`K = a.len()`, a positive multiple of
/// `pipeline.k()`); `None` where any step is outside the model.
#[inline]
pub fn chain<W: Copy>(
    pipeline: &GroupSum,
    a: &[W],
    b: &[W],
    product: impl Fn(W, W) -> Option<Term> + Copy,
) -> Option<u32> {
    let k = pipeline.k();
    if a.len() != b.len() || a.is_empty() || a.len() % k != 0 {
        return None;
    }
    let mut acc = 0u32;
    for s in 0..a.len() / k {
        acc = step(pipeline, acc, &a[s * k..(s + 1) * k], &b[s * k..(s + 1) * k], product)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tc;

    fn to_tc(t: Term) -> tc::Term {
        tc::Term { negative: t.negative, exponent: t.exponent as i64, magnitude: t.magnitude as u64 }
    }

    /// A small deterministic generator (xorshift64*), so the tests need no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// This module's `group_sum` equals `tc::group_sum` (the Ampere port, u128 arithmetic) at every width a
    /// Table 2 pipeline uses, on random terms spanning subnormal results, cancellation and wide exponent gaps.
    #[test]
    fn group_sum_agrees_with_the_ampere_port_at_every_width() {
        let mut rng = Rng(0x5EED_0001);
        for &(width, floor, n) in &[(25u32, -132i32, 9usize), (26, -133, 17), (14, -139, 17), (14, -139, 33)] {
            for _ in 0..20_000 {
                let mut terms = [ZERO; 33];
                for t in terms.iter_mut().take(n) {
                    let r = rng.next();
                    let magnitude = match r % 4 {
                        0 => 0,
                        1 => (r >> 8) as u32 & 0xFF_FFFF,
                        _ => ((r >> 8) as u32 & 0x1FF_FFFF) | 1,
                    };
                    let exponent = -150 + ((r >> 40) % 300) as i32;
                    *t = Term { negative: (r >> 63) == 1, exponent, magnitude };
                }
                let ours = group_sum(&terms[..n], width, floor);
                let tcs: Vec<tc::Term> = terms[..n].iter().map(|&t| to_tc(t)).collect();
                let theirs = tc::group_sum(&tcs, width, floor as i64);
                assert_eq!(to_tc(ours), theirs, "width {width} terms {:?}", &terms[..n]);
            }
        }
    }

    #[test]
    fn bit_length_below_2_32() {
        for m in (1u64..1 << 20).chain((0..32).flat_map(|b| [1u64 << b, (1u64 << b) | 1, (2u64 << b) - 1])) {
            assert_eq!(bit_length(m), 64 - m.leading_zeros() as i32, "{m}");
        }
        for m in (1u64..1 << 16).chain((0..64).flat_map(|b| [1u64 << b, (1u64 << b) | 1, (1u64 << b).wrapping_mul(2).wrapping_sub(1)])) {
            assert_eq!(bit_length64(m), 64 - m.leading_zeros() as i32, "{m}");
        }
    }

    #[test]
    fn fp32_round_trip() {
        for bits in [0u32, 0x8000_0000, 1, 0x007F_FFFF, 0x0080_0000, 0x3F80_0000, 0xBF80_0001, 0x7F7F_FFFF] {
            let t = fp32_term(bits).unwrap();
            let expect = if bits == 0x8000_0000 { 0 } else { bits };
            assert_eq!(pack_fp32(t), expect, "{bits:#x}");
        }
        assert_eq!(fp32_term(0x7F80_0000), None);
        assert_eq!(fp32_term(0xFFC0_0000), None);
    }
}
