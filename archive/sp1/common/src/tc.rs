//! One Ampere `mma.sync.m16n8k16` step on BF16 operands with an FP32
//! accumulator, bit for bit as `veritor/core/silicon.py` models it and
//! `lean/VeritorSpec/TC.lean` specifies it (Hawkeye's Ampere parameters: two
//! groups of eight products, a 25-bit adder, exponent floor `-132`).
//!
//! Everything is integer arithmetic on the raw encodings: exact BF16 products
//! carried as unnormalised terms on a 25-bit grid ([`bf16_product`]), a grouped
//! truncating accumulation ([`group_sum`]), and the FP32 packing of the result
//! ([`pack_fp32`]).  Non-finite operands (a BF16 or FP32 word with exponent
//! field `0xFF`) are outside the modelled domain: the Python raises, the Lean
//! `Op.eval` is `none`, and here every function returns `None`, which
//! `GateSet::evaluate` turns into a rejection of the obligation.
//!
//! The FP32 -> BF16 cast of the `veritor.tc-ampere-bf16@1` family
//! ([`f32_to_bf16`]) is round-to-nearest-even with every NaN mapped to the
//! software canonical quiet NaN word `0x7FC0` (`silicon.f32_to_bf16`, the
//! word the pinned CUDA-core kernels `gpu/gpt2/pinned_ops.cu` store).
//! Version `@2` of the family ([`f32_to_bf16_hw`], lane L8) is the same RNE
//! with every NaN mapped to `0x7FFF`, the word `cvt.rn.bf16.f32` produces on
//! the A100 and the RTX 4090 for every measured NaN input (L6,
//! `docs/data/tc-total-2026-09-07/*/cvt_special.json`).  The two agree on
//! every non-NaN word ([`tests::the_two_casts_differ_only_on_nan`], exhaustive
//! over `2^32` in a release build).  [`bf16_to_f32`] is the exact widening.
//!
//! Widths: the magnitudes never exceed 2^30 (nine 26-bit aligned terms), so
//! `u64`/`i128` arithmetic is exact everywhere; shifts by more than the word
//! are made explicit (they yield `0`, as Python's big-integer shifts do).

/// An unnormalised signed binary term `(-1)^negative * magnitude * 2^(exponent - 23)`
/// (`silicon.Term`); `magnitude == 0` is the (unsigned) zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub negative: bool,
    pub exponent: i64,
    pub magnitude: u64,
}

/// `silicon.ZERO`.
pub const ZERO: Term = Term {
    negative: false,
    exponent: 0,
    magnitude: 0,
};

/// The silicon parameters of one `mma` shape (`silicon.Pipeline`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    /// How many products each grouped sum takes, in k order.
    pub groups: &'static [usize],
    /// The internal significand width of the adder.
    pub width: u32,
    /// The exponent floor (the maximum exponent used when every term is zero).
    pub zero_exponent: i64,
}

impl Pipeline {
    pub fn k(&self) -> usize {
        self.groups.iter().sum()
    }
}

/// `silicon.AMPERE_BF16_M16N8K16` (Hawkeye's A100 parameters; identical to the
/// RTX 4090 measurement `ADA_BF16_M16N8K16`).
pub const AMPERE_BF16_M16N8K16: Pipeline = Pipeline {
    groups: &[8, 8],
    width: 25,
    zero_exponent: -132,
};

/// `silicon.bf16_product`: the exact product of two BF16 words as a
/// 25-bit-scaled term; `None` for a non-finite operand.  A subnormal keeps its
/// unnormalised significand at exponent `-126`.
pub fn bf16_product(a: u64, b: u64) -> Option<Term> {
    let ea = (a >> 7) & 0xFF;
    let eb = (b >> 7) & 0xFF;
    if ea == 0xFF || eb == 0xFF {
        return None;
    }
    let sa = (a & 0x7F) | if ea != 0 { 0x80 } else { 0 };
    let sb = (b & 0x7F) | if eb != 0 { 0x80 } else { 0 };
    let magnitude = (sa * sb) << 9;
    if magnitude == 0 {
        return Some(ZERO);
    }
    let ea = if ea == 0 { 1 } else { ea } as i64;
    let eb = if eb == 0 { 1 } else { eb } as i64;
    Some(Term {
        negative: ((a ^ b) >> 15) & 1 == 1,
        exponent: ea + eb - 254,
        magnitude,
    })
}

/// `silicon.fp32_term`: an FP32 word as a 24-bit term; `None` if non-finite.
pub fn fp32_term(bits: u64) -> Option<Term> {
    let exponent_field = (bits >> 23) & 0xFF;
    if exponent_field == 0xFF {
        return None;
    }
    let mantissa = bits & 0x7F_FFFF;
    let negative = (bits >> 31) & 1 == 1;
    if exponent_field == 0 {
        if mantissa == 0 {
            return Some(ZERO);
        }
        return Some(Term {
            negative,
            exponent: -126,
            magnitude: mantissa,
        });
    }
    Some(Term {
        negative,
        exponent: exponent_field as i64 - 127,
        magnitude: (1 << 23) | mantissa,
    })
}

/// `silicon.pack_fp32`: a term as FP32 bits (zero is `+0`; an exponent above
/// `127` is `+-inf`; a magnitude without bit 23 at exponent `-126` is a
/// subnormal).  Only ever applied to the output of [`group_sum`], whose
/// exponent is at least `-126`, so the exponent field is never negative.
pub fn pack_fp32(term: Term) -> u64 {
    if term.magnitude == 0 {
        return 0;
    }
    let sign = (term.negative as u64) << 31;
    if term.exponent > 127 {
        return sign | 0x7F80_0000;
    }
    let mut exponent_field = term.exponent + 127;
    if term.magnitude & (1 << 23) == 0 {
        exponent_field -= 1;
    }
    debug_assert!(exponent_field >= 0);
    sign | ((exponent_field as u64) << 23) | (term.magnitude & 0x7F_FFFF)
}

/// `m << r` for `r >= 0`, `m >> -r` otherwise (`TC.shiftInt`).
fn shift_int(m: u128, r: i64) -> u128 {
    if r >= 0 {
        shl(m, r as u64)
    } else {
        shr(m, (-r) as u64)
    }
}

/// `m >> r` for `r >= 0`, `m << -r` otherwise (`TC.unshiftInt`).
fn unshift_int(m: u128, r: i64) -> u128 {
    if r >= 0 {
        shr(m, r as u64)
    } else {
        shl(m, (-r) as u64)
    }
}

fn shr(m: u128, s: u64) -> u128 {
    if s >= 128 {
        0
    } else {
        m >> s
    }
}

fn shl(m: u128, s: u64) -> u128 {
    // never reached with a shift that could overflow (rescale is 1 for the Ampere pipeline)
    if s >= 128 {
        0
    } else {
        m << s
    }
}

/// `silicon.group_sum`: one grouped, truncating accumulation step of an adder
/// of `width` significand bits with exponent floor `zero_exponent`.  Every term
/// is rescaled from the 24-bit FP32 grid to `width` bits, aligned to the
/// largest exponent by a truncating right shift, summed exactly; the sum is
/// normalised to `width` bits by truncation, denormalised if below the FP32
/// normal range, and truncated back to 24 bits.
pub fn group_sum(terms: &[Term], width: u32, zero_exponent: i64) -> Term {
    let rescale = width as i64 - 24;
    let mut max_exponent = zero_exponent;
    for term in terms {
        if term.magnitude != 0 && term.exponent > max_exponent {
            max_exponent = term.exponent;
        }
    }
    let mut total: i128 = 0;
    for term in terms {
        if term.magnitude == 0 {
            continue;
        }
        let scaled = shift_int(term.magnitude as u128, rescale);
        let aligned = shr(scaled, (max_exponent - term.exponent) as u64);
        if term.negative {
            total -= aligned as i128;
        } else {
            total += aligned as i128;
        }
    }
    if total == 0 {
        return ZERO;
    }
    let negative = total < 0;
    let mut magnitude: u128 = total.unsigned_abs();
    let bit_length = 128 - magnitude.leading_zeros() as i64;
    let mut exponent = max_exponent + bit_length - width as i64;
    if bit_length > width as i64 {
        magnitude >>= (bit_length - width as i64) as u32;
    } else {
        magnitude <<= (width as i64 - bit_length) as u32;
    }
    if exponent < -126 {
        magnitude = shr(magnitude, (-126 - exponent) as u64);
        exponent = -126;
    }
    let magnitude = unshift_int(magnitude, rescale);
    if magnitude == 0 {
        return ZERO;
    }
    Term {
        negative,
        exponent,
        magnitude: magnitude as u64,
    }
}

/// `silicon.tc_dot`: `D = acc + sum_k a[k] * b[k]` exactly as `pipeline`'s
/// silicon computes it, on an FP32 accumulator word and `k` operand words each
/// side (the caller has range-checked every word).  `None` for a non-finite
/// accumulator or operand.  A group whose result leaves the FP32 range
/// saturates to an infinity of its sign, sticky through the remaining groups,
/// whose operands are then not examined (as in the Python and the Lean).
pub fn tc_dot(pipeline: &Pipeline, acc_bits: u64, a: &[u64], b: &[u64]) -> Option<u64> {
    let k = pipeline.k();
    if a.len() != k || b.len() != k {
        return None;
    }
    let mut acc = fp32_term(acc_bits)?;
    let mut terms: Vec<Term> = Vec::with_capacity(1 + pipeline.groups.iter().copied().max().unwrap_or(0));
    let mut start = 0;
    for &size in pipeline.groups {
        terms.clear();
        terms.push(acc);
        for i in start..start + size {
            terms.push(bf16_product(a[i], b[i])?);
        }
        acc = group_sum(&terms, pipeline.width, pipeline.zero_exponent);
        if acc.magnitude != 0 && acc.exponent > 127 {
            return Some(pack_fp32(acc));
        }
        start += size;
    }
    Some(pack_fp32(acc))
}

/// The BF16 NaN word `f32_to_bf16` of `veritor.tc-ampere-bf16@1` produces (software canonical).
pub const BF16_NAN_SOFTWARE: u64 = 0x7FC0;
/// The BF16 NaN word `f32_to_bf16` of `veritor.tc-ampere-bf16@2` produces
/// (`cvt.rn.bf16.f32` on the A100 and the RTX 4090).
pub const BF16_NAN_HARDWARE: u64 = 0x7FFF;

/// Whether a 32-bit word is an FP32 NaN (exponent field `0xFF`, non-zero mantissa).
pub fn is_nan_f32(bits: u64) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0
}

/// `silicon.f32_to_bf16` on one word: round to nearest, ties to even; every
/// NaN becomes the canonical quiet NaN `0x7FC0`.  Total on 32-bit words.
pub fn f32_to_bf16(bits: u64) -> u64 {
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        return 0x7FC0;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
}

/// `silicon.f32_to_bf16_hw` / `silicon_total.cvt_rn_bf16_f32` on one word:
/// round to nearest, ties to even; every NaN becomes the hardware word
/// `0x7FFF`.  Total on 32-bit words; equal to [`f32_to_bf16`] off NaN.
pub fn f32_to_bf16_hw(bits: u64) -> u64 {
    if is_nan_f32(bits) {
        return BF16_NAN_HARDWARE;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
}

/// `silicon.bf16_to_f32` on one word: the exact widening (the low 16 bits are zero).
pub fn bf16_to_f32(word: u64) -> u64 {
    word << 16
}

#[cfg(test)]
mod tests {
    use super::*;

    const BF16_ONE: u64 = 0x3F80;
    const FP32_ONE: u64 = 0x3F80_0000;

    fn dot0(a: &[u64], b: &[u64]) -> Option<u64> {
        tc_dot(&AMPERE_BF16_M16N8K16, 0, a, b)
    }

    #[test]
    fn exact_small_cases_match_the_python_model() {
        let ones = [BF16_ONE; 16];
        let zeros = [0u64; 16];
        let neg_ones = [BF16_ONE | 0x8000; 16];
        assert_eq!(dot0(&ones, &ones), Some(0x4180_0000)); // 16.0
        assert_eq!(tc_dot(&AMPERE_BF16_M16N8K16, FP32_ONE, &ones, &ones), Some(0x4188_0000)); // 17.0
        assert_eq!(dot0(&zeros, &ones), Some(0));
        assert_eq!(tc_dot(&AMPERE_BF16_M16N8K16, 0x8000_0000, &zeros, &zeros), Some(0)); // -0 acc -> +0
        assert_eq!(dot0(&ones, &neg_ones), Some(0xC180_0000)); // -16.0
        let mut half = ones;
        half[8..].copy_from_slice(&neg_ones[..8]);
        assert_eq!(dot0(&half, &ones), Some(0)); // cancellation across groups
        assert_eq!(
            tc_dot(&AMPERE_BF16_M16N8K16, FP32_ONE, &ones, &neg_ones),
            Some(0xC170_0000) // 1 - 16 = -15
        );
    }

    #[test]
    fn non_finite_operands_are_rejected_and_overflow_is_sticky() {
        let ones = [BF16_ONE; 16];
        assert_eq!(dot0(&[0x7F80; 16], &[0x7F80; 16]), None);
        assert_eq!(tc_dot(&AMPERE_BF16_M16N8K16, 0x7F80_0000, &ones, &ones), None);
        assert_eq!(tc_dot(&AMPERE_BF16_M16N8K16, 0x7FC0_0000, &[0; 16], &[0; 16]), None);
        let max_finite = (254u64 << 7) | 127;
        assert_eq!(dot0(&[max_finite; 16], &ones), Some(0x7F80_0000)); // overflow to +inf
        // group-1 overflow: group 2's NaNs are never examined
        let mut a = [max_finite; 16];
        a[8..].copy_from_slice(&[0x7FC0; 8]);
        let mut b = [max_finite; 16];
        b[8..].copy_from_slice(&[0x7F80; 8]);
        assert_eq!(dot0(&a, &b), Some(0x7F80_0000));
        // a NaN in group 2 without an overflow before it is rejected
        let mut c = ones;
        c[8..].copy_from_slice(&[0x7FC0; 8]);
        assert_eq!(tc_dot(&AMPERE_BF16_M16N8K16, 0, &c, &ones), None);
    }

    #[test]
    fn casts() {
        assert_eq!(f32_to_bf16(0x3F80_0000), 0x3F80);
        assert_eq!(f32_to_bf16(0x3F80_8000), 0x3F80); // tie -> even (down)
        assert_eq!(f32_to_bf16(0x3F81_8000), 0x3F82); // tie -> even (up)
        assert_eq!(f32_to_bf16(0x3F80_8001), 0x3F81); // above the tie
        assert_eq!(f32_to_bf16(0x7F7F_FFFF), 0x7F80); // rounds up to +inf
        assert_eq!(f32_to_bf16(0xFF7F_FFFF), 0xFF80);
        assert_eq!(f32_to_bf16(0x7F80_0000), 0x7F80);
        assert_eq!(f32_to_bf16(0x7FC0_0000), 0x7FC0);
        assert_eq!(f32_to_bf16(0xFFFF_FFFF), 0x7FC0); // every NaN is the canonical word
        assert_eq!(f32_to_bf16(0x7F80_0001), 0x7FC0);
        assert_eq!(bf16_to_f32(0x3F80), 0x3F80_0000);
        assert_eq!(bf16_to_f32(0xFFFF), 0xFFFF_0000);
    }

    #[test]
    fn hardware_cast_matches_the_measured_nan_word_and_the_old_cast_elsewhere() {
        // finite and infinite words: exactly the `@1` cast
        for bits in [
            0x3F80_0000u64,
            0x3F80_8000,
            0x3F81_8000,
            0x3F80_8001,
            0x7F7F_FFFF,
            0xFF7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007F_FFFF,
        ] {
            assert!(!is_nan_f32(bits));
            assert_eq!(f32_to_bf16_hw(bits), f32_to_bf16(bits), "{bits:#010x}");
        }
        // the NaN classes L6 measured (`cvt_special.json`, both devices): every one is 0x7FFF
        for bits in [
            0x7FC0_0000u64, // +qNaN canonical
            0xFFC0_0000,    // -qNaN canonical
            0x7FC0_0001,    // +qNaN payload 1
            0x7FFF_FFFF,    // +qNaN all ones
            0xFFFF_FFFF,    // -qNaN all ones
            0x7F80_0001,    // +sNaN min payload
            0xFF80_0001,    // -sNaN min payload
            0x7FBF_FFFF,    // +sNaN max payload
            0xFFA0_0000,    // -sNaN mid
            0x7F80_8000,    // sNaN whose payload sits on the RNE tie bit
            0x7FC0_7FFF,    // qNaN whose payload sits just below the tie
            0xFFE5_6DC8,    // a measured random -qNaN
        ] {
            assert!(is_nan_f32(bits));
            assert_eq!(f32_to_bf16_hw(bits), BF16_NAN_HARDWARE, "{bits:#010x}");
            assert_eq!(f32_to_bf16(bits), BF16_NAN_SOFTWARE, "{bits:#010x}");
        }
    }

    /// The narrow delta, exhaustively over every 32-bit word: the two casts
    /// agree off NaN, and on NaN `@1` gives `0x7FC0` while `@2` gives `0x7FFF`
    /// (`lean/VeritorSpec/TCBHw.lean`: `castF32ToBf16Hw_eq_of_not_nan`,
    /// `castF32ToBf16Hw_of_nan`).  Runs in a release build (`cargo test
    /// --release -p veritor-zk-common`, a few seconds); ignored under debug
    /// assertions, where it takes minutes.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn the_two_casts_differ_only_on_nan() {
        let mut nan_words = 0u64;
        for bits in 0..=u32::MAX {
            let bits = bits as u64;
            let old = f32_to_bf16(bits);
            let new = f32_to_bf16_hw(bits);
            if is_nan_f32(bits) {
                nan_words += 1;
                assert_eq!(old, BF16_NAN_SOFTWARE, "{bits:#010x}");
                assert_eq!(new, BF16_NAN_HARDWARE, "{bits:#010x}");
            } else {
                assert_eq!(old, new, "{bits:#010x}");
            }
        }
        // 2 signs x (2^23 - 1) non-zero mantissas
        assert_eq!(nan_words, 2 * ((1 << 23) - 1));
    }
}
