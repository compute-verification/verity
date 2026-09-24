//! The H100 BF16 Table 2 row (`bf16-hopper`): `GemmCoordinate<1536>` over
//! `mma.sync.m16n8k16.bf16` on the H100, bit for bit as
//! `verity.ml.tc.models.HOPPER_BF16_M16N8K16` computes it -- one group of 16
//! products, a 26-bit adder, exponent floor `-133` (Hawkeye's Hopper
//! parameters, re-measured on the mma.sync path by lane tc-probe-fp8) -- then
//! the software round-to-nearest-even cast to BF16 (`cast.f32_to_bf16_word`,
//! NaN -> `0x7FC0`), the word the relation publishes
//! (`backends/direct/ligero/relations.py`, `y_public=f32_to_bf16_word`).
//!
//! A VU is `1536 / 16 = 96` chained steps from `c_0 = +0`.  The product is the
//! BF16 product of [`crate::tc::bf16_product`] (Hawkeye's
//! `multiply_bfloat16_to_gfloat`, the same for every BF16 pipeline); the
//! accumulation is [`crate::groupsum`] at width 26.

use crate::groupsum::{self, GroupSum, Term, ZERO};

/// `models.HOPPER_BF16_M16N8K16`.
pub const HOPPER_BF16_M16N8K16: GroupSum = GroupSum { groups: &[16], width: 26, zero_exponent: -133 };

/// Elements per VU row (`relations.K_VU`).
pub const K_VU: usize = 1536;
/// Bytes of one VU row of little-endian BF16 words.
pub const ROW_BYTES: usize = 2 * K_VU;

/// `term.bf16_product`: the exact product of two BF16 words as a 25-bit-scaled
/// term; `None` for a non-finite operand (exponent field `0xFF`).
#[inline(always)]
pub fn bf16_product(a: u16, b: u16) -> Option<Term> {
    let ea = ((a >> 7) & 0xFF) as u32;
    let eb = ((b >> 7) & 0xFF) as u32;
    if ea == 0xFF || eb == 0xFF {
        return None;
    }
    let sa = (a & 0x7F) as u32 | if ea != 0 { 0x80 } else { 0 };
    let sb = (b & 0x7F) as u32 | if eb != 0 { 0x80 } else { 0 };
    let magnitude = (sa * sb) << 9;
    if magnitude == 0 {
        return Some(ZERO);
    }
    Some(Term {
        negative: (a ^ b) >> 15 == 1,
        exponent: ea.max(1) as i32 + eb.max(1) as i32 - 254,
        magnitude,
    })
}

/// One instruction step (`GroupSum.step`) on BF16 operands.
pub fn step(acc: u32, a: &[u16], b: &[u16]) -> Option<u32> {
    groupsum::step(&HOPPER_BF16_M16N8K16, acc, a, b, bf16_product)
}

/// The final FP32 accumulator of one VU (the chain from `+0`); `None` for rows
/// that are not `K_VU` words each or a non-finite operand.
pub fn accumulator(x: &[u16], w: &[u16]) -> Option<u32> {
    if x.len() != K_VU || w.len() != K_VU {
        return None;
    }
    groupsum::chain(&HOPPER_BF16_M16N8K16, x, w, bf16_product)
}

/// The public word of one VU: `f32_to_bf16(accumulator)` ([`crate::tc::f32_to_bf16`], the software cast).
pub fn vu(x: &[u16], w: &[u16]) -> Option<u16> {
    accumulator(x, w).map(|acc| crate::tc::f32_to_bf16(acc as u64) as u16)
}

/// [`vu`] on rows of little-endian BF16 words ([`ROW_BYTES`] each), by the kernel.
pub fn vu_bytes(x: &[u8], w: &[u8]) -> Option<u16> {
    if x.len() != ROW_BYTES || w.len() != ROW_BYTES {
        return None;
    }
    const G: usize = 16;
    const STEPS: usize = K_VU / G;
    let mut acc = ZERO;
    for s in 0..STEPS {
        let mut bad = false;
        let at = 2 * G * s;
        acc = group(acc, &x[at..at + 2 * G], &w[at..at + 2 * G], &mut bad);
        if bad {
            return None;
        }
        if acc.magnitude != 0 && acc.exponent > 127 {
            if s != STEPS - 1 {
                return None;
            }
            break;
        }
    }
    Some(crate::tc::f32_to_bf16(groupsum::pack_fp32(acc) as u64) as u16)
}

// --- the kernel ------------------------------------------------------------------------------------------------------
//
// `groupsum::step` specialised to one group of 16 BF16 products, a 26-bit adder and floor -133: a group's nonzero
// products kept as packed words and summed in a second pass once the group maximum is known, the accumulator
// carried as a `Term` between steps (exact: see `tc_fp8`'s kernel).

const FLOOR: i32 = -133;
const ADDER_WIDTH: i32 = 26;
/// `ADDER_WIDTH - FP32_SIGNIFICAND_WIDTH`: every term is scaled up by this many bits before the aligned sum.
const RESCALE: u32 = 2;
/// A product's `E = max(ea, 1) + max(eb, 1)` is its `Term` exponent + 254.
const E_BIAS: i32 = 254;

/// `group_sum(acc, products of 16 BF16 word pairs)` on little-endian row bytes (32 each); `bad` is set when an
/// operand is non-finite.  A product `p = sa * sb < 2^16` is `p << 9` on the 25-bit grid, `p << 11` in the adder.
#[inline(always)]
fn group(acc: Term, a: &[u8], b: &[u8], bad: &mut bool) -> Term {
    let (a, b): (&[u8; 32], &[u8; 32]) = (a.try_into().unwrap(), b.try_into().unwrap());
    let mut kept = [0u32; 16];
    let mut n = 0;
    let mut max_e = 0u32;
    for i in 0..16 {
        let ua = a[2 * i] as u32 | (a[2 * i + 1] as u32) << 8;
        let ub = b[2 * i] as u32 | (b[2 * i + 1] as u32) << 8;
        let ea = (ua >> 7) & 0xFF;
        let eb = (ub >> 7) & 0xFF;
        if ea == 0xFF || eb == 0xFF {
            *bad = true;
        }
        let sa = (ua & 0x7F) | ((ea != 0) as u32) << 7;
        let sb = (ub & 0x7F) | ((eb != 0) as u32) << 7;
        let p = sa * sb;
        if p != 0 {
            // p | E << 16 | sign << 31; E <= 508
            let e = ea + (ea == 0) as u32 + eb + (eb == 0) as u32;
            if e > max_e {
                max_e = e;
            }
            kept[n] = p | e << 16 | ((ua ^ ub) & 0x8000) << 16;
            n += 1;
        }
    }
    let mut max_exp = if n > 0 { max_e as i32 - E_BIAS } else { FLOOR };
    if acc.magnitude != 0 && acc.exponent > max_exp {
        max_exp = acc.exponent;
    }
    let mut total: i64 = 0;
    if acc.magnitude != 0 {
        // acc.magnitude << 2 < 2^26 vanishes from 26
        let d = (max_exp - acc.exponent) as u32;
        let v = if d < 26 { ((acc.magnitude << RESCALE) >> d) as i64 } else { 0 };
        total = if acc.negative { -v } else { v };
    }
    // p << 11 < 2^27 vanishes from 27
    let top = max_exp + E_BIAS;
    for &t in &kept[..n] {
        let d = (top - ((t >> 16) & 0x7FFF) as i32) as u32;
        if d < 27 {
            let v = (((t & 0xFFFF) << 11) >> d) as i64;
            if t >> 31 != 0 {
                total -= v;
            } else {
                total += v;
            }
        }
    }
    if total == 0 {
        return ZERO;
    }
    let negative = total < 0;
    let mut m = total.unsigned_abs();
    let bits = 64 - m.leading_zeros() as i32;
    let mut exponent = max_exp + bits - ADDER_WIDTH;
    if bits > ADDER_WIDTH {
        m >>= bits - ADDER_WIDTH;
    } else {
        m <<= ADDER_WIDTH - bits;
    }
    if exponent < -126 {
        let s = -126 - exponent;
        m = if s >= 64 { 0 } else { m >> s };
        exponent = -126;
    }
    let m = m >> RESCALE;
    if m == 0 {
        return ZERO;
    }
    Term { negative, exponent, magnitude: m as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tc;

    const ONE: u16 = 0x3F80;

    #[test]
    fn product_matches_the_ampere_port() {
        for a in (0u32..=0xFFFF).step_by(97) {
            for b in (0u32..=0xFFFF).step_by(89) {
                let ours = bf16_product(a as u16, b as u16);
                let theirs = tc::bf16_product(a as u64, b as u64);
                match (ours, theirs) {
                    (None, None) => {}
                    (Some(o), Some(t)) => {
                        assert_eq!((o.negative, o.exponent as i64, o.magnitude as u64), (t.negative, t.exponent, t.magnitude))
                    }
                    other => panic!("{a:#x} {b:#x}: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn exact_small_steps() {
        let ones = [ONE; 16];
        assert_eq!(step(0, &ones, &ones), Some(0x4180_0000)); // 16.0
        assert_eq!(step(0x3F80_0000, &ones, &ones), Some(0x4188_0000)); // 17.0
        assert_eq!(step(0x8000_0000, &[0; 16], &[0; 16]), Some(0));
        // 2^24 + 1 + 1: each 1 aligns to 2 on the 26-bit grid of the 2^24 term, the sum 2^24 + 2 is exact
        let mut a = [0u16; 16];
        a[0] = 0x4B80; // 2^24
        a[1] = ONE;
        a[2] = ONE;
        let mut b = [0u16; 16];
        b[..3].copy_from_slice(&[ONE; 3]);
        assert_eq!(step(0, &a, &b), Some(0x4B80_0001));
        assert_eq!(step(0, &[0x7F80; 16], &ones), None);
    }

    #[test]
    fn vu_casts_to_bf16() {
        let x = vec![ONE; K_VU];
        assert_eq!(accumulator(&x, &x), Some(0x44C0_0000)); // 1536.0
        assert_eq!(vu(&x, &x), Some(0x44C0));
        assert_eq!(vu(&x[..16], &x[..16]), None);
        let bytes: Vec<u8> = x.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(vu_bytes(&bytes, &bytes), Some(0x44C0));
        assert_eq!(vu_bytes(&bytes[..32], &bytes[..32]), None);
    }

    /// splitmix64.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A BF16 word with a random sign and mantissa, an exponent field from `lo..=hi`; zero / subnormal /
    /// non-finite at the given per-mille rates.
    fn word(r: &mut Rng, lo: u64, hi: u64, zero_pm: u64, sub_pm: u64, nonfinite_pm: u64) -> u16 {
        let sign = (r.next() & 1) << 15;
        let roll = r.below(1000);
        let mant = r.next() & 0x7F;
        let v = if roll < zero_pm {
            0
        } else if roll < zero_pm + sub_pm {
            mant.max(1)
        } else if roll < zero_pm + sub_pm + nonfinite_pm {
            (0xFF << 7) | if r.next() & 1 == 0 { 0 } else { mant.max(1) }
        } else {
            ((lo + r.below(hi - lo + 1)) << 7) | mant
        };
        (sign | v) as u16
    }

    /// The kernel equals `vu` on rows from several regimes: the frozen set's exponents, tiny magnitudes (subnormal
    /// results, the floor), wide exponents (saturation), values near the top, sparse non-finite words, mostly
    /// zeros, exact cancellation, and saturation or a non-finite word inside the last step.
    #[test]
    fn kernel_matches_the_reference() {
        let mut r = Rng(0xB16_2026_0924);
        let cases = if cfg!(debug_assertions) { 300 } else { 6000 };
        let mut counts = [0usize; 3]; // [finite, +-inf, rejected]
        for case in 0..cases {
            let regime = case % 9;
            let (mut x, mut w) = (vec![0u16; K_VU], vec![0u16; K_VU]);
            for k in 0..K_VU {
                (x[k], w[k]) = match regime {
                    0 => (word(&mut r, 110, 135, 5, 0, 0), word(&mut r, 110, 135, 5, 0, 0)),
                    1 => (word(&mut r, 0, 20, 50, 200, 0), word(&mut r, 0, 20, 50, 200, 0)),
                    2 => (word(&mut r, 0, 254, 20, 20, 0), word(&mut r, 60, 190, 20, 20, 0)),
                    3 => (word(&mut r, 180, 254, 0, 0, 0), word(&mut r, 120, 140, 0, 0, 0)),
                    4 => (word(&mut r, 100, 150, 10, 10, 1), word(&mut r, 100, 150, 10, 10, 1)),
                    5 => (word(&mut r, 100, 150, 900, 30, 0), word(&mut r, 100, 150, 500, 30, 0)),
                    6 => (word(&mut r, 0, 70, 10, 100, 0), word(&mut r, 55, 70, 10, 100, 0)),
                    _ => (word(&mut r, 115, 140, 5, 5, 0), word(&mut r, 115, 140, 5, 5, 0)),
                };
            }
            match regime {
                7 => {
                    for k in (0..K_VU).step_by(2) {
                        if r.next() & 3 != 0 {
                            x[k + 1] = x[k] ^ 0x8000;
                            w[k + 1] = w[k];
                        }
                    }
                }
                8 => {
                    let k = K_VU - 16 + r.below(16) as usize;
                    match (case / 9) % 3 {
                        0 => (x[k], w[k]) = (0x7F7F, 0x7F7F),
                        1 => x[k] = 0x7FC0,
                        _ => {
                            x[k - 16] = 0x7F7F;
                            w[k - 16] = 0x7F7F;
                        }
                    }
                }
                _ => {}
            }
            let want = vu(&x, &w);
            let (xb, wb): (Vec<u8>, Vec<u8>) =
                (x.iter().flat_map(|v| v.to_le_bytes()).collect(), w.iter().flat_map(|v| v.to_le_bytes()).collect());
            assert_eq!(vu_bytes(&xb, &wb), want, "case {case} (regime {regime})");
            counts[match want {
                None => 2,
                Some(v) if v & 0x7FFF == 0x7F80 => 1,
                Some(_) => 0,
            }] += 1;
        }
        eprintln!("hopper bf16 kernel: {} finite, {} +-inf, {} rejected", counts[0], counts[1], counts[2]);
        assert!(counts.iter().all(|&c| c > cases / 50), "every outcome exercised: {counts:?}");
    }
}
