//! The H100 BF16 Table 2 row (`bf16-hopper`): `GemmCoordinate<1536>` over
//! `mma.sync.m16n8k16.bf16` on the H100, bit for bit as
//! `verity.ml.circuits.silicon.models.HOPPER_BF16_M16N8K16` computes it -- one group of 16
//! products, a 26-bit adder, exponent floor `-133` (Hawkeye's Hopper
//! parameters, re-measured on the mma.sync path by lane tc-probe-fp8) -- then
//! the software round-to-nearest-even cast to BF16 (`cast.f32_to_bf16_word`,
//! NaN -> `0x7FC0`), the word the relation publishes
//! (`archive/direct/ligero/relations.py`, `y_public=f32_to_bf16_word`).
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

/// u64 words of one VU row: BF16 word `k` is bits `16 (k % 4)..` of u64 `k / 4`.
pub const ROW_WORDS: usize = K_VU / 4;

/// [`vu`] on rows of little-endian BF16 words ([`ROW_BYTES`] each), by the kernel.
pub fn vu_bytes(x: &[u8], w: &[u8]) -> Option<u16> {
    if x.len() != ROW_BYTES || w.len() != ROW_BYTES {
        return None;
    }
    let words = |row: &[u8]| -> Vec<u64> { row.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect() };
    vu_words(&words(x), &words(w))
}

/// [`vu`] on rows of [`ROW_WORDS`] u64 words each, by the kernel.
pub fn vu_words(x: &[u64], w: &[u64]) -> Option<u16> {
    let x: &[u64; ROW_WORDS] = x.try_into().ok()?;
    let w: &[u64; ROW_WORDS] = w.try_into().ok()?;
    const STEPS: usize = K_VU / 16;
    let mut acc = ZERO;
    for s in 0..STEPS {
        let mut bad = 0u64;
        let i = 4 * s;
        acc = group(acc, [x[i], x[i + 1], x[i + 2], x[i + 3]], [w[i], w[i + 1], w[i + 2], w[i + 3]], &mut bad);
        if bad & L100 != 0 {
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
// `groupsum::step` specialised to one group of 16 BF16 products, a 26-bit adder and floor -133, on four u64 words of
// each operand: the lane-parallel decode of `bare`'s Ampere kernel (two products per 64x64 -> 128 multiply), the
// group maximum, then the aligned sum; the accumulator is carried as a `Term` between steps (exact: see `tc_fp8`).

const LFF: u64 = 0x00FF_00FF_00FF_00FF;
const L7F: u64 = 0x007F_007F_007F_007F;
const L80: u64 = 0x0080_0080_0080_0080;
const L01: u64 = 0x0001_0001_0001_0001;
/// Bit 8 of a lane of `exponent field + 1`: the field was `0xFF` (a non-finite word).
const L100: u64 = 0x0100_0100_0100_0100;
/// Lanes 0 and 2 of a SWAR word, one byte each: two products per 64x64 -> 128 multiply.
const M02: u64 = 0x0000_00FF_0000_00FF;
/// A product's `max(ea, 1) + max(eb, 1) - 2` is its `Term` exponent + 252.
const BIAS: i32 = 252;
/// The adder's exponent floor `-133`, biased.
const FLOOR: u64 = (-133 + BIAS) as u64;
const ADDER_WIDTH: i32 = 26;
/// `ADDER_WIDTH - FP32_SIGNIFICAND_WIDTH`: every term is scaled up by this many bits before the aligned sum.
const RESCALE: u32 = 2;

/// Four products of a lane quadruple: significand products, biased exponents (lane `j` of the second word), and the
/// word whose bit `15 + 16 j` is product `j`'s sign.  `bad` collects bit 8 of `field + 1` for every operand.
#[inline(always)]
fn quad(x: u64, w: u64, bad: &mut u64) -> ([u64; 4], u64, u64) {
    let ex = (x >> 7) & LFF;
    let ew = (w >> 7) & LFF;
    *bad |= (ex + L01) | (ew + L01);
    // 0x80 in the lanes whose exponent field is non-zero: the implicit bit
    let ix = ((ex + LFF) >> 1) & L80;
    let iw = ((ew + LFF) >> 1) & L80;
    let sx = (x & L7F) | ix;
    let sw = (w & L7F) | iw;
    // max(e, 1) - 1 per operand, summed: at most 508 per lane, so no lane carries
    let g = (ex - (ix >> 7)) + (ew - (iw >> 7));
    let x02 = sx & M02;
    let x13 = (sx >> 16) & M02;
    let w02 = sw & M02;
    let w13 = (sw >> 16) & M02;
    // low 64 bits: p0 + (cross terms << 32), below 2^49, so the high 64 bits are exactly p2
    let t02 = x02 as u128 * w02 as u128;
    let t13 = x13 as u128 * w13 as u128;
    ([t02 as u64 & 0xFFFF, t13 as u64 & 0xFFFF, (t02 >> 64) as u64, (t13 >> 64) as u64], g, x ^ w)
}

/// `group_sum(acc, the 16 products of four word pairs)`.  A product `p = sa * sb < 2^16` is `p << 9` on the 25-bit
/// grid, `p << 11` in the adder.
#[inline(always)]
fn group(acc: Term, x: [u64; 4], w: [u64; 4], bad: &mut u64) -> Term {
    let mut p = [0u64; 16];
    let mut g = [0u64; 4];
    let mut s = [0u64; 4];
    for q in 0..4 {
        let (pq, gq, sq) = quad(x[q], w[q], bad);
        p[4 * q..4 * q + 4].copy_from_slice(&pq);
        g[q] = gq;
        s[q] = sq;
    }
    // a non-zero accumulator has exponent >= -126, above the floor
    let acc_e = (acc.exponent + BIAS) as u64;
    let mut mx = if acc.magnitude != 0 { acc_e } else { FLOOR };
    for q in 0..4 {
        for j in 0..4 {
            let e = (g[q] >> (16 * j)) & 0xFFFF;
            if p[4 * q + j] != 0 && e > mx {
                mx = e;
            }
        }
    }
    let mut total: i64 = 0;
    if acc.magnitude != 0 {
        // acc.magnitude << 2 < 2^26 vanishes from 26
        let d = mx - acc_e;
        if d < 26 {
            let v = (((acc.magnitude as u64) << RESCALE) >> d) as i64;
            total = if acc.negative { -v } else { v };
        }
    }
    // p << 11 < 2^27 vanishes from 27; a zero product whose exponent exceeds the maximum wraps `d` past 27 and adds
    // nothing, as `group_sum` skips it
    for q in 0..4 {
        for j in 0..4 {
            let d = mx.wrapping_sub((g[q] >> (16 * j)) & 0xFFFF);
            if d < 27 {
                let v = ((p[4 * q + j] << 11) >> d) as i64;
                if (s[q] >> (15 + 16 * j)) & 1 != 0 {
                    total -= v;
                } else {
                    total += v;
                }
            }
        }
    }
    let max_exp = mx as i32 - BIAS;
    if total == 0 {
        return ZERO;
    }
    let negative = total < 0;
    let mut m = total.unsigned_abs();
    let bits = groupsum::bit_length(m);
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

    /// One kernel step equals `groupsum::step` on single steps: accumulators zero, subnormal, small and large, and
    /// operands whose products straddle the floor (-133) and the FP32 subnormal range, as well as arbitrary words.
    #[test]
    fn group_matches_step() {
        let mut r = Rng(0xF100_2026);
        let trials = if cfg!(debug_assertions) { 20_000 } else { 1_000_000 };
        let mut below_floor = 0usize;
        for t in 0..trials {
            let acc_bits: u32 = match t % 5 {
                0 | 1 => (r.next() as u32) & 0x8000_0000,
                2 => (r.next() as u32) & 0x807F_FFFF,
                3 => ((r.next() as u32) & 0x81FF_FFFF) | 0x0080_0000,
                _ => loop {
                    let v = r.next() as u32;
                    if (v >> 23) & 0xFF != 0xFF {
                        break v;
                    }
                },
            };
            let (mut a, mut b) = ([0u16; 16], [0u16; 16]);
            for k in 0..16 {
                (a[k], b[k]) = if t % 3 == 2 {
                    (r.next() as u16, r.next() as u16)
                } else {
                    // ea + eb in 85..=140: products from 2^-169 to 2^-114
                    let ea = 30 + r.below(60);
                    let eb = (85 + r.below(56)).saturating_sub(ea).min(254);
                    let (ma, mb) = (r.next() & 0x7F, r.next() & 0x7F);
                    let (za, zb) = (r.below(20) == 0, r.below(20) == 0);
                    let wa = if za { 0 } else { (ea << 7) | ma };
                    let wb = if zb { 0 } else { (eb << 7) | mb };
                    ((((r.next() & 1) << 15) | wa) as u16, (((r.next() & 1) << 15) | wb) as u16)
                };
            }
            let want = step(acc_bits, &a, &b);
            let word = |v: &[u16; 16], q: usize| (0..4).fold(0u64, |acc, j| acc | (v[4 * q + j] as u64) << (16 * j));
            let mut bad = 0u64;
            let acc = groupsum::fp32_term(acc_bits).unwrap();
            let got = group(acc, [0, 1, 2, 3].map(|q| word(&a, q)), [0, 1, 2, 3].map(|q| word(&b, q)), &mut bad);
            let got = if bad & L100 != 0 { None } else { Some(groupsum::pack_fp32(got)) };
            assert_eq!(got, want, "trial {t}: acc {acc_bits:#010x} a {a:04x?} b {b:04x?}");
            if acc.magnitude == 0 && t % 3 != 2 {
                below_floor += 1;
            }
        }
        assert!(below_floor > trials / 10, "the zero-accumulator floor window exercised: {below_floor}");
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
