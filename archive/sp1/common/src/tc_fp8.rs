//! The two FP8 Table 2 rows: `GemmAccumulator<1536>` over E4M3 operands with an
//! FP32 accumulator, bit for bit as `verity.ml.circuits.silicon.models` computes them.
//!
//! * `fp8-ada` (RTX 4090 FP8, `ADA_E4M3_M16N8K32`): `mma.sync.m16n8k32.e4m3`,
//!   two groups of 16 products, a 14-bit adder, exponent floor `-139`.
//! * `fp8-hopper` (H100 FP8, `HOPPER_E4M3_K32`): `wgmma.m64n8k32.e4m3`, one
//!   group of 32 products, the same adder and floor.
//!
//! A VU is `1536 / 32 = 48` chained steps from `c_0 = +0`; its public word is
//! the final FP32 accumulator (no epilogue cast: the relation is
//! `GemmAccumulator`, `archive/direct/ligero/relations.py`).  The product is
//! `term.e4m3_product` (Hawkeye's `multiply_fp8e4m3_to_gfloat`): OCP E4M3 has
//! no infinities, only `0x7F` / `0xFF` are NaN (outside the model: `None`),
//! subnormals keep their unnormalised significand at exponent field 1.  The
//! accumulation is [`crate::groupsum`], whose narrow-adder path truncates the
//! incoming accumulator to 14 bits exactly as the Python's negative rescale.

use crate::groupsum::{self, GroupSum, Term, ZERO};

/// `models.ADA_E4M3_M16N8K32`.
pub const ADA_E4M3_M16N8K32: GroupSum = GroupSum { groups: &[16, 16], width: 14, zero_exponent: -139 };

/// `models.HOPPER_E4M3_K32`.
pub const HOPPER_E4M3_K32: GroupSum = GroupSum { groups: &[32], width: 14, zero_exponent: -139 };

/// Elements per VU row (`relations.K_VU`).
pub const K_VU: usize = 1536;

/// `term.e4m3_product`: the exact product of two E4M3 bytes as a 25-bit-scaled
/// term; `None` for a NaN operand.
#[inline(always)]
pub fn e4m3_product(a: u8, b: u8) -> Option<Term> {
    if a & 0x7F == 0x7F || b & 0x7F == 0x7F {
        return None;
    }
    let ea = ((a >> 3) & 0xF) as u32;
    let eb = ((b >> 3) & 0xF) as u32;
    let sa = (a & 0x7) as u32 | if ea != 0 { 0x8 } else { 0 };
    let sb = (b & 0x7) as u32 | if eb != 0 { 0x8 } else { 0 };
    let magnitude = (sa * sb) << 17;
    if magnitude == 0 {
        return Some(ZERO);
    }
    Some(Term {
        negative: (a ^ b) >> 7 == 1,
        exponent: ea.max(1) as i32 + eb.max(1) as i32 - 14,
        magnitude,
    })
}

/// One instruction step of `pipeline` on E4M3 operands (`GroupSum.step`).
pub fn step(pipeline: &GroupSum, acc: u32, a: &[u8], b: &[u8]) -> Option<u32> {
    groupsum::step(pipeline, acc, a, b, e4m3_product)
}

/// The public word of one FP8 VU: the final FP32 accumulator of the chain over
/// `pipeline` from `+0`.  `None` for rows that are not `K_VU` bytes each or a
/// NaN operand.  The reference the kernels ([`vu_ada`], [`vu_hopper`]) are held to.
pub fn vu(pipeline: &GroupSum, x: &[u8], w: &[u8]) -> Option<u32> {
    if x.len() != K_VU || w.len() != K_VU {
        return None;
    }
    groupsum::chain(pipeline, x, w, e4m3_product)
}

/// u64 words of one VU row.
pub const ROW_WORDS: usize = K_VU / 8;

/// `fp8-ada`: the RTX 4090 row's VU ([`vu`] over [`ADA_E4M3_M16N8K32`], by the kernel).
pub fn vu_ada(x: &[u8], w: &[u8]) -> Option<u32> {
    kernel::<16>(x, w, &words(x), &words(w))
}

/// `fp8-hopper`: the H100 FP8 row's VU ([`vu`] over [`HOPPER_E4M3_K32`], by the kernel).
pub fn vu_hopper(x: &[u8], w: &[u8]) -> Option<u32> {
    kernel::<32>(x, w, &words(x), &words(w))
}

/// [`vu_ada`] on rows given twice: as bytes and as the same bytes read as little-endian u64 words ([`ROW_WORDS`]
/// each; the guest holds both views of its input buffer).  The words must be the bytes'.
pub fn vu_ada_views(x: &[u8], w: &[u8], xw: &[u64], ww: &[u64]) -> Option<u32> {
    kernel::<16>(x, w, xw, ww)
}

/// [`vu_hopper`] on both views of the rows, as [`vu_ada_views`].
pub fn vu_hopper_views(x: &[u8], w: &[u8], xw: &[u64], ww: &[u64]) -> Option<u32> {
    kernel::<32>(x, w, xw, ww)
}

fn words(row: &[u8]) -> Vec<u64> {
    row.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect()
}

// --- the kernel ------------------------------------------------------------------------------------------------------
//
// `groupsum::chain` specialised to a 14-bit adder, floor -139, 32 products per step in groups of `G`, and E4M3
// products: a group's largest product exponent found eight lanes at a time on the u64 words (which also catch the
// NaN codes), then one pass that looks each byte pair up in a product table and adds its aligned product; the
// accumulator is carried as a `Term` between steps (a step's result round-trips `pack_fp32` / `fp32_term` exactly:
// `group_sum` never returns `-0`, and its magnitude has bit 23 set unless the exponent is -126).

/// Per byte pair `(a, b)` at index `a << 8 | b`, for a nonzero product: `E = max(ea, 1) + max(eb, 1) <= 30` in bits
/// 0..4, the product sign in bit 5 and the product significand `p = sa * sb < 2^8` as `p << 7` (its 25-bit grid value
/// truncated to the adder) in bits 17..31, so that the entry shifted right by `17 + d` is the product aligned `d`
/// places down.  A zero product's entry is 0.  A NaN code's entry is never used (rejected from the words).
static E4M3_PAIRS: [u32; 65536] = {
    const fn decode(v: u32) -> (u32, u32, u32) {
        let field = (v >> 3) & 0xF;
        ((v & 7) | if field != 0 { 8 } else { 0 }, v >> 7, if field == 0 { 1 } else { field })
    }
    let mut t = [0u32; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let ((sa, na, ea), (sb, nb, eb)) = (decode(i >> 8), decode(i & 0xFF));
        let p = sa * sb;
        t[i as usize] = if p == 0 { 0 } else { (ea + eb) | (na ^ nb) << 5 | p << 24 };
        i += 1;
    }
    t
};

const B01: u64 = 0x0101_0101_0101_0101;
const B0F: u64 = 0x0F0F_0F0F_0F0F_0F0F;
const B7F: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const B80: u64 = 0x8080_8080_8080_8080;

/// Per byte lane of a word pair: `E = max(ea, 1) + max(eb, 1)` where both significands are non-zero, else 0; `bad`
/// collects bit 7 of the lanes where either byte is a NaN code (low seven bits all set).
#[inline(always)]
fn lane_exponents(x: u64, w: u64, bad: &mut u64) -> u64 {
    let (lx, lw) = (x & B7F, w & B7F);
    *bad |= (lx + B01) | (lw + B01);
    let (ex, ew) = ((x >> 3) & B0F, (w >> 3) & B0F);
    // bit 0 of a lane: its exponent field is non-zero (fields are below 0x10, so no lane carries)
    let (nx, nw) = (((ex + B7F) >> 7) & B01, ((ew + B7F) >> 7) & B01);
    let e = (ex | (nx ^ B01)) + (ew | (nw ^ B01));
    // bit 7 of a lane: both significands are non-zero
    let nz = (lx + B7F) & (lw + B7F) & B80;
    e & ((nz >> 7) * 0xFF)
}

/// The lane-wise maximum of bytes below `0x80`.
#[inline(always)]
fn lane_max(a: u64, b: u64) -> u64 {
    let ge = ((a | B80) - b) & B80;
    b ^ ((a ^ b) & ((ge >> 7) * 0xFF))
}

const FLOOR: i32 = -139;
const ADDER_WIDTH: i32 = 14;
/// `FP32_SIGNIFICAND_WIDTH - ADDER_WIDTH`: every term is truncated by this many bits before the aligned sum.
const TRUNC: u32 = 10;

/// `group_sum(acc, products of a[i] * b[i])` at width 14, floor -139, on the group's bytes and the same bytes as
/// `G / 8` u64 words; `bad` collects bit 7 of every lane holding a NaN code.  A product `p = sa * sb < 2^8` is
/// `p << 17` on the 25-bit grid, i.e. `p << 7` after the truncation to the adder, at the `Term` exponent `E - 14`
/// with `E = max(ea, 1) + max(eb, 1)`; every product's exponent is at least `2 - 14`, above the floor.
#[inline(always)]
fn group<const G: usize>(acc: Term, a: &[u8], b: &[u8], aw: &[u64], bw: &[u64], bad: &mut u64) -> Term {
    let (a, b): (&[u8; G], &[u8; G]) = (a.try_into().unwrap(), b.try_into().unwrap());
    let mut m = 0u64;
    for k in 0..G / 8 {
        m = lane_max(m, lane_exponents(aw[k], bw[k], bad));
    }
    m = lane_max(m, m >> 32);
    m = lane_max(m, m >> 16);
    m = lane_max(m, m >> 8);
    let max_e = (m & 0xFF) as i32;
    let mut max_exp = if max_e != 0 { max_e - 14 } else { FLOOR };
    if acc.magnitude != 0 && acc.exponent > max_exp {
        max_exp = acc.exponent;
    }
    let mut total: i64 = 0;
    if acc.magnitude != 0 {
        let d = (max_exp - acc.exponent) as u32;
        let v = if d < 14 { (acc.magnitude >> TRUNC >> d) as i64 } else { 0 };
        total = if acc.negative { -v } else { v };
    }
    // a product's alignment distance is d = top - E >= 0 with top = max_exp + 14; p << 7 < 2^15 vanishes from d = 15,
    // so every product does once top >= 45 (E <= 30), and below that the shift 17 + d stays under 64
    let top = max_exp + 14;
    if max_e != 0 && top < 45 {
        let t = (top + 17) as u64;
        #[inline(always)]
        fn add(total: &mut i64, t: u64, e: u32) {
            let e = e as u64;
            let v = (e >> (t - (e & 0x1F))) as i64;
            if e & 0x20 != 0 {
                *total -= v;
            } else {
                *total += v;
            }
        }
        for i in (0..G).step_by(8) {
            for j in i..i + 8 {
                add(&mut total, t, E4M3_PAIRS[(a[j] as usize) << 8 | b[j] as usize]);
            }
        }
    }
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
    let m = m << TRUNC;
    if m == 0 {
        return ZERO;
    }
    Term { negative, exponent, magnitude: m as u32 }
}

/// The chain of 48 steps of 32 products in groups of `G` (16: Ada, 32: Hopper) from `+0`, as `groupsum::chain`
/// computes it: a NaN operand of a group rejects before that group's result is used, a saturating group ends the
/// chain -- with its infinity in the last step (the later groups unexamined), else outside the model.
fn kernel<const G: usize>(x: &[u8], w: &[u8], xw: &[u64], ww: &[u64]) -> Option<u32> {
    if x.len() != K_VU || w.len() != K_VU || xw.len() != ROW_WORDS || ww.len() != ROW_WORDS {
        return None;
    }
    const STEP: usize = 32;
    let steps = K_VU / STEP;
    let mut acc = ZERO;
    for s in 0..steps {
        let mut at = s * STEP;
        while at < (s + 1) * STEP {
            let mut bad = 0u64;
            let (g, gw) = (at..at + G, at / 8..(at + G) / 8);
            acc = group::<G>(acc, &x[g.clone()], &w[g], &xw[gw.clone()], &ww[gw], &mut bad);
            if bad & B80 != 0 {
                return None;
            }
            if acc.magnitude != 0 && acc.exponent > 127 {
                return if s == steps - 1 { Some(groupsum::pack_fp32(acc)) } else { None };
            }
            at += G;
        }
    }
    Some(groupsum::pack_fp32(acc))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: u8 = 0x38; // 1.0: exponent field 7, fraction 0

    #[test]
    fn products() {
        // 1 * 1 = 1: (8 * 8) << 17 = 2^23 at exponent 7 + 7 - 14 = 0
        assert_eq!(e4m3_product(ONE, ONE), Some(Term { negative: false, exponent: 0, magnitude: 1 << 23 }));
        assert_eq!(e4m3_product(ONE | 0x80, ONE).map(|t| t.negative), Some(true));
        assert_eq!(e4m3_product(0, ONE), Some(ZERO));
        assert_eq!(e4m3_product(0x80, 0x80), Some(ZERO));
        // the smallest subnormal (2^-9) squared: 1 << 17 at exponent 1 + 1 - 14
        assert_eq!(e4m3_product(1, 1), Some(Term { negative: false, exponent: -12, magnitude: 1 << 17 }));
        // 448 (0x7E) is finite; 0x7F / 0xFF are NaN
        assert!(e4m3_product(0x7E, 0x7E).is_some());
        assert_eq!(e4m3_product(0x7F, ONE), None);
        assert_eq!(e4m3_product(ONE, 0xFF), None);
    }

    #[test]
    fn exact_small_steps() {
        let ones = [ONE; 32];
        let zeros = [0u8; 32];
        // 32 ones: 32.0 on both pipelines
        assert_eq!(step(&HOPPER_E4M3_K32, 0, &ones, &ones), Some(0x4200_0000));
        assert_eq!(step(&ADA_E4M3_M16N8K32, 0, &ones, &ones), Some(0x4200_0000));
        assert_eq!(step(&HOPPER_E4M3_K32, 0x8000_0000, &zeros, &zeros), Some(0)); // -0 acc -> +0
        // the 14-bit adder truncates the accumulator: 1 + 2^-20 carried in loses its low bits
        let acc = 0x3F80_0008; // 1 + 2^-20
        assert_eq!(step(&HOPPER_E4M3_K32, acc, &zeros, &zeros), Some(0x3F80_0000));
        // a NaN operand is outside the model
        let mut nan = ones;
        nan[31] = 0x7F;
        assert_eq!(step(&HOPPER_E4M3_K32, 0, &nan, &ones), None);
        assert_eq!(step(&HOPPER_E4M3_K32, 0x7F80_0000, &ones, &ones), None);
    }

    #[test]
    fn vu_shapes() {
        let x = vec![ONE; K_VU];
        // 48 steps of +32 each from 0: 1536.0 exactly (every partial sum is an integer below 2^14)
        assert_eq!(vu_hopper(&x, &x), Some(0x44C0_0000));
        assert_eq!(vu_ada(&x, &x), Some(0x44C0_0000));
        assert_eq!(vu_ada(&x[..32], &x[..32]), None);
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

    /// An E4M3 byte: random sign and mantissa, exponent field in `lo..=hi`, zeros at `zero_pm` per mille; never NaN.
    fn byte(r: &mut Rng, lo: u64, hi: u64, zero_pm: u64) -> u8 {
        if r.below(1000) < zero_pm {
            return ((r.next() & 1) << 7) as u8;
        }
        let v = (((r.next() & 1) << 7) | (lo + r.below(hi - lo + 1)) << 3 | (r.next() & 7)) as u8;
        if v & 0x7F == 0x7F {
            v ^ 1
        } else {
            v
        }
    }

    /// The kernels equal `vu` over both pipelines on rows from several regimes: every exponent, tiny values
    /// (subnormal operands, products far below the accumulator), mostly zeros, exact cancellation, a wide gap
    /// between a few huge and many small products, and a sparse NaN before or after a group boundary.
    #[test]
    fn kernels_match_the_reference() {
        let mut r = Rng(0xF8_2026_0924);
        let cases = if cfg!(debug_assertions) { 400 } else { 6000 };
        let mut rejected = 0;
        for case in 0..cases {
            let regime = case % 7;
            let (mut x, mut w) = (vec![0u8; K_VU], vec![0u8; K_VU]);
            for k in 0..K_VU {
                (x[k], w[k]) = match regime {
                    0 => (byte(&mut r, 0, 15, 20), byte(&mut r, 0, 15, 20)),
                    1 => (byte(&mut r, 0, 2, 50), byte(&mut r, 0, 3, 50)),
                    2 => (byte(&mut r, 0, 15, 900), byte(&mut r, 0, 15, 500)),
                    3 | 6 => (byte(&mut r, 4, 10, 10), byte(&mut r, 4, 10, 10)),
                    4 => {
                        if r.below(40) == 0 {
                            (byte(&mut r, 14, 15, 0), byte(&mut r, 14, 15, 0))
                        } else {
                            (byte(&mut r, 0, 6, 10), byte(&mut r, 0, 6, 10))
                        }
                    }
                    _ => (byte(&mut r, 5, 9, 5), byte(&mut r, 5, 9, 5)),
                };
            }
            if regime == 3 {
                for k in (0..K_VU).step_by(2) {
                    if r.next() & 3 != 0 {
                        x[k + 1] = x[k] ^ 0x80;
                        w[k + 1] = w[k];
                    }
                }
            }
            if regime == 6 {
                let k = r.below(K_VU as u64) as usize;
                if r.next() & 1 == 0 {
                    x[k] = 0x7F;
                } else {
                    w[k] = 0xFF;
                }
            }
            let (ada, hopper) = (vu(&ADA_E4M3_M16N8K32, &x, &w), vu(&HOPPER_E4M3_K32, &x, &w));
            assert_eq!(vu_ada(&x, &w), ada, "ada case {case} (regime {regime})");
            assert_eq!(vu_hopper(&x, &w), hopper, "hopper case {case} (regime {regime})");
            rejected += ada.is_none() as usize;
        }
        assert_eq!(rejected, (0..cases).filter(|c| c % 7 == 6).count(), "exactly the NaN regime rejects");
    }

    /// One kernel step (both group sizes) equals `groupsum::step` on single groups: arbitrary bytes (every code,
    /// NaN in any lane, all zero lanes), and accumulators zero, subnormal, and normal, near or far from the products.
    #[test]
    fn group_matches_step() {
        fn one<const G: usize>(r: &mut Rng, t: usize) -> (bool, bool) {
            let acc_bits: u32 = match t % 4 {
                0 => (r.next() as u32) & 0x8000_0000,
                1 => (r.next() as u32) & 0x807F_FFFF,
                2 => ((r.next() as u32) & 0x8000_0000) | ((110 + r.below(40) as u32) << 23) | (r.next() as u32 & 0x7F_FFFF),
                _ => loop {
                    let v = r.next() as u32;
                    if (v >> 23) & 0xFF != 0xFF {
                        break v;
                    }
                },
            };
            let (mut a, mut b) = ([0u8; G], [0u8; G]);
            for k in 0..G {
                (a[k], b[k]) = match (t / 4) % 3 {
                    0 => (r.next() as u8, r.next() as u8),
                    1 => (byte(r, 0, 15, 300), byte(r, 0, 15, 300)),
                    _ => (byte(r, 0, 4, 100), byte(r, 6, 15, 100)),
                };
            }
            if t % 7 == 0 {
                a[r.below(G as u64) as usize] |= 0x7F;
            }
            let pipeline = GroupSum { groups: &[G], width: 14, zero_exponent: -139 };
            let want = groupsum::step(&pipeline, acc_bits, &a, &b, e4m3_product);
            let mut bad = 0u64;
            let acc = groupsum::fp32_term(acc_bits).unwrap();
            let got = group::<G>(acc, &a, &b, &words(&a), &words(&b), &mut bad);
            let got = if bad & B80 != 0 { None } else { Some(groupsum::pack_fp32(got)) };
            assert_eq!(got, want, "G {G} trial {t}: acc {acc_bits:#010x} a {a:02x?} b {b:02x?}");
            (want.is_none(), want.is_some_and(|v| v & 0x7FFF_FFFF != 0))
        }
        let mut r = Rng(0xF8_0000_2026);
        let trials = if cfg!(debug_assertions) { 20_000 } else { 500_000 };
        let (mut nan, mut nonzero) = (0, 0);
        for t in 0..trials {
            let (n16, z16) = one::<16>(&mut r, t);
            let (n32, z32) = one::<32>(&mut r, t);
            nan += n16 as usize + n32 as usize;
            nonzero += z16 as usize + z32 as usize;
        }
        assert!(nan > trials / 20 && nonzero > trials, "NaN rejections {nan}, non-zero results {nonzero}");
    }
}
