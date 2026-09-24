//! The two FP8 Table 2 rows: `GemmAccumulator<1536>` over E4M3 operands with an
//! FP32 accumulator, bit for bit as `verity.ml.tc.models` computes them.
//!
//! * `fp8-ada` (RTX 4090 FP8, `ADA_E4M3_M16N8K32`): `mma.sync.m16n8k32.e4m3`,
//!   two groups of 16 products, a 14-bit adder, exponent floor `-139`.
//! * `fp8-hopper` (H100 FP8, `HOPPER_E4M3_K32`): `wgmma.m64n8k32.e4m3`, one
//!   group of 32 products, the same adder and floor.
//!
//! A VU is `1536 / 32 = 48` chained steps from `c_0 = +0`; its public word is
//! the final FP32 accumulator (no epilogue cast: the relation is
//! `GemmAccumulator`, `backends/direct/ligero/relations.py`).  The product is
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

/// `fp8-ada`: the RTX 4090 row's VU ([`vu`] over [`ADA_E4M3_M16N8K32`], by the kernel).
pub fn vu_ada(x: &[u8], w: &[u8]) -> Option<u32> {
    kernel::<16>(x, w)
}

/// `fp8-hopper`: the H100 FP8 row's VU ([`vu`] over [`HOPPER_E4M3_K32`], by the kernel).
pub fn vu_hopper(x: &[u8], w: &[u8]) -> Option<u32> {
    kernel::<32>(x, w)
}

// --- the kernel ------------------------------------------------------------------------------------------------------
//
// `groupsum::chain` specialised to a 14-bit adder, floor -139, 32 products per step in groups of `G`, and E4M3
// products: each byte decoded by one table load, a group's nonzero products kept as packed words and summed in a
// second pass once the group maximum is known, and the accumulator carried as a `Term` between steps (a step's
// result round-trips `pack_fp32` / `fp32_term` exactly: `group_sum` never returns `-0`, and its magnitude has bit 23
// set unless the exponent is -126).

/// Per E4M3 byte: `0` for a zero; else the significand (implicit bit included) in bits 0..4, the sign in bit 8 and
/// `max(field, 1)` in bits 16..20; `1 << 31` alone for the two NaN codes.  The sum of two entries carries the product
/// sign in bit 8 (bit 9 takes the carry) and `E = max(ea, 1) + max(eb, 1) <= 30` in bits 16..20.
static E4M3_DECODE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut v = 0u32;
    while v < 256 {
        let field = (v >> 3) & 0xF;
        let sig = (v & 7) | if field != 0 { 8 } else { 0 };
        t[v as usize] = if v & 0x7F == 0x7F {
            1 << 31
        } else if sig == 0 {
            0
        } else {
            sig | (v >> 7) << 8 | (if field == 0 { 1 } else { field }) << 16
        };
        v += 1;
    }
    t
};

/// A kept product: `p` in bits 0..8, its sign in bit 8, `E` in bits 16..21 (so the largest word has the largest `E`).
const KEPT_MASK: u32 = 0x1F_0100;

const FLOOR: i32 = -139;
const ADDER_WIDTH: i32 = 14;
/// `FP32_SIGNIFICAND_WIDTH - ADDER_WIDTH`: every term is truncated by this many bits before the aligned sum.
const TRUNC: u32 = 10;

/// `group_sum(acc, products of a[i] * b[i])` at width 14, floor -139; `bad` collects bit 31 of every operand's
/// decode (set iff one is NaN).  A product `p = sa * sb < 2^8` is `p << 17` on the 25-bit grid, i.e. `p << 7` after
/// the truncation to the adder, at the `Term` exponent `E - 14` with `E = max(ea, 1) + max(eb, 1)`.
#[inline(always)]
fn group<const G: usize>(acc: Term, a: &[u8], b: &[u8], bad: &mut u32) -> Term {
    let (a, b): (&[u8; G], &[u8; G]) = (a.try_into().unwrap(), b.try_into().unwrap());
    let mut kept = [0u32; G];
    let mut n = 0;
    let mut max_t = 0u32;
    for i in 0..G {
        let da = E4M3_DECODE[a[i] as usize];
        let db = E4M3_DECODE[b[i] as usize];
        let p = (da & 0xF) * (db & 0xF);
        if p != 0 {
            let t = ((da + db) & KEPT_MASK) | p;
            if t > max_t {
                max_t = t;
            }
            // n <= i < G, a power of two: the mask is the identity and spares the bounds check
            kept[n & (G - 1)] = t;
            n += 1;
        } else {
            // a NaN code decodes to a zero significand
            *bad |= da | db;
        }
    }
    let mut max_exp = if n > 0 { (max_t >> 16) as i32 - 14 } else { FLOOR };
    if acc.magnitude != 0 && acc.exponent > max_exp {
        max_exp = acc.exponent;
    }
    let mut total: i64 = 0;
    if acc.magnitude != 0 {
        let d = (max_exp - acc.exponent) as u32;
        let v = if d < 14 { (acc.magnitude >> TRUNC >> d) as i64 } else { 0 };
        total = if acc.negative { -v } else { v };
    }
    // a product's alignment distance is max_exp - (E - 14); p << 7 < 2^15 vanishes from 15
    let top = max_exp + 14;
    for &t in kept.iter().take(n) {
        let d = (top - (t >> 16) as i32) as u32;
        if d < 15 {
            let v = (((t & 0xFF) << 7) >> d) as i64;
            if t & 0x100 != 0 {
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
fn kernel<const G: usize>(x: &[u8], w: &[u8]) -> Option<u32> {
    if x.len() != K_VU || w.len() != K_VU {
        return None;
    }
    const STEP: usize = 32;
    let steps = K_VU / STEP;
    let mut acc = ZERO;
    for s in 0..steps {
        let mut at = s * STEP;
        while at < (s + 1) * STEP {
            let mut bad = 0u32;
            acc = group::<G>(acc, &x[at..at + G], &w[at..at + G], &mut bad);
            if bad >> 31 != 0 {
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
}
