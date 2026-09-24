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
/// NaN operand.
pub fn vu(pipeline: &GroupSum, x: &[u8], w: &[u8]) -> Option<u32> {
    if x.len() != K_VU || w.len() != K_VU {
        return None;
    }
    groupsum::chain(pipeline, x, w, e4m3_product)
}

/// `fp8-ada`: the RTX 4090 row's VU.
pub fn vu_ada(x: &[u8], w: &[u8]) -> Option<u32> {
    vu(&ADA_E4M3_M16N8K32, x, w)
}

/// `fp8-hopper`: the H100 FP8 row's VU.
pub fn vu_hopper(x: &[u8], w: &[u8]) -> Option<u32> {
    vu(&HOPPER_E4M3_K32, x, w)
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
}
