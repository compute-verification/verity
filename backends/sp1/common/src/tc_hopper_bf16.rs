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
    }
}
