//! The RTX 5090 NVFP4 Table 2 row (`fp4-nvf4`): `GemmAccumulator<1536>` over
//! `mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (E2M1 elements,
//! one UE4M3 scale byte per 16 elements, FP32 accumulator) on sm_120, bit for
//! bit as `verity.ml.circuits.silicon.models.BLACKWELL_SM120_NVF4` computes it
//! (`BlockScaledAlignAdd`: `frac_bits = 27`, `acc_window = 36`,
//! `grid_floor = -174`, `underflow_zero = "positive"`).
//!
//! One firing (64 elements, four groups of 16, four scale bytes per operand):
//!
//! 1. each group's 16 E2M1 x E2M1 products are summed exactly as an integer
//!    number of quarters;
//! 2. the group's two UE4M3 scales multiply it losslessly: `mant = gd * ma * mb`
//!    at the unit weight `2^X`, `X = -2 + ea + eb` (`(ma, ea)` = `(8 + m, e - 10)`,
//!    or `(m, -9)` for exponent field 0; a scale byte with bit 7 set, or the
//!    NaN code `0x7F`, is outside the model);
//! 3. one five-way align-add of the four scaled dots and the carried FP32
//!    accumulator on the grid `2^lsb`, every addend truncated toward zero,
//!    summed exactly and normalised toward zero to binary32, where
//!    `lsb = max(max{X_g : g participates} - 27, acc != 0 ? E_acc - 35 : -inf, -174)`
//!    with `E_acc` the accumulator's exponent field minus 127 (a subnormal
//!    counts as `-126`).  A group participates when some product of it is
//!    nonzero and both its scale mantissas are nonzero.  Every zero result is
//!    `+0`; a sum leaving binary32 is outside the model (`None`).
//!
//! A VU is `1536 / 64 = 24` chained firings from `c_0 = +0`; its public word is
//! the final FP32 accumulator.  The VU row layout is the relation's operand
//! bytes (`archive/direct/ligero/fp4`): per firing, 64 element codes (one
//! byte each, below 16) then 4 scale bytes -- [`ROW_BYTES`] = 24 x 68.
//!
//! Widths: `|gd| <= 16 * 144`, `ma, mb <= 15`, so `|mant| < 2^20`; on the grid a
//! participating group is shifted left by at most 27 and the accumulator by
//! at most 12, so the sum is below `2^50`: `i64` is exact.

use crate::groupsum::{bit_length, bit_length64};

/// Elements per firing.
pub const K: usize = 64;
/// Elements per group (and per scale block).
pub const GROUP: usize = 16;
/// Scale bytes per operand per firing.
pub const SCALES: usize = K / GROUP;
/// Operand bytes per firing: the codes then the scales.
pub const WORDS: usize = K + SCALES;
/// Firings per VU.
pub const STEPS: usize = 24;
/// Bytes of one VU row.
pub const ROW_BYTES: usize = STEPS * WORDS;

const FRAC_BITS: i32 = 27;
const ACC_WINDOW: i32 = 36;
const GRID_FLOOR: i32 = -174;
const FP32_SUBNORMAL_EXPONENT: i32 = -149;
const FP32_MIN_NORMAL_EXPONENT: i32 = -126;
const FP32_MAX_EXPONENT: i32 = 127;

/// `|value| * 2` for the eight E2M1 magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}`.
const E2M1_NUMERATOR: [i32; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

/// The E2M1 code as a signed number of halves; `None` above 4 bits.
#[inline(always)]
fn e2m1_numerator(code: u8) -> Option<i32> {
    if code >= 16 {
        return None;
    }
    let n = E2M1_NUMERATOR[(code & 7) as usize];
    Some(if code & 8 != 0 { -n } else { n })
}

/// A UE4M3 scale byte as an exact `(mantissa, exponent)`; `None` for the padding bit or the NaN code.
#[inline(always)]
pub fn ue4m3(code: u8) -> Option<(i64, i32)> {
    if code & 0x80 != 0 {
        return None;
    }
    let e = ((code >> 3) & 0xF) as i32;
    let m = (code & 7) as i64;
    if e == 15 && m == 7 {
        return None;
    }
    Some(if e == 0 { (m, -9) } else { (8 + m, e - 10) })
}

/// `floor toward zero` of `mant * 2^exp` onto the grid `2^lsb`, as a multiple of `2^lsb`.
#[inline(always)]
fn trunc_rz_to_grid(mant: i64, exp: i32, lsb: i32) -> i64 {
    if exp >= lsb {
        return mant << (exp - lsb);
    }
    let shift = lsb - exp;
    let q = if shift >= 64 { 0 } else { (mant.unsigned_abs() >> shift) as i64 };
    if mant < 0 {
        -q
    } else {
        q
    }
}

/// `dyadic.rs::to_f32(Round::Rz)` of the nonzero `n * 2^lsb` (normals, gradual
/// underflow to `+0`, no carry); `None` above the binary32 range.
#[inline]
fn to_f32_rz(n: i64, lsb: i32) -> Option<u32> {
    let negative = n < 0;
    let mag = n.unsigned_abs();
    let e_val = bit_length64(mag) - 1 + lsb;
    let grid = if e_val >= FP32_MIN_NORMAL_EXPONENT { e_val - 23 } else { FP32_SUBNORMAL_EXPONENT };
    let shift = grid - lsb;
    let q = if shift > 0 {
        if shift >= 64 {
            0
        } else {
            mag >> shift
        }
    } else {
        mag << -shift
    };
    if q == 0 {
        return Some(0);
    }
    // q < 2^24: 24 bits at a normal grid, fewer below
    let e = bit_length(q) - 1 + grid;
    if e > FP32_MAX_EXPONENT {
        return None;
    }
    let bits = if e >= FP32_MIN_NORMAL_EXPONENT {
        (((e + 127) as u32) << 23) | (q as u32 & 0x7F_FFFF)
    } else {
        q as u32
    };
    Some(bits | if negative { 0x8000_0000 } else { 0 })
}

/// One firing on its operand bytes (`a`, `b`: [`WORDS`] bytes each, the codes
/// then the scales): `BlockScaledAlignAdd.step`.  `None` outside the model (a
/// non-finite accumulator, a code above 4 bits, an invalid scale byte, a sum
/// above binary32).
pub fn step(c: u32, a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != WORDS || b.len() != WORDS {
        return None;
    }
    // the scaled group dots: (mant, unit weight exponent, participates)
    let mut terms = [(0i64, 0i32, false); SCALES];
    for (g, term) in terms.iter_mut().enumerate() {
        let mut gd: i32 = 0;
        let mut any_product = false;
        for i in g * GROUP..(g + 1) * GROUP {
            let na = e2m1_numerator(a[i])?;
            let nb = e2m1_numerator(b[i])?;
            any_product |= na != 0 && nb != 0;
            gd += na * nb;
        }
        let (ma, ea) = ue4m3(a[K + g])?;
        let (mb, eb) = ue4m3(b[K + g])?;
        *term = (gd as i64 * ma * mb, -2 + ea + eb, any_product && ma != 0 && mb != 0);
    }
    align_add(c, &terms)
}

/// `na * nb` for every pair of E2M1 codes (index `a << 4 | b`), a signed number of quarters.
static E2M1_PRODUCTS: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut i = 0;
    while i < 256 {
        let (a, b) = (i >> 4, i & 15);
        let na = E2M1_NUMERATOR[a & 7] * if a & 8 != 0 { -1 } else { 1 };
        let nb = E2M1_NUMERATOR[b & 7] * if b & 8 != 0 { -1 } else { 1 };
        t[i] = na * nb;
        i += 1;
    }
    t
};

const B07: u64 = 0x0707_0707_0707_0707;
const B7F: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const B80: u64 = 0x8080_8080_8080_8080;
const BF0: u64 = 0xF0F0_F0F0_F0F0_F0F0;

/// Per scale-byte pair `(sa, sb)` at index `sa << 8 | sb`: `ma * mb <= 225` in bits 0..7 and `ea + eb + 32` (the
/// sum is in -18..=10) in bits 8..13; `u32::MAX` when either byte is outside the model.
static UE4M3_PAIRS: [u32; 65536] = {
    const fn decode(code: u32) -> Option<(u32, i32)> {
        let (e, m) = ((code >> 3) & 0xF, code & 7);
        if code & 0x80 != 0 || (e == 15 && m == 7) {
            None
        } else if e == 0 {
            Some((m, -9))
        } else {
            Some((8 + m, e as i32 - 10))
        }
    }
    let mut t = [0u32; 65536];
    let mut i = 0u32;
    while i < 65536 {
        t[i as usize] = match (decode(i >> 8), decode(i & 0xFF)) {
            (Some((ma, ea)), Some((mb, eb))) => ma * mb | ((ea + eb + 32) as u32) << 8,
            _ => u32::MAX,
        };
        i += 1;
    }
    t
};

/// [`step`] on a firing's operands: the codes read from the rows' words starting at word `q` (half-way into it
/// when `ODD`: a firing starts at byte `68 s`, word `17 s / 2`), the scale bytes from the rows' bytes.  Every lane's
/// pair index `a << 4 | b` is formed eight at a time and its product read from [`E2M1_PRODUCTS`]; a group
/// participates when some lane has both numerators nonzero (code bits 0..3), found lane-wise as well, and both
/// scale mantissas are nonzero.  A code above 4 bits rejects (it also garbles its neighbour's index, which is never
/// used then).
#[inline(always)]
fn firing<const ODD: bool>(c: u32, xw: &[u64], ww: &[u64], q: usize, sa: &[u8], sb: &[u8]) -> Option<u32> {
    let word = |w: &[u64], k: usize| if ODD { w[q + k] >> 32 | w[q + k + 1] << 32 } else { w[q + k] };
    let (mut mant, mut unit) = ([0i64; SCALES], [0i32; SCALES]);
    // the largest anchor so far; i32::MIN while nothing participates
    let mut lsb = i32::MIN;
    let mut bad = 0u64;
    for g in 0..SCALES {
        let (mut gd, mut nz) = (0i32, 0u64);
        for k in 2 * g..2 * g + 2 {
            let (a, b) = (word(xw, k), word(ww, k));
            bad |= a | b;
            // bit 7 of a lane: both code values below 8 are nonzero (a lane sum stays below 0x100)
            nz |= ((a & B07) + B7F) & ((b & B07) + B7F);
            let idx = a << 4 | b;
            for j in 0..8 {
                gd += E2M1_PRODUCTS[(idx >> (8 * j) & 0xFF) as usize];
            }
        }
        let s = UE4M3_PAIRS[(sa[g] as usize) << 8 | sb[g] as usize];
        if s == u32::MAX {
            return None;
        }
        let m = (s & 0xFF) as i64;
        (mant[g], unit[g]) = (gd as i64 * m, (s >> 8) as i32 - 34);
        if nz & B80 != 0 && m != 0 {
            lsb = lsb.max(unit[g] - FRAC_BITS);
        }
    }
    if bad & BF0 != 0 {
        return None;
    }
    // the carried accumulator as an exact dyadic (both zeros are zero)
    let field = (c >> 23) & 0xFF;
    if field == 0xFF {
        return None;
    }
    let frac = (c & 0x7F_FFFF) as i64;
    let (acc_m, acc_e) = if field == 0 { (frac, FP32_SUBNORMAL_EXPONENT) } else { ((1 << 23) | frac, field as i32 - 150) };
    if acc_m != 0 {
        lsb = lsb.max((field.max(1) as i32) - 127 - (ACC_WINDOW - 1));
    }
    if lsb == i32::MIN {
        return Some(0); // nothing participates: +0 (a -0 accumulator included)
    }
    let lsb = lsb.max(GRID_FLOOR);
    // a nonzero scaled dot participates
    let mut total: i64 = 0;
    for g in 0..SCALES {
        if mant[g] != 0 {
            total += trunc_rz_to_grid(mant[g], unit[g], lsb);
        }
    }
    if acc_m != 0 {
        let v = trunc_rz_to_grid(acc_m, acc_e, lsb);
        total += if c >> 31 == 1 { -v } else { v };
    }
    if total == 0 {
        return Some(0);
    }
    to_f32_rz(total, lsb)
}

/// The five-way align-add of the scaled group dots `(mant, unit weight exponent, participates)` and the carried
/// accumulator `c`, normalised toward zero to binary32.
#[inline(always)]
fn align_add(c: u32, terms: &[(i64, i32, bool); SCALES]) -> Option<u32> {
    // the carried accumulator as an exact dyadic (both zeros are zero)
    let field = (c >> 23) & 0xFF;
    if field == 0xFF {
        return None;
    }
    let frac = (c & 0x7F_FFFF) as i64;
    let (acc_m, acc_e) = if field == 0 { (frac, FP32_SUBNORMAL_EXPONENT) } else { ((1 << 23) | frac, field as i32 - 150) };
    let acc_m = if c >> 31 == 1 { -acc_m } else { acc_m };

    let mut lsb: Option<i32> = None;
    for &(_, unit, participates) in terms {
        if participates {
            let anchor = unit - FRAC_BITS;
            lsb = Some(lsb.map_or(anchor, |l| l.max(anchor)));
        }
    }
    if acc_m != 0 {
        let anchor = (field.max(1) as i32) - 127 - (ACC_WINDOW - 1);
        lsb = Some(lsb.map_or(anchor, |l| l.max(anchor)));
    }
    let lsb = match lsb {
        None => return Some(0), // nothing participates: +0 (a -0 accumulator included)
        Some(l) => l.max(GRID_FLOOR),
    };
    let mut total: i64 = 0;
    for &(mant, unit, _) in terms {
        if mant != 0 {
            total += trunc_rz_to_grid(mant, unit, lsb);
        }
    }
    if acc_m != 0 {
        total += trunc_rz_to_grid(acc_m, acc_e, lsb);
    }
    if total == 0 {
        return Some(0);
    }
    to_f32_rz(total, lsb)
}

/// The public word of one NVFP4 VU: the final FP32 accumulator of the 24
/// firings from `+0`.  `None` for rows that are not [`ROW_BYTES`] each or any
/// firing outside the model.  The reference [`vu`] is held to.
pub fn vu_reference(x: &[u8], w: &[u8]) -> Option<u32> {
    if x.len() != ROW_BYTES || w.len() != ROW_BYTES {
        return None;
    }
    let mut c = 0u32;
    for s in 0..STEPS {
        c = step(c, &x[s * WORDS..(s + 1) * WORDS], &w[s * WORDS..(s + 1) * WORDS])?;
    }
    Some(c)
}

/// u64 words of one VU row.
pub const ROW_WORDS: usize = ROW_BYTES / 8;

/// [`vu_reference`] by the kernel.
pub fn vu(x: &[u8], w: &[u8]) -> Option<u32> {
    if x.len() != ROW_BYTES || w.len() != ROW_BYTES {
        return None;
    }
    vu_views(x, w, &words(x), &words(w))
}

/// [`vu`] on rows given twice: as bytes and as the same bytes read as little-endian u64 words ([`ROW_WORDS`] each;
/// the guest holds both views of its input buffer).  The words must be the bytes'.
pub fn vu_views(x: &[u8], w: &[u8], xw: &[u64], ww: &[u64]) -> Option<u32> {
    if x.len() != ROW_BYTES || w.len() != ROW_BYTES || xw.len() != ROW_WORDS || ww.len() != ROW_WORDS {
        return None;
    }
    let mut c = 0u32;
    for s in (0..STEPS).step_by(2) {
        let (at, q) = (s * WORDS + K, 17 * s / 2);
        c = firing::<false>(c, xw, ww, q, &x[at..at + SCALES], &w[at..at + SCALES])?;
        let at = at + WORDS;
        c = firing::<true>(c, xw, ww, q + 8, &x[at..at + SCALES], &w[at..at + SCALES])?;
    }
    Some(c)
}

fn words(row: &[u8]) -> Vec<u64> {
    row.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scale byte for 1.0: exponent field 7, mantissa 0 -> (8, -3), i.e. 8 * 2^-3.
    const S1: u8 = 0x38;
    /// E2M1 code for 1.0 (two halves).
    const C1: u8 = 0x2;

    fn firing(codes: [u8; K], scales: [u8; SCALES]) -> [u8; WORDS] {
        let mut out = [0u8; WORDS];
        out[..K].copy_from_slice(&codes);
        out[K..].copy_from_slice(&scales);
        out
    }

    #[test]
    fn scales() {
        assert_eq!(ue4m3(S1), Some((8, -3)));
        assert_eq!(ue4m3(0x00), Some((0, -9)));
        assert_eq!(ue4m3(0x01), Some((1, -9)));
        assert_eq!(ue4m3(0x7E), Some((14, 5))); // 448
        assert_eq!(ue4m3(0x7F), None);
        assert_eq!(ue4m3(0x80), None);
    }

    #[test]
    fn exact_small_firings() {
        let ones = firing([C1; K], [S1; SCALES]);
        // 64 products of 1 * 1 at unit scales: 64.0
        assert_eq!(step(0, &ones, &ones), Some(0x4280_0000));
        // + an accumulator of 1.0: 65.0
        assert_eq!(step(0x3F80_0000, &ones, &ones), Some(0x4282_0000));
        // all-zero codes into a -0 accumulator: +0; into 1.0: 1.0 unchanged
        let zeros = firing([0; K], [S1; SCALES]);
        assert_eq!(step(0x8000_0000, &zeros, &zeros), Some(0));
        assert_eq!(step(0x3F80_0000, &zeros, &zeros), Some(0x3F80_0000));
        // exact cancellation across groups: +0
        let mut neg = [C1; K];
        for c in neg[32..].iter_mut() {
            *c |= 8;
        }
        assert_eq!(step(0, &firing(neg, [S1; SCALES]), &ones), Some(0));
        assert_eq!(step(0x8000_0000, &firing(neg, [S1; SCALES]), &ones), Some(0));
        // outside the model
        assert_eq!(step(0x7F80_0000, &ones, &ones), None);
        assert_eq!(step(0, &firing([16; K], [S1; SCALES]), &ones), None);
        assert_eq!(step(0, &firing([C1; K], [0x7F; SCALES]), &ones), None);
    }

    #[test]
    fn the_accumulator_window_truncates_toward_zero() {
        // accumulator 1.0 (E_acc = 0 -> anchor -35), one product 1 * 1 at a tiny scale: X = -2 + (-9) + (-9) with
        // mantissas 1 * 1 -> the unit weight 2^-20 group anchors at -47; lsb = -35; the 2^-20 * 4 addend is kept
        let mut a = [0u8; K];
        a[0] = C1;
        let x = firing(a, [0x01, S1, S1, S1]);
        let w = firing(a, [0x01, S1, S1, S1]);
        // gd = 4 quarters, mant = 4 * 1 * 1 at 2^-20 -> 2^-18
        assert_eq!(step(0x3F80_0000, &x, &w), Some(0x3F80_0020)); // 1 + 2^-18
    }

    #[test]
    fn vu_shapes() {
        let row: Vec<u8> = (0..STEPS).flat_map(|_| firing([C1; K], [S1; SCALES])).collect();
        assert_eq!(row.len(), ROW_BYTES);
        assert_eq!(vu(&row, &row), Some(0x44C0_0000)); // 1536.0
        assert_eq!(vu_reference(&row, &row), Some(0x44C0_0000));
        assert_eq!(vu(&row[..WORDS], &row[..WORDS]), None);
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

    /// The kernel equals `vu_reference` on rows from several regimes: any codes and scales (zero scales, tiny and
    /// huge ones, so groups drop out, the accumulator window truncates and sums leave binary32), mostly zero codes,
    /// exact cancellation, and a sparse off-domain code or scale byte.
    #[test]
    fn kernel_matches_the_reference() {
        let mut r = Rng(0xF4_2026_0924);
        let cases = if cfg!(debug_assertions) { 2000 } else { 40000 };
        let mut counts = [0usize; 3]; // [nonzero, +0, rejected]
        for case in 0..cases {
            let regime = case % 6;
            let (mut x, mut w) = (vec![0u8; ROW_BYTES], vec![0u8; ROW_BYTES]);
            for s in 0..STEPS {
                for i in 0..WORDS {
                    let at = s * WORDS + i;
                    (x[at], w[at]) = if i < K {
                        match regime {
                            2 => ((r.below(16) * (r.below(8) == 0) as u64) as u8, (r.below(16) * (r.below(3) == 0) as u64) as u8),
                            _ => (r.below(16) as u8, r.below(16) as u8),
                        }
                    } else {
                        let scale = |r: &mut Rng| -> u8 {
                            match regime {
                                0 => (r.below(0x7F)) as u8,
                                1 => [0x00, 0x01, 0x07, 0x08, 0x38, 0x3F, 0x70, 0x7E][r.below(8) as usize],
                                _ => (0x30 + r.below(0x10)) as u8,
                            }
                        };
                        (scale(&mut r), scale(&mut r))
                    };
                }
            }
            match regime {
                3 => {
                    for s in 0..STEPS {
                        for i in (0..K).step_by(2) {
                            let at = s * WORDS + i;
                            x[at + 1] = x[at] ^ 8;
                            w[at + 1] = w[at];
                        }
                    }
                }
                4 => {
                    let s = r.below(STEPS as u64) as usize;
                    match r.below(3) {
                        0 => x[s * WORDS + r.below(K as u64) as usize] = 16 + r.below(240) as u8,
                        1 => w[s * WORDS + K + r.below(SCALES as u64) as usize] = 0x7F,
                        _ => x[s * WORDS + K + r.below(SCALES as u64) as usize] = 0x80 | r.below(0x80) as u8,
                    }
                }
                _ => {}
            }
            let want = vu_reference(&x, &w);
            assert_eq!(vu(&x, &w), want, "case {case} (regime {regime})");
            counts[match want {
                None => 2,
                Some(0) => 1,
                Some(_) => 0,
            }] += 1;
        }
        eprintln!("nvfp4 kernel: {} nonzero, {} +0, {} rejected", counts[0], counts[1], counts[2]);
        assert!(counts.iter().all(|&c| c > 0), "every outcome exercised: {counts:?}");
    }
}
