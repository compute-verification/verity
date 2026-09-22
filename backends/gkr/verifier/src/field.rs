//! BabyBear `p = 2^31 - 2^27 + 1` and the GPU prover's extension `F_p[X] / (X^6 - W)`, `W = 22` (`gpu/field.py`,
//! PROTOCOL.md: the smallest non-square non-cube, the CPU crate's `src/field_babybear.rs` and `tensor/reference.py`
//! use the same polynomial).  Canonical `u32` representation everywhere the values are hashed or serialised;
//! Montgomery form only inside the NTT (twiddles are pre-scaled by `R = 2^32`).
//!
//! History: the a-gpu prover (main up to 2e1ef3f) used `X^6 - 31`; a-gpu2 moved it to `X^6 - 22` to share
//! a-packed2's kernels, and this verifier followed (coordinator, 2026-09-22, run r20260922-101240-8f71: the W = 31
//! verifier rejected every W = 22 proof at the first extension-field check, `LogUp POW level 0`).  Proofs written by
//! the W = 31 prover are incompatible with this verifier (a different presentation of the field: same soundness).

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const P: u32 = 2013265921;
pub const P64: u64 = P as u64;
pub const W: u32 = 22;
pub const DEG: usize = 6;
/// Hash-domain of the GPU prover's transcript (`gpu/transcript.py`).
pub const DOMAIN: &[u8] = b"verity-gkr/candidate-a/v1/babybear6";

/// 2^64 mod p (for folding a u128 accumulator).
const R64: u64 = ((1u128 << 64) % (P as u128)) as u64;
/// -p^{-1} mod 2^32 (Montgomery).
const NEG_PINV: u32 = {
    // Newton iteration for the inverse of p mod 2^32
    let mut inv: u32 = 1;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(P.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
};
/// R = 2^32 mod p.
pub const R_MOD_P: u32 = ((1u64 << 32) % P64) as u32;
/// R^2 mod p.
pub const R2_MOD_P: u32 = ((R_MOD_P as u64 * R_MOD_P as u64) % P64) as u32;

#[inline(always)]
pub fn add(a: u32, b: u32) -> u32 { let s = a + b; if s >= P { s - P } else { s } }
#[inline(always)]
pub fn sub(a: u32, b: u32) -> u32 { if a >= b { a - b } else { a + P - b } }
#[inline(always)]
pub fn mul(a: u32, b: u32) -> u32 { ((a as u64 * b as u64) % P64) as u32 }
#[inline(always)]
pub fn red64(x: u64) -> u32 { (x % P64) as u32 }
#[inline(always)]
pub fn red128(x: u128) -> u32 {
    let lo = (x as u64) % P64;
    let hi = ((x >> 64) as u64) % P64;
    (((hi * R64 as u64) % P64 + lo) % P64) as u32
}
/// Montgomery product: `a` canonical, `b_r = b R mod p` canonical -> `a b mod p` canonical.
#[inline(always)]
pub fn mont_mul(a: u32, b_r: u32) -> u32 {
    let t = a as u64 * b_r as u64;
    let m = (t as u32).wrapping_mul(NEG_PINV);
    let u = ((t + m as u64 * P64) >> 32) as u32;
    if u >= P { u - P } else { u }
}
/// `x R mod p` (Montgomery form of a canonical value).
#[inline(always)]
pub fn to_mont(x: u32) -> u32 { mont_mul(x, R2_MOD_P) }

pub fn pow(mut b: u32, mut e: u64) -> u32 {
    let mut r = 1u32;
    while e > 0 {
        if e & 1 == 1 { r = mul(r, b); }
        b = mul(b, b);
        e >>= 1;
    }
    r
}
pub fn inv(a: u32) -> u32 { assert!(a != 0, "inverse of zero"); pow(a, P64 - 2) }

/// `31` generates F_p^* (Plonky3's choice, `gpu/field.py::GENERATOR`); `p - 1 = 2^27 * 15`.
pub fn root_of_unity(order: usize) -> u32 {
    assert!(order.is_power_of_two() && order <= 1 << 27, "order {order}");
    pow(31, (P64 - 1) / order as u64)
}

/// `F_{p^6}`: coordinates `c0..c5`, `x = sum c_i X^i`, `X^6 = 31`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Ext(pub [u32; DEG]);

impl Ext {
    pub const ZERO: Ext = Ext([0; DEG]);
    pub const ONE: Ext = Ext([1, 0, 0, 0, 0, 0]);
    pub const BYTES: usize = 4 * DEG;
    #[inline(always)]
    pub fn from_base(x: u32) -> Ext { let mut c = [0u32; DEG]; c[0] = x; Ext(c) }
    pub fn from_u64(x: u64) -> Ext { Ext::from_base((x % P64) as u32) }
    #[inline(always)]
    pub fn scale(self, s: u32) -> Ext { let mut c = self.0; for x in c.iter_mut() { *x = mul(*x, s); } Ext(c) }
    pub fn is_zero(self) -> bool { self == Ext::ZERO }
    pub fn to_bytes(self) -> [u8; 24] {
        let mut b = [0u8; 24];
        for i in 0..DEG { b[4 * i..4 * i + 4].copy_from_slice(&self.0[i].to_le_bytes()); }
        b
    }
    /// Parses six LE u32 words; `None` if a coordinate is not canonical (`>= p`).
    pub fn from_bytes(b: &[u8]) -> Option<Ext> {
        let mut c = [0u32; DEG];
        for i in 0..DEG {
            let v = u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
            if v >= P { return None; }
            c[i] = v;
        }
        Some(Ext(c))
    }
    pub fn pow(self, mut e: u64) -> Ext {
        let mut b = self;
        let mut r = Ext::ONE;
        while e > 0 {
            if e & 1 == 1 { r = r * b; }
            b = b * b;
            e >>= 1;
        }
        r
    }
}

impl Add for Ext { type Output = Ext; #[inline(always)] fn add(self, o: Ext) -> Ext { Ext(std::array::from_fn(|i| add(self.0[i], o.0[i]))) } }
impl Sub for Ext { type Output = Ext; #[inline(always)] fn sub(self, o: Ext) -> Ext { Ext(std::array::from_fn(|i| sub(self.0[i], o.0[i]))) } }
impl Neg for Ext { type Output = Ext; #[inline(always)] fn neg(self) -> Ext { Ext(std::array::from_fn(|i| sub(0, self.0[i]))) } }
impl AddAssign for Ext { #[inline(always)] fn add_assign(&mut self, o: Ext) { *self = *self + o; } }
impl SubAssign for Ext { #[inline(always)] fn sub_assign(&mut self, o: Ext) { *self = *self - o; } }
impl MulAssign for Ext { #[inline(always)] fn mul_assign(&mut self, o: Ext) { *self = *self * o; } }
impl Mul for Ext {
    type Output = Ext;
    #[inline]
    fn mul(self, o: Ext) -> Ext {
        let a = self.0;
        let b = o.0;
        // acc[d] = sum_{i+j=d} a_i b_j, each product < 2^62; up to 6 terms: u128 accumulation, one fold per coefficient
        let mut acc = [0u128; 2 * DEG - 1];
        for i in 0..DEG {
            let ai = a[i] as u64;
            if ai == 0 { continue; }
            for j in 0..DEG { acc[i + j] += (ai * b[j] as u64) as u128; }
        }
        let mut c = [0u32; DEG];
        for d in 0..DEG {
            let mut x = acc[d];
            if d + DEG < 2 * DEG - 1 { x += acc[d + DEG] * W as u128; }
            c[d] = red128(x);
        }
        Ext(c)
    }
}

/// `v[i] <- v[i] + (v[i + half] - v[i]) r`, top-bit-first binding (`gpu/field.py::fold`).
pub fn fold_in_place(v: &mut Vec<Ext>, r: Ext) {
    let half = v.len() / 2;
    for i in 0..half {
        let lo = v[i];
        let hi = v[i + half];
        v[i] = lo + (hi - lo) * r;
    }
    v.truncate(half);
}

/// `eq(point, x)` for every `x` on the hypercube, top-bit-first (`gpu/field.py::eq_table`).
pub fn eq_table(point: &[Ext]) -> Vec<Ext> {
    let mut t = vec![Ext::ONE];
    for &r in point.iter().rev() {
        let one_minus = Ext::ONE - r;
        let mut nt = Vec::with_capacity(t.len() * 2);
        for &e in &t { nt.push(e * one_minus); }
        for &e in &t { nt.push(e * r); }
        t = nt;
    }
    t
}

pub fn eq_point(a: &[Ext], b: &[Ext]) -> Ext {
    let mut e = Ext::ONE;
    for (x, y) in a.iter().zip(b) { e = e * (*x * *y + (Ext::ONE - *x) * (Ext::ONE - *y)); }
    e
}

/// `s(r)` from `s(0..=d)` by Lagrange interpolation (`gpu/field.py::interpolate`).
pub fn interpolate(vals: &[Ext], r: Ext) -> Ext {
    let d = vals.len() - 1;
    let mut acc = Ext::ZERO;
    for i in 0..=d {
        let mut num = Ext::ONE;
        let mut den: i64 = 1;
        for j in 0..=d {
            if j == i { continue; }
            num = num * (r - Ext::from_u64(j as u64));
            den *= i as i64 - j as i64;
        }
        let den_f = if den < 0 { sub(0, ((-den) as u64 % P64) as u32) } else { (den as u64 % P64) as u32 };
        acc = acc + (vals[i] * num).scale(inv(den_f));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_arith() {
        assert_eq!(mul(P - 1, P - 1), 1);
        assert_eq!(mul(inv(12345), 12345), 1);
        assert_eq!(mont_mul(7, to_mont(9)), 63);
        assert_eq!(mont_mul(P - 1, to_mont(P - 1)), 1);
        let w = root_of_unity(16384);
        assert_eq!(pow(w, 16384), 1);
        assert_ne!(pow(w, 8192), 1);
        // gpu/field.py::_twiddles: root_of_unity(16384) = 31^((p-1)/2^14)
        assert_eq!(w, 1657000625);
        // W is a non-square and non-cube: X^6 - W irreducible
        assert_ne!(pow(W, (P64 - 1) / 2), 1);
        assert_ne!(pow(W, (P64 - 1) / 3), 1);
    }

    #[test]
    fn ext_mul_matches_python() {
        // gpu/field.py::e_mul on (1,2,3,4,5,6) x (6,5,4,3,2,1) computed by hand:
        // acc = [6,17,32,50,70,91,70,50,32,17,6]; c_d = acc[d] + W acc[d+6] for d<5, c_5 = 91
        let a = Ext([1, 2, 3, 4, 5, 6]);
        let b = Ext([6, 5, 4, 3, 2, 1]);
        assert_eq!(a * b, Ext([6 + W * 70, 17 + W * 50, 32 + W * 32, 50 + W * 17, 70 + W * 6, 91]));
        let x5 = Ext([0, 0, 0, 0, 0, 1]);
        let x1 = Ext([0, 1, 0, 0, 0, 0]);
        assert_eq!(x5 * x1, Ext::from_base(W));
        let big = Ext([P - 1; 6]);
        assert_eq!(big * Ext::ONE, big);
        assert_eq!((big * big) * a, big * (big * a));
    }

    #[test]
    fn interpolation_and_eq() {
        let s = |t: u64| Ext::from_u64(2 + 3 * t + 5 * t * t + 7 * t * t * t);
        let vals = [s(0), s(1), s(2), s(3)];
        assert_eq!(interpolate(&vals, Ext::from_u64(11)), s(11));
        let point = [Ext::from_u64(5), Ext::from_u64(9), Ext::from_u64(13)];
        let t = eq_table(&point);
        assert_eq!(t[6], point[0] * point[1] * (Ext::ONE - point[2]));
        let mut v = vec![Ext::ZERO; 8];
        v[6] = Ext::ONE;
        for &r in &point { fold_in_place(&mut v, r); }
        assert_eq!(v[0], t[6]);
        // sum over the cube is 1
        let mut s = Ext::ZERO;
        for e in &t { s += *e; }
        assert_eq!(s, Ext::ONE);
    }
}
