//! Goldilocks p = 2^64 - 2^32 + 1 and its cubic extension F_p[X]/(X^3 - X - 1) (irreducible: checked by
//! gcd(X^p - X, f) = 1 over F_p before this file was written).  Elements are kept canonical in [0, p).

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const P: u64 = 0xFFFF_FFFF_0000_0001;
const EPS: u64 = 0xFFFF_FFFF; // 2^64 mod p

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Fp(pub u64);

#[inline]
fn canon(x: u64) -> u64 {
    if x >= P { x - P } else { x }
}

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);
    #[inline]
    pub fn new(x: u64) -> Fp { Fp(canon(x)) }
    pub fn from_i64(x: i64) -> Fp { if x < 0 { -Fp::new((-x) as u64) } else { Fp::new(x as u64) } }
    #[inline]
    pub fn reduce128(x: u128) -> Fp {
        let lo = canon(x as u64);
        let hi = (x >> 64) as u64;
        let hi_hi = hi >> 32;
        let hi_lo = hi & EPS;
        // x = lo + hi_lo 2^64 + hi_hi 2^96 = lo + hi_lo (2^32 - 1) - hi_hi   (mod p)
        let t = Fp(lo) - Fp(hi_hi);
        t + Fp(canon(hi_lo * EPS))
    }
    pub fn pow(self, mut e: u64) -> Fp {
        let mut b = self;
        let mut r = Fp::ONE;
        while e > 0 {
            if e & 1 == 1 { r *= b; }
            b *= b;
            e >>= 1;
        }
        r
    }
    pub fn inv(self) -> Fp { assert!(self.0 != 0, "inverse of zero"); self.pow(P - 2) }
    pub fn to_bytes(self) -> [u8; 8] { self.0.to_le_bytes() }
}

impl Add for Fp { type Output = Fp; #[inline] fn add(self, o: Fp) -> Fp { let (s, c) = self.0.overflowing_add(o.0); if c || s >= P { Fp(s.wrapping_sub(P)) } else { Fp(s) } } }
impl Sub for Fp { type Output = Fp; #[inline] fn sub(self, o: Fp) -> Fp { let (d, b) = self.0.overflowing_sub(o.0); if b { Fp(d.wrapping_add(P)) } else { Fp(d) } } }
impl Mul for Fp { type Output = Fp; #[inline] fn mul(self, o: Fp) -> Fp { Fp::reduce128(self.0 as u128 * o.0 as u128) } }
impl Neg for Fp { type Output = Fp; #[inline] fn neg(self) -> Fp { Fp::ZERO - self } }
impl AddAssign for Fp { fn add_assign(&mut self, o: Fp) { *self = *self + o; } }
impl SubAssign for Fp { fn sub_assign(&mut self, o: Fp) { *self = *self - o; } }
impl MulAssign for Fp { fn mul_assign(&mut self, o: Fp) { *self = *self * o; } }

/// F_{p^3} = F_p[X]/(X^3 - X - 1); coordinates (c0, c1, c2).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Ext(pub [Fp; 3]);

impl Ext {
    pub const ZERO: Ext = Ext([Fp::ZERO, Fp::ZERO, Fp::ZERO]);
    pub const ONE: Ext = Ext([Fp::ONE, Fp::ZERO, Fp::ZERO]);
    #[inline]
    pub fn from_base(x: Fp) -> Ext { Ext([x, Fp::ZERO, Fp::ZERO]) }
    pub fn from_u64(x: u64) -> Ext { Ext::from_base(Fp::new(x)) }
    #[inline]
    pub fn scale(self, s: Fp) -> Ext { Ext([self.0[0] * s, self.0[1] * s, self.0[2] * s]) }
    pub fn is_zero(self) -> bool { self == Ext::ZERO }
    pub fn to_bytes(self) -> [u8; 24] {
        let mut b = [0u8; 24];
        for i in 0..3 { b[8 * i..8 * i + 8].copy_from_slice(&self.0[i].to_bytes()); }
        b
    }
    pub fn from_bytes(b: &[u8]) -> Ext {
        let g = |i: usize| Fp::new(u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()));
        Ext([g(0), g(1), g(2)])
    }
}

impl Add for Ext { type Output = Ext; #[inline] fn add(self, o: Ext) -> Ext { Ext([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]]) } }
impl Sub for Ext { type Output = Ext; #[inline] fn sub(self, o: Ext) -> Ext { Ext([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]]) } }
impl Neg for Ext { type Output = Ext; #[inline] fn neg(self) -> Ext { Ext([-self.0[0], -self.0[1], -self.0[2]]) } }
impl Mul for Ext {
    type Output = Ext;
    #[inline]
    fn mul(self, o: Ext) -> Ext {
        let a = self.0;
        let b = o.0;
        let d0 = a[0] * b[0];
        let d1 = a[0] * b[1] + a[1] * b[0];
        let d2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
        let d3 = a[1] * b[2] + a[2] * b[1];
        let d4 = a[2] * b[2];
        // X^3 = X + 1, X^4 = X^2 + X
        Ext([d0 + d3, d1 + d3 + d4, d2 + d4])
    }
}
impl AddAssign for Ext { fn add_assign(&mut self, o: Ext) { *self = *self + o; } }
impl SubAssign for Ext { fn sub_assign(&mut self, o: Ext) { *self = *self - o; } }
impl MulAssign for Ext { fn mul_assign(&mut self, o: Ext) { *self = *self * o; } }

/// Multilinear helpers on vectors indexed by the hypercube; variables are bound from the top bit down, and a
/// point is the list of challenges in that order.  Prover and verifier share these, so the convention is fixed.
pub fn fold_in_place(v: &mut Vec<Ext>, r: Ext) {
    let half = v.len() / 2;
    for i in 0..half {
        let lo = v[i];
        let hi = v[i + half];
        v[i] = lo + (hi - lo) * r;
    }
    v.truncate(half);
}

pub fn eval_mle(values: &[Ext], point: &[Ext]) -> Ext {
    assert_eq!(values.len(), 1 << point.len());
    let mut v = values.to_vec();
    for &r in point { fold_in_place(&mut v, r); }
    v[0]
}

/// eq(point, x) for every x on the hypercube, in the same top-bit-first convention.
pub fn eq_table(point: &[Ext]) -> Vec<Ext> {
    // Each step doubles the table with the new coordinate as the top bit, so the coordinates are consumed from
    // the last (lowest bit) to the first (top bit).
    let mut t = vec![Ext::ONE];
    for &r in point.iter().rev() {
        let mut nt = Vec::with_capacity(t.len() * 2);
        let one_minus = Ext::ONE - r;
        for &e in &t { nt.push(e * one_minus); }
        for &e in &t { nt.push(e * r); }
        t = nt;
    }
    t
}

/// s(r) from its values at 0..=d by Lagrange interpolation (denominators are small public integers).
pub fn interpolate(vals: &[Ext], r: Ext) -> Ext {
    let d = vals.len() - 1;
    let mut acc = Ext::ZERO;
    for i in 0..=d {
        let mut num = Ext::ONE;
        let mut den = Fp::ONE;
        for j in 0..=d {
            if j == i { continue; }
            num = num * (r - Ext::from_u64(j as u64));
            den = den * Fp::from_i64(i as i64 - j as i64);
        }
        acc = acc + (vals[i] * num).scale(den.inv());
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_basics() {
        let a = Fp::new(P - 1);
        assert_eq!(a + Fp::ONE, Fp::ZERO);
        assert_eq!(a * a, Fp::ONE);
        let x = Fp::new(0x1234_5678_9abc_def0);
        assert_eq!(x * x.inv(), Fp::ONE);
        let y = Fp::reduce128((P as u128 - 1) * (P as u128 - 1));
        assert_eq!(y, Fp::ONE);
    }
    #[test]
    fn ext_mul_matches_poly() {
        let a = Ext([Fp::new(3), Fp::new(5), Fp::new(7)]);
        let b = Ext([Fp::new(11), Fp::new(13), Fp::new(17)]);
        let c = a * b;
        // (3+5x+7x^2)(11+13x+17x^2) = 33 + 94x + 193x^2 + 176x^3 + 119x^4; x^3=x+1, x^4=x^2+x
        assert_eq!(c, Ext([Fp::new(33 + 176), Fp::new(94 + 176 + 119), Fp::new(193 + 119)]));
        assert_eq!(a * Ext::ONE, a);
    }
    #[test]
    fn eq_table_matches_eval() {
        let point = [Ext::from_u64(5), Ext::from_u64(9), Ext::from_u64(13)];
        let t = eq_table(&point);
        let mut v = vec![Ext::ZERO; 8];
        v[6] = Ext::ONE; // index 6 = 110b: top bit 1, middle 1, low 0
        assert_eq!(eval_mle(&v, &point), t[6]);
        let expect = point[0] * point[1] * (Ext::ONE - point[2]);
        assert_eq!(t[6], expect);
    }
    #[test]
    fn interpolation() {
        // s(t) = 2 + 3t + 5t^2
        let s = |t: u64| Ext::from_u64(2 + 3 * t + 5 * t * t);
        let vals = [s(0), s(1), s(2)];
        assert_eq!(interpolate(&vals, Ext::from_u64(7)), s(7));
    }
}
