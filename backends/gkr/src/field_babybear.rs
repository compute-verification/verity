//! BabyBear p = 2^31 - 2^27 + 1 and its degree-6 extension F_p[X]/(X^6 - 22) (PROTOCOL.md 13).  Same API as
//! `field.rs` (selected by `--features babybear`): `Fp` is a canonical `u32` in [0, p), `Ext` six coordinates
//! c0..c5 with x = Σ c_i X^i.  22 is the smallest non-square non-cube of F_p, so X^6 - 22 is irreducible
//! (2, 3 | p - 1 and 4 ∤ 6): this is the polynomial of `tensor/reference.py` (its `W = _find_w()`), which the GPU
//! lane's INT8 sumcheck kernel is checked against, so the two provers agree coordinate by coordinate.
//!
//! Multiplication is `(a·b) % p` on u64 (the compiler emits a multiply-high reduction), the extension product is
//! schoolbook (36 base products, u128 accumulation, one reduction per coefficient).  No Montgomery form: the base
//! elements are the witness's integers and are hashed / serialised as 4-byte little-endian words.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const P: u64 = 2013265921; // 2^31 - 2^27 + 1
const P32: u32 = P as u32;
/// X^6 - W: W = 22, the smallest a with a^((p-1)/2) != 1 and a^((p-1)/3) != 1.
pub const W: u32 = 22;
pub const DEGREE: usize = 6;
pub const NAME: &str = "babybear";
pub const EXT_NAME: &str = "babybear^6 (X^6-22)";
/// Fixture subdirectory of this field's exports (`fixtures/babybear/`).
#[allow(dead_code)]
pub const FIXTURES: &str = "babybear/";
pub const DOMAIN: &[u8] = b"verity-gkr/candidate-a/babybear6/v1";

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Fp(pub u32);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);
    pub const BYTES: usize = 4;
    pub fn to_u64(self) -> u64 { self.0 as u64 }
    #[inline]
    pub fn new(x: u64) -> Fp { Fp((x % P) as u32) }
    pub fn from_i64(x: i64) -> Fp { if x < 0 { -Fp::new((-x) as u64) } else { Fp::new(x as u64) } }
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
    #[inline]
    pub fn to_bytes(self) -> [u8; 4] { self.0.to_le_bytes() }
    #[inline]
    pub fn from_bytes(b: &[u8]) -> Fp { Fp::new(u32::from_le_bytes(b[..4].try_into().unwrap()) as u64) }
    /// A uniform element from a 32-byte digest by rejection sampling over its eight 4-byte words (each word is
    /// accepted iff < p); None if all eight are rejected.
    pub fn rejection_sample(d: &[u8; 32]) -> Option<Fp> {
        for i in 0..8 {
            let v = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap());
            if v < P32 { return Some(Fp(v)); }
        }
        None
    }
    /// A primitive `order`-th root of unity (order a power of two ≤ 2^27): 31 generates F_p^*, p - 1 = 2^27 · 15.
    pub fn root_of_unity(order: usize) -> Fp {
        assert!(order.is_power_of_two() && order <= 1 << 27, "order {order} not a power of two <= 2^27");
        let mut w = Fp(31).pow(15); // order 2^27
        let mut o = 1u64 << 27;
        while o > order as u64 { w = w * w; o /= 2; }
        w
    }
}

impl Add for Fp { type Output = Fp; #[inline] fn add(self, o: Fp) -> Fp { let s = self.0 + o.0; Fp(if s >= P32 { s - P32 } else { s }) } }
impl Sub for Fp { type Output = Fp; #[inline] fn sub(self, o: Fp) -> Fp { let (d, b) = self.0.overflowing_sub(o.0); Fp(if b { d.wrapping_add(P32) } else { d }) } }
impl Mul for Fp { type Output = Fp; #[inline] fn mul(self, o: Fp) -> Fp { Fp(((self.0 as u64 * o.0 as u64) % P) as u32) } }
impl Neg for Fp { type Output = Fp; #[inline] fn neg(self) -> Fp { Fp::ZERO - self } }
impl AddAssign for Fp { fn add_assign(&mut self, o: Fp) { *self = *self + o; } }
impl SubAssign for Fp { fn sub_assign(&mut self, o: Fp) { *self = *self - o; } }
impl MulAssign for Fp { fn mul_assign(&mut self, o: Fp) { *self = *self * o; } }

/// F_{p^6} = F_p[X]/(X^6 - 22); coordinates (c0, ..., c5).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Ext(pub [Fp; DEGREE]);

#[inline]
fn red128(x: u128) -> u64 {
    // x < 6 p^2 < 2^65: fold the high word (2^64 mod p) into the low one
    let lo = (x as u64) % P;
    let hi = (x >> 64) as u64;
    (lo + hi * ((1u128 << 64) % P as u128) as u64) % P
}

impl Ext {
    pub const ZERO: Ext = Ext([Fp::ZERO; DEGREE]);
    pub const ONE: Ext = Ext([Fp::ONE, Fp::ZERO, Fp::ZERO, Fp::ZERO, Fp::ZERO, Fp::ZERO]);
    pub const BYTES: usize = DEGREE * Fp::BYTES;
    #[inline]
    pub fn from_base(x: Fp) -> Ext { let mut c = [Fp::ZERO; DEGREE]; c[0] = x; Ext(c) }
    pub fn from_u64(x: u64) -> Ext { Ext::from_base(Fp::new(x)) }
    #[inline]
    pub fn scale(self, s: Fp) -> Ext { let mut c = self.0; for x in c.iter_mut() { *x = *x * s; } Ext(c) }
    pub fn is_zero(self) -> bool { self == Ext::ZERO }
    pub fn to_bytes(self) -> [u8; 24] {
        let mut b = [0u8; 24];
        for i in 0..DEGREE { b[4 * i..4 * i + 4].copy_from_slice(&self.0[i].to_bytes()); }
        b
    }
    pub fn from_bytes(b: &[u8]) -> Ext { Ext(std::array::from_fn(|i| Fp::from_bytes(&b[4 * i..4 * i + 4]))) }
}

impl Add for Ext { type Output = Ext; #[inline] fn add(self, o: Ext) -> Ext { Ext(std::array::from_fn(|i| self.0[i] + o.0[i])) } }
impl Sub for Ext { type Output = Ext; #[inline] fn sub(self, o: Ext) -> Ext { Ext(std::array::from_fn(|i| self.0[i] - o.0[i])) } }
impl Neg for Ext { type Output = Ext; #[inline] fn neg(self) -> Ext { Ext(std::array::from_fn(|i| -self.0[i])) } }
impl Mul for Ext {
    type Output = Ext;
    #[inline]
    fn mul(self, o: Ext) -> Ext {
        let a = self.0;
        let b = o.0;
        let mut acc = [0u128; 2 * DEGREE - 1];
        for i in 0..DEGREE {
            let ai = a[i].0 as u64;
            if ai == 0 { continue; }
            for j in 0..DEGREE { acc[i + j] += (ai * b[j].0 as u64) as u128; }
        }
        // X^6 = W
        let mut c = [Fp::ZERO; DEGREE];
        for d in 0..DEGREE {
            let lo = red128(acc[d]);
            let hi = if d + DEGREE < 2 * DEGREE - 1 { red128(acc[d + DEGREE]) } else { 0 };
            c[d] = Fp(((lo + hi * W as u64) % P) as u32);
        }
        Ext(c)
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
    fn ext(v: [u64; 6]) -> Ext { Ext(std::array::from_fn(|i| Fp::new(v[i]))) }

    #[test]
    fn field_basics_against_python() {
        // tensor/reference.py, seed 2026: x, y, x*y, x^-1
        let (x, y) = (Fp::new(1195201396), Fp::new(903263729));
        assert_eq!(x * y, Fp::new(1136544590));
        assert_eq!(x.inv(), Fp::new(851042159));
        assert_eq!(x * x.inv(), Fp::ONE);
        let a = Fp::new(P - 1);
        assert_eq!(a + Fp::ONE, Fp::ZERO);
        assert_eq!(a * a, Fp::ONE);
        assert_eq!(Fp::new(P + 5), Fp::new(5));
        assert_eq!(Fp::from_i64(-1), a);
        // 22 is neither a square nor a cube: X^6 - 22 is irreducible
        assert_ne!(Fp(W).pow((P - 1) / 2), Fp::ONE);
        assert_ne!(Fp(W).pow((P - 1) / 3), Fp::ONE);
        for a in 2..W { assert!(Fp(a).pow((P - 1) / 2) == Fp::ONE || Fp(a).pow((P - 1) / 3) == Fp::ONE, "{a} is a smaller non-residue"); }
    }

    #[test]
    fn ext_mul_matches_reference_py() {
        // reference.py mul() on random_ext(Random(2026)) twice: a, b, a*b, a*a
        let a = ext([255808012, 686087736, 1079144365, 1099130909, 1389847499, 1846837192]);
        let b = ext([220410946, 1898632805, 479499921, 1921583681, 1290703502, 1334626946]);
        assert_eq!(a * b, ext([175958147, 1851771178, 1307694065, 1448085535, 205114162, 1846661519]));
        assert_eq!(a * a, ext([1934860814, 620066652, 856038404, 1361338718, 171897836, 283944028]));
        assert_eq!(a * Ext::ONE, a);
        assert_eq!(a * b, b * a);
        // X^5 · X = W
        let mut x5 = [Fp::ZERO; 6];
        x5[5] = Fp::ONE;
        let mut x1 = [Fp::ZERO; 6];
        x1[1] = Fp::ONE;
        assert_eq!(Ext(x5) * Ext(x1), Ext::from_u64(W as u64));
        // bytes round trip: 6 x 4-byte LE words
        assert_eq!(Ext::from_bytes(&a.to_bytes()), a);
    }

    #[test]
    fn roots_of_unity() {
        assert_eq!(Fp::root_of_unity(1 << 14), Fp::new(1657000625)); // 31^((p-1)/2^14)
        let w = Fp::root_of_unity(16384);
        assert_eq!(w.pow(16384), Fp::ONE);
        assert_ne!(w.pow(8192), Fp::ONE);
    }

    #[test]
    fn eq_table_matches_eval() {
        let point = [Ext::from_u64(5), Ext::from_u64(9), Ext::from_u64(13)];
        let t = eq_table(&point);
        let mut v = vec![Ext::ZERO; 8];
        v[6] = Ext::ONE;
        assert_eq!(eval_mle(&v, &point), t[6]);
        assert_eq!(t[6], point[0] * point[1] * (Ext::ONE - point[2]));
    }

    #[test]
    fn interpolation() {
        let s = |t: u64| Ext::from_u64(2 + 3 * t + 5 * t * t);
        let vals = [s(0), s(1), s(2)];
        assert_eq!(interpolate(&vals, Ext::from_u64(7)), s(7));
    }
}
