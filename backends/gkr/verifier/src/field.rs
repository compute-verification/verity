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
pub const NEG_PINV: u32 = {
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
/// Montgomery reduction `x R^-1 mod p` (canonical) of an accumulator `x < 2^62`.
#[inline(always)]
pub fn mont_reduce(x: u64) -> u32 {
    let m = (x as u32).wrapping_mul(NEG_PINV);
    let u = ((x + m as u64 * P64) >> 32) as u32;
    if u >= P { u - P } else { u }
}
/// A product `x < 2^62` folded to a congruent value `< 2^58 + 2^32` (`2^32 = R_MOD_P mod p`), so six of them sum in a
/// `u64` without overflow.
#[inline(always)]
pub fn fold64(x: u64) -> u64 { (x >> 32) * R_MOD_P as u64 + (x & 0xffff_ffff) }

/// The `6 x 6` matrix of "multiply by `v`" in Montgomery form: `(x v)_d = sum_i x_i M[i][d]`, `M[i][d] = v_{d-i}`
/// (`d >= i`) or `W v_{d-i+6}`; stored `[i * 6 + d]`, scaled by `R` so `mont_reduce(sum_i fold64(x_i M[i][d]))` is
/// canonical `(x v)_d`.
pub fn mul_matrix(v: Ext) -> [u32; DEG * DEG] {
    let mut m = [0u32; DEG * DEG];
    for i in 0..DEG {
        for d in 0..DEG {
            let raw = if d >= i { v.0[d - i] } else { mul(W, v.0[d - i + DEG]) };
            m[i * DEG + d] = to_mont(raw);
        }
    }
    m
}

/// `x * v` through its matrix (same result as `x * v`).
#[inline(always)]
pub fn mat_apply(x: &Ext, m: &[u32; DEG * DEG]) -> Ext {
    let mut c = [0u32; DEG];
    for d in 0..DEG {
        let mut s = 0u64;
        for i in 0..DEG { s += fold64(x.0[i] as u64 * m[i * DEG + d] as u64); }
        c[d] = mont_reduce(s);
    }
    Ext(c)
}

/// Planar Montgomery matrices of `n` extension elements (`[(i * 6 + d) * n + c]`), for `x * v_c` over all `c` as a
/// vectorised loop (`matvec_planar`).
pub struct PlanarMats { pub n: usize, pub m: Vec<u32> }
impl PlanarMats {
    pub fn new(vs: &[Ext]) -> PlanarMats {
        let n = vs.len();
        let mut m = vec![0u32; DEG * DEG * n];
        for (c, v) in vs.iter().enumerate() {
            let mm = mul_matrix(*v);
            for i in 0..DEG { for d in 0..DEG { m[(i * DEG + d) * n + c] = mm[i * DEG + d]; } }
        }
        PlanarMats { n, m }
    }
    /// `out[d * n + c] = (x * v_c)_d` (planar output).
    pub fn matvec(&self, x: &Ext, out: &mut [u32]) {
        let n = self.n;
        debug_assert_eq!(out.len(), DEG * n);
        for d in 0..DEG {
            let (x0, x1, x2, x3, x4, x5) = (x.0[0] as u64, x.0[1] as u64, x.0[2] as u64, x.0[3] as u64, x.0[4] as u64, x.0[5] as u64);
            let m0 = &self.m[d * n..(d + 1) * n];
            let m1 = &self.m[(DEG + d) * n..(DEG + d + 1) * n];
            let m2 = &self.m[(2 * DEG + d) * n..(2 * DEG + d + 1) * n];
            let m3 = &self.m[(3 * DEG + d) * n..(3 * DEG + d + 1) * n];
            let m4 = &self.m[(4 * DEG + d) * n..(4 * DEG + d + 1) * n];
            let m5 = &self.m[(5 * DEG + d) * n..(5 * DEG + d + 1) * n];
            let o = &mut out[d * n..(d + 1) * n];
            for c in 0..n {
                let s = fold64(x0 * m0[c] as u64) + fold64(x1 * m1[c] as u64) + fold64(x2 * m2[c] as u64)
                    + fold64(x3 * m3[c] as u64) + fold64(x4 * m4[c] as u64) + fold64(x5 * m5[c] as u64);
                o[c] = mont_reduce(s);
            }
        }
    }
}

/// `out[d * n + r] = (a_r * g)_d` for planar accumulators `a[i * n + r]` and one matrix `g` (the R16 flush).
pub fn planar_times_matrix(a: &[u32], n: usize, g: &[u32; DEG * DEG], out: &mut [u32]) {
    debug_assert_eq!(a.len(), DEG * n);
    for d in 0..DEG {
        let (g0, g1, g2, g3, g4, g5) = (g[d] as u64, g[DEG + d] as u64, g[2 * DEG + d] as u64, g[3 * DEG + d] as u64, g[4 * DEG + d] as u64, g[5 * DEG + d] as u64);
        let (a0, a1, a2, a3, a4, a5) = (&a[..n], &a[n..2 * n], &a[2 * n..3 * n], &a[3 * n..4 * n], &a[4 * n..5 * n], &a[5 * n..6 * n]);
        let o = &mut out[d * n..(d + 1) * n];
        for r in 0..n {
            let s = fold64(a0[r] as u64 * g0) + fold64(a1[r] as u64 * g1) + fold64(a2[r] as u64 * g2)
                + fold64(a3[r] as u64 * g3) + fold64(a4[r] as u64 * g4) + fold64(a5[r] as u64 * g5);
            o[r] = mont_reduce(s);
        }
    }
}

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
    fn matrix_products_match_mul() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % P64) as u32 };
        let mut vs = Vec::new();
        for _ in 0..37 { vs.push(Ext(std::array::from_fn(|_| rnd()))); }
        vs.push(Ext([P - 1; 6]));
        vs.push(Ext::ZERO);
        let xs = [Ext(std::array::from_fn(|_| rnd())), Ext([P - 1; 6]), Ext::ONE, Ext::ZERO];
        let pm = PlanarMats::new(&vs);
        let mut out = vec![0u32; DEG * vs.len()];
        for x in &xs {
            for v in &vs { assert_eq!(mat_apply(x, &mul_matrix(*v)), *x * *v); }
            pm.matvec(x, &mut out);
            for (c, v) in vs.iter().enumerate() {
                let e: [u32; DEG] = std::array::from_fn(|d| out[d * vs.len() + c]);
                assert_eq!(Ext(e), *x * *v);
            }
            // planar accumulators times one matrix
            let n = vs.len();
            let mut a = vec![0u32; DEG * n];
            for (r, v) in vs.iter().enumerate() { for i in 0..DEG { a[i * n + r] = v.0[i]; } }
            planar_times_matrix(&a, n, &mul_matrix(*x), &mut out);
            for (r, v) in vs.iter().enumerate() {
                let e: [u32; DEG] = std::array::from_fn(|d| out[d * n + r]);
                assert_eq!(Ext(e), *v * *x);
            }
        }
        assert_eq!(mont_reduce(to_mont(12345) as u64), 12345);
        assert_eq!(fold64((P as u64 - 1) * (P as u64 - 1)) % P64, ((P as u64 - 1) * (P as u64 - 1)) % P64);
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
