//! BabyBear p = 15 * 2^27 + 1 and its degree-6 extension F_{p^6} = F_p[X] / (X^6 - 31) (31 = the multiplicative
//! generator of F_p is neither a square nor a cube and 6 | p - 1, so X^6 - 31 is irreducible; the same tower as
//! `archive/direct/ligero/v2/fc.py` and `archive/direct/ligerito/proto/ext.py`).  The base-field part is the
//! ligero-verify `field.rs` (Barrett reduction), copied so the two crates stay independent.
//!
//! Extension elements are 6 reduced coordinates, coordinate d = the coefficient of X^d.  Serialized as 6 little-endian
//! u32 words (coordinate 0 first), exactly the numpy `(..., 6)` int64 layout of the Python side.

pub const P: u32 = 15 * (1 << 27) + 1;
pub const GEN: u32 = 31;
pub const BETA: u32 = 31;
pub const TWO_ADICITY: u32 = 27;
pub const EXT_DEG: usize = 6;
const P64: u64 = P as u64;

#[inline(always)]
pub fn add(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P { s - P } else { s }
}

#[inline(always)]
pub fn sub(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + P - b }
}

/// floor(2^64 / p): Barrett reduction of a product < 2^62 (the estimate is short by at most one p; two conditional
/// subtractions make it exact).
const BARRETT_M: u64 = ((1u128 << 64) / P as u128) as u64;

#[inline(always)]
pub fn reduce62(x: u64) -> u32 {
    debug_assert!(x < 1 << 62);
    let q = ((x as u128 * BARRETT_M as u128) >> 64) as u64;
    let mut r = x - q * P64;
    if r >= P64 {
        r -= P64;
    }
    if r >= P64 {
        r -= P64;
    }
    r as u32
}

/// Barrett for any u64 (the quotient estimate can be short by a few p; loop).
#[inline(always)]
pub fn reduce64(x: u64) -> u32 {
    let q = ((x as u128 * BARRETT_M as u128) >> 64) as u64;
    let mut r = x - q * P64;
    while r >= P64 {
        r -= P64;
    }
    r as u32
}

#[inline(always)]
pub fn mul(a: u32, b: u32) -> u32 {
    reduce62(a as u64 * b as u64)
}

#[inline(always)]
pub fn neg(a: u32) -> u32 {
    if a == 0 { 0 } else { P - a }
}

pub fn pow(mut a: u32, mut e: u64) -> u32 {
    let mut r = 1u32;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, a);
        }
        a = mul(a, a);
        e >>= 1;
    }
    r
}

pub fn inv(a: u32) -> u32 {
    pow(a, P64 - 2)
}

pub fn root_of_unity(n: usize) -> u32 {
    assert!(n.is_power_of_two() && n <= 1 << TWO_ADICITY, "root_of_unity({n})");
    pow(GEN, (P64 - 1) / n as u64)
}

/// Twiddles and bit-reversal for one size (the reference prover encodes with it; the verifier never needs an NTT).
pub struct Ntt {
    pub n: usize,
    tw: Vec<u32>,
    tw_inv: Vec<u32>,
    n_inv: u32,
    bitrev: Vec<u32>,
}

impl Ntt {
    pub fn new(n: usize) -> Ntt {
        let w = root_of_unity(n);
        let wi = inv(w);
        let mut tw = Vec::with_capacity(n / 2);
        let mut tw_inv = Vec::with_capacity(n / 2);
        let (mut a, mut b) = (1u32, 1u32);
        for _ in 0..n / 2 {
            tw.push(a);
            tw_inv.push(b);
            a = mul(a, w);
            b = mul(b, wi);
        }
        let bits = n.trailing_zeros();
        let bitrev = (0..n as u32).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (32 - bits) }).collect();
        Ntt { n, tw, tw_inv, n_inv: inv(n as u32), bitrev }
    }

    /// In place, natural order in and out.  Forward: X[k] = sum_i x[i] w^(ik).  Inverse divides by n.
    pub fn transform(&self, x: &mut [u32], inverse: bool) {
        let n = self.n;
        assert_eq!(x.len(), n);
        for i in 0..n {
            let j = self.bitrev[i] as usize;
            if j > i {
                x.swap(i, j);
            }
        }
        let tw = if inverse { &self.tw_inv } else { &self.tw };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut i = 0;
            while i < n {
                for j in 0..half {
                    let u = x[i + j];
                    let v = mul(x[i + j + half], tw[j * step]);
                    x[i + j] = add(u, v);
                    x[i + j + half] = sub(u, v);
                }
                i += len;
            }
            len *= 2;
        }
        if inverse {
            for v in x.iter_mut() {
                *v = mul(*v, self.n_inv);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------ F_{p^6} ----

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Ext(pub [u32; EXT_DEG]);

impl Ext {
    pub const ZERO: Ext = Ext([0; EXT_DEG]);
    pub const ONE: Ext = Ext([1, 0, 0, 0, 0, 0]);

    #[inline]
    pub fn from_base(a: u32) -> Ext {
        Ext([a, 0, 0, 0, 0, 0])
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&c| c == 0)
    }

    /// True when the element lies in the base field (coordinates 1..5 zero).
    #[inline]
    pub fn is_base(&self) -> bool {
        self.0[1..].iter().all(|&c| c == 0)
    }

    #[inline]
    pub fn add(&self, o: &Ext) -> Ext {
        let mut r = [0u32; EXT_DEG];
        for i in 0..EXT_DEG {
            r[i] = add(self.0[i], o.0[i]);
        }
        Ext(r)
    }

    #[inline]
    pub fn sub(&self, o: &Ext) -> Ext {
        let mut r = [0u32; EXT_DEG];
        for i in 0..EXT_DEG {
            r[i] = sub(self.0[i], o.0[i]);
        }
        Ext(r)
    }

    #[inline]
    pub fn neg(&self) -> Ext {
        let mut r = [0u32; EXT_DEG];
        for i in 0..EXT_DEG {
            r[i] = neg(self.0[i]);
        }
        Ext(r)
    }

    /// Schoolbook product; every coordinate product is reduced before the (<= 6-term) sums, then the X^6 = 31 wrap.
    #[inline]
    pub fn mul(&self, o: &Ext) -> Ext {
        let a = &self.0;
        let b = &o.0;
        let mut t = [0u64; 2 * EXT_DEG - 1];
        for i in 0..EXT_DEG {
            if a[i] == 0 {
                continue;
            }
            for j in 0..EXT_DEG {
                t[i + j] += reduce62(a[i] as u64 * b[j] as u64) as u64;
            }
        }
        let mut r = [0u32; EXT_DEG];
        for c in 0..EXT_DEG {
            let mut v = t[c];
            if c + EXT_DEG < 2 * EXT_DEG - 1 {
                v += t[c + EXT_DEG] * BETA as u64;
            }
            r[c] = reduce64(v);
        }
        Ext(r)
    }

    /// Extension times base scalar.
    #[inline]
    pub fn mul_base(&self, s: u32) -> Ext {
        let mut r = [0u32; EXT_DEG];
        for i in 0..EXT_DEG {
            r[i] = mul(self.0[i], s);
        }
        Ext(r)
    }

    pub fn pow(&self, mut e: u64) -> Ext {
        let mut r = Ext::ONE;
        let mut b = *self;
        while e > 0 {
            if e & 1 == 1 {
                r = r.mul(&b);
            }
            b = b.mul(&b);
            e >>= 1;
        }
        r
    }

    /// Frobenius a -> a^p: diagonal in the power basis since X^p = omega6 X mod (X^6 - 31), omega6 = 31^((p-1)/6).
    fn frobenius(&self) -> Ext {
        let omega6 = pow(BETA, (P64 - 1) / 6);
        let mut r = [0u32; EXT_DEG];
        let mut w = 1u32;
        for i in 0..EXT_DEG {
            r[i] = mul(self.0[i], w);
            w = mul(w, omega6);
        }
        Ext(r)
    }

    /// Inverse by the norm: a^-1 = a^(p + .. + p^5) / N(a), N(a) in F_p.  Zero -> zero.
    pub fn inv(&self) -> Ext {
        if self.is_zero() {
            return Ext::ZERO;
        }
        let mut f = self.frobenius();
        let mut r = f;
        for _ in 0..4 {
            f = f.frobenius();
            r = r.mul(&f);
        }
        let norm = self.mul(&r);
        debug_assert!(norm.is_base());
        r.mul_base(inv(norm.0[0]))
    }

    /// 24 little-endian bytes (6 x u32 LE).
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut o = [0u8; 24];
        for i in 0..EXT_DEG {
            o[4 * i..4 * i + 4].copy_from_slice(&self.0[i].to_le_bytes());
        }
        o
    }

    /// From 24 bytes; `None` when a coordinate is not reduced (the verifier rejects non-canonical encodings).
    pub fn from_bytes(b: &[u8]) -> Option<Ext> {
        if b.len() != 24 {
            return None;
        }
        let mut r = [0u32; EXT_DEG];
        for i in 0..EXT_DEG {
            let v = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
            if v >= P {
                return None;
            }
            r[i] = v;
        }
        Some(Ext(r))
    }
}

/// eq-pair of a point coordinate: (1 - z, z).
#[inline]
pub fn eq_pair(z: &Ext) -> (Ext, Ext) {
    (Ext::ONE.sub(z), *z)
}

/// sum_i a_i b_i for an extension vector against a base vector.
pub fn dot_ext_base(a: &[Ext], b: &[u32]) -> Ext {
    debug_assert_eq!(a.len(), b.len());
    // accumulate per coordinate in u64 (each product < 2^62; reduce every 3 products to stay < 2^64)
    let mut acc = [0u64; EXT_DEG];
    let mut pending = 0;
    for (x, &s) in a.iter().zip(b.iter()) {
        for c in 0..EXT_DEG {
            acc[c] += x.0[c] as u64 * s as u64;
        }
        pending += 1;
        if pending == 3 {
            for c in 0..EXT_DEG {
                acc[c] = reduce64(acc[c]) as u64;
            }
            pending = 0;
        }
    }
    let mut r = [0u32; EXT_DEG];
    for c in 0..EXT_DEG {
        r[c] = reduce64(acc[c]);
    }
    Ext(r)
}

/// sum_i a_i b_i for two extension vectors.
pub fn dot_ext(a: &[Ext], b: &[Ext]) -> Ext {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = Ext::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc.add(&x.mul(y));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    fn rand_ext(x: &mut u64) -> Ext {
        let mut r = [0u32; EXT_DEG];
        for c in r.iter_mut() {
            *c = (xorshift(x) % P64) as u32;
        }
        Ext(r)
    }

    #[test]
    fn barrett_matches_the_divide() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..100_000 {
            let a = (xorshift(&mut x) as u32) % P;
            let b = (xorshift(&mut x) as u32) % P;
            assert_eq!(mul(a, b), ((a as u64 * b as u64) % P64) as u32);
            let big = xorshift(&mut x);
            assert_eq!(reduce64(big), (big % P64) as u32);
        }
        assert_eq!(mul(P - 1, P - 1), 1);
    }

    #[test]
    fn ext_ring_axioms_and_inverse() {
        let mut x = 0xDEADBEEFu64;
        for _ in 0..2000 {
            let (a, b, c) = (rand_ext(&mut x), rand_ext(&mut x), rand_ext(&mut x));
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
            assert_eq!(a.mul(&Ext::ONE), a);
            if !a.is_zero() {
                assert_eq!(a.mul(&a.inv()), Ext::ONE);
            }
            let s = (xorshift(&mut x) % P64) as u32;
            assert_eq!(a.mul_base(s), a.mul(&Ext::from_base(s)));
        }
    }

    #[test]
    fn x6_equals_beta() {
        let xx = Ext([0, 1, 0, 0, 0, 0]);
        assert_eq!(xx.pow(6), Ext::from_base(BETA));
        // order of X divides p^6 - 1, and X is not in a proper subfield: X^(p^k) != X for k = 1, 2, 3
        let frob = xx.frobenius();
        assert_ne!(frob, xx);
        assert_eq!(frob, xx.pow(P64));
    }

    #[test]
    fn ntt_matches_naive() {
        let n = 16;
        let ntt = Ntt::new(n);
        let coefs: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % P).collect();
        let mut x = coefs.clone();
        ntt.transform(&mut x, false);
        let w = root_of_unity(n);
        for k in 0..n {
            let pt = pow(w, k as u64);
            let mut acc = 0u32;
            for c in coefs.iter().rev() {
                acc = add(mul(acc, pt), *c);
            }
            assert_eq!(x[k], acc);
        }
        ntt.transform(&mut x, true);
        assert_eq!(x, coefs);
    }

    #[test]
    fn bytes_roundtrip_and_reject_unreduced() {
        let e = Ext([1, 2, 3, P - 1, 5, 6]);
        assert_eq!(Ext::from_bytes(&e.to_bytes()), Some(e));
        let mut b = e.to_bytes();
        b[0..4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(Ext::from_bytes(&b), None);
    }

    #[test]
    fn dots_agree() {
        let mut x = 7u64;
        let a: Vec<Ext> = (0..50).map(|_| rand_ext(&mut x)).collect();
        let b: Vec<u32> = (0..50).map(|_| (xorshift(&mut x) % P64) as u32).collect();
        let be: Vec<Ext> = b.iter().map(|&s| Ext::from_base(s)).collect();
        assert_eq!(dot_ext_base(&a, &b), dot_ext(&a, &be));
    }
}
