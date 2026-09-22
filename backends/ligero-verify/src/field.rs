//! BabyBear p = 15 * 2^27 + 1 (PROTOCOL.md section 2), the radix-2 NTT and the coset code (section 3):
//! a row of `l` values = evaluations of a polynomial of degree < l on H = <w_l> (index i -> w_l^i); its codeword =
//! evaluations on the coset g <w_n>, g = 31, i.e. INTT_l, zero-pad, scale coefficient i by g^i, NTT_n.

pub const P: u32 = 15 * (1 << 27) + 1;
pub const GEN: u32 = 31;
pub const TWO_ADICITY: u32 = 27;
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
/// subtractions make it exact).  A `%` by a u64 is a 20-40 cycle divide on Zen3; this is two multiplies.
const BARRETT_M: u64 = ((1u128 << 64) / P as u128) as u64;

#[inline(always)]
pub fn reduce62(x: u64) -> u32 {
    debug_assert!(x < 1 << 62);
    let q = ((x as u128 * BARRETT_M as u128) >> 64) as u64;
    let mut r = x - q * P64; // q <= x / p, so this is x mod p or x mod p + p
    if r >= P64 {
        r -= P64;
    }
    if r >= P64 {
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

/// `acc + a * b mod p` for a reduced `acc` (< p): the product is < 2^62 and the sum < 2^62 + 2^31 < 2^62 + p, so one
/// Barrett reduction of the sum is exact.  Callers keep the running sum in a u64 and get a reduced value back.
#[inline(always)]
pub fn mul_acc(acc: u64, a: u32, b: u32) -> u64 {
    debug_assert!(acc < P64);
    reduce62_plus(acc + a as u64 * b as u64) as u64
}

/// Barrett for x < 2^62 + 2^32 (the quotient estimate can still only be short by one).
#[inline(always)]
fn reduce62_plus(x: u64) -> u32 {
    let q = ((x as u128 * BARRETT_M as u128) >> 64) as u64;
    let mut r = x - q * P64;
    while r >= P64 {
        r -= P64;
    }
    r as u32
}

pub fn root_of_unity(n: usize) -> u32 {
    assert!(n.is_power_of_two() && n <= 1 << TWO_ADICITY);
    pow(GEN, (P64 - 1) / n as u64)
}

/// Twiddles and bit-reversal for one size.
pub struct Ntt {
    pub n: usize,
    tw: Vec<u32>,      // w^j, j < n/2
    tw_inv: Vec<u32>,  // w^-j
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

/// eta_c = g w_n^c for every column.
pub fn coset_points(n: usize) -> Vec<u32> {
    let w = root_of_unity(n);
    let mut out = Vec::with_capacity(n);
    let mut cur = GEN;
    for _ in 0..n {
        out.push(cur);
        cur = mul(cur, w);
    }
    out
}

/// g^i for i < n (the coset scaling of coefficients).
pub fn coset_scale(n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    let mut cur = 1u32;
    for _ in 0..n {
        out.push(cur);
        cur = mul(cur, GEN);
    }
    out
}

/// Evaluations on the coset g <w_n> of the polynomial with coefficients `coefs` (len <= n): one NTT.
pub fn eval_on_coset(coefs: &[u32], ntt_n: &Ntt, scale: &[u32]) -> Vec<u32> {
    let n = ntt_n.n;
    assert!(coefs.len() <= n);
    let mut x = vec![0u32; n];
    for (i, c) in coefs.iter().enumerate() {
        x[i] = mul(*c, scale[i]);
    }
    ntt_n.transform(&mut x, false);
    x
}

/// Codeword of a message row (evaluations on H, len l): INTT_l, pad, scale, NTT_n.
pub fn encode(row: &[u32], ntt_l: &Ntt, ntt_n: &Ntt, scale: &[u32]) -> Vec<u32> {
    let mut c = row.to_vec();
    ntt_l.transform(&mut c, true);
    eval_on_coset(&c, ntt_n, scale)
}

#[cfg(test)]
mod tests {
    #[test]
    fn barrett_matches_the_divide() {
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..200_000 {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let (a, b) = ((x as u32) % super::P, ((x >> 32) as u32) % super::P);
            assert_eq!(super::mul(a, b), ((a as u64 * b as u64) % super::P as u64) as u32);
            let acc = (x >> 3) as u32 % super::P;
            assert_eq!(super::mul_acc(acc as u64, a, b), (acc as u64 + a as u64 * b as u64) % super::P as u64);
        }
        assert_eq!(super::mul(super::P - 1, super::P - 1), 1);
        assert_eq!(super::mul_acc((super::P - 1) as u64, super::P - 1, super::P - 1), 0);
    }

    use super::*;

    fn naive_eval(coefs: &[u32], x: u32) -> u32 {
        let mut acc = 0u32;
        for c in coefs.iter().rev() {
            acc = add(mul(acc, x), *c);
        }
        acc
    }

    #[test]
    fn ntt_matches_naive_and_inverts() {
        let n = 16;
        let ntt = Ntt::new(n);
        let coefs: Vec<u32> = (0..n as u32).map(|i| (i * 7919 + 13) % P).collect();
        let mut x = coefs.clone();
        ntt.transform(&mut x, false);
        let w = root_of_unity(n);
        for k in 0..n {
            assert_eq!(x[k], naive_eval(&coefs, pow(w, k as u64)));
        }
        ntt.transform(&mut x, true);
        assert_eq!(x, coefs);
    }

    #[test]
    fn coset_encode_is_evaluation_on_the_coset() {
        let (l, n) = (8, 32);
        let (nl, nn) = (Ntt::new(l), Ntt::new(n));
        let sc = coset_scale(n);
        let pts = coset_points(n);
        let coefs: Vec<u32> = (0..l as u32).map(|i| (i * 104729 + 5) % P).collect();
        // message = evaluations of coefs on H
        let mut msg = coefs.clone();
        nl.transform(&mut msg, false);
        let cw = encode(&msg, &nl, &nn, &sc);
        for c in 0..n {
            assert_eq!(cw[c], naive_eval(&coefs, pts[c]));
        }
        // Z_H(eta) = eta^l - 1 never vanishes on the coset
        for c in 0..n {
            assert_ne!(pow(pts[c], l as u64), 1);
        }
    }
}
