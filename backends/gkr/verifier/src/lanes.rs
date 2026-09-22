//! Lane-parallel BabyBear kernels for the interleaved (`L` lanes per NTT point) layout of `ntt.rs`: one twiddle,
//! `L` independent values.  With AVX-512 (runtime-detected) the Montgomery product runs on 16 x u32 per register
//! in the even/odd `vpmuludq` split (2 x 3 products + 1 blend per 16 lanes, no 64-bit `vpmullq`); otherwise the
//! scalar loop of `field.rs` (which LLVM vectorises on 8 x u64 lanes).  Every kernel is bit-exact with the scalar
//! path: same Montgomery constants, same canonical `[0, p)` outputs.

use crate::field::{add, mont_mul, sub};

#[cfg(target_arch = "x86_64")]
fn avx512() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static FLAG: AtomicU8 = AtomicU8::new(0);
    match FLAG.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = std::is_x86_feature_detected!("avx512f") && std::env::var("VERITY_GKR_NO_AVX512").map(|v| v != "1").unwrap_or(true);
            FLAG.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn avx512() -> bool { false }

/// Decimation-in-time butterfly: `t = b w; a' = a + t; b' = a - t` (`w` in Montgomery form).
#[inline(always)]
pub fn bf_dit<const L: usize>(a: &mut [u32; L], b: &mut [u32; L], w: u32) {
    #[cfg(target_arch = "x86_64")]
    if L % 16 == 0 && avx512() { unsafe { x86::bf_dit(a, b, w) }; return; }
    for l in 0..L {
        let t = mont_mul(b[l], w);
        let u = a[l];
        a[l] = add(u, t);
        b[l] = sub(u, t);
    }
}

/// Decimation-in-frequency butterfly: `a' = a + b; b' = (a - b) w`.
#[inline(always)]
pub fn bf_dif<const L: usize>(a: &mut [u32; L], b: &mut [u32; L], w: u32) {
    #[cfg(target_arch = "x86_64")]
    if L % 16 == 0 && avx512() { unsafe { x86::bf_dif(a, b, w) }; return; }
    for l in 0..L {
        let u = a[l];
        let x = b[l];
        a[l] = add(u, x);
        b[l] = mont_mul(sub(u, x), w);
    }
}

/// `dst = src w`.
#[inline(always)]
pub fn mul_lanes<const L: usize>(dst: &mut [u32; L], src: &[u32; L], w: u32) {
    #[cfg(target_arch = "x86_64")]
    if L % 16 == 0 && avx512() { unsafe { x86::mul_lanes(dst, src, w) }; return; }
    for l in 0..L { dst[l] = mont_mul(src[l], w); }
}

/// `o += y w`.
#[inline(always)]
pub fn mul_add_lanes<const L: usize>(o: &mut [u32; L], y: &[u32; L], w: u32) {
    #[cfg(target_arch = "x86_64")]
    if L % 16 == 0 && avx512() { unsafe { x86::mul_add_lanes(o, y, w) }; return; }
    for l in 0..L { o[l] = add(o[l], mont_mul(y[l], w)); }
}

/// `a += e` / `a -= e` on one extension element (six canonical lanes), for the lookup accumulators.
#[inline(always)]
pub fn acc6_add(a: &mut [u32; 6], e: &[u32; 6]) {
    #[cfg(target_arch = "x86_64")]
    if avx512() { unsafe { x86::acc6(a, e, false) }; return; }
    for i in 0..6 { a[i] = add(a[i], e[i]); }
}
#[inline(always)]
pub fn acc6_sub(a: &mut [u32; 6], e: &[u32; 6]) {
    #[cfg(target_arch = "x86_64")]
    if avx512() { unsafe { x86::acc6(a, e, true) }; return; }
    for i in 0..6 { a[i] = sub(a[i], e[i]); }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::field::{NEG_PINV, P};
    use std::arch::x86_64::*;

    /// Montgomery product of 16 lanes by a broadcast `w` (Montgomery form): `(x w_R + m p) >> 32`, canonical.
    #[inline(always)]
    unsafe fn mont16(x: __m512i, w: __m512i, mu: __m512i, p: __m512i) -> __m512i {
        let x_odd = _mm512_srli_epi64::<32>(x);
        let prod_evn = _mm512_mul_epu32(x, w);
        let prod_odd = _mm512_mul_epu32(x_odd, w);
        let m_evn = _mm512_mul_epu32(prod_evn, mu);
        let m_odd = _mm512_mul_epu32(prod_odd, mu);
        let d_evn = _mm512_add_epi64(prod_evn, _mm512_mul_epu32(m_evn, p));
        let d_odd = _mm512_add_epi64(prod_odd, _mm512_mul_epu32(m_odd, p));
        // high 32 bits of every 64-bit lane, interleaved back: even results from d_evn >> 32, odd from d_odd's
        // high halves (already in the odd u32 positions)
        let t = _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64::<32>(d_evn), d_odd);
        // t in [0, 2p): canonicalise
        _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
    }
    #[inline(always)]
    unsafe fn add16(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
        let s = _mm512_add_epi32(a, b);
        _mm512_min_epu32(s, _mm512_sub_epi32(s, p))
    }
    #[inline(always)]
    unsafe fn sub16(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
        let d = _mm512_sub_epi32(a, b);
        _mm512_min_epu32(d, _mm512_add_epi32(d, p))
    }

    /// Masked 6-lane add/sub mod p (AVX512VL 256-bit ops; the two upper lanes are never touched).
    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn acc6(a: &mut [u32; 6], e: &[u32; 6], negate: bool) {
        let p = _mm256_set1_epi32(P as i32);
        let x = _mm256_maskz_loadu_epi32(0x3f, a.as_ptr() as *const i32);
        let y = _mm256_maskz_loadu_epi32(0x3f, e.as_ptr() as *const i32);
        let r = if negate {
            let d = _mm256_sub_epi32(x, y);
            _mm256_min_epu32(d, _mm256_add_epi32(d, p))
        } else {
            let s = _mm256_add_epi32(x, y);
            _mm256_min_epu32(s, _mm256_sub_epi32(s, p))
        };
        _mm256_mask_storeu_epi32(a.as_mut_ptr() as *mut i32, 0x3f, r);
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn bf_dit<const L: usize>(a: &mut [u32; L], b: &mut [u32; L], w: u32) {
        let (wv, mu, p) = (_mm512_set1_epi32(w as i32), _mm512_set1_epi32(NEG_PINV as i32), _mm512_set1_epi32(P as i32));
        for i in (0..L).step_by(16) {
            let pa = a.as_mut_ptr().add(i) as *mut i32;
            let pb = b.as_mut_ptr().add(i) as *mut i32;
            let u = _mm512_loadu_si512(pa as *const _);
            let t = mont16(_mm512_loadu_si512(pb as *const _), wv, mu, p);
            _mm512_storeu_si512(pa as *mut _, add16(u, t, p));
            _mm512_storeu_si512(pb as *mut _, sub16(u, t, p));
        }
    }
    #[target_feature(enable = "avx512f")]
    pub unsafe fn bf_dif<const L: usize>(a: &mut [u32; L], b: &mut [u32; L], w: u32) {
        let (wv, mu, p) = (_mm512_set1_epi32(w as i32), _mm512_set1_epi32(NEG_PINV as i32), _mm512_set1_epi32(P as i32));
        for i in (0..L).step_by(16) {
            let pa = a.as_mut_ptr().add(i) as *mut i32;
            let pb = b.as_mut_ptr().add(i) as *mut i32;
            let u = _mm512_loadu_si512(pa as *const _);
            let x = _mm512_loadu_si512(pb as *const _);
            _mm512_storeu_si512(pa as *mut _, add16(u, x, p));
            _mm512_storeu_si512(pb as *mut _, mont16(sub16(u, x, p), wv, mu, p));
        }
    }
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul_lanes<const L: usize>(dst: &mut [u32; L], src: &[u32; L], w: u32) {
        let (wv, mu, p) = (_mm512_set1_epi32(w as i32), _mm512_set1_epi32(NEG_PINV as i32), _mm512_set1_epi32(P as i32));
        for i in (0..L).step_by(16) {
            let x = _mm512_loadu_si512(src.as_ptr().add(i) as *const _);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut _, mont16(x, wv, mu, p));
        }
    }
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul_add_lanes<const L: usize>(o: &mut [u32; L], y: &[u32; L], w: u32) {
        let (wv, mu, p) = (_mm512_set1_epi32(w as i32), _mm512_set1_epi32(NEG_PINV as i32), _mm512_set1_epi32(P as i32));
        for i in (0..L).step_by(16) {
            let po = o.as_mut_ptr().add(i) as *mut i32;
            let x = _mm512_loadu_si512(y.as_ptr().add(i) as *const _);
            let acc = _mm512_loadu_si512(po as *const _);
            _mm512_storeu_si512(po as *mut _, add16(acc, mont16(x, wv, mu, p), p));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{mont_mul, P};

    fn rng(seed: &mut u64) -> u32 { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; (*seed % P as u64) as u32 }

    #[test]
    fn kernels_match_scalar() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..200 {
            let a: [u32; 48] = std::array::from_fn(|_| rng(&mut s));
            let b: [u32; 48] = std::array::from_fn(|_| rng(&mut s));
            let w = rng(&mut s);
            let (mut a1, mut b1) = (a, b);
            bf_dit(&mut a1, &mut b1, w);
            for l in 0..48 { let t = mont_mul(b[l], w); assert_eq!(a1[l], add(a[l], t)); assert_eq!(b1[l], sub(a[l], t)); }
            let (mut a2, mut b2) = (a, b);
            bf_dif(&mut a2, &mut b2, w);
            for l in 0..48 { assert_eq!(a2[l], add(a[l], b[l])); assert_eq!(b2[l], mont_mul(sub(a[l], b[l]), w)); }
            let mut d = [0u32; 48];
            mul_lanes(&mut d, &a, w);
            for l in 0..48 { assert_eq!(d[l], mont_mul(a[l], w)); }
            let mut o = b;
            mul_add_lanes(&mut o, &a, w);
            for l in 0..48 { assert_eq!(o[l], add(b[l], mont_mul(a[l], w))); }
        }
        // acc6
        for _ in 0..200 {
            let a: [u32; 6] = std::array::from_fn(|_| rng(&mut s));
            let e: [u32; 6] = std::array::from_fn(|_| rng(&mut s));
            let mut x = a; acc6_add(&mut x, &e); for i in 0..6 { assert_eq!(x[i], add(a[i], e[i])); }
            let mut y = a; acc6_sub(&mut y, &e); for i in 0..6 { assert_eq!(y[i], sub(a[i], e[i])); }
        }
        // edge values
        let a = [P - 1; 48];
        let b = [P - 1; 48];
        let (mut a1, mut b1) = (a, b);
        bf_dit(&mut a1, &mut b1, P - 1);
        for l in 0..48 { let t = mont_mul(P - 1, P - 1); assert_eq!(a1[l], add(P - 1, t)); assert_eq!(b1[l], sub(P - 1, t)); }
    }
}
