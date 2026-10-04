
// ---------------------------------------------------------------------------
// rec-holo (Verity, backends/flock/pod/rec_holo_gkr_stream.rs, appended to this file by 78-rec-holo-kill.sh): a
// streaming form of `prove_batched` for the copy-constraint case f = g = w with w a bit vector.
// ---------------------------------------------------------------------------

use rayon::prelude::ParallelSlice;

/// One eq-weighted round message, as [`batched_round_message`], over a layer that is not stored: `src(j)` gives the four
/// half-slice values `(L[j], L[j + h], R[j], R[j + h])`.
fn rh_round_message<S: Fn(usize) -> [F128; 4] + Sync>(src: &S, lambda: F128, eq: &SplitEqGhash) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len();
    let n_blocks = hi.len();
    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let [l0a, l1a, r0a, r1a] = src(2 * xp);
            let [l0b, l1b, r0b, r1b] = src(2 * xp + 1);
            let v_one = l0b * l1b + lambda * (r0b * r1b);
            let v_inf = (l0a + l0b) * (l1a + l1b) + lambda * ((r0a + r0b) * (r1a + r1b));
            let el = lo[x_lo];
            s1 += el * v_one;
            s_inf += el * v_inf;
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };
    match sumcheck_round_min_len(block * n_blocks, n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(|| (F128::ZERO, F128::ZERO), |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1)),
        None => (0..n_blocks).map(block_fn).fold((F128::ZERO, F128::ZERO), |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1)),
    }
}

/// [`fold_and_message`] from a layer that is not stored: writes `half` folded values of each of `src`'s four slices into
/// `dst` and returns the next round's message over them.
fn rh_fold_and_message<S: Fn(usize) -> [F128; 4] + Sync>(
    src: &S,
    half: usize,
    rho: F128,
    dst: [&mut [F128]; 4],
    lambda: F128,
    eq_next: Option<&SplitEqGhash>,
) -> Option<(F128, F128)> {
    let one_minus = F128::ONE + rho;
    let fold_at = |x: usize| -> [F128; 4] {
        let a = src(2 * x);
        let b = src(2 * x + 1);
        [
            a[0] * one_minus + b[0] * rho,
            a[1] * one_minus + b[1] * rho,
            a[2] * one_minus + b[2] * rho,
            a[3] * one_minus + b[3] * rho,
        ]
    };
    let [d0, d1, d2, d3] = dst;
    let Some(eq) = eq_next else {
        const CH: usize = 1 << 12;
        d0[..half]
            .par_chunks_mut(CH)
            .zip(d1[..half].par_chunks_mut(CH))
            .zip(d2[..half].par_chunks_mut(CH))
            .zip(d3[..half].par_chunks_mut(CH))
            .enumerate()
            .for_each(|(ci, (((c0, c1), c2), c3))| {
                for t in 0..c0.len() {
                    let v = fold_at(ci * CH + t);
                    (c0[t], c1[t], c2[t], c3[t]) = (v[0], v[1], v[2], v[3]);
                }
            });
        return None;
    };
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len();
    let n_blocks = hi.len();
    debug_assert_eq!(block * n_blocks, half / 2);
    let chunk = 2 * block;
    let body = |x_hi: usize, c: [&mut [F128]; 4]| -> (F128, F128) {
        let base = x_hi * chunk;
        let [c0, c1, c2, c3] = c;
        for t in 0..c0.len() {
            let v = fold_at(base + t);
            (c0[t], c1[t], c2[t], c3[t]) = (v[0], v[1], v[2], v[3]);
        }
        let (mut acc1, mut acc_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let (i0, i1) = (2 * x_lo, 2 * x_lo + 1);
            let v_one = c0[i1] * c1[i1] + lambda * (c2[i1] * c3[i1]);
            let v_inf = (c0[i0] + c0[i1]) * (c1[i0] + c1[i1]) + lambda * ((c2[i0] + c2[i1]) * (c3[i0] + c3[i1]));
            let el = lo[x_lo];
            acc1 += el * v_one;
            acc_inf += el * v_inf;
        }
        let eh = hi[x_hi];
        (eh * acc1, eh * acc_inf)
    };
    let msg = match sumcheck_round_min_len(block * n_blocks, n_blocks) {
        Some(min_len) => d0[..half]
            .par_chunks_mut(chunk)
            .zip(d1[..half].par_chunks_mut(chunk))
            .zip(d2[..half].par_chunks_mut(chunk))
            .zip(d3[..half].par_chunks_mut(chunk))
            .with_min_len(min_len)
            .enumerate()
            .map(|(x_hi, (((c0, c1), c2), c3))| body(x_hi, [c0, c1, c2, c3]))
            .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d)),
        None => d0[..half]
            .chunks_mut(chunk)
            .zip(d1[..half].chunks_mut(chunk))
            .zip(d2[..half].chunks_mut(chunk))
            .zip(d3[..half].chunks_mut(chunk))
            .enumerate()
            .map(|(x_hi, (((c0, c1), c2), c3))| body(x_hi, [c0, c1, c2, c3]))
            .fold((F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d)),
    };
    Some(msg)
}

/// `ŝ_σ(ρ)` straight from `σ`'s indices: `Σ_x eq(ρ, x)·tag(σ(x))`, low variable first, as [`mle_eval`] of the tag table.
pub fn rec_holo_sigma_eval(sigma: &[u32], rho: &[F128]) -> F128 {
    let nu = rho.len().min(16);
    let eq_lo = build_eq(&rho[..nu]);
    let eq_hi = build_eq(&rho[nu..]);
    sigma
        .par_chunks(1 << nu)
        .enumerate()
        .map(|(b, c)| {
            let mut acc = F128::ZERO;
            for (j, &s) in c.iter().enumerate() {
                acc += eq_lo[j] * F128::new(s as u64, 0);
            }
            eq_hi[b] * acc
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

/// [`prove_batched`]`(w, w, σ, None, ch)` without grinding, transcript-identical, for a bit vector `w` (cell `x` is bit
/// `x & 63` of `w[x >> 6]`) and `σ` as `u32` indices. It never stores the leaves, `f`, `g` or `s_σ`: each leaf is
/// recomputed from `w` and `σ` when a pass reads it.
///
/// - `unstored`: how many layers below the leaves are recomputed rather than stored (≥ 1; 1 keeps layers 0..μ−1).
/// - `double`: on a step whose source layer is recomputed, bind the first two rounds from the source before writing any
///   folded values, so the step's buffers start at a quarter of the source instead of a half.
pub fn rec_holo_prove_batched_stream<C: Challenger>(
    w: &[u64],
    sigma: &[u32],
    unstored: usize,
    double: bool,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    let n = sigma.len();
    assert!(n.is_power_of_two() && n >= 64 && w.len() * 64 == n);
    let mu = n.trailing_zeros() as usize;
    assert!(unstored >= 1 && unstored < mu, "need 1 ≤ unstored < μ");
    ch.observe_label(DOMAIN_BATCHED);
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();

    let leaf = |x: usize| -> (F128, F128) {
        let b = if (w[x >> 6] >> (x & 63)) & 1 == 1 { F128::ONE } else { F128::ZERO };
        (b + alpha * F128::new(x as u64, 0) + beta, b + alpha * F128::new(sigma[x] as u64, 0) + beta)
    };
    // V_s[i] = Π_{t < 2^(μ−s)} leaf(i + t·2^s), both circuits.
    let value = |s: usize, i: usize| -> (F128, F128) {
        let (mut l, mut r) = leaf(i);
        for t in 1..1usize << (mu - s) {
            let (a, b) = leaf(i + (t << s));
            l *= a;
            r *= b;
        }
        (l, r)
    };
    let top = mu - unstored;
    let mut l_layers: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
    let mut r_layers: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
    let (lt, rt): (Vec<F128>, Vec<F128>) = (0..1usize << top).into_par_iter().map(|i| value(top, i)).unzip();
    l_layers[top] = lt;
    r_layers[top] = rt;
    for k in (0..top).rev() {
        l_layers[k] = build_layer(&l_layers[k + 1]);
        r_layers[k] = build_layer(&r_layers[k + 1]);
    }
    let top_lhs = l_layers[0][0];
    let top_rhs = r_layers[0][0];
    ch.observe_f128(top_lhs);
    ch.observe_f128(top_rhs);

    let mut r_pt: Vec<F128> = Vec::new();
    let mut layers = Vec::with_capacity(mu);
    let (mut claim_l, mut claim_r) = (F128::ZERO, F128::ZERO);
    for k in 0..mu {
        let lambda = ch.sample_f128();
        let h = 1usize << k;
        let s = k + 1;
        let stored = s <= top;
        let mut rounds: Vec<(F128, F128)> = Vec::with_capacity(k);
        let mut r_prime: Vec<F128> = Vec::with_capacity(k + 1);
        let src = |j: usize| -> [F128; 4] {
            let (l0, r0) = value(s, j);
            let (l1, r1) = value(s, j + h);
            [l0, l1, r0, r1]
        };
        let two = !stored && double && k >= 2;
        let first = (if two { h / 4 } else { h / 2 }).max(1);
        let mut cur: [Vec<F128>; 4] = from_fn(|_| crate::alloc_uninit_vec(first));
        let mut nxt: [Vec<F128>; 4] = from_fn(|_| crate::alloc_uninit_vec((first / 2).max(1)));
        let mut len = 0usize;
        let mut pending: Option<(F128, F128)> = None;
        let mut done = 0usize;
        if k > 0 {
            let eq0 = SplitEqGhash::new(&r_pt[1..k]);
            let m0 = if stored {
                let (l0s, l1s) = l_layers[s].split_at(h);
                let (r0s, r1s) = r_layers[s].split_at(h);
                batched_round_message(l0s, l1s, r0s, r1s, lambda, &eq0)
            } else {
                rh_round_message(&src, lambda, &eq0)
            };
            ch.observe_f128(m0.0);
            ch.observe_f128(m0.1);
            let rho0 = ch.sample_f128();
            rounds.push(m0);
            r_prime.push(rho0);
            let [d0, d1, d2, d3] = &mut cur;
            let dst = [d0, d1, d2, d3].map(|d| d.as_mut_slice());
            if two {
                let om = F128::ONE + rho0;
                let src1 = |x: usize| -> [F128; 4] {
                    let (a, b) = (src(2 * x), src(2 * x + 1));
                    [a[0] * om + b[0] * rho0, a[1] * om + b[1] * rho0, a[2] * om + b[2] * rho0, a[3] * om + b[3] * rho0]
                };
                let m1 = rh_round_message(&src1, lambda, &SplitEqGhash::new(&r_pt[2..k]));
                ch.observe_f128(m1.0);
                ch.observe_f128(m1.1);
                let rho1 = ch.sample_f128();
                rounds.push(m1);
                r_prime.push(rho1);
                let eq_next = (2 < k).then(|| SplitEqGhash::new(&r_pt[3..k]));
                pending = rh_fold_and_message(&src1, h / 4, rho1, dst, lambda, eq_next.as_ref());
                len = h / 4;
                done = 2;
            } else {
                let eq_next = (1 < k).then(|| SplitEqGhash::new(&r_pt[2..k]));
                pending = if stored {
                    let (l0s, l1s) = l_layers[s].split_at(h);
                    let (r0s, r1s) = r_layers[s].split_at(h);
                    fold_and_message([l0s, l1s, r0s, r1s], rho0, dst, lambda, eq_next.as_ref())
                } else {
                    rh_fold_and_message(&src, h / 2, rho0, dst, lambda, eq_next.as_ref())
                };
                len = h / 2;
                done = 1;
            }
        }
        for i in done..k {
            let (g1, g_inf) = pending.expect("round i's message was produced already");
            ch.observe_f128(g1);
            ch.observe_f128(g_inf);
            let rho = ch.sample_f128();
            rounds.push((g1, g_inf));
            r_prime.push(rho);
            let eq_next = (i + 1 < k).then(|| SplitEqGhash::new(&r_pt[i + 2..k]));
            let src = [&cur[0][..len], &cur[1][..len], &cur[2][..len], &cur[3][..len]];
            let [d0, d1, d2, d3] = &mut nxt;
            pending = fold_and_message(src, rho, [d0, d1, d2, d3].map(|d| d.as_mut_slice()), lambda, eq_next.as_ref());
            len /= 2;
            swap(&mut cur, &mut nxt);
        }
        let (vl0, vl1, vr0, vr1) = if k == 0 {
            if stored {
                (l_layers[1][0], l_layers[1][1], r_layers[1][0], r_layers[1][1])
            } else {
                let [a, b, c, d] = src(0);
                (a, b, c, d)
            }
        } else {
            debug_assert_eq!(len, 1, "layer {k}: folds must reduce to one element");
            (cur[0][0], cur[1][0], cur[2][0], cur[3][0])
        };
        for v in [vl0, vl1, vr0, vr1] {
            ch.observe_f128(v);
        }
        layers.push(BatchedLayerProof { rounds, vl0, vl1, vr0, vr1 });
        let c_k = ch.sample_f128();
        let one_plus_c = F128::ONE + c_k;
        claim_l = one_plus_c * vl0 + c_k * vl1;
        claim_r = one_plus_c * vr0 + c_k * vr1;
        r_prime.push(c_k);
        r_pt = r_prime;
        if stored {
            l_layers[s] = Vec::new();
            r_layers[s] = Vec::new();
        }
    }
    let rho = r_pt;
    let s_sigma_eval = rec_holo_sigma_eval(sigma, &rho);
    let f_eval = claim_l + alpha * s_id_eval(&s_id_basis(mu), &rho) + beta;
    let g_eval = claim_r + alpha * s_sigma_eval + beta;
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
    let proof =
        ProductGkrBatchedProof { top_lhs, top_rhs, layers, f_eval, g_eval, s_sigma_eval, grinding_nonces: Vec::new() };
    let claim = ProductGkrBatchedClaim { rho, f_eval, g_eval, s_sigma_eval };
    (proof, claim)
}

// ---------------------------------------------------------------------------
// The same prover on AVX-512: one register holds a pair's four values (L0, R0, L1, R1), a fold is one multiply
// (a + (a + b)·ρ), eq-weighted sums accumulate unreduced, and a leaf's tag α·v + β is four byte-table lookups (α·v is
// GF(2)-linear in v's bits).
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
pub use rh_fast::rec_holo_prove_batched_fast;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
mod rh_fast {
    use core::arch::x86_64::{
        __m512i, _mm_set_epi64x, _mm512_broadcast_i32x4, _mm512_bslli_epi128, _mm512_clmulepi64_epi128,
        _mm512_loadu_si512, _mm512_set_epi64, _mm512_setzero_si512, _mm512_shuffle_i64x2, _mm512_storeu_si512,
        _mm512_xor_si512,
    };
    use core::mem::transmute;

    use rayon::prelude::ParallelSlice;

    use super::*;
    use crate::field::gf2_128::x86_64::ghash_mul_x4;

    type V = __m512i;

    #[inline(always)]
    fn mul(x: V, y: V) -> V {
        // SAFETY: avx512f and vpclmulqdq are enabled for the whole build (the module's cfg).
        unsafe { ghash_mul_x4(x, y) }
    }

    #[inline(always)]
    fn xor(x: V, y: V) -> V {
        // SAFETY: as `mul`.
        unsafe { _mm512_xor_si512(x, y) }
    }

    #[inline(always)]
    fn zero() -> V {
        // SAFETY: as `mul`.
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    fn splat(a: F128) -> V {
        // SAFETY: as `mul`.
        unsafe { _mm512_broadcast_i32x4(_mm_set_epi64x(a.hi as i64, a.lo as i64)) }
    }

    #[inline(always)]
    fn set4(v: [F128; 4]) -> V {
        // SAFETY: F128 is repr(C) {lo, hi}, so four of them are the register's lanes 0..3 in order.
        unsafe { transmute::<[F128; 4], V>(v) }
    }

    #[inline(always)]
    fn lanes(v: V) -> [F128; 4] {
        // SAFETY: as `set4`; every bit pattern is an F128.
        unsafe { transmute::<V, [F128; 4]>(v) }
    }

    #[inline(always)]
    fn load(p: &[F128], i: usize) -> V {
        assert!(4 * i + 4 <= p.len());
        // SAFETY: four readable F128 at 4i (asserted).
        unsafe { _mm512_loadu_si512(p.as_ptr().add(4 * i).cast()) }
    }

    #[inline(always)]
    fn store(p: &mut [F128], i: usize, v: V) {
        assert!(4 * i + 4 <= p.len());
        // SAFETY: four writable F128 at 4i (asserted).
        unsafe { _mm512_storeu_si512(p.as_mut_ptr().add(4 * i).cast(), v) }
    }

    #[inline(always)]
    fn fold(a: V, b: V, r: V) -> V {
        xor(a, mul(xor(a, b), r))
    }

    /// For a pair (a, b) of entries `[L0, R0, L1, R1]`: `[b.L0·b.L1, b.R0·b.R1, s.L0·s.L1, s.R0·s.R1]` with s = a + b,
    /// the two circuits' gate values at 1 and at ∞.
    #[inline(always)]
    fn gates(a: V, b: V) -> V {
        let s = xor(a, b);
        // SAFETY: as `mul`.
        let (x, y) = unsafe { (_mm512_shuffle_i64x2::<0x44>(b, s), _mm512_shuffle_i64x2::<0xEE>(b, s)) };
        mul(x, y)
    }

    /// Four lanes of unreduced products, reduced once.
    #[derive(Clone, Copy)]
    struct Acc {
        lo: V,
        mid: V,
        hi: V,
    }

    impl Acc {
        #[inline(always)]
        fn new() -> Self {
            Self { lo: zero(), mid: zero(), hi: zero() }
        }

        #[inline(always)]
        fn add(&mut self, x: V, y: V) {
            // SAFETY: as `mul`.
            unsafe {
                self.lo = xor(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
                self.hi = xor(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
                self.mid = xor(
                    self.mid,
                    xor(_mm512_clmulepi64_epi128::<0x01>(x, y), _mm512_clmulepi64_epi128::<0x10>(x, y)),
                );
            }
        }

        /// `lo + x^64·(mid + x^64·hi)` mod the GHASH polynomial, per lane: `ghash_mul_x4`'s two reduction stages.
        #[inline(always)]
        fn reduce(self) -> V {
            // SAFETY: as `mul`.
            unsafe {
                let poly = _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87);
                let red = |t0: V, t1: V| {
                    xor(xor(t0, _mm512_bslli_epi128::<8>(t1)), _mm512_clmulepi64_epi128::<0x01>(t1, poly))
                };
                red(self.lo, red(self.mid, self.hi))
            }
        }
    }

    fn combine(v: V, lambda: F128) -> (F128, F128) {
        let [a, b, c, d] = lanes(v);
        (a + lambda * b, c + lambda * d)
    }

    /// One round's message over a source of entries: `src(j)` is entry j of a layer `[L0, R0, L1, R1]`.
    fn message<S: Fn(usize) -> V + Sync>(src: &S, lambda: F128, eq: &SplitEqGhash) -> (F128, F128) {
        let (lo, hi) = (&eq.lo, &eq.hi);
        let (block, n_blocks) = (lo.len(), hi.len());
        let block_fn = |x_hi: usize| -> V {
            let mut acc = Acc::new();
            for (x_lo, &el) in lo.iter().enumerate() {
                let xp = x_hi * block + x_lo;
                acc.add(gates(src(2 * xp), src(2 * xp + 1)), splat(el));
            }
            mul(acc.reduce(), splat(hi[x_hi]))
        };
        let v = match sumcheck_round_min_len(block * n_blocks, n_blocks) {
            Some(min_len) => (0..n_blocks).into_par_iter().with_min_len(min_len).map(block_fn).reduce(zero, xor),
            None => (0..n_blocks).map(block_fn).fold(zero(), xor),
        };
        combine(v, lambda)
    }

    /// Fold `half` entries of `src` at ρ into `dst` and, while they are in registers, the next round's message over them.
    fn fold_message<S: Fn(usize) -> V + Sync>(
        src: &S,
        half: usize,
        rho: F128,
        dst: &mut [F128],
        lambda: F128,
        eq_next: Option<&SplitEqGhash>,
    ) -> Option<(F128, F128)> {
        let r = splat(rho);
        let f = |x: usize| fold(src(2 * x), src(2 * x + 1), r);
        let dst = &mut dst[..4 * half];
        let Some(eq) = eq_next else {
            const CH: usize = 1 << 12;
            dst.par_chunks_mut(4 * CH).enumerate().for_each(|(ci, c)| {
                for t in 0..c.len() / 4 {
                    store(c, t, f(ci * CH + t));
                }
            });
            return None;
        };
        let (lo, hi) = (&eq.lo, &eq.hi);
        let (block, n_blocks) = (lo.len(), hi.len());
        debug_assert_eq!(block * n_blocks, half / 2);
        let chunk = 2 * block;
        let body = |x_hi: usize, c: &mut [F128]| -> V {
            let base = x_hi * chunk;
            let mut acc = Acc::new();
            for (x_lo, &el) in lo.iter().enumerate() {
                let (a, b) = (f(base + 2 * x_lo), f(base + 2 * x_lo + 1));
                store(c, 2 * x_lo, a);
                store(c, 2 * x_lo + 1, b);
                acc.add(gates(a, b), splat(el));
            }
            mul(acc.reduce(), splat(hi[x_hi]))
        };
        let v = match sumcheck_round_min_len(block * n_blocks, n_blocks) {
            Some(min_len) => dst
                .par_chunks_mut(4 * chunk)
                .with_min_len(min_len)
                .enumerate()
                .map(|(x_hi, c)| body(x_hi, c))
                .reduce(zero, xor),
            None => dst.chunks_mut(4 * chunk).enumerate().map(|(x_hi, c)| body(x_hi, c)).fold(zero(), xor),
        };
        Some(combine(v, lambda))
    }

    fn collect<S: Fn(usize) -> V + Sync>(n: usize, src: &S) -> Vec<F128> {
        const CH: usize = 1 << 12;
        let mut out: Vec<F128> = crate::alloc_uninit_vec(4 * n);
        out.par_chunks_mut(4 * CH).enumerate().for_each(|(ci, c)| {
            for t in 0..c.len() / 4 {
                store(c, t, src(ci * CH + t));
            }
        });
        out
    }

    /// Layer s − 1's entries from layer s's: entry y is `[A.L0·A.L1, A.R0·A.R1, B.L0·B.L1, B.R0·B.R1]` with A, B the
    /// entries y and y + n/2 of the n below.
    fn halve(q: &[F128]) -> Vec<F128> {
        let n = q.len() / 8;
        collect(n, &|y: usize| {
            let (a, b) = (load(q, y), load(q, y + n));
            // SAFETY: as `mul`.
            unsafe { mul(_mm512_shuffle_i64x2::<0x44>(a, b), _mm512_shuffle_i64x2::<0xEE>(a, b)) }
        })
    }

    struct Leaves<'a> {
        w: &'a [u64],
        sigma: &'a [u32],
        alpha: V,
        beta: V,
        mu: usize,
    }

    impl Leaves<'_> {
        /// `[i, σ(i), i + h, σ(i + h)]`'s leaves: each lane's tag `α·v + β` for its index v < 2^32 is one product's two
        /// low terms and one reduction, plus the cell's bit.
        #[inline(always)]
        fn lv(&self, i: usize, h: usize) -> V {
            let j = i + h;
            let bit = |x: usize| ((self.w[x >> 6] >> (x & 63)) & 1) as i64;
            let (bi, bj) = (bit(i), bit(j));
            // SAFETY: as `mul`.
            unsafe {
                let v = _mm512_set_epi64(0, self.sigma[j] as i64, 0, j as i64, 0, self.sigma[i] as i64, 0, i as i64);
                let t0 = _mm512_clmulepi64_epi128::<0x00>(v, self.alpha);
                let t1 = _mm512_clmulepi64_epi128::<0x10>(v, self.alpha);
                let poly = _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87);
                let t = xor(xor(t0, _mm512_bslli_epi128::<8>(t1)), _mm512_clmulepi64_epi128::<0x01>(t1, poly));
                xor(xor(t, self.beta), _mm512_set_epi64(0, bj, 0, bj, 0, bi, 0, bi))
            }
        }

        /// Entry x of layer s, `[VL_s(x), VR_s(x), VL_s(x + h), VR_s(x + h)]` with h = 2^(s−1), from the leaves.
        #[inline(always)]
        fn entry(&self, s: usize, x: usize) -> V {
            let h = 1usize << (s - 1);
            let lv = |i: usize| self.lv(i, h);
            let mut v = lv(x);
            for t in 1..1usize << (self.mu - s) {
                v = mul(v, lv(x + (t << s)));
            }
            v
        }
    }

    fn sigma_eval(sigma: &[u32], rho: &[F128]) -> F128 {
        let nu = rho.len().min(16);
        let eq_lo = build_eq(&rho[..nu]);
        let eq_hi = build_eq(&rho[nu..]);
        sigma
            .par_chunks(1 << nu)
            .enumerate()
            .map(|(b, c)| {
                let mut acc = Acc::new();
                for (j, s) in c.chunks_exact(4).enumerate() {
                    acc.add(load(&eq_lo, j), set4(from_fn(|i| F128::new(s[i] as u64, 0))));
                }
                let [a, b2, c2, d] = lanes(acc.reduce());
                eq_hi[b] * (a + b2 + c2 + d)
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    /// One layer's sumcheck (k rounds) over the entries `src` of layer k + 1, returning its rounds, its challenges and
    /// the last entry. `two`: bind the first two rounds from `src` before writing (k ≥ 2).
    fn layer<S: Fn(usize) -> V + Sync, C: Challenger>(
        src: &S,
        k: usize,
        r_pt: &[F128],
        lambda: F128,
        two: bool,
        ch: &mut C,
    ) -> (Vec<(F128, F128)>, Vec<F128>, V) {
        if k == 0 {
            return (Vec::new(), Vec::new(), src(0));
        }
        let h = 1usize << k;
        let mut rounds = Vec::with_capacity(k);
        let mut r_prime = Vec::with_capacity(k + 1);
        let mut round = |m: (F128, F128), ch: &mut C| -> F128 {
            ch.observe_f128(m.0);
            ch.observe_f128(m.1);
            let rho = ch.sample_f128();
            rounds.push(m);
            r_prime.push(rho);
            rho
        };
        let rho0 = round(message(src, lambda, &SplitEqGhash::new(&r_pt[1..k])), ch);
        let (mut cur, mut len, mut pending, done);
        if two && k >= 2 {
            let r0 = splat(rho0);
            let src1 = |x: usize| fold(src(2 * x), src(2 * x + 1), r0);
            let rho1 = round(message(&src1, lambda, &SplitEqGhash::new(&r_pt[2..k])), ch);
            len = h / 4;
            cur = crate::alloc_uninit_vec::<F128>(4 * len);
            let eq_next = (2 < k).then(|| SplitEqGhash::new(&r_pt[3..k]));
            pending = fold_message(&src1, len, rho1, &mut cur, lambda, eq_next.as_ref());
            done = 2;
        } else {
            len = h / 2;
            cur = crate::alloc_uninit_vec::<F128>(4 * len);
            let eq_next = (1 < k).then(|| SplitEqGhash::new(&r_pt[2..k]));
            pending = fold_message(src, len, rho0, &mut cur, lambda, eq_next.as_ref());
            done = 1;
        }
        let mut nxt = crate::alloc_uninit_vec::<F128>(4 * (len / 2).max(1));
        for i in done..k {
            let rho = round(pending.expect("round i's message was produced already"), ch);
            let eq_next = (i + 1 < k).then(|| SplitEqGhash::new(&r_pt[i + 2..k]));
            let c = &cur[..4 * len];
            pending = fold_message(&|x| load(c, x), len / 2, rho, &mut nxt, lambda, eq_next.as_ref());
            len /= 2;
            swap(&mut cur, &mut nxt);
        }
        debug_assert_eq!(len, 1, "layer {k}: folds must reduce to one entry");
        let last = load(&cur, 0);
        (rounds, r_prime, last)
    }

    /// [`rec_holo_prove_batched_stream`]`(w, σ, unstored, true, ch)` on these kernels: the same proof bytes.
    pub fn rec_holo_prove_batched_fast<C: Challenger>(
        w: &[u64],
        sigma: &[u32],
        unstored: usize,
        ch: &mut C,
    ) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
        let n = sigma.len();
        assert!(n.is_power_of_two() && n >= 64 && w.len() * 64 == n);
        let mu = n.trailing_zeros() as usize;
        assert!(mu <= 32 && unstored >= 1 && unstored < mu, "need μ ≤ 32 and 1 ≤ unstored < μ");
        ch.observe_label(DOMAIN_BATCHED);
        let alpha = ch.sample_f128();
        let beta = ch.sample_f128();
        let leaves = Leaves { w, sigma, alpha: splat(alpha), beta: splat(beta), mu };

        let top = mu - unstored;
        let mut q: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
        q[top] = collect(1 << (top - 1), &|x| leaves.entry(top, x));
        for s in (1..top).rev() {
            q[s] = halve(&q[s + 1]);
        }
        let [a, b, c, d] = lanes(load(&q[1], 0));
        let (top_lhs, top_rhs) = (a * c, b * d);
        ch.observe_f128(top_lhs);
        ch.observe_f128(top_rhs);

        let mut r_pt: Vec<F128> = Vec::new();
        let mut layers = Vec::with_capacity(mu);
        let (mut claim_l, mut claim_r) = (F128::ZERO, F128::ZERO);
        for k in 0..mu {
            let lambda = ch.sample_f128();
            let s = k + 1;
            let (rounds, mut r_prime, last) = if s <= top {
                let qs = take(&mut q[s]);
                layer(&|x| load(&qs, x), k, &r_pt, lambda, false, ch)
            } else {
                layer(&|x| leaves.entry(s, x), k, &r_pt, lambda, true, ch)
            };
            let [vl0, vr0, vl1, vr1] = lanes(last);
            for v in [vl0, vl1, vr0, vr1] {
                ch.observe_f128(v);
            }
            layers.push(BatchedLayerProof { rounds, vl0, vl1, vr0, vr1 });
            let c_k = ch.sample_f128();
            let one_plus_c = F128::ONE + c_k;
            claim_l = one_plus_c * vl0 + c_k * vl1;
            claim_r = one_plus_c * vr0 + c_k * vr1;
            r_prime.push(c_k);
            r_pt = r_prime;
        }
        let rho = r_pt;
        let s_sigma_eval = sigma_eval(sigma, &rho);
        let f_eval = claim_l + alpha * s_id_eval(&s_id_basis(mu), &rho) + beta;
        let g_eval = claim_r + alpha * s_sigma_eval + beta;
        observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
        let proof =
            ProductGkrBatchedProof { top_lhs, top_rhs, layers, f_eval, g_eval, s_sigma_eval, grinding_nonces: Vec::new() };
        let claim = ProductGkrBatchedClaim { rho, f_eval, g_eval, s_sigma_eval };
        (proof, claim)
    }
}
