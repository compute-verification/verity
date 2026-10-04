
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
