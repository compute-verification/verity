//! The session verifier's check of one non-ZK circuit proof (`flock-circuit serve`'s `CircuitVerifier`): upstream's
//! `verifier::verify_ligerito_extra`, call for call and check for check, timed by phase, with the lincheck's last vector
//! computed from the block's structure (`FC_LINCHECK`).
//!
//! The timers are wall-clock seconds on the verifier's thread. Nothing they read enters the transcript or a proof, and the
//! opening's (`pcs::ligerito::verify_phase_ns`) come from `flock-vtime-b684b12.patch`, which only adds clocks.
//!
//! The lincheck's final check pairs `comb_partial`, the α-batched column marginal `comb` bound at the rounds' challenges, with
//! the prover's `z_partial`. Upstream (`flat`) builds the 2^k_log quirky eq table, folds `comb` over it and binds it round by
//! round. No challenge depends on `comb`, so `partial` replays the same rounds and computes `comb_partial` once, after them,
//! from [`BlockCircuit`]'s structure (`comb_partial_structured`). Over GF(2^128) the two are equal, so the transcript, the
//! claims and the accept/reject are the same; `both` computes the two and rejects when they differ.
use std::{
    env::var,
    sync::{OnceLock, atomic::Ordering},
    time::Instant,
};

use flock_core::{
    all_core_pool,
    challenger::Challenger,
    lincheck::{
        LC_PCORES_ONLY, LincheckCircuit, LincheckClaim, LincheckError, LincheckGrinding, LincheckProof, QuirkyPoint, SkipPoint,
        build_eq_table, build_quirky_eq_table_from_weights, verify_with_grinding as upstream_lincheck,
    },
    pcs::{
        Commitment, LOG_PACKING, PcsError, PcsParams,
        ligerito::{VERIFY_LEVEL_OPENS, VERIFY_RESIDUAL, VERIFY_RING_SWITCH, verify_phase_ns},
        verify_opening_batch_ligerito_mixed_with_grinding,
    },
    proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement},
    r1cs::BlockR1cs,
    verifier::{ExtraClaim, FlockVerifyError},
    zerocheck::{ZerocheckGrinding, ZerocheckProof, verify_with_grinding as upstream_zerocheck},
};
use flock_field::F128;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde_json::{Value, json};

use crate::circuit::BlockCircuit;

/// How the lincheck's `comb_partial` is computed (`FC_LINCHECK`, default `partial`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lincheck {
    /// upstream's: the 2^k_log eq table, the fold, a bind per round
    Flat,
    /// from the block's structure, after the rounds (flat where the structure is not one it covers)
    Partial,
    /// both, rejecting when they differ
    Both,
}

impl Lincheck {
    pub fn from_env() -> Result<Self, String> {
        match var("FC_LINCHECK").as_deref() {
            Err(_) | Ok("partial") => Ok(Lincheck::Partial),
            Ok("flat") => Ok(Lincheck::Flat),
            Ok("both") => Ok(Lincheck::Both),
            Ok(x) => Err(format!("FC_LINCHECK={x}: flat, partial or both")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Lincheck::Flat => "flat",
            Lincheck::Partial => "partial",
            Lincheck::Both => "both",
        }
    }
}

/// One proof's verification, by phase (seconds).
#[derive(Clone, Debug, Default)]
pub struct Phases {
    /// the proof's bincode decode
    pub decode_s: f64,
    /// the link points' region claims (`Stmt::extra_claims`)
    pub regions_s: f64,
    /// upstream's check that C is the identity (a pass over its 2^k_log rows)
    pub c0_check_s: f64,
    pub bind_s: f64,
    pub zerocheck_s: f64,
    pub lincheck_s: f64,
    /// the lincheck's `comb_partial`, by part, and the way it was computed (`flat`, `partial`, `both`, or `flat` where
    /// `partial` does not cover the structure)
    pub comb: Vec<(&'static str, f64)>,
    pub comb_mode: &'static str,
    /// the claims' points and weights gathered, the opening's parameters checked and its verifier config built
    pub opening_setup_s: f64,
    pub opening_s: f64,
    /// of the opening: the claims' ring-switch checks, the level opens (opened rows' leaves and Merkle paths), the residual
    pub ring_switch_s: f64,
    pub merkle_s: f64,
    pub residual_s: f64,
    pub claims: usize,
    pub total_s: f64,
}

impl Phases {
    pub fn json(&self) -> Value {
        let other = self.total_s - self.decode_s - self.regions_s - self.c0_check_s - self.bind_s - self.zerocheck_s - self.lincheck_s
            - self.opening_setup_s - self.opening_s;
        json!({
            "decode_s": self.decode_s, "regions_s": self.regions_s, "c0_check_s": self.c0_check_s, "bind_s": self.bind_s, "zerocheck_s": self.zerocheck_s,
            "lincheck_s": self.lincheck_s, "lincheck_comb": self.comb.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "lincheck_mode": self.comb_mode, "opening_setup_s": self.opening_setup_s, "opening_s": self.opening_s, "opening_ring_switch_s": self.ring_switch_s,
            "opening_merkle_s": self.merkle_s, "opening_residual_s": self.residual_s,
            "opening_rest_s": self.opening_s - self.ring_switch_s - self.merkle_s - self.residual_s, "opening_claims": self.claims,
            "other_s": other, "total_s": self.total_s,
        })
    }
}

/// Upstream's verifier pool, restated (it is private there): one thread unless `FLOCK_VERIFY_THREADS`, a 64 MiB stack for
/// the recursive Ligerito verifier.
fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = var("FLOCK_VERIFY_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(1);
        ThreadPoolBuilder::new().num_threads(n).stack_size(64 * 1024 * 1024).thread_name(|_| "flock-verify".into()).build()
            .expect("the session verifier's pool")
    })
}

/// Upstream's `lincheck_all_cores_enabled`: the lincheck's fold hops to the all-core pool unless disabled.
fn all_cores() -> bool {
    !LC_PCORES_ONLY.load(Ordering::Relaxed) && var("LINCHECK_PCORES_ONLY").is_err()
        && all_core_pool().current_num_threads() > rayon::current_num_threads()
}

/// Upstream's `fold_alpha_batched_pooled` placement (its `SUMCHECK_PAR_THRESHOLD` is 2^12 columns), for any work of the fold's.
fn on_fold_pool<T: Send>(n_cols: usize, f: impl FnOnce() -> T + Send) -> T {
    if n_cols >= 1 << 12 && all_cores() { all_core_pool().install(f) } else { f() }
}

/// [`flock_core::verifier::verify_ligerito_extra`] with `circuit`'s lincheck computed as `mode` says, timed into `ph`.
/// `c0_identity` keeps `r1cs.c0_is_identity()` once computed, for the statement's later verifications.
#[allow(clippy::too_many_arguments)]
pub fn verify_extra<Ch: Challenger>(r1cs: &BlockR1cs, c0_identity: &OnceLock<bool>, commitment: &Commitment, proof: &R1csProofLigerito,
                                    circuit: &BlockCircuit, pcs: &PcsParams, extra: &[ExtraClaim], mode: Lincheck, ch: &mut Ch,
                                    ph: &mut Phases)
    -> Result<R1csClaim, FlockVerifyError> {
    let (ab, c) = pool().install(|| core(r1cs, c0_identity, &proof.zerocheck, &proof.lincheck, commitment, circuit, pcs.zerocheck_grinding(),
                                         pcs.lincheck_grinding(), mode, ch, ph))?;
    let t_setup = Instant::now();
    let full = |z: &ZClaim| [z.point.x_inner_rest.as_slice(), z.point.x_outer.as_slice()].concat();
    let mut values = vec![ab.value, c.value];
    let mut skips = vec![ab.point.z_skip.weights(LOG_PACKING - 1), c.point.z_skip.weights(LOG_PACKING - 1)];
    let mut xs = vec![full(&ab), full(&c)];
    for e in extra {
        if e.skip_weights.len() != 1 << (LOG_PACKING - 1) || e.x_full.len() != xs[1].len() {
            return Err(FlockVerifyError::PcsAb(PcsError::Ligerito));
        }
        values.push(e.value);
        skips.push(e.skip_weights.clone());
        xs.push(e.x_full.clone());
    }
    ph.claims = values.len();
    pool().install(|| {
        if &commitment.params != pcs {
            return Err(PcsError::Ligerito);
        }
        let skip_refs: Vec<&[F128]> = skips.iter().map(|v| v.as_slice()).collect();
        let x_refs: Vec<&[F128]> = xs.iter().map(|v| v.as_slice()).collect();
        let cfg = pcs.ligerito_verifier_config().expect("Ligerito default verifier config");
        ph.opening_setup_s = t_setup.elapsed().as_secs_f64();
        let (t, ns0) = (Instant::now(), verify_phase_ns());
        let r = verify_opening_batch_ligerito_mixed_with_grinding(commitment, &values, &skip_refs, &x_refs, &[], &proof.pcs_open, &cfg,
                                                                  pcs.opening_grinding(), ch);
        let ns = verify_phase_ns();
        let s = |i: usize| (ns[i] - ns0[i]) as f64 * 1e-9;
        (ph.opening_s, ph.ring_switch_s, ph.merkle_s, ph.residual_s) =
            (t.elapsed().as_secs_f64(), s(VERIFY_RING_SWITCH), s(VERIFY_LEVEL_OPENS), s(VERIFY_RESIDUAL));
        r
    })
    .map_err(FlockVerifyError::PcsAb)?;
    Ok(R1csClaim { ab, c })
}

/// Upstream's `verify_core_inner`.
#[allow(clippy::too_many_arguments)]
fn core<Ch: Challenger>(r1cs: &BlockR1cs, c0_identity: &OnceLock<bool>, zc_proof: &ZerocheckProof, lc_proof: &LincheckProof,
                        commitment: &Commitment, circuit: &BlockCircuit, zg: ZerocheckGrinding, lg: LincheckGrinding, mode: Lincheck,
                        ch: &mut Ch, ph: &mut Phases)
    -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    let t = Instant::now();
    let c0_identity = *c0_identity.get_or_init(|| r1cs.c0_is_identity());
    ph.c0_check_s = t.elapsed().as_secs_f64();
    if !c0_identity {
        return Err(FlockVerifyError::NonIdentityC);
    }
    let t = Instant::now();
    bind_statement(ch, r1cs, commitment);
    ph.bind_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let zc = upstream_zerocheck(r1cs.m, zc_proof, zg, ch).map_err(FlockVerifyError::Zerocheck)?;
    ph.zerocheck_s = t.elapsed().as_secs_f64();
    let x_ab = r1cs.x_ab_from_multilinear_values(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
    let t = Instant::now();
    let lc = lincheck(r1cs.m, r1cs.k_log, r1cs.k_skip, circuit, &x_ab, zc.a_eval, zc.b_eval, lc_proof, lg, mode, ch, ph)
        .map_err(FlockVerifyError::Lincheck)?;
    ph.lincheck_s = t.elapsed().as_secs_f64();
    let ab = ZClaim { point: r1cs.ab_claim_point(lc.r_inner_skip, &lc.r_inner_rest, &x_ab.x_outer), value: lc.w };
    let c = ZClaim { point: r1cs.c_claim_point(SkipPoint::Phi8(zc.z), &zc.r_rest), value: zc.c_eval };
    Ok((ab, c))
}

/// Upstream's `lincheck::verify_with_grinding`, check for check and challenge for challenge, with `comb_partial` computed as
/// `mode` says. An AG skip point (never the circuit statement's) goes to upstream's whole.
#[allow(clippy::too_many_arguments)]
pub fn lincheck<Ch: Challenger>(m: usize, k_log: usize, k_skip: usize, c: &BlockCircuit, x_ab: &QuirkyPoint, v_a: F128, v_b: F128,
                                proof: &LincheckProof, grinding: LincheckGrinding, mode: Lincheck, ch: &mut Ch, ph: &mut Phases)
    -> Result<LincheckClaim, LincheckError> {
    if matches!(x_ab.z_skip, SkipPoint::Ag(_)) {
        ph.comb_mode = "flat";
        return upstream_lincheck(m, k_log, k_skip, c, x_ab, v_a, v_b, proof, grinding, ch);
    }
    let k = 1usize << k_log;
    let n_log = m - k_log;
    if k_skip > k_log {
        return Err(LincheckError::KSkipExceedsKLog { k_skip, k_log });
    }
    let inner_rest_len = k_log - k_skip;
    let n_skip = 1usize << k_skip;
    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(LincheckError::BadInnerRestLength { which: "x_ab", expected: inner_rest_len, got: x_ab.x_inner_rest.len() });
    }
    if x_ab.x_outer.len() != n_log {
        return Err(LincheckError::BadOuterLength { which: "x_ab", expected: n_log, got: x_ab.x_outer.len() });
    }
    if c.n_cols() != k {
        return Err(LincheckError::BadMatrixShape { which: "circuit", expected: k, got_rows: k, got_cols: c.n_cols() });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(LincheckError::BadVectorLength { which: "rounds", expected: inner_rest_len, got: proof.rounds.len() });
    }
    if proof.z_partial.len() != n_skip {
        return Err(LincheckError::BadVectorLength { which: "z_partial", expected: n_skip, got: proof.z_partial.len() });
    }
    let expected_nonces = grinding.nonce_count(inner_rest_len, usize::from(c.const_pin_col().is_some()), k_skip);
    if proof.grinding_nonces.len() != expected_nonces {
        return Err(LincheckError::BadGrindingNonceCount { expected: expected_nonces, got: proof.grinding_nonces.len() });
    }
    ch.observe_label(b"flock-lincheck-v0");
    let mut nonce_idx = 0;
    let mut sample = |ch: &mut Ch, bits: Option<u32>, which: &'static str| match bits {
        Some(b) => {
            let r = ch.verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], b).ok_or(LincheckError::InvalidGrindingNonce { which });
            nonce_idx += 1;
            r
        }
        None => Ok(ch.sample_f128()),
    };
    let alpha = sample(ch, grinding.alpha_bits(), "alpha")?;
    let ws = x_ab.z_skip.weights(k_skip);
    let xr = &x_ab.x_inner_rest;
    // where upstream's fold would panic (or index past the block), the flat fold runs where upstream's does and panics there
    let structured = structured_covers(c, k_skip) && !first_slot_vanishes(c, &ws, xr, k_skip);
    let flat = mode != Lincheck::Partial || !structured;
    ph.comb_mode = if flat && mode == Lincheck::Partial { "flat (partial does not cover this structure)" } else { mode.name() };
    let mut comb = Vec::new();
    let mut binds_s = 0.0;
    if flat {
        let t = Instant::now();
        let eq = build_quirky_eq_table_from_weights(&ws, xr);
        ph.comb.push(("flat_eq_table_s", t.elapsed().as_secs_f64()));
        let t = Instant::now();
        comb = on_fold_pool(c.n_cols(), || c.fold_alpha_batched(alpha, &eq));
        ph.comb.push(("flat_fold_s", t.elapsed().as_secs_f64()));
    }
    let mut target = alpha * v_a + v_b;
    let mut beta = F128::ZERO;
    if let Some(col) = c.const_pin_col() {
        beta = sample(ch, grinding.beta_bits(), "beta")?;
        if flat {
            comb[col] += beta;
        }
        target += beta;
    }
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let r = sample(ch, grinding.multilinear_round_bits(), "sumcheck-round")?;
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        if flat {
            let t = Instant::now();
            bind_top(&mut comb, r);
            binds_s += t.elapsed().as_secs_f64();
        }
        r_rounds.push(r);
    }
    if flat {
        ph.comb.push(("flat_binds_s", binds_s));
    }
    ch.observe_f128_slice(&proof.z_partial);
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();
    if mode != Lincheck::Flat && structured {
        let t = Instant::now();
        let cp = on_fold_pool(c.n_cols(), || comb_partial_structured(c, alpha, beta, &ws, xr, &r_inner_rest, &mut ph.comb));
        ph.comb.push(("partial_s", t.elapsed().as_secs_f64()));
        if mode == Lincheck::Both && cp != comb {
            return Err(LincheckError::ConsistencyFailed { which: "FC_LINCHECK=both: the structured comb_partial differs from the flat" });
        }
        comb = cp;
    }
    let final_sum = inner_product(&comb, &proof.z_partial);
    if running != final_sum {
        return Err(LincheckError::ConsistencyFailed { which: "sumcheck-final" });
    }
    let r_inner_skip = match grinding.skip_bits(k_skip) {
        Some(_) => SkipPoint::Phi8(sample(ch, grinding.skip_bits(k_skip), "inner-skip")?),
        None => x_ab.z_skip.sample_fresh(ch),
    };
    let w = inner_product(&r_inner_skip.weights(k_skip), &proof.z_partial);
    Ok(LincheckClaim { r_inner_skip, r_inner_rest, w })
}

/// Upstream's `sumcheck_bind_top_in_place_par`: `v[i] ← v[i] + r·(v[i+half] + v[i])`, then the top half dropped.
fn bind_top(v: &mut Vec<F128>, r: F128) {
    let half = v.len() / 2;
    if half < 1 << 12 {
        for i in 0..half {
            v[i] = v[i] + r * (v[i + half] + v[i]);
        }
    } else {
        let (lo, hi) = v.split_at_mut(half);
        lo.par_iter_mut().zip(hi[..half].par_iter()).for_each(|(l, &h)| *l = *l + r * (h + *l));
    }
    v.truncate(half);
}

fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(F128::ZERO, |acc, (x, y)| acc + *x * *y)
}

/// Whether `comb_partial_structured` covers `c`: every slot type at least 2^k_skip columns and within the block, the ranges
/// disjoint and within it, Δ and the pin within it. Upstream's fold panics on the last three; this form then folds flat.
fn structured_covers(c: &BlockCircuit, k_skip: usize) -> bool {
    let n = 1usize << c.k_log;
    if c.types.is_empty() || c.tables.len() != c.types.len() || c.types.iter().any(|t| t.0 < k_skip || t.0 > c.k_log) || c.pin >= n {
        return false;
    }
    let mut spans = Vec::with_capacity(c.ranges.len());
    for &(t, pos0, count) in &c.ranges {
        let Some(&(sl, _)) = c.types.get(t) else { return false };
        match pos0.checked_add(count).and_then(|e| e.checked_mul(1 << sl)) {
            Some(end) if end <= n => spans.push((pos0 << sl, end)),
            _ => return false,
        }
    }
    spans.sort_unstable();
    spans.windows(2).all(|w| w[0].1 <= w[1].0)
        && c.da.iter().chain(&c.db).all(|&(i, col)| (i as usize) < n && (col as usize) < n)
}

/// Whether the quirky eq table vanishes on the smallest slot type's first slot, where upstream's fold panics ("eq tensor
/// vanishes on the first slot"): every skip weight zero, or a coordinate above that slot equal to 1 (`1 + x = 0`).
fn first_slot_vanishes(c: &BlockCircuit, ws: &[F128], xr: &[F128], k_skip: usize) -> bool {
    let lo = c.types.iter().map(|t| t.0).min().unwrap_or(k_skip);
    ws.iter().all(|w| *w == F128::ZERO) || xr[lo - k_skip..].iter().any(|x| *x == F128::ONE)
}

/// `eq(point, i)` for every `i < 2^len`, from two half tables.
struct SplitEq {
    lo_bits: usize,
    lo: Vec<F128>,
    hi: Vec<F128>,
}

impl SplitEq {
    fn new(point: &[F128]) -> Self {
        let lo_bits = point.len() / 2;
        SplitEq { lo_bits, lo: build_eq_table(&point[..lo_bits]), hi: build_eq_table(&point[lo_bits..]) }
    }

    fn at(&self, i: usize) -> F128 {
        self.lo[i & ((1 << self.lo_bits) - 1)] * self.hi[i >> self.lo_bits]
    }
}

/// `Σ_{q ∈ [a, a + count)} eq(x, q) · eq(r, q)`, by aligned dyadic blocks: a block `[b 2^d, (b + 1) 2^d)` sums to
/// `Π_{j<d} (g_j(0) + g_j(1)) · Π_{j≥d} g_j(bit_j(b 2^d))`, with `g_j(0) = (1 + x_j)(1 + r_j)` and `g_j(1) = x_j r_j`.
fn interval_eq_sum(x: &[F128], r: &[F128], mut a: usize, count: usize) -> F128 {
    let n = x.len();
    let g0: Vec<F128> = (0..n).map(|j| (F128::ONE + x[j]) * (F128::ONE + r[j])).collect();
    let g1: Vec<F128> = (0..n).map(|j| x[j] * r[j]).collect();
    let mut whole = vec![F128::ONE; n + 1];
    for j in 0..n {
        whole[j + 1] = whole[j] * (g0[j] + g1[j]);
    }
    let end = a + count;
    let mut s = F128::ZERO;
    while a < end {
        let mut d = if a == 0 { n } else { (a.trailing_zeros() as usize).min(n) };
        while a + (1 << d) > end {
            d -= 1;
        }
        s += (d..n).fold(whole[d], |acc, j| acc * if (a >> j) & 1 == 1 { g1[j] } else { g0[j] });
        a += 1 << d;
    }
    s
}

/// `comb_partial[i] = Σ_{col mod 2^k_skip = i} eq(rr, col >> k_skip) · comb[col]`, where `comb` is the α-batched column marginal
/// of `c` at the quirky eq table `(ws, xr)` plus β at the pin, from `c`'s structure: never a 2^k_log vector.
///
/// A slot of type `t` (log size `sl`) at position `q` holds `comb[(q << sl) | y] = base_t[y] · eq(xr[sl−6..], q)`, with `base_t`
/// the type's fold (and its table side) at its own eq table `(ws, xr[..sl−6])`; upstream's fold computes the same product
/// as `ratio_q · base` at the first slot's table. Its share of `comb_partial[y mod 64]` is `base_t[y] · eq(rr[..sl−6], y >> 6)`
/// times `eq(xr[sl−6..], q) · eq(rr[sl−6..], q)`, so a type costs one fold at 2^sl and its ranges a sum over positions
/// ([`interval_eq_sum`]). Δ's entry `(i, col)` adds `α·eq_row(i)·eq(rr, col >> 6)` (B's without α), the pin β·eq(rr, pin >> 6).
fn comb_partial_structured(c: &BlockCircuit, alpha: F128, beta: F128, ws: &[F128], xr: &[F128], rr: &[F128],
                           times: &mut Vec<(&'static str, f64)>) -> Vec<F128> {
    let n_skip = ws.len();
    let k_skip = n_skip.trailing_zeros() as usize;
    let mask = n_skip - 1;
    let add = |a: Vec<F128>, b: Vec<F128>| a.into_iter().zip(b).map(|(x, y)| x + y).collect::<Vec<F128>>();
    let t = Instant::now();
    let mut slot = vec![F128::ZERO; c.types.len()];
    for &(ty, pos0, count) in &c.ranges {
        let h = c.types[ty].0 - k_skip;
        slot[ty] += interval_eq_sum(&xr[h..], &rr[h..], pos0, count);
    }
    let mut cp = vec![F128::ZERO; n_skip];
    for (((sl, m), table), s) in c.types.iter().zip(&c.tables).zip(&slot) {
        let h = sl - k_skip;
        let eq_t = build_quirky_eq_table_from_weights(ws, &xr[..h]);
        let mut base = m.fold_alpha_batched(alpha, &eq_t);
        if let Some(tb) = table {
            tb.fold(&eq_t, &mut base);
        }
        let er = build_eq_table(&rr[..h]);
        let chunk = n_skip.max(1 << 12);
        let p = base.par_chunks(chunk).enumerate().map(|(b, xs)| {
            let mut acc = vec![F128::ZERO; n_skip];
            for (j, v) in xs.iter().enumerate() {
                let y = b * chunk + j;
                acc[y & mask] += *v * er[y >> k_skip];
            }
            acc
        }).reduce(|| vec![F128::ZERO; n_skip], add);
        for (x, y) in cp.iter_mut().zip(p) {
            *x += y * *s;
        }
    }
    times.push(("partial_types_s", t.elapsed().as_secs_f64()));
    let t = Instant::now();
    let (row, col) = (SplitEq::new(xr), SplitEq::new(rr));
    let delta = |d: &[(u32, u32)]| d.par_chunks(1 << 14).map(|es| {
        let mut acc = vec![F128::ZERO; n_skip];
        for &(i, cc) in es {
            let (i, cc) = (i as usize, cc as usize);
            acc[cc & mask] += ws[i & mask] * row.at(i >> k_skip) * col.at(cc >> k_skip);
        }
        acc
    }).reduce(|| vec![F128::ZERO; n_skip], add);
    let (da, db) = (delta(&c.da), delta(&c.db));
    for ((x, a), b) in cp.iter_mut().zip(da).zip(db) {
        *x += alpha * a + b;
    }
    cp[c.pin & mask] += beta * col.at(c.pin >> k_skip);
    times.push(("partial_delta_s", t.elapsed().as_secs_f64()));
    cp
}
