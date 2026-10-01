//! The session verifier's check of one non-ZK circuit proof (`flock-circuit serve`'s `CircuitVerifier`): upstream's
//! `verifier::verify_ligerito_extra`, call for call and check for check, timed by phase.
//!
//! The timers are wall-clock seconds on the verifier's thread. Nothing they read enters the transcript or a proof, and the
//! opening's (`pcs::ligerito::verify_phase_ns`) come from `flock-vtime-b684b12.patch`, which only adds clocks.
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
        build_quirky_eq_table_from_weights, verify_with_grinding as upstream_lincheck,
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
    /// the lincheck's `comb_partial`, by part
    pub comb: Vec<(&'static str, f64)>,
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
            "opening_setup_s": self.opening_setup_s, "opening_s": self.opening_s, "opening_ring_switch_s": self.ring_switch_s,
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

/// [`flock_core::verifier::verify_ligerito_extra`], timed into `ph`.
#[allow(clippy::too_many_arguments)]
pub fn verify_extra<Ch: Challenger>(r1cs: &BlockR1cs, commitment: &Commitment, proof: &R1csProofLigerito, circuit: &BlockCircuit,
                                    pcs: &PcsParams, extra: &[ExtraClaim], ch: &mut Ch, ph: &mut Phases)
    -> Result<R1csClaim, FlockVerifyError> {
    let (ab, c) = pool().install(|| core(r1cs, &proof.zerocheck, &proof.lincheck, commitment, circuit, pcs.zerocheck_grinding(),
                                         pcs.lincheck_grinding(), ch, ph))?;
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
fn core<Ch: Challenger>(r1cs: &BlockR1cs, zc_proof: &ZerocheckProof, lc_proof: &LincheckProof, commitment: &Commitment, circuit: &BlockCircuit,
                        zg: ZerocheckGrinding, lg: LincheckGrinding, ch: &mut Ch, ph: &mut Phases)
    -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    let t = Instant::now();
    let c0_identity = r1cs.c0_is_identity();
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
    let lc = lincheck(r1cs.m, r1cs.k_log, r1cs.k_skip, circuit, &x_ab, zc.a_eval, zc.b_eval, lc_proof, lg, ch, ph)
        .map_err(FlockVerifyError::Lincheck)?;
    ph.lincheck_s = t.elapsed().as_secs_f64();
    let ab = ZClaim { point: r1cs.ab_claim_point(lc.r_inner_skip, &lc.r_inner_rest, &x_ab.x_outer), value: lc.w };
    let c = ZClaim { point: r1cs.c_claim_point(SkipPoint::Phi8(zc.z), &zc.r_rest), value: zc.c_eval };
    Ok((ab, c))
}

/// Upstream's `lincheck::verify_with_grinding`, check for check and challenge for challenge, its `comb_partial`'s parts timed.
/// An AG skip point (never the circuit statement's) goes to upstream's whole.
#[allow(clippy::too_many_arguments)]
pub fn lincheck<Ch: Challenger>(m: usize, k_log: usize, k_skip: usize, c: &BlockCircuit, x_ab: &QuirkyPoint, v_a: F128, v_b: F128,
                                proof: &LincheckProof, grinding: LincheckGrinding, ch: &mut Ch, ph: &mut Phases)
    -> Result<LincheckClaim, LincheckError> {
    if matches!(x_ab.z_skip, SkipPoint::Ag(_)) {
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
    let t = Instant::now();
    let eq = build_quirky_eq_table_from_weights(&ws, xr);
    ph.comb.push(("flat_eq_table_s", t.elapsed().as_secs_f64()));
    let t = Instant::now();
    let mut comb = on_fold_pool(c.n_cols(), || c.fold_alpha_batched(alpha, &eq));
    ph.comb.push(("flat_fold_s", t.elapsed().as_secs_f64()));
    let mut target = alpha * v_a + v_b;
    if let Some(col) = c.const_pin_col() {
        let beta = sample(ch, grinding.beta_bits(), "beta")?;
        comb[col] += beta;
        target += beta;
    }
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    let mut binds_s = 0.0;
    for &(e1, einf) in &proof.rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let r = sample(ch, grinding.multilinear_round_bits(), "sumcheck-round")?;
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        let t = Instant::now();
        bind_top(&mut comb, r);
        binds_s += t.elapsed().as_secs_f64();
        r_rounds.push(r);
    }
    ph.comb.push(("flat_binds_s", binds_s));
    ch.observe_f128_slice(&proof.z_partial);
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();
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
