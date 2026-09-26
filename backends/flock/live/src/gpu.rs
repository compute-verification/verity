//! Flock-CUDA for the chunk statement (feature `gpu`): `flock_cuda_prove_chunk` (backends/flock/cuda/prove_chunk.cuh,
//! installed by backends/flock/cuda_chunk_patch.py). Every squeeze goes to the rep's `LiveChallenger` through the
//! live hook (flock-live R6), and the link exchange runs in `link_cb` after the commitment, before any coin.

use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::{null, null_mut},
    slice::from_raw_parts,
    sync::OnceLock,
};

use flock_field::F128;
use flock_prover::{
    field::{F8, F256},
    lincheck::LincheckProof,
    ntt::AdditiveNttGf8,
    pcs::{
        BatchOpeningProofLigerito, Commitment, PcsParams,
        ligerito::{FinalProof, LigeritoProof, RecursiveProof, SumcheckMessage256},
        ring_switch::RingSwitchProof,
    },
    proof::R1csProofLigerito,
    r1cs::SparseBinaryMatrix,
    r1cs_hashes::blake3::Compression,
    zerocheck::{K_SKIP, ZerocheckProof},
};

use crate::{LiveChallenger, chunk::*};

// The static CUDA library travels with this crate; naming it keeps it in the link.
extern crate flock_cuda_ffi;

pub fn device_count() -> i32 {
    flock_cuda_ffi::gpu::device_count()
}

type LiveCb = unsafe extern "C" fn(*mut c_void, *const u8, usize, u32, *mut u64) -> i32;
type LinkCb = unsafe extern "C" fn(*mut c_void, *const u8, usize, i32, *mut u64, i32) -> i32;

#[repr(C)]
struct ProveParams {
    m: i32,
    statement_digest: *const u8,
    domain: *const u8,
    domain_len: u32,
    a_col_ptr: *const u32,
    a_rows: *const u32,
    a_nnz: u32,
    b_col_ptr: *const u32,
    b_rows: *const u32,
    b_nnz: u32,
    const_pin_col: i32,
    useful_bits: i32,
    k_log: i32,
    zc_mcol: *const u8,
    zc_f8mul: *const u8,
    initial_k: i32,
    num_levels: i32,
    log_inv_rates: *const i32,
    recursive_ks: *const i32,
    queries: *const i32,
    grinding_bits: *const i32,
    claim_batch_grinding_bits: *const i32,
    consistency_batch_grinding_bits: *const i32,
    ood_samples: *const i32,
    recursive_steps: i32,
    zc_initial_bits: i32,
    zc_skip_bits: i32,
    zc_round_bits: i32,
    lc_alpha_bits: i32,
    lc_beta_bits: i32,
    lc_round_bits: i32,
    lc_skip_bits: i32,
    rs_bits: i32,
    gamma_bits: i32,
    dump_z_path: *const i8,
    live_cb: Option<LiveCb>,
    live_ctx: *mut c_void,
}

#[repr(C)]
struct ChunkParams {
    n_comp_log: i32,
    cv: *const u32,
    msg: *const u32,
    ctr: *const u64,
    blen: *const u32,
    flags: *const u32,
    delta_a: *const u32,
    n_delta_a: i32,
    delta_b: *const u32,
    n_delta_b: i32,
    sub_log: i32,
    link_cb: Option<LinkCb>,
    link_ctx: *mut c_void,
    gamma_bits: i32,
    phase_s: *mut f64,
    mode: i32,
    comp_slots: i32,
    units: i32,
    unit_log: i32,
    unit_pos0: i32,
    u_a_col_ptr: *const u32,
    u_a_rows: *const u32,
    u_a_nnz: u32,
    u_b_col_ptr: *const u32,
    u_b_rows: *const u32,
    u_b_nnz: u32,
    u_a_off: *const u32,
    u_a_col: *const u32,
    u_b_off: *const u32,
    u_b_col: *const u32,
    u_useful: i32,
    u_const: i32,
    u_n_in: i32,
    u_inputs: *const u64,
    rows_x: *const u8,
    rows_w: *const u8,
    accs: *const u32,
    vus: i32,
    row_bytes: i32,
    units_per_vu: i32,
    keys: *const u32,
    blocks_real: i32,
    unit_cache: i32,
    pub_cv: *mut u32,
    pub_y16: *mut u32,
    pub_acc: *mut u32,
    u_out_acc: i32,
    u_out_y: i32,
    per_role: i32,
    chunks_per_block: i32,
    chunk_nb0: i32,
    chunk_nb1: i32,
    blocks_per_vu: i32,
    dummy_keyed: i32,
    host_z: *const u64,
    host_a: *const u64,
    host_b: *const u64,
    sha256: i32,
    sha_mid: *const u32,
    sha_pad: *const u32,
    last_blen: i32,
    last_root: i32,
    scale_slot0: i32,
    last_data_words: i32,
}

unsafe extern "C" {
    fn flock_cuda_prove_chunk(p: *const ProveParams, c: *const ChunkParams, out: *mut *mut u8, out_len: *mut usize) -> i32;
    fn flock_cuda_free(p: *mut u8);
}

unsafe extern "C" fn live_cb(ctx: *mut c_void, content: *const u8, len: usize, n: u32, out: *mut u64) -> i32 {
    let ch = unsafe { &mut *(ctx as *mut LiveChallenger) };
    let c = unsafe { from_raw_parts(content, len) }.to_vec();
    match catch_unwind(AssertUnwindSafe(|| ch.round(c, n as usize))) {
        Ok(coins) => {
            for (i, x) in coins.iter().enumerate() {
                unsafe {
                    *out.add(2 * i) = x.lo;
                    *out.add(2 * i + 1) = x.hi;
                }
            }
            0
        }
        Err(_) => -1,
    }
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

unsafe extern "C" fn link_cb(ctx: *mut c_void, cap: *const u8, cap_len: usize, m: i32, x_out: *mut u64, max: i32) -> i32 {
    let hook = unsafe { &mut *(ctx as *mut Hook<'_>) };
    let cap = unsafe { from_raw_parts(cap, cap_len) };
    match catch_unwind(AssertUnwindSafe(|| hook(cap))) {
        Ok(pts) if pts.len() <= max as usize => {
            let w = m as usize - 6;
            for (e, p) in pts.iter().enumerate() {
                for (i, x) in p[6..].iter().enumerate() {
                    unsafe {
                        *x_out.add(2 * (e * w + i)) = x.lo;
                        *x_out.add(2 * (e * w + i) + 1) = x.hi;
                    }
                }
            }
            pts.len() as i32
        }
        _ => -1,
    }
}

fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in m.rows.iter() {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

struct Static {
    a: (Vec<u32>, Vec<u32>),
    b: (Vec<u32>, Vec<u32>),
    mcol: Vec<u8>,
    f8mul: Vec<u8>,
    da: Vec<u32>,
    db: Vec<u32>,
}

fn statics(st: &ChunkStmt) -> &'static Static {
    statics_with(&st.circuit.da, &st.circuit.db)
}

/// One statement per process: the first caller's Δ is the process's.
fn statics_with(da: &[(u32, u32)], db: &[(u32, u32)]) -> &'static Static {
    statics_for(comp(), da, db)
}

/// [`statics_with`] with another compression circuit (the CSC the lincheck folds per slot).
fn statics_for(c: &flock_prover::r1cs::BlockR1cs, da: &[(u32, u32)], db: &[(u32, u32)]) -> &'static Static {
    static S: OnceLock<Static> = OnceLock::new();
    S.get_or_init(|| {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let mut mcol = vec![0u8; 64 * 64];
        for s in 0..64 {
            let mut col = vec![F8::ZERO; 64];
            col[s] = F8(1);
            ntt_s.inverse(&mut col);
            ntt_l.forward(&mut col);
            for i in 0..64 {
                mcol[s * 64 + i] = col[i].0;
            }
        }
        let mut f8mul = vec![0u8; 256 * 256];
        for x in 0..256usize {
            for y in 0..256usize {
                f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
            }
        }
        let flat = |d: &[(u32, u32)]| d.iter().flat_map(|&(i, c)| [i, c]).collect();
        Static { a: csc_from_rows(&c.a_0), b: csc_from_rows(&c.b_0), mcol, f8mul, da: flat(da), db: flat(db) }
    })
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl Reader<'_> {
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        v
    }
    fn u64s(&mut self) -> Vec<u64> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.u64()).collect()
    }
    fn f128(&mut self) -> F128 {
        let lo = self.u64();
        let hi = self.u64();
        F128 { lo, hi }
    }
    fn f256(&mut self) -> F256 {
        let c0 = self.f128();
        let c1 = self.f128();
        F256::new(c0, c1)
    }
    fn f128s(&mut self) -> Vec<F128> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.f128()).collect()
    }
    fn hash(&mut self) -> [u8; 32] {
        let h: [u8; 32] = self.b[self.o..self.o + 32].try_into().unwrap();
        self.o += 32;
        h
    }
    fn hashes(&mut self) -> Vec<[u8; 32]> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.hash()).collect()
    }
    fn rows(&mut self) -> Vec<Vec<F128>> {
        let n_rows = self.u64() as usize;
        let row_len = self.u64() as usize;
        (0..n_rows).map(|_| (0..row_len).map(|_| self.f128()).collect()).collect()
    }
}

/// One rep on the GPU. Returns the proof, its commitment and the prover's phase split (witness, commit, link,
/// zerocheck, lincheck, ring switch, ligerito, total; seconds).
pub fn prove_chunk(
    st: &ChunkStmt,
    comps: &[Compression],
    pcs: &PcsParams,
    domain: &[u8],
    mut hook: Hook<'_>,
    ch: &mut LiveChallenger,
) -> (R1csProofLigerito, Commitment, [f64; 8]) {
    let s = statics(st);
    let m = st.m;
    let cfg = pcs.ligerito_prover_config().unwrap_or_else(|_| panic!("no ligerito config for m={m}"));
    let to_i32 = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    let log_inv_rates = to_i32(&cfg.log_inv_rates);
    let recursive_ks = to_i32(&cfg.recursive_ks);
    let queries = to_i32(&cfg.queries);
    let grinding_bits = to_i32(&cfg.grinding_bits);
    let claim_batch = to_i32(&cfg.claim_batch_grinding_bits);
    let consistency_batch = to_i32(&cfg.consistency_batch_grinding_bits);
    let ood_samples = to_i32(&cfg.ood_samples);
    let zc = pcs.zerocheck_grinding();
    let lc = pcs.lincheck_grinding();
    let og = pcs.opening_grinding();
    let opt = |b: Option<u32>| b.map_or(0, |x| x as i32);
    let n_claims = 2 + 2 + 2 * REGIONS.len();
    let params = ProveParams {
        m: m as i32,
        statement_digest: st.digest.as_ptr(),
        domain: domain.as_ptr(),
        domain_len: domain.len() as u32,
        a_col_ptr: s.a.0.as_ptr(),
        a_rows: s.a.1.as_ptr(),
        a_nnz: s.a.1.len() as u32,
        b_col_ptr: s.b.0.as_ptr(),
        b_rows: s.b.1.as_ptr(),
        b_nnz: s.b.1.len() as u32,
        const_pin_col: PIN as i32,
        useful_bits: 1 << K_LOG,
        k_log: K_LOG as i32,
        zc_mcol: s.mcol.as_ptr(),
        zc_f8mul: s.f8mul.as_ptr(),
        initial_k: cfg.initial_k as i32,
        num_levels: log_inv_rates.len() as i32,
        log_inv_rates: log_inv_rates.as_ptr(),
        recursive_ks: recursive_ks.as_ptr(),
        queries: queries.as_ptr(),
        grinding_bits: grinding_bits.as_ptr(),
        claim_batch_grinding_bits: claim_batch.as_ptr(),
        consistency_batch_grinding_bits: consistency_batch.as_ptr(),
        ood_samples: ood_samples.as_ptr(),
        recursive_steps: cfg.recursive_steps as i32,
        zc_initial_bits: opt(zc.initial_bits(m)),
        zc_skip_bits: opt(zc.skip_bits()),
        zc_round_bits: opt(zc.multilinear_round_bits()),
        lc_alpha_bits: opt(lc.alpha_bits()),
        lc_beta_bits: opt(lc.beta_bits()),
        lc_round_bits: opt(lc.multilinear_round_bits()),
        lc_skip_bits: opt(lc.skip_bits(K_SKIP)),
        rs_bits: og.ring_switch_bits as i32,
        gamma_bits: og.claim_batch_bits as i32,
        dump_z_path: null(),
        live_cb: Some(live_cb),
        live_ctx: ch as *mut LiveChallenger as *mut c_void,
    };
    let cv: Vec<u32> = comps.iter().flat_map(|c| c.0).collect();
    let msg: Vec<u32> = comps.iter().flat_map(|c| c.1).collect();
    let ctr: Vec<u64> = comps.iter().map(|c| c.2).collect();
    let blen: Vec<u32> = comps.iter().map(|c| c.3).collect();
    let flags: Vec<u32> = comps.iter().map(|c| c.4).collect();
    assert_eq!(comps.len(), 1 << (m - 14));
    let mut phase = [0f64; 8];
    let chunk = ChunkParams {
        n_comp_log: (m - 14) as i32,
        cv: cv.as_ptr(),
        msg: msg.as_ptr(),
        ctr: ctr.as_ptr(),
        blen: blen.as_ptr(),
        flags: flags.as_ptr(),
        delta_a: s.da.as_ptr(),
        n_delta_a: (s.da.len() / 2) as i32,
        delta_b: s.db.as_ptr(),
        n_delta_b: (s.db.len() / 2) as i32,
        sub_log: SUB_LOG as i32,
        link_cb: Some(link_cb),
        link_ctx: &mut hook as *mut Hook<'_> as *mut c_void,
        gamma_bits: if n_claims > 1 { og.claim_batch_bits as i32 } else { 0 },
        phase_s: phase.as_mut_ptr(),
        ..ChunkParams::none()
    };
    let (proof, commitment) = run(&params, &chunk, pcs, m);
    (proof, commitment, phase)
}

impl ChunkParams {
    fn none() -> Self {
        ChunkParams {
            n_comp_log: 0, cv: null(), msg: null(), ctr: null(), blen: null(), flags: null(), delta_a: null(), n_delta_a: 0,
            delta_b: null(), n_delta_b: 0, sub_log: 0, link_cb: None, link_ctx: null_mut(), gamma_bits: 0, phase_s: null_mut(),
            mode: 0, comp_slots: 0, units: 0, unit_log: 0, unit_pos0: 0, u_a_col_ptr: null(), u_a_rows: null(), u_a_nnz: 0,
            u_b_col_ptr: null(), u_b_rows: null(), u_b_nnz: 0, u_a_off: null(), u_a_col: null(), u_b_off: null(), u_b_col: null(),
            u_useful: 0, u_const: 0, u_n_in: 0, u_inputs: null(), rows_x: null(), rows_w: null(), accs: null(), vus: 0,
            row_bytes: 0, units_per_vu: 0, keys: null(), blocks_real: 0, unit_cache: 0, pub_cv: null_mut(), pub_y16: null_mut(),
            pub_acc: null_mut(), u_out_acc: -1, u_out_y: -1, per_role: 0, chunks_per_block: 0, chunk_nb0: 0, chunk_nb1: 0,
            blocks_per_vu: 0, dummy_keyed: 0, host_z: null(), host_a: null(), host_b: null(), sha256: 0, sha_mid: null(),
            sha_pad: null(), last_blen: 64, last_root: 0, scale_slot0: -1, last_data_words: 0,
        }
    }
}

/// The pinned Fast100 schedule and the compression circuit, as the FFI takes them (`hook`/`ch` filled by the caller).
struct Sched {
    log_inv_rates: Vec<i32>,
    recursive_ks: Vec<i32>,
    queries: Vec<i32>,
    grinding_bits: Vec<i32>,
    claim_batch: Vec<i32>,
    consistency_batch: Vec<i32>,
    ood_samples: Vec<i32>,
}

fn sched(pcs: &PcsParams) -> (Sched, flock_prover::pcs::ligerito::ProverConfig) {
    let cfg = pcs.ligerito_prover_config().unwrap_or_else(|_| panic!("no ligerito config for m={}", pcs.m));
    let to_i32 = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    (
        Sched {
            log_inv_rates: to_i32(&cfg.log_inv_rates),
            recursive_ks: to_i32(&cfg.recursive_ks),
            queries: to_i32(&cfg.queries),
            grinding_bits: to_i32(&cfg.grinding_bits),
            claim_batch: to_i32(&cfg.claim_batch_grinding_bits),
            consistency_batch: to_i32(&cfg.consistency_batch_grinding_bits),
            ood_samples: to_i32(&cfg.ood_samples),
        },
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_params(s: &Static, sc: &Sched, cfg: &flock_prover::pcs::ligerito::ProverConfig, pcs: &PcsParams, digest: &[u8; 32], domain: &[u8],
                k_log: usize, pin: usize, ch: &mut LiveChallenger) -> ProveParams {
    let m = pcs.m;
    let zc = pcs.zerocheck_grinding();
    let lc = pcs.lincheck_grinding();
    let og = pcs.opening_grinding();
    let opt = |b: Option<u32>| b.map_or(0, |x| x as i32);
    ProveParams {
        m: m as i32,
        statement_digest: digest.as_ptr(),
        domain: domain.as_ptr(),
        domain_len: domain.len() as u32,
        a_col_ptr: s.a.0.as_ptr(),
        a_rows: s.a.1.as_ptr(),
        a_nnz: s.a.1.len() as u32,
        b_col_ptr: s.b.0.as_ptr(),
        b_rows: s.b.1.as_ptr(),
        b_nnz: s.b.1.len() as u32,
        const_pin_col: pin as i32,
        useful_bits: 1 << k_log,
        k_log: k_log as i32,
        zc_mcol: s.mcol.as_ptr(),
        zc_f8mul: s.f8mul.as_ptr(),
        initial_k: cfg.initial_k as i32,
        num_levels: sc.log_inv_rates.len() as i32,
        log_inv_rates: sc.log_inv_rates.as_ptr(),
        recursive_ks: sc.recursive_ks.as_ptr(),
        queries: sc.queries.as_ptr(),
        grinding_bits: sc.grinding_bits.as_ptr(),
        claim_batch_grinding_bits: sc.claim_batch.as_ptr(),
        consistency_batch_grinding_bits: sc.consistency_batch.as_ptr(),
        ood_samples: sc.ood_samples.as_ptr(),
        recursive_steps: cfg.recursive_steps as i32,
        zc_initial_bits: opt(zc.initial_bits(m)),
        zc_skip_bits: opt(zc.skip_bits()),
        zc_round_bits: opt(zc.multilinear_round_bits()),
        lc_alpha_bits: opt(lc.alpha_bits()),
        lc_beta_bits: opt(lc.beta_bits()),
        lc_round_bits: opt(lc.multilinear_round_bits()),
        lc_skip_bits: opt(lc.skip_bits(K_SKIP)),
        rs_bits: og.ring_switch_bits as i32,
        gamma_bits: og.claim_batch_bits as i32,
        dump_z_path: null(),
        live_cb: Some(live_cb),
        live_ctx: ch as *mut LiveChallenger as *mut c_void,
    }
}

/// verity/flock-pure-block/v1 on the GPU: compressions and units witnessed on the device, the two-circuit lincheck
/// fold, the region claims (points from `hook`) in the sparse ring switch. Same return shape as [`prove_chunk`].
/// Where a pure-block rep's witness inputs come from.
pub enum PureSrc<'a> {
    /// Host compression inputs and unit inputs (any, possibly dishonest, witness).
    Host { comps: &'a [Compression], inputs: &'a (dyn Fn(usize, usize) -> [F128; 5] + Sync) },
    /// The honest witness, built on the device from the instance rows and recorded accumulators; `cache` 1 stores the
    /// unit region for the next rep, 2 reuses it.
    Device { inst: &'a crate::pure_block::Instances, cache: i32 },
    /// The whole block witness `(z, a, b)` from the host (the SHA-256 layouts).
    HostWitness { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
}

pub type PureHook<'a> = &'a mut dyn FnMut(&[u8], Option<&crate::pure_block::Publics>) -> Vec<Vec<F128>>;

#[allow(clippy::too_many_arguments)]
pub fn prove_pure(
    st: &crate::pure_block::PureStmt,
    src: PureSrc<'_>,
    pcs: &PcsParams,
    domain: &[u8],
    hook: PureHook<'_>,
    ch: &mut LiveChallenger,
) -> (R1csProofLigerito, Commitment, [f64; 8]) {
    use crate::pure_block as pb;
    use rayon::prelude::*;
    let s = statics_for(st.lay.comp(), &st.circuit.da, &st.circuit.db);
    let (sc, cfg) = sched(pcs);
    let m = pcs.m;
    let params = prove_params(s, &sc, &cfg, pcs, &st.digest, domain, st.lay.k_log(), st.lay.pin(), ch);
    static U: OnceLock<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> = OnceLock::new();
    let u = U.get_or_init(|| {
        let k = 1usize << pb::UNIT_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        let (ca, cb) = (csc_from_rows(&pad(&st.net.a)), csc_from_rows(&pad(&st.net.b)));
        let csr = |rows: &Vec<Vec<usize>>| {
            let mut off = vec![0u32];
            let mut col = Vec::new();
            for r in rows {
                col.extend(r.iter().map(|&c| c as u32));
                off.push(col.len() as u32);
            }
            (off, col)
        };
        let (ao, ac) = csr(&st.net.a);
        let (bo, bc) = csr(&st.net.b);
        (ca.0, ca.1, cb.0, cb.1, ao, ac, bo, bc)
    });
    let blocks = st.blocks();
    let lay = st.lay;
    let (units, ncv) = (lay.units(), lay.n_cv());
    let n_units = blocks * units;
    let (mut uin, mut cv, mut msg, mut ctr, mut blen, mut flags) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let keys: Vec<u32> = st.keys.iter().flatten().copied().collect();
    let br = st.blocks_real;
    let (mut pub_cv, mut pub_y16, mut pub_acc) = (vec![0u32; br * 8 * ncv], vec![0u32; br], vec![0u32; br]);
    let (dev, cache) = match &src {
        PureSrc::Host { comps, inputs } => {
            uin = (0..n_units).into_par_iter().flat_map_iter(|i| inputs(i / units, i % units).into_iter().flat_map(|w| [w.lo, w.hi])).collect();
            cv = comps.iter().flat_map(|c| c.0).collect();
            msg = comps.iter().flat_map(|c| c.1).collect();
            ctr = comps.iter().map(|c| c.2).collect();
            blen = comps.iter().map(|c| c.3).collect();
            flags = comps.iter().map(|c| c.4).collect();
            assert_eq!(comps.len(), blocks * lay.comp_slots());
            (None, 0)
        }
        PureSrc::Device { inst, cache } => (Some(*inst), *cache),
        PureSrc::HostWitness { .. } => (None, 0),
    };
    let (hz, ha, hb) = match &src {
        PureSrc::HostWitness { z, a, b } => (z.as_ptr(), a.as_ptr(), b.as_ptr()),
        _ => (null(), null(), null()),
    };
    let (pcv, py, pa) = (pub_cv.as_mut_ptr(), pub_y16.as_mut_ptr(), pub_acc.as_mut_ptr());
    let sha_mid: Vec<u32> = (0..2).flat_map(|r| pb::sha_midstate(r, lay.row_bytes())).collect();
    let sha_pad = pb::sha_pad_block(lay.row_bytes());
    let mut wrapped = move |root: &[u8]| -> Vec<Vec<F128>> {
        if dev.is_none() {
            return hook(root, None);
        }
        let (c, y, a) = unsafe { (from_raw_parts(pcv, br * 8 * ncv), from_raw_parts(py, br), from_raw_parts(pa, br)) };
        let p = crate::pure_block::Publics {
            cv: (0..br).map(|b| (0..ncv).map(|r| std::array::from_fn(|k| c[8 * (ncv * b + r) + k])).collect()).collect(),
            y16: y.to_vec(),
            acc: a.to_vec(),
        };
        hook(root, Some(&p))
    };
    let mut hook: Hook<'_> = &mut wrapped;
    let n_claims = 2 + 2 * st.regions.len();
    let og = pcs.opening_grinding();
    let mut phase = [0f64; 8];
    let chunk = ChunkParams {
        n_comp_log: (st.nbl + 5) as i32,
        cv: cv.as_ptr(),
        msg: msg.as_ptr(),
        ctr: ctr.as_ptr(),
        blen: blen.as_ptr(),
        flags: flags.as_ptr(),
        delta_a: s.da.as_ptr(),
        n_delta_a: (s.da.len() / 2) as i32,
        delta_b: s.db.as_ptr(),
        n_delta_b: (s.db.len() / 2) as i32,
        sub_log: lay.comp_log() as i32,
        link_cb: Some(link_cb),
        link_ctx: &mut hook as *mut Hook<'_> as *mut c_void,
        gamma_bits: if n_claims > 1 { og.claim_batch_bits as i32 } else { 0 },
        phase_s: phase.as_mut_ptr(),
        mode: 1,
        comp_slots: lay.comp_slots() as i32,
        units: units as i32,
        unit_log: pb::UNIT_LOG as i32,
        unit_pos0: lay.unit_pos0() as i32,
        u_a_col_ptr: u.0.as_ptr(),
        u_a_rows: u.1.as_ptr(),
        u_a_nnz: u.1.len() as u32,
        u_b_col_ptr: u.2.as_ptr(),
        u_b_rows: u.3.as_ptr(),
        u_b_nnz: u.3.len() as u32,
        u_a_off: u.4.as_ptr(),
        u_a_col: u.5.as_ptr(),
        u_b_off: u.6.as_ptr(),
        u_b_col: u.7.as_ptr(),
        u_useful: st.net.useful as i32,
        u_const: st.net.const_pos as i32,
        u_n_in: st.net.n_in as i32,
        u_inputs: if dev.is_some() { null() } else { uin.as_ptr() },
        rows_x: dev.map_or(null(), |i| i.x.as_ptr()),
        rows_w: dev.map_or(null(), |i| i.w.as_ptr()),
        accs: dev.map_or(null(), |i| i.accs.as_ptr()),
        vus: dev.map_or(0, |i| i.vus as i32),
        row_bytes: dev.map_or(0, |i| i.row_bytes as i32),
        units_per_vu: dev.map_or(0, |i| i.units as i32),
        keys: keys.as_ptr(),
        blocks_real: br as i32,
        unit_cache: cache,
        pub_cv: if dev.is_some() && ncv > 0 { pcv } else { null_mut() },
        pub_y16: if dev.is_some() { py } else { null_mut() },
        pub_acc: if dev.is_some() { pa } else { null_mut() },
        u_out_acc: st.net.out_cols[0] as i32,
        u_out_y: st.net.out_cols.get(1).map_or(-1, |&c| c as i32),
        per_role: lay.per_role() as i32,
        chunks_per_block: lay.chunks().len() as i32,
        chunk_nb0: lay.chunks()[0] as i32,
        chunk_nb1: lay.chunks().get(1).map_or(0, |&n| n as i32),
        blocks_per_vu: lay.blocks_per_vu() as i32,
        dummy_keyed: lay.per_slot_keys() as i32,
        host_z: hz as *const u64,
        host_a: ha as *const u64,
        host_b: hb as *const u64,
        sha256: lay.is_sha() as i32,
        sha_mid: sha_mid.as_ptr(),
        sha_pad: sha_pad.as_ptr(),
        last_blen: lay.blen(lay.chunks()[0] - 1, lay.chunks()[0]) as i32,
        last_root: lay.is_fp4() as i32,
        scale_slot0: if lay.is_fp4() { pb::Layout::FP4_SCALE_BLOCK as i32 } else { -1 },
        last_data_words: if lay.is_sha() { lay.last_data_words() as i32 } else { 0 },
    };
    let (proof, commitment) = run(&params, &chunk, pcs, m);
    (proof, commitment, phase)
}

/// verity/flock-vllm-block/v1 on the GPU: the host's packed witness (SHA-256 slots and units, any witness), the
/// two-circuit lincheck fold with 2^15-bit compression slots, the region claims (points from `hook`) in the sparse ring
/// switch. Same return shape as [`prove_chunk`].
/// Where a flock-vllm-block rep's witness comes from.
pub enum VllmSrc<'a> {
    /// The whole packed (z, a, b), built on the host (any witness, a flipped z bit included).
    Host { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
    /// Every SHA-256 compression's (h_in, message) and every unit's five input words: both witnessed on the device.
    Device { comps: &'a [flock_prover::r1cs_hashes::sha2::Compression], units: &'a [F128] },
}

#[allow(clippy::too_many_arguments)]
pub fn prove_vllm(
    st: &crate::vllm_block::VllmStmt,
    src: VllmSrc<'_>,
    pcs: &PcsParams,
    domain: &[u8],
    mut hook: Hook<'_>,
    ch: &mut LiveChallenger,
) -> (R1csProofLigerito, Commitment, [f64; 8]) {
    use crate::{pure_block as pb, vllm_block as vb};
    let s = statics_for(vb::sha_comp(), &st.circuit.da, &st.circuit.db);
    let (sc, cfg) = sched(pcs);
    let m = pcs.m;
    let params = prove_params(s, &sc, &cfg, pcs, &st.digest, domain, st.lay.k_log(), vb::PIN, ch);
    let n_comp = st.blocks() * st.lay.comp_slots();
    let (mut cv, mut msg, mut zeros32, mut zeros64) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut hz, mut ha, mut hb, mut uin): (*const u64, *const u64, *const u64, *const u64) = (null(), null(), null(), null());
    match &src {
        VllmSrc::Host { z, a, b } => {
            assert_eq!(z.len(), 1 << (m - 7));
            (hz, ha, hb) = (z.as_ptr() as *const u64, a.as_ptr() as *const u64, b.as_ptr() as *const u64);
        }
        VllmSrc::Device { comps, units } => {
            assert_eq!(comps.len(), n_comp);
            assert_eq!(units.len(), 5 * st.blocks() * st.lay.units);
            cv = comps.iter().flat_map(|c| c.0).collect();
            msg = comps.iter().flat_map(|c| c.1).collect();
            zeros32 = vec![0u32; n_comp];
            zeros64 = vec![0u64; n_comp];
            uin = units.as_ptr() as *const u64;
        }
    }
    let device = !uin.is_null();
    let (u_useful, u_const) = (st.net.useful as i32, st.net.const_pos as i32);
    type U8 = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);
    static U: OnceLock<U8> = OnceLock::new();
    let u = U.get_or_init(|| {
        let k = 1usize << pb::UNIT_LOG;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(k, Vec::new());
            SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
        };
        let (ca, cb) = (csc_from_rows(&pad(&st.net.a)), csc_from_rows(&pad(&st.net.b)));
        let csr = |rows: &Vec<Vec<usize>>| {
            let mut off = vec![0u32];
            let mut col = Vec::new();
            for r in rows {
                col.extend(r.iter().map(|&c| c as u32));
                off.push(col.len() as u32);
            }
            (off, col)
        };
        let (ao, ac) = csr(&st.net.a);
        let (bo, bc) = csr(&st.net.b);
        (ca.0, ca.1, cb.0, cb.1, ao, ac, bo, bc)
    });
    let n_claims = 2 + 2 * st.regions.len();
    let og = pcs.opening_grinding();
    let mut phase = [0f64; 8];
    let chunk = ChunkParams {
        delta_a: s.da.as_ptr(),
        n_delta_a: (s.da.len() / 2) as i32,
        delta_b: s.db.as_ptr(),
        n_delta_b: (s.db.len() / 2) as i32,
        sub_log: vb::SUB_LOG as i32,
        link_cb: Some(link_cb),
        link_ctx: &mut hook as *mut Hook<'_> as *mut c_void,
        gamma_bits: if n_claims > 1 { og.claim_batch_bits as i32 } else { 0 },
        phase_s: phase.as_mut_ptr(),
        mode: 1,
        comp_slots: st.lay.comp_slots() as i32,
        units: st.lay.units as i32,
        unit_log: pb::UNIT_LOG as i32,
        unit_pos0: st.lay.unit_pos0() as i32,
        u_a_col_ptr: u.0.as_ptr(),
        u_a_rows: u.1.as_ptr(),
        u_a_nnz: u.1.len() as u32,
        u_b_col_ptr: u.2.as_ptr(),
        u_b_rows: u.3.as_ptr(),
        u_b_nnz: u.3.len() as u32,
        blocks_real: st.vus as i32,
        cv: if device { cv.as_ptr() } else { null() },
        msg: if device { msg.as_ptr() } else { null() },
        ctr: if device { zeros64.as_ptr() } else { null() },
        blen: if device { zeros32.as_ptr() } else { null() },
        flags: if device { zeros32.as_ptr() } else { null() },
        u_a_off: u.4.as_ptr(),
        u_a_col: u.5.as_ptr(),
        u_b_off: u.6.as_ptr(),
        u_b_col: u.7.as_ptr(),
        u_useful,
        u_const,
        u_n_in: 544,
        u_inputs: uin,
        u_out_acc: st.net.out_cols[0] as i32,
        u_out_y: -1,
        host_z: hz,
        host_a: ha,
        host_b: hb,
        sha256: device as i32,
        ..ChunkParams::none()
    };
    let (proof, commitment) = run(&params, &chunk, pcs, m);
    (proof, commitment, phase)
}

fn run(params: &ProveParams, chunk: &ChunkParams, pcs: &PcsParams, m: usize) -> (R1csProofLigerito, Commitment) {
    let mut out: *mut u8 = null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { flock_cuda_prove_chunk(params, chunk, &mut out, &mut out_len) };
    assert_eq!(rc, 0, "CUDA chunk prover returned error {rc} at m={m}");
    let bytes = unsafe { from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };

    let mut r = Reader { b: &bytes, o: 0 };
    let cap = r.hashes();
    let round1_ab = r.f128s();
    let round1_c = r.f128s();
    let n_mlv = r.u64() as usize;
    let multilinear_rounds: Vec<(F128, F128)> = (0..n_mlv).map(|_| (r.f128(), r.f128())).collect();
    let final_a_eval = r.f128();
    let final_b_eval = r.f128();
    let final_c_eval = r.f128();
    let zc_nonces = r.u64s();
    let n_lc = r.u64() as usize;
    let lc_rounds: Vec<(F128, F128)> = (0..n_lc).map(|_| (r.f128(), r.f128())).collect();
    let z_partial = r.f128s();
    let lc_nonces = r.u64s();
    let n_cl = r.u64() as usize;
    let ring_switches: Vec<RingSwitchProof> =
        (0..n_cl).map(|_| RingSwitchProof { s_hat_v: r.f128s(), grinding_nonce: r.u64() }).collect();
    let batching_nonces = r.u64s();
    let n_rcaps = r.u64() as usize;
    let recursive_caps: Vec<Vec<[u8; 32]>> = (0..n_rcaps).map(|_| r.hashes()).collect();
    let n_opens = r.u64() as usize;
    let mut opens: Vec<(Vec<Vec<F128>>, Vec<[u8; 32]>)> = (0..n_opens).map(|_| (r.rows(), r.hashes())).collect();
    let yr = r.f128s();
    let n_sc = r.u64() as usize;
    let sumcheck_transcript_f256: Vec<SumcheckMessage256> = (0..n_sc).map(|_| SumcheckMessage256 { u_0: r.f256(), u_2: r.f256() }).collect();
    let ood_values = r.f128s();
    let grinding_nonces = r.u64s();
    let claim_batch_grinding_nonces = r.u64s();
    let consistency_batch_grinding_nonces = r.u64s();
    assert_eq!(r.o, bytes.len(), "trailing bytes in the FFI stream");
    let (initial_rows, initial_mp) = opens.remove(0);
    let (final_rows, final_mp) = opens.pop().expect("final open");
    let recursive_proofs: Vec<RecursiveProof> =
        opens.into_iter().map(|(rows, mp)| RecursiveProof { opened_rows: rows, merkle_proof: mp }).collect();
    let proof = R1csProofLigerito {
        zerocheck: ZerocheckProof { round1_ab, round1_c, multilinear_rounds, final_a_eval, final_b_eval, final_c_eval, grinding_nonces: zc_nonces },
        lincheck: LincheckProof { rounds: lc_rounds, z_partial, matrix_evals: Vec::new(), grinding_nonces: lc_nonces },
        pcs_open: BatchOpeningProofLigerito {
            ring_switches,
            batching_nonces,
            ligerito: LigeritoProof {
                initial_cap: cap.clone(),
                initial_proof: RecursiveProof { opened_rows: initial_rows, merkle_proof: initial_mp },
                recursive_caps,
                recursive_proofs,
                final_proof: FinalProof { yr, opened_rows: final_rows, merkle_proof: final_mp },
                sumcheck_transcript: Vec::new(),
                sumcheck_transcript_f256,
                grinding_nonces,
                ood_values,
                fold_grinding_nonces: Vec::new(),
                claim_batch_grinding_nonces,
                consistency_batch_grinding_nonces,
            },
        },
    };
    (proof, Commitment { cap, params: pcs.clone() })
}
