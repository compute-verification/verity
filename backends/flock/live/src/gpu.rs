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
    static S: OnceLock<Static> = OnceLock::new();
    S.get_or_init(|| {
        let c = comp();
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
        Static { a: csc_from_rows(&c.a_0), b: csc_from_rows(&c.b_0), mcol, f8mul, da: flat(&st.circuit.da), db: flat(&st.circuit.db) }
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
    };
    let mut out: *mut u8 = null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { flock_cuda_prove_chunk(&params, &chunk, &mut out, &mut out_len) };
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
    (proof, Commitment { cap, params: pcs.clone() }, phase)
}
