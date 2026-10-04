//! rec-holo: the kill experiment's circuits (rec_holo_kill.rs `build`) proved as C-Flock proves them, M0 and `--zk`, on a
//! C-Flock checkout (Flock b684b12 + the flock-link, flock-gpu-link, sha512, glue, zk and vtime patches, flock-live in the
//! workspace). 79-rec-holo-zk.sh copies this file to crates/flock-live/examples. Each mode is its own process and prints one
//! JSON line per measurement.
//!
//! A session is flock-circuit's CPU session without the network: the witness once, then two reps, each a Fast100 proof whose
//! Merkle trees are hm96-sha512 under the session's salt key. `m0` proves a rep as flock-circuit's `prove_cpu` does; `zk` as
//! its `prove_cpu_zk` and `zk_finish` do (Ligerito level 0's hiding, the rep's pads and their commitment, the masking
//! challenger with the rank check at the opening, the replay, the inner proof). An FsChallenger stands in for the live stream,
//! and the extra claims are random points (the verifier's link points). Every statement keeps a mask slot of whole words
//! (A = B = I, uniform bits) of at least 576 words, the rank check's 4 · 128 + 64.
//! - `base DIR COPIES m0|zk`: the circuit's C-Flock rows tiled COPIES ≥ 8 times, the last 2^14 rows of each block its mask
//!   slot; 4 extra claims (two I/O regions at two link points).
//! - `gate DIR m0|zk`: route P's gate table, the last words of each block its mask slot; 7 extra claims (the wiring's three and
//!   the I/O regions' four). A build with READ_D = D > 0 adds each block's READ_SLOTS D-bit read comparators and 2 claims for
//!   their public region.
//! - `acyc DIR m0|zk [UNIT_LOG]`: the acyclicity table over σ's predecessor map (`acyc`); 3 extra claims.
//! - `sigma DIR m0|zk [UNIT_LOG]`: σ committed (zk: with level 0's hiding) and opened at one point; zk adds a
//!   re-registration, a fresh commitment of the same σ with both opened at one point.
//!
//! env: RUNS=2 (timed sessions after a first, cold one that also gives peak memory), ACYC_KLOG=16 (the acyclicity table's
//! block).
use std::{
    env::{args, var},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, PoisonError},
    time::Instant,
};

use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::pack_z_lincheck_from_packed,
    merkle::HashKind,
    pcs::{
        PcsParams,
        commit::{commit, commit_zk},
        ligerito::{
            LigeritoProfile, embedded_initial_k_or_default,
            zk::{EXTRA_LANES, Level0Zk, prover_config},
        },
        open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding_zk,
    },
    proof::R1csProofLigerito,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    test_rng::Rng,
    zerocheck::PaddingSpec,
};
use flock_field::F128;
use flock_live::{
    zk_hooks::{LeafSalts, ProverRng, ProverSeed},
    zk_veil::{self as zkv, Layout},
};
use flock_prover::{
    pcs::Commitment,
    prover::{prove_fast_ligerito_from_witness_extra, prove_fast_ligerito_from_witness_extra_zk},
};
use rayon::prelude::*;

const MERKLE: HashKind = HashKind::Hm96Sha512;
const MASK_MIN: usize = 128 * 4 + 64;
const BASE_MASK_ROWS: usize = 1 << 14;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn status_mb(key: &str) -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()))
        .map_or(0.0, |kb| kb / 1024.0)
}

fn reset_peak() -> f64 {
    match fs::write("/proc/self/clear_refs", "5") {
        Ok(()) => status_mb("VmRSS:"),
        Err(_) => -1.0,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn read_u64s(p: &Path) -> Vec<u64> {
    fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn read_u32s(p: &Path) -> Vec<u32> {
    fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn meta(dir: &Path, key: &str) -> usize {
    let s = fs::read_to_string(dir.join("meta.json")).unwrap();
    let at = s.find(&format!(r#""{key}":"#)).unwrap_or_else(|| panic!("meta.json has no {key}")) + key.len() + 3;
    s[at..].split([',', '}']).next().unwrap().parse().unwrap()
}

fn meta_or(dir: &Path, key: &str, default: usize) -> usize {
    let s = fs::read_to_string(dir.join("meta.json")).unwrap();
    if s.contains(&format!(r#""{key}":"#)) { meta(dir, key) } else { default }
}

fn bits_to_packed(z: &[u64]) -> Vec<F128> {
    z.par_chunks(2).map(|c| F128::new(c[0], c[1])).collect()
}

fn params(m: usize) -> PcsParams {
    let lig = LigeritoProfile::Fast100;
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: MERKLE }
}

fn block_r1cs(m: usize, k_log: usize, rows_a: Vec<Vec<usize>>, rows_b: Vec<Vec<usize>>, pin: usize) -> BlockR1cs {
    let k = 1usize << k_log;
    BlockR1cs {
        m,
        k_log,
        k_skip: 6,
        useful_bits: k,
        a_0: SparseBinaryMatrix::new(k, k, rows_a),
        b_0: SparseBinaryMatrix::new(k, k, rows_b),
        c_0: SparseBinaryMatrix::new(k, k, (0..k).map(|i| vec![i]).collect()),
        layout: WitnessLayout::RowMajor,
        const_pin: Some(pin),
        digest_cache: OnceLock::new(),
        csc_cache: OnceLock::new(),
    }
}

/// flock-circuit's `zk_queries0`: Ligerito level 0's query count under its padding.
fn queries0(m: usize) -> usize {
    assert!(m >= 25, "m = {m}: --zk needs m >= 25");
    let lc = params(m).ligerito_prover_config().unwrap();
    let l = 1usize << lc.initial_log_msg_cols;
    let q = match m {
        25 => 277,
        26 => 242,
        27 => 229,
        _ => zkv::level0_queries(l, lc.queries[0]).unwrap_or_else(|| panic!("m = {m}: no level-0 query count reaches 100 bits")),
    };
    assert!(zkv::level0_t_pad(q) <= l && zkv::level0_bits(l, q) >= 100.0, "m = {m}: {q} level-0 queries fall short");
    q
}

/// flock-circuit's `level0_zk`: every lane's padding and the four uniform lanes, from the prover's streams 6 and 7.
fn level0(m: usize, rng: &mut ProverRng) -> Level0Zk {
    let lc = params(m).ligerito_prover_config().unwrap();
    let q = queries0(m);
    let (lanes, l, t) = ((1usize << lc.initial_k) + EXTRA_LANES, 1usize << lc.initial_log_msg_cols, zkv::level0_t_pad(q));
    Level0Zk {
        queries0: q,
        t_pad: t,
        rep: 0,
        extra: std::array::from_fn(|i| rng.f128s(6, (i as u64) << 32, l)),
        pads: (0..lanes).map(|i| rng.f128s(7, (i as u64) << 32, t)).collect(),
        tamper: 0,
    }
}

/// The salt context of one proof: the session's level-0 salts for its first tree, a fresh id for each later one.
fn salts_begin(level0: &Arc<Vec<u8>>, salts: &Arc<Mutex<LeafSalts>>) {
    let s = salts.clone();
    flock_merkle::hm96::begin(
        level0.clone(),
        Box::new(move |n| {
            let mut g = s.lock().unwrap_or_else(PoisonError::into_inner);
            let id = g.next_id();
            g.expand(id, n)
        }),
    );
}

// ---------------------------------------------------------------------------
// a session (flock-circuit's CPU prover, both reps)
// ---------------------------------------------------------------------------

struct St {
    r1cs: Arc<BlockR1cs>,
    mask: Vec<usize>,
    extra: usize,
}

type Wit = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

struct ZkRep {
    lay: Layout,
    pads: zkv::Pads,
    ic: zkv::InnerCommit,
    level0: Level0Zk,
}

struct Rep {
    prove_s: f64,
    bytes: usize,
    buckets: String,
}

fn make_zkrep(st: &St, prng: &mut ProverRng, level0: &Level0Zk, rep: usize, salts: &Arc<Mutex<LeafSalts>>) -> ZkRep {
    let lc = params(st.r1cs.m).ligerito_prover_config().unwrap();
    let lay = Layout::new(st.r1cs.m, st.r1cs.k_log, lc.ood_samples[0], lc.initial_k);
    let pads = zkv::Pads::draw(prng, &lay, rep);
    let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
    let sb = {
        let mut s = salts.lock().unwrap_or_else(PoisonError::into_inner);
        let id = s.next_id();
        s.expand(id, n)
    };
    let ic = zkv::InnerCommit::new(&pads, sb);
    ZkRep { lay, pads, ic, level0: Level0Zk { rep, ..level0.clone() } }
}

fn claim_word_points(r1cs: &BlockR1cs, lay: &Layout, coins: &[F128]) -> [Vec<F128>; 2] {
    let (ab, c) = zkv::points_from_coins(r1cs, lay, true, coins).unwrap_or_else(|e| panic!("{e}"));
    [zkv::point_parts(&ab).1[1..].to_vec(), zkv::point_parts(&c).1[1..].to_vec()]
}

fn prove_m0(st: &St, w: Wit, extra: Vec<Vec<F128>>, ch: &mut FsChallenger) -> Rep {
    let (z, a, b, zl) = w;
    let tp = Instant::now();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs,
        &params(st.r1cs.m),
        z,
        a,
        b,
        zl,
        st.r1cs.csc_lincheck_circuit(),
        &mut |_: &Commitment| extra.clone(),
        ch,
    );
    let prove_s = tp.elapsed().as_secs_f64();
    Rep { prove_s, bytes: bincode::serialize(&(commitment, proof)).unwrap().len(), buckets: String::new() }
}

fn prove_zk(st: &St, w: &Wit, extra: Vec<Vec<F128>>, ch: FsChallenger, zr: &ZkRep, points: &Arc<Mutex<Vec<Vec<F128>>>>) -> Rep {
    let (z, a, b, zl) = w.clone();
    let tp = Instant::now();
    let sent = zkv::Pads { h: zr.pads.h.clone(), mu: vec![], pad_h: vec![], pad_mu: vec![], tau_salt: vec![] };
    let mut mch = zkv::Masking::new(ch, &zr.lay, &sent, zr.ic.root());
    let (r1cs, lay, pts, words) = (st.r1cs.clone(), zr.lay, points.clone(), st.mask.clone());
    mch.at_open = Some(Box::new(move |coins: &[F128]| {
        let mut all = pts.lock().unwrap_or_else(PoisonError::into_inner);
        all.extend(claim_word_points(&r1cs, &lay, coins));
        let (rank, ok) = zkv::mask_rank_ok(&all, &words);
        if ok { Ok(()) } else { Err(format!("degenerate coins: rank {rank} < {}", 128 * all.len())) }
    }));
    let circuit = st.r1cs.csc_lincheck_circuit();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra_zk(
        &st.r1cs,
        &params(st.r1cs.m),
        z,
        a,
        b,
        zl,
        circuit,
        &mut |_: &Commitment| extra.clone(),
        Some(&zr.level0),
        &mut mch,
    );
    let t_upstream = tp.elapsed().as_secs_f64();
    let (mut inner_ch, coins, _) = mch.finish();
    let tz = Instant::now();
    let (zc, lc) = zkv::mask_proof(&proof.zerocheck, &proof.lincheck, &zr.lay, &sent.h);
    let rp = zkv::replay(&st.r1cs, &zc, &lc, circuit, &zr.lay, &zr.ic.root(), Some(&proof.lincheck.comb_partial), &mut zkv::Tape::new(&coins))
        .unwrap_or_else(|e| panic!("{e}"));
    let t_replay = tz.elapsed().as_secs_f64();
    let rs = &proof.pcs_open.ring_switches;
    let (_, con_ab) = zkv::ring_switch_constraint(&rp.ab_point, &rs[0].s_hat_v, &rp.ab_value);
    let (_, con_c) = zkv::ring_switch_constraint(&rp.c_point, &rs[1].s_hat_v, &rp.c_value);
    let mut cons = rp.cons;
    cons.push(("ring switch ab", con_ab));
    cons.push(("ring switch c", con_c));
    let mut pcs_open = proof.pcs_open;
    let rec = pcs_open.ligerito.zk_record.take().expect("the zk prover records level 0's lane phase");
    zkv::mask_ligerito(&mut pcs_open.ligerito, &zr.lay, &sent.h);
    let t_prime = pcs_open.ligerito.zk_level0.as_ref().expect("level 0's hiding messages").t_prime;
    cons.extend(zkv::ligerito_constraints(&zr.lay, &rec, &zkv::ligerito_exposed(&pcs_open.ligerito, &zr.lay), t_prime).unwrap_or_else(|e| panic!("{e}")));
    let fails = zkv::failing(&cons, &zr.pads.h);
    let n_cons = cons.len();
    let mut lch = zkv::Logging { inner: &mut inner_ch, coins: Vec::new() };
    let inner = zkv::prove_inner(&zr.ic, &zr.pads, &zr.lay, zc.final_c_eval, cons, &mut lch).unwrap_or_else(|e| panic!("{e}"));
    let t_zk = tz.elapsed().as_secs_f64();
    let prove_s = tp.elapsed().as_secs_f64();
    let masked = R1csProofLigerito { zerocheck: zc, lincheck: lc, pcs_open };
    let bytes = bincode::serialize(&(commitment, masked, inner)).unwrap().len();
    assert!(fails.is_empty(), "constraints failing on the honest pads: {fails:?}");
    Rep {
        prove_s,
        bytes,
        buckets: format!(
            r#""t_upstream":{t_upstream:.6},"t_zk_replay":{t_replay:.6},"t_zk_inner":{t_zk:.6},"constraints":{n_cons},"constraints_failing_on_pads":{},"#,
            fails.len()
        ),
    }
}

/// One session: the witness (A·z, B·z, the lincheck packing), then two reps. Returns (witness s, the reps, the pads' and level
/// 0's draw and commitment per rep).
fn session(st: &St, z: &[F128], zk: bool, rng: &mut Rng) -> (f64, Vec<(Rep, f64)>) {
    let m = st.r1cs.m;
    let seed = ProverSeed::from_os();
    let salts = Arc::new(Mutex::new(LeafSalts::for_seed(&seed).table(0)));
    let level0_salts = {
        let s = salts.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::new(s.expand(s.level0_id(), params(m).n_leaves()))
    };
    let t = Instant::now();
    let a = st.r1cs.apply_a_packed(z);
    let b = st.r1cs.apply_b_packed(z);
    let zl = pack_z_lincheck_from_packed(z, m, st.r1cs.k_log);
    let mut wt = Some((z.to_vec(), a, b, zl));
    let witness_s = t.elapsed().as_secs_f64();
    let mut prng = ProverRng::from_seed(seed);
    let tl0 = Instant::now();
    let l0 = zk.then(|| level0(m, &mut prng));
    let t_level0 = tl0.elapsed().as_secs_f64();
    let points = Arc::new(Mutex::new(Vec::new()));
    let mut reps = Vec::new();
    for rep in 0..2 {
        let tz0 = Instant::now();
        let zr = l0.as_ref().map(|l| make_zkrep(st, &mut prng, l, rep, &salts));
        let t_commit = tz0.elapsed().as_secs_f64() + if rep == 0 { t_level0 } else { 0.0 };
        let extra: Vec<Vec<F128>> = (0..st.extra).map(|_| rng.f128_vec(m - 6)).collect();
        let ch = FsChallenger::new(format!("rec-holo-zk/rep{rep}").as_bytes());
        let tp = Instant::now();
        let w = if rep == 0 { wt.clone().unwrap() } else { wt.take().unwrap() };
        salts_begin(&level0_salts, &salts);
        let mut r = match &zr {
            None => {
                let mut ch = ch;
                prove_m0(st, w, extra, &mut ch)
            }
            Some(zr) => prove_zk(st, &w, extra, ch, zr, &points),
        };
        flock_merkle::hm96::end();
        r.prove_s = tp.elapsed().as_secs_f64() + t_commit;
        reps.push((r, t_commit));
    }
    (witness_s, reps)
}

fn sessions(label: &str, st: &St, z: &[F128], zk: bool, json: &str) {
    assert!(st.r1cs.satisfies_packed(z), "{label}: the witness does not satisfy the R1CS");
    let mode = if zk { "zk" } else { "m0" };
    if zk {
        assert!(st.mask.len() >= MASK_MIN, "{label}: {} mask words, the rank check needs {MASK_MIN}", st.mask.len());
    }
    let t = Instant::now();
    st.r1cs.statement_digest();
    st.r1cs.csc_lincheck_circuit();
    let setup = t.elapsed().as_secs_f64();
    let mut rng = Rng::new(0x2C_F10C);
    let rss = reset_peak();
    let (w0, reps0) = session(st, z, zk, &mut rng);
    let peak = status_mb("VmHWM:");
    let (mut tw, mut t0, mut t1) = (vec![], vec![], vec![]);
    let mut last = reps0;
    for _ in 0..env_or("RUNS", 2usize) {
        let (w, reps) = session(st, z, zk, &mut rng);
        tw.push(w);
        t0.push(reps[0].0.prove_s);
        t1.push(reps[1].0.prove_s);
        last = reps;
    }
    let (w, p0, p1) = (median(tw), median(t0), median(t1));
    println!(
        r#"{{"kind":"{label}","mode":"{mode}","m":{},"k_log":{},"blocks":{},"mask_words":{},"extra_claims":{},{json}"threads":{},"setup_s":{setup:.3},"witness_s":{w:.6},"rep_prove_s":[{p0:.6},{p1:.6}],"prove_s":{:.6},"total_s":{:.6},"cold_witness_s":{w0:.6},"cold_rep_prove_s":[{:.6},{:.6}],"rep0_zk_commit_s":{:.6},"rep0":{{{}"bytes":{}}},"rep1":{{{}"bytes":{}}},"proof_bytes":{},"rss_at_reset_mb":{rss:.1},"peak_mb":{peak:.1}}}"#,
        st.r1cs.m,
        st.r1cs.k_log,
        1usize << (st.r1cs.m - st.r1cs.k_log),
        st.mask.len(),
        st.extra,
        rayon::current_num_threads(),
        p0 + p1,
        w + p0 + p1,
        last[0].0.prove_s,
        last[1].0.prove_s,
        last[0].1,
        last[0].0.buckets,
        last[0].0.bytes,
        last[1].0.buckets,
        last[1].0.bytes,
        last[0].0.bytes + last[1].0.bytes,
    );
}

// ---------------------------------------------------------------------------
// statements
// ---------------------------------------------------------------------------

fn load_rows(p: &Path, k: usize) -> (Vec<Vec<usize>>, usize) {
    let v = read_u32s(p);
    let (n, nnz) = (v[0] as usize, v[1] as usize);
    let ptr = &v[2..3 + n];
    let idx = &v[3 + n..3 + n + nnz];
    let mut rows: Vec<Vec<usize>> = (0..n).into_par_iter().map(|i| idx[ptr[i] as usize..ptr[i + 1] as usize].iter().map(|&c| c as usize).collect()).collect();
    rows.resize(k, Vec::new());
    (rows, nnz)
}

/// Rows `[k − rows, k)` of every block become the mask slot: identity rows (A = B = I), every bit of z there uniform. Returns
/// the slot's word indices.
fn mask_slot(z: &mut [u64], ra: &mut [Vec<usize>], rb: &mut [Vec<usize>], k_log: usize, m: usize, rows: usize, rng: &mut Rng) -> Vec<usize> {
    let k = 1usize << k_log;
    for i in k - rows..k {
        ra[i] = vec![i];
        rb[i] = vec![i];
    }
    let mut words = Vec::new();
    for b in 0..1usize << (m - k_log) {
        for x in ((b << k_log) + k - rows) / 64..((b + 1) << k_log) / 64 {
            z[x] = rng.next_u64();
        }
        words.extend(((b << k_log) + k - rows) / 128..((b + 1) << k_log) / 128);
    }
    words
}

fn base(dir: &Path, copies: usize, zk: bool) {
    assert!(copies.is_power_of_two() && copies >= 8);
    let k_log = meta(dir, "base_klog");
    let k = 1usize << k_log;
    let m = k_log + copies.trailing_zeros() as usize;
    let block = read_u64s(&dir.join("base_z.bin"));
    let (mut ra, nnz_a) = load_rows(&dir.join("base_a.bin"), k);
    let (mut rb, nnz_b) = load_rows(&dir.join("base_b.bin"), k);
    let lo = k - BASE_MASK_ROWS;
    assert!(ra[lo..].iter().chain(&rb[lo..]).all(|r| r.is_empty()), "the circuit uses rows of the mask slot");
    assert!(ra.par_iter().chain(rb.par_iter()).all(|r| r.iter().all(|&c| c < lo)), "the circuit reads columns of the mask slot");
    assert!(block[lo / 64..].iter().all(|&x| x == 0));
    let mut z: Vec<u64> = block.iter().cycle().take(block.len() * copies).copied().collect();
    let mut rng = Rng::new(0x3A5C);
    let mask = mask_slot(&mut z, &mut ra, &mut rb, k_log, m, BASE_MASK_ROWS, &mut rng);
    let st = St { r1cs: Arc::new(block_r1cs(m, k_log, ra, rb, 0)), mask, extra: 4 };
    let json = format!(r#""copies":{copies},"nnz_a":{nnz_a},"nnz_b":{nnz_b},"#);
    sessions("base", &st, &bits_to_packed(&z), zk, &json);
}

/// rec_holo_kill.rs's `cmp_rows`: each block's `n` read comparators, `d` groups (p_k, c_k, t_k, x_k) each, after its
/// constant row; a unit's reads use its first block's.
fn cmp_rows(d: usize, n: usize, pin: usize) -> Vec<(usize, Vec<usize>, Vec<usize>)> {
    let xor = |v: Vec<usize>| xor_row(&v, &[], false);
    let mut rows = Vec::new();
    for g in 0..d * n {
        let (k, q) = (g % d, 4 * (1 + g));
        rows.push((q, vec![q], vec![q]));
        rows.push((q + 1, vec![q + 1], vec![q + 1]));
        let lt = if k == 0 { vec![] } else { vec![q - 3, q - 2] };
        rows.push((q + 2, xor(vec![q, q + 1, pin]), xor([lt, vec![q + 1]].concat())));
        rows.push(if k + 1 == d { (q + 3, xor(vec![q + 1, q + 2]), vec![pin]) } else { (q + 3, vec![], vec![]) });
    }
    rows
}

/// Route P's gate table. With the build's `read_d` > 0, every block's comparators too (`cmp_rows`), and 2 more extra claims
/// for their public region (base(u)'s bits and the results).
fn gate(dir: &Path, zk: bool) {
    let (mu, k_log, read_d) = (meta(dir, "mu"), meta(dir, "gate_klog"), meta_or(dir, "read_d", 0));
    let read_n = meta_or(dir, "read_slots", 0);
    let k = 1usize << k_log;
    let mut z = read_u64s(&dir.join("rp_z.bin"));
    let (mut ra, mut rb) = (vec![Vec::new(); k], vec![Vec::new(); k]);
    for g in 0..k / 4 {
        let c = 4 * g;
        ra[c] = vec![c];
        rb[c] = vec![c];
        ra[c + 1] = vec![c + 1];
        rb[c + 1] = vec![c + 1];
        ra[c + 2] = vec![c];
        rb[c + 2] = vec![c + 1];
    }
    for (c, a, b) in cmp_rows(read_d, read_n, 0) {
        ra[c] = a;
        rb[c] = b;
    }
    let blocks = 1usize << (mu - k_log);
    let words = MASK_MIN.div_ceil(blocks).next_power_of_two();
    assert!(4 * (1 + read_d * read_n) <= k - 128 * words, "the comparators meet the mask slot");
    let mut rng = Rng::new(0x6A7E);
    let mask = mask_slot(&mut z, &mut ra, &mut rb, k_log, mu, 128 * words, &mut rng);
    let st = St { r1cs: Arc::new(block_r1cs(mu, k_log, ra, rb, 0)), mask, extra: 7 + if read_d > 0 { 2 } else { 0 } };
    let json = format!(
        r#""mask_rows_per_block":{},"read_d":{read_d},"read_slots":{read_n},"comparator_cells":{},"#,
        128 * words,
        4 * read_d * read_n * blocks
    );
    sessions("gate", &st, &bits_to_packed(&z), zk, &json);
}

// ---------------------------------------------------------------------------
// acyc: the one-time acyclicity proof over σ's predecessor map
// ---------------------------------------------------------------------------

/// σ in canonical order and its inverse, for unit 0's cells (`unit_log`; the full form is one unit). Route P's cell `4g + t`
/// is gate row g's l, r, a, x (t = 0..3): l and r are consumers, a and x sources. Every net's cycle is put in increasing cell
/// order, so a consumer's predecessor π(j) = σ^{-1}(j) is the net's previous cell. An unused consumer (a padding row's or a
/// block constant row's l or r, a fixed point of σ) joins the unit's constant net of its value (cell 0: constant 1, cell 3:
/// constant 0). Returns (π, consumers whose predecessor is larger: the I/O rows' input cells, heads of their nets, which
/// a production layout places at public positions and types as sources; unused consumers joined).
fn predecessors(dir: &Path, unit_log: usize) -> (Vec<u32>, usize, usize) {
    let mut s = read_u32s(&dir.join("rp_sigma.bin"));
    s.truncate(1 << unit_log);
    assert!(s.iter().all(|&v| (v as usize) < s.len()), "unit 0's σ_u leaves the unit");
    let w = read_u64s(&dir.join("rp_w.bin"));
    let bit = |i: usize| (w[i >> 6] >> (i & 63)) & 1 == 1;
    let n = s.len();
    let mut seen = vec![false; n];
    let mut joined = [Vec::new(), Vec::new()];
    for j in 0..n {
        if s[j] as usize == j && j % 4 < 2 && j >= 4 {
            joined[bit(j) as usize].push(j as u32);
            seen[j] = true;
        }
    }
    let n_joined = joined[0].len() + joined[1].len();
    let mut cyc = Vec::new();
    for j0 in 0..n {
        if seen[j0] {
            continue;
        }
        cyc.clear();
        let mut x = j0;
        while !seen[x] {
            seen[x] = true;
            cyc.push(x as u32);
            x = s[x] as usize;
        }
        for (v, head) in [(1usize, 0usize), (0, 3)] {
            if j0 == head {
                assert_eq!(bit(head), v == 1);
                cyc.extend_from_slice(&joined[v]);
            }
        }
        cyc.sort_unstable();
        for t in 0..cyc.len() {
            s[cyc[t] as usize] = cyc[(t + 1) % cyc.len()];
        }
    }
    let mut pi = vec![0u32; n];
    for (j, &v) in s.iter().enumerate() {
        pi[v as usize] = j as u32;
        assert_eq!(bit(j), bit(v as usize), "w is not constant on σ's cycle through {j}");
    }
    let io = (0..n).filter(|&j| j % 4 < 2 && pi[j] as usize > j).count();
    (pi, io, n_joined)
}

#[derive(Clone, Copy)]
enum Bit {
    K(bool),
    H(usize),
}

/// The spare rows of a block's first cells that hold the block index's bits (row 60 of cell 0 is the pinned constant).
fn hcol(t: usize) -> usize {
    if t < 3 { 61 + t } else { 64 * (1 + (t - 3) / 4) + 60 + (t - 3) % 4 }
}

const PIN: usize = 60;

/// An XOR of columns and constants as a row of A or B (the constant 1 is the pinned column).
fn xor_row(cols: &[usize], bits: &[Bit], one: bool) -> Vec<usize> {
    let mut one = one;
    let mut v: Vec<usize> = cols.to_vec();
    for b in bits {
        match *b {
            Bit::K(k) => one ^= k,
            Bit::H(c) => v.push(c),
        }
    }
    if one {
        v.push(PIN);
    }
    v.sort_unstable();
    let mut out: Vec<usize> = Vec::new();
    for c in v {
        if out.last() == Some(&c) {
            out.pop();
        } else {
            out.push(c);
        }
    }
    out
}

/// The acyclicity table: one 64-row word per consumer cell j (the l and r cells of every gate row, in order), holding the bits
/// of π(j) (rows 0..μ, the link to com_σ's 32-bit slot; rows μ..32 forced 0), the comparator π(j) < j bit by bit from the
/// low end (row 32 + k: `(π_k ⊕ j_k ⊕ 1) · (lt_k ⊕ j_k)`, with lt_{k+1} = j_k ⊕ that row and lt_0 = 0) and its result
/// (row 32 + μ). The low bits of j are constants of the cell's position in its block; the high bits, the block's index, are
/// cells of the block's spare rows that a public region fixes, with the result rows (1 at every consumer). The last words of
/// each block are its mask slot (they cover 2·words cells per block that a production layout moves to a slot of their own).
fn acyc(dir: &Path, zk: bool, unit_log: Option<usize>) {
    let mu = unit_log.unwrap_or_else(|| meta(dir, "mu"));
    assert!(mu <= 27, "μ = {mu}: the comparator's rows and the spare rows need μ ≤ 27");
    let t = Instant::now();
    let (pi, io, joined) = predecessors(dir, mu);
    let canon_s = t.elapsed().as_secs_f64();
    let k_log: usize = env_or("ACYC_KLOG", 16);
    let (k, cpb) = (1usize << k_log, 1usize << (k_log - 6));
    let n_cons = 1usize << (mu - 1);
    let m = mu + 5;
    assert!(m >= k_log + 3, "the table needs at least 8 blocks");
    let blocks = 1usize << (m - k_log);
    let words = MASK_MIN.div_ceil(blocks).next_power_of_two();
    let masked_cells = 2 * words;
    let jbit = |p: usize, kk: usize| -> Bit {
        match kk {
            0 => Bit::K(p & 1 == 1),
            1 => Bit::K(false),
            _ if kk <= k_log - 6 => Bit::K((p >> (kk - 1)) & 1 == 1),
            _ => Bit::H(hcol(kk - (k_log - 5))),
        }
    };
    let (mut ra, mut rb) = (vec![Vec::new(); k], vec![Vec::new(); k]);
    for p in 0..cpb - masked_cells {
        let q = 64 * p;
        for kk in 0..mu {
            ra[q + kk] = vec![q + kk];
            rb[q + kk] = vec![q + kk];
            let c = jbit(p, kk);
            ra[q + 32 + kk] = xor_row(&[q + kk], &[c], true);
            rb[q + 32 + kk] = if kk == 0 { xor_row(&[], &[c], false) } else { xor_row(&[q + 32 + kk - 1], &[jbit(p, kk - 1), c], false) };
        }
        ra[q + 32 + mu] = xor_row(&[q + 32 + mu - 1], &[jbit(p, mu - 1)], false);
        rb[q + 32 + mu] = vec![PIN];
    }
    let hbits = mu - (k_log - 5);
    for c in std::iter::once(PIN).chain((0..hbits).map(hcol)) {
        ra[c] = vec![c];
        rb[c] = vec![c];
    }
    let mut z = vec![0u64; n_cons];
    let bad: usize = z
        .par_chunks_mut(cpb)
        .enumerate()
        .map(|(b, zb)| {
            let mut bad = 0;
            for (p, word) in zb.iter_mut().enumerate().take(cpb - masked_cells) {
                let jp = (b << (k_log - 6)) | p;
                let j = 4 * (jp >> 1) + (jp & 1);
                let x = pi[j] as u64;
                let (mut lt, mut ands) = (0u64, 0u64);
                for kk in 0..mu {
                    let (xb, cb) = ((x >> kk) & 1, (j as u64 >> kk) & 1);
                    let zk = (xb ^ cb ^ 1) & (lt ^ cb);
                    ands |= zk << kk;
                    lt = cb ^ zk;
                }
                assert_eq!(lt == 1, (pi[j] as usize) < j);
                if lt == 0 && (pi[j] as usize) <= j {
                    bad += 1;
                }
                *word = x | ands << 32 | lt << (32 + mu);
            }
            zb[0] |= 1 << PIN;
            for t in 0..hbits {
                let c = hcol(t);
                zb[c / 64] |= ((b >> t) as u64 & 1) << (c % 64);
            }
            bad
        })
        .sum();
    assert_eq!(bad, 0, "consumers whose predecessor is not smaller and that are not inputs");
    let mut rng = Rng::new(0xAC7C);
    let mask = mask_slot(&mut z, &mut ra, &mut rb, k_log, m, 128 * words, &mut rng);
    let build_s = t.elapsed().as_secs_f64();
    let st = St { r1cs: Arc::new(block_r1cs(m, k_log, ra, rb, PIN)), mask, extra: 3 };
    let json = format!(
        r#""cells_log":{mu},"consumer_cells":{n_cons},"input_cells":{io},"unused_joined":{joined},"masked_cells":{},"canon_s":{canon_s:.3},"build_s":{build_s:.3},"#,
        masked_cells * blocks
    );
    sessions("acyc", &st, &bits_to_packed(&z), zk, &json);
}

// ---------------------------------------------------------------------------
// sigma: the registered wiring commitment, opened at one point; re-registration
// ---------------------------------------------------------------------------

fn sigma(dir: &Path, zk: bool, unit_log: Option<usize>) {
    let mut s = read_u32s(&dir.join("rp_sigma.bin"));
    if let Some(u) = unit_log {
        s.truncate(1 << u);
        assert!(s.iter().all(|&v| (v as usize) < s.len()), "unit 0's σ_u leaves the unit");
    }
    let mu = s.len().trailing_zeros() as usize;
    let m = mu + 5;
    let packed: Vec<F128> = s.par_chunks(4).map(|c| F128::new(c[0] as u64 | (c[1] as u64) << 32, c[2] as u64 | (c[3] as u64) << 32)).collect();
    drop(s);
    let pp = params(m);
    let lc = pp.ligerito_prover_config().unwrap();
    let lig = if zk { prover_config(lc, queries0(m)) } else { lc };
    let mut rng = Rng::new(0x51_6A);
    // one commitment's proof context: its salts, its level-0 hiding; commit, then open at each point
    let run = |xs: &[Vec<F128>], rng: &mut Rng| -> (f64, f64, Vec<f64>, usize, usize) {
        let seed = ProverSeed::from_os();
        let salts = Arc::new(Mutex::new(LeafSalts::for_seed(&seed).table(0)));
        let l0s = {
            let g = salts.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::new(g.expand(g.level0_id(), pp.n_leaves()))
        };
        let mut prng = ProverRng::from_seed(seed);
        let t = Instant::now();
        let z0 = zk.then(|| level0(m, &mut prng));
        salts_begin(&l0s, &salts);
        let (cm, pd) = match &z0 {
            Some(z0) => commit_zk(&packed, &pp, z0),
            None => commit(&packed, &pp),
        };
        let tc = t.elapsed().as_secs_f64();
        let peak_c = status_mb("VmHWM:");
        let mut opens = Vec::new();
        let mut bytes = 0;
        for x in xs {
            let mut ch = FsChallenger::new(b"rec-holo-zk/sigma");
            ch.observe_f128(rng.f128());
            let t = Instant::now();
            let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding_zk(
                packed.clone(),
                &pd,
                &cm,
                &[x.as_slice()],
                &[None],
                &[],
                &PaddingSpec::dense(m),
                &lig,
                pp.opening_grinding(),
                z0.as_ref(),
                &mut ch,
            );
            opens.push(t.elapsed().as_secs_f64());
            bytes = bincode::serialize(&proof).unwrap().len();
        }
        flock_merkle::hm96::end();
        (tc, peak_c, opens, bytes, bincode::serialize(&cm).unwrap().len())
    };
    let mode = if zk { "zk" } else { "m0" };
    let rss = reset_peak();
    let x0 = rng.f128_vec(m - 6);
    let (tc0, peak_c, o0, bytes, cm_bytes) = run(&[x0.clone()], &mut rng);
    let peak = status_mb("VmHWM:");
    let (mut tc, mut to) = (vec![tc0], vec![o0[0]]);
    for _ in 0..env_or("RUNS", 2usize) {
        let (c, _, o, _, _) = run(&[rng.f128_vec(m - 6)], &mut rng);
        tc.push(c);
        to.push(o[0]);
    }
    let (c, o) = (median(tc), median(to));
    println!(
        r#"{{"kind":"sigma","mode":"{mode}","cells_log":{mu},"m":{m},"unit":{},"threads":{},"commit_s":{c:.6},"open_s":{o:.6},"proof_bytes":{bytes},"commitment_bytes":{cm_bytes},"rss_at_reset_mb":{rss:.1},"peak_commit_mb":{peak_c:.1},"peak_mb":{peak:.1}}}"#,
        unit_log.is_some(),
        rayon::current_num_threads(),
    );
    if !zk {
        return;
    }
    // re-registration: the old commitment opened once more, a fresh one committed and opened, both at one point ρ
    let (mut t_old, mut t_new_c, mut t_new_o) = (vec![], vec![], vec![]);
    for _ in 0..env_or("RUNS", 2usize).max(1) {
        let x = rng.f128_vec(m - 6);
        let (_, _, o, _, _) = run(&[x.clone()], &mut rng);
        t_old.push(o[0]);
        let (c, _, o, _, _) = run(&[x], &mut rng);
        t_new_c.push(c);
        t_new_o.push(o[0]);
    }
    let (a, b, cc) = (median(t_old), median(t_new_o), median(t_new_c));
    println!(
        r#"{{"kind":"rereg","mode":"zk","cells_log":{mu},"m":{m},"unit":{},"threads":{},"commit_new_s":{cc:.6},"open_old_s":{a:.6},"open_new_s":{b:.6},"total_s":{:.6},"bytes":{},"peak_mb":{:.1}}}"#,
        unit_log.is_some(),
        rayon::current_num_threads(),
        a + b + cc,
        cm_bytes + 2 * bytes,
        status_mb("VmHWM:"),
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let a: Vec<String> = args().collect();
    let dir = || PathBuf::from(&a[2]);
    let zk = |i: usize| match a.get(i).map(String::as_str) {
        Some("zk") => true,
        Some("m0") => false,
        other => panic!("expected m0 or zk, got {other:?}"),
    };
    let opt = |i: usize| a.get(i).map(|s| s.parse().unwrap());
    match a.get(1).map(String::as_str) {
        Some("base") => base(&dir(), a[3].parse().unwrap(), zk(4)),
        Some("gate") => gate(&dir(), zk(3)),
        Some("acyc") => acyc(&dir(), zk(3), opt(4)),
        Some("sigma") => sigma(&dir(), zk(3), opt(4)),
        _ => panic!("usage: rec_holo_zk base DIR COPIES m0|zk | gate DIR m0|zk | acyc DIR m0|zk [UNIT_LOG] | sigma DIR m0|zk [UNIT_LOG]"),
    }
}
