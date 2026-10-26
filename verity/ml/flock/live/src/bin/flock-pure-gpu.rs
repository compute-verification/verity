//! flock-pure-gpu: the Flock backend's relation as ONE block R1CS (`flock_live::pure_block`, `verity/flock-pure-block/v1`),
//! proved by the CPU prover or by Flock-CUDA (`--gpu`, feature `gpu`) in a flock-128-r2 live session. The CLI and the
//! output lines are flock-backend's `flock-pure` (so its bench.py drives either); the statement differs (no union, no
//! wiring argument), so it is its own Table 1 configuration (`prover=flock-cuda-block`).
//!
//!   flock-pure-gpu serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256 [--sessions N] [--operator LANE]
//!   flock-pure-gpu prove    --verifier ADDR --instances FILE --netlist FILE [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-pure-gpu selftest --instances FILE --netlist FILE [--gpu]
//!
//! Session: `Hello(Σ)`, `Commit(root_f = Σ, root_B, publics = each block's x and W chunk chaining values and unit 31's
//! y16)`, two verifier points (the public regions' claims), `Link(y = 0, 0)` (no cross-field link: y is pinned to zero),
//! then Flock's coins. The verifier recomputes the frame-v3 roots and the row digests from its instance file.

use std::{
    array::from_fn,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Instant,
};

use flock_core::{
    merkle::HashKind,
    pcs::{PcsParams, ligerito::{LigeritoProfile, embedded_initial_k_or_default}},
    proof::R1csProofLigerito,
    verifier::verify_ligerito_extra,
};
use flock_field::F128;
use flock_live::{chunk::digest_of, pure_block::*, *};
#[allow(unused_imports)]
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::blake3::Compression};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "pure";
const SIGMA_TAG: &[u8] = b"verity/flock-pure-block/sigma/v2";
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn flag(k: &str) -> bool {
    std::env::args().any(|x| x == k)
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-pure-block-v2/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

// frame-v3 (flock-backend's flock-pure.rs, `verity.ml.commitments.merkle`)
fn fv3_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(FRAME);
    h.update((tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize().into()
}

fn uint(x: usize) -> Vec<u8> {
    let b = (x as u64).to_be_bytes();
    let z = b.iter().position(|&c| c != 0).unwrap_or(7);
    b[z..].to_vec()
}

fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 32]>) -> [u8; 32] {
    let n = leaves.len();
    let width = n.next_power_of_two();
    let mut level = leaves;
    level.extend((n..width).map(|r| fv3_hash(b"pad", &[domain_id, &uint(r)])));
    let mut depth = 0;
    while level.len() > 1 {
        level = level.par_chunks(2).enumerate().map(|(i, p)| fv3_hash(b"node", &[domain_id, &uint(depth), &uint(i), &p[0], &p[1]])).collect();
        depth += 1;
    }
    level[0]
}

/// The row digests (x, W) and the three frame-v3 roots, from the values.
fn commit(inst: &Instances) -> ([Vec<[u8; 32]>; 2], [[u8; 32]; 3]) {
    let keys = [flock_live::chunk::row_key(b"x"), flock_live::chunk::row_key(b"w")];
    let sha = inst.header["schemas"]["a"].as_str().unwrap_or("").starts_with("sha256/");
    let digests: [Vec<[u8; 32]>; 2] = from_fn(|r| {
        (0..inst.vus)
            .into_par_iter()
            .map(|v| {
                if sha {
                    let row = inst.row(r, v);
                    let mut h = Sha256::new();
                    h.update(flock_live::pure_block::sha_row_prefix(r, inst.row_bytes));
                    h.update(row);
                    h.finalize().into()
                } else {
                    *blake3::keyed_hash(&keys[r], inst.row(r, v)).as_bytes()
                }
            })
            .collect()
    });
    let hexs = |k: &str, n: &str| unhex(inst.header[k][n].as_str().unwrap());
    let roots: [[u8; 32]; 3] = from_fn(|t| {
        let n = ["a", "b", "y"][t];
        let dom = hexs("domain_ids", n);
        let s = inst.header["schemas"][n].as_str().unwrap().to_string();
        let leaves: Vec<[u8; 32]> = if t < 2 {
            digests[t].par_iter().enumerate().map(|(r, d)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), s.as_bytes(), d])).collect()
        } else {
            let nb = inst.header["y_bytes"].as_u64().unwrap() as usize;
            inst.y.par_iter().enumerate().map(|(r, &y)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), s.as_bytes(), &y.to_be_bytes()[4 - nb..]])).collect()
        };
        fv3_root(&dom, leaves)
    });
    for (t, n) in ["a", "b", "y"].iter().enumerate() {
        assert_eq!(roots[t].to_vec(), hexs("roots", n), "root {n}: the native frame-v3 root differs from the instance file's");
    }
    (digests, roots)
}

fn sigma(st: &PureStmt, inst: &Instances, roots: &[[u8; 32]; 3]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
    h.update(inst.relation.as_bytes());
    h.update([0]);
    for x in [inst.vus, inst.units, inst.row_bytes, st.nbl, st.m] {
        h.update((x as u64).to_le_bytes());
    }
    h.update(st.net.sha256);
    h.update(st.digest);
    h.update(serde_json::to_string(&inst.header["instances"]).unwrap().as_bytes());
    for r in roots {
        h.update(r);
    }
    h.finalize().into()
}

struct PureVerifier {
    st: Arc<PureStmt>,
    inst: Arc<Instances>,
    digests: [Vec<[u8; 32]>; 2],
}

impl StatementVerifier for PureVerifier {
    fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        if table != TABLE {
            return Err(format!("C4: {table} is not in the statement"));
        }
        let (lay, st) = (self.st.lay, &self.st);
        let p = Publics::parse(publics, st.blocks_real, lay.n_cv(), lay.tail().is_some())?;
        if let (Layout::Chunk(n), false) = (lay, st.net.epilogue()) {
            let n = n as usize;
            for v in 0..self.inst.vus {
                if p.acc[n * v + n - 1] != self.inst.out_word(v) {
                    return Err(format!("C4: VU {v}'s committed final accumulator is not its output word"));
                }
            }
        }
        for v in 0..if lay.n_cv() == 0 { 0 } else { self.inst.vus } {
            for r in 0..2 {
                let (bpv, nch) = (lay.blocks_per_vu(), lay.chunks().len());
                // ChunkTail: the partial chunk's chaining value is its block's slot 7 / 23 value (CvMid)
                let cvs: Vec<[u32; 8]> = (bpv * v..bpv * (v + 1))
                    .flat_map(|b| (0..nch).map(move |c| (b, c)))
                    .map(|(b, c)| if Some(b % bpv) == lay.tail() { p.cv[b][2 + r] } else { p.cv[b][r * nch + c] })
                    .collect();
                if digest_of(&self.st.keys[r], &cvs) != self.digests[r][v] {
                    return Err(format!("C4: the chunk chaining values of VU {v} role {r} do not hash to its row digest"));
                }
            }
        }
        Ok(None)
    }

    fn verify(&self, _t: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let st = &self.st;
        let (commitment, proof): (Commitment, R1csProofLigerito) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        let pcs = params(st.m, LigeritoProfile::Fast100);
        if commitment.params != pcs {
            return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        if link.points.len() != 2 || link.y.iter().any(|y| *y != F128::ZERO) {
            return Err("session: two points and y = 0 expected (no cross-field link)".into());
        }
        let p = Publics::parse(link.publics, st.blocks_real, st.lay.n_cv(), st.lay.tail().is_some())?;
        let extra = st.extra_claims(link.points, &self.inst, &p);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn config(st: &PureStmt, sigma: [u8; 32]) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: st.m_pts, points: 2 });
    cfg
}

fn server(st: &Arc<PureStmt>, inst: &Arc<Instances>) -> Server {
    let (digests, roots) = commit(inst);
    Server::new(config(st, sigma(st, inst, &roots)), Box::new(PureVerifier { st: st.clone(), inst: inst.clone(), digests }))
}

type PHook<'a> = &'a mut dyn FnMut(&[u8], Option<&Publics>) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness.
struct Wit {
    comps: Arc<Comps>,
    inputs: Box<dyn Fn(usize, usize) -> [F128; 5] + Sync>,
    flip_z: Option<usize>,
}

fn prove_cpu(st: &PureStmt, w: &Wit, lig: LigeritoProfile, hook: PHook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let t = Instant::now();
    let (mut z, a, b, zl) = st.witness(&w.comps, &*w.inputs);
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    let zl = if w.flip_z.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, K_LOG) } else { zl };
    let tw = t.elapsed().as_secs_f64();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs,
        &params(st.m, lig),
        z,
        a,
        b,
        zl,
        &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(cm.cap.as_flattened(), None).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({"t.witness": tw}))
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &PureStmt, src: PureSrc<'_>, lig: LigeritoProfile, rep: usize, hook: PHook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (proof, commitment, ph) = flock_live::gpu::prove_pure(st, src, &params(st.m, lig), &domain(rep), hook, ch);
    let b = json!({"t.witness": ph[0], "t.encoding_commitment": ph[1], "t.link": ph[2], "t.zerocheck": ph[3], "t.lincheck": ph[4],
                   "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                   "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6]});
    eprintln!("GPUPHASE\t{b}");
    (bincode::serialize(&(commitment, proof)).unwrap(), b)
}

#[cfg(not(feature = "gpu"))]
#[allow(dead_code)]
enum PureSrc<'a> {
    Host { comps: &'a [Compression], inputs: &'a (dyn Fn(usize, usize) -> [F128; 5] + Sync) },
    Device { inst: &'a Instances, cache: i32 },
    HostWitness { z: &'a [F128], a: &'a [F128], b: &'a [F128] },
}

#[cfg(feature = "gpu")]
use flock_live::gpu::PureSrc;

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &PureStmt, _: PureSrc<'_>, _: LigeritoProfile, _: usize, _: PHook<'_>, _: &mut LiveChallenger) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone, Copy, PartialEq)]
enum CommitMode {
    Honest,
    OtherRootB,
    NoY,
}

#[derive(Clone)]
struct Plan {
    gpu: bool,
    commit: CommitMode,
    forge_middle: bool,
    bad_params: Option<usize>,
    key_substituted: bool,
    cv_flip: bool,
    y_flip: bool,
    unit_flip: bool,
    operand_flip: bool,
    acc_in_wrong: bool,
    acc_public_flip: bool,
    rep1_other_witness: bool,
    /// ChunkTail: 0 drops CHUNK_END on the partial chunk's last block, 1 forges its public chaining value, 2 forges a
    /// block of it, 3 flips a bit of its last unit
    tail_tamper: Option<u8>,
    profile: [LigeritoProfile; 2],
    substitute: Option<Vec<Vec<u8>>>,
}

impl Plan {
    fn honest(gpu: bool) -> Self {
        Plan { gpu, commit: CommitMode::Honest, forge_middle: false, bad_params: None, key_substituted: false, cv_flip: false, y_flip: false,
               unit_flip: false, operand_flip: false, acc_in_wrong: false, acc_public_flip: false, rep1_other_witness: false,
               tail_tamper: None, profile: [LigeritoProfile::Fast100; 2], substitute: None }
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// Unit 31's y16 and c_out words of every real block (the prover's publics), from its own evaluation.
fn unit31_words(st: &PureStmt, inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<u32>, Vec<u32>) {
    if st.lay == Layout::Fp8 {
        return (vec![0; st.blocks_real], vec![0; st.blocks_real]);
    }
    let cols = st.net.out_cols.clone();
    (0..st.blocks_real)
        .into_par_iter()
        .map(|b| {
            let r = inputs(b, st.lay.units() - 1);
            let (z, _, _) = st.net.eval64(|bit| {
                let w = r[bit >> 7];
                let j = bit & 127;
                (if j < 64 { w.lo >> j } else { w.hi >> (j - 64) }) & 1
            });
            let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
            (if cols.len() > 1 { word(cols[1], 16) } else { 0 }, word(cols[0], 32))
        })
        .unzip()
}

fn session(st: &Arc<PureStmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let (_, roots) = commit(inst);
    let commit_s = t0.elapsed().as_secs_f64();
    let sig = sigma(st, inst, &roots);
    let hello = config(st, sig).hello();
    let tr = Instant::now();
    // Honest GPU witnesses are built on the device from the rows (publics read back); a tampered witness needs the host.
    let sha = st.lay.is_sha();
    // ChunkTail proves from host-built inputs (the device chain kernel has no per-block flags)
    let device = plan.gpu && st.lay.tail().is_none() && plan.tail_tamper.is_none() && !(plan.forge_middle || plan.bad_params.is_some() || plan.key_substituted || plan.operand_flip
        || plan.acc_in_wrong || plan.rep1_other_witness || plan.unit_flip);
    let (mut comps, cvs) = if device {
        (Comps::B3(Vec::new()), Vec::new())
    } else if sha {
        (Comps::Sha(st.sha_compressions(inst)), vec![Vec::new(); st.blocks_real])
    } else {
        let (c, v) = st.compressions(inst);
        (Comps::B3(c), v)
    };
    match &mut comps {
        Comps::B3(comps) => {
            if plan.forge_middle {
                comps[5].1[2] ^= 1;
            }
            if let Some(n) = st.lay.tail() {
                let tail = n * st.lay.comp_slots();
                match plan.tail_tamper {
                    Some(0) => comps[tail + 7].4 &= !flock_live::chunk::CHUNK_END,
                    Some(2) => comps[tail + 3].1[0] ^= 1,
                    _ => {}
                }
            }
            if let Some(what) = plan.bad_params {
                let r = &mut comps[st.lay.chunks()[0] - 1];
                match what {
                    0 => r.4 &= !flock_live::chunk::CHUNK_END,
                    1 => r.2 = 7,
                    _ => r.3 = if r.3 == 32 { 64 } else { 32 },
                }
            }
            if plan.key_substituted {
                let mut cv = st.keys[1];
                for j in 0..st.lay.chunks()[0] {
                    comps[j].0 = cv;
                    let c = comps[j];
                    cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
                }
            }
        }
        Comps::Sha(comps) => {
            if plan.forge_middle {
                comps[5].1[2] ^= 1;
            }
            if plan.bad_params.is_some() {
                comps[st.lay.per_role() - 1].1[15] ^= 1;
            }
            if plan.key_substituted {
                comps[0].0 = flock_live::pure_block::sha_midstate(1, st.lay.row_bytes());
            }
        }
    }
    let comps = Arc::new(comps);
    let (operand_flip, acc_in_wrong) = (plan.operand_flip, plan.acc_in_wrong);
    let make_inputs = |zero: bool| -> Box<dyn Fn(usize, usize) -> [F128; 5] + Sync> {
        let (st2, inst2, c2) = (st.clone(), inst.clone(), comps.clone());
        Box::new(move |b: usize, u: usize| -> [F128; 5] {
            if zero {
                return [F128::ZERO; 5];
            }
            let mut c_in = st2.c_in(&inst2, b, u);
            if acc_in_wrong && b == 1 && u == 0 {
                c_in ^= 1;
            }
            let mut r = match &*c2 {
                Comps::B3(c) => st2.unit_inputs(c, b, u, c_in),
                Comps::Sha(_) => st2.unit_inputs_rows(&inst2, b, u, c_in),
            };
            if operand_flip && b == 0 && u == 3 {
                r[0].lo ^= 1 << 9;
            }
            r
        })
    };
    let inputs = make_inputs(false);
    let (y16, acc) = if device { (Vec::new(), Vec::new()) } else { unit31_words(st, &*inputs) };
    let mid = if st.lay.tail().is_some() { unit_words_at(st, &*inputs, 15) } else { Vec::new() };
    let tamper = |mut p: Publics| -> Vec<u8> {
        if plan.cv_flip {
            p.cv[1][0][3] ^= 1;
        }
        if plan.y_flip {
            p.y16[0] ^= 1;
        }
        if plan.acc_public_flip {
            p.acc[0] ^= 1;
        }
        if let (Some(1), Some(n)) = (plan.tail_tamper, st.lay.tail()) {
            p.cv[n][2][0] ^= 1;
        }
        p.bytes()
    };
    let host_publics = if device { None } else { Some(Publics { cv: cvs, y16, acc, mid }) };
    let flip_z = match (plan.tail_tamper, st.lay.tail()) {
        (Some(3), Some(n)) => Some((n << st.lay.k_log()) + 700 + ((st.lay.unit_pos0(st.ul) + 15) << st.ul)),
        _ => plan.unit_flip.then_some(700 + ((st.lay.unit_pos0(st.ul) + 5) << st.ul)),
    };
    let wit = Wit { comps: comps.clone(), inputs, flip_z };
    let wit1 = Wit { comps: comps.clone(), inputs: make_inputs(plan.rep1_other_witness), flip_z };
    let rows_s = tr.elapsed().as_secs_f64();
    let mut prove_s = Vec::new();
    let mut buckets = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let mut rtt_ms = 0.0;
    let mut host_witness: Option<(Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)> = None;
    let tp0 = Instant::now();
    let res: Result<(), String> = (|| {
        let th = Instant::now();
        match call(&t, Req::Hello(hello))? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        rtt_ms = th.elapsed().as_secs_f64() * 1e3;
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            let tt = t.clone();
            let mut hook = |root: &[u8], dev_pub: Option<&Publics>| -> Vec<Vec<F128>> {
                if let Some(c) = &cached {
                    return c.clone();
                }
                let publics = tamper(dev_pub.cloned().or_else(|| host_publics.clone()).expect("publics"));
                let mut root_b = root.to_vec();
                if plan.commit == CommitMode::OtherRootB {
                    root_b[0] ^= 1;
                }
                let pts: Vec<Vec<F128>> = match call(&tt, Req::Commit { root_f: sig.to_vec(), roots: vec![(TABLE.into(), root_b)], publics: vec![(TABLE.into(), publics.clone())] }) {
                    Ok(Resp::Coins(c)) if c.len() == 2 * st.m_pts => c.chunks(st.m_pts).map(|r| r.to_vec()).collect(),
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                };
                if plan.commit != CommitMode::NoY {
                    match call(&tt, Req::Link(f128_bytes(&[F128::ZERO, F128::ZERO]))) {
                        Ok(Resp::Ok) => {}
                        other => panic!("LIVE-REFUSED y: {other:?}"),
                    }
                }
                let e = st.extra_points(&pts);
                cached = Some(e.clone());
                e
            };
            let tp = Instant::now();
            let w = if rep == 0 { &wit } else { &wit1 };
            let lig = plan.profile[rep];
            let (p, bk) = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu {
                    let src = if device {
                        PureSrc::Device { inst, cache: if rep == 0 { 1 } else { 2 } }
                    } else if let Comps::B3(c) = &*w.comps {
                        PureSrc::Host { comps: c, inputs: &*w.inputs }
                    } else {
                        // the host witness is built once when both reps prove the same witness
                        if rep == 0 || plan.rep1_other_witness || host_witness.is_none() {
                            host_witness = Some(st.witness(&w.comps, &*w.inputs));
                        }
                        let hw = host_witness.as_ref().unwrap();
                        PureSrc::HostWitness { z: &hw.0, a: &hw.1, b: &hw.2 }
                    };
                    prove_gpu(st, src, lig, rep, &mut hook, &mut lch)
                } else {
                    prove_cpu(st, w, lig, &mut hook, &mut lch)
                }
            }))
            .map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            buckets.push(bk);
            proofs.push(p);
        }
        Ok(())
    })();
    let e2e_s = commit_s + rows_s + tp0.elapsed().as_secs_f64();
    if let Err(e) = res {
        let v = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        return (json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "prover_stopped": e.chars().take(200).collect::<String>()}), proofs);
    }
    let sent = plan.substitute.clone().unwrap_or_else(|| proofs.clone());
    for (rep, p) in sent.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: p.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (ws, wr, calls) = t.lock().unwrap().stats();
    let wait_s = t.lock().unwrap().wait_s();
    let hexs = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": "verity/flock-pure-block/v2", "prover": "flock-cuda-block",
               "gpu": plan.gpu, "relation": inst.relation, "vus": inst.vus, "units": inst.units, "n_blake3": st.lay.comp_slots() * st.blocks_real,
               "n_units": st.lay.units() * st.blocks_real, "dense_m": st.m, "nbl": st.nbl, "e2e_s": e2e_s, "commit_s": commit_s, "rows_s": rows_s,
               "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(), "buckets": buckets, "wait_s": wait_s, "calls": calls,
               "wire_sent": ws, "wire_recv": wr, "rtt_ms": rtt_ms,
               "rtt_method": "the session's Hello request/response time (one round trip, the verifier's work is a string compare)", "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(),
               "roots": {"a": hexs(&roots[0]), "b": hexs(&roots[1]), "y": hexs(&roots[2])}, "sigma": hexs(&sig),
               "netlist_sha256": hexs(&st.net.sha256), "statement_digest": hexs(&st.digest)}),
        proofs,
    )
}

const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("root_b_other_than_bound_root", false, false),
    ("coins_before_y", false, false),
    ("forged_middle_block", false, true),
    ("chunk_end_flag_dropped", false, true),
    ("wrong_counter", false, true),
    ("wrong_block_len", false, true),
    ("key_substituted", false, true),
    ("chunk_value_forged", false, false),
    ("y16_public_forged", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("unit_operand_differs_from_message", false, true),
    ("next_block_acc_in_differs_from_committed_acc", false, true),
    ("committed_acc_differs_from_block_acc_out", false, true),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("tail_end_flag_dropped", false, true),
    ("tail_chunk_value_forged", false, false),
    ("tail_block_forged", false, true),
    ("tail_unit_bit_flipped", false, true),
];

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan::honest(gpu);
    match name {
        "root_b_other_than_bound_root" => Plan { commit: CommitMode::OtherRootB, ..h },
        "coins_before_y" => Plan { commit: CommitMode::NoY, ..h },
        "forged_middle_block" => Plan { forge_middle: true, ..h },
        "chunk_end_flag_dropped" => Plan { bad_params: Some(0), ..h },
        "wrong_counter" => Plan { bad_params: Some(1), ..h },
        "wrong_block_len" => Plan { bad_params: Some(2), ..h },
        "key_substituted" => Plan { key_substituted: true, ..h },
        "chunk_value_forged" => Plan { cv_flip: true, ..h },
        "y16_public_forged" => Plan { y_flip: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, gpu: false, ..h },
        "unit_operand_differs_from_message" => Plan { operand_flip: true, ..h },
        "next_block_acc_in_differs_from_committed_acc" => Plan { acc_in_wrong: true, ..h },
        "committed_acc_differs_from_block_acc_out" => Plan { acc_public_flip: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, ..h },
        "tail_end_flag_dropped" => Plan { tail_tamper: Some(0), ..h },
        "tail_chunk_value_forged" => Plan { tail_tamper: Some(1), ..h },
        "tail_block_forged" => Plan { tail_tamper: Some(2), ..h },
        "tail_unit_bit_flipped" => Plan { tail_tamper: Some(3), gpu: false, ..h },
        "fast_profile_proofs" => Plan { profile: [LigeritoProfile::Fast; 2], ..h },
        _ => h,
    }
}

fn run_case(st: &Arc<PureStmt>, inst: &Arc<Instances>, name: &str, gpu: bool) -> Value {
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan::honest(gpu)).1);
    }
    let s = Arc::new(Mutex::new(server(st, inst)));
    let (v, _) = session(st, inst, shared(InProc(s.clone())), &plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let pass = acc == want.1 && (!want.2 || both);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"]})
}

/// NV1–NV3: an instance file or netlist pairing that admission must refuse (before any coin), each by its own check.
fn admission_negatives(st: &PureStmt, inst: &Instances) -> Vec<Value> {
    use flock_live::pure_block::admit;
    let mut y_tampered = inst.clone();
    y_tampered.y[0] ^= 1;
    let mut relabelled = inst.clone();
    let other = if st.lay.row_schema() == "blake3-keyed/row/v2" { "blake3-keyed/row-nvfp4/v1" } else { "blake3-keyed/row/v2" };
    relabelled.header["schemas"]["a"] = json!(other);
    let other_layout = if st.lay.is_fp4() { Layout::Fp8 } else { Layout::Fp4 };
    // G4: the y leaf re-encoded at the other width (u32 relations: u16, dropping the word's high half; bf16: u32 with
    // y = out = a word past 16 bits), and a bf16 y word past its u16 leaf
    let mut y_other_width = inst.clone();
    let wide = st.net.epilogue();
    y_other_width.header["schemas"]["y"] = json!(if wide { "u32" } else { "u16" });
    y_other_width.header["y_bytes"] = json!(if wide { 4 } else { 2 });
    if wide {
        y_other_width.y[0] |= 1 << 16;
        y_other_width.out[0] = y_other_width.y[0];
    }
    let mut y_past_leaf = inst.clone();
    y_past_leaf.y[0] |= 1 << 16;
    y_past_leaf.out[0] = y_past_leaf.y[0];
    let mut cases: Vec<(&str, &str, Result<(), String>)> = vec![
        ("admission_honest", "", admit(st.lay, &st.net, inst)),
        ("y_word_differs_from_checked_output", "NV1", admit(st.lay, &st.net, &y_tampered)),
        ("netlist_paired_with_other_layout_kind", "NV2", admit(other_layout, &st.net, inst)),
        ("row_schema_relabelled", "NV3", admit(st.lay, &st.net, &relabelled)),
        ("y_leaf_other_width", "NV5", admit(st.lay, &st.net, &y_other_width)),
    ];
    if wide {
        cases.push(("y_word_exceeds_u16_leaf", "NV5", admit(st.lay, &st.net, &y_past_leaf)));
    }
    // CN2: one VU more than m = 35 holds (the admission refuses before reading any row); CN3: the 1024-byte-row layout
    let mut too_many = inst.clone();
    let per = st.lay.blocks_per_vu();
    too_many.vus = (1usize << (flock_live::pure_block::MAX_M - st.lay.k_log())) / per + 1;
    cases.push(("batch_past_m35", "CN2", admit(st.lay, &st.net, &too_many)));
    cases.push(("chunk1_rows", "CN3", admit(Layout::Chunk(1), &st.net, inst)));
    cases.push(("lone_partial_chunk_rows", "CN3", admit(Layout::ChunkTail(0), &st.net, inst)));
    cases
        .into_iter()
        .map(|(name, tag, r)| {
            let pass = match (&r, tag) {
                (Ok(()), "") => true,
                (Err(e), t) => !t.is_empty() && e.starts_with(t),
                _ => false,
            };
            json!({"case": name, "expect": if tag.is_empty() {"admit"} else {"refuse"}, "admitted": r.is_ok(), "why": r.err(), "pass": pass})
        })
        .collect()
}

fn selftest(st: &Arc<PureStmt>, inst: &Arc<Instances>, gpu: bool) {
    let only = arg("--only", "");
    if !only.is_empty() {
        println!("NEG\t{}", run_case(st, inst, &only, gpu));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
        if name.starts_with("tail_") && st.lay.tail().is_none() {
            continue;
        }
        if !st.lay.is_chunked() && matches!(name, "y16_public_forged" | "committed_acc_differs_from_block_acc_out") {
            continue;
        }
        if !st.net.epilogue() && name == "y16_public_forged" {
            continue;
        }
        if (st.lay.is_sha() && matches!(name, "chunk_end_flag_dropped" | "wrong_counter")) || (st.lay.n_cv() == 0 && name == "chunk_value_forged") {
            continue;
        }
        let v = if gpu {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["selftest", "--gpu", "--instances", &arg("--instances", ""), "--netlist", &arg("--netlist", ""), "--only", name])
                .output()
                .expect("spawn a case");
            let text = String::from_utf8_lossy(&out.stdout);
            match text.lines().find(|l| l.starts_with("NEG\t")) {
                Some(l) => serde_json::from_str(&l[4..]).unwrap(),
                None => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    let refused = err.contains("the verifier refused a coin") || err.contains("LIVE-REFUSED");
                    json!({"case": name, "gpu": true, "accepted": false, "pass": !want_accept && refused,
                           "prover_aborted": err.lines().filter(|l| l.contains("REFUSED") || l.contains("refused")).collect::<Vec<_>>().join(" | ")})
                }
            }
        } else {
            run_case(st, inst, name, gpu)
        };
        all &= v["pass"] == true;
        println!("NEG\t{v}");
    }
    for v in admission_negatives(st, inst) {
        all &= v["pass"] == true;
        println!("NEG\t{v}");
    }
    println!("SELFTEST\t{}", json!({"statement": "verity/flock-pure-block/v2", "layout": format!("{:?}", st.lay), "relation": inst.relation, "vus": inst.vus, "m": st.m,
        "gpu": gpu, "all_pass": all}));
}

fn main() {
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let tb = Instant::now();
    let inst = Arc::new(Instances::load(&arg("--instances", "instances.bin")));
    let net = UnitNet::load(&arg("--netlist", "net.txt"));
    let pin = arg("--pin", "");
    if !pin.is_empty() {
        let got: String = net.sha256.iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(got, pin, "the netlist's sha256 is not the pinned lowering");
    }
    assert_eq!(inst.relation, net.relation, "instances and netlist name different relations");
    let lay = Layout::of(inst.row_bytes, inst.units, inst.header["schemas"]["a"].as_str().unwrap_or("").starts_with("sha256/"));
    if let Err(e) = flock_live::pure_block::admit(lay, &net, &inst) {
        eprintln!("REFUSED\t{e}");
        std::process::exit(2);
    }
    let st = Arc::new(PureStmt::new(net, inst.vus, lay));
    st.prepare(&inst);
    eprintln!("statement {} vus={} blocks={} nbl={} m={} built in {:.2}s", inst.relation, inst.vus, st.blocks_real, st.nbl, st.m, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "serve" => serve_loop(move || server(&st, &inst)),
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            let plan = Plan::honest(flag("--gpu"));
            for run in 0..warm + runs {
                let mut tcp = TcpTransport::connect(&arg("--verifier", "127.0.0.1:7200")).unwrap();
                let (rtt, rtt_min, rtt_n) = tcp.ping_rtt(30);
                let t = shared(tcp);
                let (mut v, proofs) = session(&st, &inst, t, &plan);
                v["run"] = json!(run);
                v["warm"] = json!(run < warm);
                v["open_rtt_ms"] = json!(rtt);
                v["open_rtt_min_ms"] = json!(rtt_min);
                v["open_rtt_n"] = json!(rtt_n);
                if !dump.is_empty() {
                    std::fs::create_dir_all(&dump).unwrap();
                    for (r, p) in proofs.iter().enumerate() {
                        std::fs::write(format!("{dump}/run{run:02}-rep{r}.proof"), p).unwrap();
                    }
                }
                println!("LIVE\t{v}");
            }
        }
        "replay" => {
            if pin.is_empty() {
                eprintln!("replay needs --pin (the pinned lowering's sha256)");
                std::process::exit(2);
            }
            let rec: Value = serde_json::from_slice(&std::fs::read(arg("--session", "session.json")).unwrap()).unwrap();
            let proofs: Vec<Vec<u8>> = arg("--proofs", "").split(',').map(|f| std::fs::read(f).unwrap()).collect();
            match replay(&st, &inst, &rec, &proofs) {
                Ok(v) => println!("REPLAY\t{v}"),
                Err(e) => {
                    println!("REPLAY\t{}", json!({"accepted": false, "why": e}));
                    std::process::exit(1);
                }
            }
        }
        _ => eprintln!("usage: flock-pure-gpu serve|prove|selftest|replay --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}

/// The honest device prover's y16 and acc publics (`pure_unit_outputs`): the last unit of every real block, both layouts.
fn last_unit_words(st: &PureStmt, inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync)) -> (Vec<u32>, Vec<u32>) {
    unit_words_at(st, inputs, st.lay.units() - 1).into_iter().unzip()
}

/// Unit u's (y16, c_out) words of every real block, from its own evaluation.
fn unit_words_at(st: &PureStmt, inputs: &(dyn Fn(usize, usize) -> [F128; 5] + Sync), u: usize) -> Vec<(u32, u32)> {
    let cols = st.net.out_cols.clone();
    (0..st.blocks_real)
        .into_par_iter()
        .map(|b| {
            let r = inputs(b, u);
            let (z, _, _) = st.net.eval64(|bit| {
                let w = r[bit >> 7];
                let j = bit & 127;
                (if j < 64 { w.lo >> j } else { w.hi >> (j - 64) }) & 1
            });
            let word = |col: usize, n: usize| (0..n).fold(0u32, |w, t| w | (((z[col * 128 + t] & 1) as u32) << t));
            (if cols.len() > 1 { word(cols[1], 16) } else { 0 }, word(cols[0], 32))
        })
        .collect()
}

fn f128_of_hex(s: &str) -> F128 {
    F128::new(u64::from_str_radix(&s[16..32], 16).unwrap(), u64::from_str_radix(&s[..16], 16).unwrap())
}

/// Re-verify a verifier's recorded session from files with this machine's own statement (instance set, pinned lowering,
/// frame-v3 roots, honest publics): the coins are the recorded ones, so this is not transferable evidence, but every
/// message the proofs carry must hash to what the verifier recorded before issuing each coin, the link exchange must
/// hash to the recorded link_sha256, and the Flock verifier must accept both reps on one committed root.
fn replay(st: &Arc<PureStmt>, inst: &Arc<Instances>, rec: &Value, proofs: &[Vec<u8>]) -> Result<Value, String> {
    let (digests, roots) = commit(inst);
    let sig = sigma(st, inst, &roots);
    if rec["kind"] != "flock-live-session/v1" {
        return Err("not a flock-live session record".into());
    }
    if rec["hello"].as_str() != Some(config(st, sig).hello().as_str()) {
        return Err("the record's session parameters (Σ, statement digest, profile) are not this statement's".into());
    }
    if rec["config"]["require_link"] != true || rec["link_sha256"].is_null() || rec["link_mode"] != "exchange" {
        return Err("F1: the record is not a session whose statement was committed before any coin".into());
    }
    if rec["aborted"].is_string() || rec["verdict"]["accepted"] != true {
        return Err("the live verifier did not accept this session".into());
    }
    let (comps, cvs) = st.compressions(inst);
    let inputs = |b: usize, u: usize| st.unit_inputs(&comps, b, u, st.c_in(inst, b, u));
    let (y16, acc) = last_unit_words(st, &inputs);
    let mid = if st.lay.tail().is_some() { unit_words_at(st, &inputs, 15) } else { Vec::new() };
    let publics = Publics { cv: cvs, y16, acc, mid }.bytes();
    if rec["link"]["publics_sha256"][TABLE].as_str() != Some(hex(&sha256(&publics)).as_str()) {
        return Err("C4: the record's committed publics are not this instance set's chunk values, accumulators and outputs".into());
    }
    let v = PureVerifier { st: st.clone(), inst: inst.clone(), digests };
    v.statement_publics(TABLE, &publics)?;
    let want_proofs: Vec<String> = rec["proof_sha256"].as_array().ok_or("no proofs recorded")?.iter()
        .map(|x| x["sha256"].as_str().unwrap_or_default().to_string()).collect();
    if proofs.len() != 2 || want_proofs.len() != 2 || proofs.iter().zip(&want_proofs).any(|(b, w)| hex(&sha256(b)) != *w) {
        return Err("the proof files are not the recorded proofs".into());
    }
    let points: Vec<Vec<F128>> = rec["link"]["points"].as_array().ok_or("no link points")?.iter()
        .map(|p| p.as_array().unwrap().iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect()).collect();
    let y: Vec<F128> = rec["link"]["y"].as_array().ok_or("no y")?.iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect();
    if points.len() != 2 || points.iter().any(|p| p.len() != st.m_pts) || y != [F128::ZERO, F128::ZERO] {
        return Err("session: two points of the statement's dimension and y = 0 expected".into());
    }
    let mut streams = std::collections::BTreeMap::new();
    let mut stream_roots = std::collections::BTreeMap::new();
    for s in rec["streams"].as_array().ok_or("no streams")? {
        let name = s["stream"].as_str().unwrap().to_string();
        let rounds = s["rounds"].as_array().unwrap().iter().map(|r| RoundRec {
            sha: unhex(r["msg_sha256"].as_str().unwrap()).try_into().unwrap(),
            len: r["msg_len"].as_u64().unwrap() as usize,
            n: r["n"].as_u64().unwrap() as u32,
            coins: r["coins"].as_array().unwrap().iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect(),
        }).collect();
        stream_roots.insert(name.clone(), s["root_sha256"].as_str().map(|x| x.to_string()));
        streams.insert(name, StreamRec { parent: s["parent"].as_str().unwrap().to_string(), label: s["label"].as_str().unwrap().as_bytes().to_vec(),
                                         at: s["at"].as_u64().unwrap(), rounds, cap: None });
    }
    let record = Arc::new(streams);
    let ctx = LinkCtx { points: &points, y: &y, publics: &publics };
    let t = Instant::now();
    let mut caps: Vec<Vec<u8>> = Vec::new();
    for rep in 0..2 {
        let root = rep_stream(TABLE, rep);
        let mut ch = ReplayChallenger::new(record.clone(), &root, &domain(rep));
        let cap = v.verify(TABLE, rep, &proofs[rep], &mut ch, &ctx).map_err(|e| match ch.first_failure() {
            Some(r) => format!("{r} (then the verifier: {e})"),
            None => e,
        })?;
        ch.finish()?;
        if stream_roots.get(&root).cloned().flatten() != Some(hex(&sha256(&cap))) {
            return Err(format!("{root}: the proof's root is not the root the session committed"));
        }
        caps.push(cap);
    }
    if caps[0] != caps[1] {
        return Err("R1: the reps are on different roots".into());
    }
    let root_b = hex(&sha256(&caps[0]));
    if rec["link"]["root_b_sha256"][TABLE].as_str() != Some(root_b.as_str()) || rec["link"]["root_f_sha256"].as_str() != Some(hex(&sha256(&sig)).as_str()) {
        return Err("L1: the committed root_F / root_B are not Σ and the proofs' root".into());
    }
    let mut lb = sig.to_vec();
    lb.extend_from_slice(&caps[0]);
    for p in points.iter().flatten() {
        lb.extend_from_slice(&f128_bytes(&[*p]));
    }
    lb.extend_from_slice(&f128_bytes(&y));
    if rec["link_sha256"].as_str() != Some(hex(&sha256(&lb)).as_str()) {
        return Err("L1: the recorded link_sha256 is not root_F || root_B || points || y".into());
    }
    Ok(json!({"accepted": true, "statement": "verity/flock-pure-block/v2", "relation": inst.relation, "vus": inst.vus,
              "instances": inst.header["instances"], "sigma": hex(&sig), "roots": roots.iter().map(|r| hex(r)).collect::<Vec<_>>(),
              "netlist_sha256": hex(&st.net.sha256), "statement_digest": hex(&st.digest), "root_b_sha256": root_b,
              "link_sha256": rec["link_sha256"], "proof_sha256": want_proofs, "verify_s": t.elapsed().as_secs_f64()}))
}

fn serve_loop(make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    let listen = arg("--listen", "127.0.0.1:7200");
    let out = std::path::PathBuf::from(arg("--out", "sessions"));
    std::fs::create_dir_all(&out).unwrap();
    let l = std::net::TcpListener::bind(&listen).unwrap();
    let max: Option<usize> = arg("--sessions", "").parse().ok();
    println!("SERVING {listen}");
    for (i, conn) in l.incoming().enumerate() {
        let conn = conn.unwrap();
        conn.set_nodelay(true).unwrap();
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut w = BufWriter::new(conn);
        let mut s = make();
        let id = format!("l{:04}-{}", i, std::process::id());
        loop {
            let Ok(b) = read_frame(&mut r) else { break };
            let resp = match Req::decode(&b) {
                Ok(req) => s.handle(req),
                Err(e) => Resp::Refused(e),
            };
            let done = matches!(resp, Resp::Verdict(_));
            if write_frame(&mut w, &resp.encode()).is_err() || done {
                break;
            }
        }
        let rec = s.record_json();
        let d = out.join(&id);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("session.json"), flock_live::record_text::canonical_json(&rec)).unwrap();
        for ((t, r), b) in s.proofs() {
            std::fs::write(d.join(format!("{}.bin", rep_stream(t, *r).replace('/', "."))), b).unwrap();
        }
        let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
        std::fs::write(d.join("verifier.json"), json!({"operator": arg("--operator", ""), "host": host.trim(),
            "runpod_pod_id": std::env::var("RUNPOD_POD_ID").ok(), "research_run": std::env::var("RESEARCH_RUN_ID").ok(),
            "listen": listen}).to_string()).unwrap();
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "why": rec["verdict"]["why"], "aborted": rec["aborted"],
                          "link_mode": rec["link_mode"], "link_sha256": rec["link_sha256"], "require_link": rec["config"]["require_link"],
                          "verify_s": rec["verdict"]["verify_s"], "rounds": rec["verdict"]["rounds"]});
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        writeln!(idx, "{line}").unwrap();
        println!("SESSION\t{line}");
        if Some(i + 1) == max {
            break;
        }
    }
}
