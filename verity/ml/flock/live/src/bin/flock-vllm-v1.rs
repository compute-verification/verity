//! flock-vllm-v1: the Flock backend's relation over operands committed under `vllm-v1` (`flock_live::vllm_block`,
//! `verity/flock-vllm-block/v1`), proved by the CPU prover or by Flock-CUDA (`--gpu`, feature `gpu`) in a flock-128-r2
//! live session. The CLI and output lines are flock-pure-gpu's, so flock-backend's bench.py drives it.
//!
//!   flock-vllm-v1 serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256 [--sessions N] [--operator LANE]
//!   flock-vllm-v1 prove    --verifier ADDR --instances FILE --netlist FILE [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-vllm-v1 selftest --instances FILE --netlist FILE [--gpu] [--only CASE]
//!
//! Session: `Hello(Σ)`, `Commit(root_f = Σ, root_B, publics = each VU's x and W position-leaf digests)`, two verifier
//! points (the public regions' claims), `Link(y = 0, 0)`, then Flock's coins. Σ names the three bound roots and the
//! three port domains (`domain_digest`, PROTOCOL.md §9.1). The verifier derives each port's `StepDomain` from its own
//! instance reference, folds the digests it was sent into the a and b trees (VU v's as leaf v) and the output words'
//! position leaves into the y tree, and accepts `Commit` only if each step root is the bound root it holds.

use std::{
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
use flock_live::{
    pure_block::{Instances, UNIT_LOG, UnitNet},
    vllm_block::{v1, *},
    *,
};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::sha2 as sh};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "vllm";
const STATEMENT: &str = "verity/flock-vllm-block/v1";
const SIGMA_TAG: &[u8] = b"verity/flock-vllm-block/sigma/v1";

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn flag(k: &str) -> bool {
    std::env::args().any(|x| x == k)
}

fn unhex32(s: &str) -> [u8; 32] {
    (0..32).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<_>>().try_into().unwrap()
}

fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-vllm-block-v1/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

/// The verifier's statement: the three ports' domains (derived from its instance reference) and the bound roots it holds.
#[derive(Clone)]
struct Ports {
    shape: v1::Shape,
    domains: [v1::Domain; 3],
    roots: [[u8; 32]; 3],
    y_bytes: usize,
}

/// How a verifier-side negative changes what the verifier holds.
#[derive(Clone, Copy, PartialEq)]
enum Held {
    Honest,
    /// The verifier's domain of port a names another range (a statement over the same root, other positions' meaning).
    OtherDomain,
    /// The verifier holds port a's bare tree root instead of its step-root binding.
    BareTreeRoot,
}

/// The verifier's statement from its own instance file. Every bound root the file names is recomputed natively from the
/// file's values (x rows, W columns, output words: position leaves, the vllm-v1 fold, the step-root binding under the
/// verifier's domain), so a file whose roots are not its values' is refused before any session.
fn ports(inst: &Instances, held: Held) -> Result<Ports, String> {
    let h = &inst.header;
    if h["scheme"] != "vllm-v1" {
        return Err("not a vllm-v1 instance file (verity_flock.instances --scheme vllm-v1)".into());
    }
    let y_bytes = h["y_bytes"].as_u64().unwrap();
    let shape = v1::Shape { k: h["k"].as_u64().unwrap(), word_bits: h["word_bits"].as_u64().unwrap(), y_bytes };
    let mut refs = h["instances"].clone();
    let domains: [v1::Domain; 3] = std::array::from_fn(|t| v1::Domain::of(t, &shape, &refs));
    let mut roots: [[u8; 32]; 3] = std::array::from_fn(|t| unhex32(h["roots"][v1::TREES[t]].as_str().unwrap()));
    for (t, d) in domains.iter().enumerate() {
        let n = v1::TREES[t];
        if v1::hex(&d.digest()) != h["domain_digests"][n].as_str().unwrap_or("") || d.n as usize != inst.vus {
            return Err(format!("port {n}: the instance file's domain is not the verifier's"));
        }
    }
    let (la, lb) = row_leaves(inst);
    let leaves = [la, lb, v1::y_leaves(&inst.y, y_bytes as usize)];
    for (t, l) in leaves.iter().enumerate() {
        if domains[t].bind(&v1::fold(l)) != roots[t] {
            return Err(format!("VL1: port {}: the instance file's bound root is not the step root of its own values", v1::TREES[t]));
        }
    }
    let mut domains = domains;
    match held {
        Held::Honest => {}
        Held::OtherDomain => {
            refs["range"][0] = json!(refs["range"][0].as_u64().unwrap() + 1);
            refs["range"][1] = json!(refs["range"][1].as_u64().unwrap() + 1);
            domains[0] = v1::Domain::of(0, &shape, &refs);
        }
        Held::BareTreeRoot => roots[0] = v1::fold(&leaves[0]),
    }
    Ok(Ports { shape, domains, roots, y_bytes: y_bytes as usize })
}

/// The position leaves of the x rows and the W columns.
fn row_leaves(inst: &Instances) -> (Vec<[u8; 32]>, Vec<[u8; 32]>) {
    let f = |r: usize| (0..inst.vus).into_par_iter().map(|v| v1::pos_leaf(inst.row(r, v))).collect();
    (f(0), f(1))
}

fn sigma(st: &VllmStmt, inst: &Instances, p: &Ports) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
    h.update(inst.relation.as_bytes());
    h.update([0]);
    for x in [inst.vus, inst.units, inst.row_bytes, st.nbl, st.m, p.y_bytes] {
        h.update((x as u64).to_le_bytes());
    }
    h.update(st.net.sha256);
    h.update(st.digest);
    h.update(serde_json::to_string(&inst.header["instances"]).unwrap().as_bytes());
    for (d, r) in p.domains.iter().zip(&p.roots) {
        h.update(d.digest());
        h.update(r);
    }
    let _ = p.shape;
    h.finalize().into()
}

struct VllmVerifier {
    st: Arc<VllmStmt>,
    inst: Arc<Instances>,
    ports: Ports,
}

impl StatementVerifier for VllmVerifier {
    fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        if table != TABLE {
            return Err(format!("C4: {table} is not in the statement"));
        }
        let p = Publics::parse(publics, self.st.vus)?;
        for r in 0..2 {
            let leaves: Vec<[u8; 32]> = p.digest.iter().map(|d| digest_bytes(&d[r])).collect();
            if self.ports.domains[r].bind(&v1::fold(&leaves)) != self.ports.roots[r] {
                return Err(format!("C4/vllm-v1: the {} digests' tree, bound under the verifier's domain, is not the bound root it holds",
                                   ["x", "W"][r]));
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
        let p = Publics::parse(link.publics, st.vus)?;
        let extra = st.extra_claims(link.points, &self.inst, &p);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn config(st: &VllmStmt, sigma: [u8; 32]) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: st.m_pts, points: 2 });
    cfg
}

fn server(st: &Arc<VllmStmt>, inst: &Arc<Instances>, held: Held) -> Result<Server, String> {
    Ok(server_with(st, inst, ports(inst, held)?))
}

fn server_with(st: &Arc<VllmStmt>, inst: &Arc<Instances>, p: Ports) -> Server {
    Server::new(config(st, sigma(st, inst, &p)), Box::new(VllmVerifier { st: st.clone(), inst: inst.clone(), ports: p }))
}

type PHook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness.
struct Wit {
    comps: Vec<sh::Compression>,
    inputs: Box<dyn Fn(usize, usize) -> [F128; 5] + Sync>,
    flip_z: Option<usize>,
}

fn build(st: &VllmStmt, w: &Wit) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let (mut z, a, b, zl) = st.witness(&w.comps, &*w.inputs);
    match w.flip_z {
        None => (z, a, b, zl),
        Some(bit) => {
            z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
            let zl = flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.lay.k_log());
            (z, a, b, zl)
        }
    }
}

type Built = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

fn prove_cpu(st: &VllmStmt, w: &Built, lig: LigeritoProfile, hook: PHook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (z, a, b, zl) = w.clone();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs,
        &params(st.m, lig),
        z,
        a,
        b,
        zl,
        &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(cm.cap.as_flattened()).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({}))
}

/// What a GPU rep proves: the host's whole packed witness, or the compression and unit inputs the device witnesses.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
enum GpuWit<'a> {
    Host(&'a Built),
    Device(&'a [sh::Compression], &'a [F128]),
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &VllmStmt, w: GpuWit<'_>, lig: LigeritoProfile, rep: usize, hook: PHook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    use flock_live::gpu::VllmSrc;
    let src = match w {
        GpuWit::Host(b) => VllmSrc::Host { z: &b.0, a: &b.1, b: &b.2 },
        GpuWit::Device(comps, units) => VllmSrc::Device { comps, units },
    };
    let (proof, commitment, ph) = flock_live::gpu::prove_vllm(st, src, &params(st.m, lig), &domain(rep), hook, ch);
    let bk = json!({"t.witness": ph[0], "t.encoding_commitment": ph[1], "t.link": ph[2], "t.zerocheck": ph[3],
                    "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                    "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6]});
    eprintln!("GPUPHASE\t{bk}");
    (bincode::serialize(&(commitment, proof)).unwrap(), bk)
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &VllmStmt, _: GpuWit<'_>, _: LigeritoProfile, _: usize, _: PHook<'_>, _: &mut LiveChallenger) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone, Copy, PartialEq)]
enum CommitMode {
    Honest,
    OtherRootB,
    NoY,
}

/// Where a dishonest prover departs from the honest one (every field off = honest).
#[derive(Clone, Default)]
struct Plan {
    gpu: bool,
    held: Option<Held>,
    /// what the prover's Σ names when it differs from what the verifier holds
    prover_held: Option<Held>,
    commit: Option<CommitMode>,
    /// flip a message bit of VU 0's x slot 5, keeping the honest chain after it
    forge_middle: bool,
    /// VU 0's x row changed (byte 700) and rehashed, the honest digest published
    row_forged_honest_digest: bool,
    /// VU 0's x row changed and rehashed, its own digest published
    row_forged_own_digest: bool,
    /// VU 0's x leaf hashed with another value length in the prefix and padding
    length_prefix: Option<usize>,
    /// VU 0's x chain started from another IV word
    wrong_iv: bool,
    /// the published digests: x/W of every VU swapped, or VU 0 and 1 swapped, or one bit of VU 0's x flipped
    digests_swapped_roles: bool,
    digests_swapped_positions: bool,
    digest_flip: bool,
    unit_flip: bool,
    operand_flip: bool,
    acc_in_wrong: bool,
    rep1_other_witness: bool,
    fast_profile: bool,
    substitute: Option<Vec<Vec<u8>>>,
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// Rehash VU 0's role-r slots from `msg` (a full padded message), from `iv`.
fn rechain(comps: &mut [sh::Compression], st: &VllmStmt, role: usize, msg: &[u8], iv: [u32; 8]) -> [u32; 8] {
    let base = role * st.lay.per_role();
    let mut h = iv;
    for (j, blk) in msg.chunks(64).enumerate().take(st.lay.per_role()) {
        let m: [u32; 16] = std::array::from_fn(|i| u32::from_be_bytes(blk[4 * i..4 * i + 4].try_into().unwrap()));
        comps[base + j] = (h, m);
        h = sh::sha256_compress(&h, &m);
    }
    h
}

fn session(st: &Arc<VllmStmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    // A verifier-side negative's prover names the verifier's statement (Σ agrees), so only the root check can refuse it.
    let held = ports(inst, plan.prover_held.or(plan.held).unwrap_or(Held::Honest)).expect("the prover's instance file");
    let commit_s = t0.elapsed().as_secs_f64();
    let sig = sigma(st, inst, &held);
    let hello = config(st, sig).hello();
    let tr = Instant::now();
    let (mut comps, mut digests) = st.compressions(inst);
    let row0: Vec<u8> = inst.row(0, 0).to_vec();
    if plan.forge_middle {
        comps[5].1[3] ^= 1 << 9;
    }
    if plan.row_forged_honest_digest || plan.row_forged_own_digest {
        let mut row = row0.clone();
        row[700] ^= 0x10;
        let d = rechain(&mut comps, st, 0, &pos_leaf_message(&row), sh::SHA256_IV);
        if plan.row_forged_own_digest {
            digests[0][0] = d;
        }
    }
    if let Some(n) = plan.length_prefix {
        let mut msg = pos_leaf_message(&row0[..n.min(row0.len())]);
        msg.resize(64 * st.lay.per_role(), 0);
        rechain(&mut comps, st, 0, &msg, sh::SHA256_IV);
    }
    if plan.wrong_iv {
        let mut iv = sh::SHA256_IV;
        iv[3] ^= 1;
        rechain(&mut comps, st, 0, &pos_leaf_message(&row0), iv);
    }
    let (operand_flip, acc_in_wrong) = (plan.operand_flip, plan.acc_in_wrong);
    let mk_inputs = |comps: Vec<sh::Compression>| -> Box<dyn Fn(usize, usize) -> [F128; 5] + Sync> {
        let (st2, inst2) = (st.clone(), inst.clone());
        Box::new(move |b: usize, u: usize| -> [F128; 5] {
            let mut c_in = st2.c_in(&inst2, b, u);
            if acc_in_wrong && b == 1 && u == 1 {
                c_in ^= 1;
            }
            let mut r = st2.unit_inputs(&comps, b, u, c_in);
            if operand_flip && b == 0 && u == 3 {
                r[0].lo ^= 1 << 9;
            }
            r
        })
    };
    let mut publics = Publics { digest: digests };
    if plan.digests_swapped_roles {
        for d in publics.digest.iter_mut() {
            d.swap(0, 1);
        }
    }
    if plan.digests_swapped_positions {
        publics.digest.swap(0, 1);
    }
    if plan.digest_flip {
        publics.digest[0][0][2] ^= 1 << 5;
    }
    let publics = publics.bytes();
    let flip_z = plan.unit_flip.then_some(700 + ((st.lay.unit_pos0() + 5) << UNIT_LOG));
    let wit = Wit { comps: comps.clone(), inputs: mk_inputs(comps.clone()), flip_z };
    let wit1 = if plan.rep1_other_witness {
        Wit { comps: comps.clone(), inputs: Box::new(|_, _| [F128::ZERO; 5]), flip_z: None }
    } else {
        Wit { comps: comps.clone(), inputs: mk_inputs(comps), flip_z }
    };
    // CPU (and a GPU witness with a flipped z bit): the host builds the packed witness once, and rep 1 proves the same one
    // unless the plan gives it another. GPU otherwise: the host sends the compressions and the units' input words.
    let tw = Instant::now();
    let host = !plan.gpu || plan.unit_flip;
    let empty: Built = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let built0 = if host { build(st, &wit) } else { empty.clone() };
    let built1 = (host && plan.rep1_other_witness).then(|| build(st, &wit1));
    let unit_words = |w: &Wit| -> Vec<F128> {
        let units = st.lay.units;
        (0..st.blocks() * units).into_par_iter().flat_map_iter(|i| (w.inputs)(i / units, i % units)).collect()
    };
    let units0 = if host { Vec::new() } else { unit_words(&wit) };
    let units1 = (!host && plan.rep1_other_witness).then(|| unit_words(&wit1));
    let host_witness_s = tw.elapsed().as_secs_f64();
    let rows_s = tr.elapsed().as_secs_f64();
    let mut prove_s = Vec::new();
    let mut buckets = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let mut rtt_ms = 0.0;
    let commit_mode = plan.commit.unwrap_or(CommitMode::Honest);
    let lig = if plan.fast_profile { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
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
            let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                if let Some(c) = &cached {
                    return c.clone();
                }
                let mut root_b = root.to_vec();
                if commit_mode == CommitMode::OtherRootB {
                    root_b[0] ^= 1;
                }
                let pts: Vec<Vec<F128>> = match call(&tt, Req::Commit { root_f: sig.to_vec(), roots: vec![(TABLE.into(), root_b)], publics: vec![(TABLE.into(), publics.clone())] }) {
                    Ok(Resp::Coins(c)) if c.len() == 2 * st.m_pts => c.chunks(st.m_pts).map(|r| r.to_vec()).collect(),
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                };
                if commit_mode != CommitMode::NoY {
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
            let w = if rep == 0 { &built0 } else { built1.as_ref().unwrap_or(&built0) };
            let gw = if host {
                GpuWit::Host(w)
            } else {
                GpuWit::Device(if rep == 0 { &wit.comps } else { &wit1.comps }, if rep == 0 { &units0 } else { units1.as_ref().unwrap_or(&units0) })
            };
            let (p, bk) = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu { prove_gpu(st, gw, lig, rep, &mut hook, &mut lch) } else { prove_cpu(st, w, lig, &mut hook, &mut lch) }
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
    let hexs = v1::hex;
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": STATEMENT, "scheme": "vllm-v1",
               "prover": if plan.gpu { "flock-cuda-block" } else { "flock-cpu-block" }, "gpu": plan.gpu, "relation": inst.relation,
               "vus": inst.vus, "units": inst.units, "n_sha256": st.lay.comp_slots() * st.vus, "n_units": st.lay.units * st.vus,
               "dense_m": st.m, "nbl": st.nbl, "e2e_s": e2e_s, "commit_s": commit_s, "rows_s": rows_s, "host_witness_s": host_witness_s,
               "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(), "buckets": buckets, "wait_s": wait_s, "calls": calls,
               "wire_sent": ws, "wire_recv": wr, "rtt_ms": rtt_ms,
               "rtt_method": "the session's Hello request/response time (one round trip, the verifier's work is a string compare)",
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(),
               "roots": {"a": hexs(&held.roots[0]), "b": hexs(&held.roots[1]), "y": hexs(&held.roots[2])},
               "domain_digests": {"a": hexs(&held.domains[0].digest()), "b": hexs(&held.domains[1].digest()), "y": hexs(&held.domains[2].digest())},
               "sigma": hexs(&sig), "netlist_sha256": hexs(&st.net.sha256), "statement_digest": hexs(&st.digest)}),
        proofs,
    )
}

/// (case, accept?, both reps must reject)
const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("root_b_other_than_bound_root", false, false),
    ("coins_before_y", false, false),
    ("verifier_domain_other_range", false, false),
    ("bare_tree_root", false, false),
    ("prover_names_other_domain", false, false),
    ("verifier_instance_file_root_tampered", false, false),
    ("digest_forged", false, false),
    ("digests_swapped_x_w", false, false),
    ("digests_swapped_positions", false, false),
    ("row_forged_own_digest", false, false),
    ("row_forged_honest_digest", false, true),
    ("forged_middle_block", false, true),
    ("value_length_prefix_other", false, true),
    ("wrong_iv", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("unit_operand_differs_from_message", false, true),
    ("acc_in_differs_from_previous_c_out", false, true),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
];

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan { gpu, ..Plan::default() };
    match name {
        "root_b_other_than_bound_root" => Plan { commit: Some(CommitMode::OtherRootB), ..h },
        "coins_before_y" => Plan { commit: Some(CommitMode::NoY), ..h },
        "verifier_domain_other_range" => Plan { held: Some(Held::OtherDomain), ..h },
        "bare_tree_root" => Plan { held: Some(Held::BareTreeRoot), ..h },
        "prover_names_other_domain" => Plan { prover_held: Some(Held::OtherDomain), ..h },
        "digest_forged" => Plan { digest_flip: true, ..h },
        "digests_swapped_x_w" => Plan { digests_swapped_roles: true, ..h },
        "digests_swapped_positions" => Plan { digests_swapped_positions: true, ..h },
        "row_forged_own_digest" => Plan { row_forged_own_digest: true, ..h },
        "row_forged_honest_digest" => Plan { row_forged_honest_digest: true, ..h },
        "forged_middle_block" => Plan { forge_middle: true, ..h },
        "value_length_prefix_other" => Plan { length_prefix: Some(1535), ..h },
        "wrong_iv" => Plan { wrong_iv: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..h },
        "unit_operand_differs_from_message" => Plan { operand_flip: true, ..h },
        "acc_in_differs_from_previous_c_out" => Plan { acc_in_wrong: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { fast_profile: true, ..h },
        _ => h,
    }
}

fn run_case(st: &Arc<VllmStmt>, inst: &Arc<Instances>, name: &str, gpu: bool) -> Value {
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst, Held::Honest).unwrap()));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan { gpu, ..Plan::default() }).1);
    }
    // VL1: the verifier's own instance file names another a root than its x rows' (the prover's file is honest)
    let vinst = if name == "verifier_instance_file_root_tampered" {
        let mut t = (**inst).clone();
        let a = t.header["roots"]["a"].as_str().unwrap().to_string();
        t.header["roots"]["a"] = json!(format!("{}{}", if a.starts_with('0') { "1" } else { "0" }, &a[1..]));
        Arc::new(t)
    } else {
        inst.clone()
    };
    let s = match server(st, &vinst, plan.held.unwrap_or(Held::Honest)) {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            return json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": false,
                          "pass": !want.1, "both_reps_required": want.2, "why": e, "verifier_refused_instance_file": true});
        }
    };
    let (v, _) = session(st, inst, shared(InProc(s.clone())), &plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let pass = acc == want.1 && (!want.2 || both);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"]})
}

fn selftest(st: &Arc<VllmStmt>, inst: &Arc<Instances>, gpu: bool) {
    let only = arg("--only", "");
    if !only.is_empty() {
        println!("NEG\t{}", run_case(st, inst, &only, gpu));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
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
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "scheme": "vllm-v1", "relation": inst.relation, "vus": inst.vus, "m": st.m,
        "k_log": st.lay.k_log(), "sha256_slots": st.lay.comp_slots(), "gpu": gpu, "cases": CASES.len(), "all_pass": all}));
}

fn main() {
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let tb = Instant::now();
    let inst = Arc::new(Instances::load(&arg("--instances", "instances.bin")));
    let net = UnitNet::load(&arg("--netlist", "net.txt"));
    let pin = arg("--pin", "");
    if !pin.is_empty() {
        assert_eq!(v1::hex(&net.sha256), pin, "the netlist's sha256 is not the pinned lowering");
    }
    assert_eq!(inst.relation, net.relation, "instances and netlist name different relations");
    let lay = Layout::of(inst.row_bytes, inst.units);
    let st = Arc::new(VllmStmt::new(net, inst.vus, lay));
    eprintln!("statement {} vus={} nbl={} m={} k_log={} built in {:.2}s", inst.relation, inst.vus, st.nbl, st.m, lay.k_log(), tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "serve" => {
            let p = ports(&inst, Held::Honest).unwrap_or_else(|e| panic!("the verifier refuses its instance file: {e}"));
            serve_loop(move || server_with(&st, &inst, p.clone()))
        }
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            let plan = Plan { gpu: flag("--gpu"), ..Plan::default() };
            for run in 0..warm + runs {
                let t = shared(TcpTransport::connect(&arg("--verifier", "127.0.0.1:7200")).unwrap());
                let (mut v, proofs) = session(&st, &inst, t, &plan);
                v["run"] = json!(run);
                v["warm"] = json!(run < warm);
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
        _ => eprintln!("usage: flock-vllm-v1 serve|prove|selftest|replay --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}

fn f128_of_hex(s: &str) -> F128 {
    F128::new(u64::from_str_radix(&s[16..32], 16).unwrap(), u64::from_str_radix(&s[..16], 16).unwrap())
}

/// Re-verify a verifier's recorded session from files with this machine's own statement (input set with its vllm-v1
/// step roots and port domains, pinned lowering, honest position-leaf digests): the coins are the recorded ones, so this
/// is not transferable evidence, but every message the proofs carry must hash to what the verifier recorded before each
/// coin, the link exchange must hash to the recorded link_sha256, and Flock must accept both reps on one committed root.
fn replay(st: &Arc<VllmStmt>, inst: &Arc<Instances>, rec: &Value, proofs: &[Vec<u8>]) -> Result<Value, String> {
    let p = ports(inst, Held::Honest)?;
    let sig = sigma(st, inst, &p);
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
    let (_, digests) = st.compressions(inst);
    let publics = Publics { digest: digests }.bytes();
    if rec["link"]["publics_sha256"][TABLE].as_str() != Some(hex(&sha256(&publics)).as_str()) {
        return Err("C4: the record's committed publics are not this instance set's position-leaf digests".into());
    }
    let v = VllmVerifier { st: st.clone(), inst: inst.clone(), ports: p.clone() };
    v.statement_publics(TABLE, &publics)?;
    let want_proofs: Vec<String> = rec["proof_sha256"].as_array().ok_or("no proofs recorded")?.iter()
        .map(|x| x["sha256"].as_str().unwrap_or_default().to_string()).collect();
    if proofs.len() != 2 || want_proofs.len() != 2 || proofs.iter().zip(&want_proofs).any(|(b, w)| hex(&sha256(b)) != *w) {
        return Err("the proof files are not the recorded proofs".into());
    }
    let points: Vec<Vec<F128>> = rec["link"]["points"].as_array().ok_or("no link points")?.iter()
        .map(|q| q.as_array().unwrap().iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect()).collect();
    let y: Vec<F128> = rec["link"]["y"].as_array().ok_or("no y")?.iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect();
    if points.len() != 2 || points.iter().any(|q| q.len() != st.m_pts) || y != [F128::ZERO, F128::ZERO] {
        return Err("session: two points of the statement's dimension and y = 0 expected".into());
    }
    let mut streams = std::collections::BTreeMap::new();
    let mut stream_roots = std::collections::BTreeMap::new();
    for s in rec["streams"].as_array().ok_or("no streams")? {
        let name = s["stream"].as_str().unwrap().to_string();
        let rounds = s["rounds"].as_array().unwrap().iter().map(|r| RoundRec {
            sha: unhex32(r["msg_sha256"].as_str().unwrap()),
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
    for q in points.iter().flatten() {
        lb.extend_from_slice(&f128_bytes(&[*q]));
    }
    lb.extend_from_slice(&f128_bytes(&y));
    if rec["link_sha256"].as_str() != Some(hex(&sha256(&lb)).as_str()) {
        return Err("L1: the recorded link_sha256 is not root_F || root_B || points || y".into());
    }
    Ok(json!({"accepted": true, "statement": STATEMENT, "relation": inst.relation, "vus": inst.vus,
              "instances": inst.header["instances"], "sigma": hex(&sig), "roots": p.roots.iter().map(|r| hex(r)).collect::<Vec<_>>(),
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
            "listen": listen, "statement": STATEMENT, "scheme": "vllm-v1"}).to_string()).unwrap();
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
