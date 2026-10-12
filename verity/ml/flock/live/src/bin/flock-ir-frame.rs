//! flock-ir-frame: an IR template's units bound to their frame-v3 rows (`flock_live::ir_frame`, `verity/flock-ir-frame/v3`),
//! proved by the CPU prover or by Flock-CUDA's host-witness mode (`--gpu`, feature `gpu`) in a flock-128-r2 live session.
//! Lane flock-ir-lowering; the session protocol is flock-gpu-link's flock-pure-gpu.
//!
//!   flock-ir-frame serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256 [--tables DIR] [--sessions N] [--operator LANE]
//!   flock-ir-frame prove    --verifier ADDR --instances FILE --netlist FILE [--tables DIR] [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-ir-frame selftest --instances FILE --netlist FILE [--tables DIR] [--gpu] [--only CASE]
//!   flock-ir-frame replay   --session session.json --proofs rep0,rep1 --instances FILE --netlist FILE --pin SHA256 [--tables DIR]
//!   any of them with `--class MANIFEST` (a key-count class: `--pin` is the manifest's sha256; see `check_class`)
//!
//! Before anything the verifier checks its own instance file (`verity_flock.ir_frame.stage`, never a producer's): the unit
//! netlist's order, the block table against the plan, the frame-v3 roots recomputed from its row digests and output words,
//! the wiring, output mapping, block assignment and row key against the netlist's pinned leaf maps (IR6), and for a cut
//! template its cut words against the netlist's pinned tail (on the sha256-pinned MUFU tables under
//! `--tables`). Session: `Hello(Σ)`, `Commit(root_f = Σ, root_B, publics = every real block's run output values)` (the
//! verifier chains them into its regions and folds each row's chunk values into its digest), two verifier points (the
//! regions' claims), `Link(y = 0, 0)`, Flock's coins.

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
    ir_block::{IrUnitNet, check_order},
    ir_frame::*,
    ir_tail::{Tables, tables_needed},
    *,
};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::blake3::Compression};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "irf";
const SIGMA_TAG: &[u8] = b"verity/flock-ir-frame/sigma/v3";
const STATEMENT: &str = "verity/flock-ir-frame/v3";

static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
/// A class-pinned statement's pin (`--class`): Σ binds it.
static CLASS_PIN: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn flag(k: &str) -> bool {
    std::env::args().any(|x| x == k)
}

fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-ir-frame-v3/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

fn sigma(st: &FrameStmt, inst: &FrameInstances) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
    if let Some(c) = CLASS_PIN.get() {
        h.update(b"class");
        h.update(c);
    }
    h.update(inst.unit.as_bytes());
    h.update([0]);
    h.update(st.net.sha256);
    h.update(st.digest);
    h.update(inst.public_sha256);
    h.finalize().into()
}

struct FrameVerifier {
    st: Arc<FrameStmt>,
    inst: Arc<FrameInstances>,
}

impl StatementVerifier for FrameVerifier {
    fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        if table != TABLE {
            return Err(format!("C4: {table} is not in the statement"));
        }
        let cvs = self.st.parse_publics(publics)?;
        self.st.check_digests(&self.inst, &cvs)?;
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
        let cvs = st.parse_publics(link.publics)?;
        let extra = st.extra_claims(link.points, &self.inst, &cvs);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn config(st: &FrameStmt, sigma: [u8; 32]) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: st.m_pts, points: 2 });
    cfg
}

fn server(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>) -> Server {
    Server::new(config(st, sigma(st, inst)), Box::new(FrameVerifier { st: st.clone(), inst: inst.clone() }))
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness.
struct Wit {
    comps: Vec<Compression>,
    inputs: Box<dyn Fn(usize, usize) -> Vec<F128> + Sync>,
    flip_z: Option<usize>,
}

fn witness(st: &FrameStmt, w: &Wit) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>, f64) {
    let t = Instant::now();
    let (mut z, a, b, zl) = st.witness(&w.comps, &*w.inputs);
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    let zl = if w.flip_z.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.lay.k_log) } else { zl };
    (z, a, b, zl, t.elapsed().as_secs_f64())
}

fn prove_cpu(st: &FrameStmt, w: &Wit, lig: LigeritoProfile, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (z, a, b, zl, tw) = witness(st, w);
    let tp = Instant::now();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs, &params(st.m, lig), z, a, b, zl, &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(cm.cap.as_flattened()).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({"t.witness": tw, "t.prove": tp.elapsed().as_secs_f64()}))
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &FrameStmt, w: &Wit, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (z, a, b, _zl, tw) = witness(st, w);
    let s = flock_live::gpu::SlotStmt { digest: st.digest, k_log: st.lay.k_log, pin: PIN, da: &st.circuit.da, db: &st.circuit.db,
                                        comp_slots: st.lay.comp_slots, units: st.lay.units, unit_log: st.lay.unit_log, unit_pos0: st.lay.unit_pos0,
                                        net: &st.net, regions: st.regions.len(), blocks: st.blocks() };
    let (proof, commitment, ph) = flock_live::gpu::prove_slots(&s, &z, &a, &b, &params(st.m, lig), &domain(rep), hook, ch);
    let bk = json!({"t.witness_host": tw, "t.witness": ph[0], "t.encoding_commitment": ph[1], "t.link": ph[2], "t.zerocheck": ph[3],
                    "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                    "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6]});
    eprintln!("GPUPHASE\t{bk}");
    (bincode::serialize(&(commitment, proof)).unwrap(), bk)
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &FrameStmt, _: &Wit, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone, Default)]
struct Plan {
    gpu: bool,
    no_y: bool,
    message_flip: bool,
    input_flip: bool,
    unit_flip: bool,
    cv_flip: bool,
    key_substituted: bool,
    flags_forged: bool,
    run_input_forged: bool,
    cut_flip: bool,
    rep1_other_witness: bool,
    fast: bool,
    substitute: Option<Vec<Vec<u8>>>,
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn wit(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, comps: &[Compression], plan: &Plan, zero: bool) -> Wit {
    let (st2, i2, c2) = (st.clone(), inst.clone(), comps.to_vec());
    let (input_flip, cut_flip) = (plan.input_flip, plan.cut_flip);
    let cut_col = st.cm.unit_in.first().and_then(|p| p.first()).map(|p| p.0);
    Wit {
        comps: comps.to_vec(),
        inputs: Box::new(move |b, u| {
            let mut r = st2.unit_inputs(&i2, &c2, b, u);
            if zero {
                r.iter_mut().for_each(|w| *w = F128::ZERO);
            }
            if input_flip && b == 0 && u == st2.lay.units.min(4) - 1 {
                r[0].lo ^= 1 << 2;
            }
            if let (true, 0, 0, Some(c)) = (cut_flip, b, u, cut_col) {
                let t = c % 128 + 3;
                if t < 64 { r[c / 128].lo ^= 1 << t } else { r[c / 128].hi ^= 1 << (t - 64) }
            }
            r
        }),
        flip_z: plan.unit_flip.then_some(((st.lay.unit_pos0 + (st.lay.units - 1).min(5)) << st.lay.unit_log) + st.net.in_words * 128 + 40),
    }
}

fn session(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let sig = sigma(st, inst);
    let hello = config(st, sig).hello();
    let (mut comps, mut cvs) = st.compressions(inst);
    let rechain = |comps: &mut Vec<Compression>, at: usize, nb: usize| -> [u32; 8] {
        let mut cv = comps[at].0;
        for c in &mut comps[at..at + nb] {
            c.0 = cv;
            cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
        }
        cv
    };
    if plan.flags_forged {
        comps[0].4 ^= 1;
        rechain(&mut comps, 0, st.lay.nb);
    }
    if plan.run_input_forged {
        let (b, q) = inst.blocks.iter().enumerate()
            .find_map(|(b, bl)| bl.0.iter().position(|c| c.is_some_and(|x| x.3 > 0)).map(|q| (b, q)))
            .expect("a run past a chunk's first");
        let at = b * st.lay.comp_slots + q * st.lay.nb;
        comps[at].0[0] ^= 1;
        rechain(&mut comps, at, st.lay.nb);
    }
    if plan.message_flip {
        comps[1].1[3] ^= 1 << 7;
        let (mut cv, lay) = (comps[0].0, &st.lay);
        for j in 0..lay.nb {
            let c = &mut comps[j];
            c.0 = cv;
            cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
        }
        cvs[0][0] = cv;
    }
    if plan.key_substituted {
        comps[0].0[0] ^= 1;
    }
    if plan.cv_flip {
        cvs[0][0][2] ^= 1;
    }
    let rows_s = t0.elapsed().as_secs_f64();
    let publics = FrameStmt::publics_bytes(&cvs);
    let w0 = wit(st, inst, &comps, plan, false);
    let w1 = wit(st, inst, &comps, plan, plan.rep1_other_witness);
    let mut prove_s = Vec::new();
    let mut buckets = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let tp0 = Instant::now();
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello))? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            let tt = t.clone();
            let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                if let Some(c) = &cached {
                    return c.clone();
                }
                let pts: Vec<Vec<F128>> = match call(&tt, Req::Commit { root_f: sig.to_vec(), roots: vec![(TABLE.into(), root.to_vec())],
                                                                         publics: vec![(TABLE.into(), publics.clone())] }) {
                    Ok(Resp::Coins(c)) if c.len() == 2 * st.m_pts => c.chunks(st.m_pts).map(|r| r.to_vec()).collect(),
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                };
                if !plan.no_y {
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
            let w = if rep == 0 { &w0 } else { &w1 };
            let lig = if plan.fast { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
            let (p, bk) = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu { prove_gpu(st, w, lig, rep, &mut hook, &mut lch) } else { prove_cpu(st, w, lig, &mut hook, &mut lch) }
            })).map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            buckets.push(bk);
            proofs.push(p);
        }
        Ok(())
    })();
    let e2e_s = rows_s + tp0.elapsed().as_secs_f64();
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
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": STATEMENT, "scheme": "frame-v3/blake3-keyed",
               "prover": if plan.gpu {"flock-cuda-block"} else {"flock-cpu"}, "gpu": plan.gpu, "unit": inst.unit, "set": inst.header["set"],
               "subcircuit": inst.header["subcircuit"], "instances": inst.n, "units_real": inst.units, "units_per_block": st.lay.units,
               "chunks_per_block": st.lay.chunks, "n_blake3": st.lay.comp_slots * st.blocks_real, "n_units": inst.units,
               "k_log": st.lay.k_log, "nbl": st.nbl, "dense_m": st.m, "e2e_s": e2e_s, "rows_s": rows_s, "commit_s": 0.0, "prove_s": prove_s,
               "prove_total_s": prove_s.iter().sum::<f64>(), "buckets": buckets, "wait_s": wait_s, "calls": calls, "wire_sent": ws,
               "wire_recv": wr, "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "sigma": hex(&sig),
               "roots": inst.header["frame_v3"]["roots"], "netlist_sha256": hex(&st.net.sha256), "statement_digest": hex(&st.digest),
               "public_sha256": hex(&inst.public_sha256)}),
        proofs,
    )
}

/// (case, expect accept, both reps must reject)
const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("output_claim_false", false, true),
    ("message_byte_differs_from_the_committed_row", false, false),
    ("unit_input_differs_from_its_message_bits", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("chunk_value_public_forged", false, false),
    ("key_substituted", false, true),
    ("flags_forged", false, true),
    ("chained_run_input_forged", false, true),
    ("coins_before_y", false, false),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("prover_file_differs_from_verifier_file", false, false),
    ("wiring_swapped", false, false),
    ("output_mapping_swapped", false, false),
    ("units_swapped_between_slots", false, false),
    ("row_key_changed", false, false),
    ("row_scalar_forged", false, false),
    ("eps_forged_tail_words", false, false),
    ("aggregate_forged_with_its_scalar", false, true),
    ("cut_input_differs_from_the_cut_word", false, true),
    ("public_word_forged", false, false),
    ("zero_leaf_rewired", false, false),
    ("tail_output_forged", false, false),
];

/// Cases that only a cut template has, and one that needs a chunk split into several runs.
const CUT_ONLY: &[&str] = &["row_scalar_forged", "eps_forged_tail_words", "aggregate_forged_with_its_scalar", "cut_input_differs_from_the_cut_word"];
const RUNS_ONLY: &[&str] = &["chained_run_input_forged"];
/// Cases of units that return output leaves, and of tensor-core templates (a public port, zero leaves, tail outputs).
const RET_ONLY: &[&str] = &["output_claim_false", "output_mapping_swapped"];
const TAIL_OUT_ONLY: &[&str] = &["public_word_forged", "zero_leaf_rewired", "tail_output_forged"];

fn skipped(st: &FrameStmt, inst: &FrameInstances, name: &str) -> bool {
    let multi_run = inst.blocks.iter().any(|bl| bl.0.iter().flatten().any(|c| c.3 > 0));
    (st.cm.words == 0 && CUT_ONLY.contains(&name)) || (!multi_run && RUNS_ONLY.contains(&name))
        || (st.lay.ret_words == 0 && RET_ONLY.contains(&name)) || (st.cm.outputs.is_empty() && TAIL_OUT_ONLY.contains(&name))
}

/// The verifier's file with instance 0's cut word `flip.0` changed by `flip.1`, then every instance's tail words (and the
/// outputs, when the tail computes them) recomputed by `tail` from its units' words and the public words.
fn retail(st: &FrameStmt, tail: &[flock_live::ir_tail::Op], flip: Option<(usize, u32)>) -> FrameInstances {
    let mut v = load(&arg("--instances", ""));
    let kc = st.cm.words;
    if let Some((k, bit)) = flip {
        v.cuts[k] ^= bit;
    }
    let produced: std::collections::BTreeSet<usize> = st.cm.unit_out.iter().flatten().map(|p| p.1).chain(st.cm.public.iter().map(|p| p.0)).collect();
    let total: usize = v.out_ports.iter().map(|p| p.1).sum();
    for i in 0..v.n {
        let w = &mut v.cuts[i * kc..(i + 1) * kc];
        let mut slots: Vec<Option<u32>> = (0..kc).map(|k| produced.contains(&k).then_some(w[k])).collect();
        flock_live::ir_tail::eval(tail, &mut slots, TABLES.get().unwrap());
        for (k, x) in w.iter_mut().enumerate() {
            *x = slots[k].unwrap();
        }
        for (l, &s) in st.cm.outputs.iter().enumerate() {
            v.inst_outputs[i * total + l] = slots[s].unwrap() as u16;
        }
    }
    v
}

/// An IR6 case: a structurally valid file (the block table and frame-v3 roots still check) whose leaf maps or key differ from
/// the netlist's; pass iff `check_leaf_maps` refuses it.
fn leaf_case(st: &FrameStmt, name: &str) -> Option<Value> {
    let mut v = load(&arg("--instances", ""));
    match name {
        "wiring_swapped" => v.lay.wiring.iter_mut().for_each(|w| w.swap(0, 1)),
        "output_mapping_swapped" => {
            // every instance-local unit 0's first two returned words swapped with its mapping: the instance outputs (and roots) stay
            v.lay.out_leaf[0].swap(0, 1);
            let (c0, c1, rw) = (v.lay.ret_cols[0], v.lay.ret_cols[1], v.lay.ret_words);
            let get = |w: &[F128], c: usize| ((if c % 128 < 64 { w[c / 128].lo >> (c % 128) } else { w[c / 128].hi >> (c % 128 - 64) }) & 0xFFFF) as u16;
            let set = |w: &mut [F128], c: usize, x: u16| {
                let t = c % 128;
                let m = 0xFFFFu64;
                if t < 64 { w[c / 128].lo = (w[c / 128].lo & !(m << t)) | ((x as u64) << t) } else { w[c / 128].hi = (w[c / 128].hi & !(m << (t - 64))) | ((x as u64) << (t - 64)) }
            };
            for g in (0..v.units).step_by(v.upi) {
                let w = &mut v.outputs[g * rw..(g + 1) * rw];
                let (a, b) = (get(w, c0), get(w, c1));
                set(w, c0, b);
                set(w, c1, a);
            }
        }
        "units_swapped_between_slots" => {
            let slots: Vec<(usize, usize)> = v.blocks.iter().enumerate()
                .flat_map(|(b, bl)| bl.1.iter().enumerate().filter(|x| x.1.is_some()).map(move |(u, _)| (b, u))).take(2).collect();
            let (x, y) = (v.blocks[slots[0].0].1[slots[0].1], v.blocks[slots[1].0].1[slots[1].1]);
            v.blocks[slots[0].0].1[slots[0].1] = y;
            v.blocks[slots[1].0].1[slots[1].1] = x;
        }
        "row_key_changed" => v.key = flock_live::chunk::row_key(b"w"),
        // the first P.V step slot's last b port (a masked key's zero word in a boundary block) rewired to its first (a real key)
        "zero_leaf_rewired" => {
            let u = v.lay.units / 2;
            v.lay.wiring[u][15] = v.lay.wiring[u][0];
        }
        _ => return None,
    }
    let shape = check_blocks(&v).and_then(|_| check_roots(&v));
    let r = check_leaf_maps(&v, &st.net);
    Some(json!({"case": name, "expect": "refused at load (the netlist's leaf maps)", "shape_checks_pass": shape.is_ok(), "refused": r.is_err(),
                "pass": shape.is_ok() && r.is_err(), "why": r.err().or(shape.err())}))
}

/// A load-time case: pass iff the verifier's checks refuse the altered file.
fn load_case(st: &FrameStmt, name: &str) -> Option<Value> {
    if let Some(v) = leaf_case(st, name) {
        return Some(v);
    }
    let t = TABLES.get().unwrap();
    let (v, consistent) = match name {
        "public_word_forged" => {
            let mut v = load(&arg("--instances", ""));
            v.cuts[st.cm.public[0].0] ^= 1;
            let r = check_public_ports(&v, &st.cm);
            return Some(json!({"case": name, "expect": "refused at load (the public port's committed digest)", "refused": r.is_err(),
                               "pass": r.is_err(), "why": r.err()}));
        }
        "tail_output_forged" => {
            let mut v = load(&arg("--instances", ""));
            v.inst_outputs[3] ^= 1;
            let r = check_cut_words(&v, &st.cm, t);
            return Some(json!({"case": name, "expect": "refused at load (the tail's outputs)", "refused": r.is_err(), "pass": r.is_err(),
                               "why": r.err()}));
        }
        "row_scalar_forged" => {
            // the last cut word the tail computes (RMSNorm's row scalar; attention's last P or rescale word)
            let given: std::collections::BTreeSet<usize> = st.cm.unit_out.iter().flatten().map(|p| p.1).chain(st.cm.public.iter().map(|p| p.0)).collect();
            let k = (0..st.cm.words).rev().find(|k| !given.contains(k)).expect("a tail-computed cut word");
            let mut v = load(&arg("--instances", ""));
            v.cuts[k] ^= 1;
            (v, true)
        }
        "eps_forged_tail_words" => {
            // every positive tail constant x 100 (RMSNorm's N and eps; attention's per-block exp2 scale): a whole forged tail
            let mut tail = st.cm.tail.clone();
            let mut forged = 0;
            for o in tail.iter_mut() {
                if let flock_live::ir_tail::Prim::Const(x) = &mut o.prim
                    && f32::from_bits(*x) > 0.0
                    && f32::from_bits(*x).is_finite()
                {
                    *x = (f32::from_bits(*x) * 100.0).to_bits();
                    forged += 1;
                }
            }
            assert!(forged > 0, "a positive tail constant");
            let v = retail(st, &tail, None);
            let forged = flock_live::ir_block::CutMap { tail, ..st.cm.clone() };
            let ok = check_cut_words(&v, &forged, t).is_ok();
            (v, ok)
        }
        _ => return None,
    };
    let r = check_cut_words(&v, &st.cm, t);
    Some(json!({"case": name, "expect": "refused at load", "consistent_under_forged_tail": consistent, "refused": r.is_err(),
                "pass": consistent && r.is_err(), "why": r.err()}))
}

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan { gpu, ..Plan::default() };
    match name {
        "message_byte_differs_from_the_committed_row" => Plan { message_flip: true, ..h },
        "unit_input_differs_from_its_message_bits" => Plan { input_flip: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..h },
        "chunk_value_public_forged" => Plan { cv_flip: true, ..h },
        "key_substituted" => Plan { key_substituted: true, ..h },
        "flags_forged" => Plan { flags_forged: true, ..h },
        "chained_run_input_forged" => Plan { run_input_forged: true, ..h },
        "cut_input_differs_from_the_cut_word" => Plan { cut_flip: true, ..h },
        "coins_before_y" => Plan { no_y: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { fast: true, ..h },
        _ => h,
    }
}

fn load(path: &str) -> FrameInstances {
    FrameInstances::load(path).unwrap_or_else(|e| refuse(e))
}

fn run_case(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, name: &str, gpu: bool) -> Value {
    if let Some(v) = load_case(st, name) {
        return v;
    }
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan { gpu, ..Plan::default() }).1);
    }
    let reload = || load(&arg("--instances", ""));
    let (pinst, vinst) = match name {
        "output_claim_false" => {
            let mut v = reload();
            v.outputs[7 * v.lay.ret_words].lo ^= 1;
            (inst.clone(), Arc::new(v))
        }
        "prover_file_differs_from_verifier_file" => {
            let mut p = reload();
            p.public_sha256[0] ^= 1;
            (Arc::new(p), inst.clone())
        }
        "aggregate_forged_with_its_scalar" => {
            let k0 = st.cm.unit_out[0][0].1;
            let v = retail(st, &st.cm.tail, Some((k0, 1 << 12)));
            if let Err(e) = check_cut_words(&v, &st.cm, TABLES.get().unwrap()) {
                return json!({"case": name, "pass": false, "why": format!("the self-consistent forgery was refused at load: {e}")});
            }
            (inst.clone(), Arc::new(v))
        }
        _ => (inst.clone(), inst.clone()),
    };
    let s = Arc::new(Mutex::new(Server::new(config(st, sigma(st, inst)), Box::new(FrameVerifier { st: st.clone(), inst: vinst }))));
    let (v, _) = session(st, &pinst, shared(InProc(s.clone())), &plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let pass = acc == want.1 && (!want.2 || both);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"]})
}

fn selftest(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, gpu: bool) {
    let only = arg("--only", "");
    if !only.is_empty() {
        println!("NEG\t{}", run_case(st, inst, &only, gpu));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
        if skipped(st, inst, name) {
            continue;
        }
        let v = if gpu && load_case(st, name).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["selftest", "--gpu", "--instances", &arg("--instances", ""), "--netlist", &arg("--netlist", ""), "--tables", &arg("--tables", "tables"),
                       "--only", name])
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
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "scheme": "frame-v3/blake3-keyed", "unit": inst.unit, "subcircuit": inst.header["subcircuit"],
        "instances": inst.n, "units": inst.units, "run": st.lay.nb, "k_log": st.lay.k_log, "m": st.m, "cut_words": st.cm.words,
        "gpu": gpu, "all_pass": all}));
}

fn serve_loop(make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    let listen = arg("--listen", "127.0.0.1:7400");
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
        std::fs::write(d.join("session.json"), serde_json::to_string_pretty(&rec).unwrap()).unwrap();
        for ((t, r), b) in s.proofs() {
            std::fs::write(d.join(format!("{}.bin", rep_stream(t, *r).replace('/', "."))), b).unwrap();
        }
        let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
        std::fs::write(d.join("verifier.json"), json!({"operator": arg("--operator", ""), "host": host.trim(),
            "runpod_pod_id": std::env::var("RUNPOD_POD_ID").ok(), "research_run": std::env::var("RESEARCH_RUN_ID").ok(),
            "listen": listen}).to_string()).unwrap();
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "why": rec["verdict"]["why"], "aborted": rec["aborted"],
                          "link_mode": rec["link_mode"], "link_sha256": rec["link_sha256"], "require_link": rec["config"]["require_link"],
                          "verify_s": rec["verdict"]["verify_s"], "handle_s": rec["verdict"]["handle_s"], "rounds": rec["verdict"]["rounds"]});
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        writeln!(idx, "{line}").unwrap();
        println!("SESSION\t{line}");
        if Some(i + 1) == max {
            break;
        }
    }
}

fn refuse(why: String) -> ! {
    println!("REFUSED\t{}", json!({"statement": STATEMENT, "why": why}));
    std::process::exit(2);
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let tb = Instant::now();
    let inst = Arc::new(load(&arg("--instances", "frame.bin")));
    let net_text = std::fs::read_to_string(arg("--netlist", "net.txt")).expect("netlist");
    let net = IrUnitNet::parse(&net_text);
    let pin = arg("--pin", "");
    if cmd == "serve" && pin.is_empty() {
        refuse("serve needs --pin (the pinned lowering's sha256)".into());
    }
    let class = arg("--class", "");
    if !class.is_empty() {
        let c = check_class(&class, &pin, &net, &net_text, &inst).unwrap_or_else(|e| refuse(e));
        CLASS_PIN.set(c).unwrap();
    } else if !pin.is_empty() && hex(&net.sha256) != pin {
        refuse(format!("the netlist's sha256 {} is not the pinned lowering {pin}", hex(&net.sha256)));
    }
    if inst.unit != net.name || inst.header["unit_sha256"].as_str() != Some(hex(&net.sha256).as_str()) {
        refuse("the instance file was staged for another unit netlist".into());
    }
    for c in [check_order(&net), check_blocks(&inst), check_roots(&inst), check_leaf_maps(&inst, &net)] {
        if let Err(e) = c {
            refuse(e);
        }
    }
    let st = Arc::new(FrameStmt::new(net, &inst).unwrap_or_else(|e| refuse(e)));
    let tables = TABLES.get_or_init(|| Tables::load(&arg("--tables", "tables"), &tables_needed(&st.cm.tail)).unwrap_or_else(|e| refuse(e)));
    if let Err(e) = check_public_ports(&inst, &st.cm).and_then(|_| check_cut_words(&inst, &st.cm, tables)) {
        refuse(e);
    }
    eprintln!("statement {} instances={} units={} blocks={} comp_slots={} run={} units_per_block={} k_log={} nbl={} m={} cut_words={} built in {:.2}s",
              inst.unit, inst.n, inst.units, st.blocks_real, st.lay.comp_slots, st.lay.nb, st.lay.units, st.lay.k_log, st.nbl, st.m,
              st.cm.words, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "serve" => serve_loop(move || server(&st, &inst)),
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            let plan = Plan { gpu: flag("--gpu"), ..Plan::default() };
            for run in 0..warm + runs {
                let mut tcp = TcpTransport::connect(&arg("--verifier", "127.0.0.1:7400")).unwrap();
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
                refuse("replay needs --pin (the pinned lowering's sha256)".into());
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
        _ => eprintln!("usage: flock-ir-frame serve|prove|selftest|replay --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}

/// sha256 of a unit netlist without its name and its per-T lines (`verity_flock.templates.attention_head.unit_rows_sha256`): the
/// header line minus its name token, then the matrix rows up to `LEAVES`.
fn unit_rows_sha256(text: &str) -> String {
    let (head, rest) = text.split_once('\n').unwrap_or((text, ""));
    let mut s = head.split(' ').enumerate().filter(|(i, _)| *i != 1).map(|(_, t)| t).collect::<Vec<_>>().join(" ");
    s.push('\n');
    s.push_str(rest.split("\nLEAVES ").next().unwrap_or(""));
    hex(&sha256(s.as_bytes()))
}

/// A key-count class (`flock-ir-class/v1`, `--class MANIFEST`): the pin is the manifest's sha256 (serve and replay must name it;
/// prove takes the manifest's own); the manifest lists every T of its class `[lo, hi]`; the file's own key count (its key port's
/// words / words per key, which its committed rows bind) is in the class; the netlist is that T's entry; and its unit rows are
/// the class's, so one block circuit serves the class. The statement is then this netlist's, and Σ binds the class pin.
fn check_class(path: &str, pin: &str, net: &IrUnitNet, text: &str, inst: &FrameInstances) -> Result<[u8; 32], String> {
    let b = std::fs::read(path).map_err(|e| format!("class manifest: {e}"))?;
    let c = sha256(&b);
    if !pin.is_empty() && hex(&c) != pin {
        return Err(format!("the class manifest's sha256 {} is not the pin {pin}", hex(&c)));
    }
    let m: Value = serde_json::from_slice(&b).map_err(|e| format!("class manifest: {e}"))?;
    // CP2: the pinned bytes are the manifest's own canonical serialization (sorted keys, no whitespace, one newline), so a
    // duplicate key (serde_json keeps the last) or any other spelling of the same content is refused
    if serde_json::to_string(&m).map_err(|e| e.to_string())? + "\n" != String::from_utf8_lossy(&b) {
        return Err("the class manifest is not its canonical serialization".into());
    }
    if m["format"] != "flock-ir-class/v1" {
        return Err("not a flock-ir-class/v1 manifest".into());
    }
    let (lo, hi) = (m["T"][0].as_u64().ok_or("class: no T range")? as usize, m["T"][1].as_u64().ok_or("class: no T range")? as usize);
    let nets = m["nets"].as_object().ok_or("class: no nets")?;
    if lo == 0 || lo > hi || nets.len() != hi - lo + 1 || (lo..=hi).any(|t| !nets.contains_key(&t.to_string())) {
        return Err(format!("the class manifest does not list every T of [{lo}, {hi}] once"));
    }
    let port = m["key_port"].as_str().ok_or("class: no key_port")?;
    let per = m["words_per_key"].as_u64().ok_or("class: no words_per_key")? as usize;
    let words = inst.in_ports.iter().find(|p| p.0 == port).ok_or(format!("class: no input port {port}"))?.1;
    if per == 0 || words % per != 0 || words / per < lo || words / per > hi {
        return Err(format!("the file's key count ({words} words of {port}, {per} per key) is not in the class [{lo}, {hi}]"));
    }
    let t = words / per;
    if nets[&t.to_string()].as_str() != Some(hex(&net.sha256).as_str()) {
        return Err(format!("the netlist is not the class's netlist for T = {t}"));
    }
    if m["unit_rows"].as_str() != Some(unit_rows_sha256(text).as_str()) {
        return Err("the netlist's unit rows are not the class's".into());
    }
    Ok(c)
}

fn f128_of_hex(s: &str) -> F128 {
    F128::new(u64::from_str_radix(&s[16..32], 16).unwrap(), u64::from_str_radix(&s[..16], 16).unwrap())
}

fn unhex32(s: &str) -> [u8; 32] {
    (0..32).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<_>>().try_into().unwrap()
}

/// Re-verify a verifier's recorded session from files with this machine's own statement (its own staged instance file,
/// checked above as `serve` checks it, the pinned netlist and tables, the honest run outputs): the coins are the recorded
/// ones, so this is not transferable evidence, but every message the proofs carry must hash to what the verifier recorded
/// before each coin, the link exchange must hash to the recorded link_sha256, and Flock must accept both reps on one root.
fn replay(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, rec: &Value, proofs: &[Vec<u8>]) -> Result<Value, String> {
    let sig = sigma(st, inst);
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
    let (_, cvs) = st.compressions(inst);
    let publics = FrameStmt::publics_bytes(&cvs);
    if rec["link"]["publics_sha256"][TABLE].as_str() != Some(hex(&sha256(&publics)).as_str()) {
        return Err("C4: the record's committed publics are not this instance file's run outputs".into());
    }
    let v = FrameVerifier { st: st.clone(), inst: inst.clone() };
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
    Ok(json!({"accepted": true, "statement": STATEMENT, "unit": inst.unit, "instances": inst.n, "sigma": hex(&sig),
              "netlist_sha256": hex(&st.net.sha256), "statement_digest": hex(&st.digest), "public_sha256": hex(&inst.public_sha256),
              "root_b_sha256": root_b, "link_sha256": rec["link_sha256"], "proof_sha256": want_proofs, "verify_s": t.elapsed().as_secs_f64()}))
}
