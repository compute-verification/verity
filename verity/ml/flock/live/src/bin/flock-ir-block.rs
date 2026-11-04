//! flock-ir-block: an IR subcircuit template's identical units (`flock_live::ir_block`, `verity/flock-ir-block/v1`) as ONE
//! block R1CS, proved by the CPU prover or by Flock-CUDA's host-witness mode (`--gpu`, feature `gpu`) in a flock-128-r2
//! live session. The session protocol is flock-pure-gpu's, minus the frame-v3
//! commitments (the unit inputs and outputs are public, read by the verifier from its own instance file).
//!
//!   flock-ir-block serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256 [--tables DIR] [--sessions N] [--operator LANE]
//!   flock-ir-block prove    --verifier ADDR --instances FILE --netlist FILE [--tables DIR] [--gpu] [--runs N] [--warm W]
//!   flock-ir-block selftest --instances FILE --netlist FILE [--tables DIR] [--gpu] [--only CASE]
//!
//! Before anything, the instance file is checked against the netlist (`ir_block::cut_map`, `check_cuts`): a cut template's
//! cut words must be exactly what the native tail computes from the units' words, evaluated here (`ir_tail`, the MUFU
//! tables under `--tables`, pinned by sha256); a file with a cut word the netlist and tail do not account for is refused.
//! The instance file is the verifier's own staging of the input set (`verity_flock.ir_lower.stage`), never a producer's.
//!
//! Session: `Hello(Σ)`, `Commit(root_f = Σ, root_B, no publics)`, two verifier points (the regions' claims),
//! `Link(y = 0, 0)`, then Flock's coins.

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
use flock_live::{ir_block::*, ir_tail::{Op, Prim, Tables, tables_needed}, *};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "ir";
const SIGMA_TAG: &[u8] = b"verity/flock-ir-block/sigma/v1";
const STATEMENT: &str = "verity/flock-ir-block/v1";

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
    format!("flock-ir-block-v1/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

fn sigma(st: &IrStmt, inst: &IrInstances) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
    h.update(inst.unit.as_bytes());
    h.update([0]);
    h.update(st.net.sha256);
    h.update(st.digest);
    h.update(inst.sha256);
    h.finalize().into()
}

struct IrVerifier {
    st: Arc<IrStmt>,
    inst: Arc<IrInstances>,
}

impl StatementVerifier for IrVerifier {
    fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
        domain(rep)
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
        let extra = st.extra_claims(link.points, &self.inst);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn config(st: &IrStmt, sigma: [u8; 32]) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: st.m_pts, points: 2 });
    cfg
}

fn server(st: &Arc<IrStmt>, inst: &Arc<IrInstances>) -> Server {
    Server::new(config(st, sigma(st, inst)), Box::new(IrVerifier { st: st.clone(), inst: inst.clone() }))
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness.
struct Wit {
    inputs: Box<dyn Fn(usize) -> Vec<F128> + Sync>,
    flip_z: Option<usize>,
}

fn witness(st: &IrStmt, w: &Wit) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>, f64) {
    let t = Instant::now();
    let (mut z, a, b, zl) = st.witness(&*w.inputs);
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    let zl = if w.flip_z.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.k_log) } else { zl };
    (z, a, b, zl, t.elapsed().as_secs_f64())
}

fn prove_cpu(st: &IrStmt, w: &Wit, lig: LigeritoProfile, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (z, a, b, zl, tw) = witness(st, w);
    let tp = Instant::now();
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
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({"t.witness": tw, "t.prove": tp.elapsed().as_secs_f64()}))
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &IrStmt, w: &Wit, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let (z, a, b, _zl, tw) = witness(st, w);
    let (proof, commitment, ph) = flock_live::gpu::prove_units(st, &z, &a, &b, &params(st.m, lig), &domain(rep), hook, ch);
    let bk = json!({"t.witness_host": tw, "t.witness": ph[0], "t.encoding_commitment": ph[1], "t.link": ph[2], "t.zerocheck": ph[3],
                    "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                    "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6]});
    eprintln!("GPUPHASE\t{bk}");
    (bincode::serialize(&(commitment, proof)).unwrap(), bk)
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &IrStmt, _: &Wit, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone)]
struct Plan {
    gpu: bool,
    no_y: bool,
    input_flip: bool,
    unit_flip: bool,
    rep1_other_witness: bool,
    profile: [LigeritoProfile; 2],
    substitute: Option<Vec<Vec<u8>>>,
}

impl Plan {
    fn honest(gpu: bool) -> Self {
        Plan { gpu, no_y: false, input_flip: false, unit_flip: false, rep1_other_witness: false, profile: [LigeritoProfile::Fast100; 2],
               substitute: None }
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn wit(st: &Arc<IrStmt>, inst: &Arc<IrInstances>, input_flip: bool, flip_z: Option<usize>, zero: bool) -> Wit {
    let (st2, inst2) = (st.clone(), inst.clone());
    Wit {
        inputs: Box::new(move |g| {
            if zero {
                return vec![F128::ZERO; st2.net.in_words];
            }
            let mut r = st2.honest_inputs(&inst2)(g);
            if input_flip && g == 3 {
                r[0].lo ^= 1 << 5;
            }
            r
        }),
        flip_z,
    }
}

fn session(st: &Arc<IrStmt>, inst: &Arc<IrInstances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let sig = sigma(st, inst);
    let hello = config(st, sig).hello();
    // a flipped committed bit inside unit 5's AND rows
    let flip = plan.unit_flip.then_some((5 << st.unit_log) + st.net.in_words * 128 + 40);
    let w0 = wit(st, inst, plan.input_flip, flip, false);
    let w1 = wit(st, inst, plan.input_flip, flip, plan.rep1_other_witness);
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
                let pts: Vec<Vec<F128>> = match call(&tt, Req::Commit { root_f: sig.to_vec(), roots: vec![(TABLE.into(), root.to_vec())], publics: vec![] }) {
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
            let lig = plan.profile[rep];
            let (p, bk) = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu { prove_gpu(st, w, lig, rep, &mut hook, &mut lch) } else { prove_cpu(st, w, lig, &mut hook, &mut lch) }
            }))
            .map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            buckets.push(bk);
            proofs.push(p);
        }
        Ok(())
    })();
    let e2e_s = tp0.elapsed().as_secs_f64();
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
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": STATEMENT, "prover": if plan.gpu {"flock-cuda-block"} else {"flock-cpu"},
               "gpu": plan.gpu, "unit": inst.unit, "set": inst.header["set"], "subcircuit": inst.header["subcircuit"],
               "instances": inst.header["instances"], "units_real": st.units_real, "units_per_block": st.units, "unit_log": st.unit_log,
               "k_log": st.k_log, "nbl": st.nbl, "dense_m": st.m, "e2e_s": e2e_s, "prove_s": prove_s,
               "prove_total_s": prove_s.iter().sum::<f64>(), "buckets": buckets, "wait_s": wait_s, "calls": calls, "wire_sent": ws,
               "wire_recv": wr, "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "sigma": hex(&sig),
               "netlist_sha256": hex(&st.net.sha256), "statement_digest": hex(&st.digest), "instances_sha256": hex(&inst.sha256)}),
        proofs,
    )
}

/// (case, expect accept, both reps must reject)
const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("output_claim_false", false, true),
    ("unit_input_differs_from_public_input", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("coins_before_y", false, false),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("cut_claimed_without_its_tail", false, false),
    ("header_carries_its_own_cut", false, false),
    ("row_scalar_forged", false, false),
    ("eps_forged_with_every_word_recomputed", false, false),
    ("aggregate_forged_with_its_scalar", false, true),
    ("prover_file_differs_from_verifier_file", false, false),
];

/// Cases that only a cut template has.
const CUT_ONLY: &[&str] = &["row_scalar_forged", "eps_forged_with_every_word_recomputed", "aggregate_forged_with_its_scalar"];

/// A copy of the file with different bytes (its sha256 changes with them).
fn altered(inst: &IrInstances, f: impl FnOnce(&mut IrInstances)) -> IrInstances {
    let mut x = inst.clone();
    f(&mut x);
    x.sha256[0] ^= 1;
    x
}

fn set_word32(words: &mut [F128], col: usize, v: u32) {
    let w = &mut words[col / 128];
    let t = col % 128;
    let m = 0xFFFF_FFFFu64;
    if t < 64 { w.lo = (w.lo & !(m << t)) | ((v as u64) << t) } else { w.hi = (w.hi & !(m << (t - 64))) | ((v as u64) << (t - 64)) }
}

fn get_word32(words: &[F128], col: usize) -> u32 {
    let w = &words[col / 128];
    let t = col % 128;
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) as u32
}

/// Instance 0 with unit 0's first cut output (a warp aggregate) off by one bit and every cut word the tail computes from it
/// rewritten into every unit's cut inputs: a self-consistent forgery the cut checks accept, which only the proof refutes.
fn forge_aggregate(inst: &IrInstances, cm: &CutMap, t: &Tables) -> IrInstances {
    altered(inst, |x| {
        let (c0, _) = cm.unit_out[0][0];
        let ow = x.out_words;
        let v = get_word32(&x.outputs[..ow], c0) ^ (1 << 12);
        set_word32(&mut x.outputs[..ow], c0, v);
        let upi = cm.unit_in.len();
        let mut slots: Vec<Option<u32>> = vec![None; cm.words];
        for u in 0..upi {
            for &(c, k) in &cm.unit_out[u] {
                slots[k] = Some(get_word32(&x.outputs[u * ow..(u + 1) * ow], c));
            }
        }
        flock_live::ir_tail::eval(&cm.tail, &mut slots, t);
        let iw = x.in_words;
        for u in 0..upi {
            for &(c, k) in &cm.unit_in[u] {
                set_word32(&mut x.inputs[u * iw..(u + 1) * iw], c, slots[k].unwrap());
            }
        }
    })
}

/// Red-team-flock-2's IR4 file: the tail's eps (its smallest positive constant) x 100, every instance's cut words recomputed
/// with that tail and every unit's outputs recomputed from its inputs. Returns the file and the forged tail.
fn forge_eps(inst: &IrInstances, cm: &CutMap, net: &IrUnitNet, t: &Tables) -> (IrInstances, Vec<Op>) {
    let mut tail = cm.tail.clone();
    let eps = tail
        .iter_mut()
        .filter_map(|o| match &mut o.prim {
            Prim::Const(v) if f32::from_bits(*v) > 0.0 => Some(v),
            _ => None,
        })
        .min_by(|a, b| f32::from_bits(**a).total_cmp(&f32::from_bits(**b)))
        .expect("a positive tail constant");
    *eps = (f32::from_bits(*eps) * 100.0).to_bits();
    let (upi, iw, ow) = (cm.unit_in.len(), inst.in_words, inst.out_words);
    let x = altered(inst, |x| {
        for i in 0..x.units / upi {
            let mut slots: Vec<Option<u32>> = vec![None; cm.words];
            for u in 0..upi {
                for &(c, k) in &cm.unit_out[u] {
                    slots[k] = Some(get_word32(x.output(i * upi + u), c));
                }
            }
            flock_live::ir_tail::eval(&tail, &mut slots, t);
            for u in 0..upi {
                let g = i * upi + u;
                for &(c, k) in &cm.unit_in[u] {
                    set_word32(&mut x.inputs[g * iw..(g + 1) * iw], c, slots[k].unwrap());
                }
                let o = net.outputs(x.input(g));
                x.outputs[g * ow..(g + 1) * ow].copy_from_slice(&o);
            }
        }
    });
    (x, tail)
}

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan::honest(gpu);
    match name {
        "unit_input_differs_from_public_input" => Plan { input_flip: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..h },
        "coins_before_y" => Plan { no_y: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { profile: [LigeritoProfile::Fast; 2], ..h },
        _ => h,
    }
}

/// The verifier's instance file with unit 7's first output word off by one bit (a false claim about the computation).
fn false_claim(inst: &IrInstances) -> IrInstances {
    let mut bad = inst.clone();
    bad.outputs[7 * bad.out_words].lo ^= 1;
    bad
}

struct Ctx {
    st: Arc<IrStmt>,
    inst: Arc<IrInstances>,
    net_text: String,
    cm: CutMap,
    tables: Tables,
}

/// A load-time case: pass iff the checks refuse the altered file (or, for the forgery, accept it).
fn load_case(ctx: &Ctx, name: &str) -> Option<Value> {
    let net = IrUnitNet::parse(&ctx.net_text);
    let verdict = |x: &IrInstances| -> Result<(), String> {
        let cm = cut_map(x, &net)?;
        check_cuts(x, &cm, &ctx.tables)
    };
    let r = match name {
        "cut_claimed_without_its_tail" => verdict(&altered(&ctx.inst, |x| x.header["cut_words"] = json!(ctx.cm.words + 1))),
        "header_carries_its_own_cut" => verdict(&altered(&ctx.inst, |x| {
            let mut cut = net.cut.clone().unwrap_or(json!({"words": 0, "tail": [], "in": [], "out": []}));
            if let Some(op) = cut["tail"].as_array_mut().and_then(|t| t.iter_mut().find(|o| o[1] == "Const")) {
                op[2] = json!((f32::from_bits(op[2].as_u64().unwrap() as u32) * 100.0).to_bits());
            }
            x.header["cut"] = cut;
        })),
        "eps_forged_with_every_word_recomputed" => {
            let (f, tail) = forge_eps(&ctx.inst, &ctx.cm, &net, &ctx.tables);
            let forged = CutMap { tail, ..ctx.cm.clone() };
            let consistent = check_cuts(&f, &forged, &ctx.tables).is_ok() && (0..f.units).all(|g| net.outputs(f.input(g)) == f.output(g));
            let r = verdict(&f);
            return Some(json!({"case": name, "expect": "refused at load (the pinned tail)", "consistent_under_forged_tail": consistent,
                               "refused": r.is_err(), "pass": consistent && r.is_err(), "why": r.err()}));
        }
        "row_scalar_forged" => verdict(&altered(&ctx.inst, |x| {
            let (c, _) = ctx.cm.unit_in[0][0];
            let iw = x.in_words;
            let v = get_word32(&x.inputs[..iw], c) ^ 1;
            set_word32(&mut x.inputs[..iw], c, v);
        })),
        _ => return None,
    };
    Some(json!({"case": name, "expect": "refused at load", "refused": r.is_err(), "pass": r.is_err(), "why": r.err()}))
}

fn run_case(ctx: &Ctx, name: &str, gpu: bool) -> Value {
    let (st, inst) = (&ctx.st, &ctx.inst);
    if let Some(v) = load_case(ctx, name) {
        return v;
    }
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan::honest(gpu)).1);
    }
    // (the prover's file, the verifier's file, the Σ the verifier expects)
    let (pinst, vinst, vsig_of) = match name {
        "output_claim_false" => (inst.clone(), Arc::new(false_claim(inst)), inst.clone()),
        "prover_file_differs_from_verifier_file" => {
            let p = Arc::new(altered(inst, |x| x.outputs[7 * x.out_words].lo ^= 1));
            (p, inst.clone(), inst.clone())
        }
        "aggregate_forged_with_its_scalar" => {
            let f = Arc::new(forge_aggregate(inst, &ctx.cm, &ctx.tables));
            let net = IrUnitNet::parse(&ctx.net_text);
            if let Err(e) = cut_map(&f, &net).and_then(|cm| check_cuts(&f, &cm, &ctx.tables)) {
                return json!({"case": name, "pass": false, "why": format!("the self-consistent forgery was refused at load: {e}")});
            }
            (f.clone(), f.clone(), f)
        }
        _ => (inst.clone(), inst.clone(), inst.clone()),
    };
    let s = Arc::new(Mutex::new(Server::new(config(st, sigma(st, &vsig_of)), Box::new(IrVerifier { st: st.clone(), inst: vinst }))));
    let (v, _) = session(st, &pinst, shared(InProc(s.clone())), &plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let pass = acc == want.1 && (!want.2 || both);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"]})
}

/// Every unit's recorded outputs are the netlist's outputs on its recorded inputs, and every region's claimed value is the
/// honest witness's multilinear extension at the claim point.
fn precheck(st: &IrStmt, inst: &IrInstances) -> Value {
    use rayon::prelude::*;
    let bad = (0..inst.units).into_par_iter().filter(|&g| st.net.outputs(inst.input(g)) != inst.output(g)).count();
    let (z, _, _, _) = st.witness(&st.honest_inputs(inst));
    let r: Vec<F128> = (0..st.m_pts).map(|i| F128::new(0x9e3779b97f4a7c15u64.wrapping_mul(i as u64 + 1), 0x7f4a7c159e3779b9u64 ^ i as u64)).collect();
    let mut regions = Vec::new();
    for reg in &st.regions {
        let p = st.region_point(reg, &r);
        let w = flock_core::proof::eq_skip_weights(&p);
        let direct = z.iter().enumerate().fold(F128::ZERO, |acc, (k, zw)| {
            (0..128).fold(acc, |a, t| if (if t < 64 { zw.lo >> t } else { zw.hi >> (t - 64) }) & 1 == 1 { a + w[128 * k + t] } else { a })
        });
        regions.push(json!({"region": format!("{:?}", reg.region), "agrees": direct == st.region_value(reg, &r, inst)}));
    }
    json!({"units": inst.units, "outputs_differ": bad, "regions": regions})
}

fn selftest(ctx: &Ctx, gpu: bool) {
    let (st, inst) = (&ctx.st, &ctx.inst);
    if flag("--precheck") {
        println!("PRECHECK\t{}", precheck(st, inst));
    }
    let only = arg("--only", "");
    if !only.is_empty() {
        println!("NEG\t{}", run_case(ctx, &only, gpu));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
        if ctx.cm.words == 0 && CUT_ONLY.contains(&name) {
            continue;
        }
        let v = if gpu && load_case(ctx, name).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["selftest", "--gpu", "--instances", &arg("--instances", ""), "--netlist", &arg("--netlist", ""), "--tables", &arg("--tables", "tables"), "--only", name])
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
            run_case(ctx, name, gpu)
        };
        all &= v["pass"] == true;
        println!("NEG\t{v}");
    }
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "unit": inst.unit, "subcircuit": inst.header["subcircuit"], "units": inst.units,
        "m": st.m, "cut_words": ctx.cm.words, "gpu": gpu, "all_pass": all}));
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
        std::fs::write(d.join("session.json"), flock_live::record_text::canonical_json(&rec)).unwrap();
        for ((t, r), b) in s.proofs() {
            std::fs::write(d.join(format!("{}.bin", rep_stream(t, *r).replace('/', "."))), b).unwrap();
        }
        let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
        std::fs::write(d.join("verifier.json"), json!({"operator": arg("--operator", ""), "host": host.trim(),
            "runpod_pod_id": std::env::var("RUNPOD_POD_ID").ok(), "research_run": std::env::var("RESEARCH_RUN_ID").ok(),
            "listen": listen}).to_string()).unwrap();
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "why": rec["verdict"]["why"], "aborted": rec["aborted"],
                          "verify_s": rec["verdict"]["verify_s"], "rounds": rec["verdict"]["rounds"]});
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
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let tb = Instant::now();
    let inst = Arc::new(IrInstances::load(&arg("--instances", "instances.bin")));
    let net_text = std::fs::read_to_string(arg("--netlist", "net.txt")).expect("netlist");
    let net = IrUnitNet::parse(&net_text);
    let pin = arg("--pin", "");
    if cmd == "serve" && pin.is_empty() {
        refuse("serve needs --pin (the pinned lowering's sha256)".into());
    }
    if !pin.is_empty() && hex(&net.sha256) != pin {
        refuse(format!("the netlist's sha256 {} is not the pinned lowering {pin}", hex(&net.sha256)));
    }
    if inst.unit != net.name || inst.header["unit_sha256"].as_str() != Some(hex(&net.sha256).as_str()) {
        refuse("the instance file was staged for another unit netlist".into());
    }
    if (inst.in_words, inst.out_words) != (net.in_words, net.out_cols.len()) {
        refuse("instance words do not match the netlist's IO".into());
    }
    if let Err(e) = check_order(&net) {
        refuse(e);
    }
    let cm = cut_map(&inst, &net).unwrap_or_else(|e| refuse(e));
    let tables = Tables::load(&arg("--tables", "tables"), &tables_needed(&cm.tail)).unwrap_or_else(|e| refuse(e));
    if let Err(e) = check_cuts(&inst, &cm, &tables) {
        refuse(e);
    }
    let st = Arc::new(IrStmt::new(net, inst.units));
    eprintln!("statement {} units={} per_block={} unit_log={} k_log={} nbl={} m={} cut_words={} built in {:.2}s", inst.unit, inst.units, st.units,
              st.unit_log, st.k_log, st.nbl, st.m, cm.words, tb.elapsed().as_secs_f64());
    let ctx = Ctx { st: st.clone(), inst: inst.clone(), net_text, cm, tables };
    match cmd.as_str() {
        "selftest" => selftest(&ctx, flag("--gpu")),
        "serve" => serve_loop(move || server(&st, &inst)),
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let plan = Plan::honest(flag("--gpu"));
            for run in 0..warm + runs {
                let t = shared(TcpTransport::connect(&arg("--verifier", "127.0.0.1:7400")).unwrap());
                let (mut v, _) = session(&st, &inst, t, &plan);
                v["run"] = json!(run);
                v["warm"] = json!(run < warm);
                println!("LIVE\t{v}");
            }
        }
        _ => eprintln!("usage: flock-ir-block serve|prove|selftest --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}
