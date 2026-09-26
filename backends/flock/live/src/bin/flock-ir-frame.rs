//! flock-ir-frame: an IR template's units bound to their frame-v3 rows (`flock_live::ir_frame`, `verity/flock-ir-frame/v1`),
//! proved by the CPU prover or by Flock-CUDA's host-witness mode (`--gpu`, feature `gpu`) in a flock-128-r2 live session.
//! Lane flock-ir-lowering; the session protocol is flock-gpu-link's flock-pure-gpu.
//!
//!   flock-ir-frame serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256 [--sessions N] [--operator LANE]
//!   flock-ir-frame prove    --verifier ADDR --instances FILE --netlist FILE [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-ir-frame selftest --instances FILE --netlist FILE [--gpu] [--only CASE]
//!
//! Before anything the verifier checks its own instance file (`verity_flock.ir_frame.stage`, never a producer's): the unit
//! netlist's order, the block table against the plan, and the frame-v3 roots recomputed from its row digests and output
//! words. Session: `Hello(Σ)`, `Commit(root_f = Σ, root_B, publics = every real block's chunk values)` (the verifier folds
//! each row's chunk values into its digest), two verifier points (the regions' claims), `Link(y = 0, 0)`, Flock's coins.

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
    *,
};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::blake3::Compression};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "irf";
const SIGMA_TAG: &[u8] = b"verity/flock-ir-frame/sigma/v1";
const STATEMENT: &str = "verity/flock-ir-frame/v1";

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
    format!("flock-ir-frame-v1/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

fn sigma(st: &FrameStmt, inst: &FrameInstances) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
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
    cfg.link = Some(LinkSpec { sigma, m: st.m_pts, points: 2 });
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

fn wit(st: &Arc<FrameStmt>, comps: &[Compression], plan: &Plan, zero: bool) -> Wit {
    let (st2, c2) = (st.clone(), comps.to_vec());
    let input_flip = plan.input_flip;
    Wit {
        comps: comps.to_vec(),
        inputs: Box::new(move |b, u| {
            let mut r = st2.unit_inputs(&c2, b, u);
            if zero {
                r.iter_mut().for_each(|w| *w = F128::ZERO);
            }
            if input_flip && b == 0 && u == 3 {
                r[0].lo ^= 1 << 2;
            }
            r
        }),
        flip_z: plan.unit_flip.then_some(((st.lay.unit_pos0 + 5) << st.lay.unit_log) + st.net.in_words * 128 + 40),
    }
}

fn session(st: &Arc<FrameStmt>, inst: &Arc<FrameInstances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let sig = sigma(st, inst);
    let hello = config(st, sig).hello();
    let (mut comps, mut cvs) = st.compressions(inst);
    if plan.message_flip {
        comps[1].1[3] ^= 1 << 7;
        let (mut cv, lay) = (st.key, &st.lay);
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
    let w0 = wit(st, &comps, plan, false);
    let w1 = wit(st, &comps, plan, plan.rep1_other_witness);
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
    ("coins_before_y", false, false),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("prover_file_differs_from_verifier_file", false, false),
];

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan { gpu, ..Plan::default() };
    match name {
        "message_byte_differs_from_the_committed_row" => Plan { message_flip: true, ..h },
        "unit_input_differs_from_its_message_bits" => Plan { input_flip: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..h },
        "chunk_value_public_forged" => Plan { cv_flip: true, ..h },
        "key_substituted" => Plan { key_substituted: true, ..h },
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
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "scheme": "frame-v3/blake3-keyed", "unit": inst.unit, "subcircuit": inst.header["subcircuit"],
        "instances": inst.n, "units": inst.units, "m": st.m, "gpu": gpu, "all_pass": all}));
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
    let net = IrUnitNet::parse(&std::fs::read_to_string(arg("--netlist", "net.txt")).expect("netlist"));
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
    if inst.header["cut_words"].as_u64().unwrap_or(0) != 0 || net.cut.is_some() {
        refuse("verity/flock-ir-frame/v1 does not take cut words yet".into());
    }
    for c in [check_order(&net), check_blocks(&inst), check_roots(&inst)] {
        if let Err(e) = c {
            refuse(e);
        }
    }
    let st = Arc::new(FrameStmt::new(net, &inst).unwrap_or_else(|e| refuse(e)));
    eprintln!("statement {} instances={} units={} blocks={} comp_slots={} units_per_block={} k_log={} nbl={} m={} built in {:.2}s", inst.unit,
              inst.n, inst.units, st.blocks_real, st.lay.comp_slots, st.lay.units, st.lay.k_log, st.nbl, st.m, tb.elapsed().as_secs_f64());
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
        _ => eprintln!("usage: flock-ir-frame serve|prove|selftest --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}
