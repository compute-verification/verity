//! flock-128-r2 live sessions on the CPU union prover (BLAKE3 table, `n` compressions).
//!
//!   flock-live serve    --listen ADDR --out DIR --n N            verifier: challenger + session verifier, one session per connection
//!   flock-live prove    --verifier ADDR --n N [--case C]         prover: one session, prints the verifier's verdict
//!   flock-live selftest --n N                                    every negative against an in-process server (NEG lines)
//!   flock-live bench    --verifier ADDR --n N --runs R           FS vs live prove time, same process
//!
//! Output lines are `LIVE\t{json}` / `NEG\t{json}` / `BENCH\t{json}`.

use std::{
    array::from_fn,
    collections::BTreeMap,
    io::{BufReader, BufWriter},
    net::TcpListener,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Instant,
};

use flock_core::{
    pcs::{
        PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    test_rng::Rng,
};
use flock_live::*;
use flock_prover::{
    challenger::FsChallenger,
    lincheck::SparseMatrixCircuit,
    merkle::HashKind,
    pcs::Commitment,
    proof::R1csProofLigerito,
    proof_io::{BundleFlavor, R1csProofBundleLigerito, peek_flavor},
    prover::{UnionSlotProverInput, prove_fast_ligerito_union},
    r1cs::BlockR1cs,
    r1cs_hashes::blake3::{Compression, build_block_r1cs, generate_witness_batch_major_partial_into, min_n_blocks_log},
    schedule::{Registry, TableType},
    union::UnionInstance,
    verifier::{verify_ligerito, verify_ligerito_union},
};
use serde_json::{Value, json};

const TABLE: &str = "blake3";

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn blocks(seed: u64, n: usize) -> Vec<Compression> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            (cv, m, rng.next_u32() as u64, 64u32, 16u32 | (rng.next_u32() & 3))
        })
        .collect()
}

fn profile(s: &str) -> LigeritoProfile {
    match s {
        "fast" => LigeritoProfile::Fast,
        "fast100" => LigeritoProfile::Fast100,
        o => panic!("profile {o}"),
    }
}

/// The statement: one BLAKE3 table of `n` compressions in a Flock union, as flock-128's harness proves it.
struct Stmt {
    n: usize,
    nu: usize,
    r1cs: BlockR1cs,
    registry: Registry,
}

impl Stmt {
    fn new(n: usize) -> Self {
        let nu = min_n_blocks_log(n);
        let r1cs = build_block_r1cs(nu);
        let registry = Registry::new(vec![TableType::from_block_r1cs(&r1cs)], nu);
        Self { n, nu, r1cs, registry }
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, vec![self.n])
    }

    fn params(&self, lig: LigeritoProfile) -> PcsParams {
        let u = self.union();
        let m = u.dense_m();
        let batch = embedded_initial_k_or_default(m, lig);
        PcsParams {
            m,
            log_inv_rate: lig.log_inv_rate(),
            log_batch_size: batch,
            profile: lig,
            num_lanes: u.commit_lanes(batch),
            merkle_hash: Default::default(),
        }
    }

    fn spec(&self) -> TableSpec {
        TableSpec {
            name: TABLE.into(),
            binding: Binding::Mixed { registry_digest: self.registry.digest(), counts_le: (self.n as u64).to_le_bytes().to_vec() },
        }
    }

    fn prove<C: flock_transcript::challenger::Challenger>(&self, bl: &[Compression], lig: LigeritoProfile, ch: &mut C) -> Vec<u8> {
        self.prove_w(bl, lig, false, ch)
    }

    /// `flip`: one committed witness bit flipped after generation (an invalid witness, proven honestly).
    fn prove_w<C: flock_transcript::challenger::Challenger>(&self, bl: &[Compression], lig: LigeritoProfile, flip: bool, ch: &mut C) -> Vec<u8> {
        let circ = self.r1cs.csc_lincheck_circuit();
        let nu = self.nu;
        let slots = vec![UnionSlotProverInput::in_place(
            move |dst| {
                let z_len = dst.z.len();
                let z0 = dst.z.as_mut_ptr();
                let out = generate_witness_batch_major_partial_into(bl, nu, dst);
                if flip {
                    // SAFETY: `dst.z` outlives this closure's call; the generator returned its borrow.
                    unsafe { (*z0.add(z_len / 3)).lo ^= 1 << 9 };
                }
                out
            },
            circ,
        )];
        let (proof, commitment, _) = prove_fast_ligerito_union(&self.union(), &self.params(lig), slots, ch);
        R1csProofBundleLigerito { commitment, proof }.to_bytes()
    }

    fn verify_with<C: flock_transcript::challenger::Challenger>(&self, bytes: &[u8], pcs: &PcsParams, ch: &mut C) -> Result<Vec<u8>, String> {
        match peek_flavor(bytes).map_err(|e| format!("decode: {e:?}"))? {
            BundleFlavor::R1cs => {}
            f => return Err(format!("R7: bundle flavour {f:?} refused (RS flavour only)")),
        }
        let b = R1csProofBundleLigerito::from_bytes(bytes).map_err(|e| format!("decode: {e:?}"))?;
        if &b.commitment.params != pcs {
            return Err("R7: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        let circ = self.r1cs.csc_lincheck_circuit();
        let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![circ];
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_union(&self.union(), &circs, &b.commitment, &b.proof, pcs, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(b.commitment.cap.as_flattened().to_vec())
    }
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-128/fast100x2/rep{rep}").into_bytes()
}

/// The verifier's side: the pinned profile comes from here, never from the proof.
struct Verifier(Arc<Stmt>);

impl StatementVerifier for Verifier {
    fn verify(&self, table: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, _link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        assert_eq!(table, TABLE);
        self.0.verify_with(proof, &self.0.params(LigeritoProfile::Fast100), ch)
    }
    fn domain(&self, _table: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }
}

fn server(st: &Arc<Stmt>, require_link: bool) -> Server {
    let mut cfg = SessionConfig::flock_128_r2(vec![st.spec()]);
    cfg.require_link = require_link;
    Server::new(cfg, Box::new(Verifier(st.clone())))
}

/// A stand-in for the link context flock-glue will send (root_F, the link points, y): opaque bytes that must reach
/// the verifier before Flock's first coin.
fn link_context(seed: u64) -> Vec<u8> {
    let mut v = b"flock-live/link-stub/v1".to_vec();
    v.extend_from_slice(&seed.to_le_bytes());
    v
}

/// What a (possibly dishonest) prover does in one session.
#[derive(Clone)]
struct Plan {
    hello: Option<String>,
    send_link: bool,
    reps: usize,
    witness: Vec<u64>,
    profile: Vec<&'static str>,
    tamper: Option<Tamper>,
    pow_nonce: u64,
    witness_flip: bool,
    /// Replace rep r's proof bytes after the session (e.g. rep 0's proof as rep 1's, an FS proof, a flipped byte).
    substitute: BTreeMap<usize, Subst>,
}

#[derive(Clone)]
enum Subst {
    Rep0,
    Fs,
    FlipByte(usize),
    Flavor(u8),
    RelabelFast,
    /// A proof from another (honest, accepted) session.
    Other(Vec<u8>),
}

impl Plan {
    fn honest() -> Self {
        Plan {
            hello: None,
            send_link: true,
            reps: 2,
            witness: vec![0xA, 0xA],
            profile: vec!["fast100", "fast100"],
            tamper: None,
            pow_nonce: 0,
            witness_flip: false,
            substitute: BTreeMap::new(),
        }
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

/// Run one session through `t`; returns the verdict (or where the verifier refused) and prover timings.
fn session(st: &Stmt, t: SharedTransport, plan: &Plan) -> Value {
    session_p(st, t, plan).0
}

fn session_p(st: &Stmt, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let hello = plan.hello.clone().unwrap_or_else(|| SessionConfig::flock_128_r2(vec![st.spec()]).hello());
    let t0 = Instant::now();
    let mut prove_s = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello))? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        if plan.send_link {
            call(&t, Req::Link(link_context(7)))?;
        }
        for rep in 0..plan.reps {
            let bl = blocks(plan.witness[rep], st.n);
            let mut ch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            ch.pow_nonce = plan.pow_nonce;
            ch.tamper = plan.tamper.clone();
            let tp = Instant::now();
            let p = catch_unwind(AssertUnwindSafe(|| st.prove_w(&bl, profile(plan.profile[rep]), plan.witness_flip, &mut ch)))
                .map_err(|e| {
                    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
                })?;
            prove_s.push(tp.elapsed().as_secs_f64());
            proofs.push(p);
        }
        Ok(())
    })();
    if let Err(e) = res {
        // A refused session still ends with the verifier's verdict (it rejects).
        let v = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        return (json!({"accepted": v["accepted"], "why": v["why"], "prover_stopped": e.chars().take(200).collect::<String>()}), proofs);
    }
    for (rep, s) in &plan.substitute {
        proofs[*rep] = match s {
            Subst::Rep0 => proofs[0].clone(),
            Subst::Fs => st.prove(&blocks(plan.witness[*rep], st.n), LigeritoProfile::Fast100,
                                  &mut FsChallenger::with_hash(&domain(*rep), HashKind::default())),
            Subst::FlipByte(at) => {
                let mut p = proofs[*rep].clone();
                let i = p.len() * *at / 1000;
                p[i] ^= 1;
                p
            }
            Subst::Other(p) => p.clone(),
            Subst::RelabelFast => {
                let mut b = R1csProofBundleLigerito::from_bytes(&proofs[*rep]).unwrap();
                b.commitment.params.profile = LigeritoProfile::Fast;
                b.to_bytes()
            }
            Subst::Flavor(f) => {
                let mut p = proofs[*rep].clone();
                p[6] = *f;
                p
            }
        };
    }
    for (rep, p) in proofs.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: p.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (sent, recv, calls) = t.lock().unwrap().stats();
    let wait_s = t.lock().unwrap().wait_s();
    let out = json!({
        "accepted": v["accepted"], "why": v["why"], "verdict": v, "prove_s": prove_s,
        "prove_total_s": prove_s.iter().sum::<f64>(), "session_wall_s": t0.elapsed().as_secs_f64(),
        "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(),
        "wire_sent": sent, "wire_recv": recv, "calls": calls, "wait_s": wait_s,
    });
    (out, proofs)
}

fn inproc(st: &Arc<Stmt>, require_link: bool) -> (SharedTransport, Arc<Mutex<Server>>) {
    let s = Arc::new(Mutex::new(server(st, require_link)));
    (shared(InProc(s.clone())), s)
}

fn serve(st: Arc<Stmt>) {
    let no_link = arg("--no-link", "0") == "1";
    serve_loop(move || server(&st, !no_link));
}

/// GPU tables (Flock-CUDA proves non-union `prove_ligerito`, one table per session): `--table NAME --nbl N`, with
/// `--netlist P` for a census unit (feature `verity-unit`), else the BLAKE3 statement.
fn serve_gpu() {
    let name = arg("--table", "blake3");
    let nbl: usize = arg("--nbl", "17").parse().unwrap();
    let netlist = arg("--netlist", "");
    let r1cs = Arc::new(gpu_r1cs(&netlist, nbl));
    let no_link = arg("--no-link", "0") == "1";
    serve_loop(move || {
        let spec = TableSpec { name: name.clone(), binding: Binding::R1cs { statement_digest: r1cs.statement_digest() } };
        let mut cfg = SessionConfig::flock_128_r2(vec![spec]);
        cfg.require_link = !no_link;
        Server::new(cfg, Box::new(GpuVerifier(r1cs.clone())))
    });
}

#[cfg(feature = "verity-unit")]
fn gpu_r1cs(netlist: &str, nbl: usize) -> BlockR1cs {
    if netlist.is_empty() {
        return build_block_r1cs(nbl);
    }
    let net = flock_prover::r1cs_hashes::verity_unit::Netlist::load(netlist);
    assert_eq!(net.self_check(), 0, "netlist self-check");
    net.block_r1cs(nbl)
}

#[cfg(not(feature = "verity-unit"))]
fn gpu_r1cs(netlist: &str, nbl: usize) -> BlockR1cs {
    assert!(netlist.is_empty(), "--netlist needs the verity-unit feature");
    build_block_r1cs(nbl)
}

/// Flock-CUDA's parameters, pinned: Fast100, rate 1/2, batch 6, SHA-256 Merkle trees.
fn gpu_params(r1cs: &BlockR1cs) -> PcsParams {
    PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast100,
        num_lanes: None,
        merkle_hash: HashKind::Sha256,
    }
}

struct GpuVerifier(Arc<BlockR1cs>);

impl StatementVerifier for GpuVerifier {
    fn verify(&self, _table: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, _link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let (commitment, proof): (Commitment, R1csProofLigerito) =
            bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        let pcs = gpu_params(&self.0);
        if commitment.params != pcs {
            return Err("R7: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        let r1cs = &*self.0;
        let lc = SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito(r1cs, &commitment, &proof, &lc, &pcs, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
    fn domain(&self, _table: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }
}

fn serve_loop(make: impl Fn() -> Server) {
    let listen = arg("--listen", "127.0.0.1:7100");
    let out = std::path::PathBuf::from(arg("--out", "sessions"));
    std::fs::create_dir_all(&out).unwrap();
    let l = TcpListener::bind(&listen).unwrap();
    println!("SERVING {listen}");
    for (i, conn) in l.incoming().enumerate() {
        let conn = conn.unwrap();
        conn.set_nodelay(true).unwrap();
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut w = BufWriter::new(conn);
        let mut s = make();
        let id = format!("f{:04}-{}", i, std::process::id());
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
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "aborted": rec["aborted"],
                          "rounds": rec["verdict"]["rounds"], "streams": rec["verdict"]["streams"]});
        use std::io::Write;
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        writeln!(idx, "{line}").unwrap();
        println!("SESSION\t{line}");
    }
}

fn neg(st: &Arc<Stmt>, name: &str, maps_to: &str, plan: &Plan, require_link: bool, want_accept: bool) -> bool {
    let (t, s) = inproc(st, require_link);
    let v = session(st, t, plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let ok = acc == want_accept;
    let rec = s.lock().unwrap().record_json();
    println!(
        "NEG\t{}",
        json!({"case": name, "maps_to": maps_to, "n": st.n, "expect": if want_accept {"accept"} else {"reject"},
               "accepted": acc, "pass": ok, "why": v["why"], "prover_stopped": v["prover_stopped"],
               "rounds": rec["verdict"]["rounds"], "streams": rec["verdict"]["streams"]})
    );
    ok
}

fn selftest(st: Arc<Stmt>) {
    let h = Plan::honest;
    let mut all = true;
    let t = |name: &str, maps: &str, p: Plan, link: bool, acc: bool, all: &mut bool| *all &= neg(&st, name, maps, &p, link, acc);
    t("honest_live_session", "honest", h(), true, true, &mut all);
    t("reps_with_different_roots", "neg1 R1", Plan { witness: vec![0xA, 0xB], ..h() }, true, false, &mut all);
    t("rep0_proof_replayed_as_rep1", "neg2 R2",
      Plan { substitute: BTreeMap::from([(1, Subst::Rep0)]), ..h() }, true, false, &mut all);
    t("parent_message_altered_after_coin", "neg3 R2", Plan { tamper: Some(Tamper::Alter { stream: "blake3/rep0".into(), round: 3 }), ..h() }, true, false, &mut all);
    t("fork_child_message_altered_after_coin", "neg3 R2+R3", Plan { tamper: Some(Tamper::Alter { stream: "blake3/rep1/f0".into(), round: 2 }), ..h() }, true, false, &mut all);
    t("parent_message_sent_after_its_coin", "brief R2", Plan { tamper: Some(Tamper::Defer { stream: "blake3/rep0".into(), round: 5 }), ..h() }, true, false, &mut all);
    t("fork_child_message_sent_after_its_coin", "brief R2+R3", Plan { tamper: Some(Tamper::Defer { stream: "blake3/rep0/f0".into(), round: 1 }), ..h() }, true, false, &mut all);
    t("prover_chosen_pow_nonce_honest", "neg4 R4 (coin independent: accepted)", Plan { pow_nonce: 0x5eed_1234, ..h() }, true, true, &mut all);
    t("pow_nonce_altered_after_coin", "neg4 R4", Plan { pow_nonce: 0x5eed_1234, tamper: Some(Tamper::Nonce), ..h() }, true, false, &mut all);
    t("fastx1_session", "neg5 R7", Plan { hello: Some(json!({"profile":"fast","reps":1,"flavor":"rs","tables":[TABLE]}).to_string()), reps: 1, profile: vec!["fast"], ..h() }, true, false, &mut all);
    t("fast100x1_lone_rep_hello", "neg5 R7", Plan { hello: Some(json!({"profile":"fast100","reps":1,"flavor":"rs","tables":[TABLE]}).to_string()), reps: 1, ..h() }, true, false, &mut all);
    t("lone_rep_under_r2_hello", "neg5 R7", Plan { reps: 1, ..h() }, true, false, &mut all);
    t("fast_profile_proof_in_r2_session", "neg5 R7", Plan { profile: vec!["fast", "fast"], ..h() }, true, false, &mut all);
    t("fast100_proof_relabelled_fast_params", "neg5 R7", Plan { substitute: BTreeMap::from([(0, Subst::RelabelFast)]), ..h() }, true, false, &mut all);
    t("fs_transcript_presented_as_live", "neg6 R8", Plan { substitute: BTreeMap::from([(1, Subst::Fs)]), ..h() }, true, false, &mut all);
    t("ag_flavour_bundle", "neg7 R7", Plan { substitute: BTreeMap::from([(0, Subst::Flavor(6))]), ..h() }, true, false, &mut all);
    t("byte_flip_rep1", "neg8", Plan { substitute: BTreeMap::from([(1, Subst::FlipByte(500))]), ..h() }, true, false, &mut all);
    t("coins_before_link_context", "R5", Plan { send_link: false, ..h() }, true, false, &mut all);
    let (a, other) = session_p(&st, inproc(&st, true).0, &h());
    assert_eq!(a["accepted"], true);
    t("cross_session_replay_both_reps", "neg9 (link neg 15)",
      Plan { substitute: BTreeMap::from([(0, Subst::Other(other[0].clone())), (1, Subst::Other(other[1].clone()))]), ..h() }, true, false, &mut all);
    t("witness_bit_flip_both_reps", "neg8", Plan { witness_flip: true, ..h() }, true, false, &mut all);
    // F3: child streams opened at parent position 0, or before the rep root's binding round, are refused.
    for (case, at, rounds) in [("fork_at_parent_position_0", 0u64, 0usize), ("fork_before_root_binding", 1, 0)] {
        let s = Arc::new(Mutex::new(server(&st, true)));
        let t = shared(InProc(s.clone()));
        let _ = call(&t, Req::Hello(SessionConfig::flock_128_r2(vec![st.spec()]).hello()));
        let _ = call(&t, Req::Link(link_context(7)));
        let _ = call(&t, Req::Open { stream: rep_stream(TABLE, 0), parent: String::new(), label: domain(0), at: 0 });
        for _ in 0..rounds {
            let _ = call(&t, Req::Round { stream: rep_stream(TABLE, 0), content: vec![0; 16], n: 1 });
        }
        let r = call(&t, Req::Open { stream: format!("{}/f0", rep_stream(TABLE, 0)), parent: rep_stream(TABLE, 0), label: b"x".to_vec(), at });
        let ok = matches!(r, Ok(Resp::Refused(_)));
        all &= ok;
        println!("NEG\t{}", json!({"case": case, "maps_to": "F3", "expect": "reject", "pass": ok, "why": format!("{r:?}")}));
    }
    let np = st.n - st.n / 4;
    all &= neg(&Arc::new(Stmt::new(np)), "honest_live_session_with_padding_rows", "neg10 (padding benign)", &h(), true, true);
    println!("SELFTEST\t{}", json!({"n": st.n, "all_pass": all}));
}

fn fs_session(st: &Stmt, lig: LigeritoProfile, reps: usize, seed: u64) -> (f64, Vec<usize>) {
    let bl = blocks(seed, st.n);
    let mut s = 0.0;
    let mut sz = Vec::new();
    for rep in 0..reps {
        let t = Instant::now();
        let p = st.prove(&bl, lig, &mut FsChallenger::with_hash(&domain(rep), HashKind::default()));
        s += t.elapsed().as_secs_f64();
        sz.push(p.len());
    }
    (s, sz)
}

fn bench(st: Arc<Stmt>) {
    let runs: usize = arg("--runs", "5").parse().unwrap();
    let addr = arg("--verifier", "");
    let _ = fs_session(&st, LigeritoProfile::Fast100, 1, 0xA);
    for r in 0..runs {
        let (fs_s, _) = fs_session(&st, LigeritoProfile::Fast100, 2, 0xA);
        let (fsx1_s, _) = fs_session(&st, LigeritoProfile::Fast, 1, 0xA);
        let t = if addr.is_empty() { inproc(&st, true).0 } else { shared(TcpTransport::connect(&addr).unwrap()) };
        let v = session(&st, t, &Plan::honest());
        println!("BENCH\t{}", json!({"run": r, "n": st.n, "m": st.params(LigeritoProfile::Fast100).m,
            "transport": if addr.is_empty() {"inproc"} else {"tcp"}, "verifier": addr,
            "fs_fast100x2_s": fs_s, "fs_fastx1_s": fsx1_s, "live_fast100x2_s": v["prove_total_s"],
            "live_over_fs": v["prove_total_s"].as_f64().unwrap_or(0.0) / fs_s, "accepted": v["accepted"],
            "calls": v["calls"], "wait_s": v["wait_s"], "empty_rounds": v["verdict"]["rounds_without_prover_message"], "wire_sent": v["wire_sent"], "wire_recv": v["wire_recv"],
            "verify_s": v["verdict"]["verify_s"], "rounds": v["verdict"]["rounds"]}));
    }
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let n: usize = arg("--n", "4096").parse().unwrap();
    if cmd == "serve-gpu" {
        return serve_gpu();
    }
    #[cfg(feature = "verity-unit")]
    match cmd.as_str() {
        "selftest-f2" => return f2::selftest(),
        "serve-union" => return f2::serve_union(),
        _ => {}
    }
    let st = Arc::new(Stmt::new(n));
    match cmd.as_str() {
        "serve" => serve(st),
        "selftest" => selftest(st),
        "bench" => bench(st),
        "prove" => {
            let t = shared(TcpTransport::connect(&arg("--verifier", "127.0.0.1:7100")).unwrap());
            let mut p = Plan::honest();
            if arg("--case", "honest") == "different-roots" {
                p.witness = vec![0xA, 0xB];
            }
            println!("LIVE\t{}", session(&st, t, &p));
        }
        _ => eprintln!("usage: flock-live serve|prove|selftest|bench [--n N] ..."),
    }
}

/// flock-link F2: the production statement verifiers, pinned from the verifier's configuration: the CPU census unit +
/// BLAKE3 union (registry digest of both table types, counts [n_blake3, n_units], Fast100 parameters) and the GPU unit
/// table (the unit netlist's statement digest, Flock-CUDA's Fast100 / batch 6 / SHA-256 parameters). The GPU-shaped
/// proofs here come from the CPU `prove_ligerito` (same protocol, same serialization); the H100 runs use Flock-CUDA.
#[cfg(feature = "verity-unit")]
mod f2 {
    use super::*;
    use flock_prover::{prover::prove_ligerito, r1cs_hashes::verity_unit::{CombinedSetup, Netlist}};

    pub const UNION: &str = "unit+blake3";

    pub struct UnionStmt {
        pub setup: CombinedSetup,
        pub counts: [usize; 2],
    }

    impl UnionStmt {
        pub fn new(netlist: &str, n: usize, units: usize) -> Self {
            let net = Netlist::load(netlist);
            assert_eq!(net.self_check(), 0, "netlist self-check");
            Self { setup: CombinedSetup::new(net, n.max(units)), counts: [n, units] }
        }
        pub fn spec(&self) -> TableSpec {
            TableSpec {
                name: UNION.into(),
                binding: Binding::Mixed {
                    registry_digest: self.setup.registry.digest(),
                    counts_le: self.counts.iter().flat_map(|&c| (c as u64).to_le_bytes()).collect(),
                },
            }
        }
        pub fn prove(&self, seed: u64, lig: LigeritoProfile, ch: &mut LiveChallenger) -> Vec<u8> {
            let bl = blocks(seed, self.counts[0]);
            let ids: Vec<u32> = (0..self.counts[1] as u32).collect();
            let (proof, commitment, _) = self.setup.prove_with(&bl, &ids, lig, ch);
            R1csProofBundleLigerito { commitment, proof }.to_bytes()
        }
    }

    pub struct UnionVerifier(pub Arc<UnionStmt>);

    impl StatementVerifier for UnionVerifier {
        fn verify(&self, _t: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, _l: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
            let s = &self.0;
            match peek_flavor(proof).map_err(|e| format!("decode: {e:?}"))? {
                BundleFlavor::R1cs => {}
                f => return Err(format!("R7: bundle flavour {f:?} refused (RS flavour only)")),
            }
            let b = R1csProofBundleLigerito::from_bytes(proof).map_err(|e| format!("decode: {e:?}"))?;
            let pcs = s.setup.pcs_params_with(s.counts, LigeritoProfile::Fast100);
            if b.commitment.params != pcs {
                return Err("F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
            }
            let u = UnionInstance::new(&s.setup.registry, s.counts.to_vec());
            let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> =
                vec![s.setup.blake3_r1cs.csc_lincheck_circuit(), s.setup.unit_r1cs.csc_lincheck_circuit()];
            catch_unwind(AssertUnwindSafe(|| verify_ligerito_union(&u, &circs, &b.commitment, &b.proof, &pcs, ch)))
                .map_err(|_| "verifier panicked".to_string())?
                .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
            Ok(b.commitment.cap.as_flattened().to_vec())
        }
        fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
            domain(rep)
        }
    }

    fn union_server(st: &Arc<UnionStmt>) -> Server {
        Server::new(SessionConfig::flock_128_r2(vec![st.spec()]), Box::new(UnionVerifier(st.clone())))
    }

    pub fn serve_union() {
        let st = Arc::new(UnionStmt::new(&arg("--netlist", ""), arg("--n", "4096").parse().unwrap(), arg("--units", "4096").parse().unwrap()));
        serve_loop(move || union_server(&st));
    }

    /// One two-rep session of `prover` against a server built for `verifier`'s statement.
    fn run(server: Server, table: &str, prove: &dyn Fn(usize, &mut LiveChallenger) -> Vec<u8>, hello: Option<String>) -> Value {
        let s = Arc::new(Mutex::new(server));
        let t = shared(InProc(s.clone()));
        let want = s.lock().unwrap().cfg.hello();
        let res: Result<(), String> = (|| {
            match call(&t, Req::Hello(hello.unwrap_or(want)))? {
                Resp::Ok => {}
                r => return Err(format!("hello: {r:?}")),
            }
            call(&t, Req::Link(link_context(9)))?;
            for rep in 0..2 {
                let mut ch = LiveChallenger::new(t.clone(), table, rep, &domain(rep))?;
                let p = catch_unwind(AssertUnwindSafe(|| prove(rep, &mut ch))).map_err(|e| {
                    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
                })?;
                call(&t, Req::Proof { table: table.into(), rep: rep as u32, bytes: p })?;
            }
            Ok(())
        })();
        let v: Value = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        json!({"accepted": v["accepted"], "why": v["why"], "prover_stopped": res.err().map(|e| e.chars().take(200).collect::<String>())})
    }

    fn report(all: &mut bool, case: &str, maps: &str, v: Value, want: bool) {
        let acc = v["accepted"].as_bool().unwrap_or(false);
        *all &= acc == want;
        println!("NEG\t{}", json!({"case": case, "maps_to": maps, "expect": if want {"accept"} else {"reject"}, "accepted": acc,
            "pass": acc == want, "why": v["why"], "prover_stopped": v["prover_stopped"]}));
    }

    pub fn selftest() {
        let (net, other) = (arg("--netlist", ""), arg("--netlist-other", ""));
        let (n, units, nbl): (usize, usize, usize) =
            (arg("--n", "512").parse().unwrap(), arg("--units", "512").parse().unwrap(), arg("--nbl", "10").parse().unwrap());
        let mut all = true;
        // CPU census unit + BLAKE3 union.
        let st = Arc::new(UnionStmt::new(&net, n, units));
        let ot = Arc::new(UnionStmt::new(&other, n, units));
        let fewer = Arc::new(UnionStmt::new(&net, n, units - 1));
        let honest = |st: &Arc<UnionStmt>, lig: LigeritoProfile| {
            let st = st.clone();
            move |_rep: usize, ch: &mut LiveChallenger| st.prove(0xA, lig, ch)
        };
        report(&mut all, "union_honest", "F2 honest", run(union_server(&st), UNION, &honest(&st, LigeritoProfile::Fast100), None), true);
        report(&mut all, "union_other_counts", "F2 counts pin", run(union_server(&st), UNION, &honest(&fewer, LigeritoProfile::Fast100), None), false);
        report(&mut all, "union_other_netlist", "F2 registry digest pin", run(union_server(&st), UNION, &honest(&ot, LigeritoProfile::Fast100), None), false);
        report(&mut all, "union_fast_profile", "F2 params pin", run(union_server(&st), UNION, &honest(&st, LigeritoProfile::Fast), None), false);
        // BLAKE3-only statement presented to the union verifier.
        let b3 = Arc::new(Stmt::new(n));
        let b3p = { let b3 = b3.clone(); move |_r: usize, ch: &mut LiveChallenger| b3.prove(&blocks(0xA, b3.n), LigeritoProfile::Fast100, ch) };
        report(&mut all, "union_blake3_only_statement", "F2 registry digest pin", run(union_server(&st), UNION, &b3p, None), false);

        // GPU unit table (non-union prove_ligerito, Flock-CUDA's pinned parameters).
        let gpu_server = |netlist: &str| {
            let r1cs = Arc::new(gpu_r1cs(netlist, nbl));
            let spec = TableSpec { name: "unit".into(), binding: Binding::R1cs { statement_digest: r1cs.statement_digest() } };
            Server::new(SessionConfig::flock_128_r2(vec![spec]), Box::new(GpuVerifier(r1cs)))
        };
        let gpu_prove = |netlist: &str, lig: LigeritoProfile| {
            let net = Netlist::load(netlist);
            let r1cs = net.block_r1cs(nbl);
            let (z, _, _, _) = net.witness_row_major(nbl);
            move |_rep: usize, ch: &mut LiveChallenger| {
                let mut pcs = gpu_params(&r1cs);
                pcs.profile = lig;
                let (proof, commitment, _) = prove_ligerito(&r1cs, z.clone(), &pcs, ch);
                bincode::serialize(&(commitment, proof)).unwrap()
            }
        };
        report(&mut all, "gpu_unit_honest", "F2 honest (GPU statement, CPU-made proof)", run(gpu_server(&net), "unit", &gpu_prove(&net, LigeritoProfile::Fast100), None), true);
        report(&mut all, "gpu_unit_other_netlist", "F2 statement digest pin", run(gpu_server(&net), "unit", &gpu_prove(&other, LigeritoProfile::Fast100), None), false);
        report(&mut all, "gpu_unit_fast_profile", "F2 params pin", run(gpu_server(&net), "unit", &gpu_prove(&net, LigeritoProfile::Fast), None), false);
        println!("SELFTEST\t{}", json!({"f2": true, "n": n, "units": units, "nbl": nbl, "all_pass": all}));
    }
}
