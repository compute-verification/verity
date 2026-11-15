//! flock-gpu-link: route (a)'s Flock side as ONE block R1CS per BLAKE3 chunk (`flock_live::chunk`), proved by the
//! CPU prover or by Flock-CUDA (`--gpu`, feature `gpu`), in a flock-128-r2 live session with the link exchange.
//!
//! The session is flock-link's (L1): `Hello`, then `Commit(root_F, root_B, the chunk chaining values)`, the verifier's
//! two link points, `Link(y)`, then Flock's coins. Both reps open, besides the R1CS's own ab and c claims, the two link
//! claims (L2) and the key, counter and chaining-value public claims at both points (L4 endpoints); the chain itself
//! is inside the block matrix (L4 wiring). One table carries the whole statement (L3 is the one-`Commit` rule).
//!
//!   flock-gpu-link selftest --vus V [--k K]                                 negatives against an in-process verifier (CPU)
//!   flock-gpu-link serve    --listen ADDR --out DIR --vus V --seed S [--gpu-params] [--operator LANE] [--sessions N]
//!   flock-gpu-link prove    --verifier ADDR --vus V --seed S [--gpu] [--runs R]
//!
//! Output lines are `LIVE\t{json}` / `NEG\t{json}` / `SELFTEST\t{json}`.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Instant,
};

use flock_core::{
    merkle::HashKind,
    pcs::{PcsParams, ligerito::{LigeritoProfile, embedded_initial_k_or_default}},
    proof::{R1csProofLigerito, eq_skip_weights},
    test_rng::Rng,
    verifier::verify_ligerito_extra,
};
use flock_field::F128;
use flock_live::{chunk::*, *};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::blake3::Compression};
use rayon::prelude::*;
use serde_json::{Value, json};

const TABLE: &str = "chain";

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

fn words(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n / 4).flat_map(|_| rng.next_u32().to_le_bytes()).collect()
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-gpu-link/fast100x2/rep{rep}").into_bytes()
}

/// The pinned PCS parameters: Flock-CUDA's (rate 1/2, the embedded batch, SHA-256 Merkle), also used on the CPU so
/// both provers make the same proofs.
fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

/// `Σ over set bits p of eq(r, p)`: bit p is bit p%8 of byte p/8 (operand words little-endian, LSB first).
fn link_eval(bytes: &[u8], r: &[F128]) -> F128 {
    let m = r.len();
    assert!(m >= 3 && bytes.len() * 8 <= 1usize << m);
    let lo = eq_skip_weights(&r[..3]);
    let mut t0 = [F128::ZERO; 256];
    for x in 1..256usize {
        let rest = x & (x - 1);
        t0[x] = t0[rest] + lo[(x ^ rest).trailing_zeros() as usize];
    }
    let mid_n = (m - 3).min(11);
    let mid = eq_skip_weights(&r[3..3 + mid_n]);
    let hi = eq_skip_weights(&r[3 + mid_n..]);
    bytes
        .par_chunks(1 << mid_n)
        .enumerate()
        .map(|(h, c)| {
            let mut acc = F128::ZERO;
            for (i, &x) in c.iter().enumerate() {
                if x != 0 {
                    acc += mid[i] * t0[x as usize];
                }
            }
            hi[h] * acc
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

fn leaf_digests(st: &ChunkStmt, bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes
        .chunks(2 * st.k)
        .enumerate()
        .map(|(l, row)| *blake3::keyed_hash(&row_key(if l < st.vus { b"x" } else { b"w" }), row).as_bytes())
        .collect()
}

fn cv_bytes(cvs: &[[u32; 8]]) -> Vec<u8> {
    cvs.iter().flatten().flat_map(|w| w.to_le_bytes()).collect()
}

fn parse_cvs(b: &[u8]) -> Vec<[u32; 8]> {
    b.chunks(32).map(|c| std::array::from_fn(|i| u32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap()))).collect()
}

fn sigma(st: &ChunkStmt, digests: &[[u8; 32]]) -> [u8; 32] {
    let cat: Vec<u8> = digests.iter().flatten().copied().collect();
    sha256(
        format!(
            "verity/link-L/v2-chunk\nscheme blake3-keyed/row/v2\nlambda_b pos=chunk*8192+j*512+u->chunk<<18|j<<14|512+u\n\
             points 2 gf2_128\nvus {}\nk {}\nm {}\nflock_statement {}\nflock_profile fast100 reps 2\nleaf_digests {} {}\n",
            st.vus,
            st.k,
            st.m_link,
            hex(&st.digest),
            digests.len(),
            hex(&sha256(&cat))
        )
        .as_bytes(),
    )
}

struct ChunkVerifier {
    st: Arc<ChunkStmt>,
    digests: Vec<[u8; 32]>,
}

impl StatementVerifier for ChunkVerifier {
    fn domain(&self, _table: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        let st = &self.st;
        if table != TABLE {
            return Err(format!("C4: {table} is not in the statement"));
        }
        if publics.len() != 32 * st.chunks {
            return Err(format!("C4: {} bytes of chunk chaining values, want {}", publics.len(), 32 * st.chunks));
        }
        let cvs = parse_cvs(publics);
        for (l, want) in self.digests.iter().enumerate() {
            let (key, _) = st.key_counter(l * st.cpl);
            if digest_of(&key, &cvs[l * st.cpl..(l + 1) * st.cpl]) != *want {
                return Err(format!("C4: the chunk chaining values of leaf {l} do not hash to its digest"));
            }
        }
        Ok(None)
    }

    fn verify(&self, _table: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let st = &self.st;
        let (commitment, proof): (Commitment, R1csProofLigerito) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        let pcs = params(st.m, LigeritoProfile::Fast100);
        if commitment.params != pcs {
            return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        if link.points.len() != 2 || link.y.len() != 2 {
            return Err("L2: the link claims are missing from the session".into());
        }
        let extra = st.extra_claims(link.points, link.y, &parse_cvs(link.publics));
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn server(st: &Arc<ChunkStmt>, digests: Vec<[u8; 32]>) -> Server {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma(st, &digests).to_vec(), m: st.m_link, points: 2 });
    Server::new(cfg, Box::new(ChunkVerifier { st: st.clone(), digests }))
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// One rep on the CPU: the witness of `comps`, then the linked prove; `hook` gets the flattened root after the
/// commitment and returns the extra claims' points.
fn prove_cpu(st: &ChunkStmt, comps: &[Compression], lig: LigeritoProfile, hook: Hook<'_>, ch: &mut LiveChallenger) -> Vec<u8> {
    let (z, a, b, zl) = st.witness(comps);
    let pcs = params(st.m, lig);
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs,
        &pcs,
        z,
        a,
        b,
        zl,
        &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(cm.cap.as_flattened()).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    bincode::serialize(&(commitment, proof)).unwrap()
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &ChunkStmt, comps: &[Compression], lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger) -> Vec<u8> {
    assert!(flock_live::gpu::device_count() > 0, "no CUDA device");
    let (proof, commitment, phase) = flock_live::gpu::prove_chunk(st, comps, &params(st.m, lig), &domain(rep), hook, ch);
    eprintln!("GPUPHASE\t{}", json!({"rep": rep, "m": st.m, "witness": phase[0], "commit": phase[1], "link": phase[2],
        "zerocheck": phase[3], "lincheck": phase[4], "ring_switch": phase[5], "ligerito": phase[6], "total": phase[7]}));
    bincode::serialize(&(commitment, proof)).unwrap()
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &ChunkStmt, _: &[Compression], _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger) -> Vec<u8> {
    panic!("built without the gpu feature")
}

#[derive(Clone, Copy, PartialEq)]
enum CommitMode {
    Honest,
    OtherRootB,
    Twice,
    NoRootF,
    YBeforePoints,
    NoY,
}

/// What a (possibly dishonest) prover does.
#[derive(Clone)]
struct Plan {
    gpu: bool,
    commit: CommitMode,
    y_delta: [u64; 2],
    flip_z_bit: bool,
    forge_middle: bool,
    bad_params: Option<usize>,
    publics_flip: bool,
    key_substituted: bool,
    dummy_message: bool,
    rep1_other_witness: bool,
    profile: [LigeritoProfile; 2],
    substitute: Option<Vec<Vec<u8>>>,
}

impl Plan {
    fn honest() -> Self {
        Plan {
            gpu: false,
            commit: CommitMode::Honest,
            y_delta: [0, 0],
            flip_z_bit: false,
            forge_middle: false,
            bad_params: None,
            publics_flip: false,
            key_substituted: false,
            dummy_message: false,
            rep1_other_witness: false,
            profile: [LigeritoProfile::Fast100; 2],
            substitute: None,
        }
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// (statement bytes the verifier's digests come from, z bytes Flock commits, b bytes y comes from).
fn instance(st: &ChunkStmt, seed: u64, plan: &Plan) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let b = words(seed, st.chunks * 1024);
    let mut z = b.clone();
    if plan.flip_z_bit {
        z[st.k * 2 + 77] ^= 0x10;
        return (z.clone(), z, b);
    }
    (b.clone(), z, b)
}

fn session(st: &ChunkStmt, t: SharedTransport, seed: u64, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let (sb, z, b) = instance(st, seed, plan);
    let hello = {
        let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
        cfg.link = Some(LinkSpec { sigma: sigma(st, &leaf_digests(st, &sb)).to_vec(), m: st.m_link, points: 2 });
        cfg.hello()
    };
    let t0 = Instant::now();
    let (mut comps, mut cvs) = st.compressions(&z);
    let mut b_link = b.clone();
    if plan.forge_middle {
        // Block 5 of chunk 0 carries another message; its output no longer feeds block 6, whose input is still the
        // honest chaining value, and the chunk's public output stays honest. y is computed from the forged bits.
        b_link[5 * 64 + 9] ^= 0x01;
        comps[5] = st.compressions(&b_link).0[5];
    }
    if let Some(what) = plan.bad_params {
        let r = &mut comps[15];
        match what {
            0 => r.4 &= !CHUNK_END,
            1 => r.2 = 7,
            _ => r.3 = 32,
        }
    }
    if plan.key_substituted {
        // Chunk 0 chained from the W key: every block consistent, the chaining values honest-looking but not public.
        let mut cv = st.keys[1];
        for j in 0..SUBS {
            comps[j].0 = cv;
            let c = comps[j];
            cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
        }
    }
    if plan.dummy_message {
        assert!(st.chunks < st.blocks(), "no dummy chunk at this size");
        let at = st.chunks * SUBS;
        comps[at].1[3] ^= 0x40;
        let mut cv = comps[at].0;
        for j in 0..SUBS {
            comps[at + j].0 = cv;
            let c = comps[at + j];
            cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
        }
    }
    if plan.publics_flip {
        cvs[1][2] ^= 1;
    }
    let cvb = cv_bytes(&cvs);
    let comps1 = if plan.rep1_other_witness { st.compressions(&words(seed ^ 0xB, z.len())).0 } else { comps.clone() };
    let mut prove_s = Vec::new();
    let mut y_s = 0.0;
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello))? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        if plan.commit == CommitMode::YBeforePoints {
            match call(&t, Req::Link(vec![0u8; 32]))? {
                Resp::Ok => {}
                r => return Err(format!("link: {r:?}")),
            }
        }
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            let tt = t.clone();
            let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                if let Some(c) = &cached {
                    return c.clone();
                }
                let mut root_b = root.to_vec();
                if plan.commit == CommitMode::OtherRootB {
                    root_b[0] ^= 1;
                }
                let root_f = if plan.commit == CommitMode::NoRootF {
                    Vec::new()
                } else {
                    sha256(&[b"flock-gpu-link/root-f-standin/v1".as_slice(), &seed.to_le_bytes()].concat()).to_vec()
                };
                let req = || Req::Commit { root_f: root_f.clone(), roots: vec![(TABLE.into(), root_b.clone())], publics: vec![(TABLE.into(), cvb.clone())] };
                let pts: Vec<Vec<F128>> = match call(&tt, req()) {
                    Ok(Resp::Coins(c)) if c.len() == 2 * st.m_link => c.chunks(st.m_link).map(|r| r.to_vec()).collect(),
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                };
                if plan.commit == CommitMode::Twice {
                    match call(&tt, req()) {
                        Ok(Resp::Coins(_)) => {}
                        other => panic!("LIVE-REFUSED commit twice: {other:?}"),
                    }
                }
                let ty = Instant::now();
                let y: Vec<F128> = pts
                    .iter()
                    .zip(plan.y_delta)
                    .map(|(r, d)| {
                        let mut y = link_eval(&b_link, r);
                        y.lo ^= d;
                        y
                    })
                    .collect();
                y_s = ty.elapsed().as_secs_f64();
                if plan.commit != CommitMode::NoY {
                    match call(&tt, Req::Link(f128_bytes(&y))) {
                        Ok(Resp::Ok) => {}
                        other => panic!("LIVE-REFUSED y: {other:?}"),
                    }
                }
                let e = st.extra_points(&pts);
                cached = Some(e.clone());
                e
            };
            let tp = Instant::now();
            let c = if rep == 0 { &comps } else { &comps1 };
            let lig = plan.profile[rep];
            let p = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu { prove_gpu(st, c, lig, rep, &mut hook, &mut lch) } else { prove_cpu(st, c, lig, &mut hook, &mut lch) }
            }))
            .map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            proofs.push(p);
        }
        Ok(())
    })();
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
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "vus": st.vus, "k": st.k, "chunks": st.chunks,
               "nbl": st.nbl, "m": st.m, "m_link": st.m_link, "gpu": plan.gpu, "prove_s": prove_s,
               "prove_total_s": prove_s.iter().sum::<f64>(), "y_s": y_s,
               "session_wall_s": t0.elapsed().as_secs_f64(),
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "wire_sent": ws, "wire_recv": wr,
               "calls": calls, "wait_s": wait_s}),
        proofs,
    )
}

fn inproc(st: &Arc<ChunkStmt>, statement: &[u8]) -> (SharedTransport, Arc<Mutex<Server>>) {
    let s = Arc::new(Mutex::new(server(st, leaf_digests(st, statement))));
    (shared(InProc(s.clone())), s)
}

fn neg(st: &Arc<ChunkStmt>, name: &str, maps_to: &str, plan: &Plan, want_accept: bool) -> (bool, Value) {
    let (sb, _, _) = instance(st, 0xA, plan);
    let (t, s) = inproc(st, &sb);
    let (v, _) = session(st, t, 0xA, plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    println!(
        "NEG\t{}",
        json!({"case": name, "maps_to": maps_to, "vus": st.vus, "gpu": plan.gpu, "expect": if want_accept {"accept"} else {"reject"},
               "accepted": acc, "pass": acc == want_accept, "why": v["why"], "prover_stopped": v["prover_stopped"],
               "reps_ok": reps, "prove_s": v["prove_s"], "calls": v["calls"], "link_mode": rec["link_mode"]})
    );
    (acc == want_accept, rec)
}

/// Every selftest case: (name, honest should be accepted, both reps must reject on their own).
const CASES: &[(&str, bool, bool)] = &[
    ("honest_linked_session", true, false),
    ("honest_linked_session_again", true, false),
    ("root_b_other_than_bound_root", false, false),
    ("roots_committed_twice", false, false),
    ("no_root_f", false, false),
    ("y_before_points", false, false),
    ("coins_before_y", false, false),
    ("y1_tampered", false, true),
    ("y2_tampered", false, true),
    ("y1_y2_tampered_same_delta", false, true),
    ("z_bit_differs_from_y_source", false, true),
    ("forged_middle_block_honest_endpoints", false, true),
    ("chunk_end_flag_dropped", false, true),
    ("wrong_counter", false, true),
    ("wrong_block_len", false, true),
    ("key_substituted", false, true),
    ("dummy_chunk_message_nonzero", false, true),
    ("public_chunk_value_forged", false, false),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
];

/// The GPU selftest runs every case in its own process: Flock-CUDA's live hook aborts the process when the verifier
/// refuses a coin (flock-live R6), which for a negative is the verifier rejecting it.
fn selftest_children(st: &ChunkStmt) {
    let exe = std::env::current_exe().unwrap();
    let mut all = true;
    for &(name, want_accept, want_both) in CASES {
        let out = std::process::Command::new(&exe)
            .args(["selftest", "--gpu", "--vus", &st.vus.to_string(), "--k", &st.k.to_string(), "--only", name])
            .output()
            .expect("spawn a selftest case");
        let text = String::from_utf8_lossy(&out.stdout);
        let line = text.lines().find(|l| l.starts_with("NEG\t") && l.contains(&format!("\"case\":\"{name}\"")));
        let (pass, v) = match line {
            Some(l) => {
                let v: Value = serde_json::from_str(&l[4..]).unwrap();
                let both = v["reps_ok"].as_array().map(|a| a.len() == 2 && a.iter().all(|r| r[1] == false)).unwrap_or(false);
                (v["pass"] == true && (!want_both || both), v)
            }
            None => {
                let err = String::from_utf8_lossy(&out.stderr);
                let refused = err.contains("the verifier refused a coin") || err.contains("LIVE-REFUSED");
                (!want_accept && refused, json!({"case": name, "gpu": true, "expect": "reject", "accepted": false,
                    "prover_aborted": err.lines().filter(|l| l.contains("LIVE-REFUSED") || l.contains("refused")).collect::<Vec<_>>().join(" | ")}))
            }
        };
        all &= pass;
        let mut v = v;
        v["pass"] = json!(pass);
        v["both_reps_required"] = json!(want_both);
        println!("NEG\t{v}");
    }
    println!("SELFTEST\t{}", json!({"vus": st.vus, "k": st.k, "chunks": st.chunks, "m": st.m, "m_link": st.m_link, "gpu": true,
        "mode": "one process per case", "all_pass": all}));
}

fn selftest(st: Arc<ChunkStmt>, gpu: bool) {
    let only = arg("--only", "");
    if gpu && only.is_empty() {
        return selftest_children(&st);
    }
    let h = || Plan { gpu, ..Plan::honest() };
    let all = std::cell::Cell::new(true);
    let t = |name: &str, maps: &str, p: Plan, acc: bool| -> Value {
        if !only.is_empty() && name != only {
            return Value::Null;
        }
        let (ok, rec) = neg(&st, name, maps, &p, acc);
        all.set(all.get() && ok);
        rec
    };
    let both = |rec: &Value| -> bool {
        rec["verdict"]["reps"].as_array().map(|a| a.len() == 2 && a.iter().all(|r| r["ok"] == false)).unwrap_or(false)
    };
    let honest = t("honest_linked_session", "honest", h(), true);
    let honest2 = t("honest_linked_session_again", "honest (fresh points)", h(), true);
    if !only.is_empty() {
        let replay = only == "cross_session_replay_both_reps";
        if replay {
            let (sb, _, _) = instance(&st, 0xA, &h());
            let (_, other) = session(&st, inproc(&st, &sb).0, 0xA, &h());
            t("cross_session_replay_both_reps", "link neg 15", Plan { substitute: Some(other), ..h() }, false);
        }
        let _ = (honest, honest2);
        for (name, maps, p, acc) in [
            ("root_b_other_than_bound_root", "L1", Plan { commit: CommitMode::OtherRootB, ..h() }, false),
            ("roots_committed_twice", "L1", Plan { commit: CommitMode::Twice, ..h() }, false),
            ("no_root_f", "L1 / C2", Plan { commit: CommitMode::NoRootF, ..h() }, false),
            ("y_before_points", "L1 order", Plan { commit: CommitMode::YBeforePoints, ..h() }, false),
            ("coins_before_y", "L1 / R5", Plan { commit: CommitMode::NoY, ..h() }, false),
            ("y1_tampered", "L2", Plan { y_delta: [1, 0], ..h() }, false),
            ("y2_tampered", "L2", Plan { y_delta: [0, 1 << 63], ..h() }, false),
            ("y1_y2_tampered_same_delta", "L2", Plan { y_delta: [1 << 40, 1 << 40], ..h() }, false),
            ("z_bit_differs_from_y_source", "link neg 2", Plan { flip_z_bit: true, ..h() }, false),
            ("forged_middle_block_honest_endpoints", "L4 chain", Plan { forge_middle: true, ..h() }, false),
            ("chunk_end_flag_dropped", "L4 flags", Plan { bad_params: Some(0), ..h() }, false),
            ("wrong_counter", "L4 counter", Plan { bad_params: Some(1), ..h() }, false),
            ("wrong_block_len", "L4 block_len", Plan { bad_params: Some(2), ..h() }, false),
            ("key_substituted", "L4 key", Plan { key_substituted: true, ..h() }, false),
            ("dummy_chunk_message_nonzero", "L4 dummy", Plan { dummy_message: true, ..h() }, false),
            ("public_chunk_value_forged", "C4", Plan { publics_flip: true, ..h() }, false),
            ("reps_with_different_witness", "R1", Plan { rep1_other_witness: true, ..h() }, false),
            ("fast_profile_proofs", "R7 / F2", Plan { profile: [LigeritoProfile::Fast; 2], ..h() }, false),
        ] {
            t(name, maps, p, acc);
        }
        return;
    }
    let fresh = honest["link"]["points"] != honest2["link"]["points"];
    println!("NEG\t{}", json!({"case": "points_fresh_per_session_same_roots", "maps_to": "link neg 11", "pass": fresh,
        "same_roots": honest["link"]["root_b_sha256"] == honest2["link"]["root_b_sha256"]}));
    all.set(all.get() && fresh);
    t("root_b_other_than_bound_root", "L1 (root_B is the root R1 binds)", Plan { commit: CommitMode::OtherRootB, ..h() }, false);
    t("roots_committed_twice", "L1 / link neg 11 (points drawn once)", Plan { commit: CommitMode::Twice, ..h() }, false);
    t("no_root_f", "L1 / C2", Plan { commit: CommitMode::NoRootF, ..h() }, false);
    t("y_before_points", "L1 order", Plan { commit: CommitMode::YBeforePoints, ..h() }, false);
    t("coins_before_y", "L1 / R5", Plan { commit: CommitMode::NoY, ..h() }, false);
    let mut ok_both = true;
    for (name, d) in [("y1_tampered", [1u64, 0]), ("y2_tampered", [0, 1 << 63]), ("y1_y2_tampered_same_delta", [1 << 40, 1 << 40])] {
        let r = t(name, "L2 / link neg 10", Plan { y_delta: d, ..h() }, false);
        ok_both &= both(&r);
    }
    let r = t("z_bit_differs_from_y_source", "link neg 2 (bit flip in z only)", Plan { flip_z_bit: true, ..h() }, false);
    ok_both &= both(&r);
    println!("NEG\t{}", json!({"case": "link_negatives_rejected_by_each_rep", "maps_to": "L2", "pass": ok_both}));
    all.set(all.get() && ok_both);
    let mut l4_both = true;
    for (name, maps, p) in [
        ("forged_middle_block_honest_endpoints", "L4 / link neg 13 (chain row)", Plan { forge_middle: true, ..h() }),
        ("chunk_end_flag_dropped", "L4 / link neg 14 (flags constant)", Plan { bad_params: Some(0), ..h() }),
        ("wrong_counter", "L4 / link neg 14 (counter claim)", Plan { bad_params: Some(1), ..h() }),
        ("wrong_block_len", "L4 / link neg 14 (block_len constant)", Plan { bad_params: Some(2), ..h() }),
        ("key_substituted", "L4 / C4 (key claim)", Plan { key_substituted: true, ..h() }),
        ("dummy_chunk_message_nonzero", "L4 / link (dummy chunks zero)", Plan { dummy_message: true, ..h() }),
    ] {
        let r = t(name, maps, p, false);
        l4_both &= both(&r);
    }
    println!("NEG\t{}", json!({"case": "l4_negatives_rejected_by_each_rep", "maps_to": "L4", "pass": l4_both}));
    all.set(all.get() && l4_both);
    t("public_chunk_value_forged", "L4 / C4 endpoints (refused at Commit)", Plan { publics_flip: true, ..h() }, false);
    t("reps_with_different_witness", "R1", Plan { rep1_other_witness: true, ..h() }, false);
    t("fast_profile_proofs", "R7 / F2", Plan { profile: [LigeritoProfile::Fast; 2], ..h() }, false);
    let (sb, _, _) = instance(&st, 0xA, &h());
    let (_, other) = session(&st, inproc(&st, &sb).0, 0xA, &h());
    t("cross_session_replay_both_reps", "link neg 15", Plan { substitute: Some(other), ..h() }, false);
    println!("SELFTEST\t{}", json!({"vus": st.vus, "k": st.k, "chunks": st.chunks, "m": st.m, "m_link": st.m_link, "gpu": gpu, "all_pass": all.get()}));
}

fn main() {
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let vus: usize = arg("--vus", "8").parse().unwrap();
    let k: usize = arg("--k", "1536").parse().unwrap();
    let seed: u64 = arg("--seed", "10").parse().unwrap();
    let tb = Instant::now();
    let st = Arc::new(ChunkStmt::new(vus, k));
    eprintln!("statement vus={vus} k={k} chunks={} nbl={} m={} m_link={} built in {:.2}s", st.chunks, st.nbl, st.m, st.m_link, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(st, flag("--gpu")),
        "serve" => {
            let d = leaf_digests(&st, &words(seed, st.chunks * 1024));
            serve_loop(move || server(&st, d.clone()));
        }
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let plan = Plan { gpu: flag("--gpu"), ..Plan::honest() };
            for run in 0..runs {
                let t = shared(TcpTransport::connect(&arg("--verifier", "127.0.0.1:7200")).unwrap());
                let (mut v, _) = session(&st, t, seed, &plan);
                v["run"] = json!(run);
                println!("LIVE\t{v}");
            }
        }
        _ => eprintln!("usage: flock-gpu-link selftest|serve|prove [--vus V] [--k K] [--seed S] [--gpu] ..."),
    }
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
