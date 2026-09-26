//! flock-netlist: `verity/flock-netlist/v1` (`flock_live::netlist`), proved by the CPU prover or by Flock-CUDA (`--gpu`,
//! feature `gpu`) in a flock-128-r2 live session: one commitment root, bound by both reps (R1). Lane flock-netlist.
//!
//!   flock-netlist serve    --listen ADDR --out DIR --instances PUB --netlist FILE --pin SHA256 [--sessions N] [--operator LANE]
//!   flock-netlist prove    --verifier ADDR --instances INST --netlist FILE [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-netlist selftest --instances INST --netlist FILE [--gpu] [--only CASE]
//!
//! The verifier loads only its own public file (`verity_flock.netlist.stage(..., public_only=True)`: roots, row digests,
//! outputs; no row) and the pinned netlist, from which it derives Δ, the regions and their values; it checks the frame-v3
//! roots and the pinned padding values before any session. Session: `Hello(Σ)`, `Commit(root_f = Σ, root)`, two verifier
//! points (the regions' claims), `Link(y = 0, 0)` (nothing is prover-supplied), Flock's coins.

use std::{
    collections::BTreeMap,
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
    netlist::*,
    zk_hooks::{Hooks, NoMask},
    *,
};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra, r1cs_hashes::blake3::Compression};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "nl";
const SIGMA_TAG: &[u8] = b"verity/flock-netlist/sigma/v1";

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
    format!("flock-netlist-v1/fast100x2/rep{rep}").into_bytes()
}

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: HashKind::Sha256 }
}

/// The backend identity the statement digest pins (so a NON_ZK and a ZK cell never share one).
fn identity(c: &Composite) -> Value {
    json!({"statement": STATEMENT, "backend": "C-interactive", "proof_class": "NON_ZK_PROOF", "profile": "fast100", "reps": 2,
           "commitments": "one root, both reps (R1)", "flavor": "rs", "merkle_hash": "sha256", "leaf_scheme": c.leaf_scheme,
           "hooks": NoMask::new().identity()})
}

fn sigma(st: &Stmt, inst: &Instances) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SIGMA_TAG);
    h.update(st.c.sha256);
    h.update(st.digest);
    h.update(inst.public_sha256);
    h.finalize().into()
}

struct NlVerifier {
    st: Arc<Stmt>,
    inst: Arc<Instances>,
}

impl StatementVerifier for NlVerifier {
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
            return Err("session: two points and y = 0 expected (nothing is prover-supplied)".into());
        }
        let extra = st.extra_claims(link.points, &self.inst);
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_extra(&st.r1cs, &commitment, &proof, &st.circuit, &pcs, &extra, ch)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn config(st: &Stmt, sigma: [u8; 32]) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.digest } }]);
    cfg.link = Some(LinkSpec { sigma, m: st.m_pts, points: 2 });
    cfg
}

fn server(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Server {
    Server::new(config(st, sigma(st, inst)), Box::new(NlVerifier { st: st.clone(), inst: inst.clone() }))
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness sources.
struct Wit {
    comps: Arc<dyn Fn(usize) -> Vec<Compression> + Sync + Send>,
    inputs: Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send>,
    mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send>,
    flip_z: Option<usize>,
    /// after the inputs converge: flip this bit of a unit's cut input (net, instance, column) — a tampered private cut word
    cut_flip: Option<(usize, usize, usize)>,
    /// evaluate these netlists instead of the pinned ones (a swapped tail)
    nets: Arc<BTreeMap<usize, &'static flock_live::ir_block::IrUnitNet>>,
}

type Witness = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

/// A rep's witness: the packed host witness, or (GPU) the device witness's inputs.
#[derive(Clone)]
enum WitData {
    Host(Witness),
    #[allow(dead_code)]
    Dev(Vec<Compression>, Vec<F128>),
}

fn witness(st: &Stmt, w: &Wit, device: bool) -> (WitData, f64) {
    let t = Instant::now();
    if device && w.flip_z.is_none() && st.c.wires.is_empty() {
        use rayon::prelude::*;
        let ur = st.c.unit_range();
        let comps: Vec<Compression> = (0..st.blocks()).into_par_iter().flat_map_iter(|b| (w.comps)(b)).collect();
        let inputs: Vec<F128> = (0..st.blocks() * ur.count).into_par_iter().flat_map_iter(|x| (w.inputs)(x / ur.count, x % ur.count)).collect();
        return (WitData::Dev(comps, inputs), t.elapsed().as_secs_f64());
    }
    let mut ins = st.converge(&*w.inputs, &w.nets);
    if let Some((n, x, col)) = w.cut_flip {
        let word = &mut ins[n][x][col / 128];
        if col % 128 < 64 { word.lo ^= 1 << (col % 128) } else { word.hi ^= 1 << (col % 128 - 64) }
    }
    let (mut z, a, b, zl) = st.witness(&*w.comps, &ins, &*w.mask, &w.nets);
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    let zl = if w.flip_z.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.c.k_log) } else { zl };
    (WitData::Host((z, a, b, zl)), t.elapsed().as_secs_f64())
}

fn prove_cpu(st: &Stmt, wt: &WitData, lig: LigeritoProfile, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    let WitData::Host((z, a, b, zl)) = wt.clone() else { panic!("the CPU prover takes a host witness") };
    let tp = Instant::now();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs, &params(st.m, lig), z, a, b, zl, &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(cm.cap.as_flattened()).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({"t.prove": tp.elapsed().as_secs_f64()}))
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    use flock_live::gpu_netlist::{NlSrc, prove_netlist};
    let src = match wt {
        WitData::Host((z, a, b, _)) => NlSrc::Host { z, a, b },
        WitData::Dev(c, i) => NlSrc::Device { comps: c, inputs: i },
    };
    let (proof, commitment, ph, wph) = prove_netlist(st, src, &params(st.m, lig), &domain(rep), hook, ch);
    let bk = json!({"t.witness": ph[0], "t.witness_comp": wph[0], "t.witness_units": wph[1], "t.encoding_commitment": ph[1], "t.link": ph[2],
                    "t.zerocheck": ph[3], "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                    "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6], "device_witness": matches!(wt, WitData::Dev(..))});
    eprintln!("GPUPHASE\t{bk}");
    (bincode::serialize(&(commitment, proof)).unwrap(), bk)
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone, Default)]
struct Plan {
    gpu: bool,
    no_y: bool,
    message_flip: bool,
    input_flip: bool,
    unit_flip: bool,
    chain_forged: bool,
    parent_forged: bool,
    key_substituted: bool,
    flags_forged: bool,
    mask_random: bool,
    cut_flip: bool,
    tail_swapped: bool,
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

fn rechain(cs: &mut [Compression], at: usize, len: usize) {
    let mut cv = cs[at].0;
    for c in &mut cs[at..at + len] {
        c.0 = cv;
        cv = flock_hash::blake3_compress(&c.0, &c.1, c.2, c.3, c.4)[..8].try_into().unwrap();
    }
}

fn wit(st: &Arc<Stmt>, inst: &Arc<Instances>, plan: &Plan, zero: bool) -> Wit {
    let mut rows = inst.rows.clone().expect("the prover's instance file has the rows");
    if plan.message_flip {
        rows[3] ^= 0x10;
    }
    let rows = Arc::new(rows);
    let (s1, r1) = (st.clone(), rows.clone());
    let plan1 = plan.clone();
    let first_run_len = st.c.ports[match st.c.ranges.iter().find_map(|r| if let Role::Runs { p, .. } = r.role { Some(p) } else { None }) { Some(p) => p, None => 0 }].nb;
    let comps: Arc<dyn Fn(usize) -> Vec<Compression> + Sync + Send> = Arc::new(move |b| {
        let mut cs = s1.compressions(Some(&r1), b);
        if b == 0 {
            if plan1.key_substituted {
                cs[0].0[0] ^= 1;
            }
            if plan1.flags_forged {
                cs[0].4 ^= 1;
                rechain(&mut cs, 0, first_run_len);
            }
            if plan1.chain_forged && first_run_len > 1 {
                cs[1].0[0] ^= 1;
                rechain(&mut cs, 1, first_run_len - 1);
            }
            if plan1.parent_forged
                && let Some(r) = s1.c.ranges.iter().find(|r| matches!(r.role, Role::Parents { .. }))
            {
                let idx = s1.c.ranges.iter().filter(|x| x.kind == Kind::Comp).flat_map(|x| (0..x.count).map(|s| x.slot(s))).collect::<std::collections::BTreeSet<_>>()
                    .into_iter().position(|col| col == r.slot(1)).unwrap();
                cs[idx].1[0] ^= 1;
            }
        }
        cs
    });
    let (s2, r2) = (st.clone(), rows.clone());
    let input_flip = plan.input_flip;
    let inputs: Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b, k| {
        let mut w = s2.unit_inputs(Some(&r2), b, k);
        if zero {
            w.iter_mut().for_each(|x| *x = F128::ZERO);
        }
        if input_flip && b == 0 && k == 1 {
            w[0].lo ^= 1 << 2;
        }
        w
    });
    let mask_words = 1usize << (st.c.meta["mask_log"].as_u64().unwrap_or(14) as usize - 7);
    let random = plan.mask_random;
    let hooks = Arc::new(Mutex::new(NoMask::new()));
    let mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b| {
        let mut h = hooks.lock().unwrap();
        if random { h.rng.f128s(flock_live::zk_hooks::STREAM_MASK, (b * mask_words) as u64, mask_words) } else { h.mask_fill(b, mask_words) }
    });
    let ur = st.c.unit_range();
    let unit = st.c.unit();
    let cut_flip = (plan.cut_flip && unit.in_groups.len() > 1).then(|| (st.c.unit_net, 0, unit.in_groups[1].col * 128 + 3));
    let nets = if plan.tail_swapped { forged_stages(st) } else { BTreeMap::new() };
    Wit { comps, inputs, mask, flip_z: plan.unit_flip.then_some(ur.slot(ur.count.min(6) - 1) + st.c.unit().in_words * 128 + 40), cut_flip,
          nets: Arc::new(nets) }
}

/// The tail stages of the forged netlist (`--forged`: the same template with every positive tail constant x 100, staged by
/// `verity_flock.netlist`), evaluated in place of the pinned stages: a swapped tail.
fn forged_stages(st: &Stmt) -> BTreeMap<usize, &'static flock_live::ir_block::IrUnitNet> {
    let text = std::fs::read_to_string(arg("--forged", "net-forged.txt")).expect("--forged netlist");
    let f: &'static Composite = Box::leak(Box::new(Composite::parse(&text, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e))));
    st.c.nets.iter().enumerate().filter(|(_, n)| n.0.starts_with("stage")).filter_map(|(i, n)| {
        let g = &f.nets.iter().find(|m| m.0 == n.0)?.1;
        (g.sha256 != n.1.sha256 && g.useful <= 1 << n.1.unit_log && g.in_words == n.1.in_words && g.out_cols == n.1.out_cols).then_some((i, g))
    }).collect()
}

fn session(st: &Arc<Stmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let sig = sigma(st, inst);
    let hello = config(st, sig).hello();
    let device = plan.gpu && !plan.mask_random && std::env::var("NL_HOST_WITNESS").is_err();
    let w0 = wit(st, inst, plan, false);
    let (wt0, tw0) = witness(st, &w0, device);
    let (wt1, tw1) = if plan.rep1_other_witness { witness(st, &wit(st, inst, plan, true), device) } else { (wt0.clone(), 0.0) };
    let witness_s = tw0 + tw1;
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
            let wt = if rep == 0 { &wt0 } else { &wt1 };
            let lig = if plan.fast { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
            let (p, bk) = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu { prove_gpu(st, wt, lig, rep, &mut hook, &mut lch) } else { prove_cpu(st, wt, lig, &mut hook, &mut lch) }
            })).map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            buckets.push(bk);
            proofs.push(p);
        }
        Ok(())
    })();
    let e2e_s = t0.elapsed().as_secs_f64();
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
    let c = &st.c;
    let ands = (c.unit().useful as u64, st.n as u64 * c.upv as u64);
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": STATEMENT, "scheme": "frame-v3/blake3-keyed",
               "prover": if plan.gpu {"flock-cuda-netlist"} else {"flock-cpu"}, "gpu": plan.gpu, "unit": c.name, "set": inst.header["set"],
               "subcircuit": inst.header["subcircuit"], "instances": st.n, "vus_per_block": c.g, "units": ands.1, "k_log": c.k_log, "nbl": st.nbl,
               "dense_m": st.m, "e2e_s": e2e_s, "witness_s": witness_s, "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(),
               "buckets": buckets, "wait_s": wait_s, "calls": calls, "wire_sent": ws, "wire_recv": wr,
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "sigma": hex(&sig), "roots": inst.header["frame_v3"]["roots"],
               "netlist_sha256": hex(&c.sha256), "statement_digest": hex(&st.digest), "public_sha256": hex(&inst.public_sha256),
               "identity": st.identity, "tp_s": tp0.elapsed().as_secs_f64()}),
        proofs,
    )
}

/// (case, expect accept, both reps must reject)
const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("mask_cells_random", true, false),
    ("output_claim_false", false, true),
    ("digest_claim_false", false, true),
    ("message_byte_differs_from_the_committed_row", false, true),
    ("unit_input_differs_from_its_message_bits", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("chaining_value_forged", false, true),
    ("parent_message_forged", false, true),
    ("key_substituted", false, true),
    ("flags_forged", false, true),
    ("coins_before_y", false, false),
    ("reps_with_different_witness", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("prover_file_differs_from_verifier_file", false, false),
    ("relabelled_netlist", false, false),
    ("relabelled_netlist_refused_at_load", false, false),
    ("verifier_evaluated_tail_refused", false, false),
    ("private_cut_word_tampered", false, true),
    ("tail_swapped", false, true),
    ("tail_swapped_netlist_refused", false, false),
];

fn skipped(st: &Stmt, name: &str) -> bool {
    let multi_chunk = st.c.ports.iter().any(|p| p.chunks > 1);
    let multi_block = st.c.ports.iter().any(|p| p.nb > 1);
    let cut = !st.c.wires.is_empty();
    (name == "parent_message_forged" && !multi_chunk) || (name == "chaining_value_forged" && !multi_block)
        || (!cut && ["private_cut_word_tampered", "tail_swapped", "tail_swapped_netlist_refused"].contains(&name))
}

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan { gpu, ..Plan::default() };
    match name {
        "mask_cells_random" => Plan { mask_random: true, ..h },
        "private_cut_word_tampered" => Plan { cut_flip: true, ..h },
        "tail_swapped" => Plan { tail_swapped: true, ..h },
        "message_byte_differs_from_the_committed_row" => Plan { message_flip: true, ..h },
        "unit_input_differs_from_its_message_bits" => Plan { input_flip: true, ..h },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..h },
        "chaining_value_forged" => Plan { chain_forged: true, ..h },
        "parent_message_forged" => Plan { parent_forged: true, ..h },
        "key_substituted" => Plan { key_substituted: true, ..h },
        "flags_forged" => Plan { flags_forged: true, ..h },
        "coins_before_y" => Plan { no_y: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { fast: true, ..h },
        _ => h,
    }
}

fn load_composite(path: &str) -> Composite {
    Composite::parse(&std::fs::read_to_string(path).expect("netlist"), Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e))
}

/// The verifier's file with one public word changed and its frame-v3 roots recomputed: a self-consistent, false statement.
fn false_statement(st: &Stmt, what: &str) -> Instances {
    let path = arg("--instances", "");
    let mut v = Instances::load(&path, &st.c).unwrap();
    match what {
        "output" => v.outputs[5] ^= 1,
        _ => v.digests[1][0] ^= 1,
    }
    let text = std::fs::read(&path).unwrap();
    let nl = text.iter().position(|&x| x == b'\n').unwrap();
    let head: Value = serde_json::from_slice(&text[..nl]).unwrap();
    let fv = &head["frame_v3"];
    let mut roots = fv["roots"].clone();
    let np = st.c.ports.len();
    for (p, port) in st.c.ports.iter().enumerate() {
        let dom = hexb(fv["domain_ids"][&port.name].as_str().unwrap()).unwrap();
        let leaves: Vec<[u8; 32]> = (0..v.n).map(|i| fv3_leaf(&dom, i, b"blake3-keyed/row/v2", &v.digest(i, p, np))).collect();
        roots[&port.name] = json!(hex(&fv3_root(&dom, leaves)));
    }
    let n_out: usize = st.c.out_words.iter().sum();
    let mut at = 0;
    for x in st.c.meta["out_ports"].as_array().unwrap() {
        let (name, w) = (x[0].as_str().unwrap(), x[1].as_u64().unwrap() as usize);
        let dom = hexb(fv["domain_ids"][name].as_str().unwrap()).unwrap();
        let words: Vec<u16> = (0..v.n).flat_map(|i| v.outputs[i * n_out + at..i * n_out + at + w].to_vec()).collect();
        let leaves = words.iter().enumerate().map(|(r, x)| fv3_leaf(&dom, r, b"u16", &x.to_be_bytes())).collect();
        roots[name] = json!(hex(&fv3_root(&dom, leaves)));
        at += w;
    }
    v.header["frame_v3"]["roots"] = roots;
    check_public(&st.c, &v).expect("the false statement is self-consistent");
    let mut h = Sha256::new();
    h.update(v.public_sha256);
    h.update(what.as_bytes());
    v.public_sha256 = h.finalize().into();
    v
}

fn run_case(st: &Arc<Stmt>, inst: &Arc<Instances>, name: &str, gpu: bool) -> Value {
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    if name == "relabelled_netlist_refused_at_load" {
        let text = std::fs::read_to_string(arg("--netlist", "")).unwrap();
        let other = relabel(&text);
        let c2 = Composite::parse(&other, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
        let refused_pin = hex(&c2.sha256) != hex(&st.c.sha256);
        let refused_file = Instances::load(&arg("--instances", ""), &c2).is_err();
        return json!({"case": name, "expect": "refused at load (pin and instance file)", "pass": refused_pin && refused_file,
                      "pin_differs": refused_pin, "file_refused": refused_file});
    }
    if name == "verifier_evaluated_tail_refused" {
        let text = std::fs::read_to_string(arg("--netlist", "")).unwrap();
        let (cut_line, meta_tail) = with_tail(&text);
        let r1 = Composite::parse(&cut_line, Some(&arg("--tables", "tables"))).err();
        let r2 = Composite::parse(&meta_tail, Some(&arg("--tables", "tables"))).err();
        return json!({"case": name, "expect": "refused at parse (a CUT line in a NET; a tail in META)", "pass": r1.is_some() && r2.is_some(),
                      "why_cut_line": r1, "why_meta_tail": r2});
    }
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan { gpu, ..Plan::default() }).1);
    }
    let (pst, pinst, vinst) = match name {
        "output_claim_false" | "digest_claim_false" => {
            let v = Arc::new(false_statement(st, if name.starts_with("output") { "output" } else { "digest" }));
            let mut p = Instances::load(&arg("--instances", ""), &st.c).unwrap();
            p.public_sha256 = v.public_sha256;
            (st.clone(), Arc::new(p), v)
        }
        "prover_file_differs_from_verifier_file" => {
            let mut p = Instances::load(&arg("--instances", ""), &st.c).unwrap();
            p.public_sha256[0] ^= 1;
            (st.clone(), Arc::new(p), inst.clone())
        }
        "tail_swapped_netlist_refused" => {
            let text = std::fs::read_to_string(arg("--forged", "net-forged.txt")).unwrap();
            let c2 = Composite::parse(&text, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
            let st2 = Arc::new(Stmt::new(c2, st.n, st.identity.clone()).unwrap_or_else(|e| refuse(e)));
            (st2, inst.clone(), inst.clone())
        }
        "relabelled_netlist" => {
            let text = std::fs::read_to_string(arg("--netlist", "")).unwrap();
            let c2 = Composite::parse(&relabel(&text), Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
            let st2 = Arc::new(Stmt::new(c2, st.n, st.identity.clone()).unwrap_or_else(|e| refuse(e)));
            (st2, inst.clone(), inst.clone())
        }
        _ => (st.clone(), inst.clone(), inst.clone()),
    };
    let s = Arc::new(Mutex::new(Server::new(config(st, sigma(st, &vinst)), Box::new(NlVerifier { st: st.clone(), inst: vinst.clone() }))));
    let (v, _) = session(&pst, &pinst, shared(InProc(s.clone())), &plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let pass = acc == want.1 && (!want.2 || both);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"],
           "witness_s": v["witness_s"]})
}

/// Two forms of the netlist that hand the verifier part of the computation: the unit NET with a `CUT` line (the flock-ir-frame
/// native tail: the 36-op RMSNorm row tail's shape, one F32Add over two cut words), and META with a `tail` program.
fn with_tail(text: &str) -> (String, String) {
    let lines: Vec<&str> = text.split('\n').collect();
    let at = lines.iter().position(|l| l.starts_with("NET unit ")).unwrap();
    let n: usize = lines[at].split_whitespace().nth(2).unwrap().parse().unwrap();
    let cut = r#"CUT {"in":[],"out":[],"tail":[[2,"F32Add_v1",0,1]],"words":2}"#;
    let mut a: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    a[at] = format!("NET unit {}", n + 1);
    a.insert(at + 1 + n, cut.to_string());
    let mut meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
    meta["tail"] = json!([[2, "F32Add_v1", 0, 1]]);
    let mut b: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    b[1] = format!("META {}", serde_json::to_string(&meta).unwrap());
    (a.join("\n"), b.join("\n"))
}

/// The same netlist with unit 0's first two leaf inputs swapped in its META: the rows are unchanged, the meaning is not.
fn relabel(text: &str) -> String {
    let mut lines: Vec<String> = text.split('\n').map(|s| s.to_string()).collect();
    let mut meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
    let l = meta["leaves_in"][0].as_array_mut().unwrap();
    l.swap(0, 1);
    lines[1] = format!("META {}", serde_json::to_string(&meta).unwrap());
    lines.join("\n")
}

fn selftest(st: &Arc<Stmt>, inst: &Arc<Instances>, gpu: bool) {
    let only = arg("--only", "");
    if !only.is_empty() {
        println!("NEG\t{}", run_case(st, inst, &only, gpu));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
        if skipped(st, name) {
            continue;
        }
        let v = if gpu {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["selftest", "--gpu", "--instances", &arg("--instances", ""), "--netlist", &arg("--netlist", ""), "--tables", &arg("--tables", "tables"),
                       "--forged", &arg("--forged", "net-forged.txt"), "--only", name])
                .output()
                .expect("spawn a case");
            let text = String::from_utf8_lossy(&out.stdout);
            match text.lines().find(|l| l.starts_with("NEG\t")) {
                Some(l) => serde_json::from_str(&l[4..]).unwrap(),
                None => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    let refused = err.contains("the verifier refused a coin") || err.contains("LIVE-REFUSED");
                    json!({"case": name, "gpu": true, "accepted": false, "pass": !want_accept && refused,
                           "prover_aborted": err.lines().filter(|l| l.contains("REFUSED") || l.contains("refused") || l.contains("panicked")).collect::<Vec<_>>().join(" | ")})
                }
            }
        } else {
            run_case(st, inst, name, gpu)
        };
        all &= v["pass"] == true;
        println!("NEG\t{v}");
    }
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "unit": st.c.name, "subcircuit": inst.header["subcircuit"], "instances": st.n,
        "vus_per_block": st.c.g, "k_log": st.c.k_log, "m": st.m, "gpu": gpu, "netlist_sha256": hex(&st.c.sha256), "all_pass": all}));
}

fn serve_loop(make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    let listen = arg("--listen", "127.0.0.1:7500");
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
    let c = load_composite(&arg("--netlist", "net.txt"));
    let pin = arg("--pin", "");
    if cmd == "serve" && pin.is_empty() {
        refuse("serve needs --pin (the pinned netlist's sha256)".into());
    }
    if !pin.is_empty() && hex(&c.sha256) != pin {
        refuse(format!("the netlist's sha256 {} is not the pinned netlist {pin}", hex(&c.sha256)));
    }
    let inst = Arc::new(Instances::load(&arg("--instances", "inst.bin"), &c).unwrap_or_else(|e| refuse(e)));
    if cmd == "serve" && inst.rows.is_some() {
        refuse("the verifier's instance file holds rows: stage its own public file (public_only)".into());
    }
    if let Err(e) = check_public(&c, &inst) {
        refuse(e);
    }
    let id = identity(&c);
    let st = Arc::new(Stmt::new(c, inst.n, id).unwrap_or_else(|e| refuse(e)));
    eprintln!("statement {} instances={} vus_per_block={} blocks={} k_log={} nbl={} m={} regions={} delta={} built in {:.2}s",
              st.c.name, st.n, st.c.g, st.blocks_real, st.c.k_log, st.nbl, st.m, st.regions.len(), st.circuit.da.len() / 2, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "serve" => serve_loop(move || server(&st, &inst)),
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            let plan = Plan { gpu: flag("--gpu"), ..Plan::default() };
            for run in 0..warm + runs {
                let mut tcp = TcpTransport::connect(&arg("--verifier", "127.0.0.1:7500")).unwrap();
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
        _ => eprintln!("usage: flock-netlist serve|prove|selftest --instances FILE --netlist FILE [--pin SHA256] ..."),
    }
}
