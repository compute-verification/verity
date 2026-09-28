//! flock-circuit: `verity/flock-circuit` (`flock_live::circuit`), proved by the CPU prover or by Flock-CUDA (`--gpu`,
//! feature `gpu`) in a flock-128-r2 live session: one commitment root, bound by both reps (R1). Lane flock-circuit.
//!
//!   flock-circuit serve    --listen ADDR --out DIR --instances PUB --circuit FILE --pin SHA256 [--sessions N] [--operator LANE]
//!   flock-circuit prove    --verifier ADDR --instances INST --circuit FILE [--gpu] [--runs N] [--warm W] [--dump DIR]
//!   flock-circuit selftest --instances INST --circuit FILE [--gpu] [--only CASE]
//!
//! The verifier loads only its own public file (`verity_flock.circuit.stage(..., public_only=True)`: roots, each row's `b || c`,
//! outputs; no row, no salt) and the pinned circuit, from which it derives Δ, the regions and their values; it checks the
//! frame-v3-sha512 roots and the pinned padding values before any session. Session: `Hello(Σ)`, `Commit(root_f = Σ, root)`, two verifier
//! points (the regions' claims), `Link(y = 0, 0)` (nothing is prover-supplied), Flock's coins.

use std::{
    collections::BTreeMap,
    sync::PoisonError,
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
    circuit::*,
    zk_hooks::{Hooks, NoMask, ProverSeed},
    *,
};
use flock_prover::{pcs::Commitment, prover::prove_fast_ligerito_from_witness_extra};
#[cfg(feature = "gpu")]
use flock_live::gpu_circuit::HostSlot;
use serde_json::{Value, json};
use sha2::{Digest, Sha512};

const TABLE: &str = "circuit";

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

fn domain(c: &Composite, rep: usize) -> Vec<u8> {
    format!("{}{rep}", c.tags().domain).into_bytes()
}

/// The Merkle hash of every Ligerito level: hm96-sha512/v1 leaves under SHA-512 nodes in a production build (feature `sha512`,
/// verity's Flock patch).
#[cfg(feature = "sha512")]
const MERKLE: HashKind = HashKind::Hm96Sha512;
#[cfg(not(feature = "sha512"))]
const MERKLE: HashKind = HashKind::Sha256;

fn params(m: usize, lig: LigeritoProfile) -> PcsParams {
    PcsParams { m, log_inv_rate: 1, log_batch_size: embedded_initial_k_or_default(m, lig), profile: lig, num_lanes: None, merkle_hash: MERKLE }
}

/// The backend identity the statement digest pins (so a NON_ZK and a ZK cell never share one).
fn identity(c: &Composite) -> Value {
    json!({"statement": c.statement(), "backend": "C-interactive", "proof_class": "NON_ZK_PROOF", "profile": "fast100", "reps": 2,
           "commitments": "one root, both reps (R1)", "flavor": "rs", "leaf_scheme": c.leaf_scheme,
           // a selftest build injects seeds (the determinism check); a proving build never has the feature
           "seed_injection": cfg!(feature = "seed-injection"),
           // the frame-v3 tree and the row leaf are the serving side's formats, not this statement's to choose
           "hashes": {"merkle": format!("{} (every Ligerito level)", MERKLE.as_str()),
                      "round_digest": "sha512 (framing: the coin server retains every round's bytes and the replay compares them)",
                      "statement_digest": "sha512", "sigma": "sha512", "coin_commitment": "sha512",
                      "coin_derivation": "sha256 (verity.randomness)", "frame_v3_tree": "sha512 (frame-v3-sha512)",
                      "transcript": "none (live verifier coins)", "row_leaf_in_circuit": "hm96-sha512/row/v1 over sha512/row/v1"},
           "hooks": NoMask::new(ProverSeed::from_os()).identity()})
}

/// Σ = SHA-512(the statement's Σ tag ‖ the circuit's SHA-512 ‖ the statement digest ‖ the public digest).
fn sigma(st: &Stmt, inst: &Instances) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(st.c.tags().sigma.as_bytes());
    h.update(st.c.sha512);
    h.update(st.digest);
    h.update(inst.public_sha512);
    h.finalize().into()
}

struct CircuitVerifier {
    st: Arc<Stmt>,
    inst: Arc<Instances>,
}

impl StatementVerifier for CircuitVerifier {
    fn domain(&self, _t: &str, rep: usize) -> Vec<u8> {
        domain(&self.st.c, rep)
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
        Ok(merkle_digest_bytes(&commitment.cap).to_vec())
    }
}

fn config(st: &Stmt, inst: &Instances) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.flock_digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma(st, inst).to_vec(), m: st.m_pts, points: 2 });
    cfg.coin_seed = true;
    cfg.retain_rounds = true;
    cfg.sha512_digests = true;
    cfg.unit_draw = inst.header.get("unit_draw").cloned();
    cfg
}

/// The drawn units' statement over a registered population (both sides derive it from the population file and the draw).
fn drawn(st: &Stmt, inst: &Instances, draw: &Value) -> Result<(Arc<Stmt>, Arc<Instances>), String> {
    let di = inst.drawn(&st.c, draw)?;
    check_public(&st.c, &di)?;
    let ds = Stmt::new(st.c.clone(), di.n, st.identity.clone())?;
    Ok((Arc::new(ds), Arc::new(di)))
}

/// A shared transport used as a plain one (so it can be wrapped).
struct SharedCall(SharedTransport);

impl Transport for SharedCall {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).call(req)
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).stats()
    }
    fn wait_s(&self) -> f64 {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).wait_s()
    }
}

/// Every coin response the prover received, in order (to check them against the seed the verifier reveals).
struct Logged<T: Transport> {
    inner: T,
    coins: Arc<Mutex<Vec<Vec<F128>>>>,
}

impl<T: Transport> Transport for Logged<T> {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let hello = matches!(req, Req::Hello(_));
        let r = self.inner.call(req)?;
        if let (Resp::Coins(c), false) = (&r, hello) {
            self.coins.lock().unwrap_or_else(PoisonError::into_inner).push(c.clone());
        }
        Ok(r)
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.inner.stats()
    }
    fn wait_s(&self) -> f64 {
        self.inner.wait_s()
    }
}

/// Step 0 checked by the prover: the revealed seed opens the commitment it received at `Hello`, and every coin it was sent
/// is the seed's derivation, in order.
fn check_coin_seed(commit: &[F128], hello: &str, verdict: &Value, coins: &[Vec<F128>]) -> Result<(), String> {
    use flock_live::coin_seed;
    let cs = &verdict["coin_seed"];
    let h32 = |k: &str| -> Result<[u8; 32], String> {
        flock_live::circuit::hexb(cs[k].as_str().ok_or(format!("verdict: coin_seed.{k} missing"))?)?.try_into().map_err(|_| format!("coin_seed.{k} length"))
    };
    let (seed, nonce) = (h32("seed")?, h32("nonce")?);
    let c: Vec<u8> = commit.iter().flat_map(|w| [w.lo.to_le_bytes(), w.hi.to_le_bytes()].concat()).collect();
    if commit.len() != 4 || coin_seed::commitment(&seed, &nonce).to_vec() != c {
        return Err("step 0: the revealed seed does not open the commitment sent at Hello".into());
    }
    let key = coin_seed::coin_key(&seed, hello);
    for (k, got) in coins.iter().enumerate() {
        let want = coin_seed::coins(&key, k as u64, got.len());
        let g: Vec<u8> = got.iter().flat_map(|w| [w.lo.to_le_bytes(), w.hi.to_le_bytes()].concat()).collect();
        if g != want {
            return Err(format!("step 0: coin request {k} is not the committed seed's derivation"));
        }
    }
    Ok(())
}

fn server(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Server {
    let s = Server::new(config(st, inst), Box::new(CircuitVerifier { st: st.clone(), inst: inst.clone() }));
    let file = arg("--draw-file", "");
    if !file.is_empty() {
        // the verifier's own draw, made outside this server: served at Register as it is (checked there, U2, and against the
        // registered population when the drawn statement is built)
        let draw: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap_or_else(|e| refuse(format!("--draw-file: {e}"))))
            .unwrap_or_else(|e| refuse(format!("--draw-file: {e}")));
        flock_live::unit_draw::check(&draw).unwrap_or_else(|e| refuse(e));
        let mut s = with_draw(s, st, inst, flock_live::unit_draw::Law::Subset { k: 0 });
        s.cfg.draw_law = None;
        s.cfg.draw_given = Some(draw);
        return s;
    }
    let law = arg("--draw", "");
    if law.is_empty() {
        return s;
    }
    with_draw(s, st, inst, flock_live::unit_draw::Law::parse(&law).unwrap_or_else(|e| refuse(e)))
}

/// A server that draws `law` over the registered population at `Register` and serves the drawn units' statement.
fn with_draw(mut s: Server, st: &Arc<Stmt>, inst: &Arc<Instances>, law: flock_live::unit_draw::Law) -> Server {
    s.cfg.draw_law = Some((law, inst.n as u64));
    let (st, inst) = (st.clone(), inst.clone());
    s.with_draw(Box::new(move |draw: &Value| {
        let (ds, di) = drawn(&st, &inst, draw)?;
        Ok((config(&ds, &di), Box::new(CircuitVerifier { st: ds, inst: di }) as Box<dyn StatementVerifier>))
    }))
}

type Hook<'a> = &'a mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>;

/// A (possibly dishonest) prover's witness sources.
struct Wit {
    /// block `b`'s row slots' inputs (`Stmt::row_inputs`)
    rowin: Arc<dyn Fn(usize) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) + Sync + Send>,
    inputs: Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send>,
    mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send>,
    flip_z: Option<usize>,
    /// after the inputs converge: flip this bit of a unit's cut input (net, instance, column) — a tampered private cut word
    cut_flip: Option<(usize, usize, usize)>,
    /// a typed template's: a part's bound input row (root instance, unit coordinate) flipped as it's evaluated
    typed_flip: Option<(usize, usize)>,
    /// a typed template's swapped tail: the forged statement's instance evaluates every instance
    typed_forged: Option<Arc<flock_live::typed::Instance>>,
    /// evaluate these circuits instead of the pinned ones (a swapped tail)
    nets: Arc<BTreeMap<usize, &'static flock_live::ir_block::IrUnitNet>>,
}

type Witness = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

/// A rep's witness: the packed host witness, or (GPU) the device witness's inputs.
#[derive(Clone)]
enum WitData {
    Host(Witness),
    /// the device witness's inputs: every compression slot's input words, the hm96 slots (host slots), every unit's input words
    #[cfg(feature = "gpu")]
    Dev(Vec<F128>, Vec<HostSlot>, Vec<F128>),
    /// a cut template on the device: as `Dev` (the late words come from the host tail between the units' passes)
    #[cfg(feature = "gpu")]
    DevCut(Vec<F128>, Vec<HostSlot>, Vec<F128>, Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send>),
}

/// A rep's witness: for an honest prover on the GPU (`device`), the device witness's inputs (the compression slots' and the
/// units' input words, the hm96 slots computed here); otherwise the packed host witness.
fn witness(st: &Stmt, w: &Wit, device: bool) -> (WitData, f64) {
    let t = Instant::now();
    #[cfg(feature = "gpu")]
    // a tail that returns the outputs interleaves with the units round by round: the device's two passes cannot follow it
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && st.c.out_net == st.c.unit_net && st.c.typed.is_none() {
        use rayon::prelude::*;
        let ur = st.c.unit_range();
        let ri: Vec<_> = (0..st.blocks()).into_par_iter().map(|b| (w.rowin)(b)).collect();
        let rows: Vec<F128> = ri.iter().flat_map(|x| x.0.iter().flatten().copied()).collect();
        let hm: Vec<Vec<F128>> = ri.into_iter().flat_map(|x| x.1).collect();
        let host = flock_live::gpu_circuit::host_slots(st, st.c.hm_net, &hm);
        let inputs: Vec<F128> = (0..st.blocks() * ur.count).into_par_iter().flat_map_iter(|x| (w.inputs)(x / ur.count, x % ur.count)).collect();
        if st.c.wires.iter().any(|x| x.dst.0 != st.c.sha_net && x.dst.0 != st.c.hm_net) {
            return (WitData::DevCut(rows, host, inputs, w.inputs.clone()), t.elapsed().as_secs_f64());
        }
        return (WitData::Dev(rows, host, inputs), t.elapsed().as_secs_f64());
    }
    #[cfg(feature = "gpu")]
    // a tail that returns the outputs (a chain of units, the tail's stages): the host converges every slot's inputs and computes
    // the stages and lookups; the device the compressions, and the units from their converged inputs
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && st.c.typed.is_none() {
        use rayon::prelude::*;
        let c = &st.c;
        let ri: Vec<_> = (0..st.blocks()).into_par_iter().map(|b| (w.rowin)(b)).collect();
        let rows: Vec<F128> = ri.iter().flat_map(|x| x.0.iter().flatten().copied()).collect();
        let hm: Vec<Vec<F128>> = ri.into_iter().flat_map(|x| x.1).collect();
        let ins = st.converge_with(&*w.inputs, None, &w.nets, None);
        let mut host = flock_live::gpu_circuit::host_slots(st, c.hm_net, &hm);
        for ni in (0..c.nets.len()).filter(|&n| n != c.sha_net && n != c.hm_net && n != c.unit_net) {
            host.extend(flock_live::gpu_circuit::host_slots(st, ni, &ins[ni]));
        }
        let inputs: Vec<F128> = ins[c.unit_net].iter().flatten().copied().collect();
        return (WitData::Dev(rows, host, inputs), t.elapsed().as_secs_f64());
    }
    let _ = device;
    let mut ins = st.converge(&*w.inputs, &*w.rowin, &w.nets);
    if let Some((n, x, col)) = w.cut_flip {
        let word = &mut ins[n][x][col / 128];
        if col % 128 < 64 { word.lo ^= 1 << (col % 128) } else { word.hi ^= 1 << (col % 128 - 64) }
    }
    let (mut z, a, b, zl) = st.witness_with(&ins, &*w.mask, &w.nets, w.typed_flip, w.typed_forged.as_deref());
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    let zl = if w.flip_z.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.c.k_log) } else { zl };
    (WitData::Host((z, a, b, zl)), t.elapsed().as_secs_f64())
}

fn prove_cpu(st: &Stmt, wt: WitData, lig: LigeritoProfile, hook: Hook<'_>, ch: &mut LiveChallenger) -> (Vec<u8>, Value) {
    #[allow(irrefutable_let_patterns)]
    let WitData::Host((z, a, b, zl)) = wt else { panic!("the CPU prover takes a host witness") };
    let tp = Instant::now();
    let (proof, commitment) = prove_fast_ligerito_from_witness_extra(
        &st.r1cs, &params(st.m, lig), z, a, b, zl, &st.circuit,
        &mut |cm: &Commitment| -> Vec<Vec<F128>> { hook(merkle_digest_bytes(&cm.cap)).into_iter().map(|p| p[6..].to_vec()).collect() },
        ch,
    );
    (bincode::serialize(&(commitment, proof)).unwrap(), json!({"t.prove": tp.elapsed().as_secs_f64()}))
}

#[cfg(feature = "gpu")]
fn prove_gpu(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger,
             salts: &Mutex<flock_live::zk_hooks::LeafSalts>) -> (Vec<u8>, Value) {
    use flock_live::gpu_circuit::{CircuitSrc, prove_circuit};
    let sched = flock_live::gpu_circuit::schedule(st.c.unit());
    let (cut_rows, late_rows): (Vec<usize>, Vec<usize>) = (sched.cut_rows.iter().map(|&r| r as usize).collect(), sched.late_rows.iter().map(|&r| r as usize).collect());
    let mut mid_fn;
    let src = match wt {
        WitData::Host((z, a, b, _)) => CircuitSrc::Host { z, a, b },
        WitData::Dev(r, h, i) => CircuitSrc::Device { rows: r, host: h, inputs: i },
        WitData::DevCut(r, h, i, ui) => {
            mid_fn = move |cut: &[u32]| st.host_tail(&**ui, cut, &cut_rows, &late_rows);
            CircuitSrc::DeviceCut { rows: r, host: h, inputs: i, mid: &mut mid_fn }
        }
    };
    let (proof, commitment, ph, wph) = prove_circuit(st, src, &params(st.m, lig), &domain(&st.c, rep), hook, ch, Some(salts));
    let bk = json!({"t.witness": ph[0], "t.witness_comp": wph[0], "t.witness_units": wph[1], "t.encoding_commitment": ph[1], "t.link": ph[2],
                    "t.zerocheck": ph[3], "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
                    "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6], "device_witness": !matches!(wt, WitData::Host(..))});
    eprintln!("GPUPHASE\t{bk}");
    (bincode::serialize(&(commitment, proof)).unwrap(), bk)
}

#[cfg(not(feature = "gpu"))]
fn prove_gpu(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger,
             _: &Mutex<flock_live::zk_hooks::LeafSalts>) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[derive(Clone, Default)]
struct Plan {
    gpu: bool,
    no_y: bool,
    message_flip: bool,
    input_flip: bool,
    unit_flip: bool,
    /// row witness tampers (`RowTamper`)
    chain_forged: bool,
    salt_forged: bool,
    midstate_forged: bool,
    pad_forged: bool,
    mask_random: bool,
    cut_flip: bool,
    tail_swapped: bool,
    rep1_other_witness: bool,
    /// the cheating path: send rep 1 even when its level-0 root is not rep 0's (the verifier's R1 refuses it)
    skip_level0_check: bool,
    /// prove the registered population's statement whatever units the verifier drew
    ignore_draw: bool,
    fast: bool,
    /// send each rep's proof with one byte of its first opened hm96 salt flipped
    salt_flip: bool,
    substitute: Option<Vec<Vec<u8>>>,
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap_or_else(PoisonError::into_inner).call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn wit(st: &Arc<Stmt>, inst: &Arc<Instances>, plan: &Plan, zero: bool, seed: ProverSeed) -> Wit {
    let mut rows = inst.instance_rows(&st.c).expect("the prover's instance file has the rows");
    if plan.message_flip {
        rows[3] ^= 0x10;
    }
    let rows = Arc::new(rows);
    let salts = Arc::new(inst.instance_salts(st.c.ports.len()).expect("the prover's instance file has the salts"));
    let (s1, r1, sa1) = (st.clone(), rows.clone(), salts.clone());
    let tamper = RowTamper { cv_flip: plan.chain_forged, midstate_flip: plan.midstate_forged, pad_flip: plan.pad_forged, salt_flip: plan.salt_forged };
    let rowin: Arc<dyn Fn(usize) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) + Sync + Send> = Arc::new(move |b| {
        if zero { s1.row_inputs(None, None, b, tamper) } else { s1.row_inputs(Some(&r1), Some(&sa1), b, tamper) }
    });
    let (s2, r2) = (st.clone(), rows.clone());
    let input_flip = plan.input_flip;
    // unit 1 of block 0, or unit 0 when a block holds one unit (else the case would prove an honest witness)
    let flip_k = st.c.unit_range().count.min(2) - 1;
    let inputs: Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b, k| {
        let mut w = s2.unit_inputs(Some(&r2), b, k);
        if zero {
            w.iter_mut().for_each(|x| *x = F128::ZERO);
        }
        if input_flip && b == 0 && k == flip_k {
            w[0].lo ^= 1 << 2;
        }
        w
    });
    let mask_words = 1usize << (st.c.meta["mask_log"].as_u64().unwrap_or(14) as usize - 7);
    let random = plan.mask_random;
    let hooks = Arc::new(Mutex::new(NoMask::new(seed)));
    let mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b| {
        let mut h = hooks.lock().unwrap_or_else(PoisonError::into_inner);
        if random { h.rng.f128s(flock_live::zk_hooks::STREAM_MASK, (b * mask_words) as u64, mask_words) } else { h.mask_fill(b, mask_words) }
    });
    let ur = st.c.unit_range();
    let unit = st.c.unit();
    let cut_flip = (plan.cut_flip && unit.in_groups.len() > 1 && st.c.typed.is_none()).then(|| (st.c.unit_net, 0, unit.in_groups[1].col * 128 + 3));
    // a typed template's cut words are its parts' bound inputs: the first one of instance 0
    let typed_flip = st.c.typed.as_ref().filter(|_| plan.cut_flip).and_then(|t| t.inst.bind.iter().flatten().next()).map(|b| (0, b.0));
    let nets = if plan.tail_swapped { forged_stages(st) } else { BTreeMap::new() };
    // a unit's internal row; a typed template's first AND row, the root's (else a part's)
    let flip_z = match &st.c.typed {
        Some(t) => t.inst.first_and().map(|u| t.col(&st.c, ur.count.min(6) - 1, u)),
        None => Some(ur.slot(ur.count.min(6) - 1) + st.c.unit().in_words * 128 + 40),
    };
    let typed_forged = if plan.tail_swapped { forged_instance(st) } else { None };
    Wit { rowin, inputs, mask, flip_z: flip_z.filter(|_| plan.unit_flip), cut_flip, typed_flip, typed_forged, nets: Arc::new(nets) }
}

/// A typed template's forged instance (`--forged`, derived once): its parts where the pinned instance's are and its root
/// inside the pinned root's slot, so it evaluates in the pinned block; `None` if the file isn't such a typed statement.
fn forged_instance(st: &Stmt) -> Option<Arc<flock_live::typed::Instance>> {
    static FORGED: std::sync::OnceLock<Option<Arc<flock_live::typed::Instance>>> = std::sync::OnceLock::new();
    let pinned = &st.c.typed.as_ref()?.inst;
    FORGED.get_or_init(|| {
        let text = std::fs::read_to_string(arg("--forged", "circuit-forged.txt")).ok()?;
        let meta: Value = serde_json::from_str(text.split('\n').nth(1)?.strip_prefix("META ")?).ok()?;
        let (f, _) = flock_live::typed::instance(&meta, Some(&arg("--tables", "tables"))).ok()?;
        let fits = f.parts == pinned.parts && f.extent == pinned.extent && f.root.rows.len() <= 1 << st.c.unit().unit_log
            && f.root.in_cols == pinned.root.in_cols && f.root.rows != pinned.root.rows;
        fits.then(|| Arc::new(f))
    }).clone()
}

/// The tail stages of the forged circuit (`--forged`: the same template with every positive tail constant x 4, staged by
/// `verity_flock.circuit`), evaluated in place of the pinned stages: a swapped tail.
fn forged_stages(st: &Stmt) -> BTreeMap<usize, &'static flock_live::ir_block::IrUnitNet> {
    let text = std::fs::read_to_string(arg("--forged", "circuit-forged.txt")).expect("--forged circuit");
    let f: &'static Composite = Box::leak(Box::new(Composite::parse(&text, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e))));
    st.c.nets.iter().enumerate().filter(|(_, n)| n.0.starts_with("stage")).filter_map(|(i, n)| {
        let g = &f.nets.iter().find(|m| m.0 == n.0)?.1;
        (g.sha256 != n.1.sha256 && g.useful <= 1 << n.1.unit_log && g.in_words == n.1.in_words && g.out_cols == n.1.out_cols).then_some((i, g))
    }).collect()
}

fn session(st: &Arc<Stmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan, seed: ProverSeed) -> (Value, Vec<Vec<u8>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let t: SharedTransport = shared(Logged { inner: SharedCall(t), coins: log.clone() });
    let t0 = Instant::now();
    // registration done: the verifier draws the units (or none), and the statement is the drawn units'
    let drawn_st = match call(&t, Req::Register) {
        Ok(Resp::Draw(d)) if d == "null" || plan.ignore_draw => None,
        Ok(Resp::Draw(d)) => {
            let draw: Value = serde_json::from_str(&d).unwrap_or(Value::Null);
            match drawn(st, inst, &draw) {
                Ok(x) => Some(x),
                Err(e) => return (json!({"accepted": false, "why": e, "prover_stopped": format!("the drawn statement: {e}")}), Vec::new()),
            }
        }
        other => return (json!({"accepted": false, "why": format!("register: {other:?}"), "prover_stopped": "register refused"}), Vec::new()),
    };
    let (st, inst) = match &drawn_st { Some((a, b)) => (a, b), None => (st, inst) };
    let sig = sigma(st, inst);
    let hello = config(st, inst).hello();
    let mut commit: Vec<F128> = Vec::new();
    let device = plan.gpu && !plan.mask_random && std::env::var("FC_HOST_WITNESS").is_err();
    let w0 = wit(st, inst, plan, false, seed);
    let (wt0, tw0) = witness(st, &w0, device);
    // rep 1 proves from rep 0's witness unless it has its own: the CPU prover consumes a witness, so rep 0 takes a copy and rep 1
    // the original (two copies at most, never one per rep plus the original)
    let (mut wt0, (mut wt1, tw1)) = (Some(wt0), if plan.rep1_other_witness {
        let (w, t) = witness(st, &wit(st, inst, plan, true, seed), device);
        (Some(w), t)
    } else {
        (None, 0.0)
    });
    let witness_s = tw0 + tw1;
    let mut prove_s = Vec::new();
    let mut buckets = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let mut root0: Option<Vec<u8>> = None;
    let mut stop_why: Option<String> = None;
    let tp0 = Instant::now();
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello.clone()))? {
            Resp::Coins(c) if c.len() == 4 => commit = c,
            r => return Err(format!("hello: step 0 expects the verifier's seed commitment, got {r:?}")),
        }
        let lig = if plan.fast { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
        // hm96-sha512 leaf salts: one ChaCha20 key per proof; the level-0 tree is id 0 (both reps bind one root), every later
        // tree a fresh id. The device expands its own trees' salts; a CPU proof takes host expansions.
        let salts = Arc::new(Mutex::new(flock_live::zk_hooks::LeafSalts::for_seed(&seed)));
        #[cfg(feature = "sha512")]
        let level0 = (!plan.gpu).then(|| Arc::new(salts.lock().unwrap_or_else(PoisonError::into_inner).expand(0, params(st.m, lig).n_leaves())));
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(&st.c, rep))?;
            let tt = t.clone();
            let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                // red-team-hm96 C1: both reps' level-0 trees take salt stream 0, so they must be one tree; stop before rep 1
                // opens anything (its openings would reveal id-0 salts over other leaves)
                match &root0 {
                    None => root0 = Some(root.to_vec()),
                    Some(r0) if r0.as_slice() != root && !plan.skip_level0_check => {
                        // the device prover's callback swallows the panic, so the reason is kept here too
                        let why = "PROVER-STOPPED: rep 1's level-0 root is not rep 0's; the shared level-0 salts would open over two trees";
                        stop_why = Some(why.into());
                        panic!("{why}")
                    }
                    _ => {}
                }
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
            // FC_WITNESS_PER_REP: rep 1 rebuilds its witness rather than rep 0 proving from a copy (one copy at a time)
            let per_rep = std::env::var("FC_WITNESS_PER_REP").is_ok();
            let own = |wt0: &mut Option<WitData>, wt1: &mut Option<WitData>| -> WitData {
                match (rep, wt1.is_some()) {
                    (0, true) => wt0.take(),
                    (0, false) if per_rep => wt0.take(),
                    (0, false) => wt0.clone(),
                    _ => wt1.take().or_else(|| wt0.take()).or_else(|| Some(witness(st, &w0, device).0)),
                }.expect("a rep's witness")
            };
            #[cfg(feature = "sha512")]
            if let Some(l0) = &level0 {
                let s = salts.clone();
                flock_merkle::hm96::begin(l0.clone(), Box::new(move |n| {
                    let mut g = s.lock().unwrap_or_else(PoisonError::into_inner);
                    let id = g.next_id();
                    g.expand(id, n)
                }));
            }
            let proved = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu {
                    let wt = if rep == 0 { wt0.as_ref() } else { wt1.as_ref().or(wt0.as_ref()) }.expect("a rep's witness");
                    prove_gpu(st, wt, lig, rep, &mut hook, &mut lch, &salts)
                } else {
                    prove_cpu(st, own(&mut wt0, &mut wt1), lig, &mut hook, &mut lch)
                }
            }));
            #[cfg(feature = "sha512")]
            if level0.is_some() {
                flock_merkle::hm96::end();
            }
            let (p, bk) = proved.map_err(|e| stop_why.clone().unwrap_or_else(|| panic_msg(e)))?;
            prove_s.push(tp.elapsed().as_secs_f64());
            buckets.push(bk);
            proofs.push(p);
        }
        Ok(())
    })();
    let e2e_s = t0.elapsed().as_secs_f64();
    let wait_e2e_s = t.lock().unwrap_or_else(PoisonError::into_inner).wait_s();
    if let Err(e) = res {
        let v = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        return (json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "prover_stopped": e.chars().take(200).collect::<String>()}), proofs);
    }
    let mut sent = plan.substitute.clone().unwrap_or_else(|| proofs.clone());
    if plan.salt_flip {
        sent = sent.iter().map(|p| flip_salt(p)).collect();
    }
    for (rep, p) in sent.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: p.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (ws, wr, calls) = t.lock().unwrap_or_else(PoisonError::into_inner).stats();
    let wait_s = t.lock().unwrap_or_else(PoisonError::into_inner).wait_s();
    let step0 = if v["accepted"] == true { check_coin_seed(&commit, &hello, &v, &log.lock().unwrap_or_else(PoisonError::into_inner)) } else { Ok(()) };
    let v = if let Err(e) = &step0 { json!({"accepted": false, "why": e, "verdict": v}) } else { v };
    let c = &st.c;
    let (rows_per_vu, ands_per_vu) = c.sizes_per_vu();
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": c.statement(), "scheme": SCHEME,
               "prover": if plan.gpu {"flock-cuda-circuit"} else {"flock-cpu"}, "gpu": plan.gpu, "unit": c.name, "set": inst.header["set"],
               "subcircuit": inst.header["subcircuit"], "instances": st.n, "vus_per_block": c.g, "units": st.n * c.upv, "rows_per_vu": rows_per_vu,
               "ands_per_vu": ands_per_vu, "k_log": c.k_log, "nbl": st.nbl,
               "dense_m": st.m, "e2e_s": e2e_s, "witness_s": witness_s, "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(),
               "buckets": buckets, "wait_s": wait_s, "wait_e2e_s": wait_e2e_s, "calls": calls, "wire_sent": ws, "wire_recv": wr,
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "sigma": hex(&sig), "roots": inst.header["frame_v3"]["roots"],
               "circuit_sha512": hex(&c.sha512), "statement_digest": hex(&st.digest), "public_sha512": hex(&inst.public_sha512),
               "unit_draw": inst.header.get("unit_draw"),
               "identity": st.identity, "tp_s": tp0.elapsed().as_secs_f64(), "coin_seed": v["coin_seed"].clone(),
               "step0_checked": step0.is_ok() && v["accepted"] == true}),
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
    ("row_ref_claim_false", false, true),
    ("message_byte_differs_from_the_committed_row", false, true),
    ("unit_input_differs_from_its_message_bits", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("chaining_value_forged", false, true),
    ("row_salt_forged", false, true),
    ("row_midstate_forged", false, true),
    ("padding_block_forged", false, true),
    ("coins_before_y", false, false),
    ("reps_with_different_witness", false, false),
    ("reps_differ_prover_stops", false, false),
    ("fast_profile_proofs", false, false),
    ("cross_session_replay_both_reps", false, false),
    ("prover_file_differs_from_verifier_file", false, false),
    ("relabelled_circuit", false, false),
    ("relabelled_circuit_refused_at_load", false, false),
    ("verifier_evaluated_tail_refused", false, false),
    ("leaf_salts_from_prover_seed_refused", false, false),
    ("device_salts_match_host", true, false),
    ("leaf_key_witness_refused", false, false),
    ("typed_statement_altered_refused", false, false),
    ("private_cut_word_tampered", false, true),
    ("tail_swapped", false, true),
    ("tail_swapped_circuit_refused", false, false),
    ("coin_seed_opening_forged", false, false),
    ("opened_salt_altered", false, true),
    ("coin_seed_record_replays", true, false),
    ("record_replays_offline", true, false),
    ("unit_draw_session", true, false),
    ("unit_draw_from_file", true, false),
    ("unit_draw_ignored_refused", false, false),
    ("prover_is_deterministic", true, false),
];

fn skipped(st: &Stmt, name: &str) -> bool {
    let multi_block = st.c.ports.iter().any(|p| p.comps > 1);
    let cut = st.c.wires.iter().any(|w| w.dst.0 != st.c.sha_net && w.dst.0 != st.c.hm_net)
        || st.c.typed.as_ref().is_some_and(|t| t.inst.bind.iter().any(|b| !b.is_empty()));
    // a tail with no positive constant to forge (GEMM's: the zero accumulator and the cast) forges to itself: nothing to swap
    let forged_same = std::fs::read_to_string(arg("--forged", "circuit-forged.txt"))
        .map(|f| f == std::fs::read_to_string(arg("--circuit", "")).unwrap_or_default()).unwrap_or(true);
    (name == "chaining_value_forged" && !multi_block)
        || (!cut && ["private_cut_word_tampered", "tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
        || (forged_same && ["tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
        || (name == "tail_swapped" && st.c.typed.is_some() && forged_instance(st).is_none())
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
        "row_salt_forged" => Plan { salt_forged: true, ..h },
        "row_midstate_forged" => Plan { midstate_forged: true, ..h },
        "padding_block_forged" => Plan { pad_forged: true, ..h },
        "coins_before_y" => Plan { no_y: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, skip_level0_check: true, ..h },
        "reps_differ_prover_stops" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { fast: true, ..h },
        "opened_salt_altered" => Plan { salt_flip: true, ..h },
        _ => h,
    }
}

fn load_composite(path: &str) -> Composite {
    Composite::parse(&std::fs::read_to_string(path).expect("circuit"), Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e))
}

/// The verifier's file with one public claim changed and its frame-v3 roots recomputed: a self-consistent, false statement. `ref`
/// (a shared-row file) points instance 0 at another row of a table, whose roots are unchanged.
fn false_statement(st: &Stmt, what: &str) -> Instances {
    let path = arg("--instances", "");
    let mut v = Instances::load(&path, &st.c).unwrap();
    let np = st.c.ports.len();
    match what {
        "output" => {
            let k = 5.min(v.outputs.len() - 1);
            v.outputs[k] ^= 1
        }
        "ref" => {
            let t = v.tables.as_mut().expect("a shared-row file");
            let p = (0..np).find(|&p| t.counts[p] > 1).expect("a table of two rows or more");
            t.refs[p] = (t.refs[p] + 1) % t.counts[p] as u32;
        }
        _ => {
            let k = if v.tables.is_some() { v.bc_index(0, 0, np) } else { 1 };
            v.bcs[k][0] ^= 1
        }
    }
    for (name, root) in frame_roots(&st.c, &v).unwrap() {
        v.header["frame_v3"]["roots"][&name] = json!(hex(&root));
    }
    check_public(&st.c, &v).expect("the false statement is self-consistent");
    let mut h = Sha512::new();
    h.update(v.public_sha512);
    h.update(what.as_bytes());
    v.public_sha512 = h.finalize().into();
    v
}

fn run_case(st: &Arc<Stmt>, inst: &Arc<Instances>, name: &str, gpu: bool) -> Value {
    let want = CASES.iter().find(|c| c.0 == name).expect("case");
    if name == "relabelled_circuit_refused_at_load" {
        let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
        let other = relabel(&text);
        let c2 = Composite::parse(&other, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
        let refused_pin = hex(&c2.sha512) != hex(&st.c.sha512);
        let refused_file = Instances::load(&arg("--instances", ""), &c2).is_err();
        return json!({"case": name, "expect": "refused at load (pin and instance file)", "pass": refused_pin && refused_file,
                      "pin_differs": refused_pin, "file_refused": refused_file});
    }
    if name == "prover_is_deterministic" {
        return determinism_case(st, inst, gpu);
    }
    if name == "coin_seed_record_replays" {
        // step 0 replayed from the verifier's record alone; the same record with two rounds' coin-request indices swapped is refused
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        let (v, _) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let honest = flock_live::replay_coin_seed(&rec);
        let mut bad = rec.clone();
        let (a, b) = (bad["streams"][0]["rounds"][1]["g"].clone(), bad["streams"][1]["rounds"][1]["g"].clone());
        bad["streams"][0]["rounds"][1]["g"] = b;
        bad["streams"][1]["rounds"][1]["g"] = a;
        let tampered = flock_live::replay_coin_seed(&bad);
        return json!({"case": name, "expect": "the record replays; swapped indices do not", "accepted": v["accepted"], "requests": honest.as_ref().ok(),
                      "pass": v["accepted"] == true && honest.is_ok() && tampered.is_err(), "why": honest.err().or(tampered.err())});
    }
    if name == "record_replays_offline" {
        // G2 offline: the verifier's record and the two proofs re-verified by Server::from_record + finish (the reference of the
        // Lean verifier's agreement test); the same record with the reps' proofs swapped is refused
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let replay = |p: [&Vec<u8>; 2]| -> Result<Value, String> {
            let map = p.iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b.to_vec())).collect::<BTreeMap<_, _>>();
            let v: Box<dyn StatementVerifier> = Box::new(CircuitVerifier { st: st.clone(), inst: inst.clone() });
            Ok(Server::from_record(config(st, inst), v, &rec, map)?.finish())
        };
        let (honest, swapped) = (replay([&proofs[0], &proofs[1]]), replay([&proofs[1], &proofs[0]]));
        let accepted = |r: &Result<Value, String>| matches!(r, Ok(x) if x["accepted"] == true);
        // a retained round's bytes altered in the record: refused (they no longer hash to the round's msg_sha256)
        let mut bad = rec.clone();
        let m = bad["streams"][0]["rounds"][0]["msg"].as_str().unwrap_or("").to_string();
        bad["streams"][0]["rounds"][0]["msg"] = json!(format!("{}{}", if m.starts_with('0') { "1" } else { "0" }, m.get(1..).unwrap_or("")));
        let v2: Box<dyn StatementVerifier> = Box::new(CircuitVerifier { st: st.clone(), inst: inst.clone() });
        let map = proofs.iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b.to_vec())).collect::<BTreeMap<_, _>>();
        let altered = Server::from_record(config(st, inst), v2, &bad, map.clone()).err();
        // the same record with its link.sigma flipped: refused (Σ is the verifier's own)
        let mut bad_sigma = rec.clone();
        let sg = bad_sigma["link"]["sigma"].as_str().unwrap_or("").to_string();
        bad_sigma["link"]["sigma"] = json!(format!("{}{}", if sg.starts_with('0') { "1" } else { "0" }, sg.get(1..).unwrap_or("")));
        let v3: Box<dyn StatementVerifier> = Box::new(CircuitVerifier { st: st.clone(), inst: inst.clone() });
        let sigma_refused = Server::from_record(config(st, inst), v3, &bad_sigma, map).err();
        return json!({"case": name, "expect": "the honest record replays offline; swapped proofs, altered round bytes and another Σ do not",
                      "accepted": v["accepted"], "altered_bytes_refused": altered, "sigma_refused": sigma_refused,
                      "pass": v["accepted"] == true && accepted(&honest) && !accepted(&swapped) && !m.is_empty() && altered.is_some()
                          && sigma_refused.is_some(),
                      "why": honest.as_ref().map(|x| x["why"].clone()).unwrap_or_else(|e| json!(e)),
                      "why_swapped": swapped.map(|x| x["why"].clone()).unwrap_or_else(|e| json!(e))});
    }
    if name == "coin_seed_opening_forged" {
        // an honest session, then its verdict with the revealed seed changed: the prover's step-0 check must refuse it
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        let log = Arc::new(Mutex::new(Vec::new()));
        let t: SharedTransport = shared(Logged { inner: InProc(s0.clone()), coins: log.clone() });
        let (v, _) = session(st, inst, t, &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let mut verdict = v["verdict"].clone();
        let seed = verdict["coin_seed"]["seed"].as_str().unwrap_or("").to_string();
        let forged = format!("{}{}", if seed.starts_with('0') { "1" } else { "0" }, &seed[1..]);
        verdict["coin_seed"]["seed"] = json!(forged);
        let c = &v["coin_seed"]["commitment"];
        let honest_ok = v["step0_checked"] == true;
        let commit: Vec<F128> = flock_live::circuit::hexb(c.as_str().unwrap_or("")).map(|b| b.chunks(16).map(|x| F128::new(u64::from_le_bytes(x[..8].try_into().unwrap()), u64::from_le_bytes(x[8..].try_into().unwrap()))).collect()).unwrap_or_default();
        let hello = config(st, inst).hello();
        let r = check_coin_seed(&commit, &hello, &verdict, &[]);
        return json!({"case": name, "expect": "the prover refuses a revealed seed that does not open the step-0 commitment", "honest_checked": honest_ok,
                      "pass": honest_ok && r.is_err(), "why": r.err()});
    }
    if name == "verifier_evaluated_tail_refused" {
        let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
        let (cut_line, meta_tail) = with_tail(&text);
        let r1 = Composite::parse(&cut_line, Some(&arg("--tables", "tables"))).err();
        let r2 = Composite::parse(&meta_tail, Some(&arg("--tables", "tables"))).err();
        return json!({"case": name, "expect": "refused at parse (a CUT line in a CIRCUIT section; a tail in META)", "pass": r1.is_some() && r2.is_some(),
                      "why_cut_line": r1, "why_meta_tail": r2});
    }
    if name == "unit_draw_from_file" {
        // the verifier's own draw (units 0 and the last), served at Register: the session proves exactly those units
        let draw = json!({"law": "subset", "population": inst.n, "k": 2.min(inst.n), "units": if inst.n > 1 { json!([0, inst.n - 1]) } else { json!([0]) }});
        let mut srv = with_draw(server(st, inst), st, inst, flock_live::unit_draw::Law::Subset { k: 0 });
        srv.cfg.draw_law = None;
        srv.cfg.draw_given = Some(draw.clone());
        let s = Arc::new(Mutex::new(srv));
        let (v, _) = session(st, inst, shared(InProc(s.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let pass = v["accepted"] == true && rec["unit_draw"] == draw;
        return json!({"case": name, "gpu": gpu, "accepted": v["accepted"], "pass": pass, "unit_draw": rec["unit_draw"], "why": v["why"]});
    }
    if name == "unit_draw_session" || name == "unit_draw_ignored_refused" {
        // the verifier draws one unit of the registered population after Register; the prover proves the drawn statement
        // (or, ignoring the draw, the population's, whose Σ is not the verifier's)
        let s = Arc::new(Mutex::new(with_draw(server(st, inst), st, inst, flock_live::unit_draw::Law::Subset { k: 1 })));
        let plan = Plan { gpu, ignore_draw: name.ends_with("refused"), ..Plan::default() };
        let (v, proofs) = session(st, inst, shared(InProc(s.clone())), &plan, ProverSeed::from_os());
        let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let acc = v["accepted"] == true;
        let draw = rec["unit_draw"].clone();
        // offline: the record rebuilt under the drawn statement (U1), and refused under the population's
        let replay = if acc {
            let (ds, di) = drawn(st, inst, &draw).unwrap();
            let map: BTreeMap<(String, usize), Vec<u8>> = proofs.iter().enumerate().map(|(r, p)| ((TABLE.to_string(), r), p.clone())).collect();
            let ok = Server::from_record(config(&ds, &di), Box::new(CircuitVerifier { st: ds.clone(), inst: di.clone() }), &rec, map.clone())
                .map(|mut x| x.finish()).map(|v| v["accepted"] == true);
            // under the population's statement Σ differs (the draw is under Σ); with the record's draw altered, U1
            let pop = Server::from_record(config(st, inst), Box::new(CircuitVerifier { st: st.clone(), inst: inst.clone() }), &rec, map.clone()).err();
            let mut bad = rec.clone();
            bad["unit_draw"]["k"] = json!(draw["k"].as_u64().unwrap_or(0) + 1);
            let u1 = Server::from_record(config(&ds, &di), Box::new(CircuitVerifier { st: ds.clone(), inst: di.clone() }), &bad, map).err();
            json!({"drawn_accepted": ok, "population_refused": pop, "altered_draw_refused": u1})
        } else { Value::Null };
        let pass = if name == "unit_draw_session" {
            acc && draw["units"].as_array().is_some_and(|u| u.len() == 1) && replay["drawn_accepted"] == json!({"Ok": true})
                && replay["population_refused"].as_str().is_some_and(|w| w.contains("R7"))
                && replay["altered_draw_refused"].as_str().is_some_and(|w| w.contains("U1"))
        } else {
            !acc && v["why"].as_str().is_some_and(|w| w.contains("R7"))
        };
        return json!({"case": name, "gpu": gpu, "accepted": acc, "pass": pass, "unit_draw": draw, "why": v["why"], "replay": replay});
    }
    if name == "device_salts_match_host" {
        #[cfg(all(feature = "gpu", feature = "sha512"))]
        if gpu {
            let s = flock_live::zk_hooks::LeafSalts::for_seed(&ProverSeed::from_os());
            let idx: Vec<i64> = vec![0, 1, 2, 31, 1000, (1 << 22) + 3];
            let ok = [0u64, 1, 5, u64::MAX].iter().all(|&nonce| {
                let d = flock_live::gpu_circuit::device_salts(s.key(), nonce, &idx);
                idx.iter().enumerate().all(|(k, &i)| d[192 * k..192 * (k + 1)] == s.salt(nonce, i as u64))
            });
            return json!({"case": name, "gpu": true, "expect": "the device's salts equal LeafSalts::salt", "pass": ok});
        }
        return json!({"case": name, "pass": true, "skipped": "a check of the GPU build"});
    }
    if name == "leaf_salts_from_prover_seed_refused" || name == "leaf_key_witness_refused" {
        // the pinned leaf scheme with its salts from a seeded stream; or with the hm96 key a prover witness (red-team-hm96 F6: a
        // key chosen after commitment opens a leaf to any value), or chosen by the prover rather than the statement's constant
        let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
        let lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
        let edits: Vec<(&str, Value)> = if name == "leaf_key_witness_refused" {
            vec![("key", json!("a witness: the prover's choice")), ("key", json!("per proof, drawn by the prover"))]
        } else {
            vec![("salt_source", json!("chacha20 stream of the prover seed"))]
        };
        let why: Vec<Option<String>> = edits.into_iter().map(|(k, v)| {
            let (mut m, mut l) = (meta.clone(), lines.clone());
            m["leaf_scheme"][k] = v;
            l[1] = format!("META {}", serde_json::to_string(&m).unwrap());
            Composite::parse(&l.join("\n"), Some(&arg("--tables", "tables"))).err()
        }).collect();
        return json!({"case": name, "expect": "refused at parse (not the pinned leaf-commitment scheme)", "pass": why.iter().all(Option::is_some),
                      "why": why});
    }
    if name == "typed_statement_altered_refused" {
        // the unit's type altered under its digest; the unit's layout altered under its sha512; the unit's rows in the statement
        if st.c.statement() != flock_live::typed::STATEMENT {
            return json!({"case": name, "pass": true, "skipped": "a check of a typed statement"});
        }
        let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
        let lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
        let with_meta = |m: &Value| { let mut l = lines.clone(); l[1] = format!("META {}", serde_json::to_string(m).unwrap()); l.join("\n") };
        let mut type_altered = meta.clone();
        let ut = meta["unit_type"].as_str().unwrap_or("");
        let and = meta["types"][ut]["items"].as_array().and_then(|a| a.iter().position(|it| it[0] == 0)).unwrap_or(0);
        let c = &mut type_altered["types"][ut]["items"][and][1][1];
        *c = json!(1 - c.as_u64().unwrap_or(0));
        let mut layout_altered = meta.clone();
        let split = if meta["layouts"][0]["in_split"] == json!([]) { json!([1]) } else { json!([]) };
        layout_altered["layouts"][0]["in_split"] = split;
        let rows = match &st.c.typed {
            Some(t) => t.inst.root_text(&st.c.name),
            None => flock_live::typed::unit_rows(&meta, &st.c.name).unwrap(),
        };
        let with_rows = format!("{text}CIRCUIT unit {}\n{rows}", rows.lines().count());
        let why: Vec<Option<String>> = [with_meta(&type_altered), with_meta(&layout_altered), with_rows].iter()
            .map(|t| Composite::parse(t, Some(&arg("--tables", "tables"))).err()).collect();
        return json!({"case": name, "expect": "refused at parse (a type or a layout not its digest; the unit's rows in the statement)",
                      "pass": why.iter().all(Option::is_some), "why": why});
    }
    let mut plan = plan_of(name, gpu);
    if name == "cross_session_replay_both_reps" {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        plan.substitute = Some(session(st, inst, shared(InProc(s0)), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os()).1);
    }
    let (pst, pinst, vinst) = match name {
        "output_claim_false" | "digest_claim_false" | "row_ref_claim_false" => {
            let what = match name { "output_claim_false" => "output", "row_ref_claim_false" => "ref", _ => "digest" };
            let v = Arc::new(false_statement(st, what));
            let mut p = Instances::load(&arg("--instances", ""), &st.c).unwrap();
            p.public_sha512 = v.public_sha512;
            (st.clone(), Arc::new(p), v)
        }
        "prover_file_differs_from_verifier_file" => {
            let mut p = Instances::load(&arg("--instances", ""), &st.c).unwrap();
            p.public_sha512[0] ^= 1;
            (st.clone(), Arc::new(p), inst.clone())
        }
        "tail_swapped_circuit_refused" => {
            let text = std::fs::read_to_string(arg("--forged", "circuit-forged.txt")).unwrap();
            let c2 = Composite::parse(&text, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
            let st2 = Arc::new(Stmt::new(c2, st.n, st.identity.clone()).unwrap_or_else(|e| refuse(e)));
            (st2, inst.clone(), inst.clone())
        }
        "relabelled_circuit" => {
            let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
            let c2 = Composite::parse(&relabel(&text), Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e));
            let st2 = Arc::new(Stmt::new(c2, st.n, st.identity.clone()).unwrap_or_else(|e| refuse(e)));
            (st2, inst.clone(), inst.clone())
        }
        _ => (st.clone(), inst.clone(), inst.clone()),
    };
    let s = Arc::new(Mutex::new(Server::new(config(st, &vinst), Box::new(CircuitVerifier { st: st.clone(), inst: vinst.clone() }))));
    let (v, proofs) = session(&pst, &pinst, shared(InProc(s.clone())), &plan, ProverSeed::from_os());
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
    record_case(st.c.statement(), name, want.1, &v, &rec, &proofs);
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let stopped = v["prover_stopped"].as_str().is_some_and(|x| x.contains("level-0 root"));
    let pass = acc == want.1 && (!want.2 || both) && (name != "reps_differ_prover_stops" || stopped);
    json!({"case": name, "gpu": plan.gpu, "expect": if want.1 {"accept"} else {"reject"}, "accepted": acc, "pass": pass,
           "both_reps_required": want.2, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "prove_s": v["prove_s"],
           "witness_s": v["witness_s"]})
}

/// The proof with one byte of its level-0 opening's first salt flipped (the SHA-512 build's hm96 leaves; unchanged otherwise).
fn flip_salt(p: &[u8]) -> Vec<u8> {
    #[cfg(feature = "sha512")]
    {
        let (c, mut pr): (Commitment, R1csProofLigerito) = bincode::deserialize(p).unwrap();
        if let Some(s) = pr.pcs_open.ligerito.initial_proof.opened_salts.first_mut() {
            s[0][0] ^= 1;
        }
        return bincode::serialize(&(c, pr)).unwrap();
    }
    #[allow(unreachable_code)]
    p.to_vec()
}

/// `--record-dir DIR`: the case's session as the verifier recorded it (`DIR/<case>.json`: the expectation, the prover's result,
/// the server record) and the proofs the prover produced (`DIR/<case>.rep<r>.proof`), for replaying verifiers' vectors.
fn record_case(statement: &str, name: &str, expect_accept: bool, v: &Value, rec: &Value, proofs: &[Vec<u8>]) {
    let dir = arg("--record-dir", "");
    if dir.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    let doc = json!({"case": name, "statement": statement, "expect": if expect_accept {"accept"} else {"reject"}, "accepted": v["accepted"],
                     "why": v["why"], "prover_stopped": v["prover_stopped"], "record": rec});
    let _ = std::fs::write(format!("{dir}/{name}.json"), serde_json::to_vec_pretty(&doc).unwrap_or_default());
    for (r, p) in proofs.iter().enumerate() {
        let _ = std::fs::write(format!("{dir}/{name}.rep{r}.proof"), p);
    }
}

/// Two forms of the circuit that hand the verifier part of the computation: the unit NET with a `CUT` line (the flock-ir-frame
/// native tail: the 36-op RMSNorm row tail's shape, one F32Add over two cut words), and META with a `tail` program.
fn with_tail(text: &str) -> (String, String) {
    let lines: Vec<&str> = text.split('\n').collect();
    // the unit's section, or a typed statement's first (it has none): a CUT line is refused in any
    let at = lines.iter().position(|l| l.starts_with("CIRCUIT unit ")).or_else(|| lines.iter().position(|l| l.starts_with("CIRCUIT "))).unwrap();
    let (sec, n) = { let h: Vec<&str> = lines[at].split_whitespace().collect(); (h[1].to_string(), h[2].parse::<usize>().unwrap()) };
    let cut = r#"CUT {"in":[],"out":[],"tail":[[2,"F32Add_v1",0,1]],"words":2}"#;
    let mut a: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    a[at] = format!("CIRCUIT {sec} {}", n + 1);
    a.insert(at + 1 + n, cut.to_string());
    let mut meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
    meta["tail"] = json!([[2, "F32Add_v1", 0, 1]]);
    let mut b: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    b[1] = format!("META {}", serde_json::to_string(&meta).unwrap());
    (a.join("\n"), b.join("\n"))
}

/// The prover as a function of (statement, witness, prover seed, verifier coins): two sessions with the same injected prover seed
/// and the same injected verifier seed give byte-identical proofs (both reps) and transcripts (every round's message and coins,
/// the link exchange); a third, with another prover seed, is recorded (it differs: the hm96 leaf salts are prover randomness).
/// Needs the test harness's `seed-injection` feature.
#[cfg(feature = "seed-injection")]
fn determinism_case(st: &Arc<Stmt>, inst: &Arc<Instances>, gpu: bool) -> Value {
    let run = |pseed: [u8; 32]| {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        s0.lock().unwrap_or_else(PoisonError::into_inner).inject_coin_seed([9; 32], [8; 32]);
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::injected(pseed));
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let t = json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]});
        (v["accepted"] == true, proofs, Sha512::digest(t.to_string().as_bytes()).to_vec())
    };
    let (a, b, c) = (run([1; 32]), run([1; 32]), run([2; 32]));
    let same = a.1 == b.1 && a.2 == b.2;
    json!({"case": "prover_is_deterministic", "expect": "equal seeds and coins give byte-identical proofs and transcripts", "gpu": gpu,
           "accepted": a.0 && b.0, "proofs_equal": a.1 == b.1, "transcripts_equal": a.2 == b.2, "other_seed_equal": a.1 == c.1 && a.2 == c.2,
           "transcript_sha256": hex(&a.2), "pass": a.0 && b.0 && c.0 && same})
}

#[cfg(not(feature = "seed-injection"))]
fn determinism_case(_: &Arc<Stmt>, _: &Arc<Instances>, _: bool) -> Value {
    json!({"case": "prover_is_deterministic", "pass": true, "skipped": "built without the test harness's seed-injection feature"})
}

/// The same circuit with unit 0's first two leaf inputs swapped in its META: the rows are unchanged, the meaning is not.
fn relabel(text: &str) -> String {
    let mut lines: Vec<String> = text.split('\n').map(|s| s.to_string()).collect();
    let mut meta: Value = serde_json::from_str(&lines[1][5..]).unwrap();
    // a unit's first two leaves swapped; a typed template (no leaf maps) its instance's first two outputs
    match meta.get_mut("leaves_in").and_then(|l| l.get_mut(0)).and_then(Value::as_array_mut) {
        Some(l) => l.swap(0, 1),
        None => flock_live::typed::relabel(&mut meta).unwrap(),
    }
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
        if skipped(st, name) || (name == "row_ref_claim_false" && !inst.tables.as_ref().is_some_and(|t| t.counts.iter().any(|&r| r > 1))) {
            continue;
        }
        let v = if gpu {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["selftest", "--gpu", "--instances", &arg("--instances", ""), "--circuit", &arg("--circuit", ""), "--tables", &arg("--tables", "tables"),
                       "--forged", &arg("--forged", "circuit-forged.txt"), "--only", name, "--record-dir", &arg("--record-dir", "")])
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
    println!("SELFTEST\t{}", json!({"statement": st.c.statement(), "unit": st.c.name, "subcircuit": inst.header["subcircuit"], "instances": st.n,
        "vus_per_block": st.c.g, "k_log": st.c.k_log, "m": st.m, "gpu": gpu, "circuit_sha512": hex(&st.c.sha512),
        "unit_rows_sha256": hex(&st.c.unit().sha256), "rows_per_vu": st.c.sizes_per_vu().0, "ands_per_vu": st.c.sizes_per_vu().1, "all_pass": all}));
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
    if cmd == "replay-coins" {
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(arg("--record", "session.json")).expect("record")).expect("record JSON");
        match flock_live::replay_coin_seed(&rec) {
            Ok(n) => println!("COINS\t{}", json!({"replayed": true, "requests": n})),
            Err(e) => { println!("COINS\t{}", json!({"replayed": false, "why": e})); std::process::exit(1) }
        }
        return;
    }
    let tb = Instant::now();
    let c = load_composite(&arg("--circuit", "circuit.txt"));
    if cmd == "rows" {
        // a typed statement's rows as this prover derives them, laid out as staging's rows/ (to diff the two mirrors)
        let out = arg("--out", "rows");
        std::fs::create_dir_all(&out).expect("--out");
        for (f, text) in flock_live::typed::staged_rows(&c.meta, &c.name, Some(&arg("--tables", "tables"))).unwrap_or_else(|e| refuse(e)) {
            std::fs::write(format!("{out}/{f}"), text).expect("rows");
        }
        return;
    }
    let pin = arg("--pin", "");
    if cmd == "serve" && pin.is_empty() {
        refuse("serve needs --pin (the pinned circuit's SHA-512)".into());
    }
    if !pin.is_empty() && hex(&c.sha512) != pin {
        refuse(format!("the circuit's SHA-512 {} is not the pinned circuit {pin}", hex(&c.sha512)));
    }
    let inst = Arc::new(Instances::load(&arg("--instances", "inst.bin"), &c).unwrap_or_else(|e| refuse(e)));
    if cmd == "serve" && inst.has_rows() {
        refuse("the verifier's instance file holds rows: stage its own public file (public_only)".into());
    }
    if let Err(e) = check_public(&c, &inst) {
        refuse(e);
    }
    let id = identity(&c);
    let st = Arc::new(Stmt::new(c, inst.n, id).unwrap_or_else(|e| refuse(e)));
    // the device prover holds at most 12 slot types (`gpu_circuit::FC_MAX_TYPES`)
    if flag("--gpu") && st.circuit.types.len() > 12 {
        refuse(format!("{} slot types: the GPU prover holds at most 12", st.circuit.types.len()));
    }
    // the device's fold holds each slot type's own rows only, never a generated read's table-direct side
    if flag("--gpu") && st.circuit.tables.iter().any(Option::is_some) {
        refuse("a typed template's table reads: the GPU prover folds no table-direct side".into());
    }
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
                let (mut v, proofs) = session(&st, &inst, t, &plan, ProverSeed::from_os());
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
        _ => eprintln!("usage: flock-circuit serve|prove|selftest --instances FILE --circuit FILE [--pin SHA256] ..."),
    }
}
