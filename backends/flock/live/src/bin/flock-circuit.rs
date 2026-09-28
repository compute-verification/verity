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
    pcs::{PcsParams, ligerito::{LigeritoProfile, embedded_initial_k_or_default}, verify_opening_batch_ligerito_mixed_with_grinding_zk},
    proof::{R1csProofLigerito, bind_statement},
    verifier::verify_ligerito_extra,
};
use flock_field::F128;
use flock_live::{
    circuit::*,
    zk_hooks::{Hooks, NoMask, ProverSeed},
    zk_veil::{self as zkv, InnerProof, Layout},
    *,
};
use flock_prover::{pcs::Commitment, prover::{prove_fast_ligerito_from_witness_extra, prove_fast_ligerito_from_witness_extra_zk}};
#[cfg(feature = "gpu")]
use flock_live::gpu_circuit::HostSlots;
use serde_json::{Value, json};
use sha2::{Digest, Sha512};

const TABLE: &str = "circuit";
const SIGMA_TAG: &[u8] = b"verity/flock-circuit/sigma";

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
    format!("flock-circuit/fast100x2/rep{rep}").into_bytes()
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

/// `--zk`: M1's masked protocol (`flock_live::zk_veil`), pinned in the identity (both sides must agree).
fn zk_mode() -> bool {
    flag("--zk")
}

fn is_zk(st: &Stmt) -> bool {
    st.identity["hooks"]["zk"] == true
}

/// M1's pad layout for the statement: the algebraic phase's, then Ligerito level 0's (its OOD samples and lane rounds).
fn zk_layout(st: &Stmt) -> Layout {
    let lc = params(st.m, LigeritoProfile::Fast100).ligerito_prover_config().expect("Ligerito config");
    Layout::new(st.m, st.c.k_log, lc.ood_samples[0], lc.initial_k)
}

/// Ligerito level 0's query count per rep under its padding: at RoPE's sizes the counts the soundness lane and the red team
/// agreed (m = 25, 26, 27: 277, 242, 229, a little above the least counts, 268, 239, 227), elsewhere the least count
/// (`zk_veil::level0_queries`); either way at least 100 bits per run. `--zk` refuses m <= 24: no count works at rate 1/2.
fn zk_queries0(m: usize) -> Result<usize, String> {
    if m <= 24 {
        return Err(format!("m = {m}: --zk needs m >= 25 (Ligerito level 0's padding outgrows its lanes at rate 1/2)"));
    }
    let lc = params(m, LigeritoProfile::Fast100).ligerito_prover_config()?;
    let l = 1usize << lc.initial_log_msg_cols;
    let q = match m {
        25 => 277,
        26 => 242,
        27 => 229,
        _ => zkv::level0_queries(l, lc.queries[0]).ok_or(format!("m = {m}: no level-0 query count reaches 100 bits at rate 1/2"))?,
    };
    if zkv::level0_t_pad(q) > l || zkv::level0_bits(l, q) < 100.0 {
        return Err(format!("m = {m}: {q} level-0 queries give {:.1} bits per run", zkv::level0_bits(l, q)));
    }
    Ok(q)
}

/// Ligerito level 0's padding per lane (`zk_veil::level0_t_pad`).
fn zk_t_pad(m: usize) -> usize {
    zkv::level0_t_pad(zk_queries0(m).expect("checked when the statement is built"))
}

/// Ligerito level 0's hiding randomness for a session (both reps open the one level-0 codeword): every lane's padding, and
/// the four uniform lanes (a pair per rep), from the prover's ChaCha20 streams 6 and 7; for rep 0 (a rep's copy sets `rep`).
/// `zero`: M1's control (all zero).
fn level0_zk(st: &Stmt, rng: &mut flock_live::zk_hooks::ProverRng, zero: bool, tamper: u8) -> flock_core::pcs::ligerito::zk::Level0Zk {
    use flock_core::pcs::ligerito::zk::{EXTRA_LANES, Level0Zk};
    let lc = params(st.m, LigeritoProfile::Fast100).ligerito_prover_config().expect("Ligerito config");
    let (lanes, l, t) = ((1usize << lc.initial_k) + EXTRA_LANES, 1usize << lc.initial_log_msg_cols, zk_t_pad(st.m));
    let draw = |rng: &mut flock_live::zk_hooks::ProverRng, stream: u64, i: usize, n: usize| {
        if zero { vec![F128::ZERO; n] } else { rng.f128s(stream, (i as u64) << 32, n) }
    };
    Level0Zk { queries0: zk_queries0(st.m).expect("checked when the statement is built"), t_pad: t, rep: 0,
               extra: std::array::from_fn(|i| draw(rng, 6, i, l)), pads: (0..lanes).map(|i| draw(rng, 7, i, t)).collect(), tamper }
}

/// M1's hooks identity: what is masked, and what is not yet (the statement is not zero knowledge until the list is empty).
fn zk_identity(c: &Composite) -> Value {
    json!({"zk": true, "stage": "m2-prototype (committed verifier coins, CPU)", "compiler": "veil (ePrint 2026/683 §4.1, Remark 4.4) over gf(2^128)",
           "verifier_coins": format!("committed before the prover's first message ({}): the prover's fresh nonce in hello; the root and the \
                                      verifier's session key in its answer; every coin opened against them, or the prover stops",
                                     flock_live::coin_tree::SCHEME),
           "pads": "zerocheck and lincheck: every sent value + a pad (prover ChaCha20 stream 4); pads committed before the zerocheck's first coin",
           "pads_commitment": format!("rs over gf(2^128), {} padding symbols, rate 1/{}, {} queries, hm96-sha512/v1 leaves, sha512 nodes",
                                      zkv::K_PAD, 1 << zkv::INNER_LOG_INV_RATE, zkv::Q),
           "product": "h_ab = h_a h_b committed; sacrificed against a committed triple (x, y, x y)",
           "inner_proof": "one batched <c, h> = t by a masked dot product (tau = <c, mu> committed by an hm96 leaf, opened after the last coin; Y = h + beta mu)",
           "ring_switch": format!("non-region s_hat_v hidden by the mask slot's uniform words (2^{} bits per block); the prover refuses coins whose weights do not span",
                                  c.meta["mask_log"].as_u64().unwrap_or(14)),
           "ligerito_level0": "veil figs. 8-9: every lane padded by (W_n + c) mu with t_pad = 2 q_0 uniform coefficients (c = x); four padded \
                               uniform lanes, rep r folding its own pair 2r, 2r + 1 after its lane rounds (rho_0, rho_1 F128-independent); the ood \
                               value, the lane-round messages and the pair's sums padded; T' and ybar sent (uniform)",
           "ligerito_level0_queries": "q_0 = 277, 242, 229 at m = 25, 26, 27, else the least q_0 >= fast100's with q_0 log2(1/(sqrt(rho') + 1/50)) >= 100; \
                                       rho' = (L + 2 q_0) / 2L; refused below 100 bits and at m <= 24",
           "refused_coins": ["beta = 0", "mask-slot weights of rank < 128 k over the claims so far", "rho_0, rho_1 dependent over F128"],
           "not_yet_masked": []})
}


/// The backend identity the statement digest pins (so a NON_ZK and a ZK cell never share one).
fn identity(c: &Composite) -> Value {
    let hooks = if zk_mode() { zk_identity(c) } else { NoMask::new(ProverSeed::from_os()).identity() };
    json!({"statement": STATEMENT, "backend": "C-interactive", "proof_class": "NON_ZK_PROOF", "profile": "fast100", "reps": 2,
           "commitments": "one root, both reps (R1)", "flavor": "rs", "leaf_scheme": c.leaf_scheme,
           // a selftest build injects seeds (the determinism check); a proving build never has the feature
           "seed_injection": cfg!(feature = "seed-injection"),
           // the frame-v3 tree and the row leaf are the serving side's formats, not this statement's to choose
           "hashes": {"merkle": format!("{} (every Ligerito level)", MERKLE.as_str()),
                      "round_digest": "sha512 (framing: the coin server retains every round's bytes and the replay compares them)",
                      "statement_digest": "sha512", "sigma": "sha512", "coin_commitment": "sha512",
                      "coin_derivation": "sha256 (verity.randomness)", "frame_v3_tree": "sha512 (frame-v3-sha512)",
                      "transcript": "none (live verifier coins)", "row_leaf_in_circuit": "hm96-sha512/row/v1 over sha512/row/v1"},
           "hooks": hooks})
}

/// Σ = SHA-512(SIGMA_TAG ‖ the circuit's SHA-512 ‖ the statement digest ‖ the public digest).
fn sigma(st: &Stmt, inst: &Instances) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(SIGMA_TAG);
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
        domain(rep)
    }

    fn verify(&self, _t: &str, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        if is_zk(&self.st) {
            return self.verify_zk(rep, proof, ch, link);
        }
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

impl CircuitVerifier {
    /// M1: the zerocheck and lincheck replayed on masked values as constraints on the committed pads (`zk_veil::replay`),
    /// upstream's opening with the two masked claims' ring-switch checks turned into constraints, then the inner proof.
    fn verify_zk(&self, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        if link.points.len() != 2 || link.y.iter().any(|y| *y != F128::ZERO) {
            return Err("session: two points and y = 0 expected (nothing is prover-supplied)".into());
        }
        let (commitment, proof, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        verify_zk_with(&self.st, &self.inst, &commitment, &proof, &inner, link.points, rep, ch)?;
        Ok(merkle_digest_bytes(&commitment.cap).to_vec())
    }
}

/// Everything M1's verifier checks before the inner proof, on any challenger (the session's replay, or a simulator's coin
/// tape): the replayed zerocheck and lincheck, upstream's opening with the masked claims' ring-switch checks and level 0's lane
/// phase turned into constraints, level 0's rows checked against rep `rep`'s pair of extra lanes. Returns the constraints the
/// inner proof must prove, and level 0's record.
#[allow(clippy::too_many_arguments)]
fn zk_constraints<Ch: flock_transcript::challenger::Challenger>(st: &Stmt, inst: &Instances, commitment: &Commitment, proof: &R1csProofLigerito,
                                                                 root: &[u8; 64], points: &[Vec<F128>], rep: usize, ch: &mut Ch)
    -> Result<(Vec<(&'static str, zkv::Lin)>, flock_core::pcs::ligerito::zk::Level0ZkRecord), String> {
    let pcs = params(st.m, LigeritoProfile::Fast100);
    if commitment.params != pcs {
        return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
    }
    let lay = zk_layout(st);
    bind_statement(ch, &st.r1cs, commitment);
    let rp = zkv::replay(&st.r1cs, &proof.zerocheck, &proof.lincheck, &st.circuit, &lay, root, None, ch)?;
    let extra = st.extra_claims(points, inst);
    let rs = &proof.pcs_open.ring_switches;
    if rs.len() != 2 + extra.len() {
        return Err("zk: the opening's ring switches are not one per claim".into());
    }
    let (cc_ab, con_ab) = zkv::ring_switch_constraint(&rp.ab_point, &rs[0].s_hat_v, &rp.ab_value);
    let (cc_c, con_c) = zkv::ring_switch_constraint(&rp.c_point, &rs[1].s_hat_v, &rp.c_value);
    let (pab, pc) = (zkv::point_parts(&rp.ab_point), zkv::point_parts(&rp.c_point));
    let mut values = vec![cc_ab, cc_c];
    let mut skips = vec![pab.0, pc.0];
    let mut xs = vec![pab.1, pc.1];
    for e in &extra {
        values.push(e.value);
        skips.push(e.skip_weights.clone());
        xs.push(e.x_full.clone());
    }
    let skip_refs: Vec<&[F128]> = skips.iter().map(|v| v.as_slice()).collect();
    let x_refs: Vec<&[F128]> = xs.iter().map(|v| v.as_slice()).collect();
    let cfg = flock_core::pcs::ligerito::zk::verifier_config(pcs.ligerito_verifier_config()?, zk_queries0(st.m)?);
    let rec = verify_opening_batch_ligerito_mixed_with_grinding_zk(commitment, &values, &skip_refs, &x_refs, &[], &proof.pcs_open, &cfg,
                                                                   pcs.opening_grinding(),
                                                                   Some(flock_core::pcs::ligerito::zk::Level0Params { t_pad: zk_t_pad(st.m), rep }), ch)
        .map_err(|e| format!("opening: {e:?}"))?
        .ok_or("zk: the opening has no level-0 record")?;
    let lg = &proof.pcs_open.ligerito;
    let t_prime = lg.zk_level0.as_ref().ok_or("zk: level 0's hiding messages are missing")?.t_prime;
    let mut cons = rp.cons;
    cons.push(("ring switch ab", con_ab));
    cons.push(("ring switch c", con_c));
    cons.extend(zkv::ligerito_constraints(&lay, &rec, &zkv::ligerito_exposed(lg, &lay), t_prime)?);
    Ok((cons, rec))
}

/// M1's verifier on any challenger: [`zk_constraints`], then the inner proof.
#[allow(clippy::too_many_arguments)]
fn verify_zk_with<Ch: flock_transcript::challenger::Challenger>(st: &Stmt, inst: &Instances, commitment: &Commitment, proof: &R1csProofLigerito,
                                                                  inner: &InnerProof, points: &[Vec<F128>], rep: usize, ch: &mut Ch) -> Result<(), String> {
    let root: [u8; 64] = inner.root.clone().try_into().map_err(|_| "zk: the pads' root is not 64 bytes".to_string())?;
    catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let (cons, _) = zk_constraints(st, inst, commitment, proof, &root, points, rep, ch)?;
        zkv::verify_inner(inner, &zk_layout(st), cons, ch)
    }))
    .map_err(|_| "verifier panicked".to_string())?
    .map_err(|e| e.chars().take(200).collect::<String>())
}

/// M2's committed coin schedule: the session's stream (the link points), then the table's two reps; 256 rounds of at most
/// 512 coins each (a RoPE rep uses 100 rounds, at most 218 coins in one).
fn coin_spec() -> flock_live::coin_tree::Spec {
    use flock_live::coin_tree::{SESSION, Spec};
    Spec { streams: vec![SESSION.into(), rep_stream(TABLE, 0), rep_stream(TABLE, 1)], rounds_log: 8, block: 512 }
}

/// A test harness's fixed verifier coins: M0's step-0 seed, or under M2 the committed schedule's random tape.
#[cfg(feature = "seed-injection")]
fn fix_verifier_coins(s: &mut Server, tape: [u8; 32]) {
    if s.cfg.coin_tree.is_some() {
        s.inject_coin_tape(tape);
    } else {
        s.inject_coin_seed(tape, [8; 32]);
    }
}

fn config(st: &Stmt, inst: &Instances) -> SessionConfig {
    let mut cfg = SessionConfig::flock_128_r2(vec![TableSpec { name: TABLE.into(), binding: Binding::R1cs { statement_digest: st.flock_digest } }]);
    cfg.link = Some(LinkSpec { sigma: sigma(st, inst).to_vec(), m: st.m_pts, points: 2 });
    // M2 (--zk): every coin committed at Hello and opened round by round; M0: the step-0 seed
    if is_zk(st) {
        cfg.coin_tree = Some(coin_spec());
    } else {
        cfg.coin_seed = true;
    }
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
    /// evaluate these circuits instead of the pinned ones (a swapped tail)
    nets: Arc<BTreeMap<usize, &'static flock_live::ir_block::IrUnitNet>>,
    /// write these public regions into the witness (a dummy witness whose region claims are the public ones)
    regions: Option<Arc<Instances>>,
    /// the mask slot's words are not the device's zero fill (M1's uniform words): a device witness takes them as host slots
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    mask_host: bool,
}

type Witness = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

/// A rep's witness: the packed host witness, or (GPU) the device witness's inputs.
#[derive(Clone)]
enum WitData {
    Host(Witness),
    /// the device witness's inputs: every compression slot's input words, the host slots (hm96, M1's mask slot, deep units)
    /// packed as the device takes them, every unit's input words
    #[cfg(feature = "gpu")]
    Dev(Vec<F128>, HostSlots, Vec<F128>),
    /// a cut template on the device: as `Dev` (the late words come from the host tail between the units' passes)
    #[cfg(feature = "gpu")]
    DevCut(Vec<F128>, HostSlots, Vec<F128>, Arc<dyn Fn(usize, usize) -> Vec<F128> + Sync + Send>),
}

/// A rep's witness: for an honest prover on the GPU (`device`), the device witness's inputs (the compression slots' and the
/// units' input words, the hm96 slots computed here, and M1's mask slot); otherwise the packed host witness.
fn witness(st: &Stmt, w: &Wit, device: bool, host_units: bool) -> (WitData, f64) {
    let t = Instant::now();
    // the mask slot's words where the device's fill is not zero: z = a = b (its circuit is the identity on the slot)
    #[cfg(feature = "gpu")]
    let mask_slots = || -> Vec<flock_live::gpu_circuit::HostSlot> {
        if !w.mask_host {
            return Vec::new();
        }
        let r = st.c.ranges.iter().find(|r| r.kind == Kind::Mask).expect("a mask slot");
        (0..st.blocks()).map(|b| {
            let m = (w.mask)(b);
            ((b << (st.c.k_log - 7)) + (r.start() >> 7), m.clone(), m.clone(), m)
        }).collect()
    };
    #[cfg(feature = "gpu")]
    // a tail that returns the outputs interleaves with the units round by round: the device's two passes cannot follow it
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && w.regions.is_none() && st.c.out_net == st.c.unit_net {
        use rayon::prelude::*;
        let ur = st.c.unit_range();
        let ri: Vec<_> = (0..st.blocks()).into_par_iter().map(|b| (w.rowin)(b)).collect();
        let rows: Vec<F128> = ri.iter().flat_map(|x| x.0.iter().flatten().copied()).collect();
        let hm: Vec<Vec<F128>> = ri.into_iter().flat_map(|x| x.1).collect();
        let mut host = flock_live::gpu_circuit::host_slots(st, st.c.hm_net, &hm);
        host.extend(mask_slots());
        let unit_inputs = || (0..st.blocks() * ur.count).into_par_iter().map(|x| (w.inputs)(x / ur.count, x % ur.count));
        if st.c.wires.iter().any(|x| x.dst.0 != st.c.sha_net && x.dst.0 != st.c.hm_net) {
            let inputs: Vec<F128> = unit_inputs().flatten_iter().collect();
            return (WitData::DevCut(rows, HostSlots::pack(&host), inputs, w.inputs.clone()), t.elapsed().as_secs_f64());
        }
        if host_units {
            let ins: Vec<Vec<F128>> = unit_inputs().collect();
            host.extend(flock_live::gpu_circuit::host_slots(st, st.c.unit_net, &ins));
            return (WitData::Dev(rows, HostSlots::pack(&host), Vec::new()), t.elapsed().as_secs_f64());
        }
        return (WitData::Dev(rows, HostSlots::pack(&host), unit_inputs().flatten_iter().collect()), t.elapsed().as_secs_f64());
    }
    #[cfg(feature = "gpu")]
    // a tail that returns the outputs (a chain of units, the tail's stages): the host converges every slot's inputs and computes
    // the stages and lookups; the device the compressions, and the units from their converged inputs
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && w.regions.is_none() {
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
        host.extend(mask_slots());
        let inputs: Vec<F128> = ins[c.unit_net].iter().flatten().copied().collect();
        return (WitData::Dev(rows, HostSlots::pack(&host), inputs), t.elapsed().as_secs_f64());
    }
    let _ = (device, host_units);
    let mut ins = st.converge(&*w.inputs, &*w.rowin, &w.nets);
    if let Some((n, x, col)) = w.cut_flip {
        let word = &mut ins[n][x][col / 128];
        if col % 128 < 64 { word.lo ^= 1 << (col % 128) } else { word.hi ^= 1 << (col % 128 - 64) }
    }
    let (mut z, a, b, zl) = st.witness(&ins, &*w.mask, &w.nets);
    if let Some(bit) = w.flip_z {
        z[bit / 128] = z[bit / 128] + if bit % 128 < 64 { F128::new(1 << (bit % 128), 0) } else { F128::new(0, 1 << (bit % 128 - 64)) };
    }
    if let Some(inst) = &w.regions {
        st.write_regions(&mut z, inst);
    }
    let zl = if w.flip_z.is_some() || w.regions.is_some() { flock_core::lincheck::pack_z_lincheck_from_packed(&z, st.m, st.c.k_log) } else { zl };
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

/// A borrowed challenger (the masking wrapper owns its inner challenger; upstream's single-table path never forks).
struct ByRef<'a, C: flock_transcript::challenger::Challenger>(&'a mut C);

impl<C: flock_transcript::challenger::Challenger> flock_transcript::challenger::Challenger for ByRef<'_, C> {
    fn observe_label(&mut self, l: &[u8]) {
        self.0.observe_label(l)
    }
    fn observe_f128(&mut self, v: F128) {
        self.0.observe_f128(v)
    }
    fn observe_f128_slice(&mut self, v: &[F128]) {
        self.0.observe_f128_slice(v)
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        self.0.observe_bytes(b)
    }
    fn sample_f128(&mut self) -> F128 {
        self.0.sample_f128()
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.0.sample_f128_vec(n)
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.0.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.0.verify_pow(nonce, bits)
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the single-table prove path does not fork")
    }
}

/// One rep's masking state (M1): the pads, their commitment, and the claim points of the reps so far (the rank check).
struct ZkRep {
    lay: Layout,
    pads: zkv::Pads,
    ic: zkv::InnerCommit,
    points: Arc<Mutex<Vec<Vec<F128>>>>,
    /// a dishonest prover: absorb and send values masked with pads other than the committed ones
    pad_mismatch: bool,
    /// seconds to draw the pads and build their commitment (counted in the rep's prove time)
    t_commit: f64,
    /// Ligerito level 0's hiding, drawn once per session (both reps open the one level-0 codeword), with this rep's pair
    level0: Arc<flock_core::pcs::ligerito::zk::Level0Zk>,
}

/// A rep's pads and their commitment (M1), the pads tree under salt id `id`, with the session's level-0 hiding and this rep's pair.
#[allow(clippy::too_many_arguments)]
fn make_zkrep(st: &Stmt, plan: &Plan, prng: &mut flock_live::zk_hooks::ProverRng, level0: &flock_core::pcs::ligerito::zk::Level0Zk,
              points: &Arc<Mutex<Vec<Vec<F128>>>>, rep: usize, id: u64, salts: &Mutex<flock_live::zk_hooks::LeafSalts>) -> ZkRep {
    let tz0 = Instant::now();
    let lay = zk_layout(st);
    let mut pads = zkv::Pads::draw(prng, &lay, rep);
    if plan.zk_zero_pads {
        pads.h.iter_mut().for_each(|x| *x = F128::ZERO);
    }
    let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
    let sb = salts.lock().unwrap_or_else(PoisonError::into_inner).expand(id, n);
    let ic = zkv::InnerCommit::new(&pads, sb);
    ZkRep { lay, pads, ic, points: points.clone(), pad_mismatch: plan.zk_pad_mismatch, t_commit: tz0.elapsed().as_secs_f64(),
            level0: Arc::new(flock_core::pcs::ligerito::zk::Level0Zk { rep: if plan.zk_shared_pair { 0 } else { rep }, ..level0.clone() }) }
}

/// M1 on the device: level 0's hiding and both reps' pads with their commitments, drawn beside the witness build (they depend on
/// the seed alone). The salt ids are the sequential prover's: rep 0's pads tree takes the next id, rep 1's the one after rep 0's
/// proof's trees (levels 1 to R), which `session` consumes when rep 1 starts. Returns the level-0 draw's seconds, and each rep's
/// pads with their tree's salt id.
fn zk_prepare(st: &Stmt, plan: &Plan, seed: ProverSeed, lig: LigeritoProfile, salts: &Mutex<flock_live::zk_hooks::LeafSalts>,
              points: &Arc<Mutex<Vec<Vec<F128>>>>) -> (f64, [(ZkRep, u64); 2]) {
    let mut prng = flock_live::zk_hooks::ProverRng::from_seed(seed);
    let tl0 = Instant::now();
    let level0 = level0_zk(st, &mut prng, plan.zk_zero_pads, plan.zk_l0_tamper);
    let t_level0 = tl0.elapsed().as_secs_f64();
    let trees = params(st.m, lig).ligerito_prover_config().expect("Ligerito config").recursive_steps as u64;
    let id0 = salts.lock().unwrap_or_else(PoisonError::into_inner).next_id();
    let reps = [0usize, 1].map(|rep| {
        let id = id0 + rep as u64 * (1 + trees);
        (make_zkrep(st, plan, &mut prng, &level0, points, rep, id, salts), id)
    });
    (t_level0, reps)
}

/// The mask slot's word indices in the packed witness (every block's slot).
fn mask_words(st: &Stmt) -> Vec<usize> {
    let r = st.c.ranges.iter().find(|r| r.kind == Kind::Mask).expect("a mask slot");
    let (w0, n) = (r.start() >> 7, 1usize << (r.slot_log - 7));
    (0..st.blocks()).flat_map(|b| (0..n).map(move |j| (b << (st.c.k_log - 7)) + w0 + j)).collect()
}

/// The two z-claims' word coordinates (`x_outer[1..]` of the opening's claim point) for the phase's coins: the replay on a
/// zero proof of the right shape (the points depend on the coins alone).
fn claim_word_points(st: &Stmt, lay: &Layout, coins: &[F128]) -> Result<[Vec<F128>; 2], String> {
    use flock_core::lincheck::LincheckCircuit;
    let (ab, c) = zkv::points_from_coins(&st.r1cs, lay, st.circuit.const_pin_col().is_some(), coins)?;
    Ok([zkv::point_parts(&ab).1[1..].to_vec(), zkv::point_parts(&c).1[1..].to_vec()])
}

/// A rep's masking challenger (M1) in front of its live stream: the pads it adds (a dishonest prover's other ones), and the
/// rank check at the opening's label (the prover stops on degenerate coins).
fn zk_masking<C: flock_transcript::challenger::Challenger>(st: &Arc<Stmt>, ch: C, zk: &ZkRep) -> (zkv::Masking<C>, zkv::Pads) {
    let mut sent = zkv::Pads { h: zk.pads.h.clone(), mu: vec![], pad_h: vec![], pad_mu: vec![], tau_salt: vec![] };
    if zk.pad_mismatch {
        sent.h[zk.lay.ml(0, 0)] += F128::ONE;
    }
    let mut mch = zkv::Masking::new(ch, &zk.lay, &sent, zk.ic.root());
    let (st2, lay, pts, words) = (st.clone(), zk.lay, zk.points.clone(), mask_words(st));
    mch.at_open = Some(Box::new(move |coins: &[F128]| {
        let tr = Instant::now();
        let mut all = pts.lock().unwrap_or_else(PoisonError::into_inner);
        all.extend(claim_word_points(&st2, &lay, coins)?);
        let (rank, ok) = zkv::mask_rank_ok(&all, &words);
        eprintln!("ZKRANK\t{}", json!({"points": all.len(), "mask_words": words.len(), "rank": rank, "need": 128 * all.len(), "s": tr.elapsed().as_secs_f64()}));
        if ok { Ok(()) } else { Err(format!("degenerate coins: the mask slot's weights have rank {rank} < {} over {} claim points", 128 * all.len(), all.len())) }
    }));
    (mch, sent)
}

/// Frames every call as the device prover's challenger does (`FsChallenger` under its live hook) and hands each round to
/// `rounds` at its squeeze: with `FC_ZK_FRAMED`, M1's CPU prover takes the device path's host side (`replay_round` into the
/// masking challenger), a test of it without a GPU.
struct Framer<'a> {
    pending: Vec<u8>,
    rounds: &'a mut (dyn FnMut(&[u8], usize) -> Vec<F128> + Send),
}

impl Framer<'_> {
    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let c = std::mem::take(&mut self.pending);
        (self.rounds)(&c, n)
    }
}

impl flock_transcript::challenger::Challenger for Framer<'_> {
    fn observe_label(&mut self, l: &[u8]) {
        push_header(&mut self.pending, OP_LABEL, 0, l.len() as u64);
        push_padded(&mut self.pending, l);
    }
    fn observe_f128(&mut self, v: F128) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
        self.pending.extend_from_slice(&f128_bytes(&[v]));
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
        self.pending.extend_from_slice(&f128_bytes(vs));
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        push_header(&mut self.pending, OP_BYTES, 0, b.len() as u64);
        push_padded(&mut self.pending, b);
    }
    fn sample_f128(&mut self) -> F128 {
        self.squeeze(KIND_SCALAR, 1)[0]
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.squeeze(KIND_SLICE, n)
    }
    /// the device's live grinding: nonce 0, absorbed like any message
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        self.observe_bytes(&0u64.to_le_bytes());
        0
    }
    fn verify_pow(&mut self, _: u64, _: u32) -> bool {
        unreachable!("the prover's challenger never verifies")
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the single-table prove path does not fork")
    }
}

/// M1's CPU prover: upstream's prover under the masking challenger (or, with `FC_ZK_FRAMED`, under a [`Framer`] whose rounds are
/// replayed into it, as the device's are), then [`zk_finish`].
#[allow(clippy::too_many_arguments)]
fn prove_cpu_zk(st: &Arc<Stmt>, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger, zk: &ZkRep) -> (Vec<u8>, Value) {
    #[allow(irrefutable_let_patterns)]
    let WitData::Host((z, a, b, zl)) = wt.clone() else { panic!("M1's CPU prover takes a host witness") };
    let tp = Instant::now();
    let (mut mch, sent) = zk_masking(st, ByRef(ch), zk);
    let mut hk = |cm: &Commitment| -> Vec<Vec<F128>> { hook(merkle_digest_bytes(&cm.cap)).into_iter().map(|p| p[6..].to_vec()).collect() };
    let (proof, commitment) = if std::env::var("FC_ZK_FRAMED").is_ok() {
        let mut rounds = |content: &[u8], n: usize| replay_round(&mut mch, content, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"));
        let mut fr = Framer { pending: frame_domain(&domain(rep)), rounds: &mut rounds };
        prove_fast_ligerito_from_witness_extra_zk(&st.r1cs, &params(st.m, lig), z, a, b, zl, &st.circuit, &mut hk, Some(&zk.level0), &mut fr)
    } else {
        prove_fast_ligerito_from_witness_extra_zk(&st.r1cs, &params(st.m, lig), z, a, b, zl, &st.circuit, &mut hk, Some(&zk.level0), &mut mch)
    };
    let t_upstream = tp.elapsed().as_secs_f64();
    zk_finish(st, zk, proof, commitment, mch, &sent, tp, t_upstream, json!({}))
}

/// After upstream's prover (the CPU's, or the device's with the same unmasked proof, level-0 record and folded comb): the
/// masked proof, the constraints replayed from the logged coins, and the inner proof on the rep's stream after the opening.
#[allow(clippy::too_many_arguments)]
fn zk_finish<C: flock_transcript::challenger::Challenger>(st: &Stmt, zk: &ZkRep, mut proof: R1csProofLigerito, commitment: Commitment,
                                                           mch: zkv::Masking<C>, sent: &zkv::Pads, tp: Instant, t_upstream: f64, mut bk: Value)
                                                           -> (Vec<u8>, Value) {
    let (mut inner_ch, coins, _coins_l0) = mch.finish();
    let tz = Instant::now();
    let (zc, lc) = zkv::mask_proof(&proof.zerocheck, &proof.lincheck, &zk.lay, &sent.h);
    let rp = zkv::replay(&st.r1cs, &zc, &lc, &st.circuit, &zk.lay, &zk.ic.root(), Some(&proof.lincheck.comb_partial), &mut zkv::Tape::new(&coins))
        .unwrap_or_else(|e| panic!("{e}"));
    let t_replay = tz.elapsed().as_secs_f64();
    let wp = claim_word_points(st, &zk.lay, &coins).unwrap_or_else(|e| panic!("{e}"));
    assert!(wp[0] == zkv::point_parts(&rp.ab_point).1[1..] && wp[1] == zkv::point_parts(&rp.c_point).1[1..],
            "the rank check's claim points are not the replay's");
    let rs = &proof.pcs_open.ring_switches;
    let (_, con_ab) = zkv::ring_switch_constraint(&rp.ab_point, &rs[0].s_hat_v, &rp.ab_value);
    let (_, con_c) = zkv::ring_switch_constraint(&rp.c_point, &rs[1].s_hat_v, &rp.c_value);
    let mut cons = rp.cons;
    cons.push(("ring switch ab", con_ab));
    cons.push(("ring switch c", con_c));
    let rec = proof.pcs_open.ligerito.zk_record.take().expect("the zk prover records level 0's lane phase");
    zkv::mask_ligerito(&mut proof.pcs_open.ligerito, &zk.lay, &sent.h);
    let t_prime = proof.pcs_open.ligerito.zk_level0.as_ref().expect("level 0's hiding messages").t_prime;
    cons.extend(zkv::ligerito_constraints(&zk.lay, &rec, &zkv::ligerito_exposed(&proof.pcs_open.ligerito, &zk.lay), t_prime).unwrap_or_else(|e| panic!("{e}")));
    let fails = zkv::failing(&cons, &zk.pads.h);
    let mut lch = zkv::Logging { inner: &mut inner_ch, coins: Vec::new() };
    let inner = zkv::prove_inner(&zk.ic, &zk.pads, &zk.lay, cons, &mut lch).unwrap_or_else(|e| panic!("{e}"));
    let t_zk = tz.elapsed().as_secs_f64();
    let masked = R1csProofLigerito { zerocheck: zc, lincheck: lc, pcs_open: proof.pcs_open };
    let o = bk.as_object_mut().expect("buckets");
    for (k, v) in [("t.prove", json!(tp.elapsed().as_secs_f64())), ("t.upstream", json!(t_upstream)), ("t.zk_inner", json!(t_zk)),
                   ("t.zk_replay", json!(t_replay)), ("t.zk_pads_commit", json!(zk.t_commit)), ("constraints_failing_on_pads", json!(fails)),
                   ("zk_coins", json!(f128s_hex(&coins))), ("zk_inner_coins", json!(f128s_hex(&lch.coins)))] {
        o.insert(k.into(), v);
    }
    (bincode::serialize(&(commitment, masked, inner)).unwrap(), bk)
}

fn f128s_hex(v: &[F128]) -> Vec<String> {
    v.iter().map(|x| hex(&f128_bytes(&[*x]))).collect()
}

fn f128s_unhex(v: &Value) -> Vec<F128> {
    v.as_array().map(|a| a.iter().filter_map(|s| hexb(s.as_str()?).ok()).map(|b| F128::new(u64::from_le_bytes(b[..8].try_into().unwrap()), u64::from_le_bytes(b[8..16].try_into().unwrap()))).collect()).unwrap_or_default()
}

/// One rep's checked fields: the exposed values (masked), the two masked claims' `s_hat_v`, the inner proof's messages and
/// opened columns.
fn checked_fields(kind: &str, run: usize, rep: usize, exposed: &[F128], s: [&[F128]; 2], inner: &InnerProof) -> Value {
    json!({"kind": kind, "run": run, "rep": rep, "exposed": f128s_hex(exposed), "s_ab": f128s_hex(s[0]), "s_c": f128s_hex(s[1]),
           "inner_scalars": f128s_hex(&[inner.rho, inner.sigma, inner.tau]), "inner_y": f128s_hex(&inner.y),
           "inner_open_h": f128s_hex(&inner.opened.iter().map(|x| x.0).collect::<Vec<_>>()),
           "inner_open_mu": f128s_hex(&inner.opened.iter().map(|x| x.1).collect::<Vec<_>>())})
}

/// Ligerito's checked fields of one rep's proof: level 0's exposed values (masked), `T'` and `ybar`, the first four level-0
/// rows (the extra lanes included), and from levels 1+ the code-switch message and the next seven, the first two level-1
/// rows and the final vector.
fn ligerito_fields(v: &mut Value, lg: &flock_core::pcs::ligerito::LigeritoProof, lay: &Layout) {
    let z = lg.zk_level0.as_ref().expect("a --zk proof");
    let f256 = |x: &[flock_field::F256]| -> Vec<F128> { x.iter().flat_map(|y| y.coordinates()).collect() };
    v["lig_exposed"] = json!(f128s_hex(&zkv::ligerito_exposed(lg, lay)));
    v["lig_tprime_ybar"] = json!(f128s_hex(&[f256(&[z.t_prime]), f256(&z.ybar)].concat()));
    v["lig_rows0"] = json!(f128s_hex(&lg.initial_proof.opened_rows[..4].concat()));
    let msgs: Vec<flock_field::F256> = lg.sumcheck_transcript_f256[lay.k0..lay.k0 + 8].iter().flat_map(|m| [m.u_0, m.u_2]).collect();
    v["lig_deep"] = json!(f128s_hex(&[f256(&msgs), lg.recursive_proofs[0].opened_rows[..2].concat(), lg.final_proof.yr.clone()].concat()));
}

/// M1's simulator for one rep's whole proof, from the public statement (`inst`'s public part: roots, `b || c`, outputs), the
/// verifier's coins for the rep's stream (every squeeze, in order) and the link points: uniform exposed values and masked
/// claims' `s_hat_v`, the region claims' `s_hat_v` from the public data, the opening as the honest prover makes it on a dummy
/// (all-zero) witness with fresh masks (`dummy`: such a proof against the same coins) with level 0's exposed values drawn
/// uniformly, and the inner proof simulated for the constraints the verifier derives from all of that (on the same coins).
#[allow(clippy::too_many_arguments)]
fn simulate_full(st: &Stmt, inst: &Instances, coins: &[F128], points: &[Vec<F128>], dummy: &[u8], rng: &mut flock_live::zk_hooks::ProverRng,
                 rep: usize, index: u64) -> Result<(Commitment, R1csProofLigerito, InnerProof), String> {
    let lay = zk_layout(st);
    let (commitment, dpr, _): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(dummy).map_err(|e| format!("dummy: {e}"))?;
    let base = 3u64 << 50 | index << 36;
    let (zc, lc) = zkv::from_exposed(&rng.f128s(zkv::STREAM_PADS, base, lay.piop_exposed()), &lay);
    let mut ring_switches = dpr.pcs_open.ring_switches.clone();
    ring_switches[0].s_hat_v = rng.f128s(zkv::STREAM_PADS, base + (1 << 30), 128);
    ring_switches[1].s_hat_v = rng.f128s(zkv::STREAM_PADS, base + (1 << 31), 128);
    let mut k = 2;
    for reg in &st.regions {
        for p in points {
            ring_switches[k].s_hat_v = st.region_s_hat_v(reg, p, inst);
            k += 1;
        }
    }
    let mut lg = dpr.pcs_open.ligerito.clone();
    zkv::set_ligerito_exposed(&mut lg, &lay, &rng.f128s(zkv::STREAM_PADS, base + (1 << 32), lay.n_ood0 + 4 * lay.k0 + 4));
    let proof = R1csProofLigerito { zerocheck: zc, lincheck: lc, pcs_open: flock_core::pcs::BatchOpeningProofLigerito {
        ring_switches, batching_nonces: dpr.pcs_open.batching_nonces.clone(), ligerito: lg } };
    let mut tape = zkv::Tape::new(coins);
    let (cons, _) = zk_constraints(st, inst, &commitment, &proof, &[0; 64], points, rep, &mut tape)?;
    let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
    let salts = flock_live::zk_hooks::LeafSalts::for_seed(&ProverSeed::from_os()).expand(1, n);
    let inner = zkv::simulate_inner(&lay, rng, index, cons, salts, &mut tape)?;
    Ok((commitment, proof, inner))
}

/// One rep's proof's checked fields ([`checked_fields`] and [`ligerito_fields`]).
fn proof_fields(lay: &Layout, kind: &str, run: usize, rep: usize, pr: &R1csProofLigerito, inner: &InnerProof) -> Value {
    let e = zkv::exposed_values(&pr.zerocheck, &pr.lincheck, lay);
    let rs = &pr.pcs_open.ring_switches;
    let mut row = checked_fields(kind, run, rep, &e, [&rs[0].s_hat_v, &rs[1].s_hat_v], inner);
    ligerito_fields(&mut row, &pr.pcs_open.ligerito, lay);
    row
}

/// A rep stream's coins from a session record (every squeeze, in order), and the link points.
fn record_coins(rec: &Value, stream: &str) -> Vec<F128> {
    let s = rec["streams"].as_array().unwrap().iter().find(|s| s["stream"] == stream).unwrap_or_else(|| panic!("no stream {stream}"));
    s["rounds"].as_array().unwrap().iter().flat_map(|r| r["coins"].as_array().unwrap().iter().map(|h| f128_from_record(h.as_str().unwrap())).collect::<Vec<_>>()).collect()
}

fn f128_from_record(h: &str) -> F128 {
    let b = hexb(h).unwrap();
    F128::new(u64::from_be_bytes(b[8..16].try_into().unwrap()), u64::from_be_bytes(b[..8].try_into().unwrap()))
}

fn record_points(rec: &Value) -> Vec<Vec<F128>> {
    rec["link"]["points"].as_array().unwrap().iter().map(|p| p.as_array().unwrap().iter().map(|h| f128_from_record(h.as_str().unwrap())).collect()).collect()
}

/// `eq(r, l)` for every lane `l` (bit `i` of `l` against `r_i`), over F256.
fn eq256(r: &[flock_field::F256]) -> Vec<flock_field::F256> {
    use flock_field::F256;
    (0..1usize << r.len()).map(|l| r.iter().enumerate().fold(F256::ONE, |a, (i, &ri)| a * if (l >> i) & 1 == 1 { ri } else { F256::ONE + ri })).collect()
}

/// Two linear functionals of one session's public transcript that span fields and both reps (red-team-m1-zk F1's), each a
/// function of the witness alone if level 0's extra lanes or their padding were shared by the reps:
/// - level 0: `D(q) = N·pc^0(q) + pc^1(q) + Σ_l u_l row_q[l]` at every level-0 position either rep opened, where `pc^r` is
///   rep r's `Enc(c·ybar ‖ ybar)`, `W^r` the F128 matrix of `(a, b) ↦ ρ^r_0 a + ρ^r_1 b`, `N = W^1 (W^0)^-1` and
///   `u_l = N eq^0_l + eq^1_l`; projected onto the annihilator of the mask slot's level-0 columns;
/// - level 1: `N·v^0_j(q) + v^1_j(q)` at every level-1 position both reps opened, for each lane pair `j` (both coordinates
///   of `y_1'[x]`, `x ≡ j`) that holds no mask column.
///
/// With a pair of extra lanes per rep, both carry the pairs' uniform padding and lanes. `coins`: each rep's stream; `pairs`:
/// each rep's pair of extra lanes (`[0, 1]`; `[0, 0]` reads a session whose reps share rep 0's pair).
fn session_relations(st: &Stmt, inst: &Instances, proofs: &[Vec<u8>], coins: &[Vec<F128>], points: &[Vec<F128>], pairs: [usize; 2])
    -> Result<[Vec<F128>; 2], String> {
    use flock_core::pcs::ligerito::zk::{lane_codeword, pad_codeword};
    use flock_field::F256;
    let lc = params(st.m, LigeritoProfile::Fast100).ligerito_prover_config()?;
    let (log_msg, lanes) = (lc.initial_log_msg_cols, 1usize << lc.initial_k);
    type M2 = [[F128; 2]; 2];
    let wmat = |rho: [F256; 2]| -> M2 { [[rho[0].c0, rho[1].c0], [rho[0].c1, rho[1].c1]] };
    let mul = |a: M2, b: M2| -> M2 {
        [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
         [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
    };
    let inv = |a: M2| -> M2 {
        let d = (a[0][0] * a[1][1] + a[0][1] * a[1][0]).inv();
        [[a[1][1] * d, a[0][1] * d], [a[1][0] * d, a[0][0] * d]]
    };
    let app = |n: M2, v: F256| F256::new(n[0][0] * v.c0 + n[0][1] * v.c1, n[1][0] * v.c0 + n[1][1] * v.c1);
    let mut reps = Vec::new();
    for (rep, p) in proofs.iter().enumerate() {
        let (c, pr, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(p).map_err(|e| format!("decode: {e}"))?;
        let root: [u8; 64] = inner.root.clone().try_into().map_err(|_| "pads root")?;
        let (_, rec) = zk_constraints(st, inst, &c, &pr, &root, points, pairs[rep], &mut zkv::Tape::new(&coins[rep]))?;
        reps.push((rec, pr.pcs_open.ligerito));
    }
    let n = mul(wmat(reps[1].0.rho), inv(wmat(reps[0].0.rho)));
    let eqs: Vec<Vec<F256>> = reps.iter().map(|(r, _)| eq256(&r.challenges)).collect();
    let u: Vec<F256> = (0..lanes).map(|l| app(n, eqs[0][l]) + eqs[1][l]).collect();
    let pc: Vec<Vec<F256>> = reps.iter().map(|(_, lg)| pad_codeword(&lg.zk_level0.as_ref().expect("a --zk proof").ybar, log_msg, 1)).collect();
    let mut rows: BTreeMap<usize, &Vec<F128>> = BTreeMap::new();
    for (rec, lg) in &reps {
        for (k, &q) in rec.queries0.iter().enumerate() {
            rows.insert(q, &lg.initial_proof.opened_rows[k]);
        }
    }
    let d: Vec<(usize, F256)> = rows.iter().map(|(&q, row)| (q, (0..lanes).fold(app(n, pc[0][q]) + pc[1][q], |a, l| a + u[l] * row[l]))).collect();
    // the annihilator of the mask columns' codewords at the opened positions (word w is lane w mod lanes, column w / lanes)
    let mut mc: Vec<usize> = mask_words(st).iter().map(|w| w / lanes).collect();
    mc.sort();
    mc.dedup();
    let mut m: Vec<Vec<F128>> = mc.iter().map(|&col| {
        let mut e = vec![F128::ZERO; 1 << log_msg];
        e[col] = F128::ONE;
        let cw = lane_codeword(&[], Some(&e), log_msg, 1);
        d.iter().map(|(q, _)| cw[*q]).collect()
    }).collect();
    let mut piv = Vec::new();
    for col in 0..d.len() {
        let r = piv.len();
        let Some(p) = (r..m.len()).find(|&i| m[i][col] != F128::ZERO) else { continue };
        m.swap(r, p);
        let iv = m[r][col].inv();
        m[r].iter_mut().for_each(|x| *x = *x * iv);
        let pr = m[r].clone();
        for (i, row) in m.iter_mut().enumerate() {
            if i != r && row[col] != F128::ZERO {
                let f = row[col];
                row.iter_mut().zip(&pr).for_each(|(x, y)| *x += f * *y);
            }
        }
        piv.push(col);
    }
    let level0: Vec<F128> = (0..d.len()).filter(|j| !piv.contains(j)).flat_map(|free| {
        let t = d.iter().enumerate().fold(d[free].1, |a, (j, (_, x))| match piv.iter().position(|&p| p == j) {
            Some(i) if m[i][free] != F128::ZERO => a + *x * m[i][free],
            _ => a,
        });
        t.coordinates()
    }).collect();
    let q1: Vec<&Vec<usize>> = reps.iter().map(|(r, _)| &r.queries1).collect();
    let mut common: Vec<usize> = q1[0].iter().filter(|q| q1[1].contains(q)).copied().collect();
    common.sort();
    common.dedup();
    let level1: Vec<F128> = common.iter().flat_map(|q| {
        let k = [0, 1].map(|r| q1[r].iter().position(|x| x == q).unwrap());
        let (r0, r1) = (&reps[0].1.recursive_proofs[0].opened_rows[k[0]], &reps[1].1.recursive_proofs[0].opened_rows[k[1]]);
        let pairs = r0.len() / 2;
        (0..pairs).filter(|j| !mc.iter().any(|c| c % pairs == *j))
            .flat_map(|j| (app(n, F256::new(r0[2 * j], r0[2 * j + 1])) + F256::new(r1[2 * j], r1[2 * j + 1])).coordinates())
            .collect::<Vec<_>>()
    }).collect();
    Ok([level0, level1])
}

/// `zkstat --zk --n N`: N real sessions (fresh prover seeds) against one verifier coin seed, N simulated transcripts from those
/// coins, and N control sessions (zero pads, zero mask slot: M0's values in M1's format), each rep's checked fields printed as
/// `ZKREAL`, `ZKSIM`, `ZKCONTROL` lines, and each session's cross-rep functionals ([`session_relations`]) as `ZKREL` lines, for
/// `backends/flock/tests/zk_stats.py`. Needs the `seed-injection` harness.
#[cfg(feature = "seed-injection")]
fn zkstat(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let n: usize = arg("--n", "32").parse().unwrap();
    let lay = zk_layout(st);
    let one = |plan: &Plan| -> (Value, Vec<Vec<u8>>, Value) {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
        let (v, p) = session(st, inst, shared(InProc(s0.clone())), plan, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        (v, p, rec)
    };
    let fields = |kind: &str, run: usize, rep: usize, pr: &R1csProofLigerito, inner: &InnerProof| proof_fields(&lay, kind, run, rep, pr, inner);
    let relations = |kind: &str, run: usize, proofs: &[Vec<u8>], coins: &[Vec<F128>], points: &[Vec<F128>]| {
        let [l0, l1] = session_relations(st, inst, proofs, coins, points, [0, 1]).unwrap_or_else(|e| refuse(e));
        println!("ZKREL\t{}", json!({"kind": kind, "run": run, "rep": 0, "rel_level0": f128s_hex(&l0), "rel_level1": f128s_hex(&l1)}));
    };
    let mut tape: Option<(Vec<Vec<F128>>, Vec<Vec<F128>>)> = None;
    let mut dummies: Vec<Vec<Vec<u8>>> = Vec::new();
    // each rep's level-0 positions of the sampled rows (`lig_rows0`), from the verifier's record, and its inner proof's opened
    // columns (its last `Q` coins); the coins fix both
    let mut q0: Vec<Vec<usize>> = vec![Vec::new(); 2];
    let inner_n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
    let inner_q = |coins: &[F128]| zkv::query_positions(&coins[coins.len() - zkv::Q..], inner_n);
    for (kind, plan) in [("real", Plan::default()), ("control", Plan { zk_zero_pads: true, ..Plan::default() }),
                         ("dummy", Plan { zero_witness: true, ..Plan::default() })] {
        for run in 0..n {
            let (v, proofs, rec) = one(&plan);
            assert!(v["accepted"] == true || kind == "dummy", "{kind} session {run} rejected: {}", v["why"]);
            let t = ((0..2).map(|r| record_coins(&rec, &rep_stream(TABLE, r))).collect::<Vec<_>>(), record_points(&rec));
            match &tape {
                None => tape = Some(t),
                Some(t0) => assert!(*t0 == t, "the verifier's coins differ between sessions with one coin seed"),
            }
            if kind == "dummy" {
                dummies.push(proofs);
                continue;
            }
            for (rep, p) in proofs.iter().enumerate() {
                let (c, pr, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(p).unwrap();
                if run == 0 {
                    let (coins, points) = tape.as_ref().unwrap();
                    let ok = verify_zk_with(st, inst, &c, &pr, &inner, points, rep, &mut zkv::Tape::new(&coins[rep]));
                    assert!(ok.is_ok(), "a {kind} proof is rejected on its own coin tape: {ok:?}");
                    let root: [u8; 64] = inner.root.clone().try_into().unwrap();
                    let (_, rec) = zk_constraints(st, inst, &c, &pr, &root, points, rep, &mut zkv::Tape::new(&coins[rep])).unwrap_or_else(|e| refuse(e));
                    q0[rep] = rec.queries0[..4].to_vec();
                }
                let mut row = fields(kind, run, rep, &pr, &inner);
                row["lig_rows0_q"] = json!(q0[rep]);
                row["inner_open_q"] = json!(inner_q(&tape.as_ref().unwrap().0[rep]));
                println!("ZK{}\t{}", kind.to_uppercase(), row);
            }
            let (coins, points) = tape.as_ref().unwrap();
            relations(kind, run, &proofs, coins, points);
        }
    }
    let (coins, points) = tape.unwrap();
    let mut rng = flock_live::zk_hooks::ProverRng::from_os();
    let mut accepted = 0;
    for (run, dummy) in dummies.iter().enumerate() {
        let mut sims = Vec::new();
        for rep in 0..2 {
            let (c, pr, inner) = simulate_full(st, inst, &coins[rep], &points, &dummy[rep], &mut rng, rep, (2 * run + rep) as u64).unwrap_or_else(|e| refuse(e));
            // the simulated proof must be accepted by the verifier on the same coins
            let ok = verify_zk_with(st, inst, &c, &pr, &inner, &points, rep, &mut zkv::Tape::new(&coins[rep]));
            assert!(ok.is_ok(), "the verifier rejects a simulated proof: {ok:?}");
            accepted += 1;
            let mut row = fields("sim", run, rep, &pr, &inner);
            row["lig_rows0_q"] = json!(q0[rep]);
            row["inner_open_q"] = json!(inner_q(&coins[rep]));
            println!("ZKSIM\t{row}");
            sims.push(bincode::serialize(&(&c, &pr, &inner)).unwrap());
        }
        relations("sim", run, &sims, &coins, &points);
    }
    eprintln!("ZKSTAT\t{}", json!({"sessions_per_kind": n, "simulated_proofs_accepted": accepted}));
}

#[cfg(not(feature = "seed-injection"))]
fn zkstat(_: &Arc<Stmt>, _: &Arc<Instances>) {
    refuse("zkstat needs the test harness's seed-injection feature".into());
}

/// A framed transcript item (transcript-v2: a 16-byte header `[op][kind][0;6][len u64 LE]`, the payload padded to 16).
enum Item {
    Framing(usize),
    Label(Vec<u8>, usize),
    Bytes(usize),
    Obs(Vec<F128>),
}

fn parse_items(b: &[u8]) -> Result<Vec<Item>, String> {
    let (mut at, mut out) = (0usize, Vec::new());
    let pad = |n: usize| n.div_ceil(16) * 16;
    while at < b.len() {
        if at + 16 > b.len() {
            return Err("a truncated frame header".into());
        }
        let (op, len) = (b[at], u64::from_le_bytes(b[at + 8..at + 16].try_into().unwrap()) as usize);
        at += 16;
        match op {
            OP_DOMAIN => { out.push(Item::Framing(16 + pad(len))); at += pad(len); }
            OP_LABEL => { out.push(Item::Label(b[at..at + len].to_vec(), 16 + pad(len))); at += pad(len); }
            OP_BYTES => { out.push(Item::Bytes(len)); out.push(Item::Framing(16 + pad(len) - len)); at += pad(len); }
            OP_OBSERVE => {
                out.push(Item::Obs((0..len).map(|i| F128::new(u64::from_le_bytes(b[at + 16 * i..at + 16 * i + 8].try_into().unwrap()),
                                                              u64::from_le_bytes(b[at + 16 * i + 8..at + 16 * i + 16].try_into().unwrap()))).collect()));
                out.push(Item::Framing(16));
                at += 16 * len;
            }
            OP_SQUEEZE => out.push(Item::Framing(16)),
            x => return Err(format!("unknown frame op {x}")),
        }
    }
    if at != b.len() {
        return Err("a frame overruns its round".into());
    }
    Ok(out)
}

/// Every item one rep's stream sent, attributed to a class. Returns the tally, the masked values (algebraic phase, in order)
/// and each claim's `s_hat_v` (in opening order).
fn classify(items: &[Item]) -> (BTreeMap<&'static str, (usize, usize)>, Vec<F128>, Vec<Vec<F128>>) {
    #[derive(PartialEq)]
    enum Ph { Pre, R1cs(u8), Zc, Pads, Pcs, Rs, LigPre, L0, L0Post(u8), Lig1, Inner(u8) }
    let mut t: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut add = |c: &'static str, bytes: usize| { let e = t.entry(c).or_default(); e.0 += 1; e.1 += bytes; };
    let (mut ph, mut masked, mut rs) = (Ph::Pre, Vec::new(), Vec::new());
    for it in items {
        match it {
            Item::Framing(n) => add("public: framing (domain, labels, headers, padding)", *n),
            Item::Label(l, n) => {
                add("public: framing (domain, labels, headers, padding)", *n);
                ph = match l.as_slice() {
                    b"flock-r1cs-v0" => Ph::R1cs(0),
                    x if x == zkv::LABEL_ZC => Ph::Zc,
                    x if x == zkv::LABEL_PADS => Ph::Pads,
                    b"flock-lincheck-v0" => Ph::Zc,
                    x if x == zkv::LABEL_PCS => Ph::Pcs,
                    b"flock-ring-switch-v0" => Ph::Rs,
                    b"flock-ligerito-basis-f256-split-v0" => Ph::LigPre,
                    x if x == zkv::L0_OPEN => Ph::L0,
                    x if x == zkv::L0_CLOSE => Ph::L0Post(0),
                    x if x == zkv::LABEL_INNER => Ph::Inner(0),
                    _ => ph,
                };
            }
            Item::Bytes(n) => {
                let c = match &mut ph {
                    Ph::R1cs(k) => { *k += 1; if *k == 1 { "public: statement digest" } else { "commitment: witness cap (hm96-sha512 leaves)" } }
                    Ph::Pads => "commitment: pads root (hm96-sha512 leaves)",
                    _ if *n == 8 => "public: pow nonce (live coins: always 0)",
                    Ph::LigPre | Ph::L0Post(_) | Ph::Lig1 => "commitment: ligerito caps (hm96-sha512 leaves)",
                    Ph::Inner(_) if *n == 64 => "commitment: inner tau (an hm96-sha512 leaf, opened after the last coin)",
                    _ => "UNATTRIBUTED",
                };
                add(c, *n);
            }
            Item::Obs(v) => {
                let n = 16 * v.len();
                match &mut ph {
                    Ph::Zc | Ph::Pads => { masked.extend_from_slice(v); add("masked: zerocheck and lincheck values + committed pads", n) }
                    Ph::Rs => {
                        let c = if rs.len() < 2 { "masked: s_hat_v of the two masked claims (uniform mask-slot words)" } else { "public: s_hat_v of the region claims (public region data)" };
                        rs.push(v.clone());
                        add(c, n);
                        ph = Ph::Pcs;
                    }
                    Ph::LigPre => add("public: the opening's batched claim (from the ring switches' s_hat_v)", n),
                    Ph::L0 => { masked.extend_from_slice(v); add("masked: ligerito level 0's ood value, lane-round messages, extra lanes' sums + committed pads", n) }
                    Ph::L0Post(k) => {
                        *k += 1;
                        match *k {
                            1 => add("masked: T' (the uniform level-1 table's claim)", n),
                            2 => add("masked: ybar (the padding's combination, uniform through the extra lanes' padding)", n),
                            _ => { add("masked: ligerito levels 1+ (functions of the uniform level-1 table)", n); ph = Ph::Lig1; }
                        }
                    }
                    Ph::Lig1 => add("masked: ligerito levels 1+ (functions of the uniform level-1 table)", n),
                    Ph::Inner(k) => {
                        *k += 1;
                        add(if *k == 1 { "masked: inner rho, sigma (a fresh triple)" } else { "masked: inner Y (the dot-product mask)" }, n)
                    }
                    _ => add("UNATTRIBUTED", n),
                }
            }
        }
    }
    (t, masked, rs)
}

/// `zkaudit --zk`: nothing leaves the prover but committed, masked or public data. One real session and one control session
/// (zero pads, zero mask slot) against one verifier coin seed; every byte of every round the verifier kept is parsed and
/// attributed; the masked values must all differ from the control's (the unmasked values), the region claims' `s_hat_v` must
/// equal their recomputation from the public file alone, and nothing may be unattributed. The proofs' opened data is
/// attributed too. `complete` holds only when no class is in the clear.
#[cfg(feature = "seed-injection")]
fn zkaudit(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let run = |plan: &Plan| -> (Value, Vec<Vec<u8>>, Value) {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
        let (v, p) = session(st, inst, shared(InProc(s0.clone())), plan, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        (v, p, rec)
    };
    let gpu = flag("--gpu");
    let (v, proofs, rec) = run(&Plan { gpu, ..Plan::default() });
    let (vc, proofs_c, recc) = run(&Plan { gpu, zk_zero_pads: true, ..Plan::default() });
    assert!(v["accepted"] == true && vc["accepted"] == true, "the audit's sessions must be accepted");
    let pts: Vec<Vec<F128>> = rec["link"]["points"].as_array().unwrap().iter().map(|p| p.as_array().unwrap().iter().map(|h| {
        let b = hexb(h.as_str().unwrap()).unwrap();
        F128::new(u64::from_be_bytes(b[8..16].try_into().unwrap()), u64::from_be_bytes(b[..8].try_into().unwrap()))
    }).collect()).collect();
    let stream_items = |rec: &Value, name: &str| -> Vec<Item> {
        let s = rec["streams"].as_array().unwrap().iter().find(|s| s["stream"] == name).unwrap_or_else(|| panic!("no stream {name}"));
        s["rounds"].as_array().unwrap().iter().flat_map(|r| parse_items(&hexb(r["msg"].as_str().unwrap_or("")).unwrap()).unwrap()).collect()
    };
    let mut total: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let (mut all_differ, mut region_public, mut n_masked, mut n_region) = (true, true, 0usize, 0usize);
    let mut lig_differ: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for rep in 0..2 {
        let name = rep_stream(TABLE, rep);
        let (t, m, rs) = classify(&stream_items(&rec, &name));
        let (_, mc, rsc) = classify(&stream_items(&recc, &name));
        for (k, (a, b)) in t {
            let e = total.entry(k).or_default();
            e.0 += a;
            e.1 += b;
        }
        n_masked += m.len();
        all_differ &= m.len() == mc.len() && m.iter().zip(&mc).all(|(x, y)| x != y);
        all_differ &= rs[..2].iter().zip(&rsc[..2]).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x != y));
        let mut k = 2;
        for reg in &st.regions {
            for p in &pts {
                let want = st.region_s_hat_v(reg, p, inst);
                region_public &= rs.get(k) == Some(&want) && rsc.get(k) == Some(&want);
                n_region += 1;
                k += 1;
            }
        }
        let (cm, pr, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(&proofs[rep]).unwrap();
        let (_, prc, _): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(&proofs_c[rep]).unwrap();
        let lg = &pr.pcs_open.ligerito;
        // Ligerito's opened data against the control's (same coins, so the same positions): every value should differ
        let flat = |l: &flock_core::pcs::ligerito::LigeritoProof| -> [Vec<F128>; 3] {
            [l.initial_proof.opened_rows.concat(),
             l.recursive_proofs.iter().flat_map(|p| p.opened_rows.concat()).chain(l.final_proof.opened_rows.concat()).collect(),
             l.final_proof.yr.clone()]
        };
        for (k, (a, b)) in flat(lg).iter().zip(flat(&prc.pcs_open.ligerito).iter()).enumerate() {
            let e = lig_differ.entry(["level-0 rows", "levels 1+ rows", "final vector"][k]).or_insert((0usize, 0usize));
            e.0 += a.iter().zip(b).filter(|(x, y)| x != y).count();
            e.1 += a.len().min(b.len());
        }
        let rows = |p: &flock_core::pcs::ligerito::RecursiveProof| p.opened_rows.iter().map(|r| 16 * r.len()).sum::<usize>();
        let mut add = |c: &'static str, b: usize| { let e = total.entry(c).or_default(); e.0 += 1; e.1 += b; };
        add("masked (proof): ligerito level-0 rows ((W_n + c) mu padding per lane, uniform extra lanes)", rows(&lg.initial_proof));
        add("masked (proof): ligerito levels 1+ rows (functions of the uniform level-1 table)", lg.recursive_proofs.iter().map(rows).sum::<usize>()
            + lg.final_proof.opened_rows.iter().map(|r| 16 * r.len()).sum::<usize>());
        add("commitment (proof): merkle paths and opened salts (hm96-sha512)", 64 * (lg.initial_proof.merkle_proof.len()
            + lg.recursive_proofs.iter().map(|p| p.merkle_proof.len()).sum::<usize>() + lg.final_proof.merkle_proof.len())
            + 192 * (lg.initial_proof.opened_salts.len() + lg.recursive_proofs.iter().map(|p| p.opened_salts.len()).sum::<usize>()
                     + lg.final_proof.opened_salts.len()));
        add("masked (proof): inner opened columns (rs with 192 uniform padding symbols)", 32 * inner.opened.len());
        add("commitment (proof): inner paths and salts (hm96-sha512)", 64 * inner.paths.iter().map(|p| p.len()).sum::<usize>() + 192 * inner.salts.len());
        add("masked (proof): inner tau (<c, mu>, mu uniform) and its commitment's salt", 16 + inner.tau_salt.len());
        add("masked (proof): final_c' (never absorbed; its pad h_fc)", 16);
        let nonces_zero = pr.pcs_open.batching_nonces.iter().all(|x| *x == 0);
        add(if nonces_zero { "public (proof): the opening's batching nonces (live coins: always 0)" } else { "CLEAR (proof): nonzero batching nonces" },
            8 * pr.pcs_open.batching_nonces.len());
        add("public (proof): the commitment's parameters", bincode::serialized_size(&cm.params).unwrap() as usize);
    }
    let clear: Vec<Value> = total.iter().filter(|(k, _)| k.starts_with("CLEAR")).map(|(k, (_, b))| json!({"class": k, "bytes": b})).collect();
    let unattributed = total.get("UNATTRIBUTED").map(|x| x.1).unwrap_or(0);
    let lc = flock_core::pcs::ligerito::zk::prover_config(params(st.m, LigeritoProfile::Fast100).ligerito_prover_config().unwrap(), zk_queries0(st.m).unwrap());
    let ligerito = json!({"m": st.m, "t_pad0": zk_t_pad(st.m), "initial_k": lc.initial_k, "initial_log_msg_cols": lc.initial_log_msg_cols, "recursive_ks": lc.recursive_ks,
        "recursive_log_msg_cols": lc.recursive_log_msg_cols, "log_inv_rates": lc.log_inv_rates, "queries": lc.queries, "ood_samples": lc.ood_samples,
        "grinding_bits": lc.grinding_bits});
    let out = json!({"statement": STATEMENT, "unit": st.c.name, "instances": st.n, "ligerito": ligerito,
        "classes": total.iter().map(|(k, (n, b))| (k.to_string(), json!({"items": n, "bytes": b}))).collect::<serde_json::Map<_, _>>(),
        "masked_values": n_masked, "masked_all_differ_from_control": all_differ, "region_claims": n_region, "region_s_hat_v_is_public": region_public,
        "unattributed_bytes": unattributed, "clear": clear, "checks_pass": all_differ && region_public && unattributed == 0,
        "ligerito_values_differing_from_control": lig_differ.iter().map(|(k, (d, n))| (k.to_string(), json!({"differ": d, "of": n}))).collect::<serde_json::Map<_, _>>(),
        "attributed": all_differ && region_public && unattributed == 0 && clear.is_empty()});
    println!("ZKAUDIT\t{out}");
}

/// M2's malicious verifiers V*: the honest coin server with its coins committed from a random tape, and a strategy that sees
/// every prover message. Rewinding V* is a fresh instance on the same tape, replayed to the same point. `Abort` and
/// `AdaptiveCoins` act at the first round whose message, hashed with the tape, falls below [`TRIGGER`]: the first stops
/// there; the second answers a coin it picks from that message, under the committed coin's opening. `CoinAt(k)` does the
/// latter at rep 0's round `k`. `ZeroBeta(k)` commits to β = 0 at rep 0's round `k`. The gated strategies decide on one
/// message whether to continue for sure (half of those messages, "good") or only with probability `ε / 65536`:
/// `CapGated(ε)` on the witness cap (red-team-m2-zk's; it stops at rep 0's first round), `LateGated(ε)` on rep 0's round
/// that carries `T'` and `ybar` (it stops there). A simulator that reused messages across rewinds would fail against them.
#[cfg(feature = "seed-injection")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Strategy {
    Honest,
    Abort,
    AdaptiveCoins,
    CoinAt(usize),
    ZeroBeta(usize),
    CapGated(u16),
    LateGated(u16),
}

#[cfg(feature = "seed-injection")]
/// The adaptive strategies' trigger per round, of 65536: `(1 − q)^R ≈ 1/2` over a RoPE session's 200 rounds.
const TRIGGER: u16 = 229;

#[cfg(feature = "seed-injection")]
struct VStar {
    s: Server,
    strategy: Strategy,
    tape: [u8; 32],
    rounds: BTreeMap<String, usize>,
    stopped: bool,
    /// the gated strategies' verdict on their message (for the test harness, which conditions on it)
    gate: Arc<Mutex<Option<bool>>>,
    /// V*'s view: every request it received, with its answer
    view: Arc<Mutex<Vec<(Req, Resp)>>>,
}

#[cfg(feature = "seed-injection")]
impl VStar {
    fn new(st: &Arc<Stmt>, inst: &Arc<Instances>, strategy: Strategy, tape: [u8; 32]) -> Self {
        let mut s = server(st, inst);
        s.inject_coin_tape(tape);
        if let Strategy::ZeroBeta(k) = strategy {
            let rep0 = rep_stream(TABLE, 0);
            s.inject_coin_edit(Box::new(move |stream, round, b| {
                if stream == rep0 && round == k {
                    b[0] = F128::ZERO;
                }
            }));
        }
        VStar { s, strategy, tape, rounds: BTreeMap::new(), stopped: false, gate: Arc::new(Mutex::new(None)), view: Arc::new(Mutex::new(Vec::new())) }
    }

    fn hash(&self, what: &[u8], stream: &str, k: usize, content: &[u8]) -> [u8; 64] {
        let mut h = Sha512::new();
        for p in [&self.tape[..], what, stream.as_bytes(), &(k as u64).to_le_bytes(), content] {
            h.update(p);
        }
        h.finalize().into()
    }

    fn answer(&mut self, req: Req) -> Resp {
        if self.stopped {
            return Resp::Refused("V* stopped".into());
        }
        if let (Strategy::CapGated(_), Req::Commit { roots, .. }) = (self.strategy, &req) {
            let good = self.hash(b"cap", "", 0, &roots[0].1)[0] < 128;
            *self.gate.lock().unwrap_or_else(PoisonError::into_inner) = Some(good);
        }
        let round = if let Req::Round { stream, content, .. } = &req { Some((stream.clone(), content.clone())) } else { None };
        let Some((stream, content)) = round else { return self.s.handle(req) };
        let k = self.rounds.get(&stream).copied().unwrap_or(0);
        self.rounds.insert(stream.clone(), k + 1);
        let rep0 = stream == rep_stream(TABLE, 0);
        let pass = |v: &Self, eps: u16| u16::from_le_bytes(v.hash(b"gate", &stream, k, &content)[..2].try_into().unwrap()) < eps;
        let fire = match self.strategy {
            Strategy::Abort | Strategy::AdaptiveCoins => u16::from_le_bytes(self.hash(b"trigger", &stream, k, &content)[..2].try_into().unwrap()) < TRIGGER,
            Strategy::CoinAt(r) => rep0 && k == r,
            Strategy::CapGated(eps) => rep0 && k == 0 && *self.gate.lock().unwrap_or_else(PoisonError::into_inner) == Some(false) && !pass(self, eps),
            Strategy::LateGated(eps) if rep0 && content.windows(zkv::L0_CLOSE.len()).any(|w| w == zkv::L0_CLOSE) => {
                let good = self.hash(b"late", &stream, k, &content)[0] < 128;
                *self.gate.lock().unwrap_or_else(PoisonError::into_inner) = Some(good);
                !good && !pass(self, eps)
            }
            _ => false,
        };
        if fire && matches!(self.strategy, Strategy::Abort | Strategy::CapGated(_) | Strategy::LateGated(_)) {
            self.stopped = true;
            return Resp::Refused(format!("V* aborts at {stream} round {k}"));
        }
        let pick = fire.then(|| self.hash(b"coin", &stream, k, &content));
        match (self.s.handle(req), pick) {
            (Resp::Coins(mut w), Some(h)) => {
                w[0] = F128::new(u64::from_le_bytes(h[..8].try_into().unwrap()), u64::from_le_bytes(h[8..16].try_into().unwrap()));
                Resp::Coins(w)
            }
            (r, _) => r,
        }
    }
}

#[cfg(feature = "seed-injection")]
impl Transport for VStar {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let req = Req::decode(&req.encode())?;
        let resp = self.answer(req.clone());
        self.view.lock().unwrap_or_else(PoisonError::into_inner).push((req, resp.clone()));
        Resp::decode(&resp.encode())
    }
}

#[cfg(feature = "seed-injection")]
type View = Vec<(Req, Resp)>;

#[cfg(feature = "seed-injection")]
fn complete(view: &View) -> bool {
    view.iter().any(|(r, _)| matches!(r, Req::Proof { .. }))
}

#[cfg(feature = "seed-injection")]
/// Per rep, its rounds' coins as V* opened them in a view, each opening checked against V*'s commitment as the prover checks
/// it; and the link points.
fn extracted(st: &Stmt, view: &View) -> Result<(Vec<Vec<F128>>, Vec<Vec<Vec<F128>>>), String> {
    use flock_live::coin_tree::{KEY_BYTES, Rules, check, hello_nonce};
    let spec = coin_spec();
    let mut tree: Option<([u8; 64], Rules)> = None;
    let (mut points, mut reps): (Vec<Vec<F128>>, Vec<Vec<Vec<F128>>>) = (Vec::new(), vec![Vec::new(), Vec::new()]);
    for (req, resp) in view {
        let Resp::Coins(w) = resp else { continue };
        match req {
            // the prover's nonce from its Hello, V*'s root and session key from the answer
            Req::Hello(h) => {
                let b = f128_bytes(w);
                if b.len() != 64 + KEY_BYTES {
                    return Err(format!("V*'s answer to Hello is {} bytes, not the root and the session key", b.len()));
                }
                tree = Some((b[..64].try_into().unwrap(), Rules::v2(&hello_nonce(h)?, &b[64..].try_into().unwrap())));
            }
            Req::Commit { .. } => {
                let (root, rules) = tree.as_ref().ok_or("coins before the commitment")?;
                let n = w.len().checked_sub(spec.opening_words(0)).ok_or("a short link-point opening")?;
                points = check(&spec, rules, root, 0, 0, n, w)?.chunks(st.m_pts).map(|p| p.to_vec()).collect();
            }
            Req::Round { stream, n, .. } => {
                let (root, rules) = tree.as_ref().ok_or("coins before the commitment")?;
                let rep = (0..2).find(|&r| *stream == rep_stream(TABLE, r)).ok_or(format!("a round of {stream}"))?;
                let si = spec.stream_index(stream).ok_or(format!("{stream} is outside the schedule"))?;
                let k = reps[rep].len();
                reps[rep].push(check(&spec, rules, root, si, k, *n as usize, w)?);
            }
            _ => {}
        }
    }
    Ok((points, reps))
}

/// The `Hello` a view's session sent (its coin-tree nonce: the simulator's rewinds restart after it).
#[cfg(feature = "seed-injection")]
fn view_hello(view: &View) -> Result<String, String> {
    view.iter().find_map(|(req, _)| if let Req::Hello(h) = req { Some(h.clone()) } else { None }).ok_or("the view has no Hello".into())
}

#[cfg(feature = "seed-injection")]
enum Rewound {
    Complete,
    /// V* stopped the simulation (refused, or answered a coin its commitment does not open)
    Aborted,
}

#[cfg(feature = "seed-injection")]
/// One rewind of the Goldreich–Kahan simulator: a fresh V* on its tape, from the start of the session to its end, against a
/// scripted prover whose messages are SHVZK simulations from the extracted coins (`simulate_full` per rep, its rounds recorded
/// by replaying the verifier on it). The session's messages up to the coin commitment are the extraction run's (so V* is back
/// just after step 0); every coin V* answers is checked against its commitment and must be the extracted one (else the
/// commitment's binding failed: an error).
#[allow(clippy::too_many_arguments)]
fn rewind(st: &Stmt, inst: &Instances, v: impl Transport + 'static, hello: &str, points: &[Vec<F128>], coins: &[Vec<Vec<F128>>], dummy: &[Vec<u8>],
          rng: &mut flock_live::zk_hooks::ProverRng, index: u64) -> Result<Rewound, String> {
    let mut sims = Vec::new();
    for rep in 0..2 {
        let (c, pr, inner) = simulate_full(st, inst, &coins[rep].concat(), points, &dummy[rep], rng, rep, 2 * index + rep as u64)?;
        let mut rec = flock_live::RecordingChallenger::new(&domain(rep), coins[rep].clone());
        verify_zk_with(st, inst, &c, &pr, &inner, points, rep, &mut rec).map_err(|e| format!("rep {rep}'s simulation is rejected on the extracted coins: {e}"))?;
        if let Some(e) = rec.mismatch.or_else(|| (rec.rounds.len() != coins[rep].len()).then(|| format!("{} rounds, extracted {}", rec.rounds.len(), coins[rep].len()))) {
            return Err(format!("rep {rep}'s simulation does not follow the extracted schedule: {e}"));
        }
        sims.push((bincode::serialize(&(&c, &pr, &inner)).unwrap(), merkle_digest_bytes(&c.cap).to_vec(), rec.rounds));
    }
    let t: SharedTransport = shared(flock_live::coin_tree::Checked::new(v, coin_spec()));
    macro_rules! expect {
        ($req:expr, $ok:pat) => {
            match call(&t, $req)? {
                $ok => {}
                _ => return Ok(Rewound::Aborted),
            }
        };
    }
    expect!(Req::Register, Resp::Draw(_));
    expect!(Req::Hello(hello.to_string()), Resp::Coins(_));
    for rep in 0..2 {
        expect!(Req::Open { stream: rep_stream(TABLE, rep), parent: String::new(), label: domain(rep), at: 0 }, Resp::Ok);
        if rep == 0 {
            match call(&t, Req::Commit { root_f: sigma(st, inst).to_vec(), roots: vec![(TABLE.into(), sims[0].1.clone())], publics: vec![] })? {
                Resp::Coins(c) if c.chunks(st.m_pts).eq(points.iter().map(|p| p.as_slice())) => {}
                Resp::Coins(_) => return Err("binding: V* opened other link points than in the extraction run".into()),
                _ => return Ok(Rewound::Aborted),
            }
            expect!(Req::Link(f128_bytes(&[F128::ZERO, F128::ZERO])), Resp::Ok);
        }
        for (k, (content, n)) in sims[rep].2.iter().enumerate() {
            match call(&t, Req::Round { stream: rep_stream(TABLE, rep), content: content.clone(), n: *n as u32 })? {
                Resp::Coins(c) if c == coins[rep][k] => {}
                Resp::Coins(_) => return Err(format!("binding: V* opened other coins at rep {rep} round {k} than in the extraction run")),
                _ => return Ok(Rewound::Aborted),
            }
        }
    }
    for (rep, s) in sims.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: s.0.clone() })?;
    }
    call(&t, Req::Finish)?;
    Ok(Rewound::Complete)
}

/// A fresh dummy opening for the extracted coins, with no interaction with V*: the honest prover on the dummy witness (fresh
/// masks, level-0 padding, extra lanes and salts) against a local coin server that commits to exactly those coins.
#[cfg(feature = "seed-injection")]
fn fresh_dummy(st: &Arc<Stmt>, inst: &Arc<Instances>, points: &[Vec<F128>], coins: &[Vec<Vec<F128>>]) -> Result<Vec<Vec<u8>>, String> {
    let mut s = server(st, inst);
    let (pts, cs) = (points.concat(), coins.to_vec());
    s.inject_coin_edit(Box::new(move |stream, round, b| {
        if stream == flock_live::coin_tree::SESSION && round == 0 {
            b[..pts.len()].copy_from_slice(&pts);
        } else if let Some(c) = (0..2).find(|&r| stream == rep_stream(TABLE, r)).and_then(|r| cs[r].get(round)) {
            b[..c.len()].copy_from_slice(c);
        }
    }));
    let (v, proofs) = session(st, inst, shared(InProc(Arc::new(Mutex::new(s)))), &Plan { zero_witness: true, ..Plan::default() }, ProverSeed::from_os());
    if proofs.len() != 2 {
        return Err(format!("the fresh dummy run on the extracted coins did not reach the proofs: {}", v["prover_stopped"]));
    }
    Ok(proofs)
}

/// The Goldreich–Kahan simulator (soundness ASSUMPTIONS.md §6): V*'s view of one session from the public statement alone.
/// 1. Extraction: the honest prover on a dummy witness (zero, the public regions written in) against V*. Every message before
///    the proofs is masked, committed or public, so this run is distributed as a real one until the proofs, which follow the
///    last coin. If it does not reach the proofs (V* aborted or cheated, or the prover refused a degenerate coin), its view
///    is the output.
/// 2. Estimation (`zk_veil::gk_estimate`): fresh dummy runs against V*, uncapped, until `t` of them reach the proofs;
///    `p̃ = t / runs`, the extraction run not counted. `t = Θ(λ)`.
/// 3. At most `4t / p̃` rewinds to just after the coin commitment, each with a fresh dummy opening for the extracted coins
///    (`fresh_dummy`), so the rewinds are independent trials; the first that V* does not stop is the output. Failing that,
///    or if V* opens another coin than it did in step 1 (a binding failure), the simulator fails.
#[cfg(feature = "seed-injection")]
fn gk_simulate(st: &Arc<Stmt>, inst: &Arc<Instances>, make: &dyn Fn() -> VStar, t: usize) -> (View, Value) {
    // the extraction run draws the coin-tree nonce as the prover does; every later run restarts after its Hello
    let dummy_run = |nonce: Option<[u8; 32]>| -> View {
        let v = make();
        let view = v.view.clone();
        session(st, inst, shared(v), &Plan { zero_witness: true, coin_nonce: nonce, ..Plan::default() }, ProverSeed::from_os());
        let log = view.lock().unwrap_or_else(PoisonError::into_inner).clone();
        log
    };
    let view = dummy_run(None);
    if !complete(&view) {
        return (view, json!({"extraction_complete": false}));
    }
    let (points, coins, hello, nonce) = match extracted(st, &view).and_then(|(p, c)| {
        let h = view_hello(&view)?;
        let n = flock_live::coin_tree::hello_nonce(&h)?;
        Ok((p, c, h, n))
    }) {
        Ok(x) => x,
        Err(e) => return (Vec::new(), json!({"extraction_complete": true, "fail": format!("extraction: {e}")})),
    };
    let runs = zkv::gk_estimate(t, || complete(&dummy_run(Some(nonce))));
    let bound = zkv::gk_rewinds(runs);
    let mut rng = flock_live::zk_hooks::ProverRng::from_os();
    let stats = |rewinds: usize, fail: Option<String>| json!({"extraction_complete": true, "estimation_runs": runs, "estimation_complete": t,
                                                              "rewinds": rewinds, "bound": bound, "fail": fail});
    for attempt in 0..bound {
        let dummy = match fresh_dummy(st, inst, &points, &coins) {
            Ok(d) => d,
            Err(e) => return (Vec::new(), stats(attempt + 1, Some(e))),
        };
        let v = make();
        let view = v.view.clone();
        match rewind(st, inst, v, &hello, &points, &coins, &dummy, &mut rng, attempt as u64) {
            Ok(Rewound::Complete) => return (view.lock().unwrap_or_else(PoisonError::into_inner).clone(), stats(attempt + 1, None)),
            Ok(Rewound::Aborted) => continue,
            Err(e) => return (Vec::new(), stats(attempt + 1, Some(e))),
        }
    }
    (Vec::new(), stats(bound, Some(format!("no rewind reached the proofs in {bound} (p̃ = {t}/{runs})"))))
}

#[cfg(feature = "seed-injection")]
/// Masked values of rep 0's first rounds kept per view (`online`: present whatever the end, once the view reaches them).
const ONLINE: usize = 256;

#[cfg(feature = "seed-injection")]
/// A view's summary: how it ended (`complete`, or who stopped it after which round), the rounds V* received per stream, its
/// verdict, rep 0's first [`ONLINE`] masked values, and each proof it received (for [`proof_fields`]).
fn view_summary(kind: &str, strategy: &str, run: usize, view: &View) -> (Value, Vec<(usize, Vec<u8>)>) {
    let mut rounds: BTreeMap<String, usize> = BTreeMap::new();
    let (mut who, mut at, mut online, mut proofs, mut verdict) = ("none", Value::Null, Vec::new(), Vec::new(), Value::Null);
    for (req, resp) in view {
        match req {
            Req::Round { stream, content, .. } => {
                let k = rounds.entry(stream.clone()).or_default();
                if *stream == rep_stream(TABLE, 0) && online.len() < ONLINE {
                    online.extend(parse_items(content).unwrap_or_default().into_iter().flat_map(|it| if let Item::Obs(v) = it { v } else { vec![] }));
                }
                who = if matches!(resp, Resp::Refused(_)) { "verifier" } else { "prover" };
                at = json!([stream, *k]);
                *k += 1;
            }
            Req::Proof { rep, bytes, .. } => {
                proofs.push((*rep as usize, bytes.clone()));
                who = "complete";
            }
            Req::Finish => {
                if let Resp::Verdict(v) = resp {
                    verdict = serde_json::from_str(v).unwrap_or(Value::Null);
                }
            }
            _ => {}
        }
    }
    let online = (online.len() >= ONLINE).then(|| f128s_hex(&online[..ONLINE]));
    (json!({"kind": kind, "strategy": strategy, "run": run, "end": who, "last_round": at, "rounds": rounds, "accepted": verdict["accepted"],
            "online": online}), proofs)
}

#[cfg(feature = "seed-injection")]
/// Rep 0's β round: the one whose message is the τ commitment alone (a verifier knows the protocol's shape).
fn beta_round(view: &View) -> Option<usize> {
    let rep0 = rep_stream(TABLE, 0);
    view.iter().filter_map(|(r, _)| match r { Req::Round { stream, content, .. } if *stream == rep0 => Some(content), _ => None })
        .enumerate()
        .filter(|(_, c)| matches!(parse_items(c).as_deref(), Ok([Item::Bytes(64), Item::Framing(_), Item::Framing(16)])))
        .map(|(k, _)| k)
        .last()
}

/// `zkrewind --zk --n N`: per verifier strategy (`--strategies`, default all), N real sessions and N Goldreich–Kahan
/// simulations against V* on one tape, each view printed as `ZKVIEW` (its summary), `ZKVIEWP` (each received proof's
/// checked fields) and, when complete, `ZKVIEWREL` (the session's cross-rep relations) for
/// `backends/flock/tests/zk_rewind_stats.py`, and each strategy's simulator statistics as `ZKREWIND`.
#[cfg(feature = "seed-injection")]
fn zkrewind(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let n: usize = arg("--n", "16").parse().unwrap();
    let t: usize = arg("--gk-t", "4").parse().unwrap();
    let eps: u16 = arg("--eps", "4096").parse().unwrap();
    let tape = [arg("--tape", "7").parse::<u8>().unwrap_or_else(|_| refuse("--tape takes one byte (V*'s tape is that byte repeated)".into())); 32];
    let lay = zk_layout(st);
    let real = |make: &dyn Fn() -> VStar| -> View {
        let v = make();
        let view = v.view.clone();
        session(st, inst, shared(v), &Plan::default(), ProverSeed::from_os());
        let log = view.lock().unwrap_or_else(PoisonError::into_inner).clone();
        log
    };
    let emit = |kind: &str, name: &str, run: usize, view: &View| {
        let (s, proofs) = view_summary(kind, name, run, view);
        println!("ZKVIEW\t{s}");
        let ext = extracted(st, view).ok().map(|(points, coins)| (points, coins.iter().map(|c| c.concat()).collect::<Vec<Vec<F128>>>()));
        if let (2, Some((points, flat))) = (proofs.len(), &ext) {
            let bytes: Vec<Vec<u8>> = proofs.iter().map(|p| p.1.clone()).collect();
            if let Ok([l0, l1]) = session_relations(st, inst, &bytes, flat, points, [0, 1]) {
                println!("ZKVIEWREL\t{}", json!({"kind": kind, "strategy": name, "run": run, "rep": 0, "rel_level0": f128s_hex(&l0),
                                                  "rel_level1": f128s_hex(&l1)}));
            }
        }
        for (rep, b) in proofs {
            if let Ok((c, pr, inner)) = bincode::deserialize::<(Commitment, R1csProofLigerito, InnerProof)>(&b) {
                let mut row = proof_fields(&lay, kind, run, rep, &pr, &inner);
                row["strategy"] = json!(name);
                // the level-0 positions of the sampled rows, from the verifier's record on the view's coins
                if let (Some((points, flat)), Ok(root)) = (&ext, <[u8; 64]>::try_from(inner.root.clone())) {
                    if let Ok((_, rec)) = zk_constraints(st, inst, &c, &pr, &root, points, rep, &mut zkv::Tape::new(&flat[rep])) {
                        row["lig_rows0_q"] = json!(rec.queries0[..4]);
                    }
                    let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
                    row["inner_open_q"] = json!(zkv::query_positions(&flat[rep][flat[rep].len() - zkv::Q..], n));
                }
                println!("ZKVIEWP\t{row}");
            }
        }
    };
    for name in arg("--strategies", "honest,abort,adaptive-coins,zero-beta,cap-gated,late-gated").split(',') {
        let strategy = match name {
            "honest" => Strategy::Honest,
            "abort" => Strategy::Abort,
            "adaptive-coins" => Strategy::AdaptiveCoins,
            "zero-beta" => Strategy::ZeroBeta(beta_round(&real(&|| VStar::new(st, inst, Strategy::Honest, tape))).expect("rep 0's beta round")),
            "cap-gated" => Strategy::CapGated(eps),
            "late-gated" => Strategy::LateGated(eps),
            x => refuse(format!("unknown strategy {x}")),
        };
        let make = || VStar::new(st, inst, strategy, tape);
        let ts = Instant::now();
        for run in 0..n {
            emit("real", name, run, &real(&make));
        }
        let t_real = ts.elapsed().as_secs_f64();
        let mut stats = Vec::new();
        for run in 0..n {
            let (view, s) = gk_simulate(st, inst, &make, t);
            if s["fail"].is_null() {
                emit("sim", name, run, &view);
            }
            stats.push(s);
        }
        println!("ZKREWIND\t{}", json!({"strategy": name, "n": n, "gk_t": t, "real_s": t_real, "sim_s": ts.elapsed().as_secs_f64() - t_real,
                                         "sims": stats}));
    }
}

/// `zkgk --zk --attempts A`: the rewinds are independent trials (red-team-m2-zk G1). For each gated V* (`cap-gated`,
/// `late-gated`, `--eps`), A real sessions give its completion rate; then, conditioned on an extraction run that reached the
/// proofs with its gate "bad", A rewinds with a fresh dummy opening each (as `gk_simulate` does) must complete at the same
/// rate, not at ε. `--reuse-dummy` also runs A rewinds on the extraction run's own dummy opening, the defect, to show the
/// comparison has power. Prints `ZKGK` lines for `backends/flock/tests/zk_rewind_stats.py`.
#[cfg(feature = "seed-injection")]
fn zkgk(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let attempts: usize = arg("--attempts", "40").parse().unwrap();
    let eps: u16 = arg("--eps", "4096").parse().unwrap();
    let tape = [arg("--tape", "7").parse::<u8>().unwrap_or_else(|_| refuse("--tape takes one byte (V*'s tape is that byte repeated)".into())); 32];
    let run = |v: VStar, plan: &Plan| -> (View, Vec<Vec<u8>>, Option<bool>) {
        let (view, gate) = (v.view.clone(), v.gate.clone());
        let (_, proofs) = session(st, inst, shared(v), plan, ProverSeed::from_os());
        let log = view.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let g = *gate.lock().unwrap_or_else(PoisonError::into_inner);
        (log, proofs, g)
    };
    for name in arg("--strategies", "cap-gated,late-gated").split(',') {
        let strategy = match name {
            "cap-gated" => Strategy::CapGated(eps),
            "late-gated" => Strategy::LateGated(eps),
            x => refuse(format!("zkgk takes the gated strategies, not {x}")),
        };
        let make = || VStar::new(st, inst, strategy, tape);
        let real = (0..attempts).filter(|_| complete(&run(make(), &Plan::default()).0)).count();
        let dummy_plan = Plan { zero_witness: true, ..Plan::default() };
        let mut tries = 0;
        let (view, dummy) = loop {
            tries += 1;
            let (v, p, g) = run(make(), &dummy_plan);
            if complete(&v) && g == Some(false) {
                break (v, p);
            }
            if tries > 400 {
                refuse(format!("{name}: no extraction run reached the proofs with its gate bad in 400 tries"));
            }
        };
        let (points, coins) = extracted(st, &view).unwrap_or_else(|e| refuse(e));
        let hello = view_hello(&view).unwrap_or_else(|e| refuse(e));
        let mut rng = flock_live::zk_hooks::ProverRng::from_os();
        let mut done = |dummy: &[Vec<u8>], a: u64| matches!(rewind(st, inst, make(), &hello, &points, &coins, dummy, &mut rng, a), Ok(Rewound::Complete));
        let fresh = (0..attempts).filter(|&a| done(&fresh_dummy(st, inst, &points, &coins).unwrap_or_else(|e| refuse(e)), a as u64)).count();
        let reused = flag("--reuse-dummy").then(|| (0..attempts).filter(|&a| done(&dummy, 1000 + a as u64)).count());
        println!("ZKGK\t{}", json!({"strategy": name, "eps": eps as f64 / 65536.0, "attempts": attempts, "real_complete": real,
                                     "extraction_tries": tries, "rewinds_complete_fresh": fresh, "rewinds_complete_reused": reused}));
    }
}

#[cfg(not(feature = "seed-injection"))]
fn zkgk(_: &Arc<Stmt>, _: &Arc<Instances>) {
    refuse("zkgk needs the test harness's seed-injection feature".into());
}

#[cfg(not(feature = "seed-injection"))]
fn zkrewind(_: &Arc<Stmt>, _: &Arc<Instances>) {
    refuse("zkrewind needs the test harness's seed-injection feature".into());
}

#[cfg(not(feature = "seed-injection"))]
fn zkaudit(_: &Arc<Stmt>, _: &Arc<Instances>) {
    refuse("zkaudit needs the test harness's seed-injection feature".into());
}

/// The device prover's witness source for a rep's witness data, handed to `f`.
#[cfg(feature = "gpu")]
fn with_gpu_src<R>(st: &Stmt, wt: &WitData, f: impl FnOnce(flock_live::gpu_circuit::CircuitSrc<'_>) -> R) -> R {
    use flock_live::gpu_circuit::CircuitSrc;
    let mut mid_fn;
    let src = match wt {
        WitData::Host((z, a, b, _)) => CircuitSrc::Host { z, a, b },
        WitData::Dev(r, h, i) => CircuitSrc::Device { rows: r, host: h, inputs: i },
        WitData::DevCut(r, h, i, ui) => {
            let sched = flock_live::gpu_circuit::schedule(st.c.unit());
            let cut_rows: Vec<usize> = sched.cut_rows.iter().map(|&r| r as usize).collect();
            let late_rows: Vec<usize> = sched.late_rows.iter().map(|&r| r as usize).collect();
            mid_fn = move |cut: &[u32]| st.host_tail(&**ui, cut, &cut_rows, &late_rows);
            CircuitSrc::DeviceCut { rows: r, host: h, inputs: i, mid: &mut mid_fn }
        }
    };
    f(src)
}

#[cfg(feature = "gpu")]
/// A rep's buckets: the device's phases (`t.`), and the host's around it (`h.`: setup, staging the device's inputs, decoding and
/// serializing the proof), which with `t.total` make up the rep's wall time.
fn gpu_buckets(wt: &WitData, ph: [f64; 8], wph: [f64; 3], hph: [f64; 3], reused: bool, native_sha: bool) -> Value {
    json!({"t.witness": ph[0], "t.witness_comp": wph[0], "t.witness_units": wph[1], "t.encoding_commitment": ph[1], "t.link": ph[2],
           "t.zerocheck": ph[3], "t.lincheck": ph[4], "t.ring_switch": ph[5], "t.ligerito": ph[6], "t.total": ph[7],
           "t.arithmetic": ph[3] + ph[4] + ph[5] + ph[6], "h.setup": hph[0], "h.stage": hph[1], "h.decode": hph[2],
           "device_witness": !matches!(wt, WitData::Host(..)), "host_units": matches!(wt, WitData::Dev(_, _, i) if i.is_empty()),
           "rep_reused": reused, "native_sha": native_sha})
}

/// One rep on the GPU; with `reuse_tag`, rep 0 keeps its witness and level-0 commitment on the device and rep 1 takes them.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn prove_gpu(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger,
             salts: &Mutex<flock_live::zk_hooks::LeafSalts>, reuse_tag: Option<u64>, native_sha: bool) -> (Vec<u8>, Value) {
    use flock_live::gpu_circuit::{Live, Reuse, kept, prove_circuit};
    let reuse = match (reuse_tag, rep) {
        (None, _) => Reuse::None,
        (Some(t), 0) => Reuse::Keep(t),
        (Some(t), _) => Reuse::Take(t),
    };
    let reused = matches!(reuse, Reuse::Take(t) if kept(t));
    let (proof, commitment, ph, wph, hph) =
        with_gpu_src(st, wt, |src| prove_circuit(st, src, &params(st.m, lig), &domain(rep), hook, Live::Stream(ch), Some(salts), reuse,
                                                 native_sha));
    let ts = Instant::now();
    let bytes = bincode::serialize(&(commitment, proof)).unwrap();
    let mut bk = gpu_buckets(wt, ph, wph, hph, reused, native_sha);
    bk["h.serialize"] = json!(ts.elapsed().as_secs_f64());
    eprintln!("GPUPHASE\t{bk}");
    (bytes, bk)
}

/// M1 on the device, its device part: every live round the device frames goes to `rounds` (the rep's masking challenger, made
/// again there by `flock_live::replay_round`, so the verifier sees what M1's CPU prover sends); level 0's hiding runs on the device.
/// The caller finishes the rep with [`zk_finish`], as on the CPU.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn gpu_zk_device(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, salts: &Mutex<flock_live::zk_hooks::LeafSalts>,
                 level0: &flock_core::pcs::ligerito::zk::Level0Zk, rounds: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, native_sha: bool)
                 -> (R1csProofLigerito, Commitment, Value) {
    use flock_live::gpu_circuit::{Live, Reuse, prove_circuit};
    let (proof, commitment, ph, wph, hph) =
        with_gpu_src(st, wt, |src| prove_circuit(st, src, &params(st.m, lig), &domain(rep), hook, Live::Zk { level0, rounds }, Some(salts),
                                                 Reuse::None, native_sha));
    let bk = gpu_buckets(wt, ph, wph, hph, false, native_sha);
    eprintln!("GPUPHASE\t{bk}");
    (proof, commitment, bk)
}

#[cfg(not(feature = "gpu"))]
#[allow(clippy::too_many_arguments)]
fn prove_gpu(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger,
             _: &Mutex<flock_live::zk_hooks::LeafSalts>, _: Option<u64>, _: bool) -> (Vec<u8>, Value) {
    panic!("built without the gpu feature")
}

#[cfg(not(feature = "gpu"))]
#[allow(clippy::too_many_arguments)]
fn gpu_zk_device(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &Mutex<flock_live::zk_hooks::LeafSalts>,
                 _: &flock_core::pcs::ligerito::zk::Level0Zk, _: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, _: bool)
                 -> (R1csProofLigerito, Commitment, Value) {
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
    /// M1: mask the sent values with pads other than the committed ones
    zk_pad_mismatch: bool,
    /// M1's control: zero pads and a zero mask slot (M0's values, in M1's format)
    zk_zero_pads: bool,
    /// M1: Ligerito level 0's hiding tampered (`Level0Zk::tamper`)
    zk_l0_tamper: u8,
    /// both reps fold, sum and pad rep 0's pair of extra lanes (red-team-m1-zk F1's shared lanes; the verifier rejects rep 1)
    zk_shared_pair: bool,
    /// prove a dummy witness: all zero but the public regions (M2's extraction run; M1's simulator takes its opening)
    zero_witness: bool,
    /// the coin-tree nonce `Hello` carries, instead of the prover's own fresh one: the simulator's, drawn once per simulated
    /// session, so every run restarts after the same `Hello`
    coin_nonce: Option<[u8; 32]>,
    /// the device witness's units from the host (`Some(true)`), the device (`Some(false)`), or by depth (`deep_unit`)
    host_units: Option<bool>,
    /// rep 1 from rep 0's device witness and level-0 commitment (default: unless `FC_REP_REUSE=0`; never under `--zk`)
    rep_reuse: Option<bool>,
    /// the compression slots' witness natively (`Some(true)`) or level by level (`Some(false)`); default native unless
    /// `FC_NATIVE_SHA=0`
    native_sha: Option<bool>,
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
    // M1: the mask slot's words are uniform (they hide the non-region claims' ring-switch s_hat_v)
    let random = plan.mask_random || (is_zk(st) && !plan.zk_zero_pads);
    let hooks = Arc::new(Mutex::new(NoMask::new(seed)));
    let mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b| {
        let mut h = hooks.lock().unwrap_or_else(PoisonError::into_inner);
        if random { h.rng.f128s(flock_live::zk_hooks::STREAM_MASK, (b * mask_words) as u64, mask_words) } else { h.mask_fill(b, mask_words) }
    });
    let ur = st.c.unit_range();
    let unit = st.c.unit();
    let cut_flip = (plan.cut_flip && unit.in_groups.len() > 1).then(|| (st.c.unit_net, 0, unit.in_groups[1].col * 128 + 3));
    let nets = if plan.tail_swapped { forged_stages(st) } else { BTreeMap::new() };
    Wit { rowin, inputs, mask, flip_z: plan.unit_flip.then_some(ur.slot(ur.count.min(6) - 1) + st.c.unit().in_words * 128 + 40), cut_flip,
          nets: Arc::new(nets), regions: (zero && plan.zero_witness).then(|| inst.clone()), mask_host: random }
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
    // M2: every coin the verifier sends is checked against its step-0 commitment before the prover uses it
    let t: SharedTransport = if is_zk(st) { shared(flock_live::coin_tree::Checked::new(SharedCall(t), coin_spec())) } else { t };
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
    let hello = if is_zk(st) {
        config(st, inst).hello_with_nonce(&plan.coin_nonce.unwrap_or_else(|| flock_live::zk_hooks::coin_nonce(&seed)))
    } else {
        config(st, inst).hello()
    };
    let mut commit: Vec<F128> = Vec::new();
    let device = plan.gpu && !plan.mask_random && std::env::var("FC_HOST_WITNESS").is_err();
    #[cfg(feature = "gpu")]
    let host_units = device && plan.host_units.unwrap_or_else(|| flock_live::gpu_circuit::deep_unit(st));
    #[cfg(not(feature = "gpu"))]
    let host_units = false;
    // on the device, rep 1 proves from rep 0's witness and level-0 commitment (the session's tag keys what rep 0 keeps); not
    // under M1, whose level 0 and lane phase are its own
    static SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let native_sha = plan.native_sha.unwrap_or_else(|| std::env::var("FC_NATIVE_SHA").as_deref() != Ok("0"));
    let reuse_tag = (plan.gpu && !plan.rep1_other_witness && !is_zk(st)
        && plan.rep_reuse.unwrap_or_else(|| std::env::var("FC_REP_REUSE").as_deref() != Ok("0")))
        .then(|| SESSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let lig = if plan.fast { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
    // hm96-sha512 leaf salts: one ChaCha20 key per proof; the level-0 tree is id 0 (both reps bind one root), every later
    // tree a fresh id. The device expands its own trees' salts; a CPU proof takes host expansions.
    let salts = Arc::new(Mutex::new(flock_live::zk_hooks::LeafSalts::for_seed(&seed)));
    let zk_points = Arc::new(Mutex::new(Vec::new()));
    // M1 on the device: level 0's hiding and both reps' pads commitments beside the witness build, and rep 0's replay and inner
    // proof beside rep 1's device run (FC_ZK_SEQUENTIAL: neither, the reference for the overlap's gain)
    let overlap = std::env::var_os("FC_ZK_SEQUENTIAL").is_none();
    let zk_pre = (plan.gpu && is_zk(st) && overlap).then(|| {
        let (st, plan, salts, points) = (st.clone(), plan.clone(), salts.clone(), zk_points.clone());
        std::thread::spawn(move || zk_prepare(&st, &plan, seed, lig, &salts, &points))
    });
    let w0 = wit(st, inst, plan, plan.zero_witness, seed);
    let (wt0, tw0) = witness(st, &w0, device, host_units);
    // rep 1 proves from rep 0's witness unless it has its own: the CPU prover consumes a witness, so rep 0 takes a copy and rep 1
    // the original (two copies at most, never one per rep plus the original)
    let (mut wt0, (mut wt1, tw1)) = (Some(wt0), if plan.rep1_other_witness {
        let (w, t) = witness(st, &wit(st, inst, plan, true, seed), device, host_units);
        (Some(w), t)
    } else {
        (None, 0.0)
    });
    let witness_s = tw0 + tw1;
    let tj = Instant::now();
    let mut zk_pre = match zk_pre.map(|h| h.join()) {
        Some(Ok((_, reps))) => Some(reps.map(Some)),
        Some(Err(e)) => panic!("M1's randomness: {}", panic_msg(e)),
        None => None,
    };
    // the preparation's time past the witness build (what it adds to the session)
    let t_zk_pre_wait = tj.elapsed().as_secs_f64();
    let mut prove_s = Vec::new();
    let mut buckets = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<Vec<F128>>> = None;
    let mut root0: Option<Vec<u8>> = None;
    let mut stop_why: Option<String> = None;
    let tp0 = Instant::now();
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello.clone()))? {
            // M0's seed commitment (4 words), or the keyed coin tree's root and session key (20; `coin_tree::Checked` checked them)
            Resp::Coins(c) if c.len() == if is_zk(st) { 20 } else { 4 } => commit = c,
            r => return Err(format!("hello: step 0 expects the verifier's seed commitment, got {r:?}")),
        }
        #[cfg(feature = "sha512")]
        let level0 = (!plan.gpu).then(|| Arc::new(salts.lock().unwrap_or_else(PoisonError::into_inner).expand(0, params(st.m, lig).n_leaves())));
        let mut prng = flock_live::zk_hooks::ProverRng::from_seed(seed);
        let tl0 = Instant::now();
        let zk_level0 = (is_zk(st) && zk_pre.is_none()).then(|| level0_zk(st, &mut prng, plan.zk_zero_pads, plan.zk_l0_tamper));
        let t_level0 = if zk_pre.is_some() { t_zk_pre_wait } else { tl0.elapsed().as_secs_f64() };
        // M1 on the device: rep 0's replay and inner proof run beside rep 1's device run; rep 1's stream opens (its first round)
        // after rep 0's last round, so the wire carries the sequential prover's messages in its order
        let mut rep0_finish: Option<std::thread::JoinHandle<(Vec<u8>, Value)>> = None;
        let mut rep0_done: Option<(Vec<u8>, Value)> = None;
        for rep in 0..2 {
            // M1: this rep's pads and their commitment (a fresh salt tree id), drawn before anything of the rep is sent (on the
            // device path, beside the witness build: rep 1's tree id is consumed here, where the sequential prover takes it)
            let tz0 = Instant::now();
            let (zkrep, pre_id) = match zk_pre.as_mut() {
                Some(z) => { let (r, id) = z[rep].take().expect("a rep's pads"); (Some(r), Some(id)) }
                None => (is_zk(st).then(|| {
                    let id = salts.lock().unwrap_or_else(PoisonError::into_inner).next_id();
                    make_zkrep(st, plan, &mut prng, zk_level0.as_ref().unwrap(), &zk_points, rep, id, &salts)
                }), None),
            };
            if let (Some(id), 1) = (pre_id, rep) {
                let got = salts.lock().unwrap_or_else(PoisonError::into_inner).next_id();
                assert_eq!(got, id, "rep 1's pads tree takes the salt id after rep 0's proof's trees");
            }
            let t_zk_commit = if pre_id.is_some() { if rep == 0 { t_level0 } else { 0.0 } }
                              else { tz0.elapsed().as_secs_f64() + if rep == 0 { t_level0 } else { 0.0 } };
            // rep 1's stream on the device path opens at its first round, after rep 0's last one (the order on the wire)
            let mut lch = if rep0_finish.is_some() { None } else { Some(LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?) };
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
                    _ => wt1.take().or_else(|| wt0.take()).or_else(|| Some(witness(st, &w0, device, host_units).0)),
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
            let mut zkrep = zkrep;
            let proved = catch_unwind(AssertUnwindSafe(|| {
                if plan.gpu {
                    let wt = if rep == 0 { wt0.as_ref() } else { wt1.as_ref().or(wt0.as_ref()) }.expect("a rep's witness");
                    match zkrep.take() {
                        // M1 on the device, rep 0: the device part, then its replay and inner proof on a thread
                        Some(z) if rep == 0 && overlap => {
                            let (mut mch, sent) = zk_masking(st, lch.take().expect("rep 0's stream"), &z);
                            let tp = Instant::now();
                            let (proof, cm, bk) = {
                                let mut rounds = |c: &[u8], n: usize| replay_round(&mut mch, c, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"));
                                gpu_zk_device(st, wt, lig, rep, &mut hook, &salts, &z.level0, &mut rounds, native_sha)
                            };
                            let (t_up, st2) = (tp.elapsed().as_secs_f64(), st.clone());
                            rep0_finish = Some(std::thread::spawn(move || zk_finish(&st2, &z, proof, cm, mch, &sent, tp, t_up, bk)));
                            (Vec::new(), Value::Null)
                        }
                        // rep 1 beside rep 0's finish: the device starts now, its first round waits for rep 0's last one and then
                        // opens rep 1's stream; otherwise (sequential) the rep's stream is open already
                        Some(z) => {
                            let mut first = rep0_finish.take();
                            let mut opened: Option<(zkv::Masking<LiveChallenger>, zkv::Pads)> = lch.take().map(|l| zk_masking(st, l, &z));
                            let mut r0: Option<Result<(Vec<u8>, Value), String>> = None;
                            let tp = Instant::now();
                            let dev = catch_unwind(AssertUnwindSafe(|| {
                                let mut rounds = |c: &[u8], n: usize| -> Vec<F128> {
                                    if opened.is_none() {
                                        let done = first.take().expect("rep 0's replay and inner proof").join().map_err(panic_msg);
                                        let stop = done.as_ref().err().cloned();
                                        r0 = Some(done);
                                        if let Some(e) = stop {
                                            panic!("{e}");
                                        }
                                        let l = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep)).unwrap_or_else(|e| panic!("LIVE-REFUSED open: {e}"));
                                        opened = Some(zk_masking(st, l, &z));
                                    }
                                    let m = &mut opened.as_mut().expect("rep 1's stream").0;
                                    replay_round(m, c, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"))
                                };
                                gpu_zk_device(st, wt, lig, rep, &mut hook, &salts, &z.level0, &mut rounds, native_sha)
                            }));
                            if let Some(h) = first.take() {
                                r0 = Some(h.join().map_err(panic_msg));
                            }
                            // rep 0's stop comes first on the wire, so it is the session's
                            if let Some(r) = r0 {
                                rep0_done = Some(r.unwrap_or_else(|e| panic!("{e}")));
                            }
                            let (proof, cm, bk) = dev.unwrap_or_else(|e| std::panic::resume_unwind(e));
                            let t_up = tp.elapsed().as_secs_f64();
                            let (mch, sent) = opened.take().expect("rep 1's stream opened");
                            zk_finish(st, &z, proof, cm, mch, &sent, tp, t_up, bk)
                        }
                        None => prove_gpu(st, wt, lig, rep, &mut hook, lch.as_mut().expect("the rep's stream"), &salts, reuse_tag, native_sha),
                    }
                } else if let Some(z) = &zkrep {
                    prove_cpu_zk(st, &own(&mut wt0, &mut wt1), lig, rep, &mut hook, lch.as_mut().expect("the rep's stream"), z)
                } else {
                    prove_cpu(st, own(&mut wt0, &mut wt1), lig, &mut hook, lch.as_mut().expect("the rep's stream"))
                }
            }));
            #[cfg(feature = "sha512")]
            if level0.is_some() {
                flock_merkle::hm96::end();
            }
            let (p, bk) = proved.map_err(|e| stop_why.clone().unwrap_or_else(|| panic_msg(e)))?;
            if rep0_finish.is_some() {
                // rep 0 on the device path: its proof comes with rep 1's
                continue;
            }
            if let Some((p0, bk0)) = rep0_done.take() {
                prove_s.push(bk0["t.prove"].as_f64().unwrap_or(0.0) + t_level0);
                buckets.push(bk0);
                proofs.push(p0);
            }
            prove_s.push(tp.elapsed().as_secs_f64() + t_zk_commit);
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
    // M0's seed opens at the end; under M2 every coin was checked against the commitment before it was used
    let step0 = if v["accepted"] == true && !is_zk(st) { check_coin_seed(&commit, &hello, &v, &log.lock().unwrap_or_else(PoisonError::into_inner)) } else { Ok(()) };
    let v = if let Err(e) = &step0 { json!({"accepted": false, "why": e, "verdict": v}) } else { v };
    let c = &st.c;
    let ands = (c.unit().useful as u64, st.n as u64 * c.upv as u64);
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": STATEMENT, "scheme": SCHEME,
               "prover": if plan.gpu {"flock-cuda-circuit"} else {"flock-cpu"}, "gpu": plan.gpu, "unit": c.name, "set": inst.header["set"],
               "subcircuit": inst.header["subcircuit"], "instances": st.n, "vus_per_block": c.g, "units": ands.1, "k_log": c.k_log, "nbl": st.nbl,
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
    ("k_log_outside_the_proved_range_refused_at_load", false, false),
    ("verifier_evaluated_tail_refused", false, false),
    ("leaf_salts_from_prover_seed_refused", false, false),
    ("device_salts_match_host", true, false),
    ("leaf_key_witness_refused", false, false),
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
    ("gpu_proofs_match_cpu", true, false),
    ("gpu_paths_agree", true, false),
    ("host_units_match_witness", true, false),
    ("native_sha_matches_eval64", true, false),
    ("region_word_shared_refused", false, false),
    ("zk_pads_not_the_committed_ones", false, true),
    ("zk_level0_folds_another_extra_lane", false, true),
    ("zk_level0_blinded_claim_forged", false, true),
    ("zk_level0_ybar_forged", false, true),
    ("zk_level0_extra_lane_sum_forged", false, true),
    ("zk_verifier_coin_off_its_commitment", false, false),
    ("zk_verifier_commits_beta_zero", false, false),
    ("zk_session_relations_vary_with_the_seed", true, false),
    ("zk_dummy_witness_passes_the_region_claims", false, true),
];

/// Cases that exercise the statement's parse or the non-ZK proof format only (M1's `--zk` selftest skips them).
const NOT_ZK: &[&str] = &["coin_seed_opening_forged", "coin_seed_record_replays", "relabelled_circuit_refused_at_load", "verifier_evaluated_tail_refused", "leaf_salts_from_prover_seed_refused",
                          "device_salts_match_host", "leaf_key_witness_refused", "tail_swapped_circuit_refused", "mask_cells_random",
                          "opened_salt_altered"];

fn skipped(st: &Stmt, name: &str) -> bool {
    let multi_block = st.c.ports.iter().any(|p| p.comps > 1);
    let cut = st.c.wires.iter().any(|w| w.dst.0 != st.c.sha_net && w.dst.0 != st.c.hm_net);
    // a tail with no positive constant to forge (GEMM's: the zero accumulator and the cast) forges to itself: nothing to swap
    let forged_same = std::fs::read_to_string(arg("--forged", "circuit-forged.txt"))
        .map(|f| f == std::fs::read_to_string(arg("--circuit", "")).unwrap_or_default()).unwrap_or(true);
    (name == "chaining_value_forged" && !multi_block)
        || (!cut && ["private_cut_word_tampered", "tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
        || (is_zk(st) && NOT_ZK.contains(&name))
        || (!is_zk(st) && name.starts_with("zk_"))
        || (forged_same && ["tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
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
        "zk_pads_not_the_committed_ones" => Plan { zk_pad_mismatch: true, ..h },
        "zk_level0_folds_another_extra_lane" => Plan { zk_l0_tamper: 1, ..h },
        "zk_level0_blinded_claim_forged" => Plan { zk_l0_tamper: 2, ..h },
        "zk_level0_ybar_forged" => Plan { zk_l0_tamper: 3, ..h },
        "zk_level0_extra_lane_sum_forged" => Plan { zk_l0_tamper: 4, ..h },
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
    if name == "k_log_outside_the_proved_range_refused_at_load" {
        // the same circuit with META's k_log one past Stmt.InRange: the ranges still fit the larger block, so only the range
        // check can refuse it
        let text = std::fs::read_to_string(arg("--circuit", "")).unwrap();
        let mut lines: Vec<&str> = text.splitn(3, '\n').collect();
        let meta = lines[1].replacen(&format!("\"k_log\":{}", st.c.k_log), &format!("\"k_log\":{}", IN_RANGE_K_LOG + 1), 1);
        lines[1] = &meta;
        let why = Composite::parse(&lines.join("\n"), Some(&arg("--tables", "tables"))).err();
        return json!({"case": name, "expect": "refused at load: k_log past Stmt.InRange", "why": why,
                      "pass": meta != text.splitn(3, '\n').nth(1).unwrap_or("") && why.as_deref().is_some_and(|e| e.contains("Stmt.InRange"))});
    }
    if name == "prover_is_deterministic" {
        return determinism_case(st, inst, gpu);
    }
    if name == "gpu_proofs_match_cpu" {
        return gpu_cpu_case(st, inst);
    }
    if name == "gpu_paths_agree" {
        return gpu_paths_case(st, inst);
    }
    if name == "host_units_match_witness" {
        // the device witness's host units (a deep unit's slots, Stmt::slot_zab from their inputs) are the CPU witness's unit
        // slots word for word
        let w = wit(st, inst, &Plan::default(), false, ProverSeed::from_os());
        let ins = st.converge(&*w.inputs, &*w.rowin, &w.nets);
        let (z, a, b, _) = st.witness(&ins, &*w.mask, &w.nets);
        let (ur, wpb) = (st.c.unit_range(), 1usize << (st.c.k_log - 7));
        let slots = st.slot_zab(st.c.unit_net, st.c.unit(), &ins[st.c.unit_net]);
        let same = slots.iter().all(|(x, zs, xa, xb)| {
            let at = (x / ur.count) * wpb + ur.slot(x % ur.count) / 128;
            z[at..at + zs.len()] == zs[..] && a[at..at + xa.len()] == xa[..] && b[at..at + xb.len()] == xb[..]
        });
        return json!({"case": name, "expect": "host-evaluated unit slots equal the witness's", "slots": slots.len(),
                      "pass": same && slots.len() == st.blocks() * ur.count});
    }
    if name == "native_sha_matches_eval64" {
        // the compression and hm96 slots' native witness (sha512_native: SHA-512 in 64-bit words, each row read off it by a plan
        // derived from the pinned slot circuit) is Stmt::slot_zab's (eval64 of the rows, then the lane transpose), word for word,
        // on every slot of every block, padding instances included
        use flock_live::sha512_native::NativeSha;
        use rayon::prelude::*;
        let w = wit(st, inst, &Plan::default(), false, ProverSeed::from_os());
        let (mut si, mut hi) = (Vec::new(), Vec::new());
        for b in 0..st.blocks() {
            let (s, h) = (w.rowin)(b);
            si.extend(s);
            hi.extend(h);
        }
        let mut nets = serde_json::Map::new();
        let mut pass = true;
        for (ni, ins) in [(st.c.sha_net, &si), (st.c.hm_net, &hi)] {
            let net = &st.c.nets[ni].1;
            let v = match NativeSha::derive(net) {
                Err(e) => json!({"pass": false, "why": e}),
                Ok(plan) => {
                    let host = st.slot_zab(ni, net, ins);
                    let same = host.len() == ins.len()
                        && host.par_iter().all(|(x, z, a, b)| plan.slot_zab(&ins[*x]) == (z.clone(), a.clone(), b.clone()));
                    json!({"pass": same, "slots": host.len(), "rows": plan.rows.len(), "window": plan.window(), "census": plan.census(), "circuit_sha256": hex(&net.sha256)})
                }
            };
            pass &= v["pass"] == json!(true);
            nets.insert(st.c.nets[ni].0.clone(), v);
        }
        return json!({"case": name, "expect": "the native SHA-512 witness equals eval64's on every compression and hm96 slot",
                      "nets": nets, "pass": pass});
    }
    if name == "region_word_shared_refused" {
        // docs/region-word-check.md's negatives on the loaded circuit: a column sharing a partial region's word is refused with an
        // entry in its slot type's A row, or with a Δ wire into it; a Δ pair that cancels leaves it forced to zero
        let Some(&(_, q)) = region_shared_columns(&st.regions).first() else {
            return json!({"case": name, "pass": true, "skipped": "no partial region: every region covers whole words"});
        };
        let mut c = st.c.clone();
        let r = c.ranges.iter().find(|r| q >= r.start() && q < r.start() + (r.count << r.slot_log)).cloned().expect("a slot holds the column");
        let Kind::Net(i) = r.kind else {
            return json!({"case": name, "pass": false, "why": format!("column {q} is in the mask slot")});
        };
        let row = q & ((1usize << r.slot_log) - 1);
        let a = &mut c.nets[i].1.a;
        if a.len() <= row {
            a.resize(row + 1, Vec::new());
        }
        a[row].push(row);
        let entry = Stmt::new(c, st.n, st.identity.clone()).err();
        let (q32, pin) = (q as u32, st.pin as u32);
        let with = |extra: &[(u32, u32)]| {
            let (mut da, mut db) = (st.circuit.da.clone(), st.circuit.db.clone());
            da.extend_from_slice(extra);
            db.extend_from_slice(extra);
            region_word_violation(&st.c, &st.regions, &da, &db)
        };
        let (wire, cancels) = (with(&[(q32, q32), (q32, pin)]), with(&[(q32, pin), (q32, pin)]));
        let pinned = region_word_violation(&st.c, &st.regions, &st.circuit.da, &st.circuit.db);
        let refused = |e: &Option<String>| e.as_deref().is_some_and(|m| m.contains(&format!("column {q} shares a word")));
        return json!({"case": name, "expect": "refused with an A entry or a Δ wire; a cancelling Δ pair and the pinned circuit pass", "column": q,
                      "entry_refused": entry, "wire_refused": wire, "cancelling_pair_passes": cancels.is_none(), "pinned_passes": pinned.is_none(),
                      "pass": refused(&entry) && refused(&wire) && cancels.is_none() && pinned.is_none()});
    }
    #[cfg(feature = "seed-injection")]
    if name == "zk_verifier_coin_off_its_commitment" || name == "zk_verifier_commits_beta_zero" {
        // M2: V* answers rep 0's round 3 with a coin of its choosing under the committed coin's opening, or commits to β = 0;
        // the prover stops at that coin, before it sends anything that depends on it
        let (strategy, want) = if name.ends_with("beta_zero") {
            let v = VStar::new(st, inst, Strategy::Honest, [7; 32]);
            let view = v.view.clone();
            session(st, inst, shared(v), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
            let k = beta_round(&view.lock().unwrap_or_else(PoisonError::into_inner)).expect("rep 0's beta round");
            (Strategy::ZeroBeta(k), "degenerate coin β = 0")
        } else {
            (Strategy::CoinAt(3), "do not open its step-0 commitment")
        };
        let v = VStar::new(st, inst, strategy, [7; 32]);
        let view = v.view.clone();
        let (r, _) = session(st, inst, shared(v), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let (s, _) = view_summary("real", name, 0, &view.lock().unwrap_or_else(PoisonError::into_inner));
        let stopped = r["prover_stopped"].as_str().is_some_and(|x| x.contains(want));
        return json!({"case": name, "gpu": gpu, "expect": "the prover stops at the coin", "accepted": r["accepted"], "pass": stopped && r["accepted"] != true,
                      "prover_stopped": r["prover_stopped"], "last_round": s["last_round"], "end": s["end"]});
    }
    if name == "zk_dummy_witness_passes_the_region_claims" {
        // red-team-m1-zk F4: M2's extraction run is live, so the dummy witness carries the public region words; its region
        // claims' s_hat_v, sent before any coin opens, are the public ones, and only the inner proof rejects it
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        let (v, _) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, zero_witness: true, ..Plan::default() }, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default();
        let inner_only = reps.len() == 2 && reps.iter().all(|r| r["ok"] == false && r["why"].as_str().is_some_and(|w| w.starts_with("zk inner")));
        return json!({"case": name, "gpu": gpu, "expect": "rejected by the inner proof alone (the region claims pass)", "accepted": v["accepted"],
                      "pass": v["accepted"] != true && inner_only, "reps_ok": reps.iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect::<Vec<_>>()});
    }
    #[cfg(feature = "seed-injection")]
    if name == "zk_session_relations_vary_with_the_seed" {
        // red-team-m1-zk F1: under one verifier coin seed, the two cross-rep functionals are the same for any two prover seeds
        // when the reps share level 0's extra lanes and their padding (a prover folding rep 0's pair in both reps, which the
        // verifier rejects); with a pair per rep they differ in every word
        let rel = |shared_pair: bool| -> Result<[Vec<F128>; 2], String> {
            let s0 = Arc::new(Mutex::new(server(st, inst)));
            fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
            let plan = Plan { gpu, zk_shared_pair: shared_pair, ..Plan::default() };
            let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &plan, ProverSeed::from_os());
            if v["accepted"] != !shared_pair {
                return Err(format!("the session's verdict: {}", v["why"]));
            }
            let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
            let coins: Vec<Vec<F128>> = (0..2).map(|r| record_coins(&rec, &rep_stream(TABLE, r))).collect();
            session_relations(st, inst, &proofs, &coins, &record_points(&rec), if shared_pair { [0, 0] } else { [0, 1] })
        };
        let equal = |shared_pair: bool| -> Result<(Vec<usize>, Vec<usize>), String> {
            let (a, b) = (rel(shared_pair)?, rel(shared_pair)?);
            Ok(((0..2).map(|k| a[k].iter().zip(&b[k]).filter(|(x, y)| x == y).count()).collect(), a.iter().map(|x| x.len()).collect()))
        };
        return match (equal(false), equal(true)) {
            (Ok((eq, len)), Ok((eq_shared, len_shared))) => {
                let pass = (0..2).all(|k| len[k] > 0 && eq[k] == 0 && eq_shared[k] == len_shared[k]);
                json!({"case": name, "gpu": gpu, "expect": "the functionals differ in every word across two prover seeds, and are equal if the reps share a pair",
                       "pass": pass, "f256_values": [len[0] / 2, len[1] / 2], "f128_words_equal": eq, "shared_pair_words_equal": eq_shared,
                       "shared_pair_words": len_shared})
            }
            (Err(e), _) | (_, Err(e)) => json!({"case": name, "gpu": gpu, "pass": false, "why": e}),
        };
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
    record_case(name, want.1, &v, &rec, &proofs);
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
fn record_case(name: &str, expect_accept: bool, v: &Value, rec: &Value, proofs: &[Vec<u8>]) {
    let dir = arg("--record-dir", "");
    if dir.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all(&dir);
    let doc = json!({"case": name, "statement": STATEMENT, "expect": if expect_accept {"accept"} else {"reject"}, "accepted": v["accepted"],
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
    let at = lines.iter().position(|l| l.starts_with("CIRCUIT unit ")).unwrap();
    let n: usize = lines[at].split_whitespace().nth(2).unwrap().parse().unwrap();
    let cut = r#"CUT {"in":[],"out":[],"tail":[[2,"F32Add_v1",0,1]],"words":2}"#;
    let mut a: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    a[at] = format!("CIRCUIT unit {}", n + 1);
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
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
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

/// The first place two JSON values differ (a path, and the two values there).
fn first_diff(a: &Value, b: &Value, path: String) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            x.keys().chain(y.keys()).find_map(|k| first_diff(x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null), format!("{path}.{k}")))
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x.iter().zip(y).enumerate().find_map(|(i, (u, v))| first_diff(u, v, format!("{path}[{i}]"))),
        _ if a == b => None,
        _ => {
            let (u, v) = (a.to_string(), b.to_string());
            let at = u.bytes().zip(v.bytes()).position(|(p, q)| p != q).unwrap_or(u.len().min(v.len()));
            Some(format!("{path} (lengths {} / {}, first differing char {at}: {:?} / {:?})", u.len(), v.len(),
                         &u[at.saturating_sub(40)..(at + 40).min(u.len())], &v[at.saturating_sub(40)..(at + 40).min(v.len())]))
        }
    }
}

/// The device prover against the CPU prover (M0 or M1, the statement's mode): equal seeds and verifier coins give byte-identical
/// proofs and transcripts, so the CPU prover's statistics, audit and exact checks cover the device's proofs.
#[cfg(feature = "seed-injection")]
fn gpu_cpu_case(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Value {
    let run = |gpu: bool| {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::injected([1; 32]));
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        (v["accepted"] == true, proofs, json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]}))
    };
    let (c, g) = (run(false), run(true));
    let proof_diff: Vec<Value> = c.1.iter().zip(&g.1).map(|(x, y)| json!({"lengths": [x.len(), y.len()], "first_differing_byte": x.iter().zip(y).position(|(p, q)| p != q)}))
        .collect();
    let (pe, te) = (c.1 == g.1, c.2 == g.2);
    json!({"case": "gpu_proofs_match_cpu", "expect": "the device's proofs and transcripts are the CPU prover's, byte for byte", "gpu": true, "zk": is_zk(st),
           "accepted": c.0 && g.0, "proofs_equal": pe, "transcripts_equal": te, "proofs": proof_diff,
           "transcript_first_difference": first_diff(&c.2, &g.2, String::new()), "pass": c.0 && g.0 && pe && te})
}

#[cfg(not(feature = "seed-injection"))]
fn gpu_cpu_case(_: &Arc<Stmt>, _: &Arc<Instances>) -> Value {
    json!({"case": "gpu_proofs_match_cpu", "pass": true, "skipped": "built without the test harness's seed-injection feature"})
}

/// The device prover's paths against each other: the device's units, a rebuilt rep 1 and the compressions level by level, against
/// the host's units, rep 1 from rep 0's witness and level-0 commitment, and the compressions natively (host units only where the
/// device witness takes them, a statement whose units return its outputs; reuse never under M1). Equal seeds and verifier coins
/// give byte-identical proofs and transcripts.
#[cfg(feature = "seed-injection")]
fn gpu_paths_case(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Value {
    let run = |plan: Plan| {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &plan, ProverSeed::injected([1; 32]));
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        (v, proofs, json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]}))
    };
    let a = run(Plan { gpu: true, host_units: Some(false), rep_reuse: Some(false), native_sha: Some(false), ..Plan::default() });
    let b = run(Plan { gpu: true, host_units: Some(true), rep_reuse: Some(true), native_sha: Some(true), ..Plan::default() });
    let used = |k: &str| b.0["buckets"].as_array().map(|x| x.iter().map(|r| r[k].clone()).collect::<Vec<_>>());
    let accepted = a.0["accepted"] == true && b.0["accepted"] == true;
    let (pe, te) = (a.1 == b.1, a.2 == b.2);
    json!({"case": "gpu_paths_agree", "expect": "byte-identical proofs and transcripts", "gpu": true, "zk": is_zk(st), "accepted": accepted,
           "proofs_equal": pe, "transcripts_equal": te, "transcript_first_difference": first_diff(&a.2, &b.2, String::new()),
           "host_units": used("host_units"), "rep_reused": used("rep_reused"), "native_sha": used("native_sha"), "prove_s": [a.0["prove_s"].clone(), b.0["prove_s"].clone()],
           "pass": accepted && pe && te})
}

#[cfg(not(feature = "seed-injection"))]
fn gpu_paths_case(_: &Arc<Stmt>, _: &Arc<Instances>) -> Value {
    json!({"case": "gpu_paths_agree", "pass": true, "skipped": "built without the test harness's seed-injection feature"})
}

/// The same circuit with unit 0's first two leaf inputs swapped in its META: the rows are unchanged, the meaning is not.
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
        if skipped(st, name) || (name == "row_ref_claim_false" && !inst.tables.as_ref().is_some_and(|t| t.counts.iter().any(|&r| r > 1)))
            || ((name == "gpu_proofs_match_cpu" || name == "gpu_paths_agree") && !gpu) {
            continue;
        }
        let v = if gpu {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("selftest")
                .args(if zk_mode() { &["--zk"][..] } else { &[] })
                .args(["--gpu", "--instances", &arg("--instances", ""), "--circuit", &arg("--circuit", ""), "--tables", &arg("--tables", "tables"),
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
    println!("SELFTEST\t{}", json!({"statement": STATEMENT, "unit": st.c.name, "subcircuit": inst.header["subcircuit"], "instances": st.n,
        "vus_per_block": st.c.g, "k_log": st.c.k_log, "m": st.m, "gpu": gpu, "circuit_sha512": hex(&st.c.sha512), "all_pass": all}));
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
    if zk_mode() {
        zk_queries0(st.m).unwrap_or_else(|e| refuse(e));
    }
    eprintln!("statement {} instances={} vus_per_block={} blocks={} k_log={} nbl={} m={} regions={} delta={} built in {:.2}s",
              st.c.name, st.n, st.c.g, st.blocks_real, st.c.k_log, st.nbl, st.m, st.regions.len(), st.circuit.da.len() / 2, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "zkstat" => zkstat(&st, &inst),
        "zkaudit" => zkaudit(&st, &inst),
        "zkrewind" => zkrewind(&st, &inst),
        "zkgk" => zkgk(&st, &inst),
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
