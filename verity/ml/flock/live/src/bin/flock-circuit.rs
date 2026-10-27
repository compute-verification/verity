//! flock-circuit: `verity/flock-circuit` (`flock_live::circuit`), proved by the CPU prover or by Flock-CUDA (`--gpu`,
//! feature `gpu`) in a flock-128-r2 live session: one commitment root, bound by both reps (R1).
//!
//!   flock-circuit serve    --listen ADDR --out DIR --instances PUB --circuit FILE --pin SHA256 [--sessions N] [--operator LANE] [--session-tables J]
//!   flock-circuit prove    --verifier ADDR --instances INST --circuit FILE [--gpu] [--runs N] [--warm W] [--dump DIR] [--session-tables J]
//!   flock-circuit selftest --instances INST --circuit FILE [--gpu] [--only CASE]
//!   flock-circuit statement --instances PUB --circuit FILE    (the statement the verifier builds, its digest; nothing proved)
//!   flock-circuit stacked  --table R.bin (--root-out ROOT.json | --points P.json --statement-digest HEX --verifier ADDR [--dump F])   (`stacked`)
//!
//! The verifier loads only its own public file (`verity_flock.circuit.stage(..., public_only=True)`: roots and each row's
//! `b || c`, the output rows' too; no row, no salt, no output word) and the pinned circuit, from which it derives Δ, the regions and their values; it checks the
//! frame-v3-sha512 roots and the pinned padding values before any session. Session: `Hello(Σ)`, `Commit(root_f = Σ, root)`, two verifier
//! points (the regions' claims), `Link(y = 0, 0)` (nothing is prover-supplied), Flock's coins.
//!
//! `--session-tables J` (both sides, CPU prover): the session's units in J equal tables (`session_tables`), each with its own
//! randomness, all J roots in one `Commit`, and every proof after the session's last coin (the barrier).

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
};
use flock_field::F128;
use flock_live::{
    circuit::*,
    firewall::{Framer, Gate},
    session_verify as sverify,
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
/// The most tables in one session (`--session-tables`); their names sort in table order (`circuit-00` to `circuit-63`).
const MAX_TABLES: usize = 64;
/// The Lean verifier's `Draw.TABLES_LAW`: several tables split one draw of fixed count, a subset or a stratified one.
const TABLES_LAW: &str = "tables: several tables need a subset or stratified draw, whose unit count is fixed (a Bernoulli \
                          draw's count is random, so the tables could not split it evenly, and a work draw's closure is not split)";

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

/// A rep stream's transcript domain in a session of `tables` tables: [`domain`] for one; for several, the table's name and a
/// `/` before the domain tag's final `rep` (`flock-circuit/fast100x2/circuit-00/rep0`).
fn domain_of(c: &Composite, tables: usize, table: &str, rep: usize) -> Vec<u8> {
    if tables <= 1 {
        return domain(c, rep);
    }
    let base = c.tags().domain.strip_suffix("rep").expect("a statement's domain tag ends in `rep`");
    format!("{base}{table}/rep{rep}").into_bytes()
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

/// A statement's Merkle hash: [`MERKLE`], or plain SHA-512 leaves for a plain-leaf statement (`circuit::leaf_scheme_plain`).
fn merkle(c: &Composite) -> HashKind {
    if c.plain_leaves() { HashKind::Sha512 } else { MERKLE }
}

/// The statement's trees take hm96 salts.
fn salted(st: &Stmt) -> bool {
    merkle(&st.c) == MERKLE && cfg!(feature = "sha512")
}

fn pcs_of(st: &Stmt, lig: LigeritoProfile) -> PcsParams {
    PcsParams { merkle_hash: merkle(&st.c), ..params(st.m, lig) }
}

/// `--zk`: M1's masked protocol (`flock_live::zk_veil`), pinned in the identity (both sides must agree).
fn zk_mode() -> bool {
    flag("--zk")
}

fn is_zk(st: &Stmt) -> bool {
    st.identity["hooks"]["zk"] == true
}

/// A non-ZK session's coins: M0's step-0 seed by default (`coin_seed`: the verifier's OS-random seed under a SHA-512
/// commitment sent at Hello and opened in the verdict, every round's coins its SHA-256 counter-mode PRF), or under
/// `FC_COINS=os` every round's coins drawn from the OS, so soundness rests on SHA-512 collision resistance alone. The Hello and
/// the identity carry the mode, so a prover and a server that disagree refuse the session. A selftest build always seeds: its
/// harness fixes the coins through the seed.
fn seed_coins() -> bool {
    match std::env::var("FC_COINS").as_deref() {
        Err(_) | Ok("" | "seed") => true,
        Ok("os") => cfg!(feature = "seed-injection"),
        Ok(v) => refuse(format!("FC_COINS={v}: \"seed\" (the default) or \"os\"")),
    }
}

/// M1's pad layout for the statement: the algebraic phase's, then Ligerito level 0's (its OOD samples and lane rounds).
fn zk_layout(st: &Stmt) -> Layout {
    let lc = pcs_of(st, LigeritoProfile::Fast100).ligerito_prover_config().expect("Ligerito config");
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

/// The claim points a table's rank check covers at its last opening (ab and c, for both reps).
const RANK_POINTS: usize = 4;
/// Mask words beyond the rank check's `128 · RANK_POINTS`. For uniformly random rows the rank falls short with probability
/// below `2^(128 · RANK_POINTS − N)` at `N` words; a shortfall is a refused coin at an opening, which costs completeness only.
const RANK_SLACK: usize = 64;
/// The least mask slot `--zk` takes, `2^13` bits per block (the deployed `mask_log` is 14): the bound on the rank check
/// refusing a coin, its completeness loss, covers no smaller slot.
const SLOT_LOG_MIN: usize = 13;

/// A statement `--zk` can prove: level 0's query count ([`zk_queries0`]), a mask slot of at least `2^SLOT_LOG_MIN`
/// bits, and mask words enough for the rank check (§6) to be reachable. A table of a split session has fewer blocks, so
/// fewer mask words. Checked when the statement is built, before `Hello`, never at an opening.
fn zk_shape_ok(st: &Stmt) -> Result<(), String> {
    zk_queries0(st.m)?;
    let r = st.c.ranges.iter().find(|r| r.kind == Kind::Mask).ok_or("zk: the circuit has no mask slot")?;
    if r.slot_log < SLOT_LOG_MIN {
        return Err(format!("slot_log = {}: --zk needs a mask slot of slot_log >= {SLOT_LOG_MIN}", r.slot_log));
    }
    let (words, need) = (mask_words(st).len(), 128 * RANK_POINTS);
    if words < need + RANK_SLACK {
        return Err(format!("{words} mask words: the rank check needs {need} for its {RANK_POINTS} claim points, and {RANK_SLACK} more"));
    }
    Ok(())
}

/// Ligerito level 0's hiding randomness for a session (both reps open the one level-0 codeword): every lane's padding, and
/// the four uniform lanes (a pair per rep), read from the OS (an injected seed's ChaCha20 streams 6 and 7); for rep 0 (a rep's
/// copy sets `rep`). `zero`: M1's control (all zero). The lanes are drawn in parallel (`ProverRng::f128s_lanes`).
fn level0_zk(st: &Stmt, rng: &mut flock_live::zk_hooks::ProverRng, zero: bool, tamper: u8) -> flock_core::pcs::ligerito::zk::Level0Zk {
    use flock_core::pcs::ligerito::zk::{EXTRA_LANES, Level0Zk};
    let tr = Instant::now();
    let lc = pcs_of(st, LigeritoProfile::Fast100).ligerito_prover_config().expect("Ligerito config");
    let (lanes, l, t) = ((1usize << lc.initial_k) + EXTRA_LANES, 1usize << lc.initial_log_msg_cols, zk_t_pad(st.m));
    let draw = |stream: u64, lanes: usize, n: usize| -> Vec<Vec<F128>> {
        if zero { vec![vec![F128::ZERO; n]; lanes] } else { rng.f128s_lanes(stream, lanes, n) }
    };
    let z = Level0Zk { queries0: zk_queries0(st.m).expect("checked when the statement is built"), t_pad: t, rep: 0,
                       extra: draw(6, EXTRA_LANES, l).try_into().expect("EXTRA_LANES lanes"), pads: draw(7, lanes, t), tamper };
    eprintln!("ZKL0\t{}", json!({"lanes": lanes, "t_pad": t, "extra_words": l, "s": tr.elapsed().as_secs_f64()}));
    z
}

/// Under the Lean session loop (`plan.lean_frames`), a firewalled table's level-0 hiding and salts are the loop's: it draws
/// them for the host's packed witness `z` and commits level 0 on them (`Flock.Firewall.Level0`). They replace `l0`'s and
/// the table's level-0 salts, so the prover commits the codeword the loop did, and the loop's root is returned: the root of
/// record. The loop keeps it for table `j` of the session whose `Hello` was answered `coins`, until that session ends, and
/// draws a table's level 0 once. Without the loop, nothing changes and nothing is returned. Under `FC_LEVEL0_TAMPER=MODE`
/// (the selftest's `zk_worker_level0`), table 0 asks the loop for one more level 0 after its own: `second` for table 0
/// again, `past` for a table the session does not have (index [`LEVEL0_PAST`]); the loop's refusal goes to
/// [`LEVEL0_REFUSED`] (a draw it answers goes there as `drawn again`), and the session goes on on the first draw.
#[cfg(feature = "sha512")]
fn lean_level0(plan: &Plan, st: &Stmt, lig: LigeritoProfile, coins: &[F128], j: usize, z: &[F128],
               l0: &mut flock_core::pcs::ligerito::zk::Level0Zk, salts: &Mutex<flock_live::zk_hooks::LeafSalts>)
               -> Result<Option<Vec<u8>>, String> {
    use flock_core::pcs::ligerito::zk::EXTRA_LANES;
    let Some(addr) = plan.lean_frames.as_deref() else { return Ok(None) };
    if plan.zk_zero_pads {
        return Err("FIREWALL-REFUSED: the Lean session loop draws level 0's hiding; zero pads are Rust's control".into());
    }
    let pcs = pcs_of(st, lig);
    let lc = pcs.ligerito_prover_config().expect("Ligerito config");
    let (session, t_pad) = (flock_live::firewall::level0_session(coins), l0.t_pad);
    let draw = |table: usize| flock_live::firewall::lean_level0(addr, &session, table, st.m, z, 1 << lc.initial_log_msg_cols,
                                                                (1 << lc.initial_k) + EXTRA_LANES, t_pad, pcs.n_leaves());
    let got = draw(j)?;
    let tamper = std::env::var("FC_LEVEL0_TAMPER").unwrap_or_default();
    if j == 0 && !tamper.is_empty() {
        let again = match tamper.as_str() {
            "second" => draw(0),
            "past" => draw(LEVEL0_PAST),
            m => panic!("FC_LEVEL0_TAMPER={m}: second or past"),
        };
        *LEVEL0_REFUSED.lock().unwrap_or_else(PoisonError::into_inner) = Some(match again {
            Ok(_) => "drawn again".to_string(),
            Err(e) => e,
        });
    }
    l0.extra = got.extra.try_into().map_err(|_| "the Lean loop's level 0: not four extra lanes".to_string())?;
    l0.pads = got.pads;
    salts.lock().unwrap_or_else(PoisonError::into_inner).set_level0(got.salts)?;
    Ok(Some(got.root))
}

/// The table index `FC_LEVEL0_TAMPER=past` asks the loop's level 0 for: one past any session's tables.
const LEVEL0_PAST: usize = 1_000_000;
/// What the loop answered `FC_LEVEL0_TAMPER`'s extra level-0 request: its refusal, or `drawn again`.
static LEVEL0_REFUSED: Mutex<Option<String>> = Mutex::new(None);

#[cfg(not(feature = "sha512"))]
fn lean_level0(_: &Plan, _: &Stmt, _: LigeritoProfile, _: &[F128], _: usize, _: &[F128], _: &mut flock_core::pcs::ligerito::zk::Level0Zk,
               _: &Mutex<flock_live::zk_hooks::LeafSalts>) -> Result<Option<Vec<u8>>, String> {
    Ok(None)
}

/// M1's hooks identity: what is masked, and what is not yet (the statement is not zero knowledge until the list is empty).
/// `refused_coins`: the coins at which the prover stops, each before it sends anything that depends on them and reading only
/// opened coins (live PROTOCOL.md, "Refused coins"):
/// - `β = 0` (the inner proof);
/// - mask-slot weights of rank `< 128·k` over the claims so far (the rank check, §6);
/// - `ρ_0, ρ_1` dependent over F128;
/// - a zerocheck outer coin `r_j = 1`, as it opens ([`zkv::Masking`]). Completeness: under uniform coins the honest prover
///   refuses with probability at most `(#outer coins) · 2^-128` per rep, `m − 13` outer coins.
fn zk_identity(c: &Composite) -> Value {
    json!({"zk": true, "stage": "m2-prototype (committed verifier coins, CPU)", "compiler": "veil (ePrint 2026/683 §4.1, Remark 4.4) over gf(2^128)",
           "verifier_coins": format!("committed before the prover's first message ({}): the prover's fresh nonce in hello; the root and the \
                                      verifier's session key in its answer; every coin opened against them, or the prover stops",
                                     flock_live::coin_tree::SCHEME),
           "pads": "zerocheck and lincheck: every sent value + a pad (os bytes, getrandom); pads committed before the zerocheck's first coin",
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
           "refused_coins": ["beta = 0", "mask-slot weights of rank < 128 k over the claims so far", "rho_0, rho_1 dependent over F128",
                             "a zerocheck outer coin r_j = 1"],
           "not_yet_masked": []})
}


/// The backend identity the statement digest pins (so a NON_ZK and a ZK cell never share one).
fn identity(c: &Composite) -> Value {
    let hooks = if zk_mode() { zk_identity(c) } else { NoMask::new(ProverSeed::from_os()).identity() };
    let mut id = json!({"statement": c.statement(), "backend": "C-interactive", "proof_class": "NON_ZK_PROOF", "profile": "fast100", "reps": 2,
           "commitments": "one root, both reps (R1)", "flavor": "rs", "leaf_scheme": c.leaf_scheme,
           "outputs": "hidden: each output port an hm96-sha512/row/v1 row hashed in the circuit; the public file has no output word",
           // a selftest build injects seeds (the determinism check); a proving build never has the feature
           "seed_injection": cfg!(feature = "seed-injection"),
           // the frame-v3 tree and the row leaf are the serving side's formats, not this statement's to choose
           "hashes": {"merkle": format!("{} (every Ligerito level)", merkle(c).as_str()),
                      "round_digest": "sha512 (framing: the coin server retains every round's bytes and the replay compares them)",
                      "statement_digest": "sha512", "sigma": "sha512", "coin_commitment": "sha512",
                      "coin_derivation": "sha512 (verity.randomness)", "frame_v3_tree": "sha512 (frame-v3-sha512)",
                      "transcript": "none (live verifier coins)", "row_leaf_in_circuit": "hm96-sha512/row/v1 over sha512/row/v1"},
           "hooks": hooks});
    // only under FC_COINS=os: the statement digest pins the identity, and the default's must not move
    if !zk_mode() && !seed_coins() {
        id["coins"] = json!("live-os (every round's coins the verifier's OS draws: no seed, no PRF)");
        id["hashes"]["coin_commitment"] = json!("none (FC_COINS=os)");
        id["hashes"]["coin_derivation"] = json!("none (FC_COINS=os)");
    }
    // only for bit rows (META `row_schemas`): a `sha512/row/v1` statement's digest must not move
    if c.meta.get("row_schemas").is_some() {
        id["hashes"]["row_leaf_in_circuit"] = json!("hm96-sha512/row/v2 over sha512/row/v2");
        id["outputs"] = json!("hidden: each output port an hm96-sha512/row/v2 row hashed in the circuit; the public file has no output word");
    }
    // only with a segmented row: a statement of v2 rows alone keeps its digest
    if c.ports.iter().any(|p| p.segs.is_some()) {
        id["hashes"]["row_leaf_in_circuit"] = json!(ROW_LEAF_SEG_IDENTITY);
    }
    // only with runs of output ports (META `out_row_ports`): a statement of one output row per port keeps its digest
    if c.meta.get("out_row_ports").is_some() {
        // a run committed as a v2 row names both leaves; runs of segmented rows alone keep their digest
        let v2_run = c.out_rows().iter().enumerate().any(|(j, p)| p.segs.is_none() && c.out_group(j).1 > 1);
        id["outputs"] = json!(if v2_run { OUT_ROWS_V2_IDENTITY } else { OUT_ROWS_SEG_IDENTITY });
    }
    if c.pinned > 0 {
        id["outputs"] = json!(PINNED_IDENTITY);
    }
    id
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

/// A session's Σ: its one table's, or for several SHA-512(the statement's Σ tag ‖ `-tables` ‖ the table count (u64 LE) ‖ every
/// table's Σ, in table order).
fn session_sigma(tabs: &[(&str, &Stmt, &Instances)]) -> [u8; 64] {
    if let [(_, st, inst)] = tabs {
        return sigma(st, inst);
    }
    let mut h = Sha512::new();
    h.update(tabs[0].1.c.tags().sigma.as_bytes());
    h.update(b"-tables");
    h.update((tabs.len() as u64).to_le_bytes());
    for (_, st, inst) in tabs {
        h.update(sigma(st, inst));
    }
    h.finalize().into()
}

/// One table of a session (the public-circuit ZK proof's table `j`): its name, and the statement it proves.
#[derive(Clone)]
struct Table {
    name: String,
    st: Arc<Stmt>,
    inst: Arc<Instances>,
}

/// A session's statement: its tables, and the verifier's draw they prove (`None`: the registered population's units).
#[derive(Clone)]
struct Tables {
    list: Vec<Table>,
    draw: Option<Value>,
}

impl Tables {
    fn one(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Self {
        Tables { list: vec![Table { name: TABLE.into(), st: st.clone(), inst: inst.clone() }], draw: None }
    }

    fn parts(&self) -> Vec<(&str, &Stmt, &Instances)> {
        self.list.iter().map(|t| (t.name.as_str(), &*t.st, &*t.inst)).collect()
    }

    fn names(&self) -> Vec<String> {
        self.list.iter().map(|t| t.name.clone()).collect()
    }

    /// The session's rep streams, table by table: the coin schedule's order after `session`, and the prover's.
    fn streams(&self) -> Vec<String> {
        self.list.iter().flat_map(|t| [rep_stream(&t.name, 0), rep_stream(&t.name, 1)]).collect()
    }
}

/// The table names of a session of `tables` tables: `circuit` alone, or `circuit-00`, `circuit-01`, ...
fn table_names(tables: usize) -> Vec<String> {
    if tables <= 1 { vec![TABLE.into()] } else { (0..tables).map(|j| format!("{TABLE}-{j:02}")).collect() }
}

/// A session's tables over a registered population, which both sides derive from the population file, the draw and the
/// count. One table `circuit` proves the drawn units (the population's with no draw). For several, the units split into
/// `count` equal consecutive parts, and table `j` proves part `j` as a subset draw of the population, under an identity that
/// names the session's rules for several tables.
fn session_tables(st: &Arc<Stmt>, inst: &Arc<Instances>, draw: Option<&Value>, count: usize) -> Result<Tables, String> {
    if count <= 1 {
        let (st, inst) = match draw { Some(d) => drawn(st, inst, d)?, None => (st.clone(), inst.clone()) };
        if draw.is_some() && is_zk(&st) {
            zk_shape_ok(&st)?;
        }
        return Ok(Tables { list: vec![Table { name: TABLE.into(), st, inst }], draw: draw.cloned() });
    }
    if count > MAX_TABLES {
        return Err(format!("tables: at most {MAX_TABLES} per session, not {count}"));
    }
    let units = match draw {
        Some(d) if d["population"].as_u64() != Some(inst.n as u64) => return Err("U: the draw's population is not this registered population".into()),
        Some(d) if d["law"] != "subset" && d["law"] != "stratified" => return Err(TABLES_LAW.into()),
        Some(d) => flock_live::unit_draw::check(d)?,
        None => (0..inst.n).collect(),
    };
    if units.is_empty() || units.len() % count != 0 {
        return Err(format!("tables: {} units do not split into {count} equal tables", units.len()));
    }
    let k = units.len() / count;
    let list = units.chunks(k).zip(table_names(count)).enumerate().map(|(j, (part, name))| {
        let di = inst.drawn(&st.c, &json!({"law": "subset", "population": inst.n, "k": k, "units": part}))?;
        check_public(&st.c, &di)?;
        let mut id = st.identity.clone();
        id["session"] = json!({"tables": count, "table": j,
                               "units": "the draw's units (without a draw, the population's) in equal consecutive parts; table j proves part j as a subset draw of the population"});
        if is_zk(st) {
            id["hooks"]["tables"] = json!("per table: os bytes of its own (getrandom) and hm96 salt trees from id j << 32 (table 0's are a \
                                           one-table session's); a rank check per table; every table's level-0 root in one commit; every proof \
                                           after the last coin of every stream (the barrier)");
        }
        let ds = Stmt::new(st.c.clone(), di.n, id)?;
        if is_zk(&ds) {
            zk_shape_ok(&ds)?;
        }
        Ok(Table { name, st: Arc::new(ds), inst: Arc::new(di) })
    }).collect::<Result<Vec<_>, String>>()?;
    if list.iter().any(|t| t.st.m_pts != list[0].st.m_pts) {
        return Err("tables: the tables' link points have different lengths".into());
    }
    Ok(Tables { list, draw: draw.cloned() })
}

/// `--session-tables J`: the tables of every session (1, the default: one table).
fn tables_arg() -> usize {
    match arg("--session-tables", "1").parse() {
        Ok(j) if (1..=MAX_TABLES).contains(&j) => j,
        _ => refuse(format!("--session-tables: 1 to {MAX_TABLES}")),
    }
}

struct CircuitVerifier {
    tabs: Tables,
    /// how the lincheck's last vector is computed (`FC_LINCHECK`)
    lincheck: sverify::Lincheck,
    /// the last `verify` call's phase timers
    phases: Mutex<Option<Value>>,
    /// `proved_scope::check` of the statement: the refusal of a form the soundness proof does not cover
    scope: Result<(), String>,
}

impl StatementVerifier for CircuitVerifier {
    fn domain(&self, t: &str, rep: usize) -> Vec<u8> {
        domain_of(&self.tabs.list[0].st.c, self.tabs.list.len(), t, rep)
    }

    fn verify(&self, t: &str, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let tb = self.tabs.list.iter().find(|x| x.name == t).ok_or(format!("no table {t} in the statement"))?;
        if is_zk(&tb.st) {
            return Self::verify_zk(tb, rep, proof, ch, link);
        }
        let t0 = Instant::now();
        let mut ph = sverify::Phases::default();
        let r = self.verify_circuit(tb, proof, ch, link, &mut ph);
        ph.total_s = t0.elapsed().as_secs_f64();
        *self.phases.lock().unwrap_or_else(PoisonError::into_inner) = Some(ph.json());
        r
    }

    fn take_phases(&self) -> Option<Value> {
        self.phases.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    fn refusal(&self) -> Option<String> {
        self.scope.clone().err()
    }
}

impl CircuitVerifier {
    fn new(tabs: Tables) -> Self {
        let t0 = &tabs.list[0];
        let scope = proved_scope::check(&t0.st.c, &t0.inst, is_zk(&t0.st), tabs.list.len());
        CircuitVerifier { tabs, lincheck: sverify::Lincheck::from_env().unwrap_or_else(|e| refuse(e)), phases: Mutex::new(None), scope }
    }

    /// The verifier of one-table sessions.
    fn one(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Self {
        Self::new(Tables::one(st, inst))
    }

    /// M0: upstream's `verify_ligerito_extra` on the session's challenger (`session_verify::verify_extra`), timed into `ph`.
    fn verify_circuit(&self, tb: &Table, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>, ph: &mut sverify::Phases) -> Result<Vec<u8>, String> {
        let st = &tb.st;
        let t = Instant::now();
        let (commitment, proof): (Commitment, R1csProofLigerito) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        ph.decode_s = t.elapsed().as_secs_f64();
        let pcs = pcs_of(st, LigeritoProfile::Fast100);
        if commitment.params != pcs {
            return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        if link.points.len() != 2 || link.y.iter().any(|y| *y != F128::ZERO) {
            return Err("session: two points and y = 0 expected (nothing is prover-supplied)".into());
        }
        let t = Instant::now();
        let extra = st.extra_claims(link.points, &tb.inst);
        ph.regions_s = t.elapsed().as_secs_f64();
        catch_unwind(AssertUnwindSafe(|| sverify::verify_extra(&st.r1cs, &st.c0_identity, &commitment, &proof, &st.circuit, &pcs, &extra, self.lincheck, ch, ph)))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(merkle_digest_bytes(&commitment.cap).to_vec())
    }

    /// M1: the zerocheck and lincheck replayed on masked values as constraints on the committed pads (`zk_veil::replay`),
    /// upstream's opening with the two masked claims' ring-switch checks turned into constraints, then the inner proof.
    fn verify_zk(tb: &Table, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        if link.points.len() != 2 || link.y.iter().any(|y| *y != F128::ZERO) {
            return Err("session: two points and y = 0 expected (nothing is prover-supplied)".into());
        }
        let (commitment, proof, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        verify_zk_with(&tb.st, &tb.inst, &commitment, &proof, &inner, link.points, rep, ch)?;
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
    let pcs = pcs_of(st, LigeritoProfile::Fast100);
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
        zkv::verify_inner(inner, &zk_layout(st), proof.zerocheck.final_c_eval, cons, ch)
    }))
    .map_err(|_| "verifier panicked".to_string())?
    .map_err(|e| e.chars().take(200).collect::<String>())
}

/// M2's committed coin schedule: the session's stream (the link points), then every table's two reps, in table order (a
/// subtree per table and rep); 512 rounds of at most 512 coins each (a RoPE rep uses 100 rounds, at most 218 coins in one).
fn coin_spec_of(tables: &[String]) -> flock_live::coin_tree::Spec {
    use flock_live::coin_tree::{SESSION, Spec};
    let streams = std::iter::once(SESSION.to_string()).chain(tables.iter().flat_map(|t| [rep_stream(t, 0), rep_stream(t, 1)])).collect();
    Spec { streams, rounds_log: 9, block: 512 }
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
    config_with(&[(TABLE, st, inst)], inst.header.get("unit_draw").cloned())
}

/// [`config`] over a session's tables: for several, the session's draw (each table's header has its part).
fn config_of(tabs: &Tables) -> SessionConfig {
    let parts = tabs.parts();
    let draw = if let [(_, _, inst)] = parts[..] { inst.header.get("unit_draw").cloned() } else { tabs.draw.clone() };
    config_with(&parts, draw)
}

/// The session's configuration over its tables: one link exchange, whose points every table reads (the tables share `m_pts`,
/// [`session_tables`]), and each table's own root, y and rep streams.
fn config_with(tabs: &[(&str, &Stmt, &Instances)], unit_draw: Option<Value>) -> SessionConfig {
    let st = tabs[0].1;
    let specs = tabs.iter().map(|(name, st, _)| TableSpec { name: name.to_string(), binding: Binding::R1cs { statement_digest: st.flock_digest } }).collect();
    let mut cfg = SessionConfig::flock_128_r2(specs);
    cfg.link = Some(LinkSpec { sigma: session_sigma(tabs).to_vec(), m: st.m_pts, points: 2 });
    // M2 (--zk): every coin committed at Hello and opened round by round; M0: the step-0 seed, or OS coins (seed_coins)
    if is_zk(st) {
        cfg.coin_tree = Some(coin_spec_of(&tabs.iter().map(|t| t.0.to_string()).collect::<Vec<_>>()));
    } else {
        cfg.coin_seed = seed_coins();
    }
    cfg.retain_rounds = true;
    cfg.sha512_digests = true;
    cfg.unit_draw = unit_draw;
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
    server_tables(st, inst, 1)
}

/// The verifier's server for sessions of `count` tables. With a draw (`--draw`, `--draw-law`, `--draw-file`) the tables are the drawn
/// units' (until `Register`, the population's one-table statement stands in); without, the population's.
fn server_tables(st: &Arc<Stmt>, inst: &Arc<Instances>, count: usize) -> Server {
    server_of(&serving(st, inst, count), st, inst, count)
}

/// What every session of a server serves: the verifier's own draw (`--draw-file`) or its law (`--draw`, `--draw-law`), and
/// without either the population's tables. Built once, when the server starts.
struct Serving {
    given: Option<Value>,
    law: Option<flock_live::unit_draw::Law>,
    tabs: Tables,
}

fn serving(st: &Arc<Stmt>, inst: &Arc<Instances>, count: usize) -> Serving {
    use flock_live::unit_draw::Law;
    let (file, law, law_file) = (arg("--draw-file", ""), arg("--draw", ""), arg("--draw-law", ""));
    let read = |flag: &str, path: &str| -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|e| refuse(format!("{flag}: {e}"))))
            .unwrap_or_else(|e| refuse(format!("{flag}: {e}")))
    };
    if [&file, &law, &law_file].iter().filter(|s| !s.is_empty()).count() > 1 {
        refuse("one of --draw-file, --draw and --draw-law".into());
    }
    // the verifier's own draw, made outside this server: served at Register as it is (checked there, U2, and against the
    // registered population when the drawn statement is built)
    let given = (!file.is_empty()).then(|| {
        let draw = read("--draw-file", &file);
        flock_live::unit_draw::check(&draw).unwrap_or_else(|e| refuse(e));
        draw
    });
    // `--draw-law`: a law object, a stratified one the verifier's own expansion over its program and query, drawn afresh at
    // each session's Register over this registered population
    let law = if !law_file.is_empty() {
        let law = Law::of_json(&read("--draw-law", &law_file)).unwrap_or_else(|e| refuse(format!("--draw-law: {e}")));
        if let Law::Stratified { k, strata } = &law {
            if let Some(why) = flock_live::unit_draw::strata_problem(*k, strata, inst.n as u64) {
                refuse(format!("--draw-law over the {} registered units: {why}", inst.n));
            }
        }
        Some(law)
    } else {
        (!law.is_empty()).then(|| Law::parse(&law).unwrap_or_else(|e| refuse(e)))
    };
    // several tables are refused here, before any session, for a draw they cannot split or whose tables `--zk` cannot prove. A
    // subset or stratified law's table shapes depend on its counts alone, so a draw of the first units stands for every draw.
    if count > 1 {
        let probe = match (&given, &law) {
            (Some(d), _) => Some(d.clone()),
            (None, Some(Law::Bernoulli { .. })) => refuse(TABLES_LAW.into()),
            (None, Some(l)) => l.first(inst.n as u64),
            (None, None) => None,
        };
        if let Some(d) = probe {
            session_tables(st, inst, Some(&d), count).unwrap_or_else(|e| refuse(e));
        }
    }
    let tabs = if count > 1 && given.is_none() && law.is_none() {
        session_tables(st, inst, None, count).unwrap_or_else(|e| refuse(e))
    } else {
        Tables::one(st, inst)
    };
    Serving { given, law, tabs }
}

/// One session's server.
fn server_of(sv: &Serving, st: &Arc<Stmt>, inst: &Arc<Instances>, count: usize) -> Server {
    let s = Server::new(config_of(&sv.tabs), Box::new(CircuitVerifier::new(sv.tabs.clone())));
    if let Some(draw) = &sv.given {
        let mut s = with_draw(s, st, inst, flock_live::unit_draw::Law::Subset { k: 0 }, count);
        s.cfg.draw_law = None;
        s.cfg.draw_given = Some(draw.clone());
        return s;
    }
    match &sv.law {
        Some(l) => with_draw(s, st, inst, l.clone(), count),
        None => s,
    }
}

/// A server that draws `law` over the registered population at `Register` and serves the drawn units' statement, in `count`
/// tables.
fn with_draw(mut s: Server, st: &Arc<Stmt>, inst: &Arc<Instances>, law: flock_live::unit_draw::Law, count: usize) -> Server {
    s.cfg.draw_law = Some((law, inst.n as u64));
    let (st, inst) = (st.clone(), inst.clone());
    s.with_draw(Box::new(move |draw: &Value| {
        let tabs = session_tables(&st, &inst, Some(draw), count)?;
        Ok((config_of(&tabs), Box::new(CircuitVerifier::new(tabs)) as Box<dyn StatementVerifier>))
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
/// units' input words, the hm96 slots computed here, and M1's mask slot; with `mapped`, the host slots in mapped host memory);
/// otherwise the packed host witness.
fn witness(st: &Stmt, w: &Wit, device: bool, host_units: bool, mapped: bool) -> (WitData, f64) {
    let t = Instant::now();
    #[cfg(feature = "gpu")]
    let pack = |h: &[flock_live::gpu_circuit::HostSlot]| if mapped { HostSlots::pack(h) } else { HostSlots::pack_pageable(h) };
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
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && w.regions.is_none() && st.c.out_net == st.c.unit_net && st.c.typed.is_none() {
        use rayon::prelude::*;
        let ur = st.c.unit_range();
        // FC_WITNESS_TIMES: the host witness's parts on stderr (WITNESS lines)
        let times = std::env::var_os("FC_WITNESS_TIMES").is_some();
        let lap = |what: &str, t: &mut Instant, v: &mut serde_json::Map<String, Value>| {
            v.insert(what.into(), json!(t.elapsed().as_secs_f64()));
            *t = Instant::now();
        };
        let (mut tl, mut parts) = (Instant::now(), serde_json::Map::new());
        let ri: Vec<_> = (0..st.blocks()).into_par_iter().map(|b| (w.rowin)(b)).collect();
        let rows: Vec<F128> = ri.iter().flat_map(|x| x.0.iter().flatten().copied()).collect();
        let hm: Vec<Vec<F128>> = ri.into_iter().flat_map(|x| x.1).collect();
        lap("rows", &mut tl, &mut parts);
        let mut host = flock_live::gpu_circuit::host_slots(st, st.c.hm_net, &hm);
        host.extend(mask_slots());
        lap("hm96", &mut tl, &mut parts);
        let unit_inputs = || (0..st.blocks() * ur.count).into_par_iter().map(|x| (w.inputs)(x / ur.count, x % ur.count));
        if st.c.wires.iter().any(|x| x.dst.0 != st.c.sha_net && x.dst.0 != st.c.hm_net) {
            let inputs: Vec<F128> = unit_inputs().flatten_iter().collect();
            return (WitData::DevCut(rows, pack(&host), inputs, w.inputs.clone()), t.elapsed().as_secs_f64());
        }
        if host_units {
            let ins: Vec<Vec<F128>> = unit_inputs().collect();
            lap("unit_inputs", &mut tl, &mut parts);
            if mapped {
                let packed = HostSlots::pack_units(st, &host, st.c.unit_net, &ins);
                lap("units_pack", &mut tl, &mut parts);
                if times {
                    eprintln!("WITNESS\t{}", Value::Object(parts));
                }
                return (WitData::Dev(rows, packed, Vec::new()), t.elapsed().as_secs_f64());
            }
            host.extend(flock_live::gpu_circuit::host_slots(st, st.c.unit_net, &ins));
            lap("units", &mut tl, &mut parts);
            let packed = pack(&host);
            lap("pack", &mut tl, &mut parts);
            if times {
                eprintln!("WITNESS\t{}", Value::Object(parts));
            }
            return (WitData::Dev(rows, packed, Vec::new()), t.elapsed().as_secs_f64());
        }
        return (WitData::Dev(rows, pack(&host), unit_inputs().flatten_iter().collect()), t.elapsed().as_secs_f64());
    }
    #[cfg(feature = "gpu")]
    // a tail that returns the outputs (a chain of units, the tail's stages): the host converges every slot's inputs and computes
    // the stages and lookups; the device the compressions, and the units from their converged inputs
    if device && w.flip_z.is_none() && w.cut_flip.is_none() && w.nets.is_empty() && w.regions.is_none() && st.c.typed.is_none() {
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
        return (WitData::Dev(rows, pack(&host), inputs), t.elapsed().as_secs_f64());
    }
    let _ = (device, host_units, mapped);
    let mut ins = st.converge(&*w.inputs, &*w.rowin, &w.nets);
    if let Some((n, x, col)) = w.cut_flip {
        let word = &mut ins[n][x][col / 128];
        if col % 128 < 64 { word.lo ^= 1 << (col % 128) } else { word.hi ^= 1 << (col % 128 - 64) }
    }
    let (mut z, a, b, zl) = st.witness_with(&ins, &*w.mask, &w.nets, w.typed_flip, w.typed_forged.as_deref());
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
        &st.r1cs, &pcs_of(st, lig), z, a, b, zl, &st.circuit,
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
    /// a dishonest prover: re-solve `final_c'` after the inner proof's coins so that the batched constraint holds
    final_c_solved: bool,
    /// whether the inner proof stops when its batched constraint fails on the pads
    self_check: zkv::SelfCheck,
    /// seconds to draw the pads and build their commitment (counted in the rep's prove time)
    t_commit: f64,
    /// Ligerito level 0's hiding, drawn once per session (both reps open the one level-0 codeword), with this rep's pair
    level0: Arc<flock_core::pcs::ligerito::zk::Level0Zk>,
    /// under the firewall, the rep's frame in the Lean session loop, which drew `pads` and the pads tree's salts (or why it
    /// could not)
    lean: Mutex<Option<Result<flock_live::firewall::LoopRep, String>>>,
}

/// The rep's frame in the Lean session loop and its draws, Rust's tree over them held to the frame's root: table `table`'s
/// rep `rep` in the session the loop names `name`, at its `frames` address `addr` (`flock_live::firewall::LoopRep::start`).
fn lean_draws(lay: &Layout, frames: Option<(&str, &str, usize, usize)>) -> Result<(flock_live::firewall::LoopRep, zkv::Pads, zkv::InnerCommit), String> {
    let (addr, name, table, rep) = frames.ok_or_else(|| flock_live::firewall::refused("a firewalled rep's frame is the Lean session loop's, and none serves the session"))?;
    let (frame, d) = flock_live::firewall::LoopRep::start(addr, name, table, rep, lay)?;
    let ic = zkv::InnerCommit::new(&d.pads, d.salts);
    if ic.root() != d.root {
        return Err(flock_live::firewall::refused(format!("the pads tree over the Lean frame's draws has root {} where the frame's is {}",
                                                         hex(&ic.root()[..8]), hex(&d.root[..8]))));
    }
    Ok((frame, d.pads, ic))
}

/// A rep's pads and their commitment (M1), the pads tree under salt id `id`, with the session's level-0 hiding and this rep's pair.
/// Under the firewall the rep's frame in the Lean session loop draws them instead, from the OS: of the session the loop
/// names and the table (`frame`).
#[allow(clippy::too_many_arguments)]
fn make_zkrep(st: &Stmt, plan: &Plan, prng: &mut flock_live::zk_hooks::ProverRng, level0: &flock_core::pcs::ligerito::zk::Level0Zk,
              points: &Arc<Mutex<Vec<Vec<F128>>>>, rep: usize, id: u64, salts: &Mutex<flock_live::zk_hooks::LeafSalts>,
              frame: Option<(&str, usize)>) -> ZkRep {
    let tz0 = Instant::now();
    let lay = zk_layout(st);
    let at = plan.lean_frames.as_deref().zip(frame).map(|(a, (name, table))| (a, name, table, rep));
    let (mut pads, mut ic, lean) = match firewall_on(st).then(|| lean_draws(&lay, at)) {
        Some(Ok((frame, pads, ic))) => (pads, Some(ic), Some(Ok(frame))),
        Some(Err(why)) => (zkv::Pads::draw(prng, &lay, rep), None, Some(Err(why))),
        None => (zkv::Pads::draw(prng, &lay, rep), None, None),
    };
    if plan.zk_zero_pads {
        pads.h.iter_mut().for_each(|x| *x = F128::ZERO);
        ic = None;
    }
    let ic = ic.unwrap_or_else(|| {
        let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
        zkv::InnerCommit::new(&pads, salts.lock().unwrap_or_else(PoisonError::into_inner).expand(id, n))
    });
    let self_check = if plan.zero_witness { zkv::SelfCheck::Skip } else { plan.zk_self_check };
    let t_commit = tz0.elapsed().as_secs_f64();
    // each rep's copy of the extra lanes (4 · 2^22 words at m = 35) written on rayon's threads, not one
    let level0 = flock_core::pcs::ligerito::zk::Level0Zk {
        queries0: level0.queries0, t_pad: level0.t_pad, rep: if plan.zk_shared_pair { 0 } else { rep }, tamper: level0.tamper,
        extra: std::array::from_fn(|i| { use rayon::prelude::*; level0.extra[i].par_iter().copied().collect() }), pads: level0.pads.clone(),
    };
    ZkRep { lay, pads, ic, points: points.clone(), pad_mismatch: plan.zk_pad_mismatch, final_c_solved: plan.zk_final_c_solved, self_check,
            t_commit, level0: Arc::new(level0), lean: Mutex::new(lean) }
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
    let trees = pcs_of(st, lig).ligerito_prover_config().expect("Ligerito config").recursive_steps as u64;
    let id0 = salts.lock().unwrap_or_else(PoisonError::into_inner).next_id();
    let reps = [0usize, 1].map(|rep| {
        let id = id0 + rep as u64 * (1 + trees);
        (make_zkrep(st, plan, &mut prng, &level0, points, rep, id, salts, None), id)
    });
    eprintln!("ZKPRE\t{}", json!({"level0_s": t_level0, "s": tl0.elapsed().as_secs_f64()}));
    (t_level0, reps)
}

/// A table's level-0 root before any stream opens, in a session of several tables (whose one `Commit` carries every table's
/// root): the commitment the table's reps compute again, from its witness, its level-0 hiding and its level-0 salts.
fn level0_root(st: &Stmt, wt: &WitData, lig: LigeritoProfile, level0: Option<&flock_core::pcs::ligerito::zk::Level0Zk>,
               salts: &Mutex<flock_live::zk_hooks::LeafSalts>) -> Vec<u8> {
    #[allow(irrefutable_let_patterns)]
    let WitData::Host((z, ..)) = wt else { panic!("a table committed before the session opens takes a host witness") };
    let pcs = pcs_of(st, lig);
    #[cfg(feature = "sha512")]
    {
        let s = salts.lock().unwrap_or_else(PoisonError::into_inner);
        flock_merkle::hm96::begin(s.level0(pcs.n_leaves()), Box::new(|_| panic!("the level-0 commitment draws no other tree")));
    }
    #[cfg(not(feature = "sha512"))]
    let _ = salts;
    let (cm, _) = match level0 {
        Some(z0) => flock_core::pcs::commit::commit_zk(z, &pcs, z0),
        None => flock_core::pcs::commit::commit(z, &pcs),
    };
    #[cfg(feature = "sha512")]
    flock_merkle::hm96::end();
    merkle_digest_bytes(&cm.cap).to_vec()
}

/// The mask slot: its first word, its words per block, and the blocks.
fn mask_slot(st: &Stmt) -> flock_live::firewall::MaskSlot {
    let r = st.c.ranges.iter().find(|r| r.kind == Kind::Mask).expect("a mask slot");
    flock_live::firewall::MaskSlot { w0: r.start() >> 7, n: 1usize << (r.slot_log - 7), blocks: st.blocks() }
}

/// The mask slot's word indices in the packed witness (every block's slot).
fn mask_words(st: &Stmt) -> Vec<usize> {
    let s = mask_slot(st);
    (0..s.blocks).flat_map(|b| (0..s.n).map(move |j| (b << (st.c.k_log - 7)) + s.w0 + j)).collect()
}

/// The two z-claims' word coordinates (`x_outer[1..]` of the opening's claim point) for the phase's coins: the replay on a
/// zero proof of the right shape (the points depend on the coins alone).
fn claim_word_points(st: &Stmt, lay: &Layout, coins: &[F128]) -> Result<[Vec<F128>; 2], String> {
    use flock_core::lincheck::LincheckCircuit;
    let (ab, c) = zkv::points_from_coins(&st.r1cs, lay, st.circuit.const_pin_col().is_some(), coins)?;
    Ok([zkv::point_parts(&ab).1[1..].to_vec(), zkv::point_parts(&c).1[1..].to_vec()])
}

/// The pads a rep's values are sent under: its own (a dishonest prover's other ones).
fn sent_pads(zk: &ZkRep) -> zkv::Pads {
    let mut sent = zkv::Pads { h: zk.pads.h.clone(), mu: vec![], pad_h: vec![], pad_mu: vec![], tau_salt: vec![] };
    if zk.pad_mismatch {
        sent.h[zk.lay.ml(0, 0)] += F128::ONE;
    }
    sent
}

/// A rep's masking challenger (M1) in front of its live stream: the pads it adds (a dishonest prover's other ones), and the
/// rank check at the opening's label (the prover stops on degenerate coins). Under the firewall the rep's frame in the Lean
/// session loop decides the refusals, and this challenger's are its cross-check (`defer`).
fn zk_masking<C: flock_transcript::challenger::Challenger>(st: &Arc<Stmt>, ch: C, zk: &ZkRep) -> (zkv::Masking<Gate<C>>, zkv::Pads) {
    let sent = sent_pads(zk);
    let mut mch = zkv::Masking::new(Gate::new(ch), &zk.lay, &sent, zk.ic.root());
    mch.defer = firewall_on(st);
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

/// Whether the firewall's send check (PROTOCOL.md §10) runs: `FC_FIREWALL=1` on a `--zk` statement.
fn firewall_on(st: &Stmt) -> bool {
    is_zk(st) && std::env::var("FC_FIREWALL").as_deref() == Ok("1")
}

/// The Lean verifier's name of the statement a firewalled session's `Hello` is set up under: `FC_FIREWALL_STATEMENT`, else
/// this build's (the circuit's statement, with `+seed-injection` in a build whose identity pins `seed_injection`).
fn lean_statement(st: &Stmt) -> String {
    std::env::var("FC_FIREWALL_STATEMENT")
        .unwrap_or_else(|_| format!("{}{}", st.c.statement(), if cfg!(feature = "seed-injection") { "+seed-injection" } else { "" }))
}

/// The verifier's own public file of `inst` (`Instances::public_file`), written where a Lean process `what` reads it.
fn lean_public(inst: &Instances, what: &str, tables: usize) -> Result<std::path::PathBuf, String> {
    let public = std::env::temp_dir().join(format!("flock-{what}-{}-{tables}.pub", std::process::id()));
    std::fs::write(&public, inst.public_file()?).map_err(|e| format!("the Lean {what}'s public file {}: {e}", public.display()))?;
    Ok(public)
}

/// The statement flags the Lean session loop sets a firewalled session's `Hello` up on (`Flock.Firewall.Hello.Setup`): the statement, the
/// circuit, the public file `public`, the lookup tables and `tables`, then `FC_FIREWALL_HELLO_ARGS`' flags.
fn lean_statement_args(st: &Stmt, public: &std::path::Path, tables: usize) -> Vec<String> {
    let mut args: Vec<String> = vec!["--statement".into(), lean_statement(st), "--circuit".into(), arg("--circuit", "circuit.txt"),
        "--public".into(), public.display().to_string(), "--tables".into(), arg("--tables", "tables"), "--session-tables".into(), tables.to_string()];
    args.extend(std::env::var("FC_FIREWALL_HELLO_ARGS").unwrap_or_default().split_whitespace().map(String::from));
    args
}

/// Where a session of `tables` tables against the auditor at `verifier` connects, and, where that is the firewall's Lean
/// session loop (`firewall::LeanSession`, one process for every such session), which then owns the auditor's connection,
/// where its reps' Lean frames connect: a firewalled statement's ([`firewall_on`]); otherwise the auditor's own address.
/// With `fw` (`--firewall-state`) the loop is `core-firewall session`, the release side around it (PROTOCOL.md §10.5).
fn session_addr(st: &Stmt, inst: &Instances, tables: usize, verifier: &str, fw: Option<&ReleaseArgs>) -> Result<(String, Option<String>), String> {
    if !firewall_on(st) {
        return Ok((verifier.to_string(), None));
    }
    static SESSIONS: Mutex<Vec<(String, usize, flock_live::firewall::LeanSession)>> = Mutex::new(Vec::new());
    let mut g = SESSIONS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((.., s)) = g.iter().find(|(v, j, _)| v == verifier && *j == tables) {
        return Ok((s.addr().to_string(), Some(s.frames().to_string())));
    }
    let public = lean_public(inst, "session", tables)?;
    let mut args = lean_statement_args(st, &public, tables);
    let exe = match fw {
        Some(fw) => {
            args.extend(["--state".into(), fw.state.display().to_string(), "--session".into(), fw.session.clone(), "--part".into(), fw.part.clone()]);
            fw.lean.clone()
        }
        None => flock_live::firewall::lean_frame_exe(),
    };
    let s = flock_live::firewall::LeanSession::start(&exe, verifier, &args);
    let _ = std::fs::remove_file(&public);
    let s = s?;
    let (addr, frames) = (s.addr().to_string(), s.frames().to_string());
    g.push((verifier.to_string(), tables, s));
    Ok((addr, Some(frames)))
}

/// Where the salt context's draws go under the firewall: with a CPU device (`two`), the device's own trees as the sequential
/// prover takes them, its level-0 commitment from the level-0 salts and every later tree from the table's salts, a rep's
/// levels ≥ 1 the Lean finish's ([`Firewall::begin_rep`] presets them); any other draw from the table's salts.
struct FirewallDraws {
    shared: Arc<Mutex<flock_live::zk_hooks::LeafSalts>>,
    level0: Arc<Vec<u8>>,
    device_trees: usize,
    two: bool,
    /// `cut-tree`: the table's device round at or after which the device's next tree draws its salts one leaf larger, the
    /// rounds handed to the firewall so far, and whether it did
    tree_cut: Option<(usize, Arc<std::sync::atomic::AtomicUsize>, Arc<std::sync::atomic::AtomicBool>)>,
}

/// `FC_FINISH_DUMP=DIR`: each firewalled table's level-0 request to `flock-firewall finish` (`table{T}.req`, `T` the session's
/// table), the root of record (`table{T}.root`) and the session's Lean finish's (`.finish_root`); and each rep's request
/// (`Flock.Firewall.Finish.parseRep`) beside the session's Lean finish's proof file of it (`table{T}-rep{R}.req`, `.lean`),
/// and where the request's passed values sit (`.vals`: their byte offset and word count): `firewall_finish_agree`'s inputs.
fn finish_dump() -> Option<std::path::PathBuf> {
    std::env::var_os("FC_FINISH_DUMP").map(Into::into)
}

/// A rep's request to `flock-firewall finish`, but the rep's draws, which [`zk_finish_fw`] adds, and the inner proof's coins,
/// which it collects.
struct RepReq {
    table: usize,
    rep: usize,
    vals: Vec<F128>,
    extras: Vec<Vec<F128>>,
    rounds: Vec<Vec<F128>>,
    dump: Option<std::path::PathBuf>,
}

impl RepReq {
    /// The request's bytes, on the rep's draws `pads` and the pads tree's salts.
    fn body(&self, pads: &zkv::Pads, salts: &[u8]) -> Vec<u8> {
        flock_live::firewall::rep_body(pads, salts, &self.vals, &self.extras, &self.rounds)
    }

    /// Under `FC_FINISH_DUMP`, writes the request `body` and where the passed values sit in it.
    fn dump(&self, body: &[u8]) {
        let Some(dir) = &self.dump else { return };
        let after = 16 * self.vals.len() + 8 + self.extras.iter().map(|x| 8 + 16 * x.len()).sum::<usize>()
            + 8 + self.rounds.iter().map(|c| 8 + 16 * c.len()).sum::<usize>();
        let vals_at = format!("{} {}\n", body.len() - after, self.vals.len());
        let stem = format!("table{}-rep{}", self.table, self.rep);
        for (ext, bytes) in [("req", body), ("vals", vals_at.as_bytes())] {
            let p = dir.join(format!("{stem}.{ext}"));
            std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("FC_FINISH_DUMP: {}: {e}", p.display()));
        }
    }

    /// Under `FC_FINISH_DUMP`, writes the session's Lean finish's proof file `lean`.
    fn dump_lean(&self, lean: &[u8]) {
        let Some(dir) = &self.dump else { return };
        let p = dir.join(format!("table{}-rep{}.lean", self.table, self.rep));
        std::fs::write(&p, lean).unwrap_or_else(|e| panic!("FC_FINISH_DUMP: {}: {e}", p.display()));
    }
}

/// The firewall's Rust side on one table (PROTOCOL.md §10, `FC_FIREWALL=1`), around the Lean session loop, whose frame of each
/// rep places, checks and masks every round that leaves: the level-0 recheck, the end check on the returned proof, and the
/// rep's worker. A failed check stops nothing before the session's `Finish` (§10.1): its reason is kept in `why`, the rep's
/// rounds from then on are the loop's places with no device round in them (the loop's frame fails each and releases the
/// statement's round in its place), and the session ends in a stop at `Finish`. `FC_FIREWALL_TAMPER` makes the device's side
/// dishonest in one named way (the negative controls): `root` (its level-0 root), `witness` (set by the caller: the device
/// proves another witness), `label` (a label of the algebraic phase), `opening` (the opening's first round's first value),
/// `opening-frame` (its second round's), `algebra` (a masked value, in the round and the proof alike: the relation check's
/// case), `algebra-proof` (the proof alone) and `shape` (a value appended to the proof's `round1_ab`, past the ones the
/// firewall compares); `later-NAME` is `NAME` on every table but a several-table session's first; `cut-MODE` a worker that
/// fails ([`Cut`]).
struct Firewall {
    st: Arc<Stmt>,
    cm: Commitment,
    root: Vec<u8>,
    draws: Arc<Mutex<FirewallDraws>>,
    tamper: String,
    why: Option<String>,
    rep: Option<FirewallRep>,
    /// the reps that ended with no check failed: the table's proofs leave only when every rep did
    ended: usize,
    /// the device rounds the firewall was handed in the table so far, and whether a `cut-` control's worker failed yet
    seen: Arc<std::sync::atomic::AtomicUsize>,
    cut_fired: Arc<std::sync::atomic::AtomicBool>,
    t_recheck: f64,
    cpu_recheck: f64,
    /// the table's index in the session, the session's Lean finish (each rep's proof file of record), and `FC_FINISH_DUMP`'s
    /// directory
    table: usize,
    finish: Arc<Mutex<flock_live::firewall::LeanFinish>>,
    dump: Option<std::path::PathBuf>,
    /// each rep's salts for levels ≥ 1, the Lean finish's, drawn with the table and taken at the rep's start
    lean_hi: [Option<Vec<Vec<u8>>>; 2],
    /// the statement's replay's constraint names, which the Lean finish's constraints are held to
    /// ([`flock_live::firewall::relations_held`])
    relations: Vec<&'static str>,
}

/// `FC_FIREWALL_TAMPER=cut-MODE`: the worker fails in one named way (§10.1's worker failures), in the rep that holds the table's
/// device round `FC_FIREWALL_CUT_AT` (default 0, counted over the table's reps), once. At that round it returns an error
/// (`cut-rc`, a nonzero return code), claims the coins degenerate (`cut-rc240`, the device's return code 240), panics
/// (`cut-panic`) or falls silent (`cut-silent`: it ticks the firewall's injected clock and asks nothing more); after its last
/// round, its output does not decode (`cut-decode`). `cut-tree` is the firewall's salt context's: the device draws its first tree
/// after level 0 at or after that round one leaf larger than the statement's.
#[derive(Clone)]
struct Cut {
    mode: String,
    /// the rep's own round index of the cut
    at: usize,
    fired: Arc<std::sync::atomic::AtomicBool>,
}

fn cut_at() -> usize {
    std::env::var("FC_FIREWALL_CUT_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

impl Cut {
    /// The worker's round `k`: a cut that fails here does (it never returns), once.
    fn round<T>(&self, k: usize, end: &flock_live::firewall::WorkerEnd<T>) {
        if k != self.at || self.mode == "cut-decode" || self.fired.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        match self.mode.as_str() {
            "cut-rc" => std::panic::panic_any(CutErr("CUDA circuit prover returned error 1".into())),
            "cut-rc240" => std::panic::panic_any(CutErr("PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128".into())),
            "cut-silent" => {
                end.tick();
                end.round(&[], 0);
                unreachable!("a worker given up gets no coins")
            }
            _ => panic!("the worker panics at its round {k}"),
        }
    }

    /// After the worker's last round (`rounds` of them): `cut-decode`'s output, which does not decode, if the cut's round was one.
    fn output(&self, rounds: usize, proof: &R1csProofLigerito, cm: &Commitment) -> Result<(), String> {
        if self.mode != "cut-decode" || rounds <= self.at || self.fired.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Ok(());
        }
        let bytes = bincode::serialize(&(cm, proof)).unwrap();
        bincode::deserialize::<(Commitment, R1csProofLigerito)>(&bytes[..bytes.len() / 2]).map(|_| ())
            .map_err(|e| format!("the device's output does not decode: {e}"))
    }
}

/// A cut's error return, raised from inside the CPU prover's round (the device returns it).
struct CutErr(String);

/// A CPU worker's result: its panic as the error it stands for (a [`CutErr`] is the device's error return).
fn cut_result<T>(r: std::thread::Result<Result<T, String>>) -> Result<T, String> {
    r.unwrap_or_else(|e| match e.downcast::<CutErr>() {
        Ok(c) => Err(c.0),
        Err(e) => std::panic::resume_unwind(e),
    })
}

struct FirewallRep {
    rep: usize,
    lay: Layout,
    /// every coin the stream answered the rep, in order (the end check's replay reads the algebraic phase's), and the
    /// extra claims' points from the hook
    coins: Vec<F128>,
    extras: Option<Vec<Vec<F128>>>,
    algebra_tampered: bool,
    /// a check failed: every later round is the loop's place with no device round in it, never the device's
    failed: bool,
    /// the rep's frame in the Lean session loop, which names each round's place and keeps the values it took
    lean: flock_live::firewall::LoopRep,
    /// the coins of each round the stream answered, for its Lean finish
    rounds: Vec<Vec<F128>>,
    /// the opening's rounds the device sent so far, counted under the opening's tamper controls (their place)
    open_rounds: usize,
}

/// A rep after the firewall's end check ([`Firewall::finish`]): the inner proof's rounds, the session's Lean finish's
/// (`finish`, on the request `req`), leave onto the rep's stream ([`zk_finish_fw`]), and its proof file is the Lean
/// finish's too. `failed`: a check failed the rep.
struct RepEnd {
    failed: bool,
    req: RepReq,
    finish: Arc<Mutex<flock_live::firewall::LeanFinish>>,
}

/// Flips the low bit of the first payload byte of `content`'s first frame of op `op`; whether there was one.
fn flip_first(content: &mut [u8], op: u8) -> bool {
    let mut at = 0usize;
    while at + 16 <= content.len() {
        let (o, len) = (content[at], u64::from_le_bytes(content[at + 8..at + 16].try_into().unwrap()) as usize);
        if o == op && len > 0 {
            content[at + 16] ^= 1;
            return true;
        }
        at += 16 + match o { OP_OBSERVE => 16 * len, OP_SQUEEZE => 0, _ => len.div_ceil(16) * 16 };
    }
    false
}

impl Firewall {
    /// Installs the table's salt context (level 0's salts, then the draws [`FirewallDraws`] routes) and takes the table's
    /// level-0 root of record, timed: the Lean session loop's (`lean_root`, its `Flock.Firewall.Level0`, which drew `l0` and
    /// the salts for the host witness `z`), which the session's Lean finish computes again from the loop's level 0 of that
    /// root ([`flock_live::firewall::LeanFinish::level0`]) and keeps for its reps' proof files. The recheck's commitment is
    /// that root's cap.
    #[allow(clippy::too_many_arguments)]
    fn new(st: &Arc<Stmt>, z: &[F128], lig: LigeritoProfile, l0: &flock_core::pcs::ligerito::zk::Level0Zk,
           salts: &Arc<Mutex<flock_live::zk_hooks::LeafSalts>>, two: bool, table: usize,
           finish: &Arc<Mutex<flock_live::firewall::LeanFinish>>, lean_root: Option<Vec<u8>>) -> Result<Firewall, String> {
        use flock_live::firewall::refused;
        let pcs = params(st.m, lig);
        let level0 = salts.lock().unwrap_or_else(PoisonError::into_inner).level0(pcs.n_leaves());
        let tamper = std::env::var("FC_FIREWALL_TAMPER").unwrap_or_default();
        let (seen, cut_fired) = (Arc::new(std::sync::atomic::AtomicUsize::new(0)), Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let tree_cut = (tamper == "cut-tree").then(|| (cut_at(), seen.clone(), cut_fired.clone()));
        let draws = Arc::new(Mutex::new(FirewallDraws { shared: salts.clone(), level0: level0.clone(), device_trees: 0, two, tree_cut }));
        let d = draws.clone();
        flock_merkle::hm96::begin(level0.clone(), Box::new(move |n| {
            use std::sync::atomic::Ordering::SeqCst;
            let mut g = d.lock().unwrap_or_else(PoisonError::into_inner);
            let d = &mut *g;
            d.device_trees += 1;
            if d.two && d.device_trees == 1 {
                return d.level0.to_vec();
            }
            let cut = d.two && d.tree_cut.as_ref().is_some_and(|(k, seen, fired)| seen.load(SeqCst) >= *k && !fired.swap(true, SeqCst));
            let mut s = d.shared.lock().unwrap_or_else(PoisonError::into_inner);
            let id = s.next_id();
            if cut {
                let mut b = s.expand(id, n + 1);
                b.truncate(n * flock_live::zk_hooks::SALT_BYTES);
                return b;
            }
            s.expand(id, n)
        }));
        let body = flock_live::firewall::link_body(z, &l0.extra, &l0.pads, &level0);
        let (t0, c0) = (Instant::now(), flock_live::firewall::cpu_seconds());
        let root = lean_root.ok_or_else(|| refused("level-0 recheck: the table has no level 0 of the Lean session loop's"))?;
        let fin_root = finish.lock().unwrap_or_else(PoisonError::into_inner).level0(table, &root)?;
        let (t_recheck, cpu_recheck) = (t0.elapsed().as_secs_f64(), flock_live::firewall::cpu_seconds() - c0);
        let short = |r: &[u8]| hex(&r[..r.len().min(8)]);
        eprintln!("FIREWALL\t{}", json!({"recheck_root": hex(&root), "m": st.m, "t.firewall_recheck": t_recheck, "cpu.firewall_recheck": cpu_recheck}));
        if fin_root != root {
            return Err(refused(format!("level-0 recheck: the Lean finish's root {} is not the loop's {}", short(&fin_root), short(&root))));
        }
        let w = std::mem::size_of::<flock_hash::Digest>();
        if root.is_empty() || root.len() % w != 0 {
            return Err(refused(format!("level-0 recheck: the Lean root's {} bytes are not whole {w}-byte digests", root.len())));
        }
        let cm = Commitment { cap: root.chunks(w).map(merkle_digest_read).collect(), params: pcs };
        let lean_hi = {
            let mut f = finish.lock().unwrap_or_else(PoisonError::into_inner);
            [Some(f.salts(table, 0)?), Some(f.salts(table, 1)?)]
        };
        let dump = finish_dump();
        if let Some(dir) = &dump {
            for (ext, bytes) in [("req", body.as_slice()), ("root", root.as_slice()), ("finish_root", fin_root.as_slice())] {
                let p = dir.join(format!("table{table}.{ext}"));
                std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("FC_FINISH_DUMP: {}: {e}", p.display()));
            }
        }
        Ok(Firewall { st: st.clone(), cm, root, draws, tamper, why: None, rep: None, ended: 0, seen, cut_fired, t_recheck, cpu_recheck,
                      table, finish: finish.clone(), dump, lean_hi, relations: Vec::new() })
    }

    /// The rep's `cut-` control (the worker's failure), if its round is this rep's or later and it has not fired yet.
    fn cut(&self) -> Option<Cut> {
        let (seen, at) = (self.seen.load(std::sync::atomic::Ordering::SeqCst), cut_at());
        (self.tamper.starts_with("cut-") && self.tamper != "cut-tree" && !self.cut_fired.load(std::sync::atomic::Ordering::SeqCst) && at >= seen)
            .then(|| Cut { mode: self.tamper.clone(), at: at - seen, fired: self.cut_fired.clone() })
    }

    /// Whether the rep's `Commit` has been made (the hook took its root).
    fn committed(&self) -> bool {
        self.rep.as_ref().is_some_and(|r| r.extras.is_some())
    }

    /// The table's stop at `Finish`, if any: a failed check, or a rep that never ended. A table's proofs leave only after
    /// both reps ended with no check failed.
    fn verdict(&self) -> Option<String> {
        self.why.clone().or_else(|| (self.ended < 2).then(|| flock_live::firewall::refused(
            format!("the firewall ended {} of the table's 2 reps", self.ended))))
    }

    fn refuse(&mut self, why: String) -> String {
        self.why.get_or_insert_with(|| why.clone());
        why
    }

    /// A rep starts (after its pads tree took its salt id): the device's trees for levels ≥ 1 take the Lean finish's salts
    /// from here.
    fn begin_rep(&mut self, rep: usize, zk: &ZkRep) -> Result<(), String> {
        let hi = self.lean_hi.get_mut(rep).and_then(Option::take).ok_or_else(|| flock_live::firewall::refused(
            format!("table {} rep {rep}: no Lean salts for levels ≥ 1 (a table's reps are 0 and 1, each begun once)", self.table)))?;
        {
            let mut d = self.draws.lock().unwrap_or_else(PoisonError::into_inner);
            d.shared.lock().unwrap_or_else(PoisonError::into_inner).preset(hi)
                .map_err(|e| flock_live::firewall::refused(format!("table {} rep {rep}: the Lean finish's salts: {e}", self.table)))?;
            // rep 0's level-0 tree is the salt context's first, which hm96 salts with level 0's salts without a draw
            d.device_trees = usize::from(rep == 0);
        }
        let lean = zk.lean.lock().unwrap_or_else(PoisonError::into_inner).take()
            .unwrap_or_else(|| Err(flock_live::firewall::refused("a firewalled rep without its Lean frame")))?;
        self.rep = Some(FirewallRep { rep, lay: zk.lay, coins: Vec::new(), extras: None, algebra_tampered: false, failed: false, lean,
                                  rounds: Vec::new(), open_rounds: 0 });
        Ok(())
    }

    /// A check failed: its reason is kept for `Finish`, and the rep releases the statement's schedule from here on.
    fn fail(&mut self, why: String) {
        self.refuse(why);
        if let Some(r) = self.rep.as_mut() {
            r.failed = true;
        }
    }

    /// The device's level-0 root, at the session's `Commit`: the root that leaves, the recheck's (a device root that differs
    /// fails the rep).
    fn root(&mut self, root: &[u8]) -> Vec<u8> {
        let mut got = root.to_vec();
        if self.tamper == "root" {
            got[0] ^= 1;
        }
        if got != self.root {
            self.fail(flock_live::firewall::refused(format!("level-0 root: the device committed {} where the recheck has {}",
                                                            hex(&got[..8]), hex(&self.root[..8]))));
        }
        self.root.clone()
    }

    /// The extra claims' points the hook hands the opening (their multilinear part, as the device takes them).
    fn extras(&mut self, e: &[Vec<F128>]) {
        if let Some(r) = self.rep.as_mut() {
            r.extras = Some(e.iter().map(|p| p[6..].to_vec()).collect());
        }
    }

    /// One device round before anything of it is sent: the round that leaves (the device's, which the rep's frame in the
    /// Lean session loop checks against the statement's round in its place, and masks) and its coin count; or nothing, once the loop names no place of the statement's left. A rep a check here
    /// failed sends the loop's place with no device round in it ([`Firewall::scheduled`]). An error is the firewall's own fault.
    fn round(&mut self, content: &[u8], n: usize) -> Result<Option<(Vec<u8>, usize)>, String> {
        self.seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let r = self.rep.as_ref().ok_or_else(|| flock_live::firewall::refused("a round outside a firewalled rep"))?;
        if r.failed {
            return self.scheduled();
        }
        Ok(Some((self.tampered(content)?, n)))
    }

    /// The device round `content` as a tamper control (`label`, `algebra`, `opening`, `opening-frame`) sends it, at the place
    /// the Lean session loop names; any other round as it is.
    fn tampered(&mut self, content: &[u8]) -> Result<Vec<u8>, String> {
        use flock_live::firewall::Place;
        let mut c = content.to_vec();
        let t = self.tamper.clone();
        if !matches!(t.as_str(), "label" | "algebra" | "opening" | "opening-frame") {
            return Ok(c);
        }
        let r = self.rep.as_mut().unwrap();
        match r.lean.place()? {
            Place::Alg(_) if (t == "label" || t == "algebra") && !r.algebra_tampered => {
                r.algebra_tampered = flip_first(&mut c, if t == "label" { OP_LABEL } else { OP_OBSERVE });
            }
            Place::Open(_) => {
                if (t == "opening" && r.open_rounds == 0) || (t == "opening-frame" && r.open_rounds == 1) {
                    flip_first(&mut c, OP_OBSERVE);
                }
                r.open_rounds += 1;
            }
            _ => {}
        }
        Ok(c)
    }

    /// A failed rep's next round: the round of the place the Lean session loop names, sent as no device round (its squeeze
    /// alone, after the stream's domain frame at its first), which the loop's frame fails, releasing the statement's round
    /// in its place; nothing once the loop names no place of the statement's.
    fn scheduled(&mut self) -> Result<Option<(Vec<u8>, usize)>, String> {
        use flock_live::firewall::Place;
        let r = self.rep.as_ref().unwrap();
        let (Place::Alg(k) | Place::Open(k)) = r.lean.place()? else { return Ok(None) };
        let mut c = if r.rounds.is_empty() { frame_domain(&domain(&self.st.c, r.rep)) } else { Vec::new() };
        flock_live::push_header(&mut c, OP_SQUEEZE, KIND_SLICE, k as u64);
        Ok(Some((c, k)))
    }

    /// The device returned: a rep whose places the Lean session loop still names fails, and the firewall sends the rest of
    /// them itself, a round a call, until there is none.
    fn rest(&mut self) -> Result<Option<(Vec<u8>, usize)>, String> {
        use flock_live::firewall::{refused, Place};
        let r = self.rep.as_mut().ok_or_else(|| refused("a proof outside a firewalled rep"))?;
        if !r.failed {
            match r.lean.place()? {
                Place::Own => return Ok(None),
                Place::Alg(_) => self.fail(refused("the device returned before its opening")),
                Place::Open(_) => self.fail(refused("the device returned before the firewall's opening ended")),
            }
        }
        self.scheduled()
    }

    /// The coins the stream answered the round the firewall released, for the end check's replay and the Lean finish.
    fn answer(&mut self, coins: &[F128]) {
        if let Some(r) = self.rep.as_mut() {
            r.rounds.push(coins.to_vec());
            r.coins.extend_from_slice(coins);
        }
    }

    /// The rep's end check (after [`Firewall::rest`]): the device's proof if it returned one (`device`), whose every exposed
    /// value must be the one the rep's frame in the Lean session loop took (`final_c'` the one the firewall computes from
    /// them) and its commitment the recheck's. The rep's proof is the Lean finish's whatever these find, or whether the
    /// device returned at all ([`zk_finish_fw`]), on the values the frame took and every round's coins. A failed check fails
    /// the rep, for `Finish`. Returns the rep's firewall counts, with the rep's request to the Lean finish, for the inner
    /// proof; an error is the firewall's own fault.
    fn finish(&mut self, mut device: Option<(&mut R1csProofLigerito, &Commitment)>) -> Result<(Value, RepEnd), String> {
        use flock_live::firewall::refused;
        let st = self.st.clone();
        if let Some((proof, _)) = device.as_mut() {
            if self.tamper == "algebra" || self.tamper == "algebra-proof" {
                if let Some(v) = proof.zerocheck.round1_ab.first_mut() {
                    v.lo ^= 1;
                }
            }
            if self.tamper == "shape" {
                if let Some(&v) = proof.zerocheck.round1_ab.first() {
                    proof.zerocheck.round1_ab.push(v);
                }
            }
        }
        let r = self.rep.as_mut().ok_or_else(|| refused("a proof outside a firewalled rep"))?;
        let mut why: Option<String> = None;
        let mut check = |ok: bool, what: &dyn Fn() -> String| {
            if !ok && why.is_none() {
                why = Some(refused(what()));
            }
        };
        if let Some((proof, _)) = device.as_ref() {
            check(proof.zerocheck.grinding_nonces.is_empty() && proof.lincheck.grinding_nonces.is_empty() && proof.lincheck.matrix_evals.is_empty(),
                  &|| "the proof carries grinding nonces or matrix evaluations the transcript did not".into());
        }
        let lay = r.lay;
        let took = r.lean.vals()?;
        if took.len() + 1 != lay.piop_exposed() {
            return Err(refused(format!("the Lean session loop's frame took {} of the algebraic phase's {} values", took.len(), lay.piop_exposed() - 1)));
        }
        let mut vals = vec![F128::ZERO; lay.piop_exposed()];
        for (&i, v) in lay.absorbed().iter().zip(took) {
            vals[i] = v;
        }
        let (zc0, lc0) = zkv::from_exposed(&vals, &lay);
        let rp = catch_unwind(AssertUnwindSafe(|| {
            zkv::replay(&st.r1cs, &zc0, &lc0, &st.circuit, &lay, &[0; 64], Some(&[F128::ZERO; zkv::SKIP]), &mut zkv::Tape::new(&r.coins))
        })).map_err(|e| refused(format!("the replay of the passed values: {}", panic_msg(e))))?.map_err(refused)?;
        vals[lay.fc()] = rp.cons.iter().find(|c| c.0 == "zerocheck c_eval").ok_or_else(|| refused("no c_eval row"))?.1.c;
        let relations: Vec<&'static str> = rp.cons.iter().map(|c| c.0).collect();
        if let Some((proof, cm)) = device.as_ref() {
            let (zc, lc) = (&proof.zerocheck, &proof.lincheck);
            let shape = [zc.round1_ab.len(), zc.round1_c.len(), zc.multilinear_rounds.len(), lc.rounds.len(), lc.z_partial.len()];
            let want = [zkv::SKIP, zkv::SKIP, lay.n_mlv, lay.n_lc, zkv::SKIP];
            check(shape == want, &|| format!("the proof's shape (round1_ab, round1_c, multilinear_rounds, lincheck rounds, z_partial) is {shape:?} where the statement's is {want:?}"));
            if shape == want {
                let got = zkv::exposed_values(&proof.zerocheck, &proof.lincheck, &lay);
                let i = (0..vals.len()).find(|&i| got[i] != vals[i]);
                check(i.is_none(), &|| if i == Some(lay.fc()) { "the proof's final_c is not the one the firewall computes from the passed values".to_string() }
                                      else { format!("the proof's exposed value {} is not the one the firewall passed", i.unwrap_or(0)) });
            }
            check(bincode::serialize(*cm).unwrap() == bincode::serialize(&self.cm).unwrap(), &|| "the proof's commitment is not the recheck's".into());
        }
        self.relations = relations;
        if let Some(e) = why {
            self.fail(e);
        }
        let r = self.rep.as_mut().unwrap();
        let req = RepReq {
            table: self.table, rep: r.rep, vals: vals.clone(), extras: r.extras.clone().unwrap_or_default(),
            rounds: std::mem::take(&mut r.rounds), dump: self.dump.clone(),
        };
        // firewall_device_rounds: the table's device rounds the firewall was handed so far (FC_FIREWALL_CUT_AT counts them)
        let mut v = json!({"firewall_failed": r.failed,
                           "firewall_device_rounds": self.seen.load(std::sync::atomic::Ordering::SeqCst)});
        if r.rep == 0 {
            v["t.firewall_recheck"] = json!(self.t_recheck);
            v["cpu.firewall_recheck"] = json!(self.cpu_recheck);
        }
        let end = RepEnd { failed: r.failed, req, finish: self.finish.clone() };
        Ok((v, end))
    }

    /// The rep's end, after the inner proof's rounds left: the table's proofs leave only after both reps ended with no
    /// check failed. Only the session's verdict changes here, as the next rep may have begun.
    fn rep_end(&mut self, failed: bool) {
        if !failed {
            self.ended += 1;
        }
    }
}

impl Drop for Firewall {
    fn drop(&mut self) {
        flock_merkle::hm96::end();
    }
}

/// One round of a firewalled rep's device (or of the rep's device without a firewall): the firewall's turn, then the masking
/// challenger's replay of the round that leaves onto the stream, and the coins back to the firewall. A failed rep's device
/// gets its coin count back, the place's coins cut or padded with zeros: nothing it does reaches the stream.
fn firewalled_round<C: flock_transcript::challenger::Challenger>(firewall: &Option<Arc<Mutex<Firewall>>>, mch: &mut zkv::Masking<Gate<C>>, c: &[u8],
                                                                 n: usize) -> Vec<F128> {
    let Some(g) = firewall else { return replay_round(mch, c, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}")) };
    let next = g.lock().unwrap_or_else(PoisonError::into_inner).round(c, n).unwrap_or_else(|why| panic!("{why}"));
    let Some(mut coins) = next.map(|x| released(g, mch, x)) else { return vec![F128::ZERO; n] };
    coins.resize(n, F128::ZERO);
    coins
}

/// A round that leaves (the device's unmasked round, which the rep's frame in the Lean session loop masks, and its coin
/// count) onto the stream through the masking challenger: its coins, which the firewall gets too. The masking challenger's
/// deferred refusal of them (the loop's frame decides the rep's refusals) stops the proof here.
fn released<C: flock_transcript::challenger::Challenger>(g: &Arc<Mutex<Firewall>>, mch: &mut zkv::Masking<Gate<C>>,
                                                         (c, k): (Vec<u8>, usize)) -> Vec<F128> {
    mch.inner.release = Some(c.clone());
    let coins = replay_round(mch, &c, k).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"));
    g.lock().unwrap_or_else(PoisonError::into_inner).answer(&coins);
    if let Some(why) = mch.stopped.take() {
        panic!("{}", flock_live::firewall::refused(format!("Rust's masking refused the rep ({why})")));
    }
    coins
}

/// The firewall's end of a rep, whatever its worker did (PROTOCOL.md §10.1). A worker that failed (`dev`'s error: an error it
/// returned, output that does not decode, a panic, silence past the deadline) fails the rep, and one that never reached its
/// `Commit` has the hook taken here with the recheck's root. Then the rest of the places the Lean session loop names, if any
/// are left (onto the rep's stream `mch`), and the end check ([`Firewall::finish`]). Returns the rep's buckets and its end,
/// for the inner proof ([`zk_finish_fw`]); its error is the firewall's own fault.
fn firewalled_end<C: flock_transcript::challenger::Challenger>(g: &Arc<Mutex<Firewall>>, mch: &mut zkv::Masking<Gate<C>>, hook: Hook<'_>,
                                                               dev: Result<(R1csProofLigerito, Commitment, Value), String>)
                                                               -> Result<(Value, RepEnd), String> {
    let lock = || g.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut dev, mut bk) = match dev {
        Ok((proof, cm, bk)) => (Some((proof, cm)), bk),
        Err(why) => {
            lock().fail(flock_live::firewall::refused(format!("the device failed: {why}")));
            (None, json!({"firewall_device_failed": why}))
        }
    };
    if !lock().committed() {
        let root = lock().root.clone();
        hook(&root);
    }
    loop {
        let next = lock().rest()?;
        let Some(x) = next else { break };
        released(g, mch, x);
    }
    let (v, end) = lock().finish(dev.as_mut().map(|(p, c)| (p, &*c)))?;
    if let (Some(o), Some(v)) = (bk.as_object_mut(), v.as_object()) {
        o.extend(v.clone());
    }
    Ok((bk, end))
}

/// How long the firewall waits for a worker's next ask before it gives the worker up: `FC_FIREWALL_DEADLINE_S` seconds (default
/// an hour).
fn worker_wait<T>() -> impl FnMut(&std::sync::mpsc::Receiver<flock_live::firewall::Ask<T>>) -> Option<flock_live::firewall::Ask<T>> {
    flock_live::firewall::deadline(std::env::var("FC_FIREWALL_DEADLINE_S").ok().and_then(|v| v.parse().ok()).unwrap_or(3600.0))
}

/// A rep's worker under the firewall: `work` (the device prover, given the worker's end) on a thread of its own, its rounds
/// counted for the rep's `cut-` control, which fails it at its round and after its output.
fn worker<T: Send + 'static>(cut: Option<Cut>, work: impl FnOnce(&mut (dyn FnMut(&[u8], usize) -> Vec<F128> + Send), &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>)
                                                                 -> Result<(R1csProofLigerito, Commitment, T), String> + Send + 'static,
                             round: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, hook: Hook<'_>)
                             -> Result<(R1csProofLigerito, Commitment, T), String> {
    flock_live::firewall::run_worker(move |end| {
        let mut k = 0;
        let out = {
            let mut rounds = |c: &[u8], n: usize| {
                if let Some(x) = &cut {
                    x.round(k, &end);
                }
                k += 1;
                end.round(c, n)
            };
            let mut h = |root: &[u8]| end.hook(root);
            cut_result(catch_unwind(AssertUnwindSafe(|| work(&mut rounds, &mut h))))?
        };
        if let Some(x) = &cut {
            x.output(k, &out.0, &out.1)?;
        }
        Ok(out)
    }, round, hook, &mut worker_wait())
}

/// M1's CPU prover: upstream's prover under the masking challenger (or, with `FC_ZK_FRAMED`, under a [`Framer`] whose rounds are
/// replayed into it, as the device's are), then [`zk_finish`], or [`zk_finish_fw`] under the firewall.
#[allow(clippy::too_many_arguments)]
fn prove_cpu_zk(st: &Arc<Stmt>, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger, zk: &ZkRep,
                firewall: &Option<Arc<Mutex<Firewall>>>) -> Result<(Vec<u8>, Value), String> {
    #[allow(irrefutable_let_patterns)]
    let WitData::Host((z, a, b, zl)) = wt.clone() else { panic!("M1's CPU prover takes a host witness") };
    let tp = Instant::now();
    let (mut mch, sent) = zk_masking(st, ByRef(ch), zk);
    if let Some(g) = firewall {
        // the device on its own thread (the firewall serves its rounds and its Commit here), whatever it does
        let cut = g.lock().unwrap_or_else(PoisonError::into_inner).cut();
        let (st2, level0, dom) = (st.clone(), zk.level0.clone(), domain(&st.c, rep));
        let dev = worker(cut, move |rounds, hook| {
            let mut hk = |cm: &Commitment| -> Vec<Vec<F128>> { hook(merkle_digest_bytes(&cm.cap)).into_iter().map(|p| p[6..].to_vec()).collect() };
            let mut fr = Framer { pending: frame_domain(&dom), rounds };
            let (proof, cm) = prove_fast_ligerito_from_witness_extra_zk(&st2.r1cs, &pcs_of(&st2, lig), z, a, b, zl, &st2.circuit, &mut hk,
                                                                        Some(&level0), &mut fr);
            Ok((proof, cm, json!({})))
        }, &mut |c: &[u8], n: usize| firewalled_round(firewall, &mut mch, c, n), hook);
        let t_upstream = tp.elapsed().as_secs_f64();
        let (bk, end) = firewalled_end(g, &mut mch, hook, dev)?;
        return Ok(zk_finish_fw(zk, mch, tp, t_upstream, bk, g, end));
    }
    let mut hk = |cm: &Commitment| -> Vec<Vec<F128>> { hook(merkle_digest_bytes(&cm.cap)).into_iter().map(|p| p[6..].to_vec()).collect() };
    let (proof, commitment) = if std::env::var("FC_ZK_FRAMED").is_ok() {
        let mut rounds = |content: &[u8], n: usize| firewalled_round(firewall, &mut mch, content, n);
        let mut fr = Framer { pending: frame_domain(&domain(&st.c, rep)), rounds: &mut rounds };
        prove_fast_ligerito_from_witness_extra_zk(&st.r1cs, &pcs_of(st, lig), z, a, b, zl, &st.circuit, &mut hk, Some(&zk.level0), &mut fr)
    } else {
        prove_fast_ligerito_from_witness_extra_zk(&st.r1cs, &pcs_of(st, lig), z, a, b, zl, &st.circuit, &mut hk, Some(&zk.level0), &mut mch)
    };
    let t_upstream = tp.elapsed().as_secs_f64();
    Ok(zk_finish(st, zk, proof, commitment, mch, &sent, tp, t_upstream, json!({})))
}

/// After upstream's prover (the CPU's, or the device's with the same unmasked proof, level-0 record and folded comb): the
/// masked proof, the constraints replayed from the logged coins, and the inner proof on the rep's stream after the opening.
#[allow(clippy::too_many_arguments)]
fn zk_finish<C: flock_transcript::challenger::Challenger + Send>(st: &Stmt, zk: &ZkRep, mut proof: R1csProofLigerito, commitment: Commitment,
                                                                  mch: zkv::Masking<Gate<C>>, sent: &zkv::Pads, tp: Instant, t_upstream: f64,
                                                                  mut bk: Value) -> (Vec<u8>, Value) {
    let (mut inner_ch, coins, _coins_l0) = mch.finish();
    let tz = Instant::now();
    let (mut zc, lc) = zkv::mask_proof(&proof.zerocheck, &proof.lincheck, &zk.lay, &sent.h);
    let own_comb = (!proof.lincheck.comb_partial.is_empty()).then_some(proof.lincheck.comb_partial.as_slice());
    let rp = zkv::replay(&st.r1cs, &zc, &lc, &st.circuit, &zk.lay, &zk.ic.root(), own_comb, &mut zkv::Tape::new(&coins))
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
    let kept = zk.final_c_solved.then(|| cons.clone());
    let mut lch = zkv::Logging { inner: &mut inner_ch, coins: Vec::new() };
    let inner = zkv::prove_inner(&zk.ic, &zk.pads, &zk.lay, zc.final_c_eval, cons, zk.self_check, &mut lch).unwrap_or_else(|e| panic!("{e}"));
    let inner_coins = lch.coins;
    if let Some(cons) = kept {
        zc.final_c_eval += final_c_shift(&zk.lay, &zk.pads.h, cons, &inner, &inner_coins);
    }
    let t_zk = tz.elapsed().as_secs_f64();
    let masked = R1csProofLigerito { zerocheck: zc, lincheck: lc, pcs_open: proof.pcs_open };
    let o = bk.as_object_mut().expect("buckets");
    for (k, v) in [("t.prove", json!(tp.elapsed().as_secs_f64())), ("t.upstream", json!(t_upstream)), ("t.zk_inner", json!(t_zk)),
                   ("t.zk_replay", json!(t_replay)), ("t.zk_pads_commit", json!(zk.t_commit)), ("constraints_failing_on_pads", json!(fails)),
                   ("zk_coins", json!(f128s_hex(&coins))), ("zk_inner_coins", json!(f128s_hex(&inner_coins)))] {
        o.insert(k.into(), v);
    }
    (bincode::serialize(&(commitment, masked, inner)).unwrap(), bk)
}

/// A constraint's name as the replay's rows carry it: each distinct name the Lean finish sends is leaked once.
fn interned(name: String) -> &'static str {
    static NAMES: Mutex<std::collections::BTreeSet<&'static str>> = Mutex::new(std::collections::BTreeSet::new());
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(n) = names.get(name.as_str()) {
        return n;
    }
    let n: &'static str = Box::leak(name.into_boxed_str());
    names.insert(n);
    n
}

/// A firewalled rep after [`firewalled_end`] (the device's or the CPU's, whatever it did): its inner proof is the session's
/// Lean finish's (`Flock.Firewall.Finish.innerLive`), on the rep's request (`end`'s, with the rep's draws and the coins of its
/// PIOP and opening). The finish sends the constraints it replays, which must be the statement's relations name for name
/// ([`flock_live::firewall::relations_held`]; any other is the firewall's fault), then each round of its inner proof, which
/// leaves onto the rep's stream, where the rep's frame in the Lean session loop decides its stop on `β = 0` (the Lean
/// finish's refusal where the frame went on is the firewall's fault), and then the rep ends ([`Firewall::rep_end`]). It
/// speaks on schedule, as M1's simulator's does: the session stops at `Finish` when a constraint fails on the pads
/// (`constraints_failing_on_pads`). The rep's proof file is the Lean finish's on every round's coins (`rep`), and only when
/// no check failed the rep and no constraint fails on its pads; else none leaves (an empty file, which the session never
/// sends: it stops at its `Finish`).
fn zk_finish_fw<C: flock_transcript::challenger::Challenger + Send>(zk: &ZkRep, mch: zkv::Masking<Gate<C>>, tp: Instant, t_upstream: f64,
                                                                     mut bk: Value, g: &Arc<Mutex<Firewall>>, end: RepEnd) -> (Vec<u8>, Value) {
    use flock_live::firewall::refused;
    let (mut inner_ch, coins, _coins_l0) = if end.failed { mch.finish_failed() } else { mch.finish() };
    let tz = Instant::now();
    let RepEnd { failed, mut req, finish: fin } = end;
    let relations = g.lock().unwrap_or_else(PoisonError::into_inner).relations.clone();
    let body = req.body(&zk.pads, &zk.ic.salts);
    let (mut fails, mut inner_rounds) = (Vec::new(), Vec::<Vec<F128>>::new());
    let inner = fin.lock().unwrap_or_else(PoisonError::into_inner).inner(req.table, req.rep, &body, &mut |c: &[u8], n: usize| {
        inner_ch.release = Some(c.to_vec());
        let coins = replay_round(&mut inner_ch, c, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"));
        inner_rounds.push(coins.clone());
        Some(coins)
    }, &mut |b: &[u8]| {
        let cons = flock_live::firewall::parse_cons(b)?;
        flock_live::firewall::relations_held(&cons, &relations, zk.lay.ligerito())?;
        let cons: Vec<(&'static str, zkv::Lin)> = cons.into_iter().map(|(n, l)| (interned(n), l)).collect();
        fails = zkv::failing(&cons, &zk.pads.h);
        Ok(())
    });
    if let Err(e) = inner {
        panic!("{}", refused(format!("the Lean finish's inner proof refused the rep ({e}) where the Lean frame went on")));
    }
    let t_zk = tz.elapsed().as_secs_f64();
    let leaves = !failed && fails.is_empty();
    g.lock().unwrap_or_else(PoisonError::into_inner).rep_end(failed);
    let inner_coins = inner_rounds.concat();
    req.rounds.extend(inner_rounds);
    let o = bk.as_object_mut().expect("buckets");
    for (k, v) in [("t.prove", json!(tp.elapsed().as_secs_f64())), ("t.upstream", json!(t_upstream)), ("t.zk_inner", json!(t_zk)),
                   ("t.zk_pads_commit", json!(zk.t_commit)), ("constraints_failing_on_pads", json!(fails)), ("zk_coins", json!(f128s_hex(&coins))),
                   ("zk_inner_coins", json!(f128s_hex(&inner_coins)))] {
        o.insert(k.into(), v);
    }
    let body = req.body(&zk.pads, &zk.ic.salts);
    req.dump(&body);
    if !leaves {
        // the session stops at `Finish`: no proof file of the rep leaves
        return (Vec::new(), bk);
    }
    let t = Instant::now();
    let tamper = std::env::var("FC_FINISH_TAMPER").unwrap_or_default();
    let lean = {
        let mut f = fin.lock().unwrap_or_else(PoisonError::into_inner);
        let last = FINISH_SENT.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 == FINISH_REPS.load(std::sync::atomic::Ordering::SeqCst);
        let body = if !tamper.is_empty() && last { finish_tampered(&req, &zk.pads, &zk.ic.salts, &zk.lay, &tamper) } else { body };
        f.rep(req.table, req.rep, &body)
    };
    let lean = match lean {
        Ok(l) => l,
        Err(e) if !tamper.is_empty() => {
            *FINISH_REFUSED.lock().unwrap_or_else(PoisonError::into_inner) = Some(e);
            return (Vec::new(), bk);
        }
        Err(e) => panic!("{e}"),
    };
    req.dump_lean(&lean);
    bk.as_object_mut().expect("buckets").insert("t.lean_finish".into(), json!(t.elapsed().as_secs_f64()));
    (lean, bk)
}

/// `FC_FINISH_TAMPER=MODE` (the selftest's `zk_worker_finish`): the session's last rep request to the Lean finish in time
/// (the [`FINISH_REPS`]-th, counted under the finish's lock, so every other rep's requests went before it) doctored in one
/// named way: `pads-zero` (the pads `h` all zero), `vals-chosen` (the first passed value plus one), `coins` (the first
/// round's first coin plus one) or `final-c` (`final_c` plus one). The loop's finish refuses each (`Flock.Firewall.Finish.RepIn.differs`, and
/// for `final_c`, which no live inner proof answers, its live-only rule: a rep's file is only the one its `inner` request
/// computed), and the refusal goes to [`FINISH_REFUSED`].
static FINISH_REPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FINISH_SENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FINISH_REFUSED: Mutex<Option<String>> = Mutex::new(None);

/// The rep request `req` on the draws `pads` and the pads tree's `salts`, doctored per `FC_FINISH_TAMPER`'s `mode`.
fn finish_tampered(req: &RepReq, pads: &zkv::Pads, salts: &[u8], lay: &Layout, mode: &str) -> Vec<u8> {
    let mut p = zkv::Pads { h: pads.h.clone(), mu: pads.mu.clone(), pad_h: pads.pad_h.clone(), pad_mu: pads.pad_mu.clone(),
                            tau_salt: pads.tau_salt.clone() };
    let (mut vals, mut rounds, one) = (req.vals.clone(), req.rounds.clone(), F128::new(1, 0));
    match mode {
        "pads-zero" => p.h.iter_mut().for_each(|h| *h = F128::ZERO),
        "vals-chosen" => vals[0] = vals[0] + one,
        "coins" => rounds[0][0] = rounds[0][0] + one,
        "final-c" => vals[lay.fc()] = vals[lay.fc()] + one,
        m => panic!("FC_FINISH_TAMPER={m}: pads-zero, vals-chosen, coins or final-c"),
    }
    flock_live::firewall::rep_body(&p, salts, &vals, &req.extras, &rounds)
}

/// The shift of `final_c'` that zeroes the inner proof's batched constraint on the committed pads, from its coins (`ε`, then
/// the batching coins): `final_c'` is a constant of the c_eval and ring switch c rows.
fn final_c_shift(lay: &Layout, h: &[F128], mut cons: Vec<(&'static str, zkv::Lin)>, inner: &InnerProof, coins: &[F128]) -> F128 {
    cons.extend(zkv::triple_constraints(lay, coins[0], inner.rho, inner.sigma));
    let g = &coins[1..1 + cons.len()];
    let (c, t) = zkv::batch(&cons, g);
    let sum = c.iter().zip(h).fold(t, |a, (x, y)| a + *x * *y);
    let at = |name: &str| cons.iter().position(|r| r.0 == name).expect("a row of final_c'");
    sum * (g[at("zerocheck c_eval")] + g[at("ring switch c")]).inv()
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

/// M1's simulator for one rep's whole proof, from the public statement (`inst`'s public part: roots and `b || c`), the
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
    session_relations_q(st, inst, proofs, coins, points, pairs).map(|(r, _)| r)
}

/// `session_relations`, with the level-1 positions each rep opened and the number both opened.
fn session_relations_q(st: &Stmt, inst: &Instances, proofs: &[Vec<u8>], coins: &[Vec<F128>], points: &[Vec<F128>], pairs: [usize; 2])
    -> Result<([Vec<F128>; 2], [usize; 3]), String> {
    use flock_core::pcs::ligerito::zk::{lane_codeword, pad_codeword};
    use flock_field::F256;
    let lc = pcs_of(st, LigeritoProfile::Fast100).ligerito_prover_config()?;
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
    Ok(([level0, level1], [q1[0].len(), q1[1].len(), common.len()]))
}

/// `zkstat --zk --n N`: N real sessions (fresh prover seeds) against one verifier coin seed, N simulated transcripts from those
/// coins, and N control sessions (zero pads, zero mask slot: M0's values in M1's format), each rep's checked fields printed as
/// `ZKREAL`, `ZKSIM`, `ZKCONTROL` lines, and each session's cross-rep functionals ([`session_relations`]) as `ZKREL` lines, for
/// `verity/ml/flock/tests/zk_stats.py`. Needs the `seed-injection` harness.
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
            assert!(verified(&v) || kind == "dummy", "{kind} session {run} rejected: {}", v["why"]);
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

/// The classes of the inner proof's observes, in the order it absorbs them (`zk_veil::prove_inner`): `final_c'`, `[ρ, σ]`,
/// then `Y`.
const INNER_CLASSES: [&str; 3] = ["masked: final_c' (its pad h_fc)", "masked: inner rho, sigma (a fresh triple)", "masked: inner Y (the dot-product mask)"];

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
                        add(INNER_CLASSES[(*k as usize).min(3) - 1], n)
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
    assert!(verified(&v) && verified(&vc), "the audit's sessions must be accepted");
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
        let nonces_zero = pr.pcs_open.batching_nonces.iter().all(|x| *x == 0);
        add(if nonces_zero { "public (proof): the opening's batching nonces (live coins: always 0)" } else { "CLEAR (proof): nonzero batching nonces" },
            8 * pr.pcs_open.batching_nonces.len());
        add("public (proof): the commitment's parameters", bincode::serialized_size(&cm.params).unwrap() as usize);
    }
    let clear: Vec<Value> = total.iter().filter(|(k, _)| k.starts_with("CLEAR")).map(|(k, (_, b))| json!({"class": k, "bytes": b})).collect();
    let unattributed = total.get("UNATTRIBUTED").map(|x| x.1).unwrap_or(0);
    let lc = flock_core::pcs::ligerito::zk::prover_config(pcs_of(st, LigeritoProfile::Fast100).ligerito_prover_config().unwrap(), zk_queries0(st.m).unwrap());
    let ligerito = json!({"m": st.m, "t_pad0": zk_t_pad(st.m), "initial_k": lc.initial_k, "initial_log_msg_cols": lc.initial_log_msg_cols, "recursive_ks": lc.recursive_ks,
        "recursive_log_msg_cols": lc.recursive_log_msg_cols, "log_inv_rates": lc.log_inv_rates, "queries": lc.queries, "ood_samples": lc.ood_samples,
        "grinding_bits": lc.grinding_bits});
    let out = json!({"statement": st.c.statement(), "unit": st.c.name, "instances": st.n, "ligerito": ligerito,
        "classes": total.iter().map(|(k, (n, b))| (k.to_string(), json!({"items": n, "bytes": b}))).collect::<serde_json::Map<_, _>>(),
        "masked_values": n_masked, "masked_all_differ_from_control": all_differ, "region_claims": n_region, "region_s_hat_v_is_public": region_public,
        "unattributed_bytes": unattributed, "clear": clear, "checks_pass": all_differ && region_public && unattributed == 0,
        "ligerito_values_differing_from_control": lig_differ.iter().map(|(k, (d, n))| (k.to_string(), json!({"differ": d, "of": n}))).collect::<serde_json::Map<_, _>>(),
        "attributed": all_differ && region_public && unattributed == 0 && clear.is_empty()});
    println!("ZKAUDIT\t{out}");
}

/// `zkaudit`'s inner-proof classes on one honest session: in each rep's stream, the observes after the inner proof's label are
/// the masked zerocheck's `final_c'`, then the proof's `[ρ, σ]`, then its `Y`, and `classify` tallies each once, under
/// [`INNER_CLASSES`] in that order.
fn zk_audit_inner_case(st: &Arc<Stmt>, inst: &Arc<Instances>, gpu: bool) -> Value {
    let name = "zk_audit_attributes_the_inner_proof";
    let s0 = Arc::new(Mutex::new(server(st, inst)));
    let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
    let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
    let mut pass = verified(&v) && proofs.len() == 2;
    let mut reps = Vec::new();
    for (rep, proof) in proofs.iter().enumerate() {
        let stream = rep_stream(TABLE, rep);
        let Some(s) = rec["streams"].as_array().and_then(|a| a.iter().find(|s| s["stream"] == stream)) else {
            return json!({"case": name, "pass": false, "why": format!("no stream {stream}")});
        };
        let items: Vec<Item> = s["rounds"].as_array().unwrap().iter().flat_map(|r| parse_items(&hexb(r["msg"].as_str().unwrap_or("")).unwrap()).unwrap()).collect();
        let at = items.iter().position(|it| matches!(it, Item::Label(l, _) if l.as_slice() == zkv::LABEL_INNER)).map_or(items.len(), |i| i + 1);
        let obs: Vec<&Vec<F128>> = items[at..].iter().filter_map(|it| if let Item::Obs(o) = it { Some(o) } else { None }).collect();
        let (_, pr, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(proof).unwrap();
        let want = [vec![pr.zerocheck.final_c_eval], vec![inner.rho, inner.sigma], inner.y.clone()];
        let (t, _, _) = classify(&items);
        let tally: Vec<(usize, usize)> = INNER_CLASSES.iter().map(|c| t.get(c).copied().unwrap_or_default()).collect();
        let values = obs.len() == want.len() && obs.iter().zip(&want).all(|(a, b)| *a == b);
        let classes = tally.iter().zip(&want).all(|(&(n, b), w)| n == 1 && b == 16 * w.len());
        pass &= values && classes;
        reps.push(json!({"rep": rep, "observes_after_label": obs.iter().map(|o| o.len()).collect::<Vec<_>>(), "values_are_the_proofs": values,
                         "classes": INNER_CLASSES.iter().zip(&tally).map(|(c, (n, b))| json!({"class": c, "items": n, "bytes": b})).collect::<Vec<_>>()}));
    }
    json!({"case": name, "expect": "the inner proof's observes are final_c', [rho, sigma] and Y, each tallied once under its class",
           "accepted": v["accepted"], "reps": reps, "pass": pass})
}

/// M2's malicious verifiers V*: the honest coin server with its coins committed from a random tape, and a strategy that sees
/// every prover message. Rewinding V* is a fresh instance on the same tape, replayed to the same point. `Abort` and
/// `AdaptiveCoins` act at the first round whose message, hashed with the tape, falls below [`TRIGGER`]: the first stops
/// there; the second answers a coin it picks from that message, under the committed coin's opening. `CoinAt(k)` does the
/// latter at rep 0's round `k`. `ZeroBeta(k)` commits to β = 0 at rep 0's round `k`, `OuterOne(k)` to a zerocheck outer coin
/// `r_0 = 1` at rep 0's round `k` (the round that opens the outer coins, [`outer_round`]). The gated strategies decide on one
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
    OuterOne(usize),
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
    /// the stream the strategies target: table 0's rep 0
    rep0: String,
}

#[cfg(feature = "seed-injection")]
impl VStar {
    fn new(st: &Arc<Stmt>, inst: &Arc<Instances>, strategy: Strategy, tape: [u8; 32]) -> Self {
        Self::with_server(server(st, inst), TABLE, strategy, tape)
    }

    /// V* for sessions of the given tables (the population's, without a draw).
    fn of(tabs: &Tables, strategy: Strategy, tape: [u8; 32]) -> Self {
        Self::with_server(Server::new(config_of(tabs), Box::new(CircuitVerifier::new(tabs.clone()))), &tabs.list[0].name, strategy, tape)
    }

    fn with_server(mut s: Server, table0: &str, strategy: Strategy, tape: [u8; 32]) -> Self {
        s.inject_coin_tape(tape);
        let rep0 = rep_stream(table0, 0);
        if let Strategy::ZeroBeta(k) | Strategy::OuterOne(k) = strategy {
            let (r0, x) = (rep0.clone(), if matches!(strategy, Strategy::ZeroBeta(_)) { F128::ZERO } else { F128::ONE });
            s.inject_coin_edit(Box::new(move |stream, round, b| {
                if stream == r0 && round == k {
                    b[0] = x;
                }
            }));
        }
        VStar { s, strategy, tape, rounds: BTreeMap::new(), stopped: false, gate: Arc::new(Mutex::new(None)), view: Arc::new(Mutex::new(Vec::new())), rep0 }
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
        let rep0 = stream == self.rep0;
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
/// Per rep stream of the session's tables (in [`Tables::streams`]'s order), its rounds' coins as V* opened them in a view,
/// each opening checked against V*'s commitment as the prover checks it; and the link points.
fn extracted(tabs: &Tables, view: &View) -> Result<(Vec<Vec<F128>>, Vec<Vec<Vec<F128>>>), String> {
    use flock_live::coin_tree::{KEY_BYTES, Rules, check, hello_nonce};
    let (spec, streams, m_pts) = (coin_spec_of(&tabs.names()), tabs.streams(), tabs.list[0].st.m_pts);
    let mut tree: Option<([u8; 64], Rules)> = None;
    let (mut points, mut reps): (Vec<Vec<F128>>, Vec<Vec<Vec<F128>>>) = (Vec::new(), vec![Vec::new(); streams.len()]);
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
                points = check(&spec, rules, root, 0, 0, n, w)?.chunks(m_pts).map(|p| p.to_vec()).collect();
            }
            Req::Round { stream, n, .. } => {
                let (root, rules) = tree.as_ref().ok_or("coins before the commitment")?;
                let s = streams.iter().position(|x| x == stream).ok_or(format!("a round of {stream}"))?;
                let si = spec.stream_index(stream).ok_or(format!("{stream} is outside the schedule"))?;
                let k = reps[s].len();
                reps[s].push(check(&spec, rules, root, si, k, *n as usize, w)?);
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
/// scripted prover whose messages are SHVZK simulations from the extracted coins (`simulate_full` per rep of every table, its
/// rounds recorded by replaying the verifier on it), sent in the prover's order: the tables' streams one after another, the
/// one Commit with every table's root in table 0's rep 0, every proof after the last round. The session's messages up to the
/// coin commitment are the extraction run's (so V* is back just after step 0); every coin V* answers is checked against its
/// commitment and must be the extracted one (else the commitment's binding failed: an error).
#[allow(clippy::too_many_arguments)]
fn rewind(tabs: &Tables, v: impl Transport + 'static, hello: &str, points: &[Vec<F128>], coins: &[Vec<Vec<F128>>], dummy: &[Vec<u8>],
          rng: &mut flock_live::zk_hooks::ProverRng, index: u64) -> Result<Rewound, String> {
    let (count, streams) = (tabs.list.len(), tabs.streams());
    let mut sims = Vec::new();
    for (s, stream) in streams.iter().enumerate() {
        let (tb, rep) = (&tabs.list[s / 2], s % 2);
        let (st, inst) = (&*tb.st, &*tb.inst);
        let (c, pr, inner) = simulate_full(st, inst, &coins[s].concat(), points, &dummy[s], rng, rep, streams.len() as u64 * index + s as u64)?;
        let mut rec = flock_live::RecordingChallenger::new(&domain_of(&st.c, count, &tb.name, rep), coins[s].clone());
        verify_zk_with(st, inst, &c, &pr, &inner, points, rep, &mut rec).map_err(|e| format!("{stream}'s simulation is rejected on the extracted coins: {e}"))?;
        if let Some(e) = rec.mismatch.or_else(|| (rec.rounds.len() != coins[s].len()).then(|| format!("{} rounds, extracted {}", rec.rounds.len(), coins[s].len()))) {
            return Err(format!("{stream}'s simulation does not follow the extracted schedule: {e}"));
        }
        sims.push((bincode::serialize(&(&c, &pr, &inner)).unwrap(), merkle_digest_bytes(&c.cap).to_vec(), rec.rounds));
    }
    let t: SharedTransport = shared(flock_live::coin_tree::Checked::new(v, coin_spec_of(&tabs.names())));
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
    let m_pts = tabs.list[0].st.m_pts;
    for (s, stream) in streams.iter().enumerate() {
        let (tb, rep) = (&tabs.list[s / 2], s % 2);
        expect!(Req::Open { stream: stream.clone(), parent: String::new(), label: domain_of(&tb.st.c, count, &tb.name, rep), at: 0 }, Resp::Ok);
        if s == 0 {
            let roots = tabs.list.iter().enumerate().map(|(j, x)| (x.name.clone(), sims[2 * j].1.clone())).collect();
            match call(&t, Req::Commit { root_f: session_sigma(&tabs.parts()).to_vec(), roots, publics: vec![] })? {
                Resp::Coins(c) if c.chunks(m_pts).eq(points.iter().map(|p| p.as_slice())) => {}
                Resp::Coins(_) => return Err("binding: V* opened other link points than in the extraction run".into()),
                _ => return Ok(Rewound::Aborted),
            }
            expect!(Req::Link(f128_bytes(&vec![F128::ZERO; 2 * count])), Resp::Ok);
        }
        for (k, (content, n)) in sims[s].2.iter().enumerate() {
            match call(&t, Req::Round { stream: stream.clone(), content: content.clone(), n: *n as u32 })? {
                Resp::Coins(c) if c == coins[s][k] => {}
                Resp::Coins(_) => return Err(format!("binding: V* opened other coins at {stream} round {k} than in the extraction run")),
                _ => return Ok(Rewound::Aborted),
            }
        }
    }
    for (s, sim) in sims.iter().enumerate() {
        call(&t, Req::Proof { table: tabs.list[s / 2].name.clone(), rep: (s % 2) as u32, bytes: sim.0.clone() })?;
    }
    call(&t, Req::Finish)?;
    Ok(Rewound::Complete)
}

/// A fresh dummy opening for the extracted coins, with no interaction with V*: the honest prover on the dummy witness (fresh
/// masks, level-0 padding, extra lanes and salts) against a local coin server that commits to exactly those coins, for every
/// table of the session.
#[cfg(feature = "seed-injection")]
fn fresh_dummy(st: &Arc<Stmt>, inst: &Arc<Instances>, tabs: &Tables, points: &[Vec<F128>], coins: &[Vec<Vec<F128>>]) -> Result<Vec<Vec<u8>>, String> {
    let mut s = Server::new(config_of(tabs), Box::new(CircuitVerifier::new(tabs.clone())));
    let (pts, cs, streams) = (points.concat(), coins.to_vec(), tabs.streams());
    s.inject_coin_edit(Box::new(move |stream, round, b| {
        if stream == flock_live::coin_tree::SESSION && round == 0 {
            b[..pts.len()].copy_from_slice(&pts);
        } else if let Some(c) = streams.iter().position(|x| x == stream).and_then(|s| cs[s].get(round)) {
            b[..c.len()].copy_from_slice(c);
        }
    }));
    let plan = Plan { zero_witness: true, tables: tabs.list.len(), split: Some(tabs.clone()), ..Plan::default() };
    let (v, proofs) = session(st, inst, shared(InProc(Arc::new(Mutex::new(s)))), &plan, ProverSeed::from_os());
    if proofs.len() != 2 * tabs.list.len() {
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
///
/// A session of several tables (`tabs`) is one view: every table's streams are extracted, simulated and replayed together.
#[cfg(feature = "seed-injection")]
fn gk_simulate(st: &Arc<Stmt>, inst: &Arc<Instances>, tabs: &Tables, make: &dyn Fn() -> VStar, t: usize) -> (View, Value) {
    // the extraction run draws the coin-tree nonce as the prover does; every later run restarts after its Hello
    let dummy_run = |nonce: Option<[u8; 32]>| -> View {
        let v = make();
        let view = v.view.clone();
        let plan = Plan { zero_witness: true, coin_nonce: nonce, tables: tabs.list.len(), split: Some(tabs.clone()), ..Plan::default() };
        session(st, inst, shared(v), &plan, ProverSeed::from_os());
        let log = view.lock().unwrap_or_else(PoisonError::into_inner).clone();
        log
    };
    let view = dummy_run(None);
    if !complete(&view) {
        return (view, json!({"extraction_complete": false}));
    }
    let (points, coins, hello, nonce) = match extracted(tabs, &view).and_then(|(p, c)| {
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
        let dummy = match fresh_dummy(st, inst, tabs, &points, &coins) {
            Ok(d) => d,
            Err(e) => return (Vec::new(), stats(attempt + 1, Some(e))),
        };
        let v = make();
        let view = v.view.clone();
        match rewind(tabs, v, &hello, &points, &coins, &dummy, &mut rng, attempt as u64) {
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
/// verdict, the first [`ONLINE`] masked values of `rep0` (table 0's rep 0), and each proof it received with its table (for
/// [`proof_fields`]).
fn view_summary(kind: &str, strategy: &str, run: usize, view: &View, rep0: &str) -> (Value, Vec<(String, usize, Vec<u8>)>) {
    let mut rounds: BTreeMap<String, usize> = BTreeMap::new();
    let (mut who, mut at, mut online, mut proofs, mut verdict) = ("none", Value::Null, Vec::new(), Vec::new(), Value::Null);
    for (req, resp) in view {
        match req {
            Req::Round { stream, content, .. } => {
                let k = rounds.entry(stream.clone()).or_default();
                if stream == rep0 && online.len() < ONLINE {
                    online.extend(parse_items(content).unwrap_or_default().into_iter().flat_map(|it| if let Item::Obs(v) = it { v } else { vec![] }));
                }
                who = if matches!(resp, Resp::Refused(_)) { "verifier" } else { "prover" };
                at = json!([stream, *k]);
                *k += 1;
            }
            Req::Proof { table, rep, bytes } => {
                proofs.push((table.clone(), *rep as usize, bytes.clone()));
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
/// `rep0`'s β round (table 0's rep 0): the one whose message is the τ commitment alone (a verifier knows the protocol's shape).
fn beta_round(view: &View, rep0: &str) -> Option<usize> {
    view.iter().filter_map(|(r, _)| match r { Req::Round { stream, content, .. } if stream == rep0 => Some(content), _ => None })
        .enumerate()
        .filter(|(_, c)| matches!(parse_items(c).as_deref(), Ok([Item::Bytes(64), Item::Framing(_), Item::Framing(16)])))
        .map(|(k, _)| k)
        .last()
}

#[cfg(feature = "seed-injection")]
/// Rep 0's round that opens the zerocheck's outer coins: the one after the round whose message carries the zerocheck's label
/// (that round's squeeze opens the skip coins).
fn outer_round(view: &View, rep0: &str) -> Option<usize> {
    view.iter().filter_map(|(r, _)| match r { Req::Round { stream, content, .. } if stream == rep0 => Some(content), _ => None })
        .position(|c| c.windows(zkv::LABEL_ZC.len()).any(|w| w == zkv::LABEL_ZC))
        .map(|k| k + 1)
}

/// `zkrewind --zk --n N`: per verifier strategy (`--strategies`, default all), N real sessions and N Goldreich–Kahan
/// simulations against V* on one tape, each view printed as `ZKVIEW` (its summary), `ZKVIEWP` (each received proof's
/// checked fields) and, when complete, `ZKVIEWREL` (the session's cross-rep relations) for
/// `verity/ml/flock/tests/zk_rewind_stats.py`, and each strategy's simulator statistics as `ZKREWIND`. With
/// `--session-tables J`, every view is a session of J tables: each proof and relation line names its table, and V*'s targeted
/// round is table 0's rep 0's.
#[cfg(feature = "seed-injection")]
fn zkrewind(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let n: usize = arg("--n", "16").parse().unwrap();
    let t: usize = arg("--gk-t", "4").parse().unwrap();
    let eps: u16 = arg("--eps", "4096").parse().unwrap();
    let tape = [arg("--tape", "7").parse::<u8>().unwrap_or_else(|_| refuse("--tape takes one byte (V*'s tape is that byte repeated)".into())); 32];
    let tabs = session_tables(st, inst, None, tables_arg()).unwrap_or_else(|e| refuse(e));
    let (count, rep0) = (tabs.list.len(), rep_stream(&tabs.list[0].name, 0));
    let real = |make: &dyn Fn() -> VStar| -> View {
        let v = make();
        let view = v.view.clone();
        session(st, inst, shared(v), &Plan { tables: count, split: Some(tabs.clone()), ..Plan::default() }, ProverSeed::from_os());
        let log = view.lock().unwrap_or_else(PoisonError::into_inner).clone();
        log
    };
    let emit = |kind: &str, name: &str, run: usize, view: &View| {
        let (s, proofs) = view_summary(kind, name, run, view, &rep0);
        println!("ZKVIEW\t{s}");
        let ext = extracted(&tabs, view).ok().map(|(points, coins)| (points, coins.iter().map(|c| c.concat()).collect::<Vec<Vec<F128>>>()));
        for (j, tb) in tabs.list.iter().enumerate() {
            let bytes: Vec<Vec<u8>> = proofs.iter().filter(|p| p.0 == tb.name).map(|p| p.2.clone()).collect();
            if let (2, Some((points, flat))) = (bytes.len(), &ext)
                && let Ok([l0, l1]) = session_relations(&tb.st, &tb.inst, &bytes, &flat[2 * j..2 * j + 2], points, [0, 1])
            {
                println!("ZKVIEWREL\t{}", json!({"kind": kind, "strategy": name, "run": run, "rep": 0, "table": j, "rel_level0": f128s_hex(&l0),
                                                  "rel_level1": f128s_hex(&l1)}));
            }
        }
        for (table, rep, b) in proofs {
            let Some(j) = tabs.list.iter().position(|x| x.name == table) else { continue };
            let (tb, s) = (&tabs.list[j], 2 * j + rep);
            let lay = zk_layout(&tb.st);
            if let Ok((c, pr, inner)) = bincode::deserialize::<(Commitment, R1csProofLigerito, InnerProof)>(&b) {
                let mut row = proof_fields(&lay, kind, run, rep, &pr, &inner);
                row["strategy"] = json!(name);
                row["table"] = json!(j);
                // the level-0 positions of the sampled rows, from the verifier's record on the view's coins
                if let (Some((points, flat)), Ok(root)) = (&ext, <[u8; 64]>::try_from(inner.root.clone())) {
                    if let Ok((_, rec)) = zk_constraints(&tb.st, &tb.inst, &c, &pr, &root, points, rep, &mut zkv::Tape::new(&flat[s])) {
                        row["lig_rows0_q"] = json!(rec.queries0[..4]);
                    }
                    let n = zkv::Code::new(lay.len() + zkv::K_PAD).n;
                    row["inner_open_q"] = json!(zkv::query_positions(&flat[s][flat[s].len() - zkv::Q..], n));
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
            "zero-beta" => Strategy::ZeroBeta(beta_round(&real(&|| VStar::of(&tabs, Strategy::Honest, tape)), &rep0).expect("rep 0's beta round")),
            "outer-one" => Strategy::OuterOne(outer_round(&real(&|| VStar::of(&tabs, Strategy::Honest, tape)), &rep0).expect("rep 0's outer round")),
            "cap-gated" => Strategy::CapGated(eps),
            "late-gated" => Strategy::LateGated(eps),
            x => refuse(format!("unknown strategy {x}")),
        };
        let make = || VStar::of(&tabs, strategy, tape);
        let ts = Instant::now();
        for run in 0..n {
            emit("real", name, run, &real(&make));
        }
        let t_real = ts.elapsed().as_secs_f64();
        let mut stats = Vec::new();
        for run in 0..n {
            let (view, s) = gk_simulate(st, inst, &tabs, &make, t);
            if s["fail"].is_null() {
                emit("sim", name, run, &view);
            }
            stats.push(s);
        }
        println!("ZKREWIND\t{}", json!({"strategy": name, "n": n, "gk_t": t, "tables": count, "real_s": t_real,
                                         "sim_s": ts.elapsed().as_secs_f64() - t_real, "sims": stats}));
    }
}

/// `zkgk --zk --attempts A`: the rewinds are independent trials (red-team-m2-zk G1). For each gated V* (`cap-gated`,
/// `late-gated`, `--eps`), A real sessions give its completion rate; then, conditioned on an extraction run that reached the
/// proofs with its gate "bad", A rewinds with a fresh dummy opening each (as `gk_simulate` does) must complete at the same
/// rate, not at ε. `--reuse-dummy` also runs A rewinds on the extraction run's own dummy opening, the defect, to show the
/// comparison has power. Prints `ZKGK` lines for `verity/ml/flock/tests/zk_rewind_stats.py`. `--session-tables J`: sessions of
/// J tables, as in `zkrewind`.
#[cfg(feature = "seed-injection")]
fn zkgk(st: &Arc<Stmt>, inst: &Arc<Instances>) {
    let attempts: usize = arg("--attempts", "40").parse().unwrap();
    let eps: u16 = arg("--eps", "4096").parse().unwrap();
    let tape = [arg("--tape", "7").parse::<u8>().unwrap_or_else(|_| refuse("--tape takes one byte (V*'s tape is that byte repeated)".into())); 32];
    let tabs = session_tables(st, inst, None, tables_arg()).unwrap_or_else(|e| refuse(e));
    let count = tabs.list.len();
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
        let make = || VStar::of(&tabs, strategy, tape);
        let real_plan = Plan { tables: count, split: Some(tabs.clone()), ..Plan::default() };
        let real = (0..attempts).filter(|_| complete(&run(make(), &real_plan).0)).count();
        let dummy_plan = Plan { zero_witness: true, ..real_plan.clone() };
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
        let (points, coins) = extracted(&tabs, &view).unwrap_or_else(|e| refuse(e));
        let hello = view_hello(&view).unwrap_or_else(|e| refuse(e));
        let mut rng = flock_live::zk_hooks::ProverRng::from_os();
        let mut done = |dummy: &[Vec<u8>], a: u64| matches!(rewind(&tabs, make(), &hello, &points, &coins, dummy, &mut rng, a), Ok(Rewound::Complete));
        let fresh = (0..attempts).filter(|&a| done(&fresh_dummy(st, inst, &tabs, &points, &coins).unwrap_or_else(|e| refuse(e)), a as u64)).count();
        let reused = flag("--reuse-dummy").then(|| (0..attempts).filter(|&a| done(&dummy, 1000 + a as u64)).count());
        println!("ZKGK\t{}", json!({"strategy": name, "eps": eps as f64 / 65536.0, "attempts": attempts, "tables": count, "real_complete": real,
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
           "rep_reused": reused, "native_sha": native_sha,
           "host_slots": match wt { WitData::Dev(_, h, _) | WitData::DevCut(_, h, _, _) => h.layout(), WitData::Host(..) => "none" }})
}

/// One rep on the GPU; with `reuse_tag`, rep 0 keeps its witness and level-0 commitment on the device and rep 1 takes them.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn prove_gpu(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, ch: &mut LiveChallenger,
             salts: &Mutex<flock_live::zk_hooks::LeafSalts>, reuse_tag: Option<u64>, native_sha: bool) -> Result<(Vec<u8>, Value), String> {
    use flock_live::gpu_circuit::{Live, Reuse, kept, prove_circuit};
    let reuse = match (reuse_tag, rep) {
        (None, _) => Reuse::None,
        (Some(t), 0) => Reuse::Keep(t),
        (Some(t), _) => Reuse::Take(t),
    };
    let reused = matches!(reuse, Reuse::Take(t) if kept(t));
    let (proof, commitment, ph, wph, hph) =
        with_gpu_src(st, wt, |src| prove_circuit(st, src, &pcs_of(st, lig), &domain(&st.c, rep), hook, Live::Stream(ch), salted(st).then_some(salts), reuse,
                                                 native_sha))?;
    let ts = Instant::now();
    let bytes = bincode::serialize(&(commitment, proof)).unwrap();
    let mut bk = gpu_buckets(wt, ph, wph, hph, reused, native_sha);
    bk["h.serialize"] = json!(ts.elapsed().as_secs_f64());
    eprintln!("GPUPHASE\t{bk}");
    Ok((bytes, bk))
}

/// M1 on the device, its device part: every live round the device frames goes to `rounds` (the rep's masking challenger, made
/// again there by `flock_live::replay_round`, so the verifier sees what M1's CPU prover sends); level 0's hiding runs on the device.
/// The caller finishes the rep with [`zk_finish`], or [`zk_finish_fw`] under the firewall, as on the CPU.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn gpu_zk_device(st: &Stmt, wt: &WitData, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, salts: &Mutex<flock_live::zk_hooks::LeafSalts>,
                 level0: &flock_core::pcs::ligerito::zk::Level0Zk, rounds: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, native_sha: bool)
                 -> Result<(R1csProofLigerito, Commitment, Value), String> {
    use flock_live::gpu_circuit::{Live, Reuse, prove_circuit};
    let (proof, commitment, ph, wph, hph) =
        with_gpu_src(st, wt, |src| prove_circuit(st, src, &pcs_of(st, lig), &domain(&st.c, rep), hook, Live::Zk { level0, rounds }, Some(salts),
                                                 Reuse::None, native_sha))?;
    let bk = gpu_buckets(wt, ph, wph, hph, false, native_sha);
    eprintln!("GPUPHASE\t{bk}");
    Ok((proof, commitment, bk))
}

#[cfg(not(feature = "gpu"))]
#[allow(clippy::too_many_arguments)]
fn prove_gpu(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &mut LiveChallenger,
             _: &Mutex<flock_live::zk_hooks::LeafSalts>, _: Option<u64>, _: bool) -> Result<(Vec<u8>, Value), String> {
    Err("built without the gpu feature".into())
}

#[cfg(not(feature = "gpu"))]
#[allow(clippy::too_many_arguments)]
fn gpu_zk_device(_: &Stmt, _: &WitData, _: LigeritoProfile, _: usize, _: Hook<'_>, _: &Mutex<flock_live::zk_hooks::LeafSalts>,
                 _: &flock_core::pcs::ligerito::zk::Level0Zk, _: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, _: bool)
                 -> Result<(R1csProofLigerito, Commitment, Value), String> {
    Err("built without the gpu feature".into())
}

/// [`gpu_zk_device`] for a rep: under the firewall on a worker thread of its own ([`worker`], with the rep's `cut-` control), its
/// rounds (`round`) and its hook served here, on the session's thread; without the firewall, here. Its error is the device's.
#[allow(clippy::too_many_arguments)]
fn gpu_device(st: &Arc<Stmt>, wt: &Arc<WitData>, lig: LigeritoProfile, rep: usize, hook: Hook<'_>, salts: &Arc<Mutex<flock_live::zk_hooks::LeafSalts>>,
              level0: &Arc<flock_core::pcs::ligerito::zk::Level0Zk>, round: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, native_sha: bool,
              firewall: &Option<Arc<Mutex<Firewall>>>) -> Result<(R1csProofLigerito, Commitment, Value), String> {
    let Some(g) = firewall else { return gpu_zk_device(st, wt, lig, rep, hook, salts, level0, round, native_sha) };
    let cut = g.lock().unwrap_or_else(PoisonError::into_inner).cut();
    let (st, wt, salts, level0) = (st.clone(), wt.clone(), salts.clone(), level0.clone());
    worker(cut, move |rounds, hook| gpu_zk_device(&st, &wt, lig, rep, hook, &salts, &level0, rounds, native_sha), round, hook)
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
    fill_forged: bool,
    mixed_pad_forged: bool,
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
    /// M1: `final_c'` re-solved after the inner proof's batching coins, cancelling the failing rows
    zk_final_c_solved: bool,
    /// M1's control: zero pads and a zero mask slot (M0's values, in M1's format)
    zk_zero_pads: bool,
    /// M1: Ligerito level 0's hiding tampered (`Level0Zk::tamper`)
    zk_l0_tamper: u8,
    /// both reps fold, sum and pad rep 0's pair of extra lanes (red-team-m1-zk F1's shared lanes; the verifier rejects rep 1)
    zk_shared_pair: bool,
    /// M1: whether the inner proof checks its batched constraint before it speaks (`zk_veil::SelfCheck`). Every honest plan
    /// stops; `Skip` is for the negative controls in `plan_of` whose witness or pads fail by design, so that their proofs
    /// reach the verifier
    zk_self_check: zkv::SelfCheck,
    /// prove a dummy witness: all zero but the public regions (M2's extraction run; M1's simulator takes its opening). Its
    /// inner proof skips the self-check: the dummy fails the constraints, and the simulator's runs must reach the proofs as
    /// the prover's do
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
    /// the host slots in mapped host memory, read by the device beside the compressions (`Some(true)`, the default unless
    /// `FC_HOST_PINNED=0`), or pageable and copied slot by slot (`Some(false)`)
    host_mapped: Option<bool>,
    /// this session's witness, built beside the previous session (`prebuild`); taken when it is the one the session would build
    prebuilt: Option<Arc<Mutex<Option<std::thread::JoinHandle<Prebuilt>>>>>,
    /// the session's tables ([`session_tables`]; 0 or 1: one table)
    tables: usize,
    /// those tables over the population, built once for a harness's repeated sessions without a draw
    split: Option<Tables>,
    /// a prover without the barrier: each table's proofs as soon as its reps finish, before the next table's coins (the
    /// verifier accepts them; the barrier's order check must not). A test harness's only: a proving build has no such prover
    #[cfg(feature = "seed-injection")]
    early_proofs: bool,
    /// the transport is the firewall's Lean session loop ([`session_addr`]), which sends its own `Hello` and checks every coin
    lean_session: bool,
    /// that loop's frames' address: each rep's Lean frame is one of the loop's, whose coins are the ones the loop opened
    lean_frames: Option<String>,
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap_or_else(PoisonError::into_inner).call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// Table `table`'s witness sources (its mask slot from the table's own streams).
fn wit(st: &Arc<Stmt>, inst: &Arc<Instances>, plan: &Plan, zero: bool, seed: ProverSeed, table: usize) -> Wit {
    let mut rows = inst.instance_rows(&st.c).expect("the prover's instance file has the rows");
    if plan.message_flip {
        rows[3] ^= 0x10;
    }
    let rows = Arc::new(rows);
    let salts = Arc::new(inst.instance_salts(st.c.ports.len()).expect("the prover's instance file has the salts"));
    let (s1, r1, sa1) = (st.clone(), rows.clone(), salts.clone());
    let tamper = RowTamper { cv_flip: plan.chain_forged, midstate_flip: plan.midstate_forged, pad_flip: plan.pad_forged, salt_flip: plan.salt_forged,
                             fill_flip: plan.fill_forged, mixed_pad_flip: plan.mixed_pad_forged };
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
    // M1: the mask slot's words are uniform (they hide the non-region claims' ring-switch s_hat_v). A block's are drawn once and
    // kept: a proving build's are OS reads, and every build of this witness (rep 1's own, the device's) commits one level 0
    let random = plan.mask_random || (is_zk(st) && !plan.zk_zero_pads);
    let hooks = Arc::new(Mutex::new(NoMask::for_table(seed, table as u64)));
    let drawn: Arc<Vec<std::sync::OnceLock<Vec<F128>>>> = Arc::new((0..if random { st.blocks() } else { 0 }).map(|_| Default::default()).collect());
    let mask: Arc<dyn Fn(usize) -> Vec<F128> + Sync + Send> = Arc::new(move |b| {
        let draw = || {
            let mut h = hooks.lock().unwrap_or_else(PoisonError::into_inner);
            if random { h.rng.f128s(flock_live::zk_hooks::STREAM_MASK, (b * mask_words) as u64, mask_words) } else { h.mask_fill(b, mask_words) }
        };
        if random { drawn[b].get_or_init(draw).clone() } else { draw() }
    });
    let ur = st.c.unit_range();
    let unit = st.c.unit();
    // the unit's first cut input of instance 0; a unit that reads no cut word (the cut feeds the stages): the first stage input a
    // wire fills, in its slot of block 0
    let cut_flip = if !plan.cut_flip || st.c.typed.is_some() {
        None
    } else if unit.in_groups.len() > 1 {
        Some((st.c.unit_net, 0, unit.in_groups[1].col * 128 + 3))
    } else {
        st.c.wires.iter().find(|w| ![st.c.sha_net, st.c.hm_net, st.c.unit_net].contains(&w.dst.0)).map(|w| {
            let (col, width) = port_cols(&st.c.nets[w.dst.0].1.in_groups)[w.dst.2];
            (w.dst.0, w.dst.1, col + 3.min(width - 1))
        })
    };
    // a typed template's cut words are its parts' bound inputs: the first one of instance 0
    let typed_flip = st.c.typed.as_ref().filter(|_| plan.cut_flip).and_then(|t| t.inst.bind.iter().flatten().next()).map(|b| (0, b.0));
    let nets = if plan.tail_swapped { forged_stages(st) } else { BTreeMap::new() };
    // a unit's internal row; a typed template's first AND row, the root's (else a part's)
    let flip_z = match &st.c.typed {
        Some(t) => t.inst.first_and().map(|u| t.col(&st.c, ur.count.min(6) - 1, u)),
        None => Some(ur.slot(ur.count.min(6) - 1) + st.c.unit().in_words * 128 + 40),
    };
    let typed_forged = if plan.tail_swapped { forged_instance(st) } else { None };
    Wit { rowin, inputs, mask, flip_z: flip_z.filter(|_| plan.unit_flip), cut_flip, typed_flip, typed_forged, nets: Arc::new(nets),
          regions: (zero && plan.zero_witness).then(|| inst.clone()), mask_host: random }
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

/// How a session builds its witness: on the device (inputs and host slots) or on the host, with the units' slots on the host,
/// and the host slots in mapped memory.
fn witness_mode(st: &Arc<Stmt>, plan: &Plan) -> (bool, bool, bool) {
    let device = plan.gpu && !plan.mask_random && std::env::var("FC_HOST_WITNESS").is_err();
    #[cfg(feature = "gpu")]
    let host_units = device && plan.host_units.unwrap_or_else(|| flock_live::gpu_circuit::deep_unit(st));
    #[cfg(not(feature = "gpu"))]
    let host_units = { let _ = st; false };
    (device, host_units, plan.host_mapped.unwrap_or(true))
}

/// A one-table session's witness, built ahead of it: for this statement, seed and witness mode only.
struct Prebuilt {
    st: Arc<Stmt>,
    seed: ProverSeed,
    mode: (bool, bool, bool),
    w0: Wit,
    wt0: WitData,
    built_s: f64,
}

/// Build the witness a one-table session of `st` with `seed` will take, on a thread of its own (the prover's host work beside the
/// previous session's device work).
fn prebuild(st: &Arc<Stmt>, inst: &Arc<Instances>, plan: &Plan, seed: ProverSeed) -> std::thread::JoinHandle<Prebuilt> {
    let (st, inst, plan) = (st.clone(), inst.clone(), plan.clone());
    flock_live::spawn(move || {
        let mode = witness_mode(&st, &plan);
        let w0 = wit(&st, &inst, &plan, plan.zero_witness, seed, 0);
        let (wt0, built_s) = witness_uploaded(&st, &w0, mode);
        Prebuilt { st, seed, mode, w0, wt0, built_s }
    })
}

/// [`witness`], and with `FC_DEV_PREFETCH=1` its host slots and compression rows copied to device memory
/// (`HostSlots::upload`), which the session's device then reads; otherwise its compression rows copied to page-locked memory
/// (`HostSlots::pin_rows`). Built ahead (`prebuild`), the copies run beside the previous session's device work; either way their
/// time counts as the build's.
fn witness_uploaded(st: &Stmt, w: &Wit, mode: (bool, bool, bool)) -> (WitData, f64) {
    #[allow(unused_mut)]
    let (mut wt, mut s) = witness(st, w, mode.0, mode.1, mode.2);
    #[cfg(feature = "gpu")]
    if let WitData::Dev(rows, h, _) = &mut wt {
        let t = Instant::now();
        if std::env::var("FC_DEV_PREFETCH").as_deref() == Ok("1") {
            h.upload(rows);
        } else {
            h.pin_rows(rows);
        }
        s += t.elapsed().as_secs_f64();
    }
    (wt, s)
}

fn session(st: &Arc<Stmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan, seed: ProverSeed) -> (Value, Vec<Vec<u8>>) {
    match session_proved(st, inst, t, plan, seed) {
        Ok(p) => session_finish(p),
        Err(stopped) => stopped,
    }
}

/// A session past its last coin: its proofs made, not yet sent. [`session_finish`] sends them, takes the verdict and checks
/// step 0; it touches no device state, so it can run on its own thread while the next session proves.
struct Proved {
    t: SharedTransport,
    log: Arc<Mutex<Vec<Vec<F128>>>>,
    tabs: Tables,
    plan: Plan,
    hello: String,
    commit: Vec<F128>,
    sig: [u8; 64],
    res: Result<(), String>,
    proofs: Vec<Vec<u8>>,
    early: usize,
    e2e_s: f64,
    wait_e2e_s: f64,
    witness_s: f64,
    witness_prebuilt_s: Option<f64>,
    prove_s: Vec<f64>,
    buckets: Vec<Value>,
    tp0: Instant,
}

/// [`session`] up to its proofs; `Err` is a session the prover stopped before proving (its record).
fn session_proved(st: &Arc<Stmt>, inst: &Arc<Instances>, t: SharedTransport, plan: &Plan, seed: ProverSeed) -> Result<Proved, (Value, Vec<Vec<u8>>)> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let names = table_names(plan.tables);
    if firewall_on(st) && !plan.lean_session {
        let why = flock_live::firewall::refused("a firewalled session goes through the Lean session loop (`prove`, which starts it)");
        return Err((json!({"accepted": false, "why": why, "prover_stopped": why}), Vec::new()));
    }
    // M2: every coin the verifier sends is checked against its step-0 commitment before the prover uses it; under the
    // firewall the Lean session loop checks them (`flock-firewall session`), and only the coins come back
    let t: SharedTransport = if is_zk(st) && !plan.lean_session {
        shared(flock_live::coin_tree::Checked::new(SharedCall(t), coin_spec_of(&names)))
    } else {
        t
    };
    let t: SharedTransport = shared(Logged { inner: SharedCall(t), coins: log.clone() });
    let t0 = Instant::now();
    // registration done: the verifier draws the units (or none), and the statement is the drawn units', in the session's tables
    let reg = call(&t, Req::Register);
    let draw = match &reg {
        Ok(Resp::Draw(d)) if !plan.ignore_draw => d.clone(),
        _ => "null".to_string(),
    };
    let tabs = match reg {
        Ok(Resp::Draw(d)) if d == "null" || plan.ignore_draw => match &plan.split {
            Some(s) if s.names() == names => Ok(s.clone()),
            _ => session_tables(st, inst, None, names.len()),
        },
        Ok(Resp::Draw(d)) => session_tables(st, inst, Some(&serde_json::from_str(&d).unwrap_or(Value::Null)), names.len()),
        other => return Err((json!({"accepted": false, "why": format!("register: {other:?}"), "prover_stopped": "register refused"}), Vec::new())),
    };
    let tabs = match tabs {
        Ok(x) if plan.gpu && x.list.len() > 1 => {
            let why = "the device prover proves one table per session (several: the CPU prover)";
            return Err((json!({"accepted": false, "why": why, "prover_stopped": why}), Vec::new()));
        }
        Ok(x) => x,
        Err(e) => return Err((json!({"accepted": false, "why": e, "prover_stopped": format!("the drawn statement: {e}")}), Vec::new())),
    };
    let st = &tabs.list[0].st;
    // the firewall's proof files of record: the Lean session loop's finish on the session's statement, at the verifier's draw
    let lean_fin = match firewall_on(st).then(|| {
        let addr = plan.lean_frames.as_deref().ok_or_else(|| flock_live::firewall::refused("a firewalled session without the Lean session loop's frames"))?;
        flock_live::firewall::LeanFinish::connect(addr, &draw)
    }).transpose() {
        Ok(f) => f.map(|f| Arc::new(Mutex::new(f))),
        Err(e) => return Err((json!({"accepted": false, "why": e, "prover_stopped": format!("the firewall's Lean finish: {e}")}), Vec::new())),
    };
    let sig = session_sigma(&tabs.parts());
    // under the firewall the Lean session loop sends its own Hello, at a nonce it draws, in place of whatever the prover sends
    let hello = if plan.lean_session {
        String::new()
    } else if is_zk(st) {
        config_of(&tabs).hello_with_nonce(&plan.coin_nonce.unwrap_or_else(|| flock_live::zk_hooks::coin_nonce(&seed)))
    } else {
        config_of(&tabs).hello()
    };
    let mut commit: Vec<F128> = Vec::new();
    let (device, host_units, mapped) = witness_mode(st, plan);
    // on the device, rep 1 proves from rep 0's witness and level-0 commitment (the session's tag keys what rep 0 keeps); not
    // under M1, whose level 0 and lane phase are its own
    static SESSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let native_sha = plan.native_sha.unwrap_or_else(|| std::env::var("FC_NATIVE_SHA").as_deref() != Ok("0"));
    let reuse_tag = (plan.gpu && !plan.rep1_other_witness && !is_zk(st)
        && plan.rep_reuse.unwrap_or_else(|| std::env::var("FC_REP_REUSE").as_deref() != Ok("0")))
        .then(|| SESSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let lig = if plan.fast { LigeritoProfile::Fast } else { LigeritoProfile::Fast100 };
    // hm96-sha512 leaf salts, read from the OS: the level-0 tree is id 0 (both reps bind one root), every later tree a fresh
    // id (table j's from j << 32). The device takes the host's salts tree by tree, as a CPU proof does.
    let salts_t: Vec<Arc<Mutex<flock_live::zk_hooks::LeafSalts>>> = {
        let s = flock_live::zk_hooks::LeafSalts::for_seed(&seed);
        (0..tabs.list.len()).map(|j| Arc::new(Mutex::new(s.table(j as u64)))).collect()
    };
    // the rank check's claim points: per table (each has its own mask slot)
    let points_t: Vec<Arc<Mutex<Vec<Vec<F128>>>>> = tabs.list.iter().map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
    // M1 on the device: level 0's hiding and both reps' pads commitments beside the witness build, and rep 0's replay and inner
    // proof beside rep 1's device run (FC_ZK_SEQUENTIAL: neither, the reference for the overlap's gain)
    let overlap = std::env::var_os("FC_ZK_SEQUENTIAL").is_none();
    let zk_pre = (plan.gpu && is_zk(st) && overlap && plan.lean_frames.is_none()).then(|| {
        let (st, plan, salts, points) = (st.clone(), plan.clone(), salts_t[0].clone(), points_t[0].clone());
        flock_live::spawn(move || zk_prepare(&st, &plan, seed, lig, &salts, &points))
    });
    // every table's witness. Rep 1 proves from rep 0's unless it has its own: the CPU prover consumes a witness, so rep 0 takes a
    // copy and rep 1 the original (two copies at most, never one per rep plus the original)
    let mut witness_s = 0.0;
    let mut witness_prebuilt_s: Option<f64> = None;
    let mut wits = Vec::new();
    for (j, tb) in tabs.list.iter().enumerate() {
        // a witness built beside the previous session: its wait counts as this session's witness time
        let tw = Instant::now();
        let pre = (tabs.list.len() == 1 && !plan.rep1_other_witness)
            .then(|| plan.prebuilt.as_ref().and_then(|m| m.lock().unwrap_or_else(PoisonError::into_inner).take()))
            .flatten()
            .map(|h| h.join().unwrap_or_else(|e| panic!("the prebuilt witness: {}", panic_msg(e))))
            .filter(|p| Arc::ptr_eq(&p.st, &tb.st) && p.seed == seed && p.mode == (device, host_units, mapped));
        let (w0, wt0, tw0) = match pre {
            Some(p) => {
                witness_prebuilt_s = Some(p.built_s);
                (p.w0, p.wt0, tw.elapsed().as_secs_f64())
            }
            None => {
                let w0 = wit(&tb.st, &tb.inst, plan, plan.zero_witness, seed, j);
                let (wt0, tw0) = witness_uploaded(&tb.st, &w0, (device, host_units, mapped));
                (w0, wt0, tw0)
            }
        };
        let (wt1, tw1) = if plan.rep1_other_witness {
            let (w, t) = witness(&tb.st, &wit(&tb.st, &tb.inst, plan, true, seed, j), device, host_units, mapped);
            (Some(w), t)
        } else {
            (None, 0.0)
        };
        witness_s += tw0 + tw1;
        wits.push((w0, Some(wt0), wt1));
    }
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
    let mut link_pts: Option<Vec<Vec<F128>>> = None;
    let mut stop_why: Option<String> = None;
    // the proofs a prover without the barrier sent before the session's last coin (`early_proofs`)
    #[cfg_attr(not(feature = "seed-injection"), allow(unused_mut))]
    let mut early = 0;
    let tp0 = Instant::now();
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello.clone()))? {
            // M0's seed commitment (4 words), or the keyed coin tree's root and session key (20; `coin_tree::Checked` checked them)
            Resp::Coins(c) if c.len() == if is_zk(st) { 20 } else { 4 } && (is_zk(st) || seed_coins()) => commit = c,
            // FC_COINS=os: nothing committed, every round's coins the verifier's OS draws
            Resp::Ok if !is_zk(st) && !seed_coins() => {}
            r => return Err(format!("hello: step 0 expects the verifier's {}, got {r:?}",
                                    if is_zk(st) || seed_coins() { "seed commitment" } else { "Ok (FC_COINS=os)" })),
        }
        // several tables: every table after the first commits its level-0 root (and draws its level-0 hiding) before any stream
        // opens, so the session's one Commit, in table 0's first rep, carries every table's root
        let mut later = Vec::new();
        for (j, tb) in tabs.list.iter().enumerate().skip(1) {
            let tc = Instant::now();
            let mut prng = flock_live::zk_hooks::ProverRng::for_table(seed, j as u64);
            let mut l0 = is_zk(&tb.st).then(|| level0_zk(&tb.st, &mut prng, plan.zk_zero_pads, plan.zk_l0_tamper));
            // under the firewall the root is the Lean session loop's
            let lean = match (firewall_on(&tb.st), l0.as_mut(), wits[j].1.as_ref()) {
                (true, Some(l), Some(WitData::Host((z, ..)))) => Some(lean_level0(plan, &tb.st, lig, &commit, j, z, l, &salts_t[j])?
                    .ok_or_else(|| flock_live::firewall::refused("a firewalled table without the Lean session loop's level 0"))?),
                _ => None,
            };
            let root = match lean {
                Some(root) => root,
                None => level0_root(&tb.st, wits[j].1.as_ref().expect("a table's witness"), lig, l0.as_ref(), &salts_t[j]),
            };
            later.push((root, l0, tc.elapsed().as_secs_f64()));
        }
        let later_roots: Vec<Vec<u8>> = later.iter().map(|x| x.0.clone()).collect();
        let mut later = later.into_iter();
        let mut stop_at_finish: Option<String> = None;
        for (j, (tb, (w0, mut wt0, mut wt1))) in tabs.list.iter().zip(wits).enumerate() {
            let table_buckets = buckets.len();
            let (st, salts, zk_points) = (&tb.st, &salts_t[j], &points_t[j]);
            let dom = |rep: usize| domain_of(&tb.st.c, tabs.list.len(), &tb.name, rep);
            // a later table's reps bind the root it committed
            let (mut root0, table_level0, t_table) = if j == 0 {
                (None, None, 0.0)
            } else {
                let (r, l, s) = later.next().expect("a later table's committed root");
                (Some(r), l, s)
            };
            let firewalled = firewall_on(st);
            // under the firewall its own salt context spans the table (`Firewall::new`)
            #[cfg(feature = "sha512")]
            let level0 = (!plan.gpu && !firewalled && salted(st))
                .then(|| salts.lock().unwrap_or_else(PoisonError::into_inner).level0(pcs_of(st, lig).n_leaves()));
            let mut prng = flock_live::zk_hooks::ProverRng::for_table(seed, j as u64);
            let tl0 = Instant::now();
            let mut zk_level0 = if j == 0 { (is_zk(st) && zk_pre.is_none()).then(|| level0_zk(st, &mut prng, plan.zk_zero_pads, plan.zk_l0_tamper)) } else { table_level0 };
            let lean_root0 = match (j, firewalled, zk_level0.as_mut(), wt0.as_ref()) {
                (0, true, Some(l0), Some(WitData::Host((z, ..)))) => lean_level0(plan, st, lig, &commit, 0, z, l0, salts)?,
                _ => None,
            };
            let t_level0 = if zk_pre.is_some() { t_zk_pre_wait } else { tl0.elapsed().as_secs_f64() + t_table };
            // the firewall's level-0 recheck, before anything of the table is sent (counted in rep 0's prove time)
            let tg = Instant::now();
            let firewall: Option<Arc<Mutex<Firewall>>> = match firewalled {
                false => None,
                true => {
                    let Some(WitData::Host((z, ..))) = wt0.as_ref() else {
                        return Err("FIREWALL-REFUSED: the firewall rechecks a host witness (FC_HOST_WITNESS=1)".into());
                    };
                    let l0 = match zk_pre.as_ref() {
                        Some(z) => z[0].as_ref().expect("rep 0's pads").0.level0.clone(),
                        None => Arc::new(zk_level0.clone().expect("level 0's hiding")),
                    };
                    let fin = lean_fin.as_ref().ok_or_else(|| flock_live::firewall::refused("a firewalled table without the session's Lean finish"))?;
                    let lean_root = if j == 0 { lean_root0.clone() } else { root0.clone() };
                    let mut g = Firewall::new(st, z, lig, &l0, salts, !plan.gpu, j, fin, lean_root)?;
                    if let Some(t) = g.tamper.strip_prefix("later-") {
                        g.tamper = if j == 0 { String::new() } else { t.to_string() };
                    }
                    if g.tamper == "cut-tree" && plan.gpu {
                        return Err("FIREWALL-REFUSED: cut-tree is the CPU device's control (its trees draw through the salt context)".into());
                    }
                    // a later table's root left in table 0's Commit (`level0_root`, the host's): the recheck must have it
                    if let Some(r) = root0.as_ref().filter(|r| **r != g.root) {
                        return Err(format!("FIREWALL-REFUSED: {}'s level-0 root: the session's Commit carried {} where the recheck has {}",
                                           tb.name, hex(&r[..8]), hex(&g.root[..8])));
                    }
                    Some(Arc::new(Mutex::new(g)))
                }
            };
            let t_level0 = t_level0 + if firewalled { tg.elapsed().as_secs_f64() } else { 0.0 };
            // FC_FIREWALL_TAMPER=witness: the device proves a witness one bit off the host's (its level-0 root is not the recheck's)
            let tampered = |t: &str| firewall.as_ref().is_some_and(|g| g.lock().unwrap_or_else(PoisonError::into_inner).tamper == t);
            let mut wt_dev: Option<WitData> = tampered("witness").then(|| match wt0.clone() {
                Some(WitData::Host((mut z, a, b, zl))) => {
                    z[0].lo ^= 1;
                    WitData::Host((z, a, b, zl))
                }
                _ => unreachable!("the firewall takes a host witness"),
            });
            // the device's witness, shared by both reps unless rep 1 has its own (a worker thread holds it while it runs)
            let mut shared0: Option<Arc<WitData>> = None;
            let mut cached: Option<Vec<Vec<F128>>> = None;
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
                        let name = flock_live::firewall::level0_session(&commit);
                        make_zkrep(st, plan, &mut prng, zk_level0.as_ref().unwrap(), zk_points, rep, id, salts, Some((&name, j)))
                    }), None),
                };
                if let (Some(id), 1) = (pre_id, rep) {
                    let got = salts.lock().unwrap_or_else(PoisonError::into_inner).next_id();
                    if got != id {
                        // rep 0's replay and inner proof may still be sending: it ends before the stop, so the wire holds whole messages
                        if let Some(h) = rep0_finish.take() {
                            let _ = h.join();
                        }
                        return Err(format!("PROVER-STOPPED: rep 1's pads tree takes salt id {id}, but the salt counter is at {got} after rep 0's proof's trees"));
                    }
                }
                if let (Some(g), Some(z)) = (&firewall, &zkrep) {
                    g.lock().unwrap_or_else(PoisonError::into_inner).begin_rep(rep, z)?;
                }
                let t_zk_commit = if pre_id.is_some() { if rep == 0 { t_level0 } else { 0.0 } }
                                  else { tz0.elapsed().as_secs_f64() + if rep == 0 { t_level0 } else { 0.0 } };
                // rep 1's stream on the device path opens at its first round, after rep 0's last one (the order on the wire)
                let mut lch = if rep0_finish.is_some() { None } else { Some(LiveChallenger::new(t.clone(), &tb.name, rep, &dom(rep))?) };
                let tt = t.clone();
                let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                    // red-team-hm96 C1: both reps' level-0 trees take salt stream 0, so they must be one tree; stop before rep 1
                    // opens anything (its openings would reveal id-0 salts over other leaves). A later table's rep 0 checks the
                    // root it committed the same way.
                    match &root0 {
                        None => root0 = Some(root.to_vec()),
                        Some(r0) if r0.as_slice() != root && !plan.skip_level0_check => {
                            // the device prover's callback swallows the panic, so the reason is kept here too
                            let why = if j == 0 {
                                "PROVER-STOPPED: rep 1's level-0 root is not rep 0's; the shared level-0 salts would open over two trees".to_string()
                            } else {
                                format!("PROVER-STOPPED: {}'s level-0 root is not the one it committed; its level-0 salts would open over two trees", tb.name)
                            };
                            stop_why = Some(why.clone());
                            panic!("{why}")
                        }
                        _ => {}
                    }
                    if let Some(c) = &cached {
                        return c.clone();
                    }
                    let pts = match &link_pts {
                        Some(p) => p.clone(),
                        None => {
                            // the session's one Commit: this root (table 0's) and every later table's
                            let roots = tabs.list.iter().map(|x| x.name.clone()).zip(std::iter::once(root.to_vec()).chain(later_roots.iter().cloned())).collect();
                            let p: Vec<Vec<F128>> = match call(&tt, Req::Commit { root_f: sig.to_vec(), roots, publics: vec![] }) {
                                Ok(Resp::Coins(c)) if c.len() == 2 * st.m_pts => c.chunks(st.m_pts).map(|r| r.to_vec()).collect(),
                                other => panic!("LIVE-REFUSED commit: {other:?}"),
                            };
                            if !plan.no_y {
                                match call(&tt, Req::Link(f128_bytes(&vec![F128::ZERO; 2 * tabs.list.len()]))) {
                                    Ok(Resp::Ok) => {}
                                    other => panic!("LIVE-REFUSED y: {other:?}"),
                                }
                            }
                            link_pts = Some(p.clone());
                            p
                        }
                    };
                    let e = st.extra_points(&pts);
                    cached = Some(e.clone());
                    e
                };
                // the firewall holds the device's root to the recheck's before the hook sends it (the recheck's leaves either way),
                // and keeps the extra points
                let mut hook = |root: &[u8]| -> Vec<Vec<F128>> {
                    let Some(g) = &firewall else { return hook(root) };
                    let root = g.lock().unwrap_or_else(PoisonError::into_inner).root(root);
                    let e = hook(&root);
                    g.lock().unwrap_or_else(PoisonError::into_inner).extras(&e);
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
                        _ => wt1.take().or_else(|| wt0.take()).or_else(|| Some(witness(st, &w0, device, host_units, mapped).0)),
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
                let proved = catch_unwind(AssertUnwindSafe(|| -> Result<(Vec<u8>, Value), String> {
                    if plan.gpu {
                        let wt: Arc<WitData> = if rep == 0 { wt_dev.take() } else { wt1.take() }.map(Arc::new)
                            .unwrap_or_else(|| shared0.get_or_insert_with(|| Arc::new(wt0.take().expect("a rep's witness"))).clone());
                        match zkrep.take() {
                            // M1 on the device, rep 0: the device part, then its replay and inner proof on a thread
                            Some(z) if rep == 0 && overlap => {
                                let (mut mch, sent) = zk_masking(st, lch.take().expect("rep 0's stream"), &z);
                                let tp = Instant::now();
                                let dev = gpu_device(st, &wt, lig, rep, &mut hook, salts, &z.level0,
                                                     &mut |c: &[u8], n: usize| firewalled_round(&firewall, &mut mch, c, n), native_sha, &firewall);
                                rep0_finish = Some(match &firewall {
                                    Some(g) => {
                                        let (bk, end) = firewalled_end(g, &mut mch, &mut hook, dev)?;
                                        let (t_up, g) = (tp.elapsed().as_secs_f64(), g.clone());
                                        flock_live::spawn(move || zk_finish_fw(&z, mch, tp, t_up, bk, &g, end))
                                    }
                                    None => {
                                        let (proof, cm, bk) = dev?;
                                        let (t_up, st2) = (tp.elapsed().as_secs_f64(), st.clone());
                                        flock_live::spawn(move || zk_finish(&st2, &z, proof, cm, mch, &sent, tp, t_up, bk))
                                    }
                                });
                                Ok((Vec::new(), Value::Null))
                            }
                            // rep 1 beside rep 0's finish: the device starts now, its first round waits for rep 0's last one and then
                            // opens rep 1's stream; otherwise (sequential) the rep's stream is open already
                            Some(z) => {
                                let mut first = rep0_finish.take();
                                let mut opened: Option<(zkv::Masking<Gate<LiveChallenger>>, zkv::Pads)> = lch.take().map(|l| zk_masking(st, l, &z));
                                let mut r0: Option<Result<(Vec<u8>, Value), String>> = None;
                                let open = |opened: &mut Option<(zkv::Masking<Gate<LiveChallenger>>, zkv::Pads)>,
                                            first: &mut Option<std::thread::JoinHandle<(Vec<u8>, Value)>>, r0: &mut Option<Result<(Vec<u8>, Value), String>>| {
                                    if opened.is_none() {
                                        let done = first.take().expect("rep 0's replay and inner proof").join().map_err(panic_msg);
                                        let stop = done.as_ref().err().cloned();
                                        *r0 = Some(done);
                                        if let Some(e) = stop {
                                            panic!("{e}");
                                        }
                                        let l = LiveChallenger::new(t.clone(), &tb.name, rep, &dom(rep)).unwrap_or_else(|e| panic!("LIVE-REFUSED open: {e}"));
                                        *opened = Some(zk_masking(st, l, &z));
                                    }
                                };
                                let tp = Instant::now();
                                let dev = catch_unwind(AssertUnwindSafe(|| {
                                    let mut rounds = |c: &[u8], n: usize| -> Vec<F128> {
                                        open(&mut opened, &mut first, &mut r0);
                                        firewalled_round(&firewall, &mut opened.as_mut().expect("rep 1's stream").0, c, n)
                                    };
                                    gpu_device(st, &wt, lig, rep, &mut hook, salts, &z.level0, &mut rounds, native_sha, &firewall)
                                }));
                                // a device that asked no round: the firewall opens rep 1's stream for the schedule it releases
                                if firewall.is_some() && dev.is_ok() {
                                    open(&mut opened, &mut first, &mut r0);
                                }
                                if let Some(h) = first.take() {
                                    r0 = Some(h.join().map_err(panic_msg));
                                }
                                // rep 0's stop comes first on the wire, so it is the session's
                                if let Some(r) = r0 {
                                    rep0_done = Some(r.unwrap_or_else(|e| panic!("{e}")));
                                }
                                let dev = dev.unwrap_or_else(|e| std::panic::resume_unwind(e));
                                match &firewall {
                                    Some(g) => {
                                        let (mut mch, _) = opened.take().expect("rep 1's stream opened");
                                        let (bk, end) = firewalled_end(g, &mut mch, &mut hook, dev)?;
                                        Ok(zk_finish_fw(&z, mch, tp, tp.elapsed().as_secs_f64(), bk, g, end))
                                    }
                                    None => {
                                        let (proof, cm, bk) = dev?;
                                        let (mch, sent) = opened.take().expect("rep 1's stream opened");
                                        Ok(zk_finish(st, &z, proof, cm, mch, &sent, tp, tp.elapsed().as_secs_f64(), bk))
                                    }
                                }
                            }
                            None => prove_gpu(st, &wt, lig, rep, &mut hook, lch.as_mut().expect("the rep's stream"), salts,
                                             reuse_tag.map(|t| (t << 8) | j as u64), native_sha),
                        }
                    } else if let Some(z) = &zkrep {
                        let wt = match (rep, wt_dev.as_ref()) {
                            (0, Some(w)) => w.clone(),
                            _ => own(&mut wt0, &mut wt1),
                        };
                        prove_cpu_zk(st, &wt, lig, rep, &mut hook, lch.as_mut().expect("the rep's stream"), z, &firewall)
                    } else {
                        Ok(prove_cpu(st, own(&mut wt0, &mut wt1), lig, &mut hook, lch.as_mut().expect("the rep's stream")))
                    }
                }));
                #[cfg(feature = "sha512")]
                if level0.is_some() {
                    flock_merkle::hm96::end();
                }
                // the firewall's refusal comes first: nothing after it was sent
                let firewall_why = firewall.as_ref().and_then(|g| g.lock().unwrap_or_else(PoisonError::into_inner).why.clone());
                let (p, bk) = proved.map_err(panic_msg).and_then(|r| r).map_err(|e| firewall_why.or_else(|| stop_why.clone()).unwrap_or(e))?;
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
            // a failed check (a failed worker's among them), a rep whose Lean frame did not answer ok at its end, or a constraint
            // that fails on the pads, stops the session at its Finish (§10.1): every round of the table left on schedule
            if let Some(g) = &firewall {
                let rows = buckets[table_buckets..].iter()
                    .filter_map(|b| b["constraints_failing_on_pads"].as_array().filter(|r| !r.is_empty()))
                    .map(|r| r.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).next();
                let why = g.lock().unwrap_or_else(PoisonError::into_inner).verdict()
                    .or_else(|| rows.map(|r| format!("FIREWALL-REFUSED: {}: the constraints {r} fail on a rep's pads", tb.name)));
                if let Some(w) = why {
                    stop_at_finish.get_or_insert(w);
                }
            }
            #[cfg(feature = "seed-injection")]
            if plan.early_proofs && j + 1 < tabs.list.len() && stop_at_finish.is_none() {
                for (rep, p) in proofs[early..].iter().enumerate() {
                    call(&t, Req::Proof { table: tb.name.clone(), rep: rep as u32, bytes: p.clone() })?;
                }
                early = proofs.len();
            }
        }
        match stop_at_finish {
            Some(w) => Err(w),
            None => Ok(()),
        }
    })();
    let e2e_s = t0.elapsed().as_secs_f64();
    let wait_e2e_s = t.lock().unwrap_or_else(PoisonError::into_inner).wait_s();
    Ok(Proved { t, log, tabs, plan: plan.clone(), hello, commit, sig, res, proofs, early, e2e_s, wait_e2e_s, witness_s, witness_prebuilt_s,
                prove_s, buckets, tp0 })
}

fn session_finish(p: Proved) -> (Value, Vec<Vec<u8>>) {
    let Proved { t, log, tabs, plan, hello, commit, sig, res, proofs, early, e2e_s, wait_e2e_s, witness_s, witness_prebuilt_s, prove_s,
                 buckets, tp0 } = p;
    let (st, inst) = (&tabs.list[0].st, &tabs.list[0].inst);
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
    // the barrier (the public-circuit ZK proof's §2.3, line 12): a stream's proof exists only after its last coin, and every
    // proof leaves here, after every table's reps, so none before the session's last coin
    for (k, p) in sent.iter().enumerate().skip(early) {
        call(&t, Req::Proof { table: tabs.list[k / 2].name.clone(), rep: (k % 2) as u32, bytes: p.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (ws, wr, calls) = t.lock().unwrap_or_else(PoisonError::into_inner).stats();
    let wait_s = t.lock().unwrap_or_else(PoisonError::into_inner).wait_s();
    // M0's seed opens at the end; under M2 every coin was checked against the commitment before it was used, and OS coins
    // commit to nothing
    let step0 = if verified(&v) && !is_zk(st) && seed_coins() { check_coin_seed(&commit, &hello, &v, &log.lock().unwrap_or_else(PoisonError::into_inner)) } else { Ok(()) };
    let v = if let Err(e) = &step0 { json!({"accepted": false, "why": e, "verdict": v}) } else { v };
    let c = &st.c;
    let (rows_per_vu, ands_per_vu) = c.sizes_per_vu();
    let mut live = json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "statement": c.statement(), "scheme": SCHEME,
               "prover": if plan.gpu {"flock-cuda-circuit"} else {"flock-cpu"}, "gpu": plan.gpu, "unit": c.name, "set": inst.header["set"],
               "subcircuit": inst.header["subcircuit"], "instances": st.n, "vus_per_block": c.g, "units": st.n * c.upv, "rows_per_vu": rows_per_vu,
               "ands_per_vu": ands_per_vu, "k_log": c.k_log, "nbl": st.nbl,
               "dense_m": st.m, "e2e_s": e2e_s, "witness_s": witness_s, "witness_prebuilt_s": witness_prebuilt_s, "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(),
               "buckets": buckets, "wait_s": wait_s, "wait_e2e_s": wait_e2e_s, "calls": calls, "wire_sent": ws, "wire_recv": wr,
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "sigma": hex(&sig), "roots": inst.header["frame_v3"]["roots"],
               "circuit_sha512": hex(&c.sha512), "statement_digest": hex(&st.digest), "public_sha512": hex(&inst.public_sha512),
               "unit_draw": if tabs.list.len() == 1 { inst.header.get("unit_draw").cloned() } else { tabs.draw.clone() },
               "tables": tabs.list.iter().map(|t| &t.name).collect::<Vec<_>>(),
               "identity": st.identity, "tp_s": tp0.elapsed().as_secs_f64(), "coin_seed": v["coin_seed"].clone(),
               "step0_checked": step0.is_ok() && verified(&v) && (is_zk(st) || seed_coins())});
    // past json!'s recursion limit in the literal above
    live["public_file_sha512"] = json!(hex(&inst.public_file_sha512));
    (live, proofs)
}

/// (case, expect accept, both reps must reject)
const CASES: &[(&str, bool, bool)] = &[
    ("honest", true, false),
    ("honest_again", true, false),
    ("mask_cells_random", true, false),
    ("output_claim_false", false, true),
    ("output_row_committed_false", false, true),
    ("digest_claim_false", false, true),
    ("row_ref_claim_false", false, true),
    ("public_input_claim_false", false, true),
    ("message_byte_differs_from_the_committed_row", false, true),
    ("unit_input_differs_from_its_message_bits", false, true),
    ("unit_internal_bit_flipped", false, true),
    ("chaining_value_forged", false, true),
    ("row_salt_forged", false, true),
    ("row_midstate_forged", false, true),
    ("padding_block_forged", false, true),
    ("row_fill_bit_set", false, true),
    ("row_padding_bit_forged_beside_its_bytes", false, true),
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
    ("typed_statement_altered_refused", false, false),
    ("private_cut_word_tampered", false, true),
    ("tail_swapped", false, true),
    ("tail_swapped_circuit_refused", false, false),
    ("coin_seed_opening_forged", false, false),
    ("opened_salt_altered", false, true),
    ("coin_seed_record_replays", true, false),
    ("record_replays_offline", true, false),
    ("lincheck_modes_agree", true, false),
    ("unit_draw_session", true, false),
    ("unit_draw_from_file", true, false),
    ("unit_draw_ignored_refused", false, false),
    ("prover_is_deterministic", true, false),
    ("gpu_proofs_match_cpu", true, false),
    ("gpu_paths_agree", true, false),
    ("verify_ahead_matches_serial", true, false),
    ("host_units_match_witness", true, false),
    ("native_sha_matches_eval64", true, false),
    ("region_word_shared_refused", false, false),
    ("zk_pads_not_the_committed_ones", false, true),
    ("zk_final_c_solved_after_the_batching_coins", false, true),
    ("zk_level0_folds_another_extra_lane", false, true),
    ("zk_level0_blinded_claim_forged", false, true),
    ("zk_level0_ybar_forged", false, true),
    ("zk_level0_extra_lane_sum_forged", false, true),
    ("zk_verifier_coin_off_its_commitment", false, false),
    ("zk_verifier_commits_beta_zero", false, false),
    ("zk_verifier_commits_outer_coin_one", false, false),
    ("zk_failing_witness_stops_before_tau", false, false),
    ("zk_session_relations_vary_with_the_seed", true, false),
    ("zk_audit_attributes_the_inner_proof", true, false),
    ("zk_dummy_witness_passes_the_region_claims", false, true),
    ("tables_two_in_one_session", true, false),
    ("tables_proofs_before_the_last_coin_detected", true, false),
    ("tables_unsplittable_draw_refused", false, false),
    ("zk_tables_share_no_randomness", true, false),
    ("zk_worker_cut_rc", false, false),
    ("zk_worker_cut_rc240", false, false),
    ("zk_worker_cut_panic", false, false),
    ("zk_worker_cut_silent", false, false),
    ("zk_worker_cut_decode", false, false),
    ("zk_worker_cut_tree", false, false),
    ("zk_worker_wire", false, false),
    ("zk_worker_finish", false, false),
    ("zk_worker_level0", false, false),
];

/// Cases that exercise the statement's parse or the non-ZK proof format only (M1's `--zk` selftest skips them).
const NOT_ZK: &[&str] = &["coin_seed_opening_forged", "coin_seed_record_replays", "relabelled_circuit_refused_at_load", "verifier_evaluated_tail_refused", "leaf_salts_from_prover_seed_refused",
                          "device_salts_match_host", "leaf_key_witness_refused", "tail_swapped_circuit_refused", "mask_cells_random",
                          "opened_salt_altered"];

fn skipped(st: &Stmt, name: &str) -> bool {
    let multi_block = st.c.ports.iter().any(|p| p.comps > 1);
    let cut = st.c.wires.iter().any(|w| w.dst.0 != st.c.sha_net && w.dst.0 != st.c.hm_net)
        || st.c.typed.as_ref().is_some_and(|t| t.inst.bind.iter().any(|b| !b.is_empty()));
    // a tail with no positive constant to forge (GEMM's: the zero accumulator and the cast) forges to itself: nothing to swap.
    // A forged file that can't be read is no reason to skip: the case fails, naming it (`forged_missing`)
    let forged = std::fs::read_to_string(arg("--forged", "circuit-forged.txt")).ok();
    let forged_same = forged.as_ref().is_some_and(|f| *f == std::fs::read_to_string(arg("--circuit", "")).unwrap_or_default());
    (name == "chaining_value_forged" && !multi_block)
        || (name == "row_fill_bit_set" && !st.c.ports.iter().any(|p| p.v2 && p.bits % 8 != 0))
        || (name == "row_padding_bit_forged_beside_its_bytes" && !st.c.ports.iter().any(|p| p.v2 && p.bytes() % 128 != 0))
        || (!cut && ["private_cut_word_tampered", "tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
        || (is_zk(st) && NOT_ZK.contains(&name))
        || (!is_zk(st) && name.starts_with("zk_"))
        || (forged_same && ["tail_swapped", "tail_swapped_circuit_refused"].contains(&name))
        || (name == "tail_swapped" && st.c.typed.is_some() && forged.is_some() && forged_instance(st).is_none())
        || (name == "public_input_claim_false" && st.c.public_inputs.is_empty())
}

/// A case that swaps the tail (`tail_swapped`, `tail_swapped_circuit_refused`) on a statement it isn't skipped on, when the
/// forged circuit can't be read: the reason its record fails with, in place of a skip that would hide an unstaged forgery.
fn forged_missing(name: &str) -> Option<String> {
    let path = arg("--forged", "circuit-forged.txt");
    (["tail_swapped", "tail_swapped_circuit_refused"].contains(&name) && std::fs::metadata(&path).is_err()).then(|| {
        format!("no forged circuit at {path}: stage circuit-forged.txt beside the circuit (`verity_flock.circuit.stage`) and pass --forged")
    })
}

fn plan_of(name: &str, gpu: bool) -> Plan {
    let h = Plan { gpu, ..Plan::default() };
    // a control whose witness or pads fail by design: its prover skips the self-check, so that the verifier gets its proofs
    let f = Plan { zk_self_check: zkv::SelfCheck::Skip, ..h.clone() };
    match name {
        "mask_cells_random" => Plan { mask_random: true, ..h },
        "output_row_committed_false" => f,
        "private_cut_word_tampered" => Plan { cut_flip: true, ..f },
        "tail_swapped" => Plan { tail_swapped: true, ..f },
        "message_byte_differs_from_the_committed_row" => Plan { message_flip: true, ..f },
        "unit_input_differs_from_its_message_bits" => Plan { input_flip: true, ..f },
        "unit_internal_bit_flipped" => Plan { unit_flip: true, ..f },
        "chaining_value_forged" => Plan { chain_forged: true, ..f },
        "row_salt_forged" => Plan { salt_forged: true, ..f },
        "row_midstate_forged" => Plan { midstate_forged: true, ..f },
        "padding_block_forged" => Plan { pad_forged: true, ..f },
        "row_fill_bit_set" => Plan { fill_forged: true, ..f },
        "row_padding_bit_forged_beside_its_bytes" => Plan { mixed_pad_forged: true, ..f },
        "coins_before_y" => Plan { no_y: true, ..h },
        "reps_with_different_witness" => Plan { rep1_other_witness: true, skip_level0_check: true, ..h },
        "reps_differ_prover_stops" => Plan { rep1_other_witness: true, ..h },
        "fast_profile_proofs" => Plan { fast: true, ..h },
        "opened_salt_altered" => Plan { salt_flip: true, ..h },
        "zk_pads_not_the_committed_ones" => Plan { zk_pad_mismatch: true, ..f },
        "zk_final_c_solved_after_the_batching_coins" => Plan { zk_pad_mismatch: true, zk_final_c_solved: true, ..f },
        "zk_level0_folds_another_extra_lane" => Plan { zk_l0_tamper: 1, ..f },
        "zk_level0_blinded_claim_forged" => Plan { zk_l0_tamper: 2, ..f },
        "zk_level0_ybar_forged" => Plan { zk_l0_tamper: 3, ..f },
        "zk_level0_extra_lane_sum_forged" => Plan { zk_l0_tamper: 4, ..f },
        _ => h,
    }
}

/// `--tmpl-unit F`: a templated statement's unit rows at their positions (`flock-tmpl-unit`, `tests/lean/TmplUnit.lean`).
fn load_composite(path: &str) -> Composite {
    let unit = Some(arg("--tmpl-unit", "")).filter(|p| !p.is_empty()).map(|p| std::fs::read_to_string(p).expect("--tmpl-unit"));
    Composite::parse_with(&std::fs::read_to_string(path).expect("circuit"), Some(&arg("--tables", "tables")), unit.as_deref())
        .unwrap_or_else(|e| refuse(e))
}

/// A row's `b || c` under its salt (the commitment scheme's own function: `b = x XOR M(key) y`, `c = SHA-512(salt_prefix || y)`).
fn row_bc(port: &Port, row: &[u8], salt: &[u8]) -> [u8; BC_BYTES] {
    let mut bc = row_bc_zero_salt(port, row);
    for (b, m) in bc[..64].iter_mut().zip(flock_merkle::hm96::mask(salt)) {
        *b ^= m;
    }
    let mut h = Sha512::new();
    h.update(flock_merkle::hm96::consts().salt_prefix);
    h.update(salt);
    bc[64..].copy_from_slice(&h.finalize());
    bc
}

/// The verifier's file with one public claim changed and its frame-v3 roots recomputed: a self-consistent, false statement. `ref`
/// (a shared-row file) points instance 0 at another row of a table, whose roots are unchanged; `row` (the prover's file too)
/// flips instance 0's first output word in its committed output row and recomputes the row's `b || c` under its salt.
fn false_statement(st: &Stmt, what: &str) -> Instances {
    let path = arg("--instances", "");
    let mut v = Instances::load(&path, &st.c).unwrap();
    let np = st.c.ports.len();
    match what {
        "row" => {
            let (n_in, words) = (st.c.n_in, st.c.ports[st.c.n_in].words);
            let salt = v.instance_salts(np).expect("the prover's file")[n_in * SALT_BYTES..(n_in + 1) * SALT_BYTES].to_vec();
            let at = match &v.tables {
                Some(t) => (0..n_in).map(|p| t.counts[p] * 2 * st.c.ports[p].words).sum(),
                None => 2 * st.c.in_words(),
            };
            let rows = match &mut v.tables {
                Some(t) => std::sync::Arc::make_mut(t.rows.as_mut().expect("the prover's file")),
                None => v.rows.as_mut().expect("the prover's file"),
            };
            rows[at] ^= 1;
            let bc = row_bc(&st.c.ports[n_in], &rows[at..at + 2 * words], &salt);
            let k = v.bc_index(0, n_in, np);
            v.bcs[k] = bc;
        }
        "output" => {
            let k = v.bc_index(0, st.c.n_in, np);
            v.bcs[k][0] ^= 1
        }
        "public" => {
            let q = st.c.leaves_public.iter().flatten().copied().find(|&q| q >= 0).expect("a public leaf") as usize;
            v.public[q] ^= 1;
        }
        "ref" => {
            let t = v.tables.as_mut().expect("a shared-row file");
            let p = (0..st.c.n_in).find(|&p| t.counts[p] > 1).expect("an input table of two rows or more");
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
    // a recorded case's false statement is the verifier's own file (DIR/<case>.pub.bin) under its real public digest, so the
    // record replays from files and a replay reaches the proof, not the session parameters
    let dir = arg("--record-dir", "");
    if !dir.is_empty() {
        let case = match what {
            "output" => "output_claim_false",
            "ref" => "row_ref_claim_false",
            "row" => "output_row_committed_false",
            "public" => "public_input_claim_false",
            _ => "digest_claim_false",
        };
        let mut bytes = serde_json::to_vec(&v.header).unwrap();
        bytes.push(b'\n');
        for bc in &v.bcs {
            bytes.extend_from_slice(bc);
        }
        if let Some(t) = &v.tables {
            for r in &t.refs {
                bytes.extend_from_slice(&r.to_le_bytes());
            }
        }
        for w in &v.public {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::create_dir_all(&dir).unwrap();
        let p = std::path::Path::new(&dir).join(format!("{case}.pub.bin"));
        std::fs::write(&p, &bytes).unwrap();
        v.public_sha512 = Instances::load(p.to_str().unwrap(), &st.c).unwrap().public_sha512;
        return v;
    }
    let mut h = Sha512::new();
    h.update(v.public_sha512);
    h.update(what.as_bytes());
    v.public_sha512 = h.finalize().into();
    v
}

/// A verdict as it would be were the statement's form one the soundness proof covers: a refused one's `without_refusal`
/// (`proved_scope`), else itself. `v` is a server's verdict or a session's result holding it; the selftest's checks of
/// anything but the refusal read this.
fn in_scope(v: &Value) -> &Value {
    let w = if v["without_refusal"].is_object() { &v["without_refusal"] } else { &v["verdict"]["without_refusal"] };
    if w.is_object() && v["why"].as_str().is_some_and(proved_scope::is_refusal) { w } else { v }
}

fn verified(v: &Value) -> bool {
    in_scope(v)["accepted"] == true
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
    if let Some(mode) = name.strip_prefix("zk_worker_cut_") {
        return worker_cut_case(name, mode, gpu);
    }
    if name == "zk_worker_wire" {
        return worker_wire_case(gpu);
    }
    if name == "zk_worker_finish" {
        return worker_finish_case(gpu);
    }
    if name == "zk_worker_level0" {
        return worker_level0_case(gpu);
    }
    if name == "lincheck_modes_agree" {
        return lincheck_modes_case(st, inst, gpu);
    }
    if name == "zk_audit_attributes_the_inner_proof" {
        return zk_audit_inner_case(st, inst, gpu);
    }
    if name == "gpu_proofs_match_cpu" {
        return gpu_cpu_case(st, inst);
    }
    if name == "gpu_paths_agree" {
        return gpu_paths_case(st, inst);
    }
    if name == "verify_ahead_matches_serial" {
        return verify_ahead_case(st, inst, gpu);
    }
    if name == "host_units_match_witness" {
        // the device witness's host units (a deep unit's slots, Stmt::slot_zab from their inputs) are the CPU witness's unit
        // slots word for word
        let w = wit(st, inst, &Plan::default(), false, ProverSeed::from_os(), 0);
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
        let w = wit(st, inst, &Plan::default(), false, ProverSeed::from_os(), 0);
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
    if ["zk_verifier_coin_off_its_commitment", "zk_verifier_commits_beta_zero", "zk_verifier_commits_outer_coin_one"].contains(&name) {
        // M2: V* answers rep 0's round 3 with a coin of its choosing under the committed coin's opening, or commits to β = 0 or
        // to a zerocheck outer coin r_0 = 1; the prover stops at that coin, before it sends anything that depends on it
        let (strategy, want) = if name.ends_with("beta_zero") || name.ends_with("outer_coin_one") {
            let v = VStar::new(st, inst, Strategy::Honest, [7; 32]);
            let view = v.view.clone();
            session(st, inst, shared(v), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
            let (view, rep0) = (view.lock().unwrap_or_else(PoisonError::into_inner), rep_stream(TABLE, 0));
            if name.ends_with("beta_zero") {
                (Strategy::ZeroBeta(beta_round(&view, &rep0).expect("rep 0's beta round")), "degenerate coin β = 0")
            } else {
                (Strategy::OuterOne(outer_round(&view, &rep0).expect("rep 0's outer round")), "zerocheck outer coin r_0 = 1")
            }
        } else {
            (Strategy::CoinAt(3), "do not open its step-0 commitment")
        };
        let v = VStar::new(st, inst, strategy, [7; 32]);
        let view = v.view.clone();
        let (r, _) = session(st, inst, shared(v), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let (s, _) = view_summary("real", name, 0, &view.lock().unwrap_or_else(PoisonError::into_inner), &rep_stream(TABLE, 0));
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
    if name == "zk_failing_witness_stops_before_tau" {
        // the prover checks before it speaks: an honest plan's prover on a witness with a flipped unit bit fails its batched
        // constraint and stops after the batching coins, so rep 0's stream ends with the round that opens them (its message
        // `[ρ, σ]`, after the inner proof's label) and no τ commitment, β or proof goes out
        let v = VStar::new(st, inst, Strategy::Honest, [7; 32]);
        let view = v.view.clone();
        let (r, _) = session(st, inst, shared(v), &Plan { gpu, unit_flip: true, ..Plan::default() }, ProverSeed::from_os());
        let (view, rep0) = (view.lock().unwrap_or_else(PoisonError::into_inner), rep_stream(TABLE, 0));
        let rounds: Vec<&Vec<u8>> = view.iter()
            .filter_map(|(q, _)| match q { Req::Round { stream, content, .. } if *stream == rep0 => Some(content), _ => None }).collect();
        let gamma_last = rounds.last().and_then(|c| parse_items(c).ok())
            .is_some_and(|it| matches!(it.as_slice(), [Item::Obs(v), Item::Framing(16), Item::Framing(16)] if v.len() == 2));
        let after_label = rounds.len() >= 2 && rounds[rounds.len() - 2].windows(zkv::LABEL_INNER.len()).any(|w| w == zkv::LABEL_INNER);
        let stopped = r["prover_stopped"].as_str().is_some_and(|x| x.contains("PROVER-STOPPED: the batched constraint fails"));
        let (no_tau, no_proof) = (beta_round(&view, &rep0).is_none(), !complete(&view));
        return json!({"case": name, "gpu": gpu, "expect": "the prover stops after the batching coins", "accepted": r["accepted"],
                      "pass": stopped && r["accepted"] != true && gamma_last && after_label && no_tau && no_proof,
                      "prover_stopped": r["prover_stopped"], "rep0_rounds": rounds.len(), "last_round_opens_the_batching_coins": gamma_last && after_label,
                      "no_tau_commitment": no_tau, "no_proof": no_proof});
    }
    #[cfg(feature = "seed-injection")]
    if name == "zk_session_relations_vary_with_the_seed" {
        // red-team-m1-zk F1: under one verifier coin seed, the two cross-rep functionals are the same for any two prover seeds
        // when the reps share level 0's extra lanes and their padding (a prover folding rep 0's pair in both reps, which the
        // verifier rejects); with a pair per rep they differ in every word. The level-1 functional exists only at the
        // positions both reps open at level 1, which the coins draw for each rep independently: on some statements
        // (RMSNormTriton_v5 at N 2048, under each of 16 tapes) the reps share none, and there is no level-1 relation to leak.
        // So level 1 is checked at every shared position, and each rep must have opened level-1 positions: an empty level-1
        // functional is then a disjoint draw, never a record that lost its queries
        let rel = |shared_pair: bool| -> Result<([Vec<F128>; 2], [usize; 3]), String> {
            let s0 = Arc::new(Mutex::new(server(st, inst)));
            fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
            let plan = Plan { gpu, zk_shared_pair: shared_pair, ..Plan::default() };
            let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &plan, ProverSeed::from_os());
            if verified(&v) != !shared_pair {
                return Err(format!("the session's verdict: {}", v["why"]));
            }
            let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
            let coins: Vec<Vec<F128>> = (0..2).map(|r| record_coins(&rec, &rep_stream(TABLE, r))).collect();
            session_relations_q(st, inst, &proofs, &coins, &record_points(&rec), if shared_pair { [0, 0] } else { [0, 1] })
        };
        let sessions = (|| Ok::<_, String>([rel(false)?, rel(false)?, rel(true)?, rel(true)?]))();
        return match sessions {
            Ok([(a, qa), (b, qb), (sa, qsa), (sb, qsb)]) => {
                let words_equal = |x: &[Vec<F128>; 2], y: &[Vec<F128>; 2]| -> Vec<usize> {
                    (0..2).map(|k| x[k].iter().zip(&y[k]).filter(|(u, w)| u == w).count()).collect()
                };
                let (eq, eq_shared) = (words_equal(&a, &b), words_equal(&sa, &sb));
                let (len, len_shared): (Vec<usize>, Vec<usize>) = ((0..2).map(|k| a[k].len()).collect(), (0..2).map(|k| sa[k].len()).collect());
                let lengths_agree = (0..2).all(|k| b[k].len() == len[k] && sb[k].len() == len_shared[k]);
                // the coins alone draw the positions: one tape, the same positions in both sessions of each kind
                let same_draws = qa == qb && qsa == qsb;
                let level1_read = [(&a, qa), (&b, qb), (&sa, qsa), (&sb, qsb)].iter()
                    .all(|(r, q)| q[0] > 0 && q[1] > 0 && (q[2] == 0 || !r[1].is_empty()));
                let pass = lengths_agree && same_draws && level1_read
                    && len[0] > 0 && eq[0] == 0 && len_shared[0] > 0 && eq_shared[0] == len_shared[0]
                    && eq[1] == 0 && eq_shared[1] == len_shared[1];
                json!({"case": name, "gpu": gpu, "expect": "the functionals differ in every word across two prover seeds, and are equal if the reps share a pair; level 1 at every level-1 position both reps opened",
                       "pass": pass, "f256_values": [len[0] / 2, len[1] / 2], "f128_words_equal": eq, "shared_pair_words_equal": eq_shared,
                       "shared_pair_words": len_shared, "level1_positions": [qa[0], qa[1]], "level1_shared_positions": qa[2],
                       "shared_pair_level1_shared_positions": qsa[2], "lengths_agree": lengths_agree, "same_draws": same_draws,
                       "level1_read": level1_read})
            }
            Err(e) => json!({"case": name, "gpu": gpu, "pass": false, "why": e}),
        };
    }
    if name.starts_with("tables_") || name == "zk_tables_share_no_randomness" {
        if gpu {
            return json!({"case": name, "pass": true, "skipped": "several tables per session: the CPU prover"});
        }
        let tabs = match session_tables(st, inst, None, 2) {
            Ok(t) => t,
            Err(e) => return json!({"case": name, "pass": true, "skipped": format!("no two-table split of this set: {e}")}),
        };
        if name == "zk_tables_share_no_randomness" {
            return tables_randomness_case(&tabs);
        }
        if name == "tables_unsplittable_draw_refused" {
            // both sides refuse, before Hello, a draw two tables cannot split: a Bernoulli law's (its count is random), and an
            // odd count, subset or stratified; the server also refuses them at configuration (`server_tables`)
            let n = inst.n as u64;
            let units = |k: u64| (0..k).collect::<Vec<_>>();
            let strat = |k: u64| json!({"law": "stratified", "population": n, "k": k,
                                        "strata": [{"template": "t", "units": [[0, n]], "k": k}], "units": units(k)});
            let bern = session_tables(st, inst, Some(&json!({"law": "bernoulli", "p": "1/1", "population": n, "units": units(n)})), 2).err();
            let odd = (n >= 3).then(|| session_tables(st, inst, Some(&json!({"law": "subset", "population": n, "k": 3, "units": units(3)})), 2).err());
            let odd_strat = (n >= 3).then(|| session_tables(st, inst, Some(&strat(3)), 2).err());
            let every = session_tables(st, inst, Some(&json!({"law": "subset", "population": n, "k": n, "units": units(n)})), 2).err();
            let stratified = (n >= 2).then(|| session_tables(st, inst, Some(&strat(2)), 2).err());
            let splits_not = |e: &Option<Option<String>>| e.as_ref().is_none_or(|e| e.as_deref().is_some_and(|w| w.contains("do not split")));
            let pass = bern.as_deref() == Some(TABLES_LAW) && splits_not(&odd) && splits_not(&odd_strat) && every.is_none()
                && stratified.as_ref().is_none_or(|e| e.is_none());
            return json!({"case": name, "expect": "a Bernoulli draw and an odd count refused; a subset draw of every unit and an even stratified draw split",
                          "bernoulli": bern, "odd": odd, "odd_stratified": odd_strat, "every_unit": every, "stratified": stratified, "pass": pass});
        }
        // the public-circuit ZK proof's J = 2: both tables' roots in one Commit, a coin subtree per table and rep, and every
        // proof after the session's last coin. A prover without the barrier sends table 0's proofs before table 1's coins: the
        // verifier accepts them, and the order check must not.
        let early = name.ends_with("_detected");
        #[cfg(not(feature = "seed-injection"))]
        if early {
            return json!({"case": name, "pass": true, "skipped": "a prover without the barrier is the test harness's (the seed-injection feature)"});
        }
        let s0 = Arc::new(Mutex::new(server_tables(st, inst, 2)));
        let order = Arc::new(Mutex::new(Vec::new()));
        #[cfg(feature = "seed-injection")]
        let plan = Plan { tables: 2, early_proofs: early, ..Plan::default() };
        #[cfg(not(feature = "seed-injection"))]
        let plan = Plan { tables: 2, ..Plan::default() };
        let (v, proofs) = session(st, inst, shared(Order { inner: InProc(s0.clone()), log: order.clone() }), &plan, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default();
        let roots = rec["link"]["root_b"].as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
        let barrier = barrier_holds(&order.lock().unwrap_or_else(PoisonError::into_inner));
        let accepted = verified(&v) && reps.len() == 4 && reps.iter().all(|r| r["ok"] == true) && roots == table_names(2) && proofs.len() == 4;
        return json!({"case": name, "expect": if early { "accepted, and the order check finds a proof before the last coin" } else { "accepted, every proof after the last coin" },
                      "accepted": v["accepted"], "why": v["why"], "prover_stopped": v["prover_stopped"], "committed_roots": roots,
                      "tables_m": tabs.list.iter().map(|t| t.st.m).collect::<Vec<_>>(), "reps_ok": reps.iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect::<Vec<_>>(),
                      "barrier_holds": barrier, "pass": accepted && barrier != early});
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
                      "pass": verified(&v) && honest.is_ok() && tampered.is_err(), "why": honest.err().or(tampered.err())});
    }
    if name == "record_replays_offline" {
        // G2 offline: the verifier's record and the two proofs re-verified by Server::from_record + finish (the reference of the
        // Lean verifier's agreement test); the same record with the reps' proofs swapped is refused
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let replay = |p: [&Vec<u8>; 2]| -> Result<Value, String> {
            let map = p.iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b.to_vec())).collect::<BTreeMap<_, _>>();
            let v: Box<dyn StatementVerifier> = Box::new(CircuitVerifier::one(st, inst));
            Ok(Server::from_record(config(st, inst), v, &rec, map)?.finish())
        };
        let (honest, swapped) = (replay([&proofs[0], &proofs[1]]), replay([&proofs[1], &proofs[0]]));
        let accepted = |r: &Result<Value, String>| matches!(r, Ok(x) if verified(x));
        // a retained round's bytes altered in the record: refused (they no longer hash to the round's msg_sha256)
        let mut bad = rec.clone();
        let m = bad["streams"][0]["rounds"][0]["msg"].as_str().unwrap_or("").to_string();
        bad["streams"][0]["rounds"][0]["msg"] = json!(format!("{}{}", if m.starts_with('0') { "1" } else { "0" }, m.get(1..).unwrap_or("")));
        let v2: Box<dyn StatementVerifier> = Box::new(CircuitVerifier::one(st, inst));
        let map = proofs.iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b.to_vec())).collect::<BTreeMap<_, _>>();
        let altered = Server::from_record(config(st, inst), v2, &bad, map.clone()).err();
        // the same record with its link.sigma flipped: refused (Σ is the verifier's own)
        let mut bad_sigma = rec.clone();
        let sg = bad_sigma["link"]["sigma"].as_str().unwrap_or("").to_string();
        bad_sigma["link"]["sigma"] = json!(format!("{}{}", if sg.starts_with('0') { "1" } else { "0" }, sg.get(1..).unwrap_or("")));
        let v3: Box<dyn StatementVerifier> = Box::new(CircuitVerifier::one(st, inst));
        let sigma_refused = Server::from_record(config(st, inst), v3, &bad_sigma, map).err();
        return json!({"case": name, "expect": "the honest record replays offline; swapped proofs, altered round bytes and another Σ do not",
                      "accepted": v["accepted"], "altered_bytes_refused": altered, "sigma_refused": sigma_refused,
                      "pass": verified(&v) && accepted(&honest) && !accepted(&swapped) && !m.is_empty() && altered.is_some()
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
        let mut srv = with_draw(server(st, inst), st, inst, flock_live::unit_draw::Law::Subset { k: 0 }, 1);
        srv.cfg.draw_law = None;
        srv.cfg.draw_given = Some(draw.clone());
        let s = Arc::new(Mutex::new(srv));
        let (v, _) = session(st, inst, shared(InProc(s.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
        let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let pass = verified(&v) && rec["unit_draw"] == draw;
        return json!({"case": name, "gpu": gpu, "accepted": v["accepted"], "pass": pass, "unit_draw": rec["unit_draw"], "why": v["why"]});
    }
    if name == "unit_draw_session" || name == "unit_draw_ignored_refused" {
        // the verifier draws one unit of the registered population after Register; the prover proves the drawn statement
        // (or, ignoring the draw, the population's, whose Σ is not the verifier's)
        let s = Arc::new(Mutex::new(with_draw(server(st, inst), st, inst, flock_live::unit_draw::Law::Subset { k: 1 }, 1)));
        let plan = Plan { gpu, ignore_draw: name.ends_with("refused"), ..Plan::default() };
        let (v, proofs) = session(st, inst, shared(InProc(s.clone())), &plan, ProverSeed::from_os());
        let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        let acc = verified(&v);
        let draw = rec["unit_draw"].clone();
        // offline: the record rebuilt under the drawn statement (U1), and refused under the population's
        let replay = if acc {
            let (ds, di) = drawn(st, inst, &draw).unwrap();
            let map: BTreeMap<(String, usize), Vec<u8>> = proofs.iter().enumerate().map(|(r, p)| ((TABLE.to_string(), r), p.clone())).collect();
            let ok = Server::from_record(config(&ds, &di), Box::new(CircuitVerifier::one(&ds, &di)), &rec, map.clone())
                .map(|mut x| x.finish()).map(|v| verified(&v));
            // under the population's statement Σ differs (the draw is under Σ); with the record's draw altered, U1
            let pop = Server::from_record(config(st, inst), Box::new(CircuitVerifier::one(st, inst)), &rec, map.clone()).err();
            let mut bad = rec.clone();
            bad["unit_draw"]["k"] = json!(draw["k"].as_u64().unwrap_or(0) + 1);
            let u1 = Server::from_record(config(&ds, &di), Box::new(CircuitVerifier::one(&ds, &di)), &bad, map).err();
            json!({"drawn_accepted": ok, "population_refused": pop, "altered_draw_refused": u1})
        } else { Value::Null };
        let pass = if name == "unit_draw_session" {
            acc && draw["units"].as_array().is_some_and(|u| u.len() == 1) && replay["drawn_accepted"] == json!({"Ok": true})
                && replay["population_refused"].as_str().is_some_and(|w| w.contains("R7"))
                && replay["altered_draw_refused"].as_str().is_some_and(|w| w.contains("U1"))
        } else {
            !acc && in_scope(&v)["why"].as_str().is_some_and(|w| w.contains("R7"))
        };
        return json!({"case": name, "gpu": gpu, "accepted": acc, "pass": pass, "unit_draw": draw, "why": v["why"], "replay": replay});
    }
    if name == "device_salts_match_host" {
        #[cfg(all(feature = "gpu", feature = "sha512"))]
        if gpu {
            let n = 1000usize;
            let x: Vec<u8> = (0..64 * n).map(|i| (i * 7 + 3) as u8).collect();
            let salts = flock_live::zk_hooks::LeafSalts::for_seed(&ProverSeed::from_os()).expand(1, n);
            let d = flock_live::gpu_circuit::device_leaves(&x, &salts);
            let ok = (0..n).all(|i| {
                let mut xi = flock_merkle::Hash::default();
                xi.as_flattened_mut().copy_from_slice(&x[64 * i..64 * (i + 1)]);
                d[64 * i..64 * (i + 1)] == *flock_merkle::hm96::leaf(&xi, &salts[192 * i..192 * (i + 1)]).as_flattened()
            });
            return json!({"case": name, "gpu": true, "expect": "the device's leaves from the host's salts equal hm96::leaf", "pass": ok});
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
        "output_claim_false" | "digest_claim_false" | "row_ref_claim_false" | "public_input_claim_false" => {
            let what = match name { "output_claim_false" => "output", "row_ref_claim_false" => "ref", "public_input_claim_false" => "public", _ => "digest" };
            let v = Arc::new(false_statement(st, what));
            let mut p = Instances::load(&arg("--instances", ""), &st.c).unwrap();
            p.public_sha512 = v.public_sha512;
            (st.clone(), Arc::new(p), v)
        }
        // the prover commits to a false output: its row and b || c agree, so only the row's copy of the unit's output refuses it
        "output_row_committed_false" => {
            let v = Arc::new(false_statement(st, "row"));
            (st.clone(), v.clone(), v)
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
    // drawn (`--draw`) only when the verifier's file is the prover's: a drawn statement's digest reads the file, so R7 would
    // refuse a case whose files differ before the check it is for
    let s = Arc::new(Mutex::new(if Arc::ptr_eq(&pinst, &vinst) { server(st, &vinst) }
                                else { Server::new(config(st, &vinst), Box::new(CircuitVerifier::one(st, &vinst))) }));
    let (v, proofs) = session(&pst, &pinst, shared(InProc(s.clone())), &plan, ProverSeed::from_os());
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap_or_else(PoisonError::into_inner).record_json();
    // a statement form the soundness proof does not cover: an honest session's verdict is the refusal, and only that refuses it
    let refusal = proved_scope::check(&st.c, &vinst, is_zk(st), 1).err();
    let want_acc = want.1 && refusal.is_none();
    record_case(st.c.statement(), name, want_acc, &v, &rec, &proofs);
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"], r["why"]])).collect();
    let both = reps.len() == 2 && reps.iter().all(|r| r[1] == false);
    let stopped = v["prover_stopped"].as_str().is_some_and(|x| x.contains("level-0 root"));
    let pass = acc == want_acc && verified(&v) == want.1 && (want_acc == want.1 || v["why"].as_str() == refusal.as_deref())
        && (!want.2 || both) && (name != "reps_differ_prover_stops" || stopped);
    json!({"case": name, "gpu": plan.gpu, "expect": if want_acc {"accept"} else {"reject"}, "refusal": refusal, "accepted": acc, "pass": pass,
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

/// The session verifier's lincheck three ways (`FC_LINCHECK`: upstream's flat fold, the structured `comb_partial`, both): one
/// honest session's record and proofs re-verified offline under each, then with rep 0's lincheck `z_partial` or first round
/// altered (re-encoded: the replay reports the changed messages and keeps issuing the recorded coins, so the lincheck's own
/// final check decides). Every mode accepts the honest proofs and rejects each alteration with the same reason. The honest proofs
/// are verified three times, the modes interleaved, and each verification's `verify_s` and per-rep phases are kept. Under
/// `--zk` the lincheck is the zk replay's (`zk_veil::replay` folds its own comb; `FC_LINCHECK` is not read), so there is one
/// mode: the proofs are re-encoded with their `InnerProof`, and each alteration must be refused by the inner proof's batched
/// constraint.
fn lincheck_modes_case(st: &Arc<Stmt>, inst: &Arc<Instances>, gpu: bool) -> Value {
    let s0 = Arc::new(Mutex::new(server(st, inst)));
    let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &Plan { gpu, ..Plan::default() }, ProverSeed::from_os());
    let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
    if !verified(&v) || proofs.len() != 2 {
        return json!({"case": "lincheck_modes_agree", "accepted": v["accepted"], "pass": false, "why": v["why"]});
    }
    let zk = is_zk(st);
    let alter = |f: &dyn Fn(&mut R1csProofLigerito)| -> Vec<Vec<u8>> {
        let mut ps = proofs.clone();
        if zk {
            let (c, mut p, i): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(&ps[0]).unwrap();
            f(&mut p);
            ps[0] = bincode::serialize(&(c, p, i)).unwrap();
        } else {
            let (c, mut p): (Commitment, R1csProofLigerito) = bincode::deserialize(&ps[0]).unwrap();
            f(&mut p);
            ps[0] = bincode::serialize(&(c, p)).unwrap();
        }
        ps
    };
    let reencoded = alter(&|_| {}) == proofs;
    let sets = [("honest", proofs.clone()), ("honest", proofs.clone()), ("honest", proofs.clone()),
                ("z_partial", alter(&|p| p.lincheck.z_partial[3] += F128::ONE)), ("round", alter(&|p| p.lincheck.rounds[0].0 += F128::ONE))];
    // the record binds each proof's digest: an altered proof's is written in, so that the verifier, not the record, judges it
    let cfg = config(st, inst);
    let rec_for = |ps: &[Vec<u8>]| {
        let mut r = rec.clone();
        if let Some(ds) = r[format!("proof_{}", cfg.dk())].as_array_mut() {
            for d in ds.iter_mut().filter(|d| d["stream"] == rep_stream(TABLE, 0)) {
                d[cfg.dk()] = json!(hex(&cfg.dg(&ps[0])));
            }
        }
        r
    };
    let modes: &[sverify::Lincheck] =
        if zk { &[sverify::Lincheck::Partial] } else { &[sverify::Lincheck::Flat, sverify::Lincheck::Partial, sverify::Lincheck::Both] };
    let name = |m: sverify::Lincheck| if zk { "zk replay".to_string() } else { format!("{m:?}") };
    let runs: Vec<(&str, Vec<Value>)> = sets.iter().map(|(what, ps)| (*what, modes.iter().map(|&m| {
        let map = ps.iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b.clone())).collect::<BTreeMap<_, _>>();
        let ver = CircuitVerifier { lincheck: m, ..CircuitVerifier::one(st, inst) };
        match Server::from_record(config(st, inst), Box::new(ver), &rec_for(ps), map) {
            Ok(mut s) => {
                let f = s.finish();
                let reps = f["reps"].as_array().cloned().unwrap_or_default();
                json!({"mode": name(m), "accepted": f["accepted"], "why": f["why"], "without_refusal": f["without_refusal"],
                       "verify_s": f["verify_s"], "phases": reps.iter().map(|r| r["timing"]["phases"].clone()).collect::<Vec<_>>()})
            }
            Err(e) => json!({"mode": name(m), "accepted": false, "why": e}),
        }
    }).collect())).collect();
    let all = |what: &str, ok: &dyn Fn(&Value) -> bool| runs.iter().filter(|r| r.0 == what).all(|r| r.1.iter().all(ok));
    let same_why = runs.iter().all(|r| r.1.iter().all(|x| in_scope(x)["why"] == in_scope(&r.1[0])["why"]));
    let refusal = if zk { "zk inner: the batched constraint fails" } else { "sumcheck-final" };
    let lincheck_rejects = |x: &Value| !verified(x) && in_scope(x)["why"].as_str().is_some_and(|w| w.contains(refusal));
    let pass = reencoded && same_why && all("honest", &verified) && all("z_partial", &lincheck_rejects)
        && all("round", &lincheck_rejects);
    let expect = if zk {
        "the zk replay: honest proofs accepted, altered lincheck proofs refused by the inner proof's batched constraint"
    } else {
        "flat, partial and both: the same verdict on honest and altered lincheck proofs"
    };
    json!({"case": "lincheck_modes_agree", "expect": expect,
           "gpu": gpu, "accepted": v["accepted"], "reencoded_identical": reencoded, "same_why": same_why,
           "runs": runs.iter().map(|(w, r)| json!({"proofs": w, "modes": r})).collect::<Vec<_>>(), "pass": pass})
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
        (verified(&v), proofs, Sha512::digest(t.to_string().as_bytes()).to_vec())
    };
    let (a, b, c) = (run([1; 32]), run([1; 32]), run([2; 32]));
    let same = a.1 == b.1 && a.2 == b.2;
    // `--expect-transcript HEX`: another build's digest (the one-table session before a change: byte-identical, or not)
    let pin = arg("--expect-transcript", "");
    let pinned = pin.is_empty() || pin == hex(&a.2);
    json!({"case": "prover_is_deterministic", "expect": "equal seeds and coins give byte-identical proofs and transcripts", "gpu": gpu,
           "accepted": a.0 && b.0, "proofs_equal": a.1 == b.1, "transcripts_equal": a.2 == b.2, "other_seed_equal": a.1 == c.1 && a.2 == c.2,
           "transcript_sha256": hex(&a.2), "transcript_pinned": (!pin.is_empty()).then_some(pinned), "pass": a.0 && b.0 && c.0 && same && pinned})
}

/// Every round and proof request in the order the prover sends them (`round STREAM`, `proof STREAM`), for the barrier's order
/// check.
struct Order<T: Transport> {
    inner: T,
    log: Arc<Mutex<Vec<String>>>,
}

impl<T: Transport> Transport for Order<T> {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let at = match &req {
            Req::Round { stream, .. } => Some(format!("round {stream}")),
            Req::Proof { table, rep, .. } => Some(format!("proof {}", rep_stream(table, *rep as usize))),
            _ => None,
        };
        if let Some(a) = at {
            self.log.lock().unwrap_or_else(PoisonError::into_inner).push(a);
        }
        self.inner.call(req)
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.inner.stats()
    }
    fn wait_s(&self) -> f64 {
        self.inner.wait_s()
    }
}

/// The barrier on the wire: rounds and proofs were sent, and every proof after the last round of every stream.
fn barrier_holds(order: &[String]) -> bool {
    match (order.iter().rposition(|x| x.starts_with("round ")), order.iter().position(|x| x.starts_with("proof "))) {
        (Some(last_round), Some(first_proof)) => last_round < first_proof,
        _ => false,
    }
}

/// A request that reaches the verifier, as its kind, its stream and its sizes (never its bytes): what a session shows the
/// challenger beyond the values themselves.
fn seen(req: &Req) -> Value {
    let sizes = |b: &[(String, Vec<u8>)]| b.iter().map(|(k, v)| json!([k, v.len()])).collect::<Vec<_>>();
    match req {
        Req::Hello(h) => json!(["hello", h.len()]),
        Req::Link(y) => json!(["link", y.len()]),
        Req::Commit { root_f, roots, publics } => json!(["commit", root_f.len(), sizes(roots), sizes(publics)]),
        Req::Open { stream, parent, label, at } => json!(["open", stream, parent, label.len(), at]),
        Req::Round { stream, content, n } => json!(["round", stream, content.len(), n]),
        Req::Proof { table, rep, bytes } => json!(["proof", rep_stream(table, *rep as usize), bytes.len()]),
        Req::Finish => json!(["finish"]),
        Req::Ping(_) => json!(["ping"]),
        Req::Prime { state, n } => json!(["prime", state.len(), n]),
        Req::Register => json!(["register"]),
    }
}

/// `FC_WIRE_TAMPER=MODE`: the prover's frames to the firewall's session loop doctored on the wire in one named way, each of
/// which the loop drops or refuses (live PROTOCOL.md §10.4): `round-trailing` puts bytes after every `Round`'s count,
/// `commit-doctored` sends the `Commit` with a `root_F` that is not Σ, a public word and bytes after it, `proof-worker`
/// sends each `Proof` as one byte of junk, and `unknown` sends a frame of tag 99 before `Finish` (its answer kept in
/// `refused`).
struct Wire {
    inner: TcpTransport,
    mode: String,
    refused: Arc<Mutex<Option<String>>>,
}

impl Transport for Wire {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let mut b = match (&req, self.mode.as_str()) {
            (Req::Commit { roots, .. }, "commit-doctored") => {
                let mut b = Req::Commit { root_f: vec![0xab; 63], roots: roots.clone(), publics: vec![("x".into(), vec![1; 16])] }.encode();
                b.extend_from_slice(b"trailing");
                b
            }
            (Req::Proof { table, rep, .. }, "proof-worker") => Req::Proof { table: table.clone(), rep: *rep, bytes: vec![0x5a] }.encode(),
            _ => req.encode(),
        };
        if self.mode == "round-trailing" && matches!(req, Req::Round { .. }) {
            b.extend_from_slice(b"trailing");
        }
        if self.mode == "unknown" && matches!(req, Req::Finish) {
            let r = self.inner.call_raw(&[99, 1, 2, 3]);
            *self.refused.lock().unwrap_or_else(PoisonError::into_inner) = Some(format!("{r:?}"));
        }
        self.inner.call_raw(&b)
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.inner.stats()
    }
    fn wait_s(&self) -> f64 {
        self.inner.wait_s()
    }
}

/// The selftest's own process for one firewalled session (`--only zk_worker_session`), its controls from this process's
/// environment (`FC_FIREWALL`, `FC_FIREWALL_TAMPER`, `FC_FIREWALL_CUT_AT`, `FC_WIRE_TAMPER`): the session goes through the
/// Lean session loop as `prove`'s does ([`session_addr`]), to a verifier of this process's on loopback, and the record is the
/// session's outcome and every request that reached the verifier, behind the loop ([`seen`]; a frame that is not its
/// request's own encoding as `["noncanonical", …]`, one that does not decode as `["undecodable", LEN]`, and a refusal of the
/// verifier's after its request as `["refused", WHY]`).
const WORKER_SESSION: &str = "zk_worker_session";

fn worker_session(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Value {
    use std::io::{BufReader, BufWriter};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let verifier = l.local_addr().unwrap().to_string();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (mut s, at) = (server(st, inst), log.clone());
    // the verifier logs each request before it answers it, so every request of the session is logged by the time the
    // session's last answer reaches the prover
    flock_live::spawn(move || {
        let Ok((conn, _)) = l.accept() else { return };
        conn.set_nodelay(true).unwrap();
        let (mut r, mut w) = (BufReader::new(conn.try_clone().unwrap()), BufWriter::new(conn));
        while let Ok(b) = read_frame(&mut r) {
            let resp = match Req::decode(&b) {
                Ok(req) => {
                    let x = if req.encode() == b { seen(&req) } else { json!(["noncanonical", seen(&req), b.len()]) };
                    at.lock().unwrap_or_else(PoisonError::into_inner).push(x);
                    s.handle(req)
                }
                Err(e) => {
                    at.lock().unwrap_or_else(PoisonError::into_inner).push(json!(["undecodable", b.len()]));
                    Resp::Refused(e)
                }
            };
            if let Resp::Refused(why) = &resp {
                at.lock().unwrap_or_else(PoisonError::into_inner).push(json!(["refused", why.chars().take(3000).collect::<String>()]));
            }
            let done = matches!(resp, Resp::Verdict(_));
            if write_frame(&mut w, &resp.encode()).is_err() || done {
                break;
            }
        }
    });
    let plan = Plan::default();
    FINISH_REPS.store(2 * table_names(plan.tables).len(), std::sync::atomic::Ordering::SeqCst);
    let (addr, frames) = session_addr(st, inst, table_names(plan.tables).len(), &verifier, None).unwrap_or_else(|e| refuse(e));
    let plan = Plan { lean_session: frames.is_some(), lean_frames: frames, ..plan };
    let tcp = TcpTransport::connect(&addr).unwrap_or_else(|e| refuse(format!("the firewall's session loop at {addr}: {e}")));
    let wire = std::env::var("FC_WIRE_TAMPER").unwrap_or_default();
    let refused = Arc::new(Mutex::new(None));
    let t = if wire.is_empty() { shared(tcp) } else { shared(Wire { inner: tcp, mode: wire, refused: refused.clone() }) };
    let (v, _) = session(st, inst, t, &plan, ProverSeed::from_os());
    let seen = std::mem::take(&mut *log.lock().unwrap_or_else(PoisonError::into_inner));
    let device_rounds = v["buckets"].as_array().and_then(|b| b.iter().filter_map(|x| x["firewall_device_rounds"].as_u64()).max());
    let refused = refused.lock().unwrap_or_else(PoisonError::into_inner).clone();
    json!({"accepted": v["accepted"], "why": v["why"], "prover_stopped": v["prover_stopped"], "device_rounds": device_rounds, "seen": seen,
           "wire_refused": refused, "finish_refused": FINISH_REFUSED.lock().unwrap_or_else(PoisonError::into_inner).clone(),
           "level0_refused": LEVEL0_REFUSED.lock().unwrap_or_else(PoisonError::into_inner).clone()})
}

/// One firewalled CPU session in a process of its own ([`WORKER_SESSION`]), its controls `env` and none other: its record.
fn worker_subsession(env: &[(&str, String)]) -> Value {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.arg("selftest").args(if zk_mode() { &["--zk"][..] } else { &[] })
        .args(["--instances", &arg("--instances", ""), "--circuit", &arg("--circuit", ""), "--tables", &arg("--tables", "tables"),
               "--only", WORKER_SESSION])
        .env("FC_FIREWALL", "1").env("FC_HOST_WITNESS", "1").env_remove("FC_FIREWALL_TAMPER").env_remove("FC_FIREWALL_CUT_AT")
        .env_remove("FC_FIREWALL_DEADLINE_S").env_remove("FC_WIRE_TAMPER").env_remove("FC_FINISH_TAMPER")
        .env_remove("FC_LEVEL0_TAMPER");
    // the verifier's draw, as this process's server has it ([`serving`])
    for flag in ["--draw", "--draw-law", "--draw-file"] {
        let v = arg(flag, "");
        if !v.is_empty() {
            cmd.args([flag, &v]);
        }
    }
    cmd.envs(env.iter().map(|(k, v)| (*k, v.as_str())));
    let out = cmd.output().expect("spawn a firewalled session");
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().find_map(|l| l.strip_prefix("NEG\t")).and_then(|l| serde_json::from_str(l).ok()).unwrap_or_else(|| {
        let err = String::from_utf8_lossy(&out.stderr);
        json!({"error": format!("no session record (exit {:?}): {}", out.status.code(), err.lines().rev().take(5).collect::<Vec<_>>().join(" | "))})
    })
}

/// `zk_worker_wire` (live PROTOCOL.md §10.4): every frame that reaches the verifier through the firewall's session loop is
/// the loop's own encoding, whatever bytes the prover put on the wire. Beside an honest firewalled CPU session, one session
/// per `FC_WIRE_TAMPER` mode, each in a process of its own: bytes after every `Round`, a doctored `Commit` (`root_F` not Σ, a
/// public word, bytes after it) and the worker's own `Proof`s (junk) are dropped, the session accepted and the verifier's
/// every request the honest session's, canonical and at its sizes; a frame of an unknown tag is refused, and the verifier
/// gets the honest session's requests up to its proofs, then nothing.
fn worker_wire_case(gpu: bool) -> Value {
    let name = "zk_worker_wire";
    if gpu {
        return json!({"case": name, "pass": true, "skipped": "the wire's controls run on the CPU device"});
    }
    let lean = flock_live::firewall::lean_frame_exe();
    if !lean.exists() {
        return json!({"case": name, "pass": true, "skipped": format!("no Lean frame at {} (FC_FIREWALL_LEAN)", lean.display())});
    }
    let seen = |v: &Value| v["seen"].as_array().cloned().unwrap_or_default();
    let kind = |x: &Value| x[0].as_str().unwrap_or("").to_string();
    let honest = worker_subsession(&[]);
    let h = seen(&honest);
    let honest_ok = honest["accepted"] == true && h.iter().filter(|x| kind(x) == "proof").count() > 0
        && h.iter().all(|x| kind(x) != "noncanonical" && kind(x) != "undecodable");
    let before_proofs: Vec<Value> = h.iter().take_while(|x| kind(x) != "proof").cloned().collect();
    let modes: Vec<Value> = ["round-trailing", "commit-doctored", "proof-worker", "unknown"].iter().map(|&m| {
        let v = worker_subsession(&[("FC_WIRE_TAMPER", m.to_string())]);
        let s = seen(&v);
        let (want, refused) = if m == "unknown" {
            (&before_proofs, v["wire_refused"].as_str().is_some_and(|r| r.contains("a request the session loop does not serve")))
        } else {
            (&h, v["wire_refused"].is_null())
        };
        let accepted_ok = if m == "unknown" { v["accepted"] != true } else { v["accepted"] == true };
        let first_miss = (0..want.len().max(s.len())).find(|&i| want.get(i) != s.get(i));
        json!({"mode": m, "ok": honest_ok && accepted_ok && refused && first_miss.is_none(), "accepted": v["accepted"],
               "why": v["why"].as_str().unwrap_or("").chars().take(160).collect::<String>(), "wire_refused": v["wire_refused"],
               "error": v["error"], "requests": s.len(),
               "first_miss": first_miss.map(|i| json!({"at": i, "honest": want.get(i), "wire": s.get(i)}))})
    }).collect();
    json!({"case": name, "expect": "each doctored frame dropped (the honest session's requests, accepted) or refused (unknown)",
           "honest": {"accepted": honest["accepted"], "error": honest["error"], "requests": h.len()},
           "modes": modes, "pass": honest_ok && modes.iter().all(|c| c["ok"] == true)})
}

/// `zk_worker_finish` (live PROTOCOL.md §10.4): the proof files that leave are the loop's finish's on its own frames' state,
/// whatever the host's rep requests carry. Beside an honest firewalled CPU session, one session per `FC_FINISH_TAMPER` mode,
/// each in a process of its own, whose last rep request to the Lean finish carries zero pads, a chosen passed value, a
/// chosen coin or a chosen `final_c`: the finish refuses it (the request is not the frame's, or no live inner proof answers
/// it), no proof leaves, and the verifier gets the honest session's every request but its proofs, `Finish`
/// included, and does not accept.
fn worker_finish_case(gpu: bool) -> Value {
    let name = "zk_worker_finish";
    if gpu {
        return json!({"case": name, "pass": true, "skipped": "the finish's controls run on the CPU device"});
    }
    let lean = flock_live::firewall::lean_frame_exe();
    if !lean.exists() {
        return json!({"case": name, "pass": true, "skipped": format!("no Lean frame at {} (FC_FIREWALL_LEAN)", lean.display())});
    }
    let seen = |v: &Value| v["seen"].as_array().cloned().unwrap_or_default();
    let kind = |x: &Value| x[0].as_str().unwrap_or("").to_string();
    let honest = worker_subsession(&[]);
    let h = seen(&honest);
    let honest_ok = honest["accepted"] == true && h.iter().filter(|x| kind(x) == "proof").count() > 0 && honest["finish_refused"].is_null();
    let want: Vec<Value> = h.iter().filter(|x| kind(x) != "proof").cloned().collect();
    let modes: Vec<Value> = [("pads-zero", "its pads are not the frame's draws"), ("vals-chosen", "its passed values are not the ones the frame took"),
                             ("coins", "its coins are not the ones the stream answered"), ("final-c", "computed live")].iter().map(|&(m, why)| {
        let v = worker_subsession(&[("FC_FINISH_TAMPER", m.to_string())]);
        let s: Vec<Value> = seen(&v).into_iter().filter(|x| kind(x) != "refused").collect();
        let refused = v["finish_refused"].as_str().is_some_and(|r| r.contains(why));
        let first_miss = (0..want.len().max(s.len())).find(|&i| want.get(i) != s.get(i));
        json!({"mode": m, "ok": honest_ok && v["accepted"] != true && refused && first_miss.is_none(), "accepted": v["accepted"],
               "why": v["why"].as_str().unwrap_or("").chars().take(160).collect::<String>(), "finish_refused": v["finish_refused"],
               "error": v["error"], "requests": s.len(),
               "first_miss": first_miss.map(|i| json!({"at": i, "honest": want.get(i), "tampered": s.get(i)}))})
    }).collect();
    json!({"case": name, "expect": "each doctored rep request refused by the loop's finish, no proof sent, the honest session's other requests",
           "honest": {"accepted": honest["accepted"], "error": honest["error"], "requests": h.len()},
           "modes": modes, "pass": honest_ok && modes.iter().all(|c| c["ok"] == true)})
}

/// `zk_worker_level0` (live PROTOCOL.md §10.4): the level 0 a session binds is the loop's one draw for each table, never one
/// the prover picks among several. Beside an honest firewalled CPU session, one session per `FC_LEVEL0_TAMPER` mode, each in
/// a process of its own, whose table 0 asks the loop's level 0 for one more after its own: for table 0 again (`second`), or
/// for a table the session does not have (`past`). The loop refuses it, nothing more is drawn, and the session goes on on the
/// first draw: the verifier gets the honest session's every request, at its sizes, and accepts.
fn worker_level0_case(gpu: bool) -> Value {
    let name = "zk_worker_level0";
    if gpu {
        return json!({"case": name, "pass": true, "skipped": "the level-0 controls run on the CPU device"});
    }
    let lean = flock_live::firewall::lean_frame_exe();
    if !lean.exists() {
        return json!({"case": name, "pass": true, "skipped": format!("no Lean frame at {} (FC_FIREWALL_LEAN)", lean.display())});
    }
    let seen = |v: &Value| v["seen"].as_array().cloned().unwrap_or_default();
    let kind = |x: &Value| x[0].as_str().unwrap_or("").to_string();
    let honest = worker_subsession(&[]);
    let h = seen(&honest);
    let honest_ok = honest["accepted"] == true && h.iter().filter(|x| kind(x) == "proof").count() > 0 && honest["level0_refused"].is_null();
    let past = format!("a level-0 request for table {LEVEL0_PAST} of a session with ");
    let modes: Vec<Value> = [("second", "a second level-0 request for table 0 of this session"), ("past", past.as_str())].iter().map(|&(m, why)| {
        let v = worker_subsession(&[("FC_LEVEL0_TAMPER", m.to_string())]);
        let s = seen(&v);
        let refused = v["level0_refused"].as_str().is_some_and(|r| r.contains(why));
        let first_miss = (0..h.len().max(s.len())).find(|&i| h.get(i) != s.get(i));
        json!({"mode": m, "ok": honest_ok && v["accepted"] == true && refused && first_miss.is_none(), "accepted": v["accepted"],
               "why": v["why"].as_str().unwrap_or("").chars().take(160).collect::<String>(), "level0_refused": v["level0_refused"],
               "error": v["error"], "requests": s.len(),
               "first_miss": first_miss.map(|i| json!({"at": i, "honest": h.get(i), "tampered": s.get(i)}))})
    }).collect();
    json!({"case": name, "expect": "each extra level-0 request refused by the loop, the session on its one draw: the honest session's requests, accepted",
           "honest": {"accepted": honest["accepted"], "error": honest["error"], "requests": h.len()},
           "modes": modes, "pass": honest_ok && modes.iter().all(|c| c["ok"] == true)})
}

/// `zk_worker_cut_MODE` (PROTOCOL.md §10.1): a worker that fails in one way (`FC_FIREWALL_TAMPER=cut-MODE`) at its table's device
/// round `k` never chooses where the session ends. Beside an honest firewalled session, one session per `k` (early, half way and
/// the last of the honest session's `firewall_device_rounds`; `cut-tree` draws its trees in the opening phases, so it takes the
/// first and the half-way round), each a CPU
/// session in a process of its own: the verifier gets the honest session's every request up to its `Finish`, in its order and
/// at its sizes (the rounds, the Commit, the Link: the same bytes and public bytes), no proof, then `Finish`, which reports the
/// firewall's stop. A worker given up gets no coins, and a silent one ticks the firewall's injected clock: no wall clock.
fn worker_cut_case(name: &str, mode: &str, gpu: bool) -> Value {
    if gpu {
        return json!({"case": name, "pass": true, "skipped": "a worker's cut is the CPU device's control"});
    }
    let lean = flock_live::firewall::lean_frame_exe();
    if !lean.exists() {
        return json!({"case": name, "pass": true, "skipped": format!("no Lean frame at {} (FC_FIREWALL_LEAN)", lean.display())});
    }
    let run = |tamper: Option<(usize, &str)>| -> Value {
        let env = tamper.map(|(k, t)| vec![("FC_FIREWALL_TAMPER", format!("cut-{t}")), ("FC_FIREWALL_CUT_AT", k.to_string())]);
        worker_subsession(&env.unwrap_or_default())
    };
    let seen = |v: &Value| v["seen"].as_array().cloned().unwrap_or_default();
    let kind = |x: &Value| x[0].as_str().unwrap_or("").to_string();
    let count = |s: &[Value], k: &str| s.iter().filter(|x| kind(x) == k).count();
    let n = |x: &Value| x.as_u64().unwrap_or(0);
    let round_bytes = |s: &[Value]| s.iter().filter(|x| kind(x) == "round").map(|x| n(&x[2])).sum::<u64>();
    // Hello's, Link's, Commit's (root_F, each table's root and public words) and each stream's label
    let public_bytes = |s: &[Value]| s.iter().map(|x| match kind(x).as_str() {
        "hello" | "link" => n(&x[1]),
        "commit" => n(&x[1]) + [&x[2], &x[3]].iter().flat_map(|l| l.as_array().into_iter().flatten()).map(|p| n(&p[1])).sum::<u64>(),
        "open" => n(&x[3]),
        _ => 0,
    }).sum::<u64>();
    let honest = run(None);
    let h = seen(&honest);
    let rounds = count(&h, "round");
    // the device's rounds, which FC_FIREWALL_CUT_AT counts: the stream's rounds less those the host sends itself (the inner proof's)
    let dev = honest["device_rounds"].as_u64().unwrap_or(0) as usize;
    let honest_ok = honest["accepted"] == true && count(&h, "proof") == 2 && h.last().map(kind).as_deref() == Some("finish") && dev > 2;
    let want: Vec<Value> = h.iter().filter(|x| kind(x) != "proof").cloned().collect();
    let ks: Vec<usize> = if !honest_ok { vec![] } else if mode == "tree" { vec![0, dev / 2] } else { vec![2, dev / 2, dev - 1] };
    // a tree one leaf larger is the Lean frame's stop alone (the opening round it holds differs); every other cut Rust's
    let cuts: Vec<Value> = ks.iter().map(|&k| {
        let v = run(Some((k, mode)));
        let s = seen(&v);
        let stopped = v["prover_stopped"].as_str().unwrap_or("");
        let why = v["why"].as_str().unwrap_or("");
        let first_miss = (0..want.len().max(s.len())).find(|&i| want.get(i) != s.get(i));
        let refused = if mode == "tree" { stopped.is_empty() && why.contains("PROVER-STOPPED: the Lean frame stopped table 0 rep ") }
                      else { stopped.starts_with("FIREWALL-REFUSED: the device failed: ") };
        let ok = v["accepted"] == false && refused && first_miss.is_none();
        json!({"k": k, "ok": ok, "accepted": v["accepted"], "prover_stopped": stopped.chars().take(160).collect::<String>(),
               "why": why.chars().take(160).collect::<String>(), "error": v["error"],
               "rounds": count(&s, "round"), "round_bytes": round_bytes(&s), "public_bytes": public_bytes(&s), "proofs": count(&s, "proof"),
               "requests": s.len(), "last": s.last(), "first_miss": first_miss.map(|i| json!({"at": i, "honest": want.get(i), "cut": s.get(i)}))})
    }).collect();
    json!({"case": name, "expect": "the honest session's every request to its Finish, no proof, and the firewall's stop at Finish",
           "honest": {"accepted": honest["accepted"], "error": honest["error"], "rounds": rounds, "device_rounds": dev,
                      "round_bytes": round_bytes(&h), "public_bytes": public_bytes(&h), "proofs": count(&h, "proof"), "requests": h.len()},
           "cuts": cuts, "pass": honest_ok && cuts.iter().all(|c| c["ok"] == true)})
}

/// The public-circuit ZK proof's gap 1, at the session's own draws: two tables' level-0 hiding (every lane's padding, the extra
/// lanes), both reps' pads (h, μ, the pads' padding, τ's salt), the mask slot, and the salts of their level-0 and later trees
/// share no 16-byte word. A proving build reads every one from the OS, so the second table drawn at the first's addresses (the
/// control) shares none either. Under an injected seed (the test harness's) the control repeats every word: the per-rep
/// indexing alone would give the second table every word of the first, which would reveal the difference of their masked values.
fn tables_randomness_case(tabs: &Tables) -> Value {
    let os = tables_draws(tabs, ProverSeed::from_os());
    let pass = |d: &Value, control: bool| {
        let n = |k: &str| d[k].as_u64().unwrap_or(u64::MAX);
        let (a, b) = (&d["words"][0], &d["words"][1]);
        a.as_u64().is_some_and(|x| x > 0) && a == b && n("shared_words") == 0
            && if control { n("control_words_equal") == n("stream_words") && n("control_salt_words_shared") == n("salt_words") }
               else { n("control_words_equal") == 0 && n("control_salt_words_shared") == 0 }
    };
    #[cfg(feature = "seed-injection")]
    let injected = tables_draws(tabs, ProverSeed::injected([3; 32]));
    #[cfg(not(feature = "seed-injection"))]
    let injected = json!({"skipped": "built without the test harness's seed-injection feature"});
    let injected_ok = injected.get("skipped").is_some() || pass(&injected, true);
    json!({"case": "zk_tables_share_no_randomness",
           "expect": "no word in common across the tables, nor at one table's addresses; an injected seed's control repeats at them",
           "os": os, "injected": injected, "pass": pass(&os, false) && injected_ok})
}

/// [`tables_randomness_case`]'s draws under `seed`: the words the two tables share, and those the second table drawn at the first's
/// addresses (the control) shares with the first.
fn tables_draws(tabs: &Tables, seed: ProverSeed) -> Value {
    let salts = flock_live::zk_hooks::LeafSalts::for_seed(&seed);
    // (the prover's stream words, the leaf salts)
    let draws = |tb: &Table, at: u64| -> (Vec<[u8; 16]>, Vec<[u8; 16]>) {
        let (st, lay) = (&tb.st, zk_layout(&tb.st));
        let mut prng = flock_live::zk_hooks::ProverRng::for_table(seed, at);
        let l0 = level0_zk(st, &mut prng, false, 0);
        let mut w: Vec<F128> = l0.extra.iter().chain(&l0.pads).flatten().copied().collect();
        let mut s = salts.table(at);
        let (mut tau, mut b) = (Vec::new(), s.level0(64).to_vec());
        for rep in 0..2 {
            let p = zkv::Pads::draw(&mut prng, &lay, rep);
            w.extend(p.h.iter().chain(&p.mu).chain(&p.pad_h).chain(&p.pad_mu));
            tau.extend(&p.tau_salt);
            let id = s.next_id();
            b.extend(s.expand(id, 64));
        }
        let mask = wit(st, &tb.inst, &Plan::default(), false, seed, at as usize).mask;
        w.extend((0..st.blocks().min(4)).flat_map(|k| mask(k)));
        let words = w.iter().map(|x| {
            let mut a = [0u8; 16];
            a[..8].copy_from_slice(&x.lo.to_le_bytes());
            a[8..].copy_from_slice(&x.hi.to_le_bytes());
            a
        }).chain(tau.chunks(16).map(|c| c.try_into().expect("16 bytes"))).collect();
        (words, b.chunks(16).map(|c| c.try_into().expect("16 bytes")).collect())
    };
    let ((a, sa), (b, sb), (control, sc)) = (draws(&tabs.list[0], 0), draws(&tabs.list[1], 1), draws(&tabs.list[1], 0));
    let first: std::collections::BTreeSet<&[u8; 16]> = a.iter().chain(&sa).collect();
    let shared = b.iter().chain(&sb).filter(|x| first.contains(x)).count();
    let control_equal = a.iter().zip(&control).filter(|(x, y)| x == y).count();
    let control_salts_shared = sc.iter().filter(|x| first.contains(x)).count();
    let same_shape = !sa.is_empty() && a.len() == b.len() && sa.len() == sb.len() && control.len() == a.len() && sc.len() == sa.len();
    json!({"words": [a.len() + sa.len(), if same_shape { b.len() + sb.len() } else { 0 }], "stream_words": a.len(), "salt_words": sa.len(),
           "shared_words": shared, "control_words_equal": control_equal, "control_salt_words_shared": control_salts_shared})
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
        (verified(&v), proofs, json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]}))
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
/// device witness takes them, a statement whose units return its outputs; reuse never under M1), its witness built ahead of the
/// session as the prove loop's next session's is. Equal seeds and verifier coins give byte-identical proofs and transcripts.
#[cfg(feature = "seed-injection")]
fn gpu_paths_case(st: &Arc<Stmt>, inst: &Arc<Instances>) -> Value {
    let run = |plan: Plan| {
        let s0 = Arc::new(Mutex::new(server(st, inst)));
        fix_verifier_coins(&mut s0.lock().unwrap_or_else(PoisonError::into_inner), [9; 32]);
        let (v, proofs) = session(st, inst, shared(InProc(s0.clone())), &plan, ProverSeed::injected([1; 32]));
        let rec = s0.lock().unwrap_or_else(PoisonError::into_inner).record_json();
        (v, proofs, json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]}))
    };
    let a = run(Plan { gpu: true, host_units: Some(false), rep_reuse: Some(false), native_sha: Some(false), host_mapped: Some(false), ..Plan::default() });
    let mut pb = Plan { gpu: true, host_units: Some(true), rep_reuse: Some(true), native_sha: Some(true), host_mapped: Some(true), ..Plan::default() };
    pb.prebuilt = Some(Arc::new(Mutex::new(Some(prebuild(st, inst, &pb, ProverSeed::injected([1; 32]))))));
    let b = run(pb);
    let used = |k: &str| b.0["buckets"].as_array().map(|x| x.iter().map(|r| r[k].clone()).collect::<Vec<_>>());
    let used_a = |k: &str| a.0["buckets"].as_array().map(|x| x.iter().map(|r| r[k].clone()).collect::<Vec<_>>());
    let accepted = verified(&a.0) && verified(&b.0);
    let (pe, te) = (a.1 == b.1, a.2 == b.2);
    json!({"case": "gpu_paths_agree", "expect": "byte-identical proofs and transcripts", "gpu": true, "zk": is_zk(st), "accepted": accepted,
           "proofs_equal": pe, "transcripts_equal": te, "transcript_first_difference": first_diff(&a.2, &b.2, String::new()),
           "host_units": used("host_units"), "rep_reused": used("rep_reused"), "native_sha": used("native_sha"), "host_slots": [used_a("host_slots"), used("host_slots")], "prebuilt": b.0["witness_prebuilt_s"].is_number(), "prove_s": [a.0["prove_s"].clone(), b.0["prove_s"].clone()],
           "pass": accepted && pe && te})
}

#[cfg(not(feature = "seed-injection"))]
fn gpu_paths_case(_: &Arc<Stmt>, _: &Arc<Instances>) -> Value {
    json!({"case": "gpu_paths_agree", "pass": true, "skipped": "built without the test harness's seed-injection feature"})
}

/// The prove loop with its verdicts awaited beside the next sessions' proving (`FC_VERIFY_AHEAD`), against two servers taking
/// turns, each verifying its sessions at once (`serve --concurrency`), over loopback TCP, against the loop and one server one
/// session at a time: with each session's prover seed and the verifier's coins injected, every session's proofs and transcript
/// are byte-identical.
#[cfg(feature = "seed-injection")]
fn verify_ahead_case(st: &Arc<Stmt>, inst: &Arc<Instances>, gpu: bool) -> Value {
    const SESSIONS: usize = 3;
    let seeds: Vec<ProverSeed> = (0..SESSIONS).map(|k| ProverSeed::injected([1 + k as u8; 32])).collect();
    let base = Plan { gpu, ..Plan::default() };
    let run = |ahead: usize, servers: usize, tag: &str| {
        let ls: Vec<std::net::TcpListener> = (0..servers).map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let addrs: Vec<String> = ls.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
        let dirs: Vec<std::path::PathBuf> = (0..servers).map(|j| std::env::temp_dir().join(format!("fc-verify-ahead-{}-{tag}-{j}", std::process::id()))).collect();
        for d in &dirs {
            let _ = std::fs::remove_dir_all(d);
        }
        let mut got: Vec<(Value, Vec<Vec<u8>>)> = Vec::new();
        let ts = Instant::now();
        std::thread::scope(|sc| {
            for (j, (l, (d, a))) in ls.iter().zip(dirs.iter().zip(&addrs)).enumerate() {
                flock_live::spawn_scoped(sc, move || serve_on(l, d, Some((SESSIONS - j).div_ceil(servers)), (ahead + 1).div_ceil(servers), a, || {
                    let mut s = server(st, inst);
                    fix_verifier_coins(&mut s, [9; 32]);
                    s
                }));
            }
            prove_sessions(st, inst, &base, &seeds, 0, &addrs.join(","), ahead, None, &mut |_, v, proofs| got.push((v, proofs)));
        });
        let wall_s = ts.elapsed().as_secs_f64();
        let recs: Vec<Value> = (0..SESSIONS).map(|k| {
            let p = dirs[k % servers].join(format!("l{:04}-{}", k / servers, std::process::id())).join("session.json");
            let rec: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_default()).unwrap_or(Value::Null);
            json!({"streams": rec["streams"], "link": rec["link"], "proofs": rec["proof_sha512"]})
        }).collect();
        for d in &dirs {
            let _ = std::fs::remove_dir_all(d);
        }
        (got, recs, wall_s)
    };
    let (a, b) = (run(0, 1, "serial"), run(SESSIONS - 1, 2, "ahead"));
    let accepted = a.0.len() == SESSIONS && b.0.len() == SESSIONS && a.0.iter().chain(&b.0).all(|(v, _)| verified(v) && v["step0_checked"] == true);
    let pe = a.0.iter().zip(&b.0).all(|(x, y)| x.1 == y.1) && a.0.iter().all(|(_, p)| !p.is_empty());
    let te = a.1 == b.1 && a.1.iter().all(|r| !r["streams"].is_null());
    let seeds_differ = a.0.windows(2).all(|w| w[0].1 != w[1].1);
    json!({"case": "verify_ahead_matches_serial", "expect": "each session's proofs and transcript are its serial twin's, byte for byte", "gpu": gpu,
           "zk": is_zk(st), "sessions": SESSIONS, "verify_ahead": SESSIONS - 1, "verify_servers": 2, "accepted": accepted, "proofs_equal": pe, "transcripts_equal": te,
           "sessions_differ": seeds_differ, "transcript_first_difference": first_diff(&json!(a.1), &json!(b.1), String::new()),
           "wall_s": [a.2, b.2], "pass": accepted && pe && te && seeds_differ})
}

#[cfg(not(feature = "seed-injection"))]
fn verify_ahead_case(_: &Arc<Stmt>, _: &Arc<Instances>, _: bool) -> Value {
    json!({"case": "verify_ahead_matches_serial", "pass": true, "skipped": "built without the test harness's seed-injection feature"})
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
    let missing = |name: &str| forged_missing(name).map(|why| json!({"case": name, "gpu": gpu, "accepted": false, "pass": false, "why": why}));
    if only == WORKER_SESSION {
        println!("NEG\t{}", worker_session(st, inst));
        return;
    }
    if !only.is_empty() {
        println!("NEG\t{}", missing(&only).unwrap_or_else(|| run_case(st, inst, &only, gpu)));
        return;
    }
    let mut all = true;
    for &(name, want_accept, _) in CASES {
        if skipped(st, name) || (name == "row_ref_claim_false" && !inst.tables.as_ref().is_some_and(|t| t.counts[..st.c.n_in].iter().any(|&r| r > 1)))
            || (["gpu_proofs_match_cpu", "gpu_paths_agree", "verify_ahead_matches_serial"].contains(&name) && !gpu) {
            println!("SKIP\t{}", json!({"case": name}));
            continue;
        }
        if let Some(v) = missing(name) {
            all = false;
            println!("NEG\t{v}");
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
    println!("SELFTEST\t{}", json!({"statement": st.c.statement(), "unit": st.c.name, "subcircuit": inst.header["subcircuit"], "instances": st.n,
        "vus_per_block": st.c.g, "k_log": st.c.k_log, "m": st.m, "gpu": gpu, "circuit_sha512": hex(&st.c.sha512),
        "unit_rows_sha256": hex(&st.c.unit().sha256), "rows_per_vu": st.c.sizes_per_vu().0, "ands_per_vu": st.c.sizes_per_vu().1, "all_pass": all}));
}

/// The prove loop: one session per seed, each on its own connection to the verifier at `addr`, the first `warm` of them warm
/// (`addr` a comma-separated list: session k's verifier is the (k mod n)th, so n verifier processes take turns).
/// The next sessions' witnesses are built beside this one, FC_PIPELINE_DEPTH (default 1) of them at a time: a deep unit's host
/// witness is a few lane groups, so several share the cores (FC_PIPELINE=0: each session builds its own when it starts). Up to
/// `verify_ahead` proved sessions send their proofs and wait for their verdicts on their own threads while the next session
/// proves (0: each verdict before the next session starts); every warm verdict comes before the first timed session starts.
/// `emit` takes each session's record and proofs, in session order.
/// With `fw` (`--firewall-state`), each session is an outer session of `fw`'s recursive session at its part, through the
/// release side around the Lean session loop (`core-firewall session`, [`session_addr`]): it takes the firewall's outer
/// slot and is admitted or refused on the firewall's record before anything of it reaches the challenger, its proofs and
/// `Finish` leave at their release time if the recursive session has a schedule, and every stop reaches the challenger as
/// a cut, never as `Finish`. The session's record carries the outcome the loop recorded ([`firewall_outcome`]).
#[allow(clippy::too_many_arguments)]
fn prove_sessions(st: &Arc<Stmt>, inst: &Arc<Instances>, base: &Plan, seeds: &[ProverSeed], warm: usize, addr: &str, verify_ahead: usize,
                  fw: Option<&ReleaseArgs>, emit: &mut dyn FnMut(usize, Value, Vec<Vec<u8>>)) {
    let n = seeds.len();
    let pipeline = base.tables <= 1 && std::env::var("FC_PIPELINE").as_deref() != Ok("0");
    let depth = std::env::var("FC_PIPELINE_DEPTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    // FC_HOST_PREPIN=1: the first session's witness is built before any other's and pins `depth` more buffers of its mapped
    // host slots' size, so the next sessions' builds meet a warm buffer pool and a deep unit's flat forms built, as they do
    // past a prover's first sessions
    let prepin = pipeline && std::env::var("FC_HOST_PREPIN").as_deref() == Ok("1");
    let mut ahead: std::collections::VecDeque<(usize, std::thread::JoinHandle<Prebuilt>)> = std::collections::VecDeque::new();
    type Verdict = std::thread::JoinHandle<(Value, Vec<Vec<u8>>)>;
    let mut verdicts: std::collections::VecDeque<(usize, (f64, f64, usize), Verdict)> = std::collections::VecDeque::new();
    let mut emit_one = |run: usize, (mut v, proofs): (Value, Vec<Vec<u8>>), (rtt, rtt_min, rtt_n): (f64, f64, usize)| {
        v["run"] = json!(run);
        v["warm"] = json!(run < warm);
        v["pipeline_depth"] = json!(if pipeline { depth } else { 0 });
        v["verify_ahead"] = json!(verify_ahead);
        v["open_rtt_ms"] = json!(rtt);
        v["open_rtt_min_ms"] = json!(rtt_min);
        v["open_rtt_n"] = json!(rtt_n);
        emit(run, v, proofs);
    };
    let join = |h: Verdict| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
    let addrs: Vec<&str> = addr.split(',').collect();
    for run in 0..n {
        let mut plan = base.clone();
        if pipeline {
            if run == 0 && prepin {
                let p = prebuild(st, inst, base, seeds[0]).join().expect("the first session's witness");
                #[cfg(feature = "gpu")]
                if let WitData::Dev(_, h, _) | WitData::DevCut(_, h, _, _) = &p.wt0 {
                    if h.zab.is_mapped() {
                        flock_live::gpu_circuit::HostBuf::prepin(h.zab.len(), depth);
                    }
                }
                ahead.push_back((0, flock_live::spawn(move || p)));
            }
            if ahead.front().is_some_and(|(r, _)| *r == run) {
                let (_, h) = ahead.pop_front().expect("this run's witness");
                plan.prebuilt = Some(Arc::new(Mutex::new(Some(h))));
            }
            let from = ahead.back().map_or(run + 1, |(r, _)| r + 1);
            for r in from..(run + 1 + depth).min(n) {
                ahead.push_back((r, prebuild(st, inst, base, seeds[r])));
            }
        }
        if run == warm {
            while let Some((r, rtt, h)) = verdicts.pop_front() {
                emit_one(r, join(h), rtt);
            }
        }
        let (addr, frames) = session_addr(st, inst, table_names(plan.tables).len(), addrs[run % addrs.len()], fw).unwrap_or_else(|e| refuse(e));
        plan.lean_session = frames.is_some();
        plan.lean_frames = frames;
        let addr = addr.as_str();
        let seen = fw.map(|fw| firewall_events(&fw.state).len());
        if let (Some(fw), Some(seen)) = (fw, seen) {
            let mut tcp = TcpTransport::connect(addr).unwrap_or_else(|e| refuse(format!("the firewall's session loop at {addr}: {e}")));
            let rtt = tcp.ping_rtt(30);
            let (mut v, proofs) = match session_proved(st, inst, shared(tcp), &plan, seeds[run]) {
                Ok(p) => session_finish(p),
                Err(stopped) => stopped,
            };
            v["firewall"] = firewall_outcome(fw, seen, &v);
            emit_one(run, (v, proofs), rtt);
            continue;
        }
        let mut tcp = TcpTransport::connect(addr).unwrap();
        let rtt = tcp.ping_rtt(30);
        let proved = session_proved(st, inst, shared(tcp), &plan, seeds[run]);
        let finish = move || match proved {
            Ok(p) => session_finish(p),
            Err(stopped) => stopped,
        };
        if verify_ahead == 0 {
            emit_one(run, finish(), rtt);
            continue;
        }
        verdicts.push_back((run, rtt, flock_live::spawn(finish)));
        while verdicts.len() > verify_ahead {
            let (r, rtt, h) = verdicts.pop_front().expect("a session awaiting its verdict");
            emit_one(r, join(h), rtt);
        }
    }
    while let Some((r, rtt, h)) = verdicts.pop_front() {
        emit_one(r, join(h), rtt);
    }
}

/// `prove --firewall-state DIR --session ID --part NAME`: the firewall's release side (PROTOCOL.md §10.5) on the state
/// directory `verity_flock.firewall` shares, run by the Lean session loop it goes through (`core-firewall session`:
/// `FC_CORE_FIREWALL`, else the build in this checkout's verity/core/service/firewall). It runs behind the send check
/// only, through the Lean session loop ([`firewall_on`]), proves one session (the firewall serves each statement of a
/// recursive session once), and refuses `FC_VERIFY_AHEAD`: the outer slot is held until each session's verdict.
struct ReleaseArgs {
    state: std::path::PathBuf,
    session: String,
    part: String,
    lean: std::path::PathBuf,
}

fn release_args(st: &Stmt, sessions: usize) -> Option<ReleaseArgs> {
    let state = arg("--firewall-state", "");
    if state.is_empty() {
        return None;
    }
    if !firewall_on(st) {
        refuse("--firewall-state: the release side runs around the Lean session loop (--zk with FC_FIREWALL=1)".into());
    }
    if sessions != 1 {
        refuse(format!("--firewall-state proves one session (--warm 0 --runs 1), not {sessions}: the firewall serves each statement once"));
    }
    if std::env::var("FC_VERIFY_AHEAD").is_ok_and(|v| !v.is_empty() && v != "0") {
        refuse("--firewall-state refuses FC_VERIFY_AHEAD: the outer slot is held until each session's verdict".into());
    }
    let (session, part) = (arg("--session", ""), arg("--part", ""));
    if session.is_empty() || part.is_empty() {
        refuse("--firewall-state takes --session ID (the recursive session: rec_live's firewall.json) and --part NAME".into());
    }
    let lean = std::env::var_os("FC_CORE_FIREWALL").map(Into::into)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../core/service/firewall/.lake/build/bin/core-firewall"));
    Some(ReleaseArgs { state: state.into(), session, part, lean })
}

/// The firewall's record (`record.jsonl` in `state`), its events in order; none where it can't be read.
fn firewall_events(state: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(state.join("record.jsonl")).unwrap_or_default().lines().filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// What the firewall recorded of `fw`'s outer session after its first `seen` events: the last outcome of its recursive
/// session (`release`, `stop` or `refused`) and why (a stop's, else the verdict `v`'s), or `"outcome": null` where it
/// recorded none.
fn firewall_outcome(fw: &ReleaseArgs, seen: usize, v: &Value) -> Value {
    let last = firewall_events(&fw.state).into_iter().skip(seen).filter(|e| e["session"] == fw.session.as_str())
        .filter(|e| matches!(e["event"].as_str(), Some("release" | "stop" | "refused"))).last();
    let (outcome, why) = match &last {
        Some(e) if e["event"] == "release" => (e["event"].clone(), Value::Null),
        Some(e) => (e["event"].clone(), if e["why"].is_string() { e["why"].clone() } else { v["why"].clone() }),
        None => (Value::Null, v["why"].clone()),
    };
    json!({"session": fw.session, "part": fw.part, "outcome": outcome, "why": why})
}

fn serve_loop(make: impl Fn() -> Server) {
    let listen = arg("--listen", "127.0.0.1:7500");
    let out = std::path::PathBuf::from(arg("--out", "sessions"));
    let l = std::net::TcpListener::bind(&listen).unwrap();
    let max: Option<usize> = arg("--sessions", "").parse().ok();
    // --concurrency C: up to C sessions at once, each on its own thread (a prover's FC_VERIFY_AHEAD sessions awaiting their
    // verdicts, and the one it proves); a connection past them waits to be accepted
    let concurrency = arg("--concurrency", "1").parse::<usize>().unwrap_or(1).max(1);
    // the bound address: --listen 127.0.0.1:0 serves on a port the system picks
    let bound = l.local_addr().unwrap().to_string();
    println!("SERVING {bound}");
    serve_on(&l, &out, max, concurrency, &bound, make);
}

/// Serves `l`'s connections, one session each, up to `concurrency` at once, and `max` of them (unbounded: `None`); each
/// session's record, proofs and verifier file under `out/<session>`, its line in `out/index.jsonl` and on stdout.
fn serve_on(l: &std::net::TcpListener, out: &std::path::Path, max: Option<usize>, concurrency: usize, listen: &str, make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    std::fs::create_dir_all(out).unwrap();
    let free = (Mutex::new(concurrency), std::sync::Condvar::new());
    let index = Mutex::new(());
    let one = |i: usize, conn: std::net::TcpStream, mut s: Server| {
        conn.set_nodelay(true).unwrap();
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut w = BufWriter::new(conn);
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
                          "verify_s": rec["verdict"]["verify_s"], "handle_s": rec["verdict"]["handle_s"], "rounds": rec["verdict"]["rounds"]});
        let _whole_lines = index.lock().unwrap_or_else(PoisonError::into_inner);
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        // one write per line: servers sharing `out` append to the same index
        idx.write_all(format!("{line}\n").as_bytes()).unwrap();
        println!("SESSION\t{line}");
    };
    std::thread::scope(|sc| {
        for i in 0.. {
            if Some(i) == max {
                break;
            }
            {
                let mut n = free.0.lock().unwrap_or_else(PoisonError::into_inner);
                while *n == 0 {
                    n = free.1.wait(n).unwrap_or_else(PoisonError::into_inner);
                }
                *n -= 1;
            }
            let (conn, _) = l.accept().unwrap();
            let s = make();
            let (one, free) = (&one, &free);
            flock_live::spawn_scoped(sc, move || {
                // a session's panic stops the server
                if catch_unwind(AssertUnwindSafe(|| one(i, conn, s))).is_err() {
                    std::process::exit(101);
                }
                *free.0.lock().unwrap_or_else(PoisonError::into_inner) += 1;
                free.1.notify_one();
            });
        }
    });
}

/// `replay [--zk] [--session-tables J] --sessions DIR...`: the verifier's offline verdict on each recorded session
/// (`DIR/session.json` and `DIR/<table>.rep<r>.bin`), `Server::from_record` then `finish` over the session's tables (its draw
/// split into `J`), one `UPSTREAM` line per session: the reference the Lean verifier's agreement check
/// (`verity/ml/flock/verifier/agree.py`) reads. The record's coin tree is not re-checked (the live server draws the coins).
fn replay(st: &Arc<Stmt>, inst: &Arc<Instances>, count: usize) {
    let a: Vec<String> = std::env::args().collect();
    let dirs = &a[a.iter().position(|x| x == "--sessions").map_or(a.len(), |i| i + 1)..];
    for d in dirs {
        let v = (|| -> Result<Value, String> {
            let rec: Value = serde_json::from_slice(&std::fs::read(format!("{d}/session.json")).map_err(|e| format!("session files: {e}"))?)
                .map_err(|e| format!("session files: {e}"))?;
            let tabs = session_tables(st, inst, rec.get("unit_draw").filter(|x| x.is_object()), count)?;
            let proofs = tabs.names().into_iter()
                .flat_map(|t| (0..2).map(move |r| (t.clone(), r)))
                .map(|(t, r)| { let p = std::fs::read(format!("{d}/{t}.rep{r}.bin")).unwrap_or_default(); ((t, r), p) })
                .collect::<BTreeMap<_, _>>();
            let mut s = Server::from_record(config_of(&tabs), Box::new(CircuitVerifier::new(tabs)), &rec, proofs)?;
            let v = s.finish();
            Ok(json!({"accepted": v["accepted"], "why": v["why"]}))
        })().unwrap_or_else(|e| json!({"accepted": false, "why": e}));
        println!("UPSTREAM\t{}", json!({"session": d, "verdict": v}));
    }
}

fn refuse(why: String) -> ! {
    println!("REFUSED\t{}", json!({"statement": STATEMENT, "why": why}));
    std::process::exit(2);
}

const STACKED_TABLE: &str = "stacked";
const STACKED_DOMAIN: &str = "flock-circuit/stacked/fast100/rep0";

/// R, the stacked table (`--table`): 2^m cells, cell i at byte i >> 3, bit i & 7, packed 128 cells an element.
fn stacked_table(path: &str) -> (usize, Vec<u8>, Vec<F128>) {
    let b = std::fs::read(path).unwrap_or_else(|e| refuse(format!("--table {path}: {e}")));
    let m = (b.len() * 8).trailing_zeros() as usize;
    if !b.len().is_power_of_two() || !(22..=35).contains(&m) {
        refuse(format!("--table {path}: {} bytes, not 2^m cells with 22 <= m <= 35 (fast100)", b.len()));
    }
    let word = |c: &[u8]| u64::from_le_bytes(c.try_into().unwrap());
    let z = b.chunks_exact(16).map(|c| F128::new(word(&c[..8]), word(&c[8..]))).collect();
    (m, b, z)
}

/// `--points`: `{"x": [x_0, x_1, ...]}`, claim k's x = its point[6:] (`rec_sparse.RegisteredClaim.x`), each coordinate an
/// `f128_hex`.
fn stacked_points(m: usize) -> Vec<Vec<F128>> {
    let path = arg("--points", "points.json");
    let v: Value = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
        .unwrap_or_else(|e| refuse(format!("--points {path}: {e}")));
    let xs = v["x"].as_array().filter(|a| !a.is_empty()).unwrap_or_else(|| refuse(format!("--points {path}: no claim points")));
    xs.iter().map(|x| {
        let x: Vec<F128> = x.as_array().into_iter().flatten().map(|h| {
            let u = h.as_str().filter(|s| s.len() == 32).and_then(|s| u128::from_str_radix(s, 16).ok())
                .unwrap_or_else(|| refuse(format!("--points {path}: {h} is not an f128_hex")));
            F128::new(u as u64, (u >> 64) as u64)
        }).collect();
        if x.len() != m - 6 {
            refuse(format!("--points {path}: a point's x of {} coordinates, R's wants m - 6 = {}", x.len(), m - 6));
        }
        x
    }).collect()
}

/// `stacked`: candidate 3's registered opening, `rec_algebra.Shape(m, 6, claims, salt_len=0, opening_only=True)`. R is committed
/// under plain SHA-512 leaves at fast100(m) and opened at the relay's claim points by upstream's batch opening, on the live stream
/// `stacked/rep0`, bound as a rep is (`bind_statement`'s label, the inner statement's digest, R's cap): the circuit reps' own
/// statement digest (`--statement-digest`, which the firewall holds to theirs), so the stream's first coin follows the
/// statement's META and the circuit registration it binds; no zerocheck or lincheck precedes it.
///
///   flock-circuit stacked --table R.bin --root-out ROOT.json                         R's registered root, at staging
///   flock-circuit stacked --table R.bin --points P.json --statement-digest HEX --verifier ADDR [--dump F]   the opening
///
/// `ROOT.json` is `{m, cap, digest}` in hex, the cap table `stacked`'s Commit root. The opening commits R again (plain leaves
/// take no salt, so to the same cap), sends only `Open`, its rounds, `Proof` and `Finish`, and its proof is bincode((Commitment,
/// R1csProofLigerito)) with an empty zerocheck and lincheck: a circuit rep's container, as V* decodes it.
fn stacked() {
    use flock_transcript::challenger::Challenger;
    let (m, raw, z) = stacked_table(&arg("--table", "R.bin"));
    let pcs = PcsParams { merkle_hash: HashKind::Sha512, ..params(m, LigeritoProfile::Fast100) };
    let t0 = Instant::now();
    let (cm, pd) = flock_core::pcs::commit::commit(&z, &pcs);
    let root = merkle_digest_bytes(&cm.cap).to_vec();
    let digest = Sha512::digest(&raw)[..32].to_vec();
    let mut out = json!({"m": m, "digest": hex(&digest), "t.commit": t0.elapsed().as_secs_f64()});
    let root_out = arg("--root-out", "");
    if !root_out.is_empty() {
        std::fs::write(&root_out, json!({"m": m, "cap": hex(&root), "digest": hex(&digest)}).to_string()).expect("--root-out");
        println!("STACKED\t{out}");
        return;
    }
    let xs = stacked_points(m);
    let sd = arg("--statement-digest", "");
    let statement_digest: Vec<u8> = (sd.len() == 64 && sd.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| (0..32).map(|i| u8::from_str_radix(&sd[2 * i..2 * i + 2], 16).unwrap()).collect())
        .unwrap_or_else(|| refuse(format!("--statement-digest {sd:?}: the inner statement's 32-byte digest, 64 hex digits")));
    let addr = arg("--verifier", "127.0.0.1:7500");
    let t = shared(TcpTransport::connect(&addr).unwrap_or_else(|e| refuse(format!("the challenger at {addr}: {e}"))));
    let mut ch = LiveChallenger::new(t.clone(), STACKED_TABLE, 0, STACKED_DOMAIN.as_bytes()).unwrap_or_else(|e| refuse(format!("Open: {e}")));
    ch.observe_label(b"flock-r1cs-v0");
    ch.observe_bytes(&statement_digest);
    ch.observe_bytes(&root);
    let lc = pcs.ligerito_prover_config().unwrap_or_else(|e| refuse(e));
    let refs: Vec<&[F128]> = xs.iter().map(Vec::as_slice).collect();
    let pre: Vec<Option<&[F128]>> = vec![None; refs.len()];
    let tp = Instant::now();
    let pcs_open = flock_core::pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        z, &pd, &cm, &refs, &pre, &[], &flock_core::zerocheck::PaddingSpec::dense(m), &lc, pcs.opening_grinding(), &mut ch);
    let zerocheck = flock_core::zerocheck::ZerocheckProof { round1_ab: vec![], round1_c: vec![], multilinear_rounds: vec![],
        final_a_eval: F128::ZERO, final_b_eval: F128::ZERO, final_c_eval: F128::ZERO, grinding_nonces: vec![] };
    let lincheck = flock_core::lincheck::LincheckProof { rounds: vec![], z_partial: vec![], matrix_evals: vec![], grinding_nonces: vec![],
        comb_partial: vec![] };
    let bytes = bincode::serialize(&(&cm, &R1csProofLigerito { zerocheck, lincheck, pcs_open })).unwrap();
    out["t.open"] = json!(tp.elapsed().as_secs_f64());
    out["proof_bytes"] = json!(bytes.len());
    let dump = arg("--dump", "");
    if !dump.is_empty() {
        std::fs::write(&dump, &bytes).expect("--dump");
    }
    match call(&t, Req::Proof { table: STACKED_TABLE.into(), rep: 0, bytes }) {
        Ok(Resp::Ok) => {}
        other => refuse(format!("Proof: {other:?}")),
    }
    out["verdict"] = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap_or(Value::String(v)),
        Ok(Resp::Ok) => Value::Null,
        other => refuse(format!("Finish: {other:?}")),
    };
    println!("STACKED\t{out}");
}

fn main() {
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    if cmd == "replay-coins" {
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(arg("--record", "session.json")).expect("record")).expect("record JSON");
        match flock_live::replay_coin_seed(&rec) {
            Ok(n) => println!("COINS\t{}", json!({"replayed": true, "requests": n})),
            Err(e) => { println!("COINS\t{}", json!({"replayed": false, "why": e})); std::process::exit(1) }
        }
        return;
    }
    if cmd == "stacked" {
        return stacked();
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
    // the verifier's commands check Linking on the population before any session; the prover's take no links file
    if matches!(cmd.as_str(), "serve" | "replay" | "statement") {
        let path = arg("--links", "");
        let file: Option<Value> = (!path.is_empty()).then(|| {
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| refuse(format!("--links {path}: {e}"))))
                .unwrap_or_else(|e| refuse(format!("--links {path}: {e}")))
        });
        let links = flock_live::circuit::check_links(&c, &inst, file.as_ref()).unwrap_or_else(|e| refuse(e));
        if links > 0 {
            eprintln!("links: {links} cut wire(s) checked on the population's public file");
        }
    }
    // --zk's level 0 hides behind hm96 salts: a plain-leaf statement is a recursive session's inner one, ZK off
    if zk_mode() && c.plain_leaves() {
        refuse("--zk: the statement's trees have plain SHA-512 leaves (flock-leaf/sha512-unsalted), a ZK-off statement".into());
    }
    let id = identity(&c);
    let st = Arc::new(Stmt::new(c, inst.n, id).unwrap_or_else(|e| refuse(e)));
    // the device prover holds at most 12 slot types (`gpu_circuit::FC_MAX_TYPES`)
    if flag("--gpu") && st.circuit.types.len() > 12 {
        refuse(format!("{} slot types: the GPU prover holds at most 12", st.circuit.types.len()));
    }
    if flag("--gpu") {
        st.c.gpu_nnz_ok().unwrap_or_else(|e| refuse(e));
    }
    // the device's fold holds each slot type's own rows only, never a generated read's table-direct side
    if flag("--gpu") && st.circuit.tables.iter().any(Option::is_some) {
        refuse("a typed template's table reads: the GPU prover folds no table-direct side".into());
    }
    if zk_mode() {
        zk_shape_ok(&st).unwrap_or_else(|e| refuse(e));
    }
    let tables = tables_arg();
    if tables > 1 && flag("--gpu") {
        refuse("--session-tables above 1: the CPU prover (the device prover proves one table per session)".into());
    }
    sverify::Lincheck::from_env().unwrap_or_else(|e| refuse(e));
    eprintln!("statement {} instances={} vus_per_block={} blocks={} k_log={} nbl={} m={} regions={} delta={} built in {:.2}s",
              st.c.name, st.n, st.c.g, st.blocks_real, st.c.k_log, st.nbl, st.m, st.regions.len(), st.circuit.da.len() / 2, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(&st, &inst, flag("--gpu")),
        "statement" => println!("STATEMENT\t{}", json!({"statement": st.c.statement(), "unit": st.c.name, "instances": st.n,
            "vus_per_block": st.c.g, "blocks": st.blocks_real, "k_log": st.c.k_log, "nbl": st.nbl, "dense_m": st.m,
            "regions": st.regions.len(), "circuit_sha512": hex(&st.c.sha512), "statement_digest": hex(&st.digest),
            "public_sha512": hex(&inst.public_sha512), "identity": st.identity})),
        "zkstat" => zkstat(&st, &inst),
        "zkaudit" => zkaudit(&st, &inst),
        "zkrewind" => zkrewind(&st, &inst),
        "zkgk" => zkgk(&st, &inst),
        "replay" => replay(&st, &inst, tables),
        "serve" => {
            let sv = serving(&st, &inst, tables);
            serve_loop(move || server_of(&sv, &st, &inst, tables))
        }
        "prove" => {
            if firewall_on(&st) {
                flock_live::firewall::lean_tap(&flock_live::firewall::lean_frame_exe()).unwrap_or_else(|e| refuse(e));
            }
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            let split = (tables > 1).then(|| session_tables(&st, &inst, None, tables).ok()).flatten();
            // FC_ZK_SELF_CHECK=skip: a negative control's prover, which sends its inner proofs when the witness fails
            let zk_self_check = match std::env::var("FC_ZK_SELF_CHECK").as_deref() {
                Err(_) | Ok("stop") => zkv::SelfCheck::Stop,
                Ok("skip") => zkv::SelfCheck::Skip,
                Ok(v) => refuse(format!("FC_ZK_SELF_CHECK={v}: stop or skip")),
            };
            let base = Plan { gpu: flag("--gpu"), tables, split, zk_self_check, ..Plan::default() };
            let seeds: Vec<ProverSeed> = (0..warm + runs).map(|_| ProverSeed::from_os()).collect();
            let verify_ahead = std::env::var("FC_VERIFY_AHEAD").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let fw = release_args(&st, warm + runs);
            prove_sessions(&st, &inst, &base, &seeds, warm, &arg("--verifier", "127.0.0.1:7500"), verify_ahead, fw.as_ref(), &mut |run, v, proofs| {
                if !dump.is_empty() {
                    std::fs::create_dir_all(&dump).unwrap();
                    for (r, p) in proofs.iter().enumerate() {
                        std::fs::write(format!("{dump}/run{run:02}-rep{r}.proof"), p).unwrap();
                    }
                }
                println!("LIVE\t{v}");
            });
        }
        _ => eprintln!("usage: flock-circuit serve|prove|selftest|statement|replay --instances FILE --circuit FILE [--pin SHA256] ... | stacked --table R.bin ..."),
    }
}
