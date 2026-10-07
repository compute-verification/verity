//! What the outer firewall (PROTOCOL.md §10.1, `FC_FIREWALL=1`, the CPU prover) lets out of a `--zk` session, as a
//! `verity/firewall-transcript/v0` transcript for `Flock.Firewall.Contract` (`flock-firewall contract`): with
//! `FC_FIREWALL_TRANSCRIPT=DIR`, each session's is `DIR/session-<k>.json`, written when the session sends `Finish`, with the
//! public input it is held to beside it (`DIR/public-<k>.json`, `verity/firewall-public/v0`: the relations the verifier
//! checks, each statement item's value by name, and the names of the values fixed by the firewall's recheck, shadow or own
//! computation; the transcript carries none of it, and the outer agreement holds every session to the honest sessions'
//! one). The recorder
//! reads the wire's requests and coins (a [`Recording`] transport after `coin_tree::Checked`) and the firewall's candidates and
//! wants (one-line hooks in `Firewall`, `ShadowLink::check` and `zk_finish`); it changes nothing the firewall sends.
//!
//! Items, in the order they leave:
//! - `Register` (fixed by the statement), `Hello` (the firewall's own, on its word), each rep's `Open` (the statement: the
//!   rep's domain), the `Commit` (σ, the firewall's own on its word; the level-0 root by the recheck), the `Link` (the
//!   statement: zeros);
//! - each device round, frame by frame: in the algebraic phase a label, bytes, a squeeze or an observe's header by the
//!   statement's skeleton (its level-0 cap by the recheck), each observed value `masked` with its pad in `Layout::absorbed` order, and the pads' label and root
//!   that the masking challenger inserts (the statement, the firewall's own); in the opening every frame by the shadow, level
//!   0's observed values `masked`; then the round's `coins` (the wire's, after the coin tree's opening is checked and
//!   stripped) and the ones the firewall relayed;
//! - the inner proof's rounds (the firewall's own stream, nothing relayed): `final_c'`, `ρ = ε h_fa + h_x` and `σ = h_fb + h_y`
//!   masked; then one `check` per constraint row the verifier checks (`zk_finish`'s rows and the triple's), its outcome
//!   `Lin.eval(h) = 0` with its affine `form` beside it; then `τ`'s commitment and `Y` (the firewall's own, recomputed here
//!   from the pads, on every pad used);
//! - each rep's `Proof`: its shape (the statement), masked zerocheck and lincheck (the firewall's own, from the values it
//!   passed), commitment (the recheck), opening (the shadow's, level 0 masked) and inner proof (own, on its word); `finish`
//!   when both reps' proofs went out.
//!
//! Where the firewall refuses, the candidate it held back is written as the items it would have sent (masked values with their
//! pads), from `firewall.gate.from` up to the first item the contract must refuse, and the transcript stops there
//! (`firewall.gate.at`). The candidate is written where the firewall checked it, which need not be where its items leave in
//! an honest session (a Commit's roots are checked before σ goes out, a rep's proof before its inner proof's rounds). Where
//! the prover stops (its own self-check), the transcript stops after what it sent. An item taken on its word carries
//! `"on_its_word": true`; items the recorder made up to stand for a refusal it cannot itemize carry `"stand_in": true`.
//!
//! The recorder is one per process (`start` replaces it), so a process records its gated sessions one at a time: sessions
//! run concurrently in one process would interleave their items.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::proof::R1csProofLigerito;
use flock_field::F128;
use flock_prover::pcs::Commitment;
use serde_json::{Value, json};

use crate::firewall::{Item, parse};
use crate::zk_veil::{self as zkv, InnerProof, L0_CLOSE, L0_OPEN, LABEL_INNER, LABEL_PADS, LABEL_PCS, LABEL_ZC, Layout, Lin};
use crate::{KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, Req, Resp, Transport, hex, push_header, push_padded};

pub const FORMAT: &str = "verity/firewall-transcript/v0";
pub const PUBLIC_FORMAT: &str = "verity/firewall-public/v0";

/// The rows `zk_finish` adds to the replay's: the ring switch's, then level 0's lane phase.
const RING_SWITCH: [&str; 2] = ["ring switch ab", "ring switch c"];
const LANE_PHASE: [&str; 2] = ["ligerito lane phase (c0)", "ligerito lane phase (c1)"];
/// `firewall::skeleton`'s item that is the recheck commitment's cap (after the domain, the R1CS label and the digest): a value
/// of the session's, fixed by the recheck, not by the statement.
const SKELETON_CAP: usize = 3;

static REC: Mutex<Option<Rec>> = Mutex::new(None);
static SESSIONS: AtomicUsize = AtomicUsize::new(0);

/// Runs `f` on the session's recorder, if one is recording.
pub fn with<R>(f: impl FnOnce(&mut Rec) -> R) -> Option<R> {
    REC.lock().unwrap_or_else(PoisonError::into_inner).as_mut().map(f)
}

/// A new session's recorder, when `FC_FIREWALL_TRANSCRIPT` names a directory: whether one started.
pub fn start() -> bool {
    let Ok(dir) = std::env::var("FC_FIREWALL_TRANSCRIPT") else { return false };
    let k = SESSIONS.fetch_add(1, Ordering::SeqCst);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("FC_FIREWALL_TRANSCRIPT {dir}: {e}"));
    *REC.lock().unwrap_or_else(PoisonError::into_inner) =
        Some(Rec::new(PathBuf::from(&dir).join(format!("session-{k}.json")), PathBuf::from(&dir).join(format!("public-{k}.json")),
                      std::env::var("FC_FIREWALL_TAMPER").unwrap_or_default()));
    true
}

/// The session's transport as the recorder sees it: every request that left, with its answer.
pub struct Recording<T: Transport>(pub T);

impl<T: Transport> Transport for Recording<T> {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let r = self.0.call(req.clone());
        if let Ok(resp) = &r {
            with(|x| x.wire(&req, resp));
        }
        r
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.0.stats()
    }
    fn wait_s(&self) -> f64 {
        self.0.wait_s()
    }
}

fn f128_hex(v: F128) -> String {
    format!("{:016x}{:016x}", v.hi, v.lo)
}

fn push_f128(out: &mut Vec<u8>, v: F128) {
    out.extend_from_slice(&v.lo.to_le_bytes());
    out.extend_from_slice(&v.hi.to_le_bytes());
}

/// An item's frame as the live challenger writes it (`firewall::parse`'s inverse).
pub fn frame(it: &Item) -> Vec<u8> {
    let mut v = Vec::new();
    match it {
        Item::Domain(b) | Item::Label(b) | Item::Bytes(b) => {
            push_header(&mut v, match it { Item::Domain(_) => OP_DOMAIN, Item::Label(_) => OP_LABEL, _ => OP_BYTES }, 0, b.len() as u64);
            push_padded(&mut v, b);
        }
        Item::Obs { kind, vals } => {
            push_header(&mut v, OP_OBSERVE, *kind, vals.len() as u64);
            for x in vals {
                push_f128(&mut v, *x);
            }
        }
        Item::Squeeze { kind, n } => push_header(&mut v, OP_SQUEEZE, *kind, *n as u64),
    }
    v
}

fn obs_header(it: Option<&Item>) -> Vec<u8> {
    let mut v = Vec::new();
    if let Some(Item::Obs { kind, vals }) = it {
        push_header(&mut v, OP_OBSERVE, *kind, vals.len() as u64);
    }
    v
}

fn u64s(xs: &[usize]) -> Vec<u8> {
    xs.iter().flat_map(|x| (*x as u64).to_le_bytes()).collect()
}

fn bin<T: serde::Serialize>(x: &T) -> Vec<u8> {
    bincode::serialize(x).expect("bincode")
}

/// The masking challenger's pad order (`zk_veil::Masking`), replayed on the device's items.
#[derive(Clone)]
struct Mask {
    order: Vec<usize>,
    at: usize,
    lig: bool,
    on: bool,
    done: bool,
    l0: bool,
    done_l0: bool,
}

impl Mask {
    fn new(lay: &Layout) -> Self {
        Mask { order: lay.absorbed(), at: 0, lig: lay.ligerito(), on: false, done: false, l0: false, done_l0: false }
    }
    /// A label passes: whether the pads' label and root follow it on the wire.
    fn label(&mut self, l: &[u8]) -> bool {
        if l == LABEL_PCS && self.on {
            self.on = false;
            self.done = true;
        }
        if l == L0_CLOSE && self.l0 {
            self.l0 = false;
            self.on = false;
            self.done_l0 = true;
        }
        let pads = l == LABEL_ZC && !self.done;
        if pads {
            self.on = true;
        }
        if l == L0_OPEN && self.done && self.lig && !self.done_l0 {
            self.on = true;
            self.l0 = true;
        }
        pads
    }
    fn pad(&mut self) -> Result<usize, String> {
        let i = *self.order.get(self.at).ok_or("more masked values than the layout's")?;
        self.at += 1;
        Ok(i)
    }
}

struct RepRec {
    rep: usize,
    base: usize,
    lay: Layout,
    pads: zkv::Pads,
    pads_root: Vec<u8>,
    domain: Vec<u8>,
    cm: Vec<u8>,
    skeleton: Vec<Item>,
    sk_at: usize,
    alg_rounds: usize,
    open_rounds: usize,
    mask: Mask,
    stream: Option<String>,
    wire_rounds: usize,
    used: BTreeSet<usize>,
    /// the values the firewall passed, `final_c'` its own (set when the firewall passes the device's proof)
    vals: Option<Vec<F128>>,
    opening: Option<Vec<u8>>,
    inner: Option<usize>,
    cons: Option<Vec<(&'static str, Lin)>>,
    eps: F128,
    c: Vec<F128>,
    beta: F128,
}

enum Pending {
    Round { alg: bool, k: usize, content: Vec<u8>, n: usize, want: Option<(Vec<u8>, usize)> },
    Root { got: Vec<u8>, want: Vec<u8> },
    Proof { proof: Box<R1csProofLigerito> },
}

pub struct Rec {
    path: PathBuf,
    public: PathBuf,
    tamper: String,
    items: Vec<Value>,
    relations: Vec<String>,
    /// each statement item's value (hex), by name
    statement: BTreeMap<String, String>,
    /// the names of the items the firewall's recheck, shadow or own computation fixes
    computes: BTreeSet<String>,
    pads: Vec<F128>,
    reps: Vec<RepRec>,
    pending: Option<Pending>,
    root_want: Option<Vec<u8>>,
    /// the last gated round's coins on the wire (stream, round), for the firewall's answer
    coins: Option<(String, usize, Vec<F128>)>,
    link: Option<Vec<F128>>,
    tables: usize,
    proofs: usize,
    /// the refusal, the item the contract must refuse and where the held-back candidate's items begin
    refused: Option<(String, usize, usize)>,
    stopped: Option<String>,
    errors: Vec<String>,
}

impl Rec {
    fn new(path: PathBuf, public: PathBuf, tamper: String) -> Self {
        Rec { path, public, tamper, items: Vec::new(), relations: Vec::new(), statement: BTreeMap::new(), computes: BTreeSet::new(),
              pads: Vec::new(), reps: Vec::new(), pending: None, root_want: None,
              coins: None, link: None, tables: 0, proofs: 0, refused: None, stopped: None, errors: Vec::new() }
    }

    fn halted(&self) -> bool {
        self.refused.is_some() || !self.errors.is_empty()
    }

    fn fixed(&mut self, what: String, by: &str, want: &[u8], sent: &[u8], pads: Vec<usize>) -> bool {
        self.public_item(&what, by, want);
        self.items.push(json!({"fixed": {"what": what, "by": by, "want": hex(want), "sent": hex(sent), "pads": pads}}));
        want == sent
    }

    fn on_its_word(&mut self, what: String, by: &str, sent: &[u8], pads: Vec<usize>) {
        self.public_item(&what, by, sent);
        self.items.push(json!({"fixed": {"what": what, "by": by, "want": hex(sent), "sent": hex(sent), "pads": pads, "on_its_word": true}}));
    }

    /// The public input's part for one fixed item: a statement item's value under its name (one value per name), or the
    /// name of a value the firewall fixes itself.
    fn public_item(&mut self, what: &str, by: &str, want: &[u8]) {
        if by != "statement" {
            self.computes.insert(what.to_string());
            return;
        }
        match self.statement.get(what) {
            Some(w) if *w != hex(want) => self.errors.push(format!("{what}: two statement values under one name")),
            Some(_) => {}
            None => {
                self.statement.insert(what.to_string(), hex(want));
            }
        }
    }

    fn masked(&mut self, ri: usize, value: F128, pad: usize, sent: F128) -> bool {
        let r = &mut self.reps[ri];
        let p = r.base + pad;
        r.used.insert(p);
        self.items.push(json!({"masked": {"value": f128_hex(value), "pad": p, "sent": f128_hex(sent)}}));
        sent == value + self.pads[p]
    }

    fn used(&self, ri: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
        let r = &self.reps[ri];
        r.used.iter().copied().filter(|p| keep(p - r.base)).collect()
    }

    fn cur(&self) -> Option<usize> {
        self.reps.len().checked_sub(1)
    }

    /// A rep starts: its layout, pads and their root, domain, the recheck's commitment, the statement's skeleton and the
    /// replay's constraint names (the relations, with `zk_finish`'s and the triple's).
    #[allow(clippy::too_many_arguments)]
    pub fn begin_rep(&mut self, rep: usize, lay: Layout, pads: &zkv::Pads, pads_root: &[u8], domain: &[u8], cm: &Commitment,
                     skeleton: &[Item], replay_names: &[&str]) {
        let base = self.pads.len();
        self.pads.extend_from_slice(&pads.h);
        let triple: Vec<&str> = zkv::triple_constraints(&lay, F128::ZERO, F128::ZERO, F128::ZERO).iter().map(|c| c.0).collect();
        let lane: &[&str] = if lay.ligerito() { &LANE_PHASE } else { &[] };
        for n in replay_names.iter().chain(&RING_SWITCH).chain(lane).chain(&triple) {
            if !self.relations.iter().any(|r| r == n) {
                self.relations.push(n.to_string());
            }
        }
        let pads = zkv::Pads { h: pads.h.clone(), mu: pads.mu.clone(), pad_h: pads.pad_h.clone(), pad_mu: pads.pad_mu.clone(), tau_salt: pads.tau_salt.clone() };
        self.reps.push(RepRec { rep, base, lay, pads, pads_root: pads_root.to_vec(), domain: domain.to_vec(), cm: bin(cm), skeleton: skeleton.to_vec(),
                                sk_at: 0, alg_rounds: 0, open_rounds: 0, mask: Mask::new(&lay), stream: None, wire_rounds: 0, used: BTreeSet::new(),
                                vals: None, opening: None, inner: None, cons: None, eps: F128::ZERO, c: Vec::new(), beta: F128::ZERO });
    }

    /// The device's level-0 root (after any tamper) and the recheck's, before the `Commit` that would carry it.
    pub fn root(&mut self, got: &[u8], want: &[u8]) {
        self.root_want = Some(want.to_vec());
        self.pending = Some(Pending::Root { got: got.to_vec(), want: want.to_vec() });
    }

    /// A device round the firewall is about to check (after any tamper): the algebraic phase's (`alg`) or the opening's.
    pub fn candidate(&mut self, alg: bool, content: &[u8], n: usize) {
        let Some(r) = self.reps.last_mut() else { return };
        let k = if alg { r.alg_rounds += 1; r.alg_rounds - 1 } else { r.open_rounds += 1; r.open_rounds - 1 };
        self.pending = Some(Pending::Round { alg, k, content: content.to_vec(), n, want: None });
    }

    /// The shadow's round the firewall compares the device's opening round with.
    pub fn shadow_round(&mut self, want: &[u8], n: usize) {
        if let Some(Pending::Round { want: w, .. }) = self.pending.as_mut() {
            *w = Some((want.to_vec(), n));
        }
    }

    /// The coins the firewall relayed to the device for the round it passed.
    pub fn answer(&mut self, relayed: &[F128]) {
        let Some((stream, k, coins)) = self.coins.take() else { return };
        self.coins_item(&stream, k, &coins, relayed);
    }

    fn coins_item(&mut self, stream: &str, k: usize, coins: &[F128], relayed: &[F128]) {
        let h = |v: &[F128]| v.iter().map(|x| f128_hex(*x)).collect::<Vec<_>>();
        self.items.push(json!({"coins": {"stream": stream, "round": k, "coins": h(coins), "relayed": h(relayed)}}));
    }

    /// The link points as the session hands them on (the hook's `extra_points` reads them).
    pub fn relayed_link(&mut self, pts: &[F128]) {
        if let Some(c) = self.link.take() {
            self.coins_item("commit", 0, &c, pts);
        }
    }

    /// The values the firewall passed, `final_c'` the one it computed from them.
    pub fn passed(&mut self, vals: &[F128]) {
        if let Some(r) = self.reps.last_mut() {
            r.vals = Some(vals.to_vec());
        }
    }

    /// The firewall passed the device's proof: the shadow's opening, which is what leaves (level 0 masked).
    pub fn shadow_opening(&mut self, open: &BatchOpeningProofLigerito) {
        let Some(r) = self.reps.last_mut() else { return };
        let mut o = open.clone();
        o.ligerito.zk_record = None;
        zkv::mask_ligerito(&mut o.ligerito, &r.lay, &r.pads.h);
        r.opening = Some(bin(&o));
        r.inner = Some(0);
    }

    /// The device's proof the firewall refuses (after any tamper).
    pub fn proof_candidate(&mut self, proof: &R1csProofLigerito) {
        self.pending = Some(Pending::Proof { proof: Box::new(proof.clone()) });
    }

    /// The constraints `zk_finish` hands the inner proof (the triple's are added after `ε`).
    pub fn constraints(&mut self, cons: &[(&'static str, Lin)]) {
        if let Some(r) = self.reps.last_mut() {
            r.cons = Some(cons.to_vec());
        }
    }

    /// The prover stopped the session (what it sent before stays in the transcript).
    pub fn stopped(&mut self, why: &str) {
        self.stopped = Some(why.chars().take(400).collect());
    }

    /// The firewall refused: the candidate it held back, as the items it would have sent, up to the first the contract refuses.
    pub fn refused(&mut self, why: &str) {
        if self.halted() {
            return;
        }
        let from = self.items.len();
        let ok = match self.pending.take() {
            Some(Pending::Round { alg, k, content, want, .. }) => self.stop_round(alg, k, &content, want),
            Some(Pending::Root { got, want }) => self.commit_roots(&[("circuit".to_string(), got)], &want),
            Some(Pending::Proof { proof }) => self.stop_proof(&proof),
            None => true,
        };
        if ok {
            self.items.push(json!({"fixed": {"what": format!("the firewall's refusal: {why}"), "by": "own", "want": "", "sent": "01", "pads": [],
                                              "stand_in": true}}));
        }
        self.refused = Some((why.to_string(), self.items.len() - 1, from));
    }

    fn wants(&self, ri: usize, alg: bool, cand: &[Item], want: &Option<(Vec<u8>, usize)>) -> Result<Vec<Option<Item>>, String> {
        let r = &self.reps[ri];
        if alg {
            return Ok((0..cand.len()).map(|f| r.skeleton.get(r.sk_at + f).cloned()).collect());
        }
        let w = match want {
            Some((w, _)) => parse(w).map_err(|e| format!("the shadow's round: {e}"))?,
            None => Vec::new(),
        };
        Ok((0..cand.len()).map(|f| w.get(f).cloned()).collect())
    }

    /// Items of one device round: `wire` the round as it left (None: the firewall held it back, and what would have left is the
    /// device's frames with their values masked). Returns whether every item may leave (stops at the first that may not when
    /// `wire` is None).
    fn round_items(&mut self, ri: usize, alg: bool, k: usize, cand: &[Item], wire: Option<&[Item]>, want: &Option<(Vec<u8>, usize)>)
                   -> Result<bool, String> {
        let wants = self.wants(ri, alg, cand, want)?;
        let (rep, by) = (self.reps[ri].rep, if alg { "statement" } else { "shadow" });
        let phase = if alg { "algebraic phase" } else { "opening" };
        let mut wi = 0usize;
        let mut next = |synth: Item| -> Result<Item, String> {
            match wire {
                Some(w) => {
                    let x = w.get(wi).cloned().ok_or("the wire's round has fewer frames than the device's")?;
                    wi += 1;
                    Ok(x)
                }
                None => Ok(synth),
            }
        };
        let stop = wire.is_none();
        for (f, c) in cand.iter().enumerate() {
            let by = if alg && self.reps[ri].sk_at + f == SKELETON_CAP { "recheck" } else { by };
            let what = format!("rep {rep} {phase} round {k} frame {f}");
            let w = wants[f].as_ref();
            let wf = w.map(frame).unwrap_or_default();
            match c {
                Item::Label(l) => {
                    let pads = self.reps[ri].mask.label(l);
                    let s = next(c.clone())?;
                    if !self.fixed(format!("{what}: label"), by, &wf, &frame(&s), vec![]) && stop {
                        return Ok(false);
                    }
                    if pads {
                        let root = self.reps[ri].pads_root.clone();
                        let (pl, pr) = (Item::Label(LABEL_PADS.to_vec()), Item::Bytes(root));
                        let s = next(pl.clone())?;
                        self.fixed(format!("rep {rep}: the pads' label"), "statement", &frame(&pl), &frame(&s), vec![]);
                        let s = next(pr.clone())?;
                        if !self.fixed(format!("rep {rep}: the pads' root"), "own", &frame(&pr), &frame(&s), vec![]) && stop {
                            return Ok(false);
                        }
                    }
                }
                Item::Obs { vals, .. } if self.reps[ri].mask.on => {
                    let idx: Vec<usize> = (0..vals.len()).map(|_| self.reps[ri].mask.pad()).collect::<Result<_, _>>()?;
                    let h = &self.reps[ri].pads.h;
                    let synth = match c {
                        Item::Obs { kind, vals } => Item::Obs { kind: *kind, vals: vals.iter().zip(&idx).map(|(v, i)| *v + h[*i]).collect() },
                        _ => unreachable!(),
                    };
                    if let (true, Some(Item::Obs { vals: tags, .. })) = (alg, w) {
                        if let Some(t) = tags.iter().zip(&idx).position(|(t, i)| t.lo as usize != *i) {
                            self.errors.push(format!("{what}: the skeleton's exposed index {} is not the masking order's {}", tags[t].lo, idx[t]));
                        }
                    }
                    let s = next(synth)?;
                    if !self.fixed(format!("{what}: observe header"), by, &obs_header(w), &obs_header(Some(&s)), vec![]) && stop {
                        return Ok(false);
                    }
                    let Item::Obs { vals: sent, .. } = s else { return Err(format!("{what}: the wire's frame is not an observe")) };
                    for ((v, i), x) in vals.iter().zip(&idx).zip(&sent) {
                        if !self.masked(ri, *v, *i, *x) && stop {
                            return Ok(false);
                        }
                    }
                }
                _ => {
                    let s = next(c.clone())?;
                    if !self.fixed(format!("{what}: {c}"), by, &wf, &frame(&s), vec![]) && stop {
                        return Ok(false);
                    }
                }
            }
        }
        if let Some(w) = wire {
            if wi != w.len() {
                return Err(format!("rep {rep} {phase} round {k}: the wire's round has frames the device's has not"));
            }
        }
        if alg {
            self.reps[ri].sk_at += cand.len();
        }
        Ok(true)
    }

    fn stop_round(&mut self, alg: bool, k: usize, content: &[u8], want: Option<(Vec<u8>, usize)>) -> bool {
        let Some(ri) = self.cur() else { return true };
        let cand = match parse(content) {
            Ok(c) => c,
            Err(e) => {
                self.errors.push(format!("the refused round does not parse: {e}"));
                return true;
            }
        };
        match self.round_items(ri, alg, k, &cand, None, &want) {
            Ok(ok) => ok,
            Err(e) => {
                self.errors.push(e);
                true
            }
        }
    }

    fn shape(lay: &Layout, zc: &flock_core::zerocheck::ZerocheckProof, lc: &flock_core::lincheck::LincheckProof) -> (Vec<u8>, Vec<u8>) {
        let got = [zc.round1_ab.len(), zc.round1_c.len(), zc.multilinear_rounds.len(), lc.rounds.len(), lc.z_partial.len()];
        (u64s(&[zkv::SKIP, zkv::SKIP, lay.n_mlv, lay.n_lc, zkv::SKIP]), u64s(&got))
    }

    fn piop_want(r: &RepRec) -> Option<Vec<u8>> {
        let (zc, lc) = zkv::from_exposed(r.vals.as_ref()?, &r.lay);
        let (z, l) = zkv::mask_proof(&zc, &lc, &r.lay, &r.pads.h);
        Some([bin(&z), bin(&l)].concat())
    }

    fn stop_proof(&mut self, p: &R1csProofLigerito) -> bool {
        let Some(ri) = self.cur() else { return true };
        let (lay, rep) = (self.reps[ri].lay, self.reps[ri].rep);
        let (w, s) = Self::shape(&lay, &p.zerocheck, &p.lincheck);
        if !self.fixed(format!("rep {rep} proof: shape (round1_ab, round1_c, multilinear_rounds, lincheck rounds, z_partial)"), "statement", &w, &s, vec![]) {
            return false;
        }
        let Some(want) = Self::piop_want(&self.reps[ri]) else { return true };
        if !p.zerocheck.grinding_nonces.is_empty() || !p.lincheck.grinding_nonces.is_empty() || !p.lincheck.matrix_evals.is_empty() {
            return true;
        }
        let (z, l) = zkv::mask_proof(&p.zerocheck, &p.lincheck, &lay, &self.reps[ri].pads.h);
        let pads = self.used(ri, |i| i < lay.piop_exposed());
        self.fixed(format!("rep {rep} proof: the masked zerocheck and lincheck"), "own", &want, &[bin(&z), bin(&l)].concat(), pads)
    }

    fn commit_roots(&mut self, sent: &[(String, Vec<u8>)], want: &[u8]) -> bool {
        let enc = |rs: &[(String, Vec<u8>)]| rs.iter().flat_map(|(n, r)| [n.as_bytes(), r.as_slice()].concat()).collect::<Vec<u8>>();
        let w: Vec<(String, Vec<u8>)> = sent.iter().map(|(n, _)| (n.clone(), want.to_vec())).collect();
        self.fixed("the session's Commit: the level-0 roots".into(), "recheck", &enc(&w), &enc(sent), vec![])
    }

    /// A request that left, with its answer.
    pub fn wire(&mut self, req: &Req, resp: &Resp) {
        if self.halted() && !matches!(req, Req::Finish) {
            return;
        }
        if let Err(e) = self.wire_inner(req, resp) {
            self.errors.push(e);
        }
        if matches!(req, Req::Finish) {
            self.write();
        }
    }

    fn wire_inner(&mut self, req: &Req, resp: &Resp) -> Result<(), String> {
        match req {
            Req::Register => {
                self.fixed("Register".into(), "statement", &Req::Register.encode(), &req.encode(), vec![]);
            }
            Req::Hello(_) => self.on_its_word("Hello: the session's configuration and coin nonce".into(), "own", &req.encode(), vec![]),
            Req::Open { stream, .. } => {
                let ri = self.cur().ok_or("an Open before any gated rep")?;
                let want = Req::Open { stream: stream.clone(), parent: String::new(), label: self.reps[ri].domain.clone(), at: 0 }.encode();
                self.reps[ri].stream = Some(stream.clone());
                let rep = self.reps[ri].rep;
                self.fixed(format!("rep {rep}: Open"), "statement", &want, &req.encode(), vec![]);
            }
            Req::Commit { root_f, roots, publics } => {
                self.pending = None;
                self.on_its_word("the session's Commit: σ (root_f)".into(), "own", root_f, vec![]);
                let want = self.root_want.clone().ok_or("a Commit before the firewall saw the device's root")?;
                self.tables = roots.len();
                self.commit_roots(roots, &want);
                let enc: Vec<u8> = publics.iter().flat_map(|(n, p)| [n.as_bytes(), p.as_slice()].concat()).collect();
                self.fixed("the session's Commit: public words".into(), "statement", &[], &enc, vec![]);
                if let Resp::Coins(c) = resp {
                    self.link = Some(c.clone());
                }
            }
            Req::Link(b) => {
                self.fixed("the session's Link: y".into(), "statement", &vec![0u8; 32 * self.tables], b, vec![]);
            }
            Req::Round { stream, content, n } => {
                let Resp::Coins(coins) = resp else { return Err(format!("{stream}: a round answered {resp:?}")) };
                let ri = self.reps.iter().rposition(|r| r.stream.as_deref() == Some(stream.as_str())).ok_or_else(|| format!("{stream}: a round on no rep's stream"))?;
                let k = self.reps[ri].wire_rounds;
                self.reps[ri].wire_rounds += 1;
                let wire = parse(content)?;
                match self.pending.take() {
                    Some(Pending::Round { alg, k: dk, content: dc, n: dn, want }) => {
                        if dn != *n as usize {
                            return Err(format!("{stream} round {k}: the wire squeezes {n}, the device {dn}"));
                        }
                        self.round_items(ri, alg, dk, &parse(&dc)?, Some(&wire), &want)?;
                        self.coins = Some((stream.clone(), k, coins.clone()));
                    }
                    Some(Pending::Proof { .. }) => return Err(format!("{stream} round {k}: a round while the firewall holds a proof")),
                    // a later rep's root, held to the recheck's but sent once, in rep 0's Commit
                    Some(Pending::Root { .. }) | None => self.inner_round(ri, &wire, coins)?,
                }
            }
            Req::Proof { rep, bytes, .. } => self.proof(*rep as usize, bytes)?,
            Req::Finish => {
                if self.refused.is_none() && self.stopped.is_none() && self.proofs == self.reps.len() && self.proofs > 0 {
                    self.items.push(json!({"finish": {}}));
                }
            }
            Req::Ping(_) | Req::Prime { .. } => {}
        }
        Ok(())
    }

    /// The inner proof's rounds on the rep's own stream (no coin is relayed).
    fn inner_round(&mut self, ri: usize, wire: &[Item], coins: &[F128]) -> Result<(), String> {
        let j = self.reps[ri].inner.ok_or("a round the firewall never saw, before the rep's inner proof")?;
        self.reps[ri].inner = Some(j + 1);
        let (rep, base, lay) = (self.reps[ri].rep, self.reps[ri].base, self.reps[ri].lay);
        let what = |f: usize| format!("rep {rep} inner proof round {j} frame {f}");
        let st = |me: &mut Self, f: usize, want: Item, got: &Item| {
            me.fixed(format!("{}: {want}", what(f)), "statement", &frame(&want), &frame(got), vec![]);
        };
        let all = |me: &Self| me.used(ri, |_| true);
        match (j, wire) {
            (0, [l, o @ Item::Obs { vals, .. }, sq]) if vals.len() == 1 => {
                st(self, 0, Item::Label(LABEL_INNER.to_vec()), l);
                let fc = self.reps[ri].vals.as_ref().ok_or("final_c' before the firewall passed the proof")?[lay.fc()];
                self.fixed(format!("{}: observe header", what(1)), "statement", &obs_header(Some(&Item::Obs { kind: KIND_SCALAR, vals: vec![F128::ZERO] })), &obs_header(Some(o)), vec![]);
                self.masked(ri, fc, lay.fc(), vals[0]);
                st(self, 2, Item::Squeeze { kind: KIND_SCALAR, n: 1 }, sq);
                self.reps[ri].eps = coins[0];
            }
            (1, [o @ Item::Obs { vals, .. }, sq]) if vals.len() == 2 => {
                let (eps, h) = (self.reps[ri].eps, self.reps[ri].pads.h.clone());
                self.fixed(format!("{}: observe header", what(0)), "statement", &obs_header(Some(&Item::Obs { kind: KIND_SLICE, vals: vec![F128::ZERO; 2] })), &obs_header(Some(o)), vec![]);
                self.masked(ri, eps * h[lay.fa()], lay.tx(), vals[0]);
                self.masked(ri, h[lay.fb()], lay.ty(), vals[1]);
                let mut cons = self.reps[ri].cons.clone().ok_or("the inner proof's batching before zk_finish's constraints")?;
                cons.extend(zkv::triple_constraints(&lay, eps, vals[0], vals[1]));
                st(self, 1, Item::Squeeze { kind: KIND_SLICE, n: cons.len() }, sq);
                let pads = all(self);
                for (name, l) in &cons {
                    if !self.relations.iter().any(|r| r == name) {
                        self.errors.push(format!("a constraint row {name} not among the relations"));
                    }
                    let a: Vec<Value> = l.a.iter().enumerate().filter(|(_, x)| **x != F128::ZERO).map(|(i, x)| json!([base + i, f128_hex(*x)])).collect();
                    self.items.push(json!({"check": {"relation": name, "pads": pads, "ok": l.eval(&h) == F128::ZERO,
                                                     "form": {"c": f128_hex(l.c), "a": a}}}));
                }
                self.reps[ri].c = zkv::batch(&cons, coins).0;
            }
            (2, [b, sq]) => {
                let r = &self.reps[ri];
                let tau = r.c.iter().zip(&r.pads.mu).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
                let want = Item::Bytes(zkv::tau_commitment(tau, &r.pads.tau_salt));
                let pads = all(self);
                self.fixed(format!("{}: τ's commitment", what(0)), "own", &frame(&want), &frame(b), pads);
                st(self, 1, Item::Squeeze { kind: KIND_SCALAR, n: 1 }, sq);
                self.reps[ri].beta = coins[0];
            }
            (3, [y, sq]) => {
                let r = &self.reps[ri];
                let beta = r.beta;
                let p = &r.pads;
                let yv: Vec<F128> = p.h.iter().zip(&p.mu).map(|(a, b)| *a + beta * *b).chain(p.pad_h.iter().zip(&p.pad_mu).map(|(a, b)| *a + beta * *b)).collect();
                let pads = all(self);
                self.fixed(format!("{}: Y", what(0)), "own", &frame(&Item::Obs { kind: KIND_SLICE, vals: yv }), &frame(y), pads);
                st(self, 1, Item::Squeeze { kind: KIND_SLICE, n: zkv::Q }, sq);
            }
            _ => {
                let pads = all(self);
                for (f, x) in wire.iter().enumerate() {
                    self.on_its_word(format!("{}: {x}", what(f)), "own", &frame(x), pads.clone());
                }
                self.errors.push(format!("rep {rep} inner proof round {j}: not a round the recorder knows"));
            }
        }
        Ok(())
    }

    fn proof(&mut self, rep: usize, bytes: &[u8]) -> Result<(), String> {
        let ri = self.reps.iter().position(|r| r.rep == rep).ok_or_else(|| format!("a proof of rep {rep}, which never began"))?;
        let (cm, p, inner): (Commitment, R1csProofLigerito, InnerProof) = bincode::deserialize(bytes).map_err(|e| format!("rep {rep}'s proof: {e}"))?;
        let lay = self.reps[ri].lay;
        let (w, s) = Self::shape(&lay, &p.zerocheck, &p.lincheck);
        self.fixed(format!("rep {rep} proof: shape (round1_ab, round1_c, multilinear_rounds, lincheck rounds, z_partial)"), "statement", &w, &s, vec![]);
        let want = Self::piop_want(&self.reps[ri]).ok_or_else(|| format!("rep {rep}'s proof before the firewall passed it"))?;
        let pads = self.used(ri, |i| i < lay.piop_exposed());
        self.fixed(format!("rep {rep} proof: the masked zerocheck and lincheck"), "own", &want, &[bin(&p.zerocheck), bin(&p.lincheck)].concat(), pads);
        let want = self.reps[ri].cm.clone();
        self.fixed(format!("rep {rep} proof: the level-0 commitment"), "recheck", &want, &bin(&cm), vec![]);
        let want = self.reps[ri].opening.clone().ok_or_else(|| format!("rep {rep}'s opening before the firewall passed it"))?;
        let pads = self.used(ri, |i| i >= lay.piop_exposed() && i < lay.hab());
        self.fixed(format!("rep {rep} proof: the opening"), "shadow", &want, &bin(&p.pcs_open), pads);
        let pads = self.used(ri, |_| true);
        self.on_its_word(format!("rep {rep} proof: the inner proof"), "own", &bin(&inner), pads);
        self.proofs += 1;
        Ok(())
    }

    fn write(&mut self) {
        let h = |v: &[F128]| v.iter().map(|x| f128_hex(*x)).collect::<Vec<_>>();
        let t = json!({
            "kind": FORMAT, "phase": "outer",
            "tape": {"salts": [], "pads": h(&self.pads)}, "exposed": {"salts": [], "pads": []},
            "items": self.items,
            "firewall": {"gate": {"refused": self.refused.as_ref().map(|r| r.0.clone()), "at": self.refused.as_ref().map(|r| r.1),
                                  "from": self.refused.as_ref().map(|r| r.2)},
                         "stopped": self.stopped, "tamper": self.tamper, "recorder_errors": self.errors,
                         "reps": self.reps.iter().map(|r| json!({"rep": r.rep, "pad_base": r.base, "layout": r.lay.to_json()})).collect::<Vec<_>>()},
        });
        let public = json!({"kind": PUBLIC_FORMAT, "schedule": null, "relations": self.relations, "statement": self.statement,
                            "computes": self.computes});
        for (path, v) in [(&self.public, &public), (&self.path, &t)] {
            if let Err(e) = std::fs::write(path, serde_json::to_vec(v).expect("JSON")) {
                eprintln!("FC_FIREWALL_TRANSCRIPT: {}: {e}", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_parse_back_to_their_items() {
        let items = vec![Item::Domain(b"d".to_vec()), Item::Label(b"flock-r1cs-v0".to_vec()), Item::Bytes(vec![7; 33]),
                         Item::Obs { kind: KIND_SLICE, vals: vec![F128::new(1, 2), F128::new(3, 4)] }, Item::Obs { kind: KIND_SCALAR, vals: vec![F128::new(5, 6)] },
                         Item::Squeeze { kind: KIND_SLICE, n: 0 }, Item::Squeeze { kind: KIND_SCALAR, n: 1 }];
        let bytes: Vec<u8> = items.iter().flat_map(frame).collect();
        assert_eq!(parse(&bytes).unwrap(), items);
    }

    /// The masking order: values between the zerocheck's label and the opening's take the layout's piop pads, level 0's the
    /// rest, and nothing else is masked.
    #[test]
    fn mask_follows_the_masking_challengers_phases() {
        let lay = Layout::new(25, 22, 2, 3);
        let mut m = Mask::new(&lay);
        assert!(!m.label(b"flock-r1cs-v0") && !m.on);
        assert!(m.label(LABEL_ZC) && m.on);
        let piop: Vec<usize> = (0..lay.piop_absorbed()).map(|_| m.pad().unwrap()).collect();
        assert_eq!(piop, lay.absorbed()[..lay.piop_absorbed()]);
        assert!(!m.label(LABEL_PCS) && !m.on && m.done);
        assert!(!m.label(LABEL_ZC) && !m.on, "a second zerocheck label inside the opening is not masked");
        m.label(L0_OPEN);
        assert!(m.on && m.l0);
        let l0: Vec<usize> = (0..lay.absorbed().len() - lay.piop_absorbed()).map(|_| m.pad().unwrap()).collect();
        assert_eq!(l0, lay.absorbed()[lay.piop_absorbed()..]);
        m.label(L0_CLOSE);
        assert!(!m.on && m.pad().is_err());
    }
}
